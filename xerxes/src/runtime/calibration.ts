// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Calibration (M3.3): record evaluated outcomes per workload, strategy and
 * hardware class, then recommend the strategy that works. OFF by default.
 *
 * Records never contain task text, only a workload tag, a strategy name, the
 * outcome and its measurements. Unknown costs stay undefined; nothing is
 * invented. The store is a local append-only JSONL file.
 */
import { appendFileSync, closeSync, fstatSync, mkdirSync, openSync, readSync } from 'node:fs'
import { dirname, join } from 'node:path'

import type { HardwareProfile } from '../local/hardware.js'
import type { MatrixReport } from '../../playground/matrixReport.js'
import { ValidationError } from '../core/errors.js'
import { xerxesHome } from '../daemon/paths.js'
import type { CalibrationLookup, CalibrationRecord, StrategyStats } from '../routing/types.js'

export type CalibrationMode = 'off' | 'record'

const MODES: readonly CalibrationMode[] = ['off', 'record']
const NAME_PATTERN = /^[\w.:+-]{1,80}$/
const MAX_READ_BYTES = 4 * 1024 * 1024
const GIB = 1024 ** 3
const MEMORY_BUCKETS_GIB: readonly number[] = [8, 12, 16, 24, 32, 48, 64, 96, 128]
const DEFAULT_SIMPLICITY_ORDER: readonly string[] = [
  'local-only',
  'cloud-cheap',
  'local-advisor',
  'cloud-strong',
  'hybrid-planned',
]
/** A simpler strategy may be this much slower than the winner and still be kept. */
const SIMPLER_WALL_SLACK = 1.2

function parseMode(key: string, raw: unknown): CalibrationMode | undefined {
  if (raw === undefined || raw === null) return undefined
  if (typeof raw !== 'string') throw new ValidationError(key, `expected one of ${MODES.join('|')}`, raw)
  const value = raw.trim().toLowerCase()
  if (value === '') return undefined
  const match = MODES.find(mode => mode === value)
  if (match === undefined) throw new ValidationError(key, `expected one of ${MODES.join('|')}`, raw)
  return match
}

/** `runtime.calibration` (env XERXES_CALIBRATION wins); default `off`. */
export function parseCalibrationConfig(
  runtime: Record<string, unknown>,
  env: Record<string, string | undefined>,
): CalibrationMode {
  return parseMode('XERXES_CALIBRATION', env.XERXES_CALIBRATION) ?? parseMode('calibration', runtime.calibration) ?? 'off'
}

function memoryBucket(bytes: number): string {
  const gib = bytes / GIB
  // 5% tolerance: a "24 GB" card reports slightly under 24 GiB.
  let bucket: number | undefined
  for (const candidate of MEMORY_BUCKETS_GIB) {
    if (candidate <= gib * 1.05) bucket = candidate
  }
  return bucket === undefined ? 'lt8g' : `${bucket}g`
}

/**
 * Coarse, stable hardware class so similar machines share statistics, e.g.
 * `darwin-arm64-apple-24g` or `linux-x64-nvidia-24g-vram`. Discrete GPUs key on
 * VRAM; everything else on system memory.
 */
export function hardwareKey(hw: HardwareProfile): string {
  const arch = hw.arch.toLowerCase().replace(/[^\w]/g, '') || 'unknown'
  if (hw.gpu.vendor === 'nvidia' && hw.gpu.vramBytes !== undefined && hw.gpu.vramBytes > 0) {
    return `${hw.platform}-${arch}-nvidia-${memoryBucket(hw.gpu.vramBytes)}-vram`
  }
  const kind = hw.gpu.vendor === 'apple' || hw.appleSilicon ? 'apple' : hw.gpu.vendor === 'nvidia' ? 'nvidia-novram' : 'cpu'
  return `${hw.platform}-${arch}-${kind}-${memoryBucket(hw.totalMemBytes)}`
}

function validName(field: string, value: unknown): string {
  if (typeof value !== 'string' || !NAME_PATTERN.test(value)) {
    throw new ValidationError(field, 'must match [A-Za-z0-9_.:+-]{1,80}', value)
  }
  return value
}

function validateRecord(input: Omit<CalibrationRecord, 'at'>): Omit<CalibrationRecord, 'at'> {
  const workload = validName('workload', input.workload)
  const key = validName('hardwareKey', input.hardwareKey)
  const strategy = validName('strategy', input.strategy)
  if (typeof input.accepted !== 'boolean') throw new ValidationError('accepted', 'must be a boolean', input.accepted)
  if (typeof input.wallMs !== 'number' || !Number.isFinite(input.wallMs) || input.wallMs < 0) {
    throw new ValidationError('wallMs', 'must be a non-negative finite number', input.wallMs)
  }
  if (input.costUsd !== undefined && (typeof input.costUsd !== 'number' || !Number.isFinite(input.costUsd) || input.costUsd < 0)) {
    throw new ValidationError('costUsd', 'must be a non-negative finite number when reported', input.costUsd)
  }
  if (input.retries !== undefined && (!Number.isInteger(input.retries) || input.retries < 0)) {
    throw new ValidationError('retries', 'must be a non-negative integer', input.retries)
  }
  // Rebuilt field by field so no extra property (such as task text) can ride along.
  return {
    workload,
    hardwareKey: key,
    strategy,
    accepted: input.accepted,
    wallMs: input.wallMs,
    ...(input.costUsd === undefined ? {} : { costUsd: input.costUsd }),
    ...(input.retries === undefined ? {} : { retries: input.retries }),
  }
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const middle = Math.floor(sorted.length / 2)
  const high = sorted[middle] ?? 0
  const low = sorted[middle - 1] ?? high
  return sorted.length % 2 === 1 ? high : (low + high) / 2
}

function parseLine(line: string): CalibrationRecord | undefined {
  let value: unknown
  try {
    value = JSON.parse(line)
  } catch {
    return undefined
  }
  if (typeof value !== 'object' || value === null) return undefined
  const row = value as Record<string, unknown>
  const { at, workload, hardwareKey: key, strategy, accepted, wallMs, costUsd, retries } = row
  if (typeof at !== 'string') return undefined
  try {
    const checked = validateRecord({
      workload: workload as string,
      hardwareKey: key as string,
      strategy: strategy as string,
      accepted: accepted as boolean,
      wallMs: wallMs as number,
      ...(costUsd === undefined ? {} : { costUsd: costUsd as number }),
      ...(retries === undefined ? {} : { retries: retries as number }),
    })
    return { at, ...checked }
  } catch {
    return undefined
  }
}

export interface CalibrationStoreOptions {
  readonly path?: string
  readonly now?: () => Date
}

export interface MatrixImportOptions {
  readonly hardwareKey: string
  /** Map a matrix task to a workload tag; default is the benchmark id. Must not return task text. */
  readonly workloadOf?: (taskId: string, benchmark: string) => string
}

export interface MatrixImportResult {
  readonly imported: number
  /** Runs without a gradable outcome (`needs_review`). */
  readonly skipped: number
}

export class CalibrationStore implements CalibrationLookup {
  readonly path: string
  private readonly now: () => Date

  constructor(options: CalibrationStoreOptions = {}) {
    this.path = options.path ?? join(xerxesHome(), 'calibration', 'outcomes.jsonl')
    this.now = options.now ?? (() => new Date())
  }

  /** Validate and append one outcome. Throws `ValidationError` on a bad field. */
  record(input: Omit<CalibrationRecord, 'at'>): CalibrationRecord {
    const checked = validateRecord(input)
    const full: CalibrationRecord = { at: this.now().toISOString(), ...checked }
    this.append([full])
    return full
  }

  private append(records: readonly CalibrationRecord[]): void {
    if (records.length === 0) return
    mkdirSync(dirname(this.path), { recursive: true, mode: 0o700 })
    appendFileSync(this.path, records.map(record => `${JSON.stringify(record)}\n`).join(''), { mode: 0o600 })
  }

  /** Records from the last 4 MiB of the file; malformed lines are ignored. */
  readAll(): CalibrationRecord[] {
    let fd: number
    try {
      fd = openSync(this.path, 'r')
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return []
      throw error
    }
    try {
      const size = fstatSync(fd).size
      const start = Math.max(0, size - MAX_READ_BYTES)
      const buffer = Buffer.alloc(size - start)
      let offset = 0
      while (offset < buffer.length) {
        const read = readSync(fd, buffer, offset, buffer.length - offset, start + offset)
        if (read === 0) break
        offset += read
      }
      let text = buffer.subarray(0, offset).toString('utf8')
      if (start > 0) {
        const newline = text.indexOf('\n')
        text = newline === -1 ? '' : text.slice(newline + 1)
      }
      const records: CalibrationRecord[] = []
      for (const line of text.split('\n')) {
        if (line === '') continue
        const parsed = parseLine(line)
        if (parsed !== undefined) records.push(parsed)
      }
      return records
    } finally {
      closeSync(fd)
    }
  }

  stats(workload: string, hardwareKeyValue: string): readonly StrategyStats[] {
    const groups = new Map<string, CalibrationRecord[]>()
    for (const record of this.readAll()) {
      if (record.workload !== workload || record.hardwareKey !== hardwareKeyValue) continue
      const group = groups.get(record.strategy)
      if (group === undefined) groups.set(record.strategy, [record])
      else group.push(record)
    }
    const result: StrategyStats[] = []
    for (const [strategy, records] of [...groups].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) {
      const accepted = records.filter(record => record.accepted).length
      const costs = records.flatMap(record => (record.costUsd === undefined ? [] : [record.costUsd]))
      result.push({
        strategy,
        runs: records.length,
        accepted,
        acceptRate: accepted / records.length,
        medianWallMs: median(records.map(record => record.wallMs)),
        ...(costs.length === 0 ? {} : { meanCostUsd: costs.reduce((sum, cost) => sum + cost, 0) / costs.length }),
      })
    }
    return result
  }

  /**
   * Turn a benchmark matrix result into records. The config id is the strategy;
   * the workload defaults to the benchmark id. `needs_review` runs have no
   * outcome and are skipped; errored runs count as not accepted. Everything is
   * validated before anything is written.
   */
  importMatrixReport(report: Pick<MatrixReport, 'records'>, options: MatrixImportOptions): MatrixImportResult {
    const at = this.now().toISOString()
    const records: CalibrationRecord[] = []
    let skipped = 0
    for (const run of report.records) {
      if (run.status === 'needs_review') {
        skipped += 1
        continue
      }
      const checked = validateRecord({
        workload: options.workloadOf === undefined ? run.benchmark : options.workloadOf(run.taskId, run.benchmark),
        hardwareKey: options.hardwareKey,
        strategy: run.configId,
        accepted: run.accepted,
        wallMs: run.wallMs,
        ...(run.costUsd === undefined ? {} : { costUsd: run.costUsd }),
      })
      records.push({ at, ...checked })
    }
    this.append(records)
    return { imported: records.length, skipped }
  }
}

export interface RecommendOptions {
  readonly minRuns?: number
  readonly simplicityOrder?: readonly string[]
  readonly tolerance?: { readonly acceptRate: number }
}

export interface Recommendation {
  readonly strategy: string | undefined
  readonly reason: string
  readonly considered: StrategyStats[]
}

function pct(value: number): string {
  return `${(value * 100).toFixed(0)}%`
}

/** True when `candidate` beats `current` on cost, then wall time. Unknown costs are not compared. */
function cheaperOrFaster(candidate: StrategyStats, current: StrategyStats): boolean {
  if (candidate.meanCostUsd !== undefined && current.meanCostUsd !== undefined && candidate.meanCostUsd !== current.meanCostUsd) {
    return candidate.meanCostUsd < current.meanCostUsd
  }
  return candidate.medianWallMs < current.medianWallMs
}

export function recommend(stats: readonly StrategyStats[], opts: RecommendOptions = {}): Recommendation {
  const minRuns = opts.minRuns ?? 3
  const order = opts.simplicityOrder ?? DEFAULT_SIMPLICITY_ORDER
  const tolerance = opts.tolerance?.acceptRate ?? 0.02
  const rank = (strategy: string): number => {
    const index = order.indexOf(strategy)
    return index === -1 ? Number.MAX_SAFE_INTEGER : index
  }

  const considered = stats.filter(entry => entry.runs >= minRuns)
  if (considered.length === 0) {
    const most = stats.reduce((max, entry) => Math.max(max, entry.runs), 0)
    return {
      strategy: undefined,
      reason: `not enough data: no strategy has ${minRuns} runs yet (most has ${most})`,
      considered: [],
    }
  }

  const bestRate = Math.max(...considered.map(entry => entry.acceptRate))
  const tied = considered
    .filter(entry => bestRate - entry.acceptRate <= tolerance + 1e-12)
    .sort((a, b) => rank(a.strategy) - rank(b.strategy))

  let winner = tied[0] as StrategyStats
  for (const candidate of tied.slice(1)) {
    if (cheaperOrFaster(candidate, winner)) winner = candidate
  }

  const simpler = tied
    .filter(entry => rank(entry.strategy) < rank(winner.strategy) && entry.medianWallMs <= winner.medianWallMs * SIMPLER_WALL_SLACK)
    .sort((a, b) => rank(a.strategy) - rank(b.strategy))[0]
  if (simpler !== undefined) {
    return {
      strategy: simpler.strategy,
      reason:
        `${simpler.strategy} kept over ${winner.strategy}: coordination adds no benefit ` +
        `(accept ${pct(simpler.acceptRate)} vs ${pct(winner.acceptRate)}, within ${pct(tolerance)}; not more than 20% slower)`,
      considered,
    }
  }

  const alone = tied.length === 1
  return {
    strategy: winner.strategy,
    reason: alone
      ? `${winner.strategy} has the best accept rate (${pct(winner.acceptRate)} over ${winner.runs} runs)`
      : `${winner.strategy} ties on quality (within ${pct(tolerance)} of ${pct(bestRate)}) and is cheaper or faster`,
    considered,
  }
}

/** Compact table for a `/calibration` command. */
export function renderCalibration(stats: readonly StrategyStats[], recommendation: Recommendation): string {
  if (stats.length === 0) return 'No calibration data yet for this workload and hardware.'
  const rows = stats.map(entry => [
    entry.strategy,
    String(entry.runs),
    `${entry.accepted}/${entry.runs} (${pct(entry.acceptRate)})`,
    `${Math.round(entry.medianWallMs)} ms`,
    entry.meanCostUsd === undefined ? 'n/a' : `$${entry.meanCostUsd.toFixed(4)}`,
  ])
  const header = ['strategy', 'runs', 'accepted', 'median wall', 'mean cost']
  const widths = header.map((title, column) => Math.max(title.length, ...rows.map(row => (row[column] as string).length)))
  const format = (cells: readonly string[]): string => cells.map((cell, column) => cell.padEnd(widths[column] as number)).join('  ').trimEnd()
  return [
    format(header),
    ...rows.map(format),
    '',
    recommendation.strategy === undefined
      ? `Recommendation: none. ${recommendation.reason}`
      : `Recommendation: ${recommendation.strategy}. ${recommendation.reason}`,
  ].join('\n')
}
