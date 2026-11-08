// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import type { MatrixReport, TaskRecord } from '../playground/matrixReport.js'
import { ValidationError } from '../src/core/errors.js'
import type { HardwareProfile } from '../src/local/hardware.js'
import {
  CalibrationStore,
  hardwareKey,
  recommend,
  renderCalibration,
} from '../src/runtime/calibration.js'
import type { StrategyStats } from '../src/routing/types.js'

const GIB = 1024 ** 3

let dir: string
let path: string

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'xerxes-calibration-'))
  path = join(dir, 'nested', 'outcomes.jsonl')
})

afterEach(() => {
  rmSync(dir, { recursive: true, force: true })
})

function stat(strategy: string, over: Partial<StrategyStats> = {}): StrategyStats {
  return { strategy, runs: 5, accepted: 4, acceptRate: 0.8, medianWallMs: 1000, ...over }
}

describe('hardwareKey', () => {
  test('mac unified memory buckets by system memory', () => {
    const mac: HardwareProfile = {
      platform: 'darwin',
      arch: 'arm64',
      totalMemBytes: 24 * GIB,
      appleSilicon: true,
      gpu: { vendor: 'apple' },
    }
    expect(hardwareKey(mac)).toBe('darwin-arm64-apple-24g')
    expect(hardwareKey({ ...mac, totalMemBytes: 36 * GIB })).toBe('darwin-arm64-apple-32g')
    expect(hardwareKey({ ...mac, totalMemBytes: 6 * GIB })).toBe('darwin-arm64-apple-lt8g')
  })

  test('nvidia keys on vram, tolerating slightly under-reported cards', () => {
    const box: HardwareProfile = {
      platform: 'linux',
      arch: 'x64',
      totalMemBytes: 64 * GIB,
      appleSilicon: false,
      gpu: { vendor: 'nvidia', name: 'RTX 4090', vramBytes: Math.floor(23.9 * GIB) },
    }
    expect(hardwareKey(box)).toBe('linux-x64-nvidia-24g-vram')
    expect(hardwareKey({ ...box, gpu: { vendor: 'nvidia', vramBytes: 256 * GIB } })).toBe('linux-x64-nvidia-128g-vram')
  })

  test('no gpu keys on system memory', () => {
    const cpu: HardwareProfile = {
      platform: 'linux',
      arch: 'x64',
      totalMemBytes: 16 * GIB,
      appleSilicon: false,
      gpu: { vendor: 'none' },
    }
    expect(hardwareKey(cpu)).toBe('linux-x64-cpu-16g')
  })

  test('similar machines share a key', () => {
    const base: HardwareProfile = {
      platform: 'linux',
      arch: 'x64',
      totalMemBytes: 16 * GIB,
      appleSilicon: false,
      gpu: { vendor: 'none' },
    }
    expect(hardwareKey({ ...base, totalMemBytes: 15.6 * GIB })).toBe(hardwareKey(base))
  })
})

describe('CalibrationStore', () => {
  test('record validates fields', () => {
    const store = new CalibrationStore({ path })
    const good = { workload: 'csv-reconcile', hardwareKey: 'darwin-arm64-apple-24g', strategy: 'local-only', accepted: true, wallMs: 10 }
    expect(() => store.record({ ...good, workload: 'has space' })).toThrow(ValidationError)
    expect(() => store.record({ ...good, strategy: '' })).toThrow(ValidationError)
    expect(() => store.record({ ...good, strategy: 'x'.repeat(81) })).toThrow(ValidationError)
    expect(() => store.record({ ...good, wallMs: -1 })).toThrow(ValidationError)
    expect(() => store.record({ ...good, wallMs: Number.NaN })).toThrow(ValidationError)
    expect(() => store.record({ ...good, costUsd: -0.1 })).toThrow(ValidationError)
    expect(store.readAll()).toEqual([])
    expect(() => store.record(good)).not.toThrow()
    expect(store.readAll()).toHaveLength(1)
  })

  test('missing cost stays undefined and is never written as 0', () => {
    const store = new CalibrationStore({ path, now: () => new Date('2026-11-03T00:00:00Z') })
    store.record({ workload: 'w', hardwareKey: 'h', strategy: 'local-only', accepted: true, wallMs: 5 })
    const line = readFileSync(path, 'utf8').trim()
    expect(line).not.toContain('costUsd')
    expect(JSON.parse(line).at).toBe('2026-11-03T00:00:00.000Z')
  })

  test('creates the directory 0700 and the file 0600', () => {
    const store = new CalibrationStore({ path })
    store.record({ workload: 'w', hardwareKey: 'h', strategy: 's', accepted: true, wallMs: 1 })
    expect(statSync(join(dir, 'nested')).mode & 0o777).toBe(0o700)
    expect(statSync(path).mode & 0o777).toBe(0o600)
  })

  test('stats: median with even counts, accept rate, and per-workload+hardware filtering', () => {
    const store = new CalibrationStore({ path })
    const walls = [400, 100, 300, 200]
    walls.forEach((wallMs, index) =>
      store.record({ workload: 'w', hardwareKey: 'h', strategy: 'local-only', accepted: index < 3, wallMs }),
    )
    store.record({ workload: 'w', hardwareKey: 'h', strategy: 'local-only', accepted: true, wallMs: 1_000_000 - 999_999 })
    store.record({ workload: 'other', hardwareKey: 'h', strategy: 'local-only', accepted: false, wallMs: 1 })
    store.record({ workload: 'w', hardwareKey: 'other', strategy: 'local-only', accepted: false, wallMs: 1 })
    const [row] = store.stats('w', 'h')
    expect(store.stats('w', 'h')).toHaveLength(1)
    expect(row?.runs).toBe(5)
    expect(row?.accepted).toBe(4)
    expect(row?.acceptRate).toBeCloseTo(0.8)
    expect(row?.medianWallMs).toBe(200)
  })

  test('stats: even-count median averages the middle pair', () => {
    const store = new CalibrationStore({ path })
    for (const wallMs of [100, 200, 300, 400]) {
      store.record({ workload: 'w', hardwareKey: 'h', strategy: 's', accepted: true, wallMs })
    }
    expect(store.stats('w', 'h')[0]?.medianWallMs).toBe(250)
  })

  test('stats: meanCostUsd covers only runs that reported a cost, undefined when none did', () => {
    const store = new CalibrationStore({ path })
    store.record({ workload: 'w', hardwareKey: 'h', strategy: 'free', accepted: true, wallMs: 1 })
    store.record({ workload: 'w', hardwareKey: 'h', strategy: 'free', accepted: true, wallMs: 1 })
    store.record({ workload: 'w', hardwareKey: 'h', strategy: 'paid', accepted: true, wallMs: 1, costUsd: 0.1 })
    store.record({ workload: 'w', hardwareKey: 'h', strategy: 'paid', accepted: true, wallMs: 1, costUsd: 0.3 })
    store.record({ workload: 'w', hardwareKey: 'h', strategy: 'paid', accepted: true, wallMs: 1 })
    const stats = store.stats('w', 'h')
    const free = stats.find(entry => entry.strategy === 'free')
    const paid = stats.find(entry => entry.strategy === 'paid')
    expect(free?.meanCostUsd).toBeUndefined()
    expect(free && 'meanCostUsd' in free).toBe(false)
    expect(paid?.meanCostUsd).toBeCloseTo(0.2)
    expect(paid?.runs).toBe(3)
  })

  test('reading skips malformed lines and a missing file is empty', () => {
    const store = new CalibrationStore({ path })
    expect(store.stats('w', 'h')).toEqual([])
    store.record({ workload: 'w', hardwareKey: 'h', strategy: 's', accepted: true, wallMs: 1 })
    writeFileSync(path, `not json\n{"at":"x"}\n${readFileSync(path, 'utf8')}`)
    expect(store.readAll()).toHaveLength(1)
  })

  test('never writes task text, even if an extra field is smuggled in', () => {
    const store = new CalibrationStore({ path })
    const canary = 'CANARY-PRIVATE-TASK-TEXT-9f3a'
    const smuggled = Object.assign(
      { workload: 'w', hardwareKey: 'h', strategy: 's', accepted: true, wallMs: 1 },
      { prompt: canary, task: canary },
    )
    store.record(smuggled)
    expect(() => store.record({ workload: canary + ' x', hardwareKey: 'h', strategy: 's', accepted: true, wallMs: 1 })).toThrow(
      ValidationError,
    )
    const content = readFileSync(path, 'utf8')
    expect(content).not.toContain(canary)
    expect(Object.keys(JSON.parse(content.trim())).sort()).toEqual(['accepted', 'at', 'hardwareKey', 'strategy', 'wallMs', 'workload'])
  })

  test('importMatrixReport maps config id to strategy and skips needs_review', () => {
    const base = { details: 'ok', disclosureFailure: false }
    const records: TaskRecord[] = [
      { ...base, configId: 'local-only', taskId: 't1', benchmark: 'dabstep', status: 'accepted', accepted: true, score: 1, wallMs: 1200 },
      { ...base, configId: 'local-only', taskId: 't2', benchmark: 'dabstep', status: 'rejected', accepted: false, score: 0, wallMs: 800 },
      { ...base, configId: 'cloud-strong', taskId: 't1', benchmark: 'dabstep', status: 'accepted', accepted: true, score: 1, wallMs: 500, costUsd: 0.05 },
      { ...base, configId: 'cloud-strong', taskId: 't3', benchmark: 'dabstep', status: 'error', accepted: false, score: 0, wallMs: 50, error: 'boom' },
      { ...base, configId: 'cloud-strong', taskId: 't4', benchmark: 'dabstep', status: 'needs_review', accepted: false, score: 0, wallMs: 70 },
    ]
    const report: MatrixReport = { generatedAt: '2026-11-03T00:00:00Z', constrained: [], unrestricted: [], records, excluded: [] }
    const store = new CalibrationStore({ path })
    const result = store.importMatrixReport(report, { hardwareKey: 'darwin-arm64-apple-24g' })
    expect(result).toEqual({ imported: 4, skipped: 1 })

    const local = store.stats('dabstep', 'darwin-arm64-apple-24g').find(entry => entry.strategy === 'local-only')
    const cloud = store.stats('dabstep', 'darwin-arm64-apple-24g').find(entry => entry.strategy === 'cloud-strong')
    expect(local).toMatchObject({ runs: 2, accepted: 1, acceptRate: 0.5, medianWallMs: 1000 })
    expect(local?.meanCostUsd).toBeUndefined()
    expect(cloud).toMatchObject({ runs: 2, accepted: 1 })
    expect(cloud?.meanCostUsd).toBeCloseTo(0.05)
  })

  test('importMatrixReport honours workloadOf and is all-or-nothing on a bad field', () => {
    const base = { details: '', disclosureFailure: false, score: 1, status: 'accepted' as const, accepted: true }
    const records: TaskRecord[] = [
      { ...base, configId: 'local-only', taskId: 'a', benchmark: 'gdpval', wallMs: 10 },
      { ...base, configId: 'bad id with spaces', taskId: 'b', benchmark: 'gdpval', wallMs: 10 },
    ]
    const report: MatrixReport = { generatedAt: 'x', constrained: [], unrestricted: [], records, excluded: [] }
    const store = new CalibrationStore({ path })
    expect(() => store.importMatrixReport(report, { hardwareKey: 'h' })).toThrow(ValidationError)
    expect(store.readAll()).toEqual([])

    store.importMatrixReport({ records: records.slice(0, 1) }, { hardwareKey: 'h', workloadOf: (_task, benchmark) => `wl-${benchmark}` })
    expect(store.stats('wl-gdpval', 'h')).toHaveLength(1)
  })
})

describe('recommend', () => {
  test('not enough data returns no strategy with a reason', () => {
    const result = recommend([stat('local-only', { runs: 2 }), stat('cloud-strong', { runs: 1 })])
    expect(result.strategy).toBeUndefined()
    expect(result.reason).toContain('not enough data')
    expect(result.considered).toEqual([])
    expect(recommend([]).strategy).toBeUndefined()
  })

  test('picks the clearly best accept rate', () => {
    const result = recommend([
      stat('local-only', { acceptRate: 0.5, accepted: 3 }),
      stat('cloud-strong', { acceptRate: 0.9, accepted: 5, meanCostUsd: 0.5 }),
    ])
    expect(result.strategy).toBe('cloud-strong')
  })

  test('ignores strategies below minRuns even if they look best', () => {
    const result = recommend([stat('local-only', { acceptRate: 0.6 }), stat('cloud-strong', { acceptRate: 1, runs: 2 })])
    expect(result.strategy).toBe('local-only')
    expect(result.considered.map(entry => entry.strategy)).toEqual(['local-only'])
  })

  test('within tolerance, the simpler strategy is kept when not much slower', () => {
    const result = recommend([
      stat('local-only', { acceptRate: 0.8, medianWallMs: 1000 }),
      stat('cloud-strong', { acceptRate: 0.82, medianWallMs: 900, meanCostUsd: 0.5 }),
    ])
    expect(result.strategy).toBe('local-only')
    expect(result.reason).toContain('coordination adds no benefit')
  })

  test('the simpler strategy loses when more than 20% slower', () => {
    const result = recommend([
      stat('local-only', { acceptRate: 0.8, medianWallMs: 2000 }),
      stat('cloud-strong', { acceptRate: 0.82, medianWallMs: 900, meanCostUsd: 0.5 }),
    ])
    expect(result.strategy).toBe('cloud-strong')
  })

  test('among equals on quality, lower cost wins over speed', () => {
    const result = recommend(
      [
        stat('a', { acceptRate: 0.9, meanCostUsd: 0.4, medianWallMs: 100 }),
        stat('b', { acceptRate: 0.9, meanCostUsd: 0.1, medianWallMs: 5000 }),
      ],
      { simplicityOrder: [] },
    )
    expect(result.strategy).toBe('b')
  })

  test('tolerance is configurable and unknown strategies rank as most complex', () => {
    const stats = [stat('custom', { acceptRate: 0.9 }), stat('local-only', { acceptRate: 0.85 })]
    expect(recommend(stats).strategy).toBe('custom')
    expect(recommend(stats, { tolerance: { acceptRate: 0.1 } }).strategy).toBe('local-only')
    expect(recommend(stats, { minRuns: 6 }).strategy).toBeUndefined()
  })
})

describe('renderCalibration', () => {
  test('renders a table with n/a for unknown cost and the recommendation', () => {
    const stats = [stat('local-only'), stat('cloud-strong', { meanCostUsd: 0.1234, medianWallMs: 900 })]
    const text = renderCalibration(stats, recommend(stats))
    expect(text).toContain('strategy')
    expect(text).toContain('local-only')
    expect(text).toContain('n/a')
    expect(text).toContain('$0.1234')
    expect(text).toContain('Recommendation: local-only')
  })

  test('empty and no-recommendation cases', () => {
    expect(renderCalibration([], recommend([]))).toContain('No calibration data')
    const stats = [stat('local-only', { runs: 1 })]
    expect(renderCalibration(stats, recommend(stats))).toContain('Recommendation: none')
  })
})
