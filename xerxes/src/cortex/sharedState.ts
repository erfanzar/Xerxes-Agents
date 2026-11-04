// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Shared execution state for a run that mixes local and cloud workers (M3.2).
 *
 * One object keeps artifacts, per-worker access, step progress and committed
 * external actions consistent, so that:
 *
 *  - a worker only ever sees what its trust level allows (a cloud-trust worker
 *    can never read a private artifact while a data boundary is on);
 *  - private data cannot be laundered: sensitivity is sticky per ref, and
 *    anything produced by a worker that has read private data is private;
 *  - a failed or interrupted run can be resumed without redoing completed work
 *    and without silently repeating a committed external action.
 *
 * Journal (optional, `journalPath`): one JSON object per line, appended on
 * every mutation (dirs 0700, file 0600). `SharedExecutionState.restore`
 * rebuilds the exact state from it.
 *
 * What the journal stores: artifact VALUES are written only when
 * `persistValues: true`; by default the journal holds ref, version, sha256,
 * byte size, sensitivity and producer, never the content. A state restored
 * from a value-less journal therefore knows an artifact exists but cannot
 * serve it: `read` throws `VALUE_UNAVAILABLE`, and recovery treats a step
 * whose outputs cannot be served as needing a rerun (the producer must run
 * again or put the value back). Step failure reasons, check details and
 * external-action descriptions are clipped to 500 chars and ARE journaled, so
 * keep them free of private content.
 *
 * Only the latest version of an artifact keeps its value in memory; older
 * versions keep metadata (hash, size, producer, check).
 *
 * Off by default: nothing instantiates this unless hybrid plans are enabled.
 */

import { createHash } from 'node:crypto'
import { appendFileSync, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'

import type { DataBoundaryMode, DataTrust, Sensitivity } from '../routing/types.js'
import { DataBoundaryError } from '../security/boundaryTypes.js'

const CLIP = 500
const MAX_ID = 200

export type SharedStateErrorCode =
  | 'INVALID_ARGUMENT'
  | 'INVALID_STATE'
  | 'UNKNOWN_WORKER'
  | 'UNKNOWN_ARTIFACT'
  | 'UNKNOWN_STEP'
  | 'VALUE_UNAVAILABLE'
  | 'JOURNAL_EXISTS'
  | 'JOURNAL_NOT_FOUND'
  | 'JOURNAL_CORRUPT'

/** A typed shared-state failure. Never converted into a silent success. */
export class SharedStateError extends Error {
  readonly code: SharedStateErrorCode
  /** 1-based journal line for JOURNAL_CORRUPT, when known. */
  readonly line?: number

  constructor(code: SharedStateErrorCode, message: string, line?: number) {
    super(message)
    this.name = 'SharedStateError'
    this.code = code
    if (line !== undefined) this.line = line
  }
}

export interface SharedStateOptions {
  readonly runId: string
  readonly boundaryMode: DataBoundaryMode
  /** Append-only JSONL journal; omit for an in-memory state. */
  readonly journalPath?: string
  /** Write artifact values into the journal. Default false (hash + size only). */
  readonly persistValues?: boolean
  readonly now?: () => Date
}

export interface WorkerInput {
  readonly id: string
  readonly trust: DataTrust
  readonly role?: string
}

export interface ArtifactCheck {
  readonly passed: boolean
  readonly detail?: string
}

export interface ArtifactInput {
  readonly ref: string
  readonly value: string
  readonly sensitivity: Sensitivity
  readonly producedBy: string
  readonly check?: ArtifactCheck
}

export interface ArtifactVersionInfo {
  readonly ref: string
  readonly version: number
  /** Effective sensitivity of this version (may be stricter than declared). */
  readonly sensitivity: Sensitivity
  /** Why the declared sensitivity was raised to private, if it was. */
  readonly forced?: 'sticky' | 'taint'
  readonly sha256: string
  readonly bytes: number
}

export interface StepOutcome {
  /** Artifact refs the step produced. */
  readonly outputs: readonly string[]
  readonly accepted: boolean
}

export interface ExternalActionInput {
  /** Idempotency key for the committed side effect; unique within the run. */
  readonly key: string
  readonly description: string
}

export type StepStatus = 'running' | 'completed' | 'failed'

export interface StepSummary {
  readonly id: string
  readonly workerId: string
  readonly status: StepStatus
  readonly attempts: number
  readonly outputs: readonly string[]
  readonly accepted?: boolean
  readonly failure?: string
  readonly externalKeys: readonly string[]
}

export interface RecoveryPlan {
  /** Completed, accepted, all outputs still servable and passing checks. */
  readonly skip: readonly string[]
  /**
   * Safe to run again: failed, interrupted, rejected, outputs lost, or stale
   * because an upstream step must rerun. Steps with a committed external
   * action are NOT here, see `blockedExternal`.
   */
  readonly rerun: readonly string[]
  /**
   * Steps that need a rerun but already committed an external action. They
   * must not be repeated unless the caller confirms; the action wrapper
   * should consult `hasExternalAction(key)` before performing the effect.
   */
  readonly blockedExternal: readonly { readonly stepId: string; readonly keys: readonly string[] }[]
  /** Never started; nothing to repeat, they simply still have to run. */
  readonly pending: readonly string[]
  /** Human-readable reason per rerun/blocked step. */
  readonly reasons: Readonly<Record<string, string>>
}

export interface SharedStateSnapshot {
  readonly runId: string
  readonly boundaryMode: DataBoundaryMode
  readonly workers: readonly {
    readonly id: string
    readonly trust: DataTrust
    readonly role?: string
    /** Has read private data, so everything it writes is private. */
    readonly tainted: boolean
  }[]
  readonly artifacts: readonly {
    readonly ref: string
    readonly latestVersion: number
    readonly versions: number
    readonly sensitivity: Sensitivity
    readonly bytes: number
    readonly sha256: string
    readonly producedBy: string
    readonly createdAt: string
    readonly check?: ArtifactCheck
    readonly valueAvailable: boolean
  }[]
  readonly steps: readonly StepSummary[]
  readonly reads: { readonly total: number; readonly denied: number }
}

export interface RestoreReport {
  readonly events: number
  /** The last journal line was a partial write and was dropped. */
  readonly truncatedTail: boolean
  readonly droppedBytes: number
}

// -- journal events ---------------------------------------------------------

type JournalEvent =
  | { readonly t: 'run'; readonly at: string; readonly runId: string; readonly boundaryMode: DataBoundaryMode }
  | { readonly t: 'worker'; readonly at: string; readonly id: string; readonly trust: DataTrust; readonly role?: string }
  | {
    readonly t: 'artifact'
    readonly at: string
    readonly ref: string
    readonly version: number
    readonly sensitivity: Sensitivity
    readonly forced?: 'sticky' | 'taint'
    readonly producedBy: string
    readonly sha256: string
    readonly bytes: number
    readonly check?: ArtifactCheck
    readonly value?: string
  }
  | {
    readonly t: 'read'
    readonly at: string
    readonly worker: string
    readonly ref: string
    readonly version: number
    readonly sensitivity: Sensitivity
    readonly denied: boolean
  }
  | { readonly t: 'step_started'; readonly at: string; readonly step: string; readonly worker: string }
  | {
    readonly t: 'step_completed'
    readonly at: string
    readonly step: string
    readonly outputs: readonly string[]
    readonly accepted: boolean
  }
  | { readonly t: 'step_failed'; readonly at: string; readonly step: string; readonly reason: string }
  | { readonly t: 'external'; readonly at: string; readonly step: string; readonly key: string; readonly description: string }

function asRecord(v: unknown): Record<string, unknown> {
  if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new Error('journal line is not an object')
  return Object.fromEntries(Object.entries(v))
}

function reqStr(r: Record<string, unknown>, k: string): string {
  const v = r[k]
  if (typeof v !== 'string') throw new Error(`field "${k}" must be a string`)
  return v
}

function optStr(r: Record<string, unknown>, k: string): string | undefined {
  const v = r[k]
  if (v === undefined) return undefined
  if (typeof v !== 'string') throw new Error(`field "${k}" must be a string`)
  return v
}

function reqNum(r: Record<string, unknown>, k: string): number {
  const v = r[k]
  if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`field "${k}" must be a number`)
  return v
}

function reqBool(r: Record<string, unknown>, k: string): boolean {
  const v = r[k]
  if (typeof v !== 'boolean') throw new Error(`field "${k}" must be a boolean`)
  return v
}

function reqStrArr(r: Record<string, unknown>, k: string): string[] {
  const v = r[k]
  if (!Array.isArray(v) || v.some(x => typeof x !== 'string')) throw new Error(`field "${k}" must be a string array`)
  return v.map(String)
}

function reqSensitivity(r: Record<string, unknown>, k: string): Sensitivity {
  const v = reqStr(r, k)
  if (v !== 'private' && v !== 'shareable') throw new Error(`field "${k}" must be private|shareable`)
  return v
}

function reqTrust(r: Record<string, unknown>, k: string): DataTrust {
  const v = reqStr(r, k)
  if (v !== 'local' && v !== 'cloud') throw new Error(`field "${k}" must be local|cloud`)
  return v
}

function reqMode(r: Record<string, unknown>, k: string): DataBoundaryMode {
  const v = reqStr(r, k)
  if (v !== 'off' && v !== 'private' && v !== 'hybrid' && v !== 'cloud') throw new Error(`field "${k}" is not a boundary mode`)
  return v
}

function parseCheck(r: Record<string, unknown>): ArtifactCheck | undefined {
  const raw = r['check']
  if (raw === undefined) return undefined
  const c = asRecord(raw)
  const detail = optStr(c, 'detail')
  return { passed: reqBool(c, 'passed'), ...(detail !== undefined ? { detail } : {}) }
}

function parseEvent(raw: unknown): JournalEvent {
  const r = asRecord(raw)
  const at = reqStr(r, 'at')
  switch (r['t']) {
    case 'run':
      return { t: 'run', at, runId: reqStr(r, 'runId'), boundaryMode: reqMode(r, 'boundaryMode') }
    case 'worker': {
      const role = optStr(r, 'role')
      return { t: 'worker', at, id: reqStr(r, 'id'), trust: reqTrust(r, 'trust'), ...(role !== undefined ? { role } : {}) }
    }
    case 'artifact': {
      const forced = optStr(r, 'forced')
      if (forced !== undefined && forced !== 'sticky' && forced !== 'taint') throw new Error('field "forced" is invalid')
      const check = parseCheck(r)
      const value = optStr(r, 'value')
      return {
        t: 'artifact',
        at,
        ref: reqStr(r, 'ref'),
        version: reqNum(r, 'version'),
        sensitivity: reqSensitivity(r, 'sensitivity'),
        producedBy: reqStr(r, 'producedBy'),
        sha256: reqStr(r, 'sha256'),
        bytes: reqNum(r, 'bytes'),
        ...(forced !== undefined ? { forced } : {}),
        ...(check !== undefined ? { check } : {}),
        ...(value !== undefined ? { value } : {}),
      }
    }
    case 'read':
      return {
        t: 'read',
        at,
        worker: reqStr(r, 'worker'),
        ref: reqStr(r, 'ref'),
        version: reqNum(r, 'version'),
        sensitivity: reqSensitivity(r, 'sensitivity'),
        denied: reqBool(r, 'denied'),
      }
    case 'step_started':
      return { t: 'step_started', at, step: reqStr(r, 'step'), worker: reqStr(r, 'worker') }
    case 'step_completed':
      return { t: 'step_completed', at, step: reqStr(r, 'step'), outputs: reqStrArr(r, 'outputs'), accepted: reqBool(r, 'accepted') }
    case 'step_failed':
      return { t: 'step_failed', at, step: reqStr(r, 'step'), reason: reqStr(r, 'reason') }
    case 'external':
      return { t: 'external', at, step: reqStr(r, 'step'), key: reqStr(r, 'key'), description: reqStr(r, 'description') }
    default:
      throw new Error(`unknown event type ${JSON.stringify(r['t'])}`)
  }
}

// -- in-memory state --------------------------------------------------------

interface WorkerState {
  readonly id: string
  readonly trust: DataTrust
  readonly role: string | undefined
  tainted: boolean
}

interface VersionMeta {
  readonly version: number
  readonly sensitivity: Sensitivity
  readonly producedBy: string
  readonly sha256: string
  readonly bytes: number
  readonly createdAt: string
  readonly check: ArtifactCheck | undefined
}

interface ArtifactState {
  readonly ref: string
  /** Sticky: once private, always private. */
  sensitivity: Sensitivity
  readonly versions: VersionMeta[]
  value: string | undefined
}

interface StepState {
  readonly id: string
  workerId: string
  status: StepStatus
  attempts: number
  outputs: string[]
  accepted: boolean | undefined
  failure: string | undefined
  readonly externals: { readonly key: string; readonly description: string }[]
}

function clip(s: string, n = CLIP): string {
  return s.length <= n ? s : `${s.slice(0, n)}…`
}

function sha256Hex(s: string): string {
  return createHash('sha256').update(s).digest('hex')
}

function requireId(what: string, v: string): void {
  if (typeof v !== 'string' || v.length === 0 || v.length > MAX_ID) {
    throw new SharedStateError('INVALID_ARGUMENT', `${what} must be a non-empty string of at most ${MAX_ID} characters`)
  }
}

export class SharedExecutionState {
  readonly runId: string
  readonly boundaryMode: DataBoundaryMode
  private readonly persistValues: boolean
  private readonly now: () => Date
  private journalPath: string | undefined
  private headerSeen = false
  private restoreInfo: RestoreReport | undefined

  private readonly workers = new Map<string, WorkerState>()
  private readonly artifacts = new Map<string, ArtifactState>()
  private readonly steps = new Map<string, StepState>()
  /** idempotency key -> step id */
  private readonly externalKeys = new Map<string, string>()
  private readReads = 0
  private deniedReads = 0

  constructor(opts: SharedStateOptions) {
    requireId('runId', opts.runId)
    this.runId = opts.runId
    this.boundaryMode = opts.boundaryMode
    this.persistValues = opts.persistValues === true
    this.now = opts.now ?? (() => new Date())
    if (opts.journalPath !== undefined) {
      const path = opts.journalPath
      if (existsSync(path) && statSync(path).size > 0) {
        throw new SharedStateError(
          'JOURNAL_EXISTS',
          `journal ${path} already has content; use SharedExecutionState.restore() to continue it`,
        )
      }
      mkdirSync(dirname(path), { recursive: true, mode: 0o700 })
      this.journalPath = path
      this.commit({ t: 'run', at: this.stamp(), runId: this.runId, boundaryMode: this.boundaryMode })
    }
  }

  /**
   * Rebuild the exact state from a journal and keep appending to it. A
   * truncated last line (crash mid-write) is dropped, the file is repaired and
   * `restoreReport` says so. A malformed line anywhere else, an inconsistent
   * event, or a run-id mismatch is a `JOURNAL_CORRUPT` error with its line.
   */
  static restore(journalPath: string, opts: Omit<SharedStateOptions, 'journalPath'>): SharedExecutionState {
    if (!existsSync(journalPath)) {
      throw new SharedStateError('JOURNAL_NOT_FOUND', `no journal at ${journalPath}`)
    }
    const text = readFileSync(journalPath, 'utf8')
    const endsWithNewline = text.endsWith('\n')
    const lines = text.length === 0 ? [] : text.split('\n')
    if (endsWithNewline) lines.pop()

    const state = new SharedExecutionState(opts)
    let events = 0
    let truncatedTail = false
    let droppedBytes = 0
    for (const [i, line] of lines.entries()) {
      let ev: JournalEvent
      try {
        ev = parseEvent(JSON.parse(line))
      } catch (err) {
        if (i === lines.length - 1 && !endsWithNewline) {
          truncatedTail = true
          droppedBytes = Buffer.byteLength(line)
          break
        }
        throw new SharedStateError('JOURNAL_CORRUPT', `${journalPath}:${i + 1}: ${errMessage(err)}`, i + 1)
      }
      try {
        state.check(ev)
        state.mutate(ev)
      } catch (err) {
        throw new SharedStateError('JOURNAL_CORRUPT', `${journalPath}:${i + 1}: ${errMessage(err)}`, i + 1)
      }
      events++
    }
    if (!state.headerSeen) {
      throw new SharedStateError('JOURNAL_CORRUPT', `${journalPath}: no run header`, 1)
    }
    if (truncatedTail) {
      const good = lines.slice(0, lines.length - 1)
      writeFileSync(journalPath, good.map(l => `${l}\n`).join(''), { mode: 0o600 })
    } else if (!endsWithNewline) {
      appendFileSync(journalPath, '\n', { mode: 0o600 })
    }
    state.journalPath = journalPath
    state.restoreInfo = { events, truncatedTail, droppedBytes }
    return state
  }

  /** What `restore` found; undefined for a state that was not restored. */
  get restoreReport(): RestoreReport | undefined {
    return this.restoreInfo
  }

  // -- workers --------------------------------------------------------------

  registerWorker(input: WorkerInput): void {
    requireId('worker id', input.id)
    if (input.trust !== 'local' && input.trust !== 'cloud') {
      throw new SharedStateError('INVALID_ARGUMENT', `worker trust must be local|cloud, got ${String(input.trust)}`)
    }
    const existing = this.workers.get(input.id)
    if (existing !== undefined) {
      if (existing.trust !== input.trust) {
        throw new SharedStateError('INVALID_STATE', `worker ${input.id} is already registered with trust ${existing.trust}`)
      }
      return
    }
    this.commit({
      t: 'worker',
      at: this.stamp(),
      id: input.id,
      trust: input.trust,
      ...(input.role !== undefined ? { role: clip(input.role, MAX_ID) } : {}),
    })
  }

  // -- artifacts ------------------------------------------------------------

  /**
   * Store a new version of `ref`. Effective sensitivity is private when the
   * caller declared it, when the ref was ever private (sticky), or when the
   * producer has read any private artifact (taint).
   */
  putArtifact(input: ArtifactInput): ArtifactVersionInfo {
    requireId('artifact ref', input.ref)
    if (typeof input.value !== 'string') throw new SharedStateError('INVALID_ARGUMENT', 'artifact value must be a string')
    if (input.sensitivity !== 'private' && input.sensitivity !== 'shareable') {
      throw new SharedStateError('INVALID_ARGUMENT', `sensitivity must be private|shareable, got ${String(input.sensitivity)}`)
    }
    const worker = this.needWorker(input.producedBy)
    const existing = this.artifacts.get(input.ref)
    let sensitivity: Sensitivity = input.sensitivity
    let forced: 'sticky' | 'taint' | undefined
    if (sensitivity === 'shareable') {
      if (existing?.sensitivity === 'private') {
        sensitivity = 'private'
        forced = 'sticky'
      } else if (worker.tainted) {
        sensitivity = 'private'
        forced = 'taint'
      }
    }
    const version = (existing?.versions.length ?? 0) + 1
    const sha256 = sha256Hex(input.value)
    const bytes = Buffer.byteLength(input.value)
    const check = input.check !== undefined
      ? { passed: input.check.passed, ...(input.check.detail !== undefined ? { detail: clip(input.check.detail) } : {}) }
      : undefined
    this.commit({
      t: 'artifact',
      at: this.stamp(),
      ref: input.ref,
      version,
      sensitivity,
      producedBy: input.producedBy,
      sha256,
      bytes,
      ...(forced !== undefined ? { forced } : {}),
      ...(check !== undefined ? { check } : {}),
      ...(this.persistValues ? { value: input.value } : {}),
    }, input.value)
    return { ref: input.ref, version, sensitivity, sha256, bytes, ...(forced !== undefined ? { forced } : {}) }
  }

  /**
   * Latest value of `ref` for `workerId`. A cloud-trust worker reading a
   * private artifact while the boundary mode is not `off` is refused with
   * DATA_BOUNDARY_DENIED (the attempt is recorded, no taint). Every
   * successful read is recorded and taints the reader when the artifact is
   * private. Mode `off` never restricts, but still records and taints.
   */
  read(workerId: string, ref: string): string {
    const worker = this.needWorker(workerId)
    const art = this.artifacts.get(ref)
    if (art === undefined) throw new SharedStateError('UNKNOWN_ARTIFACT', `no artifact ${ref}`)
    const latest = art.versions[art.versions.length - 1]
    if (latest === undefined) throw new SharedStateError('UNKNOWN_ARTIFACT', `no artifact ${ref}`)
    const denied = this.deniesRead(worker, art)
    if (!denied && art.value === undefined) {
      throw new SharedStateError(
        'VALUE_UNAVAILABLE',
        `artifact ${ref} v${latest.version} exists but its value was not persisted; rerun its producer`,
      )
    }
    this.commit({
      t: 'read',
      at: this.stamp(),
      worker: workerId,
      ref,
      version: latest.version,
      sensitivity: art.sensitivity,
      denied,
    })
    if (denied || art.value === undefined) {
      throw new DataBoundaryError(
        'DATA_BOUNDARY_DENIED',
        `worker ${workerId} (cloud) may not read private artifact ${ref} under data boundary mode ${this.boundaryMode}`,
      )
    }
    return art.value
  }

  /** Refs `workerId` may read right now. Does not count as a read. */
  view(workerId: string): readonly { ref: string; version: number; sensitivity: Sensitivity; bytes: number }[] {
    const worker = this.needWorker(workerId)
    const out: { ref: string; version: number; sensitivity: Sensitivity; bytes: number }[] = []
    for (const art of this.artifacts.values()) {
      if (this.deniesRead(worker, art)) continue
      const latest = art.versions[art.versions.length - 1]
      if (latest === undefined) continue
      out.push({ ref: art.ref, version: latest.version, sensitivity: art.sensitivity, bytes: latest.bytes })
    }
    return out
  }

  // -- progress and recovery --------------------------------------------------

  /** Begin (or restart) a step. Restarting clears its outputs; external actions stay recorded. */
  stepStarted(stepId: string, workerId: string): void {
    requireId('step id', stepId)
    this.needWorker(workerId)
    this.commit({ t: 'step_started', at: this.stamp(), step: stepId, worker: workerId })
  }

  /**
   * Finish a running step. `accepted` is forced to false when any output's
   * latest result check failed, so progress never contradicts the checks.
   */
  stepCompleted(stepId: string, outcome: StepOutcome): StepSummary {
    this.needStep(stepId)
    for (const ref of outcome.outputs) requireId('output ref', ref)
    const checkFailed = outcome.outputs.some(ref => {
      const meta = this.artifacts.get(ref)?.versions.at(-1)
      return meta?.check?.passed === false
    })
    this.commit({
      t: 'step_completed',
      at: this.stamp(),
      step: stepId,
      outputs: [...outcome.outputs],
      accepted: outcome.accepted && !checkFailed,
    })
    return this.summarize(this.needStep(stepId))
  }

  stepFailed(stepId: string, reason: string): void {
    this.needStep(stepId)
    this.commit({ t: 'step_failed', at: this.stamp(), step: stepId, reason: clip(reason) })
  }

  /**
   * Record that a step committed an external side effect (sent email, wrote a
   * remote file). Call it after the effect succeeded. Returns false when the
   * same step already recorded the key (idempotent); a key used by a different
   * step is an error. Descriptions are journaled: keep them non-sensitive.
   */
  externalAction(stepId: string, action: ExternalActionInput): boolean {
    this.needStep(stepId)
    requireId('external action key', action.key)
    const owner = this.externalKeys.get(action.key)
    if (owner !== undefined) {
      if (owner === stepId) return false
      throw new SharedStateError('INVALID_STATE', `external action key "${action.key}" already belongs to step ${owner}`)
    }
    this.commit({ t: 'external', at: this.stamp(), step: stepId, key: action.key, description: clip(action.description) })
    return true
  }

  /** True when an external action with this idempotency key was already committed. */
  hasExternalAction(key: string): boolean {
    return this.externalKeys.has(key)
  }

  /**
   * Decide what a resumed run must do. `deps` maps a step to the steps it
   * depends on. Throws INVALID_ARGUMENT for duplicate ids, deps on unknown
   * steps, or a dependency cycle.
   */
  recoveryPlan(allStepIds: readonly string[], deps: ReadonlyMap<string, readonly string[]>): RecoveryPlan {
    const ids = new Set<string>()
    for (const id of allStepIds) {
      if (ids.has(id)) throw new SharedStateError('INVALID_ARGUMENT', `duplicate step id ${id}`)
      ids.add(id)
    }
    for (const [id, list] of deps) {
      if (!ids.has(id)) throw new SharedStateError('INVALID_ARGUMENT', `deps names unknown step ${id}`)
      for (const d of list) {
        if (!ids.has(d)) throw new SharedStateError('INVALID_ARGUMENT', `step ${id} depends on unknown step ${d}`)
      }
    }

    type Verdict = 'skip' | 'rerun' | 'pending'
    const memo = new Map<string, Verdict>()
    const reasons: Record<string, string> = {}
    const visiting = new Set<string>()

    const verdict = (id: string): Verdict => {
      const known = memo.get(id)
      if (known !== undefined) return known
      if (visiting.has(id)) throw new SharedStateError('INVALID_ARGUMENT', `dependency cycle through step ${id}`)
      visiting.add(id)
      const own = this.ownRecoveryState(id)
      let result: Verdict
      if (own.kind === 'pending') {
        for (const d of deps.get(id) ?? []) verdict(d) // still walk, to surface cycles
        result = 'pending'
      } else if (own.kind === 'bad') {
        for (const d of deps.get(id) ?? []) verdict(d)
        reasons[id] = own.reason
        result = 'rerun'
      } else {
        result = 'skip'
        for (const d of deps.get(id) ?? []) {
          const dv = verdict(d)
          if (dv !== 'skip') {
            reasons[id] = `upstream step ${d} ${dv === 'rerun' ? 'must rerun' : 'has not completed'}`
            result = 'rerun'
          }
        }
      }
      visiting.delete(id)
      memo.set(id, result)
      return result
    }

    const skip: string[] = []
    const rerun: string[] = []
    const pending: string[] = []
    const blockedExternal: { stepId: string; keys: string[] }[] = []
    for (const id of allStepIds) {
      const v = verdict(id)
      if (v === 'skip') skip.push(id)
      else if (v === 'pending') pending.push(id)
      else {
        const keys = this.steps.get(id)?.externals.map(e => e.key) ?? []
        if (keys.length > 0) {
          blockedExternal.push({ stepId: id, keys })
          reasons[id] = `${reasons[id] ?? 'needs rerun'}; already committed external action(s): ${keys.join(', ')}`
        } else {
          rerun.push(id)
        }
      }
    }
    return { skip, rerun, blockedExternal, pending, reasons }
  }

  // -- status ---------------------------------------------------------------

  /** Serializable summary for a status command. Never contains artifact values. */
  snapshot(): SharedStateSnapshot {
    return {
      runId: this.runId,
      boundaryMode: this.boundaryMode,
      workers: [...this.workers.values()].map(w => ({
        id: w.id,
        trust: w.trust,
        ...(w.role !== undefined ? { role: w.role } : {}),
        tainted: w.tainted,
      })),
      artifacts: [...this.artifacts.values()].flatMap(a => {
        const latest = a.versions[a.versions.length - 1]
        if (latest === undefined) return []
        return [{
          ref: a.ref,
          latestVersion: latest.version,
          versions: a.versions.length,
          sensitivity: a.sensitivity,
          bytes: latest.bytes,
          sha256: latest.sha256,
          producedBy: latest.producedBy,
          createdAt: latest.createdAt,
          ...(latest.check !== undefined ? { check: latest.check } : {}),
          valueAvailable: a.value !== undefined,
        }]
      }),
      steps: [...this.steps.values()].map(s => this.summarize(s)),
      reads: { total: this.readReads, denied: this.deniedReads },
    }
  }

  // -- internals --------------------------------------------------------------

  private stamp(): string {
    return this.now().toISOString()
  }

  private needWorker(id: string): WorkerState {
    const w = this.workers.get(id)
    if (w === undefined) throw new SharedStateError('UNKNOWN_WORKER', `worker ${id} is not registered`)
    return w
  }

  private needStep(id: string): StepState {
    const s = this.steps.get(id)
    if (s === undefined) throw new SharedStateError('UNKNOWN_STEP', `step ${id} was never started`)
    return s
  }

  private deniesRead(worker: WorkerState, art: ArtifactState): boolean {
    return art.sensitivity === 'private' && worker.trust === 'cloud' && this.boundaryMode !== 'off'
  }

  private summarize(s: StepState): StepSummary {
    return {
      id: s.id,
      workerId: s.workerId,
      status: s.status,
      attempts: s.attempts,
      outputs: [...s.outputs],
      ...(s.accepted !== undefined ? { accepted: s.accepted } : {}),
      ...(s.failure !== undefined ? { failure: s.failure } : {}),
      externalKeys: s.externals.map(e => e.key),
    }
  }

  private ownRecoveryState(id: string): { kind: 'pending' } | { kind: 'good' } | { kind: 'bad'; reason: string } {
    const s = this.steps.get(id)
    if (s === undefined) return { kind: 'pending' }
    if (s.status === 'running') return { kind: 'bad', reason: 'interrupted: started but never finished' }
    if (s.status === 'failed') return { kind: 'bad', reason: `failed: ${s.failure ?? 'no reason recorded'}` }
    if (s.accepted !== true) return { kind: 'bad', reason: 'completed but not accepted' }
    for (const ref of s.outputs) {
      const art = this.artifacts.get(ref)
      if (art === undefined) return { kind: 'bad', reason: `output ${ref} is missing` }
      if (art.value === undefined) return { kind: 'bad', reason: `output ${ref} cannot be served (value not persisted)` }
      if (art.versions.at(-1)?.check?.passed === false) return { kind: 'bad', reason: `output ${ref} failed its result check` }
    }
    return { kind: 'good' }
  }

  /** Validate, journal, then apply. Nothing is applied if the journal write throws. */
  private commit(ev: JournalEvent, value?: string): void {
    this.check(ev)
    if (this.journalPath !== undefined) {
      appendFileSync(this.journalPath, `${JSON.stringify(ev)}\n`, { mode: 0o600 })
    }
    this.mutate(ev, value)
  }

  /** Pure consistency check of an event against the current state. */
  private check(ev: JournalEvent): void {
    switch (ev.t) {
      case 'run':
        if (this.headerSeen) throw new SharedStateError('JOURNAL_CORRUPT', 'duplicate run header')
        if (ev.runId !== this.runId) {
          throw new SharedStateError('JOURNAL_CORRUPT', `journal belongs to run ${ev.runId}, not ${this.runId}`)
        }
        return
      case 'worker': {
        const w = this.workers.get(ev.id)
        if (w !== undefined && w.trust !== ev.trust) {
          throw new SharedStateError('INVALID_STATE', `worker ${ev.id} re-registered with a different trust`)
        }
        return
      }
      case 'artifact': {
        this.needWorker(ev.producedBy)
        const art = this.artifacts.get(ev.ref)
        const expected = (art?.versions.length ?? 0) + 1
        if (ev.version !== expected) {
          throw new SharedStateError('INVALID_STATE', `artifact ${ev.ref}: expected version ${expected}, got ${ev.version}`)
        }
        if (art?.sensitivity === 'private' && ev.sensitivity !== 'private') {
          throw new SharedStateError('INVALID_STATE', `artifact ${ev.ref}: sensitivity cannot drop from private`)
        }
        return
      }
      case 'read': {
        this.needWorker(ev.worker)
        const art = this.artifacts.get(ev.ref)
        if (art === undefined) throw new SharedStateError('UNKNOWN_ARTIFACT', `read of unknown artifact ${ev.ref}`)
        if (ev.version !== art.versions.length) {
          throw new SharedStateError('INVALID_STATE', `read of ${ev.ref} v${ev.version}, latest is v${art.versions.length}`)
        }
        return
      }
      case 'step_started':
        this.needWorker(ev.worker)
        return
      case 'step_completed':
      case 'step_failed': {
        const s = this.needStep(ev.step)
        if (s.status !== 'running') {
          throw new SharedStateError('INVALID_STATE', `step ${ev.step} is ${s.status}, not running`)
        }
        return
      }
      case 'external': {
        this.needStep(ev.step)
        const owner = this.externalKeys.get(ev.key)
        if (owner !== undefined) {
          throw new SharedStateError('INVALID_STATE', `external action key "${ev.key}" already recorded for step ${owner}`)
        }
        return
      }
    }
  }

  /** Apply an already-checked event. `liveValue` is the in-memory value of a live put. */
  private mutate(ev: JournalEvent, liveValue?: string): void {
    switch (ev.t) {
      case 'run':
        this.headerSeen = true
        return
      case 'worker':
        if (!this.workers.has(ev.id)) {
          this.workers.set(ev.id, { id: ev.id, trust: ev.trust, role: ev.role, tainted: false })
        }
        return
      case 'artifact': {
        let art = this.artifacts.get(ev.ref)
        if (art === undefined) {
          art = { ref: ev.ref, sensitivity: 'shareable', versions: [], value: undefined }
          this.artifacts.set(ev.ref, art)
        }
        art.versions.push({
          version: ev.version,
          sensitivity: ev.sensitivity,
          producedBy: ev.producedBy,
          sha256: ev.sha256,
          bytes: ev.bytes,
          createdAt: ev.at,
          check: ev.check,
        })
        if (ev.sensitivity === 'private') art.sensitivity = 'private'
        art.value = liveValue ?? ev.value
        return
      }
      case 'read': {
        this.readReads++
        if (ev.denied) {
          this.deniedReads++
        } else if (ev.sensitivity === 'private') {
          const w = this.workers.get(ev.worker)
          if (w !== undefined) w.tainted = true
        }
        return
      }
      case 'step_started': {
        const s = this.steps.get(ev.step)
        if (s === undefined) {
          this.steps.set(ev.step, {
            id: ev.step,
            workerId: ev.worker,
            status: 'running',
            attempts: 1,
            outputs: [],
            accepted: undefined,
            failure: undefined,
            externals: [],
          })
        } else {
          s.workerId = ev.worker
          s.status = 'running'
          s.attempts++
          s.outputs = []
          s.accepted = undefined
          s.failure = undefined
        }
        return
      }
      case 'step_completed': {
        const s = this.steps.get(ev.step)
        if (s === undefined) return
        s.status = 'completed'
        s.outputs = [...ev.outputs]
        s.accepted = ev.accepted
        s.failure = undefined
        return
      }
      case 'step_failed': {
        const s = this.steps.get(ev.step)
        if (s === undefined) return
        s.status = 'failed'
        s.failure = ev.reason
        s.accepted = undefined
        return
      }
      case 'external': {
        const s = this.steps.get(ev.step)
        if (s === undefined) return
        s.externals.push({ key: ev.key, description: ev.description })
        this.externalKeys.set(ev.key, ev.step)
        return
      }
    }
  }
}

function errMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}
