// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Coalescing micro-batcher for daemon -> renderer Electron IPC events.
 *
 * High-frequency streaming chunks (`text_part`, `think_part`) and ephemeral
 * telemetry updates (`swarm_event` `member_activity`, routine `status_update`,
 * subagent `heartbeat`/`progress`) are coalesced within a ~16 ms frame window
 * and shipped as a single IPC call (`daemon:event` when 1 frame, or
 * `daemon:event-batch` when >= 2 frames). Interactive/lifecycle/terminal events
 * flush pending deltas first (strict FIFO ordering) and deliver immediately with
 * zero added latency.
 */

export const DAEMON_EVENT_CHANNEL = 'daemon:event'
export const DAEMON_EVENT_BATCH_CHANNEL = 'daemon:event-batch'

export interface DaemonEventFrame {
  readonly type: string
  readonly payload: Record<string, unknown>
}

export interface DaemonEventBatcherOptions {
  readonly sendSingle: (type: string, payload: Record<string, unknown>) => void
  readonly sendBatch: (frames: readonly DaemonEventFrame[]) => void
  /** Coalescing window in ms. Default: 16 ms (~60 Hz). */
  readonly flushMs?: number
  /** Upper bound on queued frames before forcing an immediate flush. Default: 128. */
  readonly maxBatchSize?: number
  readonly setTimer?: (fn: () => void, ms: number) => ReturnType<typeof setTimeout>
  readonly clearTimer?: (handle: ReturnType<typeof setTimeout>) => void
}

const DEFAULT_FLUSH_MS = 16
const DEFAULT_MAX_BATCH_SIZE = 128

function stringField(payload: Record<string, unknown>, key: string): string {
  const value = payload[key]
  return typeof value === 'string' ? value : ''
}

/** True when two text/think delta payloads share every key and value other than `deltaField`. */
function sameNonTextFields(a: Record<string, unknown>, b: Record<string, unknown>, deltaField: 'text' | 'think'): boolean {
  const aKeys = Object.keys(a)
  const bKeys = Object.keys(b)
  if (aKeys.length !== bKeys.length) return false
  for (const key of aKeys) {
    if (key === deltaField) continue
    if (!Object.prototype.hasOwnProperty.call(b, key)) return false
    if (a[key] !== b[key]) return false
  }
  return true
}

/** Coalesce key for ephemeral telemetry frames that can be safely superseded within one 16 ms window. */
function supersedeKeyOf(type: string, payload: Record<string, unknown>): string | null {
  if (type === 'swarm_event' && payload.kind === 'member_activity') {
    const member = stringField(payload, 'member_name')
    if (!member) return null
    return `swarm_activity:${stringField(payload, 'session_id')}\0${member}`
  }
  if (type === 'status_update' && payload.goal_phase === undefined) {
    return `status:${stringField(payload, 'session_id')}\0${stringField(payload, 'kind')}`
  }
  if (type === 'subagent_event') {
    const kind = stringField(payload, 'kind') || stringField(payload, 'event_type')
    if (kind === 'heartbeat' || kind === 'progress') {
      const agentId = stringField(payload, 'agent_id') || stringField(payload, 'id') || stringField(payload, 'subagent_id')
      if (agentId) return `subagent:${stringField(payload, 'session_id')}\0${agentId}\0${kind}`
    }
  }
  return null
}

function deltaFieldOf(type: string, payload: Record<string, unknown>): 'text' | 'think' | null {
  if (type === 'text_part' && typeof payload.text === 'string') return 'text'
  if (type === 'think_part') {
    if (typeof payload.think === 'string') return 'think'
    if (typeof payload.text === 'string') return 'text'
  }
  return null
}

export class DaemonEventBatcher {
  private readonly sendSingle: DaemonEventBatcherOptions['sendSingle']
  private readonly sendBatch: DaemonEventBatcherOptions['sendBatch']
  private readonly flushMs: number
  private readonly maxBatchSize: number
  private readonly setTimer: (fn: () => void, ms: number) => ReturnType<typeof setTimeout>
  private readonly clearTimer: (handle: ReturnType<typeof setTimeout>) => void

  private queue: Array<{ type: string; payload: Record<string, unknown> }> = []
  private supersedeIndex = new Map<string, number>()
  private timer: ReturnType<typeof setTimeout> | undefined
  private disposed = false

  constructor(options: DaemonEventBatcherOptions) {
    this.sendSingle = options.sendSingle
    this.sendBatch = options.sendBatch
    this.flushMs = Math.max(0, options.flushMs ?? DEFAULT_FLUSH_MS)
    this.maxBatchSize = Math.max(1, options.maxBatchSize ?? DEFAULT_MAX_BATCH_SIZE)
    this.setTimer = options.setTimer ?? ((fn, ms) => {
      const handle = setTimeout(fn, ms)
      handle.unref?.()
      return handle
    })
    this.clearTimer = options.clearTimer ?? (handle => clearTimeout(handle))
  }

  get pendingCount(): number {
    return this.queue.length
  }

  push(type: string, payload: Record<string, unknown>): void {
    if (this.disposed) return

    const deltaField = deltaFieldOf(type, payload)
    if (deltaField !== null) {
      const last = this.queue.at(-1)
      if (last && last.type === type && typeof last.payload[deltaField] === 'string' && sameNonTextFields(last.payload, payload, deltaField)) {
        last.payload = { ...last.payload, [deltaField]: (last.payload[deltaField] as string) + (payload[deltaField] as string) }
        return
      }
      this.enqueue(type, payload)
      return
    }

    const supersedeKey = supersedeKeyOf(type, payload)
    if (supersedeKey !== null) {
      const existingIndex = this.supersedeIndex.get(supersedeKey)
      if (existingIndex !== undefined) {
        const previous = this.queue[existingIndex]
        if (previous) {
          // Preserve a full swarm status snapshot if an earlier queued frame carried one and a compact tick omitted it.
          const merged = type === 'swarm_event' && previous.payload.status !== undefined && payload.status === undefined
            ? { ...payload, status: previous.payload.status }
            : payload
          this.queue[existingIndex] = { type, payload: merged }
          return
        }
      }
      this.supersedeIndex.set(supersedeKey, this.queue.length)
      this.enqueue(type, payload)
      return
    }

    // Immediate/lifecycle/interactive event: preserve strict FIFO order by flushing queued deltas first.
    if (this.queue.length > 0) this.flush()
    if (!this.disposed) this.sendSingle(type, payload)
  }

  flush(): void {
    if (this.timer !== undefined) {
      this.clearTimer(this.timer)
      this.timer = undefined
    }
    if (this.disposed || this.queue.length === 0) {
      this.queue = []
      this.supersedeIndex.clear()
      return
    }
    const batch = this.queue
    this.queue = []
    this.supersedeIndex.clear()
    if (batch.length === 1) {
      const only = batch[0]!
      this.sendSingle(only.type, only.payload)
    } else {
      this.sendBatch(batch)
    }
  }

  dispose(): void {
    this.disposed = true
    if (this.timer !== undefined) {
      this.clearTimer(this.timer)
      this.timer = undefined
    }
    this.queue = []
    this.supersedeIndex.clear()
  }

  private enqueue(type: string, payload: Record<string, unknown>): void {
    this.queue.push({ type, payload })
    if (this.queue.length >= this.maxBatchSize) {
      this.flush()
      return
    }
    if (this.timer === undefined) {
      this.timer = this.setTimer(() => {
        this.timer = undefined
        this.flush()
      }, this.flushMs)
    }
  }
}
