// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Lifecycle states for a provider-backed reinforcement-learning run. */
export enum RLRunStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  SUCCEEDED = 'succeeded',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

const TRANSITIONS: Readonly<Record<RLRunStatus, ReadonlySet<RLRunStatus>>> = Object.freeze({
  [RLRunStatus.PENDING]: new Set([RLRunStatus.RUNNING, RLRunStatus.CANCELLED, RLRunStatus.FAILED]),
  [RLRunStatus.RUNNING]: new Set([RLRunStatus.SUCCEEDED, RLRunStatus.FAILED, RLRunStatus.CANCELLED]),
  [RLRunStatus.SUCCEEDED]: new Set<RLRunStatus>(),
  [RLRunStatus.FAILED]: new Set<RLRunStatus>(),
  [RLRunStatus.CANCELLED]: new Set<RLRunStatus>(),
})

export interface RLRunStateOptions {
  readonly error?: string
  readonly iteration?: number
  readonly loss?: number | null
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly reward?: number | null
  readonly runId: string
  readonly status?: RLRunStatus
  readonly tokensSeen?: number
  readonly wandbUrl?: string
}

export interface RLRunStateUpdate {
  readonly error?: string
  readonly iteration?: number
  readonly loss?: number | null
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly reward?: number | null
  readonly tokensSeen?: number
  readonly wandbUrl?: string
}

export interface RLRunStateRecord {
  readonly error: string
  readonly iteration: number
  readonly loss: number | null
  readonly metadata: Readonly<Record<string, unknown>>
  readonly reward: number | null
  readonly run_id: string
  readonly status: RLRunStatus
  readonly tokens_seen: number
  readonly wandb_url: string
}

/** Immutable observable snapshot for one reinforcement-learning run. */
export class RLRunState {
  readonly error: string
  readonly iteration: number
  readonly loss: number | null
  readonly metadata: Readonly<Record<string, unknown>>
  readonly reward: number | null
  readonly runId: string
  readonly status: RLRunStatus
  readonly tokensSeen: number
  readonly wandbUrl: string

  constructor(options: RLRunStateOptions) {
    this.runId = requiredText(options.runId, 'runId')
    this.status = runStatus(options.status ?? RLRunStatus.PENDING)
    this.iteration = nonNegativeInteger(options.iteration ?? 0, 'iteration')
    this.reward = optionalFiniteNumber(options.reward, 'reward')
    this.loss = optionalFiniteNumber(options.loss, 'loss')
    this.tokensSeen = nonNegativeInteger(options.tokensSeen ?? 0, 'tokensSeen')
    this.wandbUrl = text(options.wandbUrl ?? '', 'wandbUrl')
    this.error = text(options.error ?? '', 'error')
    this.metadata = freezeRecord(options.metadata ?? {})
    Object.freeze(this)
  }

  /** Build a new snapshot with metrics or annotations updated in place. */
  withUpdate(update: RLRunStateUpdate = {}): RLRunState {
    return new RLRunState({
      runId: this.runId,
      status: this.status,
      iteration: update.iteration ?? this.iteration,
      reward: update.reward === undefined ? this.reward : update.reward,
      loss: update.loss === undefined ? this.loss : update.loss,
      tokensSeen: update.tokensSeen ?? this.tokensSeen,
      wandbUrl: update.wandbUrl ?? this.wandbUrl,
      error: update.error ?? this.error,
      metadata: update.metadata ?? this.metadata,
    })
  }

  /** Build a new snapshot with an explicit lifecycle status. */
  withStatus(status: RLRunStatus, update: RLRunStateUpdate = {}): RLRunState {
    return new RLRunState({
      runId: this.runId,
      status,
      iteration: update.iteration ?? this.iteration,
      reward: update.reward === undefined ? this.reward : update.reward,
      loss: update.loss === undefined ? this.loss : update.loss,
      tokensSeen: update.tokensSeen ?? this.tokensSeen,
      wandbUrl: update.wandbUrl ?? this.wandbUrl,
      error: update.error ?? this.error,
      metadata: update.metadata ?? this.metadata,
    })
  }

  /** Render the Python-compatible field names used by daemon status payloads. */
  toRecord(): RLRunStateRecord {
    return Object.freeze({
      run_id: this.runId,
      status: this.status,
      iteration: this.iteration,
      reward: this.reward,
      loss: this.loss,
      tokens_seen: this.tokensSeen,
      wandb_url: this.wandbUrl,
      error: this.error,
      metadata: { ...this.metadata },
    })
  }
}

export type RLRunEventKind = 'created' | 'observed' | 'transitioned' | 'updated'

/** One immutable event in a run's local status ledger. */
export interface RLRunEvent {
  readonly kind: RLRunEventKind
  readonly previousStatus?: RLRunStatus
  readonly runId: string
  readonly state: RLRunState
  readonly timestamp: string
  /** Whether an externally observed status change follows the local transition graph. */
  readonly transitionValid?: boolean
}

/** Injected telemetry boundary for status events. Failures cannot mutate a tracked run. */
export interface RLRunEventSink {
  record(event: RLRunEvent): void
}

export interface RLRunTrackerOptions {
  readonly clock?: () => Date
  readonly eventSink?: RLRunEventSink
}

/**
 * Keeps a local, bounded event ledger around immutable run status snapshots.
 *
 * It only permits declared local transitions. Remote provider observations are
 * recorded verbatim, with transition validity surfaced to the caller rather
 * than rewriting an out-of-order provider response.
 */
export class RLRunTracker {
  private readonly clock: () => Date
  private readonly eventSink: RLRunEventSink | undefined
  private readonly ledger: RLRunEvent[] = []
  private current: RLRunState

  constructor(initial: RLRunState, options: RLRunTrackerOptions = {}) {
    if (!(initial instanceof RLRunState)) throw new TypeError('initial must be an RLRunState')
    this.current = initial
    this.clock = options.clock ?? (() => new Date())
    this.eventSink = options.eventSink
    this.record('created')
  }

  get events(): readonly RLRunEvent[] {
    return this.ledger.map(copyEvent)
  }

  get state(): RLRunState {
    return this.current
  }

  /** Record metrics or metadata without changing lifecycle state. */
  update(update: RLRunStateUpdate = {}): RLRunEvent {
    this.current = this.current.withUpdate(update)
    return this.record('updated')
  }

  /** Advance through a legal local lifecycle transition. */
  transition(status: RLRunStatus, update: RLRunStateUpdate = {}): RLRunEvent {
    const previousStatus = this.current.status
    if (!canTransition(previousStatus, status)) {
      throw new RangeError(`cannot transition RL run ${this.current.runId} from ${previousStatus} to ${status}`)
    }
    this.current = this.current.withStatus(status, update)
    return this.record('transitioned', previousStatus, true)
  }

  /** Record a provider snapshot, including an out-of-order transition when one is returned. */
  observe(state: RLRunState): RLRunEvent {
    if (!(state instanceof RLRunState)) throw new TypeError('state must be an RLRunState')
    if (state.runId !== this.current.runId) throw new RangeError('observed state must use the tracked run id')
    const previousStatus = this.current.status
    this.current = state
    const transitionValid = previousStatus === state.status || canTransition(previousStatus, state.status)
    return this.record('observed', previousStatus, transitionValid)
  }

  private record(
    kind: RLRunEventKind,
    previousStatus?: RLRunStatus,
    transitionValid?: boolean,
  ): RLRunEvent {
    const event = Object.freeze({
      kind,
      runId: this.current.runId,
      state: this.current,
      timestamp: timestamp(this.clock),
      ...(previousStatus === undefined ? {} : { previousStatus }),
      ...(transitionValid === undefined ? {} : { transitionValid }),
    }) satisfies RLRunEvent
    this.ledger.push(event)
    try {
      this.eventSink?.record(copyEvent(event))
    } catch {
      // Telemetry must not change the locally recorded provider state.
    }
    return copyEvent(event)
  }
}

/** Return whether the state machine permits a direct lifecycle move. */
export function canTransition(fromStatus: RLRunStatus, toStatus: RLRunStatus): boolean {
  return TRANSITIONS[runStatus(fromStatus)].has(runStatus(toStatus))
}

function copyEvent(event: RLRunEvent): RLRunEvent {
  return {
    kind: event.kind,
    runId: event.runId,
    state: event.state,
    timestamp: event.timestamp,
    ...(event.previousStatus === undefined ? {} : { previousStatus: event.previousStatus }),
    ...(event.transitionValid === undefined ? {} : { transitionValid: event.transitionValid }),
  }
}

function freezeRecord(record: Readonly<Record<string, unknown>>): Readonly<Record<string, unknown>> {
  return Object.freeze({ ...record })
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) throw new TypeError(`${name} must be a non-negative integer`)
  return value
}

function optionalFiniteNumber(value: number | null | undefined, name: string): number | null {
  if (value === undefined || value === null) return null
  if (!Number.isFinite(value)) throw new TypeError(`${name} must be finite or null`)
  return value
}

function requiredText(value: string, name: string): string {
  const normalized = text(value, name).trim()
  if (!normalized) throw new TypeError(`${name} must be non-empty`)
  return normalized
}

function runStatus(value: RLRunStatus): RLRunStatus {
  if (!Object.values(RLRunStatus).includes(value)) throw new TypeError(`unknown RL run status: ${String(value)}`)
  return value
}

function text(value: string, name: string): string {
  if (typeof value !== 'string') throw new TypeError(`${name} must be a string`)
  return value
}

function timestamp(clock: () => Date): string {
  const value = clock()
  if (!(value instanceof Date) || Number.isNaN(value.getTime())) {
    throw new TypeError('RL run clock must return a valid Date')
  }
  return value.toISOString()
}
