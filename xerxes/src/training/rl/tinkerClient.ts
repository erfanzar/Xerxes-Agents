// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  RLRunState,
  RLRunStatus,
  RLRunTracker,
} from './status.js'
import type {
  RLRunEvent,
  RLRunEventSink,
  RLRunTrackerOptions,
} from './status.js'

type MaybePromise<Value> = Promise<Value> | Value

export interface TinkerRunConfigOptions {
  readonly batchSize?: number
  readonly env: string
  readonly extra?: Readonly<Record<string, unknown>>
  readonly learningRate?: number
  readonly model: string
  readonly steps?: number
}

export interface TinkerRunPayload extends Readonly<Record<string, unknown>> {
  readonly batch_size: number
  readonly env: string
  readonly learning_rate: number
  readonly model: string
  readonly steps: number
}

/** Immutable hyperparameters for one externally orchestrated Tinker run. */
export class TinkerRunConfig {
  readonly batchSize: number
  readonly env: string
  readonly extra: Readonly<Record<string, unknown>>
  readonly learningRate: number
  readonly model: string
  readonly steps: number

  constructor(options: TinkerRunConfigOptions) {
    this.model = requiredText(options.model, 'model')
    this.env = requiredText(options.env, 'env')
    this.learningRate = positiveFinite(options.learningRate ?? 0.00001, 'learningRate')
    this.batchSize = positiveInteger(options.batchSize ?? 8, 'batchSize')
    this.steps = positiveInteger(options.steps ?? 100, 'steps')
    this.extra = Object.freeze({ ...(options.extra ?? {}) })
    Object.freeze(this)
  }

  /** Render the provider payload; `extra` intentionally has the final override position. */
  toPayload(): TinkerRunPayload {
    return Object.freeze({
      model: this.model,
      env: this.env,
      learning_rate: this.learningRate,
      batch_size: this.batchSize,
      steps: this.steps,
      ...this.extra,
    }) as TinkerRunPayload
  }
}

export interface TinkerStartResponse {
  readonly id: string
}

/**
 * Explicit hosted-training boundary. The runtime never imports a vendor SDK,
 * looks up credentials, or assumes that a resolved promise means model output.
 */
export interface TinkerTransport {
  cancelRun(runId: string): MaybePromise<boolean | void>
  createRun(payload: TinkerRunPayload): MaybePromise<string | TinkerStartResponse>
  getRun(runId: string): MaybePromise<unknown>
}

export interface TinkerClientOptions {
  readonly clock?: () => Date
  readonly eventSink?: RLRunEventSink
  readonly transport?: TinkerTransport
}

/** Raised when a caller attempts submission without a host-supplied Tinker transport. */
export class TinkerTransportUnavailableError extends Error {
  constructor() {
    super('Tinker transport not configured')
    this.name = 'TinkerTransportUnavailableError'
  }
}

/** Raised when a host transport returns a response that cannot identify a run. */
export class TinkerProtocolError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'TinkerProtocolError'
  }
}

/**
 * Typed Tinker service adapter driven entirely by an injected host transport.
 *
 * No default SDK adapter is provided: application code owns credentials and
 * networking, while this layer validates payloads and keeps local status
 * histories for daemon/UI consumers.
 */
export class TinkerClient {
  private readonly trackerOptions: RLRunTrackerOptions
  private readonly trackers = new Map<string, RLRunTracker>()
  readonly transport: TinkerTransport | undefined

  constructor(options: TinkerClientOptions = {}) {
    this.transport = options.transport
    this.trackerOptions = {
      ...(options.clock === undefined ? {} : { clock: options.clock }),
      ...(options.eventSink === undefined ? {} : { eventSink: options.eventSink }),
    }
  }

  /** Submit a real remote run through the supplied transport and return its provider-assigned id. */
  async start(config: TinkerRunConfig): Promise<string> {
    if (!(config instanceof TinkerRunConfig)) throw new TypeError('config must be a TinkerRunConfig')
    const transport = this.requireTransport()
    const runId = startRunId(await transport.createRun(config.toPayload()))
    this.tracker(runId)
    return runId
  }

  /** Fetch, normalize, and record the latest provider status snapshot. */
  async status(runId: string): Promise<RLRunState> {
    const normalizedRunId = requiredText(runId, 'runId')
    const transport = this.transport
    if (transport === undefined) {
      return new RLRunState({
        runId: normalizedRunId,
        status: RLRunStatus.FAILED,
        error: 'Tinker transport not configured',
      })
    }
    const state = tinkerRunState(normalizedRunId, await transport.getRun(normalizedRunId))
    this.tracker(normalizedRunId).observe(state)
    return state
  }

  /** Cancel a remote run. A transport result of `false` is the only failed cancellation acknowledgement. */
  async cancel(runId: string): Promise<boolean> {
    const normalizedRunId = requiredText(runId, 'runId')
    const transport = this.transport
    if (transport === undefined) return false
    const result = await transport.cancelRun(normalizedRunId)
    if (result === false) return false
    const tracker = this.tracker(normalizedRunId)
    const cancelled = new RLRunState({ runId: normalizedRunId, status: RLRunStatus.CANCELLED })
    if (tracker.state.status === RLRunStatus.PENDING || tracker.state.status === RLRunStatus.RUNNING) {
      tracker.transition(RLRunStatus.CANCELLED)
    } else {
      tracker.observe(cancelled)
    }
    return true
  }

  /** Return the latest local status snapshot if this process has observed the run. */
  trackedState(runId: string): RLRunState | undefined {
    return this.trackers.get(runId)?.state
  }

  /** Return local lifecycle and provider-observation events for one known run. */
  events(runId: string): readonly RLRunEvent[] {
    return this.trackers.get(runId)?.events ?? []
  }

  private requireTransport(): TinkerTransport {
    if (this.transport === undefined) throw new TinkerTransportUnavailableError()
    return this.transport
  }

  private tracker(runId: string): RLRunTracker {
    const existing = this.trackers.get(runId)
    if (existing !== undefined) return existing
    const tracker = new RLRunTracker(new RLRunState({ runId }), this.trackerOptions)
    this.trackers.set(runId, tracker)
    return tracker
  }
}

/** Map the hosted service's vendor spellings into Xerxes' stable status vocabulary. */
export function mapTinkerStatus(value: unknown): RLRunStatus {
  const status = typeof value === 'string' ? value.trim().toLowerCase() : ''
  if (status === 'pending' || status === 'queued') return RLRunStatus.PENDING
  if (status === 'running' || status === 'active') return RLRunStatus.RUNNING
  if (status === 'succeeded' || status === 'completed' || status === 'success') return RLRunStatus.SUCCEEDED
  if (status === 'cancelled' || status === 'canceled') return RLRunStatus.CANCELLED
  return RLRunStatus.FAILED
}

/** Translate a raw provider status response without treating malformed values as success. */
export function tinkerRunState(runId: string, raw: unknown): RLRunState {
  const record = objectRecord(raw, 'Tinker status response')
  return new RLRunState({
    runId: requiredText(runId, 'runId'),
    status: mapTinkerStatus(record.status),
    iteration: nonNegativeInteger(recordValue(record, 'iteration') ?? 0, 'iteration'),
    reward: nullableFinite(recordValue(record, 'reward'), 'reward'),
    loss: nullableFinite(recordValue(record, 'loss'), 'loss'),
    tokensSeen: nonNegativeInteger(recordValue(record, 'tokens_seen', 'tokensSeen') ?? 0, 'tokensSeen'),
    wandbUrl: stringValue(recordValue(record, 'wandb_url', 'wandbUrl')),
    error: stringValue(record.error),
    metadata: metadataValue(record.metadata),
  })
}

function metadataValue(value: unknown): Readonly<Record<string, unknown>> {
  return isRecord(value) ? value : {}
}

function nullableFinite(value: unknown, name: string): number | null {
  if (value === undefined || value === null) return null
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TinkerProtocolError(`${name} must be finite or null`)
  }
  return value
}

function nonNegativeInteger(value: unknown, name: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new TinkerProtocolError(`${name} must be a non-negative integer`)
  }
  return value
}

function objectRecord(value: unknown, name: string): Readonly<Record<string, unknown>> {
  if (!isRecord(value)) throw new TinkerProtocolError(`${name} must be an object`)
  return value
}

function recordValue(record: Readonly<Record<string, unknown>>, ...names: readonly string[]): unknown {
  for (const name of names) {
    const value = record[name]
    if (value !== undefined) return value
  }
  return undefined
}

function startRunId(value: string | TinkerStartResponse): string {
  if (typeof value === 'string') return requiredText(value, 'Tinker start response')
  if (isRecord(value) && typeof value.id === 'string') return requiredText(value.id, 'Tinker start response id')
  throw new TinkerProtocolError('Tinker start response must include a non-empty id')
}

function positiveFinite(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0) throw new TypeError(`${name} must be a positive finite number`)
  return value
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 1) throw new TypeError(`${name} must be a positive integer`)
  return value
}

function requiredText(value: string, name: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new TypeError(`${name} must be non-empty`)
  return value.trim()
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
