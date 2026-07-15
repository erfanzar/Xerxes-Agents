// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export type ToolArguments = Readonly<Record<string, unknown>>

/** One observed tool invocation from an agent turn. */
export interface ToolCallEvent {
  readonly arguments: ToolArguments
  readonly durationMs: number
  readonly errorMessage?: string
  readonly errorType?: string
  readonly retryOf?: number
  readonly status: string
  readonly timestamp: number
  readonly toolName: string
}

export interface ToolCallInput {
  readonly arguments?: ToolArguments
  readonly durationMs?: number
  readonly errorMessage?: string
  readonly errorType?: string
  readonly status?: string
  readonly toolName: string
}

export interface SkillCandidateOptions {
  readonly agentId?: string
  readonly completedAt?: number
  readonly events?: readonly ToolCallEvent[]
  readonly finalResponse?: string
  readonly turnId?: string
  readonly userPrompt?: string
}

/**
 * Immutable observation snapshot used for triggering, drafting, and verification.
 *
 * It records an observed procedure but does not claim that a new skill is
 * valid or persisted.
 */
export class SkillCandidate {
  readonly agentId: string | undefined
  readonly completedAt: number
  readonly events: readonly ToolCallEvent[]
  readonly finalResponse: string
  readonly turnId: string | undefined
  readonly userPrompt: string

  constructor(options: SkillCandidateOptions = {}) {
    this.agentId = options.agentId
    this.completedAt = options.completedAt ?? Date.now()
    this.events = Object.freeze((options.events ?? []).map(copyEvent))
    this.finalResponse = options.finalResponse ?? ''
    this.turnId = options.turnId
    this.userPrompt = options.userPrompt ?? ''
  }

  get retries(): number {
    return this.events.filter(event => event.retryOf !== undefined).length
  }

  get successfulEvents(): readonly ToolCallEvent[] {
    return this.events.filter(event => event.status === 'success')
  }

  get totalDurationMs(): number {
    return this.events.reduce((total, event) => total + event.durationMs, 0)
  }

  get uniqueTools(): readonly string[] {
    const names = new Set<string>()
    const tools: string[] = []
    for (const event of this.events) {
      if (!names.has(event.toolName)) {
        names.add(event.toolName)
        tools.push(event.toolName)
      }
    }
    return tools
  }

  /** Return the observed complete call sequence, including retries and failures. */
  signature(): string {
    return this.events.map(event => event.toolName).join('>')
  }
}

export interface ToolSequenceTrackerOptions {
  /** Wall-clock time stored on events and completed candidates. */
  readonly now?: () => number
  /** Monotonic time used only for automatically measured durations. */
  readonly monotonicNow?: () => number
}

/**
 * Tracks one turn at a time and detects retries by a stable tool-and-arguments signature.
 *
 * End turn returns an immutable snapshot and resets the tracker, preventing
 * events from one turn from leaking into the next proposal.
 */
export class ToolSequenceTracker {
  private agentId: string | undefined
  private callStartedAt: number | undefined
  private readonly events: ToolCallEvent[] = []
  private readonly now: () => number
  private readonly monotonicNow: () => number
  private readonly signatures = new Map<string, number>()
  private turnId: string | undefined
  private userPrompt = ''

  constructor(options: ToolSequenceTrackerOptions = {}) {
    this.now = options.now ?? Date.now
    this.monotonicNow = options.monotonicNow ?? (() => performance.now())
  }

  get callCount(): number {
    return this.events.length
  }

  /** Return a defensive snapshot of events recorded in the active turn. */
  get observedEvents(): readonly ToolCallEvent[] {
    return this.events.map(copyEvent)
  }

  beginTurn(options: { readonly agentId?: string; readonly turnId?: string; readonly userPrompt?: string } = {}): void {
    this.events.length = 0
    this.signatures.clear()
    this.agentId = options.agentId
    this.turnId = options.turnId
    this.userPrompt = options.userPrompt ?? ''
    this.callStartedAt = undefined
  }

  /** Start elapsed-time measurement for the next call without changing the observation sequence. */
  markCallStart(): void {
    this.callStartedAt = this.monotonicNow()
  }

  /** Record one completed tool call and return an immutable event snapshot. */
  recordCall(input: ToolCallInput): ToolCallEvent {
    const toolName = input.toolName.trim()
    if (!toolName) {
      throw new TypeError('toolName must not be empty')
    }
    const arguments_ = copyArguments(input.arguments ?? {})
    const signature = toolName + '::' + stableValueKey(arguments_)
    const retryOf = this.signatures.get(signature)
    this.signatures.set(signature, this.events.length)
    const durationMs = this.resolveDuration(input.durationMs)
    const event: ToolCallEvent = Object.freeze({
      toolName,
      arguments: arguments_,
      status: input.status ?? 'success',
      durationMs,
      timestamp: this.now(),
      ...(input.errorType === undefined ? {} : { errorType: input.errorType }),
      ...(input.errorMessage === undefined ? {} : { errorMessage: input.errorMessage }),
      ...(retryOf === undefined ? {} : { retryOf }),
    })
    this.events.push(event)
    return copyEvent(event)
  }

  /** Finalize and reset the current turn. */
  endTurn(finalResponse = ''): SkillCandidate {
    const candidate = new SkillCandidate({
      ...(this.agentId === undefined ? {} : { agentId: this.agentId }),
      ...(this.turnId === undefined ? {} : { turnId: this.turnId }),
      userPrompt: this.userPrompt,
      finalResponse,
      events: this.events,
      completedAt: this.now(),
    })
    this.beginTurn()
    return candidate
  }

  private resolveDuration(explicit: number | undefined): number {
    if (explicit !== undefined) {
      if (!Number.isFinite(explicit) || explicit < 0) {
        throw new RangeError('durationMs must be a finite non-negative number')
      }
      return explicit
    }
    if (this.callStartedAt === undefined) {
      return 0
    }
    const duration = Math.max(0, this.monotonicNow() - this.callStartedAt)
    this.callStartedAt = undefined
    return duration
  }
}

function copyArguments(arguments_: ToolArguments): ToolArguments {
  return Object.freeze({ ...arguments_ })
}

function copyEvent(event: ToolCallEvent): ToolCallEvent {
  return Object.freeze({
    toolName: event.toolName,
    arguments: copyArguments(event.arguments),
    status: event.status,
    durationMs: event.durationMs,
    timestamp: event.timestamp,
    ...(event.errorType === undefined ? {} : { errorType: event.errorType }),
    ...(event.errorMessage === undefined ? {} : { errorMessage: event.errorMessage }),
    ...(event.retryOf === undefined ? {} : { retryOf: event.retryOf }),
  })
}

function stableValueKey(value: unknown, ancestors = new Set<object>()): string {
  if (value === null) return 'null'
  if (typeof value === 'boolean' || typeof value === 'number' || typeof value === 'string') {
    return typeof value + ':' + JSON.stringify(value)
  }
  if (typeof value === 'undefined') return 'undefined'
  if (Array.isArray(value)) {
    if (ancestors.has(value)) {
      throw new TypeError('tool call arguments must not contain circular values')
    }
    ancestors.add(value)
    const result = '[' + value.map(item => stableValueKey(item, ancestors)).join(',') + ']'
    ancestors.delete(value)
    return result
  }
  if (typeof value === 'object') {
    if (ancestors.has(value)) {
      throw new TypeError('tool call arguments must not contain circular values')
    }
    ancestors.add(value)
    const record = value as Record<string, unknown>
    const result = '{' + Object.keys(record)
      .sort()
      .map(key => JSON.stringify(key) + ':' + stableValueKey(record[key], ancestors))
      .join(',') + '}'
    ancestors.delete(value)
    return result
  }
  return typeof value + ':' + String(value)
}
