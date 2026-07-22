// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  type AuditEvent,
  ErrorEvent,
  SkillUsedEvent,
  ToolCallAttemptEvent,
  ToolCallCompleteEvent,
  ToolCallFailureEvent,
  TurnEndEvent,
  TurnStartEvent,
} from './events.js'
import type { AuditCollector } from './collector.js'

/** Scalar values accepted by the portion of the OpenTelemetry API used by this adapter. */
export type OTelAttributeValue = boolean | number | string
export type OTelAttributes = Readonly<Record<string, OTelAttributeValue>>

/** Structural subset of an OpenTelemetry span, keeping the runtime free of a hard package dependency. */
export interface OTelSpan {
  addEvent(name: string, attributes?: OTelAttributes): void
  end(): void
  setAttribute(key: string, value: OTelAttributeValue): void
}

/** Structural subset of an OpenTelemetry tracer accepted through dependency injection. */
export interface OTelTracer {
  startSpan(name: string, options?: Readonly<{ attributes?: OTelAttributes }>): OTelSpan
}

export interface OTelCollectorOptions {
  readonly serviceName?: string
  /** An OpenTelemetry tracer supplied by the application, if OpenTelemetry is installed. */
  readonly tracer?: OTelTracer
  /** Maximum turn spans held open at once; the oldest is ended and evicted beyond this. */
  readonly maxOpenTurns?: number
  /** Maximum retained no-op fallback entries; the oldest are dropped beyond this. */
  readonly maxFallbackEntries?: number
}

/** A recorded no-op span/event used when no tracer has been injected. */
export interface OTelFallbackEntry {
  readonly attributes: OTelAttributes
  readonly name: string
}

/**
 * Turns audit records into OpenTelemetry spans/events when supplied a tracer.
 *
 * The collector deliberately does not import OpenTelemetry itself. Applications can inject
 * their configured tracer, while Bun-only deployments retain an inspectable no-op log.
 */
export class OTelCollector implements AuditCollector {
  static readonly DEFAULT_MAX_OPEN_TURNS = 1_000
  static readonly DEFAULT_MAX_FALLBACK_ENTRIES = 10_000

  private readonly noopLog: OTelFallbackEntry[] = []
  private readonly openTurnSpans = new Map<string, OTelSpan>()
  private readonly maxOpenTurns: number
  private readonly maxFallbackEntries: number
  private evictedTurns = 0
  private evictedFallbacks = 0
  readonly serviceName: string
  readonly tracer: OTelTracer | undefined

  constructor(options?: OTelCollectorOptions)
  constructor(serviceName?: string, tracer?: OTelTracer)
  constructor(optionsOrServiceName: OTelCollectorOptions | string = {}, tracer?: OTelTracer) {
    if (typeof optionsOrServiceName === 'string') {
      this.serviceName = optionsOrServiceName
      this.tracer = tracer
      this.maxOpenTurns = OTelCollector.DEFAULT_MAX_OPEN_TURNS
      this.maxFallbackEntries = OTelCollector.DEFAULT_MAX_FALLBACK_ENTRIES
      return
    }
    this.serviceName = optionsOrServiceName.serviceName ?? 'xerxes'
    this.tracer = optionsOrServiceName.tracer
    this.maxOpenTurns = boundedSize(optionsOrServiceName.maxOpenTurns, OTelCollector.DEFAULT_MAX_OPEN_TURNS, 'maxOpenTurns')
    this.maxFallbackEntries = boundedSize(optionsOrServiceName.maxFallbackEntries, OTelCollector.DEFAULT_MAX_FALLBACK_ENTRIES, 'maxFallbackEntries')
  }

  emit(event: AuditEvent): void {
    try {
      if (event instanceof TurnStartEvent) {
        this.onTurnStart(event)
        return
      }
      if (event instanceof TurnEndEvent) {
        this.onTurnEnd(event)
        return
      }
      if (event instanceof ToolCallAttemptEvent) {
        this.recordEvent(`tool.attempt:${event.toolName}`, event)
        return
      }
      if (event instanceof ToolCallCompleteEvent) {
        this.recordEvent(`tool.complete:${event.toolName}`, event)
        return
      }
      if (event instanceof ToolCallFailureEvent) {
        this.recordEvent(`tool.failure:${event.toolName}`, event)
        return
      }
      if (event instanceof SkillUsedEvent) {
        this.recordEvent(`skill.used:${event.skillName}`, event)
        return
      }
      if (event instanceof ErrorEvent) {
        this.recordEvent(`error:${event.errorType}`, event)
        return
      }
      this.recordEvent(event.eventType, event)
    } catch {
      // Audit exports must never prevent agent work. Keep enough context for local diagnostics.
      this.recordFallback('audit.export_error', cleanOtelAttributes(event.toRecord()))
    }
  }

  flush(): void {
    for (const span of this.openTurnSpans.values()) {
      try {
        span.end()
      } catch {
        // Best effort shutdown mirrors the behavior of external telemetry SDKs.
      }
    }
    this.openTurnSpans.clear()
  }

  get hasOtel(): boolean {
    return this.tracer !== undefined
  }

  /** A copy of no-op entries, so callers cannot mutate the collector's diagnostics. */
  get fallbackLog(): OTelFallbackEntry[] {
    return this.noopLog.map(entry => ({ name: entry.name, attributes: { ...entry.attributes } }))
  }

  get openTurnCount(): number {
    return this.openTurnSpans.size
  }

  /** Turn spans evicted because cancelled turns never produced a TurnEnd event. */
  get evictedTurnCount(): number {
    return this.evictedTurns
  }

  /** No-op fallback entries dropped to keep the diagnostic log bounded. */
  get evictedFallbackCount(): number {
    return this.evictedFallbacks
  }

  private onTurnStart(event: TurnStartEvent): void {
    const attributes = cleanOtelAttributes({
      'xerxes.turn_id': event.turnId,
      'xerxes.agent_id': event.agentId,
      'xerxes.session_id': event.sessionId,
      'xerxes.prompt_preview': event.promptPreview,
      'service.name': this.serviceName,
    })
    if (this.tracer === undefined || event.turnId === undefined) {
      this.recordFallback('turn', attributes)
      return
    }
    // Refresh recency so long-lived turns are not evicted ahead of stale ones.
    this.openTurnSpans.delete(event.turnId)
    while (this.openTurnSpans.size >= this.maxOpenTurns) {
      const oldest = this.openTurnSpans.keys().next().value
      if (oldest === undefined) break
      const stale = this.openTurnSpans.get(oldest)
      this.openTurnSpans.delete(oldest)
      this.evictedTurns += 1
      try {
        stale?.end()
      } catch {
        // Ending an evicted span is best effort.
      }
    }
    this.openTurnSpans.set(event.turnId, this.tracer.startSpan('xerxes.turn', { attributes }))
  }

  private onTurnEnd(event: TurnEndEvent): void {
    if (event.turnId === undefined) return
    const span = this.openTurnSpans.get(event.turnId)
    if (span === undefined) return
    this.openTurnSpans.delete(event.turnId)
    try {
      span.setAttribute('xerxes.function_calls_count', event.functionCallsCount)
      span.end()
    } catch {
      // Do not leak a telemetry failure into the agent loop.
    }
  }

  private recordEvent(name: string, event: AuditEvent): void {
    const attributes = cleanOtelAttributes(event.toRecord())
    const turnId = event.turnId
    const turnSpan = turnId === undefined ? undefined : this.openTurnSpans.get(turnId)
    if (this.tracer !== undefined && turnSpan !== undefined) {
      turnSpan.addEvent(name, attributes)
      return
    }
    if (this.tracer === undefined) {
      this.recordFallback(name, attributes)
      return
    }
    const span = this.tracer.startSpan(name, { attributes })
    try {
      // The short span itself carries all attributes.
    } finally {
      span.end()
    }
  }

  private recordFallback(name: string, attributes: OTelAttributes): void {
    this.noopLog.push({ name, attributes: { ...attributes } })
    const overflow = this.noopLog.length - this.maxFallbackEntries
    if (overflow > 0) {
      this.noopLog.splice(0, overflow)
      this.evictedFallbacks += overflow
    }
  }
}

function boundedSize(value: number | undefined, fallback: number, name: string): number {
  if (value === undefined) return fallback
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`OTelCollector ${name} must be a positive integer`)
  }
  return value
}

/** Coerce a JSON record into OTel-safe scalar attributes and remove absent values. */
export function cleanOtelAttributes(values: Readonly<Record<string, unknown>>): OTelAttributes {
  const attributes: Record<string, OTelAttributeValue> = {}
  for (const [key, value] of Object.entries(values)) {
    if (value === undefined || value === null) continue
    if (typeof value === 'string' || typeof value === 'boolean') {
      attributes[key] = value
      continue
    }
    if (typeof value === 'number') {
      attributes[key] = Number.isFinite(value) ? value : String(value)
      continue
    }
    if (value instanceof Date) {
      attributes[key] = value.toISOString()
      continue
    }
    attributes[key] = stringifyAttribute(value)
  }
  return attributes
}

function stringifyAttribute(value: unknown): string {
  try {
    const rendered = JSON.stringify(value)
    if (rendered !== undefined) return rendered.slice(0, 200)
  } catch {
    // Fall through to String for cyclic or host objects.
  }
  return String(value).slice(0, 200)
}
