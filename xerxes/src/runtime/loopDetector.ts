// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { JsonObject } from '../types/toolCalls.js'

export type LoopSeverity = 'critical' | 'ok' | 'warning'

export interface LoopDetectionConfig {
  readonly enabled: boolean
  /** Optional absolute ceiling. Xerxes leaves distinct tool calls unbounded by default. */
  readonly maxToolCallsPerTurn?: number
  readonly pingPongCritical: number
  readonly pingPongWarning: number
  readonly sameCallCritical: number
  readonly sameCallWarning: number
}

export interface LoopEvent {
  readonly callCount: number
  readonly details: string
  readonly pattern: 'disabled' | 'max_calls' | 'none' | 'pingpong' | 'same_call'
  readonly severity: LoopSeverity
  readonly toolName: string
}

interface CallRecord {
  readonly argumentsKey: string
  readonly toolName: string
}

export const DEFAULT_LOOP_DETECTION_CONFIG: LoopDetectionConfig = {
  enabled: true,
  pingPongCritical: 6,
  pingPongWarning: 4,
  sameCallCritical: 5,
  sameCallWarning: 3,
}

export class ToolLoopError extends Error {
  readonly event: LoopEvent

  constructor(event: LoopEvent) {
    super(`Tool loop detected (${event.pattern}): ${event.details}`)
    this.name = 'ToolLoopError'
    this.event = event
  }
}

/** Constant-space per-turn repetition detector with an optional caller-owned absolute ceiling. */
export class LoopDetector {
  private alternations = 0
  private readonly config: LoopDetectionConfig
  private lastRecord: CallRecord | undefined
  private readonly listeners = new Set<(event: LoopEvent) => void>()
  private readonly recentToolNames: string[] = []
  private sameCallStreak = 0
  private totalCalls = 0

  constructor(config: Partial<LoopDetectionConfig> = {}) {
    this.config = { ...DEFAULT_LOOP_DETECTION_CONFIG, ...config }
  }

  get callCount(): number {
    return this.totalCalls
  }

  addListener(listener: (event: LoopEvent) => void): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  reset(): void {
    this.alternations = 0
    this.lastRecord = undefined
    this.recentToolNames.length = 0
    this.sameCallStreak = 0
    this.totalCalls = 0
  }

  recordCall(toolName: string, argumentsValue: JsonObject | string | undefined = undefined): LoopEvent {
    if (!this.config.enabled) {
      return { severity: 'ok', pattern: 'disabled', toolName, details: '', callCount: 0 }
    }

    const record: CallRecord = { toolName, argumentsKey: stableArguments(argumentsValue) }
    this.totalCalls += 1
    this.sameCallStreak = isSameCall(this.lastRecord, record) ? this.sameCallStreak + 1 : 1
    const previousToolName = this.recentToolNames.at(-1)
    this.alternations = previousToolName !== undefined && previousToolName !== toolName
      ? this.alternations + 1
      : 0
    this.recentToolNames.push(toolName)
    if (this.recentToolNames.length > 4) this.recentToolNames.shift()
    this.lastRecord = record
    let event: LoopEvent
    const maxToolCalls = this.config.maxToolCallsPerTurn
    if (maxToolCalls !== undefined && this.totalCalls > maxToolCalls) {
      event = {
        severity: 'critical',
        pattern: 'max_calls',
        toolName,
        details: `Reached max tool calls per turn (${maxToolCalls})`,
        callCount: this.totalCalls,
      }
    } else {
      event = this.sameCall(record)
      if (event.severity === 'ok') {
        event = this.pingPong()
      }
    }
    if (event.severity !== 'ok') {
      for (const listener of this.listeners) {
        listener(event)
      }
    }
    return event
  }

  private sameCall(current: CallRecord): LoopEvent {
    const count = this.sameCallStreak
    if (count >= this.config.sameCallCritical) {
      return { severity: 'critical', pattern: 'same_call', toolName: current.toolName, details: `Same tool+args called ${count} times consecutively`, callCount: count }
    }
    if (count >= this.config.sameCallWarning) {
      return { severity: 'warning', pattern: 'same_call', toolName: current.toolName, details: `Same tool+args called ${count} times consecutively`, callCount: count }
    }
    return { severity: 'ok', pattern: 'none', toolName: current.toolName, details: '', callCount: count }
  }

  private pingPong(): LoopEvent {
    if (this.totalCalls < 4) {
      return { severity: 'ok', pattern: 'none', toolName: '', details: '', callCount: 0 }
    }
    if (new Set(this.recentToolNames).size > 2) {
      return { severity: 'ok', pattern: 'none', toolName: '', details: '', callCount: 0 }
    }
    const alternations = this.alternations
    const toolName = this.recentToolNames.at(-1) ?? ''
    if (alternations >= this.config.pingPongCritical) {
      return { severity: 'critical', pattern: 'pingpong', toolName, details: `Ping-pong pattern detected (${alternations} alternations)`, callCount: alternations }
    }
    if (alternations >= this.config.pingPongWarning) {
      return { severity: 'warning', pattern: 'pingpong', toolName, details: `Ping-pong pattern detected (${alternations} alternations)`, callCount: alternations }
    }
    return { severity: 'ok', pattern: 'none', toolName, details: '', callCount: alternations }
  }
}

function isSameCall(previous: CallRecord | undefined, current: CallRecord): boolean {
  return previous?.toolName === current.toolName && previous.argumentsKey === current.argumentsKey
}

function stableArguments(value: JsonObject | string | undefined): string {
  if (value === undefined) {
    return 'empty'
  }
  if (typeof value === 'string') {
    return value
  }
  return JSON.stringify(sortValue(value))
}

function sortValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(sortValue)
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>).sort(([left], [right]) => left.localeCompare(right)).map(([key, item]) => [key, sortValue(item)]))
  }
  return value
}
