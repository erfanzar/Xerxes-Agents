// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { JsonObject } from '../types/toolCalls.js'

export type LoopSeverity = 'critical' | 'ok' | 'warning'

export interface LoopDetectionConfig {
  readonly enabled: boolean
  readonly maxToolCallsPerTurn: number
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
  maxToolCallsPerTurn: 25,
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

/** Per-turn tool loop detector with the same thresholds as the Python runtime. */
export class LoopDetector {
  private readonly config: LoopDetectionConfig
  private readonly history: CallRecord[] = []
  private readonly listeners = new Set<(event: LoopEvent) => void>()

  constructor(config: Partial<LoopDetectionConfig> = {}) {
    this.config = { ...DEFAULT_LOOP_DETECTION_CONFIG, ...config }
  }

  get callCount(): number {
    return this.history.length
  }

  addListener(listener: (event: LoopEvent) => void): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  reset(): void {
    this.history.length = 0
  }

  recordCall(toolName: string, argumentsValue: JsonObject | string | undefined = undefined): LoopEvent {
    if (!this.config.enabled) {
      return { severity: 'ok', pattern: 'disabled', toolName, details: '', callCount: 0 }
    }

    const record: CallRecord = { toolName, argumentsKey: stableArguments(argumentsValue) }
    this.history.push(record)
    let event: LoopEvent
    if (this.history.length >= this.config.maxToolCallsPerTurn) {
      event = {
        severity: 'critical',
        pattern: 'max_calls',
        toolName,
        details: `Reached max tool calls per turn (${this.config.maxToolCallsPerTurn})`,
        callCount: this.history.length,
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
    let count = 0
    for (let index = this.history.length - 1; index >= 0; index -= 1) {
      const record = this.history[index]
      if (record?.toolName === current.toolName && record.argumentsKey === current.argumentsKey) {
        count += 1
      } else {
        break
      }
    }
    if (count >= this.config.sameCallCritical) {
      return { severity: 'critical', pattern: 'same_call', toolName: current.toolName, details: `Same tool+args called ${count} times consecutively`, callCount: count }
    }
    if (count >= this.config.sameCallWarning) {
      return { severity: 'warning', pattern: 'same_call', toolName: current.toolName, details: `Same tool+args called ${count} times consecutively`, callCount: count }
    }
    return { severity: 'ok', pattern: 'none', toolName: current.toolName, details: '', callCount: count }
  }

  private pingPong(): LoopEvent {
    if (this.history.length < 4) {
      return { severity: 'ok', pattern: 'none', toolName: '', details: '', callCount: 0 }
    }
    const names = this.history.map(record => record.toolName)
    if (new Set(names.slice(-4)).size > 2) {
      return { severity: 'ok', pattern: 'none', toolName: '', details: '', callCount: 0 }
    }
    let alternations = 0
    for (let index = names.length - 1; index > 0; index -= 1) {
      if (names[index] !== names[index - 1]) {
        alternations += 1
      } else {
        break
      }
    }
    const toolName = names.at(-1) ?? ''
    if (alternations >= this.config.pingPongCritical) {
      return { severity: 'critical', pattern: 'pingpong', toolName, details: `Ping-pong pattern detected (${alternations} alternations)`, callCount: alternations }
    }
    if (alternations >= this.config.pingPongWarning) {
      return { severity: 'warning', pattern: 'pingpong', toolName, details: `Ping-pong pattern detected (${alternations} alternations)`, callCount: alternations }
    }
    return { severity: 'ok', pattern: 'none', toolName, details: '', callCount: alternations }
  }
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
