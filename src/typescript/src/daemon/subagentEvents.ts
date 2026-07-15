// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { DaemonEvent } from './runtime.js'

export type DaemonSubagentEventListener = (event: DaemonEvent) => void

const MAX_BUFFERED_EVENTS_PER_SESSION = 256
const MAX_BUFFERED_SESSIONS = 64

/**
 * Session-scoped live event bus for delegated turns.
 *
 * Subagents can outlive the tool call that started them, so their renderer
 * events cannot be returned as one tool result. The daemon owns this small
 * fan-out boundary and the active turn runner subscribes only to its session.
 */
export class DaemonSubagentEventBus {
  private readonly buffered = new Map<string, DaemonEvent[]>()
  private readonly listeners = new Map<string, Set<DaemonSubagentEventListener>>()

  publish(sessionId: string, event: DaemonEvent): void {
    const id = sessionId.trim()
    if (!id) return
    const listeners = this.listeners.get(id)
    if (listeners?.size) {
      for (const listener of listeners) listener(event)
      return
    }
    if (!this.buffered.has(id) && this.buffered.size >= MAX_BUFFERED_SESSIONS) {
      const oldestSession = this.buffered.keys().next().value
      if (oldestSession !== undefined) this.buffered.delete(oldestSession)
    }
    const events = this.buffered.get(id) ?? []
    events.push(event)
    if (events.length > MAX_BUFFERED_EVENTS_PER_SESSION) events.shift()
    this.buffered.set(id, events)
  }

  subscribe(sessionId: string, listener: DaemonSubagentEventListener): () => void {
    const id = sessionId.trim()
    if (!id) return () => undefined
    const listeners = this.listeners.get(id) ?? new Set<DaemonSubagentEventListener>()
    listeners.add(listener)
    this.listeners.set(id, listeners)
    const buffered = this.buffered.get(id)
    if (buffered?.length) {
      this.buffered.delete(id)
      for (const event of buffered) listener(event)
    }
    return () => {
      listeners.delete(listener)
      if (!listeners.size) this.listeners.delete(id)
    }
  }
}

export interface DaemonSubagentEventSource {
  subscribe(sessionId: string, listener: DaemonSubagentEventListener): () => void
}
