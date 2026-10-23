// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { HubEvent } from '../hub/protocol.js'

/**
 * Host-independent client bridge contract shared by the phone PWA, the desktop
 * supervisor tab and tests. A bridge carries requests and pushed events for one
 * scope (a workspace, or one session inside it); it never exposes arbitrary
 * desktop IPC or daemon methods. This module is browser-safe.
 */

export type ConnectionStatus =
  | { readonly state: 'connecting' }
  | { readonly state: 'replaying' }
  | { readonly state: 'live' }
  | { readonly state: 'reconnecting'; readonly attempt: number; readonly lastError?: string }
  | { readonly state: 'revoked'; readonly reason: string }
  | { readonly state: 'closed' }

export interface BridgeScope {
  readonly workspaceId: string
  readonly sessionId?: string
}

export interface ClientBridge {
  readonly scope: BridgeScope
  /** Resolves with the result, or rejects with the server's error. Never resolves an empty success for a failure. */
  call(method: string, params?: Readonly<Record<string, unknown>>): Promise<Readonly<Record<string, unknown>>>
  /** Subscribe to events for this scope; returns the unsubscribe function. */
  onEvent(listener: (event: HubEvent) => void): () => void
  onStatus(listener: (status: ConnectionStatus) => void): () => void
  status(): ConnectionStatus
  dispose(): void
}

export type ApplyOutcome =
  | { readonly kind: 'applied' }
  | { readonly kind: 'duplicate' }
  | { readonly kind: 'gap'; readonly expected: number; readonly received: number }

/**
 * Tracks the last event a client actually applied, so a reconnect resumes from
 * there. Events are deduplicated by id and by sequence; a jump in sequence is
 * reported as a gap so the caller can resnapshot instead of silently missing
 * events. The tracker only moves forward after `applied`.
 */
export class EventCursorTracker {
  private lastSequence: number | undefined
  private lastCursor: string | undefined
  private readonly recentIds: string[] = []
  private readonly recentSet = new Set<string>()
  private readonly memory: number

  constructor(initial?: { readonly sequence: number; readonly cursor: string }, memory = 2048) {
    this.memory = memory
    if (initial) {
      this.lastSequence = initial.sequence
      this.lastCursor = initial.cursor
    }
  }

  get cursor(): string | undefined {
    return this.lastCursor
  }

  get sequence(): number | undefined {
    return this.lastSequence
  }

  /** Decide what to do with an incoming event; call `commit` after applying it. */
  classify(event: HubEvent): ApplyOutcome {
    if (this.recentSet.has(event.event_id)) return { kind: 'duplicate' }
    if (this.lastSequence !== undefined && event.sequence <= this.lastSequence) return { kind: 'duplicate' }
    if (this.lastSequence !== undefined && event.sequence !== this.lastSequence + 1) {
      return { kind: 'gap', expected: this.lastSequence + 1, received: event.sequence }
    }
    return { kind: 'applied' }
  }

  commit(event: HubEvent): void {
    this.lastSequence = event.sequence
    this.lastCursor = event.cursor
    this.recentIds.push(event.event_id)
    this.recentSet.add(event.event_id)
    while (this.recentIds.length > this.memory) {
      const dropped = this.recentIds.shift()
      if (dropped !== undefined) this.recentSet.delete(dropped)
    }
  }

  /** After a snapshot reset the stream restarts from the snapshot boundary. */
  reset(boundary?: { readonly sequence: number; readonly cursor: string }): void {
    this.lastSequence = boundary?.sequence
    this.lastCursor = boundary?.cursor
    this.recentIds.length = 0
    this.recentSet.clear()
  }
}

/** Generate an idempotency key for a client mutation (crypto.randomUUID where available). */
export function newCommandId(): string {
  const crypto = globalThis.crypto
  if (crypto && typeof crypto.randomUUID === 'function') return crypto.randomUUID()
  const bytes = new Uint8Array(16)
  if (!crypto || typeof crypto.getRandomValues !== 'function') throw new Error('a secure random source is required to create command ids')
  crypto.getRandomValues(bytes)
  return Array.from(bytes, byte => byte.toString(16).padStart(2, '0')).join('')
}
