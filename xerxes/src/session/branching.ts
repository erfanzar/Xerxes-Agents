// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto'

import { cloneSessionRecord, SessionRecord } from './models.js'
import { isLockableSessionStore, type SessionStore } from './store.js'

export interface BranchSessionOptions {
  readonly newSessionId?: string
  readonly sourceSessionId: string
  readonly title?: string
}

/** Create a child session with a deep copy of the source conversation history. */
export function branchSession(store: SessionStore, options: BranchSessionOptions): SessionRecord {
  const source = store.loadSession(options.sourceSessionId)
  if (!source) throw new Error(`unknown source session: ${options.sourceSessionId}`)
  if (options.newSessionId !== undefined) {
    if (options.newSessionId === source.sessionId) {
      throw new Error('newSessionId must differ from sourceSessionId')
    }
    if (store.loadSession(options.newSessionId)) {
      throw new Error(`session already exists: ${options.newSessionId}`)
    }
  }

  const child = cloneSessionRecord(source)
  const now = new Date().toISOString()
  child.sessionId = options.newSessionId ?? randomUUID().replaceAll('-', '')
  child.createdAt = now
  child.updatedAt = now
  child.parentSessionId = source.sessionId
  child.metadata = {
    ...source.metadata,
    forked_from: source.sessionId,
    title: options.title || stringMetadata(source.metadata.title),
  }
  store.saveSession(child)
  return child
}

/**
 * {@link branchSession} with the whole check-then-create cycle performed under
 * the cross-process store lock. The synchronous form's existence checks and
 * final save are separate statements, so two processes branching to the same
 * explicit id could both pass the check and one silently overwrite the other;
 * under the shared lock exactly one wins and the other observes the created
 * row and fails its own check.
 */
export async function branchSessionExclusive(store: SessionStore, options: BranchSessionOptions): Promise<SessionRecord> {
  if (!isLockableSessionStore(store)) return branchSession(store, options)
  return store.withWriteLock(() => branchSession(store, options))
}

/** Follow parent pointers from a session back to its root, detecting cycles. */
export function sessionLineage(store: SessionStore, sessionId: string): string[] {
  const chain: string[] = []
  const seen = new Set<string>()
  let current: string | null = sessionId
  while (current && !seen.has(current)) {
    seen.add(current)
    const session = store.loadSession(current)
    if (!session) break
    chain.push(session.sessionId)
    current = session.parentSessionId
  }
  return chain
}

/** Compatibility alias for the original session API. */
export const lineage = sessionLineage

function stringMetadata(value: unknown): string {
  return typeof value === 'string' ? value : ''
}
