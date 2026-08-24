// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Durable per-session notification ledger.
 *
 * Background-task completions used to be announced only on the connection that
 * started them, so a TUI disconnect swallowed the notice even though the work
 * itself finished and persisted. Queuing notices in the parent session's
 * metadata makes them survive the connection: the next `session.open` drains
 * the backlog to whichever client attaches. Delivery is at-most-once by design
 * — a notice is bookkeeping, not transcript content, and duplicating it on
 * every attach would be worse than losing one.
 */

export interface QueuedSessionNotification {
  readonly at: number
  readonly level: 'error' | 'info' | 'warning'
  readonly message: string
}

const PENDING_NOTIFICATIONS_KEY = 'pending_notifications'

/** Ring ceiling: idle sessions hammered by background tasks cannot grow metadata without bound. */
export const MAX_PENDING_SESSION_NOTIFICATIONS = 32

/** Queue one notice for delivery at the next session attach. */
export function queueSessionNotification(
  metadata: Record<string, unknown>,
  notification: QueuedSessionNotification,
): void {
  const pending = [...readSessionNotifications(metadata), notification]
  metadata[PENDING_NOTIFICATIONS_KEY] = pending.slice(-MAX_PENDING_SESSION_NOTIFICATIONS)
}

/** Read the backlog without consuming it. */
export function readSessionNotifications(
  metadata: Readonly<Record<string, unknown>>,
): readonly QueuedSessionNotification[] {
  const raw = metadata[PENDING_NOTIFICATIONS_KEY]
  if (!Array.isArray(raw)) return []
  return raw.filter(isQueuedNotification)
}

/** Drain the backlog for delivery; the same notice is never delivered twice. */
export function takeSessionNotifications(metadata: Record<string, unknown>): readonly QueuedSessionNotification[] {
  const drained = readSessionNotifications(metadata)
  if (drained.length) delete metadata[PENDING_NOTIFICATIONS_KEY]
  return drained
}

function isQueuedNotification(value: unknown): value is QueuedSessionNotification {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as Record<string, unknown>
  return typeof candidate.at === 'number'
    && typeof candidate.message === 'string'
    && (candidate.level === 'error' || candidate.level === 'info' || candidate.level === 'warning')
}
