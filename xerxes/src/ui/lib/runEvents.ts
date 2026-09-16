// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { GatewayRpc } from '../app/interfaces.js'

export interface RunEvent { sequence: number; text: string; at: number }
export interface RunEventPage { events: RunEvent[]; nextCursor: number; hasMore: boolean }

/** Validate the whole page before advancing: malformed evidence must never be skipped. */
export async function readRunEvents(rpc: GatewayRpc, runId: string, scope: 'session' | 'workspace', after = 0): Promise<RunEventPage> {
  const result = await rpc('run.events', { run_id: runId, scope, after_sequence: after, limit: 20 })
  if (!result?.ok) throw new Error(String(result?.error || 'Run events unavailable. Press R to retry.'))
  if (!Array.isArray(result.events) || result.events.length > 20 || typeof result.has_more !== 'boolean'
    || typeof result.next_cursor !== 'number' || !Number.isSafeInteger(result.next_cursor)) throw new Error('Invalid run event page. Cursor was not advanced.')
  let cursor = after
  const events = result.events.map((value: unknown): RunEvent => {
    if (!value || typeof value !== 'object') throw new Error('Invalid run event. Cursor was not advanced.')
    const row = value as Record<string, unknown>
    if (typeof row.sequence !== 'number' || !Number.isSafeInteger(row.sequence) || row.sequence <= cursor
      || typeof row.text !== 'string' || typeof row.at !== 'number' || !Number.isFinite(row.at)
      || !Number.isFinite(new Date(row.at).getTime())) throw new Error('Invalid run event. Cursor was not advanced.')
    cursor = row.sequence
    return { sequence: row.sequence, text: row.text, at: row.at }
  })
  if (result.next_cursor !== cursor || (result.has_more && events.length === 0)) throw new Error('Invalid run event cursor. Page was not advanced.')
  return { events, nextCursor: cursor, hasMore: result.has_more }
}
