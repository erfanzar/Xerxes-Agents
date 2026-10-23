// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'

import { HubProtocolError } from './protocol.js'

/**
 * Server-side cursor codec. Clients treat cursors as opaque strings.
 *
 * A cursor names one position in one Hub stream and is bound to the scope it
 * was issued for: a cursor from a subscription over workspaces A,B cannot be
 * replayed against a subscription over C. Scope binding is about correctness
 * (no silently empty or foreign replay), not authorization; every event is
 * still checked against the reader's grants when it is read.
 */

export interface HubCursor {
  readonly streamId: string
  readonly sequence: number
  readonly scopeKey: string
}

const PREFIX = 'hc1.'

export interface CursorScope {
  readonly workspaceIds: readonly string[]
  readonly sessionIds?: readonly string[]
}

export function cursorScopeKey(scope: CursorScope): string {
  const workspaces = [...new Set(scope.workspaceIds)].sort()
  const sessions = [...new Set(scope.sessionIds ?? [])].sort()
  return createHash('sha256')
    .update(JSON.stringify([workspaces, sessions]))
    .digest('hex')
    .slice(0, 16)
}

export function encodeCursor(cursor: HubCursor): string {
  if (!Number.isSafeInteger(cursor.sequence) || cursor.sequence < 0) {
    throw new RangeError(`cursor sequence must be a non-negative safe integer, got ${cursor.sequence}`)
  }
  const body = JSON.stringify({ s: cursor.streamId, n: cursor.sequence, k: cursor.scopeKey })
  return PREFIX + Buffer.from(body, 'utf8').toString('base64url')
}

export function decodeCursor(text: string): HubCursor {
  if (typeof text !== 'string' || !text.startsWith(PREFIX) || text.length > 512) {
    throw new HubProtocolError('invalid_request', 'malformed cursor')
  }
  let value: unknown
  try {
    value = JSON.parse(Buffer.from(text.slice(PREFIX.length), 'base64url').toString('utf8')) as unknown
  } catch {
    throw new HubProtocolError('invalid_request', 'malformed cursor')
  }
  if (
    typeof value !== 'object' ||
    value === null ||
    typeof (value as { s?: unknown }).s !== 'string' ||
    typeof (value as { k?: unknown }).k !== 'string' ||
    !Number.isSafeInteger((value as { n?: unknown }).n) ||
    ((value as { n: number }).n) < 0
  ) {
    throw new HubProtocolError('invalid_request', 'malformed cursor')
  }
  const record = value as { s: string; n: number; k: string }
  return { streamId: record.s, sequence: record.n, scopeKey: record.k }
}

/**
 * Decode a cursor a client sent for a given stream and scope. A cursor from a
 * different stream (the Hub's store was replaced) reports `stream_replaced`;
 * one from a different scope is refused so a client cannot accidentally resume
 * a narrower or foreign subscription.
 */
export function resolveClientCursor(
  text: string | undefined,
  expected: { readonly streamId: string; readonly scopeKey: string },
): { readonly kind: 'start' } | { readonly kind: 'after'; readonly sequence: number } | { readonly kind: 'stream_replaced' } {
  if (text === undefined) return { kind: 'start' }
  const cursor = decodeCursor(text)
  if (cursor.streamId !== expected.streamId) return { kind: 'stream_replaced' }
  if (cursor.scopeKey !== expected.scopeKey) {
    throw new HubProtocolError('invalid_request', 'cursor was issued for a different subscription scope')
  }
  return { kind: 'after', sequence: cursor.sequence }
}
