// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { SlackTransportError, type SlackIdentity } from './types.js'

export function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : undefined
}

export function teamId(value: unknown): value is string {
  return typeof value === 'string' && /^T[A-Z0-9]{1,63}$/u.test(value)
}
export function userId(value: unknown): value is string {
  return typeof value === 'string' && /^[UW][A-Z0-9]{1,63}$/u.test(value)
}
export function channelId(value: unknown): value is string {
  // Slack documents D- and C-prefixed IM examples. The prefix alone proves nothing.
  return typeof value === 'string' && /^[CDG][A-Z0-9]{1,63}$/u.test(value)
}
export function userToken(value: unknown): value is string {
  return typeof value === 'string' && /^xoxp-[A-Za-z0-9_-]{1,1019}$/u.test(value)
}
export function timestamp(value: unknown): value is string {
  return typeof value === 'string' && /^\d{1,18}\.\d{6}$/u.test(value)
}

export function identity(value: unknown): SlackIdentity {
  const parsed = record(value)
  if (!teamId(parsed?.teamId) || !userId(parsed?.userId)) throw new SlackTransportError('invalid_request')
  return { teamId: parsed.teamId, userId: parsed.userId }
}
