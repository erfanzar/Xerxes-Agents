// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { timingSafeEqual } from 'node:crypto'
import { Buffer } from 'node:buffer'

import type { WebhookHeaders } from './webhooks.js'

/**
 * Read a webhook header case-insensitively.
 *
 * HTTP edges deliver adapter headers with varying casing (Bun lowercases
 * them; tests and proxies may not), while signature schemes are
 * case-sensitive about values only.
 */
export function webhookHeaderValue(headers: WebhookHeaders, name: string): string | undefined {
  const value = Object.entries(headers)
    .find(([headerName]) => headerName.toLowerCase() === name.toLowerCase())
    ?.[1]
  return typeof value === 'string' && value.length ? value : undefined
}

/**
 * Compare two strings without leaking their first differing byte through
 * early exit. Length differences still short-circuit (length itself is not
 * secret for fixed-algorithm signatures).
 */
export function constantTimeEqualStrings(left: string, right: string): boolean {
  const leftBytes = Buffer.from(left, 'utf8')
  const rightBytes = Buffer.from(right, 'utf8')
  return leftBytes.byteLength === rightBytes.byteLength && timingSafeEqual(leftBytes, rightBytes)
}
