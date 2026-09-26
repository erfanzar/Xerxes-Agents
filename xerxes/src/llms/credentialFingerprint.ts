// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'

/** Headers that carry who is calling — the parts that change when an account or key changes. */
const CREDENTIAL_HEADERS = new Set([
  'authorization',
  'x-api-key',
  'api-key',
  'x-goog-api-key',
  'chatgpt-account-id',
  'openai-organization',
  'openai-project',
])

/**
 * A short, non-reversible identity for the credential a request carries.
 *
 * Provider-agnostic on purpose: every client already builds its auth headers
 * (bearer token, API key, account id), so hashing those answers "would a
 * request now go out as someone else?" for any provider — a switched
 * subscription account, a fresh login, an edited API key — without knowing
 * how that provider stores its credentials. Undefined when there is nothing
 * to identify (no auth headers).
 */
export function credentialFingerprint(headers: Readonly<Record<string, string | undefined>>): string | undefined {
  const parts = Object.entries(headers)
    .filter(([name, value]) => value && CREDENTIAL_HEADERS.has(name.toLowerCase()))
    .map(([name, value]) => `${name.toLowerCase()}=${value}`)
    .sort()
  if (!parts.length) return undefined
  return createHash('sha256').update(parts.join('\n')).digest('hex').slice(0, 24)
}
