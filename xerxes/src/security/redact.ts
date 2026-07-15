// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Replacement text used for credential and personally identifiable data. */
export const REDACTED = '[redacted]'

export interface RedactionRule {
  readonly name: string
  readonly pattern: RegExp
  readonly replacement: string
}

function rule(name: string, pattern: RegExp, replacement = REDACTED): RedactionRule {
  return Object.freeze({ name, pattern, replacement })
}

/** Deterministic log/audit redaction rules retained from Xerxes' Python runtime. */
export const DEFAULT_REDACTION_RULES: readonly RedactionRule[] = Object.freeze([
  rule('api_key_field', /(api[_-]?key)[\s:="']+([A-Za-z0-9._-]{8,})/gi, '$1=' + REDACTED),
  rule('openai_token', /\bsk-[A-Za-z0-9_-]{16,}\b/g, REDACTED),
  rule('anthropic_token', /\bsk-ant-[A-Za-z0-9_-]{16,}\b/g, REDACTED),
  rule('bearer_header', /(authorization:\s*bearer)\s+([A-Za-z0-9._-]+)/gi, '$1 ' + REDACTED),
  rule('password_field', /(password)[\s:="']+(\S+)/gi, '$1=' + REDACTED),
  rule('jwt_token', /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/g, REDACTED),
  rule('phone_us', /\b(?:\+1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b/g, REDACTED),
  rule('phone_international', /\+\d{1,3}[\s.-]?\d{4,14}\b/g, REDACTED),
  rule('email', /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g, REDACTED),
  rule('aws_access_key', /\bAKIA[0-9A-Z]{16}\b/g, REDACTED),
  rule('github_pat', /\bghp_[A-Za-z0-9]{20,}\b/g, REDACTED),
])

/** Case-insensitive object keys that must never be emitted with their values. */
export const SENSITIVE_FIELD_NAMES = new Set<string>([
  'api_key',
  'apikey',
  'token',
  'access_token',
  'refresh_token',
  'password',
  'secret',
  'client_secret',
  'authorization',
])

/** Apply each configured rule to a string without mutating the input. */
export function redactString(text: string, rules: readonly RedactionRule[] = DEFAULT_REDACTION_RULES): string {
  let redacted = String(text)
  for (const current of rules) {
    redacted = redacted.replace(current.pattern, current.replacement)
  }
  return redacted
}

/**
 * Clone and recursively redact an arbitrary payload for logs, diagnostics, or audit records.
 *
 * Cycles become a clear marker rather than throwing while constructing an error report. Plain
 * records and arrays are retained; other host objects are converted only when they are Errors.
 */
export function redactPayload(value: unknown, rules: readonly RedactionRule[] = DEFAULT_REDACTION_RULES): unknown {
  return redactValue(value, rules, new WeakSet<object>())
}

function redactValue(value: unknown, rules: readonly RedactionRule[], seen: WeakSet<object>): unknown {
  if (typeof value === 'string') return redactString(value, rules)
  if (value === null || typeof value !== 'object') return value
  if (seen.has(value)) return '[Circular]'
  seen.add(value)
  if (Array.isArray(value)) return value.map(item => redactValue(item, rules, seen))
  if (value instanceof Error) {
    return { name: value.name, message: redactString(value.message, rules) }
  }
  const result: Record<string, unknown> = {}
  for (const [key, nested] of Object.entries(value)) {
    result[key] = SENSITIVE_FIELD_NAMES.has(key.toLowerCase()) ? REDACTED : redactValue(nested, rules, seen)
  }
  return result
}
