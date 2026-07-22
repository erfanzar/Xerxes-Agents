// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Recovery-oriented categories for provider and network failures. */
export const ErrorKind = {
  RATE_LIMIT: 'rate_limit',
  CONTEXT_OVERFLOW: 'context_overflow',
  PROVIDER_DOWN: 'provider_down',
  AUTH: 'auth',
  QUOTA_EXCEEDED: 'quota_exceeded',
  TIMEOUT: 'timeout',
  TRANSIENT: 'transient',
  BAD_REQUEST: 'bad_request',
  FATAL: 'fatal',
  UNKNOWN: 'unknown',
} as const

export type ErrorKind = (typeof ErrorKind)[keyof typeof ErrorKind]

export interface ClassifiedError {
  readonly kind: ErrorKind
  readonly message: string
  readonly original: unknown
  readonly retryable: boolean
  /** Seconds supplied by a Retry-After-style message, when present. */
  readonly suggestedBackoffSeconds?: number
}

interface ErrorDetails {
  readonly code?: string
  readonly message: string
  readonly name: string
  readonly status?: number
}

const RETRYABLE_KINDS = new Set<ErrorKind>([
  ErrorKind.RATE_LIMIT,
  ErrorKind.PROVIDER_DOWN,
  ErrorKind.TIMEOUT,
  ErrorKind.TRANSIENT,
])

const PATTERNS: ReadonlyArray<readonly [ErrorKind, readonly RegExp[]]> = [
  [ErrorKind.RATE_LIMIT, [/rate.?limit/i, /too many requests/i, /\b429\b/]],
  [
    ErrorKind.CONTEXT_OVERFLOW,
    [
      /context.{0,8}length/i,
      /maximum.{0,8}context/i,
      /context window/i,
      /too many tokens/i,
      /reduce.{0,8}messages/i,
      /prompt is too long/i,
    ],
  ],
  [ErrorKind.AUTH, [/unauthorized/i, /invalid.{0,4}api.{0,4}key/i, /\b40[13]\b/, /forbidden/i]],
  [ErrorKind.QUOTA_EXCEEDED, [/quota/i, /insufficient.{0,8}credit/i, /billing/i]],
  [ErrorKind.PROVIDER_DOWN, [/\b(?:50[0-4]|529)\b/, /service unavailable/i, /overloaded/i, /bad gateway/i]],
  [ErrorKind.TIMEOUT, [/timeout/i, /timed out/i, /\b408\b/]],
  [ErrorKind.BAD_REQUEST, [/\b400\b/, /invalid request/i, /malformed/i]],
  [ErrorKind.TRANSIENT, [/transient/i, /temporarily/i]],
]

const CONNECTION_CODES = new Set([
  'ECONNABORTED',
  'ECONNREFUSED',
  'ECONNRESET',
  'EAI_AGAIN',
  'ENETDOWN',
  'ENETUNREACH',
  'ENOTFOUND',
])

/** Stateless provider error classifier with JavaScript-native error detection. */
export class ErrorClassifier {
  classify(error: unknown): ClassifiedError {
    const details = describeError(error)
    const retryAfter = parseRetryAfter(details.message)

    if (details.name === 'AbortError' || details.name === 'InterruptedError') {
      return classified(ErrorKind.FATAL, error, 'user interrupt')
    }
    if (details.name === 'TimeoutError' || details.name === 'StreamInactivityError' || details.code === 'ETIMEDOUT') {
      return classified(ErrorKind.TIMEOUT, error, details.message, retryAfter)
    }
    if (details.name === 'ConfigurationError') {
      return classified(ErrorKind.FATAL, error, details.message, retryAfter)
    }
    if (details.name === 'ValidationError') {
      return classified(ErrorKind.BAD_REQUEST, error, details.message, retryAfter)
    }
    if (
      details.name === 'ConnectionError'
      || (details.code !== undefined && CONNECTION_CODES.has(details.code))
      || isFetchNetworkFailure(error, details.message)
    ) {
      return classified(ErrorKind.PROVIDER_DOWN, error, details.message)
    }

    const statusKind = kindForStatus(details.status)
    if (statusKind !== undefined) {
      // Providers signal context overflow as HTTP 400 (OpenAI
      // context_length_exceeded, Anthropic "prompt is too long"); let the
      // message patterns win so compaction/recovery can engage instead of a
      // hard bad-request failure.
      if (details.status === 400 && matchesPatterns(ErrorKind.CONTEXT_OVERFLOW, details.message)) {
        return classified(ErrorKind.CONTEXT_OVERFLOW, error, details.message, retryAfter)
      }
      return classified(statusKind, error, details.message, retryAfter)
    }

    for (const [kind] of PATTERNS) {
      if (matchesPatterns(kind, details.message)) {
        return classified(kind, error, details.message, retryAfter)
      }
    }
    return classified(ErrorKind.UNKNOWN, error, details.message, retryAfter)
  }

  isRetryable(error: unknown): boolean {
    return this.classify(error).retryable
  }
}

const defaultClassifier = new ErrorClassifier()

/** Classify an error with the module's shared stateless classifier. */
export function classifyError(error: unknown): ClassifiedError {
  return defaultClassifier.classify(error)
}

/** Python-compatible concise alias for `classifyError`. */
export const classify = classifyError

function classified(
  kind: ErrorKind,
  original: unknown,
  message: string,
  suggestedBackoffSeconds?: number,
): ClassifiedError {
  return {
    kind,
    original,
    message,
    retryable: RETRYABLE_KINDS.has(kind),
    ...(suggestedBackoffSeconds === undefined ? {} : { suggestedBackoffSeconds }),
  }
}

function describeError(error: unknown): ErrorDetails {
  if (error instanceof Error) {
    const record = error as unknown as Record<string, unknown>
    const code = stringProperty(record, 'code')
    const status = numberProperty(record, 'status', 'statusCode')
    const optional: { code?: string; status?: number } = {}
    if (code !== undefined) optional.code = code
    if (status !== undefined) optional.status = status
    return {
      name: error.name || 'Error',
      message: error.message || error.name || 'Error',
      ...optional,
    }
  }
  if (isRecord(error)) {
    const message = stringProperty(error, 'message') ?? String(error)
    const name = stringProperty(error, 'name') ?? 'Error'
    const code = stringProperty(error, 'code')
    const status = numberProperty(error, 'status', 'statusCode')
    const optional: { code?: string; status?: number } = {}
    if (code !== undefined) optional.code = code
    if (status !== undefined) optional.status = status
    return {
      name,
      message,
      ...optional,
    }
  }
  return { name: typeof error, message: String(error) }
}

function matchesPatterns(kind: ErrorKind, message: string): boolean {
  const entry = PATTERNS.find(([patternKind]) => patternKind === kind)
  return entry !== undefined && entry[1].some(pattern => pattern.test(message))
}

function kindForStatus(status: number | undefined): ErrorKind | undefined {
  if (status === 429) return ErrorKind.RATE_LIMIT
  if (status === 401 || status === 403) return ErrorKind.AUTH
  if (status === 400) return ErrorKind.BAD_REQUEST
  if (status === 408) return ErrorKind.TIMEOUT
  if (status !== undefined && ((status >= 500 && status <= 504) || status === 529)) return ErrorKind.PROVIDER_DOWN
  return undefined
}

function parseRetryAfter(message: string): number | undefined {
  const match = /retry[- ]after[:\s]+(\d+(?:\.\d+)?)/i.exec(message)
  if (!match?.[1]) return undefined
  const parsed = Number(match[1])
  return Number.isFinite(parsed) ? parsed : undefined
}

function isFetchNetworkFailure(error: unknown, message: string): boolean {
  return error instanceof TypeError && /(?:fetch failed|network(?: request)? failed)/i.test(message)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object'
}

function stringProperty(value: Record<string, unknown>, key: string): string | undefined {
  const candidate = value[key]
  return typeof candidate === 'string' ? candidate : undefined
}

function numberProperty(value: Record<string, unknown>, ...keys: readonly string[]): number | undefined {
  for (const key of keys) {
    const candidate = value[key]
    if (typeof candidate === 'number' && Number.isFinite(candidate)) return candidate
  }
  return undefined
}
