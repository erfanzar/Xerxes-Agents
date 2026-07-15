// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export type RateLimitHeaders = Headers | Readonly<Record<string, string | number | undefined>>

/** Latest rate-limit observation for one provider/model pair. */
export class RateLimitState {
  lastUpdated: number
  limitRequests: number | undefined
  limitTokens: number | undefined
  remainingRequests: number | undefined
  remainingTokens: number | undefined
  resetAt: number | undefined
  retryAfter: number | undefined
  retryAfterUntil: number | undefined

  constructor(now = Date.now() / 1_000) {
    this.lastUpdated = now
  }
}

export interface RateLimitTrackerOptions {
  /** Fraction of a known budget below which pre-emptive throttling starts. */
  readonly throttleRatio?: number
  /** Epoch-seconds clock; injectable for deterministic callers and tests. */
  readonly now?: () => number
}

/**
 * Tracks provider rate-limit headers per provider/model pair and offers
 * deterministic throttle and delay decisions before callers hit a 429.
 */
export class RateLimitTracker {
  static readonly HEADER_LIMIT_REQUESTS = ['x-ratelimit-limit-requests', 'x-ratelimit-limit'] as const
  static readonly HEADER_REMAINING_REQUESTS = ['x-ratelimit-remaining-requests', 'x-ratelimit-remaining'] as const
  static readonly HEADER_LIMIT_TOKENS = ['x-ratelimit-limit-tokens'] as const
  static readonly HEADER_REMAINING_TOKENS = ['x-ratelimit-remaining-tokens'] as const
  static readonly HEADER_RESET = ['x-ratelimit-reset-requests', 'x-ratelimit-reset'] as const
  static readonly HEADER_RETRY_AFTER = ['retry-after'] as const
  /** Python-compatible abbreviation retained at the public boundary. */
  static readonly HEADER_LIMIT_REQS = RateLimitTracker.HEADER_LIMIT_REQUESTS
  /** Python-compatible abbreviation retained at the public boundary. */
  static readonly HEADER_REMAINING_REQS = RateLimitTracker.HEADER_REMAINING_REQUESTS

  private readonly clock: () => number
  private readonly states = new Map<string, RateLimitState>()
  private readonly throttleRatio: number

  constructor(options: RateLimitTrackerOptions | number = {}) {
    const normalized = typeof options === 'number' ? { throttleRatio: options } : options
    const throttleRatio = normalized.throttleRatio ?? 0.05
    if (!Number.isFinite(throttleRatio) || throttleRatio <= 0 || throttleRatio >= 1) {
      throw new RangeError('throttleRatio must be in (0.0, 1.0)')
    }
    this.throttleRatio = throttleRatio
    this.clock = normalized.now ?? (() => Date.now() / 1_000)
  }

  /** Merge response headers into a provider/model snapshot without clearing absent values. */
  update(provider: string, model: string, headers: RateLimitHeaders, now = this.clock()): RateLimitState {
    const state = this.states.get(stateKey(provider, model)) ?? new RateLimitState(now)
    const normalized = normalizeHeaders(headers)
    assignInteger(state, 'limitRequests', pluck(normalized, RateLimitTracker.HEADER_LIMIT_REQUESTS))
    assignInteger(state, 'remainingRequests', pluck(normalized, RateLimitTracker.HEADER_REMAINING_REQUESTS))
    assignInteger(state, 'limitTokens', pluck(normalized, RateLimitTracker.HEADER_LIMIT_TOKENS))
    assignInteger(state, 'remainingTokens', pluck(normalized, RateLimitTracker.HEADER_REMAINING_TOKENS))

    const reset = parseFloatValue(pluck(normalized, RateLimitTracker.HEADER_RESET))
    if (reset !== undefined) state.resetAt = now + reset
    const retryAfter = parseFloatValue(pluck(normalized, RateLimitTracker.HEADER_RETRY_AFTER))
    if (retryAfter !== undefined) {
      state.retryAfter = retryAfter
      state.retryAfterUntil = now + retryAfter
    }
    state.lastUpdated = now
    this.states.set(stateKey(provider, model), state)
    return state
  }

  /** Return a provider/model snapshot, or undefined when no response has supplied headers. */
  state(provider: string, model: string): RateLimitState | undefined {
    return this.states.get(stateKey(provider, model))
  }

  /** Return true while Retry-After is active or a known request/token budget is nearly exhausted. */
  shouldThrottle(provider: string, model: string, now = this.clock()): boolean {
    const state = this.state(provider, model)
    if (!state) return false
    if (state.retryAfterUntil !== undefined && state.retryAfterUntil > now) return true
    return (
      this.budgetIsLow(state.limitRequests, state.remainingRequests)
      || this.budgetIsLow(state.limitTokens, state.remainingTokens)
    )
  }

  /**
   * Return a pre-call delay in milliseconds. Retry-After wins; otherwise a
   * throttled bucket waits until reset (capped at 60 seconds) or 250ms.
   */
  delayMs(provider: string, model: string, now = this.clock()): number {
    const state = this.state(provider, model)
    if (!state) return 0
    if (state.retryAfterUntil !== undefined) {
      const remaining = state.retryAfterUntil - now
      if (remaining > 0) return Math.trunc(remaining * 1_000)
    }
    if (!this.shouldThrottle(provider, model, now)) return 0
    if (state.resetAt !== undefined) return Math.trunc(Math.min(60, Math.max(0, state.resetAt - now)) * 1_000)
    return 250
  }

  /** Drop every provider/model snapshot. */
  clear(): void {
    this.states.clear()
  }

  private budgetIsLow(limit: number | undefined, remaining: number | undefined): boolean {
    return limit !== undefined && remaining !== undefined && limit > 0 && remaining / limit < this.throttleRatio
  }
}

function assignInteger(
  state: RateLimitState,
  field: 'limitRequests' | 'remainingRequests' | 'limitTokens' | 'remainingTokens',
  raw: string | undefined,
): void {
  const parsed = parseInteger(raw)
  if (parsed !== undefined) state[field] = parsed
}

function normalizeHeaders(headers: RateLimitHeaders): Map<string, string> {
  const normalized = new Map<string, string>()
  if (headers instanceof Headers) {
    for (const [name, value] of headers.entries()) normalized.set(name.toLowerCase(), value)
    return normalized
  }
  for (const [name, value] of Object.entries(headers)) {
    if (value !== undefined) normalized.set(name.toLowerCase(), String(value))
  }
  return normalized
}

function pluck(headers: ReadonlyMap<string, string>, names: readonly string[]): string | undefined {
  for (const name of names) {
    const value = headers.get(name)
    if (value !== undefined) return value
  }
  return undefined
}

function parseInteger(value: string | undefined): number | undefined {
  if (value === undefined || !/^[+-]?\d+$/.test(value.trim())) return undefined
  const parsed = Number(value)
  return Number.isSafeInteger(parsed) ? parsed : undefined
}

function parseFloatValue(value: string | undefined): number | undefined {
  if (value === undefined || value.trim() === '') return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

function stateKey(provider: string, model: string): string {
  return `${provider}\u0000${model}`
}
