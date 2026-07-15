// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Standard states for a per-key downstream circuit breaker. */
export const CircuitState = {
  CLOSED: 'closed',
  OPEN: 'open',
  HALF_OPEN: 'half_open',
} as const

export type CircuitState = (typeof CircuitState)[keyof typeof CircuitState]

export interface CircuitBreakerConfigOptions {
  /** Failures in the rolling window required to trip a closed circuit. */
  readonly failureThreshold?: number
  /** Seconds an open circuit rejects calls before allowing one probe. */
  readonly cooldownSeconds?: number
  /** Seconds over which failed calls are counted. */
  readonly rollingWindowSeconds?: number
  /** Consecutive successful half-open probes required to close the circuit. */
  readonly successThreshold?: number
}

/** Immutable thresholds shared by every breaker in a registry. */
export class CircuitBreakerConfig {
  readonly cooldownSeconds: number
  readonly failureThreshold: number
  readonly rollingWindowSeconds: number
  readonly successThreshold: number

  constructor(options: CircuitBreakerConfigOptions = {}) {
    this.failureThreshold = positiveInteger(options.failureThreshold ?? 5, 'failureThreshold')
    this.cooldownSeconds = nonNegativeNumber(options.cooldownSeconds ?? 30, 'cooldownSeconds')
    this.rollingWindowSeconds = positiveNumber(options.rollingWindowSeconds ?? 60, 'rollingWindowSeconds')
    this.successThreshold = positiveInteger(options.successThreshold ?? 1, 'successThreshold')
  }
}

export class CircuitOpenError extends Error {
  readonly key: string
  readonly openedAt: number

  constructor(key: string, openedAt: number) {
    super(`Circuit '${key}' is OPEN since ${openedAt.toFixed(0)}`)
    this.name = 'CircuitOpenError'
    this.key = key
    this.openedAt = openedAt
  }
}

export interface CircuitBreakerRegistryOptions {
  /** Monotonic seconds clock; injectable so state transitions are deterministic in tests. */
  readonly now?: () => number
}

interface BreakerState {
  consecutiveSuccesses: number
  failures: number[]
  openedAt: number
  probeInFlight: boolean
  state: CircuitState
}

/**
 * Maintains independent CLOSED -> OPEN -> HALF_OPEN circuit breakers.
 *
 * JavaScript executes these mutations synchronously, so a Map provides the
 * same atomic transition boundary that the Python implementation protects
 * with a lock. Use `callAsync` for promise-returning downstream operations.
 */
export class CircuitBreakerRegistry {
  readonly config: CircuitBreakerConfig
  private readonly clock: () => number
  private readonly states = new Map<string, BreakerState>()

  constructor(
    config: CircuitBreakerConfig | CircuitBreakerConfigOptions = {},
    options: CircuitBreakerRegistryOptions = {},
  ) {
    this.config = config instanceof CircuitBreakerConfig ? config : new CircuitBreakerConfig(config)
    this.clock = options.now ?? (() => performance.now() / 1_000)
  }

  /** Return whether one call may proceed, transitioning OPEN to HALF_OPEN when due. */
  shouldAllow(key: string, now = this.clock()): boolean {
    const state = this.entry(key)
    if (state.state === CircuitState.CLOSED) return true
    if (state.state === CircuitState.OPEN) {
      if (now - state.openedAt < this.config.cooldownSeconds) return false
      state.state = CircuitState.HALF_OPEN
      state.consecutiveSuccesses = 0
      state.probeInFlight = true
      return true
    }
    if (state.probeInFlight) return false
    state.probeInFlight = true
    return true
  }

  /** Record a successful call. Half-open probes close the circuit after the configured threshold. */
  recordSuccess(key: string, _now = this.clock()): void {
    const state = this.entry(key)
    if (state.state !== CircuitState.HALF_OPEN) return
    state.probeInFlight = false
    state.consecutiveSuccesses += 1
    if (state.consecutiveSuccesses < this.config.successThreshold) return
    state.state = CircuitState.CLOSED
    state.failures = []
    state.consecutiveSuccesses = 0
  }

  /** Record a failure and return whether it opened or re-opened the circuit. */
  recordFailure(key: string, now = this.clock()): boolean {
    const state = this.entry(key)
    const cutoff = now - this.config.rollingWindowSeconds
    state.failures = state.failures.filter(timestamp => timestamp >= cutoff)
    state.failures.push(now)
    if (state.state === CircuitState.HALF_OPEN) {
      state.state = CircuitState.OPEN
      state.openedAt = now
      state.consecutiveSuccesses = 0
      state.probeInFlight = false
      return true
    }
    if (state.state === CircuitState.CLOSED && state.failures.length >= this.config.failureThreshold) {
      state.state = CircuitState.OPEN
      state.openedAt = now
      return true
    }
    return false
  }

  /** Return a key's current state, creating a closed breaker on first access. */
  stateOf(key: string): CircuitState {
    return this.entry(key).state
  }

  /** Remove state for one key, or all keys when no key is supplied. */
  reset(key?: string): void {
    if (key === undefined) {
      this.states.clear()
      return
    }
    this.states.delete(key)
  }

  /** Invoke a synchronous operation through a circuit breaker. */
  call<Args extends readonly unknown[], T>(key: string, fn: (...args: Args) => T, ...args: Args): T {
    const now = this.clock()
    if (!this.shouldAllow(key, now)) throw new CircuitOpenError(key, this.entry(key).openedAt)
    try {
      const result = fn(...args)
      this.recordSuccess(key, this.clock())
      return result
    } catch (error) {
      this.recordFailure(key, this.clock())
      throw error
    }
  }

  /** Invoke a promise-returning operation through a circuit breaker. */
  async callAsync<Args extends readonly unknown[], T>(
    key: string,
    fn: (...args: Args) => Promise<T>,
    ...args: Args
  ): Promise<T> {
    const now = this.clock()
    if (!this.shouldAllow(key, now)) throw new CircuitOpenError(key, this.entry(key).openedAt)
    try {
      const result = await fn(...args)
      this.recordSuccess(key, this.clock())
      return result
    } catch (error) {
      this.recordFailure(key, this.clock())
      throw error
    }
  }

  private entry(key: string): BreakerState {
    const existing = this.states.get(key)
    if (existing) return existing
    const created: BreakerState = {
      state: CircuitState.CLOSED,
      failures: [],
      consecutiveSuccesses: 0,
      openedAt: 0,
      probeInFlight: false,
    }
    this.states.set(key, created)
    return created
  }
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 1) throw new RangeError(`${name} must be a positive integer`)
  return value
}

function positiveNumber(value: number, name: string): number {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${name} must be positive`)
  return value
}

function nonNegativeNumber(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0) throw new RangeError(`${name} must be non-negative`)
  return value
}
