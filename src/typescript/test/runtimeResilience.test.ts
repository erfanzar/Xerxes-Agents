// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  CircuitBreakerConfig,
  CircuitBreakerRegistry,
  CircuitOpenError,
  CircuitState,
  ErrorClassifier,
  ErrorKind,
  FallbackRegistry,
  HealthSnapshot,
  RateLimitTracker,
  ToolHealthProber,
  classifyError,
} from '../src/runtime/index.js'

test('circuit breakers count failures in a rolling window and admit one half-open probe', () => {
  const breaker = new CircuitBreakerRegistry(new CircuitBreakerConfig({
    failureThreshold: 2,
    cooldownSeconds: 10,
    rollingWindowSeconds: 5,
    successThreshold: 2,
  }))

  expect(breaker.stateOf('openai:gpt')).toBe(CircuitState.CLOSED)
  expect(breaker.recordFailure('openai:gpt', 0)).toBe(false)
  expect(breaker.recordFailure('openai:gpt', 1)).toBe(true)
  expect(breaker.shouldAllow('openai:gpt', 9)).toBe(false)
  expect(breaker.shouldAllow('openai:gpt', 11)).toBe(true)
  expect(breaker.shouldAllow('openai:gpt', 11)).toBe(false)
  breaker.recordSuccess('openai:gpt', 11)
  expect(breaker.stateOf('openai:gpt')).toBe(CircuitState.HALF_OPEN)
  expect(breaker.shouldAllow('openai:gpt', 12)).toBe(true)
  breaker.recordSuccess('openai:gpt', 12)
  expect(breaker.stateOf('openai:gpt')).toBe(CircuitState.CLOSED)

  const rolling = new CircuitBreakerRegistry({ failureThreshold: 3, cooldownSeconds: 1, rollingWindowSeconds: 5 })
  rolling.recordFailure('anthropic:claude', 0)
  rolling.recordFailure('anthropic:claude', 1)
  expect(rolling.recordFailure('anthropic:claude', 10)).toBe(false)
  expect(rolling.stateOf('anthropic:claude')).toBe(CircuitState.CLOSED)
})

test('circuit breaker call helpers record sync and async failures without swallowing them', async () => {
  let now = 0
  const breaker = new CircuitBreakerRegistry({ failureThreshold: 1, cooldownSeconds: 10 }, { now: () => now })
  expect(breaker.call('tool', () => 'ok')).toBe('ok')
  expect(breaker.call('math', (left: number, right: number) => left + right, 2, 3)).toBe(5)
  expect(() => breaker.call('tool', () => { throw new Error('boom') })).toThrow('boom')
  expect(() => breaker.call('tool', () => 'blocked')).toThrow(CircuitOpenError)
  now = 20
  await expect(breaker.callAsync('tool', async () => 'recovered')).resolves.toBe('recovered')
  expect(breaker.stateOf('tool')).toBe(CircuitState.CLOSED)
})

test('error classification maps built-in JavaScript failures, status codes, patterns, and retry hints', () => {
  const timeout = new Error('read timed out')
  timeout.name = 'TimeoutError'
  expect(classifyError(timeout)).toMatchObject({ kind: ErrorKind.TIMEOUT, retryable: true })
  expect(classifyError({ status: 429, message: 'slow down; Retry-After: 12.5' })).toMatchObject({
    kind: ErrorKind.RATE_LIMIT,
    retryable: true,
    suggestedBackoffSeconds: 12.5,
  })
  expect(classifyError(new Error('maximum context length is 128000'))).toMatchObject({
    kind: ErrorKind.CONTEXT_OVERFLOW,
    retryable: false,
  })
  expect(classifyError(new Error('HTTP 503 Service Unavailable'))).toMatchObject({
    kind: ErrorKind.PROVIDER_DOWN,
    retryable: true,
  })
  expect(classifyError(new Error('unrelated message'))).toMatchObject({ kind: ErrorKind.UNKNOWN, retryable: false })
  expect(new ErrorClassifier().isRetryable({ code: 'ECONNREFUSED', message: 'connection refused' })).toBe(true)
})

test('rate limit tracking merges case-insensitive headers and uses an injected clock for delays', () => {
  let now = 100
  const tracker = new RateLimitTracker({ throttleRatio: 0.1, now: () => now })
  const state = tracker.update('openai', 'gpt-4o', {
    'X-RateLimit-Limit-Requests': '100',
    'x-ratelimit-remaining-requests': '5',
    'x-ratelimit-limit-tokens': '1000',
    'x-ratelimit-remaining-tokens': '400',
    'x-ratelimit-reset-requests': '90',
  })
  expect(state.limitRequests).toBe(100)
  expect(state.remainingRequests).toBe(5)
  expect(tracker.shouldThrottle('openai', 'gpt-4o')).toBe(true)
  expect(tracker.delayMs('openai', 'gpt-4o')).toBe(60_000)

  tracker.update('openai', 'gpt-4o', { 'retry-after': '3.2' })
  expect(tracker.delayMs('openai', 'gpt-4o')).toBe(3_200)
  now = 104
  expect(tracker.delayMs('openai', 'gpt-4o')).toBe(60_000)
  tracker.clear()
  expect(tracker.delayMs('openai', 'gpt-4o')).toBe(0)
})

test('fallback registries route unique choices and health probes schedule and retain snapshots', () => {
  const fallbacks = new FallbackRegistry()
  fallbacks.set('summary', 'opus', ['opus', 'sonnet', 'haiku', 'haiku'])
  expect(fallbacks.orderFor('summary')).toEqual(['opus', 'sonnet', 'haiku'])
  expect(fallbacks.nextAfter('summary', 'sonnet')).toBe('haiku')
  expect(fallbacks.remove('summary')).toBe(true)
  expect(fallbacks.orderFor('summary')).toEqual([])

  let calls = 0
  let monotonic = 0
  const prober = new ToolHealthProber({ now: () => 0, monotonicNow: () => monotonic })
  prober.register('healthy', () => {
    calls += 1
    monotonic += 0.002
    return true
  }, { intervalSeconds: 10 })
  prober.register('degraded', () => new HealthSnapshot({ name: 'degraded', status: 'degraded', message: 'warming' }))
  expect(prober.runDue(0).map(snapshot => snapshot.status)).toEqual(['ok', 'degraded'])
  expect(calls).toBe(1)
  expect(prober.runDue(1)).toEqual([])
  expect(prober.runDue(10)).toHaveLength(1)
  expect(calls).toBe(2)
  expect(prober.healthy('healthy')).toBe(true)
  expect(prober.snapshot('degraded')?.status).toBe('degraded')
  expect(prober.runOne('missing', 12)).toMatchObject({ status: 'unknown', message: 'not registered' })
})
