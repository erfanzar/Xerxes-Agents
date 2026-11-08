// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { CompletionDeadlineError } from '../src/llms/client.js'
import {
  ErrorClassifier,
  ErrorKind,
  classifyError,
} from '../src/runtime/index.js'

test('error classification maps built-in JavaScript failures, status codes, patterns, and retry hints', () => {
  const timeout = new Error('read timed out')
  timeout.name = 'TimeoutError'
  expect(classifyError(timeout)).toMatchObject({ kind: ErrorKind.TIMEOUT, retryable: true })
  expect(classifyError({ status: 429, message: 'slow down; Retry-After: 12.5' })).toMatchObject({
    kind: ErrorKind.RATE_LIMIT,
    retryable: true,
    suggestedBackoffSeconds: 12.5,
  })
  const structured = new Error('provider is busy') as Error & {
    details: { status: number; retryAfterSeconds: number }
  }
  structured.details = { status: 429, retryAfterSeconds: 7 }
  expect(classifyError(structured)).toMatchObject({
    kind: ErrorKind.RATE_LIMIT,
    retryable: true,
    suggestedBackoffSeconds: 7,
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

test('the completion deadline error classifies as a retryable timeout, not an unknown failure', () => {
  // completeLlm's default deadline raises this class on stalled upstreams.
  // Callers relying on the internal deadline (compaction fallbacks, plan
  // generators) need the same retry policy as TimeoutError-named failures.
  const deadline = new CompletionDeadlineError(180_000)
  expect(deadline.name).toBe('CompletionDeadlineError')
  expect(classifyError(deadline)).toMatchObject({
    kind: ErrorKind.TIMEOUT,
    retryable: true,
  })
  expect((deadline as Error).message).toContain('did not finish within 180000ms')
  expect(new ErrorClassifier().isRetryable(deadline)).toBeTrue()
})

test('a spliced stream frame is a transient fault the loop retries, whatever its dump contains', async () => {
  const { StreamFrameError } = await import('../src/core/errors.js')
  // Captured from z.ai: a frame cut off mid-id with a fresh stream spliced on.
  const frame = '{"id":"202609250533140b631f052data:{"id":"20260925053438095d1c327ea0470e","created":1790285685,"object":"chat.completion.chunk","index":401,"choices":[{"index":0,"delta":{"role":"assistan'
  const error = new StreamFrameError('zhipu', `invalid SSE JSON: ${frame}`, new SyntaxError('JSON Parse error'))
  expect(classifyError(error)).toMatchObject({ kind: ErrorKind.TRANSIENT, retryable: true })
})
