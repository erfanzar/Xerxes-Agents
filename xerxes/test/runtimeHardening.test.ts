// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ErrorKind, classifyError, reportedContextLimit } from '../src/runtime/errorClassifier.js'

test('HTTP 400 with a context-overflow message classifies as context overflow, not bad request', () => {
  // OpenAI signals context overflow as HTTP 400 with this code in the body.
  expect(classifyError({
    status: 400,
    message: 'This model\'s maximum context length is 128000 tokens (context_length_exceeded)',
  })).toMatchObject({ kind: ErrorKind.CONTEXT_OVERFLOW, retryable: false })
  // Anthropic signals it as HTTP 400 with "prompt is too long".
  expect(classifyError({ status: 400, message: 'prompt is too long: 213000 tokens > 200000 maximum' }))
    .toMatchObject({ kind: ErrorKind.CONTEXT_OVERFLOW })
  expect(classifyError({ status: 400, message: 'request exceeds the available context window' }))
    .toMatchObject({ kind: ErrorKind.CONTEXT_OVERFLOW })
})

test("Kimi's model token limit refusal is a context overflow that states its window", () => {
  // The exact refusal that ended a goal round instead of compacting and retrying.
  const message = 'Client kimi-code: stream request failed (400): {"error":{"message":"Invalid request: Your request exceeded model token limit: 1048576 (requested: 1232443)","type":"invalid_request_error"}}'
  expect(classifyError({ status: 400, message })).toMatchObject({ kind: ErrorKind.CONTEXT_OVERFLOW })
  expect(reportedContextLimit(message)).toBe(1_048_576)
  expect(reportedContextLimit("This model's maximum context length is 128000 tokens")).toBe(128_000)
  expect(reportedContextLimit('prompt is too long: 213000 tokens > 200000 maximum')).toBe(200_000)
  // Numbers that are not a window are never read as one.
  expect(reportedContextLimit('Invalid request: max_tokens must be at most 32768')).toBeUndefined()
  expect(reportedContextLimit('rate limit: 429')).toBeUndefined()
})

test('HTTP 400 without a context-overflow message stays a plain bad request', () => {
  expect(classifyError({ status: 400, message: 'invalid request: unknown field "foo"' }))
    .toMatchObject({ kind: ErrorKind.BAD_REQUEST, retryable: false })
  expect(classifyError({ status: 400, message: 'malformed JSON body' }))
    .toMatchObject({ kind: ErrorKind.BAD_REQUEST })
})

test('other status codes keep their status-driven classification', () => {
  expect(classifyError({ status: 401, message: 'maximum context length reached' }))
    .toMatchObject({ kind: ErrorKind.AUTH })
  expect(classifyError({ status: 429, message: 'slow down' }))
    .toMatchObject({ kind: ErrorKind.RATE_LIMIT, retryable: true })
  expect(classifyError({ status: 503, message: 'prompt is too long' }))
    .toMatchObject({ kind: ErrorKind.PROVIDER_DOWN, retryable: true })
})
