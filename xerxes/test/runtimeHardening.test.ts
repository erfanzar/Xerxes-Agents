// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ErrorKind, classifyError } from '../src/runtime/errorClassifier.js'
import { IterationBudget, iterationBudgetFromConfig } from '../src/runtime/iterationBudget.js'

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

test('present-but-invalid iteration budget configuration throws instead of silently disabling the budget', () => {
  for (const invalid of ['abc', 0, -3, '0', '-7', '3.5', 1.5, Number.NaN, 'ten']) {
    expect(() => iterationBudgetFromConfig({ max_tool_turns: invalid }))
      .toThrow(RangeError)
  }
  // An invalid environment fallback is a configuration error too.
  expect(() => iterationBudgetFromConfig({}, { environment: { XERXES_MAX_TOOL_TURNS: 'abc' } }))
    .toThrow(RangeError)
})

test('absent iteration budget configuration falls back to the environment, then unbounded', () => {
  expect(iterationBudgetFromConfig({}, { environment: {} })).toBeInstanceOf(IterationBudget)
  expect(iterationBudgetFromConfig({}, { environment: {} }).maxIterations).toBeUndefined()
  expect(iterationBudgetFromConfig({ max_tool_turns: '' }, { environment: { XERXES_MAX_TOOL_TURNS: '6' } })
    .maxIterations).toBe(6)
  expect(iterationBudgetFromConfig({ max_tool_turns: null }, { environment: {} }).maxIterations)
    .toBeUndefined()
  expect(iterationBudgetFromConfig({ max_tool_turns: '12' }).maxIterations).toBe(12)
})
