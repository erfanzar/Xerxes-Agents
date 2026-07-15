// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  AgentError,
  ClientError,
  ConfigurationError,
  FunctionExecutionError,
  RateLimitError,
  ValidationError,
  XerxesError,
  XerxesMemoryError,
  XerxesTimeoutError,
} from '../src/core/errors.js'

test('core errors retain message, immutable details, and typed public metadata', () => {
  const base = new XerxesError('something went wrong', { key: 'value' })
  expect(base).toBeInstanceOf(Error)
  expect(base.message).toBe('something went wrong')
  expect(base.details).toEqual({ key: 'value' })
  expect(Object.isFrozen(base.details)).toBe(true)

  const agent = new AgentError('agent-1', 'failed to execute')
  expect(agent.message).toContain('agent-1')
  expect(agent.agentId).toBe('agent-1')

  const original = new Error('bad input')
  const functionError = new FunctionExecutionError('search', 'failed', original)
  expect(functionError.message).toContain('search')
  expect(functionError.functionName).toBe('search')
  expect(functionError.cause).toBe(original)

  const timeout = new XerxesTimeoutError('llm_call', 30)
  expect(timeout.message).toContain('30')
  expect(timeout.operation).toBe('llm_call')
  expect(timeout.timeout).toBe(30)
})

test('validation, rate-limit, memory, client, and configuration errors retain their payloads', () => {
  const validation = new ValidationError('age', 'must be positive', -1)
  expect(validation.message).toContain('age')
  expect(validation.field).toBe('age')
  expect(validation.value).toBe(-1)

  const rateLimit = new RateLimitError('api', 100, 'hour', 30)
  expect(rateLimit.message).toContain('30')
  expect(rateLimit).toMatchObject({ resource: 'api', limit: 100, window: 'hour', retryAfter: 30 })

  const memory = new XerxesMemoryError('store', 'disk full')
  expect(memory.message).toContain('store')
  expect(memory.operation).toBe('store')

  const original = new Error('connection reset')
  const client = new ClientError('anthropic', 'failed', original)
  expect(client.message).toContain('anthropic')
  expect(client.clientType).toBe('anthropic')
  expect(client.cause).toBe(original)

  const configuration = new ConfigurationError('llm.apiKey', 'missing')
  expect(configuration.message).toContain('llm.apiKey')
  expect(configuration.configKey).toBe('llm.apiKey')
})
