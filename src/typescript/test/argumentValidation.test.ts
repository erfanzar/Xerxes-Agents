// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { FunctionExecutionError } from '../src/core/errors.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { validateAndFormatToolArgumentError, validateToolArguments } from '../src/runtime/argumentValidation.js'

const schema = {
  type: 'object',
  additionalProperties: false,
  required: ['path', 'mode'],
  properties: {
    path: { type: 'string' },
    mode: { type: 'string', enum: ['read', 'write'] },
    retries: { type: 'integer' },
  },
} as const

test('tool argument validation reports required, unknown, type, and enum mistakes before execution', () => {
  expect(validateToolArguments('ReadFile', {}, schema)).toMatchObject({
    ok: false,
    missing: ['path', 'mode'],
    error: 'ReadFile: missing required parameter(s): path, mode',
  })
  expect(validateToolArguments('ReadFile', { path: 1, mode: 'read' }, schema).error).toContain("parameter 'path' expected string")
  expect(validateToolArguments('ReadFile', { path: 'a', mode: 'delete' }, schema).error).toContain("parameter 'mode' must be read or write")
  expect(validateToolArguments('ReadFile', { path: 'a', mode: 'read', extra: true }, schema).error).toContain("unknown parameter 'extra'")
  expect(validateToolArguments('ReadFile', { path: 'a', mode: 'read', retries: 2 }, schema)).toMatchObject({ ok: true, error: '' })
})

test('formatted validation accepts JSON object strings and rejects non-object or malformed provider payloads', () => {
  expect(validateAndFormatToolArgumentError('ReadFile', '{"path":"a","mode":"read"}', schema)).toBeUndefined()
  expect(validateAndFormatToolArgumentError('ReadFile', '{', schema)).toContain('arguments are not valid JSON')
  expect(validateAndFormatToolArgumentError('ReadFile', '[]', schema)).toBe('ReadFile: arguments must be a JSON object.')
})

test('tool registry validates its selected per-agent definition before calling a handler', async () => {
  const registry = new ToolRegistry()
  let calls = 0
  registry.register({
    type: 'function',
    function: { name: 'read', description: 'read', parameters: schema },
  }, () => {
    calls += 1
    return 'unreachable'
  })
  await expect(registry.execute({
    id: 'call-1',
    type: 'function',
    function: { name: 'read', arguments: { path: 'a' } },
  }, { metadata: {} })).rejects.toBeInstanceOf(FunctionExecutionError)
  expect(calls).toBe(0)
})
