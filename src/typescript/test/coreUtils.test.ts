// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  assertStrictObject,
  debugPrint,
  defineCallableSchema,
  estimateMessagesTokens,
  estimateTokens,
  formatDebugLine,
  functionToJson,
  getCallablePublicName,
  mergeChunk,
  mergeFields,
  runAsync,
  safeJsonStringify,
  toJsonValue,
} from '../src/core/utils.js'

test('runAsync preserves the native await boundary without synchronous event-loop blocking', async () => {
  expect(await runAsync(42)).toBe(42)
  expect(await runAsync(Promise.resolve('ready'))).toBe('ready')
  await expect(runAsync(Promise.reject(new Error('failed')))).rejects.toThrow('failed')
})

test('debug output is deterministic through injected clock and sink', () => {
  const lines: string[] = []
  const output = {
    now: () => new Date(2026, 6, 13, 9, 8, 7),
    write: (line: string) => lines.push(line),
  }
  debugPrint(false, output, 'not written')
  debugPrint(true, output, 'hello', 'world')

  expect(lines).toEqual(['\u001b[97m[\u001b[90m2026-07-13 09:08:07\u001b[97m]\u001b[90m hello world\u001b[0m'])
  expect(formatDebugLine(['x'], new Date(2026, 0, 2, 3, 4, 5))).toContain('2026-01-02 03:04:05')
})

test('stream-field and chunk merging preserve accumulator semantics without mutating incoming deltas', () => {
  const target: Record<string, unknown> = {
    content: 'Hello',
    nested: { value: 'one' },
  }
  mergeFields(target, { content: ' world', nested: { value: ' two' }, ignored: null })
  expect(target).toEqual({ content: 'Hello world', nested: { value: 'one two' } })

  const response: Record<string, unknown> = {
    content: 'Hello',
    tool_calls: [{ id: 'call_1', function: { arguments: '{"path":"', name: 'read_file' } }],
  }
  const delta = {
    content: ' world',
    role: 'assistant',
    tool_calls: [{ index: 0, function: { arguments: 'README.md"}' } }],
  }
  mergeChunk(response, delta)

  expect(response).toEqual({
    content: 'Hello world',
    tool_calls: [{ id: 'call_1', function: { arguments: '{"path":"README.md"}', name: 'read_file' } }],
  })
  expect(delta).toEqual({
    content: ' world',
    role: 'assistant',
    tool_calls: [{ index: 0, function: { arguments: 'README.md"}' } }],
  })
})

test('token estimates retain source heuristics and reject invalid divisors', () => {
  expect(estimateTokens('')).toBe(0)
  expect(estimateTokens('abcdefgh', 2)).toBe(4)
  expect(estimateMessagesTokens([
    { content: 'Hello', role: 'user' },
    { content: 'Hi there', role: 'assistant' },
  ])).toBe(11)
  expect(() => estimateTokens('text', 0)).toThrow('charsPerToken must be a positive finite number')
})

test('callable metadata builds safe typed schemas and skips hidden context variables', () => {
  function greet(_name: string, _age = 0): string {
    return 'hello'
  }

  const schema = functionToJson(greet, {
    description: 'Greet a person.',
    parameters: [
      { name: 'name', type: 'string', description: 'Person name.' },
      { name: 'age', type: 'integer', defaultValue: 0 },
      { name: 'context_variables', type: 'object' },
      { name: 'tags', type: 'array', items: 'string', required: false },
    ],
  })

  expect(schema).toEqual({
    type: 'function',
    function: {
      name: 'greet',
      description: 'Greet a person.',
      parameters: {
        type: 'object',
        properties: {
          name: { type: 'string', description: 'Person name.' },
          age: { type: 'integer', default: 0 },
          tags: { type: 'array', items: { type: 'string' } },
        },
        required: ['name'],
      },
    },
  })

  function custom(): void {}
  defineCallableSchema(custom, {
    name: 'custom_tool',
    description: 'Hand-tuned schema.',
    parameters: { type: 'object', properties: {}, required: [] },
  })
  expect(getCallablePublicName(custom)).toBe('custom_tool')
  expect(functionToJson(custom, { description: 'Source docs.' })).toEqual({
    type: 'function',
    function: {
      name: 'custom_tool',
      description: 'Hand-tuned schema.\n\nSource docs.',
      parameters: { type: 'object', properties: {}, required: [] },
    },
  })
})

test('JSON conversion and strict-object checks reject lossy or unexpected input', () => {
  const source = { nested: ['value', 2] }
  expect(toJsonValue(source)).toEqual(source)
  expect(safeJsonStringify({ b: true, a: 1 })).toBe('{"b":true,"a":1}')
  expect(() => toJsonValue({ number: Number.NaN })).toThrow('numbers must be finite JSON values')
  expect(() => assertStrictObject({ known: true, extra: false }, ['known'])).toThrow('Unexpected field: extra')
})
