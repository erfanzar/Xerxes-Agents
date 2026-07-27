// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { FunctionExecutionError, ValidationError } from '../src/core/errors.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { coerceDeclared, validateToolArguments } from '../src/runtime/argumentValidation.js'
import { optionalBoolean, optionalInteger } from '../src/tools/inputs.js'
import type { JsonObject } from '../src/types/toolCalls.js'

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

const coercionSchema = {
  type: 'object',
  properties: {
    offset: { type: 'integer' },
    ratio: { type: 'number' },
    replace_all: { type: 'boolean' },
    level: { type: 'integer', enum: [1, 2] },
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

test('validation accepts JSON object strings and rejects non-object or malformed provider payloads', () => {
  expect(validateToolArguments('ReadFile', '{"path":"a","mode":"read"}', schema)).toMatchObject({
    ok: true,
    coerced: { path: 'a', mode: 'read' },
  })
  expect(validateToolArguments('ReadFile', '{', schema).error).toContain('arguments are not valid JSON')
  expect(validateToolArguments('ReadFile', '[]', schema).error).toContain('expected arguments to be an object, got array')
})

test('declared-type coercion repairs exactly the literal provider forms and nothing looser', () => {
  // Only these exact spellings are repairs; everything else must stay untouched.
  expect(coerceDeclared('true', 'boolean')).toBe(true)
  expect(coerceDeclared('false', 'boolean')).toBe(false)
  expect(coerceDeclared('0', 'integer')).toBe(0)
  expect(coerceDeclared('-12', 'integer')).toBe(-12)
  expect(coerceDeclared('7', 'number')).toBe(7)
  expect(coerceDeclared('-1.5', 'number')).toBe(-1.5)

  // Truthiness-style spellings are NOT booleans: coercing them would run a tool with a
  // silently invented argument instead of reporting a bad call.
  for (const rejected of ['', ' ', '0', '1', 'yes', 'no', 'on', 'off', 'TRUE', 'True', ' true', 'true ']) {
    expect(coerceDeclared(rejected, 'boolean')).toBe(rejected)
  }
  for (const rejected of ['', ' ', '1.5', '1e3', '+1', ' 1', '1 ', '0x10', 'NaN', 'Infinity', '1,000', '--1']) {
    expect(coerceDeclared(rejected, 'integer')).toBe(rejected)
  }
  for (const rejected of ['', '.5', '5.', '1e3', '+1.5', 'NaN', 'Infinity', '1.2.3']) {
    expect(coerceDeclared(rejected, 'number')).toBe(rejected)
  }

  // Digits past the safe range parse to a different integer, so they stay a rejection.
  expect(coerceDeclared('9007199254740993', 'integer')).toBe('9007199254740993')
  expect(coerceDeclared('9007199254740991', 'integer')).toBe(9_007_199_254_740_991)

  // Non-strings and undeclared/other types are never rewritten.
  expect(coerceDeclared(null, 'boolean')).toBeNull()
  expect(coerceDeclared(undefined, 'integer')).toBeUndefined()
  expect(coerceDeclared(1, 'boolean')).toBe(1)
  expect(coerceDeclared('true', 'string')).toBe('true')
  expect(coerceDeclared('1', 'string')).toBe('1')
  expect(coerceDeclared('1', 'array')).toBe('1')
  expect(coerceDeclared('1', 'unknown-type')).toBe('1')
})

test('validation coerces declared literals in place and still rejects the near-miss spellings', () => {
  const repaired = validateToolArguments('Edit', {
    offset: '0',
    ratio: '1.5',
    replace_all: 'true',
    note: 'untyped stays as written',
  }, coercionSchema)
  expect(repaired.ok).toBe(true)
  expect(repaired.coerced).toEqual({ offset: 0, ratio: 1.5, replace_all: true, note: 'untyped stays as written' })

  // Enum membership is checked against the coerced value, not the raw string.
  expect(validateToolArguments('Edit', { level: '2' }, coercionSchema)).toMatchObject({ ok: true, coerced: { level: 2 } })
  expect(validateToolArguments('Edit', { level: '3' }, coercionSchema).error).toContain("parameter 'level'")

  for (const rejected of ['', 'yes', 'on', '0', '1', null]) {
    expect(validateToolArguments('Edit', { replace_all: rejected }, coercionSchema).ok).toBe(false)
  }
  for (const rejected of ['', '1.5', '1e3', ' 1', null, true]) {
    expect(validateToolArguments('Edit', { offset: rejected }, coercionSchema).ok).toBe(false)
  }
  expect(validateToolArguments('Edit', { offset: '1.5' }, coercionSchema).error)
    .toContain("parameter 'offset' expected integer, got string")

  // An untouched payload is reported as-is rather than needlessly copied.
  const clean: JsonObject = { offset: 3, replace_all: false }
  expect(validateToolArguments('Edit', clean, coercionSchema).coerced).toBe(clean)
})

test('tool registry hands the handler the coerced payload, not the raw provider strings', async () => {
  const registry = new ToolRegistry()
  let seen: JsonObject | undefined
  registry.register({
    type: 'function',
    function: { name: 'edit', description: 'edit', parameters: coercionSchema },
  }, inputs => {
    seen = inputs
    return 'ok'
  })

  expect(await registry.execute({
    id: 'call-coerce',
    type: 'function',
    function: { name: 'edit', arguments: { offset: '10', replace_all: 'false' } },
  }, { metadata: {} })).toBe('ok')
  expect(seen).toEqual({ offset: 10, replace_all: false })

  await expect(registry.execute({
    id: 'call-reject',
    type: 'function',
    function: { name: 'edit', arguments: { replace_all: 'yes' } },
  }, { metadata: {} })).rejects.toThrow("parameter 'replace_all' expected boolean, got string")
})

test('input helpers apply the same literal repair for handlers invoked without a schema', () => {
  expect(optionalBoolean({ flag: 'true' }, 'flag', false)).toBe(true)
  expect(optionalBoolean({ flag: 'false' }, 'flag', true)).toBe(false)
  expect(optionalBoolean({}, 'flag', true)).toBe(true)
  expect(optionalInteger({ limit: '25' }, 'limit', 5)).toBe(25)
  expect(optionalInteger({}, 'limit', 5)).toBe(5)

  for (const rejected of ['', '0', '1', 'yes', 'on', null]) {
    expect(() => optionalBoolean({ flag: rejected }, 'flag', false)).toThrow(ValidationError)
  }
  for (const rejected of ['', '1.5', 'ten', null, true]) {
    expect(() => optionalInteger({ limit: rejected }, 'limit', 5)).toThrow(ValidationError)
  }
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
