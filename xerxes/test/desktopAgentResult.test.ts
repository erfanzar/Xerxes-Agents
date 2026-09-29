// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { agentResult, resultLabel } from '../src/desktop/renderer/agentResult.js'

test('a complete structured result reads as its fields', () => {
  const result = agentResult('{"slice":"softmax_f16","measured":["probe.py","SEGMENTS 5"],"speedup":1.4}')
  expect(result).toEqual({ fields: { slice: 'softmax_f16', measured: ['probe.py', 'SEGMENTS 5'], speedup: 1.4 }, complete: true })
})

test('a saved excerpt keeps the fields that survived and says it is incomplete', () => {
  const whole = JSON.stringify({ slice: 'gpt2_f16', current_path: 'The whole scan is one segment.', measured: ['cd repo; python probe.py', 'SEGMENTS 5, graph_steps 3'], root_cause: 'widening' })
  const excerpt = `${whole.slice(0, whole.indexOf('graph_steps'))}…`
  const result = agentResult(excerpt)
  expect(result?.complete).toBe(false)
  expect(result?.fields).toMatchObject({ slice: 'gpt2_f16', current_path: 'The whole scan is one segment.' })
  expect(result?.fields).not.toHaveProperty('root_cause')
  // A cut inside a key drops that key rather than inventing a value.
  expect(agentResult('{"slice":"a","root_ca…')).toEqual({ fields: { slice: 'a' }, complete: false })
})

test('prose, arrays and empty objects are not structured results', () => {
  expect(agentResult('The slice is fine.')).toBeNull()
  expect(agentResult('["a","b"]')).toBeNull()
  expect(agentResult('{}')).toBeNull()
  expect(agentResult('{not json')).toBeNull()
})

test('field names read as labels', () => {
  expect(resultLabel('root_cause')).toBe('Root cause')
  expect(resultLabel('currentPath')).toBe('Current path')
  expect(resultLabel('slice')).toBe('Slice')
})
