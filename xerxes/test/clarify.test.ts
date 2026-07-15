// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { clarify, registerClarifyTool, StaticAsker } from '../src/tools/clarify.js'
import type { ToolCall } from '../src/types/toolCalls.js'

test('clarify reports an honest needs-ui result without a user interaction adapter', async () => {
  expect(await clarify({ question: 'Which deployment region?' })).toEqual({
    ok: true,
    answered: false,
    needs_ui: true,
  })
  expect(await clarify({ question: '  ', choices: ['a'] })).toEqual({ ok: false, error: 'question required' })
  expect(await clarify({ question: 'Pick', allowFreeform: false })).toEqual({
    ok: false,
    error: 'either options must be supplied or allow_freeform=true',
  })
})

test('clarify tool preserves selected indices and static skipped replies', async () => {
  const registry = new ToolRegistry()
  registerClarifyTool(registry, { asker: new StaticAsker({ index: 1 }) })
  const output = JSON.parse(await registry.execute({
    id: 'clarify',
    type: 'function',
    function: { name: 'clarify', arguments: { question: 'Pick', options: ['one', 'two'] } },
  } satisfies ToolCall, { metadata: {} }))
  expect(output).toEqual({
    ok: true,
    answered: true,
    answer: 'two',
    selected_index: 1,
    skipped: false,
  })
  expect(await clarify({ question: 'Skip?', asker: new StaticAsker({ skip: true }) })).toEqual({
    ok: true,
    answered: false,
    answer: '',
    selected_index: null,
    skipped: true,
  })
})
