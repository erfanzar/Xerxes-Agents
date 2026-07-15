// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { messageToOpenAi } from '../src/types/messages.js'

test('OpenAI conversion retains assistant reasoning when a tool call follows', () => {
  expect(messageToOpenAi({
    role: 'assistant',
    content: 'I need to inspect the file.',
    thinking: 'Read the target first.',
    tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'ReadFile', arguments: { path: 'README.md' } } }],
  })).toEqual({
    role: 'assistant',
    content: 'I need to inspect the file.',
    reasoning_content: 'Read the target first.',
    tool_calls: [{
      id: 'call-1',
      type: 'function',
      function: { name: 'ReadFile', arguments: '{"path":"README.md"}' },
    }],
  })
})

test('empty thinking is not sent as a provider reasoning field', () => {
  expect(messageToOpenAi({ role: 'assistant', content: 'ok', thinking: '' })).toEqual({ role: 'assistant', content: 'ok' })
})
