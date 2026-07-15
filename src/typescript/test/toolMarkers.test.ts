// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  extractAssistantToolCallMarkers,
  stripAssistantToolCallMarkers,
} from '../src/streaming/toolMarkers.js'

test('tool marker extraction removes JSON payloads and provider context from visible text', () => {
  const result = extractAssistantToolCallMarkers([
    'I will inspect the project.',
    'ASSISTANT_TOOL_CALLS: [{"id":"call_1","function":{"name":"ReadFile","arguments":"{\\"path\\":\\"README.md\\"}"}}]',
    '<system-reminder>hidden provider prompt</system-reminder>',
    'TOOL_CALL_ID: call_1',
    'TOOL: {"hidden":true}',
    'Then I will report the result.',
  ].join('\n'))

  expect(result).toEqual({
    text: 'I will inspect the project.\nThen I will report the result.',
    toolCalls: [{
      id: 'call_1',
      name: 'ReadFile',
      input: { path: 'README.md' },
    }],
  })
})

test('tool marker extraction normalizes invoke blocks and decodes parameter values', () => {
  const result = extractAssistantToolCallMarkers([
    'Starting.',
    '<invoke name="WriteFile">',
    '<parameter name="path">&quot;notes.md&quot;</parameter>',
    '<parameter name="content">hello &amp; goodbye</parameter>',
    '</invoke>',
    'Done.',
  ].join('\n'), 'call_cc')

  expect(result).toEqual({
    text: 'Starting.\nDone.',
    toolCalls: [{
      id: 'call_cc_0',
      name: 'WriteFile',
      input: { path: 'notes.md', content: 'hello & goodbye' },
    }],
  })
  expect(stripAssistantToolCallMarkers('hello ASSISTANT_TOOL_CALLS: {"name":"ListDir","input":{}}')).toBe('hello')
})

test('invalid marker payloads remain visible rather than silently creating malformed calls', () => {
  const result = extractAssistantToolCallMarkers('ASSISTANT_TOOL_CALLS: {"name": invalid}')
  expect(result).toEqual({ text: 'ASSISTANT_TOOL_CALLS: {"name": invalid}', toolCalls: [] })
})
