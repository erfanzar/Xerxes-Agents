// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  messagesFromAnthropic,
  messagesFromOpenAi,
  messagesToAnthropic,
  messagesToAnthropicPayload,
  messagesToOpenAi,
  normalizeOpenAiToolArguments,
} from '../src/streaming/messages.js'

test('OpenAI conversion preserves multimodal content, reasoning, and canonical tool links', () => {
  const converted = messagesToOpenAi([
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Inspect this image.' },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,aGVsbG8=', detail: 'high' } },
      ],
    },
    {
      role: 'assistant',
      content: '',
      thinking: 'I need the file first.',
      tool_calls: [{
        id: 'call-1',
        type: 'function',
        function: { name: 'ReadFile', arguments: { path: 'README.md' } },
      }],
    },
    { role: 'tool', tool_call_id: 'call-1', name: 'ReadFile', content: 'contents' },
  ], 'Follow the project instructions.')

  expect(converted).toEqual([
    { role: 'system', content: 'Follow the project instructions.' },
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Inspect this image.' },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,aGVsbG8=', detail: 'high' } },
      ],
    },
    {
      role: 'assistant',
      content: null,
      reasoning_content: 'I need the file first.',
      tool_calls: [{
        id: 'call-1',
        type: 'function',
        function: { name: 'ReadFile', arguments: '{"path":"README.md"}' },
      }],
    },
    { role: 'tool', tool_call_id: 'call-1', name: 'ReadFile', content: 'contents' },
  ])
})

test('OpenAI decoding normalizes tool arguments and keeps multimodal user content', () => {
  const decoded = messagesFromOpenAi([
    { role: 'system', content: 'system context' },
    {
      role: 'user',
      content: [
        { type: 'text', text: 'look' },
        { type: 'image_url', image_url: { url: 'https://example.test/image.png' } },
      ],
    },
    {
      role: 'assistant',
      content: null,
      reasoning_content: 'Reason before calling.',
      tool_calls: [
        { id: 'call-json', type: 'function', function: { name: 'ReadFile', arguments: '{"path":"README.md"}' } },
        { id: 'call-empty', type: 'function', function: { name: 'NoArgs', arguments: null } },
      ],
    },
    {
      role: 'tool',
      tool_call_id: 'call-json',
      content: [
        { type: 'text', text: 'done' },
        { type: 'image_url', image_url: { url: 'https://example.test/out.png' } },
      ],
    },
  ])

  expect(decoded).toEqual([
    { role: 'user', content: 'system context' },
    {
      role: 'user',
      content: [
        { type: 'text', text: 'look' },
        { type: 'image_url', image_url: { url: 'https://example.test/image.png' } },
      ],
    },
    {
      role: 'assistant',
      content: '',
      thinking: 'Reason before calling.',
      tool_calls: [
        { id: 'call-json', type: 'function', function: { name: 'ReadFile', arguments: { path: 'README.md' } } },
        { id: 'call-empty', type: 'function', function: { name: 'NoArgs', arguments: {} } },
      ],
    },
    { role: 'tool', tool_call_id: 'call-json', content: 'done[Image: https://example.test/out.png]' },
  ])

  expect(messagesFromOpenAi([{ role: 'system', content: 'system context' }], { preserveSystemRole: true }))
    .toEqual([{ role: 'system', content: 'system context' }])
  expect(normalizeOpenAiToolArguments(null)).toEqual({})
  expect(normalizeOpenAiToolArguments({ path: 'README.md' })).toEqual({ path: 'README.md' })
  expect(() => normalizeOpenAiToolArguments('[]')).toThrow('must decode to a JSON object')
  expect(() => normalizeOpenAiToolArguments('{')).toThrow('must be valid JSON')
})

test('Anthropic conversion groups tool results, replays signed thinking, and converts data images', () => {
  const messages = [
    { role: 'system' as const, content: 'System one.' },
    {
      role: 'user' as const,
      content: [
        { type: 'text' as const, text: 'Inspect both.' },
        { type: 'image_url' as const, image_url: { url: 'data:image/png;base64,aGVsbG8=' } },
        { type: 'image_url' as const, image_url: { url: 'https://example.test/remote.png' } },
      ],
    },
    {
      role: 'assistant' as const,
      content: 'Calling tools.',
      thinking: 'Use the two tools.',
      thinking_signature: 'signature-1',
      tool_calls: [{
        id: 'call-1',
        type: 'function' as const,
        function: { name: 'ReadFile', arguments: { path: 'README.md' } },
      }],
    },
    { role: 'tool' as const, tool_call_id: 'call-1', content: 'ok' },
    { role: 'tool' as const, tool_call_id: 'call-2', content: 'failed', is_error: true },
  ]

  expect(messagesToAnthropic(messages)).toEqual([
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Inspect both.' },
        { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'aGVsbG8=' } },
        { type: 'text', text: '[Image: https://example.test/remote.png]' },
      ],
    },
    {
      role: 'assistant',
      content: [
        { type: 'thinking', thinking: 'Use the two tools.', signature: 'signature-1' },
        { type: 'text', text: 'Calling tools.' },
        { type: 'tool_use', id: 'call-1', name: 'ReadFile', input: { path: 'README.md' } },
      ],
    },
    {
      role: 'user',
      content: [
        { type: 'tool_result', tool_use_id: 'call-1', content: 'ok' },
        { type: 'tool_result', tool_use_id: 'call-2', content: 'failed', is_error: true },
      ],
    },
  ])
  expect(messagesToAnthropicPayload(messages)).toEqual({
    system: 'System one.',
    messages: messagesToAnthropic(messages),
  })
  expect(messagesToAnthropic([{ role: 'assistant', content: 'No signature.', thinking: 'not replayable' }]))
    .toEqual([{ role: 'assistant', content: [{ type: 'text', text: 'No signature.' }] }])
})

test('Anthropic decoding restores thinking, tool results, and multimodal data URLs', () => {
  const decoded = messagesFromAnthropic([
    {
      role: 'assistant',
      content: [
        { type: 'thinking', thinking: 'Inspect first.', signature: 'signature-1' },
        { type: 'text', text: 'Calling it.' },
        { type: 'tool_use', id: 'call-1', name: 'ReadFile', input: { path: 'README.md' } },
      ],
    },
    {
      role: 'user',
      content: [{
        type: 'tool_result',
        tool_use_id: 'call-1',
        name: 'ReadFile',
        content: [
          { type: 'text', text: 'permission denied' },
          { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'aGVsbG8=' } },
        ],
        is_error: true,
      }],
    },
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Here is the screenshot.' },
        { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: 'aW1hZ2U=' } },
      ],
    },
  ])

  expect(decoded).toEqual([
    {
      role: 'assistant',
      content: 'Calling it.',
      thinking: 'Inspect first.',
      thinking_signature: 'signature-1',
      tool_calls: [{
        id: 'call-1',
        type: 'function',
        function: { name: 'ReadFile', arguments: { path: 'README.md' } },
      }],
    },
    {
      role: 'tool',
      tool_call_id: 'call-1',
      name: 'ReadFile',
      content: 'permission denied\n[Image: data:image/png;base64,aGVsbG8=]',
      is_error: true,
    },
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Here is the screenshot.' },
        { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,aW1hZ2U=' } },
      ],
    },
  ])
})

test('Anthropic decoding concatenates multiple thinking blocks and keeps the last signature', () => {
  const decoded = messagesFromAnthropic([
    {
      role: 'assistant',
      content: [
        { type: 'thinking', thinking: 'First step.', signature: 'signature-1' },
        { type: 'text', text: 'Working.' },
        { type: 'thinking', thinking: 'Second step.', signature: 'signature-2' },
        { type: 'tool_use', id: 'call-1', name: 'ReadFile', input: { path: 'README.md' } },
      ],
    },
  ])

  expect(decoded).toEqual([{
    role: 'assistant',
    content: 'Working.',
    thinking: 'First step.\nSecond step.',
    thinking_signature: 'signature-2',
    tool_calls: [{
      id: 'call-1',
      type: 'function',
      function: { name: 'ReadFile', arguments: { path: 'README.md' } },
    }],
  }])
})
