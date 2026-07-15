// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { AnthropicMessagesClient, messagesToAnthropic } from '../src/llms/anthropic.js'
import type { CompletionRequest } from '../src/llms/client.js'

test('Anthropic conversion preserves signed thinking and error tool results', () => {
  const converted = messagesToAnthropic([
    { role: 'system', content: 'Be concise.' },
    {
      role: 'assistant',
      content: 'I will inspect it.',
      thinking: 'Need a file read.',
      thinking_signature: 'signature-1',
      tool_calls: [{ id: 'call-1', type: 'function', function: { name: 'ReadFile', arguments: { path: 'README.md' } } }],
    },
    { role: 'tool', tool_call_id: 'call-1', name: 'ReadFile', content: 'permission denied', is_error: true },
  ])

  expect(converted).toEqual({
    system: 'Be concise.',
    messages: [
      {
        role: 'assistant',
        content: [
          { type: 'thinking', thinking: 'Need a file read.', signature: 'signature-1' },
          { type: 'text', text: 'I will inspect it.' },
          { type: 'tool_use', id: 'call-1', name: 'ReadFile', input: { path: 'README.md' } },
        ],
      },
      {
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: 'call-1', content: 'permission denied', is_error: true }],
      },
    ],
  })
})

test('Anthropic SSE adapter normalizes text, thinking, usage, and tool calls', async () => {
  const payload = [
    { type: 'message_start', message: { usage: { input_tokens: 11 } } },
    { type: 'content_block_start', index: 0, content_block: { type: 'thinking', signature: 'sig-1' } },
    { type: 'content_block_delta', index: 0, delta: { type: 'thinking_delta', thinking: 'Inspect.' } },
    { type: 'content_block_start', index: 1, content_block: { type: 'tool_use', id: 'tool-1', name: 'ReadFile', input: {} } },
    { type: 'content_block_delta', index: 1, delta: { type: 'input_json_delta', partial_json: '{"path":"README.md"}' } },
    { type: 'content_block_delta', index: 2, delta: { type: 'text_delta', text: 'Calling tool.' } },
    { type: 'message_delta', usage: { output_tokens: 7 } },
    { type: 'message_stop' },
  ]
  const response = sseResponse(payload)
  const client = new AnthropicMessagesClient({ apiKey: 'test-key', fetchImplementation: async () => response })
  const request: CompletionRequest = { model: 'claude-sonnet-4-6', messages: [{ role: 'user', content: 'read file' }] }
  const events = []
  for await (const event of client.stream(request)) {
    events.push(event)
  }

  expect(events).toContainEqual({ thinkingSignature: 'sig-1' })
  expect(events).toContainEqual({ thinking: 'Inspect.' })
  expect(events).toContainEqual({ content: 'Calling tool.' })
  expect(events).toContainEqual({ usage: { inputTokens: 11, outputTokens: 0 } })
  expect(events).toContainEqual({ usage: { inputTokens: 0, outputTokens: 7 } })
  expect(events).toContainEqual({
    toolCalls: [{ id: 'tool-1', type: 'function', function: { name: 'ReadFile', arguments: { path: 'README.md' } } }],
  })
})

test('Anthropic requests cache stable prefixes and retain cache token usage', async () => {
  let payload: Record<string, unknown> | undefined
  const response = sseResponse([
    {
      type: 'message_start',
      message: {
        usage: {
          cache_creation_input_tokens: 7,
          cache_read_input_tokens: 13,
          input_tokens: 21,
        },
      },
    },
    { type: 'message_delta', usage: { output_tokens: 5 } },
  ])
  const client = new AnthropicMessagesClient({
    apiKey: 'test-key',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return response
    },
  })
  const request: CompletionRequest = {
    model: 'claude-sonnet-4-6',
    messages: [{ role: 'system', content: 'Stable instructions.' }, { role: 'user', content: 'Use a tool.' }],
    tools: [{
      type: 'function',
      function: {
        name: 'ReadFile',
        description: 'Read one file.',
        parameters: { type: 'object', properties: {} },
      },
    }],
  }
  const events = []
  for await (const event of client.stream(request)) events.push(event)

  expect(payload?.system).toEqual([{
    type: 'text',
    text: 'Stable instructions.',
    cache_control: { type: 'ephemeral' },
  }])
  expect(payload?.tools).toEqual([expect.objectContaining({
    name: 'ReadFile',
    cache_control: { type: 'ephemeral' },
  })])
  expect(events).toContainEqual({
    usage: {
      inputTokens: 21,
      outputTokens: 0,
      cacheReadTokens: 13,
      cacheCreationTokens: 7,
    },
  })
})

test('Anthropic native completion uses a non-streaming request and retains thinking, tools, and cache usage', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new AnthropicMessagesClient({
    apiKey: 'test-key',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return Response.json({
        content: [
          { type: 'thinking', thinking: 'Inspect source.', signature: 'sig-1' },
          { type: 'text', text: 'I will read it.' },
          { type: 'tool_use', id: 'tool-1', name: 'ReadFile', input: { path: 'README.md' } },
        ],
        stop_reason: 'tool_use',
        usage: {
          input_tokens: 15,
          output_tokens: 8,
          cache_read_input_tokens: 4,
          cache_creation_input_tokens: 2,
        },
      })
    },
  })

  const completion = await client.complete({
    model: 'claude-sonnet-4-6',
    messages: [{ role: 'user', content: 'Read the README.' }],
  })

  expect(payload).toMatchObject({ model: 'claude-sonnet-4-6', stream: false })
  expect(completion).toEqual({
    content: 'I will read it.',
    thinking: 'Inspect source.',
    thinkingSignature: 'sig-1',
    finishReason: 'tool_use',
    usage: { inputTokens: 15, outputTokens: 8, cacheReadTokens: 4, cacheCreationTokens: 2 },
    toolCalls: [{ id: 'tool-1', type: 'function', function: { name: 'ReadFile', arguments: { path: 'README.md' } } }],
  })
})

function sseResponse(events: readonly Record<string, unknown>[]): Response {
  const encoder = new TextEncoder()
  return new Response(new ReadableStream({
    start(controller) {
      for (const event of events) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`))
      }
      controller.enqueue(encoder.encode('data: [DONE]\n\n'))
      controller.close()
    },
  }), { headers: { 'Content-Type': 'text/event-stream' } })
}
