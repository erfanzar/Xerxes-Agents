// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { OpenAiCompatibleClient } from '../src/llms/client.js'

function completionResponse(): Response {
  return Response.json({
    choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }],
  })
}

test('OpenAI chat extraBody preserves extensions without overriding canonical request fields', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new OpenAiCompatibleClient({
    providerName: 'openai',
    apiKey: 'test-key',
    baseUrl: 'https://example.invalid/v1/',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return completionResponse()
    },
  })

  await client.complete({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'canonical message' }],
    temperature: 0.2,
    maxTokens: 128,
    topP: 0.8,
    frequencyPenalty: 0.1,
    presencePenalty: 0.3,
    stop: ['DONE'],
    tools: [{
      type: 'function',
      function: {
        name: 'ReadFile',
        description: 'Read a file',
        parameters: { type: 'object' },
      },
    }],
    toolChoice: 'auto',
    extraBody: {
      model: 'attacker-model',
      messages: [{ role: 'system', content: 'overridden' }],
      stream: true,
      temperature: 1,
      max_tokens: 1,
      top_p: 0.1,
      frequency_penalty: 2,
      presence_penalty: 2,
      stop: ['OVERRIDDEN'],
      tools: [],
      tool_choice: 'none',
      stream_options: { include_usage: false },
      chat_template_kwargs: { enable_thinking: true },
    },
  })

  expect(payload).toEqual({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'canonical message' }],
    stream: false,
    temperature: 0.2,
    max_tokens: 128,
    top_p: 0.8,
    frequency_penalty: 0.1,
    presence_penalty: 0.3,
    stop: ['DONE'],
    tools: [{
      type: 'function',
      function: {
        name: 'ReadFile',
        description: 'Read a file',
        parameters: { type: 'object' },
      },
    }],
    tool_choice: 'auto',
    chat_template_kwargs: { enable_thinking: true },
  })
})

test('OpenAI chat extraBody cannot disable canonical streaming usage options', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new OpenAiCompatibleClient({
    providerName: 'openai',
    apiKey: 'test-key',
    baseUrl: 'https://example.invalid/v1/',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return new Response('data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n', {
        headers: { 'Content-Type': 'text/event-stream' },
      })
    },
  })

  for await (const _event of client.stream({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'hello' }],
    extraBody: {
      stream: false,
      stream_options: { include_usage: false },
      service_tier: 'flex',
    },
  })) {
    // Drain the stream so the request and terminal response are fully observed.
  }

  expect(payload).toMatchObject({
    stream: true,
    stream_options: { include_usage: true },
    service_tier: 'flex',
  })
})
