// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { AnthropicMessagesClient } from '../src/llms/anthropic.js'
import { OpenAiCompatibleClient, type CompletionRequest, type LlmDelta } from '../src/llms/client.js'
import {
  canonicalizeToolCallArguments,
  deterministicToolCallId,
} from '../src/streaming/toolCallIds.js'

test('canonical tool-call arguments sort object keys and preserve JSON structure', () => {
  const first = canonicalizeToolCallArguments({
    b: 2,
    a: { d: true, c: ['μ', null] },
  })
  const second = canonicalizeToolCallArguments({
    a: { c: ['μ', null], d: true },
    b: 2,
  })

  expect(first).toBe('{"a": {"c": ["μ", null], "d": true}, "b": 2}')
  expect(second).toBe(first)
})

test('canonical tool-call arguments safely encode non-JSON values without throwing', () => {
  class Exotic {
    toString(): string {
      return 'Exotic'
    }
  }

  const circular: Record<string, unknown> = {}
  circular.self = circular
  const guarded: Record<string, unknown> = {}
  Object.defineProperty(guarded, 'unsafe', {
    enumerable: true,
    get: () => {
      throw new Error('must not be evaluated')
    },
  })
  const input = {
    bigint: 12n,
    circular,
    exotic: new Exotic(),
    guarded,
    missing: undefined,
    values: new Set(['b', 'a']),
  }

  const canonical = canonicalizeToolCallArguments(input)
  expect(canonicalizeToolCallArguments(input)).toBe(canonical)
  expect(() => JSON.parse(canonical)).not.toThrow()
  expect(canonical).toContain('"$xerxes_type": "bigint"')
  expect(canonical).toContain('"$xerxes_type": "circular"')
  expect(canonical).toContain('"$xerxes_type": "accessor"')
  expect(canonical).toContain('"value": "Exotic"')
  expect(canonical).toContain('"values": ["a", "b"]')
})

test('deterministic tool-call IDs are stable, argument-order independent, and configurable', () => {
  const first = deterministicToolCallId('ReadFile', { path: 'README.md', recursive: false })
  const second = deterministicToolCallId('ReadFile', { recursive: false, path: 'README.md' })
  const differentName = deterministicToolCallId('ListDir', { path: 'README.md', recursive: false })
  const differentArguments = deterministicToolCallId('ReadFile', { path: 'src' })

  expect(first).toBe(second)
  expect(first).toMatch(/^call_[0-9a-f]{16}$/)
  expect(first).not.toBe(differentName)
  expect(first).not.toBe(differentArguments)
  expect(deterministicToolCallId('ReadFile', { path: 'README.md' })).toBe('call_4e74834c1ef17d1d')
  expect(deterministicToolCallId('ReadFile', { path: 'README.md' }, { length: 8, prefix: 'tc_' }))
    .toBe('tc_4e74834c')
})

test('provider stream fallbacks use deterministic IDs only when the provider omitted one', async () => {
  const openAi = new OpenAiCompatibleClient({
    apiKey: 'test-key',
    baseUrl: 'https://example.invalid/v1',
    fetchImplementation: async () => sseResponse([{
      choices: [{
        delta: {
          tool_calls: [
            { index: 0, id: 'provider-openai-id', function: { name: 'ListDir', arguments: '{"path":"."}' } },
            { index: 1, function: { name: 'ReadFile', arguments: '{"path":"README.md"}' } },
          ],
        },
        finish_reason: 'tool_calls',
      }],
    }]),
    providerName: 'openai',
  })
  const anthropic = new AnthropicMessagesClient({
    apiKey: 'test-key',
    fetchImplementation: async () => sseResponse([
      {
        type: 'content_block_start',
        index: 0,
        content_block: { type: 'tool_use', id: 'provider-anthropic-id', name: 'ListDir', input: { path: '.' } },
      },
      {
        type: 'content_block_start',
        index: 1,
        content_block: { type: 'tool_use', name: 'ReadFile', input: { path: 'README.md' } },
      },
      { type: 'message_stop' },
    ]),
  })

  const openAiEvents = await collect(openAi, openAiRequest())
  const anthropicEvents = await collect(anthropic, anthropicRequest())
  const fallbackId = deterministicToolCallId('ReadFile', { path: 'README.md' })

  expect(openAiEvents).toContainEqual({
    finishReason: 'tool_calls',
    toolCalls: [
      { id: 'provider-openai-id', type: 'function', function: { name: 'ListDir', arguments: { path: '.' } } },
      { id: fallbackId, type: 'function', function: { name: 'ReadFile', arguments: { path: 'README.md' } } },
    ],
  })
  expect(anthropicEvents).toContainEqual({
    toolCalls: [
      { id: 'provider-anthropic-id', type: 'function', function: { name: 'ListDir', arguments: { path: '.' } } },
      { id: fallbackId, type: 'function', function: { name: 'ReadFile', arguments: { path: 'README.md' } } },
    ],
  })
})

function openAiRequest(): CompletionRequest {
  return { model: 'gpt-4o', messages: [{ role: 'user', content: 'inspect files' }] }
}

function anthropicRequest(): CompletionRequest {
  return { model: 'claude-sonnet-4-6', messages: [{ role: 'user', content: 'inspect files' }] }
}

async function collect(
  client: { stream(request: CompletionRequest): AsyncIterable<LlmDelta> },
  request: CompletionRequest,
): Promise<LlmDelta[]> {
  const events: LlmDelta[] = []
  for await (const event of client.stream(request)) {
    events.push(event)
  }
  return events
}

function sseResponse(events: readonly Record<string, unknown>[]): Response {
  const body = events.map(event => `data: ${JSON.stringify(event)}\n\n`).join('') + 'data: [DONE]\n\n'
  return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } })
}
