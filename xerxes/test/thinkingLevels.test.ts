// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { AnthropicMessagesClient } from '../src/llms/anthropic.js'
import { OpenAiCompatibleClient, type CompletionRequest, type LlmDelta } from '../src/llms/client.js'
import { createAgentState, type StreamEvent } from '../src/streaming/events.js'
import { runTurn } from '../src/streaming/loop.js'
import {
  detectThinkingDirective,
  resolveTurnThinking,
  ULTRA_THINKING_DIRECTIVE,
} from '../src/runtime/thinkingLevels.js'

test('thinking keywords resolve the Claude ladder with longest phrase first', () => {
  expect(detectThinkingDirective('please think about this')?.budgetTokens).toBe(4_000)
  expect(detectThinkingDirective('think hard before editing')?.budgetTokens).toBe(10_000)
  expect(detectThinkingDirective('megathink the rollout')?.level).toBe('think_hard')
  expect(detectThinkingDirective('think harder about the proof')?.budgetTokens).toBe(20_000)
  expect(detectThinkingDirective('ULTRATHINK this architecture')?.budgetTokens).toBe(32_000)
  // Strongest phrase wins even when weaker keywords are also present.
  expect(detectThinkingDirective('think harder, not just think')?.level).toBe('think_harder')
  expect(detectThinkingDirective('ultrathink and think hard')?.level).toBe('ultrathink')
})

test('keyword detection is whole-word and case-insensitive', () => {
  expect(detectThinkingDirective('rethinking the plan')).toBeUndefined()
  expect(detectThinkingDirective('thinking is hard')).toBeUndefined()
  expect(detectThinkingDirective('a thinker thinks')).toBeUndefined()
  expect(detectThinkingDirective('Think Hard about it')?.level).toBe('think_hard')
  expect(detectThinkingDirective('')).toBeUndefined()
})

test('turn thinking precedence: ultra mode, then keyword, then session defaults, then off', () => {
  expect(resolveTurnThinking({ prompt: 'plain prompt', ultraMode: true })).toEqual(ULTRA_THINKING_DIRECTIVE)
  expect(resolveTurnThinking({ prompt: 'ultrathink this', ultraMode: false })?.level).toBe('ultrathink')
  expect(
    resolveTurnThinking({
      defaults: { budgetTokens: 24_576, effort: 'high' },
      prompt: 'plain prompt',
      ultraMode: false,
    }),
  ).toEqual({ budgetTokens: 24_576, effort: 'high', level: 'think_hard', matchedKeyword: 'session default' })
  expect(
    resolveTurnThinking({ defaults: { enabled: true }, prompt: 'plain', ultraMode: false })?.budgetTokens,
  ).toBe(10_000)
  expect(resolveTurnThinking({ defaults: { enabled: false, budgetTokens: 9_999 }, prompt: 'plain', ultraMode: false }))
    .toBeUndefined()
  expect(resolveTurnThinking({ prompt: 'plain prompt', ultraMode: false })).toBeUndefined()
  // A keyword still escalates over an explicitly disabled session default.
  expect(resolveTurnThinking({ defaults: { enabled: false }, prompt: 'think harder please', ultraMode: false })?.level)
    .toBe('think_harder')
})

test('session default effort: low is preserved and off-values disable thinking', () => {
  expect(resolveTurnThinking({ defaults: { effort: 'low' }, prompt: 'plain', ultraMode: false })?.effort).toBe('low')
  expect(resolveTurnThinking({ defaults: { budgetTokens: 8_000, effort: 'low' }, prompt: 'plain', ultraMode: false })?.effort)
    .toBe('low')
  expect(resolveTurnThinking({ defaults: { budgetTokens: 8_000, enabled: true, effort: 'off' }, prompt: 'plain', ultraMode: false }))
    .toBeUndefined()
  expect(resolveTurnThinking({ defaults: { effort: 'none' }, prompt: 'plain', ultraMode: false })).toBeUndefined()
})

test('runTurn threads the thinking directive into the provider request', async () => {
  const requests: CompletionRequest[] = []
  const events: StreamEvent[] = []
  for await (const event of runTurn({
    model: 'glm-5.2',
    permissionMode: 'accept-all',
    state: createAgentState(),
    thinking: { budgetTokens: 32_000, effort: 'high' },
    userMessage: 'ultrathink this task',
  }, {
    llm: {
      async *stream(request): AsyncGenerator<LlmDelta> {
        requests.push(request)
        yield { content: 'done' }
      },
    },
  })) {
    events.push(event)
  }

  expect(requests).toHaveLength(1)
  expect(requests[0]?.thinking).toEqual({ budgetTokens: 32_000, effort: 'high' })
  expect(events.at(-1)).toMatchObject({ type: 'turn_done' })
})

test('OpenAI-compatible payload maps neutral thinking to reasoning fields', async () => {
  let body: Record<string, unknown> = {}
  const client = new OpenAiCompatibleClient({
    apiKey: 'test-key',
    baseUrl: 'https://provider.test/v1',
    fetchImplementation: async (_url, init) => {
      body = JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>
      return new Response(sse('data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'), {
        headers: { 'Content-Type': 'text/event-stream' },
      })
    },
    providerName: 'zhipu',
  })

  for await (const _ of client.stream({
    model: 'glm-5.2',
    messages: [{ role: 'user', content: 'hi' }],
    thinking: { budgetTokens: 24_576, effort: 'high' },
  })) {
    void _
  }

  expect(body['reasoning_effort']).toBe('high')
  expect(body['thinking_budget']).toBe(24_576)
})

test('Anthropic payload maps neutral thinking to an enabled budget block', async () => {
  let body: Record<string, unknown> = {}
  const client = new AnthropicMessagesClient({
    apiKey: 'test-key',
    fetchImplementation: async (_url, init) => {
      body = JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>
      return new Response(sse(
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n' +
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}\n\n' +
        'event: message_stop\ndata: {"type":"message_stop"}\n\n',
      ), { headers: { 'Content-Type': 'text/event-stream' } })
    },
  })

  for await (const _ of client.stream({
    model: 'claude-sonnet-4-6',
    messages: [{ role: 'user', content: 'hi' }],
    thinking: { budgetTokens: 20_000, effort: 'high' },
  })) {
    void _
  }

  expect(body['thinking']).toEqual({ type: 'enabled', budget_tokens: 20_000 })
})

test('Anthropic thinking raises max_tokens past the budget and withholds incompatible sampling', async () => {
  const capture = async (request: Partial<CompletionRequest> & Pick<CompletionRequest, 'model' | 'messages'>) => {
    let body: Record<string, unknown> = {}
    const client = new AnthropicMessagesClient({
      apiKey: 'test-key',
      fetchImplementation: async (_url, init) => {
        body = JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>
        return new Response(sse(
          'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n' +
          'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}\n\n' +
          'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ), { headers: { 'Content-Type': 'text/event-stream' } })
      },
    })
    for await (const _ of client.stream(request)) {
      void _
    }
    return body
  }

  // Ultra escalation with default sampling: budget wins over max_tokens, and
  // temperature 0.6 / top_p are withheld because extended thinking rejects them.
  const escalated = await capture({
    model: 'claude-sonnet-4-6',
    messages: [{ role: 'user', content: 'hi' }],
    temperature: 0.6,
    thinking: { budgetTokens: 32_000, effort: 'high' },
    topP: 0.9,
  })
  expect(escalated['max_tokens']).toBe(32_000 + 4_096)
  expect(escalated['temperature']).toBeUndefined()
  expect(escalated['top_p']).toBeUndefined()
  expect(escalated['thinking']).toEqual({ type: 'enabled', budget_tokens: 32_000 })

  // An explicit larger maxTokens and temperature exactly 1 are preserved.
  const explicit = await capture({
    model: 'claude-sonnet-4-6',
    maxTokens: 64_000,
    messages: [{ role: 'user', content: 'hi' }],
    temperature: 1,
    thinking: { budgetTokens: 4_000, effort: 'medium' },
  })
  expect(explicit['max_tokens']).toBe(64_000)
  expect(explicit['temperature']).toBe(1)

  // Without thinking, sampling flows exactly as before.
  const plain = await capture({
    model: 'claude-sonnet-4-6',
    messages: [{ role: 'user', content: 'hi' }],
    temperature: 0.6,
    topP: 0.9,
  })
  expect(plain['max_tokens']).toBe(2048)
  expect(plain['temperature']).toBe(0.6)
  expect(plain['top_p']).toBe(0.9)
  expect(plain['thinking']).toBeUndefined()
})

function sse(payload: string): string {
  return payload
}
