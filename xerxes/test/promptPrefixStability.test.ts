// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Every provider caches by exact prefix. A conversation that grows by one
 * step must reach each provider as the previous request plus new material:
 * the same system text, and every earlier message byte-identical. Cache
 * markers are the one thing allowed to move (each adapter puts its marker on
 * the newest block), so they are stripped before comparing.
 */

import { afterAll, expect, test } from 'bun:test'

import { messagesToAnthropic } from '../src/llms/anthropic.js'
import { buildBedrockConverseInput, resolveBedrockModel } from '../src/llms/bedrock.js'
import { claudeCodeTranscript, withTranscriptCacheMark } from '../src/llms/claudeCode.js'
import { OpenAiCompatibleClient } from '../src/llms/client.js'
import { messagesToGemini } from '../src/llms/gemini.js'
import { vertexPayload } from '../src/llms/vertex.js'
import type { ChatMessage } from '../src/types/messages.js'
import { clearModelsDev, seedModelsDev } from './fixtures/modelsDev.js'

seedModelsDev()
afterAll(() => clearModelsDev())

const system: ChatMessage = { role: 'system', content: 'You are Xerxes.' }
const stepOne: ChatMessage[] = [
  system,
  { role: 'user', content: '<turn-context>\nmemory changed\n</turn-context>\n\nread a.ts' },
  { role: 'assistant', content: 'Reading.', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'read', arguments: { path: 'a.ts', limit: 40 } } }] },
  { role: 'tool', tool_call_id: 'c1', name: 'read', content: 'export const a = 1' },
]
const stepTwo: ChatMessage[] = [
  ...stepOne,
  { role: 'assistant', content: 'Now b.', tool_calls: [{ id: 'c2', type: 'function', function: { name: 'read', arguments: { path: 'b.ts' } } }] },
  { role: 'tool', tool_call_id: 'c2', name: 'read', content: 'export const b = 2' },
]

/** Drop cache markers, which each adapter deliberately moves to the newest block. */
function withoutMarks(value: unknown): unknown {
  if (Array.isArray(value)) return value.filter(item => !(item && typeof item === 'object' && 'cachePoint' in item)).map(withoutMarks)
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).filter(([key]) => key !== 'cache_control').map(([key, item]) => [key, withoutMarks(item)]))
  }
  return value
}

function expectExtension(first: { system: unknown; messages: readonly unknown[] }, second: { system: unknown; messages: readonly unknown[] }) {
  expect(JSON.stringify(withoutMarks(second.system))).toBe(JSON.stringify(withoutMarks(first.system)))
  const earlier = withoutMarks(first.messages) as unknown[]
  expect(second.messages.length).toBeGreaterThan(earlier.length)
  expect(JSON.stringify((withoutMarks(second.messages) as unknown[]).slice(0, earlier.length))).toBe(JSON.stringify(earlier))
}

test('Anthropic: step two extends step one', () => {
  const one = messagesToAnthropic(stepOne), two = messagesToAnthropic(stepTwo)
  expectExtension({ system: one.system, messages: one.messages }, { system: two.system, messages: two.messages })
})

test('Bedrock: step two extends step one', () => {
  const request = (messages: ChatMessage[]) => ({ model: 'amazon-bedrock/anthropic.claude-opus-4-1-20250805-v1:0', messages })
  const model = resolveBedrockModel(request(stepOne), {})
  const one = buildBedrockConverseInput(request(stepOne), { env: {}, model }), two = buildBedrockConverseInput(request(stepTwo), { env: {}, model })
  expectExtension({ system: one.system, messages: one.messages as unknown[] }, { system: two.system, messages: two.messages as unknown[] })
})

test('Gemini and Vertex: step two extends step one', () => {
  const one = messagesToGemini(stepOne), two = messagesToGemini(stepTwo)
  expectExtension({ system: one.systemInstruction, messages: one.contents }, { system: two.systemInstruction, messages: two.contents })
  const v1 = vertexPayload({ model: 'google-vertex/gemini-2.5-flash', messages: stepOne })
  const v2 = vertexPayload({ model: 'google-vertex/gemini-2.5-flash', messages: stepTwo })
  expectExtension({ system: v1.systemInstruction, messages: v1.contents }, { system: v2.systemInstruction, messages: v2.contents })
})

test('Claude Code: step two extends step one, and only the newest block carries the mark', () => {
  const one = withTranscriptCacheMark(claudeCodeTranscript(stepOne)), two = withTranscriptCacheMark(claudeCodeTranscript(stepTwo))
  expectExtension({ system: '', messages: one }, { system: '', messages: two })
})

test('OpenAI-compatible providers (OpenAI, OpenRouter, DeepSeek, Kimi, z.ai, …): step two extends step one', async () => {
  for (const providerName of ['openai', 'openrouter', 'deepseek', 'kimi-code', 'zhipu', 'groq'] as const) {
    const sent: Record<string, unknown>[] = []
    const client = new OpenAiCompatibleClient({
      providerName, apiKey: 'test-key', baseUrl: 'https://example.invalid/v1',
      fetchImplementation: async (_input, init) => {
        sent.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
        return new Response('data: [DONE]\n\n', { headers: { 'content-type': 'text/event-stream' } })
      },
    })
    for (const messages of [stepOne, stepTwo]) {
      for await (const _delta of client.stream({ model: 'test-model', messages, sessionId: 'session-1' })) { /* request captured */ }
    }
    const [one, two] = sent as [{ messages: unknown[]; prompt_cache_key?: unknown }, { messages: unknown[]; prompt_cache_key?: unknown }]
    expectExtension({ system: null, messages: one.messages }, { system: null, messages: two.messages })
    expect(two.prompt_cache_key).toEqual(one.prompt_cache_key)
  }
})
