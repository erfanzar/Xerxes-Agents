// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { resolvedProfileModelCapabilities, type ProviderProfile } from '../src/bridge/profiles.js'
import { runtimeConnection } from '../src/daemon/runtimeConnection.js'
import { OpenAiCompatibleClient, type CompletionRequest, type LlmDelta } from '../src/llms/client.js'
import { modelCard } from '../src/llms/modelCards.js'
import type { ProviderName } from '../src/llms/providerRegistry.js'
import {
  DEFAULT_TEMPERATURE,
  DEFAULT_TOP_K,
  nativeSamplingDefaults,
  publishedSamplingProfile,
} from '../src/llms/samplingDefaults.js'

test('Bonsai 2 ids map to the profile Prism publishes for their size', () => {
  for (const id of [
    'ternary-bonsai-2-27b-mlx',
    'ternary-bonsai-2-27b',
    'prism-ml/Ternary-Bonsai-2-27B-mlx-2bit',
    'OsaurusAI/Bonsai-2-27B-Ternary-JANG',
    'mannix/bonsai-2-27b:latest',
  ]) {
    expect(publishedSamplingProfile(id)?.thinking).toEqual({
      temperature: 1, topP: 0.95, topK: 20, minP: 0, presencePenalty: 0, repetitionPenalty: 1,
    })
    expect(modelCard(id).maxOutputTokens).toBe(32_768)
  }
  for (const id of ['Ternary-Bonsai-2-2B-mlx-2bit', 'bonsai-2-4b', 'bonsai-9b', 'local/bonsai-9b']) {
    expect(publishedSamplingProfile(id)?.thinking.presencePenalty).toBe(1)
    expect(modelCard(id).maxOutputTokens).toBeUndefined()
  }
  expect(publishedSamplingProfile('bonsai-2-27b')?.instruct).toEqual({
    temperature: 0.7, topP: 0.8, topK: 20, minP: 0, presencePenalty: 1.5, repetitionPenalty: 1,
  })
  for (const id of ['qwen3-27b', 'gpt-4o-mini', 'bonsai-90b', 'bonsai-8b']) {
    expect(publishedSamplingProfile(id)).toBeUndefined()
  }
})

test('native defaults stay for other models and step aside for a published profile', () => {
  expect(nativeSamplingDefaults('gpt-4o-mini')).toEqual({ temperature: DEFAULT_TEMPERATURE, top_k: DEFAULT_TOP_K })
  expect(nativeSamplingDefaults(undefined)).toEqual({ temperature: DEFAULT_TEMPERATURE, top_k: DEFAULT_TOP_K })
  expect(nativeSamplingDefaults('ternary-bonsai-2-27b-mlx')).toEqual({})
})

async function payloadFor(
  request: Omit<CompletionRequest, 'messages'>,
  providerName: ProviderName = 'custom',
): Promise<Record<string, unknown>> {
  let payload: Record<string, unknown> | undefined
  const client = new OpenAiCompatibleClient({
    providerName,
    apiKey: 'lm-studio',
    baseUrl: 'http://127.0.0.1:1234/v1',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      const body = `data: ${JSON.stringify({ choices: [{ delta: { content: 'ok' }, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`
      return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } })
    },
  })
  const deltas: LlmDelta[] = []
  for await (const delta of client.stream({ ...request, messages: [{ role: 'user', content: 'Hello' }] })) deltas.push(delta)
  expect(deltas.at(-1)?.finishReason).toBe('stop')
  if (!payload) throw new Error('no request was sent')
  return payload
}

test('a Bonsai request with nothing configured sends Prism\'s whole thinking profile to a local engine', async () => {
  for (const provider of ['custom', 'openai', 'lmstudio'] as const) {
    const payload = await payloadFor({ model: 'ternary-bonsai-2-27b-mlx' }, provider)
    expect(payload).toMatchObject({
      temperature: 1,
      top_p: 0.95,
      top_k: 20,
      min_p: 0,
      presence_penalty: 0,
      repetition_penalty: 1,
      repeat_penalty: 1,
    })
  }
})

test('explicit request values and extra-body fields win over the published profile', async () => {
  const payload = await payloadFor({
    model: 'ternary-bonsai-2-27b-mlx',
    temperature: 0.3,
    topP: 0.9,
    topK: 40,
    repetitionPenalty: 1.05,
    presencePenalty: 0.5,
    extraBody: { min_p: 0.05 },
  })
  expect(payload).toMatchObject({
    temperature: 0.3,
    top_p: 0.9,
    top_k: 40,
    min_p: 0.05,
    presence_penalty: 0.5,
    repetition_penalty: 1.05,
    repeat_penalty: 1.05,
  })
})

test('a Bonsai request that turns thinking off gets the non-thinking profile', async () => {
  const payload = await payloadFor({
    model: 'ternary-bonsai-2-27b-mlx',
    extraBody: { chat_template_kwargs: { enable_thinking: false } },
  })
  expect(payload).toMatchObject({ temperature: 0.7, top_p: 0.8, top_k: 20, presence_penalty: 1.5 })
})

test('thinking off reaches Bonsai as its chat-template switch, with the non-thinking profile', async () => {
  const off = await payloadFor({ model: 'ternary-bonsai-2-27b', thinking: { effort: 'off' } })
  expect(off).toMatchObject({ chat_template_kwargs: { enable_thinking: false }, temperature: 0.7, top_p: 0.8, presence_penalty: 1.5 })
  // Thinking left on, or never mentioned, sends no switch.
  expect(await payloadFor({ model: 'ternary-bonsai-2-27b', thinking: { effort: 'high' } })).not.toHaveProperty('chat_template_kwargs')
  expect(await payloadFor({ model: 'ternary-bonsai-2-27b' })).not.toHaveProperty('chat_template_kwargs')
  // A model without a published card is left alone.
  expect(await payloadFor({ model: 'qwen3-27b', thinking: { effort: 'off' } })).not.toHaveProperty('chat_template_kwargs')
})

test('other models on a custom endpoint still get no non-standard sampling fields', async () => {
  const payload = await payloadFor({ model: 'qwen3-27b', topK: 64, minP: 0.05, repetitionPenalty: 1.1 })
  expect(payload).not.toHaveProperty('top_k')
  expect(payload).not.toHaveProperty('min_p')
  expect(payload).not.toHaveProperty('repetition_penalty')
  expect(payload).not.toHaveProperty('repeat_penalty')
})

test('the CLI connection leaves Bonsai sampling unset unless someone set it', () => {
  const config = (runtime: Record<string, unknown>) => ({ runtime } as unknown as Parameters<typeof runtimeConnection>[0])
  const bonsai = runtimeConnection(config({ model: 'ternary-bonsai-2-27b-mlx', provider: 'custom' }), undefined)
  expect(bonsai?.temperature).toBeUndefined()
  expect(bonsai?.topK).toBeUndefined()
  expect(runtimeConnection(config({ model: 'ternary-bonsai-2-27b-mlx', temperature: 0.8, top_k: 30 }), undefined))
    .toMatchObject({ temperature: 0.8, topK: 30 })
  expect(runtimeConnection(config({ model: 'gpt-4o-mini' }), undefined))
    .toMatchObject({ temperature: DEFAULT_TEMPERATURE, topK: DEFAULT_TOP_K })
})

test('Bonsai 2 27B gets Prism\'s output ceiling only when nothing else states one', () => {
  expect(resolvedProfileModelCapabilities(undefined, 'ternary-bonsai-2-27b-mlx'))
    .toMatchObject({ maxOutputTokens: 32_768, outputSource: 'catalog' })
  const profile = {
    name: 'bonsai-local',
    provider: 'custom',
    base_url: 'http://127.0.0.1:1234/v1',
    model: 'ternary-bonsai-2-27b-mlx',
    model_overrides: { 'ternary-bonsai-2-27b-mlx': { max_output_tokens: 16_384 } },
  } as unknown as ProviderProfile
  expect(resolvedProfileModelCapabilities(profile, 'ternary-bonsai-2-27b-mlx'))
    .toMatchObject({ maxOutputTokens: 16_384, outputSource: 'override' })
  expect(resolvedProfileModelCapabilities(undefined, 'bonsai-9b').maxOutputTokens).toBeUndefined()
})
