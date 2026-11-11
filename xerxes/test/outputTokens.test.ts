// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { ValidationError } from '../src/core/errors.js'
import { getLlmModelInfo, OpenAiCompatibleClient, ResponsesApiClient, type CompletionRequest } from '../src/llms/client.js'
import { reportModelCapability } from '../src/llms/modelsDev.js'
import { OUTPUT_TOKEN_LIMIT, OutputTokenLimitError } from '../src/llms/outputTokenLimit.js'
import { resolveMaxOutputTokens } from '../src/llms/outputTokens.js'
import type { ProviderName } from '../src/llms/providerRegistry.js'

function capture(providerName: ProviderName, responses: boolean, streaming = false) {
  const bodies: Record<string, unknown>[] = []
  const options = { providerName, apiKey: 'offline-test', baseUrl: 'https://offline.invalid/v1', fetchImplementation: async (_url: RequestInfo | URL, init?: RequestInit) => {
    const raw = init?.body
    const text = raw instanceof Uint8Array ? new TextDecoder().decode(Bun.zstdDecompressSync(raw)) : String(raw)
    const body: unknown = JSON.parse(text)
    if (!body || typeof body !== 'object' || Array.isArray(body)) throw new Error('Expected payload object')
    bodies.push(Object.fromEntries(Object.entries(body)))
    if (streaming || providerName === 'openai-codex') {
      const event = responses ? { type: 'response.completed', response: { status: 'completed', output: [] } } : { choices: [{ delta: { content: 'ok' }, finish_reason: 'stop' }] }
      return new Response(`data: ${JSON.stringify(event)}\n\n${responses ? '' : 'data: [DONE]\n\n'}`, { headers: { 'Content-Type': 'text/event-stream' } })
    }
    return Response.json(responses ? { status: 'completed', output: [] } : { choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }] })
  } }
  const client = responses ? new ResponsesApiClient({ ...options, codexTransport: 'sse' }) : new OpenAiCompatibleClient(options)
  return { bodies, async send(maxTokens?: number, model = 'offline-output-test', extra: Partial<CompletionRequest> = {}) {
    const request: CompletionRequest = { model, messages: [{ role: 'user', content: 'hi' }], ...(maxTokens === undefined ? {} : { maxTokens }), ...extra }
    if (streaming) { for await (const _delta of client.stream(request)) { /* consume */ } }
    else await client.complete(request)
    return bodies.at(-1)?.[responses ? 'max_output_tokens' : 'max_tokens']
  } }
}
for (const provider of ['openai', 'custom', 'lmstudio', 'ollama'] as const) {
  for (const responses of [false, true]) for (const streaming of [false, true]) {
    test(`${provider}: 32k default, no cap, explicit values and guard; responses=${responses}, streaming=${streaming}`, async () => {
      const transport = capture(provider, responses, streaming)
      expect(await transport.send()).toBe(32_768)
      expect(await transport.send(2_000)).toBe(2_000)
      expect(await transport.send(-1)).toBeUndefined()
      expect(transport.bodies.at(-1)).not.toHaveProperty(responses ? 'max_output_tokens' : 'max_tokens')
      for (const invalid of [0, -2, NaN, Infinity, 1.5]) await expect(transport.send(invalid)).rejects.toBeInstanceOf(ValidationError)
      const sent = transport.bodies.length
      await expect(transport.send(-1, undefined, { [OUTPUT_TOKEN_LIMIT]: 512 })).rejects.toBeInstanceOf(OutputTokenLimitError)
      expect(transport.bodies).toHaveLength(sent)
      expect(await transport.send(256, undefined, { [OUTPUT_TOKEN_LIMIT]: 512 })).toBe(256)
      if (responses) expect(await transport.send(1)).toBe(16)
    })
    test(`${provider}: known provider ceilings remain hard limits; responses=${responses}, streaming=${streaming}`, async () => {
      const model = `offline-output-${provider}-${responses}-${streaming}`
      reportModelCapability(provider, model, { maxOutputTokens: 8_192 })
      const transport = capture(provider, responses, streaming)
      expect(await transport.send(undefined, model)).toBe(8_192)
      expect(await transport.send(20_000, model)).toBe(8_192)
      expect(await transport.send(-1, model)).toBe(8_192)
      expect(await transport.send(2_000, model)).toBe(2_000)
    })
  }
}
test('Codex and unselected providers retain their previous wire behavior', async () => {
  for (const value of [undefined, 2_000, -1, 0]) {
    expect(await capture('openai-codex', true).send(value)).toBeUndefined()
    expect(await capture('groq', false).send(value)).toBe(value)
  }
  expect(getLlmModelInfo('openai-codex/offline-test').maxTokens).toBe(2_048)
  expect(getLlmModelInfo('openai/offline-test').maxTokens).toBe(32_768)
})
test('shared resolver preserves sentinel state and ignores unusable model metadata', () => {
  expect(resolveMaxOutputTokens(undefined)).toBe(32_768)
  expect(resolveMaxOutputTokens(-1)).toBe(-1)
  expect(resolveMaxOutputTokens(-1, 64_000)).toBe(64_000)
  for (const invalid of [0, -2, NaN, Infinity, 1.5]) {
    expect(() => resolveMaxOutputTokens(invalid)).toThrow(ValidationError)
    expect(resolveMaxOutputTokens(undefined, invalid)).toBe(32_768)
  }
})
