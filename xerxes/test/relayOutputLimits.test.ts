// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { AnthropicMessagesClient } from '../src/llms/anthropic.js'
import { AzureOpenAiClient } from '../src/llms/azureOpenAi.js'
import { buildBedrockConverseInput, resolveBedrockModel } from '../src/llms/bedrock.js'
import { GeminiClient } from '../src/llms/gemini.js'
import { PiMessagesClient } from '../src/llms/piMessages.js'
import { vertexPayload } from '../src/llms/vertex.js'
import { OpenAiCompatibleClient, ResponsesApiClient, type CompletionRequest, type FetchImplementation, type LlmClient } from '../src/llms/client.js'
import { OUTPUT_TOKEN_LIMIT, OutputTokenLimitError } from '../src/llms/outputTokenLimit.js'
import { LocalProviderRelay } from '../src/security/localProviderRelay.js'
import { decodeRelayCompletion } from '../src/security/providerRelayProtocol.js'
import { LocalProviderEndpoint } from '../src/security/localProviderEndpoint.js'
import { LocalRelayClient } from '../src/llms/localRelayClient.js'
import { seedModelsDev } from './fixtures/modelsDev.js'

const messages: CompletionRequest['messages'] = [{ role: 'user', content: 'Fixture only' }]
async function capture(model: string, cap: number | null, makeClient: (fetch: FetchImplementation) => LlmClient, thinking?: CompletionRequest['thinking']) {
  const bodies: Record<string, unknown>[] = []
  const client = makeClient(async (_input, init) => {
    const body = typeof init?.body === 'string' ? init.body : Buffer.from(Bun.zstdDecompressSync(init?.body as Uint8Array)).toString()
    bodies.push(JSON.parse(body) as Record<string, unknown>)
    throw new Error('Private synthetic diagnostic; captured before networking')
  })
  const relay = new LocalProviderRelay(), peer = { destination: 'fixture', workspace: '/fixture' }
  const grant = relay.authorize(peer, { profile: 'fixture', model, maxOutputTokens: cap, expiresAt: Date.now() + 60000, maxRequests: 1, maxConcurrent: 1 }, () => ({ client, routeIdentity: 'fixture' }), 'fixture')
  let code: unknown
  try { for await (const _ of relay.stream(peer, grant.token, { model, messages, ...(thinking ? { thinking } : {}) })) {} }
  catch (error) { code = (error as { code?: unknown }).code; expect(String(error)).not.toContain('Private synthetic') }
  finally { relay.close() }
  return { bodies, code }
}

test('Anthropic budget expansion cannot exceed a local grant; sufficient grants retain normal thinking', async () => {
  const make = (fetchImplementation: FetchImplementation) => new AnthropicMessagesClient({ apiKey: 'fixture', baseUrl: 'https://fixture.invalid', fetchImplementation })
  const refused = await capture('claude-sonnet-4-20250514', 1024, make, { effort: 'high', budgetTokens: 10000 })
  expect(refused).toEqual({ bodies: [], code: 'output_limit' })
  const allowed = await capture('claude-sonnet-4-20250514', 16384, make, { effort: 'high', budgetTokens: 10000 })
  expect(allowed.bodies).toHaveLength(1)
  expect(allowed.bodies[0]?.max_tokens).toBe(16384)
  expect(allowed.bodies[0]?.thinking).toMatchObject({ budget_tokens: 10000 })
  expect(allowed.code).toBe('provider_failed') // injected fetch ends the fixture
})

test('Responses and Azure minimum output floors cannot exceed a local grant', async () => {
  for (const make of [
    (fetchImplementation: FetchImplementation) => new ResponsesApiClient({ providerName: 'openai', apiKey: 'fixture', baseUrl: 'https://fixture.invalid', fetchImplementation }),
    (fetchImplementation: FetchImplementation) => new AzureOpenAiClient({ apiKey: 'fixture', baseUrl: 'https://fixture.invalid', fetchImplementation }),
  ]) {
    expect(await capture('gpt-4o', 1, make)).toEqual({ bodies: [], code: 'output_limit' })
    const allowed = await capture('gpt-4o', 16, make)
    expect(allowed.bodies).toHaveLength(1)
    expect(allowed.bodies[0]?.max_output_tokens).toBe(16)
  }
})

test('Codex cannot drop a bounded grant silently; provider-controlled policy is a different request', async () => {
  const make = (fetchImplementation: FetchImplementation) => new ResponsesApiClient({ providerName: 'openai-codex', apiKey: 'fixture', baseUrl: 'https://fixture.invalid', codexTransport: 'sse', fetchImplementation })
  expect(await capture('gpt-5', 1024, make)).toEqual({ bodies: [], code: 'output_limit' })
  const allowed = await capture('gpt-5', null, make)
  expect(allowed.bodies).toHaveLength(1)
  expect(allowed.bodies[0]).not.toHaveProperty('max_output_tokens')
  expect(JSON.stringify(allowed.bodies)).not.toContain('outputTokenLimit')
})

test('Bedrock expansion is checked before the SDK receives its command input', () => {
  seedModelsDev()
  const request: CompletionRequest = { model: 'amazon-bedrock/anthropic.claude-opus-4-1-20250805-v1:0', messages, maxTokens: 1024, thinking: { effort: 'high', budgetTokens: 10000 }, [OUTPUT_TOKEN_LIMIT]: 1024 }
  const options = { env: {}, model: resolveBedrockModel(request, {}) }
  expect(() => buildBedrockConverseInput(request, options)).toThrow(OutputTokenLimitError)
  expect(buildBedrockConverseInput({ ...request, [OUTPUT_TOKEN_LIMIT]: 16384 }, options)).toMatchObject({ inferenceConfig: { maxTokens: 11024 } })
})

test('output policy refusal crosses the private transport safely and explicit off permits retry', async () => {
  let requests = 0
  const model = 'claude-sonnet-4-20250514', relay = new LocalProviderRelay(), peer = { destination: 'fixture', workspace: '/fixture' }
  const provider = new AnthropicMessagesClient({ apiKey: 'fixture', baseUrl: 'https://fixture.invalid', fetchImplementation: async (_url, init) => {
    requests++
    expect(JSON.parse(String(init?.body)).max_tokens).toBe(1024)
    return new Response('data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Safe retry"}}\n\ndata: {"type":"message_stop"}\n\n', { headers: { 'content-type': 'text/event-stream' } })
  } })
  const grant = relay.authorize(peer, { profile: 'fixture', model, maxOutputTokens: 1024, expiresAt: Date.now() + 60000, maxRequests: 2, maxConcurrent: 1 }, () => ({ client: provider, routeIdentity: 'fixture' }), 'fixture')
  const endpoint = new LocalProviderEndpoint(relay, peer, grant.token), client = new LocalRelayClient(frame => endpoint.handle(frame))
  try {
    await expect(client.stream({ model, messages, thinking: { budgetTokens: 10000 } })[Symbol.asyncIterator]().next()).rejects.toMatchObject({ code: 'output_limit', message: expect.stringContaining('Lower reasoning') })
    expect(requests).toBe(0)
    const output = []
    for await (const delta of client.stream({ model, messages, thinking: { effort: 'none' } })) output.push(delta.content)
    expect(output.join('')).toBe('Safe retry')
    expect(requests).toBe(1)
    expect(relay.inspect(peer, grant.token)).toMatchObject({ requestsUsed: 2, status: 'exhausted' })
  } finally { endpoint.close(); relay.close() }
})

test('other native transports retain the approved ceiling and no authority field reaches JSON', async () => {
  const fixtures = [
    { model: 'gpt-4o', make: (fetchImplementation: FetchImplementation) => new OpenAiCompatibleClient({ providerName: 'openai', apiKey: 'fixture', baseUrl: 'https://fixture.invalid', fetchImplementation }), field: 'max_tokens' },
    { model: 'gemini-2.5-flash', make: (fetchImplementation: FetchImplementation) => new GeminiClient({ apiKey: 'fixture', baseUrl: 'https://fixture.invalid', fetchImplementation }), field: 'generationConfig' },
    { model: 'radius/fixture', make: (fetchImplementation: FetchImplementation) => new PiMessagesClient('fixture', { apiKey: 'fixture', baseUrl: 'https://fixture.invalid', fetchImplementation }), field: 'options' },
  ]
  for (const { model, make, field } of fixtures) {
    const captured = await capture(model, 1024, make)
    expect(captured.bodies).toHaveLength(1)
    const body = captured.bodies[0]!
    if (field === 'generationConfig') expect(body[field]).toMatchObject({ maxOutputTokens: 1024 })
    else if (field === 'options') expect(body[field]).toMatchObject({ maxTokens: 1024 })
    else expect(body[field]).toBe(1024)
    expect(JSON.stringify(body)).not.toContain('outputTokenLimit')
  }
  const request: CompletionRequest = { model: 'google-vertex/gemini-2.5-flash', messages, maxTokens: 1024, [OUTPUT_TOKEN_LIMIT]: 1024 }
  expect(vertexPayload(request).generationConfig?.maxOutputTokens).toBe(1024)
  expect(() => vertexPayload({ ...request, maxTokens: 1025 })).toThrow(OutputTokenLimitError)
  expect(() => decodeRelayCompletion({ model: 'gpt-4o', messages, outputTokenLimit: null })).toThrow('unsupported')
})
