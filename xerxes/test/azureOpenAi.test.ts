// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { AzureOpenAiClient, resolveAzureBaseUrl, resolveAzureDeployment } from '../src/llms/azureOpenAi.js'
import type { LlmDelta } from '../src/llms/client.js'
import { ConfigurationError, ProviderError } from '../src/core/errors.js'

test('Azure base URL resolution follows the documented precedence and normalizes Azure hosts', () => {
  // Explicit option wins over the environment.
  expect(resolveAzureBaseUrl(
    { baseUrl: 'https://opt-gateway.contoso.com/v1' },
    { AZURE_OPENAI_BASE_URL: 'https://env-gateway.contoso.com/v1' },
  )).toBe('https://opt-gateway.contoso.com/v1')
  // Environment base URL wins over the resource-name default.
  expect(resolveAzureBaseUrl({}, { AZURE_OPENAI_BASE_URL: 'https://env-gateway.contoso.com/v1' }))
    .toBe('https://env-gateway.contoso.com/v1')
  // Every documented Azure host suffix normalizes to the Responses v1 path.
  expect(resolveAzureBaseUrl(
    { baseUrl: 'https://res.openai.azure.com/openai/deployments/gpt-4o?api-version=2024-02-01' },
    {},
  )).toBe('https://res.openai.azure.com/openai/v1')
  expect(resolveAzureBaseUrl({ baseUrl: 'https://res.cognitiveservices.azure.com/' }, {}))
    .toBe('https://res.cognitiveservices.azure.com/openai/v1')
  expect(resolveAzureBaseUrl({ baseUrl: 'https://res.ai.azure.com' }, {}))
    .toBe('https://res.ai.azure.com/openai/v1')
  // Non-Azure custom hosts keep their path.
  expect(resolveAzureBaseUrl({ baseUrl: 'https://gateway.contoso.com/aoai' }, {}))
    .toBe('https://gateway.contoso.com/aoai')
  // Resource-name defaults: option first, then environment.
  expect(resolveAzureBaseUrl({ resourceName: 'my-res' }, {}))
    .toBe('https://my-res.openai.azure.com/openai/v1')
  expect(resolveAzureBaseUrl({}, { AZURE_OPENAI_RESOURCE_NAME: 'env-res' }))
    .toBe('https://env-res.openai.azure.com/openai/v1')
  // Nothing resolvable is a configuration error, not a silent fallback.
  expect(() => resolveAzureBaseUrl({}, {})).toThrow(ConfigurationError)
})

test('Azure deployment resolution prefers explicit name, then maps, then the model id', () => {
  const env = { AZURE_OPENAI_DEPLOYMENT_NAME_MAP: 'gpt-4o=prod-gpt4o, gpt-5=prod-gpt5' }
  expect(resolveAzureDeployment('gpt-4o', { deploymentName: 'explicit-dep' }, env)).toBe('explicit-dep')
  // The environment map matches the full id and the prefix-stripped model id.
  expect(resolveAzureDeployment('gpt-5', {}, env)).toBe('prod-gpt5')
  expect(resolveAzureDeployment('azure/gpt-4o', {}, env)).toBe('prod-gpt4o')
  // Unmapped models fall back to their bare model id.
  expect(resolveAzureDeployment('o4-mini', {}, env)).toBe('o4-mini')
  // A static option map participates after the explicit name.
  expect(resolveAzureDeployment('gpt-5', { deploymentNameMap: { 'gpt-5': 'static-dep' } }, {})).toBe('static-dep')
})

test('Azure client authenticates with api-key, versions the endpoint, and omits rejected fields', async () => {
  let url = ''
  let headers: Record<string, string> = {}
  let payload: Record<string, unknown> = {}
  const client = new AzureOpenAiClient({
    resourceName: 'my-res',
    apiKey: 'az-key',
    apiVersion: 'preview',
    deploymentName: 'prod-gpt4o',
    fetchImplementation: async (input, init) => {
      url = String(input)
      headers = init?.headers as Record<string, string>
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return sse([{ type: 'response.completed', response: { status: 'completed' } }])
    },
  })

  for await (const _delta of client.stream({
    model: 'gpt-4o',
    messages: [
      { role: 'system', content: 'Be brief.' },
      { role: 'user', content: 'hi' },
    ],
    sessionId: 'sess-1',
    serviceTier: 'flex',
  })) {
    // Drain.
  }

  expect(url).toBe('https://my-res.openai.azure.com/openai/v1/responses?api-version=preview')
  expect(headers['api-key']).toBe('az-key')
  expect(headers.Authorization).toBeUndefined()
  expect(payload).toMatchObject({
    model: 'prod-gpt4o',
    store: false,
    stream: true,
    instructions: 'Be brief.',
    prompt_cache_key: 'sess-1',
  })
  // Pi's Azure compat data: the gateway 400s on both of these.
  expect(payload).not.toHaveProperty('service_tier')
  expect(payload).not.toHaveProperty('prompt_cache_retention')
})

test('Azure streaming translates reasoning, text, tool calls, and cached usage', async () => {
  const client = new AzureOpenAiClient({
    resourceName: 'r',
    apiKey: 'k',
    fetchImplementation: async () => sse([
      { type: 'response.reasoning_summary_text.delta', delta: 'Think.' },
      { type: 'response.output_text.delta', delta: 'Hello' },
      {
        type: 'response.output_item.added',
        item: { type: 'function_call', id: 'fc_1', call_id: 'call_1', name: 'ReadFile' },
      },
      { type: 'response.function_call_arguments.delta', item_id: 'fc_1', delta: '{"path":' },
      { type: 'response.function_call_arguments.delta', item_id: 'fc_1', delta: '"README.md"}' },
      {
        type: 'response.output_item.done',
        item: { type: 'function_call', id: 'fc_1', call_id: 'call_1', name: 'ReadFile' },
      },
      {
        type: 'response.completed',
        response: {
          status: 'completed',
          usage: {
            input_tokens: 12,
            output_tokens: 7,
            input_tokens_details: { cached_tokens: 3 },
            output_tokens_details: { reasoning_tokens: 2 },
          },
        },
      },
    ]),
  })

  const events: LlmDelta[] = []
  for await (const event of client.stream({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: 'hi' }],
  })) events.push(event)

  expect(events).toEqual([
    { thinking: 'Think.' },
    { content: 'Hello' },
    {
      finishReason: 'tool_calls',
      // input_tokens 12 includes the 3 cached, so fresh is 9 and the pair
      // still sums to the 12-token prompt the provider measured.
      usage: { inputTokens: 9, outputTokens: 7, cacheReadTokens: 3, reasoningTokens: 2 },
      toolCalls: [{
        id: 'call_1',
        type: 'function',
        function: { name: 'ReadFile', arguments: { path: 'README.md' } },
      }],
    },
  ])
})

test('Azure non-streaming completion parses output items and normalized usage', async () => {
  let url = ''
  let payload: Record<string, unknown> = {}
  const client = new AzureOpenAiClient({
    resourceName: 'r',
    apiKey: 'k',
    fetchImplementation: async (input, init) => {
      url = String(input)
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return Response.json({
        status: 'completed',
        service_tier: 'default',
        output: [
          { type: 'reasoning', summary: [{ type: 'summary_text', text: 'Look.' }] },
          { type: 'message', content: [{ type: 'output_text', text: 'Done.' }] },
          { type: 'function_call', call_id: 'call-1', name: 'ReadFile', arguments: '{"path":"README.md"}' },
        ],
        usage: { input_tokens: 14, output_tokens: 6, input_tokens_details: { cached_tokens: 3 } },
      })
    },
  })

  const completion = await client.complete({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: 'q' }],
    maxTokens: 8,
    thinking: { effort: 'high' },
  })

  // Default api-version is v1; the 8-token cap clamps to the Responses floor.
  expect(url).toBe('https://r.openai.azure.com/openai/v1/responses?api-version=v1')
  expect(payload.max_output_tokens).toBe(16)
  expect(payload.reasoning).toEqual({ effort: 'high' })
  expect(payload.include).toEqual(['reasoning.encrypted_content'])
  expect(completion).toEqual({
    content: 'Done.',
    thinking: 'Look.',
    thinkingSignature: JSON.stringify({
      type: 'reasoning',
      summary: [{ type: 'summary_text', text: 'Look.' }],
    }),
    finishReason: 'tool_calls',
    toolCalls: [{ id: 'call-1', type: 'function', function: { name: 'ReadFile', arguments: { path: 'README.md' } } }],
    usage: { inputTokens: 11, outputTokens: 6, cacheReadTokens: 3, serviceTier: 'default' },
  })
})

test('Azure HTTP failures surface as ProviderError carrying the status', async () => {
  const client = new AzureOpenAiClient({
    resourceName: 'r',
    apiKey: 'k',
    fetchImplementation: async () => new Response(
      JSON.stringify({ error: { message: 'deployment not found' } }),
      { status: 404 },
    ),
  })

  try {
    await client.complete({ model: 'gpt-4o', messages: [{ role: 'user', content: 'q' }] })
    throw new Error('expected the completion request to reject')
  } catch (error) {
    expect(error).toBeInstanceOf(ProviderError)
    expect((error as ProviderError).details.status).toBe(404)
    expect((error as Error).message).toContain('deployment not found')
  }
})

/** Frame a scripted event list as an Azure Responses SSE body. */
function sse(events: Record<string, unknown>[]): Response {
  return new Response(
    events.map(event => `data: ${JSON.stringify(event)}\n\n`).join(''),
    { headers: { 'Content-Type': 'text/event-stream' } },
  )
}
