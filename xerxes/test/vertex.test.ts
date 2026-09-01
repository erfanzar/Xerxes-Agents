// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterAll, beforeAll, expect, test } from 'bun:test'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { ConfigurationError, ProviderError } from '../src/core/errors.js'
import { GoogleVertexClient, clearVertexTokenCache, resolveVertexApiKey, vertexContentsFromMessages, vertexPayload } from '../src/llms/vertex.js'
import type { CompletionRequest } from '../src/llms/client.js'
import { createLlmClient } from '../src/llms/client.js'
import { detectProvider } from '../src/llms/providerRegistry.js'

test('express api keys pass through but pi-ai placeholders fall back to ADC', () => {
  expect(resolveVertexApiKey({ apiKey: 'real-key' })).toBe('real-key')
  expect(resolveVertexApiKey({ apiKey: '<vertex-api-key>' })).toBeUndefined()
  expect(resolveVertexApiKey({ apiKey: 'gcp-vertex-credentials' })).toBeUndefined()
  expect(resolveVertexApiKey({})).toBeUndefined()
})

test('payload conversion mirrors the native generateContent shape', () => {
  const request: CompletionRequest = {
    model: 'google-vertex/gemini-2.5-flash',
    systemSegments: [{ name: 'core', text: 'be brief' }],
    messages: [
      { role: 'system', content: 'be brief' },
      { role: 'user', content: 'hi' },
      {
        role: 'assistant',
        content: 'deploying',
        thinking: 'internal plan',
        thinking_signature: 'c2ln',
        tool_calls: [{
          id: 'call-1',
          type: 'function',
          function: { name: 'deploy', arguments: { name: 'api' } },
        }],
      },
      { role: 'tool', tool_call_id: 'call-1', content: 'done' },
      { role: 'user', content: 'thanks' },
    ],
    maxTokens: 512,
    temperature: 0.3,
    thinking: { effort: 'high' },
    toolChoice: 'any',
    tools: [{
      type: 'function',
      function: {
        name: 'deploy',
        description: 'd',
        parameters: { $schema: 'https://json-schema.org/x', type: 'object', properties: {} },
      },
    }],
  }
  const payload = vertexPayload(request)
  expect(payload.model).toBe('gemini-2.5-flash')
  expect(payload.config?.systemInstruction).toEqual({ parts: [{ text: 'be brief' }] })
  expect(payload.config?.generationConfig).toEqual({ maxOutputTokens: 512, temperature: 0.3 })
  expect(payload.config?.thinkingConfig).toEqual({ includeThoughts: true, thinkingLevel: 'HIGH' })
  expect(payload.config?.toolConfig).toEqual({ functionCallingConfig: { mode: 'ANY' } })
  // $-meta declarations are stripped from the schema (pi-ai sanitizeForOpenApi).
  expect(payload.config?.tools?.[0]?.functionDeclarations).toEqual([{
    name: 'deploy',
    description: 'd',
    parametersJsonSchema: { type: 'object', properties: {} },
  }])

  // system messages are extracted; assistant turns replay thought parts with
  // signatures; tool results merge into one user turn of functionResponse.
  expect(payload.contents).toEqual([
    { role: 'user', parts: [{ text: 'hi' }] },
    {
      role: 'model',
      parts: [
        { text: 'deploying' },
        { thought: true, text: 'internal plan', thoughtSignature: 'c2ln' },
        { functionCall: { args: { name: 'api' }, name: 'deploy' } },
      ],
    },
    {
      role: 'user',
      parts: [{ functionResponse: { name: 'deploy', response: { output: 'done' } } }],
    },
    { role: 'user', parts: [{ text: 'thanks' }] },
  ])
})

test('gemini-3 models carry tool call ids on function parts (pi-ai requiresToolCallId)', () => {
  const contents = vertexContentsFromMessages([
    {
      role: 'assistant',
      content: '',
      tool_calls: [{
        id: 'call.id.1',
        type: 'function',
        function: { name: 'deploy', arguments: { name: 'api' } },
      }],
    },
    { role: 'tool', tool_call_id: 'call.id.1', content: 'ok' },
  ], 'gemini-3-flash-preview')
  expect(contents[0]?.parts[0]?.functionCall?.id).toBe('call.id.1')
  expect(contents[1]?.parts[0]?.functionResponse?.id).toBe('call.id.1')
})

let credentialDir: string
beforeAll(() => {
  credentialDir = mkdtempSync(join(tmpdir(), 'xerxes-vertex-'))
})
afterAll(() => {
  rmSync(credentialDir, { force: true, recursive: true })
})

const SSE_STREAM_RESPONSE = {
  candidates: [{
    content: {
      parts: [
        { text: 'plan', thought: true, thoughtSignature: 'c2ln' },
        { text: 'Hel' },
        { text: 'lo' },
        { functionCall: { args: { name: 'api' }, name: 'deploy' } },
      ],
    },
    finishReason: 'STOP',
  }],
  usageMetadata: {
    cachedContentTokenCount: 40,
    candidatesTokenCount: 10,
    promptTokenCount: 100,
    thoughtsTokenCount: 5,
    totalTokenCount: 115,
  },
}

test('streaming posts to the ADC endpoint with a bearer token and maps the SSE chunks', async () => {
  clearVertexTokenCache()
  const credentialPath = join(credentialDir, 'adc-user.json')
  writeFileSync(credentialPath, JSON.stringify({
    type: 'authorized_user',
    client_id: 'cid',
    client_secret: 'secret',
    refresh_token: 'rt',
  }))
  const requests: { url: string; init?: RequestInit }[] = []
  const client = new GoogleVertexClient({
    env: {
      GOOGLE_APPLICATION_CREDENTIALS: credentialPath,
      GOOGLE_CLOUD_LOCATION: 'us-central1',
      GOOGLE_CLOUD_PROJECT: 'proj-1',
    },
    fetchImplementation: async (input, init) => {
      requests.push({ url: String(input), ...(init === undefined ? {} : { init }) })
      const url = String(input)
      if (url.includes('oauth2.googleapis.com/token')) {
        return new Response(JSON.stringify({ access_token: 'tok-1', expires_in: 3600 }))
      }
      const encoder = new TextEncoder()
      return new Response(new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(SSE_STREAM_RESPONSE)}\n\n`))
          controller.close()
        },
      }), { headers: { 'Content-Type': 'text/event-stream' } })
    },
  })

  const deltas = []
  for await (const delta of client.stream({
    model: 'google-vertex/gemini-2.5-flash',
    messages: [{ role: 'user', content: 'hi' }],
  })) deltas.push(delta)

  const tokenRequest = requests[0]
  expect(tokenRequest?.url).toBe('https://oauth2.googleapis.com/token')
  const tokenBody = String(tokenRequest?.init?.body)
  expect(tokenBody).toContain('grant_type=refresh_token')
  expect(tokenBody).toContain('refresh_token=rt')

  const streamRequest = requests[1]
  expect(streamRequest?.url).toBe(
    'https://us-central1-aiplatform.googleapis.com/v1/projects/proj-1/locations/us-central1/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse',
  )
  expect((streamRequest?.init?.headers as Record<string, string>).Authorization).toBe('Bearer tok-1')

  expect(deltas[0]).toEqual({ thinking: 'plan', thinkingSignature: 'c2ln' })
  expect(deltas[1]).toEqual({ content: 'Hel' })
  expect(deltas[2]).toEqual({ content: 'lo' })
  // pi-ai synthesizes `${name}_${Date.now()}_${counter}` ids when absent.
  const toolDelta = deltas[3] as unknown as { toolCalls: { id: string; type: string; function: Record<string, unknown> }[] }
  expect(toolDelta.toolCalls[0]?.id).toMatch(/^deploy_\d+_1$/)
  expect(toolDelta.toolCalls[0]?.type).toBe('function')
  expect(toolDelta.toolCalls[0]?.function).toEqual({ name: 'deploy', arguments: { name: 'api' } })
  expect(deltas[4]).toEqual({
    usage: {
      inputTokens: 60,
      outputTokens: 15,
      cacheReadTokens: 40,
      reasoningTokens: 5,
    },
  })
  // STOP with collected tool calls surfaces as tool_calls (pi-ai stopReason).
  expect(deltas[5]).toEqual({ finishReason: 'tool_calls' })
  expect(deltas).toHaveLength(6)
})

test('express api keys use the publisher endpoint with an api-key header', async () => {
  clearVertexTokenCache()
  const requests: { url: string; init?: RequestInit }[] = []
  const client = new GoogleVertexClient({
    apiKey: 'express-key',
    fetchImplementation: async (input, init) => {
      requests.push({ url: String(input), ...(init === undefined ? {} : { init }) })
      const encoder = new TextEncoder()
      return new Response(new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(SSE_STREAM_RESPONSE)}\n\n`))
          controller.close()
        },
      }))
    },
  })
  for await (const _ of client.stream({
    model: 'google-vertex/gemini-2.5-flash',
    messages: [{ role: 'user', content: 'hi' },
    ],
  })) { /* consume */ }
  expect(requests[0]?.url).toBe(
    'https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse',
  )
  expect((requests[0]?.init?.headers as Record<string, string>)['x-goog-api-key']).toBe('express-key')
})

test('non-streaming complete uses generateContent and parses the full response', async () => {
  clearVertexTokenCache()
  const client = new GoogleVertexClient({
    apiKey: 'express-key',
    fetchImplementation: async () => new Response(JSON.stringify({
      candidates: [{
        content: { parts: [{ text: 'Answer' }, { text: ' hidden', thought: true }] },
        finishReason: 'MAX_TOKENS',
      }],
      usageMetadata: { candidatesTokenCount: 4, promptTokenCount: 9 },
    })),
  })
  const completion = await client.complete({
    model: 'google-vertex/gemini-2.5-flash',
    messages: [{ role: 'user', content: 'question' }],
  })
  expect(completion.content).toBe('Answer')
  expect(completion.thinking).toBe(' hidden')
  expect(completion.finishReason).toBe('length')
  expect(completion.usage).toEqual({ inputTokens: 9, outputTokens: 4 })
})

test('safety finish reasons are fatal, never silent', async () => {
  clearVertexTokenCache()
  const client = new GoogleVertexClient({
    apiKey: 'express-key',
    fetchImplementation: async () => new Response(JSON.stringify({
      candidates: [{ content: { parts: [{ text: 'x' }] }, finishReason: 'SAFETY' }],
    })),
  })
  await expect(client.complete({
    model: 'google-vertex/gemini-2.5-flash',
    messages: [{ role: 'user', content: 'hi' }],
  })).rejects.toBeInstanceOf(ProviderError)
})

test('missing project or location configuration fails with actionable errors', () => {
  const client = new GoogleVertexClient({ env: {} })
  const streamPromise = (async () => {
    for await (const _ of client.stream({
      model: 'google-vertex/gemini-2.5-flash',
      messages: [{ role: 'user', content: 'hi' }],
    })) { /* consume */ }
  })()
  expect(streamPromise).rejects.toBeInstanceOf(ConfigurationError)
})

test('the factory routes google-vertex models to the native client', async () => {
  const requests: string[] = []
  const client = createLlmClient('google-vertex/gemini-2.5-flash', { api_key: 'express-key' }, {
    fetchImplementation: async (input) => {
      requests.push(String(input))
      return new Response(JSON.stringify({
        candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'STOP' }],
        usageMetadata: { candidatesTokenCount: 1, promptTokenCount: 1 },
      }))
    },
  })
  expect(client).toBeInstanceOf(GoogleVertexClient)
  expect(client.complete).toBeTypeOf('function')
  await client.complete!({
    model: 'google-vertex/gemini-2.5-flash',
    messages: [{ role: 'user', content: 'hi' }],
  })
  expect(requests[0]).toContain(':generateContent')
})

test('model routing recognizes vertex prefixes', () => {
  expect(detectProvider('google-vertex/gemini-2.5-flash')).toBe('google-vertex')
  expect(detectProvider('vertex/gemini-2.5-pro')).toBe('google-vertex')
})
