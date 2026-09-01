// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import type { CopilotCredential, CopilotSession } from '../src/auth/copilotAuth.js'
import { AzureOpenAiClient } from '../src/llms/azureOpenAi.js'
import { createLlmClient, OpenAiCompatibleClient } from '../src/llms/client.js'
import { detectProvider, resolveProvider } from '../src/llms/providerRegistry.js'

test('provider routing recognizes copilot and azure prefixes and aliases', () => {
  expect(detectProvider('github-copilot/gpt-5.2')).toBe('github-copilot')
  expect(detectProvider('azure/gpt-4o')).toBe('azure')
  expect(resolveProvider('gpt-4o', { provider: 'copilot' })).toBe('github-copilot')
  expect(resolveProvider('gpt-4o', { provider: 'azure_openai' })).toBe('azure')
})

function stubCopilotSession(credential: CopilotCredential): CopilotSession {
  return { credential: async () => credential } as unknown as CopilotSession
}

test('github-copilot factory injects session headers and the proxy-ep api host', async () => {
  const requests: { url: string; init?: RequestInit }[] = []
  const client = createLlmClient('github-copilot/gpt-5.2', {}, {
    copilotSession: stubCopilotSession({
      // proxy-ep: an enterprise-style claim must re-anchor the request URL.
      access: 'jwt.proxy-ep=proxy.enterprise.example.com;',
      refresh: 'gho_test',
      expires: Math.floor(Date.now() / 1000) + 3_600,
    }),
    fetchImplementation: async (input, init) => {
      requests.push({ url: String(input), ...(init === undefined ? {} : { init }) })
      return new Response(JSON.stringify({
        choices: [{ message: { content: 'hi' }, finish_reason: 'stop' }],
      }))
    },
  })
  expect(client).toBeInstanceOf(OpenAiCompatibleClient)

  expect(client.complete).toBeTypeOf("function")
  await client.complete!({
    model: 'github-copilot/gpt-5.2',
    messages: [
      { role: 'user', content: 'hello' },
      {
        role: 'user',
        content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } }],
      },
    ],
  })
  const request = requests[0]
  expect(request?.url).toBe('https://api.enterprise.example.com/chat/completions')
  const headers = request?.init?.headers as Record<string, string>
  expect(headers.Authorization).toBe('Bearer jwt.proxy-ep=proxy.enterprise.example.com;')
  expect(headers['Editor-Version']).toBe('vscode/1.107.0')
  expect(headers['Copilot-Integration-Id']).toBe('vscode-chat')
  // Last message is a user turn, and it carries an image.
  expect(headers['X-Initiator']).toBe('user')
  expect(headers['Copilot-Vision-Request']).toBe('true')
})

test('github-copilot X-Initiator is agent when the last message is not from the user', async () => {
  const requests: { init?: RequestInit }[] = []
  const client = createLlmClient('github-copilot/gpt-5.2', {}, {
    copilotSession: stubCopilotSession({
      access: 'token',
      refresh: 'gho_test',
      expires: Math.floor(Date.now() / 1000) + 3_600,
    }),
    fetchImplementation: async (_input, init) => {
      requests.push(init === undefined ? {} : { init })
      return new Response(JSON.stringify({
        choices: [{ message: { content: 'hi' }, finish_reason: 'stop' }],
      }))
    },
  })
  expect(client.complete).toBeTypeOf("function")
  await client.complete!({
    model: 'github-copilot/gpt-5.2',
    messages: [
      { role: 'user', content: 'hello' },
      { role: 'assistant', content: 'working' },
      { role: 'tool', tool_call_id: 't1', content: 'result' },
    ],
  })
  const headers = requests[0]?.init?.headers as Record<string, string>
  expect(headers['X-Initiator']).toBe('agent')
  expect(headers['Copilot-Vision-Request']).toBeUndefined()
})

test('azure factory builds the deployment-scoped client from overrides', async () => {
  const requests: { url: string; init?: RequestInit }[] = []
  const client = createLlmClient('azure/gpt-4o', {
    api_key: 'azure-key',
    base_url: 'https://myresource.openai.azure.com/openai/v1',
  }, {
    fetchImplementation: async (input, init) => {
      requests.push({ url: String(input), ...(init === undefined ? {} : { init }) })
      return new Response(JSON.stringify({ status: 'completed', output: [] }))
    },
  })
  expect(client).toBeInstanceOf(AzureOpenAiClient)

  expect(client.complete).toBeTypeOf("function")
  await client.complete!({ model: 'azure/gpt-4o', messages: [{ role: 'user', content: 'hi' }] })
  const request = requests[0]
  expect(request?.url).toContain('myresource.openai.azure.com')
  expect(request?.url).toContain('responses')
  const headers = request?.init?.headers as Record<string, string>
  expect(headers['api-key']).toBe('azure-key')
  expect(headers.Authorization).toBeUndefined()
})

test('codex_transport override validates its value', () => {
  expect(() => createLlmClient('codex/gpt-5.3-codex', { codex_transport: 'carrier-pigeon' }))
    .toThrow(/codex_transport/)
})

test('the pi-ai openai-compat providers register and route by prefix', () => {
  expect(detectProvider('groq/llama-3.3-70b-versatile')).toBe('groq')
  expect(detectProvider('xai/grok-4')).toBe('xai')
  expect(detectProvider('cerebras/llama-3.3-70b')).toBe('cerebras')
  expect(detectProvider('together/meta-llama/Llama-4')).toBe('together')
  expect(detectProvider('huggingface/Qwen/Qwen3')).toBe('huggingface')
  expect(resolveProvider('any/model', { provider: 'hf' })).toBe('huggingface')
  expect(resolveProvider('any/model', { provider: 'bigmodel' })).toBe('zai-coding-cn')
  expect(resolveProvider('any/model', { provider: 'vercel' })).toBe('vercel-ai-gateway')
})

test('multi-api gateways route per model by the catalog api field', async () => {
  const { AnthropicMessagesClient } = await import('../src/llms/anthropic.js')
  const { ResponsesApiClient } = await import('../src/llms/client.js')

  // OpenCode Zen Claude models only speak anthropic-messages, at /zen.
  const claude = createLlmClient('opencode/claude-haiku-4-5', { api_key: 'k' }, {
    fetchImplementation: async () => new Response('{}'),
  })
  expect(claude).toBeInstanceOf(AnthropicMessagesClient)

  // Its GPT models only speak the Responses API, at /zen/v1.
  const gpt = createLlmClient('opencode/gpt-5', { api_key: 'k' }, {
    fetchImplementation: async () => new Response('{}'),
  })
  expect(gpt).toBeInstanceOf(ResponsesApiClient)

  // And a completions model stays on chat-completions.
  const pickle = createLlmClient('opencode/big-pickle', { api_key: 'k' }, {
    fetchImplementation: async () => new Response('{}'),
  })
  expect(pickle).toBeInstanceOf(OpenAiCompatibleClient)
})

test('anthropic-protocol providers build the messages client', async () => {
  const { AnthropicMessagesClient } = await import('../src/llms/anthropic.js')
  expect(createLlmClient('minimax-cn/MiniMax-M2.7', { api_key: 'k' }, {
    fetchImplementation: async () => new Response('{}'),
  })).toBeInstanceOf(AnthropicMessagesClient)
  expect(createLlmClient('vercel-ai-gateway/alibaba/qwen-3-14b', { api_key: 'k' }, {
    fetchImplementation: async () => new Response('{}'),
  })).toBeInstanceOf(AnthropicMessagesClient)
})

test('cloudflare-ai-gateway resolves its account-templated URL or fails loudly', async () => {
  const saved = { account: process.env.CLOUDFLARE_ACCOUNT_ID, gateway: process.env.CLOUDFLARE_GATEWAY_ID }
  delete process.env.CLOUDFLARE_ACCOUNT_ID
  delete process.env.CLOUDFLARE_GATEWAY_ID
  try {
    expect(() => createLlmClient('cloudflare-ai-gateway/gpt-4.1', { api_key: 'k' }, {
      fetchImplementation: async () => new Response('{}'),
    })).toThrow(/CLOUDFLARE_ACCOUNT_ID/)

    process.env.CLOUDFLARE_ACCOUNT_ID = 'acct1'
    process.env.CLOUDFLARE_GATEWAY_ID = 'gw1'
    let seenUrl = ''
    const client = createLlmClient('cloudflare-ai-gateway/gpt-4.1', { api_key: 'k' }, {
      fetchImplementation: async (input) => {
        seenUrl = String(input)
        return new Response(JSON.stringify({ status: 'completed', output: [] }))
      },
    })
    expect(client.complete).toBeTypeOf('function')
    await client.complete!({ model: 'cloudflare-ai-gateway/gpt-4.1', messages: [{ role: 'user', content: 'hi' }] })
    expect(seenUrl).toBe('https://gateway.ai.cloudflare.com/v1/acct1/gw1/openai/responses')
  } finally {
    if (saved.account === undefined) delete process.env.CLOUDFLARE_ACCOUNT_ID
    else process.env.CLOUDFLARE_ACCOUNT_ID = saved.account
    if (saved.gateway === undefined) delete process.env.CLOUDFLARE_GATEWAY_ID
    else process.env.CLOUDFLARE_GATEWAY_ID = saved.gateway
  }
})
