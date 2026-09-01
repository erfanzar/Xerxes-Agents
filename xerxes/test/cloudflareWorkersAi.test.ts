// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ConfigurationError } from '../src/core/errors.js'
import {
  cloudflareWorkersAiBaseUrl,
  createCloudflareWorkersAiClient,
  resolveCloudflareWorkersAiConfig,
} from '../src/llms/cloudflareWorkersAi.js'
import { OpenAiCompatibleClient } from '../src/llms/client.js'
import { createLlmClient } from '../src/llms/client.js'
import { detectProvider } from '../src/llms/providerRegistry.js'

const ENV = {
  CLOUDFLARE_API_KEY: 'cf-key',
  CLOUDFLARE_ACCOUNT_ID: 'acct-123',
}

test('config resolution names the missing variable instead of guessing', () => {
  expect(() => resolveCloudflareWorkersAiConfig({})).toThrow(/CLOUDFLARE_API_KEY and CLOUDFLARE_ACCOUNT_ID/)
  expect(() => resolveCloudflareWorkersAiConfig({ CLOUDFLARE_API_KEY: 'k' })).toThrow(/CLOUDFLARE_ACCOUNT_ID/)
  expect(() => resolveCloudflareWorkersAiConfig({ CLOUDFLARE_ACCOUNT_ID: 'a' })).toThrow(/CLOUDFLARE_API_KEY/)
  expect(resolveCloudflareWorkersAiConfig(ENV)).toEqual({ accountId: 'acct-123', apiKey: 'cf-key' })
  // An override api key wins over the environment value.
  expect(resolveCloudflareWorkersAiConfig(ENV, { apiKey: 'override' }).apiKey).toBe('override')
})

test('the workers-ai base URL materializes the account id placeholder', () => {
  expect(cloudflareWorkersAiBaseUrl('acct-123')).toBe(
    'https://api.cloudflare.com/client/v4/accounts/acct-123/ai/v1',
  )
})

test('model routing sends @cf/ models and the workers-ai prefix to the provider', () => {
  expect(detectProvider('cloudflare-workers-ai/@cf/meta/llama-3-8b-instruct')).toBe('cloudflare-workers-ai')
  expect(detectProvider('workers-ai/@cf/meta/llama-3-8b-instruct')).toBe('cloudflare-workers-ai')
})

test('the built client speaks OpenAI completions on the account-scoped endpoint', async () => {
  const requests: { url: string; init?: RequestInit }[] = []
  const client = createCloudflareWorkersAiClient({
    env: ENV,
    fetchImplementation: async (input, init) => {
      requests.push({ url: String(input), ...(init === undefined ? {} : { init }) })
      return new Response(JSON.stringify({
        choices: [{ message: { content: 'hi' }, finish_reason: 'stop' }],
      }))
    },
  })
  expect(client).toBeInstanceOf(OpenAiCompatibleClient)

  expect(client.complete).toBeTypeOf('function')
  await client.complete!({
    model: 'cloudflare-workers-ai/@cf/meta/llama-3-8b-instruct',
    messages: [{ role: 'user', content: 'hello' }],
  })
  const request = requests[0]
  expect(request?.url).toBe('https://api.cloudflare.com/client/v4/accounts/acct-123/ai/v1/chat/completions')
  const headers = request?.init?.headers as Record<string, string>
  expect(headers.Authorization).toBe('Bearer cf-key')
  const body = JSON.parse(String(request?.init?.body)) as Record<string, unknown>
  expect(body.model).toBe('@cf/meta/llama-3-8b-instruct')
  expect(body.stream).toBe(false)
})

test('createLlmClient routes the provider through the factory branch', async () => {
  const requests: { url: string }[] = []
  const saved = { ...process.env }
  process.env.CLOUDFLARE_API_KEY = 'factory-key'
  process.env.CLOUDFLARE_ACCOUNT_ID = 'factory-acct'
  try {
    const client = createLlmClient('cloudflare-workers-ai/@cf/meta/llama-3-8b-instruct', {}, {
      fetchImplementation: async (input, init) => {
        requests.push({ url: String(input) })
        void init
        return new Response(JSON.stringify({
          choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }],
        }))
      },
    })
    expect(client).toBeInstanceOf(OpenAiCompatibleClient)
    expect(client.complete).toBeTypeOf('function')
    await client.complete!({
      model: 'cloudflare-workers-ai/@cf/meta/llama-3-8b-instruct',
      messages: [{ role: 'user', content: 'hello' }],
    })
    expect(requests[0]?.url).toContain('/accounts/factory-acct/ai/v1/chat/completions')
  } finally {
    process.env = saved
  }
})

test('missing ambient credentials surface an actionable configuration error', () => {
  expect(() => createCloudflareWorkersAiClient({ env: {} })).toThrow(ConfigurationError)
})
