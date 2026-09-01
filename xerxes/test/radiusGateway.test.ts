// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  type RadiusGatewayModel,
  DEFAULT_RADIUS_GATEWAY,
  getRadiusCredentialConfig,
  getRadiusModelsFromConfig,
  loadRadiusGatewayConfig,
  normalizeRadiusGatewayUrl,
  sanitizeRadiusGatewayConfig,
} from '../src/llms/radiusGateway.js'

const VALID_MODEL: RadiusGatewayModel = {
  id: 'claude-sonnet',
  name: 'Claude Sonnet (Radius)',
  reasoning: true,
  input: ['text', 'image'],
  cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
  contextWindow: 200_000,
  maxTokens: 8_192,
}

const VALID_CONFIG = {
  baseUrl: 'https://gateway.example/pi',
  models: [VALID_MODEL, { nope: true }],
}

test('normalizeRadiusGatewayUrl adds the scheme and strips trailing slashes', () => {
  expect(normalizeRadiusGatewayUrl('gateway.example')).toBe('https://gateway.example')
  expect(normalizeRadiusGatewayUrl('http://gateway.example/')).toBe('http://gateway.example')
  expect(normalizeRadiusGatewayUrl('https://gateway.example///')).toBe('https://gateway.example')
})

test('sanitizeRadiusGatewayConfig keeps only well-formed models', () => {
  const config = sanitizeRadiusGatewayConfig(VALID_CONFIG)
  expect(config?.baseUrl).toBe('https://gateway.example/pi')
  expect(config?.models).toEqual([VALID_MODEL])

  expect(sanitizeRadiusGatewayConfig('nope')).toBeUndefined()
  expect(sanitizeRadiusGatewayConfig({ models: [] })).toBeUndefined()
  expect(sanitizeRadiusGatewayConfig({ baseUrl: 'https://x', models: 'nope' })).toBeUndefined()
  // Missing required model fields (cost, contextWindow, maxTokens) drop the entry.
  expect(sanitizeRadiusGatewayConfig({
    baseUrl: 'https://x',
    models: [{ id: 'a', name: 'A', reasoning: false, input: ['text'] }],
  })?.models).toEqual([])
})

test('getRadiusModelsFromConfig resolves pi-messages specs bound to the gateway', () => {
  const config = sanitizeRadiusGatewayConfig(VALID_CONFIG)!
  expect(getRadiusModelsFromConfig('radius', config)).toEqual([{
    ...VALID_MODEL,
    api: 'pi-messages',
    provider: 'radius',
    baseUrl: 'https://gateway.example/pi',
  }])
  expect(getRadiusCredentialConfig(VALID_CONFIG)).toEqual(config)
  expect(getRadiusCredentialConfig(null)).toBeUndefined()
})

test('loadRadiusGatewayConfig GETs /v1/config with the bearer token', async () => {
  const seen: { url: string; init?: RequestInit }[] = []
  const config = await loadRadiusGatewayConfig('https://gateway.example', 'token-1', undefined, {
    fetchImplementation: async (input, init) => {
      seen.push({ url: String(input), ...(init === undefined ? {} : { init }) })
      return new Response(JSON.stringify(VALID_CONFIG), { status: 200 })
    },
  })
  expect(seen[0]?.url).toBe('https://gateway.example/v1/config')
  expect((seen[0]?.init?.headers as Record<string, string>).authorization).toBe('Bearer token-1')
  expect((seen[0]?.init?.headers as Record<string, string>).accept).toBe('application/json')
  expect(config.baseUrl).toBe('https://gateway.example/pi')
})

test('loadRadiusGatewayConfig omits the authorization header without a token', async () => {
  let headers: Record<string, string> = {}
  await loadRadiusGatewayConfig('https://gateway.example', undefined, undefined, {
    fetchImplementation: async (_input, init) => {
      headers = init?.headers as Record<string, string>
      return new Response(JSON.stringify(VALID_CONFIG), { status: 200 })
    },
  })
  expect(headers.authorization).toBeUndefined()
})

test('loadRadiusGatewayConfig reports http failures with a truncated body', async () => {
  const longBody = JSON.stringify({ detail: 'y'.repeat(1_000) })
  await expect(loadRadiusGatewayConfig('https://gateway.example', 't', undefined, {
    fetchImplementation: async () => new Response(longBody, { status: 503 }),
  })).rejects.toThrow(/^Could not load Radius config from https:\/\/gateway\.example: 503: /)

  await loadRadiusGatewayConfig('https://gateway.example', 't', undefined, {
    fetchImplementation: async () => new Response(longBody, { status: 503 }),
  }).catch((error: Error) => {
    const bodyPart = error.message.split(': 503: ')[1] ?? ''
    expect(bodyPart.length).toBeLessThanOrEqual(513)
    expect(bodyPart.endsWith('…')).toBe(true)
  })
})

test('loadRadiusGatewayConfig rejects payloads that fail validation', async () => {
  await expect(loadRadiusGatewayConfig('https://gateway.example', 't', undefined, {
    fetchImplementation: async () => new Response(JSON.stringify({ models: 'all' }), { status: 200 }),
  })).rejects.toThrow('Invalid Radius config from https://gateway.example')

  // Malformed JSON surfaces the parser's own error (pi-ai does not catch it either).
  await expect(loadRadiusGatewayConfig('https://gateway.example', 't', undefined, {
    fetchImplementation: async () => new Response('not json', { status: 200 }),
  })).rejects.toThrow()
})

test('the default gateway is the hosted pi endpoint', () => {
  expect(DEFAULT_RADIUS_GATEWAY).toBe('https://radius.pi.dev')
})
