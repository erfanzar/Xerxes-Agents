// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  ANTHROPIC_OAUTH_CLIENT_ID,
  ANTHROPIC_OAUTH_IDENTITY_PROMPT,
  ANTHROPIC_REDIRECT_URI,
  ANTHROPIC_TOKEN_URL,
  AnthropicOAuthSession,
  anthropicOAuthHeaders,
  exchangeAnthropicAuthorizationCode,
  isAnthropicOAuthToken,
  parseAnthropicAuthorizationInput,
  refreshAnthropicToken,
  toClaudeCodeToolName,
} from '../src/auth/anthropicOAuth.js'
import { AnthropicMessagesClient } from '../src/llms/anthropic.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'

async function inTemporaryHome(run: (home: string) => Promise<void>): Promise<void> {
  const home = await mkdtemp(join(tmpdir(), 'xerxes-anthropic-oauth-'))
  try {
    await run(home)
  } finally {
    await rm(home, { force: true, recursive: true })
  }
}

const NOW = 1_700_000_000

test('only sk-ant-oat tokens take the OAuth surface', () => {
  expect(isAnthropicOAuthToken('sk-ant-oat01-abc')).toBe(true)
  expect(isAnthropicOAuthToken('sk-ant-api03-abc')).toBe(false)
})

test('OAuth headers carry the Claude Code identity; bearer-only tokens stay plain', () => {
  expect(anthropicOAuthHeaders('sk-ant-oat01-x', true)).toEqual({
    Authorization: 'Bearer sk-ant-oat01-x',
    'anthropic-beta': 'claude-code-20250219,oauth-2025-04-20',
    'User-Agent': 'claude-cli/2.1.75',
    'x-app': 'cli',
  })
  expect(anthropicOAuthHeaders('ghp_other', false)).toEqual({ Authorization: 'Bearer ghp_other' })
})

test('tool names normalize to Claude Code canonical casing, unknown names pass through', () => {
  expect(toClaudeCodeToolName('read')).toBe('Read')
  expect(toClaudeCodeToolName('BASH')).toBe('Bash')
  expect(toClaudeCodeToolName('todoWrite')).toBe('TodoWrite')
  expect(toClaudeCodeToolName('my_custom_tool')).toBe('my_custom_tool')
})

test('authorization input parses URLs, code#state pairs, query strings, and bare codes', () => {
  expect(parseAnthropicAuthorizationInput(
    `https://console.anthropic.com/oauth/authorize/callback?code=abc&state=xyz`,
  )).toEqual({ code: 'abc', state: 'xyz' })
  expect(parseAnthropicAuthorizationInput('abc#xyz')).toEqual({ code: 'abc', state: 'xyz' })
  expect(parseAnthropicAuthorizationInput('code=abc&state=xyz')).toEqual({ code: 'abc', state: 'xyz' })
  expect(parseAnthropicAuthorizationInput('  bare-code  ')).toEqual({ code: 'bare-code' })
  expect(parseAnthropicAuthorizationInput('')).toEqual({})
})

test('code exchange posts the exact pi-ai wire shape and derives the expiry', async () => {
  await inTemporaryHome(async home => {
    const requests: { url: string; init?: RequestInit }[] = []
    const session = new AnthropicOAuthSession({
      xerxesHome: home,
      environment: {},
      now: () => NOW,
      fetchImplementation: async (url, init) => {
        requests.push({ url: String(url), ...(init === undefined ? {} : { init }) })
        return new Response(JSON.stringify({
          access_token: 'sk-ant-oat01-access',
          refresh_token: 'refresh-1',
          expires_in: 36_000,
        }))
      },
    })
    const credential = await session.login({
      openUrl: () => {},
      callbackPort: 0,
      timeoutMs: 2_000,
      manualInput: async () => 'auth-code',
    })
    expect(requests[0]?.url).toBe(ANTHROPIC_TOKEN_URL)
    expect(JSON.parse(String(requests[0]?.init?.body))).toMatchObject({
      grant_type: 'authorization_code',
      client_id: ANTHROPIC_OAUTH_CLIENT_ID,
      code: 'auth-code',
      redirect_uri: ANTHROPIC_REDIRECT_URI,
    })
    // The expiry derives from the wall clock at exchange time (pi-ai parity),
    // so assert with a small tolerance rather than an injected now.
    expect(credential.access).toBe('sk-ant-oat01-access')
    expect(credential.refresh).toBe('refresh-1')
    expect(Math.abs(credential.expires - (Math.floor(Date.now() / 1_000) + 36_000))).toBeLessThan(5)
  })
})

test('refresh posts only the refresh fields and persists the rotated credential', async () => {
  await inTemporaryHome(async home => {
    const session = new AnthropicOAuthSession({
      xerxesHome: home,
      now: () => NOW,
      fetchImplementation: async (url, init) => {
        expect(String(url)).toBe(ANTHROPIC_TOKEN_URL)
        expect(JSON.parse(String(init?.body))).toEqual({
          grant_type: 'refresh_token',
          client_id: ANTHROPIC_OAUTH_CLIENT_ID,
          refresh_token: 'refresh-1',
        })
        return new Response(JSON.stringify({
          access_token: 'sk-ant-oat01-next',
          refresh_token: 'refresh-2',
          expires_in: 3_600,
        }))
      },
    })
    const refreshed = await session.refresh({
      access: 'sk-ant-oat01-old',
      refresh: 'refresh-1',
      expires: NOW - 10,
    })
    expect(refreshed.access).toBe('sk-ant-oat01-next')
    expect(refreshed.refresh).toBe('refresh-2')
    const storedRaw = JSON.parse(await readFile(join(home, 'auth', 'anthropic-oauth.json'), 'utf8')) as Record<string, unknown>
    expect(storedRaw['access']).toBe('sk-ant-oat01-next')
  })
})

test('credential resolution: stored wins, expired refreshes, missing falls to ambient env token', async () => {
  await inTemporaryHome(async home => {
    // Missing: explicit error, no ambient token.
    const missing = new AnthropicOAuthSession({
      xerxesHome: home,
      environment: {},
      now: () => NOW,
    })
    await expect(missing.credential()).rejects.toThrow(/No Anthropic subscription session/)

    // Ambient env token: returned unrefreshable, never persisted.
    const ambient = new AnthropicOAuthSession({
      xerxesHome: home,
      environment: { ANTHROPIC_AUTH_TOKEN: 'sk-ant-oat01-env' },
      now: () => NOW,
    })
    const ambientCredential = await ambient.credential()
    expect(ambientCredential).toMatchObject({ access: 'sk-ant-oat01-env', refresh: '' })
    await expect(readFile(join(home, 'auth', 'anthropic-oauth.json'), 'utf8')).rejects.toThrow()
  })
})

test('an expired stored credential refreshes on resolve', async () => {
  await inTemporaryHome(async home => {
    const credentialPath = join(home, 'auth', 'anthropic-oauth.json')
    const { mkdir, writeFile } = await import('node:fs/promises')
    await mkdir(join(home, 'auth'), { recursive: true })
    await writeFile(credentialPath, `${JSON.stringify({
      access: 'sk-ant-oat01-expired',
      refresh: 'refresh-1',
      expires: NOW - 1_000,
    })}\n`, 'utf8')

    const session = new AnthropicOAuthSession({
      xerxesHome: home,
      environment: {},
      now: () => NOW,
      fetchImplementation: async (url, init) => {
        expect(String(url)).toBe(ANTHROPIC_TOKEN_URL)
        expect(JSON.parse(String(init?.body))).toMatchObject({ refresh_token: 'refresh-1' })
        return new Response(JSON.stringify({
          access_token: 'sk-ant-oat01-refreshed',
          refresh_token: 'refresh-2',
          expires_in: 3_600,
        }))
      },
    })
    const credential = await session.credential()
    expect(credential.access).toBe('sk-ant-oat01-refreshed')
    expect((JSON.parse(await readFile(credentialPath, 'utf8')) as Record<string, unknown>)['access'])
      .toBe('sk-ant-oat01-refreshed')
  })
})

test('login exchanges a manual pasted code when the browser never returns', async () => {
  await inTemporaryHome(async home => {
    const session = new AnthropicOAuthSession({
      xerxesHome: home,
      environment: {},
      now: () => NOW,
      fetchImplementation: async (url, init) => {
        const body = JSON.parse(String(init?.body)) as Record<string, string>
        expect(body.grant_type).toBe('authorization_code')
        expect(body.code).toBe('pasted-code')
        expect(body.redirect_uri).toBe(ANTHROPIC_REDIRECT_URI)
        expect(String(url)).toBe(ANTHROPIC_TOKEN_URL)
        return new Response(JSON.stringify({
          access_token: 'sk-ant-oat01-manual',
          refresh_token: 'refresh-manual',
          expires_in: 3_600,
        }))
      },
    })
    const credential = await session.login({
      openUrl: () => {},
      callbackPort: 0,
      timeoutMs: 2_000,
      manualInput: async () => 'pasted-code',
    })
    expect(credential.access).toBe('sk-ant-oat01-manual')
    expect((await session.stored())?.access).toBe('sk-ant-oat01-manual')
    expect(await session.logout()).toBe(true)
  })
})

const TOOL: ToolDefinition = {
  type: 'function',
  function: { name: 'read', description: 'Read a file.', parameters: { type: 'object', properties: {} } },
}

test('the anthropic transport switches to the OAuth surface for sk-ant-oat tokens', async () => {
  const requests: { init?: RequestInit }[] = []
  const client = new AnthropicMessagesClient({
    apiKey: 'sk-ant-api03-plain',
    baseUrl: 'https://example.invalid',
    resolveOAuthToken: async () => 'sk-ant-oat01-session',
    fetchImplementation: async (_input, init) => {
      requests.push(init === undefined ? {} : { init })
      return new Response(JSON.stringify({
        content: [{ type: 'text', text: 'ok' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 1, output_tokens: 1 },
      }))
    },
  })

  await client.complete({
    model: 'anthropic/claude-sonnet-4-6',
    messages: [{ role: 'user', content: 'hi' }],
    tools: [TOOL],
  })

  const headers = requests[0]?.init?.headers as Record<string, string>
  expect(headers.Authorization).toBe('Bearer sk-ant-oat01-session')
  expect(headers['anthropic-beta']).toBe('claude-code-20250219,oauth-2025-04-20')
  expect(headers['User-Agent']).toBe('claude-cli/2.1.75')
  expect(headers['x-app']).toBe('cli')
  expect(headers['x-api-key']).toBeUndefined()

  const payload = JSON.parse(String(requests[0]?.init?.body)) as Record<string, unknown>
  const system = payload.system as { type: string; text: string }[]
  expect(system[0]?.text).toBe(ANTHROPIC_OAUTH_IDENTITY_PROMPT)
  expect((payload.tools as { name: string }[])[0]?.name).toBe('Read')
})

test('without an OAuth token the transport keeps the API-key request untouched', async () => {
  const requests: { init?: RequestInit }[] = []
  const client = new AnthropicMessagesClient({
    apiKey: 'sk-ant-api03-plain',
    baseUrl: 'https://example.invalid',
    resolveOAuthToken: async () => undefined,
    fetchImplementation: async (_input, init) => {
      requests.push(init === undefined ? {} : { init })
      return new Response(JSON.stringify({
        content: [{ type: 'text', text: 'ok' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 1, output_tokens: 1 },
      }))
    },
  })

  await client.complete({ model: 'anthropic/claude-sonnet-4-6', messages: [{ role: 'user', content: 'hi' }] })
  const headers = requests[0]?.init?.headers as Record<string, string>
  expect(headers['x-api-key']).toBe('sk-ant-api03-plain')
  expect(headers.Authorization).toBeUndefined()
  expect(headers['anthropic-beta']).toBeUndefined()
  const payload = JSON.parse(String(requests[0]?.init?.body)) as Record<string, unknown>
  // No system prompt was configured, so none is sent — and never an OAuth
  // identity array on the API-key path.
  expect(payload.system).toBeUndefined()
})
