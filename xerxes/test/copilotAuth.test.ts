// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterAll, describe, expect, test } from 'bun:test'

import {
  COPILOT_DEVICE_CODE_URL,
  COPILOT_MODELS_API_VERSION,
  COPILOT_TOKEN_EXCHANGE_URL,
  COPILOT_TOKEN_URL,
  copilotApiBase,
  copilotAuthHeaders,
  copilotRequestHeaders,
  CopilotSession,
  fetchCopilotModels,
  type CopilotCredential,
} from '../src/auth/copilotAuth.js'

/** One observed HTTP call from the session under test. */
interface ObservedRequest {
  readonly body: string | undefined
  readonly headers: Record<string, string>
  readonly method: string
  readonly url: string
}

interface ScriptedResponse {
  readonly payload: unknown
  readonly retryAfter?: string
  readonly status?: number
}

function scriptedFetch(
  script: readonly ScriptedResponse[],
  observed: ObservedRequest[] = [],
): typeof fetch {
  let call = 0
  return (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    observed.push({
      body: typeof init?.body === 'string' ? init.body : undefined,
      headers: (init?.headers ?? {}) as Record<string, string>,
      method: init?.method ?? 'GET',
      url: String(input),
    })
    const next = script[call]
    call += 1
    if (!next) throw new Error(`unexpected extra fetch call #${call}`)
    const status = next.status ?? 200
    if (status === 429) {
      return new Response(JSON.stringify(next.payload), {
        status,
        headers: {
          'Content-Type': 'application/json',
          ...(next.retryAfter === undefined ? {} : { 'Retry-After': next.retryAfter }),
        },
      })
    }
    return Response.json(next.payload, { status })
  }) as typeof fetch
}

function session(options: Partial<ConstructorParameters<typeof CopilotSession>[0]> = {}): CopilotSession {
  return new CopilotSession(options)
}

const INDIVIDUAL_TOKEN = 'tid=abc;exp=1800000000;sku=copilot_for_individuals;proxy-ep=https://api.individual.githubcopilot.com;st=seat'
const ENTERPRISE_TOKEN = 'tid=def;sku=copilot_for_enterprise;proxy-ep=https://copilot-api.contoso.example'

afterAll(async () => {
  // Nothing global to tear down; kept for symmetry with tmp cleanup below.
})

describe('copilotApiBase', () => {
  test('derives the base from the proxy-ep claim', () => {
    expect(copilotApiBase(INDIVIDUAL_TOKEN)).toBe('https://api.individual.githubcopilot.com')
    expect(copilotApiBase(ENTERPRISE_TOKEN)).toBe('https://copilot-api.contoso.example')
  })

  test('falls back to the enterprise host spelling from a domain claim', () => {
    expect(copilotApiBase('tid=x;ent=contoso.com')).toBe('https://copilot-api.contoso.com')
  })

  test('falls back to the individual default when no claim names a base', () => {
    expect(copilotApiBase('tid=x;sku=copilot_for_individuals')).toBe(
      'https://api.individual.githubcopilot.com',
    )
  })
})

describe('copilotAuthHeaders', () => {
  test('sends the bearer token plus the pinned static headers', () => {
    const credential: CopilotCredential = { access: 'copilot-token', expires: 1, refresh: 'gho_x' }
    expect(copilotAuthHeaders(credential)).toEqual({
      Accept: 'application/json',
      Authorization: 'Bearer copilot-token',
      'Copilot-Integration-Id': 'vscode-chat',
      'Editor-Plugin-Version': 'copilot-chat/0.35.0',
      'Editor-Version': 'vscode/1.107.0',
      'User-Agent': 'GitHubCopilotChat/0.35.0',
    })
  })
})

describe('copilotRequestHeaders', () => {
  test('marks user-initiated turns as user and agent turns as agent', () => {
    expect(copilotRequestHeaders({ lastMessageRole: 'user' })['X-Initiator']).toBe('user')
    expect(copilotRequestHeaders({ lastMessageRole: 'assistant' })['X-Initiator']).toBe('agent')
    expect(copilotRequestHeaders()['X-Initiator']).toBe('agent')
  })

  test('always declares the conversation-edits intent and opts into vision only with images', () => {
    const withImages = copilotRequestHeaders({ hasImages: true, lastMessageRole: 'user' })
    expect(withImages['Openai-Intent']).toBe('conversation-edits')
    expect(withImages['Copilot-Vision-Request']).toBe('true')
    const withoutImages = copilotRequestHeaders({ hasImages: false })
    expect(withoutImages).not.toHaveProperty('Copilot-Vision-Request')
    expect(withoutImages['Openai-Intent']).toBe('conversation-edits')
  })
})

describe('CopilotSession', () => {
  test('device flow: start, pending poll, token, and copilot exchange persist a session', async () => {
    const home = await mkdtemp(join(tmpdir(), 'xerxes-copilot-login-'))
    try {
      const observed: ObservedRequest[] = []
      const sleeps: number[] = []
      const exchanges = scriptedFetch([
        // Device code start.
        {
          payload: {
            device_code: 'device-code-1',
            expires_in: 900,
            interval: 0,
            user_code: 'ABCD-1234',
            verification_uri: 'https://github.com/login/device',
          },
        },
        // First poll: still waiting for the user.
        { payload: { error: 'authorization_pending', error_description: 'not yet' } },
        // Second poll: the user finished.
        { payload: { access_token: 'gho_user_token', scope: 'read:user', token_type: 'bearer' } },
        // Copilot token exchange.
        {
          payload: {
            endpoints: { api: 'https://api.individual.githubcopilot.com' },
            expires_at: 1_800_000_000,
            token: INDIVIDUAL_TOKEN,
          },
        },
      ], observed)
      const credential = await session({
        fetchImplementation: exchanges,
        sleep: async ms => {
          sleeps.push(ms)
        },
        xerxesHome: home,
      }).login((userCode, verificationUri) => {
        expect(userCode).toBe('ABCD-1234')
        expect(verificationUri).toBe('https://github.com/login/device')
      })

      expect(credential).toEqual({
        access: INDIVIDUAL_TOKEN,
        expires: 1_800_000_000,
        refresh: 'gho_user_token',
      })
      expect(observed.map(request => request.url)).toEqual([
        COPILOT_DEVICE_CODE_URL,
        COPILOT_TOKEN_URL,
        COPILOT_TOKEN_URL,
        COPILOT_TOKEN_EXCHANGE_URL,
      ])
      const exchange = observed[3]
      expect(exchange?.headers.Authorization).toBe('Bearer gho_user_token')
      expect(exchange?.headers['Editor-Version']).toBe('vscode/1.107.0')
      const start = observed[0]
      expect(start?.body).toContain(`client_id=Iv1.b507a08c87ecfe98`)
      expect(start?.body).toContain('scope=read')
      const persisted = JSON.parse(await readFile(join(home, 'auth', 'copilot.json'), 'utf8')) as Record<string, unknown>
      expect(persisted.access).toBe(INDIVIDUAL_TOKEN)
      expect(persisted.refresh).toBe('gho_user_token')
    } finally {
      await rm(home, { recursive: true, force: true })
    }
  })

  test('slow_down stretches the poll interval by five seconds', async () => {
    const home = await mkdtemp(join(tmpdir(), 'xerxes-copilot-slow-'))
    try {
      const observed: ObservedRequest[] = []
      const sleeps: number[] = []
      const exchanges = scriptedFetch([
        {
          payload: {
            device_code: 'device-code-2',
            expires_in: 900,
            interval: 0,
            user_code: 'SLOW-0001',
            verification_uri: 'https://github.com/login/device',
          },
        },
        { payload: { error: 'slow_down' } },
        { payload: { access_token: 'gho_slow_token' } },
        { payload: { expires_at: 1_800_000_000, token: INDIVIDUAL_TOKEN } },
      ], observed)
      await session({
        fetchImplementation: exchanges,
        sleep: async ms => {
          sleeps.push(ms)
        },
        xerxesHome: home,
      }).login(() => undefined)

      expect(observed[1]?.body).toContain('device_code=device-code-2')
      expect(observed[2]?.body).toContain('device_code=device-code-2')
      expect(sleeps).toEqual([0, 5_000])
    } finally {
      await rm(home, { recursive: true, force: true })
    }
  })

  test('a credential within the five-minute skew re-exchanges and persists', async () => {
    const home = await mkdtemp(join(tmpdir(), 'xerxes-copilot-skew-'))
    try {
      const now = 1_700_000_000
      await Bun.write(join(home, 'auth', 'copilot.json'), `${JSON.stringify({
        access: 'stale-access',
        expires: now + 200,
        refresh: 'gho_refresh_token',
      })}\n`)
      const observed: ObservedRequest[] = []
      const exchanges = scriptedFetch([
        { payload: { expires_at: now + 1_800, token: INDIVIDUAL_TOKEN } },
      ], observed)
      const credential = await session({
        environment: {},
        fetchImplementation: exchanges,
        now: () => now,
        xerxesHome: home,
      }).credential()

      expect(credential.access).toBe(INDIVIDUAL_TOKEN)
      expect(credential.refresh).toBe('gho_refresh_token')
      expect(observed).toHaveLength(1)
      expect(observed[0]?.headers.Authorization).toBe('Bearer gho_refresh_token')
    } finally {
      await rm(home, { recursive: true, force: true })
    }
  })

  test('a credential outside the skew is returned without any network call', async () => {
    const home = await mkdtemp(join(tmpdir(), 'xerxes-copilot-fresh-'))
    try {
      const now = 1_700_000_000
      await Bun.write(join(home, 'auth', 'copilot.json'), `${JSON.stringify({
        access: 'fresh-access',
        expires: now + 400,
        refresh: 'gho_refresh_token',
      })}\n`)
      const strict = scriptedFetch([])
      const credential = await session({
        environment: {},
        fetchImplementation: strict,
        now: () => now,
        xerxesHome: home,
      }).credential()

      expect(credential.access).toBe('fresh-access')
    } finally {
      await rm(home, { recursive: true, force: true })
    }
  })

  test('adopts an ambient GitHub token when nothing is stored', async () => {
    const home = await mkdtemp(join(tmpdir(), 'xerxes-copilot-env-'))
    try {
      const observed: ObservedRequest[] = []
      const exchanges = scriptedFetch([
        { payload: { expires_at: 1_800_000_000, token: INDIVIDUAL_TOKEN } },
      ], observed)
      const credential = await session({
        environment: { COPILOT_GITHUB_TOKEN: 'gho_env_token' },
        fetchImplementation: exchanges,
        xerxesHome: home,
      }).credential()

      expect(credential.refresh).toBe('gho_env_token')
      expect(observed[0]?.headers.Authorization).toBe('Bearer gho_env_token')
      const persisted = JSON.parse(await readFile(join(home, 'auth', 'copilot.json'), 'utf8')) as Record<string, unknown>
      expect(persisted.refresh).toBe('gho_env_token')
    } finally {
      await rm(home, { recursive: true, force: true })
    }
  })

  test('no stored session and no ambient token names the fix instead of fetching', async () => {
    const home = await mkdtemp(join(tmpdir(), 'xerxes-copilot-missing-'))
    try {
      const strict = scriptedFetch([])
      expect(session({
        environment: { GITHUB_TOKEN: '  ', GH_TOKEN: undefined },
        fetchImplementation: strict,
        xerxesHome: home,
      }).credential()).rejects.toThrow('xerxes auth login copilot')
    } finally {
      await rm(home, { recursive: true, force: true })
    }
  })

  test('refresh re-mints from the stored GitHub token', async () => {
    const home = await mkdtemp(join(tmpdir(), 'xerxes-copilot-refresh-'))
    try {
      const observed: ObservedRequest[] = []
      const exchanges = scriptedFetch([
        { payload: { expires_at: 1_800_000_100, token: ENTERPRISE_TOKEN } },
      ], observed)
      const credential = await session({
        fetchImplementation: exchanges,
        xerxesHome: home,
      }).refresh({
        access: 'old',
        expires: 1,
        refresh: 'gho_refresh_token',
      })

      expect(credential.access).toBe(ENTERPRISE_TOKEN)
      expect(credential.expires).toBe(1_800_000_100)
      expect(observed).toHaveLength(1)
    } finally {
      await rm(home, { recursive: true, force: true })
    }
  })
})

describe('fetchCopilotModels', () => {
  const credential: CopilotCredential = {
    access: INDIVIDUAL_TOKEN,
    expires: 1_800_000_000,
    refresh: 'gho_x',
  }

  test('keeps only tool-capable, policy-enabled, picker-visible models', async () => {
    const observed: ObservedRequest[] = []
    const models = scriptedFetch([
      {
        payload: {
          data: [
            {
              id: 'gpt-4o',
              model_picker_enabled: true,
              // No policy field: falls back to allowed.
            },
            {
              capabilities: { supports: { tool_calls: false } },
              id: 'no-tools',
            },
            {
              id: 'policy-disabled',
              policy: { state: 'disabled' },
            },
            {
              id: 'policy-enabled',
              policy: { state: 'enabled' },
            },
            {
              capabilities: { supports: { tool_calls: true } },
              id: 'defaults-everything',
            },
          ],
        },
      },
    ], observed)
    const ids = await fetchCopilotModels(credential, { fetchImplementation: models })

    expect(ids).toEqual(['gpt-4o', 'policy-enabled', 'defaults-everything'])
    expect(observed[0]?.url).toBe('https://api.individual.githubcopilot.com/models')
    expect(observed[0]?.headers['X-GitHub-Api-Version']).toBe(COPILOT_MODELS_API_VERSION)
    expect(observed[0]?.headers.Authorization).toBe(`Bearer ${INDIVIDUAL_TOKEN}`)
  })

  test('retries a 429 honouring retry-after seconds', async () => {
    const observed: ObservedRequest[] = []
    const sleeps: number[] = []
    const models = scriptedFetch([
      { payload: { error: 'rate limited' }, retryAfter: '2', status: 429 },
      { payload: { data: [{ id: 'gpt-4o' }] } },
    ], observed)
    const ids = await fetchCopilotModels(credential, {
      fetchImplementation: models,
      sleep: async ms => {
        sleeps.push(ms)
      },
    })

    expect(ids).toEqual(['gpt-4o'])
    expect(observed).toHaveLength(2)
    expect(sleeps).toEqual([2_000])
  })

  test('surfaces a persistent failure after the bounded retries', async () => {
    const models = scriptedFetch([
      { payload: { error: 'rate limited' }, retryAfter: '0', status: 429 },
      { payload: { error: 'rate limited' }, retryAfter: '0', status: 429 },
      { payload: { error: 'rate limited' }, retryAfter: '0', status: 429 },
    ])
    expect(fetchCopilotModels(credential, { fetchImplementation: models }))
      .rejects.toThrow('Copilot model list request failed (429)')
  })
})
