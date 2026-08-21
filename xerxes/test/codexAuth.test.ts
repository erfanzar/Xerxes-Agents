// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  CODEX_PROVIDER,
  CodexSession,
  codexAuthHeaders,
  codexClaims,
  codexOAuthConfig,
  fetchCodexModelCatalog,
  importCodexCliTokens,
} from '../src/auth/codexAuth.js'
import { CredentialStorage } from '../src/auth/storage.js'
import { OAuthToken } from '../src/mcp/oauth.js'
import { createLlmClient } from '../src/llms/client.js'
import { detectProvider, getApiKey, resolveProvider } from '../src/llms/providerRegistry.js'
import { CODEX_DEFAULT_MODEL, CODEX_PROFILE_NAME, ProfileStore } from '../src/bridge/profiles.js'
import {
  fallbackReasoningLevels,
  isGradedEffort,
  providerReasoningLevels,
  REASONING_OFF,
  resolveEffort,
  selectableEfforts,
} from '../src/llms/reasoningLevels.js'

/** Build an unsigned JWT carrying the claims a Codex access token carries. */
function accessToken(options: {
  accountId?: string
  email?: string
  expiresAt?: number
  plan?: string
} = {}): string {
  const payload = {
    'https://api.openai.com/auth': {
      ...(options.accountId === undefined ? {} : { chatgpt_account_id: options.accountId }),
      ...(options.plan === undefined ? {} : { chatgpt_plan_type: options.plan }),
    },
    ...(options.email === undefined ? {} : { 'https://api.openai.com/profile': { email: options.email } }),
    ...(options.expiresAt === undefined ? {} : { exp: options.expiresAt }),
  }
  const encode = (value: unknown) => Buffer.from(JSON.stringify(value), 'utf8').toString('base64url')
  return `${encode({ alg: 'none' })}.${encode(payload)}.`
}

async function inTemporaryHome(
  run: (home: string, storage: CredentialStorage) => Promise<void>,
): Promise<void> {
  const home = await mkdtemp(join(tmpdir(), 'xerxes-codex-'))
  try {
    await run(home, new CredentialStorage(join(home, 'credentials')))
  } finally {
    await rm(home, { force: true, recursive: true })
  }
}

async function writeCodexCliAuth(home: string, body: unknown): Promise<void> {
  await mkdir(join(home, '.codex'), { recursive: true })
  await writeFile(join(home, '.codex', 'auth.json'), JSON.stringify(body), 'utf8')
}

test('codexClaims reads the account, plan, and expiry a Codex access token carries', () => {
  const claims = codexClaims(accessToken({
    accountId: 'acct-1',
    email: 'user@example.invalid',
    expiresAt: 1_800_000_000,
    plan: 'pro',
  }))

  expect(claims).toEqual({
    accountId: 'acct-1',
    email: 'user@example.invalid',
    expiresAt: 1_800_000_000,
    planType: 'pro',
  })
})

test('codexClaims degrades to undefined fields instead of throwing on a non-JWT', () => {
  expect(codexClaims('not-a-jwt')).toEqual({
    accountId: undefined,
    email: undefined,
    expiresAt: undefined,
    planType: undefined,
  })
})

test('the Codex CLI session on the machine is adopted when Xerxes has none', async () => {
  await inTemporaryHome(async home => {
    await writeCodexCliAuth(home, {
      tokens: { access_token: accessToken({ accountId: 'acct-9', expiresAt: 2_000_000_000 }), refresh_token: 'r1' },
    })

    const imported = await importCodexCliTokens({}, home)
    expect(imported?.accessToken).toContain('.')
    expect(imported?.refreshToken).toBe('r1')
    expect(imported?.expiresAt).toBe(2_000_000_000)
  })
})

test('an absent, malformed, or API-key-only Codex CLI auth file yields no session', async () => {
  await inTemporaryHome(async home => {
    expect(await importCodexCliTokens({}, home)).toBeUndefined()

    await writeCodexCliAuth(home, { OPENAI_API_KEY: 'sk-test' })
    expect(await importCodexCliTokens({}, home)).toBeUndefined()

    await mkdir(join(home, '.codex'), { recursive: true })
    await writeFile(join(home, '.codex', 'auth.json'), '{ not json', 'utf8')
    expect(await importCodexCliTokens({}, home)).toBeUndefined()
  })
})

test('CODEX_HOME redirects the CLI-session lookup', async () => {
  await inTemporaryHome(async home => {
    const custom = join(home, 'elsewhere')
    await mkdir(custom, { recursive: true })
    await writeFile(
      join(custom, 'auth.json'),
      JSON.stringify({ tokens: { access_token: accessToken({ accountId: 'a' }), refresh_token: 'r' } }),
      'utf8',
    )

    expect(await importCodexCliTokens({ CODEX_HOME: custom }, home)).toBeDefined()
    // The default location is empty, so only the override can have matched.
    expect(await importCodexCliTokens({}, home)).toBeUndefined()
  })
})

test('an adopted CLI session is persisted so it survives the CLI being removed', async () => {
  await inTemporaryHome(async (home, storage) => {
    await writeCodexCliAuth(home, {
      tokens: { access_token: accessToken({ accountId: 'acct-2', expiresAt: 2_000_000_000, plan: 'plus' }), refresh_token: 'r2' },
    })
    const session = new CodexSession({ environment: {}, homeDirectory: home, now: () => 1_000, storage })

    expect((await session.credential()).planType).toBe('plus')
    expect((await storage.load(CODEX_PROVIDER))?.refreshToken).toBe('r2')

    // With the CLI gone, the adopted copy is the only session left and must
    // keep answering on its own.
    await rm(join(home, '.codex'), { force: true, recursive: true })
    expect((await session.credential()).planType).toBe('plus')
  })
})

test('an expiring session is refreshed and the rotated token is stored', async () => {
  await inTemporaryHome(async (home, storage) => {
    const expiring = accessToken({ accountId: 'acct-3', expiresAt: 1_000 })
    await storage.save(CODEX_PROVIDER, new OAuthToken({
      accessToken: expiring,
      refreshToken: 'old-refresh',
      expiresAt: 1_000,
    }))

    const fresh = accessToken({ accountId: 'acct-3', expiresAt: 99_000, plan: 'pro' })
    let sentBody = ''
    const session = new CodexSession({
      environment: {},
      homeDirectory: home,
      now: () => 900,
      storage,
      fetchImplementation: (async (_url: string, init?: RequestInit) => {
        sentBody = String(init?.body ?? '')
        return new Response(
          JSON.stringify({ access_token: fresh, refresh_token: 'new-refresh', expires_in: 3_600 }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        )
      }) as never,
    })

    const credential = await session.credential()
    expect(credential.accessToken).toBe(fresh)
    expect(credential.planType).toBe('pro')
    // Form-encoded refresh_token grant, which is what the token endpoint takes.
    expect(sentBody).toContain('grant_type=refresh_token')
    expect(sentBody).toContain('refresh_token=old-refresh')
    expect((await storage.load(CODEX_PROVIDER))?.refreshToken).toBe('new-refresh')
  })
})

test('Codex refresh rejects an insecure token endpoint override before making a request', async () => {
  await inTemporaryHome(async (home, storage) => {
    await storage.save(CODEX_PROVIDER, new OAuthToken({
      accessToken: accessToken({ expiresAt: 1_000 }),
      refreshToken: 'secret-refresh',
      expiresAt: 1_000,
    }))

    let requests = 0
    const session = new CodexSession({
      environment: { CODEX_REFRESH_TOKEN_URL_OVERRIDE: 'http://attacker.invalid/oauth/token' },
      homeDirectory: home,
      now: () => 900,
      storage,
      fetchImplementation: (async () => {
        requests += 1
        return new Response('{}', { status: 200 })
      }) as never,
    })

    await expect(session.credential()).rejects.toThrow(/tokenUrl must use HTTPS/)
    expect(requests).toBe(0)
  })
})

test('Codex refresh disables automatic redirects that could leak refresh credentials', async () => {
  await inTemporaryHome(async (home, storage) => {
    await storage.save(CODEX_PROVIDER, new OAuthToken({
      accessToken: accessToken({ expiresAt: 1_000 }),
      refreshToken: 'secret-refresh',
      expiresAt: 1_000,
    }))

    let redirect: RequestRedirect | undefined
    const session = new CodexSession({
      environment: {},
      homeDirectory: home,
      now: () => 900,
      storage,
      fetchImplementation: (async (_url: string, init?: RequestInit) => {
        redirect = init?.redirect
        return new Response('redirect refused', { status: 302 })
      }) as never,
    })

    await expect(session.credential()).rejects.toThrow(/refresh failed \(302\)/)
    expect(redirect).toBe('manual')
  })
})

test('a refresh response that omits refresh_token keeps the existing one', async () => {
  await inTemporaryHome(async (home, storage) => {
    await storage.save(CODEX_PROVIDER, new OAuthToken({
      accessToken: accessToken({ expiresAt: 1_000 }),
      refreshToken: 'keep-me',
      expiresAt: 1_000,
    }))

    const session = new CodexSession({
      environment: {},
      homeDirectory: home,
      now: () => 900,
      storage,
      fetchImplementation: (async () => new Response(
        JSON.stringify({ access_token: accessToken({ expiresAt: 99_000 }), expires_in: 3_600 }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      )) as never,
    })

    await session.credential()
    // Dropping it here would strand the session at the next expiry.
    expect((await storage.load(CODEX_PROVIDER))?.refreshToken).toBe('keep-me')
  })
})

test('a quota 429 is reported as exhausted usage, not as a broken credential', async () => {
  await inTemporaryHome(async (home, storage) => {
    await storage.save(CODEX_PROVIDER, new OAuthToken({
      accessToken: accessToken({ expiresAt: 1_000 }),
      refreshToken: 'r',
      expiresAt: 1_000,
    }))

    const session = new CodexSession({
      environment: {},
      homeDirectory: home,
      now: () => 900,
      storage,
      fetchImplementation: (async () => new Response('{"detail":"limit"}', {
        status: 429,
        headers: { 'retry-after': '60' },
      })) as never,
    })

    // Telling the user to re-authenticate could not lift a usage cap.
    await expect(session.credential()).rejects.toThrow(/quota is exhausted/i)
  })
})

test('resolution without any session names the command that fixes it', async () => {
  await inTemporaryHome(async (home, storage) => {
    const session = new CodexSession({ environment: {}, homeDirectory: home, now: () => 1, storage })
    await expect(session.credential()).rejects.toThrow(/xerxes auth login codex/)
  })
})

test('a fresher Codex CLI session is re-adopted over a stale stored one', async () => {
  await inTemporaryHome(async (home, storage) => {
    // Stored credential the CLI already rotated past: its refresh token is dead.
    await storage.save(CODEX_PROVIDER, new OAuthToken({
      accessToken: accessToken({ accountId: 'acct-old', expiresAt: 2_000 }),
      refreshToken: 'stale-refresh',
      expiresAt: 2_000,
    }))
    // The CLI holds a later link in the same chain, still valid.
    await writeCodexCliAuth(home, {
      tokens: { access_token: accessToken({ accountId: 'acct-new', expiresAt: 50_000, plan: 'pro' }), refresh_token: 'fresh-refresh' },
    })

    const session = new CodexSession({ environment: {}, homeDirectory: home, now: () => 1_000, storage })
    const credential = await session.credential()

    expect(credential.accountId).toBe('acct-new')
    expect(credential.planType).toBe('pro')
    // Re-adoption is persisted so the next resolution does not depend on the CLI.
    expect((await storage.load(CODEX_PROVIDER))?.refreshToken).toBe('fresh-refresh')
  })
})

test('a stale stored session does not displace a working explicit login when the CLI is absent', async () => {
  await inTemporaryHome(async (home, storage) => {
    await storage.save(CODEX_PROVIDER, new OAuthToken({
      accessToken: accessToken({ accountId: 'acct-x', expiresAt: 50_000, plan: 'plus' }),
      refreshToken: 'r',
      expiresAt: 50_000,
    }))

    const session = new CodexSession({ environment: {}, homeDirectory: home, now: () => 1_000, storage })
    expect((await session.credential()).planType).toBe('plus')
  })
})

test('a refresh rejected as an invalid grant heals from a valid CLI session', async () => {
  await inTemporaryHome(async (home, storage) => {
    // Stored and CLI sessions carry the same expiry, so the CLI copy is not
    // fresher and cannot win by age; only the refresh rejection re-adopts it.
    await storage.save(CODEX_PROVIDER, new OAuthToken({
      accessToken: accessToken({ expiresAt: 1_000 }),
      refreshToken: 'rotated-away',
      expiresAt: 1_000,
    }))
    await writeCodexCliAuth(home, {
      tokens: { access_token: accessToken({ accountId: 'acct-heal', expiresAt: 1_000, plan: 'pro' }), refresh_token: 'cli-refresh' },
    })

    const session = new CodexSession({
      environment: {},
      homeDirectory: home,
      now: () => 900,
      storage,
      fetchImplementation: (async () => new Response(
        '{"error":{"message":"Your refresh token has already been used","type":"invalid_request_error","code":"refresh_token_reused"}}',
        { status: 401 },
      )) as never,
    })

    const credential = await session.credential()
    expect(credential.accountId).toBe('acct-heal')
    expect((await storage.load(CODEX_PROVIDER))?.refreshToken).toBe('cli-refresh')
  })
})

test('a refresh failure with no usable CLI session still surfaces the provider error', async () => {
  await inTemporaryHome(async (home, storage) => {
    await storage.save(CODEX_PROVIDER, new OAuthToken({
      accessToken: accessToken({ expiresAt: 1_000 }),
      refreshToken: 'dead',
      expiresAt: 1_000,
    }))

    const session = new CodexSession({
      environment: {},
      homeDirectory: home,
      now: () => 900,
      storage,
      fetchImplementation: (async () => new Response(
        '{"error":{"code":"refresh_token_reused"}}',
        { status: 401 },
      )) as never,
    })

    await expect(session.credential()).rejects.toThrow(/refresh failed \(401\)/)
  })
})

test('concurrent callers share one refresh instead of racing the refresh token', async () => {
  await inTemporaryHome(async (home, storage) => {
    await storage.save(CODEX_PROVIDER, new OAuthToken({
      accessToken: accessToken({ expiresAt: 1_000 }),
      refreshToken: 'r',
      expiresAt: 1_000,
    }))

    let refreshes = 0
    const session = new CodexSession({
      environment: {},
      homeDirectory: home,
      now: () => 900,
      storage,
      fetchImplementation: (async () => {
        refreshes += 1
        await Bun.sleep(5)
        return new Response(
          JSON.stringify({ access_token: accessToken({ expiresAt: 99_000 }), refresh_token: 'r2', expires_in: 3_600 }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        )
      }) as never,
    })

    await Promise.all([session.credential(), session.credential(), session.credential()])
    // A provider that rotates the refresh token on use invalidates every loser
    // of a parallel race, signing the user out mid-fan-out.
    expect(refreshes).toBe(1)
  })
})

test('codexAuthHeaders routes the request to the signed-in workspace', () => {
  const headers = codexAuthHeaders({ accessToken: 'tok', accountId: 'acct-4', planType: 'pro' }, 'sid-1')

  expect(headers.Authorization).toBe('Bearer tok')
  expect(headers['chatgpt-account-id']).toBe('acct-4')
  expect(headers.originator).toBe('codex_cli_rs')
  expect(headers.session_id).toBe('sid-1')
})

test('codexAuthHeaders omits the account header when the token carries no workspace', () => {
  expect(codexAuthHeaders({ accessToken: 'tok', accountId: undefined, planType: undefined }))
    .not.toHaveProperty('chatgpt-account-id')
})

test('codexOAuthConfig honors the same endpoint overrides the Codex CLI accepts', () => {
  const config = codexOAuthConfig('http://localhost:1455/auth/callback', {
    CODEX_APP_SERVER_LOGIN_CLIENT_ID: 'app_custom',
    CODEX_REFRESH_TOKEN_URL_OVERRIDE: 'https://staging.invalid/oauth/token',
  })

  expect(config.clientId).toBe('app_custom')
  expect(config.tokenUrl).toBe('https://staging.invalid/oauth/token')
  // offline_access is what yields the refresh token; without it every turn
  // would need a fresh browser round trip.
  expect(config.scopes).toContain('offline_access')
})

test('the codex provider is selected only by explicit routing, never by model name', () => {
  expect(resolveProvider('codex/gpt-5.3-codex')).toBe('openai-codex')
  expect(resolveProvider('chatgpt/gpt-5.3-codex')).toBe('openai-codex')
  expect(resolveProvider('gpt-5.3-codex', { base_url: 'https://chatgpt.com/backend-api/codex' }))
    .toBe('openai-codex')

  // A bare `-codex` model stays on the metered API: silently moving it onto
  // the user's ChatGPT plan would change who pays for the turn.
  expect(detectProvider('gpt-5.3-codex')).toBe('openai')
})

test('the codex provider exposes no API-key environment variable', () => {
  expect(getApiKey('openai-codex', {}, { OPENAI_API_KEY: 'sk-should-not-be-used' })).toBe('')
})

test('codex requests carry OAuth headers and omit the parameters the backend rejects', async () => {
  await inTemporaryHome(async (home, storage) => {
    await storage.save(CODEX_PROVIDER, new OAuthToken({
      accessToken: accessToken({ accountId: 'acct-5', expiresAt: 99_000 }),
      refreshToken: 'r',
      expiresAt: 99_000,
    }))

    let seenUrl = ''
    let seenHeaders: Record<string, string> = {}
    let seenBody: Record<string, unknown> = {}
    const client = createLlmClient('codex/gpt-5.3-codex', {}, {
      codexSession: new CodexSession({ environment: {}, homeDirectory: home, now: () => 1_000, storage }),
      fetchImplementation: (async (url: string, init?: RequestInit) => {
        seenUrl = String(url)
        seenHeaders = init?.headers as Record<string, string>
        seenBody = JSON.parse(String(init?.body)) as Record<string, unknown>
        return new Response(
          'data: {"type":"response.completed","response":{"status":"completed"}}\n\ndata: [DONE]\n\n',
          {
            status: 200,
            headers: { 'Content-Type': 'text/event-stream' },
          },
        )
      }) as never,
    })

    for await (const _delta of client.stream({
      model: 'codex/gpt-5.3-codex',
      messages: [{ role: 'user', content: 'hi' }],
      maxTokens: 100,
      temperature: 0.5,
      topP: 0.9,
    })) {
      // Drain; the assertions are about the request that was built.
    }

    expect(seenUrl).toBe('https://chatgpt.com/backend-api/codex/responses')
    expect(seenHeaders.Authorization).toStartWith('Bearer ')
    expect(seenHeaders['chatgpt-account-id']).toBe('acct-5')

    expect(seenBody.model).toBe('gpt-5.3-codex')
    expect(seenBody.store).toBe(false)
    expect(seenBody.stream).toBe(true)
    // The backend answers an unsupported parameter with 400, never by
    // ignoring it, so these must not be sent at all.
    expect(seenBody).not.toHaveProperty('max_output_tokens')
    expect(seenBody).not.toHaveProperty('max_tokens')
    expect(seenBody).not.toHaveProperty('temperature')
    expect(seenBody).not.toHaveProperty('top_p')
  })
})

test('the Responses transport caps output with max_output_tokens, not max_tokens', async () => {
  let seenBody: Record<string, unknown> = {}
  const client = createLlmClient('gpt-4.1', {}, {
    apiKey: 'sk-test',
    responsesApi: true,
    fetchImplementation: (async (_url: string, init?: RequestInit) => {
      seenBody = JSON.parse(String(init?.body)) as Record<string, unknown>
      return new Response(
        'data: {"type":"response.completed","response":{"status":"completed"}}\n\ndata: [DONE]\n\n',
        {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        },
      )
    }) as never,
  })

  for await (const _delta of client.stream({
    model: 'gpt-4.1',
    messages: [{ role: 'user', content: 'hi' }],
    maxTokens: 64,
    thinking: { effort: 'medium' },
  })) {
    // Drain.
  }

  // `max_tokens` is the chat-completions spelling; the Responses API rejects it.
  expect(seenBody.max_output_tokens).toBe(64)
  expect(seenBody).not.toHaveProperty('max_tokens')
  expect(seenBody.reasoning).toEqual({ effort: 'medium' })
})

test('the Codex catalog is discovered live and plan-scoped, not hard-coded', async () => {
  let seenUrl = ''
  let seenHeaders: Record<string, string> = {}
  const catalog = await fetchCodexModelCatalog(
    { accessToken: 'tok', accountId: 'acct-6', planType: 'pro' },
    {
      fetchImplementation: (async (url: string, init?: RequestInit) => {
        seenUrl = String(url)
        seenHeaders = init?.headers as Record<string, string>
        return Response.json({
          models: [
            {
              id: 'gpt-5.6-sol',
              display_name: 'GPT-5.6-Sol',
              context_window: 272_000,
              default_reasoning_level: 'low',
              supported_reasoning_levels: [
                { effort: 'low', description: 'Fast responses' },
                { effort: 'ultra' },
              ],
            },
            { slug: 'gpt-5.4', display_name: 'GPT-5.4' },
            { display_name: 'nameless entry is skipped' },
          ],
        })
      }) as never,
    },
  )

  // The catalog route is gated on client_version; omitting it is a 400.
  expect(seenUrl).toContain('/models?client_version=')
  expect(seenHeaders['chatgpt-account-id']).toBe('acct-6')
  expect(catalog).toEqual([
    {
      id: 'gpt-5.6-sol',
      displayName: 'GPT-5.6-Sol',
      contextLimit: 272_000,
      defaultReasoningLevel: 'low',
      harnessCoupled: false,
      reasoningLevels: [
        { effort: 'low', description: 'Fast responses' },
        { effort: 'ultra', description: undefined },
      ],
    },
    {
      id: 'gpt-5.4',
      displayName: 'GPT-5.4',
      contextLimit: undefined,
      defaultReasoningLevel: undefined,
      harnessCoupled: false,
      reasoningLevels: [],
    },
  ])
})

test('a failed Codex catalog request surfaces the status instead of an empty list', async () => {
  await expect(fetchCodexModelCatalog(
    { accessToken: 'tok', accountId: undefined, planType: undefined },
    { fetchImplementation: (async () => new Response('nope', { status: 401 })) as never },
  )).rejects.toThrow(/catalog request failed \(401\)/)
})

test('a model picked from the codex profile routes to the subscription backend', () => {
  const overrides = { provider: 'openai-codex', base_url: 'https://chatgpt.com/backend-api/codex' }

  // The picker hands back bare catalog ids, so routing has to come from the
  // active profile rather than the model name.
  expect(resolveProvider('gpt-5.5', overrides)).toBe('openai-codex')
  expect(resolveProvider('codex-auto-review', overrides)).toBe('openai-codex')
  // Same id with no codex profile active stays on the metered API.
  expect(resolveProvider('gpt-5.5', {})).toBe('openai')
})

test('the codex profile is built in so the picker lists it before any setup', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-profiles-'))
  try {
    const store = new ProfileStore(join(root, 'profiles.json'))
    const codex = store.list().find(profile => profile.name === CODEX_PROFILE_NAME)

    expect(codex?.provider).toBe('openai-codex')
    expect(codex?.base_url).toBe('https://chatgpt.com/backend-api/codex')
    // Subscription-backed: the credential is an OAuth session, never a key.
    expect(codex?.api_key).toBe('')
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

/** Capture the request body a Responses-transport client builds. */
async function capturedResponsesBody(
  model: string,
  overrides: Record<string, unknown>,
  request: Record<string, unknown>,
  factoryOptions: Record<string, unknown> = {},
): Promise<Record<string, unknown>> {
  let body: Record<string, unknown> = {}
  const client = createLlmClient(model, overrides, {
    responsesApi: true,
    apiKey: 'sk-test',
    ...factoryOptions,
    fetchImplementation: (async (_url: string, init?: RequestInit) => {
      body = JSON.parse(String(init?.body)) as Record<string, unknown>
      return new Response(
        'data: {"type":"response.completed","response":{"status":"completed"}}\n\ndata: [DONE]\n\n',
        {
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
        },
      )
    }) as never,
  })
  for await (const _delta of client.stream(request as never)) {
    // Drain; the assertion is about the request that was built.
  }
  return body
}

test('a stable prompt_cache_key is derived from the reusable system prefix', async () => {
  const system = { role: 'system', content: 'You are a careful assistant with a long stable preamble.' }

  const first = await capturedResponsesBody('gpt-4.1', {}, {
    model: 'gpt-4.1',
    messages: [system, { role: 'user', content: 'first question' }],
  })
  const second = await capturedResponsesBody('gpt-4.1', {}, {
    model: 'gpt-4.1',
    messages: [system, { role: 'user', content: 'a completely different second question' }],
  })

  expect(first.prompt_cache_key).toBeString()
  // Without a key the backend does not route a repeat to the machine holding
  // the prefix, so an agent loop pays full price for its preamble every turn.
  expect(second.prompt_cache_key).toBe(first.prompt_cache_key as string)
})

test('a different system prefix or model gets a different cache key', async () => {
  const base = await capturedResponsesBody('gpt-4.1', {}, {
    model: 'gpt-4.1',
    messages: [{ role: 'system', content: 'preamble A' }, { role: 'user', content: 'q' }],
  })
  const otherPrefix = await capturedResponsesBody('gpt-4.1', {}, {
    model: 'gpt-4.1',
    messages: [{ role: 'system', content: 'preamble B' }, { role: 'user', content: 'q' }],
  })
  const otherModel = await capturedResponsesBody('gpt-4.1-mini', {}, {
    model: 'gpt-4.1-mini',
    messages: [{ role: 'system', content: 'preamble A' }, { role: 'user', content: 'q' }],
  })

  expect(otherPrefix.prompt_cache_key).not.toBe(base.prompt_cache_key as string)
  expect(otherModel.prompt_cache_key).not.toBe(base.prompt_cache_key as string)
})

test('systemSegments drive the cache key so volatile tail text cannot bust it', async () => {
  const stable = { name: 'identity', text: 'stable preamble that repeats every turn' }

  const first = await capturedResponsesBody('gpt-4.1', {}, {
    model: 'gpt-4.1',
    messages: [{ role: 'user', content: 'q1' }],
    systemSegments: [stable],
  })
  const second = await capturedResponsesBody('gpt-4.1', {}, {
    model: 'gpt-4.1',
    messages: [{ role: 'user', content: 'a totally different q2' }],
    systemSegments: [stable],
  })

  expect(first.prompt_cache_key).toBe(second.prompt_cache_key as string)
})

test('a request with no reusable prefix sends no cache key at all', async () => {
  const body = await capturedResponsesBody('gpt-4.1', {}, {
    model: 'gpt-4.1',
    messages: [{ role: 'user', content: 'no system prompt here' }],
  })

  // A key over a one-off prompt would never be hit and only adds a field.
  expect(body).not.toHaveProperty('prompt_cache_key')
})

test('third-party Responses hosts are not sent a field they may reject', async () => {
  // `responses_api` can be enabled for any OpenAI-compatible endpoint, and a
  // strict one answers an unknown parameter with 400 rather than ignoring it.
  const body = await capturedResponsesBody('deepseek-chat', {}, {
    model: 'deepseek-chat',
    messages: [{ role: 'system', content: 'stable preamble' }, { role: 'user', content: 'q' }],
  })

  expect(body).not.toHaveProperty('prompt_cache_key')
})

test('reasoning levels come from the model, not a fixed four-item menu', () => {
  // The Codex catalog publishes different sets per model — some reach `ultra`,
  // others stop at `xhigh` — so a fixed list rejects valid efforts.
  const sol = providerReasoningLevels(
    [
      { effort: 'low', description: 'Fast responses' },
      { effort: 'medium' },
      { effort: 'high' },
      { effort: 'xhigh' },
      { effort: 'max' },
      { effort: 'ultra' },
    ],
    'low',
  )

  expect(selectableEfforts(sol)).toEqual(['off', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'])
  expect(sol.defaultEffort).toBe('low')
  expect(sol.source).toBe('provider')
})

test('an effort is validated against the model and returned in the provider spelling', () => {
  const set = providerReasoningLevels([{ effort: 'xhigh' }, { effort: 'ultra' }], 'xhigh')

  expect(resolveEffort(set, 'ULTRA')).toBe('ultra')
  expect(resolveEffort(set, '  xhigh ')).toBe('xhigh')
  expect(resolveEffort(set, 'off')).toBe(REASONING_OFF)
  // `high` is valid on other models but not on this one.
  expect(resolveEffort(set, 'high')).toBeUndefined()
  expect(resolveEffort(set, '')).toBeUndefined()
})

test('providers with no capability endpoint fall back per provider, not globally', () => {
  const anthropic = fallbackReasoningLevels('anthropic')
  const generic = fallbackReasoningLevels(undefined)

  expect(anthropic.source).toBe('fallback')
  expect(selectableEfforts(anthropic)).toEqual(['off', 'low', 'medium', 'high'])
  // Anthropic's budget-based thinking and OpenAI's effort scale are different
  // vocabularies; the table is keyed by provider so they can diverge.
  expect(anthropic.levels[0]?.description).not.toBe(generic.levels[0]?.description)
})

test('a model reporting no levels degrades to the fallback rather than an empty menu', () => {
  const empty = providerReasoningLevels([], undefined)

  expect(selectableEfforts(empty)).toEqual(['off'])
  expect(selectableEfforts(fallbackReasoningLevels('openai')).length).toBeGreaterThan(1)
})

test('every model the plan returns is selectable, harness-flagged or not', async () => {
  const respond = (models: unknown) => (async () => Response.json({ models })) as never
  const catalogModels = [
    { id: 'gpt-5.6-sol', tool_mode: 'code_mode_only', multi_agent_version: 'v2', use_responses_lite: true },
    { id: 'codex-auto-review', tool_mode: 'code_mode_only', multi_agent_version: 'v1', use_responses_lite: true },
    { id: 'gpt-5.5', tool_mode: null, multi_agent_version: null, use_responses_lite: false },
    { id: 'gpt-5.4-mini' },
  ]
  const credential = { accessToken: 'tok', accountId: 'a', planType: 'pro' }

  // The harness flag records how the Codex CLI drives a model; it is not a
  // restriction on calling it, and hiding these would drop capability the
  // subscription pays for.
  const offered = await fetchCodexModelCatalog(credential, { fetchImplementation: respond(catalogModels) })
  expect(offered.map(model => model.id))
    .toEqual(['gpt-5.6-sol', 'codex-auto-review', 'gpt-5.5', 'gpt-5.4-mini'])
  expect(offered.filter(model => model.harnessCoupled).map(model => model.id))
    .toEqual(['gpt-5.6-sol', 'codex-auto-review'])

  // A caller that wants only the plain-Responses subset can still ask.
  const filtered = await fetchCodexModelCatalog(credential, {
    fetchImplementation: respond(catalogModels),
    excludeHarnessModels: true,
  })
  expect(filtered.map(model => model.id)).toEqual(['gpt-5.5', 'gpt-5.4-mini'])
})

test('harness detection keys on capability flags, not on model names', async () => {
  // The flag must keep describing the right models as the catalog changes,
  // rather than tracking a naming pattern that can shift.
  const catalog = await fetchCodexModelCatalog(
    { accessToken: 'tok', accountId: 'a', planType: 'pro' },
    {
      fetchImplementation: (async () => Response.json({
        models: [
          { id: 'totally-innocuous-name', use_responses_lite: true },
          { id: 'another-plain-name', multi_agent_version: 'v3' },
          { id: 'ordinary' },
        ],
      })) as never,
    },
  )

  expect(catalog.filter(model => model.harnessCoupled).map(model => model.id))
    .toEqual(['totally-innocuous-name', 'another-plain-name'])
})

test('the built-in codex profile starts on the plan\'s newest model', () => {
  expect(CODEX_DEFAULT_MODEL).toBe('codex/gpt-5.6-sol')
})

test('toggle-shaped providers offer a switch, not a fake graded scale', () => {
  for (const provider of ['zhipu', 'qwen', 'kimi', 'kimi-code', 'minimax'] as const) {
    const set = fallbackReasoningLevels(provider)
    expect(set.shape).toBe('toggle')
    expect(selectableEfforts(set)).toEqual(['off', 'on'])
  }
})

test('a provider that chooses reasoning by model offers nothing to select', () => {
  const set = fallbackReasoningLevels('deepseek')

  expect(set.shape).toBe('inherent')
  // Offering a bare `off` would imply reasoning can be disabled, which it cannot.
  expect(selectableEfforts(set)).toEqual([])
  expect(resolveEffort(set, 'high')).toBeUndefined()
  expect(resolveEffort(set, 'off')).toBeUndefined()
})

test('effort-shaped providers keep a graded scale', () => {
  for (const provider of ['openai', 'openrouter', 'anthropic', 'gemini'] as const) {
    const set = fallbackReasoningLevels(provider)
    expect(set.shape).toBe('effort')
    expect(selectableEfforts(set)).toEqual(['off', 'low', 'medium', 'high'])
  }
})

test('local and custom endpoints assume a switch rather than invent gradations', () => {
  // The served model is whatever the user loaded, so a graded menu would be
  // a guess the backend may not honor.
  for (const provider of ['ollama', 'lmstudio', 'custom'] as const) {
    expect(fallbackReasoningLevels(provider).shape).toBe('toggle')
  }
})

test('switch positions never reach the wire as an effort word', () => {
  // `off` and `on` are Xerxes-side states; no provider documents them as levels.
  expect(isGradedEffort('off')).toBe(false)
  expect(isGradedEffort('on')).toBe(false)
  expect(isGradedEffort('ON')).toBe(false)
  expect(isGradedEffort(undefined)).toBe(false)
  expect(isGradedEffort('high')).toBe(true)
  expect(isGradedEffort('ultra')).toBe(true)
})

test('enabling thinking on a toggle provider sends no reasoning_effort field', async () => {
  let body: Record<string, unknown> = {}
  const client = createLlmClient('glm-5', { provider: 'zhipu' }, {
    apiKey: 'sk-test',
    fetchImplementation: (async (_url: string, init?: RequestInit) => {
      body = JSON.parse(String(init?.body)) as Record<string, unknown>
      return new Response('data: [DONE]\n\n', {
        status: 200,
        headers: { 'Content-Type': 'text/event-stream' },
      })
    }) as never,
  })

  for await (const _delta of client.stream({
    model: 'glm-5',
    messages: [{ role: 'user', content: 'hi' }],
    thinking: { effort: 'on' },
  })) {
    // Drain.
  }

  expect(body).not.toHaveProperty('reasoning_effort')
})
