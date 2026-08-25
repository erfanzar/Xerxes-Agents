// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  CODEX_PROVIDER,
  CodexSession,
} from '../src/auth/codexAuth.js'
import { CredentialStorage } from '../src/auth/storage.js'
import { OAuthToken } from '../src/mcp/oauth.js'

/**
 * Build an unsigned JWT carrying the claims a Codex access token carries.
 * Same encoding the production tokens use for account/plan/expiry claims.
 */
function accessToken(options: {
  accountId?: string
  expiresAt?: number
  plan?: string
} = {}): string {
  const payload = {
    'https://api.openai.com/auth': {
      ...(options.accountId === undefined ? {} : { chatgpt_account_id: options.accountId }),
      ...(options.plan === undefined ? {} : { chatgpt_plan_type: options.plan }),
    },
    ...(options.expiresAt === undefined ? {} : { exp: options.expiresAt }),
  }
  const encode = (value: unknown) => Buffer.from(JSON.stringify(value), 'utf8').toString('base64url')
  return `${encode({ alg: 'none' })}.${encode(payload)}.`
}

async function inTemporaryHome(
  run: (home: string, credentialDirectory: string) => Promise<void>,
): Promise<void> {
  const home = await mkdtemp(join(tmpdir(), 'xerxes-codex-race-'))
  try {
    await run(home, join(home, 'credentials'))
  } finally {
    await rm(home, { force: true, recursive: true })
  }
}

/** A refresh endpoint that always yields a fresh rotated link after a short stall. */
function refreshingFetch(refreshes: { count: number }, rotatedRefreshToken: string) {
  return (async () => {
    refreshes.count += 1
    await Bun.sleep(10)
    return new Response(
      JSON.stringify({
        access_token: accessToken({ accountId: 'acct-shared', expiresAt: 99_000, plan: 'pro' }),
        refresh_token: rotatedRefreshToken,
        expires_in: 3_600,
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    )
  }) as never
}

test('independent CodexSession instances share one refresh flight for the same store', async () => {
  await inTemporaryHome(async (home, credentialDirectory) => {
    // Two distinct CredentialStorage instances over ONE directory: exactly how
    // two minted LLM clients see the world. Keying the single-flight guard by
    // instance would let both POST the same rotating refresh token.
    const storageA = new CredentialStorage(credentialDirectory)
    const storageB = new CredentialStorage(credentialDirectory)
    await storageA.save(CODEX_PROVIDER, new OAuthToken({
      accessToken: accessToken({ accountId: 'acct-shared', expiresAt: 1_000 }),
      refreshToken: 'shared-expiring-refresh',
      expiresAt: 1_000,
    }))

    const refreshes = { count: 0 }
    const makeSession = (storage: CredentialStorage) => new CodexSession({
      environment: {},
      homeDirectory: home,
      now: () => 900,
      storage,
      fetchImplementation: refreshingFetch(refreshes, 'rotated-once'),
    })

    const [first, second] = await Promise.all([makeSession(storageA).credential(), makeSession(storageB).credential()])

    // OpenAI rotates the refresh token on every refresh, so a second parallel
    // POST would invalidate one of the two callers with invalid_grant.
    expect(refreshes.count).toBe(1)
    expect(first.accessToken).toBe(second.accessToken)
    expect(first.planType).toBe('pro')
    expect((await storageA.load(CODEX_PROVIDER))?.refreshToken).toBe('rotated-once')
    expect((await storageB.load(CODEX_PROVIDER))?.refreshToken).toBe('rotated-once')

    // The settled flight is deregistered: a later expiry starts a new refresh.
    const later = makeSession(storageA)
    expect((await later.credential()).accessToken).toContain('.')
    expect(refreshes.count).toBe(1) // token is fresh now; no refresh was needed
  })
})

test('a refresh lost to a concurrent rotation recovers from the freshly stored token', async () => {
  await inTemporaryHome(async (home, credentialDirectory) => {
    const storage = new CredentialStorage(credentialDirectory)
    const winnerAccess = accessToken({ accountId: 'acct-winner', expiresAt: 99_000, plan: 'pro' })
    await storage.save(CODEX_PROVIDER, new OAuthToken({
      accessToken: accessToken({ accountId: 'acct-loser', expiresAt: 1_000 }),
      refreshToken: 'our-now-stale-refresh',
      expiresAt: 1_000,
    }))

    let refreshAttempts = 0
    const session = new CodexSession({
      environment: {},
      homeDirectory: home,
      now: () => 900,
      storage,
      fetchImplementation: (async (_url: string, init?: RequestInit) => {
        refreshAttempts += 1
        const attempted = new URLSearchParams(String(init?.body)).get('refresh_token')
        if (attempted === 'our-now-stale-refresh') {
          // Another surface won the rotation between our load and our failure:
          // it persisted the newer link, then our POST of the old link died.
          await storage.save(CODEX_PROVIDER, new OAuthToken({
            accessToken: winnerAccess,
            refreshToken: 'winner-refresh',
            expiresAt: 99_000,
          }))
          return new Response(
            '{"error":{"message":"token has been used or revoked","type":"invalid_request_error","code":"invalid_grant"}}',
            { status: 401 },
          )
        }
        return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } })
      }) as never,
    })

    // The user's turn must not fail: the store holds a newer usable link.
    const credential = await session.credential()
    expect(credential.accountId).toBe('acct-winner')
    expect(credential.planType).toBe('pro')
    expect(refreshAttempts).toBe(1)
    expect((await storage.load(CODEX_PROVIDER))?.refreshToken).toBe('winner-refresh')
  })
})

test('an expired raced-in token is itself refreshed during CAS recovery', async () => {
  await inTemporaryHome(async (home, credentialDirectory) => {
    const storage = new CredentialStorage(credentialDirectory)
    await storage.save(CODEX_PROVIDER, new OAuthToken({
      accessToken: accessToken({ accountId: 'acct-a', expiresAt: 1_000 }),
      refreshToken: 'first-dead-refresh',
      expiresAt: 1_000,
    }))

    const bodies: string[] = []
    const session = new CodexSession({
      environment: {},
      homeDirectory: home,
      now: () => 900,
      storage,
      fetchImplementation: (async (_url: string, init?: RequestInit) => {
        const attempted = String(new URLSearchParams(String(init?.body)).get('refresh_token'))
        bodies.push(attempted)
        if (attempted === 'first-dead-refresh') {
          // The other surface persisted a link that has since aged past the
          // refresh skew, so recovery must refresh it rather than trust it.
          await storage.save(CODEX_PROVIDER, new OAuthToken({
            accessToken: accessToken({ accountId: 'acct-b', expiresAt: 950 }),
            refreshToken: 'second-also-expired-refresh',
            expiresAt: 950,
          }))
          return new Response(
            '{"error":{"code":"invalid_grant"}}',
            { status: 401 },
          )
        }
        return new Response(
          JSON.stringify({
            access_token: accessToken({ accountId: 'acct-c', expiresAt: 99_000, plan: 'plus' }),
            refresh_token: 'third-live-refresh',
            expires_in: 3_600,
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        )
      }) as never,
    })

    const credential = await session.credential()
    expect(bodies).toEqual(['first-dead-refresh', 'second-also-expired-refresh'])
    expect(credential.accountId).toBe('acct-c')
    expect((await storage.load(CODEX_PROVIDER))?.refreshToken).toBe('third-live-refresh')
  })
})

/** A refresh endpoint that honors its abort signal exactly like real fetch would. */
function abortableRefreshingFetch(refreshes: { count: number }, stallMs = 20) {
  return (async (_url: string, init?: RequestInit) => {
    refreshes.count += 1
    const signal = init?.signal as AbortSignal | undefined
    await new Promise<void>((resolveStall, rejectStall) => {
      const timer = setTimeout(resolveStall, stallMs)
      signal?.addEventListener('abort', () => {
        clearTimeout(timer)
        rejectStall(new DOMException('The operation was aborted.', 'AbortError'))
      }, { once: true })
    })
    return new Response(
      JSON.stringify({
        access_token: accessToken({ accountId: 'acct-shared', expiresAt: 99_000, plan: 'pro' }),
        refresh_token: 'rotated-once',
        expires_in: 3_600,
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } },
    )
  }) as never
}

test('a joiner survives the first caller aborting because shared flights bind no signal', async () => {
  await inTemporaryHome(async (home, credentialDirectory) => {
    const storageA = new CredentialStorage(credentialDirectory)
    const storageB = new CredentialStorage(credentialDirectory)
    await storageA.save(CODEX_PROVIDER, new OAuthToken({
      accessToken: accessToken({ accountId: 'acct-shared', expiresAt: 1_000 }),
      refreshToken: 'shared-expiring-refresh',
      expiresAt: 1_000,
    }))

    const refreshes = { count: 0 }
    const makeSession = (storage: CredentialStorage) => new CodexSession({
      environment: {},
      homeDirectory: home,
      now: () => 900,
      storage,
      fetchImplementation: abortableRefreshingFetch(refreshes),
    })
    const sessionA = makeSession(storageA)
    const sessionB = makeSession(storageB)

    // A carries an abort signal; B arrives signal-free and joins A's flight.
    // Both credential() calls register synchronously before any await runs.
    const controller = new AbortController()
    const first = sessionA.credential(controller.signal)
    const second = sessionB.credential()
    controller.abort()

    // B never asked to abort, so A's caller cancelling must not fail B — and
    // coalescing still means exactly one refresh POST for both.
    const [credentialA, credentialB] = await Promise.all([first, second])
    expect(refreshes.count).toBe(1)
    expect(credentialA.accessToken).toBe(credentialB.accessToken)
    expect(credentialA.planType).toBe('pro')
    expect((await storageB.load(CODEX_PROVIDER))?.refreshToken).toBe('rotated-once')
  })
})

test('concurrent callers still share one flight when it fails', async () => {
  await inTemporaryHome(async (home, credentialDirectory) => {
    const storageA = new CredentialStorage(credentialDirectory)
    const storageB = new CredentialStorage(credentialDirectory)
    await storageA.save(CODEX_PROVIDER, new OAuthToken({
      accessToken: accessToken({ accountId: 'acct-x', expiresAt: 1_000 }),
      refreshToken: 'dead-everywhere',
      expiresAt: 1_000,
    }))

    let attempts = 0
    const makeSession = (storage: CredentialStorage) => new CodexSession({
      environment: {},
      homeDirectory: home,
      now: () => 900,
      storage,
      fetchImplementation: (async () => {
        attempts += 1
        await Bun.sleep(5)
        return new Response('{"error":{"code":"invalid_grant"}}', { status: 401 })
      }) as never,
    })

    // The store stays unchanged, so CAS recovery has nothing newer to load;
    // both coalesced callers must observe the SAME single failure.
    const outcomes = await Promise.allSettled([
      makeSession(storageA).credential(),
      makeSession(storageB).credential(),
    ])
    expect(attempts).toBe(1)
    expect(outcomes.every(outcome => outcome.status === 'rejected')).toBeTrue()
    for (const outcome of outcomes) {
      if (outcome.status === 'rejected') {
        expect(String(outcome.reason)).toMatch(/refresh failed \(401\)/)
      }
    }
  })
})

test('a logout during an in-flight refresh leaves no usable credential on disk', async () => {
  await inTemporaryHome(async (home, credentialDirectory) => {
    const storage = new CredentialStorage(credentialDirectory)
    await storage.save(CODEX_PROVIDER, new OAuthToken({
      accessToken: accessToken({ accountId: 'acct-out', expiresAt: 1_000 }),
      refreshToken: 'pre-logout-refresh',
      expiresAt: 1_000,
    }))

    const winnerAccess = accessToken({ accountId: 'acct-post-logout', expiresAt: 99_000 })
    let attempts = 0
    const session = new CodexSession({
      environment: {},
      homeDirectory: home,
      now: () => 900,
      storage,
      fetchImplementation: (async () => {
        attempts += 1
        // The user signs out while the refresh POST is in flight.
        await storage.remove(CODEX_PROVIDER)
        return new Response(
          JSON.stringify({
            access_token: winnerAccess,
            refresh_token: 'post-logout-refresh',
            expires_in: 3_600,
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        )
      }) as never,
    })

    // The in-flight turn still gets its already-minted credential...
    const credential = await session.credential()
    expect(attempts).toBe(1)
    expect(credential.accessToken).toBe(winnerAccess)

    // ...but the rotated chain must NOT reappear on disk behind the logout,
    // and a second logout correctly reports there was nothing left to remove.
    expect(await session.stored()).toBeUndefined()
    expect(await session.logout()).toBeFalse()
  })
})
