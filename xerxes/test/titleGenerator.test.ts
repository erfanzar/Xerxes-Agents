// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  attemptSessionTitle,
  generateSessionTitle,
  resetTitleAttempts,
  sanitizeTitle,
  titleModelFor,
  titlePrompt,
  TITLE_MAX_CHARS,
} from '../src/daemon/titleGenerator.js'
import type { LlmClient } from '../src/llms/client.js'
import type { ProviderProfile } from '../src/bridge/profiles.js'

function profile(provider: string, over: Partial<ProviderProfile> = {}): ProviderProfile {
  return {
    api_key: 'k',
    base_url: 'http://example.invalid/v1',
    model: 'session-model',
    name: 'p',
    provider,
    sampling: {},
    ...over,
  }
}

function fakeClient(content: string): LlmClient {
  return {
    async *stream() {
      yield { type: 'text', text: content } as never
    },
    async complete() {
      return { content } as never
    },
  }
}

function failingClient(): LlmClient {
  return {
    async *stream() {
      throw new Error('provider down')
    },
  }
}

test('titleModelFor prefers the cheap tier of the session provider', () => {
  expect(titleModelFor('anthropic/claude-opus-4-1', profile('anthropic'))).toBe('claude-haiku-4-5-20251001')
  expect(titleModelFor('gpt-5.5', profile('openai'))).toBe('gpt-4o-mini')
  expect(titleModelFor('gemini-2.5-pro', profile('gemini'))).toBe('gemini-2.0-flash-lite')
})

test('titleModelFor falls back to the session model for unknown providers', () => {
  expect(titleModelFor('some-plugin-model', profile('unknown-vendor'))).toBe('some-plugin-model')
  expect(titleModelFor('some-plugin-model', undefined)).toBe('some-plugin-model')
})

test('titlePrompt carries both sides of the opening exchange, clipped', () => {
  const prompt = titlePrompt('u'.repeat(5_000), 'a'.repeat(5_000))
  expect(prompt).toContain('User: ')
  expect(prompt).toContain('Assistant: ')
  expect(prompt.length).toBeLessThan(5_000)
})

test('sanitizeTitle strips quotes, punctuation, newlines, and caps length', () => {
  expect(sanitizeTitle('"Fix the login bug."')).toBe('Fix the login bug')
  expect(sanitizeTitle('\n\n  Refactor auth flow  \n')).toBe('Refactor auth flow')
  expect(sanitizeTitle('')).toBeUndefined()
  expect(sanitizeTitle('   \n  ')).toBeUndefined()
  const long = sanitizeTitle(`"${'x'.repeat(200)}"`)
  expect(long!.length).toBeLessThanOrEqual(TITLE_MAX_CHARS)
  expect(long!.endsWith('…')).toBe(true)
})

test('generateSessionTitle returns a sanitized title from the provider', async () => {
  const title = await generateSessionTitle({
    userText: 'How do I rotate JWT keys?',
    assistantText: 'You rotate them by...',
    sessionModel: 'session-model',
    profile: profile('openai'),
    clientFactory: () => fakeClient('"JWT key rotation"\n'),
  })
  expect(title).toBe('JWT key rotation')
})

test('generateSessionTitle swallows provider failures to undefined', async () => {
  const title = await generateSessionTitle({
    userText: 'hi',
    assistantText: 'hello',
    sessionModel: 'session-model',
    profile: profile('openai'),
    clientFactory: () => failingClient(),
  })
  expect(title).toBeUndefined()
})

test('attemptSessionTitle retries a failure but stays bounded', async () => {
  resetTitleAttempts()
  let calls = 0
  const fail = async () => {
    calls += 1
    return undefined
  }

  // A transient failure used to be permanent: one miss per daemon lifetime
  // left the session unnamed forever. It now retries, but only so far.
  expect(await attemptSessionTitle('s1', fail)).toBeUndefined()
  expect(await attemptSessionTitle('s1', fail)).toBeUndefined()
  expect(await attemptSessionTitle('s1', fail)).toBeUndefined()
  expect(attemptSessionTitle('s1', fail)).toBeUndefined()
  expect(calls).toBe(3)
  resetTitleAttempts()
})

test('attemptSessionTitle keeps attempts per session id', async () => {
  resetTitleAttempts()
  let calls = 0
  const run = async () => {
    calls += 1
    return 'Title'
  }

  expect(await attemptSessionTitle('s1', run)).toBe('Title')
  expect(await attemptSessionTitle('s2', run)).toBe('Title')
  expect(calls).toBe(2)
  resetTitleAttempts()
})

test('attemptSessionTitle refuses a second concurrent attempt', async () => {
  resetTitleAttempts()
  let calls = 0
  let release: (value: string | undefined) => void = () => {}
  const pending = new Promise<string | undefined>(resolve => {
    release = resolve
  })
  const run = () => {
    calls += 1
    return pending
  }

  const first = attemptSessionTitle('s1', run)

  // A later turn can end while the first provider call is still open; without
  // the in-flight guard the same session would pay for both.
  expect(attemptSessionTitle('s1', run)).toBeUndefined()
  release('Title')
  expect(await first).toBe('Title')
  expect(calls).toBe(1)
  resetTitleAttempts()
})

test('generateSessionTitle asks for enough tokens to survive a reasoning preamble', async () => {
  let seenMaxTokens = 0
  const title = await generateSessionTitle({
    userText: 'What does add do?',
    assistantText: 'It sums a and b.',
    sessionModel: 'reasoning-model',
    profile: profile('kimi-code'),
    clientFactory: () =>
      ({
        complete: async (request: { maxTokens?: number }) => {
          seenMaxTokens = request.maxTokens ?? 0

          // A reasoning model spends the budget on thinking first. Under a
          // 40-token ceiling it returns empty content, the request succeeds,
          // and the chat is silently never named — which is exactly how this
          // shipped broken while looking healthy.
          return { content: seenMaxTokens < 128 ? '' : 'Explain add function' }
        },
        close: async () => undefined,
      }) as unknown as LlmClient,
  })

  expect(seenMaxTokens).toBeGreaterThanOrEqual(128)
  expect(title).toBe('Explain add function')
})

test('generateSessionTitle falls back to the session model when the cheap tier fails', async () => {
  const tried: string[] = []
  const title = await generateSessionTitle({
    userText: 'fix the socket leak',
    assistantText: 'found it in runtime.ts',
    sessionModel: 'gpt-5.6-sol',
    profile: profile('openai'),
    clientFactory: (model: string) => {
      tried.push(model)

      // An OpenAI-compatible proxy declared as `provider: "openai"` may not
      // serve the cheap model at all — a 100% failure, not a flaky one.
      if (model !== 'gpt-5.6-sol') {
        throw new Error('unknown model')
      }

      return {
        complete: async () => ({ content: 'Fix socket leak' }),
        close: async () => undefined,
      } as unknown as LlmClient
    },
  })

  expect(title).toBe('Fix socket leak')
  expect(tried).toEqual(['gpt-4o-mini', 'gpt-5.6-sol'])
})
