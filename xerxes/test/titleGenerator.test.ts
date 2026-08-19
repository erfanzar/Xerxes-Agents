// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  attemptSessionTitleOnce,
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

test('attemptSessionTitleOnce runs one attempt per session id', async () => {
  resetTitleAttempts()
  let calls = 0
  const run = async () => {
    calls += 1
    return 'Title'
  }

  expect(await attemptSessionTitleOnce('s1', run)).toBe('Title')
  expect(attemptSessionTitleOnce('s1', run)).toBeUndefined()
  expect(await attemptSessionTitleOnce('s2', run)).toBe('Title')
  expect(calls).toBe(2)
  resetTitleAttempts()
})
