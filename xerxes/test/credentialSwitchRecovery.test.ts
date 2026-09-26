// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ProviderError } from '../src/core/errors.js'
import type { LlmClient, LlmDelta } from '../src/llms/client.js'
import { credentialFingerprint } from '../src/llms/credentialFingerprint.js'
import { classifyError, ErrorKind } from '../src/runtime/errorClassifier.js'
import { createAgentState } from '../src/streaming/events.js'
import { runTurn } from '../src/streaming/loop.js'

/** A provider-neutral client: one credential identity, and either an answer or a used-up plan. */
function client(identity: string, outcome: 'answer' | 'exhausted' | 'burst', calls: string[]): LlmClient {
  return {
    async authFingerprint() { return credentialFingerprint({ authorization: `Bearer ${identity}` }) },
    async *stream(): AsyncGenerator<LlmDelta> {
      calls.push(identity)
      if (outcome === 'exhausted') throw new ProviderError('any-provider', 'request failed (429): {"error":{"type":"usage_limit_reached","message":"The usage limit has been reached"}}', undefined, { status: 429 })
      if (outcome === 'burst') throw new ProviderError('any-provider', 'request failed (429): Too many requests, slow down', undefined, { status: 429 })
      yield { content: `answered as ${identity}` }
    },
  }
}

async function drive(dependencies: Parameters<typeof runTurn>[1]) {
  const events: Array<Record<string, unknown>> = []
  for await (const event of runTurn({ model: 'm', state: createAgentState(), userMessage: 'hi' }, dependencies)) events.push(event as Record<string, unknown>)
  return events
}

test('a used-up plan is told apart from a burst rate limit, for any provider wording', () => {
  const quota = (message: string) => classifyError(new ProviderError('p', message, undefined, { status: 429 })).kind
  expect(quota('{"type":"usage_limit_reached"}')).toBe(ErrorKind.QUOTA_EXCEEDED)                  // OpenAI / Codex
  expect(quota('You exceeded your current quota, please check your plan')).toBe(ErrorKind.QUOTA_EXCEEDED)
  expect(quota('Claude usage limit reached. Your limit resets at 5pm')).toBe(ErrorKind.QUOTA_EXCEEDED)
  expect(quota('{"code":"1113","message":"Insufficient balance or no resource package"}')).toBe(ErrorKind.QUOTA_EXCEEDED) // Z.ai
  expect(quota('Insufficient credits')).toBe(ErrorKind.QUOTA_EXCEEDED)                            // OpenRouter
  expect(quota('Too many requests, slow down')).toBe(ErrorKind.RATE_LIMIT)
  expect(classifyError(new ProviderError('p', 'Rate limit exceeded', undefined, { status: 429 })).retryable).toBe(true)
})

test('when the account or key changes after a used-up plan, the turn retries once as the new credential', async () => {
  const calls: string[] = []
  let refreshes = 0
  const events = await drive({
    llm: client('exhausted-account', 'exhausted', calls),
    // The switch lands a moment after the refusal, as an auto switcher's does.
    refreshLlm: () => (++refreshes < 3 ? client('exhausted-account', 'exhausted', calls) : client('fresh-account', 'answer', calls)),
    credentialSwitchWaitMs: 5_000,
    delay: async () => {},
  })
  expect(calls).toEqual(['exhausted-account', 'fresh-account'])
  expect(events.some(event => event.type === 'text' && String(event.text).includes('answered as fresh-account'))).toBe(true)
  expect(events.some(event => event.type === 'provider_retry' && String(event.error).includes('newly selected account or key'))).toBe(true)
  expect(events.at(-1)).toMatchObject({ type: 'turn_done' })
})

test('with no credential change a used-up plan fails promptly instead of burning retries', async () => {
  const calls: string[] = []
  const waits: number[] = []
  const events = await drive({
    llm: client('only-account', 'exhausted', calls),
    refreshLlm: () => client('only-account', 'exhausted', calls),
    credentialSwitchWaitMs: 300,
    retryDelays: [10_000, 10_000, 10_000, 10_000],
    delay: async ms => { waits.push(ms); await new Promise(resolve => setTimeout(resolve, Math.min(ms, 50))) },
  })
  expect(calls).toEqual(['only-account'])
  expect(waits.every(ms => ms <= 500)).toBe(true) // only the short switch polls, never a 10s retry
  expect(events.some(event => event.type === 'provider_retry' && event.final === true)).toBe(true)
})

test('a burst rate limit still retries normally, and recovery never runs for it', async () => {
  const calls: string[] = []
  let attempts = 0
  const events = await drive({
    llm: {
      async authFingerprint() { return 'same' },
      async *stream(): AsyncGenerator<LlmDelta> {
        calls.push('call')
        if (++attempts === 1) throw new ProviderError('p', 'Too many requests', undefined, { status: 429 })
        yield { content: 'ok' }
      },
    },
    refreshLlm: () => { throw new Error('recovery must not run for a burst limit') },
    retryDelays: [0],
    delay: async () => {},
  })
  expect(calls).toHaveLength(2)
  expect(events.at(-1)).toMatchObject({ type: 'turn_done' })
})

import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { CodexSession } from '../src/auth/codexAuth.js'
import { CredentialStorage } from '../src/auth/storage.js'
import { createLlmClient } from '../src/llms/client.js'

function jwt(accountId: string): string {
  const encode = (value: unknown) => Buffer.from(JSON.stringify(value)).toString('base64url')
  return `${encode({ alg: 'none' })}.${encode({ 'https://api.openai.com/auth': { chatgpt_account_id: accountId }, exp: 99_000 })}.`
}

test('real clients report a new identity when the login or key underneath them changes', async () => {
  // Subscription login switched outside Xerxes (Codex CLI / an account switcher).
  const home = await mkdtemp(join(tmpdir(), 'xr-fingerprint-'))
  try {
    await mkdir(join(home, '.codex'), { recursive: true })
    const writeAccount = (id: string) => writeFile(join(home, '.codex', 'auth.json'), JSON.stringify({ tokens: { access_token: jwt(id), refresh_token: `${id}-r` } }))
    await writeAccount('first')
    const codex = createLlmClient('codex/gpt-5.3-codex', {}, { codexTransport: 'sse', codexSession: new CodexSession({ environment: {}, homeDirectory: home, now: () => 1_000, storage: new CredentialStorage(join(home, 'credentials')) }) })
    const before = await codex.authFingerprint?.()
    await writeAccount('second')
    const after = await codex.authFingerprint?.()
    expect(before).toBeString()
    expect(after).toBeString()
    expect(after).not.toBe(before)
  } finally { await rm(home, { recursive: true, force: true }) }

  // An API key edited in the profile (any OpenAI-compatible provider).
  const keyA = createLlmClient('openai/gpt-4.1', { api_key: 'sk-first' })
  const keyB = createLlmClient('openai/gpt-4.1', { api_key: 'sk-second' })
  expect(await keyA.authFingerprint?.()).not.toBe(await keyB.authFingerprint?.())
  expect(await keyA.authFingerprint?.()).toBe(await createLlmClient('openai/gpt-4.1', { api_key: 'sk-first' }).authFingerprint?.())
})
