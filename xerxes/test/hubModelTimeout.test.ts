// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { HUB_MODEL_MIN_TIMEOUT_MS, HUB_MODEL_TIMEOUT_MS, hubModelTimeoutMs } from '../src/daemon/hubModels.js'
import { completeLlm, type LlmClient } from '../src/llms/client.js'

test('a XERXES model call has no time limit unless one is set explicitly', () => {
  // Two, then ten minutes cut slow reasoning models off mid-thought.
  expect(HUB_MODEL_TIMEOUT_MS).toBe(Number.POSITIVE_INFINITY)
  expect(hubModelTimeoutMs({})).toBe(Number.POSITIVE_INFINITY)
  expect(hubModelTimeoutMs({ XERXES_HUB_MODEL_TIMEOUT_MS: '3600000' })).toBe(3_600_000)
})

test('an explicit override keeps a floor; a malformed one means no limit', () => {
  expect(hubModelTimeoutMs({ XERXES_HUB_MODEL_TIMEOUT_MS: '1000' })).toBe(HUB_MODEL_MIN_TIMEOUT_MS)
  expect(hubModelTimeoutMs({ XERXES_HUB_MODEL_TIMEOUT_MS: 'soon' })).toBe(HUB_MODEL_TIMEOUT_MS)
  expect(hubModelTimeoutMs({ XERXES_HUB_MODEL_TIMEOUT_MS: '-5' })).toBe(HUB_MODEL_TIMEOUT_MS)
})

test('completeLlm with an infinite deadline still ends on the caller signal', async () => {
  const client = {
    stream: () => (async function* () { await new Promise(() => {}) })(),
    complete: (_request: unknown, signal: AbortSignal) => new Promise((_, reject) => signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true })),
  } as unknown as LlmClient
  const controller = new AbortController()
  const pending = completeLlm(client, { model: 'm', messages: [] } as never, controller.signal, { timeoutMs: Number.POSITIVE_INFINITY })
  await Bun.sleep(20)
  controller.abort()
  await expect(pending).rejects.toThrow()
})
