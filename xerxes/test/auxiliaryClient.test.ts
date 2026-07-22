// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  AuxiliaryClient,
  DEFAULT_AUXILIARY_MAX_TOKENS,
  MAX_RENDERED_TRANSCRIPT_CHARS,
  type AuxiliaryBackendRequest,
  type AuxiliaryMessage,
} from '../src/runtime/index.js'

test('auxiliary client delivers a resolved typed request to a synchronous injected backend', async () => {
  let captured: AuxiliaryBackendRequest | undefined
  const timestamps = [100, 117]
  const client = new AuxiliaryClient({
    backend: request => {
      captured = request
      return 'summary text'
    },
    model: 'test-aux-model',
    monotonicNow: () => timestamps.shift() ?? 117,
  })

  const response = await client.call({
    purpose: 'summary',
    messages: [{ role: 'user', content: 'hi' }],
  })

  expect(client.model).toBe('test-aux-model')
  expect(response).toEqual({
    text: 'summary text',
    purpose: 'summary',
    model: 'test-aux-model',
    durationMs: 17,
    requestTokens: 0,
    responseTokens: 0,
  })
  expect(captured).toEqual({
    purpose: 'summary',
    messages: [{ role: 'user', content: 'hi' }],
    maxTokens: DEFAULT_AUXILIARY_MAX_TOKENS,
    temperature: 0,
    metadata: {},
    model: 'test-aux-model',
  })
})

test('auxiliary client supports asynchronous backends and preserves their usage accounting', async () => {
  const client = new AuxiliaryClient({
    backend: async request => ({
      text: `completed ${request.purpose}`,
      requestTokens: 12,
      responseTokens: 7,
    }),
  })

  await expect(client.call({
    purpose: 'extract',
    messages: [{ role: 'user', content: 'body' }],
    maxTokens: 80,
  })).resolves.toMatchObject({
    text: 'completed extract',
    requestTokens: 12,
    responseTokens: 7,
  })
})

test('summarize renders messages under a compaction instruction and honors its budget', async () => {
  let captured: AuxiliaryBackendRequest | undefined
  const client = new AuxiliaryClient({
    backend: request => {
      captured = request
      return 'S'
    },
  })

  await expect(client.summarize([
    { role: 'user', content: 'hi' },
    { role: 'assistant', content: 'there' },
  ], { budgetTokens: 42 })).resolves.toBe('S')

  expect(captured?.purpose).toBe('summarize')
  expect(captured?.maxTokens).toBe(42)
  expect(captured?.messages[0]).toMatchObject({ role: 'system', content: expect.stringContaining('Summarize') })
  expect(captured?.messages[1]).toMatchObject({
    role: 'user',
    content: '[user] hi\n[assistant] there',
  })
})

test('summarize preserves non-string content and rejects malformed message roles', async () => {
  let captured: AuxiliaryBackendRequest | undefined
  const client = new AuxiliaryClient({
    backend: request => {
      captured = request
      return 'S'
    },
  })

  await client.summarize([{ role: 'user', content: null }])

  expect(captured?.messages[1]).toMatchObject({ role: 'user', content: '[user] null' })
  const malformedMessage = { role: 7 } as unknown as AuxiliaryMessage
  await expect(client.summarize([malformedMessage])).rejects.toThrow('role must be a string')
})

test('title strips display delimiters and extract preserves the caller instruction', async () => {
  const requests: AuxiliaryBackendRequest[] = []
  const client = new AuxiliaryClient({
    backend: request => {
      requests.push(request)
      return request.purpose === 'title' ? '  "My Cool Title"  ' : 'extracted'
    },
  })

  await expect(client.title([{ role: 'user', content: 'hi' }])).resolves.toBe('My Cool Title')
  await expect(client.extract('body text', { instruction: 'Extract emails' })).resolves.toBe('extracted')

  expect(requests[0]).toMatchObject({ purpose: 'title', maxTokens: 64 })
  expect(requests[1]).toMatchObject({
    purpose: 'extract',
    messages: [
      { role: 'system', content: 'Extract emails' },
      { role: 'user', content: 'body text' },
    ],
  })
})

test('summarize caps oversized transcripts with an explicit head+tail truncation marker', async () => {
  let captured: AuxiliaryBackendRequest | undefined
  const client = new AuxiliaryClient({
    backend: request => {
      captured = request
      return 'S'
    },
  })

  const head = { role: 'user', content: 'H'.repeat(10_000) }
  const middle = Array.from({ length: 200 }, (_, index) => ({
    role: 'assistant',
    content: `turn ${index} ` + 'x'.repeat(2_000),
  }))
  const tail = { role: 'user', content: 'T'.repeat(10_000) }
  await client.summarize([head, ...middle, tail])

  const rendered = captured?.messages[1]?.content
  expect(typeof rendered).toBe('string')
  const text = rendered as string
  expect(text.length).toBeLessThanOrEqual(MAX_RENDERED_TRANSCRIPT_CHARS)
  expect(text.length).toBeGreaterThan(MAX_RENDERED_TRANSCRIPT_CHARS / 2)
  expect(text).toContain('transcript truncated')
  expect(text.startsWith('[user] ' + 'H'.repeat(100))).toBe(true)
  expect(text.endsWith('T'.repeat(100))).toBe(true)
  // The middle turns are what get omitted.
  expect(text).not.toContain('turn 100')
})

test('short transcripts pass through to the auxiliary model unclipped', async () => {
  let captured: AuxiliaryBackendRequest | undefined
  const client = new AuxiliaryClient({
    backend: request => {
      captured = request
      return 'S'
    },
  })

  await client.summarize([{ role: 'user', content: 'small' }])

  expect(captured?.messages[1]).toMatchObject({ role: 'user', content: '[user] small' })
})

test('backend failures propagate without a fallback response', async () => {
  const client = new AuxiliaryClient({
    backend: () => {
      throw new Error('backend unavailable')
    },
  })

  await expect(client.summarize([{ role: 'user', content: 'x' }])).rejects.toThrow('backend unavailable')
})
