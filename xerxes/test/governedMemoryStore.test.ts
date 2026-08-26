// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { GovernedMemoryStore } from '../src/memory/governedMemoryStore.js'

test('governed memory records provenance and supports review, classification, correction, and expiration', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-gov-mem-'))
  let now = 1_000
  try {
    const store = new GovernedMemoryStore({ directory, now: () => now })
    const record = await store.record({
      id: 'fact-1',
      content: 'Xerxes uses Bun.',
      source: 'conversation',
      agentId: 'agent-a',
      turnId: 'turn-1',
      sensitivity: 'internal',
    })
    expect(record.reviewStatus).toBe('pending')
    expect(record.sensitivity).toBe('internal')

    await store.review('fact-1', 'approved')
    let state = await store.load()
    expect(state.records.get('fact-1')?.reviewStatus).toBe('approved')

    await store.classify('fact-1', 'confidential')
    state = await store.load()
    expect(state.records.get('fact-1')?.sensitivity).toBe('confidential')

    const corrected = await store.correct({ originalId: 'fact-1', newId: 'fact-1-corrected', content: 'Xerxes uses Bun and TypeScript.', reason: 'add TypeScript' })
    expect(corrected.content).toBe('Xerxes uses Bun and TypeScript.')

    await store.expire('fact-1')
    state = await store.load()
    expect(state.records.has('fact-1')).toBeFalse()
    expect(state.records.has('fact-1-corrected')).toBeTrue()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
