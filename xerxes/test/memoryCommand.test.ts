// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { runMemoryCommand } from '../src/runtime/memoryCommand.js'

test('memory command records, reviews, classifies, and lists governed records', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-memory-cmd-'))
  try {
    const recordResult = await runMemoryCommand({
      action: 'record',
      id: 'fact-1',
      content: 'Xerxes uses Bun.',
      source: 'cli',
      agentId: 'agent-a',
      sensitivity: 'internal',
      directory,
    })
    expect(recordResult.ok).toBeTrue()

    const reviewResult = await runMemoryCommand({ action: 'review', id: 'fact-1', content: 'approved', directory })
    expect(reviewResult.ok).toBeTrue()

    const classifyResult = await runMemoryCommand({ action: 'classify', id: 'fact-1', sensitivity: 'confidential', directory })
    expect(classifyResult.ok).toBeTrue()

    const listResult = await runMemoryCommand({ action: 'list', directory })
    expect(listResult.ok).toBeTrue()
    expect(listResult.message).toContain('fact-1')
    expect(listResult.message).toContain('approved')
    expect(listResult.message).toContain('confidential')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
