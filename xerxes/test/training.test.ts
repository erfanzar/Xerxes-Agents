// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { BatchRunner, ContextCompressor, TrajectoryCompressor } from '../src/index.js'

test('batch runner resumes deduplicated records, captures errors, and writes JSONL', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-training-'))
  try {
    const runner = new BatchRunner(async record => {
      if (record.id === 'bad') throw new Error('failed')
      return { id: record.id, response: record.prompt.toUpperCase(), inputTokens: 2, outputTokens: 3, costUsd: 0.01 }
    }, 2)
    const output = join(directory, 'results.jsonl')
    const summary = await runner.run([
      { id: 'done', prompt: 'skip' }, { id: 'good', prompt: 'go' }, { id: 'bad', prompt: 'no' },
    ], { outPath: output, resumeIds: new Set(['done']) })
    expect(summary).toMatchObject({ total: 3, skipped: 1, succeeded: 1, failed: 1, totalInputTokens: 2 })
    expect((await readFile(output, 'utf8')).split(/\r?\n/).filter(Boolean)).toHaveLength(2)
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
})

test('trajectory compressor emits compression metrics and respects resume ids', async () => {
  const compressor = new TrajectoryCompressor({
    compressor: new ContextCompressor({
      contextWindow: 4, threshold: 0.5, protectFirst: 1, protectLast: 1, summaryMinTokens: 1,
      summarizer: messages => messages.map(message => String(message.content)).join(' | '),
    }),
  })
  const run = await compressor.run([
    { id: 'one', messages: [{ role: 'user', content: 'head' }, { role: 'assistant', content: 'middle' }, { role: 'user', content: 'tail' }] },
    { id: 'skip', messages: [] },
  ], { alreadyDone: new Set(['skip']) })
  expect(run.processed).toBe(1)
  expect(run.skipped).toBe(1)
  expect(run.metrics[0]?.strategy).toBe('first-pass')
})
