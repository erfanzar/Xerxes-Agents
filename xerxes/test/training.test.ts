// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  BatchRunner,
  ContextCompressor,
  TrajectoryCompressor,
  contentHash,
  trajectoryHash,
  type BatchRecord,
  type Trajectory,
} from '../src/index.js'

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

test('batch runner streams input lazily through the bounded worker pool', async () => {
  let yielded = 0
  let firstRunAtYielded = -1
  async function* records(): AsyncGenerator<BatchRecord> {
    for (let index = 0; index < 6; index += 1) {
      yielded += 1
      yield { id: `r${index}`, prompt: `p${index}` }
    }
  }
  const runner = new BatchRunner(async record => {
    if (firstRunAtYielded === -1) firstRunAtYielded = yielded
    await new Promise(resolve => setTimeout(resolve, 5))
    return { id: record.id, response: record.prompt.toUpperCase() }
  }, 2)
  const summary = await runner.run(records())
  expect(summary).toMatchObject({ total: 6, skipped: 0, succeeded: 6, failed: 0 })
  // The old implementation drained the whole input before any work started.
  expect(firstRunAtYielded).toBeGreaterThan(0)
  expect(firstRunAtYielded).toBeLessThan(6)
})

test('trajectory compressor appends incrementally so a late source failure keeps finished work', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-training-stream-'))
  try {
    const compressor = new TrajectoryCompressor({ workers: 1 })
    async function* trajectories(): AsyncGenerator<Trajectory> {
      for (let index = 0; index < 3; index += 1) {
        yield { id: `t${index}`, messages: [{ role: 'user', content: `hello-${index}` }] }
      }
      throw new Error('source exhausted mid-stream')
    }
    const outPath = join(directory, 'out.jsonl')
    await expect(compressor.run(trajectories(), { outPath })).rejects.toThrow('source exhausted mid-stream')
    const lines = (await readFile(outPath, 'utf8')).split(/\r?\n/).filter(Boolean)
    expect(lines).toHaveLength(3)
    for (const line of lines) {
      const parsed = JSON.parse(line) as Record<string, unknown>
      expect(typeof parsed.id).toBe('string')
    }
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
})

test('trajectory compressor redacts secrets and writes owner-only output files', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-training-redact-'))
  try {
    const compressor = new TrajectoryCompressor({
      compressor: new ContextCompressor({
        contextWindow: 4, threshold: 0.5, protectFirst: 1, protectLast: 1, summaryMinTokens: 1,
        summarizer: messages => messages.map(message => String(message.content)).join(' | '),
      }),
    })
    const secret = 'sk-ant-secretvalue1234567890'
    const outPath = join(directory, 'nested', 'corpus.jsonl')
    const run = await compressor.run([
      { id: 'leaky', messages: [
        { role: 'user', content: `use key ${secret} and mail bob@example.com` },
        { role: 'assistant', content: 'middle' },
        { role: 'user', content: 'tail' },
      ] },
    ], { outPath })
    expect(run.processed).toBe(1)

    const content = await readFile(outPath, 'utf8')
    expect(content).not.toContain(secret)
    expect(content).not.toContain('bob@example.com')
    expect(content).toContain('[redacted]')
    if (process.platform !== 'win32') {
      expect((await stat(outPath)).mode & 0o777).toBe(0o600)
    }
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
})

test('dedup hashes are full-length SHA-256 to avoid silent collision-driven data loss', async () => {
  const trajectory = { messages: [{ role: 'user', content: 'hello' }] }
  const hash = trajectoryHash(trajectory)
  expect(hash).toMatch(/^[0-9a-f]{64}$/)
  expect(trajectoryHash(trajectory)).toBe(hash)
  expect(trajectoryHash({ messages: [{ role: 'user', content: 'hello!' }] })).not.toBe(hash)

  const record: BatchRecord = { id: 'a', prompt: 'same', metadata: { b: 1, a: 2 } }
  const recordHash = contentHash(record)
  expect(recordHash).toMatch(/^[0-9a-f]{64}$/)
  // Metadata key order is normalized before hashing.
  expect(contentHash({ id: 'a', prompt: 'same', metadata: { a: 2, b: 1 } })).toBe(recordHash)
  expect(contentHash({ id: 'a', prompt: 'different' })).not.toBe(recordHash)

  const runner = new BatchRunner(record => ({ id: record.id, response: 'ok' }), 1)
  const summary = await runner.run([
    { id: 'one', prompt: 'dup' },
    { id: 'two', prompt: 'dup' },
    { id: 'three', prompt: 'unique' },
  ], { dedupBy: 'content' })
  expect(summary).toMatchObject({ total: 3, skipped: 1, succeeded: 2 })
})
