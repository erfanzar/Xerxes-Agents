// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import {
  COMPACTION_REFERENCE_PREFIX,
  COMPACTION_SUMMARY_MARKER,
  ContextCompressor,
  isCompactionSummaryMessage,
  naiveSummarizer,
} from '../src/context/index.js'
import {
  compactMessagesIfNeeded,
  compactionThresholdTokens,
  lazyCompactionCompletionPort,
  precompactArchivePathFor,
} from '../src/daemon/compactionRunner.js'
import type { LlmClient } from '../src/llms/client.js'

/**
 * Minimal client for the deferred-port tests: `completeLlm` prefers `complete`
 * when present, and `closeLlmClient` calls `close` when present, so those two are
 * the whole contract exercised here.
 */
function stubCompletionClient(content: string, onClose?: () => void): LlmClient {
  return {
    complete: async () => ({ content, toolCalls: [] }),
    stream: () => {
      throw new Error('the deferred port must use complete(), not stream()')
    },
    ...(onClose ? { close: async () => onClose() } : {}),
  } as unknown as LlmClient
}

const COMPRESSOR_OPTIONS = {
  contextWindow: 100,
  protectFirst: 2,
  protectLast: 1,
  summarizer: naiveSummarizer,
  summaryMinTokens: 1,
  threshold: 0.1,
} as const

test('a compressor summary carries a typed marker, not just its prefix', () => {
  const result = new ContextCompressor(COMPRESSOR_OPTIONS).compress([
    { role: 'system', content: 'system' },
    { role: 'user', content: 'first' },
    { role: 'user', content: 'middle '.repeat(100) },
    { role: 'user', content: 'latest' },
  ])
  const summary = result.messages.find(message => message[COMPACTION_SUMMARY_MARKER] === true)
  expect(summary).toBeDefined()
  expect(String(summary?.content)).toStartWith(COMPACTION_REFERENCE_PREFIX)
  expect(isCompactionSummaryMessage(summary)).toBe(true)
})

test('an assistant turn quoting the summary marker is content, not a prior summary', () => {
  // Regression: prefix sniffing treated any message opening with the marker as
  // a prior summary, so this turn was consumed by the iterative path and its
  // message dropped from the compacted window.
  const quoted = {
    role: 'assistant',
    content: `${COMPACTION_REFERENCE_PREFIX}\n\nthe agent read an old summary back`,
  }
  expect(isCompactionSummaryMessage(quoted)).toBe(false)

  const result = new ContextCompressor(COMPRESSOR_OPTIONS).compress([
    { role: 'system', content: 'system' },
    quoted,
    { role: 'user', content: 'middle '.repeat(100) },
    { role: 'user', content: 'latest' },
  ])
  expect(result.metadata.strategy).toBe('first-pass')
  expect(JSON.stringify(result.messages)).toContain('the agent read an old summary back')
})

test('a persisted summary without the marker is still recognised by its prefix', () => {
  // Transcripts written before the marker existed — and reloads that rebuild
  // messages from a fixed set of provider fields — only carry the prefix.
  const persisted = { role: 'user', content: `${COMPACTION_REFERENCE_PREFIX}\n\nearlier state` }
  expect(isCompactionSummaryMessage(persisted)).toBe(true)

  const result = new ContextCompressor(COMPRESSOR_OPTIONS).compress([
    { role: 'system', content: 'system' },
    persisted,
    { role: 'user', content: 'middle '.repeat(100) },
    { role: 'user', content: 'latest' },
  ])
  expect(result.metadata.strategy).toBe('iterative')
})

test('the shared threshold is a fraction of the prompt budget, and 0 disables it', () => {
  expect(compactionThresholdTokens(100_000, 0.9)).toBe(90_000)
  expect(compactionThresholdTokens(100_000, 0)).toBe(0)
  expect(compactionThresholdTokens(0, 0.9)).toBe(0)
  // Out-of-range fractions clamp instead of compacting on every single turn.
  expect(compactionThresholdTokens(100_000, 4)).toBe(100_000)
})

function transcript(): Record<string, unknown>[] {
  const filler = 'word '.repeat(4_000)
  return [
    { role: 'user', content: `first request ${filler}` },
    { role: 'assistant', content: `first answer ${filler}` },
    { role: 'user', content: 'second request' },
    { role: 'assistant', content: 'second answer' },
    { role: 'user', content: 'third request' },
    { role: 'assistant', content: 'third answer' },
  ]
}

async function inTemporaryDirectory(body: (root: string) => Promise<void>): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-precompact-'))
  try {
    await body(root)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

test('a transcript under the threshold is left alone and costs no provider call', async () => {
  let calls = 0
  const outcome = await compactMessagesIfNeeded({
    completion: () => {
      calls += 1
      return 'summary'
    },
    messages: [{ role: 'user', content: 'tiny' }],
    model: 'gpt-4',
    reason: 'auto-compact',
    thresholdTokens: 90_000,
  })
  expect(outcome.compacted).toBe(false)
  expect(calls).toBe(0)
})

test('compaction archives the transcript it replaces before returning it', async () => {
  await inTemporaryDirectory(async root => {
    const archivePath = precompactArchivePathFor(root, 'a1b2c3d4e5f6')
    const messages = transcript()
    const outcome = await compactMessagesIfNeeded({
      archivePath,
      completion: () => 'THE SUMMARY',
      messages,
      model: 'gpt-4',
      reason: 'auto-compact',
      thresholdTokens: 1_000,
    })
    if (!outcome.compacted) throw new Error(`expected compaction, got ${outcome.reason}`)
    expect(JSON.stringify(outcome.messages)).toContain('THE SUMMARY')
    expect(outcome.stamp.archive_path).toBe(archivePath)
    expect(outcome.stamp.archive_error).toBeUndefined()
    expect(outcome.stamp.tokens_after).toBeLessThan(outcome.stamp.tokens_before)
    expect(outcome.stamp.messages_summarized).toBeGreaterThan(0)
    expect(outcome.stamp.reason).toBe('auto-compact')
    expect(Date.parse(outcome.stamp.compacted_at)).toBeGreaterThan(0)

    const record = JSON.parse((await readFile(archivePath, 'utf8')).trim()) as {
      readonly messages: readonly Record<string, unknown>[]
      readonly tokens_before: number
    }
    // The whole pre-compaction transcript, not a summary of it: this file is
    // the only surviving copy once the session is flushed.
    expect(record.messages).toEqual(messages)
    expect(record.tokens_before).toBe(outcome.stamp.tokens_before)
  })
})

test('a second compaction appends rather than overwriting the first archive', async () => {
  await inTemporaryDirectory(async root => {
    const archivePath = precompactArchivePathFor(root, 'a1b2c3d4e5f6')
    for (const label of ['first pass', 'second pass']) {
      const outcome = await compactMessagesIfNeeded({
        archivePath,
        completion: () => label,
        messages: [...transcript(), { role: 'user', content: label }],
        model: 'gpt-4',
        reason: 'auto-compact',
        thresholdTokens: 1_000,
      })
      expect(outcome.compacted).toBe(true)
    }
    const lines = (await readFile(archivePath, 'utf8')).trim().split('\n')
    expect(lines).toHaveLength(2)
    expect(lines[0]).toContain('first pass')
    expect(lines[1]).toContain('second pass')
  })
})

test('an unwritable archive path degrades to a stamped warning instead of blocking compaction', async () => {
  await inTemporaryDirectory(async root => {
    // A path whose parent is a file: mkdir and append both fail.
    const archivePath = join(root, 'not-a-directory.json', 'archive.jsonl')
    await Bun.write(join(root, 'not-a-directory.json'), '{}')
    const outcome = await compactMessagesIfNeeded({
      archivePath,
      completion: () => 'THE SUMMARY',
      messages: transcript(),
      model: 'gpt-4',
      reason: 'compact',
      thresholdTokens: 1_000,
    })
    if (!outcome.compacted) throw new Error(`expected compaction, got ${outcome.reason}`)
    // Refusing to compact would leave the session facing a provider overflow it
    // cannot recover from, so the failure is reported, not fatal.
    expect(outcome.stamp.archive_path).toBeUndefined()
    expect(outcome.stamp.archive_error).toBeTruthy()
  })
})

test('a deferred compaction port does not build a client until a summary is actually requested', async () => {
  // The reason this matters: compaction answers "nothing to compact" without
  // consulting a provider for a short transcript, so building the client up front
  // made that no-op require a usable provider. On a fresh install the active
  // profile is the built-in `claude-code` entry, which has no client adapter, so
  // `/compact` and `session.compress` failed outright instead of reporting a
  // clean no-op.
  let constructed = 0
  const port = lazyCompactionCompletionPort(() => {
    constructed += 1
    return stubCompletionClient('summary text')
  }, 'test-model')

  expect(constructed).toBe(0)
  await port.close()
  // Closing a port that was never used must not construct one just to release it.
  expect(constructed).toBe(0)
})

test('a deferred compaction port builds its client once and releases only what it built', async () => {
  let constructed = 0
  let closed = 0
  const port = lazyCompactionCompletionPort(() => {
    constructed += 1
    return stubCompletionClient('summary text', () => {
      closed += 1
    })
  }, 'test-model')

  expect(await port.port({ prompt: 'first', maxTokens: 32, stream: false, temperature: 0 })).toBe('summary text')
  expect(await port.port({ prompt: 'second', maxTokens: 32, stream: false, temperature: 0 })).toBe('summary text')
  // One client for the whole compaction, not one per summary budget attempt.
  expect(constructed).toBe(1)

  await port.close()
  expect(closed).toBe(1)
})

test('a deferred compaction port reports an unconstructable provider without retrying construction', async () => {
  // The retry loop walks three shrinking summary budgets. A provider that cannot
  // be constructed is not a transient condition, so it must be attempted once and
  // re-thrown thereafter rather than reconstructed on every budget.
  let attempts = 0
  const port = lazyCompactionCompletionPort(() => {
    attempts += 1
    throw new Error('claude-code requires its dedicated adapter.')
  }, 'test-model')

  for (const attempt of [1, 2, 3]) {
    void attempt
    await expect(port.port({ prompt: 'x', maxTokens: 32, stream: false, temperature: 0 }))
      .rejects.toThrow('claude-code requires its dedicated adapter.')
  }
  expect(attempts).toBe(1)
  // Nothing was built, so there is nothing to release.
  await port.close()
})
