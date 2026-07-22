// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import {
  COMPACTION_REFERENCE_PREFIX,
  ContextCompressor,
  ProviderTokenCounter,
  RepoMapper,
  naiveSummarizer,
  pruneToolResult,
  repairToolMessageSequence,
} from '../src/context/index.js'

test('token counting serializes tool_calls and other token-bearing message fields', () => {
  const base = [{ role: 'assistant', content: 'working' }]
  const withCalls = [{
    role: 'assistant',
    content: 'working',
    tool_calls: [{ id: 'call-1', name: 'WriteFile', input: { path: 'big.ts', content: 'x'.repeat(2_000) } }],
  }]
  const baseCount = ProviderTokenCounter.countTokensForProvider(base, 'openai')
  const callCount = ProviderTokenCounter.countTokensForProvider(withCalls, 'openai')
  // The 2,000-character arguments payload alone is roughly 500 tokens at len/4.
  expect(callCount).toBeGreaterThan(baseCount + 400)

  const serialized = ProviderTokenCounter.messagesToText(withCalls)
  expect(serialized).toContain('tool_calls=')
  expect(serialized).toContain('call-1')

  const toolMessage = ProviderTokenCounter.messagesToText([
    { role: 'tool', tool_call_id: 'call-1', name: 'WriteFile', content: 'done' },
  ])
  expect(toolMessage).toContain('tool_call_id=call-1')
  expect(toolMessage).toContain('name=WriteFile')

  // Plain role/content messages keep the established wire format exactly.
  expect(ProviderTokenCounter.messagesToText([{ role: 'user', content: 'hello' }])).toBe('user: hello')
})

test('compressor returns an unchanged result when already under threshold with nothing pruned', () => {
  let summarizerCalls = 0
  const compressor = new ContextCompressor({
    contextWindow: 100_000,
    threshold: 0.75,
    protectFirst: 1,
    protectLast: 1,
    summarizer: () => {
      summarizerCalls += 1
      return 'summary'
    },
  })
  const messages = [
    { role: 'system', content: 'system' },
    { role: 'user', content: 'small question' },
    { role: 'assistant', content: 'small answer' },
    { role: 'user', content: 'latest' },
  ]
  const result = compressor.compress(messages)
  expect(result.compressed).toBe(false)
  expect(result.messages).toEqual(messages)
  expect(result.tokensAfter).toBe(result.tokensBefore)
  expect(result.compressedCount).toBe(0)
  expect(summarizerCalls).toBe(0)
})

test('pre-pruning with a zero tail does not re-append the whole body', () => {
  const text = Array.from({ length: 100 }, (_, index) => `line-${index}`).join('\n')
  const result = pruneToolResult(text, { maxChars: 80, headLines: 2, tailLines: 0 })
  expect(result.pruned).toBe(true)
  const content = String(result.content)
  expect(content).toContain('line-0')
  expect(content).toContain('line-1')
  expect(content).toContain('98 lines omitted')
  expect(content).not.toContain('line-99')
  expect(content.length).toBeLessThan(text.length)

  const zeroHead = pruneToolResult(text, { maxChars: 80, headLines: 0, tailLines: 2 })
  const zeroHeadContent = String(zeroHead.content)
  expect(zeroHeadContent).not.toContain('line-0')
  expect(zeroHeadContent).toContain('line-99')
  expect(zeroHeadContent.length).toBeLessThan(text.length)
})

test('compressor output is provider-safe without caller-side tool-pair repair', () => {
  const messages: Array<Record<string, unknown>> = [
    { role: 'system', content: 'system' },
    { role: 'assistant', content: '', tool_calls: [{ id: 'call-1', name: 'ReadFile', input: { path: 'a.ts' } }] },
    { role: 'tool', tool_call_id: 'call-1', name: 'ReadFile', content: 'middle result '.repeat(50) },
    { role: 'user', content: 'middle question '.repeat(50) },
    { role: 'tool', tool_call_id: 'orphan', name: 'GrepTool', content: 'orphan result' },
  ]
  const compressor = new ContextCompressor({
    contextWindow: 200,
    threshold: 0.1,
    protectFirst: 2,
    protectLast: 1,
    summaryMinTokens: 1,
    summarizer: naiveSummarizer,
  })
  const result = compressor.compress(messages)
  expect(result.compressed).toBe(true)

  // The head ended in an unanswered tool call: a synthetic result must follow it.
  const assistantIndex = result.messages.findIndex(message => message.role === 'assistant')
  expect(assistantIndex).toBeGreaterThanOrEqual(0)
  expect(result.messages[assistantIndex + 1]).toMatchObject({
    role: 'tool',
    tool_call_id: 'call-1',
    is_error: true,
  })

  // The tail began with an orphan tool result: it must be dropped.
  expect(result.messages.some(message => message.tool_call_id === 'orphan')).toBe(false)

  // Already repaired: running the repair pass again is a no-op.
  expect(repairToolMessageSequence(result.messages)).toEqual(result.messages)
})

test('repo mapping caps extracted symbols per file at extraction and cache time', async () => {
  await inTemporaryDirectory(async root => {
    const declarations = Array.from({ length: 40 }, (_, index) => `export function symbol${index}() {}`).join('\n')
    await writeFile(join(root, 'dense.ts'), declarations)
    const mapper = new RepoMapper({ maxSymbolsPerFile: 5, tokenBudget: 4_000 })
    const first = await mapper.build(root)
    expect(first.symbolCount).toBe(5)
    // The cached second build reuses the capped extraction instead of re-reading 40 symbols.
    const second = await mapper.build(root)
    expect(second.symbolCount).toBe(5)
  })
})

test('compressor reports protectedFirst exactly for both prior-summary positions', () => {
  const priorSummary = { role: 'user', content: `${COMPACTION_REFERENCE_PREFIX}\n\nearlier state` }
  const options = {
    contextWindow: 100,
    threshold: 0.1,
    protectLast: 1,
    summaryMinTokens: 1,
    summarizer: naiveSummarizer,
  } as const

  // Prior summary is the last protected head message: head is sliced, then credited back.
  const headLast = new ContextCompressor({ ...options, protectFirst: 2 }).compress([
    { role: 'system', content: 'system' },
    priorSummary,
    { role: 'user', content: 'middle '.repeat(100) },
    { role: 'user', content: 'latest' },
  ])
  expect(headLast.metadata.strategy).toBe('iterative')
  expect(headLast.protectedFirst).toBe(2)

  // Prior summary is the first middle message: the head was never sliced.
  const middleFirst = new ContextCompressor({ ...options, protectFirst: 1 }).compress([
    { role: 'system', content: 'system' },
    priorSummary,
    { role: 'user', content: 'middle '.repeat(100) },
    { role: 'user', content: 'latest' },
  ])
  expect(middleFirst.metadata.strategy).toBe('iterative')
  expect(middleFirst.protectedFirst).toBe(1)
})

test('token estimator weights punctuation runs and keeps the Google adjustment consistent', () => {
  const dense = '{}'.repeat(500)
  const denseCount = ProviderTokenCounter.countTokensForProvider(dense, 'openai')
  // One token per character reported ~1,000; run-weighted punctuation is far closer to BPE.
  expect(denseCount).toBeLessThanOrEqual(600)
  expect(denseCount).toBeGreaterThanOrEqual(250)

  const prose = 'the quick brown fox jumps over the lazy dog'
  const openai = ProviderTokenCounter.countTokensForProvider(prose, 'openai')
  expect(openai).toBeGreaterThan(0)
  expect(ProviderTokenCounter.countTokensForProvider(prose, 'google')).toBe(Math.ceil(openai * 1.1))
})

test('repo mapping stops scanning once maxFiles candidates are gathered', async () => {
  await inTemporaryDirectory(async root => {
    for (let index = 0; index < 30; index += 1) {
      await writeFile(join(root, `file${String(index).padStart(2, '0')}.ts`), `export function fn${index}() {}\n`)
    }
    const result = await new RepoMapper({ maxFiles: 5, tokenBudget: 4_000 }).build(root)
    expect(result.fileCount).toBe(5)
    expect(result.symbolCount).toBeLessThanOrEqual(5)
  })
})

async function inTemporaryDirectory(run: (directory: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-context-hardening-'))
  try {
    await run(directory)
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
}
