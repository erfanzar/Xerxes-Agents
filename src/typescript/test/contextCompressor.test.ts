// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { COMPACTION_REFERENCE_PREFIX, ContextCompressor, SmartTokenCounter, naiveSummarizer } from '../src/context/index.js'

test('provider-aware counter detects models and estimates message capacity', () => {
  const counter = new SmartTokenCounter({ model: 'gemini-2.5-pro' })
  expect(counter.provider).toBe('google')
  expect(counter.countTokens('one two three')).toBeGreaterThan(0)
  expect(counter.countRemainingCapacity('one two', 10)).toBeLessThan(10)
})

test('context compressor preserves head and tail while folding the middle into reference-only context', () => {
  const messages = [
    { role: 'system', content: 'system' },
    { role: 'user', content: 'first task' },
    { role: 'assistant', content: 'first reply' },
    { role: 'user', content: 'middle one' },
    { role: 'assistant', content: 'middle two' },
    { role: 'user', content: 'latest request' },
  ]
  const compressor = new ContextCompressor({
    contextWindow: 5,
    threshold: 0.5,
    protectFirst: 2,
    protectLast: 1,
    summarizer: naiveSummarizer,
    summaryMinTokens: 1,
  })
  const result = compressor.compress(messages)
  expect(result.compressed).toBe(true)
  expect(result.messages[0]).toEqual(messages[0])
  expect(result.messages.at(-1)).toEqual(messages.at(-1))
  expect(result.messages[2]?.content).toContain(COMPACTION_REFERENCE_PREFIX)
  expect(result.metadata.strategy).toBe('first-pass')
})

test('pre-pruning alone can satisfy a context budget without an LLM summarizer', () => {
  const messages = [
    { role: 'user', content: 'start' },
    { role: 'tool', content: 'x'.repeat(10_000) },
    { role: 'user', content: 'latest' },
  ]
  const result = new ContextCompressor({ contextWindow: 1_500, threshold: 0.8, protectFirst: 0, protectLast: 1 }).compress(messages)
  expect(result.compressed).toBe(true)
  expect(result.metadata.strategy).toBe('prune-only')
  expect(result.prunedToolResults).toBe(1)
})
