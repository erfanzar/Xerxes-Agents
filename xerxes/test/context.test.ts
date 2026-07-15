// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { pruneToolMessages, pruneToolResult, repairToolMessageSequence } from '../src/context/index.js'

test('tool-output pruning preserves non-strings and summarizes oversized text', () => {
  expect(pruneToolResult({ value: 1 })).toEqual({ content: { value: 1 }, pruned: false })
  const text = Array.from({ length: 100 }, (_, index) => `line-${index}`).join('\n')
  const result = pruneToolResult(text, { maxChars: 80, headLines: 2, tailLines: 1 })
  expect(result.pruned).toBe(true)
  expect(result.content).toContain('lines omitted')
  expect(result.content).toContain('line-99')
})

test('message pruning protects the recent tail without mutation', () => {
  const long = 'x'.repeat(400)
  const messages = [
    { role: 'tool', content: long, tool_call_id: 'one' },
    { role: 'tool', content: long, tool_call_id: 'two' },
  ]
  const result = pruneToolMessages(messages, { maxChars: 50, protectLast: 1 })
  expect(result.prunedCount).toBe(1)
  expect(result.messages[0]?.content).not.toBe(long)
  expect(result.messages[1]?.content).toBe(long)
  expect(messages[0]?.content).toBe(long)
})

test('tool pair repair drops orphan results and backfills missing calls', () => {
  const messages: Array<Record<string, unknown>> = [
    { role: 'tool', tool_call_id: 'orphan', content: 'drop' },
    { role: 'assistant', content: '', tool_calls: [{ id: 'call-1', name: 'ReadFile', input: {} }] },
    { role: 'user', content: 'continue' },
  ]
  const repaired = repairToolMessageSequence(messages)
  expect(repaired).toEqual([
    { role: 'assistant', content: '', tool_calls: [{ id: 'call-1', name: 'ReadFile', input: {} }] },
    { role: 'tool', tool_call_id: 'call-1', name: 'ReadFile', content: '[Tool result unavailable after context compaction]', is_error: true },
    { role: 'user', content: 'continue' },
  ])
})
