// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { compactMessagesIfNeeded } from '../src/daemon/compactionRunner.js'
import { shedToolResults } from '../src/context/toolResultPruner.js'
import { estimateContextTokens } from '../src/context/windowUsage.js'

/**
 * One assistant message carrying hundreds of calls, then their results and a
 * few short user messages: the shape of a Claude Code task that read one file
 * 1,170 times. The round cannot be split from its calls, so summarizing found
 * "nothing to compact" while the conversation was six times the window.
 */
function giantRound(calls: number, resultChars: number): Array<Record<string, unknown>> {
  const ids = Array.from({ length: calls }, (_, index) => `call_${index}`)
  return [
    { role: 'user', content: 'Review the source and update the README.' },
    { role: 'assistant', content: '', tool_calls: ids.map((id, index) => ({ id, type: 'function', function: { name: 'ReadFile', arguments: JSON.stringify({ path: 'core.py', offset: index }) } })) },
    ...ids.map((id, index) => ({ role: 'tool', tool_call_id: id, content: `line ${index}\n` + 'x'.repeat(resultChars) })),
    { role: 'user', content: 'Continue' },
    { role: 'user', content: 'continue' },
  ]
}

test('a round larger than the window is compacted by shedding its oldest results, without a summary call', async () => {
  const messages = giantRound(400, 6_000)
  const before = estimateContextTokens(messages, { model: 'claude-code/opus' })
  let summaryCalls = 0
  const outcome = await compactMessagesIfNeeded({
    model: 'claude-code/opus', messages, reason: 'test', maxContextTokens: 64_000,
    completion: async () => { summaryCalls += 1; return { content: 'summary' } },
  })
  expect(before).toBeGreaterThan(400_000)
  expect(outcome.compacted).toBe(true)
  if (!outcome.compacted) return
  const after = estimateContextTokens(outcome.messages, { model: 'claude-code/opus' })
  expect(after).toBeLessThanOrEqual(64_000)
  expect(summaryCalls).toBe(0)
  // Every call still has its result, in order, and user messages are untouched.
  const results = outcome.messages.filter(message => message.role === 'tool')
  expect(results.map(message => message.tool_call_id)).toEqual(Array.from({ length: 400 }, (_, index) => `call_${index}`))
  expect(outcome.messages.filter(message => message.role === 'user').map(message => message.content)).toEqual(['Review the source and update the README.', 'Continue', 'continue'])
  // The newest results stay verbatim; the oldest say what was dropped.
  expect(String(results.at(-1)!.content)).toStartWith('line 399')
  expect(String(results[0]!.content)).toMatch(/^\[result omitted to fit the context window \([\d,]+ characters\)\]$/)
})

test('shedding stops once the conversation fits and never touches the newest results', () => {
  const messages = giantRound(20, 4_000)
  const count = (list: readonly Record<string, unknown>[]) => list.reduce((total, message) => total + JSON.stringify(message).length, 0)
  const budget = count(messages) - 4 * 4_000 + 1_000
  const shed = shedToolResults(messages, { budgetTokens: budget, keepRecent: 8, count })
  expect(shed.shedCount).toBe(4)
  expect(count(shed.messages)).toBeLessThanOrEqual(budget)
  const contents = shed.messages.filter(message => message.role === 'tool').map(message => String(message.content))
  expect(contents.slice(0, 4).every(text => text.startsWith('[result omitted'))).toBe(true)
  expect(contents.slice(4).every(text => text.startsWith('line '))).toBe(true)
  // Nothing to shed when it already fits.
  expect(shedToolResults(messages, { budgetTokens: Number.MAX_SAFE_INTEGER, keepRecent: 8, count }).shedCount).toBe(0)
})
