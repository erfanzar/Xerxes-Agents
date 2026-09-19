// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { getTurnOutcome, readTurnOutcome, restoreTurnOutcome, setTurnOutcome } from '../src/types/turnOutcome.js'
import { pruneToolMessages } from '../src/context/toolResultPruner.js'
import type { ChatMessage } from '../src/types/messages.js'
import { supersedeScreenshotToolResults } from '../src/context/screenshotSuperseder.js'
import { compactMessagesIfNeeded } from '../src/daemon/compactionRunner.js'

test('validates stored status and preserves it through tool pruning without enumerable model data', () => {
  for (const value of [null, {}, {version: 2, reason: 'aborted'}, {version: 1, reason: 'secret'}, {version: 1, reason: 'completed', turn_id: 'bad id'}, {version: 1, reason: 'completed', secret: 'value'}]) expect(readTurnOutcome(value)).toBeUndefined()
  const message = {role: 'tool', content: 'line\n'.repeat(2000)}
  setTurnOutcome(message, 'aborted', 'abcd1234')
  const pruned = pruneToolMessages([message], {protectLast: 0, maxChars: 100}).messages[0]!
  expect(pruned.content.length).toBeLessThan(message.content.length)
  expect(getTurnOutcome(pruned)).toEqual({version: 1, reason: 'aborted', turn_id: 'abcd1234'})
  expect(JSON.stringify(pruned)).not.toContain('turn_outcome')
  const restored = restoreTurnOutcome({turn_outcome: getTurnOutcome(pruned)}, {role: 'assistant', content: 'answer'})
  expect(getTurnOutcome(restored)).toEqual(getTurnOutcome(pruned))
  expect(JSON.stringify(restored)).not.toContain('turn_outcome')
})

test('compaction archives outcomes and retains recent outcomes without sending metadata to the summarizer', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xr-outcome-compact-'))
  const messages = Array.from({length: 40}, (_, i) => ({ role: i % 2 ? 'assistant' : 'user', content: `Message ${i}: ` + 'historical context '.repeat(200) }))
  setTurnOutcome(messages[1]!, 'aborted', 'abcdef01')
  setTurnOutcome(messages[39]!, 'completed', 'abcdef02')
  try {
    const path = join(directory, 'archive.jsonl')
    const result = await compactMessagesIfNeeded({ messages, model: 'gpt-4o', archivePath: path, reason: 'test', completion: async request => {
      expect(request.prompt).not.toContain('turn_outcome')
      expect(request.prompt).not.toContain('abcdef01')
      return 'The user is reviewing historical context. Continue with the most recent request.'
    } })
    expect(result.compacted).toBe(true)
    if (!result.compacted) throw new Error('Compaction did not run')
    expect(getTurnOutcome(result.messages.at(-1)!)).toEqual(getTurnOutcome(messages[39]!))
    const archived = JSON.parse((await Bun.file(path).text()).trim())
    expect(archived.messages[1].turn_outcome).toEqual(getTurnOutcome(messages[1]!))
    expect(archived.messages[39].turn_outcome).toEqual(getTurnOutcome(messages[39]!))
  } finally { await rm(directory, {recursive: true, force: true}) }
})


test('replacing a superseded screenshot retains its turn outcome without model-visible metadata', () => {
  const messages: ChatMessage[] = [
    {role: 'tool', tool_call_id: 'old', content: '{"_multimodal":true,"content":"old capture"}'},
    {role: 'tool', tool_call_id: 'new', content: '{"_multimodal":true,"content":"new capture"}'},
  ]
  setTurnOutcome(messages[0]!, 'aborted', 'abcdef01')
  expect(supersedeScreenshotToolResults(messages)).toBe(1)
  expect(getTurnOutcome(messages[0]!)).toEqual({version: 1, reason: 'aborted', turn_id: 'abcdef01'})
  expect(JSON.stringify(messages)).not.toContain('turn_outcome')
})
