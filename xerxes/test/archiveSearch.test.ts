// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { archiveSessionSearchPort } from '../src/daemon/archiveSearch.js'
import { searchableHistory } from '../src/daemon/archiveHistory.js'
import { compactMessagesIfNeeded, compactionSummaryBudgets } from '../src/daemon/compactionRunner.js'
import { COMPACTION_ARCHIVE_NOTE, COMPACTION_REFERENCE_PREFIX } from '../src/context/compressor.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { registerSessionSearchTool } from '../src/tools/agentMetaTools.js'

type Message = { role: string; content: unknown; tool_call_id?: string; tool_calls?: unknown }

async function withArchive(records: readonly (readonly Message[])[], run: (path: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-archive-search-'))
  const path = join(directory, 'abc12345.precompact.jsonl')
  try {
    await writeFile(path, records.map(messages => JSON.stringify({ archived_at: 'x', messages, model: 'm', reason: 'auto' })).join('\n') + '\n')
    await run(path)
  } finally { await rm(directory, { recursive: true, force: true }) }
}

test('session_search finds a value that compaction summarized out of the live window', async () => {
  const archived: Message[] = [
    { role: 'user', content: 'The deploy key for orion-7 is K9-QX2-771. Remember it.' },
    { role: 'assistant', content: 'Noted.' },
  ]
  const live: Message[] = [
    { role: 'user', content: `${COMPACTION_REFERENCE_PREFIX}\n${COMPACTION_ARCHIVE_NOTE}\n\nWe set up orion-7.` },
    { role: 'user', content: 'what is next' },
  ]
  await withArchive([archived], async path => {
    const port = archiveSessionSearchPort({ liveMessages: id => id === 'abc12345' ? live as never : undefined, archivePath: () => path })
    const registry = new ToolRegistry()
    registerSessionSearchTool(registry, port)
    const result = JSON.parse(await registry.execute(
      { id: 'c1', type: 'function', function: { name: 'session_search', arguments: { query: 'orion-7 deploy key' } } },
      { metadata: {}, sessionId: 'abc12345' },
    )) as { hits: { snippet: string; in_context: boolean; role: string }[]; archived_messages: number }
    expect(result.archived_messages).toBe(2)
    expect(result.hits).toHaveLength(1)
    expect(result.hits[0]).toMatchObject({ role: 'user', in_context: false })
    expect(result.hits[0]!.snippet).toContain('K9-QX2-771')
  })
})

test('session_search searches only the calling session and every query term must match', async () => {
  await withArchive([[{ role: 'tool', tool_call_id: 't1', content: 'threshold is 0.42' }]], async path => {
    const port = archiveSessionSearchPort({ liveMessages: () => [], archivePath: () => path })
    expect(await port.search({ query: 'threshold', limit: 5, callerSessionId: 'aaaa1111', sessionId: 'bbbb2222' }))
      .toEqual({ error: 'only the current session can be searched', hits: [] })
    const both = await port.search({ query: 'threshold 0.42', limit: 5, callerSessionId: 'aaaa1111' }) as { hits: { tool_call_id?: string }[] }
    expect(both.hits).toHaveLength(1)
    expect(both.hits[0]!.tool_call_id).toBe('t1')
    const missing = await port.search({ query: 'threshold 0.99', limit: 5, callerSessionId: 'aaaa1111' }) as { count: number }
    expect(missing.count).toBe(0)
  })
  const unknown = archiveSessionSearchPort({ liveMessages: () => undefined, archivePath: () => undefined })
  expect(await unknown.search({ query: 'x', limit: 5, callerSessionId: 'aaaa1111' })).toMatchObject({ error: 'session aaaa1111 is not open' })
})

test('searchable history keeps every archived message even when the live window lacks it', () => {
  const a = { role: 'user', content: 'first' }
  const b = { role: 'assistant', content: 'second' }
  const shortened = { role: 'tool', tool_call_id: 'r', content: '[result omitted]' }
  const full = { role: 'tool', tool_call_id: 'r', content: 'the full result' }
  const history = searchableHistory([a, b, full] as never, [b, shortened, { role: 'user', content: 'third' }] as never)
  expect(history.messages.map(message => message.content)).toEqual(['first', 'second', 'the full result', 'third'])
  // The full tool output is archived; the shortened copy is what the model sees.
  expect(history.inContext).toEqual([false, true, false, true])
})

test('summary budgets scale with the prompt budget and respect the output ceiling', () => {
  expect(compactionSummaryBudgets(undefined)).toEqual([8_192, 4_096, 2_048])
  expect(compactionSummaryBudgets(128_000)).toEqual([8_192, 4_096, 2_048])
  expect(compactionSummaryBudgets(1_000_000)).toEqual([20_000, 10_000, 5_000])
  expect(compactionSummaryBudgets(10_000_000)[0]).toBe(32_768)
  expect(compactionSummaryBudgets(1_000_000, { maxOutputTokens: 8_000 })[0]).toBe(7_200)
  expect(compactionSummaryBudgets(undefined, { midTurn: true })).toEqual([2_048])
  expect(compactionSummaryBudgets(1_000_000, { midTurn: true })).toEqual([8_000])
  expect(compactionSummaryBudgets(128_000, { midTurn: true })).toEqual([2_048])
})

test('the summary points at session_search only when the replaced history was archived', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-archive-note-'))
  try {
    const messages: Record<string, unknown>[] = []
    for (let index = 0; index < 30; index++) messages.push({ role: index % 2 ? 'assistant' : 'user', content: `turn ${index} `.repeat(300) })
    const completion = async () => ({ content: 'Short summary of the earlier work.' })
    const noted = (outcome: Awaited<ReturnType<typeof compactMessagesIfNeeded>>) => {
      if (!outcome.compacted) throw new Error(`not compacted: ${outcome.error ?? outcome.reason}`)
      return outcome.messages.some(message => String(message.content).includes(COMPACTION_ARCHIVE_NOTE))
    }
    const archived = await compactMessagesIfNeeded({ model: 'gpt-4o', messages, reason: 'compact', archivePath: join(directory, 'abc12345.precompact.jsonl'), maxContextTokens: 20_000, completion })
    expect(noted(archived)).toBe(true)
    const unarchived = await compactMessagesIfNeeded({ model: 'gpt-4o', messages, reason: 'compact', maxContextTokens: 20_000, completion })
    expect(noted(unarchived)).toBe(false)
  } finally { await rm(directory, { recursive: true, force: true }) }
})

function bulkyHistory(): Record<string, unknown>[] {
  const messages: Record<string, unknown>[] = [{ role: 'user', content: 'start the long task' }]
  for (let index = 0; index < 20; index++) {
    messages.push({ role: 'assistant', content: '', tool_calls: [{ id: `c${index}`, type: 'function', function: { name: 'ReadFile', arguments: '{}' } }] })
    messages.push({ role: 'tool', tool_call_id: `c${index}`, content: `output ${index} `.repeat(800) })
  }
  messages.push({ role: 'user', content: 'keep going' })
  return messages
}

test('when every summary fails, the automatic fallback archives first and sheds only older tool outputs', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-shed-fallback-'))
  const archivePath = join(directory, 'abc12345.precompact.jsonl')
  try {
    const messages = bulkyHistory()
    const failing = async () => { throw new Error('provider unavailable') }
    // Without the opt-in the failure is reported unchanged.
    const reported = await compactMessagesIfNeeded({ model: 'gpt-4o', messages, reason: 'compact', archivePath, maxContextTokens: 20_000, completion: failing })
    expect(reported).toMatchObject({ compacted: false, reason: 'failed' })

    const outcome = await compactMessagesIfNeeded({ model: 'gpt-4o', messages, reason: 'auto-compact', archivePath, maxContextTokens: 20_000, completion: failing, deterministicFallback: true })
    if (!outcome.compacted) throw new Error(`fallback did not run: ${outcome.error}`)
    expect(outcome.stamp).toMatchObject({ fallback: 'shed-tool-results', summary_error: 'provider unavailable', messages_summarized: 0 })
    expect(outcome.stamp.tokens_after).toBeLessThan(outcome.stamp.tokens_before)
    // Same message count and order; user and assistant messages untouched.
    expect(outcome.messages).toHaveLength(messages.length)
    expect(outcome.messages.filter(message => message.role !== 'tool')).toEqual(messages.filter(message => message.role !== 'tool'))
    // The newest results are kept verbatim; the oldest were shed.
    const tools = outcome.messages.filter(message => message.role === 'tool')
    expect(tools.at(-1)!.content).toBe(messages.at(-2)!.content)
    expect(String(tools[0]!.content)).toStartWith('[result omitted to fit the context window')
    // The full original is archived before the swap.
    const archive = (await Bun.file(archivePath).text()).trim().split('\n').map(line => JSON.parse(line) as { reason: string; messages: unknown[] })
    expect(archive.at(-1)!.reason).toBe('auto-compact:shed-fallback')
    expect(archive.at(-1)!.messages).toEqual(messages)
  } finally { await rm(directory, { recursive: true, force: true }) }
})
