// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { test, expect } from 'bun:test'
import { mkdtemp, rm, appendFile, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ArchiveHistory, appendHistoryWindow } from '../src/daemon/archiveHistory.js'
import { sessionHistoryPage } from '../src/daemon/historyPage.js'
import type { DaemonSession } from '../src/daemon/runtime.js'
type Message = DaemonSession['messages'][number]
const user = (content: string): Message => ({ role: 'user', content })
const answer = (content: string): Message => ({ role: 'assistant', content })
const summary = (): Message => ({ ...user('summary'), xerxes_compaction_summary: true })

test('archive stitching restores repeated compactions without losing repeated prompts or full tool output', () => {
  const head = user('start'), repeated = user('continue')
  const tool: Message = { role: 'tool', tool_call_id: 'one', content: 'original full output' }
  const first = [head, answer('one'), repeated, answer('two'), tool]
  const second = [head, summary(), { ...tool, content: 'pruned' }, repeated, answer('three')]
  const third = [head, summary(), answer('three'), repeated, answer('four')]
  const restored = appendHistoryWindow(appendHistoryWindow(first, second), third)
  expect(restored).toEqual([...first, repeated, answer('three'), repeated, answer('four')])
  const session = { id: 'abc', messages: restored, toolExecutions: [], thinkingContent: [] }
  const last = sessionHistoryPage(session, 2)
  expect(last.has_more).toBe(true)
  const before = sessionHistoryPage(session, 100, last.before)
  expect([...before.actions, ...last.actions].filter(a => a.messages[0]?.content === 'continue')).toHaveLength(3)
  expect(third).toHaveLength(5)
})

test('archive cache notices appends, preserves current context, and rejects corrupt or redirected history', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-history-'))
  try {
    const path = join(directory, 'a.precompact.jsonl'), reader = new ArchiveHistory()
    const first = [user('start'), answer('old'), user('continue'), answer('retained')]
    await Bun.write(path, JSON.stringify({ messages: first }) + '\n')
    const current = [user('start'), summary(), answer('retained'), user('new'), answer('new answer')]
    expect(await reader.messages(path, current)).toEqual([...first, user('new'), answer('new answer')])
    expect(current).toHaveLength(5)
    expect(await reader.messages(path, [user('start'), answer('old')])).toEqual([user('start'), answer('old')])
    expect(await reader.messages(path, [user('start'), summary(), user('continue')])).toEqual(first.slice(0, 3))
    await appendFile(path, JSON.stringify({ messages: current }) + '\n')
    expect(await reader.messages(path, [user('start'), summary(), answer('new answer'), user('last')])).toEqual([...first, user('new'), answer('new answer'), user('last')])
    expect(await reader.messages(join(directory, 'missing'), current)).toEqual(current)
    await symlink(path, join(directory, 'redirect'))
    await expect(reader.messages(join(directory, 'redirect'), current)).rejects.toThrow('regular file')
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('a torn or unreadable archive record is skipped and reported instead of making the history unloadable', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-history-torn-'))
  try {
    const path = join(directory, 'a.precompact.jsonl'), warnings: string[] = []
    const reader = new ArchiveHistory(warning => { warnings.push(warning) })
    const first = [user('start'), answer('old'), user('continue'), answer('retained')]
    const second = [summary(), answer('retained'), user('more'), answer('more answer')]
    const record = (messages: Message[]) => JSON.stringify({ archived_at: '2026-10-01T00:00:00.000Z', messages })
    // A full disk tore the second append; the next compaction then appended
    // its record directly after the torn bytes, on the same line.
    const torn = record(second).slice(0, 40)
    await Bun.write(path, `${record(first)}\n${torn}${record(second)}\n{bad}\n`)
    const current = [summary(), answer('more answer'), user('last')]
    expect(await reader.messages(path, current)).toEqual([...first, user('more'), answer('more answer'), user('last')])
    expect(warnings).toEqual([expect.stringContaining('skipped 1 unreadable record')])
  } finally { await rm(directory, { recursive: true, force: true }) }
})

test('an undo inside the retained tail stays undone when a new turn follows it', () => {
  const [q1, r1, k1, k2, k3, k4] = [user('q1'), answer('r1'), user('k1'), answer('k2'), user('k3'), answer('k4')]
  const archived = [q1, r1, k1, k2, k3, k4]
  const [n1, n2] = [user('n1'), answer('n2')]
  // The compaction recorded its four-message tail [k1..k4]; undo dropped k3/k4.
  expect(appendHistoryWindow(archived, [summary(), k1, k2, n1, n2], true, 4)).toEqual([q1, r1, k1, k2, n1, n2])
  expect(appendHistoryWindow(archived, [summary(), k1, k2], true, 4)).toEqual([q1, r1, k1, k2])
  // Undo of the whole tail, with and without a new turn afterwards.
  expect(appendHistoryWindow(archived, [summary()], true, 4)).toEqual([q1, r1])
  expect(appendHistoryWindow(archived, [summary(), n1, n2], true, 4)).toEqual([q1, r1, n1, n2])
  // The next compaction folds the same window as an archive record.
  expect(appendHistoryWindow(archived, [summary(), k1, k2, n1, n2], false, 4)).toEqual([q1, r1, k1, k2, n1, n2])
  // An untouched tail plus new messages, and a zero-retention compaction.
  expect(appendHistoryWindow(archived, [summary(), k1, k2, k3, k4, n1], true, 4)).toEqual([...archived, n1])
  expect(appendHistoryWindow(archived, [summary(), n1, n2], true, 0)).toEqual([...archived, n1, n2])
  // Archives written before the tail length was recorded fall back to the
  // turn boundary where the window diverges.
  expect(appendHistoryWindow(archived, [summary(), k1, k2, n1, n2], true)).toEqual([q1, r1, k1, k2, n1, n2])
})

test('the archive reader aligns windows on each record\'s recorded tail', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-history-retained-'))
  try {
    const path = join(directory, 'a.precompact.jsonl'), reader = new ArchiveHistory()
    const [q1, r1, k1, k2, k3, k4, n1, n2] = [user('q1'), answer('r1'), user('k1'), answer('k2'), user('k3'), answer('k4'), user('n1'), answer('n2')]
    await Bun.write(path, [
      JSON.stringify({ archived_at: 'a', messages: [q1, r1, k1, k2, k3, k4], retained_messages: 4 }),
      // Compacted again after an undo of k3/k4 and a new turn.
      JSON.stringify({ archived_at: 'b', messages: [summary(), k1, k2, n1, n2], retained_messages: 1 }),
    ].join('\n') + '\n')
    expect(await reader.messages(path, [summary(), n2, user('n3')])).toEqual([q1, r1, k1, k2, n1, n2, user('n3')])
  } finally { await rm(directory, { recursive: true, force: true }) }
})
