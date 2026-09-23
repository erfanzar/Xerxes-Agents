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
    await appendFile(path, '{bad}\n')
    await expect(reader.messages(path, current)).rejects.toThrow('unreadable record')
  } finally { await rm(directory, { recursive: true, force: true }) }
})
