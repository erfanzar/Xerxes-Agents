// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { contextKey, contextLabel, contextScope, contextSessions, latestSessionIn, sameRemote, sessionsFor } from '../src/desktop/main/contextNavigation.js'

const machine = { alias: 'worker', target: 'user@server', workspacePath: '/repo' }
test('endpoint identity separates equal local and SSH paths, while aliases share the same host', () => {
  expect(contextScope(null)).not.toBe(contextScope(machine))
  expect(contextScope({ ...machine, alias: 'renamed' })).toBe(contextScope(machine))
  expect(sameRemote(null, machine)).toBe(false)
  expect(sameRemote(machine, { ...machine, workspacePath: '/other' })).toBe(false)
  expect(sameRemote(machine, { ...machine, alias: 'renamed' })).toBe(true)
})
test('context keys and labels tell two folders on one host apart', () => {
  expect(contextKey(machine)).not.toBe(contextKey({ ...machine, workspacePath: '/other' }))
  expect(contextKey(machine)).toBe(contextKey({ ...machine, alias: 'renamed', workspacePath: '/repo/' }))
  expect(contextKey(null)).toBe('local')
  expect(contextLabel('v5p-128-dev', '/home/erfan/prismcore/')).toBe('v5p-128-dev · prismcore')
  expect(contextLabel('host', '/')).toBe('host · SSH')
})
test('sessions split between folders of one host by deepest containing folder; one folder filters nothing', () => {
  const rows = [
    { id: 'a', cwd: '/repo/sub', title: 'A', status: 'idle' },
    { id: 'b', cwd: '/repo-two', title: 'B', status: 'idle' },
    { id: 'c', cwd: '/repo/nested/x', title: 'C', status: 'idle' },
  ]
  expect(sessionsFor(rows, '/repo', ['/repo']).map(row => row.id)).toEqual(['a', 'b', 'c'])
  expect(sessionsFor(rows, '/repo', ['/repo', '/repo-two']).map(row => row.id)).toEqual(['a', 'c'])
  expect(sessionsFor(rows, '/repo-two', ['/repo', '/repo-two']).map(row => row.id)).toEqual(['b'])
  expect(sessionsFor(rows, '/repo/nested', ['/repo', '/repo/nested']).map(row => row.id)).toEqual(['c'])
  expect(latestSessionIn(rows, '/repo')).toBe('a')
  expect(latestSessionIn(rows, '/nowhere')).toBeUndefined()
})
test('navigation merges saved titles and live working state without leaking subagents', () => {
  expect(contextSessions([{ id: 'task', cwd: '/repo', title: 'Saved title', status: 'idle' }], [
    { session_id: 'task', cwd: '/repo', active_turn_id: 'turn' },
    { id: 'child', cwd: '/repo', kind: 'subagent' },
    { id: 'failure', cwd: '/other', title: 'Failed task', status: 'failed' },
  ])).toEqual([
    { id: 'task', cwd: '/repo', title: 'Saved title', status: 'working' },
    { id: 'failure', cwd: '/other', title: 'Failed task', status: 'failed' },
  ])
})
test('malformed and empty daemon session lists are safe to display', () => {
  expect(contextSessions(null, { sessions: [] })).toEqual([])
  expect(contextSessions([null, 42, {}, { id: '', cwd: '/repo' }, { id: 'no-path' }], [])).toEqual([])
})
test('context navigation bounds session identities, absolute paths, titles and row counts at the boundary', () => {
  expect(contextSessions([
    { id: 'bad\nid', cwd: '/repo' }, { id: 'x'.repeat(257), cwd: '/repo' },
    { id: 'relative', cwd: 'repo' }, { id: 'control', cwd: '/bad\npath' },
    { id: 'safe', cwd: '/repo', title: 'Train\u0000\n safely' },
    { id: 'long', cwd: '/repo', title: 'x'.repeat(1000) },
  ], [])).toEqual([
    { id: 'safe', cwd: '/repo', title: 'Train   safely', status: 'idle' },
    { id: 'long', cwd: '/repo', title: 'x'.repeat(512), status: 'idle' },
  ])
  expect(contextSessions(Array.from({ length: 1000 }, (_, i) => ({ id: `chat-${i}`, cwd: '/repo' })), [])).toHaveLength(200)
})

test('contextSessions preserves runtime provider, turn start timestamp, activity age, and branch across saved and live rows', () => {
  const startedSec = 1_795_650_000
  const updatedSec = 1_795_650_300
  const merged = contextSessions(
    [
      {
        id: 'agy-1',
        cwd: '/home/erfan/easydel-src',
        title: 'Update EasyDeL TPU Slice Serving',
        status: 'idle',
        acp_provider: 'antigravity',
        updated_at: updatedSec,
        branch: 'main',
      },
      {
        id: 'claude-2',
        cwd: '/home/erfan/easydel-src',
        title: 'Audit training loop',
        status: 'idle',
        acp_agent_id: 'claude-code',
        updated_at: updatedSec - 120,
      },
    ],
    [
      {
        session_id: 'agy-1',
        cwd: '/home/erfan/easydel-src',
        active_turn_id: 'turn-7',
        inflight: { started_at: startedSec },
      },
    ],
  )
  expect(merged).toEqual([
    {
      id: 'agy-1',
      cwd: '/home/erfan/easydel-src',
      title: 'Update EasyDeL TPU Slice Serving',
      status: 'working',
      provider: 'antigravity',
      turnStartedAt: startedSec * 1000,
      activeAt: updatedSec * 1000,
      branch: 'main',
    },
    {
      id: 'claude-2',
      cwd: '/home/erfan/easydel-src',
      title: 'Audit training loop',
      status: 'idle',
      provider: 'claude-code',
      activeAt: (updatedSec - 120) * 1000,
    },
  ])
})
