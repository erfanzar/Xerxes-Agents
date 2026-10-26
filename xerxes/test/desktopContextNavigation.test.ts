// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { contextScope, contextSessions, sameRemote } from '../src/desktop/main/contextNavigation.js'

const machine = { alias: 'worker', target: 'user@server', workspacePath: '/repo' }
test('endpoint identity separates equal local and SSH paths, while aliases share the same host', () => {
  expect(contextScope(null)).not.toBe(contextScope(machine))
  expect(contextScope({ ...machine, alias: 'renamed' })).toBe(contextScope(machine))
  expect(sameRemote(null, machine)).toBe(false)
  expect(sameRemote(machine, { ...machine, workspacePath: '/other' })).toBe(false)
  expect(sameRemote(machine, { ...machine, alias: 'renamed' })).toBe(true)
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
