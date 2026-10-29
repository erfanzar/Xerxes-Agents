// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { SavedHostContexts, savedHostResult } from '../src/desktop/main/savedHosts.js'
import { Sidebar } from '../src/desktop/renderer/App.js'
import { Store } from '../src/desktop/renderer/store.js'

const machine = { alias: 'v5p-128', target: 'user@worker', workspacePath: '/srv/project' }
const attach = 'XERXES_REMOTE_ATTACH ' + JSON.stringify({ socketPath: '/remote/global.sock', projectDir: machine.workspacePath, version: '0.8.17', buildId: 'a'.repeat(16) })
const output = attach + '\nXERXES_REMOTE_SESSIONS ' + JSON.stringify({
  saved: [{ id: 'saved-chat', cwd: '/srv/other-project', title: 'Train v5p-128', status: 'idle' }],
  live: [{ session_id: 'saved-chat', cwd: '/srv/other-project', title: '', active_turn_id: 'turn' }, { id: 'child', cwd: '/srv/project', kind: 'subagent' }],
})
const flush = () => new Promise(resolve => setTimeout(resolve, 0))

test('saved-host parser merges saved titles/live status and rejects missing or malformed session lists', () => {
  expect(savedHostResult(output)).toEqual({ availability: 'online', sessions: [{ id: 'saved-chat', cwd: '/srv/other-project', title: 'Train v5p-128', status: 'working' }] })
  expect(savedHostResult('XERXES_REMOTE_MISSING')).toEqual({ availability: 'no-runtime', reason: 'No running Xerxes runtime', sessions: [] })
  for (const value of [attach, attach + '\nXERXES_REMOTE_SESSIONS null', attach + '\nXERXES_REMOTE_SESSIONS {"saved":[],"live":{}}']) expect(() => savedHostResult(value)).toThrow()
})
test('saved hosts refresh in background, cache instantly, coalesce per-host probes, and obey 60-second refresh bounds', async () => {
  let now = 0, probes = 0, lists = 0
  const replies: Array<(value: string) => void> = []
  const hosts = new SavedHostContexts({
    now: () => now,
    list: async () => { lists++; return { ok: true, machines: [machine] } },
    run: async (_binary, args, _signal, timeout) => {
      probes++
      expect(timeout).toBe(20000)
      expect(args.join(' ')).not.toContain('XERXES_REMOTE_READY')
      return new Promise(resolve => replies.push(resolve))
    },
  })
  try {
    await hosts.refresh()
    const cached = hosts.contexts()
    expect(cached[0]).toMatchObject({ alias: 'v5p-128', label: 'v5p-128 · SSH', workspacePath: machine.workspacePath, availability: 'offline', saved: true, sessions: [] })
    expect(probes).toBe(1)
    now = 60000
    await Promise.all([hosts.refresh(), hosts.refresh()])
    expect(probes).toBe(1) // Even stale hosts cannot overlap an in-flight invocation.
    replies.shift()!(output); await flush()
    expect(hosts.contexts()[0]?.sessions[0]?.title).toBe('Train v5p-128')
    expect(probes).toBe(2) // The second attempt begins only after the first settles.
    replies.shift()!(output); await flush()
    now = 119999
    await hosts.refresh()
    expect(probes).toBe(2)
    now = 120000
    await hosts.refresh()
    expect(probes).toBe(3)
    replies.shift()!(output); await flush()
    expect(lists).toBe(3)
  } finally { hosts.dispose() }
})
test('saved host contexts dedupe target aliases and open views regardless of project directory; chat clicks use their own cwd', async () => {
  let probes = 0
  const hosts = new SavedHostContexts({
    list: async () => ({ ok: true, machines: [machine, { ...machine, alias: 'another-alias', workspacePath: '/elsewhere' }] }),
    run: async () => { probes++; return output },
  })
  try {
    await hosts.refresh([{ ...machine, workspacePath: '/already-open' }])
    expect(hosts.contexts([{ ...machine, workspacePath: '/already-open' }])).toEqual([])
    expect(probes).toBe(0)
    hosts.contexts(); await flush()
    const rows = hosts.contexts()
    expect(rows).toHaveLength(1)
    const id = rows[0]!.id
    expect(id).toBeLessThan(0)
    expect(hosts.selection(id)).toEqual({ machine })
    expect(hosts.selection(id, 'saved-chat')).toEqual({ machine: { ...machine, workspacePath: '/srv/other-project' }, sessionId: 'saved-chat' })
    expect(() => hosts.selection(id, 'missing-chat')).toThrow()
    expect(() => hosts.selection(id, 'bad\nchat')).toThrow()
    expect(() => hosts.selection(12345)).toThrow()
  } finally { hosts.dispose() }
})
test('a failed host becomes offline without blocking a healthy host or leaking stderr; missing runtime is distinct', async () => {
  const hosts = new SavedHostContexts({
    list: async () => ({ ok: true, machines: [machine, { ...machine, alias: 'missing', target: 'missing' }, { ...machine, alias: 'healthy', target: 'healthy' }] }),
    run: async (_binary, args) => {
      if (args.includes(machine.target)) throw new Error('private-key: secret-token')
      return args.includes('missing') ? 'XERXES_REMOTE_MISSING' : output
    },
  })
  try {
    await hosts.refresh(); await flush()
    const rows = hosts.contexts()
    expect(rows.map(row => row.availability)).toEqual(['offline', 'no-runtime', 'online'])
    expect(JSON.stringify(rows)).not.toContain('secret-token')
    expect(rows[2]?.sessions).toHaveLength(1)
  } finally { hosts.dispose() }
})
test('saved host timeout aborts only its own probe and retains cached chats while offline', async () => {
  let now = 0, calls = 0, aborted = false
  const hosts = new SavedHostContexts({ timeoutMs: 10, now: () => now,
    list: async () => ({ ok: true, machines: [machine] }),
    run: async (_binary, _args, signal) => {
      if (++calls === 1) return output
      return new Promise((_resolve, reject) => signal.addEventListener('abort', () => { aborted = true; reject(new Error('cancelled')) }, { once: true }))
    },
  })
  try {
    await hosts.refresh(); await flush()
    now = 60000; await hosts.refresh()
    await new Promise(resolve => setTimeout(resolve, 25))
    expect(aborted).toBe(true)
    expect(hosts.contexts()[0]).toMatchObject({ availability: 'offline', reason: 'SSH probe timed out', sessions: [{ id: 'saved-chat' }] })
    expect(calls).toBe(2)
  } finally { hosts.dispose() }
})
test('saved host registry validates SSH targets and aliases before any process runs', async () => {
  for (const extra of [{ target: '-oProxyCommand=bad' }, { alias: 'bad\nalias' }, { workspacePath: '/bad\npath' }]) {
    let calls = 0
    const hosts = new SavedHostContexts({ list: async () => ({ ok: true, machines: [{ ...machine, ...extra }] }), run: async () => { calls++; return output } })
    await hosts.refresh()
    expect(hosts.contexts()).toEqual([])
    expect(calls).toBe(0)
    hosts.dispose()
  }
})
test('sidebar renders saved SSH hosts, chat titles, working state and offline/no-runtime reasons', () => {
  const store = new Store()
  const html = renderToStaticMarkup(createElement(Sidebar, { page: null, snap: { ...store.getSnapshot(), contexts: [
    { id: -1, saved: true, remote: true, label: 'v5p-128 · SSH', workspace: machine.workspacePath, availability: 'online', sessions: savedHostResult(output).sessions },
    { id: -2, saved: true, remote: true, label: 'offline · SSH', workspace: '/other', availability: 'offline', reason: 'SSH host unavailable or read-only probe failed', sessions: [] },
    { id: -3, saved: true, remote: true, label: 'missing · SSH', workspace: '/other', availability: 'no-runtime', reason: 'No running Xerxes runtime', sessions: [] },
  ] } }))
  expect(html).toContain('v5p-128 · SSH')
  expect(html).toContain('Train v5p-128')
  expect(html).toContain('other-project · working')
  expect(html).toContain('SSH host unavailable or read-only probe failed')
  expect(html).toContain('No running Xerxes runtime')
  expect(html).toContain('no runtime')
})
