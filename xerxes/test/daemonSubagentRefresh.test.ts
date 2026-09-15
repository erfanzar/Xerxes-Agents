// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'

test('session status refreshes idle child progress on the owning session without replacing its mutable state', async () => {
  const root = await mkdtemp(join(tmpdir(), 'subagent-refresh-'))
  const states = new Map<string, string>()
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: root, sessionDirectory: join(root, 'sessions'),
    refreshSubagents: session => { session.metadata.child_status = states.get(session.id) ?? 'none' },
  })
  try {
    const a = await runtime.openSession('a'), b = await runtime.openSession('b')
    states.set(a.id, 'failed')
    expect(runtime.sessionStatus('a')).toBe(a)
    expect(a.metadata.child_status).toBe('failed')
    states.set(a.id, 'completed')
    expect(runtime.sessionStatus('a')?.metadata.child_status).toBe('completed')
    expect(runtime.sessionStatus('b')).toBe(b)
    expect(b.metadata.child_status).toBe('none')
    expect(runtime.sessionStatus('missing')).toBeUndefined()
  } finally { await runtime.shutdown(); await rm(root, { recursive: true, force: true }) }
})
