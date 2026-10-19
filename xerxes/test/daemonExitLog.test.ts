// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { appendFile, mkdtemp, readFile, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { describeDaemonExit, lastDaemonExit, recordDaemonExit } from '../src/daemon/exitLog.js'

test('a recorded exit is read back as the latest one and explains who restarted the runtime', async () => {
  const root = await mkdtemp(join(tmpdir(), 'daemon-exit-'))
  try {
    const path = join(root, 'daemon', 'exits.jsonl')
    expect(lastDaemonExit(path)).toBeUndefined()
    recordDaemonExit(path, { at: '2026-10-19T10:00:00.000Z', reason: 'shutdown', pid: 1, build: 'a', sessions: 0, unsaved_sessions: 0 })
    recordDaemonExit(path, {
      at: '2026-10-19T11:00:00.000Z', reason: 'restart', pid: 2, build: 'b',
      requested_by: 'Xerxes desktop remote setup\n(build c)', sessions: 2, unsaved_sessions: 1,
    })
    // A torn final line (the process died mid-write) is skipped, not fatal.
    await appendFile(path, '{"at":"broken')
    const last = lastDaemonExit(path)
    expect(last).toMatchObject({ reason: 'restart', pid: 2, requested_by: 'Xerxes desktop remote setup (build c)', unsaved_sessions: 1 })
    expect(describeDaemonExit(last!)).toBe('The Xerxes runtime restarted at 2026-10-19T11:00:00.000Z by Xerxes desktop remote setup (build c); 1 chat without a finished turn could not be kept.')
    expect((await stat(path)).mode & 0o777).toBe(0o600)
    expect((await readFile(path, 'utf8')).split('\n').filter(Boolean)).toHaveLength(3)
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('a crash names its cause, and an unwritable log is reported instead of thrown', () => {
  expect(describeDaemonExit({ at: 't', reason: 'crash', pid: 3, build: 'x', sessions: 1, unsaved_sessions: 0, detail: 'boom' }))
    .toBe('The Xerxes runtime crashed at t (boom).')
  expect(() => recordDaemonExit('/dev/null/not-a-dir/exits.jsonl', { at: 't', reason: 'crash', pid: 3, build: 'x', sessions: 0, unsaved_sessions: 0 })).not.toThrow()
})
