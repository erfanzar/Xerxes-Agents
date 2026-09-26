// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'

test('re-opening a loaded chat does not move its latest-message clock', async () => {
  // lastActive is what session.list/active_list report as the chat's
  // recency. Bumping it on open made a merely clicked chat jump to the top
  // of the desktop sidebar and claim "now".
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-reopen-clock-'))
  try {
    const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') })
    const session = await runtime.openSession('clicked-chat')
    session.lastActive = 1_000
    const reopened = await runtime.openSession('clicked-chat')
    expect(reopened).toBe(session)
    expect(reopened.lastActive).toBe(1_000)
  } finally { await rm(directory, { recursive: true, force: true }) }
})
