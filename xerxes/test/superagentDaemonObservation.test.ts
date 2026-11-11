// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { observeWorkspace } from '../src/daemon/superagentObservation.js'
import { HUB_DAEMON_METHODS } from '../src/hub/runtimePool.js'
import { hubMethodPolicy } from '../src/security/hubIdentityVerifier.js'

test('daemon observation dispatcher uses a pinned root, a closed operation set and bounded JSON', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-native-observe-'))
  try {
    await writeFile(join(root, 'hello.txt'), 'hello')
    expect(await observeWorkspace(root, { ability: 'workspace.read', path: 'hello.txt' })).toMatchObject({ text: 'hello' })
    await expect(observeWorkspace(root, { ability: 'exec_command', cmd: 'touch bad' })).rejects.toThrow()
    await expect(observeWorkspace(root, { ability: 'workspace.read', path: '../hello.txt' })).rejects.toThrow()
    expect(HUB_DAEMON_METHODS).toContain('superagent.observe')
    expect(hubMethodPolicy('superagent.observe', {})).toEqual({ kind: 'scope', scope: 'session.read' })
  } finally { await rm(root, { recursive: true, force: true }) }
})
