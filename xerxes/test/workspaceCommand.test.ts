// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { runWorkspaceCommand } from '../src/runtime/workspaceCommand.js'

test('workspace command creates, writes, reads, executes, and destroys a local workspace', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-workspace-cmd-'))
  try {
    const createResult = await runWorkspaceCommand({ action: 'create', id: 'ws-test', workingDir: directory })
    expect(createResult.ok).toBeTrue()
    expect(createResult.connection?.workingDir).toBe(directory)

    const writeResult = await runWorkspaceCommand({ action: 'write', id: 'ws-test', path: 'hello.txt', content: 'hello workspace', workingDir: directory })
    expect(writeResult.ok).toBeTrue()

    const readResult = await runWorkspaceCommand({ action: 'read', id: 'ws-test', path: 'hello.txt', workingDir: directory })
    expect(readResult.ok).toBeTrue()
    expect(readResult.content).toBe('hello workspace')

    const execResult = await runWorkspaceCommand({ action: 'exec', id: 'ws-test', command: ['cat', 'hello.txt'], workingDir: directory })
    expect(execResult.ok).toBeTrue()
    expect(execResult.stdout).toContain('hello workspace')

    const destroyResult = await runWorkspaceCommand({ action: 'destroy', id: 'ws-test', workingDir: directory })
    expect(destroyResult.ok).toBeTrue()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
