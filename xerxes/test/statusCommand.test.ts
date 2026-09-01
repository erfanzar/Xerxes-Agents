// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { runStatusCommand } from '../src/runtime/statusCommand.js'

test('status command reports counts from persisted subsystems', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-status-'))
  try {
    await mkdir(join(directory, 'scheduler'))
    await writeFile(join(directory, 'scheduler', 'triggers.json'), '[]')
    await mkdir(join(directory, 'governed-memory'))
    await writeFile(join(directory, 'governed-memory', 'records.json'), '{}')
    await mkdir(join(directory, 'capabilities'), { recursive: true })
    await writeFile(join(directory, 'capabilities', 'manifests.json'), '[{"id":"p","capabilities":[]}]')
    await mkdir(join(directory, 'telemetry'), { recursive: true })
    await writeFile(join(directory, 'telemetry', 'events.jsonl'), '{"event":"e"}\n')
    await mkdir(join(directory, 'workspaces', 'ws1'), { recursive: true })

    const result = await runStatusCommand({ directory })
    expect(result.ok).toBeTrue()
    expect(result.message).toContain('scheduler triggers: 1')
    expect(result.message).toContain('memory records: 1')
    expect(result.message).toContain('capability manifests: 1')
    expect(result.message).toContain('telemetry events: 1')
    expect(result.message).toContain('workspaces: 1')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
