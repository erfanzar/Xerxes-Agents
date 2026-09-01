// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { runCapabilityCommand } from '../src/runtime/capabilityCommand.js'

test('capability command registers, lists, and diffs manifests', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-capability-cmd-'))
  try {
    const manifest = { id: 'plugin-a', capabilities: [{ scope: 'fs', action: 'read', resources: ['/tmp/*'] }] }
    const registerResult = await runCapabilityCommand({ action: 'register', id: manifest.id, manifestJson: JSON.stringify(manifest), directory })
    expect(registerResult.ok).toBeTrue()
    expect(registerResult.message).toContain('registered capabilities for plugin-a')

    const listResult = await runCapabilityCommand({ action: 'list', directory })
    expect(listResult.ok).toBeTrue()
    expect(listResult.message).toContain('plugin-a')

    const fromFile = join(directory, 'from.json')
    const toFile = join(directory, 'to.json')
    await writeFile(fromFile, JSON.stringify([manifest]))
    await writeFile(toFile, JSON.stringify([{ id: 'plugin-a', capabilities: [{ scope: 'fs', action: 'write', resources: ['/tmp/*'] }] }]))

    const diffResult = await runCapabilityCommand({ action: 'diff', fromSnapshot: fromFile, toSnapshot: toFile, directory })
    expect(diffResult.ok).toBeTrue()
    expect(diffResult.message).toContain('+ plugin-a: fs:write')
    expect(diffResult.message).toContain('- plugin-a: fs:read')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
