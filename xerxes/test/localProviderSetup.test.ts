// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { cliSearchDirectories, ensureLocalProviderClis, probeLocalProviderClis } from '../src/desktop/main/localProviderSetup.js'

test('a CLI installed in ~/.bun/bin counts even when PATH omits it', async () => {
  const home = await mkdtemp(join(tmpdir(), 'local-cli-home-'))
  try {
    // PATH deliberately excludes it: the desktop process rarely inherits ~/.bun/bin.
    expect(await probeLocalProviderClis({ PATH: '/nonexistent' }, home)).toEqual({ claude: false, codex: false })
    await mkdir(join(home, '.bun', 'bin'), { recursive: true })
    await writeFile(join(home, '.bun', 'bin', 'claude'), '#!/bin/sh\n', { mode: 0o755 })
    expect(await probeLocalProviderClis({ PATH: '/nonexistent' }, home)).toEqual({ claude: true, codex: false })
    expect(cliSearchDirectories(home)[0]).toBe(join(home, '.bun', 'bin'))
  } finally { await rm(home, { recursive: true, force: true }) }
})

test('only the missing CLI is installed, and one failure never hides the other', async () => {
  const installed: string[] = []
  const status = { claude: false, codex: true }
  await ensureLocalProviderClis({
    probe: async () => ({ ...status }),
    install: async name => { installed.push(name); status[name] = true },
  })
  expect(installed).toEqual(['claude'])

  installed.length = 0
  await expect(ensureLocalProviderClis({
    probe: async () => ({ claude: false, codex: false }),
    install: async name => { installed.push(name); throw new Error('network down') },
  })).rejects.toThrow('did not finish')
  // Both are attempted, so one broken install does not strand the other.
  expect(installed).toEqual(['claude', 'codex'])
})

test('an already-complete computer installs nothing', async () => {
  const installed: string[] = []
  expect(await ensureLocalProviderClis({
    probe: async () => ({ claude: true, codex: true }),
    install: async name => { installed.push(name) },
  })).toEqual({ claude: true, codex: true })
  expect(installed).toEqual([])
})