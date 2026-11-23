// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { cliSearchDirectories, ensureLocalProviderClis, probeLocalProviderClis } from '../src/desktop/main/localProviderSetup.js'

test('a CLI installed in ~/.bun/bin or ~/.local/bin counts even when PATH omits it', async () => {
  const home = await mkdtemp(join(tmpdir(), 'local-cli-home-'))
  try {
    // PATH deliberately excludes it: the desktop process rarely inherits ~/.bun/bin or ~/.local/bin.
    expect(await probeLocalProviderClis({ PATH: '/nonexistent' }, home)).toEqual({ claude: false, codex: false, agy: false })
    await mkdir(join(home, '.bun', 'bin'), { recursive: true })
    await mkdir(join(home, '.local', 'bin'), { recursive: true })
    await writeFile(join(home, '.bun', 'bin', 'claude'), '#!/bin/sh\n', { mode: 0o755 })
    await writeFile(join(home, '.local', 'bin', 'agy'), '#!/bin/sh\n', { mode: 0o755 })
    expect(await probeLocalProviderClis({ PATH: '/nonexistent' }, home)).toEqual({ claude: true, codex: false, agy: true })
    expect(cliSearchDirectories(home)[0]).toBe(join(home, '.bun', 'bin'))
    expect(cliSearchDirectories(home)[1]).toBe(join(home, '.local', 'bin'))
  } finally { await rm(home, { recursive: true, force: true }) }
})

test('only missing CLIs are installed, and one failure never hides the others', async () => {
  const installed: string[] = []
  const status = { claude: false, codex: true, agy: false }
  await ensureLocalProviderClis({
    probe: async () => ({ ...status }),
    install: async name => { installed.push(name); status[name] = true },
  })
  expect(installed).toEqual(['claude', 'agy'])

  installed.length = 0
  await expect(ensureLocalProviderClis({
    probe: async () => ({ claude: false, codex: false, agy: false }),
    install: async name => { installed.push(name); throw new Error('network down') },
  })).rejects.toThrow('did not finish')
  // All three are attempted, so one broken install does not strand the others.
  expect(installed).toEqual(['claude', 'codex', 'agy'])
})

test('an already-complete computer installs nothing', async () => {
  const installed: string[] = []
  expect(await ensureLocalProviderClis({
    probe: async () => ({ claude: true, codex: true, agy: true }),
    install: async name => { installed.push(name) },
  })).toEqual({ claude: true, codex: true, agy: true })
  expect(installed).toEqual([])
})