// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { claudeExecutable } from '../src/auth/claudeCodeLogin.js'
import { CLAUDE_CODE_PACKAGE, ensureClaudeCode } from '../src/runtime/companionInstall.js'

async function inTemporaryHome(body: (root: string) => Promise<void>): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-claude-install-'))
  try { await body(root) } finally { await rm(root, { recursive: true, force: true }) }
}

test('choosing Claude Code installs it through Bun when it is missing, once for concurrent callers', async () => {
  await inTemporaryHome(async root => {
    let installed: string | undefined
    const runs: string[][] = []
    const host = {
      autoInstall: true, bunExecutable: '/fake/bun', stateFile: join(root, 'update.json'), now: () => 1_000,
      find: () => installed,
      run: async (argv: readonly string[]) => { runs.push([...argv]); await Bun.sleep(10); installed = '/home/me/.bun/bin/claude'; return { code: 0, output: '' } },
    }
    const [first, second] = await Promise.all([ensureClaudeCode(host), ensureClaudeCode(host)])
    expect(first).toEqual({ path: '/home/me/.bun/bin/claude', status: 'installed' })
    expect(second).toEqual(first)
    expect(runs).toEqual([['/fake/bun', 'add', '--global', CLAUDE_CODE_PACKAGE]])
    expect(await Bun.file(join(root, 'update.json')).json()).toEqual({ updatedAt: 1_000 })
  })
})

test('a failed install says why and what to do; the switch turns installs off', async () => {
  await expect(ensureClaudeCode({ autoInstall: true, find: () => undefined, run: async () => ({ code: 1, output: 'error: network unreachable\n' }) }))
    .rejects.toThrow('Could not install Claude Code automatically (bun add exited 1): error: network unreachable')
  let ran = false
  await expect(ensureClaudeCode({ environment: { XERXES_AUTO_INSTALL_CLAUDE_CODE: '0' }, find: () => undefined, run: async () => { ran = true; return { code: 0, output: '' } } }))
    .rejects.toThrow('automatic install is off')
  expect(ran).toBe(false)
})

test('a Claude Code Xerxes installed is updated at most daily; one from another installer is left alone', async () => {
  await inTemporaryHome(async root => {
    const stateFile = join(root, 'update.json')
    const runs: string[][] = []
    let now = 10 * 24 * 60 * 60_000
    const managed = { autoInstall: true, environment: { HOME: '/home/me' }, bunExecutable: '/fake/bun', stateFile, now: () => now,
      find: () => '/home/me/.bun/bin/claude', run: async (argv: readonly string[]) => { runs.push([...argv]); return { code: 0, output: '' } } }
    expect(await ensureClaudeCode(managed)).toEqual({ path: '/home/me/.bun/bin/claude', status: 'present' })
    for (let i = 0; i < 50 && runs.length === 0; i++) await Bun.sleep(5)
    expect(runs).toEqual([['/fake/bun', 'add', '--global', `${CLAUDE_CODE_PACKAGE}@latest`]])
    now += 60 * 60_000
    await ensureClaudeCode(managed)
    await Bun.sleep(30)
    expect(runs).toHaveLength(1)
    now += 24 * 60 * 60_000
    await ensureClaudeCode({ ...managed, find: () => '/home/me/.local/bin/claude' })
    await Bun.sleep(30)
    expect(runs).toHaveLength(1)
  })
})

test('Claude Code installed by Xerxes (in Bun\'s global bin) is found', async () => {
  await inTemporaryHome(async root => {
    await Bun.write(join(root, '.bun', 'bin', 'claude'), '#!/bin/sh\n')
    await Bun.$`chmod +x ${join(root, '.bun', 'bin', 'claude')}`
    expect(claudeExecutable({ HOME: root, PATH: '/nonexistent' })).toBe(join(root, '.bun', 'bin', 'claude'))
    expect(claudeExecutable({ HOME: root, PATH: '/nonexistent', BUN_INSTALL: join(root, 'elsewhere') })).toBeUndefined()
  })
})
