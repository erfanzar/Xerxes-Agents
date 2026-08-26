// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { chmod, mkdtemp, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'

import { runSetupCommand } from '../src/runtime/setupCommand.js'
import { writeSetupConfig } from '../src/runtime/setupWizard.js'

test('setup command writes a validated provider configuration file', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-setup-'))
  try {
    const target = join(directory, 'setup.yaml')
    const exitCode = await runSetupCommand({
      targetPath: target,
      answers: {
        provider: 'openai',
        model: 'gpt-test',
        api_key: 'sk-test',
        permission_mode: 'manual',
        enable_voice: 'n',
        messaging_platform: 'none',
      },
    })
    expect(exitCode).toBe(0)
    const contents = await Bun.file(target).text()
    expect(contents).toContain('provider: "openai"')
    expect(contents).toContain('model: "gpt-test"')
    expect(contents).toContain('api_key: "sk-test"')
    expect(contents).toContain('permission_mode: "manual"')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('setup command rejects an unknown provider', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-setup-bad-'))
  try {
    const target = join(directory, 'setup.yaml')
    await expect(runSetupCommand({
      targetPath: target,
      answers: { provider: 'fake-provider' },
    })).rejects.toThrow(/unknown provider fake-provider/)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('the setup config holding a provider credential is owner-only', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-setup-perms-'))
  try {
    const target = join(directory, 'nested', 'setup.yaml')
    await writeSetupConfig({ provider: 'anthropic', api_key: 'sk-secret' }, target)
    // `xerxes setup --api-key` writes a credential here; the default 0644 made
    // it readable by every local user on a shared host.
    expect((await stat(target)).mode & 0o777).toBe(0o600)
    expect((await stat(dirname(target))).mode & 0o777).toBe(0o700)

    // Overwriting an existing file does not restore a looser mode — writeFile
    // only applies `mode` when it creates the file.
    await chmod(target, 0o644)
    await writeSetupConfig({ provider: 'anthropic', api_key: 'sk-rotated' }, target)
    expect((await stat(target)).mode & 0o777).toBe(0o600)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
