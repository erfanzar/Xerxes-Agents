// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { chmod, mkdtemp, rm, stat } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'

import { runSetupCommand } from '../src/runtime/setupCommand.js'
import { writeSetupConfig } from '../src/runtime/setupWizard.js'
import { SETUP_PROFILES } from '../src/runtime/setupProfiles.js'

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

test('setup command applies a profile preset and allows answer overrides', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-setup-profile-'))
  try {
    const target = join(directory, 'setup.yaml')
    const exitCode = await runSetupCommand({
      targetPath: target,
      profile: 'developer',
      answers: { provider: 'openai', model: 'custom-model' },
    })
    expect(exitCode).toBe(0)
    const contents = await Bun.file(target).text()
    expect(SETUP_PROFILES.developer.answers).not.toHaveProperty('model')
    expect(contents).toContain('provider: "openai"')
    expect(contents).toContain('model: "custom-model"')
    expect(contents).toContain('permission_mode: "manual"')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('setup takes the first model the provider lists when none is given, and refuses without a provider', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-setup-discover-'))
  try {
    const target = join(directory, 'setup.yaml')
    let asked = ''
    expect(await runSetupCommand({ targetPath: target, answers: { provider: 'openai', api_key: 'k' }, discoverModels: async input => { asked = `${input.provider} ${input.baseUrl} ${input.apiKey}`; return ['listed-first', 'listed-second'] } })).toBe(0)
    expect(asked).toBe('openai https://api.openai.com/v1 k')
    expect(await Bun.file(target).text()).toContain('model: "listed-first"')
    await expect(runSetupCommand({ targetPath: join(directory, 'b.yaml'), answers: {} })).rejects.toThrow('choose a provider with --provider')
    await expect(runSetupCommand({ targetPath: join(directory, 'c.yaml'), answers: { provider: 'openai' }, discoverModels: async () => [] })).rejects.toThrow('pass --model')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
