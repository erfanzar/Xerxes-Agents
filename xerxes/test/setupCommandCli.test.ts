// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), '..')

test('xerxes setup CLI writes a validated configuration file', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-setup-cli-'))
  try {
    const target = join(directory, 'setup.yaml')
    const child = Bun.spawn({
      cmd: [process.execPath, 'run', 'xerxes', 'setup', '--provider', 'openai', '--model', 'gpt-test', '--api-key', 'sk-test', '--permission-mode', 'manual', '--target', target],
      cwd: resolve(rootDir),
      stdout: 'pipe',
      stderr: 'pipe',
    })
    const exitCode = await child.exited
    const stdout = await new Response(child.stdout).text()
    expect(exitCode).toBe(0)
    expect(stdout).toContain(`Wrote setup configuration to ${target}`)
    const contents = await Bun.file(target).text()
    expect(contents).toContain('provider: "openai"')
    expect(contents).toContain('model: "gpt-test"')
    expect(contents).toContain('api_key: "sk-test"')
    expect(contents).toContain('permission_mode: "manual"')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
