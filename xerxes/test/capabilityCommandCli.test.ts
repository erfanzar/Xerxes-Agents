// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), '..')

function spawnCli(args: string[]): { process: ReturnType<typeof Bun.spawn>; stdout: Promise<string>; stderr: Promise<string> } {
  const child = Bun.spawn({
    cmd: [process.execPath, 'src/cli.ts', ...args],
    cwd: resolve(rootDir),
    stdout: 'pipe',
    stderr: 'pipe',
  })
  return {
    process: child,
    stdout: new Response(child.stdout).text(),
    stderr: new Response(child.stderr).text(),
  }
}

test('xerxes capability CLI registers and lists a manifest', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-capability-cli-'))
  try {
    const manifest = JSON.stringify({ id: 'cli-plugin', capabilities: [{ scope: 'fs', action: 'read', resources: ['/tmp/*'] }] })
    const register = spawnCli(['capability', 'register', '--id', 'cli-plugin', '--manifest-json', manifest, '--directory', directory])
    const registerExit = await register.process.exited
    const registerStdout = await register.stdout
    expect(registerExit).toBe(0)
    expect(registerStdout).toContain('registered capabilities for cli-plugin')

    const list = spawnCli(['capability', 'list', '--directory', directory])
    const listExit = await list.process.exited
    const listStdout = await list.stdout
    expect(listExit).toBe(0)
    expect(listStdout).toContain('cli-plugin')
    expect(listStdout).toContain('fs:read')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
