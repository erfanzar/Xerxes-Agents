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

test('xerxes schedule CLI creates and fires a webhook trigger', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-schedule-cli-'))
  try {
    const create = spawnCli(['schedule', 'create', '--id', 'cli-trigger', '--owner', 'user', '--schedule', 'webhook:/hooks/build', '--objective', 'run build', '--directory', directory])
    const createExit = await create.process.exited
    const createStdout = await create.stdout
    expect(createExit).toBe(0)
    expect(createStdout).toContain('created trigger cli-trigger')

    const list = spawnCli(['schedule', 'list', '--directory', directory])
    const listExit = await list.process.exited
    const listStdout = await list.stdout
    expect(listExit).toBe(0)
    expect(listStdout).toContain('cli-trigger')

    const fire = spawnCli(['schedule', 'fire', '--id', 'cli-trigger', '--delivery-id', 'delivery-1', '--directory', directory])
    const fireExit = await fire.process.exited
    const fireStdout = await fire.stdout
    expect(fireExit).toBe(0)
    expect(fireStdout).toContain('fired trigger cli-trigger')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
