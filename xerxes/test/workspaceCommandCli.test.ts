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

test('xerxes workspace CLI manages a local workspace end-to-end', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-workspace-cli-'))
  try {
    const create = spawnCli(['workspace', 'create', '--id', 'cli-ws', '--working-dir', directory])
    const createExit = await create.process.exited
    const createStderr = await create.stderr
    expect(createExit).toBe(0)

    const write = spawnCli(['workspace', 'write', '--id', 'cli-ws', '--path', 'note.txt', '--content', 'hello workspace', '--working-dir', directory])
    expect(await write.process.exited).toBe(0)

    const read = spawnCli(['workspace', 'read', '--id', 'cli-ws', '--path', 'note.txt', '--working-dir', directory])
    const readStdout = await read.stdout
    expect(await read.process.exited).toBe(0)
    expect(readStdout).toContain('hello workspace')

    const exec = spawnCli(['workspace', 'exec', '--id', 'cli-ws', 'cat', 'note.txt', '--working-dir', directory])
    const execStdout = await exec.stdout
    expect(await exec.process.exited).toBe(0)
    expect(execStdout).toContain('hello workspace')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('workspace exec passes flags to the command after --', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-workspace-cli-flags-'))
  try {
    const create = spawnCli(['workspace', 'create', '--id', 'flag-ws', '--working-dir', directory])
    await create.process.exited
    await Bun.write(join(directory, 'note.txt'), 'hello')

    // Every `-`-prefixed token was claimed as a workspace option, so a command
    // with its own flags could not be run at all: this died on `-n`.
    const exec = spawnCli(['workspace', 'exec', '--id', 'flag-ws', '--working-dir', directory, '--', 'head', '-n', '1', 'note.txt'])
    const stdout = await exec.stdout
    expect(await exec.process.exited).toBe(0)
    expect(stdout).toContain('hello')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
