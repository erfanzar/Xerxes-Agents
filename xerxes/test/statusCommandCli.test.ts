// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
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

test('xerxes status CLI reports subsystem counts', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-status-cli-'))
  try {
    await mkdir(join(directory, 'telemetry'), { recursive: true })
    await writeFile(join(directory, 'telemetry', 'events.jsonl'), '{"event":"cli-test"}\n')
    const status = spawnCli(['status', '--directory', directory])
    const exit = await status.process.exited
    const stdout = await status.stdout
    expect(exit).toBe(0)
    expect(stdout).toContain('telemetry events: 1')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
