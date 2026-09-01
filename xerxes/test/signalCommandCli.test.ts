// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { dirname, resolve } from 'node:path'
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

test('xerxes signal CLI requires rest url and sender number', async () => {
  const missingUrl = spawnCli(['signal'])
  expect(await missingUrl.process.exited).toBe(1)
  expect(await missingUrl.stderr).toContain('signal requires --rest-url')

  const missingNumber = spawnCli(['signal', '--rest-url', 'http://localhost:8080'])
  expect(await missingNumber.process.exited).toBe(1)
  expect(await missingNumber.stderr).toContain('signal requires --sender-number')
})
