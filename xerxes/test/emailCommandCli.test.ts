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

test('xerxes email CLI requires smtp user, password, and valid port', async () => {
  const missingUser = spawnCli(['email'])
  expect(await missingUser.process.exited).toBe(1)
  expect(await missingUser.stderr).toContain('email requires --smtp-user')

  const missingPassword = spawnCli(['email', '--smtp-user', 'user'])
  expect(await missingPassword.process.exited).toBe(1)
  expect(await missingPassword.stderr).toContain('email requires --smtp-password')

  const invalidPort = spawnCli(['email', '--smtp-user', 'user', '--smtp-password', 'pass', '--smtp-port', 'abc'])
  expect(await invalidPort.process.exited).toBe(1)
  expect(await invalidPort.stderr).toContain('email --smtp-port')
})
