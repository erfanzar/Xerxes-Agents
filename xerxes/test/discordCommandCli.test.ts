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

test('xerxes discord CLI requires token, application-id, and valid transport', async () => {
  const missingToken = spawnCli(['discord'])
  expect(await missingToken.process.exited).toBe(1)
  expect(await missingToken.stderr).toContain('discord requires --token')

  const missingApp = spawnCli(['discord', '--token', 'fake-token'])
  expect(await missingApp.process.exited).toBe(1)
  expect(await missingApp.stderr).toContain('discord requires --application-id')

  const invalidTransport = spawnCli(['discord', '--token', 'fake-token', '--application-id', '123', '--transport', 'bad'])
  expect(await invalidTransport.process.exited).toBe(1)
  expect(await invalidTransport.stderr).toContain('webhook')
})
