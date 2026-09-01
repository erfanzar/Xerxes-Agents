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

test('xerxes whatsapp CLI requires access token and phone number id', async () => {
  const missingToken = spawnCli(['whatsapp'])
  expect(await missingToken.process.exited).toBe(1)
  expect(await missingToken.stderr).toContain('whatsapp requires --access-token')

  const missingPhone = spawnCli(['whatsapp', '--access-token', 'fake-token'])
  expect(await missingPhone.process.exited).toBe(1)
  expect(await missingPhone.stderr).toContain('whatsapp requires --phone-number-id')
})
