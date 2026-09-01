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

test('xerxes doctor --json outputs JSON array', async () => {
  const doctor = spawnCli(['doctor', '--json'])
  const exit = await doctor.process.exited
  const stdout = await doctor.stdout
  expect(exit).toBe(0)
  const parsed = JSON.parse(stdout) as unknown
  expect(Array.isArray(parsed)).toBeTrue()
  expect((parsed as Array<{ name: string }>).length).toBeGreaterThan(0)
  expect(typeof (parsed as Array<{ name: string }>)[0]?.name).toBe('string')
})
