// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const CLI = join(import.meta.dir, '../src/cli.ts')

async function runCli(args: readonly string[]): Promise<{
  readonly exitCode: number
  readonly stderr: string
  readonly stdout: string
}> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-bun-cli-flags-'))
  try {
    const child = Bun.spawn([process.execPath, CLI, ...args], {
      cwd: root,
      env: { ...process.env, XERXES_HOME: join(root, 'home') },
      stderr: 'pipe',
      stdout: 'pipe',
    })
    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
      child.exited,
    ])
    return { exitCode, stderr, stdout }
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

test('daemon rejects unknown or misspelled flags instead of ignoring them', async () => {
  const misspelled = await runCli(['daemon', '--socke', '/tmp/x.sock'])
  expect(misspelled.exitCode).not.toBe(0)
  expect(misspelled.stderr).toContain('Unknown daemon option: --socke')

  const unknown = await runCli(['daemon', '--port', '8080'])
  expect(unknown.exitCode).not.toBe(0)
  expect(unknown.stderr).toContain('Unknown daemon option: --port')
})

test('daemon rejects a flag-like value where an option value is required', async () => {
  const result = await runCli(['daemon', '--socket', '--pid-file', '/tmp/x.pid'])
  expect(result.exitCode).not.toBe(0)
  expect(result.stderr).toContain('daemon option --socket requires a value')
})

test('telegram rejects unknown or misspelled flags instead of ignoring them', async () => {
  const misspelled = await runCli(['telegram', '--token', 'test-token', '--prot', '1234'])
  expect(misspelled.exitCode).not.toBe(0)
  expect(misspelled.stderr).toContain('Unknown telegram option: --prot')

  const missingValue = await runCli(['telegram', '--token'])
  expect(missingValue.exitCode).not.toBe(0)
  expect(missingValue.stderr).toContain('telegram option --token requires a value')
})
