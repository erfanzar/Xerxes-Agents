// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const CLI = join(import.meta.dir, '../src/cli.ts')

test('telegram CLI rejects a non-numeric or out-of-range --port before starting', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-bun-cli-telegram-port-'))
  try {
    for (const port of ['not-a-port', '70000']) {
      const child = Bun.spawn(
        [process.execPath, CLI, 'telegram', '--token', 'test-token', '--port', port],
        {
          cwd: root,
          env: { ...process.env, XERXES_HOME: join(root, 'home') },
          stderr: 'pipe',
          stdout: 'pipe',
        },
      )
      const [stderr, exitCode] = await Promise.all([
        new Response(child.stderr).text(),
        child.exited,
      ])
      expect(exitCode).not.toBe(0)
      expect(stderr).toContain('telegram --port must be an integer between 0 and 65535')
    }
  } finally {
    await rm(root, { recursive: true, force: true })
  }
})

test('telegram CLI accepts a numeric --port and proceeds past validation', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-bun-cli-telegram-valid-'))
  const child = Bun.spawn(
    [process.execPath, CLI, 'telegram', '--token', 'test-token', '--port', '47321'],
    {
      cwd: root,
      env: { ...process.env, XERXES_HOME: join(root, 'home') },
      stderr: 'pipe',
      stdout: 'pipe',
    },
  )
  try {
    // A validation failure crashes the process almost immediately; surviving
    // the grace window (or failing later for an unrelated reason) proves the
    // port regex accepted the numeric value.
    const exitedEarly = await Promise.race([
      child.exited.then(() => true),
      Bun.sleep(2_000).then(() => false),
    ])
    if (!exitedEarly) child.kill()
    const stderr = await new Response(child.stderr).text()
    await child.exited
    expect(stderr).not.toContain('telegram --port must be')
  } finally {
    child.kill()
    await rm(root, { recursive: true, force: true })
  }
})
