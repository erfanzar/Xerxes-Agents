// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { join } from 'node:path'

test('Bun CLI exposes a no-mutation update dry-run surface', async () => {
  const child = Bun.spawn([
    process.execPath,
    join(import.meta.dir, '../src/cli.ts'),
    'update',
    '--dry-run',
    '--spec',
    'file:./release-preview',
  ], {
    stderr: 'pipe',
    stdout: 'pipe',
  })
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
    child.exited,
  ])

  expect(exitCode).toBe(0)
  expect(stderr).toBe('')
  expect(stdout).toContain('Git: ')
  expect(stdout).toContain('Package registry: not checked')
  expect(stdout).toContain(`Would run: ${process.execPath} add --global file:./release-preview`)
})
