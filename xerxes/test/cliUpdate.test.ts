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
  // The report now presents settled facts as an aligned label/value block, so
  // the assertion targets the pairing rather than the old `Label: ` prefix.
  expect(stdout).toMatch(/\n  Git {2,}\S/)
  expect(stdout).toMatch(/\n  Registry {2,}not checked/)
  expect(stdout).toContain(`Would run: ${process.execPath} add --global file:./release-preview`)
})

test('Bun CLI renders update option errors as one clean line without a stack dump', async () => {
  for (const args of [
    ['update', '--force'],
    ['update', '--git', '--spec', 'file:./release-preview'],
  ]) {
    const child = Bun.spawn([process.execPath, join(import.meta.dir, '../src/cli.ts'), ...args], {
      stderr: 'pipe',
      stdout: 'pipe',
    })
    const [stdout, stderr, exitCode] = await Promise.all([
      new Response(child.stdout).text(),
      new Response(child.stderr).text(),
      child.exited,
    ])

    expect(exitCode).toBe(1)
    expect(stdout).toBe('')
    const lines = stderr.trim().split('\n')
    expect(lines).toHaveLength(2)
    // A status glyph precedes the conventional `error:` label, which is retained
  // so the line stays greppable.
  expect(lines[0]).toContain('error: ')
    expect(lines[1]).toBe("run 'xerxes update --help' for usage.")
    expect(stderr).not.toContain('at ')
    expect(stderr).not.toContain('cli.js')
  }
})
