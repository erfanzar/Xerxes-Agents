// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { WorkspacePathResolver } from '../src/tools/pathSafety.js'

test('a read-only root admits reads of the spilled tool results and nothing else', async () => {
  const { mkdtemp, mkdir, writeFile, rm, symlink } = await import('node:fs/promises')
  const { tmpdir } = await import('node:os')
  const { join } = await import('node:path')
  const base = await mkdtemp(join(tmpdir(), 'xerxes-ro-'))
  try {
    const workspace = join(base, 'work'), results = join(base, 'tool-results'), secret = join(base, 'secret.txt')
    await mkdir(workspace); await mkdir(join(results, 's1'), { recursive: true })
    await writeFile(join(results, 's1', 'out.json'), '{}'); await writeFile(secret, 'x')
    await symlink(secret, join(results, 's1', 'escape.txt'))
    const paths = new WorkspacePathResolver(workspace, undefined, [results])
    expect(await paths.resolveReadable(join(results, 's1', 'out.json'))).toEndWith(join('tool-results', 's1', 'out.json'))
    await expect(paths.resolveReadable(secret)).rejects.toThrow('escapes workspace root')
    await expect(paths.resolveReadable(join(results, 's1', 'escape.txt'))).rejects.toThrow()
    await expect(paths.resolveReadable(join(results, '..', 'secret.txt'))).rejects.toThrow()
    // Writes never use the read-only root.
    await expect(paths.resolve(join(results, 's1', 'out.json'))).rejects.toThrow('escapes workspace root')
  } finally {
    await rm(base, { recursive: true, force: true })
  }
})
