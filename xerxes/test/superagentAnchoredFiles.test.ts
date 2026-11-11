// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, rm, symlink, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { AnchoredWorkspaceFiles } from '../src/hub/supervisor/abilities/anchoredFiles.js'

test.skipIf(process.platform === 'win32')('descriptor-anchored reads reject symlinks in every path component and bound actual reads', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-anchored-'))
  try {
    await mkdir(join(root, 'safe')); await writeFile(join(root, 'safe', 'file'), 'a'.repeat(10000))
    await symlink(join(root, 'safe'), join(root, 'link'))
    await symlink(join(root, 'safe', 'file'), join(root, 'leaf'))
    const files = new AnchoredWorkspaceFiles(root)
    expect((await files.read('safe/file', 0, 10)).bytes.byteLength).toBe(10)
    await expect(files.read('link/file', 0, 10)).rejects.toThrow('anchor')
    await expect(files.read('leaf', 0, 10)).rejects.toThrow('anchor')
    expect((await files.list('safe')).map(entry => entry.name)).toEqual(['file'])
  } finally { await rm(root, { recursive: true, force: true }) }
})
