// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, realpath, rename, rm, symlink, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { AnchoredWorkspaceFiles } from '../src/hub/supervisor/abilities/anchoredFiles.js'

test.skipIf(process.platform === 'win32')('descriptor-anchored reads reject symlinks in every path component and bound actual reads', async () => {
  const root = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-anchored-')))
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

test.skipIf(process.platform === 'win32')('registered root and ancestor symlink substitutions cannot redirect anchored reads', async () => {
  const temporary = await mkdtemp(join(tmpdir(), 'xerxes-anchor-root-'))
  const root = await realpath(temporary)
  try {
    await mkdir(join(root, 'parent', 'workspace'), { recursive: true })
    await mkdir(join(root, 'outside', 'workspace'), { recursive: true })
    await writeFile(join(root, 'parent', 'workspace', 'note'), 'authorized')
    await writeFile(join(root, 'outside', 'workspace', 'note'), 'outside secret')
    const files = new AnchoredWorkspaceFiles(join(root, 'parent', 'workspace'))
    expect(new TextDecoder().decode((await files.read('note', 0, 64)).bytes)).toBe('authorized')
    await rename(join(root, 'parent', 'workspace'), join(root, 'parent', 'original'))
    await symlink(join(root, 'outside', 'workspace'), join(root, 'parent', 'workspace'))
    await expect(files.read('note', 0, 64)).rejects.toThrow('anchor')
    await rm(join(root, 'parent', 'workspace'))
    await rename(join(root, 'parent', 'original'), join(root, 'parent', 'workspace'))
    await rename(join(root, 'parent'), join(root, 'original-parent'))
    await symlink(join(root, 'outside'), join(root, 'parent'))
    await expect(files.read('note', 0, 64)).rejects.toThrow('anchor')
  } finally { await rm(temporary, { recursive: true, force: true }) }
})
