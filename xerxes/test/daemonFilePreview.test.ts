// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { previewWorkspaceFile } from '../src/daemon/filePreview.js'

test('file previews preserve whitespace, bound reads and reject non-text or escaping paths', async () => {
  const root = await mkdtemp(join(tmpdir(), 'file-preview-'))
  const workspace = join(root, 'workspace')
  try {
    await mkdir(workspace)
    await Bun.write(join(workspace, 'source.ts'), 'first\n  second\n')
    expect(await previewWorkspaceFile(workspace, './source.ts')).toEqual({ ok: true, path: 'source.ts', content: 'first\n  second\n', truncated: false })
    await Bun.write(join(workspace, 'large.txt'), 'x'.repeat(150_000))
    const large = await previewWorkspaceFile(workspace, 'large.txt')
    expect(large.truncated).toBe(true)
    expect(large.content.length).toBe(128 * 1024)
    await Bun.write(join(root, 'outside.txt'), 'private')
    await symlink(join(root, 'outside.txt'), join(workspace, 'escape'))
    for (const path of ['../outside.txt', 'escape', join(root, 'outside.txt')]) {
      await expect(previewWorkspaceFile(workspace, path)).rejects.toThrow('selected workspace')
    }
    await Bun.write(join(workspace, 'binary'), new Uint8Array([0, 1, 2]))
    await expect(previewWorkspaceFile(workspace, 'binary')).rejects.toThrow('binary')
    await Bun.write(join(workspace, 'invalid'), new Uint8Array([255, 255]))
    await expect(previewWorkspaceFile(workspace, 'invalid')).rejects.toThrow('UTF-8')
    await expect(previewWorkspaceFile(workspace, 'missing')).rejects.toThrow()
    await expect(previewWorkspaceFile(workspace, null)).rejects.toThrow('valid workspace')
  } finally { await rm(root, { recursive: true, force: true }) }
})
