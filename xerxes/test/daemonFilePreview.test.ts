// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { previewWorkspaceFile, writeWorkspaceFile } from '../src/daemon/filePreview.js'

test('file previews preserve whitespace, bound reads and reject non-text or escaping paths', async () => {
  const root = await mkdtemp(join(tmpdir(), 'file-preview-'))
  const workspace = join(root, 'workspace')
  try {
    await mkdir(workspace)
    await Bun.write(join(workspace, 'source.ts'), 'first\n  second\n')
    expect(await previewWorkspaceFile(workspace, './source.ts')).toMatchObject({ ok: true, path: 'source.ts', content: 'first\n  second\n', truncated: false })
    await Bun.write(join(workspace, 'large.txt'), 'x'.repeat(150_000))
    const large = await previewWorkspaceFile(workspace, 'large.txt')
    expect(large.truncated).toBe(true)
    // A partial view has no version: it cannot be edited.
    expect('version' in large).toBe(false)
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

test('an edit saves only over the version it opened, inside the workspace, as text', async () => {
  const root = await mkdtemp(join(tmpdir(), 'file-write-'))
  const workspace = join(root, 'workspace')
  try {
    await mkdir(workspace)
    await Bun.write(join(workspace, 'a.ts'), 'one\n')
    const opened = await previewWorkspaceFile(workspace, 'a.ts')
    expect(opened.version).toMatch(/^[0-9a-f]{64}$/)
    const saved = await writeWorkspaceFile(workspace, 'a.ts', 'one\ntwo\n', opened.version)
    expect(saved).toMatchObject({ ok: true, path: 'a.ts' })
    expect(await Bun.file(join(workspace, 'a.ts')).text()).toBe('one\ntwo\n')
    // Saving again over the old version is a conflict, and writes nothing.
    expect(await writeWorkspaceFile(workspace, 'a.ts', 'stale', opened.version)).toMatchObject({ ok: false, conflict: true })
    expect(await Bun.file(join(workspace, 'a.ts')).text()).toBe('one\ntwo\n')
    // The version a save returns is the one the next save builds on.
    expect(await writeWorkspaceFile(workspace, 'a.ts', 'three\n', (saved as { version: string }).version)).toMatchObject({ ok: true })
    await Bun.write(join(root, 'outside.txt'), 'private')
    await symlink(join(root, 'outside.txt'), join(workspace, 'escape'))
    const version = (await previewWorkspaceFile(root, 'outside.txt')).version!
    await expect(writeWorkspaceFile(workspace, 'escape', 'x', version)).rejects.toThrow('selected workspace')
    expect(await Bun.file(join(root, 'outside.txt')).text()).toBe('private')
    await expect(writeWorkspaceFile(workspace, 'new.ts', 'x', version)).rejects.toThrow()
    await expect(writeWorkspaceFile(workspace, 'a.ts', 'x\0', version)).rejects.toThrow('NUL')
    await expect(writeWorkspaceFile(workspace, 'a.ts', 'x'.repeat(130 * 1024), version)).rejects.toThrow('128 KiB')
    await expect(writeWorkspaceFile(workspace, 'a.ts', 'x', 'nope')).rejects.toThrow('base_version')
  } finally { await rm(root, { recursive: true, force: true }) }
})
