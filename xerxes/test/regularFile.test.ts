// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { constants, closeSync, readFileSync } from 'node:fs'
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { openRegularFile, openRegularFileSync } from '../src/security/regularFile.js'

test('verified file opens refuse a linked destination before truncating its target', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-regular-file-'))
  try {
    const target = join(root, 'target'), linked = join(root, 'linked')
    await writeFile(target, 'keep me')
    await symlink(target, linked)
    const flags = constants.O_WRONLY | constants.O_CREAT | constants.O_TRUNC
    await expect(openRegularFile(linked, flags)).rejects.toThrow('symbolic link')
    expect(() => openRegularFileSync(linked, flags)).toThrow('symbolic link')
    expect(await readFile(target, 'utf8')).toBe('keep me')
    await mkdir(join(root, 'directory'))
    await expect(openRegularFile(join(root, 'directory'), constants.O_RDONLY)).rejects.toThrow('special file')
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('verified file opens create a missing file and truncate only a verified regular file', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-regular-file-'))
  try {
    const path = join(root, 'file')
    await expect(openRegularFile(path, constants.O_RDONLY)).rejects.toMatchObject({ code: 'ENOENT' })
    const file = await openRegularFile(path, constants.O_WRONLY | constants.O_CREAT | constants.O_TRUNC)
    try { await file.writeFile('new bytes') } finally { await file.close() }
    expect(await readFile(path, 'utf8')).toBe('new bytes')
    const fd = openRegularFileSync(path, constants.O_WRONLY | constants.O_TRUNC)
    closeSync(fd)
    expect(readFileSync(path, 'utf8')).toBe('')
  } finally { await rm(root, { recursive: true, force: true }) }
})
