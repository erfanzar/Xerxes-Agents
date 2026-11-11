// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, rm, symlink } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { formatBytes, previewOptionsOf, previewWorkspaceFile, previewWorkspaceFileRich, writeWorkspaceFile } from '../src/daemon/filePreview.js'
import { documentFixture, workbookFixture } from './fixtures/officePreview.js'

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

const ALL_KINDS = { kinds: new Set(['table', 'image', 'pdf', 'sheet', 'document', 'html'] as const), sheet: 0 }
const PNG = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0x0d, 0x49, 0x48, 0x44, 0x52])

test('rich previews parse each kind the client renders and say why others cannot be shown', async () => {
  const workspace = await mkdtemp(join(tmpdir(), 'file-preview-rich-'))
  try {
    await Bun.write(join(workspace, 'discrepancies.csv'), 'id,amount\n1,"1,200"\n2,7\n')
    expect(await previewWorkspaceFileRich(workspace, 'discrepancies.csv', ALL_KINDS)).toMatchObject({
      ok: true, kind: 'table', path: 'discrepancies.csv', delimiter: ',', encoding: 'utf-8',
      header: ['id', 'amount'], rows: [['1', '1,200'], ['2', '7']], total_rows: 2, total_columns: 2, truncated: false,
    })
    await Bun.write(join(workspace, 'book.xlsx'), workbookFixture())
    const book = await previewWorkspaceFileRich(workspace, 'book.xlsx', { ...ALL_KINDS, sheet: 1 })
    expect(book).toMatchObject({ ok: true, kind: 'sheet', sheet: 1, sheets: [{ name: 'Data' }, { name: 'Secret & Co', hidden: true }, { name: 'Chart' }], total_rows: 2 })
    await Bun.write(join(workspace, 'report.docx'), documentFixture())
    expect(await previewWorkspaceFileRich(workspace, 'report.docx', ALL_KINDS)).toMatchObject({ ok: true, kind: 'document', blocks: expect.arrayContaining([{ type: 'heading', level: 1, text: 'Summary' }]) })
    await Bun.write(join(workspace, 'chart.png'), PNG)
    expect(await previewWorkspaceFileRich(workspace, 'chart.png', ALL_KINDS)).toMatchObject({ ok: true, kind: 'image', media_type: 'image/png', data: Buffer.from(PNG).toString('base64'), size: PNG.length })
    await Bun.write(join(workspace, 'logo.svg'), '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>')
    expect(await previewWorkspaceFileRich(workspace, 'logo.svg', ALL_KINDS)).toMatchObject({ ok: true, kind: 'image', media_type: 'image/svg+xml' })
    await Bun.write(join(workspace, 'paper.pdf'), '%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n')
    expect(await previewWorkspaceFileRich(workspace, 'paper.pdf', ALL_KINDS)).toMatchObject({ ok: true, kind: 'pdf', media_type: 'application/pdf' })
    await Bun.write(join(workspace, 'page.html'), '<h1>Hi</h1><script>alert(1)</script>')
    const page = await previewWorkspaceFileRich(workspace, 'page.html', ALL_KINDS)
    expect(page).toMatchObject({ ok: true, kind: 'html', content: '<h1>Hi</h1><script>alert(1)</script>', truncated: false })
    expect('version' in page && page.version).toMatch(/^[0-9a-f]{64}$/)
    // Plain text still comes back as text, now with its kind and size.
    await Bun.write(join(workspace, 'notes.md'), '# Notes\n')
    expect(await previewWorkspaceFileRich(workspace, 'notes.md', ALL_KINDS)).toMatchObject({ ok: true, kind: 'text', content: '# Notes\n', size: 8 })
  } finally { await rm(workspace, { recursive: true, force: true }) }
})

test('a kind the client leaves out comes back as text (the Source view), and the legacy reply is unchanged', async () => {
  const workspace = await mkdtemp(join(tmpdir(), 'file-preview-kinds-'))
  try {
    await Bun.write(join(workspace, 'data.csv'), 'a,b\n1,2\n')
    const source = await previewWorkspaceFileRich(workspace, 'data.csv', { kinds: new Set(['image']), sheet: 0 })
    expect(source).toMatchObject({ ok: true, kind: 'text', content: 'a,b\n1,2\n', version: expect.any(String) })
    const legacy = await previewWorkspaceFile(workspace, 'data.csv')
    expect(legacy).toEqual({ ok: true, path: 'data.csv', content: 'a,b\n1,2\n', truncated: false, version: expect.any(String) })
    expect('kind' in legacy).toBe(false)
  } finally { await rm(workspace, { recursive: true, force: true }) }
})

test('too large, unsupported and unreadable files are typed replies; paths outside the workspace still throw', async () => {
  const root = await mkdtemp(join(tmpdir(), 'file-preview-limits-'))
  const workspace = join(root, 'workspace')
  try {
    await mkdir(workspace)
    await Bun.write(join(workspace, 'scan.pdf'), new Uint8Array(10 * 1024 * 1024 + 1))
    expect(await previewWorkspaceFileRich(workspace, 'scan.pdf', ALL_KINDS)).toEqual({
      ok: false, path: 'scan.pdf', reason: 'too_large', size: 10 * 1024 * 1024 + 1, limit: 10 * 1024 * 1024,
      error: 'This file is 10 MB. Previews of PDFs are limited to 10 MB.',
    })
    await Bun.write(join(workspace, 'archive.zip'), new Uint8Array([0x50, 0x4b, 3, 4, 0, 0, 0xff, 0xfe]))
    expect(await previewWorkspaceFileRich(workspace, 'archive.zip', ALL_KINDS)).toMatchObject({ ok: false, reason: 'unsupported' })
    // The bytes decide: a .png holding HTML is not shown as an image.
    await Bun.write(join(workspace, 'fake.png'), '<html><script>alert(1)</script></html>')
    expect(await previewWorkspaceFileRich(workspace, 'fake.png', ALL_KINDS)).toMatchObject({ ok: false, reason: 'unreadable', error: expect.stringContaining('not a PNG') })
    await Bun.write(join(workspace, 'fake.pdf'), 'hello')
    expect(await previewWorkspaceFileRich(workspace, 'fake.pdf', ALL_KINDS)).toMatchObject({ ok: false, reason: 'unreadable' })
    await Bun.write(join(workspace, 'broken.xlsx'), 'not a zip at all')
    expect(await previewWorkspaceFileRich(workspace, 'broken.xlsx', ALL_KINDS)).toMatchObject({ ok: false, reason: 'unreadable', error: expect.stringContaining('workbook could not be read') })
    // Encrypted OOXML and legacy .doc/.xls are OLE compound files.
    await Bun.write(join(workspace, 'locked.docx'), new Uint8Array([0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1]))
    expect(await previewWorkspaceFileRich(workspace, 'locked.docx', ALL_KINDS)).toMatchObject({ ok: false, reason: 'unreadable', error: expect.stringContaining('password-protected') })
    await Bun.write(join(root, 'outside.csv'), 'secret\n')
    await symlink(join(root, 'outside.csv'), join(workspace, 'link.csv'))
    await expect(previewWorkspaceFileRich(workspace, 'link.csv', ALL_KINDS)).rejects.toThrow('selected workspace')
    await expect(previewWorkspaceFileRich(workspace, '../outside.csv', ALL_KINDS)).rejects.toThrow('selected workspace')
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('preview options are validated at the boundary; unknown kind names from newer clients are ignored', () => {
  expect(previewOptionsOf({ path: 'a' })).toBeUndefined()
  expect(previewOptionsOf({ kinds: ['table', 'hologram'], sheet: 2 })).toEqual({ kinds: new Set(['table']), sheet: 2 })
  expect(() => previewOptionsOf({ kinds: 'table' })).toThrow('kinds')
  expect(() => previewOptionsOf({ kinds: Array(17).fill('table') })).toThrow('kinds')
  expect(() => previewOptionsOf({ kinds: [1] })).toThrow('kinds')
  expect(() => previewOptionsOf({ kinds: [], sheet: -1 })).toThrow('sheet')
  expect(() => previewOptionsOf({ kinds: [], sheet: 1.5 })).toThrow('sheet')
  expect(formatBytes(512)).toBe('512 B')
  expect(formatBytes(48.2 * 1024 * 1024)).toBe('48 MB')
  expect(formatBytes(1.5 * 1024)).toBe('1.5 KB')
})
