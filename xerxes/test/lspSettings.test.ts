// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm, writeFile, stat, link, symlink } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { LspSettingsStore } from '../src/lsp/settingsStore.js'
const config = { servers: [{ name: 'ts', command: 'configured-server', languageId: 'typescript', extensions: ['.ts'] }] }

test('LSP settings create private files, preserve revision conflicts and validate before writing', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lsp-settings-')); const path = join(root, 'lsp.json'); const store = new LspSettingsStore(path)
  try {
    const first = store.read(); expect(first.servers).toEqual([])
    const saved = store.save(config, first.revision)
    if (process.platform !== 'win32') expect((await stat(path)).mode & 0o777).toBe(0o600)
    expect(store.read().revision).toBe(saved.revision)
    expect(() => store.save({ servers: [] }, first.revision)).toThrow('changed')
    expect(() => store.save({ servers: [{ ...config.servers[0], enabled: 'false' }] }, saved.revision)).toThrow('boolean')
    expect(store.read().servers).toHaveLength(1)
    await writeFile(path, JSON.stringify({ servers: [] }))
    expect(() => store.save(config, saved.revision)).toThrow('changed')
    expect(store.read().servers).toEqual([])
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('LSP settings reject unsafe files and do not steal locks or echo invalid JSON', async () => {
  const root = await mkdtemp(join(tmpdir(), 'lsp-settings-')); const path = join(root, 'lsp.json'); const store = new LspSettingsStore(path)
  try {
    const first = store.read()
    await writeFile(path + '.settings-lock', 'existing owner')
    expect(() => store.save(config, first.revision)).toThrow('lock')
    expect(await Bun.file(path + '.settings-lock').text()).toBe('existing owner')
    await rm(path + '.settings-lock')
    await writeFile(path, 'secret-invalid-json')
    expect(() => store.read()).toThrow('valid UTF-8 JSON')
    await link(path, join(root, 'hardlink'))
    expect(() => store.read()).toThrow('one hard link')
    await rm(path); await symlink(join(root, 'hardlink'), path)
    expect(() => store.read()).toThrow('regular file')
  } finally { await rm(root, { recursive: true, force: true }) }
})
