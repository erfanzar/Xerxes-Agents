// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, test } from 'bun:test'
import { lspChanges, readLspView } from '../src/desktop/renderer/LspPanel.js'
import { prepareLspSettingsEdit } from '../src/lsp/settings.js'

const saved = { revision: 'v1', servers: [{ name: 'ts', enabled: true, languageId: 'typescript', extensions: ['.ts'], command: 'typescript-language-server', args: ['--stdio'], env: { TOKEN: 'private' }, timeoutMs: 30000 }] }
function form(): FormData {
  const value = new FormData()
  for (const [key, entry] of Object.entries({ enabled: 'on', languageId: 'typescript', extensions: '.ts, .tsx', timeoutMs: '5000', command: '', args: '', env: '' })) value.set(key, entry)
  return value
}
describe('desktop LSP settings', () => {
  test('editing visible fields preserves masked launch configuration', () => {
    const next = prepareLspSettingsEdit(saved, { revision: 'v1', name: 'ts', action: 'update', changes: lspChanges(form()) })
    expect(next[0]).toMatchObject({ command: 'typescript-language-server', args: ['--stdio'], env: { TOKEN: 'private' }, extensions: ['.ts', '.tsx'], timeoutMs: 5000 })
  })
  test('supports creation and explicit clearing with daemon validation', () => {
    const data = form(); data.set('extensions', '.jsx'); data.set('command', 'new-server'); data.set('args', '["--stdio"]'); data.set('env', '{"MODE":"test"}')
    const next = prepareLspSettingsEdit(saved, { revision: 'v1', name: 'new', action: 'create', changes: lspChanges(data) })
    expect(next[1]?.command).toBe('new-server')
    data.set('args', 'null'); data.set('env', 'null')
    const cleared = prepareLspSettingsEdit(saved, { revision: 'v1', name: 'ts', action: 'update', changes: lspChanges(data) })
    expect(cleared[0]?.env).toBeUndefined(); expect(cleared[0]?.args).toBeUndefined()
  })
  test('rejects malformed JSON and incompatible field types', () => {
    for (const [key, raw] of [['args', '{'], ['args', '{}'], ['args', '[1]'], ['env', '[]'], ['env', '{"a":1}']]) {
      const data = form(); data.set(key!, raw!); expect(() => lspChanges(data)).toThrow()
    }
  })
  test('rejects stale revisions instead of overwriting another editor', () => {
    expect(() => prepareLspSettingsEdit(saved, { revision: 'old', name: 'ts', action: 'update', changes: lspChanges(form()) })).toThrow('reload')
  })
  test('validates response shape and discards unexpected private fields', () => {
    const view = readLspView({ ...saved, warnings: [] })
    expect(view.servers[0]).not.toHaveProperty('env')
    expect(view.servers[0]).not.toHaveProperty('command')
    expect(() => readLspView({ revision: 'v1', servers: [], warnings: 'wrong' })).toThrow()
    expect(() => readLspView({ revision: 'v1', servers: [{ name: 'ts' }], warnings: [] })).toThrow()
  })
})
