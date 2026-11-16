// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { completionEdit, completionItemsOf, completionPrefix, cursorPosition, filterCompletions, newlineIndent, shouldComplete } from '../src/desktop/renderer/codeCompletion.js'
import { findRuff, ruffFormat } from '../src/lsp/ruffFormat.js'

test('names with digits keep completing, and Enter keeps the line indentation', () => {
  expect(shouldComplete('A1', 2)).toBe(true)
  expect(shouldComplete('x = 12', 6)).toBe(false)
  expect(newlineIndent('\t\t\t\tx = 1', 9, '    ')).toBe('\t\t\t\t')
  expect(newlineIndent('    if ok:', 10, '    ')).toBe('        ')
  expect(newlineIndent('def f():  # note', 16, '    ')).toBe('    ')
  expect(newlineIndent('plain', 5, '    ')).toBe('')
})

test('Ruff formats unsaved text inside the workspace and refuses paths outside it', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-ruff-'))
  try {
    const ruff = await findRuff(root)
    if (ruff) {
      const result = await ruffFormat(root, { filePath: 'a.py', text: 'x=[1,2 ,3]\n' })
      expect(result.text).toBe('x = [1, 2, 3]\n')
    }
    await expect(ruffFormat(root, { filePath: '../escape.py', text: 'x=1\n' })).rejects.toThrow()
    await expect(ruffFormat(root, { filePath: 'a.txt', text: 'x' })).rejects.toThrow('Python')
    await expect(ruffFormat(root, { filePath: 'a.py', text: 'x=1\n' }, undefined, join(root, 'no-ruff'))).rejects.toThrow()
  } finally { await rm(root, { recursive: true, force: true }) }
})
import { completionItems, LspHost, LSP_COMPLETION_LIMIT } from '../src/lsp/host.js'
import type { LspConnectionOptions } from '../src/lsp/connection.js'
import { installPythonLsp, pythonConfiguration } from '../src/lsp/pythonCompletion.js'

test('completion triggers on names, attribute dots and import lines, not in comments or strings', () => {
  const at = (text: string) => shouldComplete(text, text.length)
  expect(at('os.pa')).toBe(true)
  expect(at('np.')).toBe(true)
  expect(at('from easydel import ')).toBe(true)
  expect(at('import ')).toBe(true)
  expect(at('x = 1 ')).toBe(false)
  expect(at('# os.pa')).toBe(false)
  expect(at('s = "os.pa')).toBe(false)
  expect(at('1.')).toBe(false)
})

test('cursor position, prefix, filtering and the applied edit', () => {
  const text = 'import os\nos.pa'
  expect(cursorPosition(text, text.length)).toEqual({ line: 1, character: 5, lineStart: 10 })
  expect(completionPrefix(text, text.length)).toBe('pa')
  const items = [{ label: 'pardir', insertText: 'pardir' }, { label: 'path', insertText: 'path' }, { label: 'sep', insertText: 'sep' }, { label: 'Path', insertText: 'Path' }]
  expect(filterCompletions(items, 'pa').map(item => item.label)).toEqual(['pardir', 'path', 'Path'])
  expect(completionEdit(text, text.length, { label: 'path', insertText: 'path' })).toEqual({ start: 13, end: 15, insert: 'path' })
  expect(completionEdit(text, text.length, { label: 'path', insertText: 'path', replace: { start: 3, end: 5 } })).toEqual({ start: 13, end: 15, insert: 'path' })
})

test('malformed completion replies are dropped, never shown', () => {
  expect(completionItemsOf(null)).toEqual([])
  expect(completionItemsOf([{ label: 'ok', insertText: 'ok', replace: { start: 3, end: 1 } }, { label: 1 }, { insertText: 'x' }])).toEqual([{ label: 'ok', insertText: 'ok' }])
})

test('LSP completion results are normalized, ordered and bounded', () => {
  const raw = { isIncomplete: false, items: Array.from({ length: LSP_COMPLETION_LIMIT + 5 }, (_, index) => ({ label: `n${String(index).padStart(3, '0')}`, kind: 6 })) }
  const result = completionItems(raw, 0)
  expect(result.items).toHaveLength(LSP_COMPLETION_LIMIT)
  expect(result.incomplete).toBe(true)
  const edited = completionItems([{ label: 'path', textEdit: { newText: 'path', range: { start: { line: 2, character: 3 }, end: { line: 2, character: 5 } } }, documentation: { kind: 'markdown', value: 'doc' } }], 2)
  expect(edited.items[0]).toMatchObject({ label: 'path', insertText: 'path', replace: { start: 3, end: 5 }, documentation: 'doc' })
})

test('the host completes on unsaved text and answers workspace/configuration with the venv interpreter', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-lsp-complete-'))
  try {
    await writeFile(join(root, 'a.py'), 'saved on disk\n')
    const notifications: { method: string; params: any }[] = []
    let onRequest: LspConnectionOptions['onRequest']
    const host = await LspHost.start({ cwd: root, command: 'fake', languageId: 'python', configuration: pythonConfiguration('/venv/bin/python'), connectionFactory: options => {
      onRequest = options.onRequest
      return {
        connected: true,
        async request(method: string, params: any) {
          if (method === 'initialize') return { capabilities: { textDocumentSync: 2, completionProvider: { triggerCharacters: ['.'] } } }
          expect(method).toBe('textDocument/completion')
          expect(params.position).toEqual({ line: 1, character: 5 })
          return { items: [{ label: 'path', kind: 9 }] }
        },
        async notify(method: string, params: unknown) { notifications.push({ method, params }) },
        async close() {},
      }
    } })
    const answer = onRequest?.('workspace/configuration', { items: [{ section: 'python' }, { section: 'unknown' }] })
    expect(answer).toEqual({ result: [{ pythonPath: '/venv/bin/python' }, {}] })
    expect(onRequest?.('some/unsupported', {})).toBeUndefined()
    const result = await host.complete({ filePath: 'a.py', text: 'import os\nos.pa', line: 1, character: 5 })
    expect(result.items).toEqual([{ label: 'path', kind: 9, insertText: 'path' }])
    expect(notifications.find(n => n.method === 'textDocument/didOpen')?.params.textDocument.text).toBe('import os\nos.pa')
    await expect(host.complete({ filePath: 'a.py', text: 'x', line: 3, character: 0 })).rejects.toThrow('outside')
    await expect(host.complete({ filePath: '../escape.py', text: 'x', line: 0, character: 0 })).rejects.toThrow()
    await host.close()
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('the Python server install refuses low disk and checksum mismatches without leaving files', async () => {
  const home = await mkdtemp(join(tmpdir(), 'xerxes-pylsp-'))
  try {
    let fetched = 0
    const fetchFake = async () => { fetched++; return new Response('not the pinned archive') }
    await expect(installPythonLsp(home, { platform: 'darwin', arch: 'arm64', freeBytes: async () => 2 * 1024 ** 3, fetch: fetchFake })).rejects.toThrow('less than 3 GB')
    expect(fetched).toBe(0)
    await expect(installPythonLsp(home, { platform: 'darwin', arch: 'arm64', freeBytes: async () => 50 * 1024 ** 3, fetch: fetchFake })).rejects.toThrow('checksum mismatch')
    const left = await Array.fromAsync(new Bun.Glob('**/*').scan({ cwd: join(home, 'lsp', 'python'), onlyFiles: true }))
    expect(left).toEqual([])
    await expect(installPythonLsp(home, { platform: 'plan9', arch: 'mips' })).rejects.toThrow('not available')
  } finally { await rm(home, { recursive: true, force: true }) }
})
