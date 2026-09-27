// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { FileToolRow, fileToolOf, lineDiff, readBody } from '../src/desktop/renderer/FileToolRow.js'
import { workspaceFilePath } from '../src/desktop/renderer/DesktopPanels.js'
import type { ToolItem } from '../src/desktop/renderer/types.js'

const item = (name: string, args: Record<string, unknown>, extra: Partial<ToolItem> = {}): ToolItem => ({
  id: 't1', verb: name, arg: '', dur: '0.4s', state: 'done', name, input: JSON.stringify(args), output: '', ...extra,
})

test('read, write and edit calls are recognised from their name and arguments; other tools are not', () => {
  expect(fileToolOf(item('ReadFile', { file_path: 'a.ts', offset: 40 }))).toEqual({ kind: 'read', path: 'a.ts', offset: 40 })
  expect(fileToolOf(item('FileEditTool', { file_path: 'a.ts', old_string: 'x', new_string: 'y' }))).toEqual({ kind: 'edit', path: 'a.ts', before: 'x', after: 'y', wholeFile: false })
  expect(fileToolOf(item('AppendFile', { file_path: 'log.md', content: 'hi' }))).toEqual({ kind: 'write', path: 'log.md', content: 'hi', append: true })
  expect(fileToolOf(item('ExecCommandTool', { command: 'ls' }))).toBeNull()
  expect(fileToolOf(item('ReadFile', {}))).toBeNull()
})

test('an edit diffs line by line: shared lines stay as context, removals come before additions', () => {
  expect(lineDiff('a\nb\nc\n', 'a\nB\nc\nd\n')).toEqual([
    { kind: 'ctx', text: 'a' }, { kind: 'del', text: 'b' }, { kind: 'add', text: 'B' }, { kind: 'ctx', text: 'c' }, { kind: 'add', text: 'd' },
  ])
  expect(lineDiff('', 'new')).toEqual([{ kind: 'add', text: 'new' }])
  expect(readBody('one\ntwo\n\n[ReadFile] Showing lines 1-2 of 90. Continue with offset=2, limit=2.')).toBe('one\ntwo')
})

test('an edit row shows its diff and stats; a failed one shows the reason and no counts', () => {
  const edit = item('FileEditTool', { file_path: 'src/a.ts', old_string: 'const a = 1\n', new_string: 'const a = 2\n' })
  const html = renderToStaticMarkup(createElement(FileToolRow, { item: edit, tool: fileToolOf(edit)!, failed: false }))
  expect(html).toContain('>Edit<')
  expect(html).toContain('src/a.ts')
  expect(html).toContain('filetool__line--del')
  expect(html).toContain('>+1<')
  const failed = { ...edit, state: 'failed' as const, error: 'Tool execution failed: FileEditTool requires reading "src/a.ts" first' }
  const bad = renderToStaticMarkup(createElement(FileToolRow, { item: failed, tool: fileToolOf(failed)!, failed: true }))
  expect(bad).toContain('FileEditTool requires reading')
  expect(bad).not.toContain('filetool__stats')
  expect(bad).not.toContain('filetool__line')
})

test('a tool path opens in Files as the tree names it', () => {
  expect(workspaceFilePath('/repo/src/a.ts', '/repo')).toBe('./src/a.ts')
  expect(workspaceFilePath('src/a.ts', '/repo/')).toBe('./src/a.ts')
  expect(workspaceFilePath('./x.md', '/repo')).toBe('./x.md')
})

test('README HTML becomes Markdown the renderer can draw; code fences are left alone', async () => {
  const { htmlToMarkdown } = await import('../src/desktop/renderer/DesktopPanels.js')
  const readme = '<!-- lint off --><div align="center"><a href="https://x.dev/"><img src="https://x.dev/logo.png" height="80"></a></div>\n<h2>Install</h2>\n```html\n<div>kept</div>\n```\nA&nbsp;b'
  const out = htmlToMarkdown(readme)
  expect(out).not.toContain('<!--')
  expect(out).not.toContain('<div align')
  expect(out).toContain('[logo.png](https://x.dev/logo.png)')
  expect(out).toContain('## Install')
  expect(out).toContain('<div>kept</div>')
  expect(out).toContain('A b')
})

test('the editor tokenizes by language and keeps every character', async () => {
  const { tokenize, syntaxFor } = await import('../src/desktop/renderer/syntax.js')
  const { indentUnit } = await import('../src/desktop/renderer/CodeEditor.js')
  const line = 'const a = "x // y" // note'
  const tokens = tokenize(line, syntaxFor('a.ts'))
  expect(tokens.map(token => token.text).join('')).toBe(line)
  expect(tokens.find(token => token.kind === 'string')?.text).toBe('"x // y"')
  expect(tokens.at(-1)).toEqual({ kind: 'comment', text: '// note' })
  expect(tokenize('def f(): return 42  # done', syntaxFor('m.py')).map(token => token.kind)).toContain('number')
  expect(indentUnit('a:\n    b\n', 'x.py')).toBe('    ')
  expect(indentUnit('{\n\tx\n}', 'x.go')).toBe('\t')
})
