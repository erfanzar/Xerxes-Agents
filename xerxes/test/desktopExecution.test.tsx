// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { ExecutionDetails, executionView } from '../src/desktop/renderer/Execution.js'
import type { ToolItem } from '../src/desktop/renderer/types.js'

const item: ToolItem = { id: 'call-1', name: 'exec_command', verb: 'exec_command', arg: 'sed', dur: '0.0s', state: 'done', input: JSON.stringify({ cmd: 'sed', args: ['-n', '10,20p', 'path with spaces/file.ts'] }), output: JSON.stringify({ stdout: 'function run() {\n  return 1\n}\n', stderr: '', exitCode: 0, cwd: '/repo' }) }
test('execution decodes output line breaks and safely displays argument boundaries', () => {
  const view = executionView(item)
  expect(view.command).toBe("sed -n 10,20p 'path with spaces/file.ts'")
  expect(view.stdout).toBe('function run() {\n  return 1\n}\n')
  const html = renderToStaticMarkup(createElement(ExecutionDetails, { item }))
  expect(html).toContain('aria-label="Command output">function run() {\n  return 1\n}')
  expect(html).toContain('<summary>Raw details</summary>')
  expect(html).toContain('Copy command')
})
test('nonzero results expose failure and stderr even without a transport error', () => {
  const html = renderToStaticMarkup(createElement(ExecutionDetails, { item: { ...item, output: JSON.stringify({ stdout: '', stderr: 'Permission denied\n', exitCode: 2 }) } }))
  expect(html).toContain('Failed · Exit 2')
  expect(html).toContain('Permission denied\n')
})
test('plain output and malformed payloads remain visible without invented success data', () => {
  const html = renderToStaticMarkup(createElement(ExecutionDetails, { item: { ...item, output: 'partial { output', error: 'Cancelled by user', state: 'failed' } }))
  expect(html).toContain('partial { output')
  expect(html).toContain('Cancelled by user')
  expect(html).not.toContain('Exit 0')
})

test('activity grouping preserves prose and keeps approval operations outside disclosures', async () => {
  const { groupActivity } = await import('../src/desktop/renderer/activityGroups.js')
  const blocks = [{ kind: 'thinking' as const, id: 1, text: 'Inspect', streaming: false }, { kind: 'tools' as const, id: 2, items: [item], running: false }, { kind: 'agent' as const, id: 3, text: 'Result', streaming: false }]
  expect(groupActivity(blocks).map(group => group.length)).toEqual([2, 1])
  expect(groupActivity(blocks, item.id).map(group => group.length)).toEqual([1, 1, 1])
  expect(groupActivity(blocks).flat()).toEqual(blocks)
})
