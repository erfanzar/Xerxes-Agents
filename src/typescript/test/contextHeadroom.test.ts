// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  HeadroomResult,
  compressToolResult,
} from '../src/context/headroom.js'
import {
  estimateContextTokens,
  estimateRequestOverheadTokens,
  requestScaffoldingMessages,
} from '../src/context/windowUsage.js'

test('headroom compresses large JSON while retaining error-bearing samples within its conservative cap', () => {
  const payload: Array<Record<string, unknown>> = Array.from({ length: 50 }, (_, index) => ({
    id: index,
    status: 'ok',
    payload: 'x'.repeat(120),
  }))
  payload.push({ id: 99, error: 'boom', traceback: 'frame\n'.repeat(80) })

  const result = compressToolResult('JSONProcessor', JSON.stringify(payload), { maxChars: 900 })
  expect(result).toBeInstanceOf(HeadroomResult)
  expect(result).toMatchObject({ contentType: 'json' })
  expect(result.originalChars).toBeGreaterThan(result.compressedChars)
  expect(result.compressed).toContain('boom')
  expect(result.compressed).toContain('omitted')
  expect(result.compressed.length).toBeLessThanOrEqual(900)
  expect(result.metadataLine()).toContain('json preview:')
})

test('headroom routes unified diffs, search matches, logs, and binary-like content to specialized previews', () => {
  const diffLines: string[] = []
  for (let file = 0; file < 14; file += 1) {
    diffLines.push('diff --git a/src/file_' + file + '.ts b/src/file_' + file + '.ts')
    diffLines.push('--- a/src/file_' + file + '.ts', '+++ b/src/file_' + file + '.ts')
    for (let hunk = 0; hunk < 8; hunk += 1) {
      diffLines.push(
        '@@ -' + hunk * 10 + ',6 +' + hunk * 10 + ',6 @@',
        ' context before',
        '-old value ' + file + '-' + hunk,
        '+new value ' + file + '-' + hunk,
        ' context after',
      )
    }
  }
  const diff = compressToolResult('exec_command', diffLines.join('\n'), { maxChars: 1_800 })
  expect(diff).toMatchObject({ contentType: 'diff' })
  expect(diff.compressed).toContain('Xerxes headroom diff preview')
  expect(diff.compressed).toContain('diff --git a/src/file_0.ts b/src/file_0.ts')
  expect(diff.compressed).toContain('hunks omitted')
  expect(diff.compressed.length).toBeLessThanOrEqual(1_800)

  const searchLines = Array.from({ length: 5 }, (_, file) => Array.from(
    { length: 12 },
    (_, line) => 'src/module_' + file + '.ts:' + line + ':export const value = ' + line,
  )).flat()
  const search = compressToolResult('GrepTool', searchLines.join('\n'), { maxChars: 1_200 })
  expect(search).toMatchObject({ contentType: 'search' })
  expect(search.compressed).toContain('Xerxes headroom search preview')
  expect(search.compressed).toContain('more matches in this file')

  const logLines = Array.from({ length: 120 }, (_, index) => 'INFO build line ' + index)
  logLines[70] = 'ERROR tests/test_runtime.ts::test_case failed'
  logLines[71] = 'Traceback (most recent call last):'
  logLines.push('FAILED tests/test_runtime.ts::test_case - AssertionError', '1 failed, 200 passed in 12.34s')
  const log = compressToolResult('exec_command', logLines.join('\n'), { maxChars: 1_100 })
  expect(log).toMatchObject({ contentType: 'log' })
  expect(log.compressed).toContain('ERROR tests/test_runtime.ts::test_case failed')
  expect(log.compressed).toContain('1 failed, 200 passed')
  expect(log.compressed).toContain('log lines omitted')

  const binary = compressToolResult('read_binary', 'a\u0000b', { maxChars: 600 })
  expect(binary).toMatchObject({ contentType: 'binary' })
  expect(binary.compressed).toContain('Binary-like')
})

test('window usage counts provider scaffolding separately and includes it in live request estimates', () => {
  const schemas = [{
    name: 'BigTool',
    description: 'large schema '.repeat(100),
    input_schema: { type: 'object', properties: { value: { type: 'string' } } },
  }]
  const messages = [{ role: 'user', content: 'short prompt' }]
  const scaffolding = requestScaffoldingMessages({
    systemPrompt: 'system prompt '.repeat(20),
    toolSchemas: schemas,
  })
  expect(scaffolding).toHaveLength(2)
  expect(scaffolding[1]?.content).toContain('[available tool schemas]')

  const overhead = estimateRequestOverheadTokens({
    model: 'gpt-4o',
    systemPrompt: 'system prompt '.repeat(20),
    toolSchemas: schemas,
  })
  const messageOnly = estimateContextTokens(messages, { model: 'gpt-4o' })
  const full = estimateContextTokens(messages, {
    model: 'gpt-4o',
    systemPrompt: 'system prompt '.repeat(20),
    toolSchemas: schemas,
  })
  expect(overhead).toBeGreaterThan(0)
  expect(full).toBeGreaterThan(messageOnly)
  expect(estimateContextTokens([], { model: 'gpt-4o' })).toBe(0)
})
