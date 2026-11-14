// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { highlightLines } from '../src/desktop/renderer/codeHighlight.js'
import { grammarFor } from '../src/desktop/renderer/syntax.js'

const join = (lines: ReturnType<typeof highlightLines>) => lines!.map(line => line.map(span => span.text).join(''))

test('Python: keywords, function names and a multi-line docstring are classed on every line', () => {
  const code = 'def run(x):\n    """Doc line one.\n\n    Doc line three.\n    """\n    return None\n'
  const lines = highlightLines(code, grammarFor('pkg/api.py'))
  expect(lines).not.toBeNull()
  expect(join(lines)).toEqual(code.split('\n'))
  expect(lines![0]!.find(span => span.text === 'def')?.className).toContain('hljs-keyword')
  expect(lines![0]!.find(span => span.text === 'run')?.className).toContain('hljs-title')
  // The docstring's middle line keeps the string class, not just its opening line.
  expect(lines![3]!.every(span => span.className?.includes('hljs-string'))).toBe(true)
})

test('TypeScript: export/const are keywords; text always round-trips', () => {
  const code = 'export function parse(value: unknown): number {\n  const n = 1 // one\n  return n\n}'
  const lines = highlightLines(code, grammarFor('src/protocol.ts'))
  expect(join(lines)).toEqual(code.split('\n'))
  expect(lines![0]!.find(span => span.text === 'export')?.className).toContain('hljs-keyword')
  expect(lines![1]!.find(span => span.text.includes('// one'))?.className).toContain('hljs-comment')
})

test('unknown grammar or oversized input falls back to null', () => {
  expect(highlightLines('x', grammarFor('notes.unknownext'))).toBeNull()
  expect(highlightLines('a'.repeat(200_000), 'python')).toBeNull()
})
