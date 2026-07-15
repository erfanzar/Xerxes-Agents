// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  MEMORY_CONTEXT_SYSTEM_NOTE,
  buildMemoryContextBlock,
  sanitizeMemoryContext,
} from '../src/memory/index.js'

test('memory context sanitation removes forged opening and closing fences case-insensitively', () => {
  expect(sanitizeMemoryContext('hello <memory-context> world </memory-context>')).toBe('hello  world ')
  expect(sanitizeMemoryContext('<MEMORY-CONTEXT>hi</ Memory-Context >')).toBe('hi')
  expect(sanitizeMemoryContext('ordinary recalled data')).toBe('ordinary recalled data')
})

test('memory context block wraps one sanitized recalled-memory boundary', () => {
  const block = buildMemoryContextBlock('<memory-context>Use dark mode.</memory-context>')

  expect(block).toBe([
    '<memory-context>',
    MEMORY_CONTEXT_SYSTEM_NOTE,
    '',
    'Use dark mode.',
    '</memory-context>',
  ].join('\n'))
  expect(block.match(/<memory-context>/gi)).toHaveLength(1)
  expect(block.match(/<\/memory-context>/gi)).toHaveLength(1)
})

test('blank recalled context does not create a prompt block', () => {
  expect(buildMemoryContextBlock('')).toBe('')
  expect(buildMemoryContextBlock(' \n\t ')).toBe('')
})
