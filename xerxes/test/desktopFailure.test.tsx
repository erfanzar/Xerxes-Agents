// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { FailedCard, withoutDuplicateFailure } from '../src/desktop/renderer/App.js'
import type { Block } from '../src/desktop/renderer/types.js'

const error = 'Automatic context compaction failed: provider returned 400. Original conversation retained.'
test('compaction failure keeps technical evidence and offers compaction instead of resubmitting work', () => {
  const html = renderToStaticMarkup(createElement(FailedCard, { failed: { error, turn: 109, lastUser: 'Continue the goal' } }))
  expect(html).toContain('Original history preserved')
  expect(html).toContain('Technical details')
  expect(html).toContain(error)
  expect(html).toContain('Retry compaction')
  expect(html).not.toContain('resubmit')
  const auth = renderToStaticMarkup(createElement(FailedCard, { failed: { error: 'Authentication failed', turn: 1, lastUser: 'Hello' } }))
  expect(auth).toContain('Provider settings')
  expect(auth).not.toContain('Retry compaction')
})
test('failure presentation removes only exact duplicates in the latest turn without mutating history', () => {
  const blocks: Block[] = [
    { kind: 'notice', id: 1, error: true, text: error },
    { kind: 'user', id: 2, text: 'Continue' },
    { kind: 'notice', id: 3, error: true, text: error },
    { kind: 'agent', id: 4, streaming: false, text: `[Error: ${error}]` },
    { kind: 'notice', id: 5, error: true, text: 'Different failure' },
    { kind: 'agent', id: 6, streaming: false, text: 'Useful partial answer' },
  ]
  expect(withoutDuplicateFailure(blocks, error).map(block => block.id)).toEqual([1, 2, 5, 6])
  expect(withoutDuplicateFailure(blocks, 'Automatic context compaction failed: new rejection').map(block => block.id)).toEqual([1, 2, 5, 6])
  expect(blocks).toHaveLength(6)
})
