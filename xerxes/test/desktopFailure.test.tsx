// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { FailedCard, withoutDuplicateFailure } from '../src/desktop/renderer/App.js'
import type { Block } from '../src/desktop/renderer/types.js'
import { failureView, resetTimeOf } from '../src/desktop/renderer/turnFailure.js'

const error = 'Automatic context compaction failed: provider returned 400. Original conversation retained.'
test('compaction failure keeps technical evidence and offers compaction instead of resubmitting work', () => {
  const html = renderToStaticMarkup(createElement(FailedCard, { failed: { error, turn: 109, lastUser: 'Continue the goal' } }))
  expect(html).toContain('Original history preserved')
  expect(html).toContain('What the provider said')
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

/**
 * A usage limit is the most common recoverable provider failure and it
 * used to render as the provider's raw JSON with a Retry that could only
 * fail again. It has to name the cause and offer the action that works.
 */
test('a provider usage limit is explained and routed to a model switch', () => {
  const error = 'Client openai-codex: Responses API stream request failed (429): {"error":{"type":"usage_limit_reached","message":"The usage limit has been reached","plan_type":"pro","resets_at":1790054678}}'
  const view = failureView(error, 1790050000000)
  expect(view.kind).toBe('rate-limit')
  expect(view.retryIsFutile).toBe(true)
  expect(view.offerProviderSettings).toBe(true)
  expect(view.summary).toContain('out of capacity')
  // The unix stamp becomes a wall-clock time the reader can act on.
  expect(view.summary).toMatch(/resets around \d/)

  const html = renderToStaticMarkup(createElement(FailedCard, { failed: { error, turn: 3, lastUser: 'keep going' } }))
  expect(html).toContain('Switch model or provider')
  expect(html).not.toContain('usage_limit_reached</div>')
  // The provider's own words stay one disclosure away, never discarded.
  expect(html).toContain('What the provider said')
  expect(html).toContain('usage_limit_reached')
})

test('failure classes that retry cannot fix are separated from ones that can', () => {
  const at = 1790050000000
  expect(failureView('context_length_exceeded: maximum context is 200000 tokens', at).kind).toBe('context-length')
  expect(failureView('Authentication failed: invalid credential', at).kind).toBe('credentials')
  expect(failureView('Automatic context compaction failed: model refused', at).kind).toBe('compaction')
  // An unclassified failure keeps the raw text and a plain retry.
  const unknown = failureView('socket hang up', at)
  expect(unknown.kind).toBe('unknown')
  expect(unknown.summary).toBe('')
  expect(unknown.retryIsFutile).toBe(false)
  expect(renderToStaticMarkup(createElement(FailedCard, { failed: { error: 'socket hang up', turn: 1, lastUser: 'go' } }))).toContain('socket hang up')
})

test('a reset timestamp outside a plausible window is not invented', () => {
  expect(resetTimeOf('{"resets_at":1}', 1790050000000)).toBe('')
  expect(resetTimeOf('no stamp here', 1790050000000)).toBe('')
  expect(resetTimeOf('{"resets_at":1790054678}', 1790050000000)).toMatch(/\d/)
})
