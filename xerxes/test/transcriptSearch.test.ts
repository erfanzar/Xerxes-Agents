// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { TranscriptSearchIndex, extractSearchableText } from '../src/session/transcriptSearch.js'

test('search spans sessions, requires every term, and answers newest first', () => {
  const index = new TranscriptSearchIndex()
  index.index({
    sessionId: 'aaaaaaaa',
    updatedAt: '2026-07-01T00:00:00.000Z',
    title: 'older work',
    messages: [
      { role: 'user', content: 'please fix the retry backoff in the daemon' },
      { role: 'assistant', content: 'the backoff is exponential' },
    ],
  })
  index.index({
    sessionId: 'bbbbbbbb',
    updatedAt: '2026-07-20T00:00:00.000Z',
    title: 'newer work',
    messages: [
      { role: 'user', content: 'the retry backoff regressed again' },
      { role: 'assistant', content: 'unrelated answer about colors' },
    ],
  })

  const hits = index.search('retry backoff')

  expect(hits.map(hit => [hit.sessionId, hit.messageIndex])).toEqual([
    ['bbbbbbbb', 0],
    ['aaaaaaaa', 0],
  ])
  expect(hits[0]?.role).toBe('user')
  expect(hits[0]?.title).toBe('newer work')
  expect(hits[0]?.excerpt).toContain('retry backoff regressed')
  // Every term must be present: "backoff" alone reaches the second message too.
  expect(index.search('backoff').length).toBe(3)
  expect(index.search('backoff nonexistentterm')).toEqual([])
  expect(index.search('   ')).toEqual([])
})

test('a scoped or capped search stays within its bounds', () => {
  const index = new TranscriptSearchIndex()
  index.index({
    sessionId: 'aaaaaaaa',
    updatedAt: '2026-07-01T00:00:00.000Z',
    messages: Array.from({ length: 12 }, () => ({ role: 'user', content: 'marker text' })),
  })
  index.index({
    sessionId: 'bbbbbbbb',
    updatedAt: '2026-07-02T00:00:00.000Z',
    messages: [{ role: 'user', content: 'marker text' }],
  })

  // One noisy session must not crowd every other session out of the answer.
  expect(index.search('marker').map(hit => hit.sessionId)).toEqual([
    'bbbbbbbb',
    'aaaaaaaa',
    'aaaaaaaa',
    'aaaaaaaa',
    'aaaaaaaa',
    'aaaaaaaa',
  ])
  expect(index.search('marker', { limit: 2 })).toHaveLength(2)
  expect(index.search('marker', { sessionId: 'bbbbbbbb', perSession: 20 })).toHaveLength(1)
})

test('an unrecognized row indexes as empty and is counted rather than serialized', () => {
  const index = new TranscriptSearchIndex()
  index.index({
    sessionId: 'aaaaaaaa',
    updatedAt: '2026-07-01T00:00:00.000Z',
    messages: [
      { role: 'user', content: 'plain text row' },
      // Content shapes this indexer does not model.
      { role: 'user', content: 42 },
      { role: 'tool', content: { unexpected: 'record shape' } },
      'not a message at all',
    ],
  })

  expect(index.stats()).toEqual({
    indexedMessages: 4,
    searchableMessages: 1,
    sessions: 1,
    truncatedMessages: 0,
    unrecognizedMessages: 3,
  })
  // A serialized fallback would match the field names and highlight nothing.
  expect(index.search('unexpected')).toEqual([])
  expect(index.search('role')).toEqual([])
  expect(index.search('plain text')).toHaveLength(1)
})

test('typed blocks without text are understood; shapeless blocks are not', () => {
  expect(extractSearchableText({ role: 'assistant', content: 'hello' })).toEqual({
    recognized: true,
    text: 'hello',
  })
  expect(
    extractSearchableText({
      role: 'assistant',
      content: [
        { type: 'text', text: 'visible answer' },
        { type: 'tool_use', name: 'ReadFile', input: { path: 'a.txt' } },
        'raw string part',
      ],
    }),
  ).toEqual({ recognized: true, text: 'visible answer\nraw string part' })
  expect(extractSearchableText({ role: 'assistant', content: [{ noTypeNoText: true }] })).toEqual({
    recognized: false,
    text: '',
  })
  // A tool-call-only assistant message legitimately carries no text.
  expect(extractSearchableText({ role: 'assistant', tool_calls: [] })).toEqual({
    recognized: true,
    text: '',
  })
  expect(extractSearchableText({ role: 'user', text: 'display text' })).toEqual({
    recognized: true,
    text: 'display text',
  })
  expect(extractSearchableText({ role: 'user', text: 99 })).toEqual({ recognized: false, text: '' })
})

test('re-indexing replaces a session and removal drops it entirely', () => {
  const index = new TranscriptSearchIndex()
  index.index({
    sessionId: 'aaaaaaaa',
    updatedAt: '2026-07-01T00:00:00.000Z',
    messages: [{ role: 'user', content: 'first revision' }],
  })
  index.index({
    sessionId: 'aaaaaaaa',
    updatedAt: '2026-07-02T00:00:00.000Z',
    messages: [{ role: 'user', content: 'second revision' }],
  })

  expect(index.search('first')).toEqual([])
  expect(index.search('second')).toHaveLength(1)
  expect(index.has('aaaaaaaa')).toBe(true)
  expect(index.remove('aaaaaaaa')).toBe(true)
  expect(index.remove('aaaaaaaa')).toBe(false)
  expect(index.stats().sessions).toBe(0)
})

test('an over-long message is truncated visibly rather than silently', () => {
  const index = new TranscriptSearchIndex()
  index.index({
    sessionId: 'aaaaaaaa',
    updatedAt: '2026-07-01T00:00:00.000Z',
    messages: [{ role: 'user', content: `${'a '.repeat(9_000)}needleatend` }],
  })

  expect(index.stats().truncatedMessages).toBe(1)
  expect(index.search('needleatend')).toEqual([])
})
