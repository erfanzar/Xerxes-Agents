// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { draftKey, readDraft, writeDraft, transitionDraft, acceptedDraft } from '../src/desktop/renderer/drafts.js'

test('accepted text clears only its originating draft and preserves edits made while sending', () => {
  const a = { key: draftKey('/acceptance', 'a'), workspace: '/acceptance', sessionId: 'a' }
  const b = { key: draftKey('/acceptance', 'b'), workspace: '/acceptance', sessionId: 'b' }
  const sent = 'Draft\n  with indentation'
  writeDraft(a.key, sent)
  expect(acceptedDraft(a, a, sent, sent)).toBe('')
  expect(readDraft(a.key)).toBe('')
  writeDraft(a.key, sent + '\nNew thought')
  expect(acceptedDraft(a, a, sent, sent + '\nNew thought')).toBe(sent + '\nNew thought')
  expect(readDraft(a.key)).toBe(sent + '\nNew thought')
  writeDraft(a.key, sent)
  writeDraft(b.key, 'Another task')
  expect(acceptedDraft(a, b, sent, 'Another task')).toBe('Another task')
  expect(readDraft(b.key)).toBe('Another task')
  expect(readDraft(a.key)).toBe('')
})

test('draft storage preserves whitespace, isolates workspaces and clears submitted drafts', () => {
  const saved = new Map<string, string>()
  const storage = { getItem: (key: string) => saved.get(key) ?? null, setItem: (key: string, value: string) => { saved.set(key, value) }, removeItem: (key: string) => { saved.delete(key) } }
  const a = draftKey('/project-a', 'session'), b = draftKey('/project-b', 'session')
  writeDraft(a, 'Draft\n  indentation\n', storage)
  expect(saved.get(a)).toBe('Draft\n  indentation\n')
  expect(readDraft(a, storage)).toBe('Draft\n  indentation\n')
  expect(readDraft(b, storage)).toBe('')
  writeDraft(a, '', storage)
  expect(saved.has(a)).toBe(false)
  expect(readDraft(a, storage)).toBe('')
})

test('unavailable storage retains in-renderer drafts and allows clearing them', () => {
  const key = draftKey('/unavailable', 'quota')
  const storage = { getItem: () => { throw Error('blocked') }, setItem: () => { throw Error('full') }, removeItem: () => { throw Error('blocked') } }
  writeDraft(key, 'still here', storage)
  expect(readDraft(key, storage)).toBe('still here')
  writeDraft(key, '', storage)
  expect(readDraft(key, storage)).toBe('')
})

test('typing before session creation moves the draft into that conversation only', async () => {
  const previous = { key: draftKey('/opening', 'pending'), workspace: '/opening', sessionId: '' }
  const next = { key: draftKey('/opening', 'created'), workspace: '/opening', sessionId: 'created' }
  expect(transitionDraft(previous, next, 'Keep this\n  draft')).toBe('Keep this\n  draft')
  expect(readDraft(next.key)).toBe('Keep this\n  draft')
  expect(readDraft(previous.key)).toBe('')
  const other = { key: draftKey('/other', 'created'), workspace: '/other', sessionId: 'created' }
  expect(transitionDraft(next, other, 'Keep this\n  draft')).toBe('')
  expect(readDraft(next.key)).toBe('Keep this\n  draft')
})
test('provisional drafts never overwrite an existing resumed draft or cross endpoints', async () => {
  const previous = { key: draftKey('/resume', 'pending'), workspace: '/resume', sessionId: '' }
  const next = { key: draftKey('/resume', 'saved'), workspace: '/resume', sessionId: 'saved' }
  writeDraft(next.key, 'Existing draft')
  expect(transitionDraft(previous, next, 'New draft')).toBe('Existing draft')
  expect(readDraft(previous.key)).toBe('New draft')
  const remote = { key: draftKey('ssh:worker:/resume', 'saved'), workspace: 'ssh:worker:/resume', sessionId: 'saved' }
  expect(transitionDraft(previous, remote, 'New draft')).toBe('')
})
