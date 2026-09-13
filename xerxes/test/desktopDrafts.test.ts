// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { draftKey, readDraft, writeDraft } from '../src/desktop/renderer/drafts.js'

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
