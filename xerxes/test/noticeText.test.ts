// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { noticeText } from '../src/desktop/renderer/blocks.js'

test('a run notice keeps its failure text and drops the TUI-only command', () => {
  // Exactly what the daemon emits for a failed terminal run.
  expect(noticeText('terminal failed: git\n/runs inspect e6bcde6f-af32-4ba8-8821-5f7c2d4931b7'))
    .toBe('terminal failed: git')
  expect(noticeText('terminal succeeded: bun test\n/runs inspect 27ca998c-e290-4cd5-94ea-53cfe55c2d3a'))
    .toBe('terminal succeeded: bun test')
})

test('unrelated notices are untouched', () => {
  expect(noticeText('Command finished')).toBe('Command finished')
  expect(noticeText('multi\nline\nmessage')).toBe('multi\nline\nmessage')
  // A mention of the command inside prose is not the trailing affordance.
  expect(noticeText('try /runs inspect later for details')).toBe('try /runs inspect later for details')
})

test('a notice that is only the command still renders something', () => {
  // Stripping everything would leave a blank row, which reads as a bug.
  expect(noticeText('/runs inspect abc123')).toBe('/runs inspect abc123')
})
