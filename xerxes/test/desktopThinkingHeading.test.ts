// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { liveActivityPhrase, thinkingHeadingOf } from '../src/desktop/renderer/activityPhrase.js'
import type { Block } from '../src/desktop/renderer/types.js'

const reasoning = [
  "Okay, I've got the core data from the paper read and checked.",
  '',
  '**Analyzing Technical Details**',
  '',
  "I've just finished grounding the technical report into our artifact.",
  '',
  '**Reviewing Algorithm Enhancements**',
  '',
  "I'm now diving deep into enhancements for GRPO, with **token-level** loss.",
].join('\n')

test('the newest bold heading in the reasoning is what the model is working on', () => {
  // `**token-level**` inside a sentence is bold too, but only a wholly bold line is a heading.
  expect(thinkingHeadingOf('**Analyzing Technical Details**\n\nsome text')).toBe('Analyzing Technical Details')
  expect(thinkingHeadingOf(reasoning)).toBe('Reviewing Algorithm Enhancements')
  expect(thinkingHeadingOf('plain reasoning with no headings')).toBeUndefined()
})

test('the live phrase shows the heading while reasoning streams, else Thinking', () => {
  const streaming: Block[] = [{ kind: 'thinking', id: 1, text: '**Exploring RL Enhancements**\n\nI am now…', streaming: true }]
  expect(liveActivityPhrase(streaming)).toBe('Exploring RL Enhancements')
  const plain: Block[] = [{ kind: 'thinking', id: 1, text: 'just thinking', streaming: true }]
  expect(liveActivityPhrase(plain)).toBe('Thinking')
})
