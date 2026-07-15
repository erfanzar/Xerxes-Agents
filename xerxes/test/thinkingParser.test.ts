// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ThinkingParser, splitThinkingTags } from '../src/streaming/thinkingParser.js'

test('thinking parser keeps tags split across streamed chunks out of visible text', () => {
  const parser = new ThinkingParser()
  expect(parser.process('Visible <thi')).toEqual([{ type: 'text', text: 'Visible ' }])
  expect(parser.process('nk>private')).toEqual([])
  expect(parser.process('</think> answer')).toEqual([
    { type: 'thinking', text: 'private' },
    { type: 'text', text: ' answer' },
  ])
})

test('thinking parser flushes an unclosed reasoning block at end of stream', () => {
  expect(splitThinkingTags('before <thinking>unfinished')).toEqual({ visible: 'before ', thinking: 'unfinished' })
})
