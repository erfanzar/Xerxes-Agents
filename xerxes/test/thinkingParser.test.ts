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
  expect(splitThinkingTags('<thinking>unfinished')).toEqual({ visible: '', thinking: 'unfinished' })
  expect(splitThinkingTags('before\n<thinking>unfinished')).toEqual({ visible: 'before\n', thinking: 'unfinished' })
})

test('a tag the reply only mentions never hides the rest of the answer as thinking', () => {
  const answer = 'Done. DeepSeek wraps its reasoning in <think> tags, so the parser strips them.\n\n## Summary\n- fixed A\n- fixed B'
  for (const size of [answer.length, 5, 1]) {
    const parser = new ThinkingParser()
    const parts = []
    for (let index = 0; index < answer.length; index += size) parts.push(...parser.process(answer.slice(index, index + size)))
    parts.push(...parser.process(''))
    expect(parts.filter(part => part.type === 'thinking')).toEqual([])
    expect(parts.map(part => part.text).join('')).toBe(answer)
  }
  // Inline code is a mention even when the closing tag is mentioned too.
  expect(splitThinkingTags('Use `<think>` and `</think>` around reasoning.')).toEqual({
    visible: 'Use `<think>` and `</think>` around reasoning.',
    thinking: '',
  })
})
