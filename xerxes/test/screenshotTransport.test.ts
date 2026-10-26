// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'

import { screenshotParts } from '../src/context/screenshotSuperseder.js'
import { messagesToAnthropic } from '../src/llms/anthropic.js'
import { claudeCodeTranscript } from '../src/llms/claudeCode.js'
import type { ChatMessage } from '../src/types/messages.js'

const DATA = 'QUJDRA=='
const shot = JSON.stringify({
  _multimodal: true,
  content: [
    { type: 'text', text: 'Screen capture: 1568x1018\nMode: som' },
    { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${DATA}` } },
  ],
  text_summary: 'Screen capture: 1568x1018\nMode: som',
})
const messages: ChatMessage[] = [
  { role: 'user', content: 'look' },
  { role: 'assistant', content: '', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'computer_use', arguments: { action: 'capture' } } }] },
  { role: 'tool', tool_call_id: 'c1', name: 'computer_use', content: shot, is_error: false },
]

test('a screenshot tool result splits into its summary and image', () => {
  expect(screenshotParts(shot)).toEqual({ text: 'Screen capture: 1568x1018\nMode: som', images: [{ mediaType: 'image/jpeg', data: DATA }] })
  expect(screenshotParts('plain output')).toBeUndefined()
  expect(screenshotParts('{"_multimodal":true,"content":[]}')).toBeUndefined()
})

test('Claude Code receives the screenshot as an image block, not base64 text', () => {
  const blocks = claudeCodeTranscript(messages)
  const result = blocks.find(block => block.type === 'text' && block.text.includes('tool_result'))
  expect(result?.type === 'text' ? result.text : '').toContain('Screen capture: 1568x1018')
  expect(result?.type === 'text' ? result.text : '').not.toContain(DATA)
  expect(blocks.at(-1)).toEqual({ type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: DATA } })
})

test('the Anthropic API receives the screenshot as an image inside its tool_result', () => {
  const converted = messagesToAnthropic(messages)
  const last = converted.messages.at(-1)
  const block = Array.isArray(last?.content) ? last.content[0] : undefined
  expect(block).toMatchObject({
    type: 'tool_result',
    tool_use_id: 'c1',
    content: [
      { type: 'text', text: 'Screen capture: 1568x1018\nMode: som' },
      { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: DATA } },
    ],
  })
})
