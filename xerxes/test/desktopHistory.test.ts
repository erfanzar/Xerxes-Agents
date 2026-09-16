// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { historyBlocks } from '../src/desktop/renderer/history.js'
import { BlockBuilder } from '../src/desktop/renderer/blocks.js'

test('paged history has unique stable keys and does not replace live streaming output', () => {
  const ids = new Map<string, number>()
  const page = (start: number) => ({ before: null, has_more: false, actions: Array.from({ length: 100 }, (_, i) => ({ id: String(start + i), messages: [{ role: 'user', content: `Action ${start + i}` }], executions: [], thinking: [] })) })
  const tail = historyBlocks(page(100), ids)
  const builder = new BlockBuilder()
  builder.reset(tail)
  builder.push('text_part', { text: 'Still streaming' })
  builder.prepend(historyBlocks(page(0), ids))
  builder.push('text_part', { text: ' after paging' })
  const blocks = builder.snapshot(true)
  expect(blocks).toHaveLength(201)
  expect(new Set(blocks.map(block => block.id)).size).toBe(201)
  expect(blocks.at(-1)).toMatchObject({ text: 'Still streaming after paging', streaming: true })
  expect(historyBlocks(page(100), ids).map(block => block.id)).toEqual(tail.map(block => block.id))
})
