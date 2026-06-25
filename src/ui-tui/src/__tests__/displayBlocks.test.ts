// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { normalizeDisplayBlocks, summarizeResult, todoItems } from '../lib/displayBlocks.js'
import { initialState, reduce } from '../app/gatewayState.js'

describe('normalizeDisplayBlocks', () => {
  it('keeps known block types and coerces fields', () => {
    const blocks = normalizeDisplayBlocks([
      { type: 'brief', body: 'hi' },
      { type: 'diff', diff: '+a\n-b', language: 'ts' },
      { type: 'background_task', title: 'build', status: 'running' },
      { type: 'generic', content: 'x' },
      { type: 'todo', items: [{ status: 'completed', content: 'done it' }] }
    ])
    expect(blocks).toHaveLength(5)
    expect(blocks[0]).toEqual({ type: 'brief', body: 'hi' })
    expect(blocks[1]).toMatchObject({ type: 'diff', language: 'ts' })
  })

  it('drops unknown / malformed blocks', () => {
    expect(normalizeDisplayBlocks([{ type: 'mystery' }, null, 42, { body: 'no type' }])).toEqual([])
    expect(normalizeDisplayBlocks('not an array')).toEqual([])
  })

  it('defaults missing string fields', () => {
    expect(normalizeDisplayBlocks([{ type: 'brief' }])).toEqual([{ type: 'brief', body: '' }])
  })
})

describe('todoItems', () => {
  it('uniformizes status/content with defaults', () => {
    expect(todoItems({ type: 'todo', items: [{ content: 'a' }, { status: 'done', content: 'b' }] })).toEqual([
      { status: 'pending', content: 'a' },
      { status: 'done', content: 'b' }
    ])
  })
})

describe('summarizeResult', () => {
  it('uses the first line, truncating long output', () => {
    expect(summarizeResult('first line\nsecond', 0)).toBe('first line')
    expect(summarizeResult('x'.repeat(200), 0)).toHaveLength(118) // 117 + ellipsis
  })
  it('falls back to a duration when empty', () => {
    expect(summarizeResult('', 1234)).toBe('done (1234ms)')
    expect(summarizeResult('', 0)).toBe('done')
  })
})

describe('tool_result reducer integration', () => {
  it('adds a tool row carrying normalized blocks', () => {
    const s = reduce(initialState, {
      type: 'tool_result',
      payload: {
        tool_call_id: 'tc1',
        return_value: 'ignored when blocks present',
        duration_ms: 12,
        display_blocks: [{ type: 'diff', diff: '+added', language: '' }]
      }
    })
    expect(s.transcript).toHaveLength(1)
    expect(s.transcript[0]!.role).toBe('tool')
    expect(s.transcript[0]!.blocks).toHaveLength(1)
    expect(s.transcript[0]!.blocks![0]).toMatchObject({ type: 'diff' })
  })

  it('falls back to a text summary when there are no blocks', () => {
    const s = reduce(initialState, {
      type: 'tool_result',
      payload: { tool_call_id: 'tc1', return_value: 'all good\nmore', duration_ms: 5, display_blocks: [] }
    })
    expect(s.transcript[0]!.text).toBe('all good')
    expect(s.transcript[0]!.blocks).toBeUndefined()
  })
})
