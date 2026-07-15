// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import {
  buildOffsets,
  computeVisibleWindow,
  estimatedMsgHeight,
  estimateMarkdownHeight,
  estimateRowHeight,
  messageHeightKey,
  resolveScrollTop,
  wrappedLines
} from '../lib/virtualHeights.js'

describe('wrappedLines', () => {
  it('counts wrapped rows by width', () => {
    expect(wrappedLines('', 10)).toBe(1)
    expect(wrappedLines('abcde', 10)).toBe(1)
    expect(wrappedLines('abcdefghij', 5)).toBe(2)
    expect(wrappedLines('abcdefghijk', 5)).toBe(3)
  })
  it('counts explicit newlines', () => {
    expect(wrappedLines('a\nb\nc', 80)).toBe(3)
  })
  it('caps very long input', () => {
    expect(wrappedLines('x'.repeat(100000), 1, 50)).toBe(50)
  })
})

describe('estimateMarkdownHeight', () => {
  it('sums block heights', () => {
    // heading(2) + paragraph(1)
    expect(estimateMarkdownHeight('# Hi\n\nshort', 80)).toBe(3)
  })
  it('counts fenced code lines plus chrome', () => {
    // 2 code lines + 2 borders + 2 margin
    expect(estimateMarkdownHeight('```\na\nb\n```', 80)).toBe(6)
  })
})

describe('estimateRowHeight', () => {
  it('adds a bottom margin for assistant rows', () => {
    expect(estimateRowHeight({ role: 'assistant', text: 'hello' }, 80)).toBe(2) // 1 line + margin
    expect(estimateRowHeight({ role: 'user', text: 'hello' }, 80)).toBe(1)
  })
})

describe('estimatedMsgHeight detail visibility', () => {
  it('never reserves transcript rows for panel-owned subagent details', () => {
    const msg = {
      kind: 'trail' as const,
      role: 'system' as const,
      text: '',
      subagents: [
        {
          depth: 1,
          goal: 'review the renderer',
          id: 'reviewer-1',
          index: 0,
          notes: ['checking output'],
          parentId: null,
          status: 'running' as const,
          taskCount: 1,
          thinking: [],
          toolCount: 0,
          tools: []
        }
      ]
    }
    const visible = estimatedMsgHeight(msg, 80, { compact: false, details: false, subagentsVisible: true })
    const hidden = estimatedMsgHeight(msg, 80, { compact: false, details: false, subagentsVisible: false })

    expect(visible).toBe(hidden)
    expect(visible).toBe(1)
    expect(hidden).toBe(1)
  })

  it('does not invalidate transcript height keys for panel-only agent updates', () => {
    const base = {
      kind: 'trail' as const,
      role: 'system' as const,
      text: '',
      subagents: [
        {
          depth: 0,
          goal: 'audit runtime',
          id: 'agent-1',
          index: 0,
          notes: [],
          parentId: null,
          status: 'running' as const,
          taskCount: 1,
          thinking: [],
          toolCount: 0,
          tools: []
        }
      ]
    }

    expect(messageHeightKey(base)).toBe(
      messageHeightKey({
        ...base,
        subagents: [{ ...base.subagents[0]!, notes: ['new panel progress'], status: 'completed' }]
      })
    )
  })
})

describe('buildOffsets', () => {
  it('is a prefix sum with a trailing total', () => {
    expect(buildOffsets([2, 3, 5])).toEqual([0, 2, 5, 10])
    expect(buildOffsets([])).toEqual([0])
  })
})

describe('computeVisibleWindow', () => {
  const heights = [3, 3, 3, 3, 3] // total 15, offsets [0,3,6,9,12,15]

  it('returns the bottom window when scrolled past the end (sticky)', () => {
    const w = computeVisibleWindow(heights, 6, Number.MAX_SAFE_INTEGER)
    expect(w.maxScrollTop).toBe(9) // 15 - 6
    expect(w.totalHeight).toBe(15)
    // top=9 → rows starting at offset>=9: indices 3,4
    expect(w.start).toBe(3)
    expect(w.end).toBe(4)
  })

  it('returns the top window at scrollTop 0', () => {
    const w = computeVisibleWindow(heights, 6, 0)
    expect(w.start).toBe(0)
    expect(w.end).toBe(1) // rows 0 (0-3) and 1 (3-6)
    expect(w.padTop).toBe(0)
  })

  it('reports padTop when the first row is partially scrolled off', () => {
    const w = computeVisibleWindow(heights, 6, 4) // top in the middle of row 1
    expect(w.start).toBe(1)
    expect(w.padTop).toBe(1) // 4 - offsets[1]=3
  })

  it('handles an empty transcript', () => {
    const w = computeVisibleWindow([], 10, 0)
    expect(w).toMatchObject({ start: 0, end: -1, totalHeight: 0, maxScrollTop: 0 })
  })

  it('shows everything when content fits the viewport', () => {
    const w = computeVisibleWindow(heights, 100, 0)
    expect(w.start).toBe(0)
    expect(w.end).toBe(4)
    expect(w.maxScrollTop).toBe(0)
  })
})

describe('resolveScrollTop', () => {
  it('pins to bottom when sticky', () => {
    expect(resolveScrollTop(9, true, 2)).toBe(9)
  })
  it('clamps within range when not sticky', () => {
    expect(resolveScrollTop(9, false, -5)).toBe(0)
    expect(resolveScrollTop(9, false, 20)).toBe(9)
    expect(resolveScrollTop(9, false, 4)).toBe(4)
  })
})
