// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { estimatedMsgHeight, messageHeightKey, wrappedLines } from '../lib/virtualHeights.js'

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

describe('transcript spacing contract', () => {
  // These pin the renderer/estimator agreement that Phase 4 established.
  // They previously disagreed in four places, which only showed up as
  // scroll drift when jumping into rows that had never been mounted.
  it('counts one leading blank row for a user turn, not two', () => {
    const user = estimatedMsgHeight({ role: 'user', text: 'hello' }, 80, { compact: false, details: true })
    const bare = estimatedMsgHeight({ role: 'assistant', text: 'hello' }, 80, { compact: false, details: true })

    expect(user - bare).toBe(1)
  })

  it('gives a diff its row from leadGap instead of a hardcoded pair', () => {
    const withGap = estimatedMsgHeight(
      { kind: 'diff', role: 'assistant', text: 'x' },
      80,
      { compact: false, details: true, leadGap: true }
    )
    const without = estimatedMsgHeight(
      { kind: 'diff', role: 'assistant', text: 'x' },
      80,
      { compact: false, details: true, leadGap: false }
    )

    expect(withGap - without).toBe(1)
  })

  it('reserves nothing for the separator that was never rendered', () => {
    // `withSeparator` used to add 2 rows per non-first user turn for a rule
    // no renderer ever drew. Passing the dead option must change nothing.
    const plain = estimatedMsgHeight({ role: 'user', text: 'hello' }, 80, { compact: false, details: true })
    const legacy = estimatedMsgHeight(
      { role: 'user', text: 'hello' },
      80,
      { compact: false, details: true, withSeparator: true } as never
    )

    expect(legacy).toBe(plain)
  })
})
