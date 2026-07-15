// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { liveTailScrollKey, shouldAutoScrollLiveTail } from '../app/liveTailScroll.js'
import type { TurnState } from '../app/turnStore.js'
import type { ScrollBoxHandle } from '../lib/terminalTypes.js'

const turnState = (patch: Partial<TurnState> = {}): TurnState => ({
  activity: [],
  outcome: '',
  reasoning: '',
  reasoningActive: false,
  reasoningStreaming: false,
  reasoningTokens: 0,
  streamPendingTools: [],
  streamSegments: [],
  streaming: '',
  subagents: [],
  todoCollapsed: false,
  todos: [],
  toolTokens: 0,
  tools: [],
  turnTrail: [],
  ...patch
})

const scroll = (sticky: boolean): ScrollBoxHandle => ({ isSticky: () => sticky }) as ScrollBoxHandle

describe('liveTailScrollKey', () => {
  it('changes for live assistant text, tool progress, and subagent progress', () => {
    const base = liveTailScrollKey(turnState())

    expect(liveTailScrollKey(turnState({ streaming: 'hello' }))).not.toBe(base)
    expect(
      liveTailScrollKey(
        turnState({
          tools: [{ context: 'running tests', id: 't1', name: 'exec_command', startedAt: 1 }]
        })
      )
    ).not.toBe(base)
    expect(
      liveTailScrollKey(
        turnState({
          subagents: [
            {
              depth: 0,
              goal: 'scan repo',
              id: 'a1',
              index: 0,
              notes: ['reading'],
              parentId: null,
              status: 'running',
              taskCount: 0,
              thinking: [],
              toolCount: 2,
              tools: ['ReadFile']
            }
          ]
        })
      )
    ).not.toBe(base)
  })

  it('changes when a completed tool lands in the live pending shelf', () => {
    const before = liveTailScrollKey(turnState({ streamPendingTools: ['✓ ReadFile foo.py'] }))
    const after = liveTailScrollKey(turnState({ streamPendingTools: ['✓ ReadFile foo.py', '✓ GrepTool pattern'] }))

    expect(after).not.toBe(before)
  })
})

describe('shouldAutoScrollLiveTail', () => {
  it('only follows live output while the transcript is sticky', () => {
    expect(shouldAutoScrollLiveTail(true, scroll(true))).toBe(true)
    expect(shouldAutoScrollLiveTail(true, scroll(false))).toBe(false)
    expect(shouldAutoScrollLiveTail(false, scroll(true))).toBe(false)
    expect(shouldAutoScrollLiveTail(true, null)).toBe(false)
  })
})
