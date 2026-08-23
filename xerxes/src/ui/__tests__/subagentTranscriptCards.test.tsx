// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
//
// Inline subagent cards on trail rows (redesign mockups 02/03⑥) and the
// collapsed-thinking header (element ③). The card model in lib/subagentCards
// is the shared source of truth for the painted rows AND the virtualization
// height correction in useMainApp, so these tests pin both its shape and its
// painted output.
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import { patchUiState, resetUiState } from '../app/uiStore.js'
import { subagentCardAccent, subagentCardModel, subagentCardRowCount, subagentCardRows } from '../lib/subagentCards.js'
import { MessageLine, thinkingHeaderLabel } from '../opentui/messageLine.js'
import { DEFAULT_THEME, themeForMode } from '../theme.js'
import type { SubagentProgress } from '../types.js'

const theme = themeForMode(DEFAULT_THEME, 'code')

const base = {
  depth: 0,
  id: 'child',
  index: 0,
  notes: [],
  parentId: null,
  taskCount: 1,
  thinking: [],
  toolCount: 0,
  tools: []
} satisfies Partial<SubagentProgress>

const researcher: SubagentProgress = {
  ...base,
  goal: 'survey token-parse patterns in the repo',
  id: 'researcher-2',
  inputTokens: 8200,
  name: 'researcher',
  notes: ['reading src/session/tokenCache.ts'],
  status: 'running',
  toolCount: 3
}

const tester: SubagentProgress = {
  ...base,
  goal: 'flaky-test triage',
  id: 'tester-3',
  inputTokens: 21000,
  name: 'tester#3',
  outputTokens: 700,
  status: 'completed',
  summary: 'retry budget exhausted by clock skew; fix queued',
  toolCount: 9
}

describe('subagent card model', () => {
  it('keeps each card at three lines or fewer', () => {
    expect(subagentCardRowCount(researcher)).toBe(3)
    expect(subagentCardRowCount(tester)).toBe(3)
    expect(subagentCardRows([researcher, tester])).toBe(6)
  })

  it('counts nothing for an absent tree', () => {
    expect(subagentCardRows(undefined)).toBe(0)
    expect(subagentCardRows([])).toBe(0)
  })

  it('formats the budget line from tokens and tools', () => {
    expect(subagentCardModel(researcher).budget).toBe('8.2k tok · 3 tools')
    expect(subagentCardModel(tester).budget).toBe('22k tok · 9 tools')
  })

  it('shows latest activity while running and a result sentence when done', () => {
    const running = subagentCardModel(researcher)
    expect(running.activity).toBe('reading src/session/tokenCache.ts')
    expect(running.result).toBe('')

    const done = subagentCardModel(tester)
    expect(done.activity).toBe('')
    expect(done.result).toBe('Result: retry budget exhausted by clock skew; fix queued')
  })

  it('derives the headline from the name and the summary from the task', () => {
    const model = subagentCardModel(researcher)
    expect(model.headline).toBe('researcher')
    // Panel vocabulary: kebab/snake split to words, title-cased, clamped.
    expect(model.summary).toBe('Survey Token Parse Patt…')
  })

  it('never repeats the derived title as the summary', () => {
    // No name: the derived title IS the headline, so the dash segment drops.
    const model = subagentCardModel({ ...base, goal: 'audit the runtime', id: 'solo', status: 'running' })
    expect(model.headline).toBe('Audit The Runtime')
    expect(model.summary).toBe('')
  })

  it('maps statuses to the agent-panel voice colours', () => {
    expect(subagentCardAccent('running', theme)).toBe('#6ea8fe')
    expect(subagentCardAccent('queued', theme)).toBe('#6ea8fe')
    expect(subagentCardAccent('completed', theme)).toBe('#57ca85')
    expect(subagentCardAccent('failed', theme)).toBe('#f47067')
    expect(subagentCardAccent('interrupted', theme)).toBe('#f47067')
  })
})

describe('inline card rendering', () => {
  afterEach(() => {
    resetUiState()
  })

  it('paints one compact card per agent on the trail', async () => {
    const setup = await testRender(
      <box flexDirection="column">
        <MessageLine msg={{ kind: 'trail', role: 'system', text: '', subagents: [researcher, tester] }} t={theme} />
      </box>,
      { height: 16, width: 80 }
    )

    try {
      await setup.flush()
      const frame = setup.captureCharFrame()

      expect(frame).toContain('● researcher')
      expect(frame).toContain('— Survey Token Parse Patt…')
      expect(frame).toContain('└ reading src/session/tokenCache.ts')
      expect(frame).toContain('8.2k tok · 3 tools')
      expect(frame).toContain('● tester#3')
      expect(frame).toContain('Result: retry budget exhausted by clock skew; fix queued')
    } finally {
      await setup.waitForVisualIdle()
      act(() => setup.renderer.destroy())
    }
  })

  it('/details hidden suppresses the cards entirely', async () => {
    patchUiState({ detailsMode: 'hidden', detailsModeCommandOverride: true })

    const setup = await testRender(
      <box flexDirection="column">
        <MessageLine msg={{ kind: 'trail', role: 'system', text: '', subagents: [researcher] }} t={theme} />
      </box>,
      { height: 10, width: 80 }
    )

    try {
      await setup.flush()
      expect(setup.captureCharFrame()).not.toContain('researcher')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})

describe('thinking header', () => {
  it('keeps the ▸/▾ thinking affordance when no duration is known', () => {
    expect(thinkingHeaderLabel({ expanded: false })).toBe('▸ thinking')
    expect(thinkingHeaderLabel({ expanded: false })).toContain('▸ thinking')
    expect(thinkingHeaderLabel({ expanded: true })).toContain('▾ thinking')
  })

  it('reports the thinking duration when one is known', () => {
    expect(thinkingHeaderLabel({ durationSeconds: 6, expanded: false })).toBe('▸ thought for 6s')
    expect(thinkingHeaderLabel({ durationSeconds: 75, expanded: true })).toBe('▾ thought for 1m 15s')
    expect(thinkingHeaderLabel({ durationSeconds: -3, expanded: false })).toBe('▸ thought for 0s')
  })
})
