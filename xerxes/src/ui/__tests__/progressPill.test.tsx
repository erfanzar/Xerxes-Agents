// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
// Mockup 02: one quiet pill floats at the transcript end while a turn runs.
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { describe, expect, it } from 'vitest'

import { patchTurnState, resetTurnState } from '../app/turnStore.js'
import { LiveProgressPill } from '../opentui/appLayout.js'

const renderPill = async () => {
  const session = await testRender(<LiveProgressPill />, { height: 6, width: 90 })
  await session.flush()
  return session
}

describe('LiveProgressPill', () => {
  it('renders verb, clock, tool count and tokens in one row while live', async () => {
    patchTurnState({
      streamPendingTools: ['Read File("a.ts") ✓', 'Spawn Agents("2 agents: scout, tester") ✓'],
      toolTokens: 4200,
      tools: [{ id: 't1', name: 'Bash' }]
    })
    const session = await renderPill()
    try {
      const out = session.captureCharFrame()
      expect(out).toContain('✻')
      expect(out).toContain('(esc interrupt)')
      expect(out).toContain('3 tools')
      expect(out).toContain('4.2K tok')
    } finally {
      resetTurnState()
      session.renderer.destroy()
    }
  })

  it('disappears on completion', async () => {
    patchTurnState({ tools: [{ id: 't1', name: 'Bash' }] })
    const session = await renderPill()
    try {
      expect(session.captureCharFrame()).toContain('✻')

      act(() => resetTurnState())
      await session.flush()
      expect(session.captureCharFrame()).not.toContain('✻')
    } finally {
      resetTurnState()
      session.renderer.destroy()
    }
  })

  it('stays hidden while idle', async () => {
    resetTurnState()
    const session = await renderPill()
    try {
      expect(session.captureCharFrame()).not.toContain('esc interrupt')
    } finally {
      session.renderer.destroy()
    }
  })
})
