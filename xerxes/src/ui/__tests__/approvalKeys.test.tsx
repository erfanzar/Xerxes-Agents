// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
// Mockup 10: direct approval letters y/a/n beside the numeric quick-select.
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { describe, expect, it, vi } from 'vitest'

import type { ApprovalReq } from '../types.js'
import { resetOverlayState, patchOverlayState } from '../app/overlayStore.js'
import { PromptZone, approvalKeyChoice } from '../opentui/appLayout.js'
import { DEFAULT_THEME } from '../theme.js'

const approval = (overrides: Partial<ApprovalReq> = {}): ApprovalReq => ({
  allowPermanent: true,
  command: 'git push origin main',
  description: 'Push to the shared remote',
  requestId: 'req-1',
  ...overrides
})

describe('approvalKeyChoice', () => {
  it('binds y/a/n to once/session/deny', () => {
    expect(approvalKeyChoice('y')).toBe('once')
    expect(approvalKeyChoice('a')).toBe('session')
    expect(approvalKeyChoice('n')).toBe('deny')
  })

  it('accepts uppercase and rejects everything else', () => {
    expect(approvalKeyChoice('Y')).toBe('once')
    // 'a' is the session-scoped option — never the permanent always.
    expect(approvalKeyChoice('a')).not.toBe('always')
    expect(approvalKeyChoice('1')).toBeNull()
    expect(approvalKeyChoice('escape')).toBeNull()
  })
})

describe('PromptZone approval keys', () => {
  const harness = async () => {
    const answer = vi.fn()
    // Tall enough that the whole card — hint row included — fits the frame.
    const session = await testRender(<PromptZone actions={{ answerApproval: answer } as never} />, {
      height: 44,
      width: 90
    })
    await session.flush()
    return { answer, session }
  }

  it('states each answer with its consequence and honours the direct letters', async () => {
    act(() => patchOverlayState({ approval: approval() }))
    const { answer, session } = await harness()
    try {
      const out = session.captureCharFrame()

      // Three statements, always in this order.
      expect(out).toContain('WHAT WILL RUN')
      expect(out).toContain('WHO ASKED')
      expect(out).toContain('WHY YOU ARE SEEING THIS')
      // Every answer carries what it costs you next time.
      expect(out).toContain('y run it once')
      expect(out).toContain('asks again next time')
      expect(out).toContain('a allow for this session')
      expect(out).toContain('n deny and tell the agent why')

      act(() => session.mockInput.pressKey('y'))
      expect(answer).toHaveBeenLastCalledWith('once')
      act(() => session.mockInput.pressKey('a'))
      expect(answer).toHaveBeenLastCalledWith('session')
      act(() => session.mockInput.pressKey('n'))
      expect(answer).toHaveBeenLastCalledWith('deny')
    } finally {
      resetOverlayState()
      session.renderer.destroy()
    }
  })

  it('keeps numeric quick-select working and honors hidden permanent option', async () => {
    act(() => patchOverlayState({ approval: approval({ allowPermanent: false }) }))
    const { answer, session } = await harness()
    try {
      act(() => session.mockInput.pressKey('2'))
      // opts without 'always': once, session, deny → 2 is still session.
      expect(answer).toHaveBeenLastCalledWith('session')
      act(() => session.mockInput.pressKey('3'))
      expect(answer).toHaveBeenLastCalledWith('deny')
      // Esc denies too, but OpenTUI's mock keyboard cannot emit Escape
      // (pressKey('escape') types the literal word); that branch is
      // unchanged production code guarded by name === 'escape'.
    } finally {
      resetOverlayState()
      session.renderer.destroy()
    }
  })
})
