// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it, vi } from 'vitest'

import { planGatewayRecovery } from '../app/gatewayRecovery.js'
import { applyVoiceRecordResponse, shouldFallThroughForScroll } from '../app/useInputHandlers.js'
import { liveSessionInflightMessages } from '../app/useSessionLifecycle.js'
import { approvalAction } from '../domain/approval.js'
import { toTranscriptMessages } from '../domain/messages.js'
import { shouldShowStartupWelcome, startupComposerWidth } from '../domain/startupLayout.js'
import { capTranscriptHistory, visibleTranscriptMessages } from '../lib/messages.js'

const promptScrollKey = (patch: Partial<Parameters<typeof shouldFallThroughForScroll>[0]> = {}) => ({
  downArrow: false,
  pageDown: false,
  pageUp: false,
  shift: false,
  upArrow: false,
  wheelDown: false,
  wheelUp: false,
  ...patch
})

describe('native OpenTUI semantic parity', () => {
  it('leaves the welcome surface for submitted turns and pending interactions before transcript hydration', () => {
    const idle = { busy: false, hasLiveTurn: false, pendingInteraction: false, transcriptEmpty: true }

    expect(shouldShowStartupWelcome(idle)).toBe(true)
    expect(shouldShowStartupWelcome({ ...idle, busy: true })).toBe(false)
    expect(shouldShowStartupWelcome({ ...idle, hasLiveTurn: true })).toBe(false)
    expect(shouldShowStartupWelcome({ ...idle, pendingInteraction: true })).toBe(false)
    expect(shouldShowStartupWelcome({ ...idle, transcriptEmpty: false })).toBe(false)
  })

  it('uses more of ultra-wide welcome screens without overflowing narrow terminals', () => {
    expect(startupComposerWidth(40)).toBe(36)
    expect(startupComposerWidth(80)).toBe(75)
    expect(startupComposerWidth(160)).toBe(88)
    expect(startupComposerWidth(200)).toBe(104)
  })

  it('dispatches native approval choices without relying on Prompt Toolkit sentinels', () => {
    expect(approvalAction('2', {}, 0)).toEqual({ kind: 'choose', choice: 'session' })
    expect(approvalAction('', { return: true }, 3)).toEqual({ kind: 'choose', choice: 'deny' })
    expect(approvalAction('', { escape: true }, 0)).toEqual({ kind: 'choose', choice: 'deny' })
    expect(approvalAction('', { downArrow: true }, 0)).toEqual({ kind: 'move', delta: 1 })
    expect(approvalAction('', { upArrow: true }, 0)).toEqual({ kind: 'noop' })
  })

  it('does not expose a permanent allow choice when the daemon disallows it', () => {
    const restrictedChoices = ['once', 'session', 'deny'] as const

    expect(approvalAction('3', {}, 0, restrictedChoices)).toEqual({ kind: 'choose', choice: 'deny' })
    expect(approvalAction('4', {}, 0, restrictedChoices)).toEqual({ kind: 'noop' })
  })

  it('keeps transcript and in-flight user context when resuming a native session', () => {
    expect(
      toTranscriptMessages([
        { role: 'user', text: 'inspect the auth flow' },
        { context: '{"path":"src/auth.ts"}', name: 'ReadFile', role: 'tool' },
        { context: '{"pattern":"token"}', name: 'GrepTool', role: 'tool' },
        { role: 'assistant', text: 'The flow starts in auth.ts.' },
        { role: 'tool', text: 'orphaned tool result' },
        { role: 'system', text: 'resumed session' }
      ])
    ).toEqual([
      { role: 'user', text: 'inspect the auth flow' },
      {
        role: 'assistant',
        text: 'The flow starts in auth.ts.',
        tools: ['Read File("{\"path\":\"src/auth.ts\"}") ✓', 'Grep Tool("{\"pattern\":\"token\"}") ✓']
      },
      { role: 'system', text: 'resumed session' }
    ])
    expect(liveSessionInflightMessages({ user: '  keep this turn alive  ' })).toEqual([
      { role: 'user', text: 'keep this turn alive' }
    ])
    expect(liveSessionInflightMessages({ assistant: 'only a partial reply' })).toEqual([])
  })

  it('keeps prompt-time transcript scrolling available while an approval or clarify overlay is active', () => {
    expect(shouldFallThroughForScroll(promptScrollKey({ wheelUp: true }))).toBe(true)
    expect(shouldFallThroughForScroll(promptScrollKey({ pageDown: true }))).toBe(true)
    expect(shouldFallThroughForScroll(promptScrollKey({ shift: true, upArrow: true }))).toBe(true)
    expect(shouldFallThroughForScroll(promptScrollKey({ downArrow: true }))).toBe(false)
  })

  it('reconciles start-recording responses without clobbering an already-stopped optimistic state', () => {
    const setProcessing = vi.fn()
    const setRecording = vi.fn()
    const sys = vi.fn()
    const voice = { setProcessing, setRecording }

    applyVoiceRecordResponse({ status: 'recording' }, true, voice, sys)
    expect(setRecording).not.toHaveBeenCalled()

    applyVoiceRecordResponse({ status: 'busy' }, true, voice, sys)
    expect(setRecording).toHaveBeenLastCalledWith(false)
    expect(setProcessing).toHaveBeenLastCalledWith(true)
    expect(sys).toHaveBeenLastCalledWith('voice: still transcribing; try again shortly')

    applyVoiceRecordResponse({ status: 'stopped' }, false, voice, sys)
    expect(setRecording).toHaveBeenCalledTimes(1)
    expect(setProcessing).toHaveBeenCalledTimes(1)
  })

  it('caps gateway crash recovery while retaining the session selected for resume', () => {
    const now = 1_000_000
    const first = planGatewayRecovery('session-1', null, [], now)
    const second = planGatewayRecovery(null, first.sid, first.attempts, now + 1)
    const third = planGatewayRecovery(null, second.sid, second.attempts, now + 2)
    const exhausted = planGatewayRecovery(null, third.sid, third.attempts, now + 3)

    expect(first).toMatchObject({ attempts: [now], recover: true, sid: 'session-1' })
    expect(second.recover).toBe(true)
    expect(third.recover).toBe(true)
    expect(exhausted).toEqual({ attempts: third.attempts, recover: false, sid: 'session-1' })
    expect(planGatewayRecovery('session-2', null, [now - 60_001], now)).toEqual({
      attempts: [now],
      recover: true,
      sid: 'session-2'
    })
  })

  it('does not give renderer-empty bookkeeping trails virtual scroll height', () => {
    const visible = visibleTranscriptMessages([
      { role: 'user', text: 'hello' },
      { kind: 'trail', role: 'system', text: '', toolTokens: 9 },
      { kind: 'trail', role: 'system', text: '', thinking: 'working' }
    ])

    expect(visible).toEqual([
      { role: 'user', text: 'hello' },
      { kind: 'trail', role: 'system', text: '', thinking: 'working' }
    ])
  })

  it('caps hydrated transcripts while retaining their metadata intro', () => {
    const intro = { kind: 'intro' as const, role: 'system' as const, text: '' }
    const rows = Array.from({ length: 7 }, (_, index) => ({ role: 'user' as const, text: `message ${index}` }))

    expect(capTranscriptHistory([intro, ...rows], 5)).toEqual([intro, ...rows.slice(-4)])
    expect(capTranscriptHistory([intro, ...rows], 1)).toEqual([intro])
    expect(capTranscriptHistory(rows, 5)).toEqual(rows.slice(-5))
  })
})
