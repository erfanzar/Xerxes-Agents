// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import type { GatewayEventHandlerContext } from '../app/interfaces.js'
import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { resetUiState } from '../app/uiStore.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { GatewayEvent } from '../gatewayTypes.js'
import { formatAbandonedClarify } from '../lib/text.js'
import type { Msg } from '../types.js'

const buildHarness = (overrides: { bellOnComplete?: boolean; isTTY?: boolean } = {}) => {
  const appended: Msg[] = []
  const sys = vi.fn()
  const ctx: GatewayEventHandlerContext = {
    composer: { setInput: vi.fn() },
    gateway: {
      gw: {} as GatewayClient,
      rpc: vi.fn(async () => null)
    },
    session: {
      STARTUP_RESUME_ID: '',
      colsRef: { current: 80 },
      newSession: vi.fn(),
      recoverSidRef: { current: null },
      resetSession: vi.fn(),
      resumeById: vi.fn(),
      setCatalog: vi.fn()
    },
    submission: { submitRef: { current: vi.fn() } },
    system: {
      bellOnComplete: overrides.bellOnComplete ?? false,
      stdout: { isTTY: overrides.isTTY ?? false } as NodeJS.WriteStream,
      sys
    },
    transcript: {
      appendMessage: message => appended.push(message),
      panel: vi.fn(),
      setHistoryItems: vi.fn()
    },
    voice: {
      setProcessing: vi.fn(),
      setRecording: vi.fn(),
      setVoiceEnabled: vi.fn(),
      setVoiceTts: vi.fn()
    }
  }

  return { appended, handler: createGatewayEventHandler(ctx), sys }
}

const liveClarify = () =>
  patchOverlayState({
    clarify: {
      choices: ['alpha', 'beta'],
      question: 'Which model?',
      requestId: 'req-1:q'
    }
  })

describe('createGatewayEventHandler', () => {
  afterEach(() => {
    turnController.fullReset()
    resetOverlayState()
    resetUiState()
    vi.restoreAllMocks()
  })

  it('passes replayed thinking through to the transcript row', () => {
    const { appended, handler } = buildHarness()

    handler({
      payload: { role: 'assistant', text: 'old answer', thinking: 'old trace' },
      type: 'transcript.append'
    } as GatewayEvent)

    expect(appended).toEqual([
      { role: 'assistant', text: 'old answer', thinking: 'old trace' }
    ])
  })

  it('records an abandoned clarify prompt instead of dropping it at message.complete', () => {
    const { appended, handler } = buildHarness()
    liveClarify()
    turnController.startMessage()

    handler({ payload: { text: 'done' }, type: 'message.complete' } as GatewayEvent)

    const abandoned = formatAbandonedClarify('Which model?', ['alpha', 'beta'], 'timed out')

    expect(appended.some(message => message.role === 'system' && message.text === abandoned)).toBe(true)
    expect(getOverlayState().clarify).toBeNull()
    // The turn's real completion still lands after the abandoned record.
    expect(appended.at(-1)).toEqual({ role: 'assistant', text: 'done' })
  })

  it('records an abandoned clarify prompt on a turn-level error too', () => {
    const { appended, handler } = buildHarness()
    liveClarify()
    turnController.startMessage()

    handler({ payload: { message: 'boom' }, type: 'error' } as GatewayEvent)

    expect(appended.some(message => message.text === formatAbandonedClarify('Which model?', ['alpha', 'beta'], 'timed out'))).toBe(
      true
    )
    expect(getOverlayState().clarify).toBeNull()
  })

  it('flushes an abandoned clarify only once across tool.complete and message.complete', () => {
    const { appended, handler } = buildHarness()
    liveClarify()
    turnController.startMessage()

    handler({ payload: { name: 'clarify', tool_id: 'clarify-1' }, type: 'tool.complete' } as GatewayEvent)
    handler({ payload: { text: 'done' }, type: 'message.complete' } as GatewayEvent)

    const abandoned = formatAbandonedClarify('Which model?', ['alpha', 'beta'], 'timed out')

    expect(appended.filter(message => message.text === abandoned)).toHaveLength(1)
  })

  it('keeps a live clarify overlay untouched when the turn answered it first', () => {
    const { appended, handler } = buildHarness()

    turnController.startMessage()
    handler({ payload: { text: 'done' }, type: 'message.complete' } as GatewayEvent)

    expect(appended.every(message => !message.text.includes('no selection'))).toBe(true)
  })

  it('rings the completion bell on the real stdout, past the guarded proxy', () => {
    const write = vi.spyOn(process.stdout, 'write').mockImplementation(() => true)
    const { handler } = buildHarness({ bellOnComplete: true, isTTY: true })

    turnController.startMessage()
    handler({ payload: { text: 'done' }, type: 'message.complete' } as GatewayEvent)

    expect(write).toHaveBeenCalledWith('\x07')
  })

  it('stays silent when the bell is disabled or stdout is not a TTY', () => {
    const write = vi.spyOn(process.stdout, 'write').mockImplementation(() => true)

    for (const overrides of [{ bellOnComplete: false, isTTY: true }, { bellOnComplete: true, isTTY: false }]) {
      write.mockClear()
      const { handler } = buildHarness(overrides)

      turnController.startMessage()
      handler({ payload: { text: 'done' }, type: 'message.complete' } as GatewayEvent)

      expect(write).not.toHaveBeenCalled()
    }
  })

  it('appends the archived turn ending without a bell on a daemon-confirmed interrupt', () => {
    const write = vi.spyOn(process.stdout, 'write').mockImplementation(() => true)
    const { appended, handler, sys } = buildHarness({ bellOnComplete: true, isTTY: true })
    const request = vi.fn().mockResolvedValue({ ok: true })

    turnController.startMessage()
    turnController.recordMessageDelta({ text: 'Partial draft' })
    turnController.interruptTurn({ gw: { request }, sid: 'session-cut', sys })
    handler({ payload: { interrupted: true }, type: 'message.complete' } as GatewayEvent)

    expect(appended).toEqual([{ role: 'assistant', text: 'Partial draft\n\n*[interrupted]*' }])
    expect(write).not.toHaveBeenCalled()
  })

  it('emits the bare interrupted note when a confirmed interrupt has nothing to archive', () => {
    const { appended, handler, sys } = buildHarness()
    const request = vi.fn().mockResolvedValue({ ok: true })

    turnController.startMessage()
    turnController.interruptTurn({ gw: { request }, sid: 'session-empty', sys })
    handler({ payload: { interrupted: true }, type: 'message.complete' } as GatewayEvent)

    expect(appended).toEqual([])
    expect(sys).toHaveBeenCalledWith('interrupted')
  })

  it('renders the real final messages when a natural completion races the Esc interrupt', () => {
    const { appended, handler, sys } = buildHarness()
    const request = vi.fn().mockResolvedValue({ ok: true })

    turnController.startMessage()
    turnController.recordMessageDelta({ text: 'The real answer.' })
    turnController.interruptTurn({ gw: { request }, sid: 'session-race', sys })
    handler({ payload: {}, type: 'message.complete' } as GatewayEvent)

    expect(appended).toEqual([{ role: 'assistant', text: 'The real answer.' }])
    expect(sys).not.toHaveBeenCalled()
  })
})
