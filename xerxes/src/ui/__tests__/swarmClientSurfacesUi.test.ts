// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { EventEmitter } from 'node:events'
import type { Socket } from 'node:net'

import { afterEach, describe, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { getOverlayState, patchOverlayState, resetFlowOverlays, resetOverlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { GatewayClient, MAX_GATEWAY_FRAME_BYTES } from '../gatewayClient.js'
import type { Msg, SlashCatalog } from '../types.js'

const flushMicrotasks = async (): Promise<void> => {
  await Promise.resolve()
  await Promise.resolve()
  await Promise.resolve()
}

function makeSlashFixture(request: ReturnType<typeof vi.fn>, catalog: null | SlashCatalog = null) {
  const sys: string[] = []
  const page: Array<{ text: string; title?: string }> = []
  const send: string[] = []
  let historyItems: Msg[] = []
  const setHistoryItems = (update: Msg[] | ((items: Msg[]) => Msg[])) => {
    historyItems = typeof update === 'function' ? update(historyItems) : update
  }

  return {
    context: {
      composer: {
        enqueue: vi.fn(),
        hasSelection: false,
        paste: vi.fn(),
        queueRef: { current: [] },
        selection: {
          captureScrolledRows: vi.fn(),
          clearSelection: vi.fn(),
          copySelection: vi.fn(),
          copySelectionNoClear: vi.fn(),
          getState: vi.fn(),
          shiftAnchor: vi.fn(),
          shiftSelection: vi.fn(),
          version: vi.fn(),
        },
        setInput: vi.fn(),
      },
      gateway: {
        gw: { request },
        rpc: request,
      },
      local: {
        catalog,
        getHistoryItems: vi.fn(() => historyItems),
        getLastUserMsg: vi.fn(() => ''),
        maybeWarn: vi.fn(),
        setCatalog: vi.fn(),
      },
      session: {
        closeSession: vi.fn(),
        die: vi.fn(),
        dieWithCode: vi.fn(),
        guardBusySessionSwitch: vi.fn(),
        newLiveSession: vi.fn(),
        newSession: vi.fn(),
        resetVisibleHistory: vi.fn(),
        resumeById: vi.fn(),
        setSessionStartedAt: vi.fn(),
      },
      slashFlightRef: { current: 0 },
      transcript: {
        dispatch: vi.fn(),
        page: (text: string, title?: string) => page.push({ text, ...(title ? { title } : {}) }),
        panel: vi.fn(),
        send: (text: string) => send.push(text),
        setHistoryItems: vi.fn(setHistoryItems),
        sys: (text: string) => sys.push(text),
        trimLastExchange: vi.fn((items: Msg[]) => items),
      },
      voice: {
        setVoiceEnabled: vi.fn(),
        setVoiceRecordKey: vi.fn(),
        setVoiceTts: vi.fn(),
      },
    } as never,
    page,
    send,
    sys,
  }
}

describe('OpenTUI client/gateway surface fixes', () => {
  afterEach(() => {
    turnController.fullReset()
    resetOverlayState()
    resetUiState()
  })

  it('preserves the /usage overlay across turn completion, interruption, and error boundaries while clearing flow prompts', () => {
    patchOverlayState({
      usage: true,
      contextInspector: true,
      approval: { command: 'ExecCommand', description: 'Run build', requestId: 'req-1' },
      clarify: { choices: ['yes'], question: 'Proceed?', requestId: 'q-1', source: 'agent' },
    })

    resetFlowOverlays()
    expect(getOverlayState().usage).toBe(true)
    expect(getOverlayState().contextInspector).toBe(true)
    expect(getOverlayState().approval).toBeNull()
    expect(getOverlayState().clarify).toBeNull()

    // Also verify end-to-end through turnController.recordMessageComplete and recordError.
    turnController.startMessage()
    patchOverlayState({ usage: true })
    turnController.recordMessageComplete({})
    expect(getOverlayState().usage).toBe(true)

    turnController.startMessage()
    turnController.recordError()
    expect(getOverlayState().usage).toBe(true)
  })

  it('handles bare "/" long slash responses without throwing on empty command name', async () => {
    patchUiState({ sid: 's1' })
    const multiLineHelp = ['Available commands:', '/help — show help', '/new — start a session', '/usage — inspect usage'].join('\n')
    const request = vi.fn().mockResolvedValue({ ok: true, output: multiLineHelp })
    const { context, page, sys } = makeSlashFixture(request)

    createSlashHandler(context)('/')
    await flushMicrotasks()

    expect(sys).toEqual([])
    expect(page).toEqual([{ text: multiLineHelp, title: 'Command' }])
  })

  it('detects and breaks both synchronous catalog alias loops and async command.dispatch alias loops', async () => {
    patchUiState({ sid: 's1' })
    const cyclicCatalog: SlashCatalog = {
      canon: { '/alpha': '/beta', '/beta': '/alpha' },
      categories: [],
      pairs: [],
      skillCount: 0,
      sub: {},
    }
    const syncReq = vi.fn()
    const syncFixture = makeSlashFixture(syncReq, cyclicCatalog)

    expect(() => createSlashHandler(syncFixture.context)('/alpha arg1')).not.toThrow()
    expect(syncReq).not.toHaveBeenCalled()
    expect(syncFixture.sys).toEqual(['error: command alias loop for /beta'])

    // Async command.dispatch alias loop (/ping -> /pong -> /ping)
    const asyncReq = vi
      .fn()
      .mockRejectedValueOnce(new Error('no slash.exec'))
      .mockResolvedValueOnce({ type: 'alias', target: 'pong' })
      .mockRejectedValueOnce(new Error('no slash.exec'))
      .mockResolvedValueOnce({ type: 'alias', target: 'ping' })
    const asyncFixture = makeSlashFixture(asyncReq)

    createSlashHandler(asyncFixture.context)('/ping')
    for (let i = 0; i < 8; i++) await flushMicrotasks()

    expect(asyncFixture.sys).toEqual(['error: command alias loop for /pong'])
  })

  it('forwards session_key on complete.path, complete.slash, and skill_suggestions RPCs', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'tui:default-tab' })
    const calls: Array<{ method: string; params: Record<string, unknown> }> = []
    const internals = client as unknown as {
      rememberSessionKey: (id: string, key: string) => void
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }
    internals.rememberSessionKey('tab-2', 'daemon:tab-2')
    internals.rawRequest = async (method, params = {}) => {
      calls.push({ method, params })
      return { ok: true, completions: [{ label: 'src/index.ts', value: 'src/index.ts' }], suggestions: [] }
    }

    await client.request('complete.path', { word: 'src/', path_prefix: 'src/', session_id: 'tab-2' })
    await client.request('complete.slash', { text: '/pre' })
    await client.request('skill_suggestions', { prefix: 'rev', session_id: 'tab-2' })

    expect(calls).toEqual([
      { method: 'complete', params: { path_prefix: 'src/', session_key: 'daemon:tab-2' } },
      { method: 'complete', params: { text: '/pre', session_key: 'tui:default-tab' } },
      { method: 'skill_suggestions', params: { prefix: 'rev', session_id: 'tab-2', session_key: 'daemon:tab-2' } },
    ])
  })

  it('enforces MAX_GATEWAY_FRAME_BYTES accurately for both ASCII and multi-byte UTF-8 frames', () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'tui:framing' })
    const emitter = new EventEmitter()
    const destroy = vi.fn()
    const fakeSock = Object.assign(emitter, {
      destroy,
      end: vi.fn(),
      setEncoding: vi.fn(),
      write: vi.fn(),
    }) as unknown as Socket

    ;(client as unknown as { attachSocket: (s: Socket) => void }).attachSocket(fakeSock)
    const errors: unknown[] = []
    client.on('gateway.protocol_error', err => errors.push(err))

    // 2-byte UTF-8 characters ('é'): 9M chars < 16M UTF-16 code units, but 18MB UTF-8 > MAX_GATEWAY_FRAME_BYTES.
    const multiByteOversized = 'é'.repeat(Math.floor(MAX_GATEWAY_FRAME_BYTES / 2) + 32)
    emitter.emit('data', `${multiByteOversized}\n`)

    expect(errors).toHaveLength(1)
    expect(destroy).toHaveBeenCalledTimes(1)
  })
})
