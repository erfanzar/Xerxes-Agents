// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { EventEmitter } from 'node:events'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import type { Socket } from 'node:net'
import { join, resolve } from 'node:path'

import { testRender } from '@opentui/react/test-utils'
import { act, createElement } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { createSlashHandler } from '../app/createSlashHandler.js'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayEventHandlerContext, GatewayServices } from '../app/interfaces.js'
import {
  getOverlayState,
  OVERLAY_BLOCKS_BACKGROUND_HOTKEYS,
  overlayBlocksBackgroundHotkeys,
  patchOverlayState,
  resetFlowOverlays,
  resetOverlayState
} from '../app/overlayStore.js'
import {
  getActiveSideQuestionSlice,
  resetSideQuestionStore
} from '../app/sideQuestionStore.js'
import {
  applySwarmStatus,
  getActiveSwarmSlice,
  resetSwarmStore
} from '../app/swarmStore.js'
import { turnController } from '../app/turnController.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { GatewayClient } from '../gatewayClient.js'
import type {
  GatewayEvent,
  SwarmBoardMessage,
  SwarmStatusSnapshot
} from '../gatewayTypes.js'
import { completionRequestForInput, tuiSlashCompletions } from '../hooks/useCompletion.js'
import {
  formatSwarmMarkdownHint,
  SWARM_BOARD_TUI_HINT,
  tokenizeSwarmMessageText
} from '../lib/swarmFormat.js'
import { summarizeToolStartDisplay } from '../lib/toolStartDisplay.js'
import { SessionHeader } from '../opentui/appChrome.js'
import { MessageLine } from '../opentui/messageLine.js'
import { SideQuestionOverlay } from '../opentui/sideQuestionOverlay.js'
import { SwarmOverlay } from '../opentui/swarmOverlay.js'
import { DARK_THEME, DEFAULT_THEME } from '../theme.js'
import type { Msg, SlashCatalog } from '../types.js'

const flushMicrotasks = async (): Promise<void> => {
  for (let i = 0; i < 5; i += 1) {
    await Promise.resolve()
  }
}

function collectUiSourceFiles(dir: string): string[] {
  const out: string[] = []
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry)
    const st = statSync(full)
    if (st.isDirectory()) {
      out.push(...collectUiSourceFiles(full))
    } else if (/\.(?:ts|tsx)$/.test(entry)) {
      out.push(full)
    }
  }
  return out
}

function makeSwarmStatus(overrides: Partial<SwarmStatusSnapshot> = {}): SwarmStatusSnapshot {
  return {
    active: true,
    session_id: 's1',
    title: 'Auth Audit Swarm',
    board: 'local',
    open_board: '[Open the Immortals UI](sidecar://immortals/board/?auto_open=false)',
    head: 4,
    overview: {
      markdown: '### Live Progress\n- `@worker-1` auditing token validation\n- `@helper-1` checking session store',
      summary: '2 active slices on track',
      author: 'narrator',
      updated_at: '2026-11-21T07:05:00Z'
    },
    channels: [
      { name: 'main', topic: 'Main coordination board', created_by: 'coordinator', archived: false, subscriber_count: 5 },
      { name: 'security', topic: 'Auth findings', created_by: 'coordinator', archived: false, subscriber_count: 2 },
      { name: 'legacy', topic: 'Old notes', created_by: 'coordinator', archived: true, subscriber_count: 1 }
    ],
    members: [
      {
        name: 'coordinator',
        role: 'coordinator',
        model: 'claude-sonnet-4-5',
        model_display_name: 'Claude Sonnet 4.5',
        workspace: 'inherit',
        channels: ['*'],
        state: 'waiting',
        killed: false,
        last_read_seq: 4,
        last_post_seq: 1
      },
      {
        name: 'worker-1',
        role: 'worker',
        added_by: 'coordinator',
        model: 'claude-sonnet-4-5',
        model_display_name: 'Claude Sonnet 4.5',
        workspace: 'share',
        channels: ['main', 'security'],
        state: 'working',
        killed: false,
        current_activity: 'Reading src/auth/token.ts',
        last_read_seq: 3,
        last_post_seq: 3
      },
      {
        name: 'helper-1',
        role: 'worker',
        added_by: 'worker-1',
        model: 'gpt-5-mini',
        model_display_name: 'GPT-5 Mini',
        workspace: 'branch',
        channels: ['main', 'security'],
        state: 'waiting',
        killed: false,
        last_read_seq: 4,
        last_post_seq: 4
      },
      {
        name: 'narrator',
        role: 'narrator',
        added_by: 'coordinator',
        model: 'claude-sonnet-4-5',
        model_display_name: 'Claude Sonnet 4.5',
        workspace: 'inherit',
        channels: ['*'],
        state: 'waiting',
        killed: false,
        last_read_seq: 4
      },
      {
        name: 'monitor',
        role: 'monitor',
        added_by: 'coordinator',
        model: 'claude-sonnet-4-5',
        model_display_name: 'Claude Sonnet 4.5',
        workspace: 'inherit',
        channels: ['*'],
        state: 'stopped',
        killed: false,
        last_read_seq: 2
      }
    ],
    ...overrides
  }
}

function makeBoardMessages(): SwarmBoardMessage[] {
  return [
    {
      seq: 1,
      agent: 'coordinator',
      channel: 'main',
      text: '@worker-1 Audit token verification and report back.',
      mentions: ['worker-1'],
      channel_mentions: [],
      broadcast: false,
      time: '2026-11-21 07:01:10'
    },
    {
      seq: 2,
      agent: 'monitor',
      channel: 'main',
      text: '@worker-1 [stall] Waiting on #1 status check.',
      mentions: ['worker-1'],
      channel_mentions: [],
      broadcast: false,
      time: '2026-11-21 07:02:15'
    },
    {
      seq: 3,
      agent: 'worker-1',
      channel: 'security',
      text: '@#security Found unsigned refresh fallback; spawned @helper-1.',
      mentions: ['helper-1'],
      channel_mentions: ['security'],
      broadcast: false,
      time: '2026-11-21 07:03:20'
    },
    {
      seq: 4,
      agent: 'helper-1',
      channel: 'main',
      text: '@worker-1 [done] Verified fix for #3 across all session stores.',
      mentions: ['worker-1'],
      channel_mentions: [],
      broadcast: false,
      time: '2026-11-21 07:04:45'
    }
  ]
}

function makeSlashFixture(request: ReturnType<typeof vi.fn>, catalog: null | SlashCatalog = null) {
  const sys: string[] = []
  const page: Array<{ text: string; title?: string }> = []
  const send: string[] = []
  let historyItems: Msg[] = [{ role: 'user', text: 'Existing user message' }]
  const setHistoryItems = vi.fn((update: Msg[] | ((items: Msg[]) => Msg[])) => {
    historyItems = typeof update === 'function' ? update(historyItems) : update
  })
  const resetVisibleHistory = vi.fn()

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
          version: vi.fn()
        },
        setInput: vi.fn()
      },
      gateway: {
        gw: { request },
        rpc: request
      },
      local: {
        catalog,
        getHistoryItems: vi.fn(() => historyItems),
        getLastUserMsg: vi.fn(() => ''),
        maybeWarn: vi.fn(),
        setCatalog: vi.fn()
      },
      session: {
        closeSession: vi.fn(),
        die: vi.fn(),
        dieWithCode: vi.fn(),
        guardBusySessionSwitch: vi.fn(),
        newLiveSession: vi.fn(),
        newSession: vi.fn(),
        resetVisibleHistory,
        resumeById: vi.fn(),
        setSessionStartedAt: vi.fn()
      },
      slashFlightRef: { current: 0 },
      transcript: {
        dispatch: vi.fn(),
        page: (text: string, title?: string) => page.push({ text, ...(title ? { title } : {}) }),
        panel: vi.fn(),
        send: (text: string) => send.push(text),
        setHistoryItems,
        sys: (text: string) => sys.push(text),
        trimLastExchange: vi.fn((items: Msg[]) => items)
      },
      voice: {
        setVoiceEnabled: vi.fn(),
        setVoiceRecordKey: vi.fn(),
        setVoiceTts: vi.fn()
      }
    } as never,
    getHistoryItems: () => historyItems,
    page,
    resetVisibleHistory,
    send,
    setHistoryItems,
    sys
  }
}

describe('Native Swarm TUI', () => {
  afterEach(() => {
    turnController.fullReset()
    resetOverlayState()
    resetSideQuestionStore()
    resetSwarmStore()
    resetUiState()
  })

  it('keeps xerxes/src/ui strictly protocol-independent with zero imports from src/swarm', () => {
    const uiRoot = resolve(import.meta.dirname, '..')
    const files = collectUiSourceFiles(uiRoot).filter(f => !f.endsWith('nativeSwarmUi.test.ts'))
    const offenders: string[] = []
    for (const file of files) {
      const content = readFileSync(file, 'utf8')
      if (/(?:from|import)\s+['"][^'"]*\/swarm(?:\/|['"])/.test(content)) {
        offenders.push(file)
      }
    }
    expect(offenders).toEqual([])
  })

  it('preserves OverlayState.swarm and OverlayState.sideQuestion across turn completion, error, and resetFlowOverlays while blocking background hotkeys', () => {
    patchOverlayState({
      swarm: true,
      sideQuestion: true,
      approval: { command: 'ExecCommand', description: 'Run check', requestId: 'perm-1' },
      clarify: { choices: ['yes'], question: 'Continue?', requestId: 'clarify-1', source: 'agent' },
      confirm: { title: 'Confirm', detail: 'Proceed?', onConfirm: () => undefined }
    })

    expect(OVERLAY_BLOCKS_BACKGROUND_HOTKEYS.swarm).toBe(true)
    expect(OVERLAY_BLOCKS_BACKGROUND_HOTKEYS.sideQuestion).toBe(true)
    expect(overlayBlocksBackgroundHotkeys(getOverlayState())).toBe(true)

    resetFlowOverlays()
    expect(getOverlayState().swarm).toBe(true)
    expect(getOverlayState().sideQuestion).toBe(true)
    expect(getOverlayState().approval).toBeNull()
    expect(getOverlayState().clarify).toBeNull()
    expect(getOverlayState().confirm).toBeNull()

    turnController.startMessage()
    turnController.recordMessageComplete({})
    expect(getOverlayState().swarm).toBe(true)
    expect(getOverlayState().sideQuestion).toBe(true)

    turnController.startMessage()
    turnController.recordError()
    expect(getOverlayState().swarm).toBe(true)
    expect(getOverlayState().sideQuestion).toBe(true)
  })

  it('wires GatewayClient v35 swarm.* RPC helpers and live swarm_event stream updates per session', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'tui:tab-1' })
    const calls: Array<{ method: string; params: Record<string, unknown> }> = []
    const internals = client as unknown as {
      rememberSessionKey: (id: string, key: string) => void
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
      attachSocket: (s: Socket) => void
    }
    internals.rememberSessionKey('s1', 'daemon:s1')
    internals.rawRequest = async (method, params = {}) => {
      calls.push({ method, params })
      if (method === 'swarm.status') return makeSwarmStatus() as unknown as Record<string, unknown>
      if (method === 'swarm.board') return { head: 4, messages: makeBoardMessages() }
      if (method === 'swarm.post') return { status: 'posted', seq: 5, channel: String(params.channel ?? 'main') }
      if (method === 'swarm.kill') return { status: 'killed', killed: ['worker-1'] }
      if (method === 'swarm.model') return { status: 'switched', name: 'helper-1', model: 'claude-opus-4-1' }
      if (method === 'swarm.activate') return { ok: true, active: true, prompt_overlay: '<swarm>' }
      return { ok: true }
    }

    const status = await client.swarmStatus({ session_id: 's1' })
    expect(status.title).toBe('Auth Audit Swarm')
    expect(status.members).toHaveLength(5)

    const board = await client.swarmBoard({ session_id: 's1', since: 1, limit: 20, channel: 'security' })
    expect(board.head).toBe(4)
    expect(board.messages).toHaveLength(4)

    const posted = await client.swarmPost({ session_id: 's1', text: ' focus on refresh tokens ', as_feedback: true })
    expect(posted).toEqual({ status: 'posted', seq: 5, channel: 'main' })

    const killed = await client.swarmKill({ session_id: 's1', name: 'worker-1' })
    expect(killed.killed).toEqual(['worker-1'])

    const switched = await client.swarmModel({ session_id: 's1', name: 'helper-1', model: 'claude-opus-4-1' })
    expect(switched).toMatchObject({ status: 'switched', name: 'helper-1', model: 'claude-opus-4-1' })

    const activated = await client.swarmActivate({ session_id: 's1', enabled: true })
    expect(activated).toMatchObject({ ok: true, active: true })

    expect(calls.map(c => c.method)).toEqual([
      'swarm.status',
      'swarm.board',
      'swarm.post',
      'swarm.kill',
      'swarm.model',
      'swarm.activate'
    ])
    expect(calls[1]?.params).toEqual({
      session_id: 's1',
      session_key: 'daemon:s1',
      since: 1,
      limit: 20,
      channel: 'security'
    })
    expect(calls[2]?.params).toEqual({
      session_id: 's1',
      session_key: 'daemon:s1',
      text: ' focus on refresh tokens ',
      as_feedback: true
    })

    // Verify socket event demux -> createGatewayEventHandler -> $swarmStore live update + session isolation.
    patchUiState({ sid: 's1' })
    const emitter = new EventEmitter()
    const fakeSock = Object.assign(emitter, {
      destroy: vi.fn(),
      end: vi.fn(),
      setEncoding: vi.fn(),
      write: vi.fn()
    }) as unknown as Socket
    internals.attachSocket(fakeSock)

    const handler = createGatewayEventHandler({
      composer: { setInput: vi.fn() },
      gateway: { gw: client, rpc: vi.fn().mockResolvedValue(null) },
      session: {
        STARTUP_RESUME_ID: null,
        newSession: vi.fn(),
        recoverSidRef: { current: null },
        resumeById: vi.fn(),
        setCatalog: vi.fn()
      },
      submission: { submitRef: { current: vi.fn() } },
      system: { bellOnComplete: false, stdout: null, sys: vi.fn() },
      transcript: { appendMessage: vi.fn(), panel: vi.fn(), setHistoryItems: vi.fn() },
      voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
    } as unknown as GatewayEventHandlerContext)
    client.on('event', (ev: GatewayEvent) => handler(ev))

    emitter.emit(
      'data',
      `${JSON.stringify({
        jsonrpc: '2.0',
        method: 'event',
        params: {
          type: 'swarm_event',
          payload: {
            session_id: 's1',
            kind: 'message_posted',
            head: 6,
            status: makeSwarmStatus({ head: 6, title: 'Live Updated Swarm' }),
            message: {
              seq: 6,
              agent: 'worker-1',
              channel: 'main',
              text: '@coordinator [done] Completed audit.',
              mentions: ['coordinator'],
              channel_mentions: [],
              broadcast: false,
              time: '2026-11-21 07:06:00'
            }
          }
        }
      })}\n`
    )

    expect(getActiveSwarmSlice().status?.title).toBe('Live Updated Swarm')
    expect(getActiveSwarmSlice().head).toBe(6)
    expect(getActiveSwarmSlice().messages.map(m => m.seq)).toEqual([6])

    // Foreign session event must not clobber s1's active slice.
    emitter.emit(
      'data',
      `${JSON.stringify({
        jsonrpc: '2.0',
        method: 'event',
        params: {
          type: 'swarm_event',
          payload: {
            session_id: 'other-session',
            kind: 'titled',
            head: 99,
            status: makeSwarmStatus({ session_id: 'other-session', head: 99, title: 'Foreign Swarm' })
          }
        }
      })}\n`
    )
    expect(getActiveSwarmSlice().status?.title).toBe('Live Updated Swarm')
    expect(getActiveSwarmSlice().head).toBe(6)
  })

  it('handles /immortals (and /swarm alias) board, status, channels, stop, kill, and <task> without erasing transcript', async () => {
    patchUiState({ sid: 's1' })
    const request = vi.fn(async (method: string) => {
      if (method === 'swarm.status') return makeSwarmStatus()
      if (method === 'swarm.board') return { head: 4, messages: makeBoardMessages() }
      if (method === 'swarm.kill') return { status: 'killed', killed: ['worker-1', 'helper-1'] }
      if (method === 'swarm.activate') return { ok: true, active: true }
      return { ok: true }
    })

    const { context, getHistoryItems, page, resetVisibleHistory, send, setHistoryItems, sys } = makeSlashFixture(request)
    const runSlash = createSlashHandler(context)

    // 1. Bare /immortals opens overlay and fetches status + board without touching transcript.
    runSlash('/immortals')
    await flushMicrotasks()
    expect(getOverlayState().swarm).toBe(true)
    expect(request).toHaveBeenCalledWith('swarm.status', { session_id: 's1' }, { reportError: false })
    expect(request).toHaveBeenCalledWith('swarm.board', { session_id: 's1', limit: 100 }, { reportError: false })
    expect(resetVisibleHistory).not.toHaveBeenCalled()
    expect(setHistoryItems).not.toHaveBeenCalled()
    expect(getHistoryItems()).toEqual([{ role: 'user', text: 'Existing user message' }])

    // 2. Legacy /swarm board alias also opens overlay cleanly.
    patchOverlayState({ swarm: false })
    runSlash('/swarm board')
    await flushMicrotasks()
    expect(getOverlayState().swarm).toBe(true)

    // 3. /immortals status renders formatted status page.
    runSlash('/immortals status')
    await flushMicrotasks()
    expect(page.at(-1)?.title).toBe('Immortals')
    expect(page.at(-1)?.text).toContain('Auth Audit Swarm')
    expect(page.at(-1)?.text).toContain('@helper-1 [waiting] (worker · by @worker-1)')

    // 4. /immortals channels lists channels including [archived].
    runSlash('/immortals channels')
    await flushMicrotasks()
    expect(page.at(-1)?.title).toBe('Immortals Channels')
    expect(page.at(-1)?.text).toContain('#legacy [archived]')

    // 5. /immortals stop kills 'all'; /swarm kill worker-1 kills 'worker-1'.
    runSlash('/immortals stop')
    await flushMicrotasks()
    expect(request).toHaveBeenCalledWith('swarm.kill', { session_id: 's1', name: 'all' })
    expect(sys.at(-1)).toContain('Killed Immortals members: @worker-1, @helper-1')

    runSlash('/immortals kill')
    expect(sys.at(-1)).toBe('usage: /immortals kill <name|all>')

    runSlash('/swarm kill worker-1')
    await flushMicrotasks()
    expect(request).toHaveBeenCalledWith('swarm.kill', { session_id: 's1', name: 'worker-1' })

    // 6. /immortals <task prompt> calls swarm.activate and submits the coordinator task prompt.
    runSlash('/immortals Audit the authentication layer in parallel')
    await flushMicrotasks()
    expect(request).toHaveBeenCalledWith('swarm.activate', { session_id: 's1', enabled: true })
    expect(send).toEqual(['Audit the authentication layer in parallel'])
    expect(resetVisibleHistory).not.toHaveBeenCalled()
  })

  it('completes /immortals (and legacy /swarm) subcommands and live workers without hijacking multi-word task prompts', () => {
    patchUiState({ sid: 's1' })
    applySwarmStatus(makeSwarmStatus(), 's1')

    expect(completionRequestForInput('/immortals bo')).toEqual({
      method: 'complete.slash',
      params: { text: '/immortals bo' },
      replaceFrom: 1
    })
    expect(tuiSlashCompletions('/immortals bo', null).map(c => c.text)).toEqual(['/immortals board'])
    expect(tuiSlashCompletions('/immortals kill ', null).map(c => c.text)).toContain('/immortals kill worker-1')
    expect(tuiSlashCompletions('/swarm bo', null).map(c => c.text)).toEqual(['/swarm board'])
    expect(tuiSlashCompletions('/swarm kill ', null).map(c => c.text)).toContain('/swarm kill worker-1')
    expect(completionRequestForInput('/immortals Refactor all auth handlers across packages')).toBeNull()
    expect(completionRequestForInput('/swarm Refactor all auth handlers across packages')).toBeNull()
  })

  it('formats sidecar://immortals/board and legacy sidecar://swarm/swarm_board Markdown links, tool start summaries, and live SessionHeader badge', async () => {
    const rawMd = 'Immortals launched.\n\n[Open the Immortals UI](sidecar://immortals/board/?auto_open=false)'
    const formatted = formatSwarmMarkdownHint(rawMd)
    expect(formatted.hasSwarmBoardLink).toBe(true)
    expect(formatted.content).toBe('Immortals launched.')

    const legacyFormatted = formatSwarmMarkdownHint('Swarm launched.\n\n[Open the Swarm UI](sidecar://swarm/swarm_board/?auto_open=false)')
    expect(legacyFormatted.hasSwarmBoardLink).toBe(true)
    expect(legacyFormatted.content).toBe('Swarm launched.')

    expect(tokenizeSwarmMessageText('@worker-1 [done] Fixed #3 on @#security @all')).toEqual([
      { kind: 'mention', value: '@worker-1' },
      { kind: 'text', value: ' ' },
      { kind: 'done', value: '[done]' },
      { kind: 'text', value: ' Fixed ' },
      { kind: 'seq_ref', value: '#3' },
      { kind: 'text', value: ' on ' },
      { kind: 'channel_mention', value: '@#security' },
      { kind: 'text', value: ' ' },
      { kind: 'broadcast', value: '@all' }
    ])

    expect(summarizeToolStartDisplay('swarm', '', JSON.stringify({ action: 'add', name: 'auditor', n: 3 }))).toEqual({
      context: 'add @auditor-1..3'
    })
    expect(summarizeToolStartDisplay('read_board', '', JSON.stringify({ since: 12 }))).toEqual({
      context: 'since #12'
    })
    expect(summarizeToolStartDisplay('swarm_wait', '', JSON.stringify({ since: 4, mentions_only: false }))).toEqual({
      context: 'since #4 · all posts'
    })

    patchUiState({ sid: 's1' })
    const screen = await testRender(
      createElement(
        'box',
        { flexDirection: 'column', width: '100%' },
        createElement(SessionHeader, { sessionId: 's1', sessionTitle: 'Audit Chat', t: DEFAULT_THEME }),
        createElement(MessageLine, {
          cols: 100,
          msg: { role: 'assistant', text: rawMd },
          t: DEFAULT_THEME
        })
      ),
      { width: 110, height: 20 }
    )
    try {
      await screen.flush()
      const initialFrame = screen.captureCharFrame()
      expect(initialFrame).toContain(SWARM_BOARD_TUI_HINT)
      expect(initialFrame).not.toContain('sidecar://')
      expect(initialFrame).not.toContain('Auth Audit Swarm')

      act(() => {
        applySwarmStatus(makeSwarmStatus(), 's1')
      })
      await screen.flush()
      const activeFrame = screen.captureCharFrame()
      expect(activeFrame).toContain('Auth Audit Swarm')
      expect(activeFrame).toContain('1 working · 3 waiting · #4 · /immortals')
    } finally {
      act(() => screen.renderer.destroy())
    }
  })

  it.each([
    [130, 38],
    [56, 20]
  ])('renders clean SwarmOverlay views, overview, channels, lineage, current_activity, and #seq messages at %ix%i', async (width, height) => {
    patchUiState({ sid: 's1' })
    patchOverlayState({ swarm: true })
    const rpc = vi.fn(async (method: string) => {
      if (method === 'swarm.status') return makeSwarmStatus()
      if (method === 'swarm.board') return { head: 4, messages: makeBoardMessages() }
      return { ok: true }
    })

    const screen = await testRender(
      createElement(
        GatewayProvider,
        { value: { rpc, gw: new EventEmitter() } as unknown as GatewayServices },
        createElement(SwarmOverlay, { sessionId: 's1', t: DARK_THEME })
      ),
      { width, height }
    )
    try {
      await act(async () => {
        await flushMicrotasks()
      })
      await screen.flush()
      const frame = screen.captureCharFrame()
      expect(frame).toContain('Auth Audit Swarm')
      expect(frame).toContain('1 working · 3 waiting')
      expect(frame).toContain('1 Board')
      expect(frame).toContain('2 Members')
      expect(frame).toContain('3 Overview')
      expect(frame).toContain('#main')
      expect(frame).toContain('#legacy [archived]')
      expect(frame).toContain('#4')
      expect(frame).toContain('[done]')

      if (width >= 120) {
        // Wide >= 120 col split view shows Board + Members + quiet Overview ribbon simultaneously
        expect(frame).toContain('Overview · 2 active slices on track')
        expect(frame).toContain('@worker-1')
        expect(frame).toContain('Reading src/auth/token.ts')
        expect(frame).toContain('← @worker-1')
        expect(frame).toContain('ws:share')
        // Selected member idx 0 is coordinator -> x stop is hidden!
        expect(frame).not.toContain('x stop')
      } else {
        // Narrow/medium < 120 col view renders clean single-pane tabs: switch to 2 Members & 3 Overview
        act(() => screen.mockInput.pressKey('2'))
        await screen.flush()
        expect(screen.captureCharFrame()).toContain('@worker-1')
        expect(screen.captureCharFrame()).toContain('Reading src/auth/token.ts')

        act(() => screen.mockInput.pressKey('3'))
        await screen.flush()
        expect(screen.captureCharFrame()).toContain('Narrator Overview')
      }
    } finally {
      act(() => screen.renderer.destroy())
    }
  })

  it('hides the channel strip when only #main exists', async () => {
    patchUiState({ sid: 's1' })
    patchOverlayState({ swarm: true })
    const mainOnlyMessages = makeBoardMessages().filter(m => m.channel === 'main')
    const rpc = vi.fn(async (method: string) => {
      if (method === 'swarm.status') {
        return makeSwarmStatus({
          channels: [{ name: 'main', topic: 'Main board', created_by: 'coordinator', archived: false, subscriber_count: 5 }]
        })
      }
      if (method === 'swarm.board') return { head: 4, messages: mainOnlyMessages }
      return { ok: true }
    })

    const screen = await testRender(
      createElement(
        GatewayProvider,
        { value: { rpc, gw: new EventEmitter() } as unknown as GatewayServices },
        createElement(SwarmOverlay, { sessionId: 's1', t: DARK_THEME })
      ),
      { width: 110, height: 28 }
    )
    try {
      await act(async () => {
        await flushMicrotasks()
      })
      await screen.flush()
      const frame = screen.captureCharFrame()
      expect(frame).toContain('1 Board (3)')
      expect(frame).not.toContain('#main')
      expect(frame).not.toContain('[archived]')
      expect(frame).not.toContain('Tab/[ ] channel')
    } finally {
      act(() => screen.renderer.destroy())
    }
  })

  it('supports channel tabs, @mention completion, narrator feedback, Enter worker inspector, kill guard, and Esc restoration', async () => {
    patchUiState({ sid: 's1' })
    patchOverlayState({ swarm: true })
    let currentStatus = makeSwarmStatus()
    const messages = makeBoardMessages()

    const rpc = vi.fn(async (method: string, params: Record<string, unknown> = {}) => {
      if (method === 'swarm.status') return currentStatus
      if (method === 'swarm.board') return { head: messages.at(-1)?.seq ?? 0, messages }
      if (method === 'swarm.inspect') {
        return {
          ok: true,
          session_id: 's1',
          member: currentStatus.members.find(m => m.name === params.name) ?? currentStatus.members[1],
          conversation_id: `s1:swarm:${String(params.name)}`,
          activity: {
            phase: 'tool',
            summary: 'Running ReadFile on src/auth/token.ts',
            active_tools: [
              { id: 't1', name: 'ReadFile', summary: 'src/auth/token.ts', status: 'running', started_at: '2026-11-21T07:05:00Z' }
            ],
            recent_tools: [],
            live_text_preview: 'Checking JWT signature verification path'
          },
          messages: [
            { index: 0, role: 'user', text: 'Audit token verification and report back.' },
            { index: 1, role: 'assistant', text: 'Inspecting token verifier now.' }
          ],
          total_messages: 2
        }
      }
      if (method === 'swarm.post') {
        const nextSeq = messages.length + 1
        messages.push({
          seq: nextSeq,
          agent: 'user',
          channel: String(params.channel ?? 'main'),
          text: params.as_feedback ? `@narrator user feedback: ${String(params.text)}` : String(params.text ?? ''),
          mentions: params.as_feedback ? ['narrator'] : [],
          channel_mentions: [],
          broadcast: false,
          time: '2026-11-21 07:08:00'
        })
        currentStatus = { ...currentStatus, head: nextSeq }
        return { status: 'posted', seq: nextSeq, channel: String(params.channel ?? 'main') }
      }
      if (method === 'swarm.kill') {
        const target = String(params.name ?? '')
        currentStatus = {
          ...currentStatus,
          members: currentStatus.members.map(m =>
            m.name === target ? { ...m, state: 'killed', killed: true } : m
          )
        }
        return { status: 'killed', killed: [target] }
      }
      if (method === 'swarm.model') {
        return { status: 'switched', name: params.name, model: params.model }
      }
      return { ok: true }
    })

    const screen = await testRender(
      createElement(
        GatewayProvider,
        { value: { rpc, gw: new EventEmitter() } as unknown as GatewayServices },
        createElement(SwarmOverlay, { sessionId: 's1', t: DARK_THEME })
      ),
      { width: 132, height: 38 }
    )

    const pressEscape = async () => {
      await act(async () => {
        screen.mockInput.pressKey('ESCAPE')
        await new Promise(resolve => setTimeout(resolve, 30))
      })
      await screen.flush()
    }

    try {
      await act(async () => {
        await flushMicrotasks()
      })
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('Auth Audit Swarm')

      // 1. Cycle channel tabs: All (*) -> #main -> #legacy (archived)
      act(() => screen.mockInput.pressKey('TAB')) // -> main
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('1 Board (#main)')
      expect(screen.captureCharFrame()).not.toContain('Found unsigned refresh fallback')

      act(() => screen.mockInput.pressKey(']')) // -> legacy [archived]
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('1 Board (#legacy)')

      // Attempting to post in archived channel warns instead of opening composer
      act(() => screen.mockInput.pressKey('i'))
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('Channel #legacy is archived')

      act(() => screen.mockInput.pressKey('[')) // -> back to #main
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('1 Board (#main)')

      // 2. Inline board compose ('i') with @mention completion hint + Tab completion
      act(() => screen.mockInput.pressKey('i'))
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('Message #main')
      expect(screen.captureCharFrame()).toContain('Mentions: @all')
      await pressEscape()
      expect(getOverlayState().swarm).toBe(true)
      expect(screen.captureCharFrame()).not.toContain('Message #main (Enter post')

      act(() => screen.mockInput.pressKey('i'))
      await screen.flush()
      await act(async () => {
        await screen.mockInput.typeText('@wor')
      })
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('Tab complete: @worker-1')
      act(() => screen.mockInput.pressKey('TAB'))
      await screen.flush()
      await act(async () => {
        await screen.mockInput.typeText('check session expiry')
      })
      await screen.flush()
      await act(async () => {
        screen.mockInput.pressKey('RETURN')
        await flushMicrotasks()
      })
      await screen.flush()
      expect(rpc).toHaveBeenCalledWith(
        'swarm.post',
        { session_id: 's1', text: '@worker-1 check session expiry', channel: 'main' },
        { reportError: false }
      )

      // 3. Narrator feedback ('f') sends as_feedback: true
      act(() => screen.mockInput.pressKey('f'))
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('Feedback to @narrator')
      await act(async () => {
        await screen.mockInput.typeText('group findings by severity')
      })
      await screen.flush()
      await act(async () => {
        screen.mockInput.pressKey('RETURN')
        await flushMicrotasks()
      })
      await screen.flush()
      expect(rpc).toHaveBeenCalledWith(
        'swarm.post',
        { session_id: 's1', text: 'group findings by severity', as_feedback: true },
        { reportError: false }
      )

      // 4. Switch to 2 Members ('2'), move down to worker-1 (idx 1), press Enter to inspect live worker!
      act(() => screen.mockInput.pressKey('2'))
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('› 2 Members')

      act(() => screen.mockInput.pressArrow('down')) // worker-1 (idx 1)
      await screen.flush()
      // x stop is now visible because selected member is worker-1 (not coordinator)
      expect(screen.captureCharFrame()).toContain('x stop')

      await act(async () => {
        screen.mockInput.pressKey('RETURN')
        await flushMicrotasks()
      })
      await screen.flush()
      expect(rpc).toHaveBeenCalledWith(
        'swarm.inspect',
        { session_id: 's1', name: 'worker-1', limit: 60 },
        { reportError: false }
      )
      expect(screen.captureCharFrame()).toContain('← Members · @worker-1')
      expect(screen.captureCharFrame()).toContain('Running ReadFile on src/auth/token.ts')
      expect(screen.captureCharFrame()).toContain('Inspecting token verifier now.')

      // Esc from inspector returns to Members without closing SwarmOverlay!
      await pressEscape()
      expect(getOverlayState().swarm).toBe(true)
      expect(screen.captureCharFrame()).not.toContain('← Members · @worker-1')

      // 5. Kill worker-1 directly with 'x'
      await act(async () => {
        screen.mockInput.pressKey('x')
        await flushMicrotasks()
      })
      await screen.flush()
      expect(rpc).toHaveBeenCalledWith(
        'swarm.kill',
        { session_id: 's1', name: 'worker-1' },
        { reportError: false }
      )

      // 6. Move down to helper-1 (idx 2, waiting) and switch model via 'm'
      act(() => screen.mockInput.pressArrow('down')) // helper-1 (idx 2)
      await screen.flush()
      act(() => screen.mockInput.pressKey('m'))
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('Switch model for @helper-1')
      await act(async () => {
        await screen.mockInput.typeText('claude-opus-4-1')
      })
      await screen.flush()
      await act(async () => {
        screen.mockInput.pressKey('RETURN')
        await flushMicrotasks()
      })
      await screen.flush()
      expect(rpc).toHaveBeenCalledWith(
        'swarm.model',
        { session_id: 's1', name: 'helper-1', model: 'claude-opus-4-1' },
        { reportError: false }
      )

      // 7. Move down to narrator (idx 3): pressing 'x' triggers confirmation guard first!
      act(() => screen.mockInput.pressArrow('down')) // narrator (idx 3)
      await screen.flush()

      const killCallsBeforeGuard = rpc.mock.calls.filter(([m]) => m === 'swarm.kill').length
      act(() => screen.mockInput.pressKey('x'))
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('Confirm killing narrator @narrator?')
      expect(rpc.mock.calls.filter(([m]) => m === 'swarm.kill')).toHaveLength(killCallsBeforeGuard)

      // Esc cancels guard without closing SwarmOverlay
      await pressEscape()
      expect(getOverlayState().swarm).toBe(true)
      expect(screen.captureCharFrame()).not.toContain('Confirm killing narrator @narrator?')

      // Trigger guard again and confirm with 'y'
      act(() => screen.mockInput.pressKey('x'))
      await screen.flush()
      await act(async () => {
        screen.mockInput.pressKey('y')
        await flushMicrotasks()
      })
      await screen.flush()
      expect(rpc).toHaveBeenCalledWith(
        'swarm.kill',
        { session_id: 's1', name: 'narrator' },
        { reportError: false }
      )

      // 8. Final Esc closes SwarmOverlay cleanly restoring transcript view
      await pressEscape()
      expect(getOverlayState().swarm).toBe(false)
    } finally {
      act(() => screen.renderer.destroy())
    }
  })

  it('scrolls long chronological message boards smoothly with Up/Down, Ctrl+B/Ctrl+F paging, Home, and End', async () => {
    patchUiState({ sid: 's1' })
    patchOverlayState({ swarm: true })
    const manyMessages: SwarmBoardMessage[] = Array.from({ length: 24 }, (_, idx) => ({
      seq: idx + 1,
      agent: idx % 2 === 0 ? 'worker-1' : 'helper-1',
      channel: 'main',
      text: `@coordinator Progress update item-${idx + 1}`,
      mentions: ['coordinator'],
      channel_mentions: [],
      broadcast: false,
      time: '2026-11-21 07:10:00'
    }))

    const rpc = vi.fn(async (method: string) => {
      if (method === 'swarm.status') return makeSwarmStatus({ head: 24, overview: null })
      if (method === 'swarm.board') return { head: 24, messages: manyMessages }
      return { ok: true }
    })

    const screen = await testRender(
      createElement(
        GatewayProvider,
        { value: { rpc, gw: new EventEmitter() } as unknown as GatewayServices },
        createElement(SwarmOverlay, { sessionId: 's1', t: DARK_THEME })
      ),
      { width: 120, height: 26 }
    )

    try {
      await act(async () => {
        await flushMicrotasks()
      })
      await screen.flush()
      // Initially pinned to newest message #24 on 1 Board
      expect(screen.captureCharFrame()).toContain('Progress update item-24')

      // Page up via Ctrl+B (laptop PgUp equivalent)
      act(() => screen.mockInput.pressKey('b', { ctrl: true }))
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('(↑ scrolled)')

      // Jump to oldest via Home, then back to tail via End
      act(() => screen.mockInput.pressKey('HOME'))
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('Progress update item-1')

      act(() => screen.mockInput.pressKey('END'))
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('Progress update item-24')
      expect(screen.captureCharFrame()).not.toContain('(↑ scrolled)')
    } finally {
      act(() => screen.renderer.destroy())
    }
  })

  it('wires GatewayClient.sideQuestion, GatewayClient.swarmInspect, live btw_event streaming, and /btw + /side slash commands without touching main transcript', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'tui:tab-1' })
    const calls: Array<{ method: string; params: Record<string, unknown> }> = []
    const internals = client as unknown as {
      rememberSessionKey: (id: string, key: string) => void
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
      attachSocket: (s: Socket) => void
    }
    internals.rememberSessionKey('s1', 'daemon:s1')
    internals.rawRequest = async (method, params = {}) => {
      calls.push({ method, params })
      if (method === 'session.side_question') {
        return {
          ok: true,
          request_id: String(params.request_id ?? 'req-1'),
          session_id: 's1',
          question: String(params.question ?? ''),
          answer: 'normalizeEventType converts PascalCase bridge aliases to snake_case.',
          model: 'claude-sonnet-4-5'
        }
      }
      if (method === 'swarm.inspect') {
        return {
          ok: true,
          session_id: 's1',
          member: makeSwarmStatus().members[1],
          conversation_id: 's1:swarm:worker-1',
          activity: { phase: 'thinking', summary: 'Thinking…', active_tools: [], recent_tools: [] },
          messages: [{ index: 0, role: 'user', text: 'Audit auth' }],
          total_messages: 1
        }
      }
      return { ok: true }
    }

    const sideRes = await client.sideQuestion({
      session_id: 's1',
      request_id: 'req-42',
      question: 'What does normalizeEventType do?',
      history: [{ role: 'user', content: 'Earlier q' }, { role: 'assistant', content: 'Earlier a' }]
    })
    expect(sideRes).toEqual({
      ok: true,
      request_id: 'req-42',
      session_id: 's1',
      question: 'What does normalizeEventType do?',
      answer: 'normalizeEventType converts PascalCase bridge aliases to snake_case.',
      model: 'claude-sonnet-4-5'
    })

    const inspected = await client.swarmInspect({ session_id: 's1', name: 'worker-1', limit: 40 })
    expect(inspected.member.name).toBe('worker-1')
    expect(inspected.activity.phase).toBe('thinking')
    expect(inspected.total_messages).toBe(1)

    expect(calls[0]).toEqual({
      method: 'session.side_question',
      params: {
        session_id: 's1',
        session_key: 'daemon:s1',
        request_id: 'req-42',
        question: 'What does normalizeEventType do?',
        history: [{ role: 'user', content: 'Earlier q' }, { role: 'assistant', content: 'Earlier a' }]
      }
    })
    expect(calls[1]).toEqual({
      method: 'swarm.inspect',
      params: {
        session_id: 's1',
        session_key: 'daemon:s1',
        name: 'worker-1',
        limit: 40
      }
    })

    // Socket btw_event stream updates $sideQuestionStore without touching transcript!
    patchUiState({ sid: 's1', busy: true })
    const emitter = new EventEmitter()
    const fakeSock = Object.assign(emitter, {
      destroy: vi.fn(),
      end: vi.fn(),
      setEncoding: vi.fn(),
      write: vi.fn()
    }) as unknown as Socket
    internals.attachSocket(fakeSock)

    const appendMessage = vi.fn()
    const setHistoryItems = vi.fn()
    const handler = createGatewayEventHandler({
      composer: { setInput: vi.fn() },
      gateway: { gw: client, rpc: vi.fn().mockResolvedValue(null) },
      session: {
        STARTUP_RESUME_ID: null,
        newSession: vi.fn(),
        recoverSidRef: { current: null },
        resumeById: vi.fn(),
        setCatalog: vi.fn()
      },
      submission: { submitRef: { current: vi.fn() } },
      system: { bellOnComplete: false, stdout: null, sys: vi.fn() },
      transcript: { appendMessage, panel: vi.fn(), setHistoryItems },
      voice: { setProcessing: vi.fn(), setRecording: vi.fn(), setVoiceEnabled: vi.fn() }
    } as unknown as GatewayEventHandlerContext)
    client.on('event', (ev: GatewayEvent) => handler(ev))

    for (const payload of [
      { session_id: 's1', request_id: 'live-1', kind: 'begin', question: 'Why is turnActive unchanged?', model: 'claude-sonnet-4-5' },
      { session_id: 's1', request_id: 'live-1', kind: 'delta', delta: 'Because side questions ' },
      { session_id: 's1', request_id: 'live-1', kind: 'delta', delta: 'fork a read-only snapshot.' },
      { session_id: 's1', request_id: 'live-1', kind: 'end', answer: 'Because side questions fork a read-only snapshot.', model: 'claude-sonnet-4-5' }
    ]) {
      emitter.emit('data', `${JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type: 'btw_event', payload } })}\n`)
    }

    expect(getActiveSideQuestionSlice().entries).toEqual([
      expect.objectContaining({
        requestId: 'live-1',
        question: 'Why is turnActive unchanged?',
        answer: 'Because side questions fork a read-only snapshot.',
        model: 'claude-sonnet-4-5',
        status: 'done'
      })
    ])
    expect(appendMessage).not.toHaveBeenCalled()
    expect(setHistoryItems).not.toHaveBeenCalled()

    // Slash commands: bare /btw opens overlay; /btw <q> and /side <q> call session.side_question with history!
    resetSideQuestionStore()
    resetOverlayState()
    const slashCalls: Array<{ method: string; params: Record<string, unknown> }> = []
    const slashRpc = vi.fn(async (method: string, params: Record<string, unknown> = {}) => {
      slashCalls.push({ method, params })
      return {
        ok: true,
        request_id: String(params.request_id ?? 'r1'),
        session_id: 's1',
        question: String(params.question ?? ''),
        answer: `Answered: ${String(params.question ?? '')}`,
        model: 'gpt-5-mini'
      }
    })
    const slashFixture = makeSlashFixture(slashRpc)
    const runSlash = createSlashHandler(slashFixture.context)

    runSlash('/btw')
    expect(getOverlayState().sideQuestion).toBe(true)
    expect(slashCalls).toHaveLength(0)
    expect(slashFixture.resetVisibleHistory).not.toHaveBeenCalled()
    expect(slashFixture.setHistoryItems).not.toHaveBeenCalled()

    runSlash('/btw What is QUEUE_BYPASS_METHODS?')
    await flushMicrotasks()
    expect(slashCalls).toHaveLength(1)
    expect(slashCalls[0]?.method).toBe('session.side_question')
    expect(slashCalls[0]?.params).toMatchObject({
      session_id: 's1',
      question: 'What is QUEUE_BYPASS_METHODS?'
    })

    // Follow-up via /side alias automatically includes prior Q&A in history!
    runSlash('/side Is it read-only?')
    await flushMicrotasks()
    expect(slashCalls).toHaveLength(2)
    expect(slashCalls[1]?.params).toMatchObject({
      session_id: 's1',
      question: 'Is it read-only?',
      history: [
        { role: 'user', content: 'What is QUEUE_BYPASS_METHODS?' },
        { role: 'assistant', content: 'Answered: What is QUEUE_BYPASS_METHODS?' }
      ]
    })
    expect(slashFixture.getHistoryItems()).toEqual([{ role: 'user', text: 'Existing user message' }])
  })

  it('renders SideQuestionOverlay with live btw_event deltas, interactive follow-ups, c to clear, and non-destructive Esc while busy', async () => {
    patchUiState({ sid: 's1', busy: true })
    patchOverlayState({ sideQuestion: true })

    const gwEmitter = new EventEmitter()
    let resolveRpc: ((value: Record<string, unknown>) => void) | null = null
    const rpc = vi.fn((method: string, params: Record<string, unknown> = {}) => {
      if (method === 'session.side_question') {
        const reqId = String(params.request_id ?? 'req-stream')
        queueMicrotask(() => {
          gwEmitter.emit('btw_event', {
            type: 'btw_event',
            payload: {
              session_id: 's1',
              request_id: reqId,
              kind: 'begin',
              question: String(params.question ?? '')
            }
          })
          gwEmitter.emit('btw_event', {
            type: 'btw_event',
            payload: {
              session_id: 's1',
              request_id: reqId,
              kind: 'delta',
              delta: 'Streaming live aside delta…'
            }
          })
        })
        return new Promise<Record<string, unknown>>(resolve => {
          resolveRpc = resolve
        })
      }
      return Promise.resolve({ ok: true })
    })

    const screen = await testRender(
      createElement(
        GatewayProvider,
        { value: { rpc, gw: gwEmitter } as unknown as GatewayServices },
        createElement(SideQuestionOverlay, { sessionId: 's1', t: DARK_THEME })
      ),
      { width: 100, height: 28 }
    )

    try {
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('Side Question')
      expect(screen.captureCharFrame()).toContain('main turn running')

      // Type initial question in bare /btw overlay and submit with Enter
      await act(async () => {
        await screen.mockInput.typeText('Where are slash commands defined?')
      })
      await screen.flush()
      await act(async () => {
        screen.mockInput.pressKey('RETURN')
        await flushMicrotasks()
      })
      await screen.flush()

      const waitForText = async (needle: string): Promise<string> => {
        for (let pass = 0; pass < 40; pass += 1) {
          await screen.flush()
          const frame = screen.captureCharFrame()
          if (frame.includes(needle)) return frame
          await new Promise(resolve => setTimeout(resolve, 15))
        }
        return screen.captureCharFrame()
      }

      expect(await waitForText('Where are slash commands defined?')).toContain('Where are slash commands defined?')
      expect(await waitForText('Streaming live aside delta…')).toContain('Streaming live aside delta…')

      // Settle RPC response
      await act(async () => {
        resolveRpc?.({
          ok: true,
          request_id: String(rpc.mock.calls[0]?.[1]?.request_id ?? ''),
          session_id: 's1',
          question: 'Where are slash commands defined?',
          answer: 'In xerxes/src/bridge/commands.ts.',
          model: 'claude-sonnet-4-5'
        })
        await flushMicrotasks()
      })
      expect(await waitForText('In xerxes/src/bridge/commands.ts.')).toContain('In xerxes/src/bridge/commands.ts.')

      // Press 'c' to clear the ephemeral thread
      act(() => screen.mockInput.pressKey('c'))
      await screen.flush()
      expect(getActiveSideQuestionSlice().entries).toEqual([])
      expect(screen.captureCharFrame()).toContain('Ephemeral side thread')

      // Press Esc: closes SideQuestionOverlay ONLY without calling turn.cancel or mutating busy state!
      await act(async () => {
        screen.mockInput.pressKey('ESCAPE')
        await new Promise(resolve => setTimeout(resolve, 30))
      })
      await screen.flush()
      expect(getOverlayState().sideQuestion).toBe(false)
      expect(rpc.mock.calls.some(([method]) => method === 'turn.cancel' || method === 'cancel')).toBe(false)
    } finally {
      act(() => screen.renderer.destroy())
    }
  })
})
