// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import type { GatewayEventHandlerContext } from '../app/interfaces.js'
import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { turnController } from '../app/turnController.js'
import { getTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
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

  it('accumulates live LLM timing without mixing a final round with cumulative TTFT', () => {
    const { handler } = buildHarness()
    patchUiState({
      usage: {
        calls: 11,
        input: 340_000,
        llm_ms: 180_000,
        llm_steps: 11,
        output: 6_000,
        total: 346_000,
        ttft_avg_ms: 16_000,
        ttft_samples: 11,
        ttft_total_ms: 176_000
      }
    })

    handler({
      payload: {
        telemetry_delta: { llm_ms: 12_000, ttft_ms: 16_100 },
        text: 'working',
        usage: { calls: 12, input: 353_500, output: 6_900, tok_per_sec: 52, total: 360_400 }
      },
      type: 'status.update'
    })

    expect(getUiState().usage).toMatchObject({
      calls: 12,
      input: 353_500,
      llm_ms: 192_000,
      llm_steps: 12,
      output: 6_900,
      tok_per_sec: 52,
      ttft_avg_ms: 16_008.333333333334,
      ttft_samples: 12,
      ttft_total_ms: 192_100
    })
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

  it('isolates background deltas while still clearing completed task chrome', () => {
    const { appended, handler, sys } = buildHarness()
    patchUiState({ bgTasks: new Set(['bg-1']), sid: 'foreground' })

    handler({ payload: { text: 'background text' }, session_id: 'bg-1', type: 'message.delta' })
    handler({ payload: { task_id: 'bg-1', text: 'finished' }, type: 'background.complete' })

    expect(appended).toEqual([])
    expect(getTurnState().streaming).toBe('')
    expect(getUiState().bgTasks.has('bg-1')).toBe(false)
    expect(sys).toHaveBeenCalledWith('[bg bg-1] finished')
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

  it('pairs a subagent tool result with its call so the inspector can show a duration', () => {
    const { handler } = buildHarness()
    const base = { depth: 0, goal: 'audit policy', subagent_id: 'child-1', task_index: 0 }

    handler({ payload: { ...base, status: 'running' }, type: 'subagent.start' } as GatewayEvent)
    handler({
      payload: { ...base, tool_call_id: 'call-1', tool_name: 'ReadFile', tool_preview: '{"file_path":"src/auth.ts"}' },
      type: 'subagent.tool'
    } as GatewayEvent)
    handler({
      payload: { ...base, tool_call_id: 'call-1', tool_duration_ms: 1_250, tool_name: 'ReadFile', tool_ok: true },
      type: 'subagent.tool_result'
    } as GatewayEvent)

    const [agent] = getTurnState().subagents
    const [call] = agent?.toolCalls ?? []

    expect(agent?.toolCalls).toHaveLength(1)
    expect(call).toMatchObject({
      id: 'call-1',
      name: 'ReadFile',
      ok: true,
      preview: '{"file_path":"src/auth.ts"}'
    })
    expect(agent?.tools.at(-1)).toContain('src/auth.ts')
    expect(agent?.tools.at(-1)).not.toContain('{"file_path"')
    // The duration is reported by the daemon, so it is used verbatim rather
    // than measured against this client's clock.
    expect((call?.endedAt ?? 0) - (call?.startedAt ?? 0)).toBe(1_250)
  })

  it('keeps a denied tool call visible as a failure rather than dropping it', () => {
    const { handler } = buildHarness()
    const base = { depth: 0, goal: 'audit policy', subagent_id: 'child-2', task_index: 0 }

    handler({ payload: { ...base, status: 'running' }, type: 'subagent.start' } as GatewayEvent)
    handler({ payload: { ...base, tool_name: 'ExecCommand' }, type: 'subagent.tool' } as GatewayEvent)
    handler({
      payload: { ...base, tool_name: 'ExecCommand', tool_ok: false },
      type: 'subagent.tool_result'
    } as GatewayEvent)

    const agent = getTurnState().subagents.find(item => item.id === 'child-2')

    // Matched by name when the provider sent no tool_call_id, rather than
    // recorded twice as one call that never finished plus a stray result.
    expect(agent?.toolCalls).toHaveLength(1)
    expect(agent?.toolCalls?.[0]).toMatchObject({ name: 'ExecCommand', ok: false })
    expect(agent?.toolCalls?.[0]?.endedAt).toBeDefined()
  })
})
