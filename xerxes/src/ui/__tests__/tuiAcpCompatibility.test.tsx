// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act, useRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createGatewayEventHandler } from '../app/createGatewayEventHandler.js'
import { createSlashHandler } from '../app/createSlashHandler.js'
import type { ComposerActions, ComposerRefs, ComposerState, GatewayEventHandlerContext } from '../app/interfaces.js'
import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { ACP_SUBCOMMANDS } from '../app/slash/commands/acp.js'
import { findSlashCommand } from '../app/slash/registry.js'
import { turnController } from '../app/turnController.js'
import { getTurnState } from '../app/turnStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useSubmission } from '../app/useSubmission.js'
import type { QueuedMessage } from '../domain/queuedMessage.js'
import { adaptDaemonEvent, sessionInfoFromInit } from '../gatewayAdapter.js'
import { GatewayClient } from '../gatewayClient.js'
import type { AcpSessionBinding, GatewayEvent } from '../gatewayTypes.js'
import { completionRequestForInput, tuiSlashCompletions } from '../hooks/useCompletion.js'
import { PromptZone } from '../opentui/appLayout.js'
import type { Msg } from '../types.js'

const makeAntigravityBinding = (overrides: Partial<AcpSessionBinding> = {}): AcpSessionBinding => ({
  session_id: 'session-a',
  agent_id: 'antigravity',
  name: 'Antigravity',
  provider: 'antigravity',
  transport: 'acp',
  cwd: '/repo',
  connected: true,
  current_model: 'gemini-3-pro',
  current_model_name: 'Gemini 3 Pro',
  models: [
    { id: 'gemini-3-pro', name: 'Gemini 3 Pro', description: 'Reasoning model' },
    { id: 'gemini-3-flash', name: 'Gemini 3 Flash', description: 'Fast model' }
  ],
  current_mode: 'auto',
  current_mode_name: 'Auto',
  modes: [
    { id: 'auto', name: 'Auto' },
    { id: 'plan', name: 'Plan' }
  ],
  config_options: [
    {
      id: 'auto_save',
      name: 'Auto Save',
      type: 'boolean',
      current_value: true,
      options: []
    },
    {
      id: 'model',
      name: 'Model',
      category: 'model',
      type: 'select',
      current_value: 'gemini-3-pro',
      options: [
        { value: 'gemini-3-pro', name: 'Gemini 3 Pro', group: 'Google' },
        { value: 'gemini-3-flash', name: 'Gemini 3 Flash', group: 'Google' }
      ]
    }
  ],
  capabilities: {
    prompt: { image: true, audio: false, embedded_context: true },
    session: { load: true, resume: true, list: true, fork: true, close: true, additional_directories: true },
    mcp: { http: true, sse: false },
    auth_methods: [
      {
        id: 'oauth-personal',
        name: 'Google OAuth',
        description: 'Interactive login',
        type: 'terminal',
        args: ['acp', '--login'],
        env: { AGY_AUTH: '1' },
        meta: { interactive: true }
      }
    ]
  },
  ...overrides
})

const buildEventHandlerHarness = () => {
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
      bellOnComplete: false,
      stdout: { isTTY: false } as NodeJS.WriteStream,
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

async function mountSubmission(request: ReturnType<typeof vi.fn>) {
  const sys = vi.fn()
  const messages: Msg[] = []
  const queueRef = { current: [] as QueuedMessage[] }
  const composerActions: ComposerActions = {
    activateSessionQueue: vi.fn(),
    clearIn: vi.fn(),
    dequeue: vi.fn(),
    dismissCompletions: vi.fn(),
    enqueue: vi.fn((submitText: string, displayText?: string) => {
      queueRef.current.push({
        displayText: displayText ?? submitText,
        submissionId: `sub-${queueRef.current.length + 1}`,
        submitText
      })
    }),
    handleTextPaste: vi.fn(),
    holdForSwitch: vi.fn(),
    openEditor: vi.fn(async () => {}),
    pushHistory: vi.fn(),
    releaseSwitchHold: vi.fn(),
    removeQueue: vi.fn(),
    replaceQueue: vi.fn(),
    setCompIdx: vi.fn(),
    setHistoryIdx: vi.fn(),
    setInput: vi.fn(),
    setInputBuf: vi.fn(),
    setPasteSnips: vi.fn(),
    setQueueEdit: vi.fn(),
    syncQueue: vi.fn()
  }
  const composerState: ComposerState = {
    compIdx: 0,
    compReplace: 0,
    completions: [],
    historyIdx: null,
    input: '',
    inputBuf: [],
    pasteSnips: [],
    queueEditIdx: null,
    queuedDisplay: []
  }

  let submission: ReturnType<typeof useSubmission> | undefined
  function Harness() {
    const composerRefs: ComposerRefs = {
      historyDraftRef: useRef(''),
      historyRef: useRef<string[]>([]),
      queueEditRef: useRef<null | number>(null),
      queueRef,
      submitRef: useRef(() => {})
    }
    submission = useSubmission({
      appendMessage: msg => messages.push(msg),
      composerActions,
      composerRefs,
      composerState,
      gw: { request } as unknown as GatewayClient,
      maybeGoodVibes: vi.fn(),
      removeMessage: msg => {
        const index = messages.indexOf(msg)
        if (index >= 0) messages.splice(index, 1)
      },
      setLastUserMsg: vi.fn(),
      slashRef: useRef(() => false),
      submitRef: useRef(() => {}),
      sys
    })
    return null
  }

  const rendered = await testRender(<Harness />, { height: 10, width: 40 })
  await rendered.flush()
  return { messages, rendered, submission: submission!, sys }
}

const makeSlashContext = (request: ReturnType<typeof vi.fn>) => {
  const sys: string[] = []
  const page: Array<{ text: string; title?: string }> = []
  const send: string[] = []
  const resetVisibleHistory = vi.fn()
  const setHistoryItems = vi.fn()
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
      gateway: { gw: { request }, rpc: request },
      local: {
        catalog: null,
        getHistoryItems: vi.fn(() => []),
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
    page,
    resetVisibleHistory,
    send,
    setHistoryItems,
    sys
  }
}

afterEach(() => {
  turnController.fullReset()
  resetOverlayState()
  resetUiState()
  vi.restoreAllMocks()
})

describe('TUI !cmd shell follow-up guard on Antigravity ACP', () => {
  it('runs !cmd locally and displays output without submitting synthetic prompt.submit when Antigravity ACP is active', async () => {
    const request = vi.fn(async (method: string) =>
      method === 'shell.exec' ? { code: 0, stdout: 'README.md\nsrc', stderr: '' } : { ok: true }
    )
    patchUiState({
      acpBinding: makeAntigravityBinding(),
      busy: false,
      sid: 'session-a'
    })
    const fixture = await mountSubmission(request)
    try {
      await act(async () => {
        fixture.submission.dispatchSubmission('!ls')
        await Bun.sleep(0)
      })
      await fixture.rendered.flush()

      expect(request).toHaveBeenCalledTimes(1)
      expect(request).toHaveBeenCalledWith('shell.exec', { command: 'ls' })
      expect(request).not.toHaveBeenCalledWith('prompt.submit', expect.anything())
      expect(fixture.messages).toContainEqual({ role: 'user', text: '!ls' })
      expect(fixture.sys).toHaveBeenCalledWith('README.md\nsrc')
      expect(getUiState()).toMatchObject({ busy: false, status: 'ready' })
    } finally {
      act(() => fixture.rendered.renderer.destroy())
    }
  })

  it('keeps existing !cmd follow-up behavior on native Xerxes sessions', async () => {
    const request = vi.fn(async (method: string) =>
      method === 'shell.exec' ? { code: 0, stdout: 'README.md', stderr: '' } : { ok: true }
    )
    patchUiState({ acpBinding: null, busy: false, sid: 'session-a' })
    const fixture = await mountSubmission(request)
    try {
      await act(async () => {
        fixture.submission.dispatchSubmission('!ls')
        await Bun.sleep(0)
      })
      await fixture.rendered.flush()

      expect(request).toHaveBeenCalledTimes(2)
      expect(request).toHaveBeenLastCalledWith(
        'prompt.submit',
        expect.objectContaining({ session_id: 'session-a', text: expect.stringContaining('stdout:\nREADME.md') })
      )
      expect(getUiState().busy).toBe(true)
    } finally {
      act(() => fixture.rendered.renderer.destroy())
    }
  })
})

describe('TUI multi-question, multi_select, and URL elicitation clarify flow', () => {
  it('preserves all questions 1..N, multi_select, URL mode, and question_resolved in gatewayAdapter + event handler', () => {
    const { appended, handler } = buildEventHandlerHarness()
    const events = adaptDaemonEvent('question_request', {
      id: 'elicit-req-1',
      session_id: 'session-a',
      questions: [
        { id: 'env', question: 'Target environment?', options: ['staging', 'prod'], allow_free_form: false },
        { id: 'features', question: 'Enable modules', options: ['auth', 'metrics', 'tracing'], multi_select: true, allow_free_form: false }
      ]
    })
    for (const ev of events) handler(ev as GatewayEvent)

    const clarify = getOverlayState().clarify
    expect(clarify).toMatchObject({
      requestId: 'elicit-req-1:env',
      question: 'Target environment?',
      questions: [
        { id: 'env', question: 'Target environment?', choices: ['staging', 'prod'], allowFreeform: false },
        { id: 'features', question: 'Enable modules', choices: ['auth', 'metrics', 'tracing'], multiSelect: true, allowFreeform: false }
      ]
    })

    // URL mode elicitation + auto-dismiss on question_resolved (elicitation/complete)
    const urlEvents = adaptDaemonEvent('question_request', {
      id: 'url-req-1',
      session_id: 'session-a',
      mode: 'url',
      url: 'https://example.com/oauth/device?code=WXYZ',
      elicitation_id: 'elicit-42',
      message: 'Authorize your account in the browser',
      questions: [
        {
          id: 'url_consent',
          question: 'Authorize your account in the browser',
          options: ['Open link & continue', 'Decline'],
          allow_free_form: false
        }
      ]
    })
    for (const ev of urlEvents) handler(ev as GatewayEvent)

    expect(getOverlayState().clarify).toMatchObject({
      mode: 'url',
      url: 'https://example.com/oauth/device?code=WXYZ',
      elicitationId: 'elicit-42',
      message: 'Authorize your account in the browser',
      choices: ['Open link & continue', 'Decline'],
      allowFreeform: false
    })

    const resolvedEvents = adaptDaemonEvent('question_resolved', {
      id: 'url-req-1',
      elicitation_id: 'elicit-42',
      reason: 'complete'
    })
    for (const ev of resolvedEvents) handler(ev as GatewayEvent)

    expect(getOverlayState().clarify).toBeNull()
    expect(getUiState().status).toBe('running…')
    expect(appended).toContainEqual({ role: 'system', text: 'External elicitation completed.' })
  })

  it('steps through multi-question forms with multi_select toggles in PromptZone and submits { [q.id]: answer }', async () => {
    const answerClarify = vi.fn()
    act(() =>
      patchOverlayState({
        clarify: {
          allowFreeform: false,
          choices: ['staging', 'prod'],
          question: 'Target environment?',
          questionId: 'env',
          questions: [
            { id: 'env', question: 'Target environment?', choices: ['staging', 'prod'], allowFreeform: false },
            {
              id: 'features',
              question: 'Select modules',
              choices: ['auth', 'metrics', 'tracing'],
              multiSelect: true,
              allowFreeform: false
            }
          ],
          requestId: 'elicit-req-1:env',
          source: 'agent'
        }
      })
    )
    const session = await testRender(<PromptZone actions={{ answerClarify } as never} />, { height: 24, width: 90 })
    await session.flush()
    try {
      const step1Frame = session.captureCharFrame()
      expect(step1Frame).toContain('Question 1 of 2')
      expect(step1Frame).toContain('Target environment?')
      expect(step1Frame).not.toContain('Other (type your answer)')

      // Pick 'prod' (option 2) -> advances to step 2 without submitting yet
      act(() => session.mockInput.pressKey('2'))
      await session.flush()
      expect(answerClarify).not.toHaveBeenCalled()

      const step2Frame = session.captureCharFrame()
      expect(step2Frame).toContain('Question 2 of 2')
      expect(step2Frame).toContain('Select modules')
      expect(step2Frame).toContain('[ ] auth')

      // Toggle 'auth' (1) and 'tracing' (3), then press Enter to confirm
      act(() => session.mockInput.pressKey('1'))
      await session.flush()
      expect(session.captureCharFrame()).toContain('[x] auth')

      act(() => session.mockInput.pressKey('3'))
      await session.flush()
      expect(session.captureCharFrame()).toContain('[x] tracing')

      act(() => session.mockInput.pressEnter())
      await session.flush()

      expect(answerClarify).toHaveBeenCalledTimes(1)
      expect(answerClarify).toHaveBeenCalledWith({
        env: 'prod',
        features: ['auth', 'tracing']
      })
    } finally {
      session.renderer.destroy()
    }
  })

  it('renders URL elicitation visibly with Accept/Continue vs Decline options in PromptZone', async () => {
    const answerClarify = vi.fn()
    act(() =>
      patchOverlayState({
        clarify: {
          allowFreeform: false,
          choices: ['Open link & continue', 'Decline'],
          elicitationId: 'elicit-9',
          message: 'Complete OAuth device flow',
          mode: 'url',
          question: 'Complete OAuth device flow',
          questionId: 'url_consent',
          requestId: 'url-req-9:url_consent',
          source: 'agent',
          url: 'https://accounts.example.com/device?code=1234-ABCD'
        }
      })
    )
    const session = await testRender(<PromptZone actions={{ answerClarify } as never} />, { height: 24, width: 96 })
    await session.flush()
    try {
      const frame = session.captureCharFrame()
      expect(frame).toContain('Complete OAuth device flow')
      expect(frame).toContain('URL: https://accounts.example.com/device?code=1234-ABCD')
      expect(frame).toContain('Open link & continue')
      expect(frame).toContain('Decline')
      expect(frame).not.toContain('Other (type your answer)')

      act(() => session.mockInput.pressKey('2'))
      expect(answerClarify).toHaveBeenCalledWith('Decline')
    } finally {
      session.renderer.destroy()
    }
  })

  it('sends multi-question answers map and multi_select string[] arrays via GatewayClient clarify.respond', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'tui:test' })
    const calls: Array<{ method: string; params: Record<string, unknown> }> = []
    ;(client as unknown as { rawRequest: (m: string, p?: Record<string, unknown>) => Promise<unknown> }).rawRequest =
      async (method, params = {}) => {
        calls.push({ method, params })
        return { ok: true }
      }

    await client.request('clarify.respond', {
      request_id: 'elicit-req-1:env',
      answers: { env: 'prod', features: ['auth', 'tracing'] }
    })
    await client.request('clarify.respond', {
      request_id: 'single-multi:mods',
      answer: ['lint', 'test']
    })

    expect(calls[0]).toEqual({
      method: 'question_response',
      params: {
        request_id: 'elicit-req-1',
        answers: { env: 'prod', features: ['auth', 'tracing'] }
      }
    })
    expect(calls[1]).toEqual({
      method: 'question_response',
      params: {
        request_id: 'single-multi',
        answers: { mods: ['lint', 'test'] }
      }
    })
  })
})

describe('TUI ACP gateway client, events, tool updates, and granular approval overlay', () => {
  it('supports boolean configOptions, acpAuthenticate, acpSessions, acpFork, and exact option_id permission responses', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'tui:acp' })
    const calls: Array<{ method: string; params: Record<string, unknown> }> = []
    const internals = client as unknown as {
      rememberSessionKey: (id: string, key: string) => void
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }
    internals.rememberSessionKey('s1', 'daemon:s1')
    internals.rawRequest = async (method, params = {}) => {
      calls.push({ method, params })
      if (method === 'acp.set_config_option') return { ok: true, binding: makeAntigravityBinding() }
      if (method === 'acp.authenticate') {
        return {
          ok: true,
          mode: 'terminal',
          command: 'agy',
          args: ['acp', '--login'],
          env: { AGY_AUTH: '1' },
          label: 'Antigravity Login'
        }
      }
      if (method === 'acp.sessions') {
        return {
          ok: true,
          sessions: [{ session_id: 'ext-1', cwd: '/repo', title: 'Audit' }],
          next_cursor: 'cursor-2'
        }
      }
      if (method === 'acp.fork') {
        return { ok: true, binding: makeAntigravityBinding({ external_session_id: 'ext-fork-1' }) }
      }
      return { ok: true }
    }

    const boolRes = await client.acpSetConfigOption('auto_save', true)
    expect(boolRes.ok).toBe(true)
    expect(calls[0]).toEqual({
      method: 'acp.set_config_option',
      params: { session_key: 'tui:acp', config_id: 'auto_save', value: true, type: 'boolean' }
    })

    await client.acpSetConfigOption({ session_id: 's1', config_id: 'model', value: 'gemini-3-flash' })
    expect(calls[1]).toEqual({
      method: 'acp.set_config_option',
      params: { session_id: 's1', session_key: 'daemon:s1', config_id: 'model', value: 'gemini-3-flash' }
    })

    const authRes = await client.acpAuthenticate('oauth-personal', 'antigravity')
    expect(authRes).toEqual({
      ok: true,
      mode: 'terminal',
      command: 'agy',
      args: ['acp', '--login'],
      env: { AGY_AUTH: '1' },
      label: 'Antigravity Login'
    })
    expect(calls[2]).toEqual({
      method: 'acp.authenticate',
      params: { session_key: 'tui:acp', method_id: 'oauth-personal', agent_id: 'antigravity' }
    })

    const sessionsRes = await client.acpSessions({ session_id: 's1', cwd: '/repo', cursor: 'cursor-1' })
    expect(sessionsRes).toEqual({
      ok: true,
      sessions: [{ session_id: 'ext-1', cwd: '/repo', title: 'Audit' }],
      next_cursor: 'cursor-2'
    })

    const forkRes = await client.acpFork({ session_id: 's1', cwd: '/repo' })
    expect(forkRes.ok).toBe(true)
    expect(forkRes.binding?.external_session_id).toBe('ext-fork-1')

    await client.respondToPermission('perm-77', 'allow_always', 'opt-allow-repo', 's1')
    expect(calls.at(-1)).toEqual({
      method: 'permission_response',
      params: { request_id: 'perm-77', response: 'always', option_id: 'opt-allow-repo' }
    })
  })

  it('persists acp_status bindings in uiStore and applies live tool_update events to active running tools', () => {
    const { handler } = buildEventHandlerHarness()
    patchUiState({ sid: 'session-a' })

    const binding = makeAntigravityBinding()
    for (const ev of adaptDaemonEvent('acp_status', { session_id: 'session-a', binding })) {
      handler({ ...ev, session_id: 'session-a' } as GatewayEvent)
    }

    expect(getUiState().acpBinding).toEqual(binding)
    expect(getUiState().acpBinding?.capabilities?.session.fork).toBe(true)
    expect(getUiState().acpBinding?.config_options?.[0]?.current_value).toBe(true)

    // Session-isolated: background session's acp_status must not overwrite foreground binding
    for (const ev of adaptDaemonEvent('acp_status', { session_id: 'session-other', binding: null })) {
      handler({ ...ev, session_id: 'session-other' } as GatewayEvent)
    }
    expect(getUiState().acpBinding?.provider).toBe('antigravity')

    // Init info also preserves acp_binding cleanly
    const initInfo = sessionInfoFromInit({ model: 'fallback', acp_binding: binding })
    expect(initInfo.acp_binding?.provider).toBe('antigravity')

    // Live tool_call followed by incremental tool_update updates active running tool
    turnController.startMessage()
    for (const ev of adaptDaemonEvent('tool_call', { id: 'tc-1', name: 'edit_file', arguments: '{}' })) {
      handler(ev as GatewayEvent)
    }
    for (const ev of adaptDaemonEvent('tool_update', {
      tool_call_id: 'tc-1',
      name: 'edit_file',
      title: 'Editing src/app.ts',
      status: 'in_progress',
      inputs: { path: 'src/app.ts' },
      output: 'patching hunk 1',
      diff: { path: 'src/app.ts', old_text: 'const a = 1', new_text: 'const a = 2' },
      diffs: [
        { path: 'src/app.ts', old_text: 'const a = 1', new_text: 'const a = 2' },
        { path: 'src/util.ts', old_text: 'old', new_text: 'new' }
      ],
      locations: [{ path: 'src/app.ts', line: 12 }]
    })) {
      handler(ev as GatewayEvent)
    }

    const activeTool = getTurnState().tools.find(t => t.id === 'tc-1')
    expect(activeTool).toMatchObject({
      id: 'tc-1',
      name: 'edit_file',
      title: 'Editing src/app.ts',
      context: 'Editing src/app.ts',
      status: 'in_progress',
      inputs: { path: 'src/app.ts' },
      output: 'patching hunk 1',
      diff: { path: 'src/app.ts', old_text: 'const a = 1', new_text: 'const a = 2' },
      locations: [{ path: 'src/app.ts', line: 12 }]
    })
    expect(activeTool?.diffs).toHaveLength(2)
    expect(getTurnState().toolRecords['tc-1']).toMatchObject({
      name: 'edit_file',
      result: 'patching hunk 1'
    })
  })

  it('preserves approval_request options, diffs, and locations and dispatches exact option_id from PromptZone', async () => {
    const { handler } = buildEventHandlerHarness()
    const events = adaptDaemonEvent('approval_request', {
      id: 'perm-99',
      action: 'write_file',
      description: 'Edit two TypeScript files',
      cwd: '/repo',
      allow_permanent: true,
      diff: { path: 'src/a.ts', old_text: 'oldLine()', new_text: 'newLine()' },
      diffs: [
        { path: 'src/a.ts', old_text: 'oldLine()', new_text: 'newLine()' },
        { path: 'src/b.ts', old_text: 'foo', new_text: 'bar' }
      ],
      locations: [{ path: 'src/a.ts', line: 42 }],
      options: [
        { option_id: 'opt-once', name: 'Allow once', kind: 'allow_once' },
        { option_id: 'opt-file', name: 'Allow for this file', kind: 'allow_always' },
        { option_id: 'opt-repo', name: 'Allow for this repo', kind: 'allow_always' },
        { option_id: 'opt-reject-always', name: 'Always reject', kind: 'reject_always' }
      ]
    })
    for (const ev of events) handler(ev as GatewayEvent)

    expect(getOverlayState().approval).toMatchObject({
      requestId: 'perm-99',
      command: 'write_file',
      cwd: '/repo',
      locations: [{ path: 'src/a.ts', line: 42 }]
    })
    expect(getOverlayState().approval?.options).toHaveLength(4)
    expect(getOverlayState().approval?.diffs).toHaveLength(2)

    const answerApproval = vi.fn()
    const session = await testRender(<PromptZone actions={{ answerApproval } as never} />, { height: 40, width: 96 })
    await session.flush()
    try {
      const frame = session.captureCharFrame()
      expect(frame).toContain('DIFF PREVIEW (2 files)')
      expect(frame).toContain('src/a.ts')
      expect(frame).toContain('- oldLine()')
      expect(frame).toContain('+ newLine()')
      expect(frame).toContain('1 Allow once')
      expect(frame).toContain('2 Allow for this file')
      expect(frame).toContain('3 Allow for this repo')
      expect(frame).toContain('4 Always reject')

      act(() => session.mockInput.pressKey('3'))
      expect(answerApproval).toHaveBeenLastCalledWith('always', 'opt-repo')

      act(() => session.mockInput.pressKey('4'))
      expect(answerApproval).toHaveBeenLastCalledWith('deny', 'opt-reject-always')
    } finally {
      session.renderer.destroy()
    }
  })
})

describe('TUI /acp slash command registration, completions, and help', () => {
  it('registers /acp in the slash registry and exposes all daemon subcommand completions', () => {
    const cmd = findSlashCommand('acp')
    expect(cmd).toBeDefined()
    expect(cmd?.group).toBe('config')

    expect(tuiSlashCompletions('/ac', null)).toContainEqual(expect.objectContaining({ display: 'acp', text: '/acp' }))
    expect(completionRequestForInput('/acp ')).toEqual({
      method: 'complete.slash',
      params: { text: '/acp ' },
      replaceFrom: 1
    })

    const subItems = tuiSlashCompletions('/acp ', null)
    const expectedSubcommands = [
      'status',
      'list',
      'use',
      'connect',
      'disconnect',
      'model',
      'mode',
      'config',
      'catalog',
      'skills',
      'capabilities',
      'sessions',
      'fork',
      'auth'
    ]
    expect(ACP_SUBCOMMANDS.map(s => s.arg)).toEqual(expectedSubcommands)
    expect(subItems.map(item => item.text)).toEqual(expectedSubcommands.map(sub => `/acp ${sub}`))

    expect(tuiSlashCompletions('/acp m', null).map(item => item.text)).toEqual(['/acp model', '/acp mode'])

    patchUiState({ acpBinding: makeAntigravityBinding() })
    expect(tuiSlashCompletions('/acp model ', null).map(item => item.text)).toEqual([
      '/acp model gemini-3-pro',
      '/acp model gemini-3-flash'
    ])
    expect(tuiSlashCompletions('/acp config ', null).map(item => item.text)).toEqual([
      '/acp config auto_save',
      '/acp config model'
    ])
    expect(tuiSlashCompletions('/acp auth ', null).map(item => item.text)).toEqual(['/acp auth oauth-personal'])
  })

  it('routes /acp subcommands to native daemon slash without clearing transcript and lists /acp in /help', async () => {
    patchUiState({ sid: 'session-a' })
    const request = vi.fn(async () => ({ output: '' }))
    const { context, page, resetVisibleHistory, send, setHistoryItems } = makeSlashContext(request)
    const run = createSlashHandler(context)

    run('/acp')
    await Promise.resolve()
    expect(request).toHaveBeenCalledWith('slash.exec', { command: 'acp status', session_id: 'session-a' })

    run('/acp fork')
    await Promise.resolve()
    expect(request).toHaveBeenCalledWith('slash.exec', { command: 'acp fork', session_id: 'session-a' })

    run('/help')
    expect(page.some(entry => entry.text.includes('/acp ['))).toBe(true)
    expect(send).toEqual([])
    expect(resetVisibleHistory).not.toHaveBeenCalled()
    expect(setHistoryItems).not.toHaveBeenCalled()
  })
})
