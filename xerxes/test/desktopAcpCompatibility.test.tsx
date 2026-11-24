// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { ApprovalCard, QuestionCard, UrlConsentCard } from '../src/desktop/renderer/App.js'
import {
  AcpActiveBanner,
  AcpConfigOptionPicker,
  activeAcpConfigOptionLabel,
  extraAcpConfigOptions,
} from '../src/desktop/renderer/AcpRuntimePicker.js'
import {
  AGY_SCHEDULE_READ_ONLY_NOTICE,
  AcpCustomizePanel,
  ScheduleActivity,
} from '../src/desktop/renderer/DesktopPanels.js'
import { ExecutionDetails, ToolCallRow } from '../src/desktop/renderer/Execution.js'
import { FileToolRow, fileToolOf, fileToolStats } from '../src/desktop/renderer/FileToolRow.js'
import {
  ANTIGRAVITY_COMMIT_AND_PUSH_PROMPT,
  ANTIGRAVITY_PULL_REQUEST_PROMPT,
  ANTIGRAVITY_REVIEW_PROMPT,
  isAntigravityAcpSession,
  prefillComposerPrompt,
} from '../src/desktop/renderer/GitPanel.js'
import { T3ModelPicker } from '../src/desktop/renderer/T3ModelPicker.js'
import { toolImagesOf } from '../src/desktop/renderer/toolImages.js'
import { BlockBuilder } from '../src/desktop/renderer/blocks.js'
import {
  Store,
  parseAcpSessionBinding,
  type AcpSessionBinding,
  type Snapshot,
} from '../src/desktop/renderer/store.js'
import type { DaemonEvent, ToolItem } from '../src/desktop/renderer/types.js'
import { SHEETS, VscodeChat } from '../src/vscode/ui/VscodeApp.js'

const HEX_COLOR_RE = /#[0-9a-fA-F]{6}\b/g

function createMockBridge(
  handlers: Record<string, (params: Record<string, unknown>) => Record<string, unknown> | Promise<Record<string, unknown>>> = {},
) {
  const listeners = new Set<(event: DaemonEvent) => void>()
  const calls: Array<{ method: string; params: Record<string, unknown> }> = []
  const xerxes = {
    call: async (method: string, params: Record<string, unknown> = {}) => {
      calls.push({ method, params })
      const handler = handlers[method]
      if (handler) return await handler(params)
      if (method === 'initialize') {
        return {
          ok: true,
          session_id: 'sess-compat-1',
          session_key: String(params.session_key ?? 'desktop-compat'),
          cwd: '/workspace/xerxes',
          model: 'claude-opus-4-6',
          session: {
            id: 'sess-compat-1',
            title: 'ACP Compat Session',
            cwd: '/workspace/xerxes',
            model: 'claude-opus-4-6',
            messages: [],
          },
        }
      }
      if (method === 'session.list') return { ok: true, sessions: [], active: [] }
      if (method === 'commands.catalog') return { ok: true, commands: [] }
      if (method === 'acp.agents') return { ok: true, agents: [], active: null }
      if (method === 'provider_list') return { ok: true, providers: [] }
      if (method === 'permission_response') return { ok: true }
      if (method === 'question_response') return { ok: true }
      if (method === 'turn.submit') return { ok: true, session_id: 'sess-compat-1' }
      return { ok: true }
    },
    onEvent: (fn: (event: DaemonEvent) => void) => {
      listeners.add(fn)
      return () => { listeners.delete(fn) }
    },
  }
  return {
    xerxes,
    calls,
    emit: (type: string, payload: Record<string, unknown>) => {
      for (const listener of listeners) listener({ type, payload })
    },
  }
}

let activeStore: Store | null = null

async function setupStore(
  handlers: Record<string, (params: Record<string, unknown>) => Record<string, unknown> | Promise<Record<string, unknown>>> = {},
) {
  const bridge = createMockBridge(handlers)
  ;(globalThis as { window?: unknown }).window = { xerxes: bridge.xerxes }
  const store = new Store()
  activeStore = store
  store.start(bridge.xerxes)
  for (let i = 0; i < 40 && store.getSnapshot().connection !== 'online'; i += 1) {
    await Bun.sleep(5)
  }
  await Bun.sleep(10)
  return { store, bridge }
}

const agyBinding: AcpSessionBinding = {
  session_id: 'sess-compat-1',
  agent_id: 'antigravity',
  name: 'Antigravity',
  provider: 'antigravity',
  transport: 'acp',
  cwd: '/workspace/xerxes',
  external_session_id: 'agy-ext-99887766',
  connected: true,
  agent_version: '1.5.15',
  current_model: 'gemini-4-neon',
  current_model_name: 'Gemini 4 Neon',
  models: [{ id: 'gemini-4-neon', name: 'Gemini 4 Neon' }],
  current_mode: 'turbo',
  current_mode_name: 'Turbo',
  modes: [
    { id: 'default', name: 'Default' },
    { id: 'auto', name: 'Auto' },
    { id: 'turbo', name: 'Turbo' },
  ],
  config_options: [
    {
      id: 'sandbox',
      name: 'Sandbox',
      description: 'Run commands inside sandbox',
      type: 'boolean',
      current_value: true,
      options: [],
    },
    {
      id: 'profile',
      name: 'Execution Profile',
      type: 'select',
      current_value: 'fast',
      options: [
        { value: 'fast', name: 'Fast', group: 'Standard' },
        { value: 'balanced', name: 'Balanced', group: 'Standard' },
        { value: 'deep', name: 'Deep Audit', group: 'Advanced' },
      ],
    },
  ],
  capabilities: {
    prompt: { image: true, audio: false, embedded_context: true },
    session: {
      load: true,
      resume: true,
      list: true,
      fork: true,
      close: true,
      additional_directories: true,
    },
    mcp: { http: true, sse: true },
    auth_methods: [
      { id: 'oauth-personal', name: 'Google Sign-In', type: 'terminal', args: ['acp', '--login'] },
    ],
  },
  available_commands: [{ name: 'compact', description: 'Compact context', kind: 'command' }],
  discovered_skills: [{ name: 'run-workflow', description: 'Run workflow', slash_command: '/run-workflow' }],
}

function sampleSnap(overrides: Partial<Snapshot> = {}): Snapshot {
  return {
    connection: 'online',
    noWorkspace: false,
    cwd: '/workspace/xerxes',
    model: 'claude-opus-4-6',
    currentAgentPreset: 'default',
    agentPresets: [],
    agentPresetsError: null,
    channels: [],
    channelsAvailable: false,
    channelsConfigured: false,
    terminals: [],
    terminalsLoading: false,
    terminalsError: null,
    searchOpen: false,
    searchResults: [],
    searchStats: null,
    searchSearching: false,
    searchError: null,
    branch: 'main',
    daemonWarning: null,
    costUsd: null,
    costBasis: null,
    mcpStatus: {},
    models: [],
    contextTokens: null,
    computerUseWanted: false,
    contextMax: null,
    ttftMs: null,
    tokensPerSecond: null,
    llmDurationMs: 0,
    llmSteps: 0,
    toolDurationMs: 0,
    toolSteps: 0,
    inputTokens: 0,
    outputTokens: 0,
    metricPhase: null,
    metricPhaseStartedAt: null,
    cacheHitRate: null,
    cacheReadTokens: null,
    cacheWriteTokens: null,
    sessions: [],
    live: [],
    fleet: [],
    skillSuggestions: [],
    creatorTrace: [],
    backgroundJobs: [],
    currentId: 'sess-compat-1',
    sessionOpenRevision: 0,
    currentTitle: 'ACP Compat Session',
    sessionKey: 'desktop-compat',
    goal: '',
    approval: null,
    question: null,
    interactionHidden: false,
    planMode: false,
    turnActive: false,
    turnFailed: false,
    turnSeconds: 0,
    blocks: [],
    error: null,
    tab: 'activity',
    turnCount: 0,
    queue: [],
    changes: [],
    changesKept: false,
    todos: null,
    plan: null,
    log: [],
    failed: null,
    stopArmed: false,
    settingsOpen: false,
    settingsTab: 'general',
    paletteOpen: false,
    pickerOpen: false,
    modelMenuOpen: false,
    appUpdate: null,
    providerRelay: null,
    delegationMode: 'auto',
    delegationMenuOpen: false,
    contextMenuOpen: false,
    contextBreakdown: null,
    contextBreakdownLoading: false,
    reasoningPickerOpen: false,
    reasoningEffort: '',
    reasoningLevels: [],
    reasoningDefault: '',
    reasoningNote: '',
    reasoningLoading: false,
    wsMenuOpen: false,
    workspaceBusy: false,
    workspaceError: null,
    sessionMenu: null,
    streamThinking: true,
    taskModalOpen: false,
    providers: [],
    providerError: '',
    navigating: null,
    providersLoading: false,
    modelsLoading: false,
    modelsError: null,
    providerSwitching: null,
    providerSwitchError: null,
    runMode: null,
    runModeError: '',
    runModeSaving: false,
    localSetup: { status: null, job: null, error: '', loading: false },
    providerModels: {},
    providerModelLoading: [],
    providerModelWarnings: {},
    providerTypes: [],
    commands: [],
    permissionMode: 'auto',
    permissionUpdating: false,
    permissionError: null,
    snippets: {},
    acpAgents: [],
    acpBinding: null,
    savedAcpBinding: null,
    acpConnecting: false,
    acpError: null,
    acpAuthRequired: false,
    acpAuthMethods: [],
    ...overrides,
  }
}

const realWindow = globalThis.window
const realDocument = globalThis.document

afterEach(() => {
  activeStore = null
  if (realWindow === undefined) delete (globalThis as Record<string, unknown>).window
  else (globalThis as { window?: unknown }).window = realWindow
  if (realDocument === undefined) delete (globalThis as Record<string, unknown>).document
  else (globalThis as { document?: unknown }).document = realDocument
})

describe('Desktop & VS Code ACP Compatibility + Antigravity Wire Hygiene', () => {
  test('A.1 Antigravity schedule rows never submit manage_task kill prompts and render read-only notice', () => {
    const panelsSrc = readFileSync(join(import.meta.dir, '../src/desktop/renderer/DesktopPanels.tsx'), 'utf8')
    expect(panelsSrc).not.toContain('Cancel background task')
    expect(panelsSrc).not.toContain('manage_task(Action: "kill"')

    const html = renderToStaticMarkup(
      createElement(ScheduleActivity, {
        row: {
          id: 'agy-task:sched-1',
          kind: 'schedule',
          source: 'antigravity',
          synthetic: true,
          title: 'Check build status',
          state: 'scheduled',
          detail: 'Antigravity timer · 5m',
        },
        online: true,
        busy: false,
        readOnlyNotice: AGY_SCHEDULE_READ_ONLY_NOTICE,
      }),
    )
    expect(html).toContain('Check build status')
    expect(html).not.toContain('>Cancel<')
  })

  test('A.2 retryFailed submits plain "Continue" on Antigravity/ACP without Xerxes boilerplate, keeping native behavior', async () => {
    const { store, bridge } = await setupStore({
      'acp.connect': () => ({ ok: true, binding: agyBinding }),
    })

    // Connected to Antigravity: fail a turn, then retry
    expect(await store.connectAcpAgent('antigravity')).toBe(true)
    bridge.emit('turn_begin', { session_id: 'sess-compat-1', text: 'Fix the parser bug' })
    bridge.emit('notification', { session_id: 'sess-compat-1', severity: 'error', body: 'upstream timeout' })
    bridge.emit('turn_end', { session_id: 'sess-compat-1' })
    expect(store.getSnapshot().failed?.lastUser).toBe('Fix the parser bug')

    bridge.calls.length = 0
    store.retryFailed()
    await Bun.sleep(10)
    const agyRetry = bridge.calls.find(c => c.method === 'turn.submit')
    expect(agyRetry?.params.text).toBe('Continue')
    expect(agyRetry?.params.display_text).toBeUndefined()
    expect(String(agyRetry?.params.text)).not.toContain('cut off by an error')

    // Native Xerxes session keeps the full contextual retry prompt with display_text: 'Continue'
    expect(await store.disconnectAcpAgent()).toBe(true)
    bridge.emit('turn_begin', { session_id: 'sess-compat-1', text: 'Run checks' })
    bridge.emit('notification', { session_id: 'sess-compat-1', severity: 'error', body: 'network drop' })
    bridge.emit('turn_end', { session_id: 'sess-compat-1' })

    bridge.calls.length = 0
    store.retryFailed()
    await Bun.sleep(10)
    const nativeRetry = bridge.calls.find(c => c.method === 'turn.submit')
    expect(String(nativeRetry?.params.text)).toContain('Your previous reply was cut off by an error')
    expect(nativeRetry?.params.display_text).toBe('Continue')
  })

  test('A.3 Antigravity git actions prefill plain composer prompts instead of submitting Xerxes Workflow prompts', () => {
    expect(isAntigravityAcpSession({ acpBinding: agyBinding })).toBe(true)
    expect(isAntigravityAcpSession({ acpBinding: null })).toBe(false)

    const dispatched: Array<{ type: string; detail: unknown }> = []
    ;(globalThis as { window?: unknown }).window = {
      dispatchEvent: (event: CustomEvent<unknown>) => {
        dispatched.push({ type: event.type, detail: event.detail })
        return true
      },
    }

    prefillComposerPrompt(ANTIGRAVITY_PULL_REQUEST_PROMPT)
    prefillComposerPrompt(ANTIGRAVITY_REVIEW_PROMPT)
    prefillComposerPrompt(ANTIGRAVITY_COMMIT_AND_PUSH_PROMPT)

    expect(dispatched).toEqual([
      { type: 'xerxes:prefill-prompt', detail: 'Create a pull request for the current changes.' },
      { type: 'xerxes:prefill-prompt', detail: 'Review the current git changes.' },
      { type: 'xerxes:prefill-prompt', detail: 'Commit and push the current changes.' },
    ])

    // Verify all three UI action sites gate on isAntigravityAcpSession
    const gitSrc = readFileSync(join(import.meta.dir, '../src/desktop/renderer/GitPanel.tsx'), 'utf8')
    const cardSrc = readFileSync(join(import.meta.dir, '../src/desktop/renderer/ThreadDetailsCard.tsx'), 'utf8')
    const appSrc = readFileSync(join(import.meta.dir, '../src/desktop/renderer/App.tsx'), 'utf8')
    expect(gitSrc).toContain('prefillComposerPrompt(ANTIGRAVITY_REVIEW_PROMPT)')
    expect(cardSrc).toContain('prefillComposerPrompt(ANTIGRAVITY_COMMIT_AND_PUSH_PROMPT)')
    expect(cardSrc).toContain('prefillComposerPrompt(ANTIGRAVITY_PULL_REQUEST_PROMPT)')
    expect(appSrc).toContain('prefillComposerPrompt(ANTIGRAVITY_PULL_REQUEST_PROMPT)')
    expect(appSrc).toContain("'xerxes:prefill-prompt'")
  })

  test('A.4 ExitPlanMode 3-minute auto-approve timer is disabled when connected to Antigravity', async () => {
    const { store, bridge } = await setupStore({
      'acp.connect': () => ({ ok: true, binding: agyBinding }),
    })

    // On Antigravity: ExitPlanMode approval must NOT schedule autoAcceptAt
    expect(await store.connectAcpAgent('antigravity')).toBe(true)
    bridge.emit('approval_request', {
      id: 'plan-exit-agy',
      tool_name: 'ExitPlanMode',
      action: 'ExitPlanMode',
    })
    expect(store.getSnapshot().approval?.id).toBe('plan-exit-agy')
    expect(store.getSnapshot().approval?.autoAcceptAt).toBeUndefined()

    // On Native Xerxes: ExitPlanMode still schedules autoAcceptAt
    bridge.emit('approval_response', { request_id: 'plan-exit-agy' })
    expect(await store.disconnectAcpAgent()).toBe(true)
    bridge.emit('approval_request', {
      id: 'plan-exit-native',
      tool_name: 'ExitPlanMode',
      action: 'ExitPlanMode',
    })
    expect(store.getSnapshot().approval?.id).toBe('plan-exit-native')
    expect(typeof store.getSnapshot().approval?.autoAcceptAt).toBe('number')
  })

  test('B.5 Boolean & grouped configOptions preserve native booleans, fix duplicate row bug, render Fork cap, and expose sessions/fork/authenticate RPCs', async () => {
    const parsed = parseAcpSessionBinding({
      ...agyBinding,
      config_options: [
        { id: 'sandbox', name: 'Sandbox', type: 'boolean', current_value: false, options: [] },
        {
          id: 'profile',
          name: 'Execution Profile',
          type: 'select',
          current_value: 'fast',
          options: [
            { group: 'Standard', options: [{ value: 'fast', name: 'Fast' }] },
            { group: 'Advanced', options: [{ value: 'deep', name: 'Deep Audit' }] },
          ],
        },
      ],
      capabilities: {
        prompt: { image: true, audio: false, embedded_context: true },
        session: { load: true, resume: true, list: true, fork: true, close: true, additional_directories: true },
        mcp: { http: true, sse: false },
        auth_methods: [
          {
            id: 'term-login',
            name: 'CLI Sign-In',
            type: 'terminal',
            args: ['acp', '--login'],
            env: { AGY_MODE: 'interactive' },
            _meta: { 'terminal-auth': { command: 'agy', args: ['acp', '--login'] } },
          },
        ],
      },
    })

    expect(parsed?.config_options?.[0]?.current_value).toBe(false)
    expect(typeof parsed?.config_options?.[0]?.current_value).toBe('boolean')
    expect(parsed?.config_options?.[1]?.options).toEqual([
      { value: 'fast', name: 'Fast', group: 'Standard' },
      { value: 'deep', name: 'Deep Audit', group: 'Advanced' },
    ])
    expect(parsed?.capabilities?.session.fork).toBe(true)
    expect(parsed?.capabilities?.auth_methods[0]).toMatchObject({
      id: 'term-login',
      type: 'terminal',
      args: ['acp', '--login'],
      env: { AGY_MODE: 'interactive' },
    })

    const extras = extraAcpConfigOptions(agyBinding)
    expect(extras.map(o => o.id)).toEqual(['sandbox', 'profile'])
    expect(activeAcpConfigOptionLabel(extras[0]!)).toBe('On')

    // Store RPCs: boolean setAcpConfigOption, acpSessions, forkAcpSession, authenticateAcp
    const { store, bridge } = await setupStore({
      'acp.connect': () => ({ ok: true, binding: agyBinding }),
      'acp.set_config_option': () => ({ ok: true }),
      'acp.sessions': () => ({
        ok: true,
        sessions: [{ session_id: 'ext-1', cwd: '/workspace/xerxes', title: 'Prior task' }],
        next_cursor: 'cur-2',
      }),
      'acp.fork': () => ({ ok: true, binding: { ...agyBinding, external_session_id: 'agy-fork-2' } }),
      'acp.authenticate': params =>
        params.method_id === 'term-login'
          ? { ok: true, mode: 'terminal', command: 'agy', args: ['acp', '--login'], label: 'CLI Sign-In' }
          : { ok: true, mode: 'agent' },
    })
    expect(await store.connectAcpAgent('antigravity')).toBe(true)

    bridge.calls.length = 0
    expect(await store.setAcpConfigOption('sandbox', false)).toBe(true)
    expect(bridge.calls.find(c => c.method === 'acp.set_config_option')?.params).toMatchObject({
      config_id: 'sandbox',
      value: false,
      type: 'boolean',
    })
    expect(store.getSnapshot().acpBinding?.config_options?.find(o => o.id === 'sandbox')?.current_value).toBe(false)

    bridge.calls.length = 0
    expect(await store.setAcpConfigOption('profile', 'deep')).toBe(true)
    const selectParams = bridge.calls.find(c => c.method === 'acp.set_config_option')?.params
    expect(selectParams?.value).toBe('deep')
    expect(selectParams?.type).toBeUndefined()

    const listed = await store.acpSessions({ cwd: '/workspace/xerxes' })
    expect(listed.ok).toBe(true)
    expect(listed.sessions).toHaveLength(1)
    expect(listed.next_cursor).toBe('cur-2')

    expect(await store.forkAcpSession()).toBe(true)
    expect(store.getSnapshot().acpBinding?.external_session_id).toBe('agy-fork-2')

    const termAuth = await store.authenticateAcp('term-login')
    expect(termAuth).toMatchObject({ ok: true, mode: 'terminal', command: 'agy', args: ['acp', '--login'] })
    const agentAuth = await store.authenticateAcp('oauth-agent')
    expect(agentAuth).toEqual({ ok: true, mode: 'agent' })

    // AcpCustomizePanel Capabilities tab: boolean rendered ONCE with switch, select with optgroup, Fork row present, zero hex colors
    ;(globalThis as { window?: unknown }).window = { xerxes: bridge.xerxes }
    const capsHtml = renderToStaticMarkup(
      createElement(AcpCustomizePanel, { snap: sampleSnap({ acpBinding: agyBinding }), binding: agyBinding, initialTab: 'capabilities' }),
    )
    expect((capsHtml.match(/data-config-id="sandbox"/g) ?? []).length).toBe(1)
    expect(capsHtml).toContain('role="switch"')
    expect(capsHtml).toContain('aria-checked="true"')
    expect(capsHtml).toContain('<optgroup label="Standard">')
    expect(capsHtml).toContain('<optgroup label="Advanced">')
    expect(capsHtml).toContain('Fork')
    expect(capsHtml.match(HEX_COLOR_RE) ?? []).toEqual([])

    // T3ModelPicker: boolean switch + grouped dropdown popover (not click-to-cycle)
    const t3Html = renderToStaticMarkup(
      createElement(T3ModelPicker, {
        snap: sampleSnap({ acpBinding: agyBinding }),
        defaultOpen: true,
        defaultOpenExtraConfigId: 'profile',
      }),
    )
    expect(t3Html).toContain('role="switch"')
    expect(t3Html).toContain('aria-haspopup="listbox"')
    expect(t3Html).toContain('Standard')
    expect(t3Html).toContain('Advanced')
    expect(t3Html).toContain('Deep Audit')

    // AcpConfigOptionPicker: boolean renders direct switch; grouped select renders group headers
    const boolChipHtml = renderToStaticMarkup(
      createElement(AcpConfigOptionPicker, {
        snap: sampleSnap({ acpBinding: agyBinding }),
        option: agyBinding.config_options![0]!,
      }),
    )
    expect(boolChipHtml).toContain('role="switch"')
    expect(boolChipHtml).toContain('Sandbox: On')

    const selectPopHtml = renderToStaticMarkup(
      createElement(AcpConfigOptionPicker, {
        snap: sampleSnap({ acpBinding: agyBinding }),
        option: agyBinding.config_options![1]!,
        defaultOpen: true,
      }),
    )
    expect(selectPopHtml).toContain('acp-optpop__group')
    expect(selectPopHtml).toContain('Standard')
    expect(selectPopHtml).toContain('Advanced')
  })

  test('B.6 ToolApprovalCard renders all granular approval.options + multi-file diffs and approveWithOption sends exact option_id', async () => {
    const { store, bridge } = await setupStore()

    bridge.emit('approval_request', {
      id: 'appr-granular-1',
      tool_name: 'edit_file',
      action: 'edit_file',
      options: [
        { option_id: 'allow_once', name: 'Allow once', kind: 'allow_once' },
        { option_id: 'allow_file', name: 'Always allow this file', kind: 'allow_always' },
        { option_id: 'allow_repo', name: 'Always allow in repo', kind: 'allow_always' },
        { option_id: 'reject_always', name: 'Always reject', kind: 'reject_always' },
      ],
      diffs: [
        { path: 'src/first.ts', old_text: 'const a = 1\n', new_text: 'const a = 2\n' },
        { path: 'src/second.ts', old_text: 'const b = 1\n', new_text: 'const b = 3\n' },
      ],
    })

    const approval = store.getSnapshot().approval
    expect(approval?.options).toHaveLength(4)
    expect(approval?.diffs).toHaveLength(2)

    const cardHtml = renderToStaticMarkup(
      createElement(ApprovalCard, { approval: approval!, policy: 'default' }),
    )
    expect(cardHtml).toContain('data-option-id="allow_once"')
    expect(cardHtml).toContain('data-option-id="allow_file"')
    expect(cardHtml).toContain('data-option-id="allow_repo"')
    expect(cardHtml).toContain('data-option-id="reject_always"')
    expect(cardHtml).toContain('Always allow this file')
    expect(cardHtml).toContain('Always allow in repo')
    expect(cardHtml).toContain('Always reject')
    expect(cardHtml).toContain('src/first.ts')
    expect(cardHtml).toContain('src/second.ts')
    expect(cardHtml.match(HEX_COLOR_RE) ?? []).toEqual([])

    // approveWithOption dispatches exact option_id and derived decision
    bridge.calls.length = 0
    store.approveWithOption('appr-granular-1', 'allow_repo')
    await Bun.sleep(5)
    expect(bridge.calls.find(c => c.method === 'permission_response')?.params).toEqual({
      request_id: 'appr-granular-1',
      response: 'approve_for_session',
      option_id: 'allow_repo',
    })

    // Reject always option
    bridge.emit('approval_request', {
      id: 'appr-granular-2',
      tool_name: 'run_command',
      action: 'run_command',
      options: [
        { option_id: 'allow_once', name: 'Allow', kind: 'allow_once' },
        { option_id: 'reject_always', name: 'Never allow', kind: 'reject_always' },
      ],
    })
    bridge.calls.length = 0
    store.approveWithOption('appr-granular-2', 'reject_always')
    await Bun.sleep(5)
    expect(bridge.calls.find(c => c.method === 'permission_response')?.params).toEqual({
      request_id: 'appr-granular-2',
      response: 'reject',
      option_id: 'reject_always',
    })

    // Fallback 3-button card when options are absent
    const fallbackHtml = renderToStaticMarkup(
      createElement(ApprovalCard, {
        approval: { id: 'appr-fallback', action: 'run_command', description: 'ls', allowPermanent: true },
        policy: 'default',
      }),
    )
    expect(fallbackHtml).toContain('Allow once')
    expect(fallbackHtml).toContain('This session')
    expect(fallbackHtml).toContain('Deny')
  })

  test('B.7 URL-mode elicitation renders full visible URL consent card and auto-dismisses on question_resolved', async () => {
    const { store, bridge } = await setupStore()

    bridge.emit('question_request', {
      id: 'q-url-1',
      tool_call_id: 'tc-url-1',
      mode: 'url',
      url: 'https://accounts.example.com/oauth/device?user_code=XYZZ-1234',
      elicitation_id: 'elic-99',
      message: 'Complete sign-in in your browser to authorize MCP access.',
      questions: [
        {
          id: 'url_consent',
          question: 'Complete sign-in in your browser to authorize MCP access.',
          options: ['Open link & continue', 'Decline'],
          allow_free_form: false,
        },
      ],
    })

    const q = store.getSnapshot().question
    expect(q?.mode).toBe('url')
    expect(q?.url).toBe('https://accounts.example.com/oauth/device?user_code=XYZZ-1234')
    expect(q?.elicitationId).toBe('elic-99')

    const html = renderToStaticMarkup(createElement(QuestionCard, { question: q!, plan: null }))
    expect(html).toContain('https://accounts.example.com/oauth/device?user_code=XYZZ-1234')
    expect(html).toContain('Open link')
    expect(html).toContain('Continue / Done')
    expect(html).toContain('Decline')
    expect(html.match(HEX_COLOR_RE) ?? []).toEqual([])

    const directHtml = renderToStaticMarkup(createElement(UrlConsentCard, { question: q! }))
    expect(directHtml).toContain('data-elicitation-url="https://accounts.example.com/oauth/device?user_code=XYZZ-1234"')

    // question_resolved matching elicitation_id auto-dismisses the pending URL consent card
    bridge.emit('question_resolved', {
      session_id: 'sess-compat-1',
      elicitation_id: 'elic-99',
      reason: 'complete',
    })
    expect(store.getSnapshot().question).toBeNull()
  })

  test('B.8 AcpActiveBanner scopes agy auth hint strictly to Antigravity and renders structured auth_methods from connect failure', async () => {
    // Claude Code & custom ACP auth errors must NEVER suggest `agy acp --login`
    const claudeAuthHtml = renderToStaticMarkup(
      createElement(AcpActiveBanner, {
        snap: sampleSnap({ acpError: 'Authentication required (-32000) for Claude Code' }),
      }),
    )
    expect(claudeAuthHtml).toContain('Authentication required')
    expect(claudeAuthHtml).not.toContain('agy acp --login')

    const customAuthHtml = renderToStaticMarkup(
      createElement(AcpActiveBanner, {
        snap: sampleSnap({ acpError: 'Unauthenticated (-32000)' }),
      }),
    )
    expect(customAuthHtml).not.toContain('agy acp --login')

    // Capture auth_required + auth_methods on acp.connect failure and clear on clearAcpError
    const { store } = await setupStore({
      'acp.connect': () => ({
        ok: false,
        error: 'Authentication required (-32000)',
        auth_required: true,
        auth_methods: [
          { id: 'oauth-browser', name: 'Sign in with Browser', type: 'agent' },
          { id: 'cli-login', name: 'Terminal Login', type: 'terminal', args: ['login'] },
        ],
      }),
    })
    expect(await store.connectAcpAgent('custom-agent')).toBe(false)
    expect(store.getSnapshot().acpAuthRequired).toBe(true)
    expect(store.getSnapshot().acpAuthMethods).toHaveLength(2)

    const bannerHtml = renderToStaticMarkup(createElement(AcpActiveBanner, { snap: store.getSnapshot() }))
    expect(bannerHtml).toContain('data-auth-method-id="oauth-browser"')
    expect(bannerHtml).toContain('data-auth-method-type="agent"')
    expect(bannerHtml).toContain('data-auth-method-id="cli-login"')
    expect(bannerHtml).toContain('data-auth-method-type="terminal"')
    expect(bannerHtml).toContain('Sign in with Browser')
    expect(bannerHtml).toContain('Terminal Login')
    expect(bannerHtml).not.toContain('agy acp --login')

    store.clearAcpError()
    expect(store.getSnapshot().acpAuthRequired).toBe(false)
    expect(store.getSnapshot().acpAuthMethods).toEqual([])
  })

  test('B.9 Live tool_update, multi-file diffs[], rich content_blocks, and ACP usage cost', async () => {
    // BlockBuilder live tool_update + multi-diff + content_blocks
    const builder = new BlockBuilder()
    builder.push('tool_call', {
      id: 'tc-live-1',
      name: 'workspace_batch',
      title: 'Analyzing files',
      arguments: { step: 1 },
    })
    builder.push('tool_update', {
      tool_call_id: 'tc-live-1',
      title: 'Applying 2 edits',
      status: 'in_progress',
      inputs: { step: 2 },
      output: 'Editing src/a.ts and src/b.ts',
      diffs: [
        { path: 'src/a.ts', old_text: 'const a = 1\n', new_text: 'const a = 2\n' },
        { path: 'src/b.ts', old_text: 'const b = 1\n', new_text: 'const b = 3\nconst c = 4\n' },
      ],
    })
    const liveTools = builder.snapshot(true).find(b => b.kind === 'tools')
    expect(liveTools?.kind).toBe('tools')
    if (liveTools?.kind === 'tools') {
      const liveItem = liveTools.items[0]!
      expect(liveItem.state).toBe('working')
      expect(liveItem.title).toBe('Applying 2 edits')
      expect(liveItem.output).toContain('Editing src/a.ts')
      expect(liveItem.diffs).toHaveLength(2)
      expect(liveItem.diff?.adds).toBe(3)
      expect(liveItem.diff?.dels).toBe(2)

      // Multi-file diffs render via FileToolRow patch viewer
      const fileTool = fileToolOf(liveItem)
      expect(fileTool?.kind).toBe('patch')
      if (fileTool?.kind === 'patch') {
        expect(fileTool.files.map(f => f.path)).toEqual(['src/a.ts', 'src/b.ts'])
      }
      expect(fileToolStats(liveItem)).toEqual({ adds: 3, dels: 2 })
      const rowHtml = renderToStaticMarkup(createElement(FileToolRow, { item: liveItem, tool: fileTool!, failed: false }))
      expect(rowHtml).toContain('src/a.ts')
      expect(rowHtml).toContain('src/b.ts')
    }

    // Rich content_blocks on tool_result: image + resource_link + resource + terminal
    const mediaItem: ToolItem = {
      id: 'tc-media-1',
      verb: 'capture',
      arg: 'screenshot',
      dur: '0.3s',
      state: 'done',
      name: 'capture_screen',
      input: '{}',
      output: '',
      contentBlocks: [
        { type: 'image', data: 'iVBORw0KGgo=', mime_type: 'image/png', uri: 'shot.png' },
        { type: 'resource_link', uri: 'https://example.com/report', title: 'Build Report' },
        { type: 'resource', uri: 'file:///workspace/log.txt', mime_type: 'text/plain', text: 'All tests green' },
        { type: 'terminal', terminal_id: 'term-7' },
      ],
    }
    const imgs = toolImagesOf(mediaItem)
    expect(imgs).toEqual([{ src: 'data:image/png;base64,iVBORw0KGgo=', alt: 'shot.png' }])

    const detailsHtml = renderToStaticMarkup(createElement(ExecutionDetails, { item: mediaItem }))
    expect(detailsHtml).toContain('Build Report')
    expect(detailsHtml).toContain('All tests green')
    expect(detailsHtml).toContain('Terminal · term-7')
    expect(detailsHtml).not.toContain('No output')

    const genericRowHtml = renderToStaticMarkup(createElement(ToolCallRow, { item: { ...mediaItem, title: 'Capture view' }, label: 'capture_screen' }))
    expect(genericRowHtml).toContain('Capture view')

    // Store handles tool_update, folds all multi-diff files into changes, and captures ACP usage cost
    const { store, bridge } = await setupStore()
    bridge.emit('tool_call', { id: 'tc-multi', name: 'batch_edit', arguments: {} })
    bridge.emit('tool_update', {
      tool_call_id: 'tc-multi',
      title: 'Patching files…',
      status: 'in_progress',
      output: 'Working…',
    })
    bridge.emit('tool_result', {
      tool_call_id: 'tc-multi',
      name: 'batch_edit',
      output: 'ok',
      diffs: [
        { path: 'src/one.ts', old_text: 'x\n', new_text: 'x2\n' },
        { path: 'src/two.ts', old_text: 'y\n', new_text: 'y2\n' },
      ],
    })
    expect(store.getSnapshot().changes.map(c => c.path)).toEqual(['src/one.ts', 'src/two.ts'])

    bridge.emit('usage', {
      session_id: 'sess-compat-1',
      used: 12_000,
      size: 200_000,
      cost: { amount: 0.042, currency: 'USD' },
    })
    expect(store.getSnapshot().contextTokens).toBe(12_000)
    expect(store.getSnapshot().contextMax).toBe(200_000)
    expect(store.getSnapshot().costUsd).toBe(0.042)

    bridge.emit('status_update', {
      session_id: 'sess-compat-1',
      cost: { amount: 0.125, currency: 'USD' },
    })
    expect(store.getSnapshot().costUsd).toBe(0.125)
  })

  test('B.10 VS Code adds extensions sheet rendering AcpCustomizePanel and header ACP pill opens sheet instead of disconnecting', () => {
    expect(SHEETS.has('extensions')).toBe(true)

    const agySnap = sampleSnap({ acpBinding: agyBinding })
    const vscodeSheetHtml = renderToStaticMarkup(
      createElement(VscodeChat, { snap: agySnap, initialSheet: 'extensions' }),
    )
    expect(vscodeSheetHtml).toContain('studio-sheet')
    expect(vscodeSheetHtml).toContain('acp-customize')
    expect(vscodeSheetHtml).toContain('run-workflow')
    expect(vscodeSheetHtml).toContain('Switch to Xerxes')

    // Header pill opens sheet rather than immediately calling disconnectAcpAgent()
    expect(vscodeSheetHtml).toContain('xv-pill--acp')
    expect(vscodeSheetHtml).toContain('aria-haspopup="dialog"')
    expect(vscodeSheetHtml).toContain('inspect runtime &amp; customize')
    expect(vscodeSheetHtml).not.toContain('click to switch back to Xerxes')

    const vscodeSrc = readFileSync(join(import.meta.dir, '../src/vscode/ui/VscodeApp.tsx'), 'utf8')
    expect(vscodeSrc).not.toContain('onClick={() => void store.disconnectAcpAgent()}')
  })
})
