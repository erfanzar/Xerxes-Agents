// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import {
  AcpActiveBanner,
  AcpModePicker,
  AcpModelPicker,
  AcpRuntimePicker,
  AcpThinkingPicker,
  attachEscapeCloseListener,
  parseCustomArgs,
  resolveRuntimeList,
} from '../src/desktop/renderer/AcpRuntimePicker.js'
import { ActivityGroup, ApprovalCard, Shell, approvalKeyAction } from '../src/desktop/renderer/App.js'
import { BlockBuilder, editStatsOf, isEditTool } from '../src/desktop/renderer/blocks.js'
import { ExecutionDetails, ToolCallRow, classifyToolItem, executionView, stripAgyCommandEnvelope, toolHasFailed } from '../src/desktop/renderer/Execution.js'
import { FileToolRow, fileToolOf, fileToolStats } from '../src/desktop/renderer/FileToolRow.js'
import { SideQuestionPanel } from '../src/desktop/renderer/SideQuestion.js'
import {
  applyBtwEventPayload,
  formatSideQuestionToolLabel,
  parseSideQuestionSlashCommand,
} from '../src/desktop/renderer/sideQuestionState.js'
import {
  Store,
  applyInflightTurn,
  inflightStartedAtMs,
  parseAcpSessionBinding,
  type AcpExternalAgentInfo,
  type AcpSessionBinding,
  type Snapshot,
  type XerxesLike,
} from '../src/desktop/renderer/store.js'
import { ThreadDetailsCard } from '../src/desktop/renderer/ThreadDetailsCard.js'
import { rectOverlapsAny } from '../src/desktop/renderer/BrowserPanel.js'
import { extractAcpBackgroundRowsFromBlocks, ScheduleActivity } from '../src/desktop/renderer/DesktopPanels.js'
import { ChangesTab, LogTab, PlanTab } from '../src/desktop/renderer/Workspaces.js'
import { ToolTimelineGroup, summarizeToolTimeline } from '../src/desktop/renderer/ToolTimeline.js'
import type { Block, DaemonEvent, ToolItem } from '../src/desktop/renderer/types.js'
import {
  SideQuestionCard,
  foldVscodeBtwEvent,
  parseVscodeBtwEvent,
  type VscodeBtwEntry,
} from '../src/vscode/ui/SideQuestionCard.js'
import { VscodeChat } from '../src/vscode/ui/VscodeApp.js'

const sampleAgents: readonly AcpExternalAgentInfo[] = [
  {
    id: 'xerxes',
    name: 'Xerxes',
    provider: 'xerxes',
    transport: 'native',
    available: true,
    description: 'Native multi-agent runtime & Immortals',
  },
  {
    id: 'claude-code',
    name: 'Claude Code',
    provider: 'claude-code',
    transport: 'claude-stream-json',
    available: true,
    command: 'claude',
    binaryPath: '/Users/erfan/.local/bin/claude',
    version: '2.1.4',
    description: 'Anthropic Claude Code via stream-json',
  },
  {
    id: 'antigravity',
    name: 'Antigravity',
    provider: 'antigravity',
    transport: 'acp',
    available: false,
    installHint: 'Install Antigravity CLI (agy) so `agy acp` is on PATH',
    description: 'Google Antigravity via ACP stdio',
  },
]

const sampleClaudeBinding: AcpSessionBinding = {
  session_id: 'sess-acp-1',
  agent_id: 'claude-code',
  name: 'Claude Code',
  provider: 'claude-code',
  transport: 'claude-stream-json',
  cwd: '/workspace/xerxes',
  external_session_id: 'ext-claude-8f31ab92',
  connected: true,
  command: '/Users/erfan/.local/bin/claude',
  current_model: 'claude-sonnet-4-6',
  current_model_name: 'Claude Sonnet 4.6',
  models: [
    { id: 'claude-opus-4-6', name: 'Claude Opus 4.6', description: 'Most capable Anthropic model' },
    { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6', description: 'Balanced coding speed and quality' },
    { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5', description: 'Fast lightweight responses' },
  ],
}

const sampleAntigravityBinding: AcpSessionBinding = {
  session_id: 'sess-acp-1',
  agent_id: 'antigravity',
  name: 'Antigravity',
  provider: 'antigravity',
  transport: 'acp',
  cwd: '/workspace/xerxes',
  external_session_id: '39c61d78-9f1a-4b2c-98f0-123456789abc',
  connected: true,
  command: 'agy',
  current_model: 'gemini-4-carbon-high',
  current_model_name: 'Gemini 4 Carbon (High)',
  model_config_id: 'model',
  models: [
    { id: 'gemini-4-carbon-high', name: 'Gemini 4 Carbon (High)', description: 'Deep reasoning for multi-file engineering' },
    { id: 'gemini-4-carbon-low', name: 'Gemini 4 Carbon (Low)', description: 'Fast iteration with lower latency' },
    { id: 'gemini-3-pro', name: 'Gemini 3 Pro', description: 'General-purpose coding & architecture' },
    { id: 'gemini-3-flash', name: 'Gemini 3 Flash', description: 'Ultra-fast edits and Q&A' },
    { id: 'claude-opus-4-6-via-agy', name: 'Claude Opus 4.6 (Thinking)', description: 'Extended thinking via Antigravity' },
    { id: 'claude-sonnet-4-6-via-agy', name: 'Claude Sonnet 4.6', description: 'Responsive daily coding' },
    { id: 'gpt-oss-120b', name: 'GPT-OSS 120B', description: 'Open-weights reasoning model' },
  ],
  current_mode: 'planning',
  current_mode_name: 'Planning',
  mode_config_id: 'mode',
  modes: [
    { id: 'planning', name: 'Planning', description: 'Draft a structured plan before editing files' },
    { id: 'fast', name: 'Fast', description: 'Execute straightforward tasks directly' },
  ],
}

interface MockBridge {
  readonly calls: Array<{ readonly method: string; readonly params: Record<string, unknown> }>
  readonly emit: (type: string, payload: Record<string, unknown>) => void
  readonly xerxes: XerxesLike
}

function createMockBridge(
  overrides: Partial<Record<string, (params: Record<string, unknown>) => Record<string, unknown>>> = {},
): MockBridge {
  const calls: Array<{ method: string; params: Record<string, unknown> }> = []
  const listeners = new Set<(event: DaemonEvent) => void>()
  let currentActive: AcpSessionBinding | null = null

  const call = async (method: string, params: Record<string, unknown> = {}): Promise<Record<string, unknown>> => {
    calls.push({ method, params })
    const custom = overrides[method]
    if (custom) return custom(params)
    if (method === 'initialize') {
      return {
        ok: true,
        session_id: 'sess-acp-1',
        session_key: String(params.session_key ?? 'desktop-test'),
        cwd: '/workspace/xerxes',
        model: 'claude-opus-4-6',
        session: {
          id: 'sess-acp-1',
          title: 'ACP & Side Question Test',
          cwd: '/workspace/xerxes',
          model: 'claude-opus-4-6',
          messages: [],
        },
      }
    }
    if (method === 'acp.agents') {
      return { ok: true, agents: sampleAgents, active: currentActive }
    }
    if (method === 'acp.connect') {
      const agentId = String(params.agent_id ?? '')
      if (agentId === 'antigravity' && !params.command) {
        return {
          ok: false,
          error: 'Antigravity CLI not found',
          install_hint: 'Install agy CLI so `agy acp` is on PATH',
        }
      }
      currentActive =
        agentId === 'antigravity'
          ? {
              ...sampleAntigravityBinding,
              cwd: String(params.cwd ?? '/workspace/xerxes'),
              ...(typeof params.command === 'string' ? { command: params.command } : {}),
            }
          : {
              session_id: 'sess-acp-1',
              agent_id: agentId,
              name: agentId === 'claude-code' ? 'Claude Code' : agentId,
              provider: agentId === 'claude-code' ? 'claude-code' : 'acp',
              transport: agentId === 'claude-code' ? 'claude-stream-json' : 'acp',
              cwd: String(params.cwd ?? '/workspace/xerxes'),
              external_session_id: `ext-${agentId}-001`,
              connected: true,
              ...(typeof params.command === 'string' ? { command: params.command } : {}),
            }
      return { ok: true, binding: currentActive }
    }
    if (method === 'acp.set_model') {
      const modelId = String(params.model_id ?? '')
      if (!currentActive || modelId === 'bad-model') {
        return { ok: false, error: `Model ${modelId} is not supported` }
      }
      const matched = currentActive.models?.find(m => m.id === modelId)
      currentActive = {
        ...currentActive,
        current_model: modelId,
        current_model_name: matched?.name ?? modelId,
      }
      return { ok: true, binding: currentActive }
    }
    if (method === 'acp.set_mode') {
      const modeId = String(params.mode_id ?? '')
      if (!currentActive || modeId === 'bad-mode') {
        return { ok: false, error: `Mode ${modeId} is not supported` }
      }
      const matched = currentActive.modes?.find(m => m.id === modeId)
      currentActive = {
        ...currentActive,
        current_mode: modeId,
        current_mode_name: matched?.name ?? modeId,
      }
      return { ok: true, binding: currentActive }
    }
    if (method === 'acp.set_config_option') {
      const configId = String(params.config_id ?? '')
      const value = String(params.value ?? '')
      if (!currentActive || value === 'invalid') {
        return { ok: false, error: `Config ${configId}=${value} is not supported` }
      }
      const updatedConfigs = currentActive.config_options?.map(opt =>
        opt.id === configId ? { ...opt, current_value: value } : opt,
      )
      currentActive = {
        ...currentActive,
        ...(updatedConfigs ? { config_options: updatedConfigs } : {}),
        ...(configId === (currentActive.thought_level_config_id ?? 'thought_level')
          ? { thought_level: value }
          : {}),
      }
      return { ok: true, binding: currentActive }
    }
    if (method === 'acp.disconnect') {
      currentActive = null
      return { ok: true }
    }
    if (method === 'acp.cancel') {
      return { ok: true }
    }
    if (method === 'turn.cancel') {
      return { ok: true }
    }
    if (method === 'session.side_question') {
      return {
        ok: true,
        answer: `Answered with ${String(params.model || 'session-default')}: ${String(params.question ?? '')}`,
        model: String(params.model || 'claude-opus-4-6'),
      }
    }
    if (method === 'session.list' || method === 'session.active_list') {
      return { ok: true, sessions: [] }
    }
    if (method === 'swarm.status') {
      return { ok: true, active: false }
    }
    if (method === 'learning.list') {
      return { ok: true, proposals: [], lessons: [], counts: { pending_proposals: 0, pending_lessons: 0 } }
    }
    if (method === 'fetch_models') {
      return {
        ok: true,
        models: ['claude-opus-4-6', 'claude-haiku-4-5', 'gemini-2.5-flash'],
      }
    }
    return { ok: true }
  }

  const xerxes: XerxesLike = {
    call,
    onEvent: handler => {
      listeners.add(handler)
      return () => {
        listeners.delete(handler)
      }
    },
  }

  return {
    calls,
    emit: (type, payload) => {
      for (const listener of listeners) listener({ type, payload })
    },
    xerxes,
  }
}

function createMemoryStorage(): Storage {
  const map = new Map<string, string>()
  return {
    get length() {
      return map.size
    },
    clear() {
      map.clear()
    },
    getItem(key: string) {
      return map.get(key) ?? null
    },
    key(index: number) {
      return [...map.keys()][index] ?? null
    },
    removeItem(key: string) {
      map.delete(key)
    },
    setItem(key: string, value: string) {
      map.set(key, value)
    },
  }
}

const originalWindow = (globalThis as Record<string, unknown>).window
const originalLocalStorage = (globalThis as Record<string, unknown>).localStorage
let activeStore: Store | null = null

async function setupConnectedStore(
  overrides: Partial<Record<string, (params: Record<string, unknown>) => Record<string, unknown>>> = {},
): Promise<{ store: Store; bridge: MockBridge; storage: Storage }> {
  const bridge = createMockBridge(overrides)
  const storage = createMemoryStorage()
  ;(globalThis as Record<string, unknown>).localStorage = storage
  ;(globalThis as Record<string, unknown>).window = { xerxes: bridge.xerxes }
  const store = new Store()
  activeStore = store
  store.start(bridge.xerxes)
  for (let i = 0; i < 40 && store.getSnapshot().connection !== 'online'; i += 1) {
    await Bun.sleep(5)
  }
  await Bun.sleep(10)
  return { store, bridge, storage }
}

function sampleSnapshot(overrides: Partial<Snapshot> = {}): Snapshot {
  const base = new Store().getSnapshot()
  return {
    ...base,
    connection: 'online',
    noWorkspace: false,
    cwd: '/workspace/xerxes',
    branch: 'main',
    currentId: 'sess-acp-1',
    currentTitle: 'Runtime & Side Question',
    sessionKey: 'desktop-test',
    model: 'claude-opus-4-6',
    models: [
      { id: 'claude-opus-4-6', provider: 'Anthropic', label: 'Claude Opus 4.6' },
      { id: 'claude-haiku-4-5', provider: 'Anthropic', label: 'Claude Haiku 4.5' },
      { id: 'gemini-2.5-flash', provider: 'Google', label: 'Gemini 2.5 Flash' },
    ],
    acpAgents: sampleAgents,
    acpBinding: null,
    acpConnecting: false,
    acpError: null,
    ...overrides,
  }
}

afterEach(() => {
  activeStore = null
  if (originalWindow === undefined) delete (globalThis as Record<string, unknown>).window
  else (globalThis as Record<string, unknown>).window = originalWindow
  if (originalLocalStorage === undefined) delete (globalThis as Record<string, unknown>).localStorage
  else (globalThis as Record<string, unknown>).localStorage = originalLocalStorage
})

describe('Side Question state, model selector & read-only tool pills', () => {
  test('parses /btw and /side slash commands including --model / -m flags', () => {
    expect(parseSideQuestionSlashCommand('/btw')).toEqual({ kind: 'open' })
    expect(parseSideQuestionSlashCommand('/side')).toEqual({ kind: 'open' })
    expect(parseSideQuestionSlashCommand('/btw Where is ACP wired?')).toEqual({
      kind: 'ask',
      question: 'Where is ACP wired?',
    })
    expect(parseSideQuestionSlashCommand('/btw --model gemini-2.5-flash Explain server.ts')).toEqual({
      kind: 'ask',
      question: 'Explain server.ts',
      model: 'gemini-2.5-flash',
    })
    expect(parseSideQuestionSlashCommand('/side -m claude-haiku-4-5 Check diff')).toEqual({
      kind: 'ask',
      question: 'Check diff',
      model: 'claude-haiku-4-5',
    })
  })

  test('folds btw_event tool events in place, formats labels cleanly, and settles running tools on end', () => {
    const t1 = applyBtwEventPayload([], {
      request_id: 'btw-1',
      kind: 'begin',
      question: 'How does handleSideQuestionRpc pick a model?',
      model: 'gemini-2.5-flash',
    })
    expect(t1?.streaming).toBe(true)

    const t2 = applyBtwEventPayload(t1!.turns, {
      request_id: 'btw-1',
      kind: 'tool',
      tool: { name: 'ReadFile', target: 'xerxes/src/daemon/server.ts', status: 'running' },
    })
    expect(t2?.turns[0]?.tools).toEqual([
      { name: 'ReadFile', target: 'xerxes/src/daemon/server.ts', status: 'running' },
    ])

    const t3 = applyBtwEventPayload(t2!.turns, {
      request_id: 'btw-1',
      kind: 'tool',
      tool: { name: 'ReadFile', target: 'xerxes/src/daemon/server.ts', status: 'done' },
    })
    expect(t3?.turns[0]?.tools).toEqual([
      { name: 'ReadFile', target: 'xerxes/src/daemon/server.ts', status: 'done' },
    ])

    const t4 = applyBtwEventPayload(t3!.turns, {
      request_id: 'btw-1',
      kind: 'tool',
      tool: { name: 'GrepTool', target: 'handleSideQuestionRpc', status: 'running' },
    })
    const t5 = applyBtwEventPayload(t4!.turns, {
      request_id: 'btw-1',
      kind: 'end',
      answer: 'It reads `params.model` and resolves a matching profile.',
      model: 'gemini-2.5-flash',
    })
    expect(t5?.streaming).toBe(false)
    expect(t5?.turns[0]?.tools).toEqual([
      { name: 'ReadFile', target: 'xerxes/src/daemon/server.ts', status: 'done' },
      { name: 'GrepTool', target: 'handleSideQuestionRpc', status: 'done' },
    ])

    expect(formatSideQuestionToolLabel({ name: 'ReadFile', target: 'xerxes/src/daemon/server.ts', status: 'done' })).toEqual({
      verb: 'Read',
      target: 'server.ts',
    })
    expect(formatSideQuestionToolLabel({ name: 'GrepTool', target: 'handleSideQuestionRpc', status: 'done' })).toEqual({
      verb: 'Grep',
      target: 'handleSideQuestionRpc',
    })
    // Redundant target echoing tool name is omitted instead of rendering "Manage task manage_task"
    expect(formatSideQuestionToolLabel({ name: 'manage_task', target: 'manage_task', status: 'done' })).toEqual({
      verb: 'Tasks',
    })
    expect(formatSideQuestionToolLabel({ name: 'run_command', target: 'run_command', status: 'error' })).toEqual({
      verb: 'Command',
    })
  })

  test('store persists sideQuestionModel in localStorage and sends model on session.side_question', async () => {
    const { store, bridge, storage } = await setupConnectedStore()
    bridge.calls.length = 0

    store.setSideQuestionModel('gemini-2.5-flash')
    expect(store.getSnapshot().sideQuestionModel).toBe('gemini-2.5-flash')
    expect(storage.getItem('xerxes.sideQuestion.model')).toBe('gemini-2.5-flash')

    expect(await store.askSideQuestion('Summarize recent edits')).toBe(true)
    const call = bridge.calls.find(c => c.method === 'session.side_question')
    expect(call?.params.model).toBe('gemini-2.5-flash')
    expect(store.getSnapshot().sideQuestionTurns?.[0]?.model).toBe('gemini-2.5-flash')

    // Explicit per-question override via `/btw --model ...`
    bridge.calls.length = 0
    expect(await store.submit('/btw --model claude-haiku-4-5 Quick check')).toBe(true)
    const overrideCall = bridge.calls.find(c => c.method === 'session.side_question')
    expect(overrideCall?.params.model).toBe('claude-haiku-4-5')

    // Clearing restores session default (removes key)
    store.setSideQuestionModel('')
    expect(storage.getItem('xerxes.sideQuestion.model')).toBeNull()
  })

  test('Desktop SideQuestionPanel and VS Code SideQuestionCard render Model Selector, tool pills, and thinking mark', () => {
    const snap = sampleSnapshot({
      turnActive: true,
      sideQuestionModel: 'gemini-2.5-flash',
      sideQuestionStreaming: true,
      sideQuestionTurns: [
        {
          requestId: 'sq-1',
          question: 'Where is handleSideQuestionRpc defined?',
          answer: 'In `xerxes/src/daemon/server.ts`.',
          model: 'gemini-2.5-flash',
          status: 'done',
          tools: [
            { name: 'ReadFile', target: 'xerxes/src/daemon/server.ts', status: 'done' },
            { name: 'GrepTool', target: 'handleSideQuestionRpc', status: 'done' },
          ],
        },
        {
          requestId: 'sq-2',
          question: 'Inspecting live follow-up?',
          answer: '',
          model: 'gemini-2.5-flash',
          status: 'streaming',
          tools: [{ name: 'GlobTool', target: 'xerxes/src/acp/*.ts', status: 'running' }],
        },
      ],
    })

    const desktopHtml = renderToStaticMarkup(
      createElement(SideQuestionPanel, { snap, defaultModelPickerOpen: true }),
    )
    expect(desktopHtml).toContain('aria-label="Side question model"')
    expect(desktopHtml).toContain('btw-model-pop')
    expect(desktopHtml).not.toContain('<select')
    expect(desktopHtml).toContain('Session default')
    expect(desktopHtml).toContain('claude-opus-4-6')
    expect(desktopHtml).toContain('Gemini 2.5 Flash')
    expect(desktopHtml).toContain('Claude Haiku 4.5')
    expect(desktopHtml).toContain('side-question__tool-pill')
    expect(desktopHtml).toContain('server.ts')
    expect(desktopHtml).toContain('handleSideQuestionRpc')
    expect(desktopHtml).toContain('Inspecting files…')

    // VS Code btw tool event parser & card rendering
    const parsedTool = parseVscodeBtwEvent({
      session_id: 'sess-acp-1',
      request_id: 'xv-1',
      kind: 'tool',
      tool: { name: 'ReadFile', target: 'xerxes/src/daemon/server.ts', status: 'running' },
    })
    expect(parsedTool?.kind).toBe('tool')
    const folded = foldVscodeBtwEvent(
      [{ requestId: 'xv-1', question: 'Check server?', answer: '', status: 'streaming' }],
      parsedTool!,
    )
    expect(folded[0]?.tools).toHaveLength(1)

    const entries: readonly VscodeBtwEntry[] = [
      {
        requestId: 'xv-1',
        question: 'Check server?',
        answer: 'Checked `server.ts`.',
        model: 'gemini-2.5-flash',
        status: 'done',
        tools: [{ name: 'ReadFile', target: 'xerxes/src/daemon/server.ts', status: 'done' }],
      },
    ]
    const vscodeCardHtml = renderToStaticMarkup(
      createElement(SideQuestionCard, {
        snap,
        entries,
        defaultModelPickerOpen: true,
        onEntriesChange: () => {},
        onClear: () => {},
        onClose: () => {},
      }),
    )
    expect(vscodeCardHtml).toContain('aria-label="Side question model"')
    expect(vscodeCardHtml).toContain('btw-model-pop')
    expect(vscodeCardHtml).not.toContain('<select')
    expect(vscodeCardHtml).toContain('Session default')
    expect(vscodeCardHtml).toContain('claude-opus-4-6')
    expect(vscodeCardHtml).toContain('side-question__tool-pill')
    expect(vscodeCardHtml).toContain('server.ts')
  })

  test('SideQuestionPanel, SideQuestionCard, and Store isolate /btw models strictly to active external ACP runtime', async () => {
    const agyBtwSnap = sampleSnapshot({
      model: 'claude-opus-4-6',
      sideQuestionModel: 'claude-haiku-4-5', // stale native choice from an earlier session
      acpBinding: sampleAntigravityBinding,
    })

    const desktopHtml = renderToStaticMarkup(
      createElement(SideQuestionPanel, { snap: agyBtwSnap, defaultModelPickerOpen: true }),
    )
    expect(desktopHtml).toContain('Gemini 4 Carbon (High)')
    expect(desktopHtml).toContain('Gemini 3 Flash')
    // Must never leak native Xerxes provider models when Antigravity ACP is active
    expect(desktopHtml).not.toContain('title="claude-opus-4-6"')
    expect(desktopHtml).not.toContain('Anthropic')
    expect(desktopHtml).not.toContain('Claude Haiku 4.5')
    expect(desktopHtml).not.toContain('Gemini 2.5 Flash')

    const vscodeHtml = renderToStaticMarkup(
      createElement(SideQuestionCard, {
        snap: agyBtwSnap,
        entries: [],
        defaultModelPickerOpen: true,
        onEntriesChange: () => {},
        onClear: () => {},
        onClose: () => {},
      }),
    )
    expect(vscodeHtml).toContain('Gemini 4 Carbon (High)')
    expect(vscodeHtml).toContain('Gemini 3 Flash')
    expect(vscodeHtml).not.toContain('title="claude-opus-4-6"')
    expect(vscodeHtml).not.toContain('Anthropic')
    expect(vscodeHtml).not.toContain('Claude Haiku 4.5')

    // Store ignores stale native localStorage sideQuestionModel while connected to Antigravity ACP
    const { store, bridge } = await setupConnectedStore()
    store.setSideQuestionModel('claude-haiku-4-5')
    expect(await store.connectAcpAgent('antigravity', { command: 'agy', args: ['acp'] })).toBe(true)
    bridge.calls.length = 0
    expect(await store.askSideQuestion('Explain this diff')).toBe(true)
    const call = bridge.calls.find(c => c.method === 'session.side_question')
    expect(call?.params.model).toBeUndefined()
  })
})

describe('External ACP Agent Runtime Switcher (Store, Desktop & VS Code)', () => {
  test('store loads acp.agents on connect, connects/disconnects, handles errors, acp_status events, and acp.cancel on Stop', async () => {
    const { store, bridge } = await setupConnectedStore()
    expect(bridge.calls.some(c => c.method === 'acp.agents')).toBe(true)
    expect(store.getSnapshot().acpAgents).toHaveLength(3)
    expect(store.getSnapshot().acpBinding).toBeNull()

    // Connect Claude Code
    bridge.calls.length = 0
    expect(await store.connectAcpAgent('claude-code')).toBe(true)
    expect(bridge.calls.find(c => c.method === 'acp.connect')?.params).toMatchObject({
      agent_id: 'claude-code',
      cwd: '/workspace/xerxes',
    })
    expect(store.getSnapshot().acpBinding?.agent_id).toBe('claude-code')
    expect(store.getSnapshot().acpBinding?.connected).toBe(true)

    // Cancel routes only through turn.cancel (no double acp.cancel + turn.cancel)
    bridge.calls.length = 0
    store.cancel()
    await Bun.sleep(5)
    expect(bridge.calls.some(c => c.method === 'turn.cancel')).toBe(true)
    expect(bridge.calls.some(c => c.method === 'acp.cancel')).toBe(false)

    // Missing binary returns typed error with install hint
    expect(await store.connectAcpAgent('antigravity')).toBe(false)
    expect(store.getSnapshot().acpError).toContain('Antigravity CLI not found')
    expect(store.getSnapshot().acpError).toContain('agy acp')

    // Custom ACP command connects cleanly with external models & modes
    bridge.calls.length = 0
    expect(await store.connectAcpAgent('antigravity', { command: 'agy', args: ['acp'] })).toBe(true)
    expect(bridge.calls.find(c => c.method === 'acp.connect')?.params).toMatchObject({
      agent_id: 'antigravity',
      command: 'agy',
      args: ['acp'],
    })
    expect(store.getSnapshot().acpBinding?.agent_id).toBe('antigravity')
    expect(store.getSnapshot().acpBinding?.current_model).toBe('gemini-4-carbon-high')
    expect(store.getSnapshot().acpBinding?.current_model_name).toBe('Gemini 4 Carbon (High)')
    expect(store.getSnapshot().acpBinding?.models).toHaveLength(7)
    expect(store.getSnapshot().acpBinding?.modes).toHaveLength(2)

    // Switch external ACP model via acp.set_model
    bridge.calls.length = 0
    expect(await store.setAcpModel('gemini-3-pro')).toBe(true)
    expect(bridge.calls.find(c => c.method === 'acp.set_model')?.params).toMatchObject({
      model_id: 'gemini-3-pro',
    })
    expect(store.getSnapshot().acpBinding?.current_model).toBe('gemini-3-pro')
    expect(store.getSnapshot().acpBinding?.current_model_name).toBe('Gemini 3 Pro')

    // Invalid external model surfaces error without breaking binding
    expect(await store.setAcpModel('bad-model')).toBe(false)
    expect(store.getSnapshot().acpError).toContain('not supported')
    expect(store.getSnapshot().acpBinding?.current_model).toBe('gemini-3-pro')

    // Switch external ACP mode via acp.set_mode
    bridge.calls.length = 0
    expect(await store.setAcpMode('fast')).toBe(true)
    expect(bridge.calls.find(c => c.method === 'acp.set_mode')?.params).toMatchObject({
      mode_id: 'fast',
    })
    expect(store.getSnapshot().acpBinding?.current_mode).toBe('fast')
    expect(store.getSnapshot().acpBinding?.current_mode_name).toBe('Fast')

    // Live acp_status event updates binding including model metadata
    bridge.emit('acp_status', {
      session_id: 'sess-acp-1',
      binding: sampleClaudeBinding,
    })
    expect(store.getSnapshot().acpBinding?.agent_id).toBe('claude-code')
    expect(store.getSnapshot().acpBinding?.external_session_id).toBe('ext-claude-8f31ab92')
    expect(store.getSnapshot().acpBinding?.current_model).toBe('claude-sonnet-4-6')
    expect(store.getSnapshot().acpBinding?.models).toHaveLength(3)

    // Connecting 'xerxes' disconnects external ACP binding
    bridge.calls.length = 0
    expect(await store.connectAcpAgent('xerxes')).toBe(true)
    expect(bridge.calls.some(c => c.method === 'acp.disconnect')).toBe(true)
    expect(store.getSnapshot().acpBinding).toBeNull()
  })

  test('External ACP swaps composer controls: hides Xerxes native model/Plan/Agents and shows ACP model & mode pickers', () => {
    ;(globalThis as Record<string, unknown>).window = { xerxes: createMockBridge().xerxes }

    // 1. Connected to Antigravity ACP (with models + modes), while native Xerxes model is gpt-6.2-astra
    const agySnap = sampleSnapshot({
      model: 'gpt-6.2-astra',
      permissionMode: 'accept-all',
      reasoningEffort: 'high',
      reasoningLevels: [{ effort: 'high', description: 'High reasoning' }],
      acpBinding: sampleAntigravityBinding,
    })

    // Normal connected state stays calm: no noisy "Connected via" banner above the composer
    const agyBannerHtml = renderToStaticMarkup(createElement(AcpActiveBanner, { snap: agySnap }))
    expect(agyBannerHtml).toBe('')

    // Model picker popover lists Antigravity's models and shows filter bar (> 6 models)
    const modelPopoverHtml = renderToStaticMarkup(
      createElement(AcpModelPicker, { snap: agySnap, defaultOpen: true }),
    )
    expect(modelPopoverHtml).toContain('Antigravity models')
    expect(modelPopoverHtml).toContain('Gemini 4 Carbon (High)')
    expect(modelPopoverHtml).toContain('Gemini 3 Pro')
    expect(modelPopoverHtml).toContain('Filter models…')
    expect(modelPopoverHtml).toContain('aria-selected="true"')
    expect(modelPopoverHtml).not.toContain('gpt-6.2-astra')

    // Mode picker popover lists Antigravity's modes
    const modePopoverHtml = renderToStaticMarkup(
      createElement(AcpModePicker, { snap: agySnap, defaultOpen: true }),
    )
    expect(modePopoverHtml).toContain('Antigravity mode')
    expect(modePopoverHtml).toContain('Planning')
    expect(modePopoverHtml).toContain('Fast')

    // Desktop Shell composer hides Xerxes native model/Plan/Agents/effort AND hides Xerxes Permissions when AGY modes own approvals
    const shellAgyHtml = renderToStaticMarkup(createElement(Shell, { snap: agySnap }))
    expect(shellAgyHtml).toContain('Gemini 4 Carbon (High)')
    expect(shellAgyHtml).toContain('Planning')
    expect(shellAgyHtml).not.toContain('Accept all')
    expect(shellAgyHtml).not.toContain('composer__perm')
    expect(shellAgyHtml).toContain('Ask aside')
    expect(shellAgyHtml).toContain('composer__runtime')
    expect(shellAgyHtml).toContain('composer__acp-model')
    expect(shellAgyHtml).toContain('composer__acp-mode')
    expect(shellAgyHtml).not.toContain('gpt-6.2-astra')
    expect(shellAgyHtml).not.toContain('Agents · Auto')
    expect(shellAgyHtml).not.toContain('composer__plan')
    expect(shellAgyHtml).not.toContain('composer__effort')

    // VS Code VscodeChat also swaps composer controls cleanly and hides Xerxes Permissions when AGY modes exist
    const vscodeAgyHtml = renderToStaticMarkup(createElement(VscodeChat, { snap: agySnap }))
    expect(vscodeAgyHtml).toContain('Gemini 4 Carbon (High)')
    expect(vscodeAgyHtml).toContain('Planning')
    expect(vscodeAgyHtml).not.toContain('Accept all')
    expect(vscodeAgyHtml).not.toContain('composer__perm')
    expect(vscodeAgyHtml).not.toContain('gpt-6.2-astra')
    expect(vscodeAgyHtml).not.toContain('Agents · Auto')
    expect(vscodeAgyHtml).not.toContain('composer__plan')

    // External ACP without modes ALSO hides Xerxes native Permissions (external ACP owns its own approvals)
    const claudeNoModesSnap = sampleSnapshot({
      permissionMode: 'accept-all',
      acpBinding: sampleClaudeBinding,
    })
    const claudeShellHtml = renderToStaticMarkup(createElement(Shell, { snap: claudeNoModesSnap }))
    expect(claudeShellHtml).not.toContain('composer__perm')
    expect(claudeShellHtml).not.toContain('Accept all')

    // 2. External ACP agent without advertised models/modes renders calm non-misleading default label (never gpt-6.2-astra)
    const noModelsSnap = sampleSnapshot({
      model: 'gpt-6.2-astra',
      acpBinding: {
        session_id: 'sess-acp-1',
        agent_id: 'antigravity',
        name: 'Antigravity',
        provider: 'antigravity',
        transport: 'acp',
        cwd: '/workspace/xerxes',
        connected: true,
      },
    })
    const staticModelHtml = renderToStaticMarkup(createElement(AcpModelPicker, { snap: noModelsSnap }))
    expect(staticModelHtml).toContain('Antigravity default')
    expect(staticModelHtml).toContain('acp-model-chip--static')
    expect(staticModelHtml).toContain('manages model selection')
    expect(staticModelHtml).not.toContain('gpt-6.2-astra')
    expect(renderToStaticMarkup(createElement(AcpModePicker, { snap: noModelsSnap }))).toBe('')

    // 3. Disconnected back to Xerxes Native restores native model picker + Plan + Agents · Auto + effort
    const nativeSnap = sampleSnapshot({
      model: 'gpt-6.2-astra',
      permissionMode: 'accept-all',
      reasoningEffort: 'high',
      reasoningLevels: [{ effort: 'high', description: 'High reasoning' }],
      acpBinding: null,
    })
    const shellNativeHtml = renderToStaticMarkup(createElement(Shell, { snap: nativeSnap }))
    expect(shellNativeHtml).toContain('gpt-6.2-astra')
    expect(shellNativeHtml).toContain('Agents · Auto')
    expect(shellNativeHtml).toContain('composer__plan')
    expect(shellNativeHtml).toContain('composer__effort')
    expect(shellNativeHtml).not.toContain('composer__acp-model')
    expect(shellNativeHtml).not.toContain('composer__acp-mode')
    expect(shellNativeHtml).not.toContain('acp-banner')
  })

  test('parseCustomArgs and resolveRuntimeList handle quotes and fallback defaults cleanly', () => {
    expect(parseCustomArgs('acp --cwd "/tmp/my app" --stdio')).toEqual(['acp', '--cwd', '/tmp/my app', '--stdio'])
    expect(resolveRuntimeList([]).map(a => a.id)).toEqual(['xerxes', 'claude-code', 'antigravity'])
  })

  test('AcpRuntimePicker, AcpActiveBanner, Welcome brand emblem, ActivityGroup, and VscodeChat render cleanly', () => {
    const nativeSnap = sampleSnapshot()
    const pickerHtml = renderToStaticMarkup(
      createElement(AcpRuntimePicker, { snap: nativeSnap, defaultOpen: true }),
    )
    expect(pickerHtml).toContain('aria-label="Agent runtime"')
    expect(pickerHtml).toContain('Xerxes')
    expect(pickerHtml).toContain('Built-in runtime')
    expect(pickerHtml).toContain('Claude Code')
    expect(pickerHtml).toContain('Claude Code CLI')
    expect(pickerHtml).toContain('/Users/erfan/.local/bin/claude')
    expect(pickerHtml).toContain('Antigravity')
    expect(pickerHtml).toContain('Not installed')
    expect(pickerHtml).toContain('Custom ACP…')
    // Clean Claude Code menu: zero chunky header bar, transport pill badges, green status dots, backtick jargon, or footer bloat
    expect(pickerHtml).not.toContain('acp-pop__head')
    expect(pickerHtml).not.toContain('acp-row__transport')
    expect(pickerHtml).not.toContain('acp-row__dot')
    expect(pickerHtml).not.toContain('acp-pop__foot')
    expect(pickerHtml).not.toContain('stdio JSON-RPC')
    expect(pickerHtml).not.toContain('This thread · also')
    expect(pickerHtml).not.toContain('Use custom command…')

    expect(renderToStaticMarkup(createElement(AcpActiveBanner, { snap: nativeSnap }))).toBe('')

    const boundSnap = sampleSnapshot({ acpBinding: sampleClaudeBinding })
    const bannerHtml = renderToStaticMarkup(createElement(AcpActiveBanner, { snap: boundSnap }))
    expect(bannerHtml).toBe('')

    // Desktop Shell Welcome renders XerxesBrandEmblem + quiet borderless AcpRuntimePicker (no noisy connected banner)
    ;(globalThis as Record<string, unknown>).window = { xerxes: createMockBridge().xerxes }
    const shellHtml = renderToStaticMarkup(createElement(Shell, { snap: boundSnap }))
    expect(shellHtml).toContain('xerxes-brand-emblem')
    expect(shellHtml).toContain('composer__runtime')
    expect(shellHtml).toContain('Claude Code')
    expect(shellHtml).not.toContain('Connected via')

    // VS Code VscodeChat renders quiet header ACP pill + composer runtime switcher
    const vscodeHtml = renderToStaticMarkup(createElement(VscodeChat, { snap: boundSnap }))
    expect(vscodeHtml).toContain('xv-pill--acp')
    expect(vscodeHtml).toContain('composer__runtime')
    expect(vscodeHtml).toContain('Claude Code')
    expect(vscodeHtml).not.toContain('Connected via')

    // ActivityGroup delegates to ToolTimelineGroup cleanly
    const blocks: readonly Block[] = [
      {
        kind: 'tools',
        id: 1,
        running: false,
        items: [
          {
            id: 't1',
            verb: 'ReadFile',
            name: 'ReadFile',
            arg: 'App.tsx',
            dur: '0.4s',
            state: 'done',
            input: '{"path":"App.tsx"}',
            output: 'export function App() {}',
          },
        ],
      },
    ]
    const groupHtml = renderToStaticMarkup(
      createElement(ActivityGroup, { blocks, active: false, turnActive: false }),
    )
    expect(groupHtml.length).toBeGreaterThan(0)
  })
})

describe('Antigravity ACP (agy acp) Native UI Rendering, Approvals, Thinking & Chrome', () => {
  test('PascalCase AGY read_file, edit_file, write_file, run_command, structured diff, envelope stripping & foldChange', async () => {
    // 1. blocks.ts name normalization & editStatsOf
    expect(isEditTool('edit_file')).toBe(true)
    expect(isEditTool('write_file')).toBe(true)
    expect(isEditTool('read_file')).toBe(false)
    expect(editStatsOf('edit_file', { Path: 'src/app.ts', Target: 'const a = 1', Replacement: 'const a = 2\nconst b = 3' })).toEqual({
      path: 'src/app.ts',
      adds: 2,
      dels: 1,
      isNew: false,
    })
    expect(editStatsOf('write_file', { Path: 'src/new.ts', Content: 'export const x = 1\nexport const y = 2\n' })).toEqual({
      path: 'src/new.ts',
      adds: 2,
      dels: 0,
      isNew: true,
    })
    // Structured whole-file diff trims unchanged prefix/suffix for accurate stats
    expect(editStatsOf(
      'edit_file',
      {},
      {
        path: 'src/large.ts',
        old_text: 'line1\nline2\nconst oldVal = 1\nline4\nline5\n',
        new_text: 'line1\nline2\nconst newVal = 2\nconst extra = 3\nline4\nline5\n',
      },
    )).toEqual({
      path: 'src/large.ts',
      adds: 2,
      dels: 1,
      isNew: false,
    })

    // 2. FileToolRow recognizes AGY PascalCase & structured diff
    expect(fileToolOf({
      name: 'read_file',
      input: JSON.stringify({ Path: '/repo/src/index.ts', StartLine: 15 }),
    })).toEqual({ kind: 'read', path: '/repo/src/index.ts', offset: 14 })

    const agyEditItem: ToolItem = {
      id: 'agy-edit-1',
      verb: 'edit_file',
      arg: '/repo/src/index.ts',
      dur: '0.3s',
      state: 'done',
      name: 'edit_file',
      input: JSON.stringify({ Path: '/repo/src/index.ts', Target: 'const v = 1', Replacement: 'const v = 2\nconst w = 3' }),
      output: 'Edited /repo/src/index.ts',
    }
    const agyEditTool = fileToolOf(agyEditItem)
    expect(agyEditTool).toEqual({
      kind: 'edit',
      path: '/repo/src/index.ts',
      before: 'const v = 1',
      after: 'const v = 2\nconst w = 3',
      wholeFile: false,
    })
    expect(fileToolStats(agyEditItem)).toEqual({ adds: 2, dels: 1 })
    const editRowHtml = renderToStaticMarkup(createElement(FileToolRow, { item: agyEditItem, tool: agyEditTool!, failed: false }))
    expect(editRowHtml).toContain('index.ts')
    expect(editRowHtml).toContain('>+2<')
    expect(editRowHtml).toContain('>−1<')
    expect(editRowHtml).toContain('filetool__line--del')
    expect(editRowHtml).toContain('filetool__line--add')

    // Structured diff fallback when arguments have no Target/Replacement
    const builder = new BlockBuilder()
    builder.push('tool_call', { id: 'agy-diff-only', name: 'edit_file', arguments: {}, locations: [{ path: 'src/onlyDiff.ts' }] })
    builder.push('tool_result', {
      tool_call_id: 'agy-diff-only',
      name: 'edit_file',
      duration_ms: 250,
      diff: {
        path: 'src/onlyDiff.ts',
        old_text: 'a\nb\nc\nd\nOLD\ne\nf\ng\nh\n',
        new_text: 'a\nb\nc\nd\nNEW\ne\nf\ng\nh\n',
      },
      result: 'Updated src/onlyDiff.ts',
    })
    const builtTools = builder.all().flatMap(b => b.kind === 'tools' ? b.items : [])
    expect(builtTools).toHaveLength(1)
    const builtDiffItem = builtTools[0]!
    expect(builtDiffItem.path).toBe('src/onlyDiff.ts')
    expect(fileToolStats(builtDiffItem)).toEqual({ adds: 1, dels: 1 })
    const diffOnlyRowHtml = renderToStaticMarkup(createElement(ToolCallRow, { item: builtDiffItem, label: 'Edit file' }))
    expect(diffOnlyRowHtml).toContain('onlyDiff.ts')
    expect(diffOnlyRowHtml).toContain('filetool__line--del')
    expect(diffOnlyRowHtml).toContain('filetool__line--add')
    expect(diffOnlyRowHtml).toContain('OLD')
    expect(diffOnlyRowHtml).toContain('NEW')

    // 3. Execution.tsx AGY run_command + exit_code + envelope stripping
    const rawEnvelope = 'Created At: 2026-11-22T08:00:00+03:00\nCompleted At: 2026-11-22T08:00:01+03:00\nCommand exited with code 0.\n\nCommand output:\nprobe-shell-ok\n'
    expect(stripAgyCommandEnvelope(rawEnvelope)).toEqual({
      stripped: true,
      text: 'probe-shell-ok\n',
      exitCode: 0,
    })

    const agyCmdOk: ToolItem = {
      id: 'agy-cmd-1',
      verb: 'run_command',
      arg: 'echo probe-shell-ok',
      dur: '0.4s',
      state: 'done',
      name: 'run_command',
      input: JSON.stringify({ CommandLine: 'echo probe-shell-ok', toolSummary: 'Verify shell probe', WaitMsBeforeAsync: 5000 }),
      output: JSON.stringify({ output: rawEnvelope, exit_code: 0 }),
    }
    expect(toolHasFailed(agyCmdOk)).toBe(false)
    expect(executionView(agyCmdOk)).toMatchObject({
      command: 'echo probe-shell-ok',
      summary: 'Verify shell probe',
      stdout: 'probe-shell-ok\n',
      exitCode: 0,
    })
    expect(classifyToolItem(agyCmdOk)).toMatchObject({
      category: 'shell',
      badge: 'Verify shell probe',
      target: 'echo probe-shell-ok',
      metaPill: 'exit 0',
      metaTone: 'ok',
    })
    const execDetailsHtml = renderToStaticMarkup(createElement(ExecutionDetails, { item: agyCmdOk }))
    const execVisibleHtml = execDetailsHtml.split('<details class="execution__raw">')[0] ?? execDetailsHtml
    expect(execVisibleHtml).toContain('echo probe-shell-ok')
    expect(execVisibleHtml).toContain('probe-shell-ok')
    expect(execVisibleHtml).not.toContain('Created At:')
    expect(execVisibleHtml).not.toContain('Command output:')

    const agyCmdFail: ToolItem = {
      ...agyCmdOk,
      id: 'agy-cmd-fail',
      output: JSON.stringify({ output: 'Created At: 2026-11-22T08:00:00Z\nCompleted At: 2026-11-22T08:00:01Z\nCommand exited with code 2\n\nCommand output:\nfatal: bad ref\n', exit_code: 2 }),
    }
    expect(toolHasFailed(agyCmdFail)).toBe(true)
    expect(executionView(agyCmdFail)).toMatchObject({ exitCode: 2, stdout: 'fatal: bad ref\n' })

    // 4. ToolTimelineGroup accurately summarizes AGY edits, writes, reads, and commands
    const agyWriteItem: ToolItem = {
      id: 'agy-write-1',
      verb: 'write_file',
      arg: '/repo/src/created.ts',
      dur: '0.2s',
      state: 'done',
      name: 'write_file',
      input: JSON.stringify({ Path: '/repo/src/created.ts', Content: 'export const ok = true\nexport const n = 2\n' }),
      output: 'Wrote file',
    }
    const agyReadItem: ToolItem = {
      id: 'agy-read-1',
      verb: 'read_file',
      arg: '/repo/README.md',
      dur: '0.1s',
      state: 'done',
      name: 'read_file',
      input: JSON.stringify({ Path: '/repo/README.md', StartLine: 1 }),
      output: '# Xerxes\n',
    }
    const timelineBlocks: readonly Block[] = [
      { kind: 'tools', id: 1, running: false, items: [agyEditItem, agyWriteItem, agyCmdOk, agyReadItem] },
    ]
    const timelineStats = summarizeToolTimeline(timelineBlocks, false)
    expect(timelineStats.adds).toBe(4)
    expect(timelineStats.dels).toBe(1)
    expect(timelineStats.headline).toContain('Edited 2 files')
    expect(timelineStats.headline).toContain('Ran 1 command')
    expect(timelineStats.headline).toContain('Read 1 file')
    const timelineCollapsedHtml = renderToStaticMarkup(createElement(ToolTimelineGroup, { blocks: timelineBlocks, active: false, turnActive: false }))
    expect(timelineCollapsedHtml).toContain('Edited 2 files · Ran 1 command · Read 1 file')
    expect(timelineCollapsedHtml).toContain('+4 −1')
    expect(timelineCollapsedHtml).not.toContain('tool-timeline__status-icon--done')
    const timelineExpandedHtml = renderToStaticMarkup(createElement(ToolTimelineGroup, { blocks: timelineBlocks, active: false, turnActive: false, defaultOpen: true }))
    expect(timelineExpandedHtml).toContain('>+4<')
    expect(timelineExpandedHtml).toContain('>−1<')

    // 5. store.ts foldChange populates snap.changes for AGY PascalCase & structured diff calls
    const { store, bridge } = await setupConnectedStore()
    bridge.emit('turn_begin', { user_input: 'Update files' })
    bridge.emit('tool_call', {
      id: 'tc-agy-1',
      name: 'edit_file',
      arguments: { Path: 'src/a.ts', Target: 'oldA', Replacement: 'newA1\nnewA2' },
    })
    bridge.emit('tool_call', {
      id: 'tc-agy-2',
      name: 'write_file',
      arguments: { Path: 'src/b.ts', Content: 'line1\nline2\nline3\n' },
    })
    bridge.emit('tool_call', { id: 'tc-agy-3', name: 'edit_file', arguments: {} })
    bridge.emit('tool_result', {
      tool_call_id: 'tc-agy-3',
      name: 'edit_file',
      diff: { path: 'src/c.ts', old_text: 'top\nbeforeC\nend\n', new_text: 'top\nafterC\nend\n' },
      result: 'ok',
    })
    const changes = store.getSnapshot().changes
    expect(changes.map(c => c.path)).toEqual(['src/a.ts', 'src/b.ts', 'src/c.ts'])
    expect(changes[0]).toMatchObject({ path: 'src/a.ts', adds: 2, dels: 1, isNew: false })
    expect(changes[1]).toMatchObject({ path: 'src/b.ts', adds: 3, dels: 0, isNew: true })
    expect(changes[2]).toMatchObject({ path: 'src/c.ts', adds: 1, dels: 1, isNew: false })
  })

  test('Approval card displays AGY CommandLine, renders inline diff preview, respects allowPermanent === false, and updates keyboard handling', async () => {
    const { store, bridge } = await setupConnectedStore()

    // 1. AGY run_command approval with allow_permanent: false + explicit options + AGY plumbing keys hidden cleanly
    bridge.emit('approval_request', {
      id: 'appr-cmd-1',
      tool_name: 'run_command',
      action: 'run_command',
      inputs: {
        CommandLine: 'bun test xerxes/test/agyAcpIntegration.test.ts',
        WaitMsBeforeAsync: 5000,
        toolSummary: 'Run ACP integration tests',
        SafeToAutoRun: false,
        Blocking: true,
        IsDaemon: false,
      },
      cwd: '/workspace/xerxes',
      allow_permanent: false,
      options: [
        { option_id: 'allow', name: 'Allow', kind: 'allow_once' },
        { option_id: 'deny', name: 'Deny', kind: 'reject_once' },
      ],
    })
    const cmdApproval = store.getSnapshot().approval
    expect(cmdApproval?.allowPermanent).toBe(false)
    expect(cmdApproval?.options).toHaveLength(2)

    const cmdCardHtml = renderToStaticMarkup(
      createElement(ApprovalCard, { approval: cmdApproval!, policy: 'default' }),
    )
    expect(cmdCardHtml).toContain('bun test xerxes/test/agyAcpIntegration.test.ts')
    expect(cmdCardHtml).toContain('Run ACP integration tests')
    expect(cmdCardHtml).not.toContain('WaitMsBeforeAsync')
    expect(cmdCardHtml).not.toContain('waitmsbeforeasync')
    expect(cmdCardHtml).not.toContain('5000')
    expect(cmdCardHtml).not.toContain('toolSummary')
    expect(cmdCardHtml).not.toContain('toolsummary')
    expect(cmdCardHtml).not.toContain('SafeToAutoRun')
    expect(cmdCardHtml).not.toContain('IsDaemon')
    expect(cmdCardHtml).not.toContain('approval__fields')
    expect(cmdCardHtml).not.toContain('Allow for this session')
    expect(cmdCardHtml).not.toContain('This session')
    expect(cmdCardHtml).toContain('Deny <kbd>2</kbd>')
    expect(cmdCardHtml.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])

    // Keyboard action helper respects allowPermanent === false
    expect(approvalKeyAction('1', false)).toBe('allow_once')
    expect(approvalKeyAction('2', false)).toBe('deny')
    expect(approvalKeyAction('3', false)).toBe('deny')
    expect(approvalKeyAction('2', true)).toBe('allow_session')
    expect(approvalKeyAction('2', undefined)).toBe('allow_session')

    // Store refuses bogus allow_session when allowPermanent === false, and forwards option_id on allow_once
    bridge.calls.length = 0
    store.approve('appr-cmd-1', 'allow_session')
    expect(bridge.calls.some(c => c.method === 'permission_response')).toBe(false)
    store.approve('appr-cmd-1', 'allow_once')
    await Bun.sleep(5)
    expect(bridge.calls.find(c => c.method === 'permission_response')?.params).toMatchObject({
      request_id: 'appr-cmd-1',
      response: 'approve',
      option_id: 'allow',
    })

    // 2. AGY edit_file approval with structured diff renders compact theme-dynamic inline diff preview
    bridge.emit('approval_request', {
      id: 'appr-diff-1',
      tool_name: 'edit_file',
      action: 'edit_file',
      inputs: { Path: '/workspace/xerxes/src/demo.ts', Target: 'const x = 1', Replacement: 'const x = 42' },
      diff: {
        path: '/workspace/xerxes/src/demo.ts',
        old_text: 'header\nconst x = 1\nfooter\n',
        new_text: 'header\nconst x = 42\nconst y = 7\nfooter\n',
      },
      allow_permanent: false,
    })
    const diffApproval = store.getSnapshot().approval
    expect(diffApproval?.diff).toEqual({
      path: '/workspace/xerxes/src/demo.ts',
      oldText: 'header\nconst x = 1\nfooter\n',
      newText: 'header\nconst x = 42\nconst y = 7\nfooter\n',
    })
    const diffCardHtml = renderToStaticMarkup(
      createElement(ApprovalCard, { approval: diffApproval!, policy: 'default' }),
    )
    expect(diffCardHtml).toContain('approval__diff')
    expect(diffCardHtml).toContain('/workspace/xerxes/src/demo.ts')
    expect(diffCardHtml).toContain('approval__diff-line--del')
    expect(diffCardHtml).toContain('approval__diff-line--add')
    expect(diffCardHtml).toContain('const x = 1')
    expect(diffCardHtml).toContain('const x = 42')
    expect(diffCardHtml).toContain('>+2<')
    expect(diffCardHtml).toContain('>−1<')
    expect(diffCardHtml).not.toContain('This session')
    expect(diffCardHtml.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])
  })

  test('AcpThinkingPicker + dynamic configOptions update per model, /plan syncs AGY mode, and ACP slash completions badge Antigravity cleanly', async () => {
    const carbonBinding: AcpSessionBinding = {
      ...sampleAntigravityBinding,
      current_model: 'gemini-4-carbon',
      current_model_name: 'Gemini 4 Carbon',
      current_mode: 'default',
      current_mode_name: 'Default',
      modes: [
        { id: 'default', name: 'Default', description: 'Ask before edits and commands' },
        { id: 'accept-edits', name: 'Accept edits', description: 'Auto-accept file edits' },
        { id: 'auto', name: 'Auto', description: 'Auto-accept safe actions' },
        { id: 'plan', name: 'Plan', description: 'Research and plan before changing files' },
      ],
      thought_level: 'medium',
      thought_level_config_id: 'thought_level',
      thought_levels: [
        { id: 'low', name: 'Low' },
        { id: 'medium', name: 'Medium' },
        { id: 'high', name: 'High' },
      ],
      config_options: [
        {
          id: 'thought_level',
          name: 'Thinking Level',
          category: 'thought_level',
          type: 'select',
          current_value: 'medium',
          options: [
            { value: 'low', name: 'Low' },
            { value: 'medium', name: 'Medium' },
            { value: 'high', name: 'High' },
          ],
        },
      ],
      available_commands: [
        { name: 'goal', description: 'Run an Antigravity autonomous goal loop' },
        { name: 'compact', description: 'Compact Antigravity session context' },
        { name: 'boost', description: 'Boost reasoning depth for this step' },
        { name: 'grill-me', description: 'Interactive design interview' },
      ],
    }

    const neonBinding: AcpSessionBinding = {
      ...carbonBinding,
      current_model: 'gemini-4-neon',
      current_model_name: 'Gemini 4 Neon',
      thought_level: 'xhigh',
      thought_levels: [
        { id: 'low', name: 'Low' },
        { id: 'medium', name: 'Medium' },
        { id: 'high', name: 'High' },
        { id: 'xhigh', name: 'Extra' },
      ],
    }

    // parseAcpSessionBinding derives thought_level / thought_levels / available_commands from config_options
    const parsedFromConfig = parseAcpSessionBinding({
      session_id: 's1',
      agent_id: 'antigravity',
      agent_name: 'Antigravity',
      provider: 'antigravity',
      transport: 'acp',
      cwd: '/workspace/xerxes',
      connected: true,
      config_options: [
        {
          id: 'thought_level',
          name: 'Thinking Level',
          category: 'thought_level',
          type: 'select',
          current_value: 'high',
          options: [
            { value: 'low', name: 'Low' },
            { value: 'high', name: 'High' },
          ],
        },
      ],
      available_commands: [{ name: '/boost', description: 'Boost reasoning' }],
    })
    expect(parsedFromConfig?.thought_level).toBe('high')
    expect(parsedFromConfig?.thought_levels?.map(l => l.id)).toEqual(['low', 'high'])
    expect(parsedFromConfig?.thought_level_config_id).toBe('thought_level')
    expect(parsedFromConfig?.available_commands).toEqual([{ name: 'boost', description: 'Boost reasoning' }])

    // Store connects, updates thought_level via acp.set_config_option, and updates dynamically on model switch
    let activeMockBinding: AcpSessionBinding = carbonBinding
    const { store, bridge } = await setupConnectedStore({
      'acp.connect': () => {
        activeMockBinding = carbonBinding
        return { ok: true, binding: activeMockBinding }
      },
      'acp.set_config_option': params => {
        activeMockBinding = {
          ...activeMockBinding,
          thought_level: String(params.value),
        }
        return { ok: true, binding: activeMockBinding }
      },
      'acp.set_model': params => {
        activeMockBinding = params.model_id === 'gemini-4-neon' ? neonBinding : carbonBinding
        return { ok: true, binding: activeMockBinding }
      },
      'acp.set_mode': params => {
        activeMockBinding = {
          ...activeMockBinding,
          current_mode: String(params.mode_id),
          current_mode_name: String(params.mode_id) === 'plan' ? 'Plan' : 'Default',
        }
        return { ok: true, binding: activeMockBinding }
      },
      complete: params => {
        const text = String(params.text ?? '')
        if (text.startsWith('/skill')) return { ok: true, completions: [] }
        return {
          ok: true,
          completions: [
            { value: '/goal', label: '/goal', meta: 'Manage Xerxes session goal', category: 'session' },
            { value: '/acp', label: '/acp', meta: 'External ACP agent control', category: 'session' },
          ],
        }
      },
      'turn.submit': () => ({ ok: true }),
    })

    expect(await store.connectAcpAgent('antigravity')).toBe(true)
    expect(store.getSnapshot().acpBinding?.thought_level).toBe('medium')

    // setAcpThoughtLevel calls acp.set_config_option
    bridge.calls.length = 0
    expect(await store.setAcpThoughtLevel('high')).toBe(true)
    expect(bridge.calls.find(c => c.method === 'acp.set_config_option')?.params).toMatchObject({
      config_id: 'thought_level',
      value: 'high',
    })
    expect(store.getSnapshot().acpBinding?.thought_level).toBe('high')

    // Model switch dynamically updates thought_levels (Carbon -> Neon adds Extra)
    expect(await store.setAcpModel('gemini-4-neon')).toBe(true)
    expect(store.getSnapshot().acpBinding?.thought_levels?.map(l => l.name)).toEqual(['Low', 'Medium', 'High', 'Extra'])

    const carbonSnap = sampleSnapshot({ acpBinding: carbonBinding })
    const carbonThinkingHtml = renderToStaticMarkup(createElement(AcpThinkingPicker, { snap: carbonSnap, defaultOpen: true }))
    expect(carbonThinkingHtml).toContain('Medium')
    expect(carbonThinkingHtml).toContain('High')
    expect(carbonThinkingHtml).not.toContain('Extra')

    const neonSnap = sampleSnapshot({ acpBinding: neonBinding })
    const neonThinkingHtml = renderToStaticMarkup(createElement(AcpThinkingPicker, { snap: neonSnap, defaultOpen: true }))
    expect(neonThinkingHtml).toContain('Extra')
    expect(neonThinkingHtml).toContain('aria-selected="true"')

    // Shell & VscodeChat render AcpThinkingPicker + aligned banner/pill
    const shellNeonHtml = renderToStaticMarkup(createElement(Shell, { snap: neonSnap }))
    expect(shellNeonHtml).toContain('composer__acp-thinking')
    expect(shellNeonHtml).toContain('Extra')
    expect(shellNeonHtml).toContain('Default')
    expect(shellNeonHtml).not.toContain('composer__perm')

    const vscodeNeonHtml = renderToStaticMarkup(createElement(VscodeChat, { snap: neonSnap }))
    expect(vscodeNeonHtml).toContain('xv-pill--acp')
    expect(vscodeNeonHtml).toContain('Antigravity · Extra · Default')
    expect(vscodeNeonHtml).toContain('composer__acp-thinking')

    // /plan and /plan off sync AGY mode via acp.set_mode (never native plan_mode)
    bridge.calls.length = 0
    expect(await store.submit('/plan')).toBe(true)
    expect(bridge.calls.find(c => c.method === 'acp.set_mode')?.params).toMatchObject({ mode_id: 'plan' })
    expect(bridge.calls.some(c => c.method === 'plan_mode')).toBe(false)

    bridge.calls.length = 0
    expect(await store.submit('/plan off')).toBe(true)
    expect(bridge.calls.find(c => c.method === 'acp.set_mode')?.params).toMatchObject({ mode_id: 'default' })

    // ACP available_commands appear in slash completions badged Antigravity; Xerxes native commands are isolated out
    const hints = await store.completeText('/')
    const xerxesGoal = hints.find(item => item.value === '/goal' && !item.runtimeProvider)
    const agyGoal = hints.find(item => item.value === '/goal' && item.runtimeProvider === 'antigravity')
    const agyBoost = hints.find(item => item.value === '/boost' && item.runtimeProvider === 'antigravity')
    const hostAcp = hints.find(item => item.value === '/acp')
    expect(xerxesGoal).toBeUndefined()
    expect(agyGoal?.badge).toBe('Antigravity')
    expect(agyBoost?.badge).toBe('Antigravity')
    expect(hostAcp?.badge).toBe('Host')

    // Submitting colliding /goal OR AGY-only /boost routes straight to turn.submit (never native slash/goal)
    bridge.calls.length = 0
    expect(await store.submit('/boost inspect the diff')).toBe(true)
    expect(bridge.calls.some(c => c.method === 'turn.submit')).toBe(true)
    expect(bridge.calls.some(c => c.method === 'slash')).toBe(false)

    bridge.emit('turn_end', { session_id: store.getSnapshot().currentId })
    bridge.calls.length = 0
    expect(await store.submit('/goal run overnight checks')).toBe(true)
    expect(bridge.calls.some(c => c.method === 'turn.submit')).toBe(true)
    expect(bridge.calls.some(c => c.method === 'session.goal')).toBe(false)
  })

  test('Visible acpError banner with auth hint, Retry/Dismiss, savedAcpBinding Reconnect banner, crash recovery, and Escape closing popovers', async () => {
    // 1. Disconnected auth error surfaces visible banner with agy acp --login hint + Retry + Dismiss
    const authErrSnap = sampleSnapshot({
      acpBinding: null,
      acpError: 'Authentication required (-32000) for Antigravity',
    })
    const authBannerHtml = renderToStaticMarkup(createElement(AcpActiveBanner, { snap: authErrSnap }))
    expect(authBannerHtml).toContain('acp-banner--error')
    expect(authBannerHtml).toContain('acp-banner__head')
    expect(authBannerHtml).toContain('Authentication required')
    expect(authBannerHtml).toContain('agy acp --login')
    expect(authBannerHtml).toContain('Retry')
    expect(authBannerHtml).toContain('Dismiss runtime error')

    // Multi-method Antigravity auth error stacks vertically with wrapping action row and deduplicates terminal auth
    const multiAuthHtml = renderToStaticMarkup(createElement(AcpActiveBanner, {
      snap: sampleSnapshot({
        acpBinding: null,
        acpError: 'Authentication required (-32000) for Antigravity',
        acpAuthMethods: [
          { id: 'google', name: 'Log in with Google', type: 'agent' },
          { id: 'gemini-ent', name: 'Log in with Gemini Enterprise', type: 'agent' },
          { id: 'api-key', name: 'Gemini API key', type: 'agent' },
          { id: 'ent-platform', name: 'Gemini Enterprise Agent Platform', type: 'agent' },
          { id: 'terminal', name: 'Log in from the terminal', type: 'terminal' },
        ],
      }),
    }))
    expect(multiAuthHtml).toContain('acp-banner__head')
    expect(multiAuthHtml).toContain('Login with Antigravity')
    expect(multiAuthHtml).toContain('Log in with Google')
    expect(multiAuthHtml).not.toContain('Log in from the terminal')
    const appCss = await Bun.file(new URL('../src/desktop/renderer/app.css', import.meta.url)).text()
    expect(appCss).toMatch(/\.acp-banner--error\s*\{[^}]*flex-direction:\s*column/s)
    expect(appCss).toMatch(/\.acp-banner__actions\s*\{[^}]*flex-wrap:\s*wrap/s)
    // Verify balanced CSS braces (no orphaned closing brace syntax errors)
    const openBraces = (appCss.match(/\{/g) ?? []).length
    const closeBraces = (appCss.match(/\}/g) ?? []).length
    expect(openBraces).toBe(closeBraces)

    // 2. Connected session with runtime error renders a single quiet error banner with Retry, Switch to Xerxes, and Dismiss
    const connectedErrSnap = sampleSnapshot({
      acpBinding: sampleAntigravityBinding,
      acpError: 'Model switch failed',
    })
    const connectedErrHtml = renderToStaticMarkup(createElement(AcpActiveBanner, { snap: connectedErrSnap }))
    expect(connectedErrHtml).toContain('acp-banner--error')
    expect(connectedErrHtml).not.toContain('Connected via')
    expect(connectedErrHtml).toContain('Model switch failed')
    expect(connectedErrHtml).toContain('Retry')
    expect(connectedErrHtml).toContain('Switch to Xerxes')
    expect(connectedErrHtml).toContain('Dismiss runtime error')

    // 3. Reopened session with saved_acp_binding renders one-click Reconnect Antigravity banner & resumes
    const savedOnly: AcpSessionBinding = {
      ...sampleAntigravityBinding,
      connected: false,
    }
    const { store, bridge } = await setupConnectedStore({
      initialize: params => ({
        ok: true,
        session_id: 'sess-reopen-1',
        session_key: String(params.session_key ?? 'desktop-test'),
        cwd: '/workspace/xerxes',
        model: 'claude-opus-4-6',
        saved_acp_binding: savedOnly,
        session: {
          id: 'sess-reopen-1',
          title: 'Reopened AGY Session',
          cwd: '/workspace/xerxes',
          model: 'claude-opus-4-6',
          messages: [],
        },
      }),
      'acp.connect': () => ({ ok: true, binding: sampleAntigravityBinding }),
    })
    expect(store.getSnapshot().acpBinding).toBeNull()
    expect(store.getSnapshot().savedAcpBinding?.agent_id).toBe('antigravity')
    const reconnectHtml = renderToStaticMarkup(createElement(AcpActiveBanner, { snap: store.getSnapshot() }))
    expect(reconnectHtml).toContain('acp-banner--reconnect')
    expect(reconnectHtml).toContain('Reconnect Antigravity')
    expect(reconnectHtml).toContain('39c61d78')

    bridge.calls.length = 0
    expect(await store.reconnectSavedAcp()).toBe(true)
    expect(bridge.calls.find(c => c.method === 'acp.connect')?.params).toMatchObject({
      agent_id: 'antigravity',
      resume_session_id: '39c61d78-9f1a-4b2c-98f0-123456789abc',
      mode_id: 'planning',
    })
    expect(store.getSnapshot().acpBinding?.connected).toBe(true)
    expect(store.getSnapshot().savedAcpBinding).toBeNull()

    // Turbo mode renders right after Auto in AcpModePicker with description & data-mode="turbo"
    const turboBinding: AcpSessionBinding = {
      ...sampleAntigravityBinding,
      current_mode: 'turbo',
      current_mode_name: 'Turbo',
      modes: [
        { id: 'default', name: 'Default', description: 'Ask before edits and commands.' },
        { id: 'accept-edits', name: 'Accept edits', description: 'Approve file edits without asking.' },
        { id: 'auto', name: 'Auto', description: 'Approve edits and safe commands.' },
        { id: 'turbo', name: 'Turbo', description: 'Run all commands without restrictions.' },
        { id: 'plan', name: 'Plan', description: 'Plan carefully before executing a task.' },
      ],
    }
    const turboPickerHtml = renderToStaticMarkup(
      createElement(AcpModePicker, { snap: sampleSnapshot({ acpBinding: turboBinding }), defaultOpen: true }),
    )
    expect(turboPickerHtml).toContain('data-mode="turbo"')
    expect(turboPickerHtml).toContain('Run all commands without restrictions.')
    const turboListHtml = turboPickerHtml.split('role="listbox"')[1] ?? ''
    expect(turboListHtml.indexOf('>Auto<')).toBeLessThan(turboListHtml.indexOf('>Turbo<'))
    expect(turboListHtml.indexOf('>Turbo<')).toBeLessThan(turboListHtml.indexOf('>Plan<'))

    // Unexpected crash via acp_status preserves savedAcpBinding for Retry and surfaces acpError
    bridge.emit('acp_status', {
      session_id: 'sess-reopen-1',
      binding: { ...sampleAntigravityBinding, connected: false, error: 'Antigravity process exited (code 1)' },
    })
    expect(store.getSnapshot().acpBinding).toBeNull()
    expect(store.getSnapshot().acpError).toContain('exited (code 1)')
    expect(store.getSnapshot().savedAcpBinding?.agent_id).toBe('antigravity')
    bridge.calls.length = 0
    expect(await store.retryAcpConnect()).toBe(true)
    expect(bridge.calls.some(c => c.method === 'acp.connect')).toBe(true)

    // 4. Escape capture listener closes open ACP popovers even when focus moved outside
    const listeners = new Set<(event: globalThis.KeyboardEvent) => void>()
    const fakeTarget = {
      addEventListener: (_type: 'keydown', fn: (event: globalThis.KeyboardEvent) => void) => { listeners.add(fn) },
      removeEventListener: (_type: 'keydown', fn: (event: globalThis.KeyboardEvent) => void) => { listeners.delete(fn) },
    }
    let closedCount = 0
    const detach = attachEscapeCloseListener(fakeTarget, () => { closedCount += 1 })
    expect(listeners.size).toBe(1)

    let prevented = false
    let stopped = false
    let stoppedImmediate = false
    const makeKey = (key: string) => ({
      key,
      preventDefault: () => { prevented = true },
      stopPropagation: () => { stopped = true },
      stopImmediatePropagation: () => { stoppedImmediate = true },
    }) as unknown as globalThis.KeyboardEvent

    for (const fn of listeners) fn(makeKey('Enter'))
    expect(closedCount).toBe(0)
    expect(prevented).toBe(false)

    for (const fn of listeners) fn(makeKey('Escape'))
    expect(closedCount).toBe(1)
    expect(prevented).toBe(true)
    expect(stopped).toBe(true)
    expect(stoppedImmediate).toBe(true)

    detach()
    expect(listeners.size).toBe(0)
  })

  test('mid-turn reattach hydrates session.inflight prompt/tools/elapsed time and fresh reconnect_events preserve live transcript', async () => {
    // 1. Direct unit test of applyInflightTurn + deduplication
    const builder = new BlockBuilder()
    const startedSec = Math.floor((Date.now() - 870_000) / 1000) // ~14m 30s ago
    const hydrated = applyInflightTurn(builder, {
      user: 'Generate Qwen 3.8-27B eSurge performance report on v5p',
      thinking: 'Understanding DP replica management...',
      assistant: 'Measuring throughput across replicas.',
      started_at: startedSec,
      tools: [
        { id: 't1', name: 'ReadFile', arguments: '{"path":"esurge.py"}', context: 'esurge.py', ok: true, duration_ms: 120 },
        { id: 't2', name: 'BashTool', arguments: '{"command":"pytest…', context: 'pytest tests/dp' },
      ],
    })
    expect(hydrated).toEqual({
      user: 'Generate Qwen 3.8-27B eSurge performance report on v5p',
      startedAtMs: startedSec * 1000,
    })
    expect(inflightStartedAtMs({ started_at: startedSec })).toBe(startedSec * 1000)
    const blocks = builder.snapshot(true)
    expect(blocks[0]).toMatchObject({ kind: 'user', text: 'Generate Qwen 3.8-27B eSurge performance report on v5p' })
    expect(blocks.some(b => b.kind === 'thinking' && b.text.includes('Understanding DP replica management'))).toBe(true)
    const toolsBlock = blocks.find(b => b.kind === 'tools')
    expect(toolsBlock && toolsBlock.kind === 'tools' ? toolsBlock.items.map(item => ({ id: item.id, state: item.state, arg: item.arg })) : []).toEqual([
      { id: 't1', state: 'done', arg: 'esurge.py' },
      { id: 't2', state: 'working', arg: 'pytest tests/dp' },
    ])
    expect(blocks.at(-1)).toMatchObject({ kind: 'agent', text: 'Measuring throughput across replicas.' })

    // Re-applying same user prompt does not duplicate the trailing user block
    const dedupBuilder = new BlockBuilder()
    dedupBuilder.pushUser('Keep single prompt')
    applyInflightTurn(dedupBuilder, { user: 'Keep single prompt' })
    expect(dedupBuilder.snapshot(true).filter(b => b.kind === 'user')).toHaveLength(1)

    // 2. Store attach/reconnect to an in-flight Turn 1 (empty history.actions, non-empty session.inflight)
    const inflightStartedSec = Math.floor((Date.now() - 900_000) / 1000)
    let initCalls = 0
    const { store, bridge } = await setupConnectedStore({
      initialize: params => {
        initCalls += 1
        const isResumeOrReconnect = initCalls > 1
        return {
          ok: true,
          session_id: 'sess-inflight-1',
          cwd: '/repo',
          model: 'gemini-4-neon',
          session: {
            id: 'sess-inflight-1',
            key: String(params.session_key ?? 'desktop-inflight'),
            cwd: '/repo',
            active_turn_id: 'turn-live-1',
            history: { actions: [], before: null, has_more: false },
            inflight: {
              user: 'Benchmark Qwen 3.8-27B on TPU v5p',
              thinking: 'Inspecting DP replicas',
              started_at: inflightStartedSec,
              tools: [
                { id: 'tc-live', name: 'ReadFile', context: 'dp_manager.py' },
              ],
            },
          },
          ...(isResumeOrReconnect
            ? {
                reconnect_events: [
                  {
                    method: 'event',
                    params: {
                      type: 'tool_result',
                      payload: { session_id: 'sess-inflight-1', tool_call_id: 'tc-live', name: 'ReadFile', return_value: 'ok', duration_ms: 85 },
                    },
                  },
                ],
              }
            : {}),
        }
      },
    })
    const afterAttach = store.getSnapshot()
    expect(afterAttach.turnActive).toBe(true)
    expect(afterAttach.turnStartedAt).toBe(inflightStartedSec * 1000)
    expect(afterAttach.turnSeconds).toBeGreaterThanOrEqual(890)
    expect(afterAttach.blocks[0]).toMatchObject({ kind: 'user', text: 'Benchmark Qwen 3.8-27B on TPU v5p' })
    expect(afterAttach.blocks.some(b => b.kind === 'tools' && b.items.some(i => i.id === 'tc-live' && i.state === 'working'))).toBe(true)

    // 3. Reconnect WITH reconnect_events on a non-resume initialize (!resume_session_id) MUST NOT wipe the live user prompt!
    await (store as unknown as { initialize: (extra: Record<string, unknown>) => Promise<unknown> }).initialize({})
    const afterReconnect = store.getSnapshot()
    expect(afterReconnect.blocks[0]).toMatchObject({ kind: 'user', text: 'Benchmark Qwen 3.8-27B on TPU v5p' })
    expect(afterReconnect.blocks.some(b => b.kind === 'tools' && b.items.some(i => i.id === 'tc-live' && i.state === 'done'))).toBe(true)
    expect(bridge.calls.filter(c => c.method === 'initialize')).toHaveLength(2)
  })

  test('Plan & to-dos, Session edits, and Log tabs expose a Back to conversation control, ThreadDetailsCard toggles plan, and CSS keeps .chat__back visible', async () => {
    const planSnap = sampleSnapshot({
      tab: 'plan',
      todos: [
        { id: '1', content: 'Audit DP replicas', status: 'completed' },
        { id: '2', content: 'Write performance report', status: 'in_progress' },
      ],
    })
    const emptyPlanHtml = renderToStaticMarkup(createElement(PlanTab, { snap: sampleSnapshot({ tab: 'plan', plan: null, todos: null }) }))
    const todoPlanHtml = renderToStaticMarkup(createElement(PlanTab, { snap: planSnap }))
    const changesHtml = renderToStaticMarkup(createElement(ChangesTab, { snap: sampleSnapshot({ tab: 'changes' }) }))
    const logHtml = renderToStaticMarkup(createElement(LogTab, { snap: sampleSnapshot({ tab: 'log' }) }))

    for (const html of [emptyPlanHtml, todoPlanHtml, changesHtml, logHtml]) {
      expect(html).toContain('Back to conversation')
      expect(html).toContain('workspace-return__btn')
    }
    expect(todoPlanHtml).toContain('1/2 done')

    const detailsHtml = renderToStaticMarkup(
      createElement(ThreadDetailsCard, { snap: planSnap, onNavigate: () => {} }),
    )
    expect(detailsHtml).toContain('Plan &amp; to-dos')
    expect(detailsHtml).toContain('aria-pressed="true"')
    expect(detailsHtml).toContain('is-selected')

    // layout.css must NOT hide .chat__back when hiding generic .chat__view buttons
    const layoutCss = await Bun.file(new URL('../src/desktop/renderer/layout.css', import.meta.url)).text()
    expect(layoutCss).toContain('.chat__view:not(.chat__back)')
    expect(layoutCss).not.toContain(':is(.agent-menu,.chat__kind,.chat__view){display:none}')
  })

  test('Store.connectAcpAgent defaults Antigravity to Turbo mode while honoring explicit or saved modes', async () => {
    const { store, bridge } = await setupConnectedStore({
      'acp.connect': params => ({
        ok: true,
        binding: {
          ...sampleAntigravityBinding,
          current_mode: String(params.mode_id ?? 'turbo'),
          current_mode_name: String(params.mode_id ?? 'turbo') === 'plan' ? 'Plan' : 'Turbo',
        },
      }),
    })

    // Fresh Antigravity connect passes mode_id: 'turbo'
    bridge.calls.length = 0
    expect(await store.connectAcpAgent('antigravity')).toBe(true)
    expect(bridge.calls.find(c => c.method === 'acp.connect')?.params).toMatchObject({ agent_id: 'antigravity', mode_id: 'turbo' })

    // Explicit modeId overrides default Turbo
    bridge.calls.length = 0
    expect(await store.connectAcpAgent('antigravity', undefined, { modeId: 'plan' })).toBe(true)
    expect(bridge.calls.find(c => c.method === 'acp.connect')?.params).toMatchObject({ agent_id: 'antigravity', mode_id: 'plan' })

    // Non-Antigravity agent does NOT force mode_id: 'turbo'
    bridge.calls.length = 0
    expect(await store.connectAcpAgent('claude-code')).toBe(true)
    expect(bridge.calls.find(c => c.method === 'acp.connect')?.params.mode_id).toBeUndefined()
  })

  test('Activity metrics hydrate from session.inflight.tools during Turn 1 and fallback extracts AGY background jobs & schedules', async () => {
    const startedSec = Math.floor((Date.now() - 420_000) / 1000)
    const { store } = await setupConnectedStore({
      initialize: params => ({
        ok: true,
        session_id: 'sess-activity-1',
        cwd: '/repo',
        model: 'gemini-4-neon',
        session: {
          id: 'sess-activity-1',
          key: String(params.session_key ?? 'desktop-activity'),
          cwd: '/repo',
          turn_count: 0,
          active_turn_id: 'turn-1',
          history: { actions: [], before: null, has_more: false },
          inflight: {
            user: 'Serve Qwen 3.8-27B on TPU v5p',
            started_at: startedSec,
            tools: [
              { id: 'tc-1', name: 'run_command', context: 'eray tpu job status', result: 'ok', duration_ms: 1800 },
              { id: 'tc-2', name: 'schedule', context: 'in 720s · Check compilation progress' },
            ],
          },
        },
      }),
    })

    const snap = store.getSnapshot()
    expect(snap.turnActive).toBe(true)
    expect(snap.turnCount).toBe(1)
    expect(snap.toolSteps).toBe(2)
    expect(snap.toolDurationMs).toBe(1800)
    expect(snap.llmSteps).toBeGreaterThanOrEqual(1)
    expect(snap.metricPhase).toBe('tool')

    // Fallback extractor recovers live AGY schedules & background shells from blocks when remote daemon omits them
    const blocks: readonly Block[] = [
      {
        kind: 'tools',
        id: 'tg-1',
        items: [
          {
            id: 'tc-cmd',
            name: 'run_command',
            verb: 'Run command',
            arg: 'pytest tests/dp',
            input: JSON.stringify({ CommandLine: 'pytest tests/dp', toolSummary: 'DP benchmark' }),
            output: 'Tool is running as a background task with task id: conv-1/task-7\nTask logs are available at: /tmp/task-7.log',
            state: 'done',
          },
          {
            id: 'tc-sched',
            name: 'schedule',
            verb: 'Schedule',
            arg: 'in 720s · Check esurge compilation progress',
            input: JSON.stringify({ DurationSeconds: 720, Prompt: 'Check esurge compilation progress' }),
            output: 'Tool is running as a background task with task id: conv-1/task-8',
            state: 'done',
          },
          {
            id: 'tc-killed-sched',
            name: 'schedule',
            verb: 'Schedule',
            arg: 'in 60s · Old check',
            input: JSON.stringify({ DurationSeconds: 60, Prompt: 'Old check' }),
            output: 'Tool is running as a background task with task id: conv-1/task-9',
            state: 'done',
          },
          {
            id: 'tc-kill',
            name: 'manage_task',
            verb: 'Manage task',
            arg: 'kill',
            input: JSON.stringify({ Action: 'kill', TaskId: 'conv-1/task-9' }),
            output: 'Killed conv-1/task-9',
            state: 'done',
          },
        ],
      },
    ]

    const extracted = extractAcpBackgroundRowsFromBlocks(blocks, [], true, startedSec * 1000)
    expect(extracted).toHaveLength(3)
    expect(extracted.find(r => r.id === 'agy-task:conv-1/task-7')).toMatchObject({
      kind: 'shell',
      state: 'running',
      label: 'DP benchmark',
      source: 'antigravity',
      synthetic: true,
    })
    expect(extracted.find(r => r.id === 'agy-task:conv-1/task-8')).toMatchObject({
      kind: 'schedule',
      state: 'scheduled',
      title: 'Check esurge compilation progress',
      durationSeconds: 720,
      source: 'antigravity',
    })
    expect(extracted.find(r => r.id === 'agy-task:conv-1/task-9')?.state).toBe('cancelled')

    // Never duplicates when daemon already reported Antigravity rows
    const deduped = extractAcpBackgroundRowsFromBlocks(
      blocks,
      [{ id: 'agy-task:conv-1/task-8', source: 'antigravity', kind: 'schedule', state: 'scheduled' }],
      true,
      startedSec * 1000,
    )
    expect(deduped).toEqual([])

    // ScheduleActivity renders live countdown/badge & Antigravity label cleanly
    const schedHtml = renderToStaticMarkup(
      createElement(ScheduleActivity, {
        row: {
          id: 'agy-task:conv-1/task-8',
          kind: 'schedule',
          source: 'antigravity',
          title: 'Check esurge compilation progress',
          label: 'in 12m',
          detail: 'Antigravity timer · 12m',
          state: 'scheduled',
          durationSeconds: 720,
          nextRunAt: new Date(Date.now() + 300_000).toISOString(),
        },
        online: true,
        busy: false,
        onCancel: () => {},
      }),
    )
    expect(schedHtml).toContain('schedule-activity')
    expect(schedHtml).toContain('Check esurge compilation progress')
    expect(schedHtml).toContain('in ')
  })

  test('vibes.css removes the composer middle shadow seam and uses refined crimson for the stop button', async () => {
    const vibesCss = await Bun.file(new URL('../src/desktop/renderer/vibes.css', import.meta.url)).text()
    expect(vibesCss).toContain('.composer__input{')
    expect(vibesCss).toContain('z-index:1')
    expect(vibesCss).toContain('box-shadow:0 16px 16px -10px rgb(0 0 0 / 32%)')
    expect(vibesCss).not.toContain('>.composer__toolbar{grid-row:3;grid-column:1 / -1;padding:0 14px 10px 14px;border-radius:0 0 var(--v-r-card) var(--v-r-card);box-shadow:0 12px 28px')
    expect(vibesCss).toContain('--x-stop-bg:#b82937')
    expect(vibesCss).toContain('.composer__send--stop{background:var(--x-stop-bg)')
    expect(vibesCss).not.toContain('.composer__send--stop{background:var(--x-failed)')
  })

  test('truncated JSON tool arguments from older remote daemons never leak raw JSON into status phrases or fall back to Run command command', () => {
    // 1. Truncated mid-CommandLine with env prefix + python command
    const truncatedCmd = '{"CommandLine":"ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu python3 scripts/bench_esurge.py --max-len 131072 --cc 32…'
    const builder = new BlockBuilder()
    applyInflightTurn(builder, {
      user: 'Benchmark Qwen 3.8-27B',
      tools: [
        { id: 'tc-trunc-1', name: 'run_command', arguments: truncatedCmd },
        { id: 'tc-trunc-2', name: 'run_command', arguments: '{"CommandLine":"ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu","toolSummary":"Benchmark Qwen 3.8…' },
        { id: 'tc-trunc-3', name: 'run_command', arguments: '{"WaitMsBeforeAsync":10000' },
      ],
    })
    const blocks = builder.snapshot(true)
    const toolBlock = blocks.find(b => b.kind === 'tools')
    expect(toolBlock?.kind).toBe('tools')
    if (toolBlock?.kind === 'tools') {
      const [first, second, third] = toolBlock.items
      expect(first!.arg).not.toContain('{"CommandLine"')
      expect(first!.arg).toContain('ENABLE_DISTRIBUTED_INIT=0')
      const summary1 = classifyToolItem(first!)
      expect(summary1.category).toBe('shell')
      expect(summary1.command).toContain('python3 scripts/bench_esurge.py')
      expect(summary1.target).not.toBe('command')

      const summary2 = classifyToolItem(second!)
      expect(summary2.badge).toBe('Benchmark Qwen 3.8')
      expect(summary2.command).toBe('ENABLE_DISTRIBUTED_INIT=0 JAX_PLATFORMS=cpu')

      // Partial JSON with no string command never leaks raw JSON into arg
      expect(third!.arg).toBe('')
    }
  })

  test('live status replaces spinners with traveling-light text shimmer and Vibes/Neo right-panel fullscreen hides .chat and expands .desktop-rail', async () => {
    const layoutCss = await Bun.file(new URL('../src/desktop/renderer/layout.css', import.meta.url)).text()
    const vibesCss = await Bun.file(new URL('../src/desktop/renderer/vibes.css', import.meta.url)).text()

    // Spinners removed on live activity header, composer status, and working rail head
    expect(layoutCss).toContain('.tool-timeline__mark-wrap{display:none}')
    expect(layoutCss).toContain('.composer-status>.agent-orb{display:none}')
    expect(layoutCss).toContain(".railstatus[data-tone='working'] .railstatus__orb{display:none}")
    expect(layoutCss).toContain('@keyframes live-text-shine')
    expect(layoutCss).toContain('prefers-reduced-motion:reduce')

    // Vibes/Neo right-panel fullscreen overrides .chat display:flex so the right panel fills the workspace
    expect(vibesCss).toContain('.app__body:is([data-context-full],[data-review])>.chat{display:none}')
    expect(vibesCss).toContain('.app__body[data-context-full]>.desktop-rail{flex:1 1 auto;width:auto;max-width:none;min-width:0}')
    expect(vibesCss).toContain('.app__body[data-context-full]>.panel-divider:last-of-type{display:none}')
  })

  test('store.promoteQueued interrupts active turn to steer immediately and turn_begin dedupes legacy "user: " steer continuations', async () => {
    let submittedTexts: string[] = []
    let steerCalls = 0
    const { store, bridge } = await setupConnectedStore({
      'turn.steer': () => {
        steerCalls += 1
        return { ok: true, disposition: 'queued_steer' }
      },
      'turn.cancel': () => {
        setTimeout(() => bridge.emit('turn_end', { session_id: 'sess-acp-1' }), 5)
        return { ok: true }
      },
      'turn.submit': params => {
        submittedTexts.push(String(params.text ?? ''))
        setTimeout(() => bridge.emit('turn_begin', { session_id: 'sess-acp-1', user_input: String(params.text ?? '') }), 5)
        return { ok: true }
      },
    })

    // Start an active turn and queue two follow-up messages
    bridge.emit('turn_begin', { session_id: 'sess-acp-1', user_input: 'Benchmark Qwen 3.8-27B' })
    expect(store.getSnapshot().turnActive).toBe(true)

    expect(await store.steer('Optimize GDN inference')).toBe(true)
    expect(await store.steer('Also log KV usage')).toBe(true)
    expect(store.getSnapshot().queue).toHaveLength(2)

    // Queued messages card exposes a "Steer now" button while turnActive
    ;(globalThis as Record<string, unknown>).window = { xerxes: bridge.xerxes }
    const shellHtml = renderToStaticMarkup(createElement(Shell, { snap: store.getSnapshot() }))
    expect(shellHtml).toContain('composer-queue__steer')
    expect(shellHtml).toContain('Steer now')

    // Promote first queued message: cancels current turn, submits promoted message once (no duplicate bubble), and re-queues second item
    const firstId = store.getSnapshot().queue[0]!.id
    expect(await store.promoteQueued(firstId)).toBe(true)
    expect(submittedTexts).toEqual(['Optimize GDN inference'])
    expect(bridge.calls.some(c => c.method === 'turn.cancel')).toBe(true)
    const userBubbles = store.getSnapshot().blocks.filter(b => b.kind === 'user' && b.text === 'Optimize GDN inference')
    expect(userBubbles).toHaveLength(1)
    expect(steerCalls).toBe(3) // 2 initial + 1 re-queued remaining item

    // Legacy remote daemon continuation sending `user: <same text>` MUST NOT add a duplicate `user: ` bubble!
    bridge.emit('turn_end', { session_id: 'sess-acp-1' })
    bridge.emit('turn_begin', { session_id: 'sess-acp-1', user_input: 'user: Also log KV usage' })
    const gdnOrKvBubbles = store.getSnapshot().blocks.filter(b => b.kind === 'user' && b.text.includes('Also log KV usage'))
    expect(gdnOrKvBubbles).toHaveLength(1)
    expect(gdnOrKvBubbles[0]!.text).toBe('Also log KV usage')
  })

  test('store.promoteQueued removes the earlier queued bubble by blockId even when intervening tool calls finish after it', async () => {
    const cancelParams: Array<Record<string, unknown>> = []
    const submittedTexts: string[] = []
    const { store, bridge } = await setupConnectedStore({
      'turn.steer': params => {
        // Mirror daemon emitting steer_input before replying
        bridge.emit('steer_input', {
          session_id: 'sess-acp-1',
          content: String(params.content ?? ''),
          client_steer_id: String(params.client_steer_id ?? ''),
        })
        return { ok: true, disposition: 'queued_steer' }
      },
      'turn.cancel': params => {
        cancelParams.push(params)
        setTimeout(() => bridge.emit('turn_end', { session_id: 'sess-acp-1', cancelled: true }), 5)
        return { ok: true }
      },
      'turn.submit': params => {
        submittedTexts.push(String(params.text ?? ''))
        setTimeout(() => bridge.emit('turn_begin', { session_id: 'sess-acp-1', user_input: String(params.text ?? '') }), 5)
        return { ok: true }
      },
    })

    // Connect Antigravity ACP with `/swarm` advertised as an ACP command (matching real agy acp)
    bridge.emit('acp_status', {
      session_id: 'sess-acp-1',
      binding: {
        ...sampleAntigravityBinding,
        available_commands: [{ name: 'swarm', description: 'Coordinate a swarm' }],
      },
    })

    // Turn 1 starts and runs tools
    bridge.emit('turn_begin', { session_id: 'sess-acp-1', user_input: 'Search GLM 5.4 loop transformer' })
    bridge.emit('tool_start', { session_id: 'sess-acp-1', call_id: 'c1', name: 'search_web', arguments: '{"query":"glm 5.4"}' })
    bridge.emit('tool_end', { session_id: 'sess-acp-1', call_id: 'c1', name: 'search_web', result: 'ok' })

    // User submits `/swarm ...` while Turn 1 is still active -> queued via `turn.steer`
    const followUp = '/swarm transform qwen 3.8-27b into a loop transformer'
    expect(await store.submit(followUp)).toBe(true)
    const queuedItem = store.getSnapshot().queue[0]!
    expect(queuedItem.blockId).toBeGreaterThan(0)

    // Intervening tool calls stream AFTER the queued user bubble and commit on turn_end!
    bridge.emit('tool_start', { session_id: 'sess-acp-1', call_id: 'c2', name: 'search_web', arguments: '{"query":"loop transformer"}' })
    bridge.emit('tool_end', { session_id: 'sess-acp-1', call_id: 'c2', name: 'search_web', error: 'cancelled' })

    // User clicks "Steer now" on the queued item
    expect(await store.promoteQueued(queuedItem.id)).toBe(true)
    await new Promise(resolve => setTimeout(resolve, 15))

    expect(cancelParams).toHaveLength(1)
    expect(cancelParams[0]?.drop_pending_steers).toBe(true)
    expect(submittedTexts).toEqual([followUp])
    const swarmBubbles = store.getSnapshot().blocks.filter(b => b.kind === 'user' && b.text === followUp)
    expect(swarmBubbles).toHaveLength(1)
  })

  test('BrowserPanel rectOverlapsAny detects overlapping dropdown menus and ignores disjoint elements', () => {
    const browserRect = { x: 700, y: 120, width: 600, height: 700 }
    // + menu opening from right-panel header overlapping top of WebContentsView
    expect(rectOverlapsAny(browserRect, [{ x: 1020, y: 52, width: 200, height: 260 }])).toBe(true)
    // Left sidebar menu far from browser rect
    expect(rectOverlapsAny(browserRect, [{ x: 12, y: 60, width: 220, height: 300 }])).toBe(false)
    // Zero-size hidden element
    expect(rectOverlapsAny(browserRect, [{ x: 800, y: 200, width: 0, height: 0 }])).toBe(false)
  })
})
