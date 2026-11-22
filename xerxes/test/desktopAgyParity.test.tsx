// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import {
  FloatingRail,
  GlobalKeys,
  VibesSideNav,
  isImmortalsShortcutEvent,
} from '../src/desktop/renderer/App.js'
import {
  ArtifactCard,
  artifactFromToolResultPayload,
  artifactOfTool,
} from '../src/desktop/renderer/ArtifactCard.js'
import {
  ConversationChatLink,
  SwarmChatLink,
  isSwarmBoardCommand,
  linkTargetOf,
  resolveConversationTarget,
} from '../src/desktop/renderer/chatLinks.js'
import { DesktopRail } from '../src/desktop/renderer/DesktopPanels.js'
import { ToolCallRow } from '../src/desktop/renderer/Execution.js'
import { Markdown, clearMarkdownCache } from '../src/desktop/renderer/markdown.js'
import { SubagentsSection } from '../src/desktop/renderer/RailSections.js'
import { Shortcuts } from '../src/desktop/renderer/Shortcuts.js'
import {
  Store,
  isAgentFamilyTool,
  isSpawnTool,
  shouldForwardSlashToAcp,
  showsImmortalsForSession,
  spawnMembersOf,
  store as singletonStore,
  type AcpSessionBinding,
  type Snapshot,
  type XerxesLike,
} from '../src/desktop/renderer/store.js'
import { SwarmBoard, resolveSwarmBoardModels } from '../src/desktop/renderer/SwarmBoard.js'
import type { SwarmMemberSnapshot, SwarmStatusSnapshot } from '../src/desktop/renderer/swarmState.js'
import {
  SwarmToolCard,
  extractAddedMembers,
  swarmSpawnSummaryOf,
} from '../src/desktop/renderer/SwarmToolCard.js'
import type { DaemonEvent, ToolItem } from '../src/desktop/renderer/types.js'

const HEX_COLOR_RE = /#[0-9a-fA-F]{6}\b/

const agyBinding: AcpSessionBinding = {
  session_id: 'sess-agy-1',
  agent_id: 'antigravity',
  name: 'Antigravity',
  provider: 'antigravity',
  transport: 'acp',
  cwd: '/workspace/prismcore',
  external_session_id: 'eaa65661-2d98-4bac-9995-0588436d16d8',
  connected: true,
  command: 'agy acp',
  current_model: 'gemini-4-neon-xhigh',
  current_model_name: 'Gemini 4 Neon',
  models: [
    { id: 'gemini-4-neon-xhigh', name: 'Gemini 4 Neon', description: 'Flagship reasoning' },
    { id: 'gemini-4-carbon-high', name: 'Gemini 4 Carbon', description: 'Fast engineering model' },
    { id: 'gemini-3-flash', name: 'Gemini 3 Flash', description: 'Low-latency edits' },
  ],
  available_commands: [
    { name: 'swarm', description: 'Coordinate a parallel Antigravity swarm', kind: 'command', source: 'Antigravity' },
    { name: 'compact', description: 'Compact conversation context', kind: 'command', source: 'Antigravity' },
  ],
  discovered_skills: [
    { name: 'run-workflow', description: 'Orchestrate a Python workflow', slash_command: '/run-workflow' },
  ],
}

const claudeBinding: AcpSessionBinding = {
  session_id: 'sess-claude-1',
  agent_id: 'claude-code',
  name: 'Claude Code',
  provider: 'claude-code',
  transport: 'claude-stream-json',
  cwd: '/workspace/prismcore',
  connected: true,
  available_commands: [
    { name: 'compact', description: 'Compact context', kind: 'command', source: 'Claude Code' },
  ],
}

const customSwarmAcpBinding: AcpSessionBinding = {
  session_id: 'sess-custom-1',
  agent_id: 'custom-acp',
  name: 'Custom Swarm ACP',
  provider: 'acp',
  transport: 'acp',
  cwd: '/workspace/prismcore',
  connected: true,
  available_commands: [
    { name: 'swarm', description: 'Start a swarm', kind: 'command', source: 'Custom' },
  ],
}

const agyMembers: readonly SwarmMemberSnapshot[] = [
  {
    name: 'coordinator',
    role: 'coordinator',
    model: 'gemini-4-neon-xhigh',
    model_display_name: 'Gemini 4 Neon',
    workspace: 'inherit',
    channels: ['main'],
    state: 'waiting',
    killed: false,
    last_read_seq: 9,
    conversation_id: 'eaa65661-2d98-4bac-9995-0588436d16d8',
  },
  {
    name: 'tx-config',
    role: 'worker',
    model: 'gemini-4-carbon-high',
    model_display_name: 'Gemini 4 Carbon',
    workspace: 'share',
    channels: ['main'],
    state: 'working',
    killed: false,
    last_read_seq: 7,
    conversation_id: '2c8b0f21-8b42-4c7a-9e01-123456789abc',
    current_activity: 'Editing tx_config.ts',
  },
  {
    name: 'tx-tests',
    role: 'worker',
    model: 'gemini-4-neon-xhigh',
    model_display_name: 'Gemini 4 Neon',
    workspace: 'branch',
    channels: ['main'],
    state: 'working',
    killed: false,
    last_read_seq: 8,
    conversation_id: '9f11a4c0-7e22-41bb-a011-abcdefabcdef',
    current_activity: 'Running bun test',
  },
  {
    name: 'monitor',
    role: 'monitor',
    model: 'gemini-3-flash',
    model_display_name: 'Gemini 3 Flash',
    workspace: 'inherit',
    channels: ['main'],
    state: 'waiting',
    killed: false,
    last_read_seq: 9,
    conversation_id: '7d3e9201-4a5b-4c6d-8e9f-001122334455',
  },
]

const agySwarmStatus: SwarmStatusSnapshot = {
  active: true,
  session_id: 'sess-agy-1',
  title: 'Prismcore Parity Swarm',
  board: 'local',
  open_board: '[Open the Immortals UI](sidecar://immortals/board/?auto_open=false)',
  head: 9,
  overview: null,
  channels: [
    { name: 'main', topic: 'Shared Immortals board', created_by: 'coordinator', archived: false, subscriber_count: 4 },
  ],
  members: agyMembers,
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

  const call = async (method: string, params: Record<string, unknown> = {}): Promise<Record<string, unknown>> => {
    calls.push({ method, params })
    const custom = overrides[method]
    if (custom) return custom(params)
    if (method === 'initialize') {
      return {
        ok: true,
        session_id: 'sess-agy-1',
        session_key: String(params.session_key ?? 'desktop-agy-test'),
        cwd: '/workspace/prismcore',
        model: 'claude-opus-4-6',
        acp_binding: agyBinding,
        session: {
          id: 'sess-agy-1',
          title: 'Antigravity Parity Session',
          cwd: '/workspace/prismcore',
          model: 'claude-opus-4-6',
          messages: [],
          acp_binding: agyBinding,
        },
      }
    }
    if (method === 'acp.agents') {
      return { ok: true, agents: [], active: agyBinding }
    }
    if (method === 'swarm.status') {
      return { ok: true, ...agySwarmStatus }
    }
    if (method === 'swarm.board') {
      return { ok: true, head: 9, messages: [] }
    }
    if (method === 'swarm.kill') {
      return { ok: true, killed: [String(params.name ?? 'all')] }
    }
    if (method === 'swarm.inspect') {
      const target = agyMembers.find(m => m.name === String(params.name ?? '')) ?? agyMembers[1]!
      return {
        ok: true,
        session_id: 'sess-agy-1',
        member: target,
        conversation_id: target.conversation_id ?? target.name,
        activity: { phase: 'tool', summary: target.current_activity ?? 'Working', active_tools: [], recent_tools: [] },
        messages: [],
        total_messages: 0,
      }
    }
    if (method === 'slash') {
      return { ok: true, output: `Handled ${String(params.command ?? '')}` }
    }
    if (method === 'session.list' || method === 'session.active_list') {
      return { ok: true, sessions: [] }
    }
    if (method === 'learning.list') {
      return { ok: true, proposals: [], lessons: [], counts: { pending_proposals: 0, pending_lessons: 0 } }
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

function sampleSnapshot(overrides: Partial<Snapshot> = {}): Snapshot {
  const base = new Store().getSnapshot()
  return {
    ...base,
    connection: 'online',
    noWorkspace: false,
    cwd: '/workspace/prismcore',
    branch: 'main',
    currentId: 'sess-agy-1',
    currentTitle: 'Antigravity Parity Session',
    sessionKey: 'desktop-agy-test',
    model: 'native-xerxes-model',
    models: [
      { id: 'native-xerxes-model', provider: 'Xerxes', label: 'Native Xerxes Model' },
      { id: 'gpt-6.2-astra', provider: 'OpenAI', label: 'GPT 6.2 Astra' },
    ],
    acpBinding: agyBinding,
    swarm: agySwarmStatus,
    ...overrides,
  }
}

const originalWindow = (globalThis as Record<string, unknown>).window

afterEach(() => {
  clearMarkdownCache()
  if (originalWindow === undefined) delete (globalThis as Record<string, unknown>).window
  else (globalThis as Record<string, unknown>).window = originalWindow
})

describe('Track B — Antigravity ACP Swarm, Alt+S, Links, Artifacts & Subagents Parity', () => {
  test('1. /swarm UI control commands stay host-owned while /swarm <task> forwards directly to agy acp', async () => {
    // Pure routing predicate
    expect(shouldForwardSlashToAcp('/swarm', agyBinding)).toBe(false)
    expect(shouldForwardSlashToAcp('/swarm board', agyBinding)).toBe(false)
    expect(shouldForwardSlashToAcp('/swarm ui', agyBinding)).toBe(false)
    expect(shouldForwardSlashToAcp('/swarm status', agyBinding)).toBe(false)
    expect(shouldForwardSlashToAcp('/swarm channels', agyBinding)).toBe(false)
    expect(shouldForwardSlashToAcp('/swarm stop', agyBinding)).toBe(false)
    expect(shouldForwardSlashToAcp('/swarm kill tx-config', agyBinding)).toBe(false)
    expect(shouldForwardSlashToAcp('/immortals', agyBinding)).toBe(false)
    expect(shouldForwardSlashToAcp('/immortals board', agyBinding)).toBe(false)
    expect(shouldForwardSlashToAcp('/immortals stop', agyBinding)).toBe(false)
    expect(shouldForwardSlashToAcp('/immortals audit codebase', agyBinding)).toBe(false)

    // Starting an ACP swarm with a task forwards straight to agy acp
    expect(shouldForwardSlashToAcp('/swarm fix the broken parity tests', agyBinding)).toBe(true)
    expect(shouldForwardSlashToAcp('/compact', agyBinding)).toBe(true)
    expect(shouldForwardSlashToAcp('/run-workflow audit', agyBinding)).toBe(true)

    // End-to-end Store.submit behavior on connected agyBinding
    const bridge = createMockBridge()
    ;(globalThis as Record<string, unknown>).window = { xerxes: bridge.xerxes }
    const s = new Store()
    s.start(bridge.xerxes)
    for (let i = 0; i < 40 && s.getSnapshot().connection !== 'online'; i += 1) {
      await Bun.sleep(5)
    }
    expect(s.getSnapshot().acpBinding?.provider).toBe('antigravity')

    // `/swarm board` opens Immortals Board locally (never turn.submit!)
    bridge.calls.length = 0
    expect(await s.submit('/swarm board')).toBe(true)
    expect(s.getSnapshot().swarmOpen).toBe(true)
    expect(bridge.calls.some(c => c.method === 'turn.submit')).toBe(false)
    expect(bridge.calls.some(c => c.method === 'swarm.status')).toBe(true)

    // `/swarm kill tx-config` invokes host `swarm.kill` RPC (never turn.submit!)
    bridge.calls.length = 0
    expect(await s.submit('/swarm kill tx-config')).toBe(true)
    expect(bridge.calls.find(c => c.method === 'swarm.kill')?.params.name).toBe('tx-config')
    expect(bridge.calls.some(c => c.method === 'turn.submit')).toBe(false)

    // `/swarm status` runs host `slash` RPC (never turn.submit!)
    bridge.calls.length = 0
    expect(await s.submit('/swarm status')).toBe(true)
    expect(bridge.calls.find(c => c.method === 'slash')?.params.command).toBe('/swarm status')
    expect(bridge.calls.some(c => c.method === 'turn.submit')).toBe(false)

    // `/swarm <task description>` forwards directly to `turn.submit` (never swarm.activate!)
    bridge.calls.length = 0
    expect(await s.submit('/swarm audit every module in parallel')).toBe(true)
    expect(bridge.calls.find(c => c.method === 'turn.submit')?.params.text).toBe('/swarm audit every module in parallel')
    expect(bridge.calls.some(c => c.method === 'swarm.activate')).toBe(false)
    bridge.emit('turn_end', { session_id: 'sess-agy-1' })
  })

  test('2. Alt+S (code KeyS, macOS Option+S ß-safe) shortcut matcher and Shortcuts reference entry', async () => {
    // macOS Option+S reports key === 'ß' and code === 'KeyS'
    expect(isImmortalsShortcutEvent({ altKey: true, metaKey: false, ctrlKey: false, shiftKey: false, code: 'KeyS' })).toBe(true)
    // Reject extra modifiers or wrong physical keys
    expect(isImmortalsShortcutEvent({ altKey: true, metaKey: true, ctrlKey: false, shiftKey: false, code: 'KeyS' })).toBe(false)
    expect(isImmortalsShortcutEvent({ altKey: true, metaKey: false, ctrlKey: true, shiftKey: false, code: 'KeyS' })).toBe(false)
    expect(isImmortalsShortcutEvent({ altKey: true, metaKey: false, ctrlKey: false, shiftKey: true, code: 'KeyS' })).toBe(false)
    expect(isImmortalsShortcutEvent({ altKey: false, metaKey: false, ctrlKey: false, shiftKey: false, code: 'KeyS' })).toBe(false)
    expect(isImmortalsShortcutEvent({ altKey: true, metaKey: false, ctrlKey: false, shiftKey: false, code: 'KeyA' })).toBe(false)

    // GlobalKeys accepts optional toggleSwarm without breaking callers
    const snap = sampleSnapshot()
    expect(renderToStaticMarkup(createElement(GlobalKeys, { snap, closeSurface: null }))).toBe('')
    expect(renderToStaticMarkup(createElement(GlobalKeys, { snap, closeSurface: null, toggleSwarm: () => {} }))).toBe('')

    // Shortcuts reference documents ⌥S cleanly
    expect(typeof Shortcuts).toBe('function')
    const shortcutsSource = await Bun.file(new URL('../src/desktop/renderer/Shortcuts.tsx', import.meta.url)).text()
    expect(shortcutsSource).toContain('⌥S')
    expect(shortcutsSource).toContain('Toggle Immortals Board')
  })

  test('3. Immortals entry points are visible on Antigravity ACP & swarm-capable ACP, hidden on idle Claude Code', () => {
    const agyIdleSnap = sampleSnapshot({ acpBinding: agyBinding, swarm: null })
    const claudeIdleSnap = sampleSnapshot({ acpBinding: claudeBinding, swarm: null })
    const claudeSwarmSnap = sampleSnapshot({ acpBinding: claudeBinding, swarm: agySwarmStatus })
    const customSwarmSnap = sampleSnapshot({ acpBinding: customSwarmAcpBinding, swarm: null })
    const nativeSnap = sampleSnapshot({ acpBinding: null, swarm: null })

    expect(showsImmortalsForSession(agyIdleSnap)).toBe(true)
    expect(showsImmortalsForSession(customSwarmSnap)).toBe(true)
    expect(showsImmortalsForSession(nativeSnap)).toBe(true)
    expect(showsImmortalsForSession(claudeIdleSnap)).toBe(false)
    expect(showsImmortalsForSession(claudeSwarmSnap)).toBe(true)

    // FloatingRail renders Immortals on Antigravity ACP even before a swarm is active
    const agyRailHtml = renderToStaticMarkup(
      createElement(FloatingRail, {
        open: () => {},
        externalAcp: true,
        swarmActive: false,
        showImmortals: showsImmortalsForSession(agyIdleSnap),
      }),
    )
    expect(agyRailHtml).toContain('aria-label="Immortals"')

    const claudeRailHtml = renderToStaticMarkup(
      createElement(FloatingRail, {
        open: () => {},
        externalAcp: true,
        swarmActive: false,
        showImmortals: showsImmortalsForSession(claudeIdleSnap),
      }),
    )
    expect(claudeRailHtml).not.toContain('aria-label="Immortals"')

    // VibesSideNav More menu shows Immortals for Antigravity ACP, hides for idle Claude Code
    const agyMoreHtml = renderToStaticMarkup(
      createElement(VibesSideNav, {
        snap: agyIdleSnap,
        page: null,
        filter: '',
        setFilter: () => {},
        defaultMoreOpen: true,
      }),
    )
    expect(agyMoreHtml).toContain('>Immortals<')

    const claudeMoreHtml = renderToStaticMarkup(
      createElement(VibesSideNav, {
        snap: claudeIdleSnap,
        page: null,
        filter: '',
        setFilter: () => {},
        defaultMoreOpen: true,
      }),
    )
    expect(claudeMoreHtml).not.toContain('>Immortals<')

    // DesktopRail passes hideSwarm = false on Antigravity ACP
    ;(globalThis as Record<string, unknown>).window = { xerxes: createMockBridge().xerxes }
    const desktopRailHtml = renderToStaticMarkup(
      createElement(DesktopRail, {
        panel: 'subagents',
        snap: agyIdleSnap,
        close: () => {},
        activityDetails: null,
      }),
    )
    expect(desktopRailHtml).toContain('rail-add__trigger')
  })

  test('4. SwarmBoard model switcher lists Antigravity ACP models instead of native Xerxes models', () => {
    const agySnap = sampleSnapshot({ acpBinding: agyBinding })
    const resolved = resolveSwarmBoardModels(agySnap)
    expect(resolved.map(m => m.id)).toEqual([
      'gemini-4-neon-xhigh',
      'gemini-4-carbon-high',
      'gemini-3-flash',
    ])
    expect(resolved.map(m => m.label)).toEqual([
      'Gemini 4 Neon',
      'Gemini 4 Carbon',
      'Gemini 3 Flash',
    ])
    expect(resolved.every(m => m.provider === 'Antigravity')).toBe(true)

    // Native sessions still get native snap.models
    const nativeResolved = resolveSwarmBoardModels(sampleSnapshot({ acpBinding: null }))
    expect(nativeResolved.map(m => m.id)).toEqual(['native-xerxes-model', 'gpt-6.2-astra'])

    // Rendered SwarmBoard with Switch model popover open for waiting `monitor` shows AGY models only
    const html = renderToStaticMarkup(
      createElement(SwarmBoard, {
        snap: agySnap,
        initialTab: 'members',
        initialModelSwitchMember: 'monitor',
      }),
    )
    expect(html).toContain('Gemini 4 Neon')
    expect(html).toContain('Gemini 4 Carbon')
    expect(html).toContain('Gemini 3 Flash')
    expect(html).not.toContain('Native Xerxes Model')
    expect(html).not.toContain('GPT 6.2 Astra')
  })

  test('5. SwarmToolCard parses singular Antigravity add payload and falls back to live roster when output is empty', () => {
    // 5a. Singular AGY response payload (`{ status: "added", name, model, model_display_name, conversation_id }`)
    const singularArgs = {
      action: 'add',
      name: 'tx-config',
      brief: 'Fix transaction config validation',
    }
    const singularRes = {
      status: 'added',
      name: 'tx-config',
      model: 'gemini-4-carbon-high',
      model_display_name: 'Gemini 4 Carbon',
      conversation_id: '2c8b0f21-8b42-4c7a-9e01-123456789abc',
      workspace_mode: 'share',
      open_board: 'Run /swarm board (default shortcut alt+s) to open the live board.',
    }
    const parsedSingular = extractAddedMembers(singularArgs, singularRes)
    expect(parsedSingular.displayTarget).toBe('tx-config')
    expect(parsedSingular.model).toBe('Gemini 4 Carbon')
    expect(parsedSingular.workspace).toBe('share')
    expect(parsedSingular.members).toHaveLength(1)
    expect(parsedSingular.members[0]?.conversationId).toBe('2c8b0f21-8b42-4c7a-9e01-123456789abc')

    // 5b. Empty ACP tool output falls back to `snap.swarm.members` for model & workspace
    const emptyOutputItem: ToolItem = {
      id: 'call-agy-spawn-1',
      verb: 'swarm',
      arg: '',
      dur: '0.4s',
      state: 'done',
      name: 'swarm',
      input: JSON.stringify({
        action: 'add',
        name: 'tx-config',
        brief: 'Inspect config and run tests',
      }),
      output: '',
    }
    const summary = swarmSpawnSummaryOf(emptyOutputItem, false, agyMembers)
    expect(summary).not.toBeNull()
    expect(summary?.displayTarget).toBe('tx-config')
    expect(summary?.model).toBe('Gemini 4 Carbon')
    expect(summary?.workspace).toBe('share')

    const cardHtml = renderToStaticMarkup(
      createElement(SwarmToolCard, {
        item: emptyOutputItem,
        swarmMembers: agyMembers,
        defaultExpanded: true,
      }),
    )
    expect(cardHtml).toContain('tx-config')
    expect(cardHtml).toContain('Gemini 4 Carbon')
    expect(cardHtml).toContain('share · worktree')
    expect(cardHtml).toContain('Open Immortals →')
  })

  test('6. SubagentsSection renders active Immortals workers jump CTA instead of a dead empty state', () => {
    // 2 running workers -> crisp inline CTA
    const runningHtml = renderToStaticMarkup(
      createElement(SubagentsSection, {
        rows: [],
        swarm: agySwarmStatus,
        onOpenSwarm: () => {},
      }),
    )
    expect(runningHtml).toContain('2 Immortals workers running')
    expect(runningHtml).toContain('Open Immortals →')
    expect(runningHtml).not.toContain('No subagents in this task yet.')
    expect(runningHtml).not.toMatch(HEX_COLOR_RE)

    // 1 running worker -> singular grammar
    const oneWorkerSwarm: SwarmStatusSnapshot = {
      ...agySwarmStatus,
      members: [agyMembers[0]!, agyMembers[1]!],
    }
    const singularHtml = renderToStaticMarkup(
      createElement(SubagentsSection, {
        rows: [],
        swarm: oneWorkerSwarm,
      }),
    )
    expect(singularHtml).toContain('1 Immortals worker running')
    expect(singularHtml).toContain('Open Immortals →')

    // Waiting-only active swarm -> active count CTA
    const waitingSwarm: SwarmStatusSnapshot = {
      ...agySwarmStatus,
      members: [agyMembers[0]!, agyMembers[3]!],
    }
    const waitingHtml = renderToStaticMarkup(
      createElement(SubagentsSection, {
        rows: [],
        swarm: waitingSwarm,
      }),
    )
    expect(waitingHtml).toContain('1 Immortals worker active')
    expect(waitingHtml).toContain('Open Immortals →')

    // No active swarm -> normal empty state preserved
    const emptyHtml = renderToStaticMarkup(
      createElement(SubagentsSection, {
        rows: [],
        swarm: null,
      }),
    )
    expect(emptyHtml).toContain('No subagents in this task yet.')
  })

  test('7. conversation://<uuid> links route to Immortals Inspector or subagent, and /swarm board hints render as Open Immortals →', () => {
    // Link target parsing
    expect(linkTargetOf('conversation://2c8b0f21-8b42-4c7a-9e01-123456789abc')).toEqual({
      kind: 'conversation',
      conversationId: '2c8b0f21-8b42-4c7a-9e01-123456789abc',
    })
    expect(linkTargetOf('conversation://')).toEqual({ kind: 'inert' })
    expect(linkTargetOf('conversation://../../etc/passwd')).toEqual({ kind: 'inert' })

    // Resolution against swarm members first, then subagents, then swarm board fallback
    const snapWithFleet = sampleSnapshot({
      fleet: [
        {
          id: 'sub-9999-aaaa',
          key: 'sub-9999-aaaa',
          title: 'Security Auditor',
          status: 'working',
          age: '1m',
          current: false,
          kind: 'subagent',
          turns: 1,
          messages: 2,
          cwd: '/workspace/prismcore',
          untitled: false,
        },
      ],
    })
    expect(resolveConversationTarget('2c8b0f21-8b42-4c7a-9e01-123456789abc', snapWithFleet)).toEqual({
      kind: 'swarm_member',
      memberName: 'tx-config',
    })
    expect(resolveConversationTarget('sub-9999-aaaa', snapWithFleet)).toEqual({
      kind: 'subagent',
      agentId: 'sub-9999-aaaa',
    })
    expect(resolveConversationTarget('unknown-conv-id', snapWithFleet)).toEqual({
      kind: 'swarm_board',
    })

    // Clicking ConversationChatLink opens matching swarm member inspector
    const inspected: string[] = []
    const navigated: Array<{ panel: string; path?: string }> = []
    const memberLink = ConversationChatLink({
      conversationId: '2c8b0f21-8b42-4c7a-9e01-123456789abc',
      label: 'tx-config worker',
      navigate: (panel, path) => navigated.push({ panel, ...(path ? { path } : {}) }),
      getSnapshot: () => snapWithFleet,
      onOpenSwarmMember: name => inspected.push(name),
    })
    memberLink.props.onClick({
      preventDefault: () => {},
      stopPropagation: () => {},
    } as unknown as React.MouseEvent)
    expect(inspected).toEqual(['tx-config'])
    expect(navigated).toEqual([{ panel: 'swarm' }])

    // Clicking ConversationChatLink for a subagent navigates to activity/agentId
    navigated.length = 0
    const subLink = ConversationChatLink({
      conversationId: 'sub-9999-aaaa',
      label: 'Security Auditor',
      navigate: (panel, path) => navigated.push({ panel, ...(path ? { path } : {}) }),
      getSnapshot: () => snapWithFleet,
    })
    subLink.props.onKeyDown({
      key: 'Enter',
      preventDefault: () => {},
      stopPropagation: () => {},
    } as unknown as React.KeyboardEvent)
    expect(navigated).toEqual([{ panel: 'activity', path: 'sub-9999-aaaa' }])

    // Markdown renders conversation:// links as clickable chips (never inert)
    const convMdHtml = renderToStaticMarkup(
      createElement(Markdown, {
        text: 'See [tx-config](conversation://2c8b0f21-8b42-4c7a-9e01-123456789abc) for details.',
      }),
    )
    expect(convMdHtml).toContain('md__link--conversation')
    expect(convMdHtml).toContain('tx-config')
    expect(convMdHtml).not.toContain('md__link--inert')

    // Markdown turns Antigravity's open_board sentence & inline `/swarm board` into clickable `Open Immortals →`
    expect(isSwarmBoardCommand('/swarm board')).toBe(true)
    expect(isSwarmBoardCommand('/swarm ui')).toBe(true)
    expect(isSwarmBoardCommand('/immortals board')).toBe(true)
    expect(isSwarmBoardCommand('/swarm fix tests')).toBe(false)

    const hintMdHtml = renderToStaticMarkup(
      createElement(Markdown, {
        text: 'Workers spawned. Run /swarm board (default shortcut alt+s) to open the live board.',
      }),
    )
    expect(hintMdHtml).toContain('md__link--swarm')
    expect(hintMdHtml).toContain('>Open Immortals →<')
    expect(hintMdHtml).not.toContain('Open →</span>')

    const backtickHintMdHtml = renderToStaticMarkup(
      createElement(Markdown, {
        text: 'Check progress with `/swarm board` or Run `/swarm board` (default shortcut `alt+s`) to open the live board.',
      }),
    )
    expect(backtickHintMdHtml).toContain('md__link--swarm')
    expect(backtickHintMdHtml).toContain('>Open Immortals →<')

    // Fenced code blocks are never rewritten
    const fencedMdHtml = renderToStaticMarkup(
      createElement(Markdown, {
        text: '```bash\n/swarm board\n```',
      }),
    )
    expect(fencedMdHtml).not.toContain('md__link--swarm')
    expect(fencedMdHtml).toContain('/swarm board')

    // Default SwarmChatLink still preserves existing label + Open → CTA
    const defaultSwarmLinkHtml = renderToStaticMarkup(
      createElement(SwarmChatLink, {
        label: 'Open the Immortals UI',
        navigate: () => {},
      }),
    )
    expect(defaultSwarmLinkHtml).toContain('Open the Immortals UI')
    expect(defaultSwarmLinkHtml).toContain('Open →')

    // SwarmBoard honors `snap.swarmInspectTarget` to open Worker Inspector immediately
    const inspectSnap = sampleSnapshot({ swarmInspectTarget: 'tx-config' })
    const inspectBoardHtml = renderToStaticMarkup(
      createElement(SwarmBoard, { snap: inspectSnap }),
    )
    expect(inspectBoardHtml).toContain('data-inspecting="true"')
    expect(inspectBoardHtml).toContain('@tx-config')
  })

  test('8. ArtifactCard & artifactFromToolResultPayload recognize Antigravity write_file/edit_file artifacts and return_value', () => {
    // 8a. write_file with ArtifactMetadata
    const metaWrite: ToolItem = {
      id: 'wf-1',
      verb: 'write_file',
      arg: '',
      dur: '0.2s',
      state: 'done',
      name: 'write_file',
      input: JSON.stringify({
        Path: '/Users/erfan/.gemini/antigravity-cli/brain/eaa65661-2d98-4bac-9995-0588436d16d8/implementation_plan.md',
        Content: '# Parity Plan\n\nAll items mapped.',
        Overwrite: true,
        ArtifactMetadata: {
          Summary: 'Full parity implementation plan',
          UserFacing: true,
          RequestFeedback: true,
        },
      }),
      output: 'Successfully wrote file.',
    }
    const parsedMeta = artifactOfTool(metaWrite)
    expect(parsedMeta).toEqual({
      id: 'agy:implementation_plan.md',
      title: 'Parity Plan',
      version: 1,
      action: 'created',
      kind: 'markdown',
    })

    // Renders as ArtifactCard inside ToolCallRow with zero hardcoded hex colors
    const cardRowHtml = renderToStaticMarkup(
      createElement(ToolCallRow, { item: metaWrite, label: 'Write file' }),
    )
    expect(cardRowHtml).toContain('artifact-card')
    expect(cardRowHtml).toContain('>Parity Plan<')
    expect(cardRowHtml).not.toContain('execution-row')
    expect(cardRowHtml).not.toMatch(HEX_COLOR_RE)

    // 8b. edit_file targeting brain-dir HTML artifact (even without ArtifactMetadata)
    const brainEdit: ToolItem = {
      id: 'ef-1',
      verb: 'edit_file',
      arg: '',
      dur: '0.1s',
      state: 'done',
      name: 'default_api:edit_file',
      input: JSON.stringify({
        Path: '/Users/erfan/.gemini/antigravity/brain/eaa65661-2d98-4bac-9995-0588436d16d8/dashboard_preview.html',
        Target: '<h1>Old</h1>',
        Replacement: '<h1>New</h1>',
      }),
      output: '',
    }
    expect(artifactOfTool(brainEdit)).toEqual({
      id: 'agy:dashboard_preview.html',
      title: 'Dashboard Preview',
      version: 1,
      action: 'updated',
      kind: 'html',
    })

    // 8c. Ordinary workspace write_file stays a normal FileToolRow (null artifact)
    const normalWrite: ToolItem = {
      id: 'wf-2',
      verb: 'write_file',
      arg: '',
      dur: '0.1s',
      state: 'done',
      name: 'write_file',
      input: JSON.stringify({
        Path: '/workspace/prismcore/src/index.ts',
        Content: 'export const x = 1\n',
      }),
      output: 'ok',
    }
    expect(artifactOfTool(normalWrite)).toBeNull()

    // 8d. Daemon-emitted artifact_id & payload.return_value on tool_result (ArtifactToast helper)
    expect(
      artifactFromToolResultPayload({
        name: 'write_file',
        artifact_id: 'agy:audit_report.md',
        return_value: 'Wrote file',
        permitted: true,
      }),
    ).toEqual({
      id: 'agy:audit_report.md',
      title: 'Audit Report',
      version: 1,
      action: 'created',
      kind: 'markdown',
    })

    // Native ArtifactTool via return_value (not just result)
    expect(
      artifactFromToolResultPayload({
        name: 'ArtifactTool',
        return_value: 'Created artifact "Architecture Spec" (id art-77, version 2).',
        permitted: true,
      }),
    ).toEqual({
      id: 'art-77',
      title: 'Architecture Spec',
      version: 2,
      action: 'created',
      kind: 'markdown',
    })

    // Fallback input from prior tool_call when ACP tool_result omits arguments
    expect(
      artifactFromToolResultPayload(
        {
          name: 'write_file',
          return_value: '',
          permitted: true,
        },
        JSON.stringify({
          Path: '/tmp/custom/walkthrough.md',
          Content: '# Final Walkthrough\nDone.',
          ArtifactMetadata: { Summary: 'Done', UserFacing: true, RequestFeedback: false },
        }),
      ),
    ).toEqual({
      id: 'agy:walkthrough.md',
      title: 'Final Walkthrough',
      version: 1,
      action: 'created',
      kind: 'markdown',
    })

    // Denied or errored tool results never emit a toast
    expect(
      artifactFromToolResultPayload({
        name: 'write_file',
        artifact_id: 'agy:audit_report.md',
        permitted: false,
      }),
    ).toBeNull()
    expect(
      artifactFromToolResultPayload({
        name: 'write_file',
        artifact_id: 'agy:audit_report.md',
        error: 'Permission denied',
      }),
    ).toBeNull()

    // Direct ArtifactCard sanity with agy: ID
    const agyCardHtml = renderToStaticMarkup(
      createElement(ArtifactCard, { artifact: parsedMeta! }),
    )
    expect(agyCardHtml).toContain('Parity Plan')
    expect(agyCardHtml).not.toMatch(HEX_COLOR_RE)
  })

  test('9. invoke_subagent, manage_subagents, and run_workflow are recognized and parsed in spawnMembersOf & Store', async () => {
    expect(isAgentFamilyTool('invoke_subagent')).toBe(true)
    expect(isAgentFamilyTool('default_api:invoke_subagent')).toBe(true)
    expect(isAgentFamilyTool('manage_subagents')).toBe(true)
    expect(isAgentFamilyTool('run_workflow')).toBe(true)

    expect(isSpawnTool('invoke_subagent')).toBe(true)
    expect(isSpawnTool('run_workflow')).toBe(true)
    expect(isSpawnTool('manage_subagents')).toBe(false)

    expect(
      spawnMembersOf(
        'invoke_subagent',
        {
          TypeName: 'verifier',
          Role: 'Parity Verifier',
          Prompt: 'Verify all Track B requirements',
          Model: 'gemini-4-neon-xhigh',
        },
        'call-sub-1',
      ),
    ).toEqual([
      {
        key: 'call-sub-1:0',
        title: 'Parity Verifier',
        status: 'working',
        baseAgent: 'verifier',
        prompt: 'Verify all Track B requirements',
        model: 'gemini-4-neon-xhigh',
        providerProfile: '',
        reasoningEffort: '',
      },
    ])

    expect(
      spawnMembersOf(
        'invoke_subagent',
        {
          subagent_name: 'code-auditor',
          prompt: 'Inspect store.ts slash routing',
        },
        'call-sub-2',
      ),
    ).toEqual([
      {
        key: 'call-sub-2:0',
        title: 'code-auditor',
        status: 'working',
        baseAgent: 'code-auditor',
        prompt: 'Inspect store.ts slash routing',
        model: '',
        providerProfile: '',
        reasoningEffort: '',
      },
    ])

    expect(
      spawnMembersOf(
        'run_workflow',
        {
          Description: 'Fan out static checks across packages',
          Script: 'parallel([])',
        },
        'call-wf-1',
      ),
    ).toEqual([
      {
        key: 'call-wf-1:0',
        title: 'Fan out static checks across packages',
        status: 'working',
        baseAgent: '',
        prompt: '',
        model: '',
        providerProfile: '',
        reasoningEffort: '',
      },
    ])

    // Live Store opens an agents block on `invoke_subagent` tool_call and resolves receipt via conversationId
    const bridge = createMockBridge()
    ;(globalThis as Record<string, unknown>).window = { xerxes: bridge.xerxes }
    const s = new Store()
    s.start(bridge.xerxes)
    for (let i = 0; i < 40 && s.getSnapshot().connection !== 'online'; i += 1) {
      await Bun.sleep(5)
    }

    bridge.emit('turn_begin', { session_id: 'sess-agy-1', text: 'Audit schemas' })
    bridge.emit('tool_call', {
      session_id: 'sess-agy-1',
      id: 'tc-sub-1',
      tool_call_id: 'tc-sub-1',
      name: 'invoke_subagent',
      arguments: JSON.stringify({
        TypeName: 'researcher',
        Role: 'Schema Auditor',
        Prompt: 'Check agy brain schemas',
      }),
    })
    await Bun.sleep(65)
    const agentsBlock = s.getSnapshot().blocks.find(b => b.kind === 'agents')
    expect(agentsBlock?.kind).toBe('agents')
    if (agentsBlock?.kind === 'agents') {
      expect(agentsBlock.members[0]?.title).toBe('Schema Auditor')
      expect(agentsBlock.members[0]?.baseAgent).toBe('researcher')
      expect(agentsBlock.members[0]?.status).toBe('working')
    }

    bridge.emit('tool_result', {
      session_id: 'sess-agy-1',
      tool_call_id: 'tc-sub-1',
      name: 'invoke_subagent',
      return_value: JSON.stringify({
        conversationId: 'sub-conv-777',
        status: 'completed',
        title: 'Schema Auditor',
        summary: 'All schemas verified',
      }),
      permitted: true,
    })
    bridge.emit('turn_end', { session_id: 'sess-agy-1' })
    expect(s.getSnapshot().fleet.some(row => row.id === 'sub-conv-777')).toBe(true)

    // Singleton store openSwarmBoard(memberName) sets swarmInspectTarget cleanly
    singletonStore.openSwarmBoard('tx-config')
    expect(singletonStore.getSnapshot().swarmOpen).toBe(true)
    expect(singletonStore.getSnapshot().swarmInspectTarget).toBe('tx-config')
    singletonStore.closeSwarmBoard()
    expect(singletonStore.getSnapshot().swarmOpen).toBe(false)
    expect(singletonStore.getSnapshot().swarmInspectTarget).toBe('')
  })
})
