// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import {
  createElement,
  isValidElement,
  type FormEvent,
  type KeyboardEvent,
  type MouseEvent,
  type ReactElement,
  type ReactNode,
} from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { ActivityDetails, Shell, VibesSideNav } from '../src/desktop/renderer/App.js'
import {
  BlockBuilder,
  blocksFromStoredMessages,
  formatInternalWakeNotice,
  isInternalSwarmWakeText,
} from '../src/desktop/renderer/blocks.js'
import { SwarmChatLink, linkTargetOf } from '../src/desktop/renderer/chatLinks.js'
import { DesktopRail, DesktopSheet } from '../src/desktop/renderer/DesktopPanels.js'
import { Markdown } from '../src/desktop/renderer/markdown.js'
import { SectionsMenuPanel } from '../src/desktop/renderer/SectionsMenu.js'
import {
  SideQuestionPanel,
  applyBtwEventPayload,
  parseSideQuestionSlashCommand,
} from '../src/desktop/renderer/SideQuestion.js'
import { Store, type Snapshot, type XerxesLike } from '../src/desktop/renderer/store.js'
import {
  SwarmBoard,
  SwarmBoardView,
  SwarmJumpLatestButton,
  SwarmModelPicker,
  isNearScrollBottom,
  nextUnseenCount,
  resolveSwarmModelChoices,
  type SwarmBoardViewProps,
} from '../src/desktop/renderer/SwarmBoard.js'
import {
  applySwarmMentionCandidate,
  filterSwarmMembers,
  formatSwarmShortTime,
  isSwarmSidecarHref,
  mergeSwarmMessages,
  parseSwarmMemberInspect,
  parseSwarmMessageBlocks,
  parseSwarmSlashCommand,
  parseSwarmStatus,
  previewSwarmMessage,
  sortSwarmMembers,
  swarmChannelsOf,
  swarmMemberCounts,
  swarmMentionCandidates,
  tokenizeSwarmMessage,
  type SwarmBoardMessage,
  type SwarmMemberInspectResult,
  type SwarmMemberSnapshot,
  type SwarmStatusSnapshot,
} from '../src/desktop/renderer/swarmState.js'
import type { DaemonEvent } from '../src/desktop/renderer/types.js'
import { VscodeChat } from '../src/vscode/ui/VscodeApp.js'

const sampleStatus: SwarmStatusSnapshot = {
  active: true,
  session_id: 'sess-swarm-1',
  title: 'Auth & Billing Audit',
  board: 'local',
  open_board: '[Open the Swarm UI](sidecar://swarm/swarm_board/?auto_open=false)',
  head: 5,
  overview: {
    markdown: '### Current Progress\n- **Auth** verified in #3\n- Waiting on `@billing-worker`',
    summary: 'Auth verified; billing in progress.',
    updated_at: '07:15:00',
    author: 'narrator',
  },
  channels: [
    { name: 'main', topic: 'Shared swarm board', created_by: 'coordinator', archived: false, subscriber_count: 5 },
    { name: 'design', topic: 'API contracts', created_by: 'coordinator', archived: false, subscriber_count: 3 },
    { name: 'legacy', topic: 'Old notes', created_by: 'coordinator', archived: true, subscriber_count: 1 },
  ],
  members: [
    {
      name: 'dead-worker',
      role: 'worker',
      added_by: 'coordinator',
      model: 'gpt-5-mini',
      model_display_name: 'GPT-5 Mini',
      workspace: 'branch',
      workspace_path: '/repo/.xerxes/swarm/dead-worker',
      channels: ['main'],
      state: 'killed',
      killed: true,
      last_read_seq: 2,
      last_post_seq: 2,
    },
    {
      name: 'api-helper',
      role: 'worker',
      added_by: 'worker-api',
      model: 'claude-sonnet-4-6',
      model_display_name: 'Claude Sonnet 4.6',
      workspace: 'share',
      workspace_path: '/repo/.xerxes/worktrees/api-helper',
      channels: ['main', 'design'],
      state: 'waiting',
      killed: false,
      mentions_only: true,
      last_read_seq: 4,
      last_post_seq: 3,
    },
    {
      name: 'coordinator',
      role: 'coordinator',
      model: 'claude-opus-4-6',
      model_display_name: 'Claude Opus 4.6',
      workspace: 'inherit',
      channels: ['*'],
      state: 'waiting',
      killed: false,
      last_read_seq: 5,
      last_post_seq: 1,
    },
    {
      name: 'worker-api',
      group: 'worker',
      role: 'worker',
      added_by: 'coordinator',
      model: 'claude-sonnet-4-6',
      model_display_name: 'Claude Sonnet 4.6',
      workspace: 'share',
      workspace_path: '/repo/.xerxes/worktrees/worker-api',
      channels: ['main', 'design'],
      state: 'working',
      killed: false,
      last_read_seq: 5,
      last_post_seq: 4,
      current_activity: 'Reading src/auth/jwt.ts',
    },
    {
      name: 'narrator',
      role: 'narrator',
      added_by: 'coordinator',
      model: 'claude-haiku-4-5',
      model_display_name: 'Claude Haiku 4.5',
      workspace: 'inherit',
      channels: ['*'],
      state: 'stopped',
      killed: false,
      last_read_seq: 5,
      error: 'Reloaded from disk — wakes on mention',
    },
    {
      name: 'monitor',
      role: 'monitor',
      added_by: 'coordinator',
      model: 'claude-haiku-4-5',
      model_display_name: 'Claude Haiku 4.5',
      workspace: 'inherit',
      channels: ['*'],
      state: 'waiting',
      killed: false,
      last_read_seq: 5,
    },
  ],
}

const sampleMessages: readonly SwarmBoardMessage[] = [
  {
    seq: 1,
    agent: 'coordinator',
    channel: 'main',
    text: '@all Kickoff: @worker-api owns auth routes, coordinate contracts in @#design.',
    mentions: ['worker-api'],
    channel_mentions: ['design'],
    broadcast: true,
    time: '07:10:01',
  },
  {
    seq: 2,
    agent: 'worker-api',
    channel: 'design',
    text: '@api-helper Please verify token claims from #1.',
    mentions: ['api-helper'],
    channel_mentions: [],
    broadcast: false,
    time: '07:11:14',
  },
  {
    seq: 3,
    agent: 'api-helper',
    channel: 'design',
    text: '@worker-api [done] Verified JWT expiry and audience checks (#2).',
    mentions: ['worker-api'],
    channel_mentions: [],
    broadcast: false,
    time: '07:12:30',
  },
  {
    seq: 4,
    agent: 'monitor',
    channel: 'main',
    text: '@coordinator [stall] All workers parked after #3 without final handoff.',
    mentions: ['coordinator'],
    channel_mentions: [],
    broadcast: false,
    time: '07:14:00',
  },
]

const sampleInspect: SwarmMemberInspectResult = {
  session_id: 'sess-swarm-1',
  member: sampleStatus.members.find(m => m.name === 'worker-api')!,
  conversation_id: 'sess-swarm-1:swarm:worker-api',
  activity: {
    phase: 'tool',
    summary: 'Reading src/auth/jwt.ts',
    active_tools: [
      {
        id: 'tc-1',
        name: 'ReadFile',
        summary: 'src/auth/jwt.ts',
        status: 'running',
        started_at: '07:14:10',
      },
    ],
    recent_tools: [
      {
        id: 'tc-0',
        name: 'GrepTool',
        summary: 'verifyToken',
        status: 'done',
        started_at: '07:13:55',
        finished_at: '07:13:56',
      },
    ],
    live_text_preview: 'Checking JWT issuer and audience validation paths…',
  },
  messages: [
    { index: 0, role: 'user', text: 'Own auth routes and verify JWT claims.' },
    {
      index: 1,
      role: 'assistant',
      text: 'Inspecting jwt.ts now.',
      tool_calls: [{ id: 'tc-1', name: 'ReadFile', args_preview: '{"file_path":"src/auth/jwt.ts"}' }],
    },
  ],
  total_messages: 2,
}

const baseSnapshot = (overrides: Partial<Snapshot> = {}): Snapshot => ({
  connection: 'online',
  noWorkspace: false,
  cwd: '/repo',
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
  models: [
    { id: 'claude-opus-4-6', provider: 'anthropic', label: 'Claude Opus 4.6' },
    { id: 'claude-sonnet-4-6', provider: 'anthropic', label: 'Claude Sonnet 4.6' },
  ],
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
  currentId: 'sess-swarm-1',
  sessionOpenRevision: 0,
  currentTitle: 'Audit auth and billing',
  sessionKey: 'desktop-swarm-1',
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
  turnCount: 1,
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
  swarm: null,
  swarmMessages: [],
  swarmLoading: false,
  swarmError: null,
  swarmOpen: false,
  swarmOpenRevision: 0,
  ...overrides,
})

interface ElementProps {
  readonly children?: ReactNode
  readonly className?: string
  readonly 'aria-label'?: string
  readonly disabled?: boolean
  readonly value?: string
  readonly onClick?: (event?: MouseEvent) => void
  readonly onChange?: (event: { target: { value: string } }) => void
  readonly onSubmit?: (event: FormEvent) => void
}

function walkElements(node: ReactNode): ReactElement<ElementProps>[] {
  if (Array.isArray(node)) return node.flatMap(walkElements)
  if (!isValidElement<ElementProps>(node)) return []
  return [node, ...walkElements(node.props.children)]
}

class FakeSwarmBridge implements XerxesLike {
  calls: Array<{ method: string; params: Record<string, unknown> }> = []
  handlers = new Set<(event: DaemonEvent) => void>()
  status: SwarmStatusSnapshot = sampleStatus
  messages: SwarmBoardMessage[] = [...sampleMessages]
  inspectPayload: SwarmMemberInspectResult = sampleInspect

  call(method: string, params: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    this.calls.push({ method, params })
    if (method === 'initialize') {
      return Promise.resolve({
        ok: true,
        session_id: 'sess-swarm-1',
        cwd: '/repo',
        session: { id: 'sess-swarm-1', key: 'desktop-swarm-1', cwd: '/repo' },
      })
    }
    if (method === 'complete') {
      const text = String(params.text ?? '')
      if (text === '/immortals') {
        return Promise.resolve({
          completions: [
            {
              value: '/immortals',
              label: 'immortals',
              meta: 'Coordinate a parallel multi-agent Immortals force on a shared message board or open the Immortals Board',
              category: 'activity',
            },
            {
              value: '/immortals-coordinator ',
              label: 'immortals-coordinator',
              meta: 'Coordinator playbook for parallel multi-agent Immortals',
            },
            {
              value: '/immortals-monitor ',
              label: 'immortals-monitor',
              meta: 'Monitor worker playbook',
            },
            {
              value: '/immortals-narrator ',
              label: 'immortals-narrator',
              meta: 'Narrator worker playbook',
            },
          ],
        })
      }
      if (text === '/skill immortals') {
        return Promise.resolve({
          completions: [
            {
              value: '/skill immortals-coordinator ',
              label: 'immortals-coordinator',
              meta: 'Coordinator playbook for parallel multi-agent Immortals',
            },
            {
              value: '/skill immortals-monitor ',
              label: 'immortals-monitor',
              meta: 'Monitor worker playbook',
            },
            {
              value: '/skill immortals-narrator ',
              label: 'immortals-narrator',
              meta: 'Narrator worker playbook',
            },
          ],
        })
      }
      if (text === '/swarm') {
        return Promise.resolve({
          completions: [
            {
              value: '/swarm',
              label: 'swarm',
              meta: 'Coordinate a parallel multi-agent Immortals force on a shared message board or open the Immortals Board',
              category: 'activity',
            },
            {
              value: '/swarm-coordinator ',
              label: 'swarm-coordinator',
              meta: 'Coordinator playbook for parallel multi-agent swarms',
            },
            {
              value: '/swarm-monitor ',
              label: 'swarm-monitor',
              meta: 'Monitor worker playbook',
            },
            {
              value: '/swarm-narrator ',
              label: 'swarm-narrator',
              meta: 'Narrator worker playbook',
            },
          ],
        })
      }
      if (text === '/skill swarm') {
        return Promise.resolve({
          completions: [
            {
              value: '/skill swarm-coordinator ',
              label: 'swarm-coordinator',
              meta: 'Coordinator playbook for parallel multi-agent swarms',
            },
            {
              value: '/skill swarm-monitor ',
              label: 'swarm-monitor',
              meta: 'Monitor worker playbook',
            },
            {
              value: '/skill swarm-narrator ',
              label: 'swarm-narrator',
              meta: 'Narrator worker playbook',
            },
          ],
        })
      }
      return Promise.resolve({ completions: [] })
    }
    if (method === 'swarm.status') {
      return Promise.resolve(this.status as unknown as Record<string, unknown>)
    }
    if (method === 'swarm.inspect') {
      return Promise.resolve(this.inspectPayload as unknown as Record<string, unknown>)
    }
    if (method === 'swarm.board') {
      const ch = typeof params.channel === 'string' ? params.channel : ''
      const filtered = ch && ch !== '*' ? this.messages.filter(msg => msg.channel === ch) : this.messages
      return Promise.resolve({ head: this.status.head, messages: filtered })
    }
    if (method === 'swarm.post') {
      const seq = this.status.head + 1
      const channel = typeof params.channel === 'string' ? params.channel : 'main'
      const rawText = typeof params.text === 'string' ? params.text : ''
      const text = params.as_feedback === true ? `@narrator user feedback: ${rawText.replace(/@/g, '')}` : rawText
      const msg: SwarmBoardMessage = {
        seq,
        agent: 'user',
        channel,
        text,
        mentions: params.as_feedback === true ? ['narrator'] : [],
        channel_mentions: [],
        broadcast: false,
        time: '07:20:00',
      }
      this.status = { ...this.status, head: seq }
      this.messages = [...this.messages, msg]
      return Promise.resolve({ status: 'posted', seq, channel, message: msg })
    }
    if (method === 'swarm.kill') {
      const target = String(params.name ?? '')
      this.status = {
        ...this.status,
        members: this.status.members.map(member =>
          (target === 'all' && member.role !== 'coordinator') || member.name === target
            ? { ...member, state: 'killed', killed: true }
            : member,
        ),
      }
      return Promise.resolve({ status: 'killed', killed: [target] })
    }
    if (method === 'swarm.model') {
      const target = String(params.name ?? '')
      const model = typeof params.model === 'string' && params.model ? params.model : 'claude-opus-4-6'
      this.status = {
        ...this.status,
        members: this.status.members.map(member =>
          member.name === target ? { ...member, model, model_display_name: model } : member,
        ),
      }
      return Promise.resolve({ status: 'switched', name: target, model })
    }
    if (method === 'swarm.activate') {
      return Promise.resolve({ ok: true, active: true, prompt_overlay: '<swarm_coordinator>' })
    }
    if (method === 'session.side_question') {
      const requestId = String(params.request_id ?? 'req-side-1')
      const question = String(params.question ?? '')
      return Promise.resolve({
        ok: true,
        request_id: requestId,
        session_id: 'sess-swarm-1',
        question,
        answer: `Side answer for: ${question}`,
        model: 'claude-opus-4-6',
      })
    }
    if (method === 'turn.submit') {
      return Promise.resolve({ ok: true })
    }
    if (method === 'learning.list') {
      return Promise.resolve({
        ok: true,
        workspace: '/repo',
        proposals: [],
        lessons: [],
        counts: { pending_proposals: 0, pending_lessons: 0 },
      })
    }
    return Promise.resolve({ ok: true, sessions: [] })
  }

  emit(type: string, payload: Record<string, unknown>): void {
    for (const handler of this.handlers) handler({ type, payload })
  }
}

const originalWindow = globalThis.window
afterEach(() => {
  if (originalWindow === undefined) delete (globalThis as { window?: unknown }).window
  else (globalThis as { window: unknown }).window = originalWindow
})

async function setupConnectedStore(): Promise<{ store: Store; bridge: FakeSwarmBridge }> {
  const bridge = new FakeSwarmBridge()
  ;(globalThis as { window: unknown }).window = {
    xerxes: {
      call: (method: string, params?: Record<string, unknown>) => bridge.call(method, params),
      onEvent: (handler: (event: DaemonEvent) => void) => {
        bridge.handlers.add(handler)
        return () => bridge.handlers.delete(handler)
      },
    },
  }
  const instance = new Store()
  instance.start(bridge)
  await Bun.sleep(5)
  return { store: instance, bridge }
}

describe('SwarmBoard rendering: calm segmented Vibes layout, channels, roster states, live activity, @ autocomplete & tokens', () => {
  test('renders segmented views, narrator .artdoc overview, custom channel tabs, hierarchical flat roster, live activity, and @ mention popover without static Wake clutter', () => {
    const html = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: sampleStatus,
        messages: sampleMessages,
        models: baseSnapshot().models,
        initialPostDraft: 'Hey @work',
      }),
    )

    // Header: title, head seq, active/waiting/stopped/killed counts, segmented Board/Members/Overview tabs, Refresh & Stop controls
    expect(html).toContain('Auth &amp; Billing Audit')
    expect(html).toContain('#5')
    expect(html).toContain('<strong>1</strong> active')
    expect(html).toContain('<strong>3</strong> waiting')
    expect(html).toContain('<strong>1</strong> stopped')
    expect(html).toContain('<strong>1</strong> killed')
    expect(html).toContain('aria-label="Immortals views"')
    expect(html).toContain('aria-label="Refresh Immortals Board"')
    expect(html).toContain('aria-label="Stop all Immortals workers"')

    // Zero static "Wake:" chip wall clutter; replaced by live @ mention popover
    expect(html).not.toContain('Wake:')
    expect(html).toContain('aria-label="Mention an Immortals member"')
    expect(html).toContain('>@worker-api</span>')

    // Narrator Overview renders .artdoc Markdown and subtle narrator feedback input
    expect(html).toContain('aria-label="Narrator overview"')
    expect(html).toContain('class="artdoc swarm-overview-doc"')
    expect(html).toContain('@narrator')
    expect(html).toContain('<h3>Current Progress</h3>')
    expect(html).toContain('<strong>Auth</strong> verified')
    expect(html).toContain('aria-label="Narrator feedback message"')

    // Custom Channel Tabs shown because #design and #legacy exist alongside #main
    expect(html).toContain('aria-label="Immortals channels"')
    expect(html).toContain('>All</span>')
    expect(html).toContain('#main')
    expect(html).toContain('#design')
    expect(html).toContain('#legacy')
    expect(html).toContain('data-archived="true"')
    expect(html).toContain('archived')
    expect(html).toContain('title="3 subscribers"')

    // Roster hierarchy & badges: Coordinator first, worker-api before its helper api-helper, killed dead-worker last
    const coordAt = html.indexOf('>@coordinator</button>')
    const workerApiAt = html.indexOf('>@worker-api</button>')
    const helperAt = html.indexOf('>@api-helper</button>')
    const deadAt = html.indexOf('>@dead-worker</button>')
    expect(coordAt).toBeGreaterThan(-1)
    expect(coordAt).toBeLessThan(workerApiAt)
    expect(workerApiAt).toBeLessThan(helperAt)
    expect(helperAt).toBeLessThan(deadAt)

    expect(html).toContain('>Coordinator</span>')
    expect(html).toContain('Reading src/auth/jwt.ts')
    expect(html).toContain('↳ helper of @worker-api')
    expect(html).toContain('>worktree</span>')
    expect(html).toContain('>clone</span>')
    expect(html).toContain('>inherit</span>')
    expect(html).toContain('read #5 · posted #4')
    expect(html).toContain('Reloaded from disk — wakes on mention')

    // Kill/Stop button only for living non-coordinator workers; Switch Model only for waiting/stopped non-killed members
    expect(html).toContain('aria-label="Kill worker-api"')
    expect(html).toContain('aria-label="Kill api-helper"')
    expect(html).toContain('aria-label="Kill narrator"')
    expect(html).not.toContain('aria-label="Kill coordinator"')
    expect(html).not.toContain('aria-label="Kill dead-worker"')

    expect(html).toContain('aria-label="Switch model for coordinator"')
    expect(html).toContain('aria-label="Switch model for api-helper"')
    expect(html).toContain('aria-label="Switch model for narrator"')
    expect(html).not.toContain('aria-label="Switch model for worker-api"')
    expect(html).not.toContain('aria-label="Switch model for dead-worker"')

    // Message Board Feed: #seq timeline rows, @mentions, @#channel, @all, [done], [stall], and clickable #seq jump refs
    expect(html).toContain('id="swarm-msg-1"')
    expect(html).toContain('id="swarm-msg-3"')
    expect(html).toContain('swarm-token--broadcast">@all</span>')
    expect(html).toContain('swarm-token--mention" title="Mention @worker-api in composer">@worker-api</button>')
    expect(html).toContain('swarm-token--channel" title="Filter channel #design">@#design</button>')
    expect(html).toContain('swarm-badge--done">[done]</span>')
    expect(html).toContain('swarm-badge--stall">[stall]</span>')
    expect(html).toContain('aria-label="Jump to message #1"')
    expect(html).toContain('aria-label="Jump to message #2"')
    expect(html).toContain('>Graph</span>')
    expect(html).toContain('swarm-board__live')
    expect(html).toContain('swarm-member__sprite')

    // Design rule: zero hardcoded 6-digit hex colors in rendered UI chrome (excluding pixel-art Simurgh SVGs)
    const chromeHtml = html.replace(/<svg[^>]*swarm-member__sprite[\s\S]*?<\/svg>/g, '')
    expect(chromeHtml.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])
  })

  test('renders Swarm Graph tab in Battlefield (with legacy hive alias) and Mentions modes and wires member inspection from Graph', () => {
    const battlefieldHtml = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: sampleStatus,
        messages: sampleMessages,
        initialTab: 'graph',
        initialGraphMode: 'battlefield',
      }),
    )
    expect(battlefieldHtml).toContain('data-tab="graph"')
    expect(battlefieldHtml).toContain('aria-label="Immortals graph"')
    expect(battlefieldHtml).toContain('data-mode="battlefield"')
    expect(battlefieldHtml).toContain('>Map<')
    expect(battlefieldHtml).toContain('Export GIF')
    expect(battlefieldHtml).toContain('Completed')
    expect(battlefieldHtml).toContain('Dispatch sent')
    expect(battlefieldHtml).toContain('Awaiting handoff')

    // Legacy 'hive' alias seamlessly normalizes to 'battlefield'
    const legacyHiveHtml = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: sampleStatus,
        messages: sampleMessages,
        initialTab: 'graph',
        initialGraphMode: 'hive',
      }),
    )
    expect(legacyHiveHtml).toContain('data-mode="battlefield"')
    expect(legacyHiveHtml).toContain('aria-label="Immortals graph"')

    const mentionsHtml = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: sampleStatus,
        messages: sampleMessages,
        initialTab: 'graph',
        initialGraphMode: 'mentions',
      }),
    )
    expect(mentionsHtml).toContain('data-mode="mentions"')
    expect(mentionsHtml).toContain('Hide inactive')
    expect(mentionsHtml).toContain('Expand @all')
    expect(mentionsHtml).toContain('@mention, width = count')
  })

  test('hides channel strip when only #main exists, renders Live Worker Inspector, and supports @ mention candidate completion', () => {
    const mainOnlyStatus: SwarmStatusSnapshot = {
      ...sampleStatus,
      overview: null,
      channels: [{ name: 'main', topic: 'Shared swarm board', created_by: 'coordinator', archived: false, subscriber_count: 3 }],
      members: sampleStatus.members.filter(m => m.role !== 'narrator'),
    }
    const mainOnlyMessages = sampleMessages.filter(m => m.channel === 'main')
    const mainOnlyHtml = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: mainOnlyStatus,
        messages: mainOnlyMessages,
      }),
    )
    expect(mainOnlyHtml).not.toContain('aria-label="Immortals channels"')
    expect(mainOnlyHtml).toContain('No narrator overview yet')

    const archivedHtml = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: sampleStatus,
        messages: sampleMessages,
        initialChannel: 'legacy',
      }),
    )
    expect(archivedHtml).toContain('Channel #legacy is archived (read-only)')

    // Live Worker Inspector view
    const inspectHtml = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: sampleStatus,
        messages: sampleMessages,
        inspect: sampleInspect,
        initialInspectMember: 'worker-api',
      }),
    )
    expect(inspectHtml).toContain('aria-label="Worker inspector for @worker-api"')
    expect(inspectHtml).toContain('aria-label="Back to members"')
    expect(inspectHtml).toContain('swarm-member__sprite')
    expect(inspectHtml).toContain('swarm-inspector__tool-badge')
    expect(inspectHtml).toContain('data-category="read"')
    expect(inspectHtml).toContain('data-category="search"')
    expect(inspectHtml).toContain('1.0s')
    expect(inspectHtml).toContain('Checking JWT issuer and audience validation paths…')
    expect(inspectHtml).toContain('ReadFile')
    expect(inspectHtml).toContain('Own auth routes and verify JWT claims.')

    // @ mention helper functions
    const candidates = swarmMentionCandidates('Please check @wo', sampleStatus.members)
    expect(candidates.map(c => c.name)).toEqual(['worker-api'])
    expect(applySwarmMentionCandidate('Please check @wo', 'worker-api')).toBe('Please check @worker-api ')
  })

  test('tokenizeSwarmMessage keeps emails and channel headings intact while tokenizing mentions, badges, and #seq refs', () => {
    const tokens = tokenizeSwarmMessage(
      'Contact ops@example.com; @worker-1 [done] see (#12) and @#design plus @all [stall] #main',
    )
    expect(tokens).toEqual([
      { kind: 'text', raw: 'Contact ops@example.com; ' },
      { kind: 'mention', raw: '@worker-1', name: 'worker-1' },
      { kind: 'text', raw: ' ' },
      { kind: 'done', raw: '[done]' },
      { kind: 'text', raw: ' see (' },
      { kind: 'seq_ref', raw: '#12', seq: 12 },
      { kind: 'text', raw: ') and ' },
      { kind: 'channel_mention', raw: '@#design', channel: 'design' },
      { kind: 'text', raw: ' plus ' },
      { kind: 'broadcast', raw: '@all' },
      { kind: 'text', raw: ' ' },
      { kind: 'stall', raw: '[stall]' },
      { kind: 'text', raw: ' #main' },
    ])
  })
})

describe('SwarmBoardView interactive callbacks & Store Daemon v35 RPC actions', () => {
  function buildViewProps(overrides: Partial<SwarmBoardViewProps> = {}): SwarmBoardViewProps {
    const channels = swarmChannelsOf(sampleStatus, sampleMessages)
    const members = sortSwarmMembers(sampleStatus.members)
    const counts = swarmMemberCounts(sampleStatus.members)
    return {
      status: sampleStatus,
      channels,
      members,
      counts,
      messages: sampleMessages,
      visibleMessages: sampleMessages,
      models: baseSnapshot().models,
      modelListId: 'swarm-models-test',
      online: true,
      loading: false,
      busyAction: '',
      error: '',
      selectedChannel: 'design',
      highlightedSeq: null,
      feedbackOpen: true,
      feedbackDraft: 'Focus on auth token findings',
      postDraft: '@worker-api please check rate limits',
      switchingMember: 'api-helper',
      modelDrafts: { 'api-helper': 'claude-opus-4-6' },
      onRefresh: () => undefined,
      onStopAll: () => undefined,
      onToggleFeedback: () => undefined,
      onChangeFeedbackDraft: () => undefined,
      onSubmitFeedback: () => undefined,
      onCloseFeedback: () => undefined,
      onSelectChannel: () => undefined,
      onInsertMention: () => undefined,
      onInsertSeqRef: () => undefined,
      onJumpToSeq: () => undefined,
      onKillMember: () => undefined,
      onInspectMember: () => undefined,
      onCloseInspect: () => undefined,
      onToggleModelSwitch: () => undefined,
      onChangeModelDraft: () => undefined,
      onSubmitModelSwitch: () => undefined,
      onCancelModelSwitch: () => undefined,
      onChangePostDraft: () => undefined,
      onSubmitPost: () => undefined,
      ...overrides,
    }
  }

  test('invokes post, narrator-feedback (as_feedback: true), inspect, kill, stop-all, switch-model, channel select, and #seq jump actions', async () => {
    const { store, bridge } = await setupConnectedStore()
    await store.refreshSwarm()
    bridge.calls.length = 0

    let jumpedSeq = 0
    let mentioned = ''
    let pickedChannel = ''

    const view = SwarmBoardView(
      buildViewProps({
        onRefresh: () => void store.refreshSwarm(),
        onStopAll: () => void store.killSwarmMember('all'),
        onSubmitFeedback: event => {
          event?.preventDefault()
          void store.postToSwarm({ text: 'Focus on auth token findings', channel: 'main', asFeedback: true })
        },
        onSelectChannel: ch => {
          pickedChannel = ch
        },
        onInsertMention: name => {
          mentioned = name
        },
        onJumpToSeq: seq => {
          jumpedSeq = seq
        },
        onInspectMember: name => {
          void store.inspectSwarmMember(name)
        },
        onKillMember: name => void store.killSwarmMember(name),
        onSubmitModelSwitch: (member, event) => {
          event?.preventDefault()
          void store.switchSwarmMemberModel(member.name, 'claude-opus-4-6')
        },
        onSubmitPost: event => {
          event?.preventDefault()
          void store.postToSwarm({ text: '@worker-api please check rate limits', channel: 'design' })
        },
      }),
    )

    const nodes = walkElements(view)
    const prevent = () => ({ preventDefault: () => undefined }) as unknown as FormEvent

    // 1. Click #seq jump button (#1) and @mention chip (@worker-api)
    nodes.find(el => el.type === 'button' && el.props['aria-label'] === 'Jump to message #1')!.props.onClick!()
    expect(jumpedSeq).toBe(1)

    nodes.find(el => el.type === 'button' && el.props.className?.includes('swarm-token--mention'))!.props.onClick!()
    expect(mentioned).toBe('worker-api')

    nodes.find(el => el.type === 'button' && el.props.className?.includes('swarm-token--channel'))!.props.onClick!()
    expect(pickedChannel).toBe('design')

    // 2. Inspect worker-api via swarm.inspect
    nodes.find(el => el.type === 'button' && el.props['aria-label'] === 'Inspect worker-api')!.props.onClick!()
    await Bun.sleep(5)
    expect(bridge.calls.find(c => c.method === 'swarm.inspect')?.params).toMatchObject({
      name: 'worker-api',
    })
    expect(store.getSnapshot().swarmInspect?.member.name).toBe('worker-api')
    expect(parseSwarmMemberInspect({ ok: false })).toBeNull()

    // 3. Submit board post onto #design
    const composerForm = nodes.find(el => el.type === 'form' && el.props['aria-label'] === 'Post to Immortals Board')!
    composerForm.props.onSubmit!(prevent())
    await Bun.sleep(5)
    expect(bridge.calls.find(c => c.method === 'swarm.post' && !c.params.as_feedback)?.params).toMatchObject({
      session_key: 'desktop-swarm-1',
      session_id: 'sess-swarm-1',
      channel: 'design',
      text: '@worker-api please check rate limits',
    })

    // 4. Submit Narrator feedback (`as_feedback: true`)
    const feedbackForm = nodes.find(el => el.type === 'form' && el.props.className === 'swarm-board__feedback-form')!
    feedbackForm.props.onSubmit!(prevent())
    await Bun.sleep(5)
    expect(bridge.calls.find(c => c.method === 'swarm.post' && c.params.as_feedback === true)?.params).toMatchObject({
      session_key: 'desktop-swarm-1',
      session_id: 'sess-swarm-1',
      channel: 'main',
      text: 'Focus on auth token findings',
      as_feedback: true,
    })

    // 5. Sleek Switch Model popover renders for `api-helper`: single-click model pick + custom model ID fallback
    const apiHelper = sampleStatus.members.find(member => member.name === 'api-helper')!
    const pickerEl = nodes.find(el => el.type === SwarmModelPicker)!
    expect(pickerEl).toBeDefined()
    const pickerProps = pickerEl.props as unknown as {
      readonly onPickModel: (modelId: string) => void
      readonly onSubmitCustom: (event?: FormEvent) => void
    }
    pickerProps.onPickModel('claude-opus-4-6')
    await Bun.sleep(5)
    expect(bridge.calls.find(c => c.method === 'swarm.model')?.params).toMatchObject({
      session_key: 'desktop-swarm-1',
      session_id: 'sess-swarm-1',
      name: 'api-helper',
      model: 'claude-opus-4-6',
    })
    expect(
      store.getSnapshot().swarm?.members.find(member => member.name === 'api-helper')?.model,
    ).toBe('claude-opus-4-6')

    // Custom model ID fallback submits via Enter / form submit cleanly
    bridge.calls.length = 0
    const pickerCustomHtml = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: sampleStatus,
        messages: sampleMessages,
        models: [
          ...baseSnapshot().models,
          { id: 'claude-opus-4-6', provider: 'anthropic', label: 'Duplicate Opus' },
        ],
        initialTab: 'members',
        initialModelSwitchMember: 'api-helper',
        initialModelDrafts: { 'api-helper': 'custom/my-model-v2' },
      }),
    )
    expect(pickerCustomHtml).toContain('class="swarm-model-picker"')
    expect(pickerCustomHtml).toContain('aria-label="Switch model for @api-helper"')
    expect(pickerCustomHtml).toContain('role="listbox"')
    expect(pickerCustomHtml).toContain('aria-selected="true"')
    expect(pickerCustomHtml).toContain('placeholder="Custom model ID…"')
    expect(pickerCustomHtml).toContain('value="custom/my-model-v2"')
    // <= 6 models omits the filter input; > 6 models shows it
    expect(pickerCustomHtml).not.toContain('aria-label="Filter models"')
    const manyModels = Array.from({ length: 8 }, (_, idx) => ({
      id: `model-${idx + 1}`,
      provider: idx % 2 === 0 ? 'anthropic' : 'openai',
      label: `Model ${idx + 1}`,
    }))
    const manyPickerHtml = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: sampleStatus,
        messages: sampleMessages,
        models: manyModels,
        initialTab: 'members',
        initialModelSwitchMember: 'api-helper',
      }),
    )
    expect(manyPickerHtml).toContain('aria-label="Filter models"')
    // Deduplication preserves member's active model when absent from catalog
    const resolved = resolveSwarmModelChoices(manyModels, apiHelper)
    expect(resolved.activeId).toBe('claude-sonnet-4-6')
    expect(resolved.choices[0]?.id).toBe('claude-sonnet-4-6')
    // Direct custom form submit invokes switchModel with the custom draft
    let customSwitched = ''
    const customView = SwarmBoardView(
      buildViewProps({
        switchingMember: 'api-helper',
        modelDrafts: { 'api-helper': 'custom/my-model-v2' },
        onSubmitModelSwitch: (member, event) => {
          event?.preventDefault()
          customSwitched = `${member.name}:custom/my-model-v2`
        },
      }),
    )
    const customPickerEl = walkElements(customView).find(el => el.type === SwarmModelPicker)!
    ;(customPickerEl.props as unknown as { readonly onSubmitCustom: (event?: FormEvent) => void }).onSubmitCustom(prevent())
    expect(customSwitched).toBe('api-helper:custom/my-model-v2')

    // 6. Kill single living worker `worker-api`, then Stop All
    nodes.find(el => el.type === 'button' && el.props['aria-label'] === 'Kill worker-api')!.props.onClick!()
    await Bun.sleep(5)
    expect(bridge.calls.find(c => c.method === 'swarm.kill' && c.params.name === 'worker-api')).toBeDefined()
    expect(
      store.getSnapshot().swarm?.members.find(member => member.name === 'worker-api')?.state,
    ).toBe('killed')

    nodes.find(el => el.type === 'button' && el.props['aria-label'] === 'Stop all Immortals workers')!.props.onClick!()
    await Bun.sleep(5)
    expect(bridge.calls.find(c => c.method === 'swarm.kill' && c.params.name === 'all')).toBeDefined()
  })
})

describe('sidecar://immortals & sidecar://swarm Markdown link interception, /immortals completion deduplication, /btw Side Question & live swarm_event stream', () => {
  test('intercepts sidecar://immortals/board and legacy sidecar://swarm/swarm_board links in Markdown as a quiet inline bar and opens Immortals Board without external navigation', () => {
    expect(isSwarmSidecarHref('sidecar://immortals/board/?auto_open=false')).toBe(true)
    expect(isSwarmSidecarHref('sidecar://immortals/board')).toBe(true)
    expect(isSwarmSidecarHref('sidecar://swarm/swarm_board/?auto_open=false')).toBe(true)
    expect(isSwarmSidecarHref('sidecar://swarm/swarm_board')).toBe(true)
    expect(isSwarmSidecarHref('sidecar://other/panel')).toBe(false)

    expect(linkTargetOf('sidecar://immortals/board/?auto_open=false')).toEqual({ kind: 'swarm' })
    expect(linkTargetOf('sidecar://swarm/swarm_board/?auto_open=false')).toEqual({ kind: 'swarm' })
    expect(linkTargetOf('sidecar://other/panel')).toEqual({ kind: 'inert' })

    const html = renderToStaticMarkup(
      createElement(Markdown, {
        text: 'Immortals launched — [Open the Immortals UI](sidecar://immortals/board/?auto_open=false)',
      }),
    )
    expect(html).toContain('md__link--swarm')
    expect(html).toContain('Open the Immortals UI')
    expect(html).toContain('Open →')
    expect(html).not.toContain('href="sidecar://')

    const legacyHtml = renderToStaticMarkup(
      createElement(Markdown, {
        text: 'Legacy run — [Open the Swarm UI](sidecar://swarm/swarm_board/?auto_open=false)',
      }),
    )
    expect(legacyHtml).toContain('md__link--swarm')
    expect(legacyHtml).toContain('title="Open the Immortals Board"')
    expect(legacyHtml).not.toContain('href="sidecar://')

    // Omitting label falls back to default quiet bar label "Open the Immortals UI"
    const defaultLabelHtml = renderToStaticMarkup(
      createElement(SwarmChatLink, {
        navigate: () => undefined,
      }),
    )
    expect(defaultLabelHtml).toContain('Open the Immortals UI')

    const navigated: string[] = []
    let openedStore = false
    let prevented = false
    let stopped = false
    const linkEl = SwarmChatLink({
      label: 'Open the Immortals UI',
      navigate: panel => navigated.push(panel),
      onOpenBoard: () => {
        openedStore = true
      },
    })
    linkEl.props.onClick({
      preventDefault: () => {
        prevented = true
      },
      stopPropagation: () => {
        stopped = true
      },
    } as unknown as MouseEvent)
    expect(prevented).toBe(true)
    expect(stopped).toBe(true)
    expect(openedStore).toBe(true)
    expect(navigated).toEqual(['swarm'])

    let keyOpened = false
    const keyLink = SwarmChatLink({
      label: 'Open the Immortals UI',
      navigate: panel => navigated.push(`key:${panel}`),
      onOpenBoard: () => {
        keyOpened = true
      },
    })
    keyLink.props.onKeyDown({
      key: 'Enter',
      preventDefault: () => undefined,
      stopPropagation: () => undefined,
    } as unknown as KeyboardEvent)
    expect(keyOpened).toBe(true)
    expect(navigated).toEqual(['swarm', 'key:swarm'])
  })

  test('completeText("/immortals") and legacy completeText("/swarm") place Command at row #1 and deduplicate skills', async () => {
    const { store } = await setupConnectedStore()
    const immortalsItems = await store.completeText('/immortals')
    expect(immortalsItems[0]).toMatchObject({
      value: '/immortals',
      label: 'immortals',
      kind: 'command',
    })
    expect(immortalsItems.map(item => `${item.kind}:${item.label}`)).toEqual([
      'command:immortals',
      'skill:immortals-coordinator',
      'skill:immortals-monitor',
      'skill:immortals-narrator',
    ])

    const swarmItems = await store.completeText('/swarm')
    expect(swarmItems[0]).toMatchObject({
      value: '/swarm',
      label: 'swarm',
      kind: 'command',
    })
    expect(swarmItems.map(item => `${item.kind}:${item.label}`)).toEqual([
      'command:swarm',
      'skill:swarm-coordinator',
      'skill:swarm-monitor',
      'skill:swarm-narrator',
    ])
  })

  test('/immortals and legacy /swarm composer commands open Immortals Board, activate + submit <task>, and stop workers', async () => {
    expect(parseSwarmSlashCommand('/immortals')).toEqual({ kind: 'open_board' })
    expect(parseSwarmSlashCommand('/immortals board')).toEqual({ kind: 'open_board' })
    expect(parseSwarmSlashCommand('/immortals stop')).toEqual({ kind: 'kill', target: 'all' })
    expect(parseSwarmSlashCommand('/immortals kill worker-api')).toEqual({ kind: 'kill', target: 'worker-api' })
    expect(parseSwarmSlashCommand('/immortals Audit auth and billing')).toEqual({
      kind: 'activate_task',
      task: 'Audit auth and billing',
    })
    expect(parseSwarmSlashCommand('/swarm')).toEqual({ kind: 'open_board' })
    expect(parseSwarmSlashCommand('/swarm board')).toEqual({ kind: 'open_board' })
    expect(parseSwarmSlashCommand('/swarm stop')).toEqual({ kind: 'kill', target: 'all' })
    expect(parseSwarmSlashCommand('/swarm kill worker-api')).toEqual({ kind: 'kill', target: 'worker-api' })
    expect(parseSwarmSlashCommand('/swarm Audit auth and billing')).toEqual({
      kind: 'activate_task',
      task: 'Audit auth and billing',
    })

    const { store, bridge } = await setupConnectedStore()
    bridge.calls.length = 0

    // `/immortals` opens the Immortals Board locally without starting a turn
    expect(await store.submit('/immortals')).toBe(true)
    expect(store.getSnapshot().swarmOpen).toBe(true)
    expect(bridge.calls.some(c => c.method === 'turn.submit')).toBe(false)
    await Bun.sleep(5)
    expect(bridge.calls.some(c => c.method === 'swarm.status')).toBe(true)

    // `/swarm board` bumps swarmOpenRevision again (legacy alias parity)
    const revBefore = store.getSnapshot().swarmOpenRevision ?? 0
    expect(await store.submit('/swarm board')).toBe(true)
    expect(store.getSnapshot().swarmOpenRevision ?? 0).toBeGreaterThan(revBefore)

    // `/immortals <task>` calls `swarm.activate`, opens Immortals Board, and calls `turn.submit` with `<task>`
    store.closeSwarmBoard()
    expect(store.getSnapshot().swarmOpen).toBe(false)
    bridge.calls.length = 0
    expect(await store.submit('/immortals Audit auth and billing in parallel')).toBe(true)
    expect(store.getSnapshot().swarmOpen).toBe(true)
    const methods = bridge.calls.map(c => c.method)
    expect(methods.indexOf('swarm.activate')).toBeGreaterThanOrEqual(0)
    expect(methods.indexOf('turn.submit')).toBeGreaterThan(methods.indexOf('swarm.activate'))
    expect(bridge.calls.find(c => c.method === 'turn.submit')?.params.text).toBe(
      'Audit auth and billing in parallel',
    )

    // `/immortals stop` calls `swarm.kill` with `{ name: 'all' }`
    bridge.calls.length = 0
    expect(await store.submit('/immortals stop')).toBe(true)
    expect(bridge.calls.find(c => c.method === 'swarm.kill')?.params.name).toBe('all')
  })

  test('/btw and /side open the right-bar Side Question panel, stream btw_event deltas, and pass multi-turn history without steering or mutating chat', async () => {
    expect(parseSideQuestionSlashCommand('/btw')).toEqual({ kind: 'open' })
    expect(parseSideQuestionSlashCommand('/side')).toEqual({ kind: 'open' })
    expect(parseSideQuestionSlashCommand('/btw How does auth work?')).toEqual({
      kind: 'ask',
      question: 'How does auth work?',
    })
    expect(parseSideQuestionSlashCommand('/side Explain jwt.ts')).toEqual({
      kind: 'ask',
      question: 'Explain jwt.ts',
    })

    const { store, bridge } = await setupConnectedStore()
    bridge.calls.length = 0

    // Bare `/btw` opens the Side Question panel without submitting or steering
    expect(await store.submit('/btw')).toBe(true)
    expect(store.getSnapshot().sideQuestionOpen).toBe(true)
    expect(bridge.calls).toHaveLength(0)

    // Simulate a running foreground turn: `/btw <question>` must NOT call `turn.steer` or touch `blocks`
    bridge.emit('turn_begin', { session_id: 'sess-swarm-1', text: 'Implement billing webhook' })
    const blocksBefore = store.getSnapshot().blocks.length
    expect(store.getSnapshot().turnActive).toBe(true)

    expect(await store.submit('/btw What port does webhook listen on?')).toBe(true)
    expect(bridge.calls.some(c => c.method === 'turn.steer')).toBe(false)
    expect(bridge.calls.find(c => c.method === 'session.side_question')?.params).toMatchObject({
      question: 'What port does webhook listen on?',
    })
    expect(store.getSnapshot().blocks).toHaveLength(blocksBefore)
    expect(store.getSnapshot().sideQuestionTurns).toHaveLength(1)
    expect(store.getSnapshot().sideQuestionTurns?.[0]?.answer).toBe(
      'Side answer for: What port does webhook listen on?',
    )

    // Follow-up question includes prior settled turn in `history`
    bridge.calls.length = 0
    expect(await store.submit('/side And where is signature verified?')).toBe(true)
    expect(bridge.calls.find(c => c.method === 'session.side_question')?.params.history).toEqual([
      { role: 'user', content: 'What port does webhook listen on?' },
      { role: 'assistant', content: 'Side answer for: What port does webhook listen on?' },
    ])

    // Live `btw_event` delta folding helper & broadcast stream
    const folded1 = applyBtwEventPayload([], {
      request_id: 'stream-1',
      kind: 'begin',
      question: 'Check config',
      model: 'claude-opus-4-6',
    })
    const folded2 = applyBtwEventPayload(folded1!.turns, {
      request_id: 'stream-1',
      kind: 'delta',
      delta: 'Port **8080**.',
    })
    const folded3 = applyBtwEventPayload(folded2!.turns, {
      request_id: 'stream-1',
      kind: 'end',
      answer: 'Port **8080**.',
    })
    expect(folded3).toMatchObject({
      streaming: false,
      error: null,
      turns: [
        {
          requestId: 'stream-1',
          question: 'Check config',
          answer: 'Port **8080**.',
          status: 'done',
        },
      ],
    })

    store.clearSideQuestion()
    expect(store.getSnapshot().sideQuestionTurns).toEqual([])
  })

  test('live swarm_event updates active session roster/overview and merges #seq messages while ignoring foreign sessions', async () => {
    const { store, bridge } = await setupConnectedStore()
    await store.refreshSwarm()
    const initialCount = store.getSnapshot().swarmMessages?.length ?? 0

    // Foreign session event is ignored
    bridge.emit('swarm_event', {
      session_id: 'foreign-session-999',
      kind: 'message_posted',
      head: 99,
      message: {
        seq: 99,
        agent: 'intruder',
        channel: 'main',
        text: 'wrong session',
        mentions: [],
        channel_mentions: [],
        broadcast: false,
        time: '08:00:00',
      },
    })
    expect(store.getSnapshot().swarmMessages).toHaveLength(initialCount)

    // Matching session_id updates status and appends message chronologically without polling
    const updatedStatus: SwarmStatusSnapshot = {
      ...sampleStatus,
      head: 10,
      overview: {
        markdown: '### Updated Live Overview\nAll modules verified.',
        author: 'narrator',
      },
    }
    bridge.calls.length = 0
    bridge.emit('swarm_event', {
      session_id: 'sess-swarm-1',
      kind: 'message_posted',
      head: 10,
      status: updatedStatus,
      message: {
        seq: 10,
        agent: 'coordinator',
        channel: 'main',
        text: '[done] Audit finished cleanly.',
        mentions: [],
        channel_mentions: [],
        broadcast: false,
        time: '07:25:00',
      },
    })

    expect(bridge.calls).toHaveLength(0)
    expect(store.getSnapshot().swarm?.head).toBe(10)
    expect(store.getSnapshot().swarm?.overview?.markdown).toContain('Updated Live Overview')
    expect(store.getSnapshot().swarmMessages?.at(-1)?.seq).toBe(10)
    expect(store.getSnapshot().swarmMessages?.at(-1)?.text).toContain('[done] Audit finished cleanly.')

    // Out-of-order / duplicate merge stays strictly ascending and deduplicated
    const merged = mergeSwarmMessages(store.getSnapshot().swarmMessages ?? [], [
      {
        seq: 6,
        agent: 'worker-api',
        channel: 'main',
        text: 'mid-step update',
        mentions: [],
        channel_mentions: [],
        broadcast: false,
        time: '07:22:00',
      },
      store.getSnapshot().swarmMessages![0]!,
    ])
    expect(merged.map(m => m.seq)).toEqual([1, 2, 3, 4, 6, 10])
    expect(parseSwarmStatus({ ok: false, error: 'nope' })).toBeNull()
  })

  test('Desktop Shell auto-includes active Immortals tab in right bar, More menu lists Immortals & Side question, and DesktopRail renders SideQuestionPanel', () => {
    const activeSnap = baseSnapshot({
      swarm: sampleStatus,
      swarmMessages: sampleMessages,
    })

    // Desktop Shell header shows live Immortals button (`1 working`), Ask aside chip, AND auto-adds Immortals tab to right-bar tabs!
    const shellHtml = renderToStaticMarkup(createElement(Shell, { snap: activeSnap }))
    expect(shellHtml).toContain('aria-label="Open Immortals Board"')
    expect(shellHtml).toContain('1 working')
    expect(shellHtml).toContain('Ask aside')
    expect(shellHtml).toContain('aria-label="Close Immortals"')

    // VibesSideNav More menu includes Immortals and Side question
    const moreHtml = renderToStaticMarkup(
      createElement(VibesSideNav, {
        snap: activeSnap,
        page: null,
        filter: '',
        setFilter: () => undefined,
        defaultMoreOpen: true,
      }),
    )
    expect(moreHtml).toContain('<span>Immortals</span>')
    expect(moreHtml).toContain('<span>Side question</span>')

    // SectionsMenuPanel with showSwarm renders Immortals and Side question rows
    const menuHtml = renderToStaticMarkup(
      createElement(SectionsMenuPanel, {
        hover: 'swarm',
        hoverTop: 0,
        needle: '',
        setNeedle: () => undefined,
        agents: [],
        artifacts: [],
        changes: [],
        swarm: sampleStatus,
        showSwarm: true,
        onHover: () => undefined,
        onPick: () => undefined,
      }),
    )
    expect(menuHtml).toContain('Immortals')
    expect(menuHtml).toContain('Side question')
    expect(menuHtml).toContain('1 working')
    expect(menuHtml).toContain('Auth &amp; Billing Audit')
    expect(menuHtml).toContain('@worker-api')

    // DesktopRail (`panel="swarm"` and `panel="btw"`) renders SwarmBoard and SideQuestionPanel
    const railHtml = renderToStaticMarkup(
      createElement(DesktopRail, {
        panel: 'swarm',
        snap: activeSnap,
        close: () => undefined,
        activityDetails: null,
      }),
    )
    expect(railHtml).toContain('aria-label="Immortals Board"')
    expect(railHtml).toContain('Auth &amp; Billing Audit')

    const btwRailHtml = renderToStaticMarkup(
      createElement(DesktopRail, {
        panel: 'btw',
        snap: activeSnap,
        close: () => undefined,
        activityDetails: null,
      }),
    )
    expect(btwRailHtml).toContain('aria-label="Side question"')
    expect(btwRailHtml).toContain('Ephemeral · read-only')

    const sidePanelDirect = renderToStaticMarkup(createElement(SideQuestionPanel, { snap: activeSnap }))
    expect(sidePanelDirect).toContain('aria-label="Side question prompt"')

    const activityHtml = renderToStaticMarkup(createElement(ActivityDetails, { snap: activeSnap }))
    expect(activityHtml).toContain('Open Immortals Board')
    expect(activityHtml).toContain('1 working · 3 waiting')

    // VS Code VscodeChat header shows live Immortals pill and DesktopSheet (`panel="swarm"`) renders SwarmBoard
    const vscodeHtml = renderToStaticMarkup(createElement(VscodeChat, { snap: activeSnap }))
    expect(vscodeHtml).toContain('xv-pill--swarm')
    expect(vscodeHtml).toContain('Immortals · 1')

    const sheetHtml = renderToStaticMarkup(
      createElement(DesktopSheet, {
        panel: 'swarm',
        snap: activeSnap,
        close: () => undefined,
      }),
    )
    expect(sheetHtml).toContain('aria-label="Immortals Board"')
    expect(sheetHtml).toContain('Auth &amp; Billing Audit')
  })
})

describe('Large-scale Immortals board polish: ISO timestamps, inline Markdown, auto-folding, 57-member filter, and quiet wake notices', () => {
  test('formatSwarmShortTime formats ISO timestamps into HH:MM inline while keeping full ISO timestamp in title tooltip', () => {
    expect(formatSwarmShortTime('2026-11-22T01:57:02.980Z')).toBe('01:57')
    expect(formatSwarmShortTime('2026-11-22T14:05:59+03:30')).toBe('14:05')
    expect(formatSwarmShortTime('2026-11-22 09:42:11')).toBe('09:42')
    expect(formatSwarmShortTime('07:10:01')).toBe('07:10')
    expect(formatSwarmShortTime('07:10')).toBe('07:10')
    expect(formatSwarmShortTime('')).toBe('')

    const isoTime = '2026-11-22T01:57:02.980Z'
    const html = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: {
          ...sampleStatus,
          overview: {
            markdown: '### Summary',
            author: 'narrator',
            updated_at: isoTime,
          },
        },
        messages: [
          {
            seq: 149,
            agent: 'dash-read-3',
            channel: 'main',
            text: '@study-8 [done] Verified **dashboard** slice in `package.json:1-45` (#41).',
            mentions: ['study-8'],
            channel_mentions: [],
            broadcast: false,
            time: isoTime,
          },
        ],
      }),
    )
    expect(html).toContain(`title="${isoTime}">01:57</time>`)
    expect(html).not.toContain(`>${isoTime}<`)
    expect(html).toContain(`title="${isoTime}">01:57</span>`)
  })

  test('renders inline code spans, bold spans, bullet lists, @mentions, #seq refs, and [done]/[stall] tags cleanly without hardcoded hex colors', () => {
    const richText = [
      '@study-8 [done] **Dashboard** slice #41 read from pinned blobs:',
      '• `index.html:3-18` mounts `#root` -> `/src/main.tsx`',
      '- `package.json:3-22,24-44` runs **strict** `tsc --noEmit`',
      '- Check `@not-a-mention` in code and @worker-api outside [stall]',
    ].join('\n')

    const tokens = tokenizeSwarmMessage('**Verify** `foo.ts:1-20` with `#1` and @worker-1 #41 [done]')
    expect(tokens).toEqual([
      { kind: 'bold', raw: '**Verify**', text: 'Verify' },
      { kind: 'text', raw: ' ' },
      { kind: 'code', raw: '`foo.ts:1-20`', code: 'foo.ts:1-20' },
      { kind: 'text', raw: ' with ' },
      { kind: 'code', raw: '`#1`', code: '#1' },
      { kind: 'text', raw: ' and ' },
      { kind: 'mention', raw: '@worker-1', name: 'worker-1' },
      { kind: 'text', raw: ' ' },
      { kind: 'seq_ref', raw: '#41', seq: 41 },
      { kind: 'text', raw: ' ' },
      { kind: 'done', raw: '[done]' },
    ])

    const blocks = parseSwarmMessageBlocks(richText)
    expect(blocks).toHaveLength(2)
    expect(blocks[0]?.kind).toBe('paragraph')
    expect(blocks[1]).toEqual({
      kind: 'list',
      items: [
        '`index.html:3-18` mounts `#root` -> `/src/main.tsx`',
        '`package.json:3-22,24-44` runs **strict** `tsc --noEmit`',
        'Check `@not-a-mention` in code and @worker-api outside [stall]',
      ],
    })

    const html = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: sampleStatus,
        messages: [
          {
            seq: 149,
            agent: 'dash-read-3',
            channel: 'main',
            text: richText,
            mentions: ['study-8', 'worker-api'],
            channel_mentions: [],
            broadcast: false,
            time: '2026-11-22T01:57:02.980Z',
          },
        ],
      }),
    )

    expect(html).toContain('<strong')
    expect(html).toContain('>Dashboard</strong>')
    expect(html).toContain('>strict</strong>')
    expect(html).toContain('<code class="swarm-msg__code"')
    expect(html).toContain('>index.html:3-18</code>')
    expect(html).toContain('>package.json:3-22,24-44</code>')
    expect(html).toContain('>#root</code>')
    expect(html).toContain('>@not-a-mention</code>')
    expect(html).toContain('<ul class="swarm-msg__list"')
    expect(html).toContain('<li')
    expect(html).toContain('swarm-token--mention" title="Mention @study-8 in composer">@study-8</button>')
    expect(html).toContain('aria-label="Jump to message #41"')
    expect(html).toContain('swarm-badge--done">[done]</span>')
    expect(html).toContain('swarm-badge--stall">[stall]</span>')

    const chromeHtml = html.replace(/<svg[^>]*swarm-member__sprite[\s\S]*?<\/svg>/g, '')
    expect(chromeHtml.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])
  })

  test('renders Markdown headings (###), italics (*(note)*), strikethrough, ordered lists, code fences, quotes, rules and tables without breaking #seq refs or snake_case identifiers', () => {
    const auditPost = [
      '### Verification',
      '- In _resolve_shard_specs under TPU use_forced_layout=True, `pytest -q` passed.',
      '- Pure-Python check -> Exit 0 (`ALL_CONTRACT_CHECKS_PASSED`). *(Full multi-device model compute tests in `test_fused_layout_tp_portability.py` require TPU hardware and were not run on CPU.)*',
      '',
      '#### Next steps',
      '1. Run ~~legacy~~ new TPU check for #7 and @worker-api.',
      '2. Inspect `#12` inside code.',
      '',
      '> Keep _italic notes_ readable without touching linear_attention_merged_split_proj.',
      '',
      '---',
      '',
      '```python',
      '### comment inside code stays literal',
      'x = 1 * 2 * 3',
      '```',
      '',
      '| Suite | Result |',
      '| --- | --- |',
      '| CPU | **Pass** |',
    ].join('\n')

    const blocks = parseSwarmMessageBlocks(auditPost)
    expect(blocks.map(b => b.kind)).toEqual([
      'heading',
      'list',
      'heading',
      'ordered_list',
      'quote',
      'hr',
      'code',
      'table',
    ])
    expect(blocks[0]).toEqual({ kind: 'heading', level: 3, text: 'Verification' })
    expect(blocks[2]).toEqual({ kind: 'heading', level: 4, text: 'Next steps' })

    // Snake_case identifiers must never be turned into italics.
    const snakeTokens = tokenizeSwarmMessage(
      'In _resolve_shard_specs with linear_attention_merged_split_proj and _italic_ plus *(note)*',
    )
    expect(snakeTokens.filter(t => t.kind === 'italic')).toEqual([
      { kind: 'italic', raw: '_italic_', text: 'italic' },
      { kind: 'italic', raw: '*(note)*', text: '(note)' },
    ])

    // `#12` seq refs and `#main` remain paragraph text, never headings.
    expect(parseSwarmMessageBlocks('#12 confirmed on #main')).toEqual([
      { kind: 'paragraph', text: '#12 confirmed on #main' },
    ])

    const html = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: sampleStatus,
        initialExpandedSeqs: [150],
        messages: [
          {
            seq: 150,
            agent: 'test-verifier',
            channel: 'main',
            text: auditPost,
            mentions: ['worker-api'],
            channel_mentions: [],
            broadcast: false,
            time: '2026-11-24T00:15:00.000Z',
          },
        ],
      }),
    )

    expect(html).toContain('<h4 class="swarm-msg__heading" data-level="3">Verification</h4>')
    expect(html).toContain('<h4 class="swarm-msg__heading" data-level="4">Next steps</h4>')
    expect(html).not.toContain('### Verification')
    expect(html).not.toContain('#### Next steps')
    expect(html).toContain('<em class="swarm-msg__em">(Full multi-device model compute tests in <code class="swarm-msg__code"')
    expect(html).toContain('>test_fused_layout_tp_portability.py</code> require TPU hardware and were not run on CPU.)</em>')
    expect(html).toContain('_resolve_shard_specs')
    expect(html).toContain('linear_attention_merged_split_proj')
    expect(html).toContain('<em class="swarm-msg__em">italic notes</em>')
    expect(html).toContain('<del class="swarm-msg__del">legacy</del>')
    expect(html).toContain('<ol class="swarm-msg__list swarm-msg__list--ordered"')
    expect(html).toContain('aria-label="Jump to message #7"')
    expect(html).toContain('<blockquote class="swarm-msg__quote">')
    expect(html).toContain('<hr class="swarm-msg__hr"/>')
    expect(html).toContain('<pre class="swarm-msg__pre" data-lang="python"><code>### comment inside code stays literal')
    expect(html).toContain('<table class="swarm-msg__table">')
    expect(html).toContain('<th>Suite</th>')
    expect(html).toContain('>Pass</strong>')

    const chromeHtml = html.replace(/<svg[^>]*swarm-member__sprite[\s\S]*?<\/svg>/g, '')
    expect(chromeHtml.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])
  })

  test('auto-folds board messages with > 6 lines or > 420 chars and toggles cleanly between Show more and Show less', () => {
    const shortMsg = 'Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6'
    expect(previewSwarmMessage(shortMsg).foldable).toBe(false)

    const longLinesText = Array.from(
      { length: 10 },
      (_, idx) => `- \`file-${idx + 1}.ts:1-20\` verified slice ${idx + 1}`,
    ).join('\n')
    const linesPreview = previewSwarmMessage(longLinesText)
    expect(linesPreview.foldable).toBe(true)
    expect(linesPreview.displayLines).toBe(10)
    expect(linesPreview.preview).toContain('file-5.ts:1-20')
    expect(linesPreview.preview).not.toContain('file-10.ts:1-20')

    const longSingleLine = `${'Dense repository analysis of pinned sources. '.repeat(12)}END_MARKER`
    const charPreview = previewSwarmMessage(longSingleLine)
    expect(charPreview.foldable).toBe(true)
    expect(charPreview.preview.length).toBeLessThanOrEqual(385)
    expect(charPreview.preview).not.toContain('END_MARKER')

    const longMessage: SwarmBoardMessage = {
      seq: 149,
      agent: 'dash-read-3',
      channel: 'main',
      text: longLinesText,
      mentions: [],
      channel_mentions: [],
      broadcast: false,
      time: '2026-11-22T01:57:02.980Z',
    }

    const foldedHtml = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: sampleStatus,
        messages: [longMessage],
      }),
    )
    expect(foldedHtml).toContain('data-folded="true"')
    expect(foldedHtml).toContain('Show more · 10 lines')
    expect(foldedHtml).toContain('file-5.ts:1-20')
    expect(foldedHtml).not.toContain('file-10.ts:1-20')

    const expandedHtml = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: sampleStatus,
        messages: [longMessage],
        initialExpandedSeqs: [149],
      }),
    )
    expect(expandedHtml).not.toContain('data-folded="true"')
    expect(expandedHtml).toContain('Show less')
    expect(expandedHtml).toContain('file-10.ts:1-20')

    let toggledSeq = 0
    const view = SwarmBoardView({
      status: sampleStatus,
      channels: swarmChannelsOf(sampleStatus, [longMessage]),
      members: sortSwarmMembers(sampleStatus.members),
      counts: swarmMemberCounts(sampleStatus.members),
      messages: [longMessage],
      visibleMessages: [longMessage],
      models: baseSnapshot().models,
      online: true,
      loading: false,
      busyAction: '',
      error: '',
      selectedChannel: '*',
      highlightedSeq: null,
      expandedSeqs: new Set(),
      feedbackOpen: false,
      feedbackDraft: '',
      postDraft: '',
      switchingMember: '',
      modelDrafts: {},
      onRefresh: () => undefined,
      onStopAll: () => undefined,
      onToggleFeedback: () => undefined,
      onChangeFeedbackDraft: () => undefined,
      onSubmitFeedback: () => undefined,
      onCloseFeedback: () => undefined,
      onSelectChannel: () => undefined,
      onInsertMention: () => undefined,
      onInsertSeqRef: () => undefined,
      onJumpToSeq: () => undefined,
      onToggleExpandMessage: seq => {
        toggledSeq = seq
      },
      onKillMember: () => undefined,
      onToggleModelSwitch: () => undefined,
      onChangeModelDraft: () => undefined,
      onSubmitModelSwitch: () => undefined,
      onCancelModelSwitch: () => undefined,
      onChangePostDraft: () => undefined,
      onSubmitPost: () => undefined,
    })
    const foldBtn = walkElements(view).find(
      el => el.type === 'button' && el.props['aria-label'] === 'Show more of message #149',
    )!
    foldBtn.props.onClick!()
    expect(toggledSeq).toBe(149)
  })

  test('Members tab shows compact All/Active/Waiting/Retired filter + search strip only when members.length > 8 (57-member scale)', () => {
    // 6-member small run omits the filter strip
    const smallHtml = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: sampleStatus,
        messages: sampleMessages,
        initialTab: 'members',
      }),
    )
    expect(smallHtml).not.toContain('aria-label="Filter Immortals members"')
    expect(smallHtml).not.toContain('placeholder="Filter members…"')

    // 57-member large run renders quick filter buttons + search input
    const largeMembers: SwarmMemberSnapshot[] = [
      {
        name: 'coordinator',
        role: 'coordinator',
        model: 'gpt-5.2',
        model_display_name: 'GPT-5.2',
        workspace: 'inherit',
        channels: ['*'],
        state: 'waiting',
        killed: false,
        last_read_seq: 203,
      },
      ...Array.from({ length: 5 }, (_, idx): SwarmMemberSnapshot => ({
        name: `dash-read-${idx + 1}`,
        role: 'worker',
        added_by: 'coordinator',
        model: 'gpt-5.2',
        model_display_name: 'GPT-5.2',
        workspace: 'inherit',
        channels: ['main'],
        state: 'working',
        killed: false,
        last_read_seq: 203,
        current_activity: `Reading dashboard slice ${idx + 1}`,
      })),
      ...Array.from({ length: 48 }, (_, idx): SwarmMemberSnapshot => ({
        name: `study-${idx + 1}`,
        role: 'worker',
        added_by: 'coordinator',
        model: 'gpt-5.2',
        model_display_name: 'GPT-5.2',
        workspace: 'inherit',
        channels: ['main'],
        state: 'waiting',
        killed: false,
        last_read_seq: 200,
      })),
      ...Array.from({ length: 3 }, (_, idx): SwarmMemberSnapshot => ({
        name: `retired-${idx + 1}`,
        role: 'worker',
        added_by: 'coordinator',
        model: 'gpt-5.2',
        model_display_name: 'GPT-5.2',
        workspace: 'inherit',
        channels: ['main'],
        state: 'killed',
        killed: true,
        last_read_seq: 120,
      })),
    ]
    expect(largeMembers).toHaveLength(57)

    expect(filterSwarmMembers(largeMembers, 'active', '')).toHaveLength(5)
    expect(filterSwarmMembers(largeMembers, 'waiting', '')).toHaveLength(49)
    expect(filterSwarmMembers(largeMembers, 'retired', '')).toHaveLength(3)
    expect(filterSwarmMembers(largeMembers, 'all', 'dash-read-3').map(m => m.name)).toEqual(['dash-read-3'])

    const largeStatus: SwarmStatusSnapshot = { ...sampleStatus, head: 203, members: largeMembers }
    const activeFilteredHtml = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: largeStatus,
        messages: sampleMessages,
        initialTab: 'members',
        initialMemberFilter: 'active',
        initialMemberQuery: 'dash-read-3',
      }),
    )
    expect(activeFilteredHtml).toContain('aria-label="Filter Immortals members"')
    expect(activeFilteredHtml).toContain('placeholder="Filter members…"')
    expect(activeFilteredHtml).toContain('>All</span>')
    expect(activeFilteredHtml).toContain('>Active</span>')
    expect(activeFilteredHtml).toContain('>Waiting</span>')
    expect(activeFilteredHtml).toContain('>Retired</span>')
    expect(activeFilteredHtml).toContain('>@dash-read-3</button>')
    expect(activeFilteredHtml).not.toContain('>@study-1</button>')
    expect(activeFilteredHtml).not.toContain('>@retired-1</button>')
  })

  test('blocks.ts converts internal Immortals/Swarm wakeups (live turn_begin, agent/monitor origin, replay_user, stored history) into quiet notice rows instead of user chat bubbles', async () => {
    expect(isInternalSwarmWakeText('Immortals board #194 (@study-1)')).toBe(true)
    expect(isInternalSwarmWakeText('Swarm board #12 (@worker-1)')).toBe(true)
    expect(isInternalSwarmWakeText('Immortals worker error (@study-2)')).toBe(true)
    expect(isInternalSwarmWakeText('Please explain how Immortals works')).toBe(false)
    expect(formatInternalWakeNotice('Immortals board #194 (@study-1)')).toBe('Immortals · #194 from @study-1')
    expect(formatInternalWakeNotice('Immortals worker error (@study-2)')).toBe('Immortals · worker error from @study-2')

    // Direct BlockBuilder.push('turn_begin', ...) with agent/monitor origin or wake display text
    const builder = new BlockBuilder()
    builder.pushUser('Audit the entire repo')
    builder.push('turn_begin', { origin: 'agent', text: 'Immortals board #194 (@study-1)' })
    builder.push('turn_begin', { origin: 'monitor', user_input: 'Monitor stall alert on #195' })
    builder.push('turn_begin', { text: 'Swarm worker error (@worker-9)' })
    builder.push('notification', { category: 'history', type: 'replay_user', body: '✨ Immortals board #196 (@dash-read-3)' })
    const blocks = builder.all()
    expect(blocks.filter(b => b.kind === 'user').map(b => b.kind === 'user' ? b.text : '')).toEqual([
      'Audit the entire repo',
    ])
    expect(blocks.filter(b => b.kind === 'notice')).toMatchObject([
      { kind: 'notice', error: false, text: 'Immortals · #194 from @study-1' },
      { kind: 'notice', error: false, text: 'Monitor stall alert on #195' },
      { kind: 'notice', error: false, text: 'Immortals · worker error from @worker-9' },
      { kind: 'notice', error: false, text: 'Immortals · #196 from @dash-read-3' },
    ])

    // Stored history hydration
    const storedBlocks = blocksFromStoredMessages([
      { role: 'user', content: 'Map every subsystem' },
      { role: 'user', origin: 'agent', text: 'Immortals board #194 (@study-1)', content: '[board] the latest message is #194' },
      { role: 'user', text: 'Swarm worker error (@study-2)', content: 'worker failed' },
      { role: 'user', content: '[board] the latest message is #200 (you have read up to #194). New board message #200 from @study-8 in #main addressed @coordinator.' },
      { role: 'assistant', content: 'Incorporated all slices.' },
    ])
    expect(storedBlocks.filter(b => b.kind === 'user').map(b => b.kind === 'user' ? b.text : '')).toEqual([
      'Map every subsystem',
    ])
    expect(storedBlocks.filter(b => b.kind === 'notice').map(b => b.kind === 'notice' ? b.text : '')).toEqual([
      'Immortals · #194 from @study-1',
      'Immortals · worker error from @study-2',
      'Immortals · #200 from @study-8',
    ])

    // Live Store turn_begin push for Immortals coordinator wake never produces a fake right-aligned user bubble
    const { store, bridge } = await setupConnectedStore()
    bridge.emit('turn_begin', { session_id: 'sess-swarm-1', text: 'Human question' })
    bridge.emit('turn_end', { session_id: 'sess-swarm-1' })
    bridge.emit('turn_begin', {
      session_id: 'sess-swarm-1',
      origin: 'agent',
      text: 'Immortals board #194 (@study-1)',
    })
    bridge.emit('text_part', { session_id: 'sess-swarm-1', text: 'Synthesizing slice #194.' })
    bridge.emit('turn_end', { session_id: 'sess-swarm-1' })

    const liveBlocks = store.getSnapshot().blocks
    expect(liveBlocks.filter(b => b.kind === 'user').map(b => b.kind === 'user' ? b.text : '')).toEqual([
      'Human question',
    ])
    expect(liveBlocks.some(b => b.kind === 'notice' && b.text === 'Immortals · #194 from @study-1')).toBe(true)
  })

  test('Immortals Board auto-follow scroll helpers & floating ↓ Latest pill with unread count and click-to-bottom', () => {
    // 1. Scroll-bottom detection: no overflow, at bottom, within 48px threshold, and scrolled up
    expect(isNearScrollBottom({ scrollTop: 0, clientHeight: 400, scrollHeight: 320 })).toBe(true)
    expect(isNearScrollBottom({ scrollTop: 600, clientHeight: 400, scrollHeight: 1000 })).toBe(true)
    expect(isNearScrollBottom({ scrollTop: 560, clientHeight: 400, scrollHeight: 1000 })).toBe(true)
    expect(isNearScrollBottom({ scrollTop: 551, clientHeight: 400, scrollHeight: 1000 })).toBe(false)
    expect(isNearScrollBottom({ scrollTop: 0, clientHeight: 400, scrollHeight: 1000 })).toBe(false)

    // 2. Unseen counter accumulates while scrolled up (non-yanking) and clears when pinned at bottom
    expect(nextUnseenCount(0, 10, 13, false)).toBe(3)
    expect(nextUnseenCount(3, 13, 15, false)).toBe(5)
    expect(nextUnseenCount(5, 15, 15, false)).toBe(5)
    expect(nextUnseenCount(5, 15, 16, true)).toBe(0)

    // 3. SwarmJumpLatestButton hides while pinned, renders pill & unread count while scrolled up
    expect(
      renderToStaticMarkup(createElement(SwarmJumpLatestButton, { visible: false, unseenCount: 0, onClick: () => {} })),
    ).toBe('')

    let jumped = 0
    const pillEl = SwarmJumpLatestButton({
      visible: true,
      unseenCount: 3,
      onClick: () => {
        jumped += 1
      },
    })
    expect(pillEl).not.toBeNull()
    const pillHtml = renderToStaticMarkup(pillEl!)
    expect(pillHtml).toContain('swarm-jump-latest')
    expect(pillHtml).toContain('Latest')
    expect(pillHtml).toContain('Jump to latest (3 new)')
    expect(pillHtml).toContain('swarm-jump-latest__count')
    expect(pillHtml).toContain('>3<')
    expect(pillHtml).not.toMatch(/#[0-9a-fA-F]{6}\b/)
    ;(pillEl!.props as { onClick: () => void }).onClick()
    expect(jumped).toBe(1)

    // 4. SwarmBoardView renders feed inside .swarm-board__feed-stage, hides pill at bottom, shows pill when scrolled up
    const baseProps: SwarmBoardViewProps = {
      status: sampleStatus,
      channels: [{ name: 'main', topic: '', subscriber_count: 2, archived: false }],
      members: sampleStatus.members,
      counts: { total: sampleStatus.members.length, working: 1, waiting: 1, stopped: 0, killed: 0 },
      messages: sampleMessages,
      visibleMessages: sampleMessages,
      models: [],
      online: true,
      loading: false,
      busyAction: '',
      error: '',
      selectedChannel: '*',
      highlightedSeq: null,
      feedbackOpen: false,
      feedbackDraft: '',
      postDraft: '',
      switchingMember: '',
      modelDrafts: {},
      showJumpLatest: false,
      unseenLatestCount: 0,
      onJumpToLatest: () => {
        jumped += 1
      },
      onRefresh: () => {},
      onStopAll: () => {},
      onToggleFeedback: () => {},
      onChangeFeedbackDraft: () => {},
      onSubmitFeedback: () => {},
      onCloseFeedback: () => {},
      onSelectChannel: () => {},
      onInsertMention: () => {},
      onInsertSeqRef: () => {},
      onJumpToSeq: () => {},
      onKillMember: () => {},
      onToggleModelSwitch: () => {},
      onChangeModelDraft: () => {},
      onSubmitModelSwitch: () => {},
      onCancelModelSwitch: () => {},
      onChangePostDraft: () => {},
      onSubmitPost: () => {},
    }

    const pinnedHtml = renderToStaticMarkup(createElement(SwarmBoardView, baseProps))
    expect(pinnedHtml).toContain('swarm-board__feed-stage')
    expect(pinnedHtml).not.toContain('swarm-jump-latest')

    const scrolledUpHtml = renderToStaticMarkup(
      createElement(SwarmBoardView, { ...baseProps, showJumpLatest: true, unseenLatestCount: 4 }),
    )
    expect(scrolledUpHtml).toContain('swarm-jump-latest')
    expect(scrolledUpHtml).toContain('Latest')
    expect(scrolledUpHtml).toContain('Jump to latest (4 new)')
    expect(scrolledUpHtml).not.toMatch(/#[0-9a-fA-F]{6}\b/)

    // Member inspector wraps live transcript in .swarm-inspector__stage for floating Latest pill support
    const inspectedHtml = renderToStaticMarkup(
      createElement(SwarmBoardView, {
        ...baseProps,
        inspectedMember: sampleInspect.member.name,
        inspectResult: sampleInspect,
      }),
    )
    expect(inspectedHtml).toContain('swarm-inspector__stage')
    expect(inspectedHtml).not.toMatch(/#[0-9a-fA-F]{6}\b/)

    // 5. CSS uses popover surface + 999px pill radius in both atelier.css and vibes.css
    const rendererDir = join(import.meta.dir, '../src/desktop/renderer')
    const atelierCss = readFileSync(join(rendererDir, 'atelier.css'), 'utf8')
    const vibesCss = readFileSync(join(rendererDir, 'vibes.css'), 'utf8')
    expect(atelierCss).toContain('.swarm-board__feed-stage,.swarm-inspector__stage{position:relative')
    expect(atelierCss).toContain('.swarm-jump-latest{position:absolute;right:16px;bottom:14px')
    expect(atelierCss).toContain('border-radius:999px;background:var(--x-popover,var(--x-dialog))')
    expect(vibesCss).toContain('.swarm-jump-latest{background:var(--x-popover);border:1px solid var(--x-hairline);border-radius:999px')
  })

  test('renders AGY-inspired Immortals Overview, Board, Members, and Graph layouts with SimurghSprite branding and zero hardcoded hex colors', () => {
    const html = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: sampleStatus,
        messages: sampleMessages,
        models: baseSnapshot().models,
      }),
    )

    // Default tab is Overview; underline tabs follow Overview -> Board -> Members -> Graph
    expect(html).toContain('data-tab="overview"')
    const overviewIdx = html.indexOf('>Overview</span>')
    const boardIdx = html.indexOf('>Board</span>')
    const membersIdx = html.indexOf('>Members</span>')
    const graphIdx = html.indexOf('>Graph</span>')
    expect(overviewIdx).toBeGreaterThan(-1)
    expect(overviewIdx).toBeLessThan(boardIdx)
    expect(boardIdx).toBeLessThan(membersIdx)
    expect(membersIdx).toBeLessThan(graphIdx)

    // Live Overview dashboard: Running card, X of Y tasks done card, Threads groups, and The team SimurghSprite tile strip
    expect(html).toContain('swarm-ov-card--running')
    expect(html).toContain('>Running</span>')
    expect(html).toContain('tasks done')
    expect(html).toContain('>Threads</h4>')
    expect(html).toContain('>The team</h4>')
    expect(html).toContain('swarm-ov-coord')
    expect(html).toContain('aria-label="Immortals team tiles"')
    expect(html).toContain('swarm-avatar-tile--all')
    expect(html).toContain('swarm-avatar-tile__sleep')

    // Board: AGY search bar, filter toggle, author -> recipient headers, and SimurghSprite avatars
    expect(html).toContain('aria-label="Search board messages"')
    expect(html).toContain('aria-label="Filter board messages"')
    expect(html).toContain('swarm-msg__arrow')
    expect(html).toContain('swarm-msg__recipients')
    expect(html).toContain('swarm-msg__avatar')

    // Members: Search members bar, STANDING BY / RETIRED section dividers, and latest post preview
    expect(html).toContain('aria-label="Search members"')
    expect(html).toContain('STANDING BY')
    expect(html).toContain('RETIRED')
    expect(html).toContain('swarm-member__preview')

    // Idle Immortals empty state renders clean card without narrator feedback clutter
    const idleHtml = renderToStaticMarkup(
      createElement(SwarmBoard, {
        status: null,
        messages: [],
        models: [],
      }),
    )
    expect(idleHtml).toContain('Immortals ready')
    expect(idleHtml).toContain('/immortals &lt;task&gt;')
    expect(idleHtml).not.toContain('<strong>0</strong> active')
    expect(idleHtml).not.toContain('Feedback to narrator…')
    expect(html).toContain('swarm-pill-composer')
    expect(html).toContain('swarm-pill-composer__send')

    // Zero hardcoded hex colors anywhere in rendered chrome
    const chromeHtml = html.replace(/<svg[^>]*swarm-member__sprite[\s\S]*?<\/svg>/g, '')
    expect(chromeHtml.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])
  })
})
