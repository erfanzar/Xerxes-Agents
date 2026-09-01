// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Application store: one plain observable object, read by React through
 * `useSyncExternalStore`. All daemon interaction lives here as actions; the
 * transcript fold lives in a BlockBuilder and is exposed as part of the
 * snapshot. No React state, no effect cascades.
 *
 * Beyond the transcript, the store folds the same event stream into the
 * workspace surfaces: a per-file changes accumulator (edit-family tool calls
 * carry real old/new strings), a bounded raw-event ring for the Log tab, a
 * plan capture driven by plan-mode agent output, a steering queue mirrored
 * from `turn.steer` acceptances, and failed-turn state keyed off error
 * notifications (a provider failure arrives as text, never as silence).
 */

import { BlockBuilder, blocksFromStoredMessages, editStatsOf, parseArgs } from './blocks.js'
import {
  daemonCompatibilityWarning,
  DESKTOP_DAEMON_PROTOCOL,
  DESKTOP_VERSION,
  EXPECTED_DAEMON_BUILD_ID,
} from './buildInfo.js'
import { sessionToMarkdown, type ExportSession } from './exportMarkdown.js'
import type {
  AgentMember,
  AgentPreset,
  Approval,
  ApprovalResponse,
  BackgroundJob,
  Block,
  CachedModel,
  CreatorTrace,
  DaemonEvent,
  DiffFile,
  DiffLine,
  FailedTurn,
  LogEntry,
  McpServerStatus,
  ModelChoice,
  PermissionMode,
  PlanItem,
  PlanState,
  ProviderRow,
  ProviderTypeRow,
  QueueItem,
  SessionRow,
  SettingsTab,
  SkillSuggestion,
  WorkspaceTab,
} from './types.js'

export type Connection = 'connecting' | 'online' | 'offline'

const STREAM_THINKING_KEY = 'xerxes.streamThinking'

/** Persisted display preference; defaults on everywhere (SSR/test safe). */
function readStreamThinking(): boolean {
  try {
    return (typeof localStorage === 'undefined' ? null : localStorage.getItem(STREAM_THINKING_KEY)) !== '0'
  } catch {
    return true
  }
}

function writeStreamThinking(value: boolean): void {
  try {
    localStorage?.setItem(STREAM_THINKING_KEY, value ? '1' : '0')
  } catch {
    // A blocked storage area must not break the toggle — the choice just
    // stays session-scoped.
  }
}

/** Slow fleet cadence while agent-family tools run — snapshots lag the event. */
const FLEET_POLL_MS = 2_000

/**
 * Wire names whose execution can spawn, stop, or reap a subagent. The fleet
 * rail reads the parent's `subagent_snapshots`, which only move while one of
 * these runs — every other tool call leaves the panel unchanged, so only
 * these wake the poller.
 */
const AGENT_FAMILY_TOOLS = new Set([
  'agent', 'agenttool',
  'spawnagents',
  'handoff', 'handofftool',
  'sendmessage', 'sendmessagetool',
  'taskcreate', 'taskcreatetool',
  'taskstop', 'taskstoptool',
  'taskupdate', 'taskupdatetool',
  'awaitagents', 'resetagent',
])

function isAgentFamilyTool(name: unknown): boolean {
  if (typeof name !== 'string' || !name) return false
  const tail = name.split(/[.:]/).pop() ?? name
  return AGENT_FAMILY_TOOLS.has(tail.toLowerCase())
}

/** The spawn-capable subset — these open members on the turn's agents card. */
const SPAWN_TOOLS = new Set([
  'agent', 'agenttool',
  'spawnagents',
  'taskcreate', 'taskcreatetool',
  'handoff', 'handofftool',
])

function isSpawnTool(name: unknown): boolean {
  if (typeof name !== 'string' || !name) return false
  return SPAWN_TOOLS.has((name.split(/[.:]/).pop() ?? name).toLowerCase())
}

/** Snapshot statuses fold onto the card's display vocabulary. */
function agentStatusOf(status: string): string {
  const s = status.toLowerCase()
  if (s === 'working' || s === 'running' || s === 'starting' || s === 'waiting') return 'working'
  if (s === 'completed' || s === 'done' || s === 'closed') return 'completed'
  if (s === 'error' || s === 'failed') return 'failed'
  if (s === 'cancelled' || s === 'interrupted') return 'cancelled'
  return s || 'working'
}

/**
 * Members a spawn call opens on the agents card, parsed from its arguments.
 * SpawnAgents carries an `agents` batch; the single-spawn tools carry one
 * title-ish field. Keys are call-local; fleet sync matches on title.
 */
export function spawnMembersOf(name: unknown, args: unknown, callId: string): AgentMember[] {
  const parsed = parseArgs(args)
  const label = (value: Record<string, unknown>, index: number): string =>
    str(value.title) || str(value.name) || str(value.description) || str(value.prompt).slice(0, 60) || `agent ${index + 1}`
  const raw = parsed.agents
  if (Array.isArray(raw) && raw.length) {
    return raw.slice(0, 24).map((item, index) => {
      const record = (item && typeof item === 'object' ? item : {}) as Record<string, unknown>
      return { key: `${callId}:${index}`, title: label(record, index), status: 'working' }
    })
  }
  if (!isSpawnTool(name)) return []
  return [{ key: `${callId}:0`, title: label(parsed, 0), status: 'working' }]
}

/** The session context menu (mockup 08), anchored at pointer coordinates. */
export interface SessionMenuState {
  readonly id: string
  /** Daemon session key — what `session.title` mutates through. */
  readonly key: string
  readonly title: string
  readonly x: number
  readonly y: number
}

export interface ReasoningLevelRow {
  readonly effort: string
  readonly description: string
}

/** Estimated split of the next request's token budget (daemon-computed). */
export interface ContextBreakdown {
  readonly systemPromptTokens: number
  readonly toolsTokens: number
  readonly messagesTokens: number
  readonly totalTokens: number
  readonly contextLimit: number
}

export interface Snapshot {
  readonly connection: Connection
  /** True before any workspace folder is chosen — the shell shows the gate. */
  readonly noWorkspace: boolean
  readonly cwd: string
  readonly model: string
  /** Agent preset fixed for the current session after its first turn. */
  readonly currentAgentPreset: string
  /** Live DSH-style preset roster used by settings and the new-task seat. */
  readonly agentPresets: readonly AgentPreset[]
  /** Git branch of the workspace, from the daemon's initialize — '' when unknown. */
  readonly branch: string
  /** Actionable initialize-handshake mismatch; null when app and daemon agree. */
  readonly daemonWarning: string | null
  /** Session cost estimate from the daemon wire (USD); null when unpriced. */
  readonly costUsd: number | null
  /** MCP server statuses from the daemon; empty until fetched or without servers. */
  readonly mcpStatus: Readonly<Record<string, McpServerStatus>>
  readonly models: readonly ModelChoice[]
  readonly contextTokens: number | null
  readonly contextMax: number | null
  /** Session-average provider time to first output token. */
  readonly ttftMs: number | null
  /** Latest provider round's output decode rate. */
  readonly tokensPerSecond: number | null
  /** Cumulative provider and tool telemetry for the bound session. */
  readonly llmDurationMs: number
  readonly llmSteps: number
  readonly toolDurationMs: number
  readonly toolSteps: number
  readonly inputTokens: number
  readonly outputTokens: number
  /** Current live phase, used to advance the active timing bucket. */
  readonly metricPhase: 'llm' | 'tool' | null
  readonly metricPhaseStartedAt: number | null
  /** Latest provider-reported input-cache hit share, from 0 through 1. */
  readonly cacheHitRate: number | null
  readonly sessions: readonly SessionRow[]
  readonly live: readonly SessionRow[]
  readonly fleet: readonly SessionRow[]
  /** Repeatable workflows the runtime observed and proposed as skills. */
  readonly skillSuggestions: readonly SkillSuggestion[]
  /** Policy-gated declarative creator actions for the bound session. */
  readonly creatorTrace: readonly CreatorTrace[]
  /** Daemon-backgrounded turns (bg-* sessions) currently working. */
  readonly backgroundJobs: readonly BackgroundJob[]
  readonly currentId: string
  readonly currentTitle: string
  /** The connection's bound daemon session key — what key-scoped RPCs address. */
  readonly sessionKey: string
  readonly goal: string
  readonly approval: Approval | null
  readonly question: import('./types.js').TaskQuestion | null
  readonly planMode: boolean
  readonly turnActive: boolean
  readonly turnFailed: boolean
  readonly turnSeconds: number
  readonly blocks: readonly Block[]
  readonly error: string | null
  // ── workspace surfaces ──
  readonly tab: WorkspaceTab
  readonly turnCount: number
  readonly queue: readonly QueueItem[]
  readonly changes: readonly DiffFile[]
  readonly changesKept: boolean
  readonly plan: PlanState | null
  readonly log: readonly LogEntry[]
  readonly failed: FailedTurn | null
  // ── overlays ──
  readonly settingsOpen: boolean
  readonly settingsTab: SettingsTab
  readonly paletteOpen: boolean
  readonly pickerOpen: boolean
  /** The single model/effort chip's dropdown — rows drill into the pickers. */
  readonly modelMenuOpen: boolean
  /** Context-usage popover with the estimated token split. */
  readonly contextMenuOpen: boolean
  readonly contextBreakdown: ContextBreakdown | null
  readonly contextBreakdownLoading: boolean
  /** Reasoning-effort picker state — levels come from the daemon per model. */
  readonly reasoningPickerOpen: boolean
  readonly reasoningEffort: string
  readonly reasoningLevels: readonly ReasoningLevelRow[]
  readonly reasoningDefault: string
  readonly reasoningNote: string
  readonly reasoningLoading: boolean
  readonly wsMenuOpen: boolean
  readonly sessionMenu: SessionMenuState | null
  /** Display choice: show reasoning trails in the activity feed. */
  readonly streamThinking: boolean
  /** The ⌘N new-task modal (mockup 18). */
  readonly taskModalOpen: boolean
  // ── settings data ──
  readonly providers: readonly ProviderRow[]
  /** Model catalogs and editable capacities cached for each provider profile. */
  readonly providerModels: Readonly<Record<string, readonly CachedModel[]>>
  readonly providerModelLoading: readonly string[]
  readonly providerModelWarnings: Readonly<Record<string, string>>
  /** The daemon registry's adapter catalog — the add/edit form's dropdown. */
  readonly providerTypes: readonly ProviderTypeRow[]
  /** The daemon's slash catalog — name/description pairs from commands.catalog. */
  readonly commands: readonly { readonly name: string; readonly description: string }[]
  readonly permissionMode: string
  readonly snippets: Readonly<Record<string, string>>
}

const str = (value: unknown): string => (typeof value === 'string' ? value : '')

const num = (value: unknown): number | null => (typeof value === 'number' && Number.isFinite(value) ? value : null)

function cachedModelsFromResult(result: Record<string, unknown>): CachedModel[] {
  const catalog = Array.isArray(result.catalog) ? result.catalog : []
  const byId = new Map<string, CachedModel>()
  for (const value of catalog) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) continue
    const record = value as Record<string, unknown>
    const id = str(record.id).trim()
    if (!id) continue
    const contextLimit = num(record.context_limit)
    const maxOutputTokens = num(record.max_output_tokens)
    const contextSource = capabilitySource(record.context_source)
    const outputSource = capabilitySource(record.output_source)
    byId.set(id, {
      id,
      ...(contextLimit === null ? {} : { contextLimit }),
      ...(contextSource === undefined ? {} : { contextSource }),
      ...(maxOutputTokens === null ? {} : { maxOutputTokens }),
      ...(outputSource === undefined ? {} : { outputSource }),
      overridden: record.overridden === true,
    })
  }
  const ids = Array.isArray(result.models)
    ? result.models.map(value => str(value).trim()).filter(Boolean)
    : []
  for (const id of ids) {
    if (!byId.has(id)) byId.set(id, { id, overridden: false })
  }
  return [...byId.values()]
}

function agentPresetsFromResult(result: Record<string, unknown>): AgentPreset[] {
  const rows = Array.isArray(result.presets) ? result.presets : []
  return rows.flatMap(value => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return []
    const row = value as Record<string, unknown>
    const id = str(row.id).trim()
    if (!id) return []
    const trustValue = str(row.trust)
    const trust: AgentPreset['trust'] = trustValue === 'system' || trustValue === 'project' ? trustValue : 'user'
    return [{
      id,
      name: str(row.name) || id,
      description: str(row.description),
      trust,
      isDefault: row.is_default === true,
      manageable: row.manageable === true,
      ...(str(row.broken) ? { broken: str(row.broken) } : {}),
    }]
  })
}

function capabilitySource(
  value: unknown,
): CachedModel['contextSource'] | undefined {
  return value === 'catalog' || value === 'override' || value === 'provider' || value === 'unknown'
    ? value
    : undefined
}

function clientHandshake(): Record<string, unknown> {
  return {
    client_version: DESKTOP_VERSION,
    client_protocol: DESKTOP_DAEMON_PROTOCOL,
    ...(EXPECTED_DAEMON_BUILD_ID
      ? { expected_daemon_build_id: EXPECTED_DAEMON_BUILD_ID }
      : {}),
  }
}

function ageOf(when: unknown): string {
  // Saved rows carry updated_at as an ISO string; live rows carry
  // last_active as epoch seconds. Accept both or show nothing rather than
  // mislabel.
  let epoch = typeof when === 'number' && Number.isFinite(when) ? when : NaN
  if (typeof when === 'string') {
    const parsed = Date.parse(when)
    if (!Number.isNaN(parsed)) epoch = parsed / 1000
  }
  if (Number.isNaN(epoch)) return ''
  const minutes = Math.floor((Date.now() - epoch * 1000) / 60_000)
  if (minutes < 1) return 'now'
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h`
  return `${Math.floor(hours / 24)}d`
}

/** Display provider for a model id: `z-ai/glm-5.2` → `z-ai`, else family. */
export function providerOf(id: string): string {
  const slashed = id.split('/')[0]
  if (slashed && id.includes('/')) return slashed
  const lower = id.toLowerCase()
  if (/^(gpt|o1|o3|o4|codex|chatgpt)/.test(lower)) return 'openai'
  if (lower.includes('claude')) return 'anthropic'
  if (lower.includes('kimi') || lower.includes('moonshot')) return 'kimi'
  if (lower.includes('glm') || lower.includes('z-ai')) return 'z-ai'
  if (lower.includes('deepseek')) return 'deepseek'
  if (lower.includes('qwen')) return 'qwen'
  if (lower.includes('gemini')) return 'google'
  if (lower.includes('llama')) return 'meta'
  if (lower.includes('mistral') || lower.includes('codestral')) return 'mistral'
  return 'models'
}

function toChoices(ids: readonly string[]): ModelChoice[] {
  return ids.filter(Boolean).map(id => ({ id, provider: providerOf(id) }))
}

function skillSuggestionOf(value: unknown): SkillSuggestion | null {
  const row = value && typeof value === 'object' && !Array.isArray(value)
    ? value as Readonly<Record<string, unknown>>
    : {}
  const skillName = str(row.skill_name ?? row.skillName)
  if (!skillName) return null
  const rawTools = row.unique_tools ?? row.uniqueTools
  return {
    skillName,
    description: str(row.description),
    version: str(row.version),
    sourcePath: str(row.source_path ?? row.sourcePath),
    toolCount: num(row.tool_count ?? row.toolCount) ?? 0,
    uniqueTools: Array.isArray(rawTools) ? rawTools.map(tool => str(tool)).filter(Boolean) : [],
  }
}

function creatorTraceOf(value: unknown): CreatorTrace | null {
  const row = value && typeof value === 'object' && !Array.isArray(value)
    ? value as Readonly<Record<string, unknown>>
    : {}
  const action = str(row.action)
  const status = row.status === 'error' ? 'error' : row.status === 'ok' ? 'ok' : null
  if (!action || !status) return null
  return {
    action,
    status,
    name: str(row.name),
    version: str(row.version),
    detail: str(row.detail),
    at: str(row.at),
  }
}

const wireRow = (row: Record<string, unknown>, currentId: string): SessionRow | null => {
  const id = str(row.id ?? row.session_id ?? row.key)
  if (!id) return null
  const count = num(row.message_count) ?? num(row.messages) ?? 0
  const turns = num(row.turn_count) ?? 0
  const kind = str(row.kind ?? row.session_kind) === 'subagent' ? 'subagent' : 'main'
  // Untitled rows get the short id — never a wall of 'Untitled'. The lazy
  // enrichment below replaces it with a first-message snippet whenever the
  // session is still loaded in the daemon and can answer session.status.
  const hasTitle = Boolean(str(row.title))
  return {
    id,
    key: str(row.key) || id,
    title: str(row.title) || `#${id.slice(0, 6)}`,
    status: str(row.status) || '',
    age: ageOf(row.updated_at ?? row.last_active),
    current: id === currentId,
    kind,
    turns,
    messages: count,
    cwd: str(row.cwd),
    untitled: !hasTitle,
  } satisfies SessionRow
}

interface SavedWire {
  id?: unknown
  session_id?: unknown
  key?: unknown
  title?: unknown
  status?: unknown
  updated_at?: unknown
  last_active?: unknown
  message_count?: unknown
  messages?: unknown
  kind?: unknown
  session_kind?: unknown
  turn_count?: unknown
  model?: unknown
  cwd?: unknown
}

function normalize(rows: unknown, currentId: string): SessionRow[] {
  if (!Array.isArray(rows)) return []
  const out: SessionRow[] = []
  for (const raw of rows) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) continue
    const row = raw as SavedWire
    const mapped = wireRow(row as Record<string, unknown>, currentId)
    if (mapped) out.push(mapped)
  }
  return out
}

/** Parse `- [ ]` / `- [x]` checklist items out of plan markdown. */
function planItemsOf(markdown: string): PlanItem[] {
  const items: PlanItem[] = []
  for (const line of markdown.split('\n')) {
    const match = /^\s*[-*]\s+\[([ xX])]\s*(.+)$/.exec(line)
    if (match) items.push({ done: match[1]!.toLowerCase() === 'x', text: match[2]!.trim() })
  }
  return items
}

/** Does this batched question look like a plan review (approve + revise)? */
export function isPlanReview(question: import('./types.js').TaskQuestion): boolean {
  const haystack = [question.items.map(item => `${item.question} ${item.options.join(' ')}`).join(' ')].join(' ').toLowerCase()
  return haystack.includes('plan') && (haystack.includes('approve') || haystack.includes('accept'))
}

const LOG_CAP = 400
const SNIPPET_CAP = 64
const HEARTBEAT_MS = 5_000

/** Compact one-line summary of an event for the Log tab. */
function summarize(type: string, payload: Readonly<Record<string, unknown>>): string {
  const pieces: string[] = []
  for (const key of ['model', 'text', 'think', 'title', 'body', 'message', 'error', 'name', 'status', 'plan_mode', 'context_tokens', 'duration_ms', 'tool_name', 'action']) {
    const value = payload[key]
    if (typeof value === 'string' && value) pieces.push(`${key}=${value.slice(0, 120)}`)
    else if (typeof value === 'number' || typeof value === 'boolean') pieces.push(`${key}=${String(value)}`)
  }
  return pieces.length ? pieces.join(' ') : '{}'
}

export class Store {
  private listeners = new Set<() => void>()
  private frame: Snapshot
  private builder = new BlockBuilder()
  private tick: NodeJS.Timeout | null = null
  private heartbeat: NodeJS.Timeout | null = null
  /** In-flight agent-family tool calls; the fleet poll lives while any do. */
  private readonly pendingAgentCalls = new Set<string>()
  private fleetPoll: NodeJS.Timeout | null = null
  /** The turn's agents-card members by local key — merged into the builder. */
  private readonly agentMembers = new Map<string, AgentMember>()
  /** Titles the current card shows, for fleet-snapshot status matching. */
  private readonly agentMemberKeysByTitle = new Map<string, string>()
  private fleetPollRounds = 0
  private started = false
  private sessionKey = `desktop-${Math.random().toString(36).slice(2, 10)}`
  private unsubEvents: (() => void) | null = null

  // ── workspace folds (not part of the transcript) ──
  private queue: QueueItem[] = []
  private changes = new Map<string, DiffFile>()
  private logRing: LogEntry[] = []
  private planState: PlanState | null = null
  private failure: FailedTurn | null = null
  private turnCount = 0
  private seq = 1
  private turnError: string | null = null
  private agentText = ''
  private lastUser = ''
  private snippets: Record<string, string> = {}
  private enriching = new Set<string>()
  private ttftTotalMs = 0
  private ttftSamples = 0
  private activeMetricTools = new Set<string>()

  constructor() {
    this.frame = this.frozen({
      connection: 'connecting',
      noWorkspace: false,
      cwd: '',
      model: '',
      currentAgentPreset: 'default',
      agentPresets: [],
      branch: '',
      daemonWarning: null,
      costUsd: null,
      mcpStatus: {},
      models: [],
      contextTokens: null,
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
      sessions: [],
      live: [],
      fleet: [],
      skillSuggestions: [],
      creatorTrace: [],
      backgroundJobs: [],
      currentId: '',
      currentTitle: '',
      sessionKey: '',
      goal: '',
      approval: null,
      question: null,
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
      plan: null,
      log: [],
      failed: null,
      settingsOpen: false,
      settingsTab: 'general',
      paletteOpen: false,
      pickerOpen: false,
      modelMenuOpen: false,
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
      sessionMenu: null,
      streamThinking: readStreamThinking(),
      taskModalOpen: false,
      providers: [],
      providerModels: {},
      providerModelLoading: [],
      providerModelWarnings: {},
      providerTypes: [],
      commands: [],
      permissionMode: '',
      snippets: {},
    })
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  getSnapshot = (): Snapshot => this.frame

  // ── Actions ──────────────────────────────────────────────────────────

  start(bridge?: XerxesLike): void {
    if (this.started || typeof window === 'undefined') return
    this.started = true
    if (bridge) this.bridge = bridge
    this.unsubEvents = window.xerxes.onEvent(event => this.onEvent(event))
    // Workspace gate: no folder, no daemon, no initialize — the shell asks
    // for one instead of inventing a target the user never chose.
    const gate = (this.bridge as XerxesLike & { getWorkspace?: () => Promise<string | null> })
      .getWorkspace?.()
    void Promise.resolve(gate).then(saved => {
      if (saved === null) {
        this.patch({ noWorkspace: true, connection: 'offline' })
        return
      }
      this.initializeLive()
    })
  }

  private initializeLive(): void {
    void this.initializeSelfHealing().then(
      () => {
        void this.refreshGoal()
        void this.refreshSessions()
      },
      () => this.wentOffline(),
    )
    this.heartbeat = setInterval(() => void this.beat(), HEARTBEAT_MS)
    this.heartbeat.unref?.()
  }

  /**
   * Initialize, but survive a daemon whose memory for our session key
   * disagrees with disk (crash, restored backup, externally removed
   * transcripts). The daemon answers such binds with a transcript_generation
   * conflict; retrying under a brand-new key sidesteps the poisoned binding
   * instead of leaving the shell stuck on "Daemon offline" forever.
   */
  private async initializeSelfHealing(extra: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    try {
      return await this.initialize(extra)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      if (!/transcript_generation|divergent append/i.test(message)) throw error
      // A bare initialize with a fresh key — resuming would just re-bind the
      // poisoned session. Old chats stay on disk and in the sidebar.
      this.sessionKey = `desktop-${Math.random().toString(36).slice(2, 10)}`
      this.patch({ currentId: '', sessionKey: this.sessionKey })
      this.builder.reset()
      return this.initialize({})
    }
  }

  /**
   * Submit or steer. While a turn runs, text goes to `turn.steer` — the
   * daemon queues it between steps and the store mirrors it visibly; when
   * idle it is a normal `turn.submit`.
   */
  async submit(text: string): Promise<void> {
    const trimmed = text.trim()
    if (!trimmed) return
    if (this.frame.turnActive && !trimmed.startsWith('/')) {
      await this.steer(trimmed)
      return
    }
    if (!trimmed.startsWith('/')) {
      this.lastUser = trimmed
      this.builder.pushUser(trimmed)
      this.notify()
      try {
        await this.bridge.call('turn.submit', { session_key: this.sessionKey, text: trimmed })
        this.cameOnline()
      } catch (error) {
        // The daemon rejected the submit (e.g. no provider configured): the
        // optimistic bubble never happened — roll it back rather than leave
        // a delivered-looking ghost above the error.
        this.builder.rollbackUser(trimmed)
        this.notify()
        this.fail(error)
      }
      return
    }
    // /goal routes to its dedicated durable-state RPC; everything else goes
    // to the daemon's native slash table, which owns rejection.
    if (trimmed === '/goal' || trimmed.startsWith('/goal ')) {
      const result = await this.setGoal(trimmed.slice(5).trim())
      if (!result.ok) this.fail(new Error(result.text))
      return
    }
    if (trimmed === '/plan' || trimmed.startsWith('/plan ') || trimmed === '/plan off') {
      // /plan steers planning; the mode flip itself is the RPC's job.
      const rest = trimmed.slice(5).trim()
      if (!rest || rest === 'off') this.setPlanMode(!this.frame.planMode)
      else await this.steer(rest)
      return
    }
    try {
      const result = await this.bridge.call('slash', { command: trimmed })
      const message = str(result.output) || str(result.error) || (result.ok === true ? 'ok' : 'command failed')
      this.builder.push('notification', { severity: result.ok === true ? 'info' : 'error', message })
      this.notify()
      if (trimmed.startsWith('/permissions')) void this.loadProviders()
      void this.refreshSessions()
    } catch (error) {
      this.fail(error)
    }
  }

  /** Queue steering text daemon-side; mirror it locally until consumed. */
  async steer(text: string): Promise<void> {
    const cleaned = text.trim()
    if (!cleaned) return
    try {
      const result = await this.bridge.call('turn.steer', { session_key: this.sessionKey, content: cleaned })
      if (result.ok === true) {
        this.queue = [...this.queue, { id: this.seq++, text: cleaned }]
        this.patch({ queue: this.queue })
      } else {
        this.fail(new Error(str(result.error) || 'steering refused'))
      }
    } catch (error) {
      this.fail(error)
    }
  }

  dropQueued(id: number): void {
    this.queue = this.queue.filter(item => item.id !== id)
    this.patch({ queue: this.queue })
  }

  cancel(): void {
    void this.bridge
      .call('turn.cancel', { session_key: this.sessionKey })
      .catch(error => this.fail(error))
  }

  /** Ask the attached process to exit; DaemonRpc reconnects and launches this build. */
  restartDaemon(): void {
    this.patch({ connection: 'connecting' })
    void this.bridge.call('slash', { command: '/restart' }).catch(error => this.fail(error))
  }

  async openSession(id: string): Promise<void> {
    if (this.frame.turnActive) return
    try {
      // A resume re-keys the connection, but NOT to a string of our own
      // choosing: the daemon binds a resumed session under the session id
      // (resume_session_id wins over session_key — see its 'ignored-slot'
      // contract test), and every later submit/steer/cancel sends the key
      // explicitly. Adopt the key the daemon actually bound, or the next
      // message silently lands in a fresh, context-free session.
      const result = await this.bridge.call('initialize', {
        session_key: `${this.sessionKey}-r${id.slice(-8)}`,
        resume_session_id: id,
        ...clientHandshake(),
      })
      const session = this.sessionOf(result)
      this.sessionKey = str(session.key) || str(result.session_id) || id
      // Replay leans on the record's tool_executions + thinking_content so a
      // reopened chat shows the same think → tool rows the live stream did.
      this.builder.reset(
        blocksFromStoredMessages(session.transcript ?? session.messages, {
          executions: session.tool_executions,
          thinking: session.thinking_content,
        }),
      )
      this.resetWorkspaceFolds()
      this.patch({
        currentId: str(result.session_id ?? session.id ?? id),
        currentTitle: str(session.title),
        currentAgentPreset: str(session.agent_id) || this.frame.currentAgentPreset,
        sessionKey: this.sessionKey,
        daemonWarning: daemonCompatibilityWarning(result),
        planMode: session.plan_mode === true,
        approval: null,
        ...this.telemetryFromSession(session),
      })
      this.adoptFleet(session)
      this.loadSkillSuggestions()
      this.loadCreatorTrace()
      void this.refreshGoal()
      void this.refreshSessions()
    } catch (error) {
      this.fail(error)
    }
  }

  newChat(): void {
    void this.beginFreshTask()
  }

  /** Start an unmistakable blank Creator mode session from the shell chrome. */
  startCreatorMode(): void {
    void this.beginFreshTask('creator')
  }

  /**
   * Re-key and rebind. The daemon keys sessions by this string; reusing it
   * would reopen the previous conversation under an empty transcript. A
   * fresh task is a fresh key. Resolves true once the daemon has opened the
   * fresh session, so follow-up calls (plan ceiling, first submit) address
   * a session that exists; false means the rebind failed and the shell
   * went offline.
   */
  private beginFreshTask(agentPreset?: string): Promise<boolean> {
    if (this.frame.turnActive || this.frame.connection !== 'online') return Promise.resolve(false)
    this.sessionKey = `desktop-${Math.random().toString(36).slice(2, 10)}`
    this.builder.reset()
    this.resetWorkspaceFolds()
    this.patch({ currentId: '', currentTitle: '', goal: '', approval: null, failed: null, sessionKey: this.sessionKey })
    return this.initialize(agentPreset ? { agent_id: agentPreset } : {})
      .then(() => {
        void this.refreshGoal()
        void this.refreshSessions()
        return true
      })
      .catch(() => {
        this.wentOffline()
        return false
      })
  }

  approve(requestId: string, response: ApprovalResponse): void {
    const card = this.frame.approval
    // The daemon's vocabulary is approve / approve_for_session / reject —
    // anything else resolves as reject, so map our UI labels here.
    const wire = response === 'allow_once' ? 'approve'
      : response === 'allow_session' ? 'approve_for_session'
      : 'reject'
    void this.bridge
      .call('permission_response', { request_id: requestId, response: wire })
      .then(result => {
        // Clear only on confirmation: a refused response (unknown id, another
        // connection owns it) leaves the request pending daemon-side, and
        // dropping the card would strand it with no surface to answer.
        if (result.ok === false) {
          if (card?.id === requestId) this.patch({ approval: card })
          this.builder.push('notification', { severity: 'error', message: str(result.error) || 'approval refused' })
          this.notify()
        } else if (card?.id === requestId) {
          this.patch({ approval: null })
        }
      })
      .catch(error => {
        if (card?.id === requestId) this.patch({ approval: card })
        this.fail(error)
      })
  }

  answerQuestion(requestId: string, answers: Record<string, string>): void {
    const card = this.frame.question
    void this.bridge
      .call('question_response', { request_id: requestId, answers })
      .then(result => {
        if (result.ok === false) {
          if (card?.requestId === requestId) this.patch({ question: card })
          this.builder.push('notification', { severity: 'error', message: str(result.error) || 'answer refused' })
          this.notify()
        } else if (card?.requestId === requestId) {
          this.patch({ question: null })
        }
      })
      .catch(error => {
        if (card?.requestId === requestId) this.patch({ question: card })
        this.fail(error)
      })
  }

  loadModels(force = false): void {
    if (!force && this.frame.models.length) return
    void this.bridge
      .call('fetch_models', {})
      .then(result => {
        if (result.ok === false) {
          // 'no profile', 'provider refused' and 'offline' are not the same
          // state as 'zero models' — say which one happened.
          this.builder.push('notification', {
            severity: 'error',
            message: str(result.error) || 'model discovery failed',
          })
          this.notify()
          return
        }
        const ids = Array.isArray(result.models) ? (result.models as unknown[]).map(m => str(m)).filter(Boolean) : []
        this.patch({ models: toChoices(ids) })
      })
      .catch(error => this.fail(error))
  }

  /** Discover one saved profile's catalog without activating that profile. */
  loadProviderModels(profileName: string, force = false): void {
    const name = profileName.trim()
    if (!name || this.frame.providerModelLoading.includes(name)) return
    if (!force && this.frame.providerModels[name]) return
    this.patch({ providerModelLoading: [...this.frame.providerModelLoading, name] })
    void this.bridge.call('provider_models', { profile_name: name }).then(result => {
      const loading = this.frame.providerModelLoading.filter(entry => entry !== name)
      if (result.ok === false) {
        this.patch({
          providerModelLoading: loading,
          providerModelWarnings: {
            ...this.frame.providerModelWarnings,
            [name]: str(result.error) || 'model discovery failed',
          },
        })
        return
      }
      const models = cachedModelsFromResult(result)
      const warnings = { ...this.frame.providerModelWarnings }
      const warning = str(result.warning)
      if (warning) warnings[name] = warning
      else delete warnings[name]
      this.patch({
        providerModelLoading: loading,
        providerModels: { ...this.frame.providerModels, [name]: models },
        providerModelWarnings: warnings,
      })
    }).catch(error => {
      const message = error instanceof Error ? error.message : String(error)
      this.patch({
        providerModelLoading: this.frame.providerModelLoading.filter(entry => entry !== name),
        providerModelWarnings: { ...this.frame.providerModelWarnings, [name]: message },
      })
    })
  }

  /** Persist or clear one cached model's input/output capacity overrides. */
  saveModelCapabilities(
    profileName: string,
    model: string,
    contextLimit: number | null,
    maxOutputTokens: number | null,
  ): void {
    const name = profileName.trim()
    const id = model.trim()
    if (!name || !id) return
    void this.bridge.call('provider_model_override', {
      profile_name: name,
      model: id,
      context_limit: contextLimit,
      max_output_tokens: maxOutputTokens,
    }).then(result => {
      if (result.ok === false) {
        this.builder.push('notification', {
          severity: 'error',
          message: str(result.error) || 'model capacity update refused',
        })
        this.notify()
        return
      }
      this.loadProviderModels(name, true)
      if (this.frame.providers.some(provider => provider.name === name && provider.active)) {
        this.loadModels(true)
      }
      this.builder.push('notification', {
        severity: 'info',
        message: `model capacities updated for \`${id}\``,
      })
      this.notify()
    }).catch(error => this.fail(error))
  }

  /**
   * Upsert a provider profile. The daemon's provider_save persists it to
   * ~/.xerxes/profiles.json AND makes it the active profile (its runtime
   * reloads onto the new credentials), so the session re-initializes onto
   * the saved model. Refused mid-turn for the same reason as switching.
   */
  saveProvider(profile: {
    name: string
    baseUrl: string
    model: string
    provider?: string
    apiKey?: string
  }): void {
    const name = profile.name.trim()
    const baseUrl = profile.baseUrl.trim()
    const model = profile.model.trim()
    const provider = profile.provider?.trim()
    // A registry-known type supplies its default endpoint daemon-side;
    // "Provider default" (blank) is only valid then.
    const knownDefault = this.frame.providerTypes.find(t => t.name === provider)?.baseUrl ?? ''
    if (!name || !model || (!baseUrl && !knownDefault)) {
      this.builder.push('notification', {
        severity: 'error',
        message: knownDefault
          ? 'name and model are required'
          : 'name, base_url, and model are required',
      })
      this.notify()
      return
    }
    if (this.frame.turnActive) return
    const params: Record<string, unknown> = { name, model }
    if (baseUrl) params.base_url = baseUrl
    if (provider) params.provider = provider
    if (profile.apiKey?.trim()) params.api_key = profile.apiKey.trim()
    void this.bridge
      .call('provider_save', params)
      .then(async result => {
        if (result.ok === false) {
          this.builder.push('notification', {
            severity: 'error',
            message: str(result.error) || 'provider save refused',
          })
          this.notify()
          return
        }
        const saved = (result.profile && typeof result.profile === 'object'
          ? result.profile
          : {}) as Record<string, unknown>
        // Refresh the surfaces FIRST: the daemon has already moved onto the
        // new profile, and if the re-initialize below fails (e.g. the saved
        // key is rejected), a stale list would hide what actually happened.
        void this.loadProviders()
        this.loadModels(true)
        this.builder.push('notification', {
          severity: 'info',
          message: `provider \`${name}\` saved and activated`,
        })
        this.notify()
        const extra: Record<string, unknown> = { model: str(saved.model) || model }
        if (this.frame.currentId) extra.resume_session_id = this.frame.currentId
        try {
          await this.initialize(extra)
        } catch {
          // Already surfaced by initialize's caller contract: the profile
          // list above shows the truth; the chip catches up on recovery.
        }
      })
      .catch(error => this.fail(error))
  }

  /** Delete a saved profile. The active one must be switched away from first. */
  deleteProvider(name: string): void {
    const row = this.frame.providers.find(entry => entry.name === name)
    if (!row || row.active || this.frame.turnActive) return
    void this.bridge
      .call('provider_delete', { name })
      .then(result => {
        if (result.ok === false) {
          this.builder.push('notification', {
            severity: 'error',
            message: str(result.error) || 'provider delete refused',
          })
          this.notify()
          return
        }
        this.builder.push('notification', {
          severity: 'info',
          message: `provider \`${name}\` deleted`,
        })
        this.notify()
        void this.loadProviders()
        this.loadModels(true)
      })
      .catch(error => this.fail(error))
  }

  /** The daemon's own slash catalog — every TUI command, discoverable here. */
  loadCommands(): void {
    if (this.frame.commands.length) return
    void this.bridge
      .call('commands.catalog', {})
      .then(result => {
        const pairs = Array.isArray(result.pairs) ? result.pairs : []
        const commands = pairs
          .map(pair => {
            const entry = (Array.isArray(pair) ? pair : []) as unknown[]
            const name = str(entry[0]).replace(/^\//, '')
            return name ? { name, description: str(entry[1]) } : null
          })
          .filter((entry): entry is { name: string; description: string } => entry !== null)
        this.patch({ commands })
      })
      .catch(() => {})
  }

  /**
   * Live daemon completions for a partial draft: slash-command prefixes and
   * `/skill <name>` references (the same registry `/skills` lists). The one
   * completion source of truth — the TUI's — not a GUI-local copy.
   */
  async completeText(text: string): Promise<{ value: string; label: string; meta: string }[]> {
    const result = await this.bridge.call('complete', { text })
    const completions = Array.isArray(result.completions) ? result.completions : []
    return completions
      .map(raw => {
        const entry = (raw ?? {}) as Record<string, unknown>
        const value = str(entry.value)
        return { value, label: str(entry.label) || value, meta: str(entry.meta) }
      })
      .filter(entry => entry.value)
  }

  /**
   * Ask the shell for a workspace folder. On a pick the shell retargets its
   * daemon and reloads this window — the renderer's job ends at the request.
   */
  chooseWorkspace(): void {
    void this.bridge.chooseWorkspace?.().catch(() => {})
  }

  /** Enter a known workspace folder (sidebar header, switcher menu) — retargets the daemon. */
  enterWorkspace(cwd: string): void {
    if (!cwd) return
    this.patch({ wsMenuOpen: false })
    void (this.bridge as XerxesLike & { useWorkspace?: (dir: string) => Promise<unknown> })
      .useWorkspace?.(cwd)
      .catch(() => {})
  }

  pickModel(modelId: string): void {
    if (!modelId) return
    // Same mid-turn refusal as provider switching: hot-swapping the model a
    // running turn is riding on is refused daemon-side too.
    if (this.frame.turnActive) return
    // Route through the daemon's /model handler: it pins the choice to this
    // session AND persists it as the active profile's model. The old
    // initialize({ model }) path only reloaded runtime memory, so every
    // daemon restart silently fell back to the profile's stored model.
    void this.bridge.call('slash', { command: `/model ${modelId}` }).then(result => {
      if (result.ok === false) {
        this.builder.push('notification', {
          severity: 'error',
          message: str(result.error) || 'model change rejected',
        })
        this.notify()
        return
      }
      const applied = str(result.model)
      if (applied) this.patch({ model: applied })
    }).catch(error => this.fail(error))
  }

  /**
   * Switch the daemon's active provider profile and adopt its model here.
   * Refused mid-turn: provider_select swaps the live credentials the
   * in-flight request is already riding on.
   */
  selectProvider(name: string): void {
    if (this.frame.turnActive) return
    const target = this.frame.providers.find(row => row.name === name)
    if (!target || target.active) return
    void this.bridge
      .call('provider_select', { name })
      .then(async result => {
        if (result.ok === false) return
        // Refresh before re-initializing: a failed initialize (profile key
        // rejected, session gone) must not leave the list claiming the old
        // profile is still active — the daemon already switched.
        void this.loadProviders()
        this.loadModels(true)
        const extra: Record<string, unknown> = target.model ? { model: target.model } : {}
        if (this.frame.currentId) extra.resume_session_id = this.frame.currentId
        try {
          await this.initialize(extra)
        } catch {
          // The refreshed provider list already shows the truth.
        }
      })
      .catch(error => this.fail(error))
  }

  /** Toggle the session's plan mode on the daemon (plan = read-only ceiling). */
  setPlanMode(next: boolean): void {
    void this.bridge
      .call('set_plan_mode', { session_key: this.sessionKey, enabled: next })
      .then(result => {
        if (result.ok === false) {
          // Refused (e.g. the daemon restarted and lost the session): flip
          // nothing locally — the chip must not claim a ceiling the daemon
          // never armed.
          this.fail(new Error(str(result.error) || 'plan mode refused'))
          return
        }
        // The daemon answers before the status echo; apply optimistically,
        // the next status_update.plan_mode is authoritative either way.
        this.patch({ planMode: next })
      })
      .catch(error => this.fail(error))
  }

  togglePlanMode(): void {
    this.setPlanMode(!this.frame.planMode)
  }

  // ── Overlay + tab actions ────────────────────────────────────────────

  setTab(tab: WorkspaceTab): void {
    this.patch({ tab })
  }

  openSettings(tab?: SettingsTab): void {
    this.patch({ settingsOpen: true, paletteOpen: false, ...(tab ? { settingsTab: tab } : {}) })
    this.loadModels()
    void this.loadProviders()
    void this.loadAgentPresets()
  }

  closeSettings(): void {
    this.patch({ settingsOpen: false })
  }

  setSettingsTab(tab: SettingsTab): void {
    this.patch({ settingsTab: tab })
    if (tab === 'agents') void this.loadAgentPresets()
    if (tab === 'models') {
      this.loadModels()
      void this.loadProviders()
    }
  }

  togglePalette(): void {
    this.patch({ paletteOpen: !this.frame.paletteOpen })
    this.loadModels()
    this.loadCommands()
  }

  closePalette(): void {
    this.patch({ paletteOpen: false })
  }

  openPicker(): void {
    this.patch({ pickerOpen: true, settingsOpen: false, paletteOpen: false, modelMenuOpen: false })
    this.loadModels()
  }

  closePicker(): void {
    this.patch({ pickerOpen: false })
  }

  /** Reasoning levels are asked of the daemon per open — they differ per model. */
  openReasoningPicker(): void {
    this.patch({
      reasoningPickerOpen: true,
      settingsOpen: false,
      paletteOpen: false,
      pickerOpen: false,
      modelMenuOpen: false,
      wsMenuOpen: false,
      reasoningLoading: true,
    })
    void this.bridge.call('reasoning_levels', {}).then(result => {
      if (result.ok === false) {
        this.builder.push('notification', {
          severity: 'error',
          message: str(result.error) || 'reasoning levels unavailable',
        })
        this.patch({ reasoningLoading: false })
        this.notify()
        return
      }
      const levels = Array.isArray(result.levels)
        ? (result.levels as unknown[]).flatMap(entry => {
            if (typeof entry !== 'object' || entry === null) return []
            const row = entry as Record<string, unknown>
            const effort = str(row.effort)
            return effort ? [{ effort, description: str(row.description) }] : []
          })
        : []
      const current = str(result.current)
      this.patch({
        reasoningLoading: false,
        reasoningLevels: levels,
        reasoningDefault: str(result.default),
        reasoningNote: str(result.note),
        ...(current ? { reasoningEffort: current } : {}),
      })
    }).catch(error => {
      this.patch({ reasoningLoading: false })
      this.fail(error)
    })
  }

  closeReasoningPicker(): void {
    this.patch({ reasoningPickerOpen: false })
  }

  toggleReasoningPicker(): void {
    if (this.frame.reasoningPickerOpen) {
      this.closeReasoningPicker()
    } else {
      this.openReasoningPicker()
    }
  }

  /** Selection rides the daemon's /thinking handler so session pinning applies. */
  pickReasoning(effort: string): void {
    this.patch({ reasoningPickerOpen: false })
    const trimmed = effort.trim()
    if (!trimmed) return
    void this.bridge.call('slash', { command: `/thinking ${trimmed}` }).then(result => {
      if (result.ok === false) {
        this.builder.push('notification', {
          severity: 'error',
          message: str(result.error) || 'reasoning effort rejected',
        })
        this.notify()
        return
      }
      const applied = str(result.reasoning_effort)
      if (applied) this.patch({ reasoningEffort: applied })
    }).catch(error => this.fail(error))
  }

  togglePicker(): void {
    this.patch({ pickerOpen: !this.frame.pickerOpen, modelMenuOpen: false })
    this.loadModels()
  }

  /** The combined chip's dropdown; rows drill into the two pickers. */
  toggleModelMenu(): void {
    this.patch({
      modelMenuOpen: !this.frame.modelMenuOpen,
      pickerOpen: false,
      reasoningPickerOpen: false,
      paletteOpen: false,
      wsMenuOpen: false,
      contextMenuOpen: false,
    })
  }

  closeModelMenu(): void {
    this.patch({ modelMenuOpen: false })
  }

  /** Context popover: the daemon estimates the split on open, never cached. */
  toggleContextMenu(): void {
    if (this.frame.contextMenuOpen) {
      this.patch({ contextMenuOpen: false })
      return
    }
    this.patch({
      contextMenuOpen: true,
      modelMenuOpen: false,
      pickerOpen: false,
      reasoningPickerOpen: false,
      paletteOpen: false,
      wsMenuOpen: false,
      contextBreakdownLoading: true,
    })
    void this.bridge.call('context_breakdown', { session_key: this.sessionKey }).then(result => {
      if (result.ok === false) {
        this.patch({ contextBreakdownLoading: false, contextBreakdown: null })
        return
      }
      const breakdown: ContextBreakdown = {
        systemPromptTokens: Math.max(0, num(result.system_prompt_tokens) ?? 0),
        toolsTokens: Math.max(0, num(result.tools_tokens) ?? 0),
        messagesTokens: Math.max(0, num(result.messages_tokens) ?? 0),
        totalTokens: Math.max(0, num(result.total_tokens) ?? 0),
        contextLimit: Math.max(0, num(result.context_limit) ?? 0),
      }
      this.patch({ contextBreakdownLoading: false, contextBreakdown: breakdown })
    }).catch(() => {
      this.patch({ contextBreakdownLoading: false, contextBreakdown: null })
    })
  }

  closeContextMenu(): void {
    this.patch({ contextMenuOpen: false })
  }

  toggleWorkspaceMenu(): void {
    this.patch({ wsMenuOpen: !this.frame.wsMenuOpen, paletteOpen: false, pickerOpen: false })
  }

  closeWorkspaceMenu(): void {
    this.patch({ wsMenuOpen: false })
  }

  /**
   * 'Stream thinking' — a display choice, not a policy change: the daemon
   * keeps streaming reasoning trails; the feed just stops showing them.
   */
  setStreamThinking(value: boolean): void {
    this.patch({ streamThinking: value })
    writeStreamThinking(value)
  }

  // ── New-task modal (mockup 18) ───────────────────────────────────────

  openTaskModal(): void {
    if (this.frame.turnActive || this.frame.connection !== 'online') return
    this.patch({ taskModalOpen: true, paletteOpen: false, wsMenuOpen: false, sessionMenu: null })
    void this.loadAgentPresets()
  }

  closeTaskModal(): void {
    this.patch({ taskModalOpen: false })
  }

  /**
   * Start the task the modal collected: fresh session, plan ceiling applied
   * to THAT session, then the objective submitted — in this order, so the
   * ceiling and the first message land on the session they belong to.
   */
  async startTask(objective: string, planFirst: boolean, agentPreset?: string): Promise<void> {
    this.patch({ taskModalOpen: false })
    if (this.frame.turnActive || this.frame.connection !== 'online') return
    const bound = await this.beginFreshTask(agentPreset)
    if (!bound) return
    if (planFirst && !this.frame.planMode) this.setPlanMode(true)
    const text = objective.trim()
    if (text) await this.submit(text)
  }

  // ── Session context menu (mockup 08) ─────────────────────────────────

  openSessionMenu(row: { id: string; key: string; title: string }, x: number, y: number): void {
    // Keep the 216px menu on screen near the pointer. Viewport fallbacks
    // keep SSR/tests (no window) on a sane clamp.
    const vw = typeof window === 'object' && Number.isFinite(window.innerWidth) && window.innerWidth > 0 ? window.innerWidth : 4000
    const vh = typeof window === 'object' && Number.isFinite(window.innerHeight) && window.innerHeight > 0 ? window.innerHeight : 4000
    const clampedX = Math.min(Math.max(0, x), Math.max(0, vw - 232))
    const clampedY = Math.min(Math.max(0, y), Math.max(0, vh - 150))
    this.patch({ sessionMenu: { id: row.id, key: row.key, title: row.title, x: clampedX, y: clampedY }, wsMenuOpen: false })
  }

  closeSessionMenu(): void {
    this.patch({ sessionMenu: null })
  }

  /** Rename through the daemon's `session.title` — the wire owns the title. */
  async renameSession(key: string, title: string): Promise<void> {
    const clean = title.trim()
    const anchor = this.frame.sessionMenu
    this.patch({ sessionMenu: null })
    if (!clean) return
    try {
      await this.bridge.call('session.title', { session_key: key, title: clean })
      this.refreshSessions()
    } catch (error) {
      // Observable, not swallowed: the menu reopens at its anchor so the
      // failed rename is visible and retryable.
      if (anchor) this.patch({ sessionMenu: anchor })
      console.error('session.title failed', error)
    }
  }

  /**
   * Export a session transcript as a markdown download (mockup 08). The
   * record comes from the daemon's session.status; the fold is the same one
   * the replay renders, so the file shows what the transcript shows.
   */
  async exportSessionTranscript(key: string): Promise<void> {
    this.patch({ sessionMenu: null })
    if (typeof document === 'undefined') return
    try {
      const result = await this.bridge.call('session.status', { session_key: key })
      const record = (this.sessionOf(result) ?? result) as unknown as ExportSession
      const markdown = sessionToMarkdown(record)
      const anchor = document.createElement('a')
      anchor.href = URL.createObjectURL(new Blob([markdown], { type: 'text/markdown' }))
      anchor.download = `${(str(record.title) || str(record.id) || 'session').replace(/[^\w-]+/g, '-').slice(0, 64) || 'session'}.md`
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      setTimeout(() => URL.revokeObjectURL(anchor.href), 5_000)
    } catch (error) {
      // Observable, not swallowed — a failed export must say so.
      this.fail(new Error(`export failed: ${String(error)}`))
    }
  }

  /**
   * Undo recorded edits through the daemon — one file, or every file when
   * `path` is null. Undone files leave the review list; a refusal (the file
   * changed since the edit) surfaces as a visible error, never a silent one.
   */
  async undoChanges(path: string | null): Promise<void> {
    try {
      const result = await this.bridge.call('changes.undo', {
        session_key: this.sessionKey,
        ...(path ? { path } : {}),
      })
      if (result.ok === false && !Array.isArray(result.results)) {
        this.builder.push('notification', { severity: 'error', message: str(result.error) || 'undo failed' })
        this.notify()
        return
      }
      const results = Array.isArray(result.results) ? result.results : []
      const undone = new Set(
        results.filter(row => row && typeof row === 'object' && (row as Record<string, unknown>).ok === true)
          .map(row => str((row as Record<string, unknown>).path)),
      )
      const failures = results.filter(row => row && typeof row === 'object' && (row as Record<string, unknown>).ok !== true)
      this.patch({ changes: this.frame.changes.filter(change => !undone.has(change.path)) })
      const reverted = typeof result.reverted === 'number' ? result.reverted : 0
      const scope = path ? ` in ${path}` : ''
      if (failures.length) {
        const first = failures[0] as Record<string, unknown>
        this.builder.push('notification', {
          severity: 'error',
          message: `Undo refused: ${str(first.error) || 'unknown refusal'}`,
        })
      } else if (reverted > 0) {
        this.builder.push('notification', { severity: 'info', message: `Undid ${reverted} edit${reverted === 1 ? '' : 's'}${scope}` })
      }
      this.notify()
    } catch (error) {
      this.fail(error)
    }
  }

  /**
   * Create a git worktree through the daemon and switch the shell into it —
   * the per-project daemon for the worktree spawns on the reload, so the
   * next task runs in the isolated checkout.
   */
  async createWorktree(name: string): Promise<void> {
    try {
      const result = await this.bridge.call('workspace.worktree', { action: 'create', name })
      if (result.ok === false) {
        this.builder.push('notification', { severity: 'error', message: str(result.error) || 'worktree refused' })
        this.notify()
        return
      }
      const path = str(result.path)
      if (!path) return
      this.patch({ taskModalOpen: false })
      void this.bridge.useWorkspace?.(path)
    } catch (error) {
      this.fail(error)
    }
  }

  // ── Failure + retry ──────────────────────────────────────────────────

  retryFailed(): void {
    const failed = this.frame.failed
    if (!failed) return
    this.failure = null
    this.patch({ failed: null, turnFailed: false, tab: 'activity' })
    if (failed.lastUser) void this.submit(failed.lastUser)
  }

  resolveFailure(): void {
    this.failure = null
    this.patch({ failed: null, turnFailed: false })
  }

  // ── Settings data ────────────────────────────────────────────────────

  /** Restore persisted skill suggestions for the bound session. */
  loadSkillSuggestions(): void {
    void this.bridge.call('skill_suggestions', { session_key: this.sessionKey }).then(result => {
      if (result.ok === false) return
      const rows = Array.isArray(result.suggestions) ? result.suggestions : []
      const suggestions = rows.map(skillSuggestionOf).filter((row): row is SkillSuggestion => row !== null)
      this.patch({ skillSuggestions: suggestions })
    }).catch(() => {
      // Older daemons have no suggestions RPC; live structured events still work.
    })
  }

  /** Restore the creator-mode audit trail for the bound session. */
  loadCreatorTrace(): void {
    void this.bridge.call('creator_trace', { session_key: this.sessionKey }).then(result => {
      if (result.ok === false) return
      const rows = Array.isArray(result.trace) ? result.trace : []
      const trace = rows.map(creatorTraceOf).filter((row): row is CreatorTrace => row !== null)
      this.patch({ creatorTrace: trace })
    }).catch(() => {
      // Older daemons do not expose creator mode.
    })
  }

  async loadAgentPresets(): Promise<void> {
    try {
      const result = await this.bridge.call('agentPreset.list', {})
      if (result.ok === false) throw new Error(str(result.error) || 'could not load agent presets')
      this.patch({ agentPresets: agentPresetsFromResult(result) })
    } catch (error) {
      this.fail(error)
    }
  }

  async setDefaultAgentPreset(id: string): Promise<void> {
    try {
      const result = await this.bridge.call('agentPreset.setDefault', { agent_preset: id })
      if (result.ok === false) throw new Error(str(result.error) || 'could not set default agent preset')
      await this.loadAgentPresets()
    } catch (error) {
      this.fail(error)
    }
  }

  async copyAgentPreset(from: string, id: string, name?: string): Promise<boolean> {
    try {
      const result = await this.bridge.call('agentPreset.copy', {
        from,
        agent_preset: id,
        ...(name?.trim() ? { name: name.trim() } : {}),
      })
      if (result.ok === false) throw new Error(str(result.error) || 'could not duplicate agent preset')
      await this.loadAgentPresets()
      return true
    } catch (error) {
      this.fail(error)
      return false
    }
  }

  async removeAgentPreset(id: string): Promise<void> {
    try {
      const result = await this.bridge.call('agentPreset.remove', { agent_preset: id })
      if (result.ok === false) throw new Error(str(result.error) || 'could not remove agent preset')
      await this.loadAgentPresets()
    } catch (error) {
      this.fail(error)
    }
  }

  async readAgentPreset(id: string): Promise<string> {
    const result = await this.bridge.call('agentPreset.read', { agent_preset: id })
    if (result.ok === false) throw new Error(str(result.error) || 'could not read agent preset')
    return str(result.content)
  }

  async openAgentPresetLocation(id: string): Promise<void> {
    try {
      const result = await this.bridge.call('agentPreset.openDocument', { agent_preset: id })
      if (result.ok === false) throw new Error(str(result.error) || 'could not open agent preset')
      const path = str(result.path)
      if (path) await window.xerxes.openPath?.(path)
    } catch (error) {
      this.fail(error)
    }
  }

  async draftAgentPreset(): Promise<void> {
    this.patch({ settingsOpen: false })
    await this.startTask(
      'Help me create a custom Xerxes agent preset. Ask what behavior and capabilities I want, then duplicate a suitable preset, author it, and validate it.',
      false,
      'creator',
    )
  }

  async loadProviders(): Promise<void> {
    const providers = this.bridge
      .call('provider_list', {})
      .then(result => {
        const rows = Array.isArray(result.profiles) ? (result.profiles as unknown[]) : []
        return rows
          .map(raw => {
            const row = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
            const name = str(row.name)
            if (!name) return null
            return {
              name,
              provider: str(row.provider) || providerOf(str(row.model) || name),
              model: str(row.model),
              active: row.active === true,
              baseUrl: str(row.base_url),
            } satisfies ProviderRow
          })
          .filter((row): row is ProviderRow => row !== null)
      })
      .catch(() => null)
    // The adapter catalog for the add/edit form; an older daemon simply
    // lacks the method, and the form falls back to a free-text type field.
    const types = this.bridge
      .call('provider_types', {})
      .then(result => {
        const rows = Array.isArray(result.types) ? (result.types as unknown[]) : []
        return rows
          .map(raw => {
            const row = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
            const name = str(row.name)
            if (!name) return null
            return {
              name,
              baseUrl: str(row.base_url),
              apiKeyEnv: str(row.api_key_env),
            } satisfies ProviderTypeRow
          })
          .filter((row): row is ProviderTypeRow => row !== null)
      })
      .catch(() => null)
    const status = this.bridge
      .call('runtime.status', {})
      .then(result => ({ permissionMode: str(result.permission_mode), model: str(result.model) }))
      .catch(() => null)
    void Promise.all([providers, types, status]).then(([rows, typeRows, state]) => {
      if (rows) this.patch({ providers: rows })
      if (typeRows) this.patch({ providerTypes: typeRows })
      // Daemon-wide fallback only: a session-scoped /permissions pin from
      // initialize is the truth for what this app's next tool call faces.
      if (state?.permissionMode && !this.frame.permissionMode) this.patch({ permissionMode: state.permissionMode })
    })
  }

  setPermissionMode(mode: PermissionMode): void {
    void this.bridge
      .call('slash', { command: `/permissions ${mode}` })
      .then(result => {
        // The slash result carries the pinned mode; adopt it immediately so
        // the card's ✓ marker follows the click instead of waiting for a
        // re-initialize that may never come.
        const pinned = str(result.permission_mode)
        if (result.ok !== false && pinned) this.patch({ permissionMode: pinned })
        void this.loadProviders()
      })
      .catch(error => this.fail(error))
  }

  // ── Connection ───────────────────────────────────────────────────────

  retryConnection(): void {
    this.patch({ connection: 'connecting' })
    // Resume the open conversation by id when one exists — a bare
    // initialize evicts the live session and the daemon would hand back a
    // fresh, context-free one under our key.
    const extra: Record<string, unknown> = this.frame.currentId
      ? { resume_session_id: this.frame.currentId }
      : {}
    void this.initializeSelfHealing().then(
      () => {
        void this.refreshGoal()
        void this.refreshSessions()
      },
      () => this.wentOffline(),
    )
  }

  private cameOnline(): void {
    if (this.frame.connection !== 'online') this.patch({ connection: 'online' })
  }

  private wentOffline(): void {
    // A daemon that died mid-turn will never send turn_end; clear the
    // acting badge (and with it the 1s tick) or Stop/⌘N stay bricked
    // against a turn that no longer exists. The live runs must fold too:
    // otherwise they sit in the builder and resurrect — blinking carets on
    // finished messages — the next time any turn renders its active fold.
    // `connecting` counts: a failed retryConnection lands here too.
    if (this.frame.connection !== 'offline' && this.frame.turnActive) {
      this.stopTick()
      this.pendingAgentCalls.clear()
      this.stopFleetPoll()
      this.builder.finalize()
      this.patch({ connection: 'offline', turnActive: false, turnSeconds: 0 })
      return
    }
    if (this.frame.connection !== 'offline') this.patch({ connection: 'offline' })
  }

  /** Cheap liveness probe; also heals the badge after a daemon restart. */
  private async beat(): Promise<void> {
    if (this.frame.connection === 'online') {
      // Events would still be flowing; a silent socket only shows when a
      // call dies, which every action already routes through fail().
      return
    }
    try {
      await this.bridge.call('runtime.status', {})
      this.cameOnline()
      const extra: Record<string, unknown> = this.frame.currentId
        ? { resume_session_id: this.frame.currentId }
        : {}
      await this.initialize(extra)
      void this.refreshGoal()
      void this.refreshSessions()
    } catch {
      this.wentOffline()
    }
  }

  // ── RPC helpers ──────────────────────────────────────────────────────

  private bridge: XerxesLike = {
    call: (method, params) => window.xerxes.call(method, params),
    // Passthrough — the wrapper must forward EVERY preload method or the
    // optional calls silently do nothing.
    chooseWorkspace: () => window.xerxes.chooseWorkspace?.() ?? Promise.resolve(null),
    useWorkspace: dir => window.xerxes.useWorkspace?.(dir) ?? Promise.resolve(null),
    getWorkspace: () => window.xerxes.getWorkspace?.() ?? Promise.resolve(''),
  }

  private sessionOf(result: Record<string, unknown>): Record<string, unknown> {
    return result.session && typeof result.session === 'object'
      ? (result.session as Record<string, unknown>)
      : {}
  }

  private telemetryFromSession(session: Record<string, unknown>): Partial<Snapshot> {
    const executions = Array.isArray(session.tool_executions)
      ? session.tool_executions.filter((value): value is Record<string, unknown> => Boolean(value) && typeof value === 'object' && !Array.isArray(value))
      : []
    const storedToolDuration = executions.reduce((total, row) => total + Math.max(0, num(row.duration_ms) ?? 0), 0)
    this.ttftSamples = Math.max(0, Math.trunc(num(session.ttft_samples) ?? 0))
    this.ttftTotalMs = Math.max(0, num(session.ttft_total_ms) ?? 0)
    this.turnCount = Math.max(0, Math.trunc(num(session.turn_count) ?? 0))
    const ttftAverage = num(session.ttft_avg_ms)
      ?? (this.ttftSamples > 0 ? this.ttftTotalMs / this.ttftSamples : null)
    const tokensPerSecond = num(session.tokens_per_second)
    const cacheHitRate = num(session.cache_hit_rate)
    return {
      turnCount: this.turnCount,
      llmDurationMs: Math.max(0, num(session.llm_duration_ms) ?? 0),
      llmSteps: Math.max(0, Math.trunc(num(session.llm_steps) ?? num(session.calls) ?? 0)),
      toolDurationMs: Math.max(0, num(session.tool_duration_ms) ?? storedToolDuration),
      toolSteps: Math.max(0, Math.trunc(num(session.tool_steps) ?? executions.length)),
      inputTokens: Math.max(0, num(session.input_tokens) ?? 0),
      outputTokens: Math.max(0, num(session.output_tokens) ?? 0),
      metricPhase: str(session.active_turn_id) ? 'llm' : null,
      metricPhaseStartedAt: str(session.active_turn_id) ? Date.now() : null,
      tokensPerSecond: tokensPerSecond === null ? null : Math.max(0, tokensPerSecond),
      cacheHitRate: cacheHitRate === null ? null : Math.max(0, Math.min(1, cacheHitRate)),
      ttftMs: ttftAverage === null ? null : Math.max(0, ttftAverage),
    }
  }

  private async initialize(extra: Record<string, unknown>): Promise<Record<string, unknown>> {
    const result = await this.bridge.call('initialize', {
      session_key: this.sessionKey,
      ...extra,
      ...clientHandshake(),
    })
    const session = this.sessionOf(result)
    // A resume binds the session under the session id, not our requested
    // key — every session-scoped call below must target what the daemon
    // actually bound or it silently addresses a fresh session.
    const boundKey = str(session.key) || str(result.session_id)
    if (extra.resume_session_id && boundKey) this.sessionKey = boundKey
    const reportedContextLimit = num(result.context_limit)
    const contextLimit = reportedContextLimit !== null && reportedContextLimit > 0
      ? reportedContextLimit
      : null
    // The daemon owns turn truth (`active_turn_id` exists only mid-turn).
    // A fold that still believes it is acting after a restart or reconnect
    // would show ▶ act forever with blinking carets on finished messages —
    // reconcile to the daemon and close any stranded live runs.
    const daemonInTurn = str(session.active_turn_id) !== ''
    if (!daemonInTurn && this.frame.turnActive) {
      this.stopTick()
      this.builder.finalize()
    }
    this.patch({
      connection: 'online',
      currentId: str(result.session_id ?? session.id),
      currentTitle: str(session.title),
      sessionKey: this.sessionKey,
      cwd: str(result.cwd ?? session.cwd),
      model: str(result.model),
      ...(str(result.reasoning_effort)
        ? { reasoningEffort: str(result.reasoning_effort) }
        : {}),
      currentAgentPreset: str(result.agent_name ?? session.agent_id) || this.frame.currentAgentPreset,
      branch: str(result.branch),
      daemonWarning: daemonCompatibilityWarning(result),
      costUsd: typeof result.cost_usd === 'number' ? result.cost_usd : null,
      contextMax: contextLimit,
      approval: null,
      ...(daemonInTurn ? {} : { turnActive: false, turnSeconds: 0 }),
      // initialize reports THIS session's policy — /permissions pins the
      // mode per session, so the daemon-wide runtime.status would lie about
      // what the next tool call will actually face.
      ...(str(result.permission_mode) ? { permissionMode: str(result.permission_mode) } : {}),
      ...this.telemetryFromSession(session),
    })
    this.adoptFleet(session)
    this.loadSkillSuggestions()
    this.loadCreatorTrace()
    return result
  }

  /**
   * Fleet truth: the parent session's subagent_snapshots panel — the
   * sessions map behind session.active_list only holds client-opened
   * sessions, so filtering it for kind 'subagent' shows nothing while
   * subagents actually run.
   */
  private adoptFleet(session: Readonly<Record<string, unknown>>): void {
    const raw = Array.isArray(session.subagent_snapshots) ? session.subagent_snapshots : []
    const fleet = raw
      .map(item => {
        const row = (item && typeof item === 'object' ? item : {}) as Record<string, unknown>
        const id = str(row.id)
        if (!id) return null
        const label = str(row.title) || str(row.name) || str(row.agent_id) || `#${id.slice(0, 6)}`
        const entry: SessionRow = {
          id,
          key: id,
          title: label,
          status: str(row.status) || 'running',
          age: '',
          current: false,
          kind: 'subagent',
          turns: 0,
          messages: 0,
          cwd: '',
          untitled: false,
        }
        return entry
      })
      .filter(row => row !== null)
    this.patch({ fleet })
    this.syncAgentMembersFromFleet(fleet)
  }

  /**
   * Fold daemon snapshot statuses into the agents card. Snapshots persist in
   * session metadata, so this is the terminal-status path for children that
   * outlive their turn. Unseen rows are adopted only while LIVE — the
   * previous turn's terminal snapshots must not open a stale card at the
   * next turn's start.
   */
  private syncAgentMembersFromFleet(fleet: readonly SessionRow[]): void {
    if (fleet.length === 0) return
    let touched = false
    for (const row of fleet) {
      const status = agentStatusOf(row.status)
      const key = this.agentMemberKeysByTitle.get(row.title)
      if (key) {
        const member = this.agentMembers.get(key)
        if (member && member.status !== status) {
          this.agentMembers.set(key, { ...member, status })
          touched = true
        }
        continue
      }
      if (!this.agentMembers.has(row.id) && status === 'working') {
        this.agentMembers.set(row.id, { key: row.id, title: row.title, status })
        this.agentMemberKeysByTitle.set(row.title, row.id)
        touched = true
      }
    }
    if (!touched) return
    const members = [...this.agentMembers.values()]
    this.builder.pushAgents(members)
    if (members.some(m => m.status === 'working')) this.startFleetPoll()
    this.notify()
  }

  /**
   * Events tagged with a foreign session_id are background turns (the daemon
   * runs them as bg-* sessions on this connection's pipe). They never enter
   * the foreground fold — they move the header jobs chip only.
   */
  private onBackgroundEvent(type: string, payload: Readonly<Record<string, unknown>>, sessionId: string): void {
    if (type === 'turn_begin') {
      const title = (str(payload.user_input) || str(payload.text) || 'background task').slice(0, 80)
      this.patch({
        backgroundJobs: [
          ...this.frame.backgroundJobs.filter(job => job.id !== sessionId),
          { id: sessionId, title, status: 'working' },
        ],
      })
      return
    }
    if (type === 'session_title') {
      const title = str(payload.title)
      if (title) {
        this.patch({
          backgroundJobs: this.frame.backgroundJobs.map(job => (job.id === sessionId ? { ...job, title } : job)),
        })
      }
      return
    }
    if (type === 'turn_end') {
      // Running-only chip: settled work leaves the list.
      this.patch({ backgroundJobs: this.frame.backgroundJobs.filter(job => job.id !== sessionId) })
      void this.refreshSessions()
    }
  }

  /**
   * Attach-mid-run seed: the event stream only sees turns that START while
   * we're attached, so the chip also reads `session.active_list` — bg-*
   * sessions with an active turn are working background jobs by definition.
   */
  private seedBackgroundJobs(rows: unknown): void {
    if (!Array.isArray(rows)) return
    const seeded: BackgroundJob[] = []
    for (const raw of rows) {
      const row = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
      const id = str(row.id ?? row.session_id)
      if (!id || id === this.frame.currentId) continue
      if (!str(row.key).startsWith('bg-')) continue
      if (!str(row.active_turn_id)) continue
      seeded.push({ id, title: str(row.title) || 'background task', status: 'working' })
    }
    // Event-driven entries win (they carry the prompt as title).
    const live = this.frame.backgroundJobs.filter(job => !seeded.some(s => s.id === job.id))
    const merged = [...seeded, ...live]
    const same =
      merged.length === this.frame.backgroundJobs.length &&
      merged.every((job, index) => job.id === this.frame.backgroundJobs[index]?.id && job.status === this.frame.backgroundJobs[index]?.status)
    if (!same) this.patch({ backgroundJobs: merged })
  }

  private async refreshGoal(): Promise<void> {
    try {
      const result = await this.bridge.call('session.goal', { session_key: this.sessionKey, input: '' })
      this.patch({ goal: str(result.text) })
    } catch {
      // Best-effort view; absence renders an empty goal card.
    }
  }

  /** Mid-turn fleet refresh: snapshots only move while a turn runs. */
  private refreshFleet(): void {
    void this.bridge
      .call('session.status', { session_key: this.sessionKey })
      .then(result => {
        const session = this.sessionOf(result)
        this.adoptFleet(Object.keys(session).length ? session : result)
      })
      .catch(() => {})
  }

  /**
   * Poll while agent-family tools are in flight. A subagent spawn is
   * persisted inside tool execution, so the refresh fired at the `tool_call`
   * event itself can still come back empty — the poll is what guarantees the
   * rail shows the child while it actually runs.
   */
  private startFleetPoll(): void {
    if (this.fleetPoll) return
    this.fleetPoll = setInterval(() => {
      this.fleetPollRounds += 1
      const anyWorking = [...this.agentMembers.values()].some(m => m.status === 'working')
      // Background children outlive their turn's tool calls — the poll only
      // dies once the card is fully terminal (or after ~6 minutes of
      // silence, leaving the last-known states honestly displayed).
      if ((this.pendingAgentCalls.size === 0 && !anyWorking) || this.fleetPollRounds > 180) {
        this.stopFleetPoll()
        return
      }
      this.refreshFleet()
    }, FLEET_POLL_MS)
  }

  private stopFleetPoll(): void {
    if (this.fleetPoll) {
      clearInterval(this.fleetPoll)
      this.fleetPoll = null
    }
    this.fleetPollRounds = 0
  }

  /** Fetch MCP server statuses for the settings card (best-effort). */
  refreshMcpStatus(): void {
    void this.bridge
      .call('session.status', { session_key: this.sessionKey })
      .then(result => {
        const raw = (this.sessionOf(result).mcp_status ?? (result as Record<string, unknown>).mcp_status) as unknown
        if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return
        const statuses: Record<string, McpServerStatus> = {}
        for (const [name, value] of Object.entries(raw as Record<string, unknown>)) {
          if (!value || typeof value !== 'object') continue
          const record = value as Record<string, unknown>
          statuses[name] = {
            connected: record.connected === true,
            tools: typeof record.tools === 'number' ? record.tools : 0,
            resources: typeof record.resources === 'number' ? record.resources : 0,
            prompts: typeof record.prompts === 'number' ? record.prompts : 0,
            ...(typeof record.lastError === 'string' && record.lastError ? { lastError: record.lastError } : {}),
          }
        }
        this.patch({ mcpStatus: statuses })
      })
      .catch(() => {})
  }

  /** Reconnect every configured MCP server through the daemon's slash RPC. */
  async reloadMcp(): Promise<void> {
    try {
      await this.bridge.call('slash', { command: '/reload-mcp' })
      this.refreshMcpStatus()
    } catch (error) {
      this.fail(error)
    }
  }

  private refreshSessions(): void {
    const saved = this.bridge
      .call('session.list', { kind: 'main', scope: 'global', limit: 60 })
      .then(result => normalize(result.sessions, this.frame.currentId))
      .catch(() => null)
    const active = this.bridge
      .call('session.active_list', {})
      .then(result => {
        this.seedBackgroundJobs(result.sessions)
        return normalize(result.sessions, this.frame.currentId)
      })
      .catch(() => null)
    void Promise.all([saved, active]).then(([savedRows, activeRows]) => {
      if (!savedRows && !activeRows) return
      // The attached session is not 'fleet' and not a history row — it is
      // what the chat column is already showing.
      const currentId = this.frame.currentId
      const all = activeRows ?? []
      // An untitled 0-turn live row is an empty shell — a session some client
      // opened and never spoke in. It is not a task; listing it as one is how
      // the sidebar filled with "0 turns" ghosts.
      const live = all.filter(
        row => row.kind === 'main' && row.id !== currentId && !(row.turns === 0 && row.untitled),
      )
      // Fleet comes from the parent's subagent_snapshots (adoptFleet) —
      // active_list only holds client-opened sessions and would blank the
      // panel while subagents actually run.
      const liveIds = new Set(live.map(row => row.id))
      const history = (savedRows ?? []).filter(row => !liveIds.has(row.id) && row.id !== currentId)
      this.patch({ live, sessions: history })
      this.enrichUntitled(history)
    })
  }

  /**
   * Give untitled history rows a real identity: the first user message,
   * fetched lazily through session.status and capped. Rows whose sessions
   * are no longer loaded answer `{ok:false}` and keep the short-id label.
   */
  private enrichUntitled(rows: readonly SessionRow[]): void {
    for (const row of rows) {
      if (row.untitled && !this.snippets[row.id] && !this.enriching.has(row.id)) {
        this.enriching.add(row.id)
        void this.bridge
          .call('session.status', { session_key: row.key })
          .then(result => {
            const session = this.sessionOf(result)
            const transcript = session.transcript ?? session.messages
            if (!Array.isArray(transcript)) return
            for (const message of transcript as unknown[]) {
              if (!message || typeof message !== 'object') continue
              const record = message as Record<string, unknown>
              if (record.role !== 'user') continue
              const content = record.content
              const text = typeof content === 'string'
                ? content
                : Array.isArray(content)
                  ? content.map(part => (part && typeof part === 'object' && typeof (part as Record<string, unknown>).text === 'string' ? (part as Record<string, unknown>).text as string : '')).join('')
                  : ''
              const cleaned = text.replace(/\s+/g, ' ').trim()
              if (cleaned) {
                this.snippets = { ...this.snippets, [row.id]: cleaned.length > SNIPPET_CAP ? `${cleaned.slice(0, SNIPPET_CAP - 1)}…` : cleaned }
                break
              }
            }
          })
          .catch(() => {})
          .finally(() => {
            this.enriching.delete(row.id)
            this.patch({})
          })
      }
    }
  }

  private async setGoal(input: string): Promise<{ ok: boolean; text: string }> {
    try {
      const result = await this.bridge.call('session.goal', { session_key: this.sessionKey, input })
      const ok = result.ok === true
      const text = str(result.text) || (ok ? 'ok' : 'command failed')
      this.patch({ goal: ok ? text : this.frame.goal })
      this.builder.push('notification', { severity: ok ? 'info' : 'error', message: text })
      this.notify()
      return { ok, text }
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error)
      this.fail(error)
      return { ok: false, text }
    }
  }

  // ── Event stream ─────────────────────────────────────────────────────

  private onEvent(event: DaemonEvent): void {
    const { type, payload } = event
    this.pushLog(type, payload)
    // Background turns (bg-* sessions) share this connection's event pipe
    // with a session_id tag. They must never touch the foreground fold —
    // the TUI applies the same filter. They drive the jobs chip instead.
    const eventSession = str(payload.session_id)
    if (eventSession && this.frame.currentId && eventSession !== this.frame.currentId) {
      this.onBackgroundEvent(type, payload, eventSession)
      return
    }
    switch (type) {
      case 'turn_begin': {
        this.turnCount += 1
        // A new attempt supersedes the previous failure card; retry clears
        // it explicitly, and any other submit means the human moved on.
        this.failure = null
        // Defensive: an interrupted previous turn (daemon restart, dropped
        // turn_end) may have left live runs in the builder. Commit them as
        // closed blocks BEFORE the active fold renders, or they come back
        // as blinking streaming rows riding this new turn. No-op when the
        // previous turn ended cleanly.
        this.builder.finalize()
        // Current daemons echo the prompt as `text`; older ones said
        // `user_input`. Either way the local optimistic line may already
        // cover it — dedupe on content.
        const user = typeof payload.user_input === 'string'
          ? payload.user_input
          : typeof payload.text === 'string'
            ? payload.text
            : ''
        if (user) {
          this.lastUser = user
          const last = this.frame.blocks.at(-1)
          if (!(last && last.kind === 'user' && last.text === user)) this.builder.pushUser(user)
        }
        // A late delta that landed after the previous turn_end sits in the
        // scratch runs; left alone it re-enters the stream as a "live" block
        // of THIS turn. Drain it committed before the tail opens.
        this.builder.finalize()
        // The new turn's first spawn opens a fresh agents card; the previous
        // card's committed copy keeps its terminal states.
        this.builder.closeAgentsCard()
        this.agentMembers.clear()
        this.agentMemberKeysByTitle.clear()
        this.startTurn()
        // First fleet read of the turn; agent-family tool calls below keep
        // the panel polling while spawns actually run.
        this.refreshFleet()
        break
      }
      case 'text_part': {
        this.setMetricPhase('llm')
        if (typeof payload.text === 'string' && payload.text) this.agentText += payload.text
        this.builder.push(type, payload)
        this.notify()
        break
      }
      case 'think_part':
      case 'tool_result':
      case 'notification':
        if (type === 'think_part') this.setMetricPhase('llm')
        if (type === 'notification') {
          const suggestion = skillSuggestionOf(payload.skill)
          if (suggestion) {
            this.patch({
              skillSuggestions: [
                ...this.frame.skillSuggestions.filter(row => row.skillName !== suggestion.skillName),
                suggestion,
              ].slice(-32),
            })
          }
          const body = str(payload.body) || str(payload.message)
          const severity = String(payload.severity ?? payload.level ?? 'info').toLowerCase()
          if (body && (severity.includes('error') || severity.includes('fatal')) && this.frame.turnActive) {
            // A failed turn announces itself as an error notification — never
            // as silence; text may still have streamed before the failure.
            this.turnError = body
          }
        }
        if (type === 'tool_result' && typeof payload.error === 'string' && payload.error && this.frame.turnActive) {
          this.turnError = this.turnError ?? payload.error
        }
        if (type === 'tool_result') {
          // A foreground agent's terminal status lands with its result:
          // refresh now, then let the poll die once nothing agent-side runs.
          const id = str(payload.tool_call_id)
          if (id) {
            this.pendingAgentCalls.delete(id)
            this.activeMetricTools.delete(id)
          }
          const durationMs = Math.max(0, num(payload.duration_ms) ?? 0)
          this.patch({
            toolDurationMs: this.frame.toolDurationMs + durationMs,
            toolSteps: this.frame.toolSteps + 1,
          })
          if (this.activeMetricTools.size === 0) this.setMetricPhase('llm')
          if (isAgentFamilyTool(payload.name)) this.refreshFleet()
          // A failed spawn never reached the manifest — the card's only
          // honest terminal signal is the result itself.
          if (id && typeof payload.error === 'string' && payload.error && this.agentMembers.size > 0) {
            let touched = false
            for (const [key, member] of this.agentMembers) {
              if (!key.startsWith(`${id}:`) || member.status !== 'working') continue
              this.agentMembers.set(key, { ...member, status: 'failed' })
              touched = true
            }
            if (touched) this.builder.pushAgents([...this.agentMembers.values()])
          }
          if (this.pendingAgentCalls.size === 0 && ![...this.agentMembers.values()].some(m => m.status === 'working')) this.stopFleetPoll()
        }
        this.builder.push(type, payload)
        this.notify()
        break
      case 'tool_call': {
        const stats = editStatsOf(payload.name, payload.arguments)
        if (stats) this.foldChange(stats, parseArgs(payload.arguments), str(payload.name))
        this.builder.push(type, payload)
        const metricToolId = str(payload.tool_call_id) || str(payload.id)
        if (metricToolId) this.activeMetricTools.add(metricToolId)
        this.setMetricPhase('tool')
        this.notify()
        // Agent-family calls move the fleet panel while they run — spawn
        // manifests persist inside tool execution, so refresh now AND keep
        // polling until the matching tool_result lands.
        if (isAgentFamilyTool(payload.name)) {
          const id = str(payload.tool_call_id) || str(payload.id)
          if (id) this.pendingAgentCalls.add(id)
          // The in-chat agents card is event-driven, not manifest-driven:
          // even when the spawn dies inside the daemon (provider down,
          // policy refusal) and subagent_snapshots stays empty, the batch
          // appears here — and the error result below marks it failed.
          if (id && isSpawnTool(payload.name)) {
            for (const member of spawnMembersOf(payload.name, payload.arguments, id)) {
              this.agentMembers.set(member.key, member)
              if (!this.agentMemberKeysByTitle.has(member.title)) this.agentMemberKeysByTitle.set(member.title, member.key)
            }
            this.builder.pushAgents([...this.agentMembers.values()])
          }
          this.refreshFleet()
          this.startFleetPoll()
          this.notify()
        }
        break
      }
      case 'steer_input': {
        // Acceptance echo, not consumption: the daemon emits this the moment
        // it queues the text, and offers no "consumed" signal — steers drain
        // silently at the next step boundary. The mirror therefore lives
        // until turn_end, which is the one boundary we do observe.
        break
      }
      case 'session_title': {
        // Broadcast event: every session that earns a title (subagents,
        // other clients) lands here. Only ours may retitle the header.
        const title = str(payload.title)
        const forUs = !payload.session_id || str(payload.session_id) === this.frame.currentId
        if (title && forUs) this.patch({ currentTitle: title })
        break
      }
      case 'agent_preset_selected': {
        const selected = str(payload.agent_preset)
        if (selected) this.patch({ currentAgentPreset: selected })
        break
      }
      case 'status_update': {
        const patch: Record<string, unknown> = {}
        if (typeof payload.model === 'string' && payload.model) patch.model = payload.model
        if (typeof payload.context_tokens === 'number') patch.contextTokens = payload.context_tokens
        if (typeof payload.max_context === 'number') {
          patch.contextMax = payload.max_context > 0 ? payload.max_context : null
        }
        if (typeof payload.llm_duration_ms === 'number' && Number.isFinite(payload.llm_duration_ms)) {
          patch.llmDurationMs = this.frame.llmDurationMs + Math.max(0, payload.llm_duration_ms)
          patch.llmSteps = this.frame.llmSteps + 1
          patch.metricPhase = 'llm'
          patch.metricPhaseStartedAt = Date.now()
        }
        if (typeof payload.ttft_ms === 'number' && Number.isFinite(payload.ttft_ms)) {
          this.ttftSamples += 1
          this.ttftTotalMs += Math.max(0, payload.ttft_ms)
          patch.ttftMs = this.ttftTotalMs / this.ttftSamples
        }
        if (typeof payload.total_input_tokens === 'number' && Number.isFinite(payload.total_input_tokens)) {
          patch.inputTokens = Math.max(0, payload.total_input_tokens)
        } else if (typeof payload.input_tokens === 'number' && Number.isFinite(payload.input_tokens)) {
          patch.inputTokens = Math.max(0, payload.input_tokens)
        }
        if (typeof payload.total_output_tokens === 'number' && Number.isFinite(payload.total_output_tokens)) {
          patch.outputTokens = Math.max(0, payload.total_output_tokens)
        } else if (typeof payload.output_tokens === 'number' && Number.isFinite(payload.output_tokens)) {
          patch.outputTokens = Math.max(0, payload.output_tokens)
        }
        if (typeof payload.tokens_per_second === 'number' && Number.isFinite(payload.tokens_per_second)) {
          patch.tokensPerSecond = payload.tokens_per_second
        }
        const cacheReadTokens = num(payload.cache_read_tokens)
        const cumulativeInputTokens = num(payload.total_input_tokens) ?? num(payload.input_tokens)
        if (cacheReadTokens !== null && cumulativeInputTokens !== null && cacheReadTokens + cumulativeInputTokens > 0) {
          patch.cacheHitRate = cacheReadTokens / (cacheReadTokens + cumulativeInputTokens)
        } else if (typeof payload.cache_hit_rate === 'number' && Number.isFinite(payload.cache_hit_rate)) {
          patch.cacheHitRate = Math.max(0, Math.min(1, payload.cache_hit_rate))
        }
        // Live cost estimate rides the same echo the ctx counter uses.
        if (typeof payload.cost_usd === 'number') patch.costUsd = payload.cost_usd
        // The daemon logs plan-mode flips; its status echo is authoritative.
        if (typeof payload.plan_mode === 'boolean') patch.planMode = payload.plan_mode
        // The daemon echoes effort flips here; session pinning means the
        // picker may not have been the one to change it.
        if (typeof payload.reasoning_effort === 'string' && payload.reasoning_effort) {
          patch.reasoningEffort = payload.reasoning_effort
        }
        // /permissions pins per session and echoes here — without this the
        // Permissions card's ✓ marker stays on the mode it booted with.
        if (typeof payload.permission_mode === 'string' && payload.permission_mode) {
          patch.permissionMode = payload.permission_mode
        }
        this.patch(patch)
        break
      }
      case 'approval_request': {
        const id = str(payload.id) || str(payload.request_id)
        if (!id) break
        const description =
          str(payload.description) ||
          `${str(payload.name)} ${str(JSON.stringify(payload.arguments ?? ''))}`.trim()
        this.patch({
          approval: {
            id,
            action: str(payload.action),
            description,
            ...(str(payload.tool_call_id) ? { toolCallId: str(payload.tool_call_id) } : {}),
            ...(str(payload.tool_name) ? { toolName: str(payload.tool_name) } : {}),
          },
        })
        break
      }
      case 'question_request': {
        const requestId = str(payload.id)
        const rawItems = Array.isArray(payload.questions) ? payload.questions : []
        const items = rawItems.map(raw => {
          const item = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
          return {
            id: str(item.id) || 'answer',
            question: str(item.question),
            options: Array.isArray(item.options) ? item.options.filter(o => typeof o === 'string') : [],
            allowFreeform: item.allow_free_form !== false,
            ...(str(item.placeholder) ? { placeholder: str(item.placeholder) } : {}),
          }
        }).filter(item => item.question)
        if (!requestId || items.length === 0) break
        const question = {
          requestId,
          toolCallId: str(payload.tool_call_id),
          items,
        }
        this.patch({ question })
        // A plan review captures the proposal as the session's working plan.
        if (isPlanReview(question) && this.agentText.trim()) {
          this.capturePlan(this.agentText)
        }
        break
      }
      case 'question_response': {
        // The answerer's own connection gets the echo; other surfaces drop it.
        const id = str(payload.id)
        if (this.frame.question?.requestId === id) this.patch({ question: null })
        break
      }
      case 'turn_end': {
        // Plan mode: whatever the agent reasoned toward in text is the plan
        // artifact — capture it before the buffer resets for the next turn.
        if (this.frame.planMode && this.agentText.trim()) this.capturePlan(this.agentText)
        this.agentText = ''
        this.builder.finalize()
        this.builder.pushCheckpoint(this.turnCount, this.changeTotals())
        if (this.turnError) {
          this.failure = { error: this.turnError, turn: this.turnCount, lastUser: this.lastUser }
          this.turnError = null
        }
        if (this.tick) {
          clearInterval(this.tick)
          this.tick = null
        }
        this.queue = []
        this.pendingAgentCalls.clear()
        this.activeMetricTools.clear()
        // Background children may still be working — the poll only stops
        // here when the card is already terminal; their terminal statuses
        // land on the committed card as the snapshots settle.
        if (![...this.agentMembers.values()].some(m => m.status === 'working')) this.stopFleetPoll()
        this.patch({
          turnActive: false,
          turnFailed: this.failure !== null,
          failed: this.failure,
          queue: this.queue,
          metricPhase: null,
          metricPhaseStartedAt: null,
        })
        void this.refreshGoal()
        this.refreshFleet()
        this.loadSkillSuggestions()
        this.loadCreatorTrace()
        void this.refreshSessions()
        break
      }
      default:
        break
    }
    this.notify()
  }

  // ── Workspace folds ──────────────────────────────────────────────────

  private foldChange(
    stats: { path: string; adds: number; dels: number; isNew: boolean },
    args: Record<string, unknown>,
    name: string,
  ): void {
    const previous = this.changes.get(stats.path)
    const oldLines = typeof args.old_string === 'string' ? args.old_string.replace(/\n$/, '').split('\n') : []
    const newLines = typeof args.new_string === 'string'
      ? args.new_string.replace(/\n$/, '').split('\n')
      : typeof args.content === 'string'
        ? args.content.replace(/\n$/, '').split('\n')
        : []
    const hunk: DiffLine[] = [{ kind: 'hunk', text: `@@ ${stats.path} @@` }]
    for (const line of oldLines) hunk.push({ kind: 'del', text: line })
    for (const line of newLines) hunk.push({ kind: 'add', text: line })
    const merged: DiffFile = {
      path: stats.path,
      adds: (previous?.adds ?? 0) + stats.adds,
      dels: (previous?.dels ?? 0) + stats.dels,
      isNew: previous?.isNew ?? (stats.isNew || name.toLowerCase().endsWith('writefile')),
      hunks: [...(previous?.hunks ?? []), ...hunk],
      turn: this.turnCount,
    }
    this.changes.set(stats.path, merged)
    // A new edit during review reopens the surface — "Keep all" is an
    // acknowledgment, not a permanent verdict.
    this.patch({
      changes: [...this.changes.values()].sort((a, b) => a.path.localeCompare(b.path)),
      changesKept: false,
    })
  }

  /** Acknowledge the current change set (survives tab switches). */
  ackChanges(): void {
    this.patch({ changesKept: true })
  }

  private changeTotals(): { adds: number; dels: number } {
    let adds = 0
    let dels = 0
    for (const file of this.changes.values()) {
      adds += file.adds
      dels += file.dels
    }
    return { adds, dels }
  }

  private capturePlan(markdown: string): void {
    const cleaned = markdown.trim()
    this.planState = { markdown: cleaned, items: planItemsOf(cleaned), turn: this.turnCount }
    this.patch({ plan: this.planState })
  }

  private pushLog(type: string, payload: Readonly<Record<string, unknown>>): void {
    this.logRing = [...this.logRing, { id: this.seq++, turn: this.turnCount, type, summary: summarize(type, payload) }].slice(-LOG_CAP)
  }

  private resetWorkspaceFolds(): void {
    this.queue = []
    this.changes.clear()
    this.logRing = []
    this.planState = null
    this.failure = null
    this.turnCount = 0
    this.agentText = ''
    this.lastUser = ''
    this.turnError = null
    this.ttftTotalMs = 0
    this.ttftSamples = 0
    this.activeMetricTools.clear()
    this.patch({
      queue: this.queue,
      changes: [],
      changesKept: false,
      log: this.logRing,
      plan: null,
      failed: null,
      turnCount: 0,
      turnFailed: false,
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
      skillSuggestions: [],
      creatorTrace: [],
      tab: 'activity',
    })
  }

  private setMetricPhase(phase: 'llm' | 'tool'): void {
    if (this.frame.metricPhase === phase) return
    this.patch({ metricPhase: phase, metricPhaseStartedAt: Date.now() })
  }

  private startTurn(): void {
    this.activeMetricTools.clear()
    this.patch({
      turnActive: true,
      turnFailed: false,
      turnSeconds: 0,
      metricPhase: 'llm',
      metricPhaseStartedAt: Date.now(),
    })
    if (!this.tick) {
      this.tick = setInterval(() => {
        this.patch({ turnSeconds: this.frame.turnSeconds + 1 })
      }, 1000)
    }
  }

  private stopTick(): void {
    if (this.tick) {
      clearInterval(this.tick)
      this.tick = null
    }
  }

  private fail(error: unknown): void {
    const message = error instanceof Error ? error.message : String(error)
    if (/connect|socket|offline|disposed|closed|not ready|launch/i.test(message)) this.wentOffline()
    this.patch({ error: message })
    this.builder.push('notification', { severity: 'error', message })
    this.notify()
  }

  // ── Snapshot plumbing ────────────────────────────────────────────────

  private patch(merge: Partial<Snapshot> | Record<string, unknown>): void {
    this.frame = this.frozen({
      ...this.frame,
      ...merge,
      blocks: this.builder.snapshot(this.frame.turnActive),
      log: this.logRing,
      turnCount: this.turnCount,
      snippets: this.snippets,
    })
    this.emit()
  }

  private notify(): void {
    this.frame = this.frozen({
      ...this.frame,
      blocks: this.builder.snapshot(this.frame.turnActive),
      log: this.logRing,
      turnCount: this.turnCount,
      snippets: this.snippets,
    })
    this.emit()
  }

  private frozen(value: Snapshot | Record<string, unknown>): Snapshot {
    return Object.freeze({ ...value } as Snapshot)
  }

  private emit(): void {
    for (const listener of this.listeners) listener()
  }
}

/** Shape the store actually needs from the bridge (matches types.ts). */
export interface XerxesLike {
  call(method: string, params?: Record<string, unknown>): Promise<Record<string, unknown>>
  /** Present on the real preload bridge; test bridges may omit it. */
  chooseWorkspace?(): Promise<unknown>
  useWorkspace?(dir: string): Promise<unknown>
  getWorkspace?(): Promise<string | null>
}

export const store = new Store()
