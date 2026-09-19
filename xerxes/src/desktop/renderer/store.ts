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

import { todoItemsOf, todosFromResult, type TodoItem } from './todoState.js'
import type { WorkspaceContext } from '../main/contextNavigation.js'
import { BlockBuilder, blocksFromStoredMessages, editStatsOf, parseArgs } from './blocks.js'
import {
  daemonCompatibilityWarning,
  DESKTOP_DAEMON_PROTOCOL,
  DESKTOP_VERSION,
  EXPECTED_DAEMON_BUILD_ID,
} from './buildInfo.js'
import { sessionToMarkdown, type ExportSession } from './exportMarkdown.js'
import { selectedSession, rememberSession } from './sessionPreference.js'
import { historyBlocks, readHistoryPage, type HistoryPage } from './history.js'
import { connectionFailureKind } from './connectionFailure.js'
import { desktopCall, desktopError } from './desktopRpc.js'
import { foldAgentEvent } from './agentEvents.js'
import type {
  AgentMember,
  AgentPreset,
  Approval,
  ApprovalResponse,
  BackgroundJob,
  Block,
  CachedModel,
  ChannelRow,
  ChannelStatus,
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
  SessionSearchHit,
  SessionSearchStats,
  SettingsTab,
  SkillSuggestion,
  TerminalDetail,
  TerminalRow,
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
  readonly renaming?: boolean
  readonly pending?: boolean
  readonly error?: string
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
  readonly agentPresetsError: string | null
  /** Messaging gateways from `channel.list`, kept fresh by `channel_status` broadcasts. */
  readonly channels: readonly ChannelRow[]
  readonly channelsAvailable: boolean
  readonly channelsConfigured: boolean
  /** Daemon-tracked terminals from `terminal.list` (owner: the bound session). */
  readonly terminals: readonly TerminalRow[]
  readonly terminalsLoading: boolean
  readonly terminalsError: string | null
  /** Transcript search (`session.search`) state for the search overlay. */
  readonly searchOpen: boolean
  readonly searchResults: readonly SessionSearchHit[]
  readonly searchStats: SessionSearchStats | null
  readonly searchSearching: boolean
  readonly searchError: string | null
  /** Git branch of the workspace, from the daemon's initialize — '' when unknown. */
  readonly branch: string
  /** Actionable initialize-handshake mismatch; null when app and daemon agree. */
  readonly daemonWarning: string | null
  readonly runtimeUpdate?: 'checking' | 'waiting' | 'restarting' | 'failed'
  readonly runtimeUpdateMessage?: string
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
  /** Successful explicit resumes, including reopening the current session. */
  readonly sessionOpenRevision: number
  readonly currentTitle: string
  /** The connection's bound daemon session key — what key-scoped RPCs address. */
  readonly sessionKey: string
  readonly goal: string
  readonly approval: Approval | null
  readonly question: import('./types.js').TaskQuestion | null
  readonly planMode: boolean
  readonly turnActive: boolean
  readonly networkRetrying?: boolean
  readonly turnFailed: boolean
  readonly turnSeconds: number
  readonly blocks: readonly Block[]
  readonly historyMore?: boolean
  readonly historyLoading?: boolean
  readonly historyError?: string | null
  readonly error: string | null
  // ── workspace surfaces ──
  readonly tab: WorkspaceTab
  readonly turnCount: number
  readonly queue: readonly QueueItem[]
  readonly changes: readonly DiffFile[]
  readonly changesKept: boolean
  readonly todos: readonly TodoItem[] | null
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
  readonly workspaceDirectories?: readonly string[]
  readonly contexts?: readonly WorkspaceContext[]
  readonly storageScope?: string
  readonly workspaceBusy: boolean
  readonly workspaceError: string | null
  readonly sessionMenu: SessionMenuState | null
  /** Display choice: show reasoning trails in the activity feed. */
  readonly streamThinking: boolean
  /** The ⌘N new-task modal (mockup 18). */
  readonly taskModalOpen: boolean
  // ── settings data ──
  readonly providers: readonly ProviderRow[]
  readonly providerError: string
  readonly providerSwitching: string | null
  readonly providerSwitchError: string | null
  /** Model catalogs and editable capacities cached for each provider profile. */
  readonly providerModels: Readonly<Record<string, readonly CachedModel[]>>
  readonly providerModelLoading: readonly string[]
  readonly providerModelWarnings: Readonly<Record<string, string>>
  /** The daemon registry's adapter catalog — the add/edit form's dropdown. */
  readonly providerTypes: readonly ProviderTypeRow[]
  /** The daemon's slash catalog — name/description pairs from commands.catalog. */
  readonly commands: readonly { readonly name: string; readonly description: string }[]
  readonly permissionMode: string
  readonly permissionUpdating: boolean
  readonly permissionError: string | null
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

/** Fold a `channel.list` / `channel_status` payload onto the channels state. */
function channelStatusFrom(result: Record<string, unknown>): ChannelStatus {
  const rows = Array.isArray(result.channels) ? result.channels : []
  const channels: ChannelRow[] = rows.flatMap(value => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return []
    const row = value as Record<string, unknown>
    const name = str(row.name).trim()
    if (!name) return []
    return [{
      name,
      adapterName: str(row.adapter_name) || name,
      enabled: row.enabled === true,
      ...(str(row.last_operation) ? { lastOperation: str(row.last_operation) } : {}),
      ...(str(row.last_error) ? { lastError: str(row.last_error) } : {}),
    }]
  })
  return {
    channels,
    available: result.channels_available === true,
    configured: result.channels_configured === true,
  }
}

/** The terminal registry's rows pass the wire camelCase — parse them as-is. */
function terminalRowOf(value: unknown): TerminalRow | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const row = value as Record<string, unknown>
  const id = str(row.id).trim()
  if (!id) return null
  const pid = num(row.pid)
  const endedAt = num(row.endedAt)
  return {
    id,
    kind: str(row.kind) || 'background',
    label: str(row.label),
    command: str(row.command),
    cwd: str(row.cwd),
    ...(pid === null ? {} : { pid }),
    running: row.running === true,
    // Required on the wire (registry rows always carry the start epoch).
    startedAt: Math.max(0, num(row.startedAt) ?? 0),
    ...(endedAt === null ? {} : { endedAt }),
    exitCode: typeof row.exitCode === 'number' ? row.exitCode : null,
    outputChars: Math.max(0, num(row.outputChars) ?? 0),
    canWrite: row.canWrite === true,
    canInterrupt: row.canInterrupt === true,
    canKill: row.canKill === true,
  }
}

function terminalsFromResult(result: Record<string, unknown>): TerminalRow[] {
  const rows = Array.isArray(result.terminals) ? result.terminals : []
  return rows.map(terminalRowOf).filter((row): row is TerminalRow => row !== null)
}

function terminalDetailOf(value: unknown): TerminalDetail | null {
  const row = terminalRowOf(value)
  if (!row) return null
  const record = (value ?? {}) as Record<string, unknown>
  return {
    ...row,
    output: str(record.output),
    outputTruncated: record.outputTruncated === true,
  }
}

function searchHitOf(value: unknown): SessionSearchHit | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const row = value as Record<string, unknown>
  const sessionId = str(row.session_id).trim()
  if (!sessionId) return null
  return {
    sessionId,
    messageIndex: Math.max(0, num(row.message_index) ?? 0),
    role: str(row.role),
    excerpt: str(row.excerpt),
    title: str(row.title),
    updatedAt: str(row.updated_at),
  }
}

function searchStatsOf(value: unknown): SessionSearchStats | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null
  const row = value as Record<string, unknown>
  return {
    sessions: Math.max(0, num(row.sessions) ?? 0),
    indexedMessages: Math.max(0, num(row.indexed_messages) ?? 0),
    searchableMessages: Math.max(0, num(row.searchable_messages) ?? 0),
  }
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
  private updatingRuntime = false
  private terminalsLoadVersion = 0
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
  private historyBefore: string | null = null
  private historyGeneration = 0
  private historySeen = new Set<string>()
  private legacyHistory: Block[] = []
  private historyActions: HistoryPage['actions'] = []
  private historyBlockIds = new Map<string, number>()
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
  /** Latest session-search request; stale responses must not win. */
  private searchSeq = 0
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
      sessionOpenRevision: 0,
      currentTitle: '',
      sessionKey: '',
      goal: '',
      approval: null,
      question: null,
      planMode: false,
      turnActive: false, networkRetrying: false,
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
      todos: null,
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
      workspaceBusy: false,
      workspaceError: null,
      sessionMenu: null,
      streamThinking: readStreamThinking(),
      taskModalOpen: false,
      providers: [],
      providerError: '',
      providerSwitching: null,
      providerSwitchError: null,
      providerModels: {},
      providerModelLoading: [],
      providerModelWarnings: {},
      providerTypes: [],
      commands: [],
      permissionMode: '',
      permissionUpdating: false,
      permissionError: null,
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
    void Promise.all([Promise.resolve(gate), this.bridge.getContextScope?.()]).then(([saved, scope]) => {
      this.patch({ storageScope: scope && scope !== 'local' ? scope + ':' : '' })
      if (saved === null) {
        this.patch({ noWorkspace: true, connection: 'offline' })
        return
      }
      this.initializeLive(typeof saved === 'string' ? saved : '')
    }).catch(error => this.wentOffline(error))
    void this.refreshContexts()
    this.heartbeat = setInterval(() => void this.beat(), HEARTBEAT_MS)
    this.heartbeat.unref?.()
  }

  private initializeLive(workspace: string): void {
    this.patch({ cwd: workspace })
    void Promise.resolve(this.bridge.getResumeSession?.()).then(async explicitId => {
      const id = explicitId || (workspace ? selectedSession((this.frame.storageScope ?? '') + workspace) : null)
      try {
        return await this.initializeSelfHealing(id ? { resume_session_id: id } : {})
      } catch (error) {
        // Old desktop builds could save a session under the wrong workspace.
        // Keep the daemon's isolation check: never move or rebind that history.
        if (!id || connectionFailureKind(error instanceof Error ? error.message : String(error)) !== 'session') throw error
        this.sessionKey = `desktop-${Math.random().toString(36).slice(2, 10)}`
        const result = await this.initialize({})
        this.builder.push('notification', { severity: 'warning', message: 'The saved session belongs to another workspace. Opened a new session here; the original conversation is unchanged and remains available in its workspace.' })
        this.notify()
        return result
      }
    }).then(
      () => {
        void this.refreshGoal()
        void this.refreshSessions()
      },
      error => this.wentOffline(error),
    )
  }

  private refreshingContexts = false
  private async refreshContexts(): Promise<void> {
    if (!this.bridge.getContexts || this.refreshingContexts) return
    this.refreshingContexts = true
    try { this.patch({ contexts: await this.bridge.getContexts() }) }
    catch (error) { this.patch({ workspaceError: desktopError(error) }) }
    finally { this.refreshingContexts = false }
  }

  async activateContext(id: number, sessionId?: string): Promise<void> {
    try { await this.bridge.activateContext?.(id, sessionId) }
    catch (error) { this.patch({ error: desktopError(error), workspaceError: desktopError(error) }) }
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
  async restartDaemon(allowLegacy = true): Promise<void> {
    if (this.updatingRuntime) return
    if (this.frame.daemonWarning?.startsWith('The app is older')) {
      this.patch({ runtimeUpdate: 'failed', runtimeUpdateMessage: 'Update and relaunch the desktop app. Its bundled runtime is older than the running runtime.' })
      return
    }
    this.updatingRuntime = true
    this.patch({ runtimeUpdate: 'checking', runtimeUpdateMessage: 'Checking for running work…' })
    try {
      const result = await this.bridge.call('desktop.restartRuntime', { session_key: this.sessionKey, allow_legacy: allowLegacy })
      if (result.ok !== true) {
        if (result.busy === true) {
          this.patch({ runtimeUpdate: 'waiting', runtimeUpdateMessage: 'Update queued. It will install automatically when all running work finishes.' })
          return
        }
        throw new Error(str(result.error) || 'This runtime cannot update automatically. Restart the workspace runtime from its terminal, then reopen the app.')
      }
      this.patch({ runtimeUpdate: 'restarting', runtimeUpdateMessage: 'Reconnecting to the updated runtime…' })
      await this.initialize(this.frame.currentId ? { resume_session_id: this.frame.currentId } : {})
      if (this.frame.daemonWarning) throw new Error('The runtime restarted, but its build still differs. Quit and relaunch the latest app; check any custom runtime path.')
      this.patch({ runtimeUpdate: undefined, runtimeUpdateMessage: undefined })
    } catch (error) {
      this.patch({ runtimeUpdate: 'failed', runtimeUpdateMessage: error instanceof Error ? error.message : String(error) })
    } finally {
      this.updatingRuntime = false
    }
  }

  private sessionNavigation: Promise<void> = Promise.resolve()
  private sessionNavigationVersion = 0
  private openingSession = false
  private sessionNavigationNeedsRestore = false

  openSession(id: string): Promise<void> {
    const version = ++this.sessionNavigationVersion
    this.openingSession = true
    const pending = this.sessionNavigation.then(async () => {
      if (version !== this.sessionNavigationVersion) return
      try { await this.openSessionNow(id, version) }
      finally {
        if (version === this.sessionNavigationVersion) {
          this.openingSession = false
          if (this.frame.connection === 'connecting') this.patch({ connection: 'online' })
        }
      }
    })
    this.sessionNavigation = pending
    return pending
  }

  private async openSessionNow(id: string, version: number): Promise<void> {
    try {
      const row = [...this.frame.sessions, ...this.frame.live].find(session => session.id === id)
      if (this.frame.turnActive) {
        if (id === this.frame.currentId) return
        if (!this.bridge.openWorkspaceWindow) throw new Error('This desktop build cannot open another session window. Relaunch the updated app.')
        await this.bridge.openWorkspaceWindow(row?.cwd || this.frame.cwd, id)
        return
      }
      if (row?.cwd && row.cwd !== this.frame.cwd) {
        if (!this.bridge.useWorkspace) throw new Error('This host cannot switch workspaces. Open the session from its project folder.')
        await this.bridge.useWorkspace(row.cwd, id)
        return
      }
      // A resume re-keys the connection, but NOT to a string of our own
      // choosing: the daemon binds a resumed session under the session id
      // (resume_session_id wins over session_key — see its 'ignored-slot'
      // contract test), and every later submit/steer/cancel sends the key
      // explicitly. Adopt the key the daemon actually bound, or the next
      // message silently lands in a fresh, context-free session.
      this.patch({ connection: 'connecting' })
      const result = await this.bridge.call('initialize', {
        history_limit: 100,
        session_key: `${this.sessionKey}-r${id.slice(-8)}`,
        resume_session_id: id,
        ...clientHandshake(),
      })
      if (version !== this.sessionNavigationVersion) {
        if (result.ok !== false) this.sessionNavigationNeedsRestore = true
        return
      }
      if (result.ok === false) throw new Error(str(result.error) || 'Session initialization was rejected')
      this.sessionNavigationNeedsRestore = false
      this.applyInitializedSession(result, { resume_session_id: id }, true)
      this.patch({ sessionOpenRevision: this.frame.sessionOpenRevision + 1 })
      void this.refreshGoal()
      void this.refreshSessions()
    } catch (error) {
      if (version !== this.sessionNavigationVersion) return
      if (this.sessionNavigationNeedsRestore && this.frame.currentId) {
        try {
          await this.initialize({ resume_session_id: this.frame.currentId })
          this.sessionNavigationNeedsRestore = false
        } catch (restoreError) {
          this.fail(restoreError)
          return
        }
      }
      if (this.frame.turnActive) this.patch({ error: error instanceof Error ? error.message : String(error) })
      else this.fail(error)
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
   * a session that exists; false means the rebind failed and the previous
   * conversation remains selected with the error visible.
   */
  private beginFreshTask(agentPreset?: string): Promise<boolean> {
    if (this.openingSession || this.frame.turnActive || this.frame.connection !== 'online') return Promise.resolve(false)
    const sessionKey = `desktop-${Math.random().toString(36).slice(2, 10)}`
    this.openingFreshTask = true
    this.patch({ connection: 'connecting' })
    return this.initialize({ session_key: sessionKey, ...(agentPreset ? { agent_id: agentPreset } : {}) })
      .then(() => {
        void this.refreshGoal()
        void this.refreshSessions()
        return true
      })
      .catch(error => {
        this.patch({ connection: 'online' })
        this.fail(error)
        return false
      })
      .finally(() => { this.openingFreshTask = false })
  }

  approve(requestId: string, response: ApprovalResponse): void {
    const sessionKey = this.sessionKey
    const sessionId = this.frame.currentId
    const isCurrent = () => this.sessionKey === sessionKey && this.frame.currentId === sessionId
      && this.frame.approval?.id === requestId
    // The daemon's vocabulary is approve / approve_for_session / reject —
    // anything else resolves as reject, so map our UI labels here.
    const wire = response === 'allow_once' ? 'approve'
      : response === 'allow_session' ? 'approve_for_session'
      : 'reject'
    void this.bridge
      .call('permission_response', { request_id: requestId, response: wire })
      .then(result => {
        if (!isCurrent()) return
        // Clear only on confirmation: a refused response (unknown id, another
        // connection owns it) leaves the request pending daemon-side, and
        // dropping the card would strand it with no surface to answer.
        if (result.ok === false) {
          this.builder.push('notification', { severity: 'error', message: str(result.error) || 'approval refused' })
          this.notify()
        } else {
          this.patch({ approval: null })
        }
      })
      .catch(error => {
        if (!isCurrent()) return
        this.fail(error)
      })
  }

  answerQuestion(requestId: string, answers: Record<string, string>): void {
    const sessionKey = this.sessionKey
    const sessionId = this.frame.currentId
    const isCurrent = () => this.sessionKey === sessionKey && this.frame.currentId === sessionId
      && this.frame.question?.requestId === requestId
    void this.bridge
      .call('question_response', { request_id: requestId, answers })
      .then(result => {
        if (!isCurrent()) return
        if (result.ok === false) {
          this.builder.push('notification', { severity: 'error', message: str(result.error) || 'answer refused' })
          this.notify()
        } else {
          this.patch({ question: null })
        }
      })
      .catch(error => {
        if (!isCurrent()) return
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
   * reloads onto the new credentials). Apply its model to the original
   * session without reopening it. Refused mid-turn like provider switching.
   */
  async saveProvider(profile: {
    name: string
    baseUrl: string
    model: string
    provider?: string
    apiKey?: string
  }): Promise<string | null> {
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
      return knownDefault ? 'Name and model are required.' : 'Name, base URL, and model are required.'
    }
    if (this.frame.turnActive) return 'Wait for the current turn to finish before changing providers.'
    const { sessionKey, isCurrent } = this.captureSessionRequest()
    const params: Record<string, unknown> = { name, model }
    if (baseUrl) params.base_url = baseUrl
    if (provider) params.provider = provider
    if (profile.apiKey?.trim()) params.api_key = profile.apiKey.trim()
    return this.bridge
      .call('provider_save', params)
      .then(async result => {
        if (result.ok === false) {
          if (isCurrent()) this.builder.push('notification', {
            severity: 'error',
            message: str(result.error) || 'provider save refused',
          })
          this.notify()
          return str(result.error) || 'Provider save refused.'
        }
        const saved = (result.profile && typeof result.profile === 'object'
          ? result.profile
          : {}) as Record<string, unknown>
        // The save is durable even if applying its model fails. Refresh the
        // profile list and report the two outcomes separately in the editor.
        void this.loadProviders()
        this.loadModels(true)
        try {
          const applied = await this.bridge.call('set_model', {
            session_key: sessionKey, model: str(saved.model) || model, provider_profile: name,
          })
          if (applied.ok === false) throw new Error(str(applied.error) || 'Model change was refused')
          if (isCurrent()) {
            this.patch({ model: str(applied.model) || str(saved.model) || model })
            this.builder.push('notification', { severity: 'info', message: `provider \`${name}\` saved and activated` })
            this.notify()
          }
        } catch (error) {
          return `Provider saved, but its model could not be applied to the original chat: ${desktopError(error)}`
        }
        return null
      })
      .catch(error => { if (isCurrent()) this.fail(error); return desktopError(error) })
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
  async completeText(text: string): Promise<{ value: string; label: string; meta: string; kind: 'command' | 'skill' }[]> {
    const queries = /^\/[^\s]*$/.test(text) && text !== '/skill'
      ? [text, '/skill ' + text.slice(1)] : [text]
    const results = await Promise.all(queries.map(text => this.bridge.call('complete', { text })))
    const completions = results.flatMap(result => Array.isArray(result.completions) ? result.completions : [])
    return completions
      .map(raw => {
        const entry = (raw ?? {}) as Record<string, unknown>
        const value = str(entry.value)
        return { value, label: str(entry.label) || value, meta: str(entry.meta), kind: (entry.category !== undefined ? 'command' : 'skill') as 'command' | 'skill' }
      })
      .filter((entry, index, entries) => entry.value && entries.findIndex(other => other.value === entry.value) === index)
  }

  /**
   * Ask the shell for a workspace folder. On a pick the shell retargets its
   * daemon and reloads this window — the renderer's job ends at the request.
   */
  chooseWorkspace(): Promise<void> {
    return this.requestWorkspace(async () => {
      if (!this.bridge.chooseWorkspace) throw new Error('This host cannot open a workspace folder.')
      await this.bridge.chooseWorkspace()
    })
  }

  openWorkspaceWindow(directory?: string): Promise<void> {
    return this.requestWorkspace(async () => {
      if (!this.bridge.openWorkspaceWindow) throw new Error('This desktop build cannot open additional windows. Relaunch the updated app.')
      await this.bridge.openWorkspaceWindow(directory)
    })
  }

  /** Enter a known workspace folder (sidebar header, switcher menu) — retargets the daemon. */
  enterWorkspace(cwd: string): Promise<void> {
    if (!cwd) return Promise.resolve()
    return this.requestWorkspace(async () => {
      if (!this.bridge.useWorkspace) throw new Error('This host cannot switch workspace folders.')
      await this.bridge.useWorkspace(cwd)
    })
  }

  clearWorkspaceError(): void {
    this.patch({ workspaceError: null })
  }

  private async requestWorkspace(action: () => Promise<void>): Promise<void> {
    if (this.frame.workspaceBusy) return
    this.patch({ wsMenuOpen: false, workspaceBusy: true, workspaceError: null })
    try {
      await action()
    } catch (error) {
      this.patch({ workspaceError: desktopError(error) })
    } finally {
      this.patch({ workspaceBusy: false })
    }
  }

  /** A late session-scoped reply must never update a newly selected chat. */
  private captureSessionRequest(): { sessionKey: string; isCurrent: () => boolean } {
    const sessionKey = this.sessionKey, sessionId = this.frame.currentId, navigation = this.sessionNavigationVersion
    return { sessionKey, isCurrent: () => this.sessionKey === sessionKey && this.frame.currentId === sessionId && this.sessionNavigationVersion === navigation }
  }

  pickModel(modelId: string): void {
    if (!modelId) return
    // Same mid-turn refusal as provider switching: hot-swapping the model a
    // running turn is riding on is refused daemon-side too.
    if (this.frame.turnActive) return
    const { sessionKey, isCurrent } = this.captureSessionRequest()
    // Route through the daemon's /model handler: it pins the choice to this
    // session AND persists it as the active profile's model. The old
    // initialize({ model }) path only reloaded runtime memory, so every
    // daemon restart silently fell back to the profile's stored model.
    void this.bridge.call('slash', { command: `/model ${modelId}`, session_key: sessionKey }).then(result => {
      if (!isCurrent()) return
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
    }).catch(error => { if (isCurrent()) this.fail(error) })
  }

  /**
   * Switch the daemon's active provider profile and adopt its model here.
   * Refused mid-turn: provider_select swaps the live credentials the
   * in-flight request is already riding on.
   */
  selectProvider(name: string): void {
    if (this.frame.turnActive || this.frame.providerSwitching) return
    const target = this.frame.providers.find(row => row.name === name)
    if (!target || target.active) return
    const { isCurrent } = this.captureSessionRequest()
    this.patch({ providerSwitching: name, providerSwitchError: null })
    void this.bridge
      .call('provider_select', { name })
      .then(result => {
        if (result.ok === false) {
          this.patch({ providerSwitchError: str(result.error) || 'Provider switch was refused. Select a provider to retry.' })
          return
        }
        // provider_select applies the model to its session and emits status.
        // Reinitializing here would discard loaded history and could bind a
        // different chat if navigation happened while the request was pending.
        if (isCurrent() && target.model) this.patch({ model: target.model })
        void this.loadProviders()
        this.loadModels(true)
      })
      .catch(error => { this.patch({ providerSwitchError: desktopError(error) }) })
      .finally(() => { this.patch({ providerSwitching: null }) })
  }

  /** Toggle the session's plan mode on the daemon (plan = read-only ceiling). */
  setPlanMode(next: boolean): void {
    const { sessionKey, isCurrent } = this.captureSessionRequest()
    void this.bridge
      .call('set_plan_mode', { session_key: sessionKey, enabled: next })
      .then(result => {
        if (!isCurrent()) return
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
      .catch(error => { if (isCurrent()) this.fail(error) })
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
    if (tab === 'channels') this.loadChannels()
    if (tab === 'terminals') this.loadTerminals()
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
    if (tab === 'channels') this.loadChannels()
    if (tab === 'terminals') this.loadTerminals()
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
    const { sessionKey, isCurrent } = this.captureSessionRequest()
    this.patch({
      reasoningPickerOpen: true,
      settingsOpen: false,
      paletteOpen: false,
      pickerOpen: false,
      modelMenuOpen: false,
      wsMenuOpen: false,
      reasoningLoading: true,
    })
    void this.bridge.call('reasoning_levels', { session_key: sessionKey }).then(result => {
      if (!isCurrent()) return
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
      if (!isCurrent()) return
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
    const { sessionKey, isCurrent } = this.captureSessionRequest()
    this.patch({ reasoningPickerOpen: false })
    const trimmed = effort.trim()
    if (!trimmed) return
    void this.bridge.call('slash', { command: `/thinking ${trimmed}`, session_key: sessionKey }).then(result => {
      if (!isCurrent()) return
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
    }).catch(error => { if (isCurrent()) this.fail(error) })
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
    const { sessionKey, isCurrent } = this.captureSessionRequest()
    this.patch({
      contextMenuOpen: true,
      modelMenuOpen: false,
      pickerOpen: false,
      reasoningPickerOpen: false,
      paletteOpen: false,
      wsMenuOpen: false,
      contextBreakdownLoading: true,
    })
    void this.bridge.call('context_breakdown', { session_key: sessionKey }).then(result => {
      if (!isCurrent()) return
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
      if (!isCurrent()) return
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
    this.loadModels()
  }

  closeTaskModal(): void {
    this.patch({ taskModalOpen: false })
  }

  /**
   * Start the task the modal collected: fresh session, plan ceiling applied
   * to THAT session, then the objective submitted — in this order, so the
   * ceiling and the first message land on the session they belong to.
   */
  async startTask(objective: string, planFirst: boolean, agentPreset?: string, model?: string): Promise<void> {
    if (this.frame.turnActive || this.frame.connection !== 'online') return
    this.patch({ error: null })
    const bound = await this.beginFreshTask(agentPreset)
    if (!bound) return
    try {
      if (model) {
        const result = await this.bridge.call('slash', { session_key: this.sessionKey, command: `/model ${model}` })
        if (result.ok === false) throw new Error(str(result.error) || 'Model change rejected')
        this.patch({ model: str(result.model) || model })
      }
      if (planFirst !== this.frame.planMode) {
        const result = await this.bridge.call('set_plan_mode', { session_key: this.sessionKey, enabled: planFirst })
        if (result.ok === false) throw new Error(str(result.error) || 'Plan mode could not be enabled')
        this.patch({ planMode: planFirst })
      }
      this.patch({ taskModalOpen: false })
      const text = objective.trim()
      if (text) await this.submit(text)
    } catch (error) { this.fail(error) }
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
    if (anchor?.pending) return
    if (!clean) { this.closeSessionMenu(); return }
    const pending = anchor ? { ...anchor, title: clean, renaming: true, pending: true, error: '' } : null
    if (pending) this.patch({ sessionMenu: pending })
    try {
      await desktopCall(this.bridge, key, 'session.title', { title: clean })
      if (this.frame.sessionMenu === pending) this.closeSessionMenu()
      this.refreshSessions()
    } catch (error) {
      // Preserve the attempted title, but never resurrect a dismissed menu.
      if (pending && this.frame.sessionMenu === pending) {
        this.patch({ sessionMenu: { ...pending, pending: false, error: desktopError(error) } })
      } else if (!anchor) this.fail(new Error(`Rename failed: ${desktopError(error)}`))
    }
  }

  /**
   * Export a session transcript as a markdown download (mockup 08). The
   * record comes from the daemon's session.status; the fold is the same one
   * the replay renders, so the file shows what the transcript shows.
   */
  async exportSessionTranscript(key: string): Promise<void> {
    this.patch({ sessionMenu: null })
    try {
      await this.downloadSessionTranscript(key)
    } catch (error) {
      this.fail(new Error(`export failed: ${String(error)}`))
    }
  }

  /** Export for a surface that owns its own pending and error presentation. */
  async downloadSessionTranscript(key: string): Promise<void> {
    if (typeof document === 'undefined') throw new Error('Transcript downloads require a desktop or browser window')
    const result = await desktopCall(this.bridge, key, 'session.status')
    const record = (this.sessionOf(result) ?? result) as unknown as ExportSession
    const markdown = sessionToMarkdown(record)
    const anchor = document.createElement('a')
    anchor.href = URL.createObjectURL(new Blob([markdown], { type: 'text/markdown' }))
    anchor.download = `${(str(record.title) || str(record.id) || 'session').replace(/[^\w-]+/g, '-').slice(0, 64) || 'session'}.md`
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    setTimeout(() => URL.revokeObjectURL(anchor.href), 5_000)
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
   * the shared daemon binds the next task to the isolated checkout.
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

  async retryCompaction(): Promise<void> {
    const key = this.sessionKey
    const failed = this.frame.failed
    if (!failed || this.frame.turnActive) return
    try {
      const result = await this.bridge.call('slash', { session_key: key, command: '/compact' })
      if (key !== this.sessionKey || this.frame.failed !== failed) return
      if (result.ok !== true) throw new Error(str(result.error) || str(result.output) || 'Compaction failed')
      this.resolveFailure()
      this.builder.push('notification', { severity: 'info', message: str(result.output) || 'Conversation compacted. You can now continue the task.' })
      this.notify()
    } catch (error) {
      if (key !== this.sessionKey || this.frame.failed !== failed) return
      this.failure = { ...failed, error: `Automatic context compaction failed: ${error instanceof Error ? error.message : String(error)}. Original conversation retained.` }
      this.patch({ failed: this.failure })
    }
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

  clearAgentPresetError(): void {
    this.patch({ agentPresetsError: null })
  }

  async loadAgentPresets(): Promise<void> {
    this.patch({ agentPresetsError: null })
    try {
      const result = await this.bridge.call('agentPreset.list', {})
      if (result.ok === false) throw new Error(str(result.error) || 'could not load agent presets')
      this.patch({ agentPresets: agentPresetsFromResult(result) })
    } catch (error) {
      this.patch({ agentPresetsError: desktopError(error) })
    }
  }

  async setDefaultAgentPreset(id: string): Promise<void> {
    this.patch({ agentPresetsError: null })
    try {
      const result = await this.bridge.call('agentPreset.setDefault', { agent_preset: id })
      if (result.ok === false) throw new Error(str(result.error) || 'could not set default agent preset')
      await this.loadAgentPresets()
    } catch (error) {
      this.patch({ agentPresetsError: desktopError(error) })
    }
  }

  async copyAgentPreset(from: string, id: string, name?: string): Promise<boolean> {
    this.patch({ agentPresetsError: null })
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
      this.patch({ agentPresetsError: desktopError(error) })
      return false
    }
  }

  async removeAgentPreset(id: string): Promise<void> {
    this.patch({ agentPresetsError: null })
    try {
      const result = await this.bridge.call('agentPreset.remove', { agent_preset: id })
      if (result.ok === false) throw new Error(str(result.error) || 'could not remove agent preset')
      await this.loadAgentPresets()
    } catch (error) {
      this.patch({ agentPresetsError: desktopError(error) })
    }
  }

  async readAgentPreset(id: string): Promise<string> {
    const result = await this.bridge.call('agentPreset.read', { agent_preset: id })
    if (result.ok === false) throw new Error(str(result.error) || 'could not read agent preset')
    return str(result.content)
  }

  /** Persist an edited preset body (`agentPreset.write`); the daemon revalidates on save. */
  async writeAgentPreset(id: string, content: string): Promise<boolean> {
    this.patch({ agentPresetsError: null })
    try {
      const result = await this.bridge.call('agentPreset.write', { agent_preset: id, content })
      if (result.ok === false) throw new Error(str(result.error) || 'could not save agent preset')
      await this.loadAgentPresets()
      return true
    } catch (error) {
      this.patch({ agentPresetsError: desktopError(error) })
      return false
    }
  }

  /** Bind an existing preset to the current session (`agentPreset.select`). */
  async selectAgentPreset(id: string): Promise<boolean> {
    if (!this.frame.currentId || this.frame.turnActive) return false
    this.patch({ agentPresetsError: null })
    try {
      const result = await this.bridge.call('agentPreset.select', {
        agent_preset: id,
        session_key: this.sessionKey,
      })
      if (result.ok === false) throw new Error(str(result.error) || 'could not select agent preset')
      const applied = str(result.agent_preset)
      if (applied) this.patch({ currentAgentPreset: applied })
      return true
    } catch (error) {
      this.patch({ agentPresetsError: desktopError(error) })
      return false
    }
  }

  async openAgentPresetLocation(id: string): Promise<void> {
    this.patch({ agentPresetsError: null })
    try {
      const result = await this.bridge.call('agentPreset.openDocument', { agent_preset: id })
      if (result.ok === false) throw new Error(str(result.error) || 'could not open agent preset')
      const path = str(result.path)
      if (!path) throw new Error('The runtime did not return a preset location')
      if (!window.xerxes.openPath) throw new Error('Opening preset folders requires the desktop host')
      if (!await window.xerxes.openPath(path)) throw new Error('The system file manager could not open this preset folder')
    } catch (error) {
      this.patch({ agentPresetsError: desktopError(error) })
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

  // ── Channels ─────────────────────────────────────────────────────────

  /** Load gateway status; flags stay truthful even when the manager is absent (`ok:false`). */
  loadChannels(): void {
    void this.bridge
      .call('channel.list', {})
      .then(result => this.applyChannelStatus(channelStatusFrom(result)))
      .catch(error => this.fail(error))
  }

  private applyChannelStatus(status: ChannelStatus): void {
    this.patch({
      channels: status.channels,
      channelsAvailable: status.available,
      channelsConfigured: status.configured,
    })
  }

  /** Enable or disable one gateway; the response list and the `channel_status` broadcast both refresh the panel. */
  setChannelEnabled(name: string, enabled: boolean): Promise<void> {
    const method = enabled ? 'channel.enable' : 'channel.disable'
    return this.bridge
      .call(method, { name })
      .then(result => {
        if (result.ok === false) throw new Error(str(result.error) || `could not ${enabled ? 'enable' : 'disable'} ${name}`)
        // Fast path: the response carries the refreshed channel LIST but no
        // availability flags — those ride the `channel_status` broadcast the
        // daemon emits right after, or the quiet reload below.
        const status = channelStatusFrom(result)
        this.patch({ channels: status.channels })
        this.loadChannels()
      })
  }

  // ── Terminals ────────────────────────────────────────────────────────

  loadTerminals(): void {
    const sessionKey = this.sessionKey
    const version = ++this.terminalsLoadVersion
    const current = () => sessionKey === this.sessionKey && version === this.terminalsLoadVersion
    this.patch({ terminalsLoading: true, terminalsError: null })
    void this.bridge
      .call('terminal.list', { session_key: sessionKey })
      .then(result => {
        if (!current()) return
        if (result.ok === false) throw new Error(str(result.error) || 'Could not list terminals')
        this.patch({ terminals: terminalsFromResult(result), terminalsLoading: false })
      })
      .catch(error => {
        if (!current()) return
        this.patch({ terminalsLoading: false, terminalsError: desktopError(error) })
      })
  }

  /** Read one terminal's retained output tail (`terminal.inspect`). */
  async inspectTerminal(id: string): Promise<TerminalDetail | null> {
    const result = await this.bridge.call('terminal.inspect', {
      terminal_id: id,
      session_key: this.sessionKey,
      max_output_chars: 24_000,
    })
    if (result.ok === false) throw new Error(str(result.error) || 'could not inspect terminal')
    return terminalDetailOf(result.terminal)
  }

  /** Send input to, interrupt, or kill one terminal (`terminal.control`). */
  async controlTerminal(id: string, action: 'write' | 'interrupt' | 'kill', chars?: string): Promise<void> {
    const params: Record<string, unknown> = {
      action,
      terminal_id: id,
      session_key: this.sessionKey,
      ...(action === 'write' ? { chars: chars ?? '' } : {}),
    }
    await this.bridge
      .call('terminal.control', params)
      .then(result => {
        if (result.ok === false) throw new Error(str(result.error) || `terminal ${action} refused`)
        // The control result echoes the inspected terminal — fold its fresh
        // state (running flag, output size, exit code) into the open list.
        const detail = terminalDetailOf(result.terminal)
        this.patch({
          terminals: detail
            ? this.frame.terminals.map(row => (row.id === id ? detail : row))
            : this.frame.terminals,
        })
      })
  }

  // ── Session search ───────────────────────────────────────────────────

  openSessionSearch(): void {
    this.patch({
      searchOpen: true,
      paletteOpen: false,
      settingsOpen: false,
      taskModalOpen: false,
    })
  }

  closeSessionSearch(): void {
    this.patch({ searchOpen: false })
  }

  /**
   * Ask the daemon's transcript FTS (`session.search`). Stale answers drop:
   * a sequence counter makes the slowest response for an earlier needle
   * lose to the latest one.
   */
  runSessionSearch(query: string): void {
    const needle = query.trim()
    if (needle.length < 2) {
      this.searchSeq += 1
      this.patch({ searchResults: [], searchStats: null, searchSearching: false, searchError: null })
      return
    }
    const seq = ++this.searchSeq
    this.patch({ searchSearching: true, searchError: null })
    void this.bridge
      .call('session.search', { query: needle, limit: 24 })
      .then(result => {
        if (seq !== this.searchSeq) return
        if (result.ok === false) {
          this.patch({
            searchResults: [],
            searchStats: null,
            searchSearching: false,
            searchError: str(result.error) || 'search failed',
          })
          return
        }
        const rows = Array.isArray(result.results) ? result.results : []
        this.patch({
          searchResults: rows.map(searchHitOf).filter((row): row is SessionSearchHit => row !== null),
          searchStats: searchStatsOf(result.stats),
          searchSearching: false,
          searchError: null,
        })
      })
      .catch(error => {
        if (seq !== this.searchSeq) return
        this.patch({
          searchResults: [],
          searchStats: null,
          searchSearching: false,
          searchError: error instanceof Error ? error.message : String(error),
        })
      })
  }

  async loadProviders(): Promise<void> {
    this.patch({ providerError: '' })
    const providers = this.bridge
      .call('provider_list', {})
      .then(result => {
        if (result.ok === false) throw new Error(str(result.error) || 'Could not load provider profiles')
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
      .catch(error => { this.patch({ providerError: error instanceof Error ? error.message : String(error) }); return null })
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
    await Promise.all([providers, types, status]).then(([rows, typeRows, state]) => {
      if (rows) this.patch({ providers: rows })
      if (typeRows) this.patch({ providerTypes: typeRows })
      // Daemon-wide fallback only: a session-scoped /permissions pin from
      // initialize is the truth for what this app's next tool call faces.
      if (state?.permissionMode && !this.frame.permissionMode) this.patch({ permissionMode: state.permissionMode })
    })
  }

  setPermissionMode(mode: PermissionMode): void {
    if (this.frame.permissionUpdating) return
    const { sessionKey, isCurrent } = this.captureSessionRequest()
    this.patch({ permissionUpdating: true, permissionError: null })
    void this.bridge
      .call('slash', { command: `/permissions ${mode}`, session_key: sessionKey })
      .then(result => {
        if (!isCurrent()) return
        if (result.ok === false) {
          this.patch({ permissionError: str(result.error) || 'Permission change was refused. The previous mode is still active.' })
          return
        }
        // The slash result carries the pinned mode; adopt it immediately so
        // the card's ✓ marker follows the click instead of waiting for a
        // re-initialize that may never come.
        const pinned = str(result.permission_mode)
        if (result.ok !== false && pinned) this.patch({ permissionMode: pinned })
        void this.loadProviders()
      })
      .catch(error => {
        if (!isCurrent()) return
        this.patch({ permissionError: desktopError(error) })
      })
      .finally(() => { if (isCurrent()) this.patch({ permissionUpdating: false }) })
  }

  // ── Connection ───────────────────────────────────────────────────────

  private reconnecting = false
  private openingFreshTask = false
  private supportsConnectionLease = false
  async retryConnection(): Promise<void> {
    if (this.reconnecting || this.openingFreshTask || this.openingSession) return
    this.reconnecting = true
    this.patch({ connection: 'connecting' })
    // Resume the open conversation by id when one exists — a bare
    // initialize evicts the live session and the daemon would hand back a
    // fresh, context-free one under our key.
    const extra: Record<string, unknown> = this.frame.currentId
      ? { resume_session_id: this.frame.currentId }
      : {}
    await this.initializeSelfHealing(extra).then(
      () => {
        void this.refreshGoal()
        void this.refreshSessions()
      },
      error => this.wentOffline(error),
    ).finally(() => { this.reconnecting = false })
  }

  private cameOnline(): void {
    if (this.frame.connection !== 'online') this.patch({ connection: 'online', error: null })
  }

  private wentOffline(error?: unknown): void {
    if (error !== undefined) this.patch({ error: error instanceof Error ? error.message : String(error) })
    if (this.supportsConnectionLease) {
      this.patch({ connection: 'offline' })
      return
    }
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
    void this.refreshContexts()
    if (this.frame.noWorkspace) return
    void this.bridge.getWorkspaceDirectories?.().then(workspaceDirectories => this.patch({ workspaceDirectories })).catch(error => this.patch({ workspaceError: desktopError(error) }))
    if (this.updatingRuntime || this.openingFreshTask || this.openingSession) return
    if (this.frame.connection === 'online') {
      this.refreshSessions()
      if (this.frame.daemonWarning && !this.frame.daemonWarning.startsWith('The app is older') && this.frame.runtimeUpdate !== 'failed') {
        await this.restartDaemon(false)
      }
      // Events would still be flowing; a silent socket only shows when a
      // call dies, which every action already routes through fail().
      return
    }
    if (connectionFailureKind(this.frame.error) !== 'transport') return
    await this.retryConnection()
  }

  // ── RPC helpers ──────────────────────────────────────────────────────

  private bridge: XerxesLike = {
    call: (method, params) => window.xerxes.call(method, params),
    getContextScope: () => window.xerxes.getContextScope?.() ?? Promise.resolve('local'),
    getContexts: () => window.xerxes.getContexts?.() ?? Promise.resolve([]),
    activateContext: (id, sessionId) => window.xerxes.activateContext?.(id, sessionId) ?? Promise.reject(new Error('Update the desktop app to switch contexts.')),
    // Passthrough — the wrapper must forward EVERY preload method or the
    // optional calls silently do nothing.
    getWorkspaceDirectories: () => window.xerxes.getWorkspaceDirectories?.() ?? Promise.resolve([]),
    openWorkspaceWindow: (directory, resumeSessionId) => {
      if (!window.xerxes.openWorkspaceWindow) return Promise.reject(new Error('This desktop build cannot open additional windows. Relaunch the updated app.'))
      return window.xerxes.openWorkspaceWindow(directory, resumeSessionId)
    },
    chooseWorkspace: () => window.xerxes.chooseWorkspace?.() ?? Promise.resolve(null),
    useWorkspace: (dir, resumeSessionId) => window.xerxes.useWorkspace?.(dir, resumeSessionId) ?? Promise.resolve(null),
    getWorkspace: () => window.xerxes.getWorkspace?.() ?? Promise.resolve(''),
    getResumeSession: () => window.xerxes.getResumeSession?.() ?? Promise.resolve(null),
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

  private adoptHistory(session: Record<string, unknown>, preserve = false): void {
    const page = readHistoryPage(session.history)
    const previousCursor = this.historyBefore
    this.historyGeneration++
    if (!preserve) this.historyBlockIds.clear()
    this.legacyHistory = []
    if (page) {
      const overlap = preserve && page.actions.length ? this.historyActions.findIndex(action => action.id === page.actions[0]!.id && JSON.stringify(action) === JSON.stringify(page.actions[0])) : -1
      const older = overlap > 0 ? this.historyActions.slice(0, overlap) : []
      this.historyActions = [...older, ...page.actions]
      this.historyBefore = older.length ? previousCursor : page.before
      this.historySeen = new Set(this.historyActions.map(action => action.id))
      this.builder.reset(historyBlocks({ ...page, actions: this.historyActions }, this.historyBlockIds))
    } else {
      // Older runtimes lack server paging. Bound rendering until the runtime
      // can safely update; retain the received legacy data for manual paging.
      const blocks = blocksFromStoredMessages(session.transcript ?? session.messages, { executions: session.tool_executions, thinking: session.thinking_content })
      this.legacyHistory = blocks.slice(0, -100)
      this.builder.reset(blocks.slice(-100))
      this.historyActions = []
      this.historyBefore = null
      this.historySeen.clear()
    }
    this.patch({ historyMore: Boolean(this.historyBefore || this.legacyHistory.length), historyLoading: false, historyError: null })
  }

  async loadOlderHistory(): Promise<void> {
    if (this.frame.historyLoading || !this.frame.historyMore) return
    const generation = this.historyGeneration
    const key = this.sessionKey
    this.patch({ historyLoading: true, historyError: null })
    try {
      if (this.legacyHistory.length) {
        this.builder.prepend(this.legacyHistory.splice(-100))
        this.patch({ historyMore: this.legacyHistory.length > 0 })
      } else {
        const result = await this.bridge.call('session.history', { session_key: key, before: this.historyBefore, history_limit: 100 })
        if (generation !== this.historyGeneration || key !== this.sessionKey) return
        if (result.ok !== true) throw new Error(str(result.error) || 'Could not load older history')
        const page = readHistoryPage(result.history)
        if (!page || page.before === this.historyBefore) throw new Error('History page did not advance')
        if (page.actions.some(action => this.historySeen.has(action.id))) throw new Error('History changed. Reopen this session to load its current history.')
        this.builder.prepend(historyBlocks(page, this.historyBlockIds))
        this.historyActions = [...page.actions, ...this.historyActions]
        for (const action of page.actions) this.historySeen.add(action.id)
        this.historyBefore = page.before
        this.patch({ historyMore: page.has_more })
      }
    } catch (error) {
      if (generation === this.historyGeneration && key === this.sessionKey) this.patch({ historyError: error instanceof Error ? error.message : String(error) })
    } finally {
      if (generation === this.historyGeneration && key === this.sessionKey) this.patch({ historyLoading: false })
    }
  }

  private async initialize(extra: Record<string, unknown>): Promise<Record<string, unknown>> {
    const result = await this.bridge.call('initialize', {
      history_limit: 100,
      session_key: this.sessionKey,
      ...extra,
      ...clientHandshake(),
    })
    if (result.ok === false) throw new Error(str(result.error) || 'Session initialization was rejected')
    return this.applyInitializedSession(result, extra)
  }

  /** One adoption path for startup, reconnect and explicit session navigation. */
  private applyInitializedSession(result: Record<string, unknown>, extra: Record<string, unknown>, explicitNavigation = false): Record<string, unknown> {
    this.supportsConnectionLease = result.connection_lease_supported === true
    const session = this.sessionOf(result)
    const replay = Array.isArray(result.reconnect_events) ? result.reconnect_events : null
    const preserveLiveTranscript = !explicitNavigation && replay !== null && this.frame.currentId === str(result.session_id ?? session.id)
    if (!extra.resume_session_id && typeof extra.session_key === 'string' && extra.session_key !== this.sessionKey) {
      this.sessionKey = str(session.key) || extra.session_key
      this.resetWorkspaceFolds()
      this.patch({ goal: '', approval: null, question: null, failed: null })
    }
    // A resume binds the session under the session id, not our requested
    // key — every session-scoped call below must target what the daemon
    // actually bound or it silently addresses a fresh session.
    const boundKey = str(session.key) || str(result.session_id)
    if (extra.resume_session_id && boundKey) {
      this.sessionKey = boundKey
      if (!preserveLiveTranscript) {
        this.adoptHistory(session, !explicitNavigation && this.frame.currentId === str(result.session_id ?? session.id))
        this.resetWorkspaceFolds()
      }
    }
    if (!extra.resume_session_id) this.adoptHistory(session)
    const reportedContextLimit = num(result.context_limit)
    const contextLimit = reportedContextLimit !== null && reportedContextLimit > 0
      ? reportedContextLimit
      : null
    // The daemon owns turn truth (`active_turn_id` exists only mid-turn).
    // A fold that still believes it is acting after a restart or reconnect
    // would show ▶ act forever with blinking carets on finished messages —
    // reconcile to the daemon and close any stranded live runs.
    const daemonInTurn = str(session.active_turn_id) !== ''
    if (daemonInTurn && !this.frame.turnActive) this.startTurn()
    if (!daemonInTurn && this.frame.turnActive) {
      this.stopTick()
      this.builder.finalize()
    }
    this.patch({
      connection: 'online',
      error: null,
      currentId: str(result.session_id ?? session.id),
      currentTitle: str(session.title),
      sessionKey: this.sessionKey,
      cwd: str(result.cwd ?? session.cwd),
      model: str(result.model ?? session.model),
      planMode: (result.plan_mode ?? session.plan_mode) === true,
      ...(str(result.reasoning_effort)
        ? { reasoningEffort: str(result.reasoning_effort) }
        : {}),
      currentAgentPreset: str(result.agent_name ?? session.agent_id) || this.frame.currentAgentPreset,
      branch: str(result.branch),
      daemonWarning: daemonCompatibilityWarning(result),
      costUsd: typeof result.cost_usd === 'number' ? result.cost_usd : null,
      contextMax: contextLimit,
      approval: null,
      ...(this.frame.currentId !== str(result.session_id ?? session.id) ? { question: null, goal: '' } : {}),
      ...(daemonInTurn ? {} : { turnActive: false, turnSeconds: 0 }),
      // initialize reports THIS session's policy — /permissions pins the
      // mode per session, so the daemon-wide runtime.status would lie about
      // what the next tool call will actually face.
      ...(str(result.permission_mode) ? { permissionMode: str(result.permission_mode) } : {}),
      ...this.telemetryFromSession(session),
    })
    this.adoptFleet(session)
      this.adoptTodos(session)
    if (preserveLiveTranscript) {
      for (const raw of replay ?? []) {
        const frame = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {}
        const params = frame.params && typeof frame.params === 'object' ? frame.params as Record<string, unknown> : {}
        if (frame.method === 'event' && typeof params.type === 'string' && params.payload && typeof params.payload === 'object') {
          this.onEvent({ type: params.type, payload: params.payload as Record<string, unknown> })
        }
      }
    }
    for (const raw of Array.isArray(result.pending_interactions) ? result.pending_interactions : []) {
      if (raw && typeof raw === 'object' && (raw.type === 'approval_request' || raw.type === 'question_request') && raw.payload && typeof raw.payload === 'object') {
        this.onEvent({ type: raw.type, payload: raw.payload as Record<string, unknown> })
      }
    }
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
  private adoptTodos(session: Readonly<Record<string, unknown>>): void {
    const structured = todoItemsOf(session.todos)
    if (structured !== null) { this.patch({ todos: structured }); return }
    if (!Array.isArray(session.tool_executions)) return
    for (const execution of [...session.tool_executions].reverse()) {
      if (!execution || execution.permitted === false || execution.error) continue
      const todos = todosFromResult(execution.name, execution.result ?? execution.return_value)
      if (todos !== null) { this.patch({ todos }); return }
    }
  }

  private adoptFleet(session: Readonly<Record<string, unknown>>): void {
    const raw = Array.isArray(session.subagent_snapshots) ? session.subagent_snapshots : []
    const fleet = raw
      .map(item => {
        const row = (item && typeof item === 'object' ? item : {}) as Record<string, unknown>
        const id = str(row.id)
        if (!id) return null
        const previous = this.frame.fleet.find(agent => agent.id === id)
        const updatedAt = Date.parse(str(row.updated_at))
        const liveWins = previous?.agentDetails?.lastEventAt !== undefined && (!Number.isFinite(updatedAt) || updatedAt <= previous.agentDetails.lastEventAt)
        const label = str(row.title) || str(row.name) || str(row.agent_id) || `#${id.slice(0, 6)}`
        const count = (value: unknown): number | undefined => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined
        const paths = (value: unknown): string[] => Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
        const entry: SessionRow = {
          id,
          key: id,
          title: label,
          status: liveWins ? previous.status : str(row.status) || 'running',
          age: '',
          current: false,
          kind: 'subagent',
          turns: 0,
          messages: 0,
          cwd: '',
          untitled: false,
          agentDetails: {
            ...this.frame.fleet.find(agent => agent.id === id)?.agentDetails,
            summary: str(row.summary), error: str(row.error), model: str(row.model),
            toolCount: count(row.tool_count), inputTokens: count(row.input_tokens), outputTokens: count(row.output_tokens),
            filesRead: paths(row.files_read), filesWritten: paths(row.files_written),
            ...(liveWins ? previous.agentDetails : {}),
          },
        }
        return entry
      })
      .filter(row => row !== null)
    const merged = [...fleet, ...this.frame.fleet.filter(row => row.agentDetails?.lastEventAt !== undefined && !fleet.some(saved => saved.id === row.id))]
    this.patch({ fleet: merged })
    this.syncAgentMembersFromFleet(merged)
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
  async controlAgent(id: string, action: 'stop' | 'retry', message = ''): Promise<string> {
    if (!this.frame.fleet.some(row => row.id === id)) throw new Error('This agent is no longer in the current session')
    const key = this.sessionKey
    const started = Date.now()
    const result = await desktopCall(this.bridge, key, action === 'retry' ? 'subagent.retry' : 'subagent.interrupt', { task: id, ...(message.trim() ? { message: message.trim() } : {}) })
    if (action === 'stop' && result.found !== true) throw new Error('The runtime did not confirm the stop request')
    if (action === 'retry' && result.ok !== true) throw new Error('The runtime did not confirm the retry')
    if (key === this.sessionKey) {
      if (action === 'retry') {
        // An acknowledgement is not a live agent event. Giving it a timestamp
        // can mask a completed snapshot produced before the RPC reply arrives.
        this.patch({ fleet: this.frame.fleet.map(row => row.id === id && (row.agentDetails?.lastEventAt ?? 0) <= started ? { ...row, status: 'running', ...(row.agentDetails ? { agentDetails: { ...row.agentDetails, error: '', summary: '', toolCalls: [], thinking: [], notes: [], startedAt: Date.now(), lastEventAt: undefined } } : {}) } : row) })
        this.syncAgentMembersFromFleet(this.frame.fleet)
        this.startFleetPoll()
      }
      this.refreshFleet()
    }
    return action === 'retry' ? 'Retry accepted. Waiting for agent progress.' : 'Stop requested. Waiting for the runtime to confirm it ended.'
  }

  private refreshFleet(): void {
    const key = this.sessionKey
    const revision = this.frame.sessionOpenRevision
    void this.bridge
      .call('session.status', { session_key: key, history_limit: 0 })
      .then(result => {
        if (key !== this.sessionKey || revision !== this.frame.sessionOpenRevision) return
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

  /** Fetch MCP statuses without presenting an unavailable response as empty. */
  refreshMcpStatus(): Promise<void> {
    return desktopCall(this.bridge, this.sessionKey, 'session.status', { history_limit: 0 })
      .then(result => {
        const raw = (this.sessionOf(result).mcp_status ?? (result as Record<string, unknown>).mcp_status) as unknown
        if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('MCP status is unavailable from this runtime. Check the runtime version and connection.')
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
  }

  /** Reconnect every configured MCP server through the daemon's slash RPC. */
  async reloadMcp(): Promise<void> {
    try {
      await desktopCall(this.bridge, this.sessionKey, 'slash', { command: '/reload-mcp' })
    } catch (reloadError) {
      try { await this.refreshMcpStatus() }
      catch (statusError) {
        throw new AggregateError([reloadError, statusError], `MCP reload failed and current server status could not be refreshed. ${desktopError(reloadError)}; ${desktopError(statusError)}`, { cause: reloadError })
      }
      throw reloadError
    }
    await this.refreshMcpStatus()
  }

  private sessionsRefresh = 0

  private refreshSessions(): void {
    const revision = ++this.sessionsRefresh
    void this.bridge.getWorkspaceDirectories?.().then(workspaceDirectories => this.patch({ workspaceDirectories })).catch(error => this.patch({ workspaceError: desktopError(error) }))
    const saved = this.bridge
      .call('session.list', { kind: 'main', scope: 'global', limit: 60 })
      .then(result => normalize(result.sessions, this.frame.currentId))
      .catch(() => null)
    const active = this.bridge
      .call('session.active_list', { history_limit: 0 })
      .then(result => {
        this.seedBackgroundJobs(result.sessions)
        return normalize(result.sessions, this.frame.currentId)
      })
      .catch(() => null)
    void Promise.all([saved, active]).then(([savedRows, activeRows]) => {
      if (revision !== this.sessionsRefresh) return
      if (!savedRows && !activeRows) return
      // The attached session is not 'fleet' and not a history row — it is
      // what the chat column is already showing.
      const currentId = this.frame.currentId
      const all = activeRows ?? this.frame.live
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
      const history = (savedRows ?? this.frame.sessions).filter(row => !liveIds.has(row.id) && row.id !== currentId)
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
          .call('session.status', { session_key: row.key, history_limit: 0 })
          .then(result => {
            const session = this.sessionOf(result)
            if (typeof session.preview === 'string' && session.preview.trim()) {
              this.snippets = { ...this.snippets, [row.id]: session.preview }
              return
            }
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
    if (type === 'desktop_connection') {
      if (payload.online === false) this.wentOffline(new Error('Connection lost. Reconnecting to your session.'))
      else if (payload.online === true && this.frame.currentId && this.frame.connection !== 'online') this.retryConnection()
      return
    }
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
      case 'subagent_event': {
        const fleet = foldAgentEvent(this.frame.fleet, payload)
        this.patch({ fleet })
        this.syncAgentMembersFromFleet(fleet)
        return
      }
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
        if (type === 'tool_result' && payload.permitted !== false && !payload.error) {
          const todos = todosFromResult(payload.name, payload.return_value)
          if (todos !== null) this.patch({ todos })
        }
        if (type === 'tool_result' && payload.permitted !== false && !payload.error && Array.isArray(payload.display_blocks)) {
          for (const block of payload.display_blocks) {
            if (block?.type !== 'todo') continue
            const todos = todoItemsOf(block.items)
            if (todos !== null) this.patch({ todos })
          }
        }
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
      case 'channel_status': {
        // Daemon-wide broadcast (no session tag): any enable/disable — from
        // this shell, the TUI, or the webhook itself — keeps the panel true.
        this.applyChannelStatus(channelStatusFrom(payload))
        break
      }
      case 'status_update': {
        if (payload.kind === 'network_retry') this.patch({ networkRetrying: true })
        if (payload.kind === 'provider_ready') this.patch({ networkRetrying: false })
        const patch: Record<string, unknown> = {}
        if (typeof payload.model === 'string' && payload.model) patch.model = payload.model
        if (typeof payload.context_tokens === 'number') patch.contextTokens = payload.context_tokens
        if (typeof payload.max_context === 'number') {
          patch.contextMax = payload.max_context > 0 ? payload.max_context : null
        }
        // The daemon sends two status_update shapes under one type: per-round
        // deltas (usage frames — no turn_count/llm_steps) and cumulative
        // session echoes (every settled turn, plus mode/config changes —
        // those carry turn_count/llm_steps). Only the first kind may be
        // added; adopting a cumulative echo as a delta inflated model time
        // roughly N-fold (the TUI adapter discriminates the same way).
        const cumulativeTelemetry = payload.llm_steps !== undefined || payload.turn_count !== undefined
        if (typeof payload.llm_duration_ms === 'number' && Number.isFinite(payload.llm_duration_ms)) {
          if (cumulativeTelemetry) {
            patch.llmDurationMs = Math.max(0, payload.llm_duration_ms)
            patch.llmSteps = Math.max(0, Math.trunc(typeof payload.llm_steps === 'number' ? payload.llm_steps : this.frame.llmSteps))
          } else {
            patch.llmDurationMs = this.frame.llmDurationMs + Math.max(0, payload.llm_duration_ms)
            patch.llmSteps = this.frame.llmSteps + 1
            patch.metricPhase = 'llm'
            patch.metricPhaseStartedAt = Date.now()
          }
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
      reasoningPickerOpen: false,
      reasoningLoading: false,
      reasoningLevels: [],
      reasoningDefault: '',
      reasoningNote: '',
      contextMenuOpen: false,
      contextBreakdownLoading: false,
      contextBreakdown: null,
      permissionUpdating: false,
      permissionError: null,
      fleet: [],
      queue: this.queue,
      changes: [],
      changesKept: false,
      log: this.logRing,
      plan: null,
      todos: null,
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
      turnActive: true, networkRetrying: false,
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
    if (/connect|socket|offline|disposed|closed|not ready|launch/i.test(message)) {
      this.wentOffline(error)
      return
    }
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
    if ('currentId' in merge || 'cwd' in merge) rememberSession((this.frame.storageScope ?? '') + this.frame.cwd, this.frame.currentId)
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
  getContextScope?(): Promise<string>
  getContexts?(): Promise<WorkspaceContext[]>
  activateContext?(id: number, sessionId?: string): Promise<void>
  call(method: string, params?: Record<string, unknown>): Promise<Record<string, unknown>>
  /** Present on the real preload bridge; test bridges may omit it. */
  chooseWorkspace?(): Promise<unknown>
  getWorkspaceDirectories?(): Promise<string[]>
  openWorkspaceWindow?(dir?: string, resumeSessionId?: string): Promise<unknown>
  useWorkspace?(dir: string, resumeSessionId?: string): Promise<unknown>
  getWorkspace?(): Promise<string | null>
  getResumeSession?(): Promise<string | null>
}

export const store = new Store()
