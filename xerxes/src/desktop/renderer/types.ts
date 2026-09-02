// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Shared types for the desktop app: the wire-facing event vocabulary the
 * daemon pushes (ui/PROTOCOL.md is the contract), the transcript block model
 * the renderer folds events into, the workspace surfaces (changes, plan, log)
 * folded from the same stream, and the `window.xerxes` bridge surface the
 * sandboxed renderer is allowed to see.
 */

// ── Bridge (preload ↔ main ↔ renderer) ──────────────────────────────────

/** A daemon event push after bridge sanitization. Payload is frozen. */
export interface DaemonEvent {
  readonly type: string
  readonly payload: Readonly<Record<string, unknown>>
}

export interface XerxesBridge {
  call<T = Record<string, unknown>>(method: string, params?: Record<string, unknown>): Promise<T>
  onEvent(handler: (event: DaemonEvent) => void): () => void
  /** Present the folder picker; the shell switches workspace and reloads. */
  chooseWorkspace?(): Promise<unknown>
  /** Enter a workspace folder directly (no dialog). */
  useWorkspace?(dir: string): Promise<unknown>
  /** Saved workspace folder, or null when the create-workspace gate shows. */
  getWorkspace?(): Promise<string | null>
  /** Ping for needs-input / task-finished moments when the app is unfocused. */
  setNotifications?(on: boolean): Promise<boolean>
  /** Launch-at-login registration (main-process login item). */
  getLoginItem?(): Promise<boolean>
  setLoginItem?(on: boolean): Promise<boolean>
  /** Reveal a daemon-resolved local preset directory in the OS file browser. */
  openPath?(path: string): Promise<boolean>
}

declare global {
  interface Window {
    readonly xerxes: XerxesBridge
  }
}

// ── Transcript blocks ───────────────────────────────────────────────────

export interface ToolItem {
  readonly id: string
  readonly verb: string
  readonly arg: string
  readonly dur: string
  readonly state: 'working' | 'done' | 'failed'
  /** Exact tool identifier plus unabridged request/result for inspection. */
  readonly name: string
  readonly input: string
  readonly output: string
  readonly error?: string
  /** Workspace file the call touched, for edit-family tools. */
  readonly path?: string
  /** Line delta for edit-family calls, shown as `+a −d` on the trail row. */
  readonly diff?: { readonly adds: number; readonly dels: number }
}

export type Block =
  | { kind: 'user'; id: number; text: string }
  | { kind: 'agent'; id: number; text: string; streaming: boolean }
  | { kind: 'thinking'; id: number; text: string; streaming: boolean }
  | { kind: 'tools'; id: number; items: readonly ToolItem[]; running: boolean }
  | { kind: 'notice'; id: number; error: boolean; text: string }
  | { kind: 'checkpoint'; id: number; turn: number; adds: number; dels: number }
  | { kind: 'agents'; id: number; members: readonly AgentMember[] }

/** One spawned subagent inside a turn's agents card (dsh in-chat batch). */
export interface AgentMember {
  /** Store-local identity: spawn call id + index, or the daemon snapshot id. */
  readonly key: string
  readonly title: string
  /** working | completed | failed | cancelled — snapshot statuses pass through mapped. */
  readonly status: string
}

/** A daemon-backgrounded turn (bg-* session) currently working. */
export interface BackgroundJob {
  readonly id: string
  readonly title: string
  readonly status: string
}

/** One runtime-observed repeatable workflow proposed as a reusable skill. */
export interface SkillSuggestion {
  readonly skillName: string
  readonly description: string
  readonly version: string
  readonly sourcePath: string
  readonly toolCount: number
  readonly uniqueTools: readonly string[]
}

/** One persisted action in the policy-gated declarative creator forge. */
export interface AgentPreset {
  readonly id: string
  readonly name: string
  readonly description: string
  readonly trust: 'project' | 'system' | 'user'
  readonly isDefault: boolean
  readonly manageable: boolean
  readonly broken?: string
}

export interface CreatorTrace {
  readonly action: string
  readonly name: string
  readonly version: string
  readonly status: 'ok' | 'error'
  readonly detail: string
  readonly at: string
}

// ── Session rows ────────────────────────────────────────────────────────

export interface SessionRow {
  readonly id: string
  /** Daemon session key — the lazy-enrichment handle for saved rows. */
  readonly key: string
  readonly title: string
  readonly status: string
  readonly age: string
  readonly current: boolean
  readonly kind: 'main' | 'subagent'
  readonly turns: number
  readonly messages: number
  /** Project folder the chat ran in — the workspace grouping key. */
  readonly cwd: string
  /** No daemon-derived title — eligible for first-message enrichment. */
  readonly untitled: boolean
}

// ── Approval ────────────────────────────────────────────────────────────

export interface Approval {
  readonly id: string
  readonly action: string
  readonly description: string
  /** The streamed tool call this decision attaches to, when known. */
  readonly toolCallId?: string
  readonly toolName?: string
}

export type ApprovalResponse = 'allow_once' | 'allow_session' | 'deny'

// ── User questions (daemon `question_request`) ─────────────────────────

export interface QuestionItem {
  readonly id: string
  readonly question: string
  readonly options: readonly string[]
  readonly allowFreeform: boolean
  readonly placeholder?: string
}

export interface TaskQuestion {
  readonly requestId: string
  readonly toolCallId: string
  readonly items: readonly QuestionItem[]
}

// ── Workspace tabs ──────────────────────────────────────────────────────

export type WorkspaceTab = 'activity' | 'changes' | 'plan' | 'log'

export type DiffLineKind = 'hunk' | 'ctx' | 'add' | 'del'

export interface DiffLine {
  readonly kind: DiffLineKind
  readonly text: string
}

/** One reviewed file: per-path fold of the edit-family calls this session. */
export interface DiffFile {
  readonly path: string
  readonly adds: number
  readonly dels: number
  readonly isNew: boolean
  readonly hunks: readonly DiffLine[]
  readonly turn: number
}

export interface PlanItem {
  readonly text: string
  readonly done: boolean
}

/** The working plan: markdown captured from plan-mode agent output. */
export interface PlanState {
  readonly markdown: string
  readonly items: readonly PlanItem[]
  readonly turn: number
}

/** A steering message accepted by `turn.steer`, mirrored until consumed. */
export interface QueueItem {
  readonly id: number
  readonly text: string
}

/** One raw daemon event, kept in a bounded ring for the Log tab. */
export interface LogEntry {
  readonly id: number
  readonly turn: number
  readonly type: string
  readonly summary: string
}

export interface FailedTurn {
  readonly error: string
  readonly turn: number
  readonly lastUser: string
}

// ── Models & providers ──────────────────────────────────────────────────

export interface ModelChoice {
  readonly id: string
  /** Display grouping; derived from the id when the daemon sends none. */
  readonly provider: string
}

export interface CachedModel {
  readonly contextLimit?: number
  readonly contextSource?: 'catalog' | 'override' | 'provider' | 'unknown'
  readonly id: string
  readonly maxOutputTokens?: number
  readonly outputSource?: 'catalog' | 'override' | 'provider' | 'unknown'
  readonly overridden: boolean
}

export interface ProviderRow {
  readonly name: string
  readonly provider: string
  readonly model: string
  readonly active: boolean
  /** Stored endpoint — the Edit form prefills it; '' when none was sent. */
  readonly baseUrl: string
}

/**
 * One adapter the daemon's provider registry knows how to drive — the
 * add/edit form's Provider dropdown. `baseUrl` is the registry default
 * ('' → the type needs an explicit endpoint, e.g. custom); `apiKeyEnv` is
 * the environment variable a blank key falls back to ('' → none).
 */
export interface ProviderTypeRow {
  readonly name: string
  readonly baseUrl: string
  readonly apiKeyEnv: string
}

// ── Channels (daemon gateways: telegram, discord, slack, …) ─────────────

/** One messaging gateway the daemon's channel manager drives. */
export interface ChannelRow {
  readonly name: string
  readonly adapterName: string
  readonly enabled: boolean
  /** Last lifecycle op the gateway ran ('start' | 'stop' | …), when reported. */
  readonly lastOperation?: string
  /** Why the gateway is not up, when the daemon reports one. */
  readonly lastError?: string
}

/** Point-in-time channel-manager status shared by `channel.list` and the `channel_status` broadcast. */
export interface ChannelStatus {
  readonly channels: readonly ChannelRow[]
  /** False when the daemon runs without its channel manager at all. */
  readonly available: boolean
  /** False when no channel credentials are configured yet. */
  readonly configured: boolean
}

// ── Terminals (daemon-tracked shells the agents ran) ────────────────────

/** One daemon-registered terminal in list view — everything but the output. */
export interface TerminalRow {
  readonly id: string
  readonly kind: string
  readonly label: string
  readonly command: string
  readonly cwd: string
  readonly pid?: number
  readonly running: boolean
  /** Epoch ms. */
  readonly startedAt: number
  /** Epoch ms, absent while running. */
  readonly endedAt?: number
  readonly exitCode: number | null
  /** Total characters observed, including any dropped from the mirror. */
  readonly outputChars: number
  readonly canWrite: boolean
  readonly canInterrupt: boolean
  readonly canKill: boolean
}

/** Detail view: a row plus the retained tail of what it printed. */
export interface TerminalDetail extends TerminalRow {
  readonly output: string
  /** Older output was dropped from the mirror before this tail. */
  readonly outputTruncated: boolean
}

// ── Session search (`session.search` over the persisted transcripts) ────

/** One transcript message that matched a search query. */
export interface SessionSearchHit {
  readonly sessionId: string
  readonly messageIndex: number
  readonly role: string
  readonly excerpt: string
  readonly title: string
  readonly updatedAt: string
}

/** What the daemon's search index currently covers. */
export interface SessionSearchStats {
  readonly sessions: number
  readonly indexedMessages: number
  readonly searchableMessages: number
}

// ── Settings ────────────────────────────────────────────────────────────

export type SettingsTab = 'general' | 'models' | 'agents' | 'permissions' | 'mcp' | 'channels' | 'terminals'

/** One MCP server's redacted status, straight from the daemon wire. */
export interface McpServerStatus {
  readonly connected: boolean
  readonly tools: number
  readonly resources: number
  readonly prompts: number
  readonly lastError?: string
}

export type PermissionMode = 'accept-all' | 'auto' | 'manual' | 'plan'
