// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { Lesson } from '../memory/lessons.js'
import type { StoredReflectionProposal } from '../runtime/reflection.js'
import type { TurnOutcomeReason } from '../types/turnOutcome.js'
import type { SessionInfo, SlashCategory, SubagentStatus, Usage } from './types.js'

export interface GatewaySkin {
  banner_hero?: string
  banner_logo?: string
  branding?: Record<string, string>
  colors?: Record<string, string>
  help_header?: string
  tool_prefix?: string
}

export interface GatewayCompletionItem {
  display: string
  /** Daemon command category, used to order the bare-slash menu. */
  group?: string
  kind?: 'command' | 'skill'
  meta?: string
  source?: string
  text: string
}

export interface GatewayTranscriptMessage {
  outcome?: TurnOutcomeReason
  context?: string
  duration_s?: number
  error?: string
  name?: string
  role: 'assistant' | 'system' | 'tool' | 'user'
  text?: string
  thinking?: string
}

export type ApprovalResponseKind = 'always' | 'approve' | 'approve_for_session' | 'reject'

export interface QuestionItem {
  allow_free_form?: boolean
  id: string
  multi_select?: boolean
  options?: string[]
  placeholder?: string
  question: string
}

export type DisplayBlock =
  | { body: string; type: 'brief' }
  | { diff: string; language: string; type: 'diff' }
  | { items: Record<string, unknown>[]; type: 'todo' }
  | { status: string; title: string; type: 'background_task' }
  | { content: string; type: 'generic' }

export function normalizeEventType(type: string): string {
  if (!type || type.includes('_') || type.includes('.')) {
    return type
  }

  return type
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1_$2')
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .toLowerCase()
}

export function isResponseFrame(frame: unknown): boolean {
  return Boolean(frame && typeof frame === 'object' && 'id' in frame && (frame as { id?: unknown }).id !== undefined)
}

// ── Commands / completion ────────────────────────────────────────────

export interface CommandsCatalogResponse {
  canon?: Record<string, string>
  categories?: SlashCategory[]
  pairs?: [string, string][]
  skill_count?: number
  sub?: Record<string, string[]>
  warning?: string
}

export interface CompletionResponse {
  items?: GatewayCompletionItem[]
  replace_from?: number
}

export interface SlashExecResponse {
  output?: string
  warning?: string
}

// ── Credits / top-up ─────────────────────────────────────────────────

export interface CreditsViewResponse {
  balance_lines: string[]
  depleted: boolean
  identity_line: string | null
  logged_in: boolean
  topup_url: string | null
}

export type CommandDispatchResponse =
  | { output?: string; type: 'exec' | 'plugin' }
  | { target: string; type: 'alias' }
  | { message?: string; name: string; type: 'skill' }
  | { message: string; notice?: string; type: 'send' }
  | { message: string; notice?: string; type: 'prefill' }

// ── Config ───────────────────────────────────────────────────────────

export interface ConfigDisplayConfig {
  bell_on_complete?: boolean
  busy_input_mode?: string
  details_mode?: string
  inline_diffs?: boolean
  mouse_tracking?: boolean | null | number | string
  sections?: Record<string, string>
  show_cost?: boolean
  show_reasoning?: boolean
  streaming?: boolean
  thinking_mode?: string
  /**
   * Nudge the user toward the /agents spawn-tree dashboard the first time a
   * turn starts delegating, via a one-time transient activity hint.  Opens
   * nothing — just advertises the command.  Default true.
   */
  tui_agents_nudge?: boolean
  tui_auto_resume_recent?: boolean
  tui_compact?: boolean
  /** Legacy alias for display.mouse_tracking. */
  tui_mouse?: boolean | null | number | string
  // Forward-compat: backend may send styles this client doesn't know yet —
  // `normalizeIndicatorStyle` falls back to 'kaomoji' for those — but the
  // wire type is documented as `string` so consumers don't get a false
  // narrowing-and-autocomplete contract on a value that requires runtime
  // validation anyway.
  tui_status_indicator?: string
  tui_statusbar?: 'bottom' | 'off' | 'on' | 'top' | boolean
}

export interface ConfigVoiceConfig {
  // Raw `yaml.safe_load()` value from config; may be non-string if hand-edited.
  // Callers must normalize/validate at runtime (parseVoiceRecordKey()).
  record_key?: unknown
}

export interface ConfigFullResponse {
  config?: {
    display?: ConfigDisplayConfig
    voice?: ConfigVoiceConfig
    paste_collapse_threshold?: number
    paste_collapse_char_threshold?: number
  }
}

export interface ConfigMtimeResponse {
  mtime?: number
}

export interface ConfigGetValueResponse {
  display?: string
  home?: string
  value?: string
}

export interface ConfigSetResponse {
  confirm_message?: string
  confirm_required?: boolean
  credential_warning?: string
  history_reset?: boolean
  info?: SessionInfo
  value?: string
  warning?: string
}

export interface SetupStatusResponse {
  provider_configured?: boolean
}

/** Read-only local checkout status returned by the native Bun daemon. */
export interface NativeUpdateStatusResponse {
  applied?: false
  command?: string
  git?: {
    aheadCount?: number
    behindCount?: number
    branch?: string
    error?: string
    headHash?: string
    isGit?: boolean
    upstream?: string
    upstreamHash?: string
  }
  next_steps?: string[]
  summary?: string
}

// ── Human-reviewed learning ──────────────────────────────────────────

export interface LearningCounts {
  pending_proposals: number
  pending_lessons: number
}

export type LearningListResponse =
  | { ok: true; workspace: string; proposals: StoredReflectionProposal[]; lessons: Lesson[]; counts: LearningCounts }
  | { ok: false; error: string }

export type LearningReviewParams = {
  session_key?: string
  target: 'proposal' | 'lesson'
  id: string
  action: 'approve' | 'reject' | 'rollback'
  reason?: string
}

export interface LearningReviewResponse {
  ok: boolean
  report: string
  proposal?: StoredReflectionProposal
  lesson?: Lesson
  error?: string
}

// ── Session lifecycle ────────────────────────────────────────────────

export interface SessionCreateResponse {
  info?: SessionInfo & { config_warning?: string; credential_warning?: string }
  session_id: string
}

export interface SessionResumeResponse {
  /** Same owner reclaimed; retain the current transcript and apply the journal. */
  reconnected?: boolean
  recovery_pending?: boolean
  todos?: unknown[]
  inflight?: null | SessionInflightTurn
  info?: SessionInfo
  message_count?: number
  messages: GatewayTranscriptMessage[]
  resumed?: string
  running?: boolean
  session_id: string
  started_at?: number
  status?: LiveSessionStatus
  subagent_snapshots?: SubagentSnapshotPayload[]
}

export type LiveSessionStatus = 'idle' | 'starting' | 'waiting' | 'working'
export type SessionKind = 'main' | 'subagent'

export interface SessionActiveItem {
  /** Current turn input, shown as activity only; never used as a chat title. */
  activity?: string
  agent_id?: string
  current?: boolean
  id: string
  kind?: SessionKind
  last_active?: number
  message_count?: number
  model?: string
  parent_session_id?: null | string
  preview?: string
  resumable?: boolean
  root_session_id?: null | string
  session_key?: string
  started_at?: number
  status: LiveSessionStatus
  subagent_id?: null | string
  title?: string
}

export interface SessionActiveListResponse {
  sessions?: SessionActiveItem[]
}

/** One tool call from the in-flight turn, snapshotted for a mid-turn reattach. */
export interface SessionInflightTool {
  /** Compact arguments preview, never the full payload. */
  arguments?: string
  /** Display-safe semantic context retained when raw JSON was truncated. */
  context?: string
  /** Milliseconds when the call already settled; absent while running. */
  duration_ms?: number
  /** Compact diagnostic when the call failed. */
  error?: string
  id?: string
  name: string
  /** Present once the call settled; absent/false semantics follow the daemon. */
  ok?: boolean
}

export interface SessionInflightTurn {
  assistant?: string
  /** Epoch seconds the in-flight turn began, so reattach keeps elapsed continuity. */
  started_at?: number
  streaming?: boolean
  /** Thinking the in-flight turn produced so far. */
  thinking?: string
  /** Tool calls the in-flight turn produced so far, in call order. */
  tools?: SessionInflightTool[]
  user?: string
}

/** Persisted manifest row for a subagent the session spawned (daemon subagent_snapshots). */
export interface SubagentSnapshotPayload {
  group?: SubagentGroupPayload
  provider_profile?: string
  reasoning_effort?: string
  agent_id?: string
  api_calls?: number
  closed?: boolean
  created_at?: string
  creator_id?: null | string
  error?: string
  files_read?: string[]
  files_written?: string[]
  history_session_id?: string
  id: string
  input_tokens?: number
  model?: string
  name?: string
  output_tokens?: number
  parent_id?: null | string
  prompt_profile?: string
  queue_size?: number
  reasoning_tokens?: number
  rules?: string[]
  source_agent_id?: string
  status: string
  summary?: string
  title?: string
  tool_count?: number
  toolsets?: string[]
  updated_at?: string
}

export interface SessionActivateResponse {
  todos?: unknown[]
  inflight?: null | SessionInflightTurn
  info?: SessionInfo
  message_count?: number
  messages: GatewayTranscriptMessage[]
  // Events that arrived during the switch are held until the caller has
  // reset and hydrated the view; it must then call finishSessionRecovery.
  recovery_pending?: boolean
  running?: boolean
  session_id: string
  session_key?: string
  started_at?: number
  status?: LiveSessionStatus
  subagent_snapshots?: SubagentSnapshotPayload[]
}

/** Read-only live-session snapshot used by Agent View without attaching. */
export interface SessionPeekResponse {
  todos?: unknown[]
  inflight?: null | SessionInflightTurn
  messages: GatewayTranscriptMessage[]
  session_id: string
  status: LiveSessionStatus
  subagent_snapshots?: SubagentSnapshotPayload[]
}

export interface SessionListItem {
  agent_id?: string
  id: string
  kind?: SessionKind
  message_count: number
  model?: string
  parent_session_id?: null | string
  preview: string
  resumable?: boolean
  root_session_id?: null | string
  source?: string
  /** Epoch seconds of the latest conversation message. */
  last_message_at?: number
  /** Legacy alias retained for older callers; contains last-message time. */
  started_at: number
  status?: string
  subagent_id?: null | string
  title: string
}

export interface SessionListResponse {
  sessions?: SessionListItem[]
}

export interface SessionDeleteResponse {
  deleted: string
}

export interface SessionMostRecentResponse {
  session_id?: null | string
  source?: string
  started_at?: number
  title?: string
}

/** Rendered `/goal` output; `ok` is false for a refused transition. */
export interface SessionGoalResponse {
  ok?: boolean
  text?: string
}

export interface SessionTitleResponse {
  pending?: boolean
  session_key?: string
  title?: string
}

export interface SessionSaveResponse {
  file?: string
}

export interface SessionUndoResponse {
  removed?: number
}

export interface SessionUsageResponse {
  cache_read?: number
  cache_write?: number
  calls?: number
  compressions?: number
  context_max?: number
  context_percent?: number
  context_used?: number
  cost_status?: 'estimated' | 'exact'
  cost_usd?: number
  credits_lines?: string[]
  input?: number
  model?: string
  output?: number
  total?: number
}

export interface SessionStatusResponse {
  output?: string
}

export interface SessionCompressResponse {
  after_messages?: number
  after_tokens?: number
  before_messages?: number
  before_tokens?: number
  info?: SessionInfo
  messages?: GatewayTranscriptMessage[]
  removed?: number
  summary?: {
    headline?: string
    noop?: boolean
    note?: null | string
    token_line?: string
  }
  usage?: Usage
}

export interface SessionBranchResponse {
  session_id?: string
  title?: string
}

export interface SessionCloseResponse {
  closed?: boolean
  ok?: boolean
}

export interface SessionInterruptResponse {
  ok?: boolean
}

export interface SessionSteerResponse {
  ok?: boolean
  status?: 'queued' | 'rejected'
  text?: string
}

// ── Prompt / submission ──────────────────────────────────────────────

export interface PromptSubmitResponse {
  duplicate?: boolean
  ok?: boolean
}

export interface BackgroundStartResponse {
  task_id?: string
}

export interface ClarifyRespondResponse {
  error?: string
  ok?: boolean
}

export interface ApprovalRespondResponse {
  ok?: boolean
}

export interface SudoRespondResponse {
  ok?: boolean
}

export interface SecretRespondResponse {
  ok?: boolean
}

// ── Shell / clipboard / input ────────────────────────────────────────

export interface ShellExecResponse {
  code: number
  stderr?: string
  stdout?: string
}

export interface ClipboardPasteResponse {
  attached?: boolean
  count?: number
  height?: number
  message?: string
  token_estimate?: number
  width?: number
}

export interface InputDetectDropResponse {
  height?: number
  is_image?: boolean
  matched?: boolean
  name?: string
  text?: string
  token_estimate?: number
  width?: number
}

export interface TerminalResizeResponse {
  ok?: boolean
}

// ── Image attach ─────────────────────────────────────────────────────

/** One validated image ready to ride the `images` param of `prompt.submit`. */
export interface PromptSubmitImage {
  /** Canonical base64 of the decoded image bytes. */
  data: string
  /** Magic-byte-sniffed mime (image/png, image/jpeg, image/gif, image/webp). */
  media_type: string
}

export interface ImageAttachResponse {
  attached?: boolean
  /** Canonical base64 payload for the next prompt.submit `images` entry. */
  data?: string
  height?: number
  /** Magic-byte-sniffed mime, never extension-derived. */
  media_type?: string
  name?: string
  path?: string
  remainder?: string
  /** Decoded byte size of the attached image. */
  size?: number
  token_estimate?: number
  width?: number
}

// ── Voice ────────────────────────────────────────────────────────────

export interface VoiceToggleResponse {
  audio_available?: boolean
  available?: boolean
  details?: string
  enabled?: boolean
  record_key?: string
  stt_available?: boolean
  tts?: boolean
}

export interface VoiceRecordResponse {
  status?: 'busy' | 'recording' | 'stopped'
  text?: string
}

// ── Tools (TS keeps configure since it resets local history) ─────────

export interface ToolsConfigureResponse {
  changed?: string[]
  enabled_toolsets?: string[]
  info?: SessionInfo
  missing_servers?: string[]
  reset?: boolean
  unknown?: string[]
}

// ── Model picker ─────────────────────────────────────────────────────

export interface ModelOptionProvider {
  auth_type?: string
  authenticated?: boolean
  configured_model?: string
  is_current?: boolean
  key_env?: string
  models?: string[]
  name: string
  provider_type?: string
  slug: string
  total_models?: number
  warning?: string
}

export interface ModelOptionsResponse {
  model?: string
  provider?: string
  providers?: ModelOptionProvider[]
}

export interface ModelCapabilityOption {
  context_limit?: number
  context_source?: 'catalog' | 'override' | 'provider' | 'unknown'
  /** The provider's own name and description for the model, when it gives them. */
  display_name?: string
  description?: string
  id: string
  max_output_tokens?: number
  output_source?: 'catalog' | 'override' | 'provider' | 'unknown'
  overridden?: boolean
}

export interface ModelModelsResponse {
  catalog?: ModelCapabilityOption[]
  models?: string[]
  source?: string
  warning?: string
}

export interface ReasoningLevelOption {
  description?: string
  effort: string
}

export interface ReasoningLevelsResponse {
  /** Effort in force right now. */
  current?: string
  /** Effort the model applies when none is chosen. */
  default?: string
  levels?: ReasoningLevelOption[]
  /** Human-readable description of how this provider exposes reasoning. */
  note?: string
  /** `effort` (graded), `toggle` (on/off), or `inherent` (model decides). */
  shape?: string
  /** `provider` when the model reported these, `fallback` otherwise. */
  source?: string
}

// ── MCP ──────────────────────────────────────────────────────────────

export interface ReloadMcpResponse {
  status?: string
  message?: string
}

export interface ReloadEnvResponse {
  updated?: number
}

export interface ProcessStopResponse {
  killed?: number
}

export interface BrowserManageResponse {
  connected?: boolean
  kind?: 'cdp' | 'injected' | 'none'
  pages?: Array<{ ref_id?: string; refId?: string; title?: string; url?: string }>
  url?: string
}

export interface RollbackCheckpoint {
  hash: string
  message?: string
  timestamp?: string
}

export interface RollbackListResponse {
  checkpoints?: RollbackCheckpoint[]
  enabled?: boolean
}

export interface RollbackDiffResponse {
  diff?: string
  rendered?: string
  stat?: string
}

export interface RollbackRestoreResponse {
  error?: string
  history_removed?: number
  message?: string
  reason?: string
  restored_to?: string
  success?: boolean
}

// ── Subagent events ──────────────────────────────────────────────────

/** The workflow run and phase a subagent was started in. */
export interface SubagentGroupPayload {
  id: string
  label?: string
  phase?: string
}

export interface SubagentEventPayload {
  group?: SubagentGroupPayload
  provider_profile?: string
  reasoning_effort?: string
  agent_name?: string
  agent_type?: string
  api_calls?: number
  cost_usd?: number
  creator_id?: null | string
  depth?: number
  duration_seconds?: number
  files_read?: string[]
  files_written?: string[]
  goal: string
  input_tokens?: number
  iteration?: number
  model?: string
  output_tail?: { is_error?: boolean; preview?: string; tool?: string }[]
  output_tokens?: number
  parent_id?: null | string
  cache_creation_tokens?: number
  cache_read_tokens?: number
  reasoning_tokens?: number
  rules?: string[]
  status?: SubagentStatus
  subagent_id?: string
  summary?: string
  task_count?: number
  task_index: number
  text?: string
  title?: string
  tool_call_id?: string
  tool_count?: number
  /** Wall time one tool call took, reported with its result. */
  tool_duration_ms?: number
  tool_name?: string
  /** False when the call was denied by policy or ended in an error. */
  tool_ok?: boolean
  tool_preview?: string
  toolsets?: string[]
}

// ── Delegation control RPCs ──────────────────────────────────────────

export interface DelegationStatusResponse {
  active?: {
    depth?: number
    goal?: string
    model?: null | string
    parent_id?: null | string
    started_at?: number
    status?: string
    subagent_id?: string
    tool_count?: number
  }[]
  max_concurrent_children?: number
  max_spawn_depth?: number
  paused?: boolean
}

export interface DelegationPauseResponse {
  paused?: boolean
}

export interface SubagentInterruptResponse {
  ok?: boolean
  found?: boolean
  interrupted?: number
  subagent_id?: string
}

// ── Spawn-tree snapshots ─────────────────────────────────────────────

export interface SpawnTreeListEntry {
  count: number
  finished_at?: number
  label?: string
  path: string
  session_id?: string
  started_at?: number | null
}

export interface SpawnTreeListResponse {
  entries?: SpawnTreeListEntry[]
}

export interface SpawnTreeLoadResponse {
  finished_at?: number
  label?: string
  session_id?: string
  started_at?: null | number
  subagents?: unknown[]
}

// ── Native Swarm (v35) ───────────────────────────────────────────────

export type SwarmMemberRole = 'coordinator' | 'worker' | 'narrator' | 'monitor'
export type SwarmWorkspaceMode = 'inherit' | 'share' | 'branch'
export type SwarmMemberState = 'waiting' | 'working' | 'stopped' | 'killed'

export interface SwarmOverviewSnapshot {
  markdown: string
  summary?: string
  updated_at?: string
  author?: string
}

export interface SwarmChannelSnapshot {
  name: string
  topic: string
  created_by: string
  archived: boolean
  subscriber_count: number
}

export interface SwarmMemberSnapshot {
  name: string
  group?: string
  role: SwarmMemberRole
  added_by?: string
  conversation_id?: string
  model: string
  model_display_name: string
  workspace: SwarmWorkspaceMode
  workspace_path?: string
  channels: string[]
  state: SwarmMemberState
  killed: boolean
  current_activity?: string
  parked_since?: number
  mentions_only?: boolean
  last_read_seq: number
  last_post_seq?: number
  last_post_time?: string
  error?: string
}

export type SwarmActivityPhase = 'idle' | 'thinking' | 'responding' | 'tool'

export interface SwarmWorkerToolActivity {
  id: string
  name: string
  summary?: string
  status: 'running' | 'done' | 'error'
  started_at: string
  finished_at?: string
  result_preview?: string
}

export interface SwarmMemberActivitySnapshot {
  phase: SwarmActivityPhase
  summary?: string
  active_tools: SwarmWorkerToolActivity[]
  recent_tools: SwarmWorkerToolActivity[]
  live_text_preview?: string
  updated_at?: string
}

export interface SwarmInspectToolCall {
  id: string
  name: string
  args_preview: string
}

export interface SwarmInspectMessage {
  index: number
  role: 'user' | 'assistant' | 'tool'
  text: string
  tool_name?: string
  tool_calls?: SwarmInspectToolCall[]
}

export interface SwarmInspectParams {
  session_id?: string | null
  name: string
  limit?: number
}

export interface SwarmMemberInspectResponse {
  ok?: boolean
  session_id: string
  member: SwarmMemberSnapshot
  conversation_id: string
  activity: SwarmMemberActivitySnapshot
  messages: SwarmInspectMessage[]
  total_messages: number
  error?: string
}

export interface SwarmStatusSnapshot {
  active: boolean
  session_id: string
  title: string
  board: 'local'
  open_board: string
  head: number
  tail_from_seq?: number
  overview: SwarmOverviewSnapshot | null
  channels: SwarmChannelSnapshot[]
  members: SwarmMemberSnapshot[]
}

export interface SwarmBoardMessage {
  seq: number
  agent: string
  channel: string
  text: string
  mentions: string[]
  channel_mentions: string[]
  broadcast: boolean
  time: string
}

export type SwarmBoardParams = {
  session_id?: string | null
  since?: number
  limit?: number
  seqs?: number[]
  channel?: string
}

export interface SwarmBoardResponse {
  head: number
  messages: SwarmBoardMessage[]
}

export type SwarmPostParams = {
  session_id?: string | null
  text: string
  channel?: string
  as_feedback?: boolean
}

export interface SwarmPostResponse {
  status: 'posted'
  seq: number
  channel: string
}

export type SwarmKillParams = {
  session_id?: string | null
  name: string
}

export interface SwarmKillResponse {
  ok?: boolean
  status?: 'killed'
  killed?: string[]
  error?: string
}

export type SwarmModelParams = {
  session_id?: string | null
  name?: string
  model?: string
  reason?: string
}

export interface SwarmModelResponse {
  ok?: boolean
  status?: 'switched'
  name?: string
  model?: string
  error?: string
}

export type SwarmActivateParams = {
  session_id?: string | null
  enabled?: boolean
}

export interface SwarmActivateResponse {
  ok?: boolean
  active?: boolean
  prompt_overlay?: string
  error?: string
}

export interface SwarmEventPayload {
  session_id: string
  kind: string
  head: number
  status?: SwarmStatusSnapshot
  message?: SwarmBoardMessage
  member_name?: string
  member?: SwarmMemberSnapshot
  activity?: SwarmMemberActivitySnapshot
}

// ── Ephemeral Side Question (/btw · v35) ──────────────────────────────

export interface SideQuestionHistoryTurn {
  role: 'user' | 'assistant'
  content: string
}

export type SideQuestionParams = {
  session_id?: string | null
  session_key?: string
  request_id?: string
  question: string
  model?: string
  history?: readonly SideQuestionHistoryTurn[]
}

export interface SideQuestionResponse {
  ok: boolean
  request_id: string
  session_id: string
  question: string
  answer: string
  model: string
  error?: string
}

export type BtwEventKind = 'begin' | 'delta' | 'tool' | 'end' | 'error'

export interface BtwEventToolInfo {
  readonly name: string
  readonly target?: string
  readonly status: 'running' | 'done' | 'error'
}

export interface BtwEventPayload {
  session_id: string
  request_id: string
  kind: BtwEventKind
  question?: string
  delta?: string
  answer?: string
  model?: string
  error?: string
  tool?: BtwEventToolInfo
}

// ── External ACP Agents (v35) ─────────────────────────────────────────

export type AcpExternalAgentProvider = 'xerxes' | 'claude-code' | 'antigravity' | 'codex' | 'acp'
export type AcpExternalAgentTransport = 'native' | 'acp' | 'claude-stream-json'

export interface AcpExternalAgentInfo {
  readonly id: string
  readonly name: string
  readonly provider: AcpExternalAgentProvider
  readonly transport: AcpExternalAgentTransport
  readonly available: boolean
  readonly command?: string
  readonly args?: readonly string[]
  readonly binaryPath?: string
  readonly version?: string
  readonly installHint?: string
  readonly description: string
}

export interface AcpExternalModelOption {
  readonly id: string
  readonly name: string
  readonly description?: string
}

export interface AcpExternalModeOption {
  readonly id: string
  readonly name: string
  readonly description?: string
}

export interface AcpExternalConfigOptionChoice {
  readonly value: string
  readonly name: string
  readonly description?: string
  readonly group?: string
}

export type AcpConfigSelectOption = AcpExternalConfigOptionChoice

export interface AcpExternalConfigOption {
  readonly id: string
  readonly name: string
  readonly description?: string
  readonly category?: string
  readonly type: string
  readonly current_value: string | boolean
  readonly options: readonly AcpExternalConfigOptionChoice[]
}

export interface AcpExternalCommandOption {
  readonly name: string
  readonly description?: string
  readonly input_hint?: string
  readonly kind?: 'command' | 'skill'
  readonly source?: string
}

export type AcpAvailableCommand = AcpExternalCommandOption
export type AcpConfigOption = AcpExternalConfigOption

export interface AcpRuntimePromptCapabilities {
  readonly image: boolean
  readonly audio: boolean
  readonly embedded_context: boolean
}

export interface AcpRuntimeSessionCapabilities {
  readonly load: boolean
  readonly resume: boolean
  readonly list: boolean
  readonly fork: boolean
  readonly close: boolean
  readonly additional_directories: boolean
}

export interface AcpRuntimeMcpCapabilities {
  readonly http: boolean
  readonly sse: boolean
}

export interface AcpRuntimeAuthMethod {
  readonly id: string
  readonly name: string
  readonly description?: string
  readonly type?: string
  readonly args?: readonly string[]
  readonly env?: Readonly<Record<string, string>>
  readonly meta?: Readonly<Record<string, unknown>>
}

export type AcpAuthMethodInfo = AcpRuntimeAuthMethod

export interface AcpRuntimeCapabilities {
  readonly prompt: AcpRuntimePromptCapabilities
  readonly session: AcpRuntimeSessionCapabilities
  readonly mcp: AcpRuntimeMcpCapabilities
  readonly auth_methods: readonly AcpRuntimeAuthMethod[]
  readonly meta?: Readonly<Record<string, unknown>>
}

export interface AcpDiscoveredSkill {
  readonly name: string
  readonly description?: string
  readonly source?: string
  readonly path?: string
  readonly slash_command?: string
}

export interface AcpSessionBinding {
  readonly session_id: string
  readonly agent_id: string
  readonly name: string
  readonly provider: AcpExternalAgentProvider
  readonly transport: AcpExternalAgentTransport
  readonly cwd: string
  readonly external_session_id?: string
  readonly connected: boolean
  readonly command?: string
  readonly error?: string
  readonly agent_version?: string
  readonly agent_title?: string
  readonly capabilities?: AcpRuntimeCapabilities
  readonly current_model?: string
  readonly current_model_name?: string
  readonly models?: readonly AcpExternalModelOption[]
  readonly current_mode?: string
  readonly current_mode_name?: string
  readonly modes?: readonly AcpExternalModeOption[]
  readonly model_config_id?: string
  readonly mode_config_id?: string
  readonly config_options?: readonly AcpExternalConfigOption[]
  readonly thought_level?: string
  readonly thought_levels?: readonly AcpExternalModeOption[]
  readonly thought_level_config_id?: string
  readonly available_commands?: readonly AcpExternalCommandOption[]
  readonly discovered_skills?: readonly AcpDiscoveredSkill[]
}

export type AcpAgentsParams = {
  session_id?: string | null
  session_key?: string
}

export interface AcpAgentsResponse {
  ok: boolean
  agents: AcpExternalAgentInfo[]
  active: AcpSessionBinding | null
  error?: string
}

export type AcpCatalogParams = {
  session_id?: string | null
  session_key?: string
}

export interface AcpCatalogResponse {
  ok: boolean
  connected: boolean
  binding?: AcpSessionBinding
  commands: AcpAvailableCommand[]
  skills: AcpDiscoveredSkill[]
  capabilities?: AcpRuntimeCapabilities
  config_options: AcpExternalConfigOption[]
  error?: string
}

export type AcpConnectParams = {
  session_id?: string | null
  session_key?: string
  agent_id: string
  cwd?: string
  command?: string
  args?: readonly string[]
  mode_id?: string
  resume_session_id?: string
  additional_directories?: readonly string[]
  ide_mcp_port?: number
  agent?: string
  sandbox?: boolean
}

export interface AcpConnectResponse {
  ok: boolean
  binding?: AcpSessionBinding
  error?: string
  install_hint?: string
  auth_required?: boolean
  auth_methods?: AcpAuthMethodInfo[]
}

export type AcpDisconnectParams = {
  session_id?: string | null
  session_key?: string
}

export interface AcpDisconnectResponse {
  ok: boolean
  disconnected?: boolean
  binding: AcpSessionBinding | null
  error?: string
}

export type AcpCancelParams = {
  session_id?: string | null
  session_key?: string
}

export interface AcpCancelResponse {
  ok: boolean
  cancelled?: boolean
  error?: string
}

export type AcpSetModelParams = {
  session_id?: string | null
  session_key?: string
  model_id: string
}

export interface AcpSetModelResponse {
  ok: boolean
  binding?: AcpSessionBinding
  error?: string
}

export type AcpSetModeParams = {
  session_id?: string | null
  session_key?: string
  mode_id: string
}

export interface AcpSetModeResponse {
  ok: boolean
  binding?: AcpSessionBinding
  error?: string
}

export type AcpSetConfigOptionParams = {
  session_id?: string | null
  session_key?: string
  config_id: string
  value: string | boolean
  type?: 'boolean' | 'select' | string
}

export interface AcpSetConfigOptionResponse {
  ok: boolean
  binding?: AcpSessionBinding
  error?: string
}

export type AcpAuthenticateParams = {
  session_id?: string | null
  session_key?: string
  method_id: string
  agent_id?: string
}

export interface AcpAuthenticateResponse {
  ok: boolean
  mode?: 'agent' | 'terminal'
  command?: string
  args?: readonly string[]
  env?: Readonly<Record<string, string>>
  label?: string
  error?: string
}

export interface AcpSessionSummary {
  readonly session_id: string
  readonly cwd?: string
  readonly title?: string
  readonly updated_at?: string
  readonly meta?: Readonly<Record<string, unknown>>
}

export type AcpSessionsParams = {
  session_id?: string | null
  session_key?: string
  cwd?: string
  cursor?: string
}

export interface AcpSessionsResponse {
  ok: boolean
  sessions: AcpSessionSummary[]
  next_cursor?: string
  error?: string
}

export type AcpForkParams = {
  session_id?: string | null
  session_key?: string
  cwd?: string
  external_session_id?: string
}

export interface AcpForkResponse {
  ok: boolean
  binding?: AcpSessionBinding
  error?: string
}

export interface AcpStatusEventPayload {
  session_id: string
  binding: AcpSessionBinding | null
  agents?: AcpExternalAgentInfo[]
}

export type GatewayEvent =
  | { payload: { workspace: string; counts: LearningCounts }; type: 'learning.changed'; session_id?: string }
  | { payload?: { skin?: GatewaySkin }; session_id?: string; type: 'gateway.ready' }
  | { payload?: GatewaySkin; session_id?: string; type: 'skin.changed' }
  | { payload: SessionInfo; session_id?: string; type: 'session.info' }
  | { payload?: { text?: string }; session_id?: string; type: 'thinking.delta' }
  | { payload?: undefined; session_id?: string; type: 'message.start' }
  | { payload: GatewayTranscriptMessage; session_id?: string; type: 'transcript.append' }
  /** A provider attempt failed and is retried (`final: false`) or given up (`final: true`). */
  | { payload: { final: boolean; note: string }; session_id?: string; type: 'turn.retry' }
  /** Result checks ran after a round: a transcript row, and how many problems remain. */
  | { payload: { failing?: number; text: string }; session_id?: string; type: 'turn.checks' }
  | {
      payload?: {
        kind?: string
        mode?: string
        permission_mode?: SessionInfo['permission_mode']
        /** `provider_wait` only: the daemon's estimate of the prompt the model is reading. */
        prompt_tokens?: number
        reasoning_effort?: string
        telemetry_delta?: { llm_ms?: number; ttft_ms?: number }
        text?: string
        usage?: Usage
      }
      session_id?: string
      type: 'status.update'
    }
  | {
      payload?: {
        id?: string
        key?: string
        kind?: 'sticky' | 'ttl'
        level?: 'error' | 'info' | 'success' | 'warn'
        text?: string
        ttl_ms?: null | number
      }
      session_id?: string
      type: 'notification.show'
    }
  | { payload?: { key?: string }; session_id?: string; type: 'notification.clear' }
  | { payload?: { state?: 'idle' | 'listening' | 'transcribing' }; session_id?: string; type: 'voice.status' }
  | { payload?: { no_speech_limit?: boolean; text?: string }; session_id?: string; type: 'voice.transcript' }
  | { payload: { line: string }; session_id?: string; type: 'gateway.stderr' }
  | {
      payload?: { level?: 'info' | 'warn' | 'error'; message?: string }
      session_id?: string
      type: 'browser.progress'
    }
  | {
      payload?: { bun?: string; cwd?: string; entry_path?: string; stderr_tail?: string }
      session_id?: string
      type: 'gateway.start_timeout'
    }
  | { payload?: { preview?: string }; session_id?: string; type: 'gateway.protocol_error' }
  | {
      payload?: { text?: string; verbose?: boolean }
      session_id?: string
      type: 'reasoning.delta' | 'reasoning.available'
    }
  | { payload: { name?: string; preview?: string }; session_id?: string; type: 'tool.progress' }
  | { payload: { name?: string }; session_id?: string; type: 'tool.generating' }
  | {
      payload: { args_text?: string; context?: string; name?: string; reasoning?: string; tool_id: string; todos?: unknown[] }
      session_id?: string
      type: 'tool.start'
    }
  | {
      payload: {
        content_blocks?: readonly Record<string, unknown>[]
        diff?: { new_text?: string; old_text?: string; path: string }
        diffs?: { new_text?: string; old_text?: string; path: string }[]
        inputs?: Record<string, unknown>
        kind?: string
        locations?: { line?: number; path: string }[]
        name?: string
        output?: string
        status?: string
        title?: string
        tool_id: string
      }
      session_id?: string
      type: 'tool.update'
    }
  | {
      payload: {
        duration_s?: number
        error?: string
        inline_diff?: string
        name?: string
        reasoning?: string
        result_text?: string
        summary?: string
        tool_id: string
        todos?: unknown[]
      }
      session_id?: string
      type: 'tool.complete'
    }
  | {
      payload: {
        allow_free_form?: boolean
        choices: string[] | null
        elicitation_id?: string
        message?: string
        mode?: 'form' | 'url'
        multi_select?: boolean
        placeholder?: string
        question: string
        question_id?: string
        questions?: QuestionItem[]
        request_id: string
        source?: 'agent' | 'provider'
        tool_id?: string
        url?: string
      }
      session_id?: string
      type: 'clarify.request'
    }
  | {
      /** The daemon request id (without the `:question` suffix), and whether the daemon answered on its timeout. */
      payload: { answer?: string; auto: boolean; elicitation_id?: string; reason?: string; request_id: string }
      session_id?: string
      type: 'clarify.resolved'
    }
  | {
      payload: {
        allow_permanent?: boolean
        command: string
        cwd?: string
        description: string
        diff?: { new_text?: string; old_text?: string; path: string }
        diffs?: Array<{ new_text?: string; old_text?: string; path: string }>
        inputs?: Record<string, unknown>
        locations?: Array<{ line?: number; path: string }>
        options?: Array<{ kind: string; name: string; option_id: string; warning?: string }>
        request_id: string
      }
      session_id?: string
      type: 'approval.request'
    }
  | { payload: { request_id: string }; session_id?: string; type: 'sudo.request' }
  | { payload: { env_var: string; prompt: string; request_id: string }; session_id?: string; type: 'secret.request' }
  | { payload: { task_id: string; text: string }; session_id?: string; type: 'background.complete' }
  | { payload?: { text?: string }; session_id?: string; type: 'review.summary' }
  | { payload: SubagentEventPayload; session_id?: string; type: 'subagent.spawn_requested' }
  | { payload: SubagentEventPayload; session_id?: string; type: 'subagent.start' }
  | { payload: SubagentEventPayload; session_id?: string; type: 'subagent.thinking' }
  | { payload: SubagentEventPayload; session_id?: string; type: 'subagent.tool' }
  | { payload: SubagentEventPayload; session_id?: string; type: 'subagent.tool_result' }
  | { payload: SubagentEventPayload; session_id?: string; type: 'subagent.progress' }
  | { payload: SubagentEventPayload; session_id?: string; type: 'subagent.complete' }
  | { payload: { rendered?: string; text?: string }; session_id?: string; type: 'message.delta' }
  | {
      payload?: { outcome?: TurnOutcomeReason; unstarted?: boolean; interrupted?: boolean; reasoning?: string; rendered?: string; text?: string; usage?: Usage }
      session_id?: string
      type: 'message.complete'
    }
  | { payload?: { message?: string }; session_id?: string; type: 'error' }
  | { payload?: { session_id?: string; title?: string }; session_id?: string; type: 'session_title' }
  | { payload: SwarmEventPayload; session_id?: string; type: 'swarm_event' }
  | { payload: BtwEventPayload; session_id?: string; type: 'btw_event' }
  | { payload: AcpStatusEventPayload; session_id?: string; type: 'acp_status' }

export type AnyEvent = GatewayEvent | { payload?: Record<string, unknown>; session_id?: string; type: string }
