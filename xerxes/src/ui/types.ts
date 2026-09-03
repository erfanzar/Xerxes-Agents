// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
export interface ActiveTool {
  context?: string
  id: string
  name: string
  startedAt?: number
}

export interface TodoItem {
  content: string
  id: string
  status: 'cancelled' | 'completed' | 'in_progress' | 'pending'
}

export interface ActivityItem {
  id: number
  text: string
  tone: 'error' | 'info' | 'warn'
}

export type SubagentStatus = 'completed' | 'error' | 'failed' | 'interrupted' | 'queued' | 'running' | 'timeout'

export interface SubagentProgress {
  agentType?: string
  name?: string
  title?: string
  apiCalls?: number
  costUsd?: number
  creatorId?: null | string
  depth: number
  durationSeconds?: number
  filesRead?: string[]
  filesWritten?: string[]
  goal: string
  id: string
  index: number
  inputTokens?: number
  iteration?: number
  model?: string
  notes: string[]
  outputTail?: SubagentOutputEntry[]
  outputTokens?: number
  parentId: null | string
  cacheCreationTokens?: number
  cacheReadTokens?: number
  reasoningTokens?: number
  rules?: string[]
  startedAt?: number
  status: SubagentStatus
  summary?: string
  taskCount: number
  thinking: string[]
  toolCalls?: SubagentToolCall[]
  toolCount: number
  tools: string[]
  toolsets?: string[]
}

/**
 * One tool call an agent made, with how long it took.
 *
 * Separate from the `tools` string list, which is a capped tail of display
 * lines for the live rail. This is the record the inspector reads: it pairs
 * start with result, so a call still in flight is visibly in flight rather
 * than indistinguishable from one that finished instantly.
 */
export interface SubagentToolCall {
  /** Epoch ms; absent while the call is still running. */
  endedAt?: number
  id: string
  name: string
  /** False when the call was denied or errored. */
  ok?: boolean
  preview?: string
  result?: string
  startedAt: number
}

export interface SubagentOutputEntry {
  isError: boolean
  preview: string
  tool: string
}

export interface SubagentNode {
  aggregate: SubagentAggregate
  children: SubagentNode[]
  item: SubagentProgress
}

export interface SubagentAggregate {
  activeCount: number
  costUsd: number
  descendantCount: number
  filesTouched: number
  hotness: number
  inputTokens: number
  maxDepthFromHere: number
  outputTokens: number
  totalDuration: number
  totalTools: number
}

export interface DelegationStatus {
  active: {
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
  paused: boolean
}

export interface ApprovalReq {
  // false when the backend won't honor a permanent allow (tirith warning) → hide "Always allow".
  allowPermanent?: boolean
  command: string
  description: string
  requestId: string
}

export interface ConfirmReq {
  cancelLabel?: string
  confirmLabel?: string
  danger?: boolean
  detail?: string
  onConfirm: () => void
  title: string
}

export interface ClarifyReq {
  allowFreeform?: boolean
  choices: string[] | null
  placeholder?: string
  question: string
  questionId?: string
  requestId: string
  source?: 'agent' | 'provider'
  toolId?: string
}

export interface Msg {
  info?: SessionInfo
  kind?: 'config' | 'diff' | 'intro' | 'panel' | 'slash' | 'trail'
  panelData?: PanelData
  role: Role
  subagents?: SubagentProgress[]
  text: string
  thinking?: string
  thinkingTokens?: number
  toolTokens?: number
  tools?: string[]
  todos?: TodoItem[]
  todoIncomplete?: boolean
  todoCollapsedByDefault?: boolean
}

export type Role = 'assistant' | 'system' | 'tool' | 'user'
export type DetailsMode = 'hidden' | 'collapsed' | 'expanded'
export type ThinkingMode = 'collapsed' | 'truncated' | 'full'

// Per-section overrides for the agent details accordion.  Resolution order
// at lookup time is: explicit `display.sections.<name>` → built-in
// SECTION_DEFAULTS → global `details_mode`.  Today the built-in defaults
// expand `thinking`/`tools` and hide `activity`; `subagents` remains in the
// compatibility schema but is rendered exclusively by the agent panel.
export type SectionName = 'thinking' | 'tools' | 'subagents' | 'activity'
export type SectionVisibility = Partial<Record<SectionName, DetailsMode>>

export interface McpServerStatus {
  connected: boolean
  disabled?: boolean
  status?: 'configured' | 'connecting' | 'connected' | 'disabled' | 'failed'
  name: string
  tools: number
  transport: string
}

export interface SessionInfo {
  cwd?: string
  fast?: boolean
  head_hash?: string
  lazy?: boolean
  mcp_servers?: McpServerStatus[]
  model: string
  permission_mode?: 'accept-all' | 'auto' | 'manual' | 'plan'
  profile_name?: string
  reasoning_effort?: string
  release_date?: string
  service_tier?: string
  session_id?: string
  mode?: string
  skills: Record<string, string[]>
  skillDescriptions?: Record<string, string>
  system_prompt?: string
  tools: Record<string, string[]>
  update_behind?: number | null
  update_command?: string
  usage?: Usage
  version?: string
}

export interface Usage {
  cache_read?: number
  cache_write?: number
  calls: number
  compressions?: number
  context_max?: number
  context_percent?: number
  context_used?: number
  cost_status?: string
  cost_usd?: number
  dev_credits_spent_micros?: number
  input: number
  output: number
  reasoning?: number
  total: number
  // Cumulative session telemetry (daemon status_update, v35 additive):
  // wall-clock LLM/tool time, step counts, TTFT, throughput, cache hit rate.
  turns?: number
  llm_steps?: number
  tool_steps?: number
  llm_ms?: number
  tool_ms?: number
  ttft_avg_ms?: number
  /** Cumulative sample state retained so live per-round TTFT can update the average exactly. */
  ttft_samples?: number
  ttft_total_ms?: number
  tok_per_sec?: number
  /** Fraction 0..1; absent when the provider never reported cache reads. */
  cache_hit_rate?: number
}

export interface SudoReq {
  requestId: string
}

export interface SecretReq {
  envVar: string
  prompt: string
  requestId: string
}

export interface PanelData {
  sections: PanelSection[]
  title: string
}

export interface PanelSection {
  items?: string[]
  rows?: [string, string][]
  text?: string
  title?: string
}

export interface SlashCatalog {
  canon: Record<string, string>
  categories: SlashCategory[]
  pairs: [string, string][]
  skillCount: number
  sub: Record<string, string[]>
}

export interface SlashCategory {
  name: string
  pairs: [string, string][]
}
