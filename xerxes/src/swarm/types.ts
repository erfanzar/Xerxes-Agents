// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ChatMessage } from '../types/messages.js'

export const COORDINATOR_NAME = 'coordinator'
export const USER_BOARD_AUTHOR = 'user'
export const MAIN_CHANNEL = 'main'
export const WILDCARD_CHANNEL = '*'
export const SWARM_OPEN_BOARD_MARKDOWN = '[Open the Immortals UI](sidecar://immortals/board/?auto_open=false)'

export type SwarmMemberRole = 'coordinator' | 'worker' | 'narrator' | 'monitor'
export type SwarmMemberState = 'waiting' | 'working' | 'stopped' | 'killed'
export type SwarmWorkspaceMode = 'inherit' | 'share' | 'branch'
export type SwarmBoardMessageKind = 'brief' | 'message' | 'feedback' | 'system'

export interface SwarmBoardMessage {
  readonly seq: number
  readonly agent: string
  readonly channel: string
  readonly text: string
  readonly mentions: string[]
  readonly channel_mentions: string[]
  readonly broadcast: boolean
  readonly time: string
  readonly kind?: SwarmBoardMessageKind
}

export interface SwarmOverviewSnapshot {
  readonly markdown: string
  readonly summary?: string
  readonly updated_at?: string
  readonly author?: string
}

export interface SwarmChannelRecord {
  readonly name: string
  readonly topic: string
  readonly created_by: string
  readonly created_at: string
  readonly archived: boolean
}

export interface SwarmChannelSummary {
  readonly name: string
  readonly topic: string
  readonly created_by: string
  readonly archived: boolean
  readonly subscriber_count: number
}

export interface SwarmMemberRecord {
  readonly name: string
  readonly group?: string
  readonly role: SwarmMemberRole
  readonly added_by?: string
  model: string
  model_display_name: string
  readonly workspace: SwarmWorkspaceMode
  readonly workspace_path: string
  readonly workspace_handle_id?: string
  readonly workspace_base_head?: string
  readonly channels: string[]
  state: SwarmMemberState
  killed: boolean
  parked_since?: number
  mentions_only?: boolean
  last_read_seq: number
  last_post_seq?: number
  last_post_time?: string
  error?: string
}

export interface SwarmWorkerToolActivity {
  readonly id: string
  readonly name: string
  readonly summary?: string
  readonly status: 'running' | 'done' | 'error'
  readonly started_at: string
  readonly finished_at?: string
  readonly result_preview?: string
}

export interface SwarmMemberActivitySnapshot {
  readonly phase: 'idle' | 'thinking' | 'responding' | 'tool'
  readonly summary?: string
  readonly active_tools: SwarmWorkerToolActivity[]
  readonly recent_tools: SwarmWorkerToolActivity[]
  readonly live_text_preview?: string
  readonly updated_at?: string
}

export interface SwarmInspectMessage {
  readonly index: number
  readonly role: 'user' | 'assistant' | 'tool'
  readonly text: string
  readonly tool_name?: string
  readonly tool_calls?: Array<{
    readonly id: string
    readonly name: string
    readonly args_preview: string
  }>
}

export interface SwarmMemberSnapshot {
  readonly name: string
  readonly group?: string
  readonly role: SwarmMemberRole
  readonly added_by?: string
  readonly model: string
  readonly model_display_name: string
  readonly workspace: SwarmWorkspaceMode
  readonly workspace_path?: string
  readonly channels: string[]
  readonly state: SwarmMemberState
  readonly killed: boolean
  readonly conversation_id?: string
  readonly current_activity?: string
  readonly parked_since?: number
  readonly mentions_only?: boolean
  readonly last_read_seq: number
  readonly last_post_seq?: number
  readonly last_post_time?: string
  readonly error?: string
}

export interface SwarmMemberInspectResult {
  readonly session_id: string
  readonly member: SwarmMemberSnapshot
  readonly conversation_id: string
  readonly activity: SwarmMemberActivitySnapshot
  readonly messages: SwarmInspectMessage[]
  readonly total_messages: number
}

export interface SwarmRosterStatusEntry extends SwarmMemberSnapshot {
  readonly last_post_preview?: string
}

export interface SwarmStatusSnapshot {
  readonly active: boolean
  readonly session_id: string
  readonly title: string
  readonly board: 'local'
  readonly open_board: string
  readonly head: number
  readonly tail_from_seq: number
  readonly overview: SwarmOverviewSnapshot | null
  readonly channels: SwarmChannelSummary[]
  readonly members: SwarmMemberSnapshot[]
}

export type SwarmLifecycleEventKind =
  | 'activated'
  | 'titled'
  | 'member_added'
  | 'member_updated'
  | 'member_activity'
  | 'member_killed'
  | 'message_posted'
  | 'channel_updated'
  | 'overview_updated'

export interface SwarmLifecycleEvent {
  readonly session_id: string
  readonly kind: SwarmLifecycleEventKind
  readonly head: number
  readonly status: SwarmStatusSnapshot
  readonly message?: SwarmBoardMessage
  readonly member?: SwarmMemberSnapshot
  readonly channel?: SwarmChannelSummary
  readonly activity?: SwarmMemberActivitySnapshot
}

export type SwarmEventListener = (event: SwarmLifecycleEvent) => void

export interface SwarmWorkerErrorInfo {
  readonly member_name: string
  readonly message: string
  readonly quota_exhausted: boolean
}

export interface SwarmCoordinatorWakeEvent {
  readonly session_id: string
  readonly head: number
  readonly last_read_seq: number
  readonly wake_seq: number
  readonly reason?: 'board' | 'worker_error'
  readonly message?: SwarmBoardMessage
  readonly worker_error?: SwarmWorkerErrorInfo
  readonly notice: string
  readonly status: SwarmStatusSnapshot
}

export interface SwarmModelInfo {
  readonly id: string
  readonly display_name: string
  readonly suggested?: boolean
  readonly quota_remaining?: number
  readonly quota_resets?: string
  readonly quota_exhausted?: boolean
  readonly credits_available?: boolean
  readonly quota_note?: string
}

export interface SwarmModelCatalogPort {
  listModels(options?: { readonly all?: boolean }): Promise<readonly SwarmModelInfo[]> | readonly SwarmModelInfo[]
  getDefaultModel?(): string | { readonly id: string; readonly display_name?: string }
}

export interface SwarmWorkspaceAllocation {
  readonly mode: SwarmWorkspaceMode
  readonly path: string
  readonly handle_id?: string
  readonly base_head?: string
}

export interface SwarmWorkspaceCreateRequest {
  readonly mode: SwarmWorkspaceMode
  readonly baseDir: string
  readonly memberName: string
  readonly sessionId: string
  readonly signal?: AbortSignal
}

export interface SwarmWorkspaceCleanupRequest {
  readonly mode: SwarmWorkspaceMode
  readonly baseDir: string
  readonly workspacePath: string
  readonly memberName: string
  readonly sessionId: string
  readonly handleId?: string
}

export interface SwarmWorkspacePatchInspectRequest {
  readonly mode: SwarmWorkspaceMode
  readonly baseDir: string
  readonly workspacePath: string
  readonly memberName: string
  readonly baseHead?: string
  readonly targetDir?: string
  readonly signal?: AbortSignal
}

export interface SwarmWorkspacePatchInspection {
  readonly baseHead?: string
  readonly head?: string
  readonly changedFiles: readonly string[]
  readonly insertions: number
  readonly deletions: number
  readonly patchText: string
  readonly cleanWorkingTree: boolean
  readonly hasConflictMarkers: boolean
  readonly applicable: boolean
}

export interface SwarmWorkspacePatchApplyRequest {
  readonly patchText: string
  readonly targetDir: string
  readonly signal?: AbortSignal
}

export interface SwarmMemberPatchInspection extends SwarmWorkspacePatchInspection {
  readonly memberName: string
  readonly workspace: SwarmWorkspaceMode
  readonly workspacePath: string
}

export interface SwarmCandidateInput {
  readonly name: string
  readonly verified?: boolean
}

export interface SwarmCompareCandidatesOptions {
  readonly verified?: Readonly<Record<string, boolean>> | ReadonlyMap<string, boolean>
  readonly targetDir?: string
  readonly signal?: AbortSignal
}

export interface SwarmRankedCandidatePatch extends SwarmMemberPatchInspection {
  readonly rank: number
  readonly verified: boolean
  readonly eligible: boolean
  readonly totalLineChanges: number
}

export interface SwarmCandidateComparisonResult {
  readonly candidates: readonly SwarmRankedCandidatePatch[]
  readonly winner: SwarmRankedCandidatePatch | null
}

export interface SwarmExportMemberPatchOptions {
  readonly outputPath?: string
  readonly allowConflictMarkers?: boolean
  readonly signal?: AbortSignal
}

export interface SwarmExportedMemberPatch {
  readonly memberName: string
  readonly workspace: SwarmWorkspaceMode
  readonly workspacePath: string
  readonly baseHead?: string
  readonly changedFiles: readonly string[]
  readonly insertions: number
  readonly deletions: number
  readonly patchText: string
  readonly outputPath?: string
}

export interface SwarmApplyWinnerPatchOptions {
  readonly allowEmpty?: boolean
  readonly allowConflictMarkers?: boolean
  readonly signal?: AbortSignal
}

export interface SwarmApplyWinnerPatchResult {
  readonly status: 'applied'
  readonly memberName: string
  readonly targetDir: string
  readonly changedFiles: readonly string[]
  readonly insertions: number
  readonly deletions: number
  readonly patchText: string
}

export interface SwarmWorkspacePort {
  createWorkspace(request: SwarmWorkspaceCreateRequest): Promise<SwarmWorkspaceAllocation>
  cleanupWorkspace(request: SwarmWorkspaceCleanupRequest): Promise<void>
  inspectWorkspacePatch?(request: SwarmWorkspacePatchInspectRequest): Promise<SwarmWorkspacePatchInspection>
  checkPatchApplicable?(request: SwarmWorkspacePatchApplyRequest): Promise<boolean>
  applyPatchToWorkspace?(request: SwarmWorkspacePatchApplyRequest): Promise<void>
}

export type SwarmForkMessagesPort = (
  memberName: string,
) => Promise<readonly ChatMessage[] | undefined> | readonly ChatMessage[] | undefined

export type SwarmWorkerWakeReason = 'startup' | 'wake' | 'model_switch' | 'revive' | 'schedule'

/**
 * Clean turn-termination signal thrown onto a worker's per-turn AbortController
 * when `swarm_wait` parks the worker with no pending wake messages.
 */
export class SwarmParkSignal extends Error {
  readonly memberName: string
  readonly since: number
  readonly mentionsOnly: boolean

  constructor(memberName: string, since: number, mentionsOnly: boolean) {
    super(`Swarm worker "${memberName}" parked on board seq #${since}`)
    this.name = 'SwarmParkSignal'
    this.memberName = memberName
    this.since = since
    this.mentionsOnly = mentionsOnly
  }
}

export function isSwarmParkSignal(value: unknown): value is SwarmParkSignal {
  return value instanceof SwarmParkSignal
    || (typeof value === 'object' && value !== null && (value as { readonly name?: unknown }).name === 'SwarmParkSignal')
}

/**
 * Permanent cancellation signal emitted when a worker is killed via `swarm kill`.
 */
export class SwarmKillSignal extends Error {
  readonly memberName: string

  constructor(memberName: string) {
    super(`Swarm worker "${memberName}" was killed`)
    this.name = 'SwarmKillSignal'
    this.memberName = memberName
  }
}

export function isSwarmKillSignal(value: unknown): value is SwarmKillSignal {
  return value instanceof SwarmKillSignal
    || (typeof value === 'object' && value !== null && (value as { readonly name?: unknown }).name === 'SwarmKillSignal')
}

export type SwarmChannelOp = 'create' | 'subscribe' | 'unsubscribe' | 'list' | 'archive'

export interface SwarmAddInput {
  readonly caller?: string
  readonly name: string
  readonly brief: string
  readonly model?: string
  readonly n?: number
  readonly tasks?: readonly string[]
  readonly fork?: string
  readonly workspace?: SwarmWorkspaceMode
  readonly channels?: readonly string[]
}

export interface SwarmAddResult {
  readonly status: 'added'
  readonly added: SwarmMemberSnapshot[]
  readonly brief_seqs: number[]
  readonly channel: string
  readonly head: number
  readonly open_board: string
  readonly note: string
}

export interface SwarmSwitchModelInput {
  readonly caller?: string
  readonly name?: string
  readonly model?: string
  readonly reason?: string
}

export interface SwarmSwitchModelResult {
  readonly status: 'switched'
  readonly name: string
  readonly previous_model: string
  readonly model: string
  readonly model_display_name: string
  readonly resumed: boolean
  readonly reason?: string
}

export interface SwarmKillInput {
  readonly caller?: string
  readonly name: string
  /** Set only by host RPCs (`/immortals kill all` or `/immortals stop`, plus `/swarm` aliases), never by LLM tool calls. */
  readonly allowAll?: boolean
}

export interface SwarmKillResult {
  readonly status: 'killed'
  readonly killed: string[]
}

export interface SwarmTitleInput {
  readonly caller?: string
  readonly title: string
}

export interface SwarmTitleResult {
  readonly status: 'titled'
  readonly title: string
  readonly open_board: string
}

export interface SwarmChannelActionInput {
  readonly caller?: string
  readonly op: SwarmChannelOp
  readonly channel?: string
  readonly topic?: string
  readonly name?: string
}

export type SwarmChannelActionResult =
  | { readonly status: 'created'; readonly channel: SwarmChannelSummary }
  | { readonly status: 'subscribed'; readonly channel: string; readonly members: string[] }
  | { readonly status: 'unsubscribed'; readonly channel: string; readonly members: string[] }
  | { readonly channels: SwarmChannelSummary[] }
  | { readonly status: 'archived'; readonly channel: SwarmChannelSummary }

export interface SwarmPostMessageInput {
  readonly caller?: string
  readonly text: string
  readonly channel?: string
  readonly kind?: SwarmBoardMessageKind
}

export interface SwarmPostMessageResult {
  readonly status: 'posted'
  readonly seq: number
  readonly channel: string
  readonly mentions: string[]
  readonly channel_mentions: string[]
  readonly broadcast: boolean
  readonly woken: string[]
  readonly message: SwarmBoardMessage
}

export interface SwarmUserPostInput {
  readonly text: string
  readonly channel?: string
  readonly as_feedback?: boolean
}

export interface SwarmReadBoardInput {
  readonly caller?: string
  readonly since?: number
  readonly limit?: number
  readonly seqs?: readonly number[]
  readonly channel?: string
  readonly status?: boolean
}

export type SwarmReadBoardResult =
  | {
      readonly head: number
      readonly messages: SwarmBoardMessage[]
    }
  | {
      readonly head: number
      readonly active: boolean
      readonly title: string
      readonly members: SwarmRosterStatusEntry[]
    }

export interface SwarmWaitInput {
  readonly caller?: string
  readonly since: number
  readonly mentions_only?: boolean
  readonly signal?: AbortSignal
  readonly wakeSignal?: AbortSignal
  readonly onWorkerPark?: (signal: SwarmParkSignal) => void
}

export type SwarmWaitResult =
  | {
      readonly status: 'woken'
      readonly head: number
      readonly last_read_seq: number
      readonly wake_seq: number
      readonly message: string
    }
  | {
      readonly status: 'parked'
      readonly head: number
      readonly since: number
      readonly mentions_only: boolean
    }
  | {
      readonly status: 'interrupted_by_steer'
      readonly head: number
      readonly last_read_seq: number
      readonly message: string
    }
  | {
      readonly status: 'worker_error'
      readonly head: number
      readonly last_read_seq: number
      readonly worker_error: SwarmWorkerErrorInfo
      readonly message: string
    }

export interface SwarmOverviewUpdateInput {
  readonly caller?: string
  readonly markdown: string
  readonly summary?: string
}

export interface SwarmOverviewUpdateResult {
  readonly status: 'updated'
  readonly overview: SwarmOverviewSnapshot
  readonly head: number
}

export interface SwarmActivateResult {
  readonly ok: true
  readonly active: boolean
  readonly prompt_overlay: string
}
