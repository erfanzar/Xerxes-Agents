// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { PermissionMode } from '../streaming/permissions.js'
import { createAgentState, type AgentState } from '../streaming/events.js'
import type { ChatMessage, MessageContent } from '../types/messages.js'
import type { ToolCall } from '../types/toolCalls.js'
import {
  type DaemonTranscript,
  DaemonTranscriptStore,
  type RawMessage,
} from '../session/daemonTranscript.js'

export type PersistedSubagentStatus = 'cancelled' | 'completed' | 'error' | 'running'

/** In-flight provider output written without mutating the live turn state. */
export interface SubagentConversationCheckpoint {
  readonly content: string
  readonly thinking: string
}

/** Stable identity and delegated policy recorded with one child conversation. */
export interface SubagentConversationContext {
  readonly agentId: string
  readonly creatorAgentId?: string
  /** Actual execution directory, which may be an isolated worktree. */
  readonly cwd: string
  readonly handleId: string
  readonly historySessionId: string
  readonly model: string
  readonly parentAgentId?: string
  readonly parentSessionId?: string
  readonly permissionCeiling: PermissionMode
  readonly permissionMode: PermissionMode
  readonly profile: string
  readonly projectRoot: string
  readonly rules: readonly string[]
  readonly title: string
  readonly toolsAllowed: readonly string[]
  readonly toolsExcluded: readonly string[]
  readonly toolsWhitelist: readonly string[]
  readonly toolsets: readonly string[]
}

interface ActiveConversationOwners {
  direct: number
  native: number
}

const activeConversationOwners = new Map<string, ActiveConversationOwners>()

/** Reserve one child transcript against a second direct resume in this daemon. */
export function claimSubagentConversation(historySessionId: string): () => void {
  return claimConversation(historySessionId, 'native')
}

/** Claim a completed child for direct TUI use, failing closed against live native work. */
export function claimDirectSubagentConversation(historySessionId: string): () => void {
  const owners = activeConversationOwners.get(historySessionId)
  if (owners && owners.direct + owners.native > 0) {
    throw new Error('Subagent conversation is already active')
  }
  return claimConversation(historySessionId, 'direct')
}

function claimConversation(historySessionId: string, owner: 'direct' | 'native'): () => void {
  const owners = activeConversationOwners.get(historySessionId) ?? { direct: 0, native: 0 }
  if (owner === 'native' && owners.direct > 0) {
    throw new Error('Subagent conversation is open as a direct session')
  }
  owners[owner] += 1
  activeConversationOwners.set(historySessionId, owners)
  let released = false
  return () => {
    if (released) return
    released = true
    const current = activeConversationOwners.get(historySessionId)
    if (!current) return
    current[owner] = Math.max(0, current[owner] - 1)
    if (current.direct + current.native > 0) activeConversationOwners.set(historySessionId, current)
    else activeConversationOwners.delete(historySessionId)
  }
}

/** True only while this process has a native child actively writing the transcript. */
export function isSubagentConversationActive(historySessionId: string): boolean {
  return activeConversationOwners.has(historySessionId)
}

/**
 * Owns complete child-agent state independently from the bounded task summary
 * stored on the parent. The daemon injects the same atomic transcript store it
 * uses for main sessions; tests remain ephemeral unless they inject one too.
 */
export class SubagentConversationPersistence {
  private readonly states = new Map<string, Promise<AgentState>>()

  constructor(private readonly transcripts?: DaemonTranscriptStore) {}

  async stateFor(context: SubagentConversationContext): Promise<AgentState> {
    const existing = this.states.get(context.historySessionId)
    if (existing) return existing

    const loading = this.loadState(context)
    this.states.set(context.historySessionId, loading)
    try {
      return await loading
    } catch (error) {
      this.states.delete(context.historySessionId)
      throw error
    }
  }

  async save(
    context: SubagentConversationContext,
    state: AgentState,
    status: PersistedSubagentStatus,
    error?: unknown,
    checkpoint?: SubagentConversationCheckpoint,
  ): Promise<void> {
    if (!this.transcripts) return
    const metadata = conversationMetadata(context, status, error)
    state.metadata = { ...state.metadata, ...metadata }
    await this.transcripts.save(transcriptFromState(context, state, metadata, checkpoint))
  }

  private async loadState(context: SubagentConversationContext): Promise<AgentState> {
    if (!this.transcripts) return createAgentState()
    const transcript = await this.transcripts.load(context.historySessionId, {
      currentProjectDirectory: context.projectRoot,
    })
    const state = createAgentState(transcript?.messages.flatMap(rawMessageToChatMessage) ?? [])
    if (!transcript) return state

    state.apiCallsComplete = transcript.apiCallsComplete ?? transcript.turnCount === 0
    state.metadata = { ...transcript.metadata }
    state.thinkingContent = transcript.thinkingContent.filter((value): value is string => typeof value === 'string')
    state.toolExecutions = transcript.toolExecutions.filter(isToolExecutionRecord)
    state.totalApiCalls = transcript.totalApiCalls ?? 0
    state.totalInputTokens = transcript.totalInputTokens
    state.totalOutputTokens = transcript.totalOutputTokens
    state.turnCount = transcript.turnCount
    state.usageComplete = transcript.usageComplete ?? transcript.turnCount === 0
    return state
  }
}

function transcriptFromState(
  context: SubagentConversationContext,
  state: AgentState,
  metadata: Readonly<Record<string, unknown>>,
  checkpoint?: SubagentConversationCheckpoint,
): DaemonTranscript {
  const checkpointMessage = partialAssistantMessage(checkpoint)
  return {
    agentId: context.agentId,
    apiCallsComplete: state.apiCallsComplete,
    cwd: context.cwd,
    extra: {},
    format: 'bun-v2',
    interactionMode: 'code',
    key: context.historySessionId,
    messages: [
      ...state.messages.map(chatMessageToRawMessage),
      ...(checkpointMessage ? [checkpointMessage] : []),
    ],
    metadata,
    pendingResumeReplays: [],
    planMode: false,
    schemaVersion: undefined,
    sessionId: context.historySessionId,
    thinkingContent: checkpoint?.thinking
      ? [...state.thinkingContent, checkpoint.thinking]
      : state.thinkingContent,
    toolExecutions: state.toolExecutions,
    totalApiCalls: state.totalApiCalls,
    totalInputTokens: state.totalInputTokens,
    totalOutputTokens: state.totalOutputTokens,
    turnCount: state.turnCount,
    updatedAt: new Date().toISOString(),
    usageComplete: state.usageComplete,
    workspace: '',
  }
}

function partialAssistantMessage(
  checkpoint: SubagentConversationCheckpoint | undefined,
): RawMessage | undefined {
  if (!checkpoint || (!checkpoint.content && !checkpoint.thinking)) return undefined
  return {
    role: 'assistant',
    // Provider thinking signatures are not available until the round commits.
    // Keep a thinking-only crash snapshot resumable as a nonempty text message;
    // signed reasoning from committed rounds remains preserved separately.
    content: checkpoint.content || '[interrupted while reasoning]',
    ...(checkpoint.thinking ? { thinking: checkpoint.thinking } : {}),
    checkpoint_partial: true,
  }
}

function conversationMetadata(
  context: SubagentConversationContext,
  status: PersistedSubagentStatus,
  error: unknown,
): Record<string, unknown> {
  const parentSessionId = context.parentSessionId?.trim() || null
  return {
    session_kind: 'subagent',
    parent_session_id: parentSessionId,
    root_session_id: parentSessionId ?? context.historySessionId,
    subagent_id: context.handleId,
    subagent_handle_id: context.handleId,
    history_session_id: context.historySessionId,
    source_agent_id: parentSessionId,
    creator_agent_id: context.creatorAgentId ?? null,
    parent_agent_id: context.parentAgentId ?? null,
    agent_profile: context.profile,
    prompt_profile: context.profile,
    title: context.title,
    status,
    model: context.model,
    project_root: context.projectRoot,
    permission_mode: context.permissionMode,
    delegated_permission_mode: context.permissionMode,
    permission_ceiling: context.permissionCeiling,
    tools_allowed: [...context.toolsAllowed],
    tools_excluded: [...context.toolsExcluded],
    tools_whitelist: [...context.toolsWhitelist],
    rules: [...context.rules],
    toolsets: [...context.toolsets],
    tool_policy: {
      rules: [...context.rules],
      toolsets: [...context.toolsets],
    },
    last_error: error === undefined ? null : errorMessage(error),
  }
}

function chatMessageToRawMessage(message: ChatMessage): RawMessage {
  if (message.role !== 'user' || !message.displayText) return { ...message }
  const { displayText, ...providerMessage } = message
  return { ...providerMessage, text: displayText }
}

function rawMessageToChatMessage(message: RawMessage): ChatMessage[] {
  const role = message.role
  const content = message.content
  if (role === 'assistant' && isMessageContent(content)) {
    return [{
      role,
      content,
      ...(typeof message.thinking === 'string' ? { thinking: message.thinking } : {}),
      ...(typeof message.thinking_signature === 'string'
        ? { thinking_signature: message.thinking_signature }
        : {}),
      ...(Array.isArray(message.tool_calls) ? { tool_calls: message.tool_calls as readonly ToolCall[] } : {}),
    }]
  }
  if (role === 'system' && isMessageContent(content)) return [{ role, content }]
  if (role === 'user' && isMessageContent(content)) {
    return [{
      role,
      content,
      ...(typeof message.text === 'string' ? { displayText: message.text } : {}),
    }]
  }
  if (role === 'tool' && typeof content === 'string' && typeof message.tool_call_id === 'string') {
    return [{
      role,
      content,
      tool_call_id: message.tool_call_id,
      ...(typeof message.name === 'string' ? { name: message.name } : {}),
      ...(message.is_error === true ? { is_error: true } : {}),
    }]
  }
  return []
}

function isMessageContent(value: unknown): value is MessageContent {
  return typeof value === 'string' || Array.isArray(value)
}

function isToolExecutionRecord(value: unknown): value is AgentState['toolExecutions'][number] {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const record = value as Record<string, unknown>
  return typeof record.durationMs === 'number'
    && typeof record.name === 'string'
    && typeof record.permitted === 'boolean'
    && typeof record.result === 'string'
    && typeof record.toolCallId === 'string'
    && typeof record.inputs === 'object'
    && record.inputs !== null
    && !Array.isArray(record.inputs)
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
