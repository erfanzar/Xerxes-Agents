// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'

/** Current wire-schema version for durable session records. */
export const CURRENT_SESSION_SCHEMA_VERSION = 1
/** Compatibility name shared with the original runtime's session schema. */
export const CURRENT_SCHEMA_VERSION = CURRENT_SESSION_SCHEMA_VERSION

export type SessionId = string
export type WorkspaceId = string
export type SessionRecordData = Record<string, unknown>

export interface ToolCallRecordOptions {
  readonly arguments: Record<string, unknown>
  readonly callId: string
  readonly durationMs?: number | null
  readonly error?: string | null
  readonly extra?: Record<string, unknown>
  readonly metadata?: Record<string, unknown>
  readonly result?: unknown
  readonly sandboxContext?: string | null
  readonly status?: string
  readonly toolName: string
}

/** One tool invocation captured while completing a recorded turn. */
export class ToolCallRecord {
  arguments: Record<string, unknown>
  callId: string
  durationMs: number | null
  error: string | null
  extra: Record<string, unknown>
  metadata: Record<string, unknown>
  result: unknown
  sandboxContext: string | null
  status: string
  toolName: string

  constructor(options: ToolCallRecordOptions) {
    this.callId = options.callId
    this.toolName = options.toolName
    this.arguments = { ...options.arguments }
    this.result = options.result ?? null
    this.status = options.status ?? 'success'
    this.error = options.error ?? null
    this.durationMs = options.durationMs ?? null
    this.sandboxContext = options.sandboxContext ?? null
    this.metadata = { ...options.metadata }
    this.extra = { ...options.extra }
  }

  toRecord(): SessionRecordData {
    return {
      ...this.extra,
      call_id: this.callId,
      tool_name: this.toolName,
      arguments: { ...this.arguments },
      result: this.result,
      status: this.status,
      error: this.error,
      duration_ms: this.durationMs,
      sandbox_context: this.sandboxContext,
      metadata: { ...this.metadata },
    }
  }

  toJSON(): SessionRecordData {
    return this.toRecord()
  }

  static fromRecord(value: unknown): ToolCallRecord {
    const data = requiredRecord(value, 'tool_call')
    const status = stringOrUndefined(data.status)
    const error = nullableString(data.error)
    const durationMs = nullableNumber(data.duration_ms)
    const sandboxContext = nullableString(data.sandbox_context)
    return new ToolCallRecord({
      callId: requiredString(data.call_id, 'tool_call.call_id'),
      toolName: requiredString(data.tool_name, 'tool_call.tool_name'),
      arguments: recordValue(data.arguments),
      ...(hasOwn(data, 'result') ? { result: data.result } : {}),
      ...(status === undefined ? {} : { status }),
      ...(error === undefined ? {} : { error }),
      ...(durationMs === undefined ? {} : { durationMs }),
      ...(sandboxContext === undefined ? {} : { sandboxContext }),
      metadata: recordValue(data.metadata),
      extra: extraFields(data, TOOL_CALL_FIELDS),
    })
  }
}

export interface TurnRecordOptions {
  readonly agentId?: string | null
  readonly auditEventIds?: readonly string[]
  readonly endedAt?: string | null
  readonly error?: string | null
  readonly extra?: Record<string, unknown>
  readonly metadata?: Record<string, unknown>
  readonly prompt?: string
  readonly responseContent?: string | null
  readonly startedAt?: string
  readonly status?: string
  readonly toolCalls?: readonly ToolCallRecord[]
  readonly turnId: string
}

/** One user-prompt to agent-response cycle. */
export class TurnRecord {
  agentId: string | null
  auditEventIds: string[]
  endedAt: string | null
  error: string | null
  extra: Record<string, unknown>
  metadata: Record<string, unknown>
  prompt: string
  responseContent: string | null
  startedAt: string
  status: string
  toolCalls: ToolCallRecord[]
  turnId: string

  constructor(options: TurnRecordOptions) {
    this.turnId = options.turnId
    this.agentId = options.agentId ?? null
    this.prompt = options.prompt ?? ''
    this.responseContent = options.responseContent ?? null
    this.toolCalls = (options.toolCalls ?? []).map(toolCall => ToolCallRecord.fromRecord(toolCall.toRecord()))
    this.startedAt = options.startedAt ?? ''
    this.endedAt = options.endedAt ?? null
    this.status = options.status ?? 'success'
    this.error = options.error ?? null
    this.auditEventIds = [...options.auditEventIds ?? []]
    this.metadata = { ...options.metadata }
    this.extra = { ...options.extra }
  }

  toRecord(): SessionRecordData {
    return {
      ...this.extra,
      turn_id: this.turnId,
      agent_id: this.agentId,
      prompt: this.prompt,
      response_content: this.responseContent,
      tool_calls: this.toolCalls.map(toolCall => toolCall.toRecord()),
      started_at: this.startedAt,
      ended_at: this.endedAt,
      status: this.status,
      error: this.error,
      audit_event_ids: [...this.auditEventIds],
      metadata: { ...this.metadata },
    }
  }

  toJSON(): SessionRecordData {
    return this.toRecord()
  }

  static fromRecord(value: unknown): TurnRecord {
    const data = requiredRecord(value, 'turn')
    const agentId = nullableString(data.agent_id)
    const prompt = stringOrUndefined(data.prompt)
    const responseContent = nullableString(data.response_content)
    const startedAt = stringOrUndefined(data.started_at)
    const endedAt = nullableString(data.ended_at)
    const status = stringOrUndefined(data.status)
    const error = nullableString(data.error)
    return new TurnRecord({
      turnId: requiredString(data.turn_id, 'turn.turn_id'),
      ...(agentId === undefined ? {} : { agentId }),
      ...(prompt === undefined ? {} : { prompt }),
      ...(responseContent === undefined ? {} : { responseContent }),
      toolCalls: arrayValue(data.tool_calls).map(ToolCallRecord.fromRecord),
      ...(startedAt === undefined ? {} : { startedAt }),
      ...(endedAt === undefined ? {} : { endedAt }),
      ...(status === undefined ? {} : { status }),
      ...(error === undefined ? {} : { error }),
      auditEventIds: stringArrayValue(data.audit_event_ids),
      metadata: recordValue(data.metadata),
      extra: extraFields(data, TURN_FIELDS),
    })
  }
}

export interface AgentTransitionRecordOptions {
  readonly extra?: Record<string, unknown>
  readonly fromAgent?: string | null
  readonly reason?: string | null
  readonly timestamp?: string
  readonly toAgent: string
  readonly turnId?: string
}

/** A marker for handing a session from one agent to another. */
export class AgentTransitionRecord {
  extra: Record<string, unknown>
  fromAgent: string | null
  reason: string | null
  timestamp: string
  toAgent: string
  turnId: string

  constructor(options: AgentTransitionRecordOptions) {
    this.fromAgent = options.fromAgent ?? null
    this.toAgent = options.toAgent
    this.reason = options.reason ?? null
    this.turnId = options.turnId ?? ''
    this.timestamp = options.timestamp ?? ''
    this.extra = { ...options.extra }
  }

  toRecord(): SessionRecordData {
    return {
      ...this.extra,
      from_agent: this.fromAgent,
      to_agent: this.toAgent,
      reason: this.reason,
      turn_id: this.turnId,
      timestamp: this.timestamp,
    }
  }

  toJSON(): SessionRecordData {
    return this.toRecord()
  }

  static fromRecord(value: unknown): AgentTransitionRecord {
    const data = requiredRecord(value, 'agent_transition')
    const fromAgent = nullableString(data.from_agent)
    const reason = nullableString(data.reason)
    const turnId = stringOrUndefined(data.turn_id)
    const timestamp = stringOrUndefined(data.timestamp)
    return new AgentTransitionRecord({
      toAgent: requiredString(data.to_agent, 'agent_transition.to_agent'),
      ...(fromAgent === undefined ? {} : { fromAgent }),
      ...(reason === undefined ? {} : { reason }),
      ...(turnId === undefined ? {} : { turnId }),
      ...(timestamp === undefined ? {} : { timestamp }),
      extra: extraFields(data, AGENT_TRANSITION_FIELDS),
    })
  }
}

export interface SessionRecordOptions {
  readonly agentId?: string | null
  readonly agentTransitions?: readonly AgentTransitionRecord[]
  readonly createdAt?: string
  readonly extra?: Record<string, unknown>
  readonly metadata?: Record<string, unknown>
  readonly parentSessionId?: string | null
  readonly schemaVersion?: number
  readonly sessionId: string
  readonly turns?: readonly TurnRecord[]
  readonly updatedAt?: string
  readonly workspaceId?: string | null
}

/** Root persistence record for a conversation, its turns, and its handoffs. */
export class SessionRecord {
  agentId: string | null
  agentTransitions: AgentTransitionRecord[]
  createdAt: string
  extra: Record<string, unknown>
  metadata: Record<string, unknown>
  parentSessionId: string | null
  schemaVersion: number
  sessionId: string
  turns: TurnRecord[]
  updatedAt: string
  workspaceId: string | null

  constructor(options: SessionRecordOptions) {
    this.sessionId = options.sessionId
    this.workspaceId = options.workspaceId ?? null
    this.createdAt = options.createdAt ?? ''
    this.updatedAt = options.updatedAt ?? ''
    this.agentId = options.agentId ?? null
    this.turns = (options.turns ?? []).map(turn => TurnRecord.fromRecord(turn.toRecord()))
    this.agentTransitions = (options.agentTransitions ?? []).map(transition =>
      AgentTransitionRecord.fromRecord(transition.toRecord()),
    )
    this.metadata = { ...options.metadata }
    this.parentSessionId = options.parentSessionId ?? null
    this.schemaVersion = normalizeSchemaVersion(options.schemaVersion)
    this.extra = { ...options.extra }
  }

  toRecord(): SessionRecordData {
    return {
      ...this.extra,
      schema_version: this.schemaVersion,
      session_id: this.sessionId,
      workspace_id: this.workspaceId,
      created_at: this.createdAt,
      updated_at: this.updatedAt,
      agent_id: this.agentId,
      turns: this.turns.map(turn => turn.toRecord()),
      agent_transitions: this.agentTransitions.map(transition => transition.toRecord()),
      metadata: { ...this.metadata },
      parent_session_id: this.parentSessionId,
    }
  }

  toJSON(): SessionRecordData {
    return this.toRecord()
  }

  static fromRecord(value: unknown): SessionRecord {
    const data = requiredRecord(value, 'session')
    const workspaceId = nullableString(data.workspace_id)
    const createdAt = stringOrUndefined(data.created_at)
    const updatedAt = stringOrUndefined(data.updated_at)
    const agentId = nullableString(data.agent_id)
    const parentSessionId = nullableString(data.parent_session_id)
    return new SessionRecord({
      sessionId: requiredString(data.session_id, 'session.session_id'),
      ...(workspaceId === undefined ? {} : { workspaceId }),
      ...(createdAt === undefined ? {} : { createdAt }),
      ...(updatedAt === undefined ? {} : { updatedAt }),
      ...(agentId === undefined ? {} : { agentId }),
      turns: arrayValue(data.turns).map(TurnRecord.fromRecord),
      agentTransitions: arrayValue(data.agent_transitions).map(AgentTransitionRecord.fromRecord),
      metadata: recordValue(data.metadata),
      ...(parentSessionId === undefined ? {} : { parentSessionId }),
      schemaVersion: normalizeSchemaVersion(numberValue(data.schema_version)),
      extra: extraFields(data, SESSION_FIELDS),
    })
  }
}

/** Return a deep record-level copy suitable for a session branch. */
export function cloneSessionRecord(session: SessionRecord): SessionRecord {
  return SessionRecord.fromRecord(deepCopyRecord(session.toRecord()))
}

const TOOL_CALL_FIELDS = new Set([
  'call_id',
  'tool_name',
  'arguments',
  'result',
  'status',
  'error',
  'duration_ms',
  'sandbox_context',
  'metadata',
])
const TURN_FIELDS = new Set([
  'turn_id',
  'agent_id',
  'prompt',
  'response_content',
  'tool_calls',
  'started_at',
  'ended_at',
  'status',
  'error',
  'audit_event_ids',
  'metadata',
])
const AGENT_TRANSITION_FIELDS = new Set(['from_agent', 'to_agent', 'reason', 'turn_id', 'timestamp'])
const SESSION_FIELDS = new Set([
  'schema_version',
  'session_id',
  'workspace_id',
  'created_at',
  'updated_at',
  'agent_id',
  'turns',
  'agent_transitions',
  'metadata',
  'parent_session_id',
])

function arrayValue(value: unknown): unknown[] {
  return Array.isArray(value) ? value : []
}

function deepCopyRecord(record: SessionRecordData): SessionRecordData {
  const copy: SessionRecordData = {}
  for (const [key, value] of Object.entries(record)) copy[key] = deepCopyValue(value)
  return copy
}

function deepCopyValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(deepCopyValue)
  if (isPlainRecord(value)) return deepCopyRecord(value)
  return value
}

function isPlainRecord(value: unknown): value is SessionRecordData {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

function extraFields(data: SessionRecordData, fields: ReadonlySet<string>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(data).filter(([key]) => !fields.has(key)))
}

function hasOwn(data: SessionRecordData, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(data, key)
}

function normalizeSchemaVersion(value: number | undefined): number {
  if (value === undefined) return CURRENT_SESSION_SCHEMA_VERSION
  if (!Number.isInteger(value) || value < 1) return CURRENT_SESSION_SCHEMA_VERSION
  return value
}

function nullableNumber(value: unknown): number | null | undefined {
  if (value === null) return null
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function nullableString(value: unknown): string | null | undefined {
  if (value === null) return null
  return typeof value === 'string' ? value : undefined
}

function numberValue(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function recordValue(value: unknown): Record<string, unknown> {
  return isRecord(value) ? { ...value } : {}
}

function requiredRecord(value: unknown, field: string): SessionRecordData {
  if (!isRecord(value)) throw new ValidationError(field, 'must be an object', value)
  return value
}

function requiredString(value: unknown, field: string): string {
  if (typeof value !== 'string' || value.length === 0) throw new ValidationError(field, 'must be a non-empty string', value)
  return value
}

function stringArrayValue(value: unknown): string[] {
  return arrayValue(value).flatMap(item => typeof item === 'string' ? [item] : [])
}

function stringOrUndefined(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined
}

function isRecord(value: unknown): value is SessionRecordData {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
