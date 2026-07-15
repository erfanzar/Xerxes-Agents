// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Stable discriminators persisted by JSONL and OpenTelemetry audit sinks. */
export const AuditEventTypes = {
  AGENT_SWITCH: 'agent_switch',
  BASE: 'base',
  ERROR: 'error',
  HOOK_MUTATION: 'hook_mutation',
  SANDBOX_DECISION: 'sandbox_decision',
  SKILL_AUTHORED: 'skill_authored',
  SKILL_FEEDBACK: 'skill_feedback',
  SKILL_USED: 'skill_used',
  TOOL_CALL_ATTEMPT: 'tool_call_attempt',
  TOOL_CALL_COMPLETE: 'tool_call_complete',
  TOOL_CALL_FAILURE: 'tool_call_failure',
  TOOL_LOOP_BLOCK: 'tool_loop_block',
  TOOL_LOOP_WARNING: 'tool_loop_warning',
  TOOL_POLICY_DECISION: 'tool_policy_decision',
  TURN_END: 'turn_end',
  TURN_START: 'turn_start',
} as const

export type AuditEventType = (typeof AuditEventTypes)[keyof typeof AuditEventTypes]
export type AuditMetadata = Record<string, unknown>

/** JSON-compatible shape written by audit sinks. Field names match the Python JSONL contract. */
export interface AuditEventRecord extends Record<string, unknown> {
  readonly agent_id: string | null
  readonly event_type: string
  readonly metadata: Record<string, unknown>
  readonly session_id: string | null
  readonly severity: string
  readonly timestamp: string
  readonly turn_id: string | null
}

export interface AuditEventOptions {
  readonly agentId?: string | null
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly sessionId?: string | null
  readonly severity?: string
  readonly timestamp?: string
  readonly turnId?: string | null
}

/** Return the current instant as an ISO-8601 UTC string. */
export function nowIso(): string {
  return new Date().toISOString()
}

/** Base record shared by all structured audit events. */
export class AuditEvent {
  readonly eventType: AuditEventType = AuditEventTypes.BASE
  agentId: string | undefined
  metadata: AuditMetadata
  sessionId: string | undefined
  severity: string
  timestamp: string
  turnId: string | undefined

  constructor(options: AuditEventOptions = {}) {
    this.timestamp = options.timestamp ?? nowIso()
    this.agentId = options.agentId ?? undefined
    this.turnId = options.turnId ?? undefined
    this.sessionId = options.sessionId ?? undefined
    this.severity = options.severity ?? 'info'
    this.metadata = copyMetadata(options.metadata)
  }

  /** Return a detached JSON-safe record suitable for persistence. */
  toRecord(): AuditEventRecord {
    return this.baseRecord()
  }

  /** Serialize the event without requiring consumers to install a JSON extension. */
  toJson(): string {
    return JSON.stringify(this.toRecord())
  }

  protected baseRecord(): AuditEventRecord {
    return {
      event_type: this.eventType,
      timestamp: this.timestamp,
      agent_id: this.agentId ?? null,
      turn_id: this.turnId ?? null,
      session_id: this.sessionId ?? null,
      severity: this.severity,
      metadata: copyMetadata(this.metadata),
    }
  }
}

export interface TurnStartEventOptions extends AuditEventOptions {
  readonly promptPreview?: string
}

/** Emitted when an agent turn starts. */
export class TurnStartEvent extends AuditEvent {
  override readonly eventType = AuditEventTypes.TURN_START
  promptPreview: string

  constructor(options: TurnStartEventOptions = {}) {
    super(options)
    this.promptPreview = options.promptPreview ?? ''
  }

  override toRecord(): AuditEventRecord {
    return { ...this.baseRecord(), prompt_preview: this.promptPreview }
  }
}

export interface TurnEndEventOptions extends AuditEventOptions {
  readonly contentPreview?: string
  readonly functionCallsCount?: number
}

/** Emitted once an agent turn finishes. */
export class TurnEndEvent extends AuditEvent {
  override readonly eventType = AuditEventTypes.TURN_END
  contentPreview: string
  functionCallsCount: number

  constructor(options: TurnEndEventOptions = {}) {
    super(options)
    this.contentPreview = options.contentPreview ?? ''
    this.functionCallsCount = options.functionCallsCount ?? 0
  }

  override toRecord(): AuditEventRecord {
    return {
      ...this.baseRecord(),
      content_preview: this.contentPreview,
      function_calls_count: this.functionCallsCount,
    }
  }
}

export interface ToolCallAttemptEventOptions extends AuditEventOptions {
  readonly argumentsPreview?: string
  readonly toolName?: string
}

/** Emitted immediately before a post-policy tool execution starts. */
export class ToolCallAttemptEvent extends AuditEvent {
  override readonly eventType = AuditEventTypes.TOOL_CALL_ATTEMPT
  argumentsPreview: string
  toolName: string

  constructor(options: ToolCallAttemptEventOptions = {}) {
    super(options)
    this.toolName = options.toolName ?? ''
    this.argumentsPreview = options.argumentsPreview ?? ''
  }

  override toRecord(): AuditEventRecord {
    return { ...this.baseRecord(), tool_name: this.toolName, arguments_preview: this.argumentsPreview }
  }
}

export interface ToolCallCompleteEventOptions extends AuditEventOptions {
  readonly durationMs?: number
  readonly resultPreview?: string
  readonly status?: string
  readonly toolName?: string
}

/** Emitted after a tool returns normally. */
export class ToolCallCompleteEvent extends AuditEvent {
  override readonly eventType = AuditEventTypes.TOOL_CALL_COMPLETE
  durationMs: number
  resultPreview: string
  status: string
  toolName: string

  constructor(options: ToolCallCompleteEventOptions = {}) {
    super(options)
    this.toolName = options.toolName ?? ''
    this.status = options.status ?? 'success'
    this.durationMs = options.durationMs ?? 0
    this.resultPreview = options.resultPreview ?? ''
  }

  override toRecord(): AuditEventRecord {
    return {
      ...this.baseRecord(),
      tool_name: this.toolName,
      status: this.status,
      duration_ms: this.durationMs,
      result_preview: this.resultPreview,
    }
  }
}

export interface ToolCallFailureEventOptions extends AuditEventOptions {
  readonly errorMessage?: string
  readonly errorType?: string
  readonly toolName?: string
}

/** Emitted when a tool raises or returns a failure status. */
export class ToolCallFailureEvent extends AuditEvent {
  override readonly eventType = AuditEventTypes.TOOL_CALL_FAILURE
  errorMessage: string
  errorType: string
  toolName: string

  constructor(options: ToolCallFailureEventOptions = {}) {
    super({ ...options, severity: options.severity ?? 'error' })
    this.toolName = options.toolName ?? ''
    this.errorType = options.errorType ?? ''
    this.errorMessage = options.errorMessage ?? ''
  }

  override toRecord(): AuditEventRecord {
    return {
      ...this.baseRecord(),
      tool_name: this.toolName,
      error_type: this.errorType,
      error_message: this.errorMessage,
    }
  }
}

export interface ToolPolicyDecisionEventOptions extends AuditEventOptions {
  readonly action?: string
  readonly policySource?: string
  readonly toolName?: string
}

/** Emitted after policy or permission resolves a tool request. */
export class ToolPolicyDecisionEvent extends AuditEvent {
  override readonly eventType = AuditEventTypes.TOOL_POLICY_DECISION
  action: string
  policySource: string
  toolName: string

  constructor(options: ToolPolicyDecisionEventOptions = {}) {
    super(options)
    this.toolName = options.toolName ?? ''
    this.action = options.action ?? ''
    this.policySource = options.policySource ?? ''
  }

  override toRecord(): AuditEventRecord {
    return {
      ...this.baseRecord(),
      tool_name: this.toolName,
      action: this.action,
      policy_source: this.policySource,
    }
  }
}

export interface SandboxDecisionEventOptions extends AuditEventOptions {
  readonly backendType?: string
  readonly context?: string
  readonly reason?: string
  readonly toolName?: string
}

/** Emitted when the sandbox layer routes a tool to a backend. */
export class SandboxDecisionEvent extends AuditEvent {
  override readonly eventType = AuditEventTypes.SANDBOX_DECISION
  backendType: string
  context: string
  reason: string
  toolName: string

  constructor(options: SandboxDecisionEventOptions = {}) {
    super(options)
    this.toolName = options.toolName ?? ''
    this.context = options.context ?? ''
    this.reason = options.reason ?? ''
    this.backendType = options.backendType ?? ''
  }

  override toRecord(): AuditEventRecord {
    return {
      ...this.baseRecord(),
      tool_name: this.toolName,
      context: this.context,
      reason: this.reason,
      backend_type: this.backendType,
    }
  }
}

export interface ToolLoopWarningEventOptions extends AuditEventOptions {
  readonly callCount?: number
  readonly pattern?: string
  readonly severityLevel?: string
  readonly toolName?: string
}

/** Emitted when loop detection observes suspicious repeated tool use. */
export class ToolLoopWarningEvent extends AuditEvent {
  override readonly eventType = AuditEventTypes.TOOL_LOOP_WARNING
  callCount: number
  pattern: string
  severityLevel: string
  toolName: string

  constructor(options: ToolLoopWarningEventOptions = {}) {
    super({ ...options, severity: options.severity ?? 'warning' })
    this.toolName = options.toolName ?? ''
    this.pattern = options.pattern ?? ''
    this.severityLevel = options.severityLevel ?? ''
    this.callCount = options.callCount ?? 0
  }

  override toRecord(): AuditEventRecord {
    return {
      ...this.baseRecord(),
      tool_name: this.toolName,
      pattern: this.pattern,
      severity_level: this.severityLevel,
      call_count: this.callCount,
    }
  }
}

export interface ToolLoopBlockEventOptions extends AuditEventOptions {
  readonly callCount?: number
  readonly pattern?: string
  readonly toolName?: string
}

/** Emitted when loop detection blocks further execution. */
export class ToolLoopBlockEvent extends AuditEvent {
  override readonly eventType = AuditEventTypes.TOOL_LOOP_BLOCK
  callCount: number
  pattern: string
  toolName: string

  constructor(options: ToolLoopBlockEventOptions = {}) {
    super({ ...options, severity: options.severity ?? 'error' })
    this.toolName = options.toolName ?? ''
    this.pattern = options.pattern ?? ''
    this.callCount = options.callCount ?? 0
  }

  override toRecord(): AuditEventRecord {
    return {
      ...this.baseRecord(),
      tool_name: this.toolName,
      pattern: this.pattern,
      call_count: this.callCount,
    }
  }
}

export interface HookMutationEventOptions extends AuditEventOptions {
  readonly hookName?: string
  readonly mutatedField?: string
  readonly toolName?: string
}

/** Emitted when an extension hook rewrites a tool field. */
export class HookMutationEvent extends AuditEvent {
  override readonly eventType = AuditEventTypes.HOOK_MUTATION
  hookName: string
  mutatedField: string
  toolName: string

  constructor(options: HookMutationEventOptions = {}) {
    super(options)
    this.hookName = options.hookName ?? ''
    this.toolName = options.toolName ?? ''
    this.mutatedField = options.mutatedField ?? ''
  }

  override toRecord(): AuditEventRecord {
    return {
      ...this.baseRecord(),
      hook_name: this.hookName,
      tool_name: this.toolName,
      mutated_field: this.mutatedField,
    }
  }
}

export interface ErrorEventOptions extends AuditEventOptions {
  readonly errorContext?: string
  readonly errorMessage?: string
  readonly errorType?: string
}

/** Emitted for a non-tool error. */
export class ErrorEvent extends AuditEvent {
  override readonly eventType = AuditEventTypes.ERROR
  errorContext: string
  errorMessage: string
  errorType: string

  constructor(options: ErrorEventOptions = {}) {
    super({ ...options, severity: options.severity ?? 'error' })
    this.errorType = options.errorType ?? ''
    this.errorMessage = options.errorMessage ?? ''
    this.errorContext = options.errorContext ?? ''
  }

  override toRecord(): AuditEventRecord {
    return {
      ...this.baseRecord(),
      error_type: this.errorType,
      error_message: this.errorMessage,
      error_context: this.errorContext,
    }
  }
}

export interface SkillUsedEventOptions extends AuditEventOptions {
  readonly durationMs?: number
  readonly outcome?: string
  readonly skillName?: string
  readonly triggeredAutomatically?: boolean
  readonly version?: string
}

/** Emitted when a skill bundle is invoked. */
export class SkillUsedEvent extends AuditEvent {
  override readonly eventType = AuditEventTypes.SKILL_USED
  durationMs: number
  outcome: string
  skillName: string
  triggeredAutomatically: boolean
  version: string

  constructor(options: SkillUsedEventOptions = {}) {
    super(options)
    this.skillName = options.skillName ?? ''
    this.version = options.version ?? ''
    this.outcome = options.outcome ?? 'unknown'
    this.durationMs = options.durationMs ?? 0
    this.triggeredAutomatically = options.triggeredAutomatically ?? true
  }

  override toRecord(): AuditEventRecord {
    return {
      ...this.baseRecord(),
      skill_name: this.skillName,
      version: this.version,
      outcome: this.outcome,
      duration_ms: this.durationMs,
      triggered_automatically: this.triggeredAutomatically,
    }
  }
}

export interface SkillAuthoredEventOptions extends AuditEventOptions {
  readonly confirmedByUser?: boolean
  readonly skillName?: string
  readonly sourcePath?: string
  readonly toolCount?: number
  readonly uniqueTools?: readonly string[]
  readonly version?: string
}

/** Emitted when a skill bundle is authored or updated. */
export class SkillAuthoredEvent extends AuditEvent {
  override readonly eventType = AuditEventTypes.SKILL_AUTHORED
  confirmedByUser: boolean
  skillName: string
  sourcePath: string
  toolCount: number
  uniqueTools: string[]
  version: string

  constructor(options: SkillAuthoredEventOptions = {}) {
    super(options)
    this.skillName = options.skillName ?? ''
    this.version = options.version ?? ''
    this.sourcePath = options.sourcePath ?? ''
    this.toolCount = options.toolCount ?? 0
    this.uniqueTools = [...(options.uniqueTools ?? [])]
    this.confirmedByUser = options.confirmedByUser ?? false
  }

  override toRecord(): AuditEventRecord {
    return {
      ...this.baseRecord(),
      skill_name: this.skillName,
      version: this.version,
      source_path: this.sourcePath,
      tool_count: this.toolCount,
      unique_tools: [...this.uniqueTools],
      confirmed_by_user: this.confirmedByUser,
    }
  }
}

export interface SkillFeedbackEventOptions extends AuditEventOptions {
  readonly rating?: string
  readonly reason?: string
  readonly skillName?: string
  readonly source?: string
}

/** Emitted when a user or agent rates a skill execution. */
export class SkillFeedbackEvent extends AuditEvent {
  override readonly eventType = AuditEventTypes.SKILL_FEEDBACK
  rating: string
  reason: string
  skillName: string
  source: string

  constructor(options: SkillFeedbackEventOptions = {}) {
    super(options)
    this.skillName = options.skillName ?? ''
    this.rating = options.rating ?? 'neutral'
    this.reason = options.reason ?? ''
    this.source = options.source ?? 'user'
  }

  override toRecord(): AuditEventRecord {
    return {
      ...this.baseRecord(),
      skill_name: this.skillName,
      rating: this.rating,
      reason: this.reason,
      source: this.source,
    }
  }
}

export interface AgentSwitchEventOptions extends AuditEventOptions {
  readonly fromAgent?: string
  readonly reason?: string
  readonly toAgent?: string
}

/** Emitted when control transfers between agents mid-turn. */
export class AgentSwitchEvent extends AuditEvent {
  override readonly eventType = AuditEventTypes.AGENT_SWITCH
  fromAgent: string
  reason: string
  toAgent: string

  constructor(options: AgentSwitchEventOptions = {}) {
    super(options)
    this.fromAgent = options.fromAgent ?? ''
    this.toAgent = options.toAgent ?? ''
    this.reason = options.reason ?? ''
  }

  override toRecord(): AuditEventRecord {
    return {
      ...this.baseRecord(),
      from_agent: this.fromAgent,
      to_agent: this.toAgent,
      reason: this.reason,
    }
  }
}

function copyMetadata(metadata: Readonly<Record<string, unknown>> | undefined): AuditMetadata {
  return cloneAuditValue(metadata ?? {}) as AuditMetadata
}

function cloneAuditValue(value: unknown, seen = new WeakSet<object>()): unknown {
  if (value === null || value === undefined) return value ?? null
  if (typeof value === 'string' || typeof value === 'boolean') return value
  if (typeof value === 'number') return Number.isFinite(value) ? value : String(value)
  if (typeof value === 'bigint') return value.toString()
  if (typeof value === 'symbol' || typeof value === 'function') return String(value)
  if (value instanceof Date) return value.toISOString()
  if (Array.isArray(value)) return value.map(item => cloneAuditValue(item, seen))
  if (typeof value !== 'object') return String(value)
  if (seen.has(value)) return '[Circular]'
  seen.add(value)
  const record: Record<string, unknown> = {}
  for (const [key, item] of Object.entries(value)) {
    record[key] = cloneAuditValue(item, seen)
  }
  seen.delete(value)
  return record
}
