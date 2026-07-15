// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { InMemoryCollector, type AuditCollector } from './collector.js'
import { redactPayload, redactString } from '../security/redact.js'
import {
  AgentSwitchEvent,
  AuditEvent,
  ErrorEvent,
  HookMutationEvent,
  SandboxDecisionEvent,
  SkillAuthoredEvent,
  SkillFeedbackEvent,
  SkillUsedEvent,
  ToolCallAttemptEvent,
  ToolCallCompleteEvent,
  ToolCallFailureEvent,
  ToolLoopBlockEvent,
  ToolLoopWarningEvent,
  ToolPolicyDecisionEvent,
  TurnEndEvent,
  TurnStartEvent,
  type AuditEventOptions,
} from './events.js'

export interface AuditContext {
  readonly agentId?: string
  readonly sessionId?: string
  readonly turnId?: string
}

export interface LoopWarningHookPayload extends AuditContext {
  readonly count: number
  readonly pattern: string
  readonly severity: string
  readonly toolName: string
}

/** Minimal adapter implemented by the extension hook runner once it is wired into Bun. */
export interface AuditHookRunner {
  hasHooks(hookPoint: string): boolean
  run(hookPoint: string, payload: LoopWarningHookPayload): unknown
}

export interface AuditEmitterOptions {
  readonly collector?: AuditCollector
  readonly hookRunner?: AuditHookRunner
  readonly sessionId?: string
}

export interface TurnStartAuditInput extends AuditContext {
  readonly prompt?: unknown
}

export interface TurnEndAuditInput extends AuditContext {
  readonly content?: unknown
  readonly functionCallsCount?: number
}

export interface ToolCallAttemptAuditInput extends AuditContext {
  readonly args?: unknown
  readonly toolName: string
}

export interface ToolCallCompleteAuditInput extends AuditContext {
  readonly durationMs?: number
  readonly result?: unknown
  readonly status?: string
  readonly toolName: string
}

export interface ToolCallFailureAuditInput extends AuditContext {
  readonly errorMessage?: string
  readonly errorType?: string
  readonly toolName: string
}

export interface ToolPolicyDecisionAuditInput extends AuditContext {
  readonly action?: string
  readonly source?: string
  readonly toolName: string
}

export interface ToolLoopWarningAuditInput extends AuditContext {
  readonly count?: number
  readonly pattern?: string
  readonly severity?: string
  readonly toolName: string
}

export interface ToolLoopBlockAuditInput extends AuditContext {
  readonly count?: number
  readonly pattern?: string
  readonly toolName: string
}

export interface SandboxDecisionAuditInput extends AuditContext {
  readonly backendType?: string
  readonly context?: string
  readonly reason?: string
  readonly toolName: string
}

export interface HookMutationAuditInput extends AuditContext {
  readonly hookName: string
  readonly mutatedField?: string
  readonly toolName?: string
}

export interface ErrorAuditInput extends AuditContext {
  readonly context?: string
  readonly errorMessage?: string
  readonly errorType?: string
}

export interface SkillUsedAuditInput extends AuditContext {
  readonly durationMs?: number
  readonly outcome?: string
  readonly skillName: string
  readonly triggeredAutomatically?: boolean
  readonly version?: string
}

export interface SkillAuthoredAuditInput extends AuditContext {
  readonly confirmedByUser?: boolean
  readonly skillName: string
  readonly sourcePath?: string
  readonly toolCount?: number
  readonly uniqueTools?: readonly string[]
  readonly version?: string
}

export interface SkillFeedbackAuditInput extends AuditContext {
  readonly rating?: string
  readonly reason?: string
  readonly skillName: string
  readonly source?: string
}

export interface AgentSwitchAuditInput extends AuditContext {
  readonly fromAgent: string
  readonly reason?: string
  readonly toAgent: string
}

/**
 * Creates typed audit records and forwards them to one collector.
 *
 * The emitter stamps its current `sessionId` onto every emitted event. Callers may
 * update that property when a runtime opens or resumes another session.
 */
export class AuditEmitter {
  readonly collector: AuditCollector
  readonly hookRunner: AuditHookRunner | undefined
  sessionId: string | undefined

  constructor(options: AuditEmitterOptions = {}) {
    this.collector = options.collector ?? new InMemoryCollector()
    this.hookRunner = options.hookRunner
    this.sessionId = options.sessionId
  }

  /** Stamp the session id and send an already-constructed audit event. */
  emit(event: AuditEvent): void {
    if (this.sessionId !== undefined) {
      event.sessionId = this.sessionId
    }
    this.collector.emit(event)
  }

  emitTurnStart(input: TurnStartAuditInput = {}): string {
    const turnId = input.turnId ?? generateTurnId()
    this.emit(new TurnStartEvent({
      ...eventContext(input),
      turnId,
      promptPreview: preview(input.prompt),
    }))
    return turnId
  }

  emitTurnEnd(input: TurnEndAuditInput = {}): void {
    this.emit(new TurnEndEvent({
      ...eventContext(input),
      contentPreview: preview(input.content),
      functionCallsCount: input.functionCallsCount ?? 0,
    }))
  }

  emitToolCallAttempt(input: ToolCallAttemptAuditInput): void {
    this.emit(new ToolCallAttemptEvent({
      ...eventContext(input),
      toolName: input.toolName,
      argumentsPreview: preview(input.args),
    }))
  }

  emitToolCallComplete(input: ToolCallCompleteAuditInput): void {
    this.emit(new ToolCallCompleteEvent({
      ...eventContext(input),
      toolName: input.toolName,
      status: input.status ?? 'success',
      durationMs: input.durationMs ?? 0,
      resultPreview: preview(input.result),
    }))
  }

  emitToolCallFailure(input: ToolCallFailureAuditInput): void {
    this.emit(new ToolCallFailureEvent({
      ...eventContext(input),
      toolName: input.toolName,
      errorType: input.errorType ?? '',
      errorMessage: input.errorMessage ?? '',
    }))
  }

  emitToolPolicyDecision(input: ToolPolicyDecisionAuditInput): void {
    this.emit(new ToolPolicyDecisionEvent({
      ...eventContext(input),
      toolName: input.toolName,
      action: input.action ?? '',
      policySource: input.source ?? '',
    }))
  }

  /** Emit a warning directly, even when loop-warning hooks are registered. */
  emitToolLoopWarning(input: ToolLoopWarningAuditInput): void {
    this.emitLoopWarningEvent(input)
  }

  emitToolLoopBlock(input: ToolLoopBlockAuditInput): void {
    this.emit(new ToolLoopBlockEvent({
      ...eventContext(input),
      toolName: input.toolName,
      pattern: input.pattern ?? '',
      callCount: input.count ?? 0,
    }))
  }

  emitSandboxDecision(input: SandboxDecisionAuditInput): void {
    this.emit(new SandboxDecisionEvent({
      ...eventContext(input),
      toolName: input.toolName,
      context: input.context ?? '',
      reason: input.reason ?? '',
      backendType: input.backendType ?? '',
    }))
  }

  emitHookMutation(input: HookMutationAuditInput): void {
    this.emit(new HookMutationEvent({
      ...eventContext(input),
      hookName: input.hookName,
      toolName: input.toolName ?? '',
      mutatedField: input.mutatedField ?? '',
    }))
  }

  emitError(input: ErrorAuditInput = {}): void {
    this.emit(new ErrorEvent({
      ...eventContext(input),
      errorType: input.errorType ?? '',
      errorMessage: input.errorMessage ?? '',
      errorContext: input.context ?? '',
    }))
  }

  emitSkillUsed(input: SkillUsedAuditInput): void {
    this.emit(new SkillUsedEvent({
      ...eventContext(input),
      skillName: input.skillName,
      version: input.version ?? '',
      outcome: input.outcome ?? 'unknown',
      durationMs: input.durationMs ?? 0,
      triggeredAutomatically: input.triggeredAutomatically ?? true,
    }))
  }

  emitSkillAuthored(input: SkillAuthoredAuditInput): void {
    this.emit(new SkillAuthoredEvent({
      ...eventContext(input),
      skillName: input.skillName,
      version: input.version ?? '',
      sourcePath: input.sourcePath ?? '',
      toolCount: input.toolCount ?? 0,
      uniqueTools: input.uniqueTools ?? [],
      confirmedByUser: input.confirmedByUser ?? false,
    }))
  }

  emitSkillFeedback(input: SkillFeedbackAuditInput): void {
    this.emit(new SkillFeedbackEvent({
      ...eventContext(input),
      skillName: input.skillName,
      rating: input.rating ?? 'neutral',
      reason: input.reason ?? '',
      source: input.source ?? 'user',
    }))
  }

  emitAgentSwitch(input: AgentSwitchAuditInput): void {
    this.emit(new AgentSwitchEvent({
      ...eventContext(input),
      fromAgent: input.fromAgent,
      toAgent: input.toAgent,
      reason: input.reason ?? '',
    }))
  }

  /**
   * Notify a registered loop-warning hook when present; otherwise emit the normal event.
   *
   * This mirrors the runtime's extension point: hooks replace, rather than duplicate,
   * the default warning event.
   */
  emitLoopWarning(input: ToolLoopWarningAuditInput): void {
    if (!this.hookRunner?.hasHooks('on_loop_warning')) {
      this.emitLoopWarningEvent(input)
      return
    }
    this.hookRunner.run('on_loop_warning', {
      toolName: input.toolName,
      pattern: input.pattern ?? '',
      severity: input.severity ?? 'warning',
      count: input.count ?? 0,
      ...(input.agentId === undefined ? {} : { agentId: input.agentId }),
      ...(input.turnId === undefined ? {} : { turnId: input.turnId }),
    })
  }

  flush(): void {
    this.collector.flush()
  }

  private emitLoopWarningEvent(input: ToolLoopWarningAuditInput): void {
    this.emit(new ToolLoopWarningEvent({
      ...eventContext(input),
      toolName: input.toolName,
      pattern: input.pattern ?? '',
      severityLevel: input.severity ?? 'warning',
      callCount: input.count ?? 0,
    }))
  }
}

/** Generate a compact hexadecimal turn id compatible with the original daemon's audit records. */
export function generateTurnId(): string {
  return crypto.randomUUID().replaceAll('-', '').slice(0, 12)
}

function eventContext(input: AuditContext): AuditEventOptions {
  return {
    ...(input.agentId === undefined ? {} : { agentId: input.agentId }),
    ...(input.sessionId === undefined ? {} : { sessionId: input.sessionId }),
    ...(input.turnId === undefined ? {} : { turnId: input.turnId }),
  }
}

function preview(value: unknown): string {
  if (value === undefined || value === null) return ''
  if (typeof value === 'string') return redactString(value).slice(0, 200)
  try {
    return JSON.stringify(redactPayload(value)).slice(0, 200)
  } catch {
    return redactString(String(value)).slice(0, 200)
  }
}
