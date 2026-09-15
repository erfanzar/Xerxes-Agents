// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export interface GoalEvidence {
  kind: 'tool' | 'user-decision'
  toolCallId?: string
  decisionId?: string
  summary: string
  recordedAt: number
}

export interface GoalCriterion {
  id: string
  description: string
  evidence?: GoalEvidence
}

export interface InspectedGoal {
  id: string
  revision: number
  objective: string
  currentMilestone?: string
  phase: string
  roundsStarted: number
  maxGoalRounds: number
  maxDurationMs?: number
  maxTotalTokens?: number
  createdAt?: number
  blockedReason?: { code: string; message: string }
  criteria: GoalCriterion[]
  activation?: 'armed' | 'disarmed'
}

export interface GoalContinuation {
  version: 1
  id: string
  sessionId: string
  goalId: string
  revision: number
  state: 'queued' | 'running' | 'settled' | 'interrupted' | 'cancelled'
  queuedAt: number
  startedAt?: number
  settledAt?: number
  ownerId?: string
  round?: number
  reason?: string
}

export interface GoalTokenUsage {
  inputTokens: number
  outputTokens: number
  measuredCalls: number
  settledCalls: number
  pendingCalls: number
  complete: boolean
}

export interface GoalInspection {
  sessionId: string
  goal: InspectedGoal | null
  tokenUsage: GoalTokenUsage | null
  continuation: GoalContinuation | null
}

export function parseGoalInspection(value: unknown, expectedSessionId: string): GoalInspection {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid goal inspection response')
  const row = value as Record<string, unknown>
  if (row.ok !== true || row.session_id !== expectedSessionId) throw new Error(typeof row.error === 'string' ? row.error : 'Invalid goal inspection response')
  const rawGoal = row.goal
  const rawTokenUsage = row.token_usage
  let tokenUsage: GoalTokenUsage | null = null
  if (rawTokenUsage !== undefined && rawTokenUsage !== null) {
    if (!rawTokenUsage || typeof rawTokenUsage !== 'object' || Array.isArray(rawTokenUsage)) {
      throw new Error('Invalid goal inspection token usage')
    }
    const usage = rawTokenUsage as Record<string, unknown>
    const integerFields = ['inputTokens', 'outputTokens', 'measuredCalls', 'settledCalls', 'pendingCalls'] as const
    if (integerFields.some(field => typeof usage[field] !== 'number'
      || !Number.isSafeInteger(usage[field]) || (usage[field] as number) < 0)
      || typeof usage.complete !== 'boolean') {
      throw new Error('Invalid goal inspection token usage')
    }
    const inputTokens = usage.inputTokens as number
    const outputTokens = usage.outputTokens as number
    if (inputTokens > Number.MAX_SAFE_INTEGER - outputTokens) throw new Error('Invalid goal inspection token usage')
    tokenUsage = { inputTokens, outputTokens, measuredCalls: usage.measuredCalls as number,
      settledCalls: usage.settledCalls as number, pendingCalls: usage.pendingCalls as number,
      complete: usage.complete as boolean }
  }
  const rawContinuation = row.continuation
  let continuation: GoalContinuation | null = null
  if (rawContinuation !== undefined && rawContinuation !== null) {
    if (!rawContinuation || typeof rawContinuation !== 'object' || Array.isArray(rawContinuation)) throw new Error('Invalid goal continuation')
    const value = rawContinuation as Record<string, unknown>
    const states = ['queued', 'running', 'settled', 'interrupted', 'cancelled'] as const
    if (value.version !== 1 || typeof value.id !== 'string' || !value.id.trim() || value.sessionId !== expectedSessionId
      || typeof value.goalId !== 'string' || !value.goalId.trim() || typeof value.revision !== 'number' || !Number.isSafeInteger(value.revision) || value.revision < 1
      || !states.includes(value.state as typeof states[number]) || typeof value.queuedAt !== 'number' || !Number.isSafeInteger(value.queuedAt) || value.queuedAt < 0
      || (value.startedAt !== undefined && (typeof value.startedAt !== 'number' || !Number.isSafeInteger(value.startedAt) || value.startedAt < 0))
      || (value.settledAt !== undefined && (typeof value.settledAt !== 'number' || !Number.isSafeInteger(value.settledAt) || value.settledAt < 0))
      || (value.ownerId !== undefined && (typeof value.ownerId !== 'string' || !value.ownerId.trim()))
      || (value.round !== undefined && (typeof value.round !== 'number' || !Number.isSafeInteger(value.round) || value.round < 1))
      || (value.reason !== undefined && (typeof value.reason !== 'string' || !value.reason.trim()))) throw new Error('Invalid goal continuation')
    continuation = { version: 1, id: value.id, sessionId: expectedSessionId, goalId: value.goalId as string, revision: value.revision as number,
      state: value.state as GoalContinuation['state'], queuedAt: value.queuedAt as number,
      ...(value.startedAt === undefined ? {} : { startedAt: value.startedAt as number }), ...(value.settledAt === undefined ? {} : { settledAt: value.settledAt as number }),
      ...(value.ownerId === undefined ? {} : { ownerId: value.ownerId as string }), ...(value.round === undefined ? {} : { round: value.round as number }), ...(value.reason === undefined ? {} : { reason: value.reason as string }) }
  }
  if (rawGoal === null) return { sessionId: expectedSessionId, goal: null, tokenUsage, continuation: null }
  if (!rawGoal || typeof rawGoal !== 'object' || Array.isArray(rawGoal)) throw new Error('Invalid goal inspection goal')
  const goal = rawGoal as Record<string, unknown>
  if (typeof goal.id !== 'string' || !goal.id.trim() || typeof goal.revision !== 'number' || !Number.isSafeInteger(goal.revision) || goal.revision < 1
    || typeof goal.objective !== 'string' || typeof goal.phase !== 'string' || typeof goal.roundsStarted !== 'number' || !Number.isSafeInteger(goal.roundsStarted) || goal.roundsStarted < 0
    || typeof goal.maxGoalRounds !== 'number' || !Number.isSafeInteger(goal.maxGoalRounds) || goal.maxGoalRounds < 1) throw new Error('Invalid goal inspection goal')
  const hasDuration = goal.maxDurationMs !== undefined
  if (hasDuration && (typeof goal.maxDurationMs !== 'number' || !Number.isSafeInteger(goal.maxDurationMs) || goal.maxDurationMs < 1
    || typeof goal.createdAt !== 'number' || !Number.isSafeInteger(goal.createdAt) || goal.createdAt < 0
    || goal.createdAt > Number.MAX_SAFE_INTEGER - goal.maxDurationMs)) throw new Error('Invalid goal inspection time limit')
  if (goal.maxTotalTokens !== undefined
    && (typeof goal.maxTotalTokens !== 'number' || !Number.isSafeInteger(goal.maxTotalTokens) || goal.maxTotalTokens < 1)) {
    throw new Error('Invalid goal inspection token limit')
  }
  if (goal.activation !== undefined && goal.activation !== 'armed' && goal.activation !== 'disarmed') throw new Error('Invalid goal inspection activation')
  if (goal.currentMilestone !== undefined && (typeof goal.currentMilestone !== 'string' || !goal.currentMilestone.trim() || goal.currentMilestone.length > 1_000)) throw new Error('Invalid goal inspection milestone')
  const criteria: GoalCriterion[] = []
  if (goal.criteria !== undefined) {
    if (!Array.isArray(goal.criteria) || goal.criteria.length > 128) throw new Error('Invalid goal inspection criteria')
    for (const item of goal.criteria) {
      if (!item || typeof item !== 'object' || Array.isArray(item)) throw new Error('Invalid goal inspection criterion')
      const criterion = item as Record<string, unknown>
      if (typeof criterion.id !== 'string' || !criterion.id.trim() || typeof criterion.description !== 'string' || !criterion.description.trim()) {
        throw new Error('Invalid goal inspection criterion')
      }
      let evidence: GoalEvidence | undefined
      if (criterion.evidence !== undefined) {
        if (!criterion.evidence || typeof criterion.evidence !== 'object' || Array.isArray(criterion.evidence)) throw new Error('Invalid goal inspection evidence')
        const rawEvidence = criterion.evidence as Record<string, unknown>
        const userDecision = rawEvidence.kind === 'user-decision'
        if (rawEvidence.kind !== undefined && rawEvidence.kind !== 'user-decision' && rawEvidence.kind !== 'tool-result') throw new Error('Invalid goal inspection evidence')
        if (userDecision ? 'toolCallId' in rawEvidence : 'decisionId' in rawEvidence) throw new Error('Invalid goal inspection evidence')
        if (typeof rawEvidence.summary !== 'string' || !rawEvidence.summary.trim()
          || typeof rawEvidence.recordedAt !== 'number' || !Number.isFinite(rawEvidence.recordedAt) || rawEvidence.recordedAt < 0
          || (userDecision ? typeof rawEvidence.decisionId !== 'string' || !rawEvidence.decisionId.trim() : typeof rawEvidence.toolCallId !== 'string' || !rawEvidence.toolCallId.trim())) throw new Error('Invalid goal inspection evidence')
        evidence = userDecision
          ? { kind: 'user-decision', decisionId: rawEvidence.decisionId as string, summary: rawEvidence.summary, recordedAt: rawEvidence.recordedAt }
          : { kind: 'tool', toolCallId: rawEvidence.toolCallId as string, summary: rawEvidence.summary, recordedAt: rawEvidence.recordedAt }
      }
      criteria.push({ id: criterion.id, description: criterion.description, ...(evidence ? { evidence } : {}) })
    }
  }
  let blockedReason: InspectedGoal['blockedReason']
  if (goal.blockedReason !== undefined) {
    if (!goal.blockedReason || typeof goal.blockedReason !== 'object' || Array.isArray(goal.blockedReason)) throw new Error('Invalid goal inspection blocker')
    const reason = goal.blockedReason as Record<string, unknown>
    if (typeof reason.code !== 'string' || !reason.code.trim() || typeof reason.message !== 'string' || !reason.message.trim()) throw new Error('Invalid goal inspection blocker')
    blockedReason = { code: reason.code, message: reason.message }
  }
  return {
    sessionId: expectedSessionId,
    goal: {
      id: goal.id,
      revision: goal.revision,
      objective: goal.objective,
      ...(goal.currentMilestone === undefined ? {} : { currentMilestone: goal.currentMilestone }),
      phase: goal.phase,
      roundsStarted: goal.roundsStarted,
      maxGoalRounds: goal.maxGoalRounds,
      ...(hasDuration ? { maxDurationMs: goal.maxDurationMs as number, createdAt: goal.createdAt as number } : {}),
      ...(goal.maxTotalTokens === undefined ? {} : { maxTotalTokens: goal.maxTotalTokens }),
      ...(blockedReason ? { blockedReason } : {}),
      criteria,
      ...(goal.activation === undefined ? {} : { activation: goal.activation as 'armed' | 'disarmed' }),
    },
    tokenUsage,
    continuation: continuation && continuation.goalId === goal.id ? continuation : null,
  }
}
