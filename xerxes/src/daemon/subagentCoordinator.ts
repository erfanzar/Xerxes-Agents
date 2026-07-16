// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SubAgentManager } from '../agents/subagentManager.js'
import type {
  SpawnedAgentSnapshot,
  SpawnedAgentStatus,
} from '../operators/subagents.js'

const TERMINAL_STATUSES = new Set<SpawnedAgentStatus>([
  'cancelled',
  'closed',
  'completed',
  'error',
  'interrupted',
])

const MAX_RECOVERED_OUTPUT_CHARS = 16_000

const AGENT_TOOL_NAMES = new Set([
  'AgentTool',
  'AwaitAgents',
  'PeekAgent',
  'ResetAgent',
  'SpawnAgents',
  'TaskCreateTool',
  'TaskGetTool',
  'TaskListTool',
  'TaskOutputTool',
  'TaskStopTool',
  'TaskUpdateTool',
])

interface ActiveCohort {
  readonly deadline: number | undefined
  readonly ids: Set<string>
  readonly sourceId: string
  closed: boolean
}

export interface SubagentTurnCohort {
  /** Stop associating subsequently completed work with this parent turn. */
  close(): void
  /** Wait for the background work explicitly detached during this turn. */
  waitForResults(signal?: AbortSignal): Promise<readonly SpawnedAgentSnapshot[]>
}

/**
 * Coordinates explicitly backgrounded subagents with the parent turn that
 * created them. The tool adapter records detached handles; the daemon runner
 * waits for that exact cohort before it lets the parent turn finish.
 */
export interface SubagentTurnCoordinator {
  begin(sourceId: string): SubagentTurnCohort
  consume(snapshots: readonly SpawnedAgentSnapshot[]): void
  /** Rehydrate snapshots retained in a resumed parent transcript. */
  restore?(sourceId: string, snapshots: readonly SpawnedAgentSnapshot[]): number
  track(snapshots: readonly SpawnedAgentSnapshot[]): void
  /** Return the exact detached handles still owned by the active parent turn. */
  trackedIds(sourceId: string): readonly string[]
}

export class NativeSubagentTurnCoordinator implements SubagentTurnCoordinator {
  private readonly active = new Map<string, ActiveCohort>()
  private readonly delivered = new Set<string>()

  constructor(
    private readonly manager: Pick<SubAgentManager, 'waitFor'>,
    private readonly listSnapshots: () => readonly SpawnedAgentSnapshot[],
    private readonly waitTimeoutMs: number | undefined = undefined,
    private readonly now: () => number = () => Date.now(),
    private readonly restoreSnapshots: (snapshots: readonly SpawnedAgentSnapshot[]) => number = () => 0,
  ) {}

  restore(sourceId: string, snapshots: readonly SpawnedAgentSnapshot[]): number {
    const normalized = sourceId.trim()
    if (!normalized || !snapshots.length) return 0
    return this.restoreSnapshots(snapshots.filter(snapshot => snapshot.sourceAgentId === normalized))
  }

  begin(sourceId: string): SubagentTurnCohort {
    const normalized = sourceId.trim()
    if (!normalized) return inertCohort()

    const previous = this.active.get(normalized)
    if (previous) previous.closed = true
    const snapshots = this.listSnapshots()
    const visibleIds = new Set(snapshots.map(snapshot => snapshot.id))
    for (const id of this.delivered) {
      if (!visibleIds.has(id)) this.delivered.delete(id)
    }
    const cohort: ActiveCohort = {
      deadline: this.waitTimeoutMs === undefined
        ? undefined
        : this.now() + Math.max(0, this.waitTimeoutMs),
      // A parent can be interrupted or its daemon can restart while children
      // continue or leave recovered terminal snapshots behind. Reattach every
      // undelivered handle owned by the session so the next turn receives the
      // results automatically instead of requiring TaskList/Peek prompting.
      ids: new Set(snapshots
        .filter(snapshot => (
          snapshot.sourceAgentId?.trim() === normalized
          && !this.delivered.has(snapshot.id)
        ))
        .map(snapshot => snapshot.id)),
      sourceId: normalized,
      closed: false,
    }
    this.active.set(normalized, cohort)
    return {
      close: () => this.close(cohort),
      waitForResults: signal => this.waitForResults(cohort, signal),
    }
  }

  track(snapshots: readonly SpawnedAgentSnapshot[]): void {
    for (const snapshot of snapshots) {
      const sourceId = snapshot.sourceAgentId?.trim()
      if (!sourceId) continue
      const cohort = this.active.get(sourceId)
      if (!cohort || cohort.closed) continue
      if (this.delivered.has(snapshot.id)) continue
      cohort.ids.add(snapshot.id)
    }
  }

  consume(snapshots: readonly SpawnedAgentSnapshot[]): void {
    for (const snapshot of snapshots) {
      if (!TERMINAL_STATUSES.has(snapshot.status)) continue
      this.delivered.add(snapshot.id)
      const sourceId = snapshot.sourceAgentId?.trim()
      if (!sourceId) continue
      this.active.get(sourceId)?.ids.delete(snapshot.id)
    }
  }

  trackedIds(sourceId: string): readonly string[] {
    const cohort = this.active.get(sourceId.trim())
    if (!cohort || cohort.closed) return []
    return Object.freeze([...cohort.ids])
  }

  private close(cohort: ActiveCohort): void {
    cohort.closed = true
    if (this.active.get(cohort.sourceId) === cohort) {
      this.active.delete(cohort.sourceId)
    }
  }

  private async waitForResults(
    cohort: ActiveCohort,
    signal?: AbortSignal,
  ): Promise<readonly SpawnedAgentSnapshot[]> {
    if (cohort.closed || !cohort.ids.size || signal?.aborted) return []

    await this.manager.waitFor(
      () => this.cohortSettled(cohort),
      {
        extraWake: () => cohort.closed || signal?.aborted === true,
        ...(cohort.deadline === undefined
          ? {}
          : { timeoutMs: Math.max(0, cohort.deadline - this.now()) }),
      },
    )
    if (cohort.closed || signal?.aborted) return []

    const byId = new Map(this.listSnapshots().map(snapshot => [snapshot.id, snapshot]))
    const results = [...cohort.ids].flatMap(id => {
      const snapshot = byId.get(id)
      return snapshot === undefined ? [] : [snapshot]
    })
    for (const snapshot of results) {
      if (TERMINAL_STATUSES.has(snapshot.status)) this.delivered.add(snapshot.id)
    }
    cohort.ids.clear()
    // An explicitly bounded wait includes unfinished snapshots so callers
    // opting into a deadline can explain partial progress.
    return Object.freeze(results)
  }

  private cohortSettled(cohort: ActiveCohort): boolean {
    const byId = new Map(this.listSnapshots().map(snapshot => [snapshot.id, snapshot]))
    return [...cohort.ids].every(id => {
      const snapshot = byId.get(id)
      return snapshot === undefined || TERMINAL_STATUSES.has(snapshot.status)
    })
  }
}

/**
 * Recover the latest visible state of subagents from the persisted parent
 * transcript plus its complete session-metadata manifest. Native child turns
 * are process-local, so these archives are the honest state after a restart.
 */
export function recoverSubagentSnapshots(
  messages: readonly unknown[],
  sourceId: string,
  persistedSnapshots: readonly Readonly<Record<string, unknown>>[] = [],
): readonly SpawnedAgentSnapshot[] {
  const normalizedSource = sourceId.trim()
  if (!normalizedSource) return []
  const recovered = new Map<string, SpawnedAgentSnapshot>()
  for (const candidate of messages) {
    if (!isRecord(candidate) || candidate.role !== 'tool') continue
    const name = stringValue(candidate.name)
    if (!AGENT_TOOL_NAMES.has(name)) continue
    const content = decodedToolContent(candidate.content)
    if (content === undefined) continue
    if (name === 'ResetAgent' && isRecord(content)) {
      removeResetTarget(content.reset_target, recovered)
    }
    collectSnapshots(content, normalizedSource, recovered)
  }
  // The session manifest is updated outside the provider transcript and is
  // the authoritative complete view for bounded large-batch receipts.
  collectSnapshots(persistedSnapshots, normalizedSource, recovered)
  return Object.freeze(newestSnapshotsByStableName(recovered.values()).sort((left, right) => (
    left.createdAt.localeCompare(right.createdAt) || left.id.localeCompare(right.id)
  )))
}

/**
 * A stable name may be reused after its earlier handle reaches a terminal
 * state. Resumed transcripts can therefore contain several historical ids for
 * one logical task name. Keep exact-id updates above, then restore only the
 * newest generation so name-based lookups cannot select an older tombstone.
 */
function newestSnapshotsByStableName(
  snapshots: Iterable<SpawnedAgentSnapshot>,
): SpawnedAgentSnapshot[] {
  const newest = new Map<string, SpawnedAgentSnapshot>()
  for (const snapshot of snapshots) {
    const previous = newest.get(snapshot.name)
    if (previous === undefined || snapshotIsNewer(snapshot, previous)) {
      newest.set(snapshot.name, snapshot)
    }
  }
  return [...newest.values()]
}

function snapshotIsNewer(
  candidate: SpawnedAgentSnapshot,
  previous: SpawnedAgentSnapshot,
): boolean {
  const created = candidate.createdAt.localeCompare(previous.createdAt)
  if (created !== 0) return created > 0
  const updated = candidate.updatedAt.localeCompare(previous.updatedAt)
  if (updated !== 0) return updated > 0
  // Transcript traversal is chronological. Prefer the later observation when
  // two generations have indistinguishable persisted timestamps.
  return true
}

function removeResetTarget(
  value: unknown,
  recovered: Map<string, SpawnedAgentSnapshot>,
): void {
  const target = stringValue(value)
  if (!target) return
  recovered.delete(target)
  for (const [id, snapshot] of recovered) {
    if (snapshot.name === target) recovered.delete(id)
  }
}

function collectSnapshots(
  value: unknown,
  sourceId: string,
  recovered: Map<string, SpawnedAgentSnapshot>,
): void {
  if (Array.isArray(value)) {
    for (const item of value) collectSnapshots(item, sourceId, recovered)
    return
  }
  if (!isRecord(value)) return
  const rawId = stringValue(value.id)
  const snapshot = recoveredSnapshot(value, sourceId, rawId ? recovered.get(rawId) : undefined)
  if (snapshot) recovered.set(snapshot.id, snapshot)
  for (const nested of ['agents', 'new_task', 'task']) {
    if (nested in value) collectSnapshots(value[nested], sourceId, recovered)
  }
}

function recoveredSnapshot(
  value: Readonly<Record<string, unknown>>,
  sourceId: string,
  previous?: SpawnedAgentSnapshot,
): SpawnedAgentSnapshot | undefined {
  const id = stringValue(value.id)
  const name = stringValue(value.name)
  const status = spawnedAgentStatus(value.status)
  if (!id || !name || !status) return undefined
  const recordedSource = stringValue(value.source_agent_id)
    || stringValue(value.sourceAgentId)
    || previous?.sourceAgentId
    || sourceId
  if (recordedSource !== sourceId) return undefined
  const lastOutput = boundedText(value.last_output ?? value.lastOutput, MAX_RECOVERED_OUTPUT_CHARS)
    ?? previous?.lastOutput
  const completionSummary = boundedText(
    value.summary ?? value.completion_summary ?? value.completionSummary,
    500,
  ) || previous?.completionSummary || (status === 'completed' ? lastOutput?.slice(0, 500) : undefined)
  const error = boundedText(value.error, 2_000) ?? previous?.error
  const lastInput = boundedText(value.last_input ?? value.lastInput, MAX_RECOVERED_OUTPUT_CHARS)
    ?? previous?.lastInput
  const creatorAgentId = stringValue(value.creator_id)
    || stringValue(value.creatorAgentId)
    || previous?.creatorAgentId
  const parentAgentId = stringValue(value.parent_id)
    || stringValue(value.parentAgentId)
    || previous?.parentAgentId
  const model = stringValue(value.model) || previous?.model
  const rules = stringArray(value.rules)
  const toolsets = stringArray(value.toolsets)
  const apiCalls = nonNegativeInteger(value.api_calls ?? value.apiCalls) ?? previous?.apiCalls
  const toolCalls = nonNegativeInteger(value.tool_count ?? value.toolCalls) ?? previous?.toolCalls
  const inputTokens = nonNegativeInteger(value.input_tokens ?? value.inputTokens) ?? previous?.inputTokens
  const outputTokens = nonNegativeInteger(value.output_tokens ?? value.outputTokens) ?? previous?.outputTokens
  const reasoningTokens = nonNegativeInteger(value.reasoning_tokens ?? value.reasoningTokens) ?? previous?.reasoningTokens
  const filesRead = stringArray(value.files_read ?? value.filesRead)
  const filesWritten = stringArray(value.files_written ?? value.filesWritten)
  return Object.freeze({
    agentId: stringValue(value.agent_id) || stringValue(value.agentId) || previous?.agentId || 'coder',
    closed: value.closed === true || status === 'closed',
    createdAt: timestampValue(value.created_at ?? value.createdAt, previous?.createdAt),
    ...(error ? { error } : {}),
    id,
    ...(lastInput ? { lastInput } : {}),
    ...(lastOutput ? { lastOutput } : {}),
    name,
    title: stringValue(value.title) || previous?.title || name,
    ...(creatorAgentId ? { creatorAgentId } : {}),
    ...(parentAgentId ? { parentAgentId } : {}),
    ...(model ? { model } : {}),
    ...(rules.length ? { rules } : previous?.rules === undefined ? {} : { rules: previous.rules }),
    ...(toolsets.length ? { toolsets } : previous?.toolsets === undefined ? {} : { toolsets: previous.toolsets }),
    ...(apiCalls === undefined ? {} : { apiCalls }),
    ...(toolCalls === undefined ? {} : { toolCalls }),
    ...(inputTokens === undefined ? {} : { inputTokens }),
    ...(outputTokens === undefined ? {} : { outputTokens }),
    ...(reasoningTokens === undefined ? {} : { reasoningTokens }),
    ...(filesRead.length ? { filesRead } : previous?.filesRead === undefined ? {} : { filesRead: previous.filesRead }),
    ...(filesWritten.length ? { filesWritten } : previous?.filesWritten === undefined ? {} : { filesWritten: previous.filesWritten }),
    ...(completionSummary ? { completionSummary } : {}),
    promptProfile: stringValue(value.prompt_profile)
      || stringValue(value.promptProfile)
      || previous?.promptProfile
      || 'coder',
    queueSize: nonNegativeInteger(value.queue_size ?? value.queueSize) ?? previous?.queueSize ?? 0,
    sourceAgentId: recordedSource,
    status,
    updatedAt: timestampValue(value.updated_at ?? value.updatedAt, previous?.updatedAt),
  })
}

function decodedToolContent(value: unknown): unknown {
  if (typeof value !== 'string') return value
  const trimmed = value.trim()
  if (!trimmed || (!trimmed.startsWith('{') && !trimmed.startsWith('['))) return undefined
  try {
    return JSON.parse(trimmed) as unknown
  } catch {
    return undefined
  }
}

function spawnedAgentStatus(value: unknown): SpawnedAgentStatus | undefined {
  switch (value) {
    case 'cancelled':
    case 'closed':
    case 'completed':
    case 'error':
    case 'idle':
    case 'interrupted':
    case 'running':
      return value
    default:
      return undefined
  }
}

function boundedText(value: unknown, limit: number): string | undefined {
  if (typeof value !== 'string') return undefined
  const normalized = value.trim()
  if (!normalized) return undefined
  return normalized.length <= limit ? normalized : `${normalized.slice(0, limit - 1)}…`
}

function stringArray(value: unknown): readonly string[] {
  return Array.isArray(value)
    ? Object.freeze(value.filter((item): item is string => typeof item === 'string'))
    : Object.freeze([])
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function nonNegativeInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined
}

function timestampValue(value: unknown, fallback = new Date(0).toISOString()): string {
  const candidate = stringValue(value)
  return candidate && Number.isFinite(Date.parse(candidate)) ? candidate : fallback
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function inertCohort(): SubagentTurnCohort {
  return {
    close: () => undefined,
    waitForResults: async () => [],
  }
}
