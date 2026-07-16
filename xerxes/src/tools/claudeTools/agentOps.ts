// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../../core/errors.js'
import { replacePersistedSubagentSnapshots } from '../../agents/subagentPersistence.js'
import { ToolRegistry, type ToolExecutionContext } from '../../executors/toolRegistry.js'
import {
  MAX_AGENT_TITLE_LENGTH,
  normalizeAgentTitle,
  type SpawnAgentOptions,
  type SpawnedAgentDescriptor,
  type SpawnedAgentManagerPort,
  type SpawnedAgentSnapshot,
} from '../../operators/subagents.js'
import type { JsonObject, JsonValue, ToolDefinition } from '../../types/toolCalls.js'
import type { PermissionMode } from '../../streaming/permissions.js'
import { optionalBoolean, optionalString, requiredString } from '../inputs.js'

const DEFAULT_WAIT_SECONDS = 120
const MAILBOX_EVENT_LIMIT = 1_000
const MAX_SPAWN_AGENT_REQUESTS = 1_000
const MAX_CONCURRENT_SPAWN_REQUESTS = 8
const MAX_INLINE_BATCH_RECEIPT_AGENTS = 8
const DEFAULT_TASK_LIST_PAGE_SIZE = 50
const MAX_TASK_LIST_PAGE_SIZE = 50
const POLL_INTERVAL_MS = 25

const TERMINAL_STATUSES = new Set(['cancelled', 'closed', 'completed', 'error', 'interrupted'])

export interface ClaudeAgentSpec {
  readonly model?: string
  readonly name?: string
  readonly prompt: string
  readonly subagentType?: string
  readonly title: string
}

export interface ClaudeAgentEvent {
  readonly agentId: string
  readonly apiCalls?: number
  readonly completionSummary?: string
  readonly creatorAgentId?: string
  readonly event: 'agent_output' | 'agent_spawned' | 'agent_status'
  readonly filesRead?: readonly string[]
  readonly filesWritten?: readonly string[]
  readonly inputTokens?: number
  readonly model?: string
  readonly name: string
  readonly output?: string
  readonly outputTokens?: number
  readonly parentAgentId?: string
  readonly previousStatus?: string
  readonly reasoningTokens?: number
  readonly rules?: readonly string[]
  readonly seq: number
  readonly sourceAgentId?: string
  readonly status: string
  readonly title: string
  readonly toolCalls?: number
  readonly toolsets?: readonly string[]
}

/**
 * Captures observable sub-agent lifecycle changes for Claude-compatible
 * `CheckAgentMessages` calls. Hosts may also report their own events through
 * `record` when their runner has richer streamed progress than a snapshot.
 */
export class AgentEventMailbox {
  private readonly cursors = new Map<string, number>()
  private readonly events: ClaudeAgentEvent[] = []
  private floorCursor = 0
  private readonly observed = new Map<string, SpawnedAgentSnapshot>()
  private sequence = 0

  capture(snapshots: readonly SpawnedAgentSnapshot[]): void {
    for (const snapshot of snapshots) {
      const previous = this.observed.get(snapshot.id)
      if (previous === undefined) {
        this.record({
          agentId: snapshot.id,
          event: 'agent_spawned',
          name: snapshot.name,
          status: snapshot.status,
          ...snapshotEventMetadata(snapshot),
        })
      } else if (previous.status !== snapshot.status) {
        this.record({
          agentId: snapshot.id,
          event: 'agent_status',
          name: snapshot.name,
          previousStatus: previous.status,
          status: snapshot.status,
          ...snapshotEventMetadata(snapshot),
        })
      }
      if (snapshot.lastOutput !== undefined && snapshot.lastOutput !== previous?.lastOutput) {
        this.record({
          agentId: snapshot.id,
          event: 'agent_output',
          name: snapshot.name,
          output: snapshot.lastOutput,
          status: snapshot.status,
          ...snapshotEventMetadata(snapshot),
        })
      }
      this.observed.set(snapshot.id, snapshot)
    }
  }

  record(event: Omit<ClaudeAgentEvent, 'seq'>): ClaudeAgentEvent {
    const recorded = Object.freeze({ ...event, seq: ++this.sequence })
    this.events.push(recorded)
    if (this.events.length > MAILBOX_EVENT_LIMIT) {
      this.events.splice(0, this.events.length - MAILBOX_EVENT_LIMIT)
      this.floorCursor = Math.max(this.floorCursor, (this.events[0]?.seq ?? this.sequence) - 1)
    }
    return recorded
  }

  drain(sinceSeq?: number): readonly ClaudeAgentEvent[]
  drain(ownerId: string, sinceSeq?: number): readonly ClaudeAgentEvent[]
  drain(ownerOrSince: string | number = 0, scopedSince = 0): readonly ClaudeAgentEvent[] {
    const scope = mailboxScope(ownerOrSince)
    const sinceSeq = typeof ownerOrSince === 'number' ? ownerOrSince : scopedSince
    const after = Math.max(this.floorCursor, this.cursors.get(scope) ?? 0, normalizeSequence(sinceSeq))
    const events = this.events.filter(event => event.seq > after && eventMatchesScope(event, scope))
    this.cursors.set(scope, this.sequence)
    return Object.freeze(events)
  }

  peek(sinceSeq?: number): readonly ClaudeAgentEvent[]
  peek(ownerId: string, sinceSeq?: number): readonly ClaudeAgentEvent[]
  peek(ownerOrSince: string | number = 0, scopedSince = 0): readonly ClaudeAgentEvent[] {
    const scope = mailboxScope(ownerOrSince)
    const sinceSeq = typeof ownerOrSince === 'number' ? ownerOrSince : scopedSince
    const after = normalizeSequence(sinceSeq)
    return Object.freeze(this.events.filter(event => event.seq > after && eventMatchesScope(event, scope)))
  }

  latestSeq(ownerId?: string): number {
    if (ownerId === undefined) return this.sequence
    const scope = normalizeOwnerId(ownerId)
    return this.events.reduce(
      (latest, event) => eventMatchesScope(event, scope) ? Math.max(latest, event.seq) : latest,
      0,
    )
  }
}

export interface ClaudeAgentToolsOptions {
  /** Resolves an agent type to the runtime descriptor consumed by the manager. */
  readonly agentResolver?: (subagentType: string, model?: string) => SpawnedAgentDescriptor | undefined
  readonly mailbox?: AgentEventMailbox
  readonly manager: SpawnedAgentManagerPort
  readonly now?: () => number
  /** Associates explicitly detached work with the active parent turn. */
  readonly backgroundAgents?: {
    consume(snapshots: readonly SpawnedAgentSnapshot[]): void
    track(snapshots: readonly SpawnedAgentSnapshot[]): void
    trackedIds?(sourceAgentId: string): readonly string[]
  }
}

export const CLAUDE_AGENT_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  definition('AgentTool', 'Spawn one focused subagent, optionally waiting for its final result.', {
    prompt: stringSchema('The delegated task.'),
    title: titleSchema('Short human-readable title describing this delegated task.'),
    subagent_type: stringSchema('Agent definition to run.'),
    isolation: stringSchema('Requested isolation mode. Native worktree isolation is not yet available here.'),
    name: stringSchema('Stable subagent name.'),
    model: stringSchema('Optional model override.'),
    run_in_background: booleanSchema('Return immediately while the subagent keeps working.', false),
    wait: booleanSchema('Wait for the subagent to finish.', true),
    timeout: numberSchema('Maximum seconds to wait.'),
  }, ['prompt', 'title']),
  definition('SendMessageTool', 'Queue a follow-up message for a managed subagent.', {
    target: stringSchema('Subagent id or stable name.'),
    message: stringSchema('Message for the subagent.'),
  }, ['target', 'message']),
  definition('TaskCreateTool', 'Create a background subagent task without waiting.', {
    prompt: stringSchema('The delegated task.'),
    title: titleSchema('Short human-readable title describing this background task.'),
    name: stringSchema('Stable subagent name.'),
    subagent_type: stringSchema('Agent definition to run.'),
  }, ['prompt', 'title']),
  definition('SpawnAgents', 'Spawn up to 1,000 subagents as one queued batch and optionally wait for all of them.', {
    agents: {
      description: 'JSON array of {title, prompt, name?, subagent_type?, model?}. Every agent needs a short title. Work beyond the runtime concurrency is queued.',
      type: 'array',
      minItems: 1,
      maxItems: MAX_SPAWN_AGENT_REQUESTS,
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['title', 'prompt'],
        properties: {
          title: titleSchema('Short human-readable title for this delegated task.'),
          prompt: stringSchema('The delegated task.'),
          name: stringSchema('Optional stable subagent handle.'),
          subagent_type: stringSchema('Agent definition to run.'),
          model: stringSchema('Optional model override.'),
        },
      },
    },
    wait: booleanSchema('Default true. Wait for all spawned agents. If false, the parent runtime still joins required results before the turn ends.', true),
    timeout: numberSchema('Maximum seconds to wait for the batch.'),
  }, ['agents']),
  definition('TaskGetTool', 'Return one currently attached subagent status. Use only an exact id or name returned by the current TaskListTool result.', {
    task_id: stringSchema('Exact subagent id or stable name from the current TaskListTool result; do not retry stale targets.'),
  }, ['task_id']),
  definition('TaskListTool', 'List a compact page of subagent tasks attached to the current runtime. Use offset pagination while a full page is returned. If the first page returns [], stale names and ids are not queryable; do not retry them.', {
    offset: { type: 'integer', minimum: 0, default: 0, description: 'Zero-based task offset.' },
    limit: { type: 'integer', minimum: 1, maximum: MAX_TASK_LIST_PAGE_SIZE, default: DEFAULT_TASK_LIST_PAGE_SIZE, description: 'Maximum compact task rows to return.' },
  }),
  definition('TaskOutputTool', 'Return the latest output of a managed subagent.', {
    task_id: stringSchema('Subagent id or stable name.'),
  }, ['task_id']),
  definition('TaskStopTool', 'Cancel and close a managed subagent task.', {
    task_id: stringSchema('Subagent id or stable name.'),
  }, ['task_id']),
  definition('TaskUpdateTool', 'Queue an update for a managed subagent task.', {
    task_id: stringSchema('Subagent id or stable name.'),
    message: stringSchema('Update message.'),
  }, ['task_id', 'message']),
  definition('AwaitAgents', 'Wait for any or all currently attached subagents. Omit agent_ids for the tracked cohort; the default waits for all. Explicit targets must come from the current TaskListTool result.', {
    agent_ids: {
      description: 'Optional array of exact subagent ids or names returned by the current TaskListTool result. Do not retry stale targets.',
      type: 'array',
      items: { type: 'string' },
    },
    wake_on: { type: 'string', enum: ['any', 'all', 'none'], default: 'all' },
    timeout_seconds: numberSchema('Maximum seconds to wait.'),
  }),
  definition('CheckAgentMessages', 'Drain or inspect lifecycle events from managed subagents.', {
    since_seq: { type: 'integer', minimum: 0, default: 0 },
    peek: booleanSchema('Do not advance the mailbox cursor.'),
  }),
  definition('PeekAgent', 'Return one current managed-subagent snapshot. Use an exact id or name from the current TaskListTool result and do not retry stale targets.', {
    target: stringSchema('Exact subagent id or stable name from the current TaskListTool result; do not retry stale targets.'),
  }, ['target']),
  definition('ResetAgent', 'Cancel then restart a managed subagent with a replacement task.', {
    target: stringSchema('Subagent id or stable name.'),
    new_prompt: stringSchema('Replacement task; defaults to its last input.'),
  }, ['target']),
  definition('HandoffTool', 'Hand off work to a specialized subagent.', {
    target_agent: stringSchema('Agent type receiving the handoff.'),
    reason: stringSchema('Why the handoff is needed.'),
    context_summary: stringSchema('Compact context handed to the target.'),
    prompt: stringSchema('Specific task for the target.'),
    timeout: numberSchema('Maximum seconds to wait.'),
  }, ['target_agent', 'reason']),
]

/** Register the non-duplicative Claude-Code-compatible subagent surface. */
export function registerClaudeAgentTools(
  registry: ToolRegistry,
  options: ClaudeAgentToolsOptions,
  agentId = 'default',
): readonly ToolDefinition[] {
  const adapter = new ClaudeAgentTools(options)
  for (const tool of CLAUDE_AGENT_TOOL_DEFINITIONS) {
    registry.replace(tool, (inputs, context, signal) => adapter.execute(tool.function.name, inputs, context, signal), agentId)
  }
  return CLAUDE_AGENT_TOOL_DEFINITIONS
}

/** Adapter that maps Claude-style task calls onto the Bun `SpawnedAgentManager`. */
export class ClaudeAgentTools {
  private readonly mailbox: AgentEventMailbox
  private readonly now: () => number

  constructor(private readonly options: ClaudeAgentToolsOptions) {
    this.mailbox = options.mailbox ?? new AgentEventMailbox()
    this.now = options.now ?? (() => Date.now())
  }

  async execute(
    name: string,
    inputs: JsonObject,
    context: ToolExecutionContext,
    signal?: AbortSignal,
  ): Promise<unknown> {
    try {
      switch (name) {
        case 'AgentTool': return await this.agentTool(inputs, context, signal)
        case 'SendMessageTool': return await this.sendMessage(inputs, context)
        case 'TaskCreateTool': return await this.taskCreate(inputs, context)
        case 'SpawnAgents': return await this.spawnAgents(inputs, context, signal)
        case 'TaskGetTool': return this.taskGet(requiredString(inputs, 'task_id'), context, 'task_id')
        case 'TaskListTool': return this.taskList(inputs, context)
        case 'TaskOutputTool': return this.taskOutput(requiredString(inputs, 'task_id'), context, 'task_id')
        case 'TaskStopTool': return this.taskStop(requiredString(inputs, 'task_id'), context, 'task_id')
        case 'TaskUpdateTool': return await this.taskUpdate(inputs, context)
        case 'AwaitAgents': return await this.awaitAgents(inputs, context, signal)
        case 'CheckAgentMessages': return this.checkMessages(inputs, context)
        case 'PeekAgent': return this.taskGet(requiredString(inputs, 'target'), context)
        case 'ResetAgent': return await this.resetAgent(inputs, context)
        case 'HandoffTool': return await this.handoff(inputs, context, signal)
        default: throw new ValidationError('tool', 'is not handled by ClaudeAgentTools', name)
      }
    } finally {
      this.persistContext(context)
    }
  }

  private async agentTool(
    inputs: JsonObject,
    context: ToolExecutionContext,
    signal?: AbortSignal,
  ): Promise<Record<string, unknown>> {
    const isolation = optionalString(inputs, 'isolation')?.trim()
    if (isolation) {
      throw new ValidationError(
        'isolation',
        'is not supported by the Bun spawned-agent manager; use a host-provided isolated runner',
        isolation,
      )
    }
    const name = optionalString(inputs, 'name')?.trim()
    const subagentType = optionalString(inputs, 'subagent_type')?.trim()
    const model = optionalString(inputs, 'model')?.trim()
    const spec: ClaudeAgentSpec = {
      prompt: requiredString(inputs, 'prompt'),
      title: normalizeAgentTitle(requiredString(inputs, 'title')),
      ...(name ? { name } : {}),
      ...(subagentType ? { subagentType } : {}),
      ...(model ? { model } : {}),
    }
    const snapshot = await this.spawnSpec(spec, context)
    const background = optionalBoolean(inputs, 'run_in_background', false)
    if (background || !optionalBoolean(inputs, 'wait', true)) {
      this.observeBackgroundState([snapshot])
      return agentSnapshotWire(snapshot)
    }
    try {
      const settled = await this.waitFor([snapshot.id], timeoutMilliseconds(inputs, 'timeout', DEFAULT_WAIT_SECONDS), signal)
      const final = settled[0] ?? snapshot
      this.observeBackgroundState([final])
      return agentSnapshotWire(final)
    } catch (error) {
      if (signal?.aborted) this.closeAfterAbort([snapshot.id], error)
      throw error
    }
  }

  private async sendMessage(inputs: JsonObject, context: ToolExecutionContext): Promise<Record<string, unknown>> {
    const snapshot = await this.options.manager.sendInput(this.resolveId(requiredString(inputs, 'target'), context), {
      message: requiredString(inputs, 'message'),
    })
    this.capture()
    return agentSnapshotWire(snapshot)
  }

  private async taskCreate(inputs: JsonObject, context: ToolExecutionContext): Promise<Record<string, unknown>> {
    const name = optionalString(inputs, 'name')?.trim()
    const subagentType = optionalString(inputs, 'subagent_type')?.trim()
    const spec: ClaudeAgentSpec = {
      prompt: requiredString(inputs, 'prompt'),
      title: normalizeAgentTitle(requiredString(inputs, 'title')),
      ...(name ? { name } : {}),
      ...(subagentType ? { subagentType } : {}),
    }
    const snapshot = await this.spawnSpec(spec, context)
    this.observeBackgroundState([snapshot])
    return agentSnapshotWire(snapshot)
  }

  private async spawnAgents(
    inputs: JsonObject,
    context: ToolExecutionContext,
    signal?: AbortSignal,
  ): Promise<readonly Record<string, unknown>[] | Record<string, unknown>> {
    const specs = parseAgentSpecs(inputs.agents)
    if (!specs.length) throw new ValidationError('agents', 'must contain at least one agent specification', inputs.agents)
    if (specs.length > MAX_SPAWN_AGENT_REQUESTS) {
      throw new ValidationError('agents', `must contain at most ${MAX_SPAWN_AGENT_REQUESTS} agents`, specs.length)
    }
    const registration = await settleWithConcurrency(
      specs,
      MAX_CONCURRENT_SPAWN_REQUESTS,
      spec => this.spawnSpec(spec, context),
      signal,
    )
    const results = registration.results
    const failures = results.filter((result): result is PromiseRejectedResult => result.status === 'rejected')
    const snapshots = results
      .filter((result): result is PromiseFulfilledResult<SpawnedAgentSnapshot> => result.status === 'fulfilled')
      .map(result => result.value)
    const registrationFailure = (registration.stopped
      ? registration.reason
      : failures[0]?.reason) ?? new Error('Subagent spawn registration failed')
    if (registration.stopped || failures.length) {
      const cleanupErrors: unknown[] = []
      for (const snapshot of snapshots) {
        try {
          this.options.manager.close(snapshot.id)
        } catch (error) {
          cleanupErrors.push(error)
        }
      }
      this.capture()
      if (cleanupErrors.length) {
        throw new AggregateError(
          [registrationFailure, ...cleanupErrors],
          'Failed to spawn the complete subagent batch and clean up partial work',
        )
      }
      throw registrationFailure
    }
    // Persist the complete manifest before a foreground wait so a daemon crash
    // cannot reduce a large bounded receipt to only its inline preview rows.
    this.persistContext(context)
    if (!optionalBoolean(inputs, 'wait', true)) {
      this.observeSpawnBatchState(snapshots)
      return spawnBatchWire(snapshots)
    }
    const ids = snapshots.map(snapshot => snapshot.id)
    try {
      const settled = await this.waitFor(ids, timeoutMilliseconds(inputs, 'timeout', DEFAULT_WAIT_SECONDS), signal)
      this.observeSpawnBatchState(settled)
      return spawnBatchWire(settled)
    } catch (error) {
      if (signal?.aborted) this.closeAfterAbort(ids, error)
      throw error
    }
  }

  private taskGet(
    target: string,
    context: ToolExecutionContext,
    inputField = 'target',
  ): Record<string, unknown> {
    this.capture()
    const snapshot = this.requireSnapshot(target, context, inputField)
    if (isTerminal(snapshot)) this.options.backgroundAgents?.consume([snapshot])
    return agentSnapshotWire(snapshot)
  }

  private taskList(inputs: JsonObject, context: ToolExecutionContext): readonly Record<string, unknown>[] {
    this.capture()
    const snapshots = this.ownedHandles(context)
    const offset = nonnegativeInteger(inputs, 'offset', 0)
    const limit = nonnegativeInteger(inputs, 'limit', DEFAULT_TASK_LIST_PAGE_SIZE)
    if (limit < 1 || limit > MAX_TASK_LIST_PAGE_SIZE) {
      throw new ValidationError('limit', `must be between 1 and ${MAX_TASK_LIST_PAGE_SIZE}`, limit)
    }
    const page = snapshots.slice(offset, offset + limit)
    return page.map(compactAgentSnapshotWire)
  }

  private taskOutput(target: string, context: ToolExecutionContext, inputField = 'target'): string {
    const snapshot = this.requireSnapshot(target, context, inputField)
    if (snapshot.lastOutput !== undefined) {
      this.options.backgroundAgents?.consume([snapshot])
      return snapshot.lastOutput
    }
    return `No output for task '${target}' (may still be running).`
  }

  private taskStop(
    target: string,
    context: ToolExecutionContext,
    inputField = 'target',
  ): Record<string, unknown> {
    const snapshot = this.options.manager.close(this.resolveId(target, context, inputField))
    this.capture()
    this.options.backgroundAgents?.consume([snapshot])
    return agentSnapshotWire(snapshot)
  }

  private async taskUpdate(inputs: JsonObject, context: ToolExecutionContext): Promise<Record<string, unknown>> {
    const snapshot = await this.options.manager.sendInput(this.resolveId(requiredString(inputs, 'task_id'), context, 'task_id'), {
      message: requiredString(inputs, 'message'),
    })
    this.capture()
    return agentSnapshotWire(snapshot)
  }

  private async awaitAgents(
    inputs: JsonObject,
    context: ToolExecutionContext,
    signal?: AbortSignal,
  ): Promise<Record<string, unknown>> {
    const requested = parseAgentIds(inputs.agent_ids)
    const owned = this.ownedHandles(context)
    const ownedById = new Map(owned.map(snapshot => [snapshot.id, snapshot]))
    const tracked = this.options.backgroundAgents
      ?.trackedIds?.(contextOwnerId(context))
      .filter(id => ownedById.has(id)) ?? []
    const targets = requested.length
      ? requested.map(target => this.resolveId(target, context, 'agent_ids'))
      : tracked.length
        ? tracked
        : owned.filter(snapshot => !isTerminal(snapshot)).map(snapshot => snapshot.id)
    const wakeOn = normalizeWakeOn(optionalString(inputs, 'wake_on'))
    const timeout = timeoutMilliseconds(inputs, 'timeout_seconds', 30)
    const started = this.now()
    let wakeReason: 'agents_done' | 'cancelled' | 'timeout' = 'timeout'
    if (!targets.length) {
      return this.observedAwaitResult(
        wakeOn === 'none' ? 'timeout' : 'agents_done',
        wakeOn,
        started,
        [],
        tracked,
      )
    }
    while (true) {
      this.capture()
      const snapshots = targets.map(target => this.requireSnapshot(target, context, 'agent_ids'))
      if (signal?.aborted) {
        wakeReason = 'cancelled'
        return this.observedAwaitResult(wakeReason, wakeOn, started, snapshots, tracked)
      }
      if (agentsSatisfied(snapshots, wakeOn)) {
        wakeReason = 'agents_done'
        return this.observedAwaitResult(wakeReason, wakeOn, started, snapshots, tracked)
      }
      const elapsed = this.now() - started
      if (elapsed >= timeout) return this.observedAwaitResult(wakeReason, wakeOn, started, snapshots, tracked)
      await sleep(Math.min(POLL_INTERVAL_MS, timeout - elapsed))
    }
  }

  private checkMessages(inputs: JsonObject, context: ToolExecutionContext): Record<string, unknown> {
    this.capture()
    const sinceSeq = nonnegativeInteger(inputs, 'since_seq', 0)
    const peek = optionalBoolean(inputs, 'peek', false)
    const owner = contextOwnerId(context)
    const events = peek ? this.mailbox.peek(owner, sinceSeq) : this.mailbox.drain(owner, sinceSeq)
    return { latest_seq: this.mailbox.latestSeq(owner), events }
  }

  private async resetAgent(inputs: JsonObject, context: ToolExecutionContext): Promise<Record<string, unknown>> {
    const target = requiredString(inputs, 'target')
    const current = this.requireSnapshot(target, context)
    const replacement = optionalString(inputs, 'new_prompt')?.trim() || current.lastInput
    if (!replacement) throw new ValidationError('new_prompt', 'is required when the agent has no prior input')
    const closed = this.options.manager.close(current.id)
    this.options.backgroundAgents?.consume([closed])
    this.options.manager.resume(current.id)
    const snapshot = await this.options.manager.sendInput(current.id, { message: replacement, interrupt: true })
    this.capture()
    this.observeBackgroundState([snapshot])
    return { reset_target: target, new_task: agentSnapshotWire(snapshot) }
  }

  private observedAwaitResult(
    wakeReason: 'agents_done' | 'cancelled' | 'timeout',
    wakeOn: 'all' | 'any' | 'none',
    started: number,
    snapshots: readonly SpawnedAgentSnapshot[],
    trackedIds: readonly string[],
  ): Record<string, unknown> {
    // Large await receipts contain only compact status rows, so they have not
    // delivered terminal outputs and must not consume those results from the
    // exact parent-turn cohort. Already-tracked work remains available for the
    // coordinator's single bounded result continuation.
    if (snapshots.length <= MAX_INLINE_BATCH_RECEIPT_AGENTS) {
      this.observeBackgroundState(snapshots)
    } else {
      const tracked = new Set(trackedIds)
      const retained = snapshots.filter(snapshot => tracked.has(snapshot.id))
      if (retained.length) this.options.backgroundAgents?.track(retained)
    }
    return awaitResult(wakeReason, wakeOn, started, this.now(), snapshots)
  }

  private observeSpawnBatchState(snapshots: readonly SpawnedAgentSnapshot[]): void {
    if (snapshots.length > MAX_INLINE_BATCH_RECEIPT_AGENTS) {
      // The compact batch receipt deliberately omits every output body. Track
      // the complete exact batch, including agents that finished during the
      // foreground wait, so the turn coordinator delivers their bounded
      // results once before the parent can finish.
      this.options.backgroundAgents?.track(snapshots)
      return
    }
    this.observeBackgroundState(snapshots)
  }

  private observeBackgroundState(snapshots: readonly SpawnedAgentSnapshot[]): void {
    const terminal = snapshots.filter(isTerminal)
    const pending = snapshots.filter(snapshot => !isTerminal(snapshot))
    if (terminal.length) this.options.backgroundAgents?.consume(terminal)
    if (pending.length) this.options.backgroundAgents?.track(pending)
  }

  private async handoff(
    inputs: JsonObject,
    context: ToolExecutionContext,
    signal?: AbortSignal,
  ): Promise<unknown> {
    const targetAgent = requiredString(inputs, 'target_agent')
    const reason = requiredString(inputs, 'reason')
    const contextSummary = optionalString(inputs, 'context_summary')?.trim() || '(no context provided)'
    const prompt = optionalString(inputs, 'prompt')?.trim() || '(continue from where the previous agent left off)'
    const snapshot = await this.spawnSpec({
      subagentType: targetAgent,
      name: `handoff-${targetAgent}-${crypto.randomUUID().replaceAll('-', '').slice(0, 6)}`,
      title: normalizeAgentTitle(`Handoff to ${targetAgent}`),
      prompt: `## Handoff from parent agent\n\nReason: ${reason}\n\nContext: ${contextSummary}\n\nYour task: ${prompt}`,
    }, context)
    try {
      const settled = await this.waitFor([snapshot.id], timeoutMilliseconds(inputs, 'timeout', DEFAULT_WAIT_SECONDS), signal)
      const final = settled[0] ?? snapshot
      this.observeBackgroundState([final])
      return final.lastOutput ?? agentSnapshotWire(final)
    } catch (error) {
      if (signal?.aborted) this.closeAfterAbort([snapshot.id], error)
      throw error
    }
  }

  private async spawnSpec(spec: ClaudeAgentSpec, context: ToolExecutionContext): Promise<SpawnedAgentSnapshot> {
    const type = spec.subagentType?.trim() || 'general-purpose'
    const parentPermissionMode = contextPermissionMode(context)
    const resolved = this.options.agentResolver?.(type, spec.model)
    const agent: SpawnedAgentDescriptor = Object.freeze({
      id: resolved?.id ?? type,
      ...(spec.model?.trim() ? { model: spec.model.trim() } : resolved?.model === undefined ? {} : { model: resolved.model }),
      systemPrompt: resolved?.systemPrompt ?? `You are a focused ${type} subagent. Complete the delegated task and return a self-contained result.`,
      ...(resolved?.name === undefined ? { name: type } : { name: resolved.name }),
    })
    const request: SpawnAgentOptions = {
      agent,
      message: spec.prompt,
      promptProfile: type,
      title: spec.title,
      ...(typeof context.metadata.model === 'string' && context.metadata.model.trim()
        ? { parentModel: context.metadata.model.trim() }
        : {}),
      ...(parentPermissionMode === undefined ? {} : { permissionMode: parentPermissionMode }),
      ...(context.sessionId ? { sourceAgentId: context.sessionId } : context.agentId ? { sourceAgentId: context.agentId } : {}),
      ...(context.agentId ? { creatorAgentId: context.agentId, parentAgentId: context.agentId } : {}),
      ...(spec.name?.trim() ? { nickname: spec.name.trim() } : {}),
    }
    const snapshot = await this.options.manager.spawn(request)
    this.capture()
    return snapshot
  }

  private async waitFor(ids: readonly string[], timeout: number, signal?: AbortSignal): Promise<SpawnedAgentSnapshot[]> {
    if (signal?.aborted) throw signal.reason ?? new Error('Subagent wait cancelled')
    const result = await abortable(this.options.manager.wait(ids, timeout), signal)
    this.capture()
    const snapshots = new Map(
      [...result.completed, ...result.pending].map(snapshot => [snapshot.id, snapshot]),
    )
    return ids
      .map(id => snapshots.get(id))
      .filter((snapshot): snapshot is SpawnedAgentSnapshot => snapshot !== undefined)
  }

  private closeAfterAbort(ids: readonly string[], cause: unknown): void {
    const errors: unknown[] = []
    for (const id of ids) {
      try {
        this.options.manager.close(id)
      } catch (error) {
        errors.push(error)
      }
    }
    this.capture()
    if (errors.length) {
      throw new AggregateError([cause, ...errors], 'Subagent wait was cancelled but child cleanup failed')
    }
  }

  private requireSnapshot(
    target: string,
    context: ToolExecutionContext,
    inputField = 'target',
  ): SpawnedAgentSnapshot {
    const id = this.resolveId(target, context, inputField)
    const snapshot = this.ownedHandles(context).find(candidate => candidate.id === id)
    if (snapshot === undefined) this.throwMissingTarget(target, context, inputField)
    return snapshot
  }

  private resolveId(target: string, context: ToolExecutionContext, inputField = 'target'): string {
    const normalized = target.trim()
    const snapshot = this.ownedHandles(context).find(candidate => candidate.id === normalized || candidate.name === normalized)
    if (snapshot === undefined) this.throwMissingTarget(target, context, inputField)
    return snapshot.id
  }

  private throwMissingTarget(target: string, context: ToolExecutionContext, inputField: string): never {
    const available = this.ownedHandles(context)
    if (!available.length) {
      throw new ValidationError(
        inputField,
        'managed subagent not found; TaskListTool returned no tasks attached to the current runtime. Do not retry stale names or ids; respawn the required work if no persisted result is available',
        target,
      )
    }
    const targets = available
      .slice(0, 20)
      .map(snapshot => snapshot.id === snapshot.name ? snapshot.id : `${snapshot.name} (${snapshot.id})`)
      .join(', ')
    throw new ValidationError(
      inputField,
      `managed subagent not found; use an exact target from TaskListTool. Available targets: ${targets}`,
      target,
    )
  }

  private ownedHandles(context: ToolExecutionContext): SpawnedAgentSnapshot[] {
    const owner = contextOwnerId(context)
    return this.options.manager.listHandles().filter(snapshot => normalizeOwnerId(snapshot.sourceAgentId) === owner)
  }

  private capture(): void {
    this.mailbox.capture(this.options.manager.listHandles())
  }

  private persistContext(context: ToolExecutionContext): void {
    replacePersistedSubagentSnapshots(context.metadata, this.ownedHandles(context))
  }
}

function definition(
  name: string,
  description: string,
  properties: Record<string, unknown>,
  required: readonly string[] = [],
): ToolDefinition {
  return {
    type: 'function',
    function: {
      name,
      description,
      parameters: { type: 'object', additionalProperties: false, properties, ...(required.length ? { required } : {}) },
    },
  }
}

function stringSchema(description: string): Record<string, unknown> {
  return { type: 'string', description }
}

function titleSchema(description: string): Record<string, unknown> {
  return { type: 'string', description, minLength: 1, maxLength: MAX_AGENT_TITLE_LENGTH }
}

function booleanSchema(description: string, defaultValue?: boolean): Record<string, unknown> {
  return {
    type: 'boolean',
    description,
    ...(defaultValue === undefined ? {} : { default: defaultValue }),
  }
}

function numberSchema(description: string): Record<string, unknown> {
  return { type: 'number', description }
}

function parseAgentSpecs(value: JsonValue | undefined): ClaudeAgentSpec[] {
  const parsed = parseAgentPayload(value)
  const specs: ClaudeAgentSpec[] = []
  for (const [index, item] of parsed.entries()) {
    if (!isRecord(item)) throw new ValidationError('agents', `entry ${index} must be an object`, item)
    const prompt = stringField(item, 'prompt')
    if (!prompt) throw new ValidationError('agents', `entry ${index} requires a non-empty prompt`, item)
    const rawTitle = optionalRecordString(item, 'title')
    if (!rawTitle) throw new ValidationError('agents', `entry ${index} requires a non-empty title`, item)
    const title = normalizeAgentTitle(rawTitle, `agents[${index}].title`)
    const name = optionalRecordString(item, 'name')
    const subagentType = optionalRecordString(item, 'subagent_type')
    const model = optionalRecordString(item, 'model')
    specs.push({
      prompt,
      title,
      ...(name === undefined ? {} : { name }),
      ...(subagentType === undefined ? {} : { subagentType }),
      ...(model === undefined ? {} : { model }),
    })
  }
  return specs
}

/**
 * Parse the common JSON-adjacent formats models emit for SpawnAgents without
 * ever evaluating source text. Strict JSON is always attempted first; the
 * narrow retry only accepts fenced payloads, smart quotes, and basic
 * single-quoted object notation from older provider adapters.
 */
function parseAgentPayload(value: JsonValue | undefined): JsonValue[] {
  if (Array.isArray(value)) return value
  if (isRecord(value)) return [value as JsonValue]
  if (typeof value !== 'string') {
    throw new ValidationError('agents', 'must be an array, object, or JSON-encoded array', value)
  }

  const normalized = normalizeAgentPayload(value)
  const parsed = parseJsonPayload(normalized) ?? parseJsonPayload(normalized.replaceAll("'", '"'))
  if (Array.isArray(parsed)) return parsed as JsonValue[]
  if (isRecord(parsed)) return [parsed as JsonValue]
  throw new ValidationError('agents', 'must be an array, object, or JSON-encoded array', value)
}

function normalizeAgentPayload(value: string): string {
  const trimmed = value.trim()
  const fenced = /^```(?:json)?\s*\n?([\s\S]*?)\n?```$/iu.exec(trimmed)
  const body = fenced?.[1]?.trim() ?? trimmed
  return body
    .replaceAll('\u201c', '"')
    .replaceAll('\u201d', '"')
    .replaceAll('\u2018', "'")
    .replaceAll('\u2019', "'")
}

function parseJsonPayload(value: string): unknown | undefined {
  try {
    return JSON.parse(value) as unknown
  } catch {
    return undefined
  }
}

function parseAgentIds(value: JsonValue | undefined): string[] {
  if (value === undefined || value === null || value === '') return []
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) return []
    try {
      const decoded = JSON.parse(trimmed) as unknown
      if (Array.isArray(decoded)) return decoded.filter(isString).map(item => item.trim()).filter(Boolean)
      if (typeof decoded === 'string') return decoded.trim() ? [decoded.trim()] : []
    } catch {
      return trimmed.split(',').map(item => item.trim()).filter(Boolean)
    }
    throw new ValidationError('agent_ids', 'must be an array, JSON array, or comma-separated string', value)
  }
  if (!Array.isArray(value) || value.some(item => typeof item !== 'string')) {
    throw new ValidationError('agent_ids', 'must be an array of strings', value)
  }
  const ids: string[] = []
  for (const item of value) {
    if (typeof item === 'string' && item.trim()) ids.push(item.trim())
  }
  return ids
}

function parseArrayValue(value: JsonValue | undefined, name: string): JsonValue[] {
  if (Array.isArray(value)) return value
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value) as unknown
      if (Array.isArray(parsed)) return parsed as JsonValue[]
    } catch {
      // Report the same structured validation error below.
    }
  }
  throw new ValidationError(name, 'must be an array or a JSON-encoded array', value)
}

async function settleWithConcurrency<Input, Output>(
  inputs: readonly Input[],
  maxConcurrent: number,
  worker: (input: Input, index: number) => Promise<Output>,
  signal?: AbortSignal,
): Promise<
  | { readonly reason: unknown; readonly results: PromiseSettledResult<Output>[]; readonly stopped: true }
  | { readonly results: PromiseSettledResult<Output>[]; readonly stopped: false }
> {
  const settled: Array<{ readonly index: number; readonly result: PromiseSettledResult<Output> }> = []
  let nextIndex = 0
  let stop: { readonly reason: unknown } | undefined = signal?.aborted
    ? { reason: signal.reason ?? new Error('Subagent spawn cancelled') }
    : undefined
  const stopWith = (reason: unknown): void => {
    stop ??= { reason }
  }
  const run = async (): Promise<void> => {
    while (true) {
      if (signal?.aborted) stopWith(signal.reason ?? new Error('Subagent spawn cancelled'))
      if (stop !== undefined) return
      const index = nextIndex
      nextIndex += 1
      if (index >= inputs.length) return
      const input = inputs[index] as Input
      try {
        settled.push({ index, result: { status: 'fulfilled', value: await worker(input, index) } })
      } catch (reason) {
        settled.push({ index, result: { status: 'rejected', reason } })
        stopWith(reason)
        return
      }
    }
  }
  const workers = Math.min(inputs.length, maxConcurrent)
  await Promise.all(Array.from({ length: workers }, () => run()))
  if (signal?.aborted) stopWith(signal.reason ?? new Error('Subagent spawn cancelled'))
  const results = settled.sort((left, right) => left.index - right.index).map(entry => entry.result)
  return stop === undefined
    ? { results, stopped: false }
    : { reason: stop.reason, results, stopped: true }
}

function timeoutMilliseconds(inputs: JsonObject, name: string, defaultSeconds: number): number {
  const value = inputs[name]
  const seconds = value === undefined ? defaultSeconds : value
  if (typeof seconds !== 'number' || !Number.isFinite(seconds) || seconds < 0) {
    throw new ValidationError(name, 'must be a non-negative number of seconds', value)
  }
  return Math.min(Math.round(seconds * 1_000), 86_400_000)
}

function abortable<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise
  if (signal.aborted) return Promise.reject(signal.reason ?? new Error('Subagent wait cancelled'))
  return new Promise<T>((resolve, reject) => {
    const abort = () => reject(signal.reason ?? new Error('Subagent wait cancelled'))
    signal.addEventListener('abort', abort, { once: true })
    void promise.then(
      value => {
        signal.removeEventListener('abort', abort)
        resolve(value)
      },
      error => {
        signal.removeEventListener('abort', abort)
        reject(error)
      },
    )
  })
}

function nonnegativeInteger(inputs: JsonObject, name: string, defaultValue: number): number {
  const value = inputs[name]
  if (value === undefined) return defaultValue
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new ValidationError(name, 'must be a non-negative integer', value)
  }
  return value
}

function normalizeWakeOn(value: string | undefined): 'all' | 'any' | 'none' {
  const normalized = value?.trim().toLowerCase() || 'all'
  if (normalized === 'all' || normalized === 'any' || normalized === 'none') return normalized
  throw new ValidationError('wake_on', 'must be any, all, or none', value)
}

function isTerminal(snapshot: SpawnedAgentSnapshot): boolean {
  return TERMINAL_STATUSES.has(snapshot.status)
}

function agentsSatisfied(snapshots: readonly SpawnedAgentSnapshot[], wakeOn: 'all' | 'any' | 'none'): boolean {
  if (!snapshots.length || wakeOn === 'none') return false
  return wakeOn === 'all' ? snapshots.every(isTerminal) : snapshots.some(isTerminal)
}

function awaitResult(
  wakeReason: 'agents_done' | 'cancelled' | 'timeout',
  wakeOn: 'all' | 'any' | 'none',
  started: number,
  ended: number,
  snapshots: readonly SpawnedAgentSnapshot[],
): Record<string, unknown> {
  const compact = snapshots.length > MAX_INLINE_BATCH_RECEIPT_AGENTS
  return {
    wake_reason: wakeReason,
    wake_on: wakeOn,
    elapsed_seconds: Number(((ended - started) / 1_000).toFixed(3)),
    agents: compact
      ? snapshots.slice(0, MAX_INLINE_BATCH_RECEIPT_AGENTS).map(compactAgentSnapshotWire)
      : snapshots.map(agentSnapshotWire),
    ...(compact ? batchSummaryFields(snapshots) : {}),
  }
}

function spawnBatchWire(
  snapshots: readonly SpawnedAgentSnapshot[],
): readonly Record<string, unknown>[] | Record<string, unknown> {
  if (snapshots.length <= MAX_INLINE_BATCH_RECEIPT_AGENTS) return snapshots.map(agentSnapshotWire)
  return {
    accepted_count: snapshots.length,
    agents: snapshots.slice(0, MAX_INLINE_BATCH_RECEIPT_AGENTS).map(compactAgentSnapshotWire),
    ...batchSummaryFields(snapshots),
    management_hint: 'AwaitAgents without agent_ids waits for the tracked cohort. Use paged TaskListTool plus TaskGetTool or TaskOutputTool for individual details.',
  }
}

function batchSummaryFields(snapshots: readonly SpawnedAgentSnapshot[]): Record<string, unknown> {
  return {
    agent_count: snapshots.length,
    shown_count: Math.min(snapshots.length, MAX_INLINE_BATCH_RECEIPT_AGENTS),
    omitted_count: Math.max(0, snapshots.length - MAX_INLINE_BATCH_RECEIPT_AGENTS),
    status_counts: statusCounts(snapshots),
  }
}

function statusCounts(snapshots: readonly SpawnedAgentSnapshot[]): Record<string, number> {
  const counts: Record<string, number> = {}
  for (const snapshot of snapshots) counts[snapshot.status] = (counts[snapshot.status] ?? 0) + 1
  return counts
}

function compactAgentSnapshotWire(snapshot: SpawnedAgentSnapshot): Record<string, unknown> {
  const tokenCount = [snapshot.inputTokens, snapshot.outputTokens, snapshot.reasoningTokens]
    .filter((value): value is number => value !== undefined)
    .reduce((total, value) => total + value, 0)
  return {
    id: snapshot.id,
    name: snapshot.name,
    title: snapshot.title,
    status: snapshot.status,
    ...(snapshot.toolCalls === undefined ? {} : { tool_count: snapshot.toolCalls }),
    ...(snapshot.apiCalls === undefined ? {} : { api_calls: snapshot.apiCalls }),
    ...(tokenCount ? { token_count: tokenCount } : {}),
    has_output: snapshot.lastOutput !== undefined,
    ...(snapshot.error ? { error: boundedWireText(snapshot.error, 240) } : {}),
  }
}

function boundedWireText(value: string, limit: number): string {
  return value.length <= limit ? value : `${value.slice(0, limit - 1)}…`
}

function agentSnapshotWire(snapshot: SpawnedAgentSnapshot): Record<string, unknown> {
  return {
    id: snapshot.id,
    name: snapshot.name,
    title: snapshot.title,
    agent_id: snapshot.agentId,
    creator_id: snapshot.creatorAgentId ?? null,
    parent_id: snapshot.parentAgentId ?? null,
    model: snapshot.model ?? null,
    rules: snapshot.rules ?? [],
    toolsets: snapshot.toolsets ?? [],
    ...(snapshot.apiCalls === undefined ? {} : { api_calls: snapshot.apiCalls }),
    ...(snapshot.toolCalls === undefined ? {} : { tool_count: snapshot.toolCalls }),
    ...(snapshot.inputTokens === undefined ? {} : { input_tokens: snapshot.inputTokens }),
    ...(snapshot.outputTokens === undefined ? {} : { output_tokens: snapshot.outputTokens }),
    ...(snapshot.reasoningTokens === undefined ? {} : { reasoning_tokens: snapshot.reasoningTokens }),
    ...(snapshot.filesRead === undefined ? {} : { files_read: snapshot.filesRead }),
    ...(snapshot.filesWritten === undefined ? {} : { files_written: snapshot.filesWritten }),
    summary: snapshot.completionSummary ?? null,
    status: snapshot.status,
    created_at: snapshot.createdAt,
    updated_at: snapshot.updatedAt,
    prompt_profile: snapshot.promptProfile,
    source_agent_id: snapshot.sourceAgentId ?? null,
    last_input: snapshot.lastInput ?? null,
    last_output: snapshot.lastOutput ?? null,
    error: snapshot.error ?? null,
    queue_size: snapshot.queueSize,
    queued_preview: snapshot.queuedPreview ?? null,
    closed: snapshot.closed,
  }
}

function snapshotEventMetadata(snapshot: SpawnedAgentSnapshot): Omit<ClaudeAgentEvent, 'agentId' | 'event' | 'name' | 'output' | 'previousStatus' | 'seq' | 'status'> {
  return {
    title: snapshot.title,
    ...(snapshot.sourceAgentId === undefined ? {} : { sourceAgentId: snapshot.sourceAgentId }),
    ...(snapshot.creatorAgentId === undefined ? {} : { creatorAgentId: snapshot.creatorAgentId }),
    ...(snapshot.parentAgentId === undefined ? {} : { parentAgentId: snapshot.parentAgentId }),
    ...(snapshot.model === undefined ? {} : { model: snapshot.model }),
    ...(snapshot.rules === undefined ? {} : { rules: snapshot.rules }),
    ...(snapshot.toolsets === undefined ? {} : { toolsets: snapshot.toolsets }),
    ...(snapshot.apiCalls === undefined ? {} : { apiCalls: snapshot.apiCalls }),
    ...(snapshot.toolCalls === undefined ? {} : { toolCalls: snapshot.toolCalls }),
    ...(snapshot.inputTokens === undefined ? {} : { inputTokens: snapshot.inputTokens }),
    ...(snapshot.outputTokens === undefined ? {} : { outputTokens: snapshot.outputTokens }),
    ...(snapshot.reasoningTokens === undefined ? {} : { reasoningTokens: snapshot.reasoningTokens }),
    ...(snapshot.filesRead === undefined ? {} : { filesRead: snapshot.filesRead }),
    ...(snapshot.filesWritten === undefined ? {} : { filesWritten: snapshot.filesWritten }),
    ...(snapshot.completionSummary === undefined ? {} : { completionSummary: snapshot.completionSummary }),
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringField(value: Record<string, unknown>, name: string): string {
  const field = value[name]
  return typeof field === 'string' ? field.trim() : ''
}

function optionalRecordString(value: Record<string, unknown>, name: string): string | undefined {
  const field = value[name]
  if (field === undefined) return undefined
  if (typeof field !== 'string') throw new ValidationError(name, 'must be a string', field as JsonValue)
  const trimmed = field.trim()
  return trimmed || undefined
}

function isString(value: unknown): value is string {
  return typeof value === 'string'
}

function normalizeSequence(value: number): number {
  return Number.isInteger(value) && value >= 0 ? value : 0
}

const ALL_MAILBOX_EVENTS = '\0all'

function mailboxScope(ownerOrSince: string | number): string {
  return typeof ownerOrSince === 'number' ? ALL_MAILBOX_EVENTS : normalizeOwnerId(ownerOrSince)
}

function eventMatchesScope(event: ClaudeAgentEvent, scope: string): boolean {
  return scope === ALL_MAILBOX_EVENTS || normalizeOwnerId(event.sourceAgentId) === scope
}

function contextOwnerId(context: ToolExecutionContext): string {
  return normalizeOwnerId(context.sessionId) || normalizeOwnerId(context.agentId)
}

function normalizeOwnerId(value: string | undefined): string {
  return value?.trim() ?? ''
}

function contextPermissionMode(context: ToolExecutionContext): PermissionMode | undefined {
  const value = context.metadata.permission_mode ?? context.metadata.permissionMode
  return value === 'accept-all' || value === 'auto' || value === 'manual' || value === 'plan'
    ? value
    : undefined
}

function sleep(milliseconds: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, Math.max(0, milliseconds)))
}
