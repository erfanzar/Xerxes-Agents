// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { AgentDefinition } from '../agents/definitions.js'
import {
  SUBAGENT_BLOCKED_TOOLS,
  SubAgentManager,
  type SubAgentEvent,
  type SubAgentTask,
  type SubagentTaskRunRequest,
} from '../agents/subagentManager.js'
import type { ContextMessage } from '../context/compressor.js'
import { ValidationError } from '../core/errors.js'
import type { ToolExecutor } from '../executors/toolRegistry.js'
import type { LlmClient } from '../llms/client.js'
import { effectiveContextLimit } from '../llms/providerRegistry.js'
import type {
  SendAgentInputOptions,
  SpawnAgentOptions,
  SpawnedAgentManagerPort,
  SpawnedAgentSnapshot,
  SpawnedAgentStatus,
} from '../operators/subagents.js'
import { bootstrap } from '../runtime/bootstrap.js'
import { looksLikeSessionId, type DaemonTranscriptStore } from '../session/daemonTranscript.js'
import type { AgentState, StreamEvent } from '../streaming/events.js'
import { runTurn } from '../streaming/loop.js'
import type { PermissionBroker, PermissionMode } from '../streaming/permissions.js'
import type { ChatMessage } from '../types/messages.js'
import type { ToolDefinition } from '../types/toolCalls.js'
import {
  compactMessagesIfNeeded,
  compactionCompletionPort,
  compactionThresholdTokens,
  DEFAULT_AUTO_COMPACT_THRESHOLD,
  precompactArchivePath,
  type CompactionStamp,
} from './compactionRunner.js'
import type { DaemonEvent } from './runtime.js'
import {
  NativeSubagentTurnCoordinator,
  type SubagentTurnCoordinator,
} from './subagentCoordinator.js'
import {
  claimSubagentConversation,
  SubagentConversationPersistence,
  type SubagentConversationContext,
} from './subagentConversations.js'
import { DaemonSubagentEventBus } from './subagentEvents.js'

export interface NativeSubagentHostOptions {
  readonly agentDefinitions: ReadonlyMap<string, AgentDefinition>
  /**
   * Fraction of a child's prompt budget at which its conversation is compacted
   * before the turn starts. Defaults to the daemon's own auto-compaction
   * threshold so parent and children never disagree about when a context is
   * full; 0 disables child compaction.
   */
  readonly autoCompactThreshold?: number
  readonly cwd: string
  readonly eventBus: DaemonSubagentEventBus
  /** Bounded supplemental bootstrap context, such as the discovered skill catalog. */
  readonly extraContext?: string
  readonly llm: LlmClient
  readonly maxTokens?: number
  readonly model: string
  readonly permissionMode: PermissionMode
  readonly temperature?: number
  readonly toolExecutor: ToolExecutor
  readonly tools: readonly ToolDefinition[]
  readonly topK?: number
  readonly topP?: number
  /** Shared daemon store; omitted hosts keep child conversations in memory only. */
  readonly transcriptStore?: DaemonTranscriptStore
}

export interface SubagentRetryOptions {
  /**
   * Optional replacement instruction for the new attempt. Defaults to a
   * continuation nudge when the task's conversation persisted, otherwise the
   * task's original prompt.
   */
  readonly message?: string
}

/**
 * Instruction sent as the retry attempt's user message when the dead task's
 * conversation was persisted: the earlier user prompt and partial progress
 * are already in that conversation, so resubmitting the original prompt
 * would duplicate it.
 */
export const SUBAGENT_RETRY_CONTINUATION_PROMPT = [
  '[Retry] Your previous attempt ended before completion (connection failure, cancellation, or an internal error).',
  'Your earlier progress is preserved in this conversation.',
  'Review what is already done, finish the remaining work, and return the final summary.',
].join(' ')

export interface NativeSubagentHost {
  readonly manager: SubAgentManager
  readonly managerPort: SpawnedAgentManagerPort
  readonly turnCoordinator: SubagentTurnCoordinator
  /** Cancel and invalidate every child owned by this host. */
  invalidateAll(): number
  /** Cancel and invalidate every child owned by one parent session. */
  cancelSource(sourceAgentId: string): number
  /**
   * Stop the live children of one parent session after a user interrupt,
   * without invalidating or closing their handles. An interruption is a pause
   * the user may undo, so every cancelled child stays inspectable in the
   * agents panel and retryable under its stable identity.
   */
  interruptSource(sourceAgentId: string): number
  /**
   * Start a new attempt for a dead (failed/cancelled) task under its stable
   * identity. The persisted conversation continues when one survives;
   * retrying a live task returns its current snapshot instead of starting a
   * duplicate, so the operation is idempotent against repeated invocations.
   */
  retry(task: string, options?: SubagentRetryOptions): Promise<SpawnedAgentSnapshot>
  /**
   * Apply the latest daemon/provider generation without discarding delegated
   * task handles. Existing tasks keep the execution generation they were
   * created with unless permissions are tightened; subsequently spawned tasks
   * use these options.
   */
  reconfigure(options: NativeSubagentHostOptions): void
}

/** Build the real Bun delegated-turn host used by daemon and OpenTUI sessions. */
export function createNativeSubagentHost(options: NativeSubagentHostOptions): NativeSubagentHost {
  let activeGeneration = 0
  let activeOptions = options
  let activeDefinitionsFingerprint = agentDefinitionsFingerprint(options.agentDefinitions)
  const generationOptions = new Map<number, NativeSubagentHostOptions>([[activeGeneration, options]])
  const conversations = new SubagentConversationPersistence(options.transcriptStore)
  const historySessionIds = new Map<string, string>()
  /** Child depth advertised by the manager for each in-flight run, keyed by the unique running task id. */
  const runningChildDepths = new Map<string, { readonly childDepth: number, readonly taskId: string }>()
  const manager = new SubAgentManager({
    idFactory: () => {
      const taskId = `subagent_${crypto.randomUUID().replaceAll('-', '').slice(0, 12)}`
      if (options.transcriptStore) {
        historySessionIds.set(taskId, crypto.randomUUID().replaceAll('-', ''))
      }
      return taskId
    },
    onEvent: event => publishSubagentEvent(options.eventBus, event, historySessionIds.get(event.taskId)),
    pathResolver: rawPath => rawPath,
    runner: request => {
      const generation = nativeHostGeneration(request.config)
      return runNativeSubagent(
        request,
        generation === undefined ? activeOptions : generationOptions.get(generation) ?? activeOptions,
        conversations,
        historySessionIds.get(request.task.id),
        runningChildDepths,
      )
    },
  })
  const liveManagerPort = new RichSubagentManagerPort(
    manager,
    options,
    activeGeneration,
    historySessionIds,
    runningChildDepths,
  )
  const managerPort = new RecoverableSubagentManagerPort(liveManagerPort)
  const turnCoordinator = new NativeSubagentTurnCoordinator(
    manager,
    () => managerPort.listHandles(),
    undefined,
    undefined,
    snapshots => managerPort.restoreSnapshots(snapshots),
  )
  /** Drop option generations no live task can still start with; the active generation is always kept. */
  const pruneGenerationOptions = (): void => {
    const live = liveManagerPort.liveGenerations()
    for (const generation of generationOptions.keys()) {
      if (generation !== activeGeneration && !live.has(generation)) generationOptions.delete(generation)
    }
  }
  return {
    manager,
    managerPort,
    turnCoordinator,
    invalidateAll: () => managerPort.invalidateAll(),
    cancelSource: sourceAgentId => managerPort.invalidateSource(sourceAgentId),
    interruptSource: sourceAgentId => managerPort.interruptSource(sourceAgentId),
    retry: (task, retryOptions) => managerPort.retry(task, retryOptions ?? {}),
    reconfigure(nextOptions) {
      if (nextOptions.eventBus !== options.eventBus) {
        throw new Error('A native subagent host cannot be moved to a different event bus')
      }
      if (nextOptions.transcriptStore !== options.transcriptStore) {
        throw new Error('A native subagent host cannot be moved to a different transcript store')
      }
      const nextDefinitionsFingerprint = agentDefinitionsFingerprint(nextOptions.agentDefinitions)
      if (nextDefinitionsFingerprint !== activeDefinitionsFingerprint) {
        managerPort.invalidateAll()
      } else {
        managerPort.invalidateHandlesExceeding(nextOptions.permissionMode)
      }
      activeGeneration += 1
      activeOptions = nextOptions
      activeDefinitionsFingerprint = nextDefinitionsFingerprint
      generationOptions.set(activeGeneration, nextOptions)
      managerPort.reconfigure(nextOptions, activeGeneration)
      pruneGenerationOptions()
    },
  }
}

interface HandleMetadata {
  readonly agentId: string
  closed: boolean
  readonly createdAt: string
  readonly creatorAgentId: string | undefined
  readonly generation: number
  readonly historySessionId: string | undefined
  lastInput: string | undefined
  readonly parentAgentId: string | undefined
  readonly permissionMode: PermissionMode
  readonly promptProfile: string
  readonly sourceAgentId: string | undefined
}

/** Adapt the richer native manager to the Claude-compatible tool contract. */
class RichSubagentManagerPort implements SpawnedAgentManagerPort {
  private availableTools: readonly ToolDefinition[]
  private cwd: string
  private definitions: ReadonlyMap<string, AgentDefinition>
  private fallbackModel: string
  private fallbackPermissionMode: PermissionMode
  private generation: number
  private readonly handles = new Map<string, HandleMetadata>()
  private readonly invalidatedHandles = new Set<string>()
  private readonly pendingResume = new Set<string>()
  private transcripts: DaemonTranscriptStore | undefined

  constructor(
    private readonly manager: SubAgentManager,
    options: NativeSubagentHostOptions,
    generation: number,
    private readonly historySessionIds: Map<string, string>,
    private readonly runningChildDepths: ReadonlyMap<string, { readonly childDepth: number, readonly taskId: string }>,
  ) {
    this.availableTools = options.tools
    this.cwd = options.cwd
    this.definitions = options.agentDefinitions
    this.fallbackModel = options.model
    this.fallbackPermissionMode = options.permissionMode
    this.generation = generation
    this.transcripts = options.transcriptStore
  }

  reconfigure(options: NativeSubagentHostOptions, generation: number): void {
    this.availableTools = options.tools
    this.cwd = options.cwd
    this.definitions = options.agentDefinitions
    this.fallbackModel = options.model
    this.fallbackPermissionMode = options.permissionMode
    this.generation = generation
    this.transcripts = options.transcriptStore
  }

  listHandles(): SpawnedAgentSnapshot[] {
    return this.manager.listTasks().map(task => this.snapshot(task))
  }

  /** Generations still pinned by a task that has not reached a terminal state. */
  liveGenerations(): Set<number> {
    const generations = new Set<number>()
    for (const task of this.manager.listTasks()) {
      if (task.status !== 'pending' && task.status !== 'running') continue
      generations.add(this.handles.get(task.id)?.generation ?? this.generation)
    }
    return generations
  }

  async spawn(options: SpawnAgentOptions = {}): Promise<SpawnedAgentSnapshot> {
    const prompt = (options.message ?? options.taskDescription)?.trim()
    if (!prompt) throw new ValidationError('message', 'spawned agent input is required', prompt)
    const name = options.nickname?.trim()
    if (name && this.manager.listTasks().some(task => task.name === name && !this.handles.get(task.id)?.closed)) {
      throw new ValidationError('nickname', 'already identifies a spawned agent', name)
    }

    const requestedType = options.promptProfile?.trim() || options.agent?.name?.trim() || 'coder'
    const definition = this.resolveChildDefinition(options.creatorAgentId, requestedType)
    if (!definition) {
      throw new ValidationError(
        'subagent_type',
        `is not a registered agent profile; available profiles: ${visibleDefinitionNames(this.definitions).join(', ') || '(none)'}`,
        requestedType,
      )
    }
    const model = stringConfig(options.agent?.model)
      || stringConfig(definition.model)
      || stringConfig(options.parentModel)
      || this.fallbackModel
    const requestedPermissionMode = permissionModeConfig(options.permissionMode, this.fallbackPermissionMode)
    const permissionMode = delegatedPermissionExceeds(requestedPermissionMode, this.fallbackPermissionMode)
      ? this.fallbackPermissionMode
      : requestedPermissionMode
    const task = await this.spawnResolved({
      definition,
      input: prompt,
      model,
      permissionMode,
      ...(options.title ? { title: options.title } : {}),
      ...(name ? { name } : {}),
      ...(options.sourceAgentId ? { sourceAgentId: options.sourceAgentId } : {}),
      ...(options.creatorAgentId ? { creatorAgentId: options.creatorAgentId } : {}),
      ...(options.parentAgentId ? { parentAgentId: options.parentAgentId } : {}),
    })
    return this.snapshot(task)
  }

  /** Shared spawn core used by fresh spawns and identity-preserving retry respawns. */
  private async spawnResolved(resolved: {
    readonly creatorAgentId?: string
    readonly definition: AgentDefinition
    readonly historySessionId?: string
    readonly input: string
    readonly model: string
    readonly name?: string
    readonly parentAgentId?: string
    readonly permissionMode: PermissionMode
    readonly sourceAgentId?: string
    readonly taskId?: string
    readonly title?: string
  }): Promise<SubAgentTask> {
    const { definition, model, permissionMode } = resolved
    const config = {
      model,
      permissionMode,
      _nativeSubagentHostGeneration: this.generation,
      ...(definition.allowedTools === null
        ? {}
        : { _toolsAllowed: [...definition.allowedTools] }),
      ...(definition.excludeTools.length ? { _toolsExcluded: [...definition.excludeTools] } : {}),
      ...(definition.tools.length ? { _toolsWhitelist: [...definition.tools] } : {}),
    }
    const toolsets = subagentTools(this.availableTools, config).map(tool => tool.function.name)
    const rules = nativeRuleLabels(permissionMode, definition.isolation)
    const parentKey = resolved.parentAgentId ?? resolved.creatorAgentId
    const childDepth = parentKey === undefined ? undefined : this.parentRunningChildDepth(parentKey)
    // An identity-preserving respawn must register its persisted history
    // before the first turn starts so the runner continues that conversation
    // instead of opening a fresh one under the task id.
    if (resolved.taskId && resolved.historySessionId && this.transcripts) {
      this.historySessionIds.set(resolved.taskId, resolved.historySessionId)
    }
    const task = await this.manager.spawn({
      prompt: resolved.input,
      ...(resolved.title ? { title: resolved.title } : {}),
      ...(resolved.name ? { name: resolved.name } : {}),
      agentDefinition: definition,
      ...(resolved.sourceAgentId ? { sourceId: resolved.sourceAgentId } : {}),
      ...(resolved.creatorAgentId ? { creatorId: resolved.creatorAgentId } : {}),
      ...(resolved.parentAgentId ? { parentId: resolved.parentAgentId } : {}),
      ...(childDepth === undefined ? {} : { depth: childDepth }),
      ...(resolved.taskId ? { id: resolved.taskId } : {}),
      model,
      rules,
      toolsets,
      config,
    })
    this.handles.set(task.id, {
      agentId: definition.name,
      closed: false,
      createdAt: new Date().toISOString(),
      creatorAgentId: resolved.creatorAgentId,
      generation: this.generation,
      historySessionId: this.historySessionIds.get(task.id),
      lastInput: resolved.input,
      parentAgentId: resolved.parentAgentId ?? resolved.creatorAgentId,
      permissionMode,
      promptProfile: definition.name,
      sourceAgentId: resolved.sourceAgentId,
    })
    return task
  }

  /**
   * Retry a terminal task under its stable identity. A live task returns its
   * current snapshot unchanged (idempotent double-click). The new attempt
   * continues the persisted conversation when one survives, otherwise it
   * resubmits the original prompt.
   */
  async retry(handleId: string | undefined, options: SubagentRetryOptions = {}): Promise<SpawnedAgentSnapshot> {
    const value = handleId?.trim()
    if (!value) throw new ValidationError('handle_id', 'spawned agent id or name is required', handleId)
    const info = this.manager.findTask(value)
    if (info === undefined) throw new ValidationError('handle_id', 'spawned agent not found', value)
    if (this.invalidatedHandles.has(info.id)) {
      throw new ValidationError(
        'handle_id',
        'was invalidated when permissions were tightened; spawn a new agent under the current policy',
        info.id,
      )
    }
    // Idempotent fast path: a live task returns its current snapshot without
    // starting (or even planning) another attempt.
    if (info.status === 'pending' || info.status === 'running') {
      const live = this.manager.listTasks().find(candidate => candidate.id === info.id)
      if (live) return this.snapshot(live)
    }
    // Any terminal status may be retried. Provider connection failures end
    // a turn with `[Error: …]` output in the completed state, so rejecting
    // "completed" tasks would refuse exactly the dead agents retry exists for.
    const metadata = this.handles.get(info.id)
    const historySessionId = this.historySessionIds.get(info.id) ?? metadata?.historySessionId
    // Restore the persisted history link before the attempt starts so the
    // runner resumes the prior conversation rather than opening a fresh one.
    if (historySessionId && this.transcripts) this.historySessionIds.set(info.id, historySessionId)
    const input = options.message?.trim()
      || await this.continuationInput(historySessionId, metadata?.lastInput ?? '')
    const task = await this.manager.retry(info.id, input)
    if (task === undefined) {
      throw new ValidationError(
        'handle_id',
        'could not be retried because its runtime state never started or is gone; spawn a new agent instead',
        info.id,
      )
    }
    this.pendingResume.delete(task.id)
    if (metadata) {
      metadata.closed = false
      metadata.lastInput = input
    }
    return this.snapshot(task)
  }

  /**
   * Respawn a task recovered from a persisted parent transcript after a
   * daemon restart, keeping its stable task id, name, and history link.
   */
  async respawnRecovered(snapshot: SpawnedAgentSnapshot, input: string): Promise<SpawnedAgentSnapshot> {
    const definition = this.resolveChildDefinition(snapshot.creatorAgentId, snapshot.promptProfile)
    if (!definition) {
      throw new ValidationError(
        'subagent_type',
        `is not a registered agent profile; available profiles: ${visibleDefinitionNames(this.definitions).join(', ') || '(none)'}`,
        snapshot.promptProfile,
      )
    }
    const model = snapshot.model?.trim() || stringConfig(definition.model) || this.fallbackModel
    const requestedMode = snapshot.rules?.length ? permissionModeFromRules(snapshot.rules) : this.fallbackPermissionMode
    const permissionMode = delegatedPermissionExceeds(requestedMode, this.fallbackPermissionMode)
      ? this.fallbackPermissionMode
      : requestedMode
    const task = await this.spawnResolved({
      definition,
      input,
      model,
      permissionMode,
      taskId: snapshot.id,
      ...(snapshot.historySessionId ? { historySessionId: snapshot.historySessionId } : {}),
      ...(snapshot.name ? { name: snapshot.name } : {}),
      ...(snapshot.title ? { title: snapshot.title } : {}),
      ...(snapshot.creatorAgentId ? { creatorAgentId: snapshot.creatorAgentId } : {}),
      ...(snapshot.parentAgentId ? { parentAgentId: snapshot.parentAgentId } : {}),
      ...(snapshot.sourceAgentId ? { sourceAgentId: snapshot.sourceAgentId } : {}),
    })
    return this.snapshot(task)
  }

  /**
   * Choose the retry attempt's input: a continuation nudge when the task's
   * conversation persisted, otherwise the recorded original prompt so a task
   * that died before its first checkpoint still gets its instructions.
   */
  async continuationInput(historySessionId: string | undefined, fallbackInput: string): Promise<string> {
    const store = this.transcripts
    if (!historySessionId || !store) return fallbackInput
    try {
      const transcript = await store.load(historySessionId, { currentProjectDirectory: this.cwd })
      if (transcript && transcript.messages.length > 0) return SUBAGENT_RETRY_CONTINUATION_PROMPT
    } catch {
      // An unreadable history falls back to resubmitting the original prompt.
    }
    return fallbackInput
  }

  /**
   * The parent's tools advertise its task id, nickname, or shared profile
   * name as their agent id. Depth entries are keyed by the unique task id,
   * so a nickname or profile key resolves only when exactly one running task
   * matches it; concurrent siblings sharing a profile never inherit each
   * other's depth entry.
   */
  private parentRunningChildDepth(parentKey: string): number | undefined {
    const direct = this.runningChildDepths.get(parentKey)
    if (direct) return direct.childDepth
    const runningTaskIds = new Set<string>()
    for (const entry of this.runningChildDepths.values()) runningTaskIds.add(entry.taskId)
    const candidates = new Set<string>()
    for (const task of this.manager.listTasks()) {
      if (!runningTaskIds.has(task.id)) continue
      if (task.name === parentKey || task.agentDefName === parentKey) candidates.add(task.id)
    }
    if (candidates.size !== 1) return undefined
    const [taskId] = candidates
    return taskId === undefined ? undefined : this.runningChildDepths.get(taskId)?.childDepth
  }

  /** Cancel handles whose delegated policy grants capabilities absent from the new parent policy. */
  invalidateHandlesExceeding(nextMode: PermissionMode): number {
    return this.invalidateMatching(metadata => delegatedPermissionExceeds(metadata.permissionMode, nextMode))
  }

  /** Cancel and permanently close every handle owned by this host. */
  invalidateAll(): number {
    return this.invalidateMatching(() => true)
  }

  /** Cancel and permanently close children whose owning session is being removed. */
  invalidateSource(sourceAgentId: string): number {
    const source = sourceAgentId.trim()
    if (!source) return 0
    return this.invalidateMatching(metadata => metadata.sourceAgentId === source)
  }

  /**
   * Cancel the live children of one parent session on user interrupt. The
   * handle keeps its identity, history link, and open state, so the cancelled
   * child reports an honest terminal status and can still be retried.
   */
  interruptSource(sourceAgentId: string): number {
    const source = sourceAgentId.trim()
    if (!source) return 0
    let cancelled = 0
    for (const task of this.manager.listTasks()) {
      // Fall back to the task's own recorded parent the way snapshot() does,
      // so a child whose handle metadata was never registered still stops
      // instead of quietly outliving the turn that owns it.
      const owner = this.handles.get(task.id)?.sourceAgentId ?? (task.sourceId || undefined)
      if (owner !== source) continue
      if (task.status !== 'pending' && task.status !== 'running') continue
      if (this.manager.cancel(task.id)) cancelled += 1
    }
    return cancelled
  }

  private invalidateMatching(predicate: (metadata: HandleMetadata) => boolean): number {
    let cancelled = 0
    for (const task of this.manager.listTasks()) {
      const metadata = this.handles.get(task.id)
      if (!metadata || !predicate(metadata)) continue
      this.invalidatedHandles.add(task.id)
      this.pendingResume.delete(task.id)
      metadata.closed = true
      if (task.status === 'pending' || task.status === 'running') {
        if (this.manager.cancel(task.id)) cancelled += 1
      }
      this.historySessionIds.delete(task.id)
    }
    return cancelled
  }

  private resolveChildDefinition(
    creatorAgentId: string | undefined,
    requestedType: string,
  ): AgentDefinition | undefined {
    const creator = creatorAgentId?.trim()
    if (!creator) return resolveDefinition(this.definitions, requestedType)
    const creatorDefinition = this.definitions.get(creator)
    if (!creatorDefinition) {
      throw new ValidationError('creator_agent_id', 'is not a registered agent profile', creator)
    }
    const catalog = creatorDefinition.subagents ?? {}
    const catalogName = Object.hasOwn(catalog, requestedType)
      ? requestedType
      : canonicalProfileAlias(requestedType)
    const reference = catalogName ? catalog[catalogName] : undefined
    if (!catalogName || !reference) {
      const allowed = Object.keys(catalog)
      throw new ValidationError(
        'subagent_type',
        `is not allowed by agent '${creator}'; allowed profiles: ${allowed.sort().join(', ') || '(none)'}`,
        requestedType,
      )
    }
    const profileKey = reference.resolvedProfile ?? catalogName
    const definition = this.definitions.get(profileKey)
    if (!definition) {
      throw new ValidationError(
        'subagent_type',
        `catalog entry '${catalogName}' for agent '${creator}' does not resolve to a registered profile`,
        requestedType,
      )
    }
    return definition
  }

  async sendInput(handleId: string | undefined, options: SendAgentInputOptions): Promise<SpawnedAgentSnapshot> {
    const task = this.requireTask(handleId)
    const input = (options.message ?? options.taskDescription)?.trim()
    if (!input) throw new ValidationError('message', 'spawned agent input is required', input)
    if (this.pendingResume.delete(task.id)) {
      const replacement = await this.manager.reset(task.id, input)
      if (!replacement) throw new ValidationError('handle_id', 'could not restart spawned agent', task.id)
      const previous = this.handles.get(task.id)
      if (previous) {
        previous.closed = true
        this.historySessionIds.delete(task.id)
        this.handles.set(replacement.id, {
          ...previous,
          closed: false,
          historySessionId: this.historySessionIds.get(replacement.id),
          lastInput: input,
        })
      }
      return this.snapshot(replacement)
    }
    if (!(await this.manager.sendMessage(task.id, input))) {
      throw new ValidationError('handle_id', 'spawned agent is not accepting input', task.id)
    }
    const metadata = this.handles.get(task.id)
    if (metadata) metadata.lastInput = input
    return this.snapshot(task)
  }

  async wait(targets: readonly string[], timeoutMs = 30_000): Promise<{
    readonly completed: readonly SpawnedAgentSnapshot[]
    readonly pending: readonly SpawnedAgentSnapshot[]
  }> {
    const ids = targets.map(target => this.requireTask(target).id)
    const result = await this.manager.waitAll(ids, timeoutMs)
    return {
      completed: result.completed.map(snapshot => this.snapshot(this.requireTask(snapshot.id))),
      pending: result.pending.map(snapshot => this.snapshot(this.requireTask(snapshot.id))),
    }
  }

  resume(handleId: string): SpawnedAgentSnapshot {
    const task = this.requireTask(handleId)
    if (this.invalidatedHandles.has(task.id)) {
      throw new ValidationError(
        'handle_id',
        'was invalidated when permissions were tightened; spawn a new agent under the current policy',
        task.id,
      )
    }
    this.pendingResume.add(task.id)
    const metadata = this.handles.get(task.id)
    if (metadata) metadata.closed = false
    return this.snapshot(task, 'idle')
  }

  close(handleId: string): SpawnedAgentSnapshot & { readonly previousStatus: SpawnedAgentStatus } {
    const task = this.requireTask(handleId)
    const previousStatus = spawnedStatus(task)
    this.manager.cancel(task.id)
    const metadata = this.handles.get(task.id)
    if (metadata) metadata.closed = true
    this.historySessionIds.delete(task.id)
    return { ...this.snapshot(task, 'closed'), previousStatus }
  }

  private requireTask(idOrName: string | undefined): SubAgentTask {
    const value = idOrName?.trim()
    const task = value
      ? this.manager.listTasks().find(candidate => candidate.id === value || candidate.name === value)
      : this.manager.listTasks().filter(candidate => !this.handles.get(candidate.id)?.closed).at(-1)
    if (!task) throw new ValidationError('handle_id', 'spawned agent not found', value)
    return task
  }

  private snapshot(task: SubAgentTask, statusOverride?: SpawnedAgentStatus): SpawnedAgentSnapshot {
    const metadata = this.handles.get(task.id) ?? {
      agentId: task.agentDefName || task.name,
      closed: false,
      createdAt: new Date().toISOString(),
      creatorAgentId: task.creatorId || undefined,
      generation: this.generation,
      historySessionId: this.historySessionIds.get(task.id),
      lastInput: task.prompt,
      parentAgentId: task.parentId || undefined,
      permissionMode: permissionModeFromRules(task.rules),
      promptProfile: task.agentDefName || 'coder',
      sourceAgentId: task.sourceId || undefined,
    }
    this.handles.set(task.id, metadata)
    const status = statusOverride ?? spawnedStatus(task)
    const updatedAt = new Date(task.lastActivityAt ?? Date.now()).toISOString()
    return Object.freeze({
      agentId: metadata.agentId,
      attempt: task.attempt,
      closed: metadata.closed || status === 'closed',
      createdAt: metadata.createdAt,
      ...(task.error ? { error: task.error } : {}),
      ...(metadata.historySessionId ? { historySessionId: metadata.historySessionId } : {}),
      id: task.id,
      ...(metadata.lastInput ? { lastInput: metadata.lastInput } : {}),
      ...(task.result === undefined ? {} : { lastOutput: task.result }),
      name: task.name,
      title: task.title,
      ...(task.creatorId ? { creatorAgentId: task.creatorId } : {}),
      ...(task.parentId ? { parentAgentId: task.parentId } : {}),
      ...(task.model ? { model: task.model } : {}),
      rules: task.rules,
      toolsets: task.toolsets,
      ...(task.apiCalls === undefined ? {} : { apiCalls: task.apiCalls }),
      toolCalls: task.toolCallsCount,
      ...(task.inputTokens === undefined ? {} : { inputTokens: task.inputTokens }),
      ...(task.cacheReadTokens === undefined ? {} : { cacheReadTokens: task.cacheReadTokens }),
      ...(task.cacheCreationTokens === undefined ? {} : { cacheCreationTokens: task.cacheCreationTokens }),
      ...(task.outputTokens === undefined ? {} : { outputTokens: task.outputTokens }),
      ...(task.reasoningTokens === undefined ? {} : { reasoningTokens: task.reasoningTokens }),
      filesRead: Object.freeze([...task.readFiles].sort()),
      filesWritten: Object.freeze([...task.writtenFiles].sort()),
      ...(task.result === undefined ? {} : { completionSummary: task.result.slice(0, 500) }),
      promptProfile: metadata.promptProfile,
      queueSize: task.inboxSize,
      ...(metadata.sourceAgentId ? { sourceAgentId: metadata.sourceAgentId } : {}),
      status,
      updatedAt,
    })
  }
}

const RECOVERED_TERMINAL_STATUSES = new Set<SpawnedAgentStatus>([
  'cancelled',
  'closed',
  'completed',
  'error',
  'interrupted',
])

const DAEMON_RESTART_INTERRUPTION = 'Subagent execution was interrupted because its daemon process ended. The last known metadata and output were recovered from the parent transcript; use ResetAgent to rerun it.'

/**
 * Keeps honest, inspectable tombstones for tasks recorded in a resumed parent
 * transcript. A native child cannot survive its Bun process, but losing its
 * handle entirely makes TaskList/Await retry stale ids forever.
 */
class RecoverableSubagentManagerPort implements SpawnedAgentManagerPort {
  private readonly recovered = new Map<string, SpawnedAgentSnapshot>()
  private readonly pendingRestart = new Set<string>()
  private readonly tombstones = new Set<string>()

  constructor(private readonly live: RichSubagentManagerPort) {}

  reconfigure(options: NativeSubagentHostOptions, generation: number): void {
    this.live.reconfigure(options, generation)
  }

  restoreSnapshots(snapshots: readonly SpawnedAgentSnapshot[]): number {
    const liveIds = new Set(this.live.listHandles().map(snapshot => snapshot.id))
    let restored = 0
    for (const snapshot of snapshots) {
      if (liveIds.has(snapshot.id) || this.recovered.has(snapshot.id) || this.tombstones.has(snapshot.id)) continue
      this.recovered.set(snapshot.id, recoveredTombstone(snapshot))
      restored += 1
    }
    return restored
  }

  listHandles(): SpawnedAgentSnapshot[] {
    const live = this.live.listHandles()
    const liveIds = new Set(live.map(snapshot => snapshot.id))
    return [...live, ...[...this.recovered.values()].filter(snapshot => !liveIds.has(snapshot.id))]
      .sort((left, right) => left.createdAt.localeCompare(right.createdAt) || left.id.localeCompare(right.id))
  }

  async spawn(options: SpawnAgentOptions = {}): Promise<SpawnedAgentSnapshot> {
    // Recovered tombstones reserve their nicknames just like live tasks do:
    // reusing one would make every name-based lookup ambiguous between the
    // fresh task and the stale restart record.
    const nickname = options.nickname?.trim()
    if (nickname && [...this.recovered.values()].some(snapshot => !snapshot.closed && snapshot.name === nickname)) {
      throw new ValidationError('nickname', 'already identifies a spawned agent', nickname)
    }
    return this.live.spawn(options)
  }

  async sendInput(
    handleId: string | undefined,
    options: SendAgentInputOptions,
  ): Promise<SpawnedAgentSnapshot> {
    if (this.findLive(handleId)) return this.live.sendInput(handleId, options)
    const recovered = this.findRecovered(handleId)
    if (!recovered) return this.live.sendInput(handleId, options)
    if (!this.pendingRestart.delete(recovered.id)) {
      throw new ValidationError(
        'handle_id',
        'belongs to a task interrupted by a daemon restart; call ResetAgent to rerun it',
        recovered.id,
      )
    }
    const input = (options.message ?? options.taskDescription)?.trim()
    if (!input) throw new ValidationError('message', 'spawned agent input is required', input)
    const replacement = await this.live.spawn({
      agent: {
        id: recovered.agentId,
        ...(recovered.model ? { model: recovered.model } : {}),
        name: recovered.promptProfile,
      },
      message: input,
      nickname: recovered.name,
      ...(recovered.creatorAgentId ? { creatorAgentId: recovered.creatorAgentId } : {}),
      ...(recovered.parentAgentId ? { parentAgentId: recovered.parentAgentId } : {}),
      promptProfile: recovered.promptProfile,
      ...(recovered.sourceAgentId ? { sourceAgentId: recovered.sourceAgentId } : {}),
      title: recovered.title,
    })
    this.recovered.delete(recovered.id)
    this.tombstones.add(recovered.id)
    return replacement
  }

  async wait(targets: readonly string[], timeoutMs = 30_000): Promise<{
    readonly completed: readonly SpawnedAgentSnapshot[]
    readonly pending: readonly SpawnedAgentSnapshot[]
  }> {
    const active: string[] = []
    const archived: SpawnedAgentSnapshot[] = []
    for (const target of targets) {
      // Live handles accept nicknames as well as ids, so resolve both before
      // a target is treated as a restart tombstone.
      const live = this.findLive(target)
      if (live) {
        active.push(live.id)
        continue
      }
      const recovered = this.findRecovered(target)
      if (!recovered) throw new ValidationError('handle_id', 'spawned agent not found', target)
      archived.push(recovered)
    }
    const liveResult = active.length
      ? await this.live.wait(active, timeoutMs)
      : { completed: [], pending: [] }
    return Object.freeze({
      completed: Object.freeze([...liveResult.completed, ...archived]),
      pending: Object.freeze([...liveResult.pending]),
    })
  }

  resume(handleId: string): SpawnedAgentSnapshot {
    if (this.findLive(handleId)) return this.live.resume(handleId)
    const recovered = this.findRecovered(handleId)
    if (!recovered) return this.live.resume(handleId)
    this.pendingRestart.add(recovered.id)
    return Object.freeze({ ...recovered, closed: false, status: 'idle' })
  }

  /**
   * Retry a dead task under its stable identity. Live tasks delegate to the
   * rich port; restart tombstones respawn with the recovered task id, name,
   * profile, parentage, and persisted history so a dead agent stays
   * resumable in a later session after a daemon restart.
   */
  async retry(handleId: string | undefined, options: SubagentRetryOptions = {}): Promise<SpawnedAgentSnapshot> {
    if (this.findLive(handleId)) return this.live.retry(handleId, options)
    const recovered = this.findRecovered(handleId)
    if (!recovered) return this.live.retry(handleId, options)
    const input = options.message?.trim()
      || await this.live.continuationInput(recovered.historySessionId, recovered.lastInput ?? '')
    if (!input.trim()) {
      throw new ValidationError(
        'handle_id',
        'has no recorded input or persisted conversation to resume from',
        recovered.id,
      )
    }
    const replacement = await this.live.respawnRecovered(recovered, input)
    // The respawned task reuses the recovered identity, so the tombstone is
    // superseded by a live handle rather than tombstoned forever.
    this.recovered.delete(recovered.id)
    this.pendingRestart.delete(recovered.id)
    return replacement
  }

  close(handleId: string): SpawnedAgentSnapshot & { readonly previousStatus: SpawnedAgentStatus } {
    if (this.findLive(handleId)) return this.live.close(handleId)
    const recovered = this.findRecovered(handleId)
    if (!recovered) return this.live.close(handleId)
    const closed = Object.freeze({
      ...recovered,
      closed: true,
      status: 'closed' as const,
      updatedAt: new Date().toISOString(),
    })
    this.recovered.set(recovered.id, closed)
    this.pendingRestart.delete(recovered.id)
    return Object.freeze({ ...closed, previousStatus: recovered.status })
  }

  invalidateAll(): number {
    const cancelled = this.live.invalidateAll()
    for (const snapshot of this.recovered.values()) this.close(snapshot.id)
    return cancelled
  }

  invalidateSource(sourceAgentId: string): number {
    const cancelled = this.live.invalidateSource(sourceAgentId)
    for (const snapshot of this.recovered.values()) {
      if (snapshot.sourceAgentId === sourceAgentId) this.close(snapshot.id)
    }
    return cancelled
  }

  /**
   * Only live children can be interrupted. Recovered tombstones already carry
   * a terminal status from a dead daemon, so an interrupt leaves them alone
   * rather than closing handles the user may still want to retry.
   */
  interruptSource(sourceAgentId: string): number {
    return this.live.interruptSource(sourceAgentId)
  }

  invalidateHandlesExceeding(nextMode: PermissionMode): number {
    return this.live.invalidateHandlesExceeding(nextMode)
  }

  /** Live handles win over restart tombstones so a reused name never misroutes to a stale record. */
  private findLive(idOrName: string | undefined): SpawnedAgentSnapshot | undefined {
    const target = idOrName?.trim()
    if (!target) return undefined
    return this.live.listHandles().find(snapshot => snapshot.id === target || snapshot.name === target)
  }

  private findRecovered(idOrName: string | undefined): SpawnedAgentSnapshot | undefined {
    const target = idOrName?.trim()
    if (!target) return undefined
    return this.recovered.get(target)
      ?? [...this.recovered.values()].find(snapshot => snapshot.name === target)
  }
}

function recoveredTombstone(snapshot: SpawnedAgentSnapshot): SpawnedAgentSnapshot {
  if (RECOVERED_TERMINAL_STATUSES.has(snapshot.status)) return snapshot
  return Object.freeze({
    ...snapshot,
    closed: false,
    error: DAEMON_RESTART_INTERRUPTION,
    queueSize: 0,
    status: 'interrupted',
    updatedAt: new Date().toISOString(),
  })
}

async function runNativeSubagent(
  request: SubagentTaskRunRequest,
  options: NativeSubagentHostOptions,
  conversations: SubagentConversationPersistence,
  persistedHistorySessionId: string | undefined,
  runningChildDepths: Map<string, { readonly childDepth: number, readonly taskId: string }>,
): Promise<{ readonly content: string }> {
  const model = request.task.model.trim() || stringConfig(request.config.model) || options.model
  const permissionMode = permissionModeConfig(request.config.permissionMode, options.permissionMode)
  const permissionBroker = delegatedPermissionBroker(permissionMode)
  const tools = subagentTools(options.tools, request.config)
  const cwd = request.worktree?.path || options.cwd
  const conversation: SubagentConversationContext = {
    agentId: request.task.agentDefName || request.task.id,
    ...(request.task.creatorId ? { creatorAgentId: request.task.creatorId } : {}),
    cwd,
    handleId: request.task.id,
    historySessionId: persistedHistorySessionId ?? request.task.id,
    model,
    ...(request.task.parentId ? { parentAgentId: request.task.parentId } : {}),
    ...(request.task.sourceId ? { parentSessionId: request.task.sourceId } : {}),
    permissionCeiling: options.permissionMode,
    permissionMode,
    profile: request.task.agentDefName || 'coder',
    projectRoot: options.cwd,
    rules: request.task.rules,
    title: request.task.title,
    toolsAllowed: stringList(request.config._toolsAllowed),
    toolsExcluded: stringList(request.config._toolsExcluded),
    toolsWhitelist: stringList(request.config._toolsWhitelist),
    toolsets: request.task.toolsets,
  }
  const releaseConversation = claimSubagentConversation(conversation.historySessionId)
  let state: AgentState
  try {
    state = await conversations.stateFor(conversation)
  } catch (error) {
    releaseConversation()
    throw error
  }
  state.metadata.project_root = options.cwd
  // Before bootstrap, and before the first checkpoint: a reloaded conversation
  // that already fills the window is exactly what a queued follow-up or a
  // retry continuation hands us, and an uncompacted one dies as a provider
  // error the parent can only retry blind.
  await compactChildConversation({
    conversation,
    conversations,
    model,
    options,
    request,
    state,
  })
  // The run request carries the depth children of this task must be spawned at
  // (the manager precomputes task.depth + 1); publish it while the turn runs.
  // Key by the unique task id, never the shared profile name, so concurrent
  // siblings with one profile cannot overwrite each other's depth entry.
  runningChildDepths.set(request.task.id, { childDepth: request.depth, taskId: request.task.id })
  let output = ''
  const previousMessageCount = state.messages.length
  const previousTurnCount = state.turnCount
  let lastCheckpointAt = Date.now()
  let partialAssistantContent = ''
  let partialAssistantThinking = ''
  let partialBaseMessageCount = state.messages.length
  let partialCheckpointed = false

  try {
    try {
      const boot = await bootstrap({
        cwd,
        ...(options.extraContext ? { extraContext: options.extraContext } : {}),
        model,
        tools,
      })
      const events = runTurn({
        agentId: request.task.agentDefName || request.task.id,
        ...(options.maxTokens === undefined ? {} : { maxTokens: options.maxTokens }),
        model,
        permissionMode,
        sessionId: conversation.historySessionId,
        state,
        systemPrompt: [boot.systemPrompt, request.systemPrompt].filter(Boolean).join('\n\n'),
        ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
        ...(options.topK === undefined ? {} : { topK: options.topK }),
        tools,
        ...(options.topP === undefined ? {} : { topP: options.topP }),
        userMessage: request.prompt,
      }, {
        llm: options.llm,
        ...(permissionBroker === undefined ? {} : { permissionBroker }),
        toolExecutor: options.toolExecutor,
      }, request.cancelSignal)
      const iterator = events[Symbol.asyncIterator]()
      let terminalFailure: Error | undefined
      try {
        const firstEventPromise = iterator.next()
        // Attach a rejection handler before the first await so a provider stream
        // that fails during the turn-start checkpoint cannot crash the process
        // with an unhandled rejection; the await below still rethrows it.
        void firstEventPromise.catch(() => undefined)
        await waitForTurnStart(state, previousTurnCount)
        if (state.turnCount > previousTurnCount) {
          await conversations.save(conversation, state, 'running')
          partialBaseMessageCount = state.messages.length
        }
        const checkpoint = async (event: StreamEvent): Promise<void> => {
          if (state.messages.length > partialBaseMessageCount) {
            partialAssistantContent = ''
            partialAssistantThinking = ''
            partialBaseMessageCount = state.messages.length
            partialCheckpointed = false
          }
          const visibleText = reportNativeSubagentEvent(event, request)
          if (event.type === 'turn_done' && (
            event.reason === 'provider_failed' || event.reason === 'context_overflow'
          )) {
            terminalFailure = new Error(
              event.reason === 'context_overflow'
                ? 'Subagent provider context window was exhausted'
                : 'Subagent provider request failed',
            )
          }
          output += visibleText
          if (event.type === 'text') partialAssistantContent += visibleText
          if (event.type === 'thinking') partialAssistantThinking += event.text
          const now = Date.now()
          const timedCheckpoint = (event.type === 'text' || event.type === 'thinking')
            && (!partialCheckpointed || now - lastCheckpointAt >= 1_000)
          const committedCheckpoint = event.type === 'permission_request'
            || event.type === 'tool_start'
            || event.type === 'tool_end'
          if (committedCheckpoint || timedCheckpoint) {
            await conversations.save(
              conversation,
              state,
              'running',
              undefined,
              timedCheckpoint && !committedCheckpoint
                ? { content: partialAssistantContent, thinking: partialAssistantThinking }
                : undefined,
            )
            lastCheckpointAt = now
            if (timedCheckpoint) partialCheckpointed = true
          }
        }
        const firstEvent = await firstEventPromise
        if (!firstEvent.done) await checkpoint(firstEvent.value)
        for await (const event of iterator) await checkpoint(event)
        if (terminalFailure) throw terminalFailure
        await conversations.save(
          conversation,
          state,
          request.cancelSignal.aborted ? 'cancelled' : 'completed',
        )
      } finally {
        // Close the turn generator on every error or early-exit path so an
        // in-flight provider stream cannot leak past this run.
        await iterator.return(undefined)
      }
    } catch (error) {
      const attemptedInputPersisted = state.messages.slice(previousMessageCount).some(message => (
        message.role === 'user' && message.content === request.prompt
      ))
      if (!attemptedInputPersisted) state.messages.push({ role: 'user', content: request.prompt })
      if (state.turnCount === previousTurnCount) state.turnCount = previousTurnCount + 1
      try {
        await conversations.save(
          conversation,
          state,
          request.cancelSignal.aborted ? 'cancelled' : 'error',
          error,
          state.messages.length === partialBaseMessageCount
            ? { content: partialAssistantContent, thinking: partialAssistantThinking }
            : undefined,
        )
      } catch (persistenceError) {
        throw new AggregateError(
          [error, persistenceError],
          'Subagent run failed and its conversation could not be persisted',
        )
      }
      throw error
    }
    return { content: latestAssistantText(state.messages) || output }
  } finally {
    runningChildDepths.delete(request.task.id)
    releaseConversation()
  }
}

interface ChildCompactionRequest {
  readonly conversation: SubagentConversationContext
  readonly conversations: SubagentConversationPersistence
  readonly model: string
  readonly options: NativeSubagentHostOptions
  readonly request: SubagentTaskRunRequest
  readonly state: AgentState
}

/**
 * Compact a child's conversation before its turn starts.
 *
 * The rewrite is persisted here rather than left to the turn: the first
 * tool-event checkpoint saves the whole conversation, so a compaction that had
 * not been written yet would be recorded as the pre-compaction transcript
 * again and the next run would reload it. Failures are warnings — an
 * uncompacted turn may still fit, a child killed by its own housekeeping never
 * does.
 */
async function compactChildConversation(input: ChildCompactionRequest): Promise<void> {
  const { conversation, conversations, model, options, request, state } = input
  if (state.messages.length < 2) return
  // No profile overrides reach a delegated run, so the child prices its window
  // from the model registry. It is the same prompt-budget rule the parent uses.
  const thresholdTokens = compactionThresholdTokens(
    effectiveContextLimit(model),
    options.autoCompactThreshold ?? DEFAULT_AUTO_COMPACT_THRESHOLD,
  )
  if (thresholdTokens <= 0) return
  try {
    const archivePath = childArchivePath(options.transcriptStore, conversation.historySessionId)
    const outcome = await compactMessagesIfNeeded({
      ...(archivePath === undefined ? {} : { archivePath }),
      completion: compactionCompletionPort(options.llm, model),
      messages: state.messages as unknown as ContextMessage[],
      model,
      reason: 'subagent',
      thresholdTokens,
    })
    if (!outcome.compacted) return
    // Splice, not reassign: `AgentState.messages` is a shared array the turn
    // already holds a reference to.
    state.messages.splice(0, state.messages.length, ...(outcome.messages as unknown as ChatMessage[]))
    state.metadata = { ...state.metadata, last_compaction: outcome.stamp }
    await conversations.save(conversation, state, 'running')
    publishChildCompaction(options.eventBus, request, outcome.stamp)
  } catch (error) {
    console.warn(`Could not compact subagent ${request.task.id}: ${errorText(error)}`)
  }
}

/** Archive sidecar beside the child's transcript, or nothing when it has no transcript file. */
function childArchivePath(
  store: DaemonTranscriptStore | undefined,
  historySessionId: string,
): string | undefined {
  if (!store || !looksLikeSessionId(historySessionId)) return undefined
  return precompactArchivePath(store.pathFor(historySessionId))
}

/**
 * Tell the agents overlay why this child's token count dropped.
 *
 * It rides `text_part` because that is a rendered child-progress channel the
 * gateway already forwards; a new event type would need matching wire and UI
 * support and would be dropped silently until it had it.
 */
function publishChildCompaction(
  bus: DaemonSubagentEventBus,
  request: SubagentTaskRunRequest,
  stamp: CompactionStamp,
): void {
  const sourceId = request.task.sourceId
  if (!sourceId) return
  const text = `context compacted: ${stamp.tokens_before} → ${stamp.tokens_after} tokens `
    + `(${stamp.messages_summarized} message(s) summarized)`
  bus.publish(sourceId, {
    type: 'subagent_event',
    payload: {
      agent_id: request.task.id,
      agent_name: request.task.agentDefName || request.task.name,
      creator_id: request.task.creatorId || null,
      depth: request.task.depth,
      goal: request.task.prompt,
      parent_id: request.task.parentId || null,
      subagent_type: request.task.agentDefName || request.task.name,
      title: request.task.title,
      event: { type: 'text_part', payload: { text } },
    },
  })
}

async function waitForTurnStart(state: AgentState, previousTurnCount: number): Promise<void> {
  for (let attempt = 0; attempt < 16 && state.turnCount === previousTurnCount; attempt += 1) {
    await Promise.resolve()
  }
}

function reportNativeSubagentEvent(event: StreamEvent, request: SubagentTaskRunRequest): string {
  switch (event.type) {
    case 'text':
      request.report.text(event.text)
      return event.text
    case 'thinking':
      request.report.thinking(event.text)
      return ''
    case 'tool_start':
      request.report.toolStart({
        inputs: event.call.function.arguments,
        name: event.call.function.name,
        toolCallId: event.call.id,
      })
      return ''
    case 'tool_end':
      request.report.toolEnd({
        durationMs: event.result.durationMs,
        name: event.result.name,
        permitted: event.result.permitted,
        result: event.result.result,
        toolCallId: event.result.toolCallId,
      })
      return ''
    case 'usage_update':
      // Children report while they work, not only when they finish: a subagent
      // running for minutes would otherwise show "no tokens yet" throughout.
      request.report.usage({
        model: event.model,
        inputTokens: event.cumulative.inputTokens,
        outputTokens: event.cumulative.outputTokens,
        ...(event.cumulative.cacheReadTokens === undefined ? {} : { cacheReadTokens: event.cumulative.cacheReadTokens }),
        ...(event.cumulative.cacheCreationTokens === undefined ? {} : { cacheCreationTokens: event.cumulative.cacheCreationTokens }),
        ...(event.cumulative.reasoningTokens === undefined ? {} : { reasoningTokens: event.cumulative.reasoningTokens }),
      })
      return ''
    case 'turn_done':
      request.report.usage({
        ...(event.apiCallsCount === undefined ? {} : { apiCalls: event.apiCallsCount }),
        model: event.model,
        toolCalls: event.toolCallsCount,
        ...(event.usageComplete ? {
          inputTokens: event.usage.inputTokens,
          outputTokens: event.usage.outputTokens,
          ...(event.usage.cacheReadTokens === undefined ? {} : { cacheReadTokens: event.usage.cacheReadTokens }),
          ...(event.usage.cacheCreationTokens === undefined ? {} : { cacheCreationTokens: event.usage.cacheCreationTokens }),
          ...(event.usage.reasoningTokens === undefined ? {} : { reasoningTokens: event.usage.reasoningTokens }),
        } : {}),
      })
      return ''
    default:
      return ''
  }
}

const DELEGATED_PROJECT_MEMORY_WRITES = new Set([
  'agent_memory_append',
  'agent_memory_journal',
  'agent_memory_write',
])

/**
 * A parent-approved auto-mode delegation may persist only project-scoped
 * memory through tools already admitted by the child agent definition. This
 * keeps DeepScan useful without granting children global-memory or workspace
 * write access and without opening eight concurrent approval prompts.
 */
function delegatedPermissionBroker(mode: PermissionMode): PermissionBroker | undefined {
  if (mode !== 'auto') return undefined
  return {
    request: async request =>
      DELEGATED_PROJECT_MEMORY_WRITES.has(request.toolCall.function.name)
        && request.inputs.scope === 'project'
        ? 'approve'
        : 'reject',
  }
}

function subagentTools(
  definitions: readonly ToolDefinition[],
  config: Readonly<Record<string, unknown>>,
): ToolDefinition[] {
  const whitelist = stringList(config._toolsWhitelist)
  const allowed = stringList(config._toolsAllowed)
  const excluded = new Set(stringList(config._toolsExcluded))
  return definitions.filter(definition => {
    const name = definition.function.name
    if (SUBAGENT_BLOCKED_TOOLS.has(name) || excluded.has(name)) return false
    if (whitelist.length && !whitelist.includes(name)) return false
    return !allowed.length || allowed.includes(name)
  })
}

function publishSubagentEvent(
  bus: DaemonSubagentEventBus,
  event: SubAgentEvent,
  historySessionId: string | undefined,
): void {
  if (!event.sourceId) return
  const daemonEvent = daemonEventFromSubagent(event, historySessionId)
  if (daemonEvent) bus.publish(event.sourceId, daemonEvent)
}

function daemonEventFromSubagent(
  event: SubAgentEvent,
  historySessionId: string | undefined,
): DaemonEvent | undefined {
  const base = {
    agent_id: event.taskId,
    agent_name: event.agent,
    title: event.title,
    creator_id: event.creatorId || null,
    depth: event.depth,
    files_read: event.filesRead,
    files_written: event.filesWritten,
    goal: event.goal,
    ...(historySessionId ? { history_session_id: historySessionId } : {}),
    parent_id: event.parentId || null,
    model: event.model || undefined,
    rules: event.rules,
    toolsets: event.toolsets,
    tool_count: event.toolCalls,
    ...(event.apiCalls === undefined ? {} : { api_calls: event.apiCalls }),
    ...(event.inputTokens === undefined ? {} : { input_tokens: event.inputTokens }),
    ...(event.outputTokens === undefined ? {} : { output_tokens: event.outputTokens }),
    ...(event.cacheReadTokens === undefined ? {} : { cache_read_tokens: event.cacheReadTokens }),
    ...(event.cacheCreationTokens === undefined ? {} : { cache_creation_tokens: event.cacheCreationTokens }),
    ...(event.reasoningTokens === undefined ? {} : { reasoning_tokens: event.reasoningTokens }),
    ...(event.completionSummary === undefined ? {} : { summary: event.completionSummary }),
    subagent_type: event.agentType || event.agent,
    task_index: event.sequence,
  }
  const data = event.data
  switch (event.type) {
    case 'spawn':
      return { type: 'subagent_event', payload: { ...base, event: { type: 'turn_begin', payload: { status: 'running' } } } }
    case 'thinking':
      return { type: 'subagent_event', payload: { ...base, event: { type: 'think_part', payload: { think: textValue(data.preview) } } } }
    case 'text_burst':
      return { type: 'subagent_event', payload: { ...base, event: { type: 'text_part', payload: { text: textValue(data.preview) } } } }
    case 'tool_start':
      return {
        type: 'subagent_event',
        payload: {
          ...base,
          event: {
            type: 'tool_call',
            payload: {
              arguments: textValue(data.inputPreview),
              id: textValue(data.toolCallId),
              name: textValue(data.tool),
            },
          },
        },
      }
    case 'tool_end':
      return {
        type: 'subagent_event',
        payload: {
          ...base,
          event: {
            type: 'tool_result',
            payload: {
              duration_ms: numberValue(data.durationMs),
              name: textValue(data.tool),
              permitted: data.permitted !== false,
              return_value: textValue(data.resultPreview),
              tool_call_id: textValue(data.toolCallId),
            },
          },
        },
      }
    case 'cancelled':
      // Cancellation is decided synchronously while the runner turn is still
      // unwinding, and the matching `done` can therefore land after the
      // parent turn stopped listening. Publish the terminal transition now so
      // no surface is left asserting a child still runs — or that a child
      // stopped when nothing ever told it to.
      return {
        type: 'subagent_event',
        payload: {
          ...base,
          event: {
            type: 'turn_end',
            payload: {
              status: 'cancelled',
              summary: event.completionSummary ?? textValue(data.reason),
              tool_count: event.toolCalls,
            },
          },
        },
      }
    case 'done':
      return {
        type: 'subagent_event',
        payload: {
          ...base,
          event: {
            type: 'turn_end',
            payload: {
              status: textValue(data.status) || 'completed',
              summary: textValue(data.resultPreview),
              tool_count: numberValue(data.toolCalls),
            },
          },
        },
      }
    case 'coordination':
      return {
        type: 'subagent_event',
        payload: { ...base, event: { type: 'text_part', payload: { text: `re-reading ${textValue(data.path)}` } } },
      }
    default:
      return undefined
  }
}

function nativeRuleLabels(permissionMode: PermissionMode, isolation: string): readonly string[] {
  return Object.freeze([
    `permission:${permissionMode}`,
    'delegation:blocked',
    ...(isolation ? [`isolation:${isolation}`] : []),
  ])
}

function resolveDefinition(
  definitions: ReadonlyMap<string, AgentDefinition>,
  requested: string,
): AgentDefinition | undefined {
  return definitions.get(requested) ?? definitions.get(canonicalProfileAlias(requested) ?? '')
}

function canonicalProfileAlias(requested: string): string | undefined {
  if (requested === 'general-purpose' || requested === 'general') return 'coder'
  if (requested === 'explore') return 'researcher'
  return undefined
}

function visibleDefinitionNames(definitions: ReadonlyMap<string, AgentDefinition>): string[] {
  return [...definitions.keys()].filter(name => !name.startsWith('@catalog:')).sort()
}

function spawnedStatus(task: SubAgentTask): SpawnedAgentStatus {
  switch (task.status) {
    case 'pending': return 'idle'
    case 'running': return 'running'
    case 'completed': return 'completed'
    case 'cancelled': return 'cancelled'
    case 'failed': return 'error'
  }
}

function permissionModeConfig(value: unknown, fallback: PermissionMode): PermissionMode {
  return value === 'accept-all' || value === 'auto' || value === 'manual' || value === 'plan' ? value : fallback
}

function permissionModeFromRules(rules: readonly string[]): PermissionMode {
  const configured = rules.find(rule => rule.startsWith('permission:'))?.slice('permission:'.length)
  return permissionModeConfig(configured, 'manual')
}

function agentDefinitionsFingerprint(definitions: ReadonlyMap<string, AgentDefinition>): string {
  return JSON.stringify([...definitions.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, definition]) => ({
      key,
      name: definition.name,
      description: definition.description,
      systemPrompt: definition.systemPrompt,
      model: definition.model,
      source: definition.source,
      tools: definition.tools,
      allowedTools: definition.allowedTools,
      excludeTools: definition.excludeTools,
      maxDepth: definition.maxDepth,
      isolation: definition.isolation,
      subagents: Object.entries(definition.subagents ?? {})
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([name, spec]) => ({
          name,
          description: spec.description,
          path: spec.path,
          resolvedProfile: spec.resolvedProfile,
        })),
    })))
}

/**
 * Compare the effective unattended child policies, not their UI labels.
 * Delegated manual prompts have no interactive broker and are rejected, while
 * plan admits safe read-only tools; auto adds the bounded automatic surface.
 */
function delegatedPermissionExceeds(candidate: PermissionMode, ceiling: PermissionMode): boolean {
  if (candidate === ceiling || ceiling === 'accept-all') return false
  if (ceiling === 'manual') return candidate !== 'manual'
  if (ceiling === 'plan') return candidate === 'auto' || candidate === 'accept-all'
  return candidate === 'accept-all'
}

function nativeHostGeneration(config: Readonly<Record<string, unknown>>): number | undefined {
  const value = config._nativeSubagentHostGeneration
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined
}

function stringConfig(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : []
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function textValue(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function numberValue(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0
}

function latestAssistantText(messages: readonly { readonly content: unknown; readonly role: string }[]): string {
  const message = messages.slice().reverse().find(candidate => candidate.role === 'assistant')
  return typeof message?.content === 'string' ? message.content : ''
}

/** Serializable v35 wire view of a retried subagent for the `subagent.retry` response. */
export function subagentRetryWirePayload(snapshot: SpawnedAgentSnapshot): Record<string, unknown> {
  return {
    id: snapshot.id,
    name: snapshot.name,
    title: snapshot.title,
    status: snapshot.status,
    prompt_profile: snapshot.promptProfile,
    closed: snapshot.closed,
    updated_at: snapshot.updatedAt,
    ...(snapshot.historySessionId ? { history_session_id: snapshot.historySessionId } : {}),
    ...(snapshot.error ? { error: snapshot.error } : {}),
    ...(snapshot.model ? { model: snapshot.model } : {}),
    ...(snapshot.sourceAgentId ? { source_agent_id: snapshot.sourceAgentId } : {}),
    ...(snapshot.creatorAgentId ? { creator_agent_id: snapshot.creatorAgentId } : {}),
    ...(snapshot.parentAgentId ? { parent_agent_id: snapshot.parentAgentId } : {}),
  }
}
