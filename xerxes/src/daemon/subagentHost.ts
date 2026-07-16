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
import { ValidationError } from '../core/errors.js'
import type { ToolExecutor } from '../executors/toolRegistry.js'
import type { LlmClient } from '../llms/client.js'
import type {
  SendAgentInputOptions,
  SpawnAgentOptions,
  SpawnedAgentManagerPort,
  SpawnedAgentSnapshot,
  SpawnedAgentStatus,
} from '../operators/subagents.js'
import { bootstrap } from '../runtime/bootstrap.js'
import type { DaemonTranscriptStore } from '../session/daemonTranscript.js'
import type { AgentState, StreamEvent } from '../streaming/events.js'
import { runTurn } from '../streaming/loop.js'
import type { PermissionBroker, PermissionMode } from '../streaming/permissions.js'
import type { ToolDefinition } from '../types/toolCalls.js'
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
  readonly cwd: string
  readonly eventBus: DaemonSubagentEventBus
  readonly llm: LlmClient
  readonly maxTokens?: number
  readonly model: string
  readonly permissionMode: PermissionMode
  readonly temperature?: number
  readonly toolExecutor: ToolExecutor
  readonly tools: readonly ToolDefinition[]
  readonly topP?: number
  /** Shared daemon store; omitted hosts keep child conversations in memory only. */
  readonly transcriptStore?: DaemonTranscriptStore
}

export interface NativeSubagentHost {
  readonly manager: SubAgentManager
  readonly managerPort: SpawnedAgentManagerPort
  readonly turnCoordinator: SubagentTurnCoordinator
  /** Cancel and invalidate every child owned by this host. */
  invalidateAll(): number
  /** Cancel and invalidate every child owned by one parent session. */
  cancelSource(sourceAgentId: string): number
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
  const manager = new SubAgentManager({
    idFactory: () => {
      const taskId = `subagent_${crypto.randomUUID().replaceAll('-', '').slice(0, 12)}`
      if (options.transcriptStore) {
        historySessionIds.set(taskId, crypto.randomUUID().replaceAll('-', ''))
      }
      return taskId
    },
    maxConcurrent: 8,
    maxDepth: 5,
    onEvent: event => publishSubagentEvent(options.eventBus, event, historySessionIds.get(event.taskId)),
    pathResolver: rawPath => rawPath,
    runner: request => {
      const generation = nativeHostGeneration(request.config)
      return runNativeSubagent(
        request,
        generation === undefined ? activeOptions : generationOptions.get(generation) ?? activeOptions,
        conversations,
        historySessionIds.get(request.task.id),
      )
    },
  })
  const liveManagerPort = new RichSubagentManagerPort(
    manager,
    options,
    activeGeneration,
    historySessionIds,
  )
  const managerPort = new RecoverableSubagentManagerPort(liveManagerPort)
  const turnCoordinator = new NativeSubagentTurnCoordinator(
    manager,
    () => managerPort.listHandles(),
    undefined,
    undefined,
    snapshots => managerPort.restoreSnapshots(snapshots),
  )
  return {
    manager,
    managerPort,
    turnCoordinator,
    invalidateAll: () => managerPort.invalidateAll(),
    cancelSource: sourceAgentId => managerPort.invalidateSource(sourceAgentId),
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
    },
  }
}

interface HandleMetadata {
  readonly agentId: string
  closed: boolean
  readonly createdAt: string
  readonly creatorAgentId: string | undefined
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
  private definitions: ReadonlyMap<string, AgentDefinition>
  private fallbackModel: string
  private fallbackPermissionMode: PermissionMode
  private generation: number
  private readonly handles = new Map<string, HandleMetadata>()
  private readonly invalidatedHandles = new Set<string>()
  private readonly pendingResume = new Set<string>()

  constructor(
    private readonly manager: SubAgentManager,
    options: NativeSubagentHostOptions,
    generation: number,
    private readonly historySessionIds: ReadonlyMap<string, string>,
  ) {
    this.availableTools = options.tools
    this.definitions = options.agentDefinitions
    this.fallbackModel = options.model
    this.fallbackPermissionMode = options.permissionMode
    this.generation = generation
  }

  reconfigure(options: NativeSubagentHostOptions, generation: number): void {
    this.availableTools = options.tools
    this.definitions = options.agentDefinitions
    this.fallbackModel = options.model
    this.fallbackPermissionMode = options.permissionMode
    this.generation = generation
  }

  listHandles(): SpawnedAgentSnapshot[] {
    return this.manager.listTasks().map(task => this.snapshot(task))
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
    const task = await this.manager.spawn({
      prompt,
      ...(options.title ? { title: options.title } : {}),
      ...(name ? { name } : {}),
      agentDefinition: definition,
      ...(options.sourceAgentId ? { sourceId: options.sourceAgentId } : {}),
      ...(options.creatorAgentId ? { creatorId: options.creatorAgentId } : {}),
      ...(options.parentAgentId ? { parentId: options.parentAgentId } : {}),
      model,
      rules,
      toolsets,
      config,
    })
    this.handles.set(task.id, {
      agentId: definition.name,
      closed: false,
      createdAt: new Date().toISOString(),
      creatorAgentId: options.creatorAgentId,
      historySessionId: this.historySessionIds.get(task.id),
      lastInput: prompt,
      parentAgentId: options.parentAgentId ?? options.creatorAgentId,
      permissionMode,
      promptProfile: definition.name,
      sourceAgentId: options.sourceAgentId,
    })
    return this.snapshot(task)
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
    return this.live.spawn(options)
  }

  async sendInput(
    handleId: string | undefined,
    options: SendAgentInputOptions,
  ): Promise<SpawnedAgentSnapshot> {
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
    const liveIds = new Set(this.live.listHandles().map(snapshot => snapshot.id))
    const active: string[] = []
    const archived: SpawnedAgentSnapshot[] = []
    for (const target of targets) {
      if (liveIds.has(target)) {
        active.push(target)
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
    const recovered = this.findRecovered(handleId)
    if (!recovered) return this.live.resume(handleId)
    this.pendingRestart.add(recovered.id)
    return Object.freeze({ ...recovered, closed: false, status: 'idle' })
  }

  close(handleId: string): SpawnedAgentSnapshot & { readonly previousStatus: SpawnedAgentStatus } {
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

  invalidateHandlesExceeding(nextMode: PermissionMode): number {
    return this.live.invalidateHandlesExceeding(nextMode)
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
      const boot = await bootstrap({ cwd, model, tools })
      const events = runTurn({
        agentId: request.task.agentDefName || request.task.id,
        ...(options.maxTokens === undefined ? {} : { maxTokens: options.maxTokens }),
        model,
        permissionMode,
        sessionId: conversation.historySessionId,
        state,
        systemPrompt: [boot.systemPrompt, request.systemPrompt].filter(Boolean).join('\n\n'),
        ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
        tools,
        ...(options.topP === undefined ? {} : { topP: options.topP }),
        userMessage: request.prompt,
      }, {
        llm: options.llm,
        ...(permissionBroker === undefined ? {} : { permissionBroker }),
        toolExecutor: options.toolExecutor,
      }, request.cancelSignal)
      const iterator = events[Symbol.asyncIterator]()
      const firstEventPromise = iterator.next()
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
      await conversations.save(
        conversation,
        state,
        request.cancelSignal.aborted ? 'cancelled' : 'completed',
      )
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
    releaseConversation()
  }
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
    case 'turn_done':
      request.report.usage({
        ...(event.apiCallsCount === undefined ? {} : { apiCalls: event.apiCallsCount }),
        model: event.model,
        toolCalls: event.toolCallsCount,
        ...(event.usageComplete ? {
          inputTokens: event.usage.inputTokens,
          outputTokens: event.usage.outputTokens,
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
