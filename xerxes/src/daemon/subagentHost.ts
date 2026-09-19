// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { recordCompaction } from '../context/compactionHistory.js'
import { isAbsolute, resolve } from 'node:path'
import { parseWorktreeRef, parseWorktreeSource } from '../agents/worktreeOptions.js'
import { subagentCatalogForAgent, type AgentDefinition } from '../agents/definitions.js'
import {
  SUBAGENT_BLOCKED_TOOLS,
  SubAgentManager,
  type SubAgentEvent,
  type SubAgentTask,
  type SubagentTaskRunRequest,
  type SubagentWorktreePort,
  type SubagentWorktree,
} from '../agents/subagentManager.js'
import type { ContextMessage } from '../context/compressor.js'
import { ValidationError } from '../core/errors.js'
import type { ToolExecutor } from '../executors/toolRegistry.js'
import type { LlmClient } from '../llms/client.js'
import { withCapturedModelCallScopes, type ModelCallScope } from '../llms/callBudget.js'
import { effectiveContextLimit } from '../llms/providerRegistry.js'
import type {
  SendAgentInputOptions,
  SpawnAgentOptions,
  SpawnedAgentManagerPort,
  SpawnedAgentSnapshot,
  SpawnedAgentStatus,
} from '../operators/subagents.js'
import { bootstrap, bootstrapSubagentsForAgent } from '../runtime/bootstrap.js'
import { skillPromptSection, type SkillRegistry } from '../extensions/skills.js'
import type { RunHistory } from '../runtime/runHistory.js'
import { runWithActiveSession } from '../runtime/sessionContext.js'
import { looksLikeSessionId, type DaemonTranscriptStore } from '../session/daemonTranscript.js'
import { FILE_READS_METADATA_KEY, fileStateTracker } from '../tools/fileState.js'
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
import { resolveOwnedSubagentRetry } from './subagentRetryOwnership.js'
import type { DurableTaskBridge } from '../tasks/durableTaskBridge.js'

export type SourceProviderClient = Pick<NativeSubagentHostOptions, 'llm' | 'contextLimit' | 'maxTokens' | 'maxOutputTokens' | 'temperature' | 'topK' | 'topP'> & { readonly route: string }

export interface NativeSubagentHostOptions {
  /** Reconstruct original durable budget ownership before a recovered attempt starts. */
  readonly restoreModelCallScopes?: (snapshot: SpawnedAgentSnapshot) => readonly ModelCallScope[]
  readonly agentDefinitions: ReadonlyMap<string, AgentDefinition>
  /**
   * Fraction of a child's prompt budget at which its conversation is compacted
   * before the turn starts. Defaults to the daemon's own auto-compaction
   * threshold so parent and children never disagree about when a context is
   * full; 0 disables child compaction.
   */
  readonly autoCompactThreshold?: number
  /** Live provider/profile capability lookup; absence disables speculative compaction. */
  readonly contextLimit?: (model: string) => number | undefined
  readonly cwd: string
  /** Resolve the owning session's project root for a new or recovered child. */
  readonly resolveSourceWorkspace?: (sourceId: string) => string
  /** Host-owned source routing, checked before allocation and execution. Never persist the client. */
  readonly resolveSourceClient?: (sourceId: string, model: string, explicitProfile?: string) => SourceProviderClient | undefined
  readonly resolveSourceProvider?: (sourceId: string, model: string) => string | undefined
  /** Route identity captured for inherited children and checked on recovery. */
  readonly inheritedProviderRoute?: string
  /** Resolve the current route identity for an explicit provider profile. */
  readonly resolveProviderRoute?: (profile: string, model: string) => string
  /** Explicit isolation adapter, fixed for this host's lifetime. */
  readonly worktree?: SubagentWorktreePort
  /** Immutable factory; each allocation uses its captured host generation cwd. */
  readonly worktreeForWorkspace?: (cwd: string) => SubagentWorktreePort
  readonly runHistory?: RunHistory
  /**
   * Durable record of subagent attempts, so a crash mid-fan-out leaves a
   * readable account of what ran. Optional: hosts that do not want the sidecar
   * simply omit it and the manager records nothing, as before.
   */
  readonly durableTaskBridge?: DurableTaskBridge
  readonly eventBus: DaemonSubagentEventBus
  /** Bounded supplemental bootstrap context, such as the discovered skill catalog. */
  readonly extraContext?: string
  readonly skillRegistry?: SkillRegistry
  readonly validateProviderSelection?: (profile: string, model: string, effort?: string, signal?: AbortSignal) => Promise<void>
  readonly validateInheritedSelection?: (model: string, effort?: string, signal?: AbortSignal) => Promise<void>
  readonly resolveProviderProfile?: (profile: string, model: string, expectedRoute?: string) => Pick<NativeSubagentHostOptions, 'llm' | 'contextLimit' | 'maxTokens' | 'maxOutputTokens' | 'temperature' | 'topK' | 'topP'>
  readonly llm: LlmClient
  /** Explicit runtime/profile request cap. */
  readonly maxTokens?: number
  /** Per-child model fallback used when maxTokens is not configured. */
  readonly maxOutputTokens?: (model: string) => number | undefined
  readonly model: string
  readonly reasoningEffort?: string
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
  /** Host-verified requesting session. When present, resolve task names within this owner only. */
  readonly sourceAgentId?: string
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
  if (options.worktree && options.worktreeForWorkspace) throw new Error('Configure either worktree or worktreeForWorkspace')
  const taskGenerations = new Map<string, number>()
  const owners = new Map<string, { tree: SubagentWorktree; port: SubagentWorktreePort }>()
  const owner = (tree: SubagentWorktree) => {
    const found = owners.get(tree.path)
    if (!found || found.tree.branch !== tree.branch) throw new Error('Unknown worktree allocation owner')
    return found.port
  }
  const routedWorktrees: SubagentWorktreePort | undefined = options.worktreeForWorkspace ? {
    async create(request) {
      const generation = nativeHostGeneration(request.config ?? {})
      const configuration = generation === undefined ? activeOptions : generationOptions.get(generation)
      if (!configuration) throw new Error('Worktree allocation generation is no longer available')
      taskGenerations.set(request.taskId, generation ?? activeGeneration)
      const workspace = workspaceFromConfig(request.config) ?? configuration.cwd
      const port = options.worktreeForWorkspace!(workspace)
      const tree = await port.create(request)
      owners.set(tree.path, { tree, port })
      return tree
    },
    isClean: tree => owner(tree).isClean(tree),
    async remove(tree) { await owner(tree).remove(tree); owners.delete(tree.path) },
  } : options.worktree
  const conversations = new SubagentConversationPersistence(options.transcriptStore)
  const historySessionIds = new Map<string, string>()
  /** Child depth advertised by the manager for each in-flight run, keyed by the unique running task id. */
  const runningChildDepths = new Map<string, { readonly childDepth: number, readonly taskId: string }>()
  const manager = new SubAgentManager({
    ...(routedWorktrees ? { worktree: routedWorktrees } : {}),
    idFactory: () => {
      const taskId = `subagent_${crypto.randomUUID().replaceAll('-', '').slice(0, 12)}`
      if (options.transcriptStore) {
        historySessionIds.set(taskId, crypto.randomUUID().replaceAll('-', ''))
      }
      return taskId
    },
    ...(options.durableTaskBridge === undefined ? {} : { durableTaskBridge: options.durableTaskBridge }),
    onEvent: event => publishSubagentEvent(options.eventBus, event, historySessionIds.get(event.taskId)),
    pathResolver: rawPath => rawPath,
    runner: async request => {
      const generation = nativeHostGeneration(request.config) ?? taskGenerations.get(request.task.id)
      taskGenerations.set(request.task.id, generation ?? activeGeneration)
      const configuration = generation === undefined ? activeOptions : generationOptions.get(generation)
      if (!configuration) throw new Error('Agent execution generation is no longer available; refusing a different workspace')
      const workspace = workspaceFromConfig(request.config) ?? configuration.cwd
      const executionConfiguration = workspace === configuration.cwd ? configuration : { ...configuration, cwd: workspace }
      const history = executionConfiguration.runHistory
      const run = request.task.sourceId ? history?.start({
        ownerSessionId: request.task.sourceId,
        workspace: request.worktree?.path ?? workspace,
        kind: 'agent', sourceId: request.task.id,
        title: `${request.task.title || request.task.name || request.prompt} · attempt ${request.task.attempt + 1}`,
      }) : undefined
      try {
        const output = await runWithActiveSession({ cwd: request.worktree?.path ?? workspace }, () => runNativeSubagent(
        request,
        executionConfiguration,
        conversations,
        historySessionIds.get(request.task.id),
        runningChildDepths,
      ))
        request.cancelSignal.throwIfAborted()
        if (!output.content.trim()) throw new Error('Subagent completed without a final response')
        if (run) history?.finish(run.ownerSessionId, run.id, 'succeeded', { output: output.content })
        return output
      } catch (error) {
        if (run) history?.finish(run.ownerSessionId, run.id, request.cancelSignal.aborted ? 'cancelled' : 'failed', {
          error: errorText(error),
          ...(error instanceof IncompleteSubagentTurnError ? { output: error.partialOutput } : {}),
        })
        throw error
      }
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
  /** Keep execution generations for retained tasks, including terminal tasks that can be retried. */
  const pruneGenerationOptions = (): void => {
    const live = liveManagerPort.liveGenerations()
    const retained = new Set(manager.listTasks().map(task => task.id))
    // Archived (LRU-evicted) terminal tasks stay retryable through
    // listRetryTasks()/rebuildArchivedTask even though they are gone from
    // listTasks(). Without pinning their generations here, the next
    // reconfigure drops the generation and the retry fails permanently with
    // "Agent execution generation is no longer available" — and the failed
    // retry overwrites the archived task's recorded result with that error.
    for (const retryable of manager.listRetryTasks()) retained.add(retryable.id)
    for (const [id, generation] of taskGenerations) {
      if (retained.has(id)) live.add(generation)
      else taskGenerations.delete(id)
    }
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
      if (nextOptions.worktreeForWorkspace !== options.worktreeForWorkspace || nextOptions.worktree !== options.worktree || (options.worktree && nextOptions.cwd !== options.cwd)) {
        throw new Error('A native subagent host cannot change its worktree owner; create a separate host')
      }
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
  readonly providerProfile?: string
  readonly reasoningEffort?: string
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
  private restoreModelCallScopes: NativeSubagentHostOptions['restoreModelCallScopes']
  private validateProviderSelection: NativeSubagentHostOptions['validateProviderSelection']
  private validateInheritedSelection: NativeSubagentHostOptions['validateInheritedSelection']
  private resolveSourceWorkspace: NativeSubagentHostOptions['resolveSourceWorkspace']
  private resolveSourceClient: NativeSubagentHostOptions['resolveSourceClient']
  private resolveSourceProvider: NativeSubagentHostOptions['resolveSourceProvider']
  private inheritedProviderRoute: NativeSubagentHostOptions['inheritedProviderRoute']
  private resolveProviderRoute: NativeSubagentHostOptions['resolveProviderRoute']
  private readonly hasWorkspaceWorktreeFactory: boolean
  private availableTools: readonly ToolDefinition[]
  private cwd: string
  private definitions: ReadonlyMap<string, AgentDefinition>
  private fallbackModel: string
  private fallbackEffort: string | undefined
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
    this.restoreModelCallScopes = options.restoreModelCallScopes
    this.validateProviderSelection = options.validateProviderSelection
    this.validateInheritedSelection = options.validateInheritedSelection
    this.resolveSourceClient = options.resolveSourceClient
    this.resolveSourceProvider = options.resolveSourceProvider
    this.resolveSourceWorkspace = options.resolveSourceWorkspace
    this.inheritedProviderRoute = options.inheritedProviderRoute
    this.resolveProviderRoute = options.resolveProviderRoute
    this.hasWorkspaceWorktreeFactory = options.worktreeForWorkspace !== undefined
    this.availableTools = options.tools
    this.cwd = options.cwd
    this.definitions = options.agentDefinitions
    this.fallbackModel = options.model
    this.fallbackEffort = options.reasoningEffort
    this.fallbackPermissionMode = options.permissionMode
    this.generation = generation
    this.transcripts = options.transcriptStore
  }

  reconfigure(options: NativeSubagentHostOptions, generation: number): void {
    this.restoreModelCallScopes = options.restoreModelCallScopes
    this.validateProviderSelection = options.validateProviderSelection
    this.validateInheritedSelection = options.validateInheritedSelection
    this.resolveSourceClient = options.resolveSourceClient
    this.resolveSourceProvider = options.resolveSourceProvider
    this.resolveSourceWorkspace = options.resolveSourceWorkspace
    this.inheritedProviderRoute = options.inheritedProviderRoute
    this.resolveProviderRoute = options.resolveProviderRoute
    this.availableTools = options.tools
    this.cwd = options.cwd
    this.definitions = options.agentDefinitions
    this.fallbackModel = options.model
    this.fallbackEffort = options.reasoningEffort
    this.fallbackPermissionMode = options.permissionMode
    this.generation = generation
    this.transcripts = options.transcriptStore
  }

  listHandles(): SpawnedAgentSnapshot[] {
    return this.manager.listTasks().map(task => this.snapshot(task))
  }

  listRetryTasks(): ReturnType<SubAgentManager['listRetryTasks']> {
    return this.manager.listRetryTasks()
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
    options.signal?.throwIfAborted()
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
    const inheritedMode = permissionModeConfig(options.permissionMode, this.fallbackPermissionMode)
    const fileMode = agentPermissionMode(definition, inheritedMode)
    const requestedPermissionMode = delegatedPermissionExceeds(fileMode, inheritedMode) ? inheritedMode : fileMode
    const permissionMode = delegatedPermissionExceeds(requestedPermissionMode, this.fallbackPermissionMode)
      ? this.fallbackPermissionMode
      : requestedPermissionMode
    const workspace = this.resolveWorkspace(options.sourceAgentId)
    const sourceClient = options.sourceAgentId ? this.resolveSourceClient?.(options.sourceAgentId, model, options.agent?.providerProfile) : undefined
    const providerProfile = sourceClient ? undefined : options.agent?.providerProfile ?? (options.sourceAgentId ? this.resolveSourceProvider?.(options.sourceAgentId, model) : undefined)
    const providerRoute = this.captureProviderRoute(providerProfile, model, options.sourceAgentId)
    const reasoningEffort = options.agent?.reasoningEffort ?? definition.effort ?? (sourceClient ? undefined : this.fallbackEffort)
    const task = await this.spawnResolved({
      ...(options.signal ? { signal: options.signal } : {}),
      definition,
      ...(options.isolation ? { isolation: options.isolation } : {}),
      ...(options.worktreeRef === undefined ? {} : { worktreeRef: options.worktreeRef }),
      ...(options.worktreeSource === undefined ? {} : { worktreeSource: options.worktreeSource }),
      input: prompt,
      ...(providerProfile ? { providerProfile } : {}),
      ...(reasoningEffort ? { reasoningEffort } : {}),
      model,
      workspace,
      ...(providerRoute === undefined ? {} : { providerRoute }),
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
    readonly signal?: AbortSignal
    readonly isolation?: 'worktree'
    readonly worktreeRef?: string
    readonly worktreeSource?: 'working-tree'
    readonly providerProfile?: string
    readonly reasoningEffort?: string
    readonly creatorAgentId?: string
    readonly definition: AgentDefinition
    readonly historySessionId?: string
    readonly input: string
    readonly model: string
    readonly workspace: string
    readonly providerRoute?: string
    readonly name?: string
    readonly parentAgentId?: string
    readonly permissionMode: PermissionMode
    readonly sourceAgentId?: string
    readonly taskId?: string
    readonly title?: string
  }): Promise<SubAgentTask> {
    const { definition, model, permissionMode } = resolved
    if ((resolved.isolation ?? definition.isolation) === 'worktree'
      && resolved.workspace !== normalizeWorkspace(this.cwd, 'host workspace') && !this.hasWorkspaceWorktreeFactory) {
      throw new ValidationError('workspace', 'alternate workspaces require worktreeForWorkspace', resolved.workspace)
    }
    this.assertProviderRoute(resolved.providerProfile, model, resolved.providerRoute, resolved.sourceAgentId)
    const generation = this.generation
    resolved.signal?.throwIfAborted()
    const sourceClient = resolved.sourceAgentId ? this.resolveSourceClient?.(resolved.sourceAgentId, model, resolved.providerProfile) : undefined
    if (!sourceClient) {
      if (resolved.providerProfile) await this.validateProviderSelection?.(resolved.providerProfile, model, resolved.reasoningEffort, resolved.signal)
      else await this.validateInheritedSelection?.(model, resolved.reasoningEffort, resolved.signal)
    }
    resolved.signal?.throwIfAborted()
    if (generation !== this.generation) throw new Error('Agent host changed during selection validation; retry the spawn')
    this.assertProviderRoute(resolved.providerProfile, model, resolved.providerRoute, resolved.sourceAgentId)
    const worktreeRef = parseWorktreeRef(resolved.worktreeRef)
    const worktreeSource = parseWorktreeSource(resolved.worktreeSource)
    if (worktreeSource && (worktreeRef || (resolved.isolation ?? definition.isolation) !== 'worktree')) throw new ValidationError('worktree_source', 'requires isolation=worktree and no worktree_ref', worktreeSource)
    if (worktreeRef && (resolved.isolation ?? definition.isolation) !== 'worktree') throw new ValidationError('worktree_ref', 'requires isolation=worktree', worktreeRef)
    const config = {
      ...(resolved.providerProfile ? { providerProfile: resolved.providerProfile } : {}),
      ...(resolved.reasoningEffort ? { reasoningEffort: resolved.reasoningEffort } : {}),
      model,
      _nativeSubagentWorkspace: resolved.workspace,
      ...(resolved.providerRoute === undefined ? {} : { _nativeSubagentProviderRoute: resolved.providerRoute }),
      permissionMode,
      _nativeSubagentHostGeneration: generation,
      ...(worktreeRef ? { _nativeSubagentWorktreeRef: worktreeRef } : {}),
      ...(worktreeSource ? { _nativeSubagentWorktreeSource: worktreeSource } : {}),
      ...(definition.allowedTools === null
        ? {}
        : { _toolsAllowed: [...definition.allowedTools] }),
      ...(definition.excludeTools.length ? { _toolsExcluded: [...definition.excludeTools] } : {}),
      ...(definition.tools.length ? { _toolsWhitelist: [...definition.tools] } : {}),
      ...(definition.promptMode ? { _agentPromptMode: definition.promptMode } : {}),
      ...(definition.skills?.length ? { _agentSkills: [...definition.skills] } : {}),
      ...(definition.maxTurns === undefined ? {} : { _agentMaxTurns: definition.maxTurns }),
    }
    const toolsets = subagentTools(this.availableTools, config).map(tool => tool.function.name)
    const isolation = resolved.isolation ?? definition.isolation
    const rules = [...nativeRuleLabels(permissionMode, isolation), ...(definition.background ? ['background'] : []), ...(worktreeRef ? ['worktree-ref:' + worktreeRef] : []), ...(worktreeSource ? ['worktree-source:' + worktreeSource] : [])]
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
      isolation,
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
      ...(resolved.providerProfile ? { providerProfile: resolved.providerProfile } : {}),
      ...(resolved.reasoningEffort ? { reasoningEffort: resolved.reasoningEffort } : {}),
      closed: false,
      createdAt: new Date().toISOString(),
      creatorAgentId: resolved.creatorAgentId,
      generation,
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
    const taskRecord = this.manager.listRetryTasks().find(candidate => candidate.id === info.id)
    const workspace = taskRecord?.workspace ?? this.resolveWorkspace(taskRecord?.sourceAgentId)
    const input = options.message?.trim()
      || await this.continuationInput(historySessionId, metadata?.lastInput ?? '', workspace)
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
  async respawnRecovered(snapshot: SpawnedAgentSnapshot, input: string, mode: 'retry' | 'reset' = 'retry'): Promise<SpawnedAgentSnapshot> {
    // Unknown serialized scopes cannot become unrestricted work on restart.
    if (!this.restoreModelCallScopes && snapshot.modelCallBindings?.length) {
      throw new ValidationError('task', 'Cannot restore delegated budget ownership in this host; dispatch new work from the parent session', snapshot.id)
    }
    const scopes = this.restoreModelCallScopes?.(snapshot) ?? []
    for (const scope of scopes) scope.assertAdmission()
    const definition = this.resolveChildDefinition(snapshot.creatorAgentId, snapshot.promptProfile)
    if (!definition) {
      throw new ValidationError(
        'subagent_type',
        `is not a registered agent profile; available profiles: ${visibleDefinitionNames(this.definitions).join(', ') || '(none)'}`,
        snapshot.promptProfile,
      )
    }
    const model = snapshot.model?.trim() || stringConfig(definition.model) || this.fallbackModel
    const workspace = this.resolveRecoveredWorkspace(snapshot)
    const providerRoute = providerRouteOf(snapshot)
    if (providerRoute === undefined && (this.inheritedProviderRoute !== undefined || this.resolveProviderRoute !== undefined || this.resolveSourceClient !== undefined)) {
      throw new ValidationError('provider_route', 'cannot recover a child without its original provider route; dispatch new work under the current route', snapshot.id)
    }
    this.assertProviderRoute(snapshot.providerProfile, model, providerRoute, snapshot.sourceAgentId)
    const requestedMode = snapshot.rules?.length ? permissionModeFromRules(snapshot.rules) : this.fallbackPermissionMode
    const permissionMode = delegatedPermissionExceeds(requestedMode, this.fallbackPermissionMode)
      ? this.fallbackPermissionMode
      : requestedMode
    const task = await withCapturedModelCallScopes(scopes, () => this.spawnResolved({
      definition,
      ...(snapshot.rules?.includes('worktree-source:working-tree') ? { worktreeSource: 'working-tree' as const } : {}),
      ...(snapshot.rules?.includes('isolation:worktree') ? { isolation: 'worktree' as const } : {}),
      ...(snapshot.rules?.find(rule => rule.startsWith('worktree-ref:')) ? { worktreeRef: snapshot.rules.find(rule => rule.startsWith('worktree-ref:'))!.slice('worktree-ref:'.length) } : {}),
      input,
      model,
      workspace,
      ...(providerRoute === undefined ? {} : { providerRoute }),
      permissionMode,
      ...(mode === 'retry' ? { taskId: snapshot.id } : {}),
      ...(snapshot.providerProfile ? { providerProfile: snapshot.providerProfile } : {}),
      ...(snapshot.reasoningEffort ? { reasoningEffort: snapshot.reasoningEffort } : {}),
      ...(mode === 'retry' && snapshot.historySessionId ? { historySessionId: snapshot.historySessionId } : {}),
      ...(snapshot.name ? { name: snapshot.name } : {}),
      ...(snapshot.title ? { title: snapshot.title } : {}),
      ...(snapshot.creatorAgentId ? { creatorAgentId: snapshot.creatorAgentId } : {}),
      ...(snapshot.parentAgentId ? { parentAgentId: snapshot.parentAgentId } : {}),
      ...(snapshot.sourceAgentId ? { sourceAgentId: snapshot.sourceAgentId } : {}),
    }))
    return this.snapshot(task)
  }

  /**
   * Choose the retry attempt's input: a continuation nudge when the task's
   * conversation persisted, otherwise the recorded original prompt so a task
   * that died before its first checkpoint still gets its instructions.
   */
  async continuationInput(historySessionId: string | undefined, fallbackInput: string, workspace = this.cwd): Promise<string> {
    const store = this.transcripts
    if (!historySessionId || !store) return fallbackInput
    try {
      const transcript = await store.load(historySessionId, { currentProjectDirectory: workspace })
      if (transcript && transcript.messages.length > 0) return SUBAGENT_RETRY_CONTINUATION_PROMPT
    } catch {
      // An unreadable history falls back to resubmitting the original prompt.
    }
    return fallbackInput
  }

  continuationInputForSnapshot(snapshot: SpawnedAgentSnapshot): Promise<string> {
    return this.continuationInput(snapshot.historySessionId, snapshot.lastInput ?? '', this.resolveRecoveredWorkspace(snapshot))
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
    // Children still inside their spawn-setup window have no registered
    // policy yet, and their task record does not carry one either — skipping
    // them here is pinned behavior (a project-switch reconfigure must not
    // kill an in-allocation child). The known tradeoff: a ceiling tightening
    // racing a spawn can miss that one child. The eviction path
    // (invalidateSource) does not extend the same courtesy.
    return this.invalidateMatching(metadata => metadata !== undefined && delegatedPermissionExceeds(metadata.permissionMode, nextMode))
  }

  /** Cancel and permanently close every handle owned by this host. */
  invalidateAll(): number {
    return this.invalidateMatching(() => true)
  }

  /** Cancel and permanently close children whose owning session is being removed. */
  invalidateSource(sourceAgentId: string): number {
    const source = sourceAgentId.trim()
    if (!source) return 0
    return this.invalidateMatching(
      // Same owner fallback interruptSource uses: a child whose handle
      // metadata was never registered still stops instead of quietly
      // outliving the session that owns it.
      (metadata, task) => (metadata?.sourceAgentId ?? (task.sourceId || undefined)) === source,
    )
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

  private invalidateMatching(
    predicate: (metadata: HandleMetadata | undefined, task: { id: string; sourceId?: string }) => boolean,
  ): number {
    let cancelled = 0
    for (const task of this.manager.listTasks()) {
      const metadata = this.handles.get(task.id)
      // The task registers with the manager before its handle metadata does
      // (durable-task bridge + worktree creation run inside manager.spawn),
      // so a metadata-only check used to skip children inside that window.
      // Predicates receive undefined metadata and decide.
      if (!predicate(metadata, task)) continue
      this.invalidatedHandles.add(task.id)
      this.pendingResume.delete(task.id)
      if (metadata) metadata.closed = true
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
    const catalog = subagentCatalogForAgent(this.definitions, creator)
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

  private resolveWorkspace(sourceId: string | undefined): string {
    const raw = sourceId?.trim() && this.resolveSourceWorkspace
      ? this.resolveSourceWorkspace(sourceId.trim())
      : this.cwd
    return normalizeWorkspace(raw, 'source workspace')
  }

  private captureProviderRoute(profile: string | undefined, model: string, sourceId?: string): string | undefined {
    const source = sourceId ? this.resolveSourceClient?.(sourceId, model, profile) : undefined
    const route = source?.route ?? (profile?.trim() ? this.resolveProviderRoute?.(profile.trim(), model) : this.inheritedProviderRoute)
    return route === undefined ? undefined : normalizeProviderRoute(route, 'provider route')
  }

  private assertProviderRoute(profile: string | undefined, model: string, expected: string | undefined, sourceId?: string): void {
    if (expected === undefined) return
    const current = this.captureProviderRoute(profile, model, sourceId)
    if (current === undefined || normalizeProviderRoute(current, 'provider route') !== expected) {
      throw new ValidationError('provider_route', 'provider route changed since this child was created; dispatch new work under the current route', { expected, current })
    }
  }

  private resolveRecoveredWorkspace(snapshot: SpawnedAgentSnapshot): string {
    if (snapshot.workspace !== undefined) {
      if (!snapshot.workspace) throw new ValidationError('workspace', 'persisted workspace is malformed; dispatch new work from the parent session', snapshot.workspace)
      return normalizeWorkspace(snapshot.workspace, 'persisted workspace')
    }
    return this.resolveWorkspace(snapshot.sourceAgentId)
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
      throw new ValidationError('handle_id', 'spawned agent is not accepting input; use AgentTool with resume and prompt for follow-up work, or TaskOutputTool with offset pagination to read saved output', task.id)
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
      ...(task.providerProfile === undefined ? {} : { providerProfile: task.providerProfile }),
      ...(task.reasoningEffort === undefined ? {} : { reasoningEffort: task.reasoningEffort }),
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
      modelCallBindings: task.modelCallBindings,
      ...(task.providerRoute === undefined ? {} : { providerRoute: task.providerRoute }),
      ...(task.workspace === undefined ? {} : { workspace: task.workspace }),
      attempt: task.attempt,
      closed: metadata.closed || status === 'closed',
      createdAt: metadata.createdAt,
      ...(task.error ? { error: task.error } : {}),
      ...(metadata.historySessionId ? { historySessionId: metadata.historySessionId } : {}),
      id: task.id,
      ...(metadata.providerProfile ? { providerProfile: metadata.providerProfile } : {}),
      ...(metadata.reasoningEffort ? { reasoningEffort: metadata.reasoningEffort } : {}),
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
    if (!this.pendingRestart.has(recovered.id)) {
      throw new ValidationError(
        'handle_id',
        'belongs to a task interrupted by a daemon restart; call ResetAgent to rerun it',
        recovered.id,
      )
    }
    const input = (options.message ?? options.taskDescription)?.trim()
    if (!input) throw new ValidationError('message', 'spawned agent input is required', input)
    this.pendingRestart.delete(recovered.id)
    const replacement = await this.live.respawnRecovered(recovered, input, 'reset').catch(error => {
      this.pendingRestart.add(recovered.id)
      throw error
    })
    this.recovered.delete(recovered.id)
    this.pendingRestart.delete(recovered.id)
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
    if (options.sourceAgentId !== undefined) {
      const identities = new Map([...this.recovered.values(), ...this.live.listRetryTasks()].map(task => [task.id, task]))
      handleId = resolveOwnedSubagentRetry([...identities.values()], handleId, options.sourceAgentId).id
    }
    if (this.findLive(handleId)) return this.live.retry(handleId, options)
    const recovered = this.findRecovered(handleId)
    if (!recovered) return this.live.retry(handleId, options)
    const input = options.message?.trim()
      || await this.live.continuationInputForSnapshot(recovered)
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
  const providerProfile = stringConfig(request.config.providerProfile)
  const expectedRoute = providerRouteFromConfig(request.config)
  const sourceClient = request.task.sourceId ? options.resolveSourceClient?.(request.task.sourceId, model, providerProfile || undefined) : undefined
  const currentRoute = sourceClient?.route ?? (providerProfile
    ? options.resolveProviderRoute?.(providerProfile, model)
    : options.inheritedProviderRoute)
  if (expectedRoute !== undefined && currentRoute !== expectedRoute) {
    throw new ValidationError('provider_route', 'provider route changed before child execution; dispatch new work under the current route', { expected: expectedRoute, current: currentRoute })
  }
  request.cancelSignal.throwIfAborted()
  if (sourceClient) {
    const { maxTokens: _max, maxOutputTokens: _output, contextLimit: _context, temperature: _temperature, topK: _topK, topP: _topP, ...shared } = options
    options = { ...shared, ...sourceClient }
  }
  if (!sourceClient && !providerProfile) {
    request.cancelSignal.throwIfAborted()
    await options.validateInheritedSelection?.(model, stringConfig(request.config.reasoningEffort) || undefined, request.cancelSignal)
    request.cancelSignal.throwIfAborted()
    const afterValidation = options.inheritedProviderRoute
    if (expectedRoute !== undefined && afterValidation !== expectedRoute) {
      throw new ValidationError('provider_route', 'provider route changed before child execution; dispatch new work under the current route', { expected: expectedRoute, current: afterValidation })
    }
  }
  if (!sourceClient && providerProfile) {
    request.cancelSignal.throwIfAborted()
    await options.validateProviderSelection?.(providerProfile, model, stringConfig(request.config.reasoningEffort) || undefined, request.cancelSignal)
    request.cancelSignal.throwIfAborted()
    const afterValidation = options.resolveProviderRoute?.(providerProfile, model)
    if (expectedRoute !== undefined && afterValidation !== expectedRoute) {
      throw new ValidationError('provider_route', 'provider route changed before child execution; dispatch new work under the current route', { expected: expectedRoute, current: afterValidation })
    }
    if (!options.resolveProviderProfile) throw new Error('This host cannot route per-agent provider profiles')
    const selected = options.resolveProviderProfile(providerProfile, model, expectedRoute)
    const { maxTokens: _max, maxOutputTokens: _output, contextLimit: _context, temperature: _temperature, topK: _topK, topP: _topP, ...shared } = options
    options = { ...shared, ...selected }
  }
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
      const preloadedSkills = stringList(request.config._agentSkills).map(name => {
        const skill = options.skillRegistry?.get(name)
        if (!skill) throw new ValidationError('skills', `preloaded skill '${name}' is unavailable; inspect /skills diagnostics`, name)
        return skillPromptSection(skill)
      })
      const boot = await bootstrap({
        ...(request.config._agentPromptMode === 'replace' ? { baseSystemPrompt: request.systemPrompt } : {}),
        cwd,
        subagents: bootstrapSubagentsForAgent(options.agentDefinitions, request.task.agentDefName),
        ...(options.extraContext ? { extraContext: options.extraContext } : {}),
        model,
        tools,
      })
      const maxTokens = options.maxTokens ?? options.maxOutputTokens?.(model)
      const events = runTurn({
        agentId: request.task.agentDefName || request.task.id,
        ...(maxTokens === undefined ? {} : { maxTokens }),
        ...(typeof request.config._agentMaxTurns === 'number' ? { maxModelTurns: request.config._agentMaxTurns } : {}),
        model,
        ...(stringConfig(request.config.reasoningEffort) ? { thinking: { effort: stringConfig(request.config.reasoningEffort) } } : {}),
        permissionMode,
        sessionId: conversation.historySessionId,
        state,
        systemPrompt: [boot.systemPrompt, ...(request.config._agentPromptMode === 'replace' ? [] : [request.systemPrompt]), ...preloadedSkills].filter(Boolean).join('\n\n'),
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
            event.reason && event.reason !== 'completed' && event.reason !== 'objective_verified'
          )) {
            terminalFailure = new Error(
              event.reason === 'tool_budget_exhausted' && typeof request.config._agentMaxTurns === 'number'
                ? `Subagent reached maxTurns=${request.config._agentMaxTurns}. Output is partial; resume this agent to continue.`
                : event.reason === 'context_overflow'
                ? 'Subagent provider context window was exhausted'
                : event.reason === 'provider_failed'
                  ? sourceClient
                    ? 'Subagent local provider request failed. Check the parent session local connection and authorization before retrying.'
                    : 'Subagent provider request failed'
                  : `Subagent stopped before completion: ${event.reason}`,
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
        if (terminalFailure) throw new IncompleteSubagentTurnError(terminalFailure.message, output)
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
  // Unknown provider capacity disables speculative child compaction rather
  // than enforcing a model window Xerxes invented locally.
  const contextLimit = options.contextLimit?.(model)
  const maxTokens = options.maxTokens ?? options.maxOutputTokens?.(model)
  const thresholdTokens = compactionThresholdTokens(
    effectiveContextLimit({
      ...(contextLimit === undefined ? {} : { contextLimit }),
      ...(maxTokens === undefined ? {} : { requestedOutputTokens: maxTokens }),
    }),
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
    state.metadata = { ...state.metadata }
    recordCompaction(state.metadata, outcome.stamp)
    // Same rule as the main session's compaction: the summary no longer
    // carries full file contents, so the child must re-read before editing.
    fileStateTracker.clearSession(conversation.historySessionId)
    state.metadata[FILE_READS_METADATA_KEY] = []
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
    return !Array.isArray(config._toolsAllowed) || allowed.includes(name)
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
    ...(event.providerProfile ? { provider_profile: event.providerProfile } : {}),
    ...(event.reasoningEffort ? { reasoning_effort: event.reasoningEffort } : {}),
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

function agentPermissionMode(definition: AgentDefinition, inherited: PermissionMode): PermissionMode {
  switch (definition.permissionMode) {
    case 'acceptEdits': case 'auto': return 'auto'
    case 'dontAsk': case 'manual': return 'manual'
    case 'plan': return 'plan'
    case 'bypassPermissions': return 'accept-all'
    default: return inherited
  }
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
      promptMode: definition.promptMode,
      skills: definition.skills,
      maxTurns: definition.maxTurns,
      effort: definition.effort,
      background: definition.background,
      permissionMode: definition.permissionMode,
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

class IncompleteSubagentTurnError extends Error {
  constructor(message: string, readonly partialOutput: string) {
    super(message)
    this.name = 'IncompleteSubagentTurnError'
  }
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
  const providerRoute = providerRouteOf(snapshot)
  return {
    ...(snapshot.modelCallBindings === undefined ? {} : { model_call_bindings: snapshot.modelCallBindings }),
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
    ...(snapshot.providerProfile ? { provider_profile: snapshot.providerProfile } : {}),
    ...(snapshot.reasoningEffort ? { reasoning_effort: snapshot.reasoningEffort } : {}),
    ...(snapshot.workspace === undefined ? {} : { workspace: snapshot.workspace }),
    ...(providerRoute === undefined ? {} : { provider_route: providerRoute }),
    ...(snapshot.sourceAgentId ? { source_agent_id: snapshot.sourceAgentId } : {}),
    ...(snapshot.creatorAgentId ? { creator_agent_id: snapshot.creatorAgentId } : {}),
    ...(snapshot.parentAgentId ? { parent_agent_id: snapshot.parentAgentId } : {}),
  }
}

function workspaceFromConfig(config: Readonly<Record<string, unknown>> | undefined): string | undefined {
  if (!config || !Object.hasOwn(config, '_nativeSubagentWorkspace')) return undefined
  const value = config._nativeSubagentWorkspace
  if (typeof value !== 'string' || !value) throw new ValidationError('workspace', 'native subagent workspace is malformed', value)
  return normalizeWorkspace(value, 'native subagent workspace')
}

function providerRouteFromConfig(config: Readonly<Record<string, unknown>> | undefined): string | undefined {
  if (!config || !Object.hasOwn(config, '_nativeSubagentProviderRoute')) return undefined
  const value = config._nativeSubagentProviderRoute
  return normalizeProviderRoute(value, 'native subagent provider route')
}

function providerRouteOf(snapshot: SpawnedAgentSnapshot): string | undefined {
  if (!Object.hasOwn(snapshot, 'providerRoute')) return undefined
  const value = snapshot.providerRoute
  return normalizeProviderRoute(value, 'persisted provider route')
}

function normalizeProviderRoute(value: unknown, field: string): string {
  if (typeof value !== 'string' || !/^[a-f0-9]{64}$/.test(value)) {
    throw new ValidationError('provider_route', `${field} is malformed; dispatch new work under the current route`, value)
  }
  return value
}

function normalizeWorkspace(value: unknown, field: string): string {
  if (typeof value !== 'string' || !value || value.trim() !== value || value.includes('\u0000') || !isAbsolute(value)) {
    throw new ValidationError(field, 'must be a non-empty absolute path without whitespace padding or NUL bytes', value)
  }
  return resolve(value)
}
