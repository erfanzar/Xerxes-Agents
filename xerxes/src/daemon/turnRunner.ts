// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { parseTodoList } from "../runtime/todoSnapshot.js";
import { getTurnOutcome, restoreTurnOutcome } from '../types/turnOutcome.js'
import { LOCAL_PROVIDER_BINDING, type RemoteProviderBindings } from './remoteProviderBindings.js'
import { LocalProviderRelayError } from '../security/localProviderRelay.js'
import { readContextControls, type ContextControls } from '../context/controls.js'
import type { AgentDefinition } from '../agents/definitions.js'
import type { AuditEmitter } from '../audit/emitter.js'
import { compressToolResult } from '../context/headroom.js'
import {
  assembleContextLayers,
  assembleTurnContext,
  layerDigests,
  recordAssemblyProvenance,
  renderTurnContext,
} from '../context/assembly.js'
import { ToolResultStorage } from '../context/toolResultStorage.js'
import { estimateContextTokens } from '../context/windowUsage.js'
import { ValidationError } from '../core/errors.js'
import { classify, ErrorKind } from '../runtime/errorClassifier.js'
import { instructionFileUpdateLayer } from '../runtime/instructionFreshness.js'
import { appendSkillSuggestion } from '../extensions/skillSuggestions.js'
import type { HookRunner } from '../extensions/hooks.js'
import {
  renderToolGuidance,
  type ToolExecutor,
  type ToolRegistry,
} from '../executors/toolRegistry.js'
import { renderMemoryChanges, type AgentMemory, type MemorySourceText } from '../memory/agentMemory.js'
import {
  mergePersistedSubagentSnapshots,
  persistedSubagentDeliveryValues,
  persistedSubagentSnapshotValues,
  replacePersistedSubagentDeliveries,
  SUBAGENT_DELIVERY_METADATA_KEY,
  SUBAGENT_SNAPSHOT_METADATA_KEY,
} from '../agents/subagentPersistence.js'
import { SUBAGENT_BLOCKED_TOOLS } from '../agents/subagentManager.js'
import type { AgentSelfMemory } from '../memory/agentSelfMemory.js'
import { makeTurnIndexerHook } from '../memory/turnIndexer.js'
import type { Memory } from '../memory/base.js'
import type { SpawnedAgentSnapshot } from '../operators/subagents.js'
import type { LlmClient } from '../llms/client.js'
import { DEFAULT_RETRY_POLICY, type ProviderOverrides, retryPolicyForModel } from '../llms/providerRegistry.js'
import { agentNameForMode, modeSwitchHint, normalizeInteractionMode } from '../runtime/interactionModes.js'
import { DEFAULT_MAX_GOAL_ROUNDS, GOAL_CHANGES_KEY, getGoal, type GoalView } from '../runtime/goalDomain.js'
import { DEFAULT_BLOCKED_AFTER_CONSECUTIVE_ROUNDS, goalPolicyPrompt } from '../runtime/goalTools.js'
import {
  mergeContextDeltas,
  renderContextDeltas,
  takeContextDeltas,
} from '../runtime/contextDeltas.js'
import { EditFeedback } from '../runtime/editFeedback.js'
import { withActiveSession } from '../runtime/sessionContext.js'
import { resolveTurnThinking } from '../runtime/thinkingLevels.js'
import { captureUserWorkflowMemory } from '../runtime/workflowMemory.js'
import { createAgentState, type AgentState, type StreamEvent, type ToolResult } from '../streaming/events.js'
import { runTurn, type ContextReducer } from '../streaming/loop.js'
import type { SystemPromptSegment } from '../streaming/promptCaching.js'
import { fileStateTracker } from '../tools/fileState.js'
import {
  DEFAULT_PERMISSION_MODE,
  type PermissionBroker,
  type PermissionMode,
  type ToolPolicy,
} from '../streaming/permissions.js'
import type { ChatMessage, MessageContent } from '../types/messages.js'
import { isHarnessOrigin, messageText, type HarnessOrigin } from '../types/messages.js'
import { imageUrlContentParts } from './images.js'
import type { RawMessage, TranscriptMessageJournalAppend } from '../session/daemonTranscript.js'
import type { ToolCall, ToolDefinition } from '../types/toolCalls.js'
import type { DaemonInteractionBoard, DaemonQuestion } from './interactions.js'
import type { DaemonEvent, DaemonSession, RuntimeToolInventoryEntry, TurnRunControls, TurnRunner } from './runtime.js'
import {
  recoverSubagentSnapshots,
  type SubagentTurnCoordinator,
} from './subagentCoordinator.js'
import type { DaemonSubagentEventSource } from './subagentEvents.js'

export interface AgentTurnRunnerOptions {
  readonly remoteProviderBindings?: RemoteProviderBindings
  /** Definitions loaded from built-in, user, and project agent specs. */
  readonly agentDefinitions?: ReadonlyMap<string, AgentDefinition>
  /** Optional project-aware persistent memory injected into session startup context. */
  readonly agentMemory?: (session: DaemonSession) => AgentMemory | undefined | Promise<AgentMemory | undefined>
  /** Optional per-agent self-knowledge injected into session startup context. */
  readonly agentSelfMemory?: (
    session: DaemonSession,
  ) => AgentSelfMemory | undefined | Promise<AgentSelfMemory | undefined>
  /** Native bootstrap prompt provider, cached per workspace/model/agent/tool surface. */
  readonly bootstrapSystemPrompt?: BootstrapSystemPromptProvider
  /** Optional structured audit sink fed from the canonical streaming events. */
  readonly auditEmitter?: AuditEmitter
  /**
   * Extension hook sink (plugins, user shell hooks). Without it the loop's
   * hook points never fire — the daemon used to run with this unset, which
   * made the entire hooks subsystem inert outside tests.
   */
  readonly hookRunner?: HookRunner
  readonly hookRunnerForSession?: (session: DaemonSession) => HookRunner
  /**
   * Fallback model chain (Claude Code parity): when a terminal overload-class
   * provider failure arrives before any content streams, the turn restarts
   * once on this model. Requires `createLlmForModel` to build its client.
   */
  readonly fallbackModel?: string
  /** Builds an LlmClient for a fallback model using the same credentials. */
  readonly createLlmForModel?: (model: string) => LlmClient
  /** Native daemon reply board for approvals and ask-user tool calls. */
  readonly interactions?: DaemonInteractionBoard
  /** Optional tier receiving completed assistant turns for recall on later work. */
  readonly memory?: Memory
  readonly memoryMinChars?: number
  readonly resolveSessionProvider?: (session: DaemonSession, model: string) => { llm: LlmClient; createLlmForModel?: (model: string) => LlmClient; providerOverrides?: ProviderOverrides; contextLimit?: number | undefined; maxOutputTokens?: (model: string) => number | undefined }
  readonly llm: LlmClient
  /** Provider-reported context capacity for this profile/model; absent means unknown. */
  readonly contextLimit?: number
  /** Explicit runtime/profile request cap; wins over model metadata. */
  readonly maxTokens?: number
  /** Per-turn model fallback used when maxTokens is not explicitly configured. */
  readonly maxOutputTokens?: (model: string) => number | undefined
  readonly model: string
  readonly permissionBroker?: PermissionBroker
  /**
   * The connection's provider identity, threaded from the active profile.
   *
   * `retryPolicyForModel` resolves a provider from the model id when given
   * nothing else, and an OpenRouter id like
   * `stealth/ox-alpha` carries a VENDOR before the slash, not a routing
   * prefix — so resolution threw `unknown provider prefix 'stealth'` on every
   * turn that used one. The active profile already knows the answer; this
   * carries it to the two helpers that would otherwise have to guess.
   */
  readonly providerOverrides?: ProviderOverrides
  readonly permissionMode?: PermissionMode
  readonly policy?: ToolPolicy
  /**
   * Relieve a mid-turn context overflow. The loop detects the overflow and can
   * retry the round, but owns no compaction policy — without this the turn can
   * only report the failure and stop.
   */
  readonly reduceContext?: ContextReducer
  readonly autoCompactThreshold?: () => number
  /** Per-session prompt state shared across runner rebuilds; see SessionPromptSnapshots. */
  readonly promptSnapshots?: SessionPromptSnapshots
  /**
   * The live tool registry, when the host owns one. Used only to resolve the
   * per-tool usage-policy sections that ride with the request's visible tool
   * surface; the runner never executes through it.
   */
  readonly toolRegistry?: ToolRegistry
  /**
   * Session default effort hint for reasoning APIs. This is only the base
   * layer of per-turn resolution: ultra mode and escalation keywords in the
   * prompt override it for that turn, so a value here never forces a
   * thinking directive on its own.
   */
  readonly reasoningEffort?: string
  /**
   * Session default for extended thinking. `false` keeps thinking off for
   * ordinary turns but does not block escalation: an ultra-mode session or
   * a keyword in the prompt still wins per turn.
   */
  readonly thinking?: boolean
  /**
   * Session default thinking token budget, consulted only when neither ultra
   * mode nor a prompt keyword supplies a per-turn directive.
   */
  readonly thinkingBudget?: number
  /** Session-scoped delegated-turn events rendered alongside the parent turn. */
  readonly subagentEvents?: DaemonSubagentEventSource
  /** Joins explicitly detached child work back into the creating parent turn. */
  readonly subagentCoordinator?: SubagentTurnCoordinator
  readonly toolExecutor?: ToolExecutor
  /** Per-tool execution axes, normally `registry.capabilities` bound to the tool registry. */
  readonly toolCapabilities?: (
    toolName: string,
    agentId?: string,
    args?: Readonly<Record<string, unknown>>,
  ) => {
    readonly concurrencySafe: boolean
    readonly interruptBehavior: 'block' | 'cancel'
  }
  /**
   * Run the workspace type-checker at turn end and report only the diagnostics
   * this turn introduced. Off by default because it spawns a real subprocess:
   * a host that has not opted in must never pay a typecheck per turn.
   */
  readonly editDiagnostics?: boolean
  readonly createEditFeedback?: (cwd: string) => EditFeedback
  /**
   * Root for off-transcript tool-result spill. Absent it, oversized results
   * stay inline — the previous behavior — so a host that has nowhere to write
   * is not silently degraded into losing output.
   */
  readonly toolResultDirectory?: string
  readonly temperature?: number
  /**
   * Requested OpenAI service tier (`auto`/`default`/`flex`/`priority`).
   * Responses-family transports send it as `service_tier`; the priced tier
   * comes back from the provider in usage, never assumed from the request.
   */
  readonly serviceTier?: string
  readonly tools?: readonly ToolDefinition[]
  readonly topK?: number
  readonly topP?: number
}

export interface BootstrapSystemPromptInput {
  /** Effective profile supplying mode-specific prompt and child catalog. */
  readonly agentId: string
  readonly model: string
  readonly session: DaemonSession
  readonly tools: readonly ToolDefinition[] | undefined
}

export type BootstrapSystemPromptProvider = (
  input: BootstrapSystemPromptInput,
) => Promise<string> | string

/** Adapts the portable agent loop to the frozen daemon wire-event vocabulary. */
/**
 * What keeps each session's system prompt byte-identical from turn to turn:
 * the rendered bootstrap, the memory snapshot and what has been delivered
 * since, the goal status last sent, the day the prompt was dated, and
 * whether it carries the compaction notice.
 *
 * It outlives any one runner. Settings changes (/fast, /permissions, a
 * profile switch, an MCP reconnect, an agent-preset edit) rebuild the runner;
 * when these lived on the runner, every rebuild re-rendered every open
 * session's prompt and each one re-wrote its whole conversation to the
 * provider cache on its next message. The host creates one and passes it to
 * every runner it builds.
 */
export class SessionPromptSnapshots {
  readonly bootstrapPrompts = new Map<string, Promise<string>>()
  /** Per session: memory as first rendered, and what has been delivered since. */
  readonly memorySnapshots = new Map<string, SessionMemorySnapshot>()
  /** Per session: the goal status the model was last given. */
  readonly deliveredGoalStatus = new Map<string, string>()
  /** Per session: the local day its prompt was dated, and whether it compacts. */
  readonly promptDay = new Map<string, string>()
  readonly compaction = new Map<string, boolean>()

  drop(sessionId: string): void {
    this.memorySnapshots.delete(sessionId)
    this.deliveredGoalStatus.delete(sessionId)
    this.promptDay.delete(sessionId)
    this.compaction.delete(sessionId)
    for (const key of this.bootstrapPrompts.keys()) {
      if (key.startsWith(sessionId + '\u0000')) this.bootstrapPrompts.delete(key)
    }
  }
}

export class AgentTurnRunner implements TurnRunner {
  readonly managesSessionState = true

  private readonly snapshots: SessionPromptSnapshots
  private get bootstrapPrompts() { return this.snapshots.bootstrapPrompts }
  private get memorySnapshots() { return this.snapshots.memorySnapshots }
  private get deliveredGoalStatus() { return this.snapshots.deliveredGoalStatus }
  private readonly states = new Map<string, AgentState>()
  private readonly toolResultStores = new Map<string, ToolResultStorage>()

  constructor(private readonly options: AgentTurnRunnerOptions) {
    this.snapshots = options.promptSnapshots ?? new SessionPromptSnapshots()
  }

  toolInventory(session: DaemonSession): RuntimeToolInventoryEntry[] {
    const agent = this.options.agentDefinitions?.get(session.agentId)
    if (this.options.agentDefinitions && !agent) throw new ValidationError('agent_id', 'is not a registered agent profile', session.agentId)
    const mode = interactionModeAgent(this.options.agentDefinitions, session.interactionMode)
    if (mode === null) throw new ValidationError('interaction_mode', 'does not have a registered enforcement profile', session.interactionMode)
    const all = this.options.toolRegistry?.definitions() ?? this.options.tools ?? []
    const permitted = toolsForAgent(toolsForAgent(all, agent), mode) ?? []
    const effective = session.metadata.session_kind === 'subagent' ? toolsForResumedSubagent(permitted, session.metadata) : permitted
    const allowed = new Set(effective?.map(tool => tool.function.name))
    const loaded = new Set((this.options.toolRegistry?.deferredToolLoading
      ? this.options.toolRegistry.definitionsForTranscript(session.messages.flatMap(messageToChatMessage))
      : this.options.tools ?? []).map(tool => tool.function.name))
    return all.map(tool => {
      const name = tool.function.name
      const exposure = !allowed.has(name) ? 'filtered' as const : loaded.has(name) ? 'loaded' as const : this.options.toolRegistry?.deferredToolLoading ? 'deferred' as const : 'unexposed' as const
      return { name, ...(tool.function.description ? { description: tool.function.description } : {}), exposure,
        reason: exposure === 'filtered' ? 'Excluded by the current agent or interaction mode' : exposure === 'deferred' ? 'Registered; load through tool search' : exposure === 'unexposed' ? 'Registered but absent from runner schemas' : 'Schema exposed to the current session' }
    })
  }

  async *run(
    session: DaemonSession,
    text: string,
    signal: AbortSignal,
    controls: TurnRunControls = {},
  ): AsyncGenerator<DaemonEvent> {
    const selectedModel = this.options.agentDefinitions?.get(session.agentId)?.model || session.model || this.options.model
    const requiresLocal = Object.hasOwn(session.metadata, LOCAL_PROVIDER_BINDING)
    const localClient = this.options.remoteProviderBindings?.client(session, selectedModel)
    if (requiresLocal && !localClient) throw new LocalProviderRelayError('grant_unavailable')
    const routedProvider = localClient ? { llm: localClient } : this.options.resolveSessionProvider?.(session, selectedModel)
    // A local binding owns its provider settings on the local machine. This
    // host still owns tool/permission policy, but its provider defaults must
    // not enter local requests or pretend to describe the local context window.
    const providerDefaults = requiresLocal ? undefined : this.options
    const contextLimit = requiresLocal ? undefined : routedProvider?.contextLimit ?? this.options.contextLimit
    const sessionEffort = requiresLocal && !session.reasoningPinned ? undefined : session.reasoningEffort
    const defaultEffort = sessionEffort ?? providerDefaults?.reasoningEffort
    const displayText = controls.displayText?.trim() || text
    // The session is the source of truth between turns: undo, retry, compact,
    // and idle steers mutate session.messages directly, so cached state must
    // re-adopt them instead of clobbering them at the next synchronization.
    const previous = this.states.get(session.id)
    const state = stateFromSession(session)
    // Snapshot and consume notices at the same synchronous boundary as the
    // effective mode/model/tool policy above. Later RPC changes belong to the
    // next request and remain queued on the live session; draining only the
    // live copy after async setup would either misreport them on this request
    // or let the state snapshot resurrect already-consumed notices at sync.
    const contextDeltas = takeContextDeltas(session.metadata)
    takeContextDeltas(state.metadata)
    // Instruction-file freshness (DSH reconciliation parity, delivered as a
    // volatile layer): files edited mid-session announce themselves with
    // fresh content on this turn instead of silently going stale until a
    // reload. Runs against the live session metadata so the recorded digest
    // baseline persists with the transcript.
    const instructionUpdates = await instructionFileUpdateLayer(
      sessionProjectRoot(session),
      session.metadata as Record<string, unknown>,
    )
    if (instructionUpdates) {
      state.metadata.instruction_file_digests = session.metadata.instruction_file_digests
    }
    if (previous) {
      state.totalCacheReadTokens = previous.totalCacheReadTokens
      state.totalCacheCreationTokens = previous.totalCacheCreationTokens
    }
    this.states.set(session.id, state)
    installMessageJournal(state, controls.journal)
    const projectRoot = sessionProjectRoot(session)
    // Anchor the pre-mutation baseline before any tool runs. Non-blocking:
    // a whole-project typecheck costs seconds and read-only turns must not pay it.
    const editFeedback = this.options.editDiagnostics
      ? this.options.createEditFeedback?.(projectRoot) ?? new EditFeedback(projectRoot)
      : undefined
    state.metadata.project_root = projectRoot
    state.metadata.interaction_mode = session.interactionMode
    state.metadata.plan_mode = session.planMode
    // Facts the goal tools authorise against. Written per turn because
    // authority is a property of THIS turn, not of the session: a human turn
    // may open or redefine a goal, while an automatic round may only conclude
    // one. Evidence starts false so a completion claim cannot inherit proof
    // from a previous turn.
    state.metadata.goal_turn_round = controls.goalRound ?? undefined
    state.metadata.goal_turn_human = controls.goalRound === undefined && (controls.origin ?? 'human') === 'human'
    state.metadata.turn_origin = controls.origin ?? (controls.goalRound === undefined ? 'human' : 'goal')
    delete state.metadata.pending_interaction_mode
    const agent = this.options.agentDefinitions?.get(session.agentId)
    if (this.options.agentDefinitions && !agent) {
      throw new ValidationError('agent_id', 'is not a registered agent profile', session.agentId)
    }
    const model = agent?.model || session.model || this.options.model
    const modeAgent = interactionModeAgent(this.options.agentDefinitions, session.interactionMode)
    if (modeAgent === null) {
      throw new ValidationError(
        'interaction_mode',
        'does not have a registered enforcement profile',
        session.interactionMode,
      )
    }
    // Deferred loading is resolved per turn, against this session's transcript.
    // `options.tools` is a snapshot taken once at construction, so it can only
    // ever be the full surface; the registry is the only thing that knows which
    // deferred schemas the transcript has already revealed. Falls back to the
    // snapshot whenever deferral is off, which keeps every embedding that
    // passes no registry on exactly its old behaviour.
    const availableTools = this.options.toolRegistry?.deferredToolLoading
      ? this.options.toolRegistry.definitionsForTranscript(state.messages)
      : this.options.tools
    const selectedTools = toolsForAgent(availableTools, agent)
    const modeTools = toolsForAgent(selectedTools, modeAgent)
    const resumedSubagent = session.metadata.session_kind === 'subagent'
    if (resumedSubagent) state.metadata.status = 'running'
    const tools = resumedSubagent ? toolsForResumedSubagent(modeTools, session.metadata) : modeTools
    // The session's own mode wins over the runner default, so the pin reaches
    // the permission broker rather than only the status line.
    const sessionPermissionMode = permissionModeValue(session.permissionMode) ?? this.options.permissionMode
    const configuredPermissionMode = permissionModeForInteraction(session.interactionMode, sessionPermissionMode)
    const permissionMode = resumedSubagent
      ? permissionModeForResumedSubagent(configuredPermissionMode, session.metadata)
      : configuredPermissionMode
    state.metadata.permission_mode = permissionMode
    const promptAgent = modeAgent ?? agent
    // The prompt describes the always-loaded core; loaded deferred tools carry
    // their own schemas. Loading one must not re-render the system prompt.
    const deferrable = this.options.toolRegistry?.deferredToolLoading
      ? new Set(this.options.toolRegistry.deferredCatalog(session.agentId).map(entry => entry.name))
      : new Set<string>()
    const coreTools = tools?.filter(tool => !deferrable.has(tool.function.name))
    const bootstrapPrompt = await this.bootstrapSystemPrompt(
      session,
      model,
      coreTools,
      promptAgent?.name ?? session.agentId,
    )
    const memory = this.options.agentMemory ? await this.options.agentMemory(session) : undefined
    await captureUserWorkflowMemory(displayText, memory, { projectRoot })
    const contextControls = readContextControls(session.metadata)
    if (!memory && (contextControls.pins.length || contextControls.excluded.length)) throw new Error('Session context controls require an available memory host')
    const selfMemory = this.options.agentSelfMemory ? await this.options.agentSelfMemory(session) : undefined
    // Memory enters the system prompt once per session, as a snapshot; what
    // is written later travels with the next message instead. Re-rendering
    // it each turn — the agent is told to record what it learns, and other
    // sessions write the shared global files — changed the system prompt
    // between almost every pair of turns, and each change re-billed the
    // whole conversation at the provider's cache-write price.
    const memoryState = await this.memoryForTurn(session, memory, selfMemory, contextControls)
    const memorySources = memoryState.sources
    const recoveredSubagents = this.options.subagentCoordinator
      ? recoverSubagentSnapshots(
        session.messages,
        session.id,
        persistedSubagentSnapshotValues(session.metadata),
      )
      : []
    this.options.subagentCoordinator?.hydrateDelivered?.(
      persistedSubagentDeliveryValues(session.metadata),
    )
    const restoredSubagentCount = this.options.subagentCoordinator
      ?.restore?.(session.id, recoveredSubagents) ?? 0
    // Assembled through the layered pipeline: every system layer is fixed for
    // the session, so identical turns send a byte-identical prefix; what
    // changed rides with the message (turnSegments). Every layer keeps a name
    // for provenance digests.
    // The goal's rules are fixed; its status (round, phase, criteria evidence)
    // moves every goal round, so it travels with the message, and only when
    // it differs from what the model was last told.
    const goalTools = tools?.some(tool => tool.function.name === 'update_goal') ?? false
    const lastGoalStatus = this.deliveredGoalStatus.get(session.id) ?? ''
    const currentGoal = goalTools ? renderGoalStatus(getGoal(session.metadata, session.id)) : ''
    // No goal is the default and needs no announcement; a cleared one does.
    const goalStatus = currentGoal || (lastGoalStatus ? 'The goal was cleared; no goal is set for this session.' : '')
    const goalChanged = currentGoal !== lastGoalStatus
    if (goalTools) this.deliveredGoalStatus.set(session.id, currentGoal)
    // The prompt keeps the date it was built with; a new day is news for the turn.
    const today = localDay(new Date())
    const promptDay = this.snapshots.promptDay.get(session.id)
    this.snapshots.promptDay.set(session.id, today)
    const turnSegments = assembleTurnContext({
      ...(promptDay && promptDay !== today ? { dateChange: `Today's date is now ${today}.` } : {}),
      ...(goalChanged ? { goalStatus } : {}),
      contextDeltas: renderContextDeltas(contextDeltas),
      ...(instructionUpdates ? { instructionUpdates } : {}),
      memoryChanges: memoryState.changes,
      selfMemoryChanges: memoryState.selfChanges,
      recoveredSubagents: restoredSubagentCount
        ? `${restoredSubagentCount} delegated task handle(s) were recovered from this resumed transcript after their daemon process ended. TaskListTool, TaskGetTool, PeekAgent, and AwaitAgents expose honest terminal snapshots: completed output is preserved, while work last seen active is marked interrupted and must be explicitly restarted with ResetAgent or respawned. Do not retry stale ids as if they were still running.`
        : '',
    })
    // Only where the loop really compacts: with no context limit or a zero
    // threshold nothing is summarized, and the promise would be false.
    // Frozen per session: the context limit can arrive late (models.dev loads
    // in the background), and the prompt must not gain this line mid-session.
    const compacts = this.snapshots.compaction.get(session.id)
      ?? (Boolean(contextLimit && contextLimit > 0) && (this.options.autoCompactThreshold?.() ?? 0.8) > 0)
    this.snapshots.compaction.set(session.id, compacts)
    const systemSegments = assembleContextLayers({
      addendum: systemPromptAddendum(session),
      ...(compacts ? { compaction: 'Long conversations are summarized automatically when context fills, and work continues. Do not cut work short, skip verification, or hand off because the conversation is long.' } : {}),
      agentPrompt: promptAgent?.systemPrompt ?? '',
      bootstrap: bootstrapPrompt,
      memory: memoryState.section,
      modeHint: modeSwitchHint(
        session.interactionMode,
        tools?.some(tool => tool.function.name === 'SetInteractionModeTool') ?? false,
      ),
      selfMemory: memoryState.selfSection,
      subagentJoin: this.options.subagentCoordinator
        ? 'Background subagents are joined before the parent turn ends. Integrate their delivered results in this turn; do not promise synthesis in a later turn.'
        : '',
      // Deferred loading hides most of the surface from the request. Without
      // this the model is simply told it has sixteen tools and concludes the
      // rest do not exist — it answered "I can't use AgentTool, it's not in my
      // available tool list" rather than searching for it. The hiding half and
      // the discovery half only work as a pair.
      goalPolicy: goalTools ? goalPolicyPrompt(DEFAULT_BLOCKED_AFTER_CONSECUTIVE_ROUNDS) : '',
      deferredCatalog: renderDeferredCatalog(
        this.options.toolRegistry?.deferredToolLoading
          ? this.options.toolRegistry.deferredCatalog(session.agentId)
          : [],
        // Only the core is subtracted, never what was loaded since: the list
        // is fixed for the session, so loading a tool leaves the prompt alone.
        new Set((coreTools ?? []).map(tool => tool.function.name)),
      ),
      toolGuidance: this.options.toolRegistry && coreTools?.length
        ? renderToolGuidance(
          this.options.toolRegistry.guidanceForTools(
            coreTools.map(tool => tool.function.name),
            session.agentId,
          ),
        )
        : '',
    })
    // Fingerprint the assembled layers before the request fires: any later
    // "why did this turn behave differently?" is a metadata diff, not a guess.
    recordAssemblyProvenance(session.metadata, {
      ...(session.activeTurnId ? { turnId: session.activeTurnId } : {}),
      layers: [...layerDigests(systemSegments), ...layerDigests(turnSegments).map(digest => ({ ...digest, name: 'turn:' + digest.name }))],
      recordedAt: Date.now(),
    })
    const systemPrompt = systemSegments.map(segment => segment.text).join('\n\n')
    const permissionBroker = this.options.interactions?.permissionBroker(session.id) ?? this.options.permissionBroker
    // Publish the request scaffolding the daemon's context meter cannot see.
    // Pricing the window from `session.messages` alone omits the system prompt
    // and every tool schema — the largest fixed cost in the request — which is
    // why auto-compaction fired late on tool-heavy sessions.
    session.requestScaffold = {
      capturedAt: Date.now(),
      memorySources,
      systemSegments: systemSegments.map(({ name, text }) => ({ name, text })),
      ...(systemPrompt ? { systemPrompt } : {}),
      ...(tools ? { toolSchemas: tools.map(tool => tool as unknown as Readonly<Record<string, unknown>>) } : {}),
    }
    const baseToolExecutor = interactiveToolExecutor(this.options.toolExecutor, this.options.interactions, session.id)
    const toolExecutor = baseToolExecutor && editFeedback ? editFeedback.wrap(baseToolExecutor) : baseToolExecutor
    const auditContext = {
      sessionId: session.id,
      agentId: session.agentId,
      ...(session.activeTurnId ? { turnId: session.activeTurnId } : {}),
    }
    this.options.auditEmitter?.emitTurnStart({ ...auditContext, prompt: displayText })
    let auditTurnEnded = false
    let resumedSubagentOutcome: 'cancelled' | 'completed' | 'error' = 'completed'
    const subagentCohort = this.options.subagentCoordinator?.begin(session.id)
    // Resolve thinking per turn rather than once per session, because the
    // strongest signal can change on every prompt: ultra mode wins first,
    // then an escalation keyword in this turn's text, then the session
    // defaults above. `session.ultraMode === true` narrows the optional
    // in-memory flag so both absent and false mean "no ultra override".
    const thinking = resolveTurnThinking({
      defaults: {
        ...(providerDefaults?.thinking !== undefined ? { enabled: providerDefaults.thinking } : {}),
        ...(providerDefaults?.thinkingBudget !== undefined ? { budgetTokens: providerDefaults.thinkingBudget } : {}),
        // The session's own effort wins over the runner default, so two open
        // sessions can run at different efforts and a resumed one continues at
        // the effort it was held at.
        ...(defaultEffort !== undefined)
          ? { effort: defaultEffort }
          : {},
      },
      prompt: text,
      ultraMode: session.ultraMode === true,
    })
    // Validated attachments become image_url data-URL parts on the user
    // message so every existing provider mapping (OpenAI parts, Anthropic
    // image blocks) works unchanged. Text-only turns keep string content.
    const images = controls.images ?? []
    // The turn context goes in front of the user's words and is persisted
    // with them, so later requests replay it unchanged; the transcript keeps
    // showing only what the user typed (displayText).
    const turnContext = renderTurnContext(turnSegments)
    const providerText = turnContext ? `${turnContext}\n\n${text}` : text
    const userMessage: MessageContent = images.length
      ? [{ type: 'text', text: providerText }, ...imageUrlContentParts(images)]
      : providerText
    let pendingAgentEventSnapshots: readonly SpawnedAgentSnapshot[] = []
    try {
      // Fallback model chain (Claude Code fallback-model parity): when the
      // primary provider fails terminally with an overload-class error before
      // any content streamed, the turn restarts once on the configured
      // fallback model. Restarting after content would duplicate streamed
      // text, so the fallback is strictly pre-content and exactly once.
      const fallbackModel = requiresLocal ? undefined : this.options.fallbackModel
      const fallbackFactory = routedProvider?.providerOverrides ? routedProvider.createLlmForModel : this.options.createLlmForModel
      let attemptModel = model
      let attemptLlm = routedProvider?.llm ?? this.options.llm
      let fallbackAttempted = false
      for (;;) {
        const maxTokens = requiresLocal ? undefined : this.options.maxTokens ?? (routedProvider?.maxOutputTokens ?? this.options.maxOutputTokens)?.(attemptModel)
        const retryPolicy = requiresLocal ? DEFAULT_RETRY_POLICY : retryPolicyForModel(attemptModel, routedProvider?.providerOverrides ?? this.options.providerOverrides)
        // An explicit off effort must cross the relay, otherwise absence
        // would correctly mean "use the local profile's thinking default".
        // Claude Code likewise thinks by default, so off has to be said.
        const thinkingRequest = thinking ? { budgetTokens: thinking.budgetTokens, effort: thinking.effort }
          : (requiresLocal || /^claude[-_]code\//i.test(attemptModel)) && sessionEffort ? { effort: 'none' } : undefined
        const turnEvents = withActiveSession(session, runTurn({
        turnId: session.activeTurnId,
        agentId: promptAgent?.name ?? session.agentId,
        interactionMode: session.interactionMode,
        model: attemptModel,
        sessionId: session.id,
        state,
        userMessage,
        ...(providerText === displayText ? {} : { userDisplayText: displayText }),
        ...(harnessOrigin(controls) ? { userOrigin: harnessOrigin(controls)! } : {}),
        querySource: 'main',
        ...(maxTokens === undefined ? {} : { maxTokens }),
        permissionMode,
        ...(providerDefaults?.temperature !== undefined ? { temperature: providerDefaults.temperature } : {}),
        ...(providerDefaults?.serviceTier !== undefined ? { serviceTier: providerDefaults.serviceTier } : {}),
        ...(thinkingRequest === undefined ? {} : { thinking: thinkingRequest }),
        ...(providerDefaults?.topK !== undefined ? { topK: providerDefaults.topK } : {}),
        ...(tools ? { tools } : {}),
        ...(systemPrompt ? { systemPrompt, systemPromptRequestOnly: true } : {}),
        ...(systemSegments.length ? { systemSegments } : {}),
        ...(providerDefaults?.topP !== undefined ? { topP: providerDefaults.topP } : {}),
      }, {
        ...(subagentCohort ? {
          awaitAgentEvents: async signal => {
            pendingAgentEventSnapshots = await subagentCohort.waitForResults(signal)
            mergePersistedSubagentSnapshots(state.metadata, pendingAgentEventSnapshots)
            return formatSubagentResults(pendingAgentEventSnapshots)
          },
          acknowledgeAgentEvents: () => {
            if (!pendingAgentEventSnapshots.length) return
            this.options.subagentCoordinator?.consume(pendingAgentEventSnapshots)
            pendingAgentEventSnapshots = []
            const delivered = this.options.subagentCoordinator?.deliveredState?.()
            if (delivered !== undefined) {
              replacePersistedSubagentDeliveries(state.metadata, delivered)
            }
          },
        } : {}),
        ...(controls.drainSteer ? { drainSteer: controls.drainSteer } : {}),
        // Retry patience is owned by the routed provider, not a global default.
        retryDelays: retryPolicy.delaysMs,
        maxSuggestedRetryDelayMs: retryPolicy.maxSuggestedDelayMs,
        llm: attemptLlm,
        // A used-up plan can be healed mid-turn by whatever the user just
        // changed (switched account, new login, edited key): rebuild from the
        // live profile so the loop can compare and retry as the new identity.
        ...(!requiresLocal && this.options.resolveSessionProvider ? {
          refreshLlm: () => this.options.resolveSessionProvider?.(session, attemptModel)?.llm,
        } : {}),
        ...((this.options.hookRunnerForSession || this.options.hookRunner) ? { hookRunner: this.options.hookRunnerForSession?.(session) ?? this.options.hookRunner } : {}),
        ...(permissionBroker ? { permissionBroker } : {}),
        ...(this.options.policy ? { policy: this.options.policy } : {}),
        ...(toolExecutor ? { toolExecutor } : {}),
        ...(this.options.reduceContext ? { reduceContext: async (messages, signal) => {
          try {
            return await this.options.reduceContext!(messages, signal)
          } finally {
            // The host records archive/failure metadata on the session. Keep it
            // in the turn state too, which replaces session metadata on save.
            for (const key of ['compaction_history', 'last_compaction', 'last_compaction_failure']) {
              if (session.metadata[key] !== undefined) state.metadata[key] = session.metadata[key]
            }
          }
        } } : {}),
        contextCompactionDue: messages => {
          const limit = contextLimit
          if (!limit || limit <= 0) return false
          const threshold = this.options.autoCompactThreshold?.() ?? 0.8
          if (threshold <= 0) return false
          const output = this.options.maxTokens ?? routedProvider?.maxOutputTokens?.(attemptModel) ?? this.options.maxOutputTokens?.(attemptModel) ?? 8192
          return estimateContextTokens(messages as unknown as Record<string, unknown>[], {
            model: attemptModel,
            ...(systemPrompt ? { systemPrompt } : {}),
            ...(session.requestScaffold?.toolSchemas ? { toolSchemas: session.requestScaffold.toolSchemas } : {}),
          }) >= Math.max(4096, limit - output) * Math.min(1, threshold)
        },
        persistToolResult: this.toolResultPersister(session),
        // Declared per tool at registration. Absent, the loop stays strictly
        // sequential, so an undeclared tool can never be run concurrently by
        // accident.
        ...(this.options.toolCapabilities ? { capabilities: this.options.toolCapabilities } : {}),
        // Without this the denial guard still stops a refusal loop, but the
        // audit event that records why stays at zero production callers.
        ...(this.options.auditEmitter
          ? { auditToolLoopBlock: (input) => this.options.auditEmitter?.emitToolLoopBlock(input) }
          : {}),
        }, signal))
        const decorate = (item: MultiplexedTurnEvent): DaemonEvent => {
          // Agent tools persist their live manifest into the turn-local state.
          // Mirror only those bounded manifest fields at every event boundary so
          // session.status can report running children before the parent turn
          // completes; the complete state still synchronizes in finally below.
          synchronizeLiveSubagentMetadata(session, state)
          if (item.kind === 'subagent') {
            return {
              type: item.event.type,
              payload: {
                ...item.event.payload,
                session_id: session.id,
                ...(session.activeTurnId ? { turn_id: session.activeTurnId } : {}),
              },
            }
          }
          const event = item.event
          // Evidence references must resolve before the next provider round,
          // while the runner still owns the transcript's final synchronization.
          if (event.type === 'tool_end') session.toolExecutions = [...state.toolExecutions]
          accumulateSessionTelemetry(session, event)
          auditStreamEvent(this.options.auditEmitter, event, auditContext, state)
          auditTurnEnded ||= event.type === 'turn_done'
          return daemonEventFromStream(
            event,
            state,
            session,
            contextLimit,
          )
        }
        // Pre-content buffering: hold status/retry events until the attempt
        // proves it can stream. A terminal overload before any content swaps
        // the whole attempt for the fallback model instead of surfacing.
        const preContent: MultiplexedTurnEvent[] = []
        let sawContent = false
        let restartWithFallback = false
        for await (const item of multiplexTurnEvents(turnEvents, this.options.subagentEvents, session.id)) {
          if (!sawContent && item.kind === 'turn') {
            const event = item.event
            // Waiting and compaction must be visible before model output exists.
            // permission_request must too: the loop emits it in the permission
            // phase BEFORE any tool_start and then parks on the broker, so a
            // tool-only first round would otherwise never flush the buffer and
            // the client would wait on an approval it was never shown.
            // Keep only terminal failure events buffered for fallback selection.
            if (event.type === 'provider_wait' || event.type === 'compaction'
              || event.type === 'permission_request'
              || (event.type === 'provider_retry' && !event.final)) {
              yield decorate(item)
              continue
            }
            if (
              event.type === 'provider_retry'
              && event.final
              && !fallbackAttempted
              && fallbackModel !== undefined
              && fallbackFactory !== undefined
              && !signal.aborted
            ) {
              const classified = classify(event.error)
              if (classified.kind === ErrorKind.RATE_LIMIT || classified.kind === ErrorKind.PROVIDER_DOWN) {
                restartWithFallback = true
                break
              }
            }
            if (event.type !== 'text' && event.type !== 'thinking' && event.type !== 'tool_start') {
              preContent.push(item)
              continue
            }
            sawContent = true
            for (const buffered of preContent) yield decorate(buffered)
            preContent.length = 0
          }
          yield decorate(item)
        }
        if (!restartWithFallback) {
          for (const buffered of preContent) yield decorate(buffered)
          break
        }
        fallbackAttempted = true
        // The failed attempt already pushed the user message and bumped
        // turnCount — runTurn does both unconditionally on every invocation.
        // It produced no assistant content (the restart is strictly
        // pre-content), so roll both back or the fallback attempt appends the
        // prompt a second time and the transcript keeps a duplicate user turn.
        // Search backwards rather than assuming the message is last: a steer
        // drained at the round boundary may have been appended after it.
        const pushedIndex = state.messages.findLastIndex(
          message => message.role === 'user' && message.content === userMessage,
        )
        if (pushedIndex >= 0) {
          state.messages.splice(pushedIndex, 1)
          if (state.turnCount > 0) state.turnCount -= 1
        }
        // Narrowed by the restart condition above: both are defined here.
        const nextModel = fallbackModel as string
        const nextFactory = fallbackFactory as (model: string) => LlmClient
        attemptModel = nextModel
        attemptLlm = nextFactory(nextModel)
        yield {
          type: 'notification',
          payload: {
            level: 'warning',
            message: `Provider overloaded before the reply started; retrying the turn on fallback model '${nextModel}'`,
          },
        }
      }
    } catch (error) {
      if (resumedSubagent) resumedSubagentOutcome = signal.aborted ? 'cancelled' : 'error'
      this.options.auditEmitter?.emitError({
        ...auditContext,
        errorType: error instanceof Error ? error.name : 'Error',
        errorMessage: error instanceof Error ? error.message : String(error),
        context: 'agent_turn_runner',
      })
      throw error
    } finally {
      if (resumedSubagent) {
        state.metadata.status = signal.aborted ? 'cancelled' : resumedSubagentOutcome
      }
      subagentCohort?.close()
      if (!auditTurnEnded) {
        this.options.auditEmitter?.emitTurnEnd({ ...auditContext, content: latestAssistantContent(state) })
      }
      if (this.options.memory) {
        makeTurnIndexerHook(this.options.memory, {
          ...(this.options.memoryMinChars === undefined ? {} : { minChars: this.options.memoryMinChars }),
        })(
          { agentId: session.agentId, response: latestAssistantContent(state) },
        )
      }
      // Deliver the checker's verdict as a fact rather than leaving the model
      // to claim the edit compiled. Only paths this turn actually mutated are
      // reported, so a repo with pre-existing errors stays quiet.
      if (this.options.editDiagnostics && turnMutatedFiles(state)) {
        const diagnostics = await editFeedback?.report(signal).catch(() => '')
        if (diagnostics) {
          state.messages.push({ role: 'user', content: diagnostics, origin: 'harness' })
        }
      }
      recordLatestUserDisplayText(state, providerText, displayText, harnessOrigin(controls))
      synchronizeSessionState(session, state)
    }
  }

  stateFor(sessionId: string): AgentState | undefined {
    return this.states.get(sessionId)
  }

  dropSession(sessionId: string): void {
    this.snapshots.drop(sessionId)
    this.states.delete(sessionId)
    this.toolResultStores.delete(sessionId)
    // Otherwise only the tracker's LRU bounds a long-lived daemon, and a file
    // read in an evicted session keeps pinning a freshness entry forever.
    fileStateTracker.clearSession(sessionId)
  }

  /**
   * Bounded provider view of an oversized tool result.
   *
   * The bootstrap prompt has always told every agent that large results are
   * stored outside model context and replaced with a preview. Nothing built
   * the store, so that was a promise the runtime did not keep: a single
   * `exec_command` with a raised output cap, or any MCP result (which is
   * truncated nowhere), could put a megabyte into the window. Both halves —
   * the previewer and the off-transcript store — were written and tested
   * already; this is the call site they were missing.
   */
  private toolResultPersister(session: DaemonSession): (toolName: string, content: string) => string {
    const directory = this.options.toolResultDirectory
    if (!directory) return (_toolName, content) => content
    return (toolName, content) => {
      if (content.length <= TOOL_RESULT_INLINE_LIMIT_CHARS) return content
      let store = this.toolResultStores.get(session.id)
      if (!store) {
        try {
          store = new ToolResultStorage(directory, { inlineLimit: TOOL_RESULT_INLINE_LIMIT_CHARS, sessionId: session.id })
        } catch {
          // An unwritable spill directory must never fail a tool call; the
          // preview below is still worth applying on its own.
          return boundedToolResultPreview(toolName, content, undefined)
        }
        this.toolResultStores.set(session.id, store)
      }
      try {
        const stored = store.maybeStore(toolName, content)
        const reference = typeof stored === 'string' ? ToolResultStorage.parseRef(stored) : undefined
        return boundedToolResultPreview(toolName, content, reference ? store.pathFor(reference) : undefined)
      } catch {
        // The store may become unwritable after construction. Never send the
        // oversized result back into model context; retain the same bounded
        // preview while making the loss of the spill file explicit.
        return boundedToolResultPreview(toolName, content, undefined)
      }
    }
  }


  /**
   * The session's memory for this turn. The first turn (or a change of
   * workspace or context controls) renders the section and snapshots the
   * files behind it; later turns reuse that exact text and report only what
   * changed since the last delivery, for the turn context.
   */
  private async memoryForTurn(
    session: DaemonSession,
    memory: AgentMemory | undefined,
    selfMemory: AgentSelfMemory | undefined,
    controls: ContextControls,
  ): Promise<MemoryForTurn> {
    const self = selfMemory ? await selfMemory.systemPromptAddendum() : ''
    if (!memory && !self) return { section: '', selfSection: '', changes: '', selfChanges: '', sources: [] }
    const key = JSON.stringify([session.cwd, controls.revision, controls.pins, controls.excluded])
    const sourceOptions = { excludedSources: controls.excluded, pinnedMemories: controls.pins }
    const current = memory ? await memory.promptSources(sourceOptions) : new Map<string, MemorySourceText>()
    const known = this.memorySnapshots.get(session.id)
    if (!known || known.key !== key) {
      const sources: MemorySourceText[] = []
      const section = memory ? await memory.toPromptSection({ ...sourceOptions, onSource: source => sources.push(source) }) : ''
      this.memorySnapshots.set(session.id, { key, section, selfSection: self, delivered: current, deliveredSelf: self, sources })
      return { section, selfSection: self, changes: '', selfChanges: '', sources }
    }
    const changes = renderMemoryChanges(known.delivered, current)
    const selfChanges = self === known.deliveredSelf
      ? ''
      : self
        ? '## Self-memory changed since this conversation began; it now reads\n\n' + self
        : '## Self-memory was cleared since this conversation began'
    known.delivered = current
    known.deliveredSelf = self
    return { section: known.section, selfSection: known.selfSection, changes, selfChanges, sources: known.sources }
  }

  private async bootstrapSystemPrompt(
    session: DaemonSession,
    model: string,
    tools: readonly ToolDefinition[] | undefined,
    agentId: string,
  ): Promise<string> {
    const provider = this.options.bootstrapSystemPrompt
    if (!provider) return ''
    const toolSignature = (tools ?? [])
      .map(tool => tool.function.name)
      .sort()
      .join('\u0001')
    // The provider receives the whole session, so every session-scoped input
    // the prompt can reflect — plan mode and the trusted addendum, alongside
    // workspace, model, agent, and tool surface — must stay in the cache key.
    // Per session: the prompt states a git snapshot "from session start",
    // which must not be another session's. Not per day — a new day is told
    // in the turn context instead of re-rendering the prompt. Session first,
    // so drop() can evict its entries.
    const key = [
      session.id,
      session.cwd,
      model,
      session.agentId,
      agentId,
      toolSignature,
      session.planMode === true ? 'plan' : '',
      systemPromptAddendum(session),
    ].join('\u0000')
    const existing = this.bootstrapPrompts.get(key)
    if (existing) return existing
    const prompt = Promise.resolve(provider({ agentId, session, model, tools })).catch(error => {
      this.bootstrapPrompts.delete(key)
      throw error
    })
    this.bootstrapPrompts.set(key, prompt)
    return prompt
  }
}

/**
 * Inline ceiling for a single tool result. Above this the provider sees a
 * preview and a path instead of the bytes. Chosen to sit well under the
 * smallest provider window while still passing ordinary file reads and test
 * output through untouched.
 */
const TOOL_RESULT_INLINE_LIMIT_CHARS = 16_000
const TOOL_RESULT_PREVIEW_CHARS = 4_000

/**
 * Render the stand-in the provider sees. The envelope is a tag rather than the
 * historical `[tool-result-ref:…]` handle because that handle was resolvable
 * only by the host: the model was handed an opaque id and no way to act on it.
 */
function boundedToolResultPreview(toolName: string, content: string, path: string | undefined): string {
  const compressed = compressToolResult(toolName, content, { maxChars: TOOL_RESULT_PREVIEW_CHARS })
  const attributes = [
    `tool=${JSON.stringify(toolName)}`,
    `chars=${content.length}`,
    `shown=${compressed.compressed.length}`,
    ...(path ? [`path=${JSON.stringify(path)}`] : []),
  ].join(' ')
  const recovery = path
    ? 'The full output is on disk at the path above; read it only if the preview is insufficient.'
    : 'The full output was not retained.'
  return `<persisted-output ${attributes}>\n${compressed.compressed}\n${recovery}\n</persisted-output>`
}

/** Tools whose success means a file on disk changed and a checker could disagree. */
const MUTATING_TOOL_NAMES = new Set([
  'AppendFile', 'Edit', 'FileEditTool', 'NotebookEditTool', 'Write', 'WriteFile',
  'append_file', 'edit_file', 'write_file',
])

/** True when this turn wrote to the workspace, so a diagnostics pass can earn its latency. */
function turnMutatedFiles(state: AgentState): boolean {
  return state.toolExecutions.some(execution => {
    if (typeof execution !== 'object' || execution === null) return false
    const record = execution as { name?: unknown; permitted?: unknown }
    return record.permitted === true && typeof record.name === 'string' && MUTATING_TOOL_NAMES.has(record.name)
  })
}

// Must match the shared agent-event injection block cap. Formatting a larger
// batch here would let the injection layer truncate it after every snapshot was
// acknowledged as delivered.
const MAX_SUBAGENT_RESULT_CHARS = 16_000
const MAX_SINGLE_SUBAGENT_RESULT_CHARS = 16_000
const MAX_INLINE_SUBAGENT_RESULTS = 64

export function formatSubagentResults(
  snapshots: readonly SpawnedAgentSnapshot[],
): readonly string[] {
  if (!snapshots.length) return []
  const visible = snapshots.slice(0, MAX_INLINE_SUBAGENT_RESULTS)
  const omitted = snapshots.length - visible.length
  const descriptors = visible.map(snapshot => {
    const raw = snapshot.lastOutput?.trim() || snapshot.error?.trim() || '(no final output)'
    const tokens = [snapshot.inputTokens, snapshot.outputTokens, snapshot.reasoningTokens]
      .filter((value): value is number => value !== undefined)
      .reduce((total, value) => total + value, 0)
    const metrics = [
      snapshot.toolCalls === undefined ? '' : `tools=${snapshot.toolCalls}`,
      tokens ? `tokens=${tokens}` : '',
    ].filter(Boolean).join(' ')
    return {
      footer: '[/agent result]',
      header: `[agent result id=${JSON.stringify(boundedLabel(snapshot.id))} title=${JSON.stringify(boundedLabel(snapshot.title))} status=${snapshot.status}${metrics ? ` ${metrics}` : ''}]`,
      raw,
    }
  })
  const omission = omitted > 0
    ? `[agent results omitted count=${omitted} total=${snapshots.length}] The full cohort remains available through paged TaskListTool plus TaskGetTool or TaskOutputTool.`
    : ''
  const eventCount = descriptors.length + (omission ? 1 : 0)
  const fixedChars = descriptors.reduce(
    (total, descriptor) => total + descriptor.header.length + descriptor.footer.length + 2,
    0,
  ) + omission.length + Math.max(0, eventCount - 1)
  let outputBudget = Math.max(0, MAX_SUBAGENT_RESULT_CHARS - fixedChars)
  const results: string[] = []
  for (const [index, descriptor] of descriptors.entries()) {
    const remainingAgents = descriptors.length - index
    const fairShare = Math.floor(outputBudget / remainingAgents)
    const output = boundedSubagentOutput(
      descriptor.raw,
      Math.min(MAX_SINGLE_SUBAGENT_RESULT_CHARS, fairShare),
    )
    outputBudget -= output.length
    results.push([descriptor.header, output, descriptor.footer].join('\n'))
  }
  if (omission) results.push(omission)
  return Object.freeze(results)
}

function boundedSubagentOutput(output: string, limit: number): string {
  if (limit <= 0) return ''
  if (output.length <= limit) return output
  const marker = `\n… [subagent output truncated by ${output.length - limit} characters] …\n`
  if (marker.length >= limit) return marker.slice(0, limit)
  const available = Math.max(0, limit - marker.length)
  const head = Math.ceil(available * 0.7)
  return output.slice(0, head) + marker + output.slice(-(available - head))
}

function boundedLabel(value: string, limit = 128): string {
  return value.length <= limit ? value : `${value.slice(0, limit - 1)}…`
}

type MultiplexedTurnEvent =
  | { readonly event: StreamEvent; readonly kind: 'turn' }
  | { readonly event: DaemonEvent; readonly kind: 'subagent' }

/** Yield child lifecycle events while the parent iterator is awaiting a tool. */
async function* multiplexTurnEvents(
  turnEvents: AsyncIterable<StreamEvent>,
  subagentEvents: DaemonSubagentEventSource | undefined,
  sessionId: string,
): AsyncGenerator<MultiplexedTurnEvent> {
  if (!subagentEvents) {
    for await (const event of turnEvents) yield { event, kind: 'turn' }
    return
  }

  const queued: DaemonEvent[] = []
  let wake: (() => void) | undefined
  const unsubscribe = subagentEvents.subscribe(sessionId, event => {
    queued.push(event)
    wake?.()
  })
  const iterator = turnEvents[Symbol.asyncIterator]()
  let nextTurn = iterator.next()

  try {
    while (true) {
      while (queued.length) {
        const event = queued.shift()
        if (event) yield { event, kind: 'subagent' }
      }

      const eventArrived = new Promise<'subagent'>(resolve => {
        wake = () => resolve('subagent')
      })
      const result = await Promise.race([
        nextTurn.then(value => ({ kind: 'turn' as const, value })),
        eventArrived.then(kind => ({ kind })),
      ])
      wake = undefined

      if (result.kind === 'subagent') continue
      if (result.value.done) {
        while (queued.length) {
          const event = queued.shift()
          if (event) yield { event, kind: 'subagent' }
        }
        return
      }
      yield { event: result.value.value, kind: 'turn' }
      nextTurn = iterator.next()
    }
  } finally {
    wake = undefined
    // Close the turn iterator while still subscribed so events published
    // during its cleanup land in the queue, then stop listening and drain
    // the remainder; nothing a child emitted may be silently dropped.
    await iterator.return?.()
    unsubscribe()
    while (queued.length) {
      const event = queued.shift()
      if (event) yield { event, kind: 'subagent' }
    }
  }
}

function interactiveToolExecutor(
  delegate: ToolExecutor | undefined,
  interactions: DaemonInteractionBoard | undefined,
  sessionId: string,
): ToolExecutor | undefined {
  if (!interactions) {
    return delegate
  }
  return {
    async execute(call, context, signal) {
      const question = questionFromToolCall(call)
      if (question) {
        const answer = await interactions.ask(sessionId, { ...question, toolCallId: call.id }, signal)
        return JSON.stringify({ answer, question: question.question })
      }
      if (!delegate) {
        return `Tool ${call.function.name} is unavailable.`
      }
      return delegate.execute(call, context, signal)
    },
  }
}

function questionFromToolCall(call: ToolCall): DaemonQuestion | undefined {
  const inputs = call.function.arguments
  const name = call.function.name
  if (name === 'ask_user') {
    const question = stringInput(inputs.question)
    if (!question) return undefined
    return {
      question,
      ...(stringArrayInput(inputs.options).length ? { options: stringArrayInput(inputs.options) } : {}),
      ...(typeof inputs.allow_freeform === 'boolean' ? { allowFreeform: inputs.allow_freeform } : {}),
      ...(stringInput(inputs.placeholder) ? { placeholder: stringInput(inputs.placeholder) } : {}),
    }
  }
  if (name !== 'AskUserQuestionTool') {
    return undefined
  }
  const directQuestion = stringInput(inputs.question)
  if (directQuestion) {
    return { question: directQuestion }
  }
  const first = Array.isArray(inputs.questions) ? inputs.questions[0] : undefined
  if (!isRecord(first)) return undefined
  const question = stringInput(first.question)
  if (!question) return undefined
  return {
    question,
    ...(stringInput(first.id) ? { questionId: stringInput(first.id) } : {}),
    ...(stringArrayInput(first.options).length ? { options: stringArrayInput(first.options) } : {}),
    ...(typeof first.allow_free_form === 'boolean' ? { allowFreeform: first.allow_free_form } : {}),
  }
}

function stringArrayInput(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === 'string') : []
}

function stringInput(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function auditStreamEvent(
  audit: AuditEmitter | undefined,
  event: StreamEvent,
  context: { readonly agentId: string; readonly sessionId: string; readonly turnId?: string },
  state: AgentState,
): void {
  if (!audit) return
  switch (event.type) {
    case 'tool_start':
      audit.emitToolCallAttempt({ ...context, toolName: event.call.function.name, args: event.call.function.arguments })
      return
    case 'tool_end':
      if (!event.result.permitted) {
        audit.emitToolPolicyDecision({ ...context, toolName: event.result.name, action: 'deny', source: 'permission' })
      } else if (event.result.result.startsWith('Tool execution failed:')) {
        audit.emitToolCallFailure({ ...context, toolName: event.result.name, errorType: 'ToolExecutionError', errorMessage: event.result.result })
      } else {
        audit.emitToolCallComplete({
          ...context,
          toolName: event.result.name,
          durationMs: event.result.durationMs,
          result: event.result.result,
        })
      }
      return
    case 'provider_retry':
      if (event.final) audit.emitError({ ...context, errorType: 'ProviderError', errorMessage: event.error, context: 'provider_stream' })
      return
    case 'turn_done':
      audit.emitTurnEnd({ ...context, content: latestAssistantContent(state), functionCallsCount: event.toolCallsCount })
      return
    default:
      return
  }
}

function latestAssistantContent(state: AgentState): string {
  const message = state.messages.slice().reverse().find(candidate => candidate.role === 'assistant')
  if (!message) return ''
  return typeof message.content === 'string' ? message.content : JSON.stringify(message.content)
}

interface SessionMemorySnapshot {
  readonly key: string
  readonly section: string
  readonly selfSection: string
  readonly sources: readonly MemorySourceText[]
  delivered: ReadonlyMap<string, MemorySourceText>
  deliveredSelf: string
}

interface MemoryForTurn {
  /** Fixed for the session: the system prompt's memory layers. */
  readonly section: string
  readonly selfSection: string
  /** New since the last turn: delivered in the turn context. */
  readonly changes: string
  readonly selfChanges: string
  readonly sources: readonly MemorySourceText[]
}

/** The date as the bootstrap Environment line states it (YYYY-MM-DD Weekday). */
function localDay(date: Date): string {
  const pad = (value: number) => String(value).padStart(2, '0')
  const weekday = new Intl.DateTimeFormat('en-US', { weekday: 'long' }).format(date)
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${weekday}`
}

function systemPromptAddendum(session: DaemonSession): string {
  return session.systemPromptAddendum?.trim() ?? ''
}

/**
 * Structured blocks a client can render instead of re-parsing result prose.
 *
 * This shipped as a hard-coded `[]`, so the TUI's entire todo pipeline —
 * `recordTodos`, `turnState.todos`, the progress rows — was fed by a channel
 * that never carried anything, and a todo list the model maintained was
 * invisible. The wire shape and both consumers already existed; only the
 * producer was missing.
 */
function displayBlocksFor(result: ToolResult): readonly Record<string, unknown>[] {
  if (result.name !== 'TodoWriteTool' || !result.permitted) return []
  const items = parseTodoList(result.result)
  return [{ type: 'todo', items }]
}

/**
 * Read back the canonical todo rendering TodoWriteTool returns.
 *
 * Parsing our own deterministic output rather than the model's, and locked to
 * the writer by a round-trip test — the alternative is threading a structured
 * payload through ToolResult, which every other tool would then carry for one
 * tool's benefit.
 */

/**
 * Tell the model what exists but is not loaded.
 *
 * With deferred loading on, a request carries the always-loaded core plus
 * whatever ToolSearchTool already revealed — around sixteen of seventy-six
 * schemas. A model shown sixteen tools and nothing else reasonably concludes
 * that is all there is, which silently removes capabilities the product
 * advertises. Names and one-line descriptions are enough to search on and far
 * cheaper than the schemas themselves, which is the whole point of deferring.
 *
 * Tools already in this request are filtered out so the list never tells the
 * model to search for something it can already call.
 */
function renderDeferredCatalog(
  catalog: readonly { readonly name: string; readonly description: string }[],
  visible: ReadonlySet<string>,
): string {
  const hidden = catalog.filter(entry => !visible.has(entry.name))
  if (!hidden.length) return ''
  const lines = hidden.map(entry => `- ${entry.name}: ${entry.description}`)
  return [
    '[Additional tools]',
    `${hidden.length} more tools can be loaded in this session; a tool's schema is sent only once it is loaded.`,
    'Load the ones this task needs with ONE ToolSearchTool call listing their exact names space-separated, then call them normally; a loaded tool is already in your tool list.',
    'Never tell the user a capability is unavailable because it is not in your current tool list — search first.',
    '',
    ...lines,
  ].join('\n')
}

/**
 * The current goal as of this turn, delivered in the turn context when it
 * changed. It spares the model a `get_goal` call to learn whether a goal
 * exists — but the exact id and revision it must copy still come from that
 * call, because a mutation mid-turn would make this stale.
 */
function renderGoalStatus(goal: GoalView | undefined): string {
  if (!goal) return ''
  const blocked = goal.blockedReason ? ` Blocker: ${goal.blockedReason.message}` : ''
  return `Current goal: ${JSON.stringify(goal.objective)} — phase ${goal.phase}, `
    + `round ${goal.roundsStarted} of ${goal.maxGoalRounds === DEFAULT_MAX_GOAL_ROUNDS ? 'unlimited' : goal.maxGoalRounds}, ${goal.activation}.${blocked}`
    + (goal.maxDurationMs === undefined ? '' : ` Wall-time deadline (Unix milliseconds): ${goal.createdAt + goal.maxDurationMs} (includes pauses).`)
    + (goal.criteria?.length ? '\nCompletion criteria: ' + JSON.stringify(goal.criteria.map(criterion => ({ id: criterion.id, description: criterion.description, evidenceToolCallId: criterion.evidence?.toolCallId ?? null }))) : '\nNo explicit completion criteria declared.')
}

/** Apply an agent's declared tool surface without exposing unregistered tools. */
function toolsForAgent(
  available: readonly ToolDefinition[] | undefined,
  agent: AgentDefinition | undefined,
): readonly ToolDefinition[] | undefined {
  if (!available || !agent) return available
  const declared = new Set(agent.tools)
  const allowed = agent.allowedTools === null ? undefined : new Set(agent.allowedTools)
  const excluded = new Set(agent.excludeTools)
  return available.filter(tool => {
    const name = tool.function.name
    if (excluded.has(name)) return false
    if (allowed && !allowed.has(name)) return false
    return declared.size === 0 || declared.has(name)
  })
}

/** Non-code modes use their declared profile as both prompt and enforceable tool ceiling. */
function interactionModeAgent(
  definitions: ReadonlyMap<string, AgentDefinition> | undefined,
  mode: string,
): AgentDefinition | null | undefined {
  const normalized = normalizeInteractionMode(mode)
  if (normalized === 'code') return undefined
  return definitions?.get(agentNameForMode(normalized)) ?? null
}

/** Restricted interaction modes never inherit the default YOLO permission policy. */
function permissionModeForInteraction(mode: string, configured: PermissionMode | undefined): PermissionMode {
  const normalized = normalizeInteractionMode(mode)
  return normalized === 'plan' || normalized === 'researcher'
    ? 'plan'
    : configured ?? DEFAULT_PERMISSION_MODE
}

/**
 * A child transcript remains a delegated agent when opened directly from the
 * history picker. Resuming it must not silently add orchestration/mode tools
 * or widen the policy ceiling it originally ran under.
 */
function toolsForResumedSubagent(
  tools: readonly ToolDefinition[] | undefined,
  metadata: Readonly<Record<string, unknown>>,
): readonly ToolDefinition[] | undefined {
  if (tools === undefined) return undefined
  const whitelist = metadataStringSet(metadata.tools_whitelist)
  const allowed = metadataStringSet(metadata.tools_allowed)
  const excluded = metadataStringSet(metadata.tools_excluded)
  const delegatedSurface = Array.isArray(metadata.toolsets)
    ? metadataStringSet(metadata.toolsets)
    : undefined
  return tools.filter(tool => {
    const name = tool.function.name
    if (SUBAGENT_BLOCKED_TOOLS.has(name) || excluded.has(name)) return false
    if (delegatedSurface && !delegatedSurface.has(name)) return false
    if (whitelist.size && !whitelist.has(name)) return false
    return !allowed.size || allowed.has(name)
  })
}

function permissionModeForResumedSubagent(
  configured: PermissionMode,
  metadata: Readonly<Record<string, unknown>>,
): PermissionMode {
  const stored = permissionModeValue(metadata.delegated_permission_mode)
    ?? permissionModeValue(metadata.permission_mode)
  if (stored === undefined) return configured
  return permissionModeExceeds(stored, configured) ? configured : stored
}

function permissionModeValue(value: unknown): PermissionMode | undefined {
  return value === 'accept-all' || value === 'auto' || value === 'manual' || value === 'plan'
    ? value
    : undefined
}

/** Match the effective delegated-policy ordering used by the native host. */
function permissionModeExceeds(candidate: PermissionMode, ceiling: PermissionMode): boolean {
  if (candidate === ceiling || ceiling === 'accept-all') return false
  if (ceiling === 'manual') return candidate !== 'manual'
  if (ceiling === 'plan') return candidate === 'auto' || candidate === 'accept-all'
  return candidate === 'accept-all'
}

function metadataStringSet(value: unknown): ReadonlySet<string> {
  return new Set(Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [])
}

function sessionProjectRoot(session: DaemonSession): string {
  const persisted = session.metadata.project_root
  return session.metadata.session_kind === 'subagent' && typeof persisted === 'string' && persisted.trim()
    ? persisted
    : session.cwd
}

/**
 * Wire the per-message crash journal into the mutable state message buffer.
 * Only messages appended after this call are recorded; existing history is
 * already persisted in the snapshot the state was built from.
 */
function installMessageJournal(
  state: AgentState,
  journal: TranscriptMessageJournalAppend | undefined,
): void {
  if (!journal) return
  const target = state.messages
  const originalPush = target.push.bind(target)
  target.push = function journalPush(...items: ChatMessage[]): number {
    const startIndex = target.length
    const result = originalPush(...items)
    for (let index = 0; index < items.length; index += 1) {
      journal(items[index] as unknown as RawMessage, startIndex + index)
    }
    return result
  }
}

function stateFromSession(session: DaemonSession): AgentState {
  const state = createAgentState(session.messages.flatMap(messageToChatMessage))
  state.apiCallsComplete = session.apiCallsComplete ?? session.turnCount === 0
  state.metadata = { ...session.metadata }
  // Goal tools and /goal must use the same compare-and-set log during a turn.
  // A copied log hides model edits until the turn ends and can overwrite a
  // human's intervening goal edit when the final state is synchronized.
  Object.defineProperty(state.metadata, GOAL_CHANGES_KEY, {
    enumerable: true,
    configurable: true,
    get: () => session.metadata[GOAL_CHANGES_KEY],
    set: value => { session.metadata[GOAL_CHANGES_KEY] = value },
  })
  state.thinkingContent = session.thinkingContent.filter((content): content is string => typeof content === 'string')
  state.toolExecutions = session.toolExecutions.filter(isToolExecutionRecord)
  state.totalApiCalls = session.totalApiCalls ?? 0
  state.totalInputTokens = session.totalInputTokens
  state.totalOutputTokens = session.totalOutputTokens
  state.turnCount = session.turnCount
  state.usageComplete = session.usageComplete ?? session.turnCount === 0
  return state
}

function synchronizeLiveSubagentMetadata(session: DaemonSession, state: AgentState): void {
  for (const key of [SUBAGENT_SNAPSHOT_METADATA_KEY, SUBAGENT_DELIVERY_METADATA_KEY] as const) {
    const value = state.metadata[key]
    if (value !== undefined && session.metadata[key] !== value) {
      session.metadata[key] = value
    }
  }
}

function synchronizeSessionState(session: DaemonSession, state: AgentState): void {
  session.apiCallsComplete = state.apiCallsComplete
  session.messages = state.messages.map(message => {
    const outcome = getTurnOutcome(message)
    const presentation = outcome ? { turn_outcome: outcome } : {}
    if (message.role !== 'user' || !message.displayText) return { ...message, ...presentation }
    const { displayText, ...providerMessage } = message
    return { ...providerMessage, text: displayText, ...presentation }
  })
  const mergedDeltas = mergeContextDeltas(state.metadata, session.metadata)
  // The picker can change the next turn's route while this turn is streaming.
  // Its session binding must survive the finishing turn's metadata snapshot.
  const providerProfile = session.metadata.provider_profile
  // The same is true for fields other RPCs write mid-turn without an
  // active-turn guard: a /title, /save, or session.goal wake staged while a
  // turn runs must not be reverted by the turn-end snapshot restore. Presence
  // is preserved too — a mid-turn delete (e.g. /title removing title_derived)
  // must delete here as well, or a stale flag rides the restore and a later
  // generated title overwrites the user's explicit rename. (goal_wake needs
  // no such care: kickGoalWake re-validates id/revision and cancels stale
  // wakes, so restoring a superseded one self-heals.)
  const preservedTitle = session.metadata.title
  const preservedGoalWake = session.metadata.goal_wake
  const hadTitleDerived = Object.hasOwn(session.metadata, 'title_derived')
  const preservedTitleDerived = session.metadata.title_derived
  session.metadata = { ...state.metadata }
  if (providerProfile !== undefined) session.metadata.provider_profile = providerProfile
  if (preservedTitle !== undefined) session.metadata.title = preservedTitle
  if (preservedGoalWake !== undefined) session.metadata.goal_wake = preservedGoalWake
  if (hadTitleDerived) session.metadata.title_derived = preservedTitleDerived
  else delete session.metadata.title_derived
  if (mergedDeltas.length) session.metadata.context_deltas = mergedDeltas
  session.thinkingContent = [...state.thinkingContent]
  session.toolExecutions = [...state.toolExecutions]
  session.totalApiCalls = state.totalApiCalls
  session.totalInputTokens = state.totalInputTokens
  session.totalOutputTokens = state.totalOutputTokens
  session.turnCount = state.turnCount
  session.usageComplete = state.usageComplete
}

/** Who wrote this turn's prompt when it was not the human. */
function harnessOrigin(controls: TurnRunControls): HarnessOrigin | undefined {
  if (controls.goalRound !== undefined) return 'goal'
  return controls.origin === 'monitor' || controls.origin === 'schedule' ? controls.origin : undefined
}

function recordLatestUserDisplayText(state: AgentState, providerText: string, displayText: string, origin?: HarnessOrigin): void {
  if (providerText === displayText && !origin) return
  for (let index = state.messages.length - 1; index >= 0; index -= 1) {
    const message = state.messages[index]
    // Content may be a structured part list (image attachments); compare on
    // the extracted text so displayText is still recorded for those turns.
    if (message?.role !== 'user' || messageText(message) !== providerText) continue
    state.messages[index] = { ...message, ...(providerText === displayText ? {} : { displayText }), ...(origin ? { origin } : {}) }
    return
  }
}

function messageToChatMessage(message: DaemonSession['messages'][number]): ChatMessage[] {
  return providerMessagesFromTranscript(message).map(value => restoreTurnOutcome(message, value))
}

function providerMessagesFromTranscript(message: DaemonSession['messages'][number]): ChatMessage[] {
  const role = message.role
  const content = message.content
  if (role === 'assistant' && isMessageContent(content)) {
    return [{
      role: 'assistant',
      content,
      ...(typeof message.thinking === 'string' ? { thinking: message.thinking } : {}),
      ...(typeof message.thinking_signature === 'string'
        ? { thinking_signature: message.thinking_signature }
        : {}),
      ...(Array.isArray(message.tool_calls) ? { tool_calls: message.tool_calls as readonly ToolCall[] } : {}),
    }]
  }
  if (role === 'system' && isMessageContent(content)) {
    return [{ role, content }]
  }
  if (role === 'user' && isMessageContent(content)) {
    return [{
      role,
      content,
      ...(typeof message.text === 'string' ? { displayText: message.text } : {}),
      ...(isHarnessOrigin(message.origin) ? { origin: message.origin } : {}),
    }]
  }
  if (role === 'tool' && typeof content === 'string' && typeof message.tool_call_id === 'string') {
    return [{
      role,
      content,
      tool_call_id: message.tool_call_id,
      ...(typeof message.name === 'string' ? { name: message.name } : {}),
      ...(message.is_error === true ? { is_error: true } : {}),
      // Where a tool was loaded: providers with native deferred tools render
      // the result by it, and dropping it re-rendered that result next turn.
      ...(Array.isArray(message.added_tool_names) && message.added_tool_names.every(name => typeof name === 'string')
        ? { added_tool_names: message.added_tool_names as string[] }
        : {}),
    }]
  }
  return []
}

function isMessageContent(value: unknown): value is MessageContent {
  return typeof value === 'string' || Array.isArray(value)
}

function isToolExecutionRecord(value: unknown): value is AgentState['toolExecutions'][number] {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return false
  }
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

interface SessionRuntimeTelemetry {
  cacheHitRate: number
  cacheReadTokens: number
  /** Prompt tokens written to the provider cache — billed above plain input. */
  cacheWriteTokens: number
  cacheTelemetryKnown: boolean
  inputTokens: number
  llmDurationMs: number
  llmSteps: number
  toolDurationMs: number
  toolSteps: number
  tokensPerSecond: number
  ttftSamples: number
  ttftTotalMs: number
}

function accumulateSessionTelemetry(session: DaemonSession, event: StreamEvent): void {
  if (event.type !== 'usage_update' && event.type !== 'tool_end') return
  const raw = session.extra.runtime_telemetry
  const stored = raw && typeof raw === 'object' && !Array.isArray(raw)
    ? raw as Record<string, unknown>
    : {}
  const finite = (key: keyof SessionRuntimeTelemetry): number => {
    const value = stored[key]
    return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : 0
  }
  const telemetry: SessionRuntimeTelemetry = {
    cacheHitRate: finite('cacheHitRate'),
    cacheReadTokens: finite('cacheReadTokens'),
    cacheWriteTokens: finite('cacheWriteTokens'),
    cacheTelemetryKnown: stored.cacheTelemetryKnown === true,
    inputTokens: finite('inputTokens'),
    llmDurationMs: finite('llmDurationMs'),
    llmSteps: finite('llmSteps'),
    toolDurationMs: finite('toolDurationMs'),
    toolSteps: finite('toolSteps'),
    tokensPerSecond: finite('tokensPerSecond'),
    ttftSamples: finite('ttftSamples'),
    ttftTotalMs: finite('ttftTotalMs'),
  }
  if (event.type === 'usage_update') {
    telemetry.llmSteps += 1
    telemetry.llmDurationMs += Math.max(0, event.durationMs ?? 0)
    if (event.tokensPerSecond !== undefined) telemetry.tokensPerSecond = Math.max(0, event.tokensPerSecond)
    if (event.usage.cacheReadTokens !== undefined || event.usage.cacheCreationTokens !== undefined) {
      telemetry.cacheTelemetryKnown = true
      telemetry.inputTokens += Math.max(0, event.usage.inputTokens)
      telemetry.cacheReadTokens += Math.max(0, event.usage.cacheReadTokens ?? 0)
      telemetry.cacheWriteTokens += Math.max(0, event.usage.cacheCreationTokens ?? 0)
      const cacheDenominator = telemetry.inputTokens + telemetry.cacheReadTokens + telemetry.cacheWriteTokens
      if (cacheDenominator > 0) telemetry.cacheHitRate = telemetry.cacheReadTokens / cacheDenominator
    }
    if (event.ttftMs !== undefined) {
      telemetry.ttftSamples += 1
      telemetry.ttftTotalMs += Math.max(0, event.ttftMs)
    }
  } else {
    telemetry.toolSteps += 1
    telemetry.toolDurationMs += Math.max(0, event.result.durationMs)
  }
  session.extra.runtime_telemetry = telemetry
}

function daemonEventFromStream(
  event: StreamEvent,
  state: AgentState,
  session: DaemonSession,
  contextLimit?: number,
): DaemonEvent {
  switch (event.type) {
    case 'provider_wait':
      return { type: 'status_update', payload: { kind: event.active ? 'provider_wait' : 'provider_ready', text: event.active ? 'Waiting for model response…' : '' } }
    case 'compaction':
      return { type: 'status_update', payload: { kind: event.active ? 'compressing' : 'compaction', text: event.active ? 'Compacting conversation…' : 'Compaction ended.' } }
    case 'text':
      return { type: 'text_part', payload: { text: event.text } }
    case 'thinking':
      return { type: 'think_part', payload: { think: event.text } }
    case 'provider_retry':
      if (!event.final && event.maxAttempts === 0) return { type: 'status_update', payload: { kind: 'network_retry', text: 'Retrying connection…', attempt: event.attempt, delay: event.delay } }
      return { type: 'notification', payload: { level: event.final ? 'error' : 'warning', message: event.error, retry: event } }
    case 'tool_start':
      return {
        type: 'tool_call',
        payload: {
          id: event.call.id,
          tool_call_id: event.call.id,
          name: event.call.function.name,
          arguments: JSON.stringify(event.call.function.arguments),
          ...(event.reasoning ? { reasoning: event.reasoning } : {}),
        },
      }
    case 'permission_request':
      return {
        type: 'approval_request',
        payload: {
          id: event.request.requestId,
          request_id: event.request.requestId,
          name: event.request.toolCall.function.name,
          action: event.request.toolCall.function.name,
          tool_name: event.request.toolCall.function.name,
          description: event.request.description,
          inputs: event.request.inputs,
        },
      }
    case 'tool_end':
      return {
        type: 'tool_result',
        payload: {
          name: event.result.name,
          result: event.result.result,
          return_value: event.result.result,
          permitted: event.result.permitted,
          tool_call_id: event.result.toolCallId,
          duration_ms: event.result.durationMs,
          display_blocks: displayBlocksFor(event.result),
          ...(event.reasoning ? { reasoning: event.reasoning } : {}),
        },
      }
    case 'usage_update':
      // The provider's per-round input is the request context it actually saw;
      // include the generated output so the remaining-token meter moves before
      // the buffered visible deltas are replayed. Cumulative session usage is
      // billing history and must not be mistaken for current-window occupancy.
      return {
        type: 'status_update',
        payload: {
          model: event.model,
          usage: event.cumulative,
          total_input_tokens: state.totalInputTokens,
          total_output_tokens: state.totalOutputTokens,
          input_tokens: state.totalInputTokens,
          output_tokens: state.totalOutputTokens,
          total_tokens: state.totalInputTokens + state.totalOutputTokens,
          context_tokens:
            event.usage.inputTokens + (event.usage.cacheReadTokens ?? 0) + (event.usage.cacheCreationTokens ?? 0) + event.usage.outputTokens,
          ...(typeof contextLimit === 'number' && contextLimit > 0 ? { max_context: contextLimit } : {}),
          ...(state.totalCacheReadTokens ? { cache_read_tokens: state.totalCacheReadTokens } : {}),
          ...(state.totalCacheCreationTokens ? { cache_creation_tokens: state.totalCacheCreationTokens } : {}),
          ...(event.durationMs === undefined ? {} : { llm_duration_ms: event.durationMs }),
          ...(event.ttftMs === undefined ? {} : { ttft_ms: event.ttftMs }),
          ...(event.tokensPerSecond === undefined ? {} : { tokens_per_second: event.tokensPerSecond }),
          ...(event.cacheHitRate === undefined ? {} : { cache_hit_rate: event.cacheHitRate }),
        },
      }
    case 'turn_done': {
      const contextTokens = estimateContextTokens(
        state.messages as unknown as Record<string, unknown>[],
        {
          model: event.model,
          ...(session.requestScaffold?.systemPrompt ? { systemPrompt: session.requestScaffold.systemPrompt } : {}),
          ...(session.requestScaffold?.toolSchemas?.length ? { toolSchemas: session.requestScaffold.toolSchemas } : {}),
        },
      )
      return {
        type: 'status_update',
        payload: {
          model: event.model,
          usage: event.usage,
          usage_complete: state.usageComplete,
          ...(event.reason ? { stop_reason: event.reason } : {}),
          tool_calls: event.toolCallsCount,
          ...(event.apiCallsCount === undefined ? {} : { api_calls: event.apiCallsCount }),
          ...(state.apiCallsComplete
            ? { calls: state.totalApiCalls }
            : { calls_complete: false, observed_calls: state.totalApiCalls }),
          total_input_tokens: state.totalInputTokens,
          total_output_tokens: state.totalOutputTokens,
          input_tokens: state.totalInputTokens,
          output_tokens: state.totalOutputTokens,
          total_tokens: state.totalInputTokens + state.totalOutputTokens,
          context_tokens: contextTokens,
          ...(typeof contextLimit === 'number' && contextLimit > 0 ? { max_context: contextLimit } : {}),
          mode: session.interactionMode,
          plan_mode: session.planMode,
          ...(state.totalCacheReadTokens ? { cache_read_tokens: state.totalCacheReadTokens } : {}),
          ...(state.totalCacheCreationTokens ? { cache_creation_tokens: state.totalCacheCreationTokens } : {}),
        },
      }
    }
    case 'skill_suggestion':
      appendSkillSuggestion(state.metadata, event)
      return { type: 'notification', payload: { level: 'info', message: `Skill suggestion: ${event.skillName}`, skill: event } }
  }
}
