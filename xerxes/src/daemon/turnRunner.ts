// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { AgentDefinition } from '../agents/definitions.js'
import type { AuditEmitter } from '../audit/emitter.js'
import { compressToolResult } from '../context/headroom.js'
import {
  assembleContextLayers,
  layerDigests,
  recordAssemblyProvenance,
} from '../context/assembly.js'
import { ToolResultStorage } from '../context/toolResultStorage.js'
import { estimateContextTokens } from '../context/windowUsage.js'
import { ValidationError } from '../core/errors.js'
import {
  renderToolGuidance,
  type ToolExecutor,
  type ToolRegistry,
} from '../executors/toolRegistry.js'
import type { AgentMemory } from '../memory/agentMemory.js'
import {
  mergePersistedSubagentSnapshots,
  persistedSubagentDeliveryValues,
  persistedSubagentSnapshotValues,
  replacePersistedSubagentDeliveries,
} from '../agents/subagentPersistence.js'
import { SUBAGENT_BLOCKED_TOOLS } from '../agents/subagentManager.js'
import type { AgentSelfMemory } from '../memory/agentSelfMemory.js'
import { makeTurnIndexerHook } from '../memory/turnIndexer.js'
import type { Memory } from '../memory/base.js'
import type { SpawnedAgentSnapshot } from '../operators/subagents.js'
import type { LlmClient } from '../llms/client.js'
import { getContextLimit, type ProviderOverrides, retryPolicyForModel } from '../llms/providerRegistry.js'
import { agentNameForMode, modeSwitchHint, normalizeInteractionMode } from '../runtime/interactionModes.js'
import {
  readGoalLedger,
  startGoalLedger,
  updateGoalLedger,
} from '../runtime/goalState.js'
import {
  renderContextDeltas,
  takeContextDeltas,
} from '../runtime/contextDeltas.js'
import { beginEditDiagnosticsTurn, reportEditDiagnostics } from '../runtime/editDiagnostics.js'
import { withActiveSession } from '../runtime/sessionContext.js'
import { resolveTurnThinking } from '../runtime/thinkingLevels.js'
import { captureUserWorkflowMemory } from '../runtime/workflowMemory.js'
import { createAgentState, type AgentState, type StreamEvent } from '../streaming/events.js'
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
import { messageText } from '../types/messages.js'
import { imageUrlContentParts } from './images.js'
import type { RawMessage, TranscriptMessageJournalAppend } from '../session/daemonTranscript.js'
import type { ToolCall, ToolDefinition } from '../types/toolCalls.js'
import type { DaemonInteractionBoard, DaemonQuestion } from './interactions.js'
import type { DaemonEvent, DaemonSession, TurnRunControls, TurnRunner } from './runtime.js'
import {
  recoverSubagentSnapshots,
  type SubagentTurnCoordinator,
} from './subagentCoordinator.js'
import type { DaemonSubagentEventSource } from './subagentEvents.js'

export interface AgentTurnRunnerOptions {
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
  /** Native daemon reply board for approvals and ask-user tool calls. */
  readonly interactions?: DaemonInteractionBoard
  /** Optional tier receiving completed assistant turns for recall on later work. */
  readonly memory?: Memory
  readonly memoryMinChars?: number
  readonly llm: LlmClient
  readonly maxTokens?: number
  readonly model: string
  readonly permissionBroker?: PermissionBroker
  /**
   * The connection's provider identity, threaded from the active profile.
   *
   * `retryPolicyForModel` and `getContextLimit` resolve a provider from the
   * model id when given nothing else, and an OpenRouter id like
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
  /**
   * Root for off-transcript tool-result spill. Absent it, oversized results
   * stay inline — the previous behavior — so a host that has nowhere to write
   * is not silently degraded into losing output.
   */
  readonly toolResultDirectory?: string
  readonly temperature?: number
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
export class AgentTurnRunner implements TurnRunner {
  readonly managesSessionState = true

  private readonly bootstrapPrompts = new Map<string, Promise<string>>()
  private readonly states = new Map<string, AgentState>()
  private readonly toolResultStores = new Map<string, ToolResultStorage>()

  constructor(private readonly options: AgentTurnRunnerOptions) {}

  async *run(
    session: DaemonSession,
    text: string,
    signal: AbortSignal,
    controls: TurnRunControls = {},
  ): AsyncGenerator<DaemonEvent> {
    const displayText = controls.displayText?.trim() || text
    // The session is the source of truth between turns: undo, retry, compact,
    // and idle steers mutate session.messages directly, so cached state must
    // re-adopt them instead of clobbering them at the next synchronization.
    const previous = this.states.get(session.id)
    const state = stateFromSession(session)
    if (previous) {
      state.totalCacheReadTokens = previous.totalCacheReadTokens
      state.totalCacheCreationTokens = previous.totalCacheCreationTokens
    }
    this.states.set(session.id, state)
    installMessageJournal(state, controls.journal)
    const projectRoot = sessionProjectRoot(session)
    // Anchor the pre-mutation baseline before any tool runs. Non-blocking:
    // a whole-project typecheck costs seconds and read-only turns must not pay it.
    if (this.options.editDiagnostics) beginEditDiagnosticsTurn(projectRoot)
    state.metadata.project_root = projectRoot
    state.metadata.interaction_mode = session.interactionMode
    state.metadata.plan_mode = session.planMode
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
    const selectedTools = toolsForAgent(this.options.tools, agent)
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
    const bootstrapPrompt = await this.bootstrapSystemPrompt(
      session,
      model,
      tools,
      promptAgent?.name ?? session.agentId,
    )
    const memory = this.options.agentMemory ? await this.options.agentMemory(session) : undefined
    await captureUserWorkflowMemory(displayText, memory, { projectRoot })
    // Rank the memory manifest against this turn rather than emitting every
    // topic in path order. Without a query the selector is inert, so calling
    // it with no arguments — as this did — left the ranking permanently off.
    const memoryPrompt = memory ? await memory.toPromptSection({
      query: displayText,
      alreadySurfaced: recentTranscriptText(session),
      recentSuccessfulTools: recentSuccessfulToolNames(session),
    }) : ''
    const selfMemory = this.options.agentSelfMemory ? await this.options.agentSelfMemory(session) : undefined
    const selfMemoryPrompt = selfMemory ? await selfMemory.systemPromptAddendum() : ''
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
    // Assembled through the layered pipeline: identical inputs are byte-stable,
    // stable layers precede volatile ones for the cache breakpoint, and every
    // layer keeps a name for provenance digests.
    const systemSegments = assembleContextLayers({
      addendum: systemPromptAddendum(session),
      agentPrompt: promptAgent?.systemPrompt ?? '',
      bootstrap: bootstrapPrompt,
      contextDeltas: renderContextDeltas(takeContextDeltas(session.metadata)),
      memoryRecall: memoryPrompt,
      modeHint: modeSwitchHint(
        session.interactionMode,
        tools?.some(tool => tool.function.name === 'SetInteractionModeTool') ?? false,
      ),
      recoveredSubagents: restoredSubagentCount
        ? `${restoredSubagentCount} delegated task handle(s) were recovered from this resumed transcript after their daemon process ended. TaskListTool, TaskGetTool, PeekAgent, and AwaitAgents expose honest terminal snapshots: completed output is preserved, while work last seen active is marked interrupted and must be explicitly restarted with ResetAgent or respawned. Do not retry stale ids as if they were still running.`
        : '',
      selfMemory: selfMemoryPrompt,
      subagentJoin: this.options.subagentCoordinator
        ? 'Background subagents are joined before the parent turn ends. Integrate their delivered results in this turn; do not promise synthesis in a later turn.'
        : '',
      toolGuidance: this.options.toolRegistry && tools?.length
        ? renderToolGuidance(
          this.options.toolRegistry.guidanceForTools(
            tools.map(tool => tool.function.name),
            session.agentId,
          ),
        )
        : '',
    })
    // Objective mode gets a durable goal ledger: the turn's own prompt is the
    // goal statement on first entry, and every guarded round is accounted for
    // across restarts instead of living only in this turn's locals.
    if (session.interactionMode === 'objective') {
      const now = Date.now()
      const existing = readGoalLedger(session.metadata)
      const started = startGoalLedger(session.metadata, { now, text: displayText })
      const ledger = 'created' in started ? started.created : started.existing
      const outcome = updateGoalLedger(session.metadata, ledger.revision, { roundDelta: 1 }, now)
      if (!outcome.ok
        && outcome.reason === 'stale'
        && outcome.conflictWith !== undefined) {
        // A concurrent writer advanced the ledger; retry once against its view.
        updateGoalLedger(session.metadata, outcome.conflictWith.revision, { roundDelta: 1 }, now)
      }
    }
    // Fingerprint the assembled layers before the request fires: any later
    // "why did this turn behave differently?" is a metadata diff, not a guess.
    recordAssemblyProvenance(session.metadata, {
      ...(session.activeTurnId ? { turnId: session.activeTurnId } : {}),
      layers: layerDigests(systemSegments),
      recordedAt: Date.now(),
    })
    const systemPrompt = systemSegments.map(segment => segment.text).join('\n\n')
    const permissionBroker = this.options.interactions?.permissionBroker(session.id) ?? this.options.permissionBroker
    // Publish the request scaffolding the daemon's context meter cannot see.
    // Pricing the window from `session.messages` alone omits the system prompt
    // and every tool schema — the largest fixed cost in the request — which is
    // why auto-compaction fired late on tool-heavy sessions.
    session.requestScaffold = {
      ...(systemPrompt ? { systemPrompt } : {}),
      ...(tools ? { toolSchemas: tools.map(tool => tool as unknown as Readonly<Record<string, unknown>>) } : {}),
    }
    const toolExecutor = interactiveToolExecutor(this.options.toolExecutor, this.options.interactions, session.id)
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
        ...(this.options.thinking !== undefined ? { enabled: this.options.thinking } : {}),
        ...(this.options.thinkingBudget !== undefined ? { budgetTokens: this.options.thinkingBudget } : {}),
        // The session's own effort wins over the runner default, so two open
        // sessions can run at different efforts and a resumed one continues at
        // the effort it was held at.
        ...(session.reasoningEffort ?? this.options.reasoningEffort) !== undefined
          ? { effort: session.reasoningEffort ?? this.options.reasoningEffort }
          : {},
      },
      prompt: text,
      ultraMode: session.ultraMode === true,
    })
    // Validated attachments become image_url data-URL parts on the user
    // message so every existing provider mapping (OpenAI parts, Anthropic
    // image blocks) works unchanged. Text-only turns keep string content.
    const images = controls.images ?? []
    const userMessage: MessageContent = images.length
      ? [{ type: 'text', text }, ...imageUrlContentParts(images)]
      : text
    let pendingAgentEventSnapshots: readonly SpawnedAgentSnapshot[] = []
    try {
      const turnEvents = withActiveSession(session, runTurn({
        agentId: promptAgent?.name ?? session.agentId,
        interactionMode: session.interactionMode,
        model,
        sessionId: session.id,
        state,
        userMessage,
        querySource: 'main',
        ...(this.options.maxTokens !== undefined ? { maxTokens: this.options.maxTokens } : {}),
        permissionMode,
        ...(this.options.temperature !== undefined ? { temperature: this.options.temperature } : {}),
        ...(thinking === undefined ? {} : { thinking: { budgetTokens: thinking.budgetTokens, effort: thinking.effort } }),
        ...(this.options.topK !== undefined ? { topK: this.options.topK } : {}),
        ...(tools ? { tools } : {}),
        ...(systemPrompt ? { systemPrompt, systemPromptRequestOnly: true } : {}),
        ...(systemSegments.length ? { systemSegments } : {}),
        ...(this.options.topP !== undefined ? { topP: this.options.topP } : {}),
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
        retryDelays: retryPolicyForModel(model, this.options.providerOverrides).delaysMs,
        maxSuggestedRetryDelayMs: retryPolicyForModel(model, this.options.providerOverrides)
          .maxSuggestedDelayMs,
        llm: this.options.llm,
        ...(permissionBroker ? { permissionBroker } : {}),
        ...(this.options.policy ? { policy: this.options.policy } : {}),
        ...(toolExecutor ? { toolExecutor } : {}),
        ...(this.options.reduceContext ? { reduceContext: this.options.reduceContext } : {}),
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
      for await (const item of multiplexTurnEvents(turnEvents, this.options.subagentEvents, session.id)) {
        if (item.kind === 'subagent') {
          yield {
            type: item.event.type,
            payload: {
              ...item.event.payload,
              session_id: session.id,
              ...(session.activeTurnId ? { turn_id: session.activeTurnId } : {}),
            },
          }
          continue
        }
        const event = item.event
        auditStreamEvent(this.options.auditEmitter, event, auditContext, state)
        auditTurnEnded ||= event.type === 'turn_done'
        // A guard-verified completion is the ledger's terminal transition.
        if (event.type === 'turn_done' && event.reason === 'objective_verified') {
          const ledger = readGoalLedger(session.metadata)
          if (ledger && ledger.phase !== 'verified') {
            const verification = updateGoalLedger(session.metadata, ledger.revision, { phase: 'verified' }, Date.now())
            if (!verification.ok
              && verification.reason === 'stale'
              && verification.conflictWith !== undefined) {
              updateGoalLedger(
                session.metadata,
                verification.conflictWith.revision,
                { phase: 'verified' },
                Date.now(),
              )
            }
          }
        }
        yield daemonEventFromStream(event, state, session, this.options.providerOverrides)
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
        const diagnostics = await reportEditDiagnostics(projectRoot).catch(() => '')
        if (diagnostics) {
          state.messages.push({ role: 'user', content: diagnostics })
        }
      }
      recordLatestUserDisplayText(state, text, displayText)
      synchronizeSessionState(session, state)
    }
  }

  stateFor(sessionId: string): AgentState | undefined {
    return this.states.get(sessionId)
  }

  dropSession(sessionId: string): void {
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
    const key = [
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

/** Recent transcript text used to suppress memories the conversation already covered. */
function recentTranscriptText(session: DaemonSession, turns = 12): string {
  return session.messages
    .slice(-turns)
    .map(message => (typeof message.content === 'string' ? message.content : ''))
    .filter(Boolean)
    .join('\n')
}

/** Tools that recently succeeded; their reference topics rank down, their gotchas do not. */
function recentSuccessfulToolNames(session: DaemonSession, limit = 24): readonly string[] {
  const names = new Set<string>()
  for (const execution of session.toolExecutions.slice(-limit)) {
    if (typeof execution !== 'object' || execution === null) continue
    const record = execution as { name?: unknown; permitted?: unknown }
    if (record.permitted === true && typeof record.name === 'string') names.add(record.name)
  }
  return [...names]
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

function systemPromptAddendum(session: DaemonSession): string {
  return session.systemPromptAddendum?.trim() ?? ''
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
  state.thinkingContent = session.thinkingContent.filter((content): content is string => typeof content === 'string')
  state.toolExecutions = session.toolExecutions.filter(isToolExecutionRecord)
  state.totalApiCalls = session.totalApiCalls ?? 0
  state.totalInputTokens = session.totalInputTokens
  state.totalOutputTokens = session.totalOutputTokens
  state.turnCount = session.turnCount
  state.usageComplete = session.usageComplete ?? session.turnCount === 0
  return state
}

function synchronizeSessionState(session: DaemonSession, state: AgentState): void {
  session.apiCallsComplete = state.apiCallsComplete
  session.messages = state.messages.map(message => {
    if (message.role !== 'user' || !message.displayText) return { ...message }
    const { displayText, ...providerMessage } = message
    return { ...providerMessage, text: displayText }
  })
  session.metadata = { ...state.metadata }
  session.thinkingContent = [...state.thinkingContent]
  session.toolExecutions = [...state.toolExecutions]
  session.totalApiCalls = state.totalApiCalls
  session.totalInputTokens = state.totalInputTokens
  session.totalOutputTokens = state.totalOutputTokens
  session.turnCount = state.turnCount
  session.usageComplete = state.usageComplete
}

function recordLatestUserDisplayText(state: AgentState, providerText: string, displayText: string): void {
  if (providerText === displayText) return
  for (let index = state.messages.length - 1; index >= 0; index -= 1) {
    const message = state.messages[index]
    // Content may be a structured part list (image attachments); compare on
    // the extracted text so displayText is still recorded for those turns.
    if (message?.role !== 'user' || messageText(message) !== providerText) continue
    state.messages[index] = { ...message, displayText }
    return
  }
}

function messageToChatMessage(message: DaemonSession['messages'][number]): ChatMessage[] {
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
    }]
  }
  if (role === 'tool' && typeof content === 'string' && typeof message.tool_call_id === 'string') {
    return [{
      role,
      content,
      tool_call_id: message.tool_call_id,
      ...(typeof message.name === 'string' ? { name: message.name } : {}),
      ...(message.is_error === true ? { is_error: true } : {}),
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

function daemonEventFromStream(
  event: StreamEvent,
  state: AgentState,
  session: DaemonSession,
  // Passed rather than sniffed: see `TurnRunnerOptions.providerOverrides`.
  providerOverrides?: ProviderOverrides
): DaemonEvent {
  switch (event.type) {
    case 'text':
      return { type: 'text_part', payload: { text: event.text } }
    case 'thinking':
      return { type: 'think_part', payload: { think: event.text } }
    case 'provider_retry':
      return { type: 'notification', payload: { level: event.final ? 'error' : 'warning', message: event.error, retry: event } }
    case 'tool_start':
      return {
        type: 'tool_call',
        payload: {
          id: event.call.id,
          tool_call_id: event.call.id,
          name: event.call.function.name,
          arguments: JSON.stringify(event.call.function.arguments),
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
          display_blocks: [],
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
            event.usage.inputTokens + (event.usage.cacheReadTokens ?? 0) + event.usage.outputTokens,
          max_context: getContextLimit(event.model, providerOverrides),
          ...(state.totalCacheReadTokens ? { cache_read_tokens: state.totalCacheReadTokens } : {}),
          ...(state.totalCacheCreationTokens ? { cache_creation_tokens: state.totalCacheCreationTokens } : {}),
        },
      }
    case 'turn_done': {
      const contextTokens = estimateContextTokens(
        state.messages.map(message => ({ role: message.role, content: message.content })),
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
          max_context: getContextLimit(event.model, providerOverrides),
          mode: session.interactionMode,
          plan_mode: session.planMode,
          ...(state.totalCacheReadTokens ? { cache_read_tokens: state.totalCacheReadTokens } : {}),
          ...(state.totalCacheCreationTokens ? { cache_creation_tokens: state.totalCacheCreationTokens } : {}),
        },
      }
    }
    case 'skill_suggestion':
      return { type: 'notification', payload: { level: 'info', message: `Skill suggestion: ${event.skillName}`, skill: event } }
  }
}
