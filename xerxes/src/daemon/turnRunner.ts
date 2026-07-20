// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { AgentDefinition } from '../agents/definitions.js'
import type { AuditEmitter } from '../audit/emitter.js'
import { estimateContextTokens } from '../context/windowUsage.js'
import { ValidationError } from '../core/errors.js'
import type { ToolExecutor } from '../executors/toolRegistry.js'
import type { AgentMemory } from '../memory/agentMemory.js'
import {
  mergePersistedSubagentSnapshots,
  persistedSubagentSnapshotValues,
} from '../agents/subagentPersistence.js'
import { SUBAGENT_BLOCKED_TOOLS } from '../agents/subagentManager.js'
import type { AgentSelfMemory } from '../memory/agentSelfMemory.js'
import { makeTurnIndexerHook } from '../memory/turnIndexer.js'
import type { Memory } from '../memory/base.js'
import type { SpawnedAgentSnapshot } from '../operators/subagents.js'
import type { LlmClient } from '../llms/client.js'
import { getContextLimit } from '../llms/providerRegistry.js'
import { agentNameForMode, modeSwitchHint, normalizeInteractionMode } from '../runtime/interactionModes.js'
import { withActiveSession } from '../runtime/sessionContext.js'
import { resolveTurnThinking } from '../runtime/thinkingLevels.js'
import { captureUserWorkflowMemory } from '../runtime/workflowMemory.js'
import { createAgentState, type AgentState, type StreamEvent } from '../streaming/events.js'
import { runTurn } from '../streaming/loop.js'
import {
  DEFAULT_PERMISSION_MODE,
  type PermissionBroker,
  type PermissionMode,
  type ToolPolicy,
} from '../streaming/permissions.js'
import type { ChatMessage, MessageContent } from '../types/messages.js'
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
  readonly permissionMode?: PermissionMode
  readonly policy?: ToolPolicy
  /** Session default effort hint for reasoning APIs; per-turn keywords and ultra mode override it. */
  readonly reasoningEffort?: string
  /** Session default for extended thinking; false disables it unless a turn escalates. */
  readonly thinking?: boolean
  /** Session default thinking token budget. */
  readonly thinkingBudget?: number
  /** Session-scoped delegated-turn events rendered alongside the parent turn. */
  readonly subagentEvents?: DaemonSubagentEventSource
  /** Joins explicitly detached child work back into the creating parent turn. */
  readonly subagentCoordinator?: SubagentTurnCoordinator
  readonly toolExecutor?: ToolExecutor
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
    const projectRoot = sessionProjectRoot(session)
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
    const configuredPermissionMode = permissionModeForInteraction(session.interactionMode, this.options.permissionMode)
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
    const memoryPrompt = memory ? await memory.toPromptSection() : ''
    const selfMemory = this.options.agentSelfMemory ? await this.options.agentSelfMemory(session) : undefined
    const selfMemoryPrompt = selfMemory ? await selfMemory.systemPromptAddendum() : ''
    const recoveredSubagents = this.options.subagentCoordinator
      ? recoverSubagentSnapshots(
        session.messages,
        session.id,
        persistedSubagentSnapshotValues(session.metadata),
      )
      : []
    const restoredSubagentCount = this.options.subagentCoordinator
      ?.restore?.(session.id, recoveredSubagents) ?? 0
    const systemPrompt = [
      bootstrapPrompt,
      promptAgent?.systemPrompt,
      modeSwitchHint(
        session.interactionMode,
        tools?.some(tool => tool.function.name === 'SetInteractionModeTool') ?? false,
      ),
      this.options.subagentCoordinator
        ? 'Background subagents are joined before the parent turn ends. Integrate their delivered results in this turn; do not promise synthesis in a later turn.'
        : '',
      restoredSubagentCount
        ? `${restoredSubagentCount} delegated task handle(s) were recovered from this resumed transcript after their daemon process ended. TaskListTool, TaskGetTool, PeekAgent, and AwaitAgents expose honest terminal snapshots: completed output is preserved, while work last seen active is marked interrupted and must be explicitly restarted with ResetAgent or respawned. Do not retry stale ids as if they were still running.`
        : '',
      memoryPrompt,
      selfMemoryPrompt,
      systemPromptAddendum(session),
    ]
      .filter(Boolean)
      .join('\n\n')
    const permissionBroker = this.options.interactions?.permissionBroker(session.id) ?? this.options.permissionBroker
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
    const thinking = resolveTurnThinking({
      defaults: {
        ...(this.options.thinking !== undefined ? { enabled: this.options.thinking } : {}),
        ...(this.options.thinkingBudget !== undefined ? { budgetTokens: this.options.thinkingBudget } : {}),
        ...(this.options.reasoningEffort !== undefined ? { effort: this.options.reasoningEffort } : {}),
      },
      prompt: text,
      ultraMode: session.ultraMode === true,
    })
    try {
      const turnEvents = withActiveSession(session, runTurn({
        agentId: promptAgent?.name ?? session.agentId,
        interactionMode: session.interactionMode,
        model,
        sessionId: session.id,
        state,
        userMessage: text,
        ...(this.options.maxTokens !== undefined ? { maxTokens: this.options.maxTokens } : {}),
        permissionMode,
        ...(this.options.temperature !== undefined ? { temperature: this.options.temperature } : {}),
        ...(thinking === undefined ? {} : { thinking: { budgetTokens: thinking.budgetTokens, effort: thinking.effort } }),
        ...(this.options.topK !== undefined ? { topK: this.options.topK } : {}),
        ...(tools ? { tools } : {}),
        ...(systemPrompt ? { systemPrompt } : {}),
        ...(this.options.topP !== undefined ? { topP: this.options.topP } : {}),
      }, {
        ...(subagentCohort ? {
          awaitAgentEvents: async signal => {
            const snapshots = await subagentCohort.waitForResults(signal)
            mergePersistedSubagentSnapshots(state.metadata, snapshots)
            return formatSubagentResults(snapshots)
          },
        } : {}),
        ...(controls.drainSteer ? { drainSteer: controls.drainSteer } : {}),
        llm: this.options.llm,
        ...(permissionBroker ? { permissionBroker } : {}),
        ...(this.options.policy ? { policy: this.options.policy } : {}),
        ...(toolExecutor ? { toolExecutor } : {}),
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
        yield daemonEventFromStream(event, state, session)
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
      recordLatestUserDisplayText(state, text, displayText)
      synchronizeSessionState(session, state)
    }
  }

  stateFor(sessionId: string): AgentState | undefined {
    return this.states.get(sessionId)
  }

  dropSession(sessionId: string): void {
    this.states.delete(sessionId)
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
    const key = [session.cwd, model, session.agentId, agentId, toolSignature].join('\u0000')
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

const MAX_SUBAGENT_RESULT_CHARS = 64_000
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
    unsubscribe()
    await iterator.return?.()
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
    if (message?.role !== 'user' || message.content !== providerText) continue
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

function daemonEventFromStream(event: StreamEvent, state: AgentState, session: DaemonSession): DaemonEvent {
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
    case 'turn_done': {
      const contextTokens = estimateContextTokens(
        state.messages.map(message => ({ role: message.role, content: message.content })),
        { model: event.model },
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
          max_context: getContextLimit(event.model),
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
