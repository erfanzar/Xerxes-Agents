// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  errorMessage,
  type ToolExecutor,
  type ToolExecutionContext,
} from '../executors/toolRegistry.js'
import type { HookPoint, HookRunner } from '../extensions/hooks.js'
import type { LlmClient, LlmDelta, ThinkingRequest, TokenUsage } from '../llms/client.js'
import { classifyError } from '../runtime/errorClassifier.js'
import {
  inspectObjectiveResponse,
  objectiveGuardRetryLimit,
  type ObjectiveToolExecutionEvidence,
} from '../runtime/objectiveGuard.js'
import type { ChatMessage } from '../types/messages.js'
import { isJsonObject, type ToolCall, type ToolDefinition } from '../types/toolCalls.js'
import type {
  AgentState,
  PermissionRequest,
  StreamEvent,
  ToolResult,
} from './events.js'
import {
  DEFAULT_PERMISSION_MODE,
  deniedResult,
  permissionDisposition,
  permissionDescription,
  type PermissionBroker,
  type PermissionMode,
  type ToolPolicy,
} from './permissions.js'
import { ThinkingParser, type ThinkingStreamParser } from './thinkingParser.js'

/** Distinct productive tool rounds are unbounded unless the caller opts into a budget. */
export const DEFAULT_MAX_TOOL_TURNS = Number.POSITIVE_INFINITY
export const DEFAULT_RETRY_DELAYS = [1_000, 2_000] as const
/** Generous default chunk-arrival budget before a provider stream is treated as stalled. */
export const DEFAULT_STREAM_INACTIVITY_TIMEOUT_MS = 120_000
/** Cap provider-suggested Retry-After waits so a bad hint cannot park a turn for hours. */
export const MAX_SUGGESTED_RETRY_DELAY_MS = 60_000

export interface TurnRequest {
  readonly agentId?: string
  /** Session interaction mode; objective mode rejects unsupported narrative stops. */
  readonly interactionMode?: string
  readonly maxToolTurns?: number
  readonly maxTokens?: number
  readonly model: string
  /** Optional maximum retries for objective-mode text-only stopping attempts. */
  readonly objectiveGuardMaxRetries?: number
  readonly permissionMode?: PermissionMode
  readonly sessionId?: string
  readonly state: AgentState
  readonly systemPrompt?: string
  readonly temperature?: number
  /**
   * Per-turn extended-thinking directive; adapters map it to provider wire
   * fields. WHY a neutral ThinkingRequest type instead of extraBody: thinking
   * is expressed per provider dialect (reasoning_effort / thinking_budget on
   * OpenAI-compatible transports, thinking.budget_tokens on Anthropic), and
   * extraBody only merges into OpenAI-style payloads. A single typed,
   * provider-neutral shape keeps the resolution in runtime/thinkingLevels.ts
   * decoupled from wire formats, stays type-checked, and lets every adapter
   * translate the same directive into its own dialect.
   */
  readonly thinking?: ThinkingRequest
  readonly topK?: number
  readonly topP?: number
  readonly tools?: readonly ToolDefinition[]
  readonly userMessage: string
}

export interface TurnDependencies {
  /** Waits for explicitly backgrounded subagents before a text-only stop. */
  readonly awaitAgentEvents?: (signal?: AbortSignal) => Promise<readonly string[]>
  readonly delay?: (milliseconds: number, signal?: AbortSignal) => Promise<void>
  /** Supplies passive sub-agent status lines at safe provider/tool boundaries. */
  readonly drainAgentEvents?: () => readonly string[]
  /** Supplies steering text at safe provider/tool boundaries for daemon turns. */
  readonly drainSteer?: () => readonly string[]
  /** Optional plugin hook dispatch surface; when absent the turn dispatches no hooks. */
  readonly hookRunner?: HookRunner
  readonly llm: LlmClient
  /**
   * Observes provider-requested tools that were not included in the model-visible
   * surface. Returning `stop` ends the current turn without executing or retrying
   * those calls.
   */
  readonly onUnconfiguredToolCalls?: (calls: readonly ToolCall[]) => 'continue' | 'stop'
  readonly permissionBroker?: PermissionBroker
  readonly policy?: ToolPolicy
  readonly retryDelays?: readonly number[]
  /**
   * Abort a provider attempt that yields no chunk within this budget (ms), so a
   * socket held open without data stalls one attempt instead of the whole turn.
   * Defaults to {@link DEFAULT_STREAM_INACTIVITY_TIMEOUT_MS}; non-positive or
   * non-finite values disable the watchdog.
   */
  readonly streamInactivityTimeoutMs?: number
  /** Override parser behavior for an isolated diagnostic streaming surface. */
  readonly thinkingParserFactory?: () => ThinkingStreamParser
  readonly toolExecutor?: ToolExecutor
}

/**
 * Drive a complete agent turn. Events are fully serializable so daemon, HTTP,
 * MCP, channels, and the OpenTUI client share one internal vocabulary.
 */
export async function* runTurn(
  request: TurnRequest,
  dependencies: TurnDependencies,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const state = request.state
  const permissionMode = request.permissionMode ?? DEFAULT_PERMISSION_MODE
  const maxToolTurns = request.maxToolTurns ?? DEFAULT_MAX_TOOL_TURNS
  const retryDelays = dependencies.retryDelays ?? DEFAULT_RETRY_DELAYS
  const streamInactivityTimeoutMs =
    dependencies.streamInactivityTimeoutMs ?? DEFAULT_STREAM_INACTIVITY_TIMEOUT_MS
  const hookRunner = dependencies.hookRunner
  const toolContext: ToolExecutionContext = {
    ...(request.agentId ? { agentId: request.agentId } : {}),
    ...(request.sessionId ? { sessionId: request.sessionId } : {}),
    metadata: state.metadata,
  }
  ensureSystemPrompt(state.messages, request.systemPrompt)
  state.messages.push({ role: 'user', content: request.userMessage })
  state.metadata.model = request.model
  state.turnCount += 1
  await dispatchHook(hookRunner, 'on_turn_start', {
    ...(request.agentId ? { agentId: request.agentId } : {}),
    model: request.model,
    ...(request.sessionId ? { sessionId: request.sessionId } : {}),
    turnCount: state.turnCount,
    userMessage: request.userMessage,
  })

  let inputTokens = 0
  let outputTokens = 0
  let cacheReadTokens = 0
  let cacheCreationTokens = 0
  let reasoningTokens = 0
  let reasoningUsageComplete = true
  let usageComplete = true
  let apiCallsCount = 0
  let objectiveGuardRetries = 0
  const objectiveToolExecutions: ObjectiveToolExecutionEvidence[] = []
  let toolCallsCount = 0
  let forceToolFreeSummary = false
  let latestToolRoundText: string | undefined
  let turnLimit = maxToolTurns
  const objectiveGuardLimit = objectiveGuardRetryLimit(
    request.objectiveGuardMaxRetries === undefined
      ? {}
      : { objective_guard_max_retries: request.objectiveGuardMaxRetries },
  )
  /** Record one tool result, letting tool_result_persist hooks rewrite it first. */
  const recordToolResult = async (result: ToolResult, call: ToolCall): Promise<ToolResult> => {
    if (hookRunner === undefined) {
      appendToolResult(state, result, call, objectiveToolExecutions)
      return result
    }
    const mutated = hookMutation(await dispatchHook(hookRunner, 'tool_result_persist', {
      name: result.name,
      permitted: result.permitted,
      result: result.result,
      toolCallId: result.toolCallId,
    }), result.result)
    const recorded = typeof mutated === 'string' && mutated !== result.result
      ? { ...result, result: mutated }
      : result
    appendToolResult(state, recorded, call, objectiveToolExecutions)
    return recorded
  }

  for (let toolTurn = 0; toolTurn < turnLimit; toolTurn += 1) {
    appendAgentEventMessage(state, dependencies.drainAgentEvents?.())
    for (const steer of dependencies.drainSteer?.() ?? []) {
      const content = steer.trim()
      if (content) {
        state.messages.push({
          role: 'user',
          content: `[steer from user]\n${content}`,
        })
      }
    }
    // Per-attempt accumulators sit at round scope so the surviving attempt is
    // readable after the retry loop, but every attempt starts from a clean
    // slate: partial text, thinking, usage, and tool calls from a failed
    // attempt must never leak into the persisted assistant message.
    let parser = dependencies.thinkingParserFactory?.() ?? new ThinkingParser()
    let textParts: string[] = []
    let thinkingParts: string[] = []
    let thinkingSignature: string | undefined
    let roundToolCalls: readonly ToolCall[] = []
    let lastUsage: TokenUsage | undefined
    let streamCompleted = false
    let textDeduper = new ToolRoundTextDeduper(latestToolRoundText)

    for (let attempt = 0; attempt <= retryDelays.length; attempt += 1) {
      parser = dependencies.thinkingParserFactory?.() ?? new ThinkingParser()
      textParts = []
      thinkingParts = []
      thinkingSignature = undefined
      roundToolCalls = []
      lastUsage = undefined
      textDeduper = new ToolRoundTextDeduper(latestToolRoundText)
      const attemptSignal = linkAttemptSignal(signal)
      try {
        apiCallsCount += 1
        for await (const delta of watchProviderStream(
          dependencies.llm.stream(
            completionRequest(
              request,
              state.messages,
              forceToolFreeSummary ? [] : request.tools,
            ),
            attemptSignal.controller.signal,
          ),
          streamInactivityTimeoutMs,
          attemptSignal,
        )) {
          const parts = processDelta(delta, parser, textParts, thinkingParts)
          for (const part of parts) {
            for (const visible of textDeduper.push(part)) yield visible
          }
          if (delta.toolCalls) {
            roundToolCalls = delta.toolCalls
          }
          if (delta.thinkingSignature) {
            thinkingSignature = delta.thinkingSignature
          }
          if (delta.usage) {
            lastUsage = mergeUsage(lastUsage, delta.usage)
          }
        }
        for (const flushed of parser.process('')) {
          if (flushed.type === 'text') {
            textParts.push(flushed.text)
            for (const visible of textDeduper.push({ type: 'text', text: flushed.text })) yield visible
          } else {
            thinkingParts.push(flushed.text)
            yield { type: 'thinking', text: flushed.text }
          }
        }
        streamCompleted = true
        break
      } catch (error) {
        // A failed provider attempt may have consumed tokens without returning
        // usage. Keep the exact API-call count, but do not present later
        // successful-round usage as a complete total for the turn.
        usageComplete = false
        reasoningUsageComplete = false
        const classified = classifyError(error)
        await dispatchHook(hookRunner, 'on_error', {
          ...(request.agentId ? { agentId: request.agentId } : {}),
          attempt: attempt + 1,
          error: errorMessage(error),
          kind: classified.kind,
          ...(request.sessionId ? { sessionId: request.sessionId } : {}),
        })
        // Only transient failures earn another attempt. Auth, validation,
        // configuration, and other terminal errors fail the round at once.
        const final = attempt === retryDelays.length
          || !classified.retryable
          || signal?.aborted === true
        const suggestedDelay = classified.suggestedBackoffSeconds === undefined
          ? 0
          : Math.min(MAX_SUGGESTED_RETRY_DELAY_MS, classified.suggestedBackoffSeconds * 1_000)
        const delay = final ? 0 : Math.max(retryDelays[attempt] ?? 0, suggestedDelay)
        yield {
          type: 'provider_retry',
          error: errorMessage(error),
          attempt: attempt + 1,
          maxAttempts: retryDelays.length + 1,
          delay,
          final,
        }
        if (final) {
          const errorText = `[Error: ${errorMessage(error)}]`
          textParts.push(errorText)
          yield { type: 'text', text: errorText }
          streamCompleted = true
          break
        }
        await (dependencies.delay ?? defaultDelay)(delay, signal)
      } finally {
        attemptSignal.release()
      }
    }

    if (!streamCompleted) {
      throw new Error('LLM stream exited without completion or error')
    }

    if (lastUsage === undefined) {
      usageComplete = false
      reasoningUsageComplete = false
    } else if (lastUsage.reasoningTokens === undefined) {
      reasoningUsageComplete = false
    }
    accumulateUsage(lastUsage, state, (usage) => {
      inputTokens += usage.inputTokens
      outputTokens += usage.outputTokens
      cacheReadTokens += usage.cacheReadTokens ?? 0
      cacheCreationTokens += usage.cacheCreationTokens ?? 0
      reasoningTokens += usage.reasoningTokens ?? 0
    })

    const rawAssistantText = textParts.join('')
    const deduplication = textDeduper.finish()
    for (const visible of deduplication.events) yield visible
    const assistantText = rawAssistantText.slice(deduplication.suppressedPrefix)
    const providerToolCalls = roundToolCalls
    const visibleTools = forceToolFreeSummary ? [] : request.tools
    const { exposed, unconfigured } = partitionToolCalls(providerToolCalls, visibleTools)
    roundToolCalls = exposed
    const assistant: ChatMessage = {
      role: 'assistant',
      content: assistantText,
      ...(thinkingParts.length ? { thinking: thinkingParts.join('') } : {}),
      ...(thinkingSignature ? { thinking_signature: thinkingSignature } : {}),
      ...(providerToolCalls.length ? { tool_calls: providerToolCalls } : {}),
    }
    state.messages.push(assistant)
    if (providerToolCalls.length && assistantText) {
      latestToolRoundText = assistantText
    }
    if (thinkingParts.length) {
      state.thinkingContent.push(thinkingParts.join(''))
    } else {
      state.thinkingContent.push('')
    }

    if (unconfigured.length) {
      toolCallsCount += unconfigured.length
      for (const call of unconfigured) {
        const result = await recordToolResult(unconfiguredToolResult(call), call)
        yield { type: 'tool_end', result }
      }
      if (dependencies.onUnconfiguredToolCalls?.(unconfigured) === 'stop') {
        break
      }
      if (forceToolFreeSummary) {
        break
      }
      if (!roundToolCalls.length) {
        continue
      }
    }

    if (!roundToolCalls.length) {
      const agentEvents = await dependencies.awaitAgentEvents?.(signal) ?? []
      const appendedAgentEvents = appendAgentEventMessage(state, agentEvents)
      // A coordinator can acknowledge returned snapshots before cancellation
      // becomes observable here. Persist the delivered results first so an
      // interrupted parent either synthesizes them now or receives them from
      // its durable history on the next turn.
      if (signal?.aborted) break
      if (appendedAgentEvents) {
        if (toolTurn + 1 >= turnLimit) turnLimit += 1
        continue
      }
      for (const steer of dependencies.drainSteer?.() ?? []) {
        const content = steer.trim()
        if (!content) continue
        state.messages.push({
          role: 'user',
          content: `[steer from user saved for next turn]\n${content}`,
        })
        yield {
          type: 'text',
          text: `\n[Steer saved for next turn: ${content}]`,
        }
      }
      const objectiveDecision = inspectObjectiveResponse(assistantText, {
        evidence: { toolExecutions: objectiveToolExecutions },
        mode: currentInteractionMode(state, request.interactionMode),
      })
      if (!objectiveDecision.shouldContinue) {
        break
      }
      objectiveGuardRetries += 1
      if (objectiveGuardRetries > objectiveGuardLimit) {
        yield {
          type: 'text',
          text:
            '\n[Stopped: objective guard could not get a verified completion or concrete blocker after ' +
            objectiveGuardLimit +
            ' retries. The last issue was: ' +
            objectiveDecision.reason +
            '.]',
        }
        break
      }
      state.messages.push({ role: 'user', content: objectiveDecision.reminder })
      yield {
        type: 'text',
        text:
          '\n[Objective gate: ' + objectiveDecision.reason + '. Continuing.]',
      }
      continue
    }

    toolCallsCount += roundToolCalls.length
    for (let index = 0; index < roundToolCalls.length; index += 1) {
      const call = roundToolCalls[index]
      if (!call) {
        continue
      }
      if (signal?.aborted) {
        for (const cancelled of roundToolCalls.slice(index)) {
          const result = await recordToolResult(cancelledToolResult(cancelled), cancelled)
          yield { type: 'tool_end', result }
        }
        break
      }
      const permission = permissionDisposition(
        call,
        permissionMode,
        dependencies.policy,
        request.agentId,
      )
      if (permission === 'deny') {
        const result = await recordToolResult(deniedToolResult(call), call)
        yield { type: 'tool_end', result }
        continue
      }
      if (permission === 'prompt') {
        const permissionRequest = createPermissionRequest(call)
        yield { type: 'permission_request', request: permissionRequest }
        const decision =
          (await dependencies.permissionBroker?.request(
            permissionRequest,
            signal,
          )) ?? 'reject'
        // Injected brokers are allowed to resolve asynchronously. Cancellation
        // may land while a prompt is open, so an approval that races the abort
        // must not start a privileged tool with an already-aborted signal.
        if (signal?.aborted) {
          const result = await recordToolResult(cancelledToolResult(call), call)
          yield { type: 'tool_end', result }
          continue
        }
        if (decision === 'reject') {
          const result = await recordToolResult(deniedToolResult(call), call)
          yield { type: 'tool_end', result }
          continue
        }
      }

      const beforeResult = await dispatchHook(hookRunner, 'before_tool_call', {
        ...(request.agentId ? { agentId: request.agentId } : {}),
        arguments: call.function.arguments,
        name: call.function.name,
        ...(request.sessionId ? { sessionId: request.sessionId } : {}),
        toolCallId: call.id,
      })
      const effectiveCall = applyToolArgumentsMutation(
        call,
        hookMutation(beforeResult, call.function.arguments),
      )

      yield { type: 'tool_start', call: effectiveCall }
      const startedAt = performance.now()
      try {
        const output = dependencies.toolExecutor
          ? await dependencies.toolExecutor.execute(effectiveCall, toolContext, signal)
          : `Tool ${effectiveCall.function.name} is unavailable.`
        let result: ToolResult = {
          name: effectiveCall.function.name,
          result: output,
          permitted: true,
          toolCallId: effectiveCall.id,
          durationMs: performance.now() - startedAt,
        }
        const afterResult = await dispatchHook(hookRunner, 'after_tool_call', {
          ...(request.agentId ? { agentId: request.agentId } : {}),
          arguments: effectiveCall.function.arguments,
          name: effectiveCall.function.name,
          result: result.result,
          ...(request.sessionId ? { sessionId: request.sessionId } : {}),
          toolCallId: effectiveCall.id,
        })
        const mutatedOutput = hookMutation(afterResult, result.result)
        if (typeof mutatedOutput === 'string' && mutatedOutput !== result.result) {
          result = { ...result, result: mutatedOutput }
        }
        const recorded = await recordToolResult(result, effectiveCall)
        yield { type: 'tool_end', result: recorded }
      } catch (error) {
        const result = await recordToolResult(
          failedToolResult(effectiveCall, error, performance.now() - startedAt),
          effectiveCall,
        )
        yield { type: 'tool_end', result }
      }
    }
    if (signal?.aborted) {
      break
    }
    let needsFinalization = false
    if (toolTurn + 1 >= turnLimit) {
      const agentEvents = await dependencies.awaitAgentEvents?.(signal) ?? []
      const appendedAgentEvents = appendAgentEventMessage(state, agentEvents)
      if (signal?.aborted) break
      if (appendedAgentEvents) {
        forceToolFreeSummary = true
        needsFinalization = true
      }
      if (needsFinalization) turnLimit += 1
    }
  }

  state.totalApiCalls += apiCallsCount
  state.usageComplete &&= usageComplete
  await dispatchHook(hookRunner, 'on_turn_end', {
    ...(request.agentId ? { agentId: request.agentId } : {}),
    apiCallsCount,
    model: request.model,
    ...(request.sessionId ? { sessionId: request.sessionId } : {}),
    toolCallsCount,
    turnCount: state.turnCount,
    usageComplete,
  })
  yield {
    type: 'turn_done',
    apiCallsCount,
    model: request.model,
    toolCallsCount,
    usageComplete,
    usage: {
      inputTokens,
      outputTokens,
      ...(cacheReadTokens ? { cacheReadTokens } : {}),
      ...(cacheCreationTokens ? { cacheCreationTokens } : {}),
      ...(usageComplete && reasoningUsageComplete ? { reasoningTokens } : {}),
    },
  }
}

function currentInteractionMode(state: AgentState, initialMode: string | undefined): string {
  const liveMode = state.metadata.interaction_mode
  return typeof liveMode === 'string' && liveMode.trim() ? liveMode : initialMode ?? 'code'
}

type IncrementalTextEvent = Extract<StreamEvent, { readonly type: 'text' | 'thinking' }>

/**
 * Hold only the prefix that could be an exact replay of the latest assistant
 * text attached to a tool round. As soon as it differs, normal streaming
 * resumes. Exact repeats stay suppressed so live output and saved history show
 * the same assistant text once.
 */
class ToolRoundTextDeduper {
  private candidate = ''
  private diverged = false
  private readonly overlapLengths: readonly number[]
  private readonly pending: IncrementalTextEvent[] = []
  private suppressedPrefix = 0

  constructor(private readonly previous: string | undefined) {
    this.overlapLengths = previous === undefined ? [] : eligibleTextOverlapLengths(previous)
  }

  push(event: IncrementalTextEvent): readonly IncrementalTextEvent[] {
    if (event.type === 'thinking' || this.previous === undefined || this.diverged) {
      return [event]
    }
    this.candidate += event.text
    this.pending.push(event)
    const viable = this.overlapLengths.filter(length => {
      const suffix = this.previous?.slice(-length) ?? ''
      const compared = Math.min(length, this.candidate.length)
      return this.candidate.slice(0, compared) === suffix.slice(0, compared)
    })
    if (viable.some(length => length >= this.candidate.length)) return []

    const overlap = Math.max(0, ...viable.filter(length => this.candidate.startsWith(this.previous?.slice(-length) ?? '')))
    this.diverged = true
    this.suppressedPrefix = overlap
    return stripTextEventPrefix(this.pending.splice(0), overlap)
  }

  finish(): { readonly events: readonly IncrementalTextEvent[]; readonly suppressedPrefix: number } {
    if (!this.diverged && this.previous !== undefined) {
      this.suppressedPrefix = Math.max(
        0,
        ...this.overlapLengths.filter(length => (
          length <= this.candidate.length
          && this.candidate.startsWith(this.previous?.slice(-length) ?? '')
        )),
      )
    }
    return {
      events: stripTextEventPrefix(this.pending.splice(0), this.suppressedPrefix),
      suppressedPrefix: this.suppressedPrefix,
    }
  }
}

const MIN_PARTIAL_TEXT_OVERLAP = 12

function eligibleTextOverlapLengths(previous: string): number[] {
  const lengths: number[] = []
  for (let length = MIN_PARTIAL_TEXT_OVERLAP; length <= previous.length; length += 1) {
    lengths.push(length)
  }
  if (previous.length > 0 && previous.length < MIN_PARTIAL_TEXT_OVERLAP) {
    lengths.push(previous.length)
  }
  return lengths
}

function stripTextEventPrefix(
  events: readonly IncrementalTextEvent[],
  count: number,
): IncrementalTextEvent[] {
  let remaining = count
  const visible: IncrementalTextEvent[] = []
  for (const event of events) {
    if (event.type === 'thinking' || remaining === 0) {
      visible.push(event)
      continue
    }
    if (event.text.length <= remaining) {
      remaining -= event.text.length
      continue
    }
    visible.push({ type: 'text', text: event.text.slice(remaining) })
    remaining = 0
  }
  return visible
}

function appendAgentEventMessage(
  state: AgentState,
  events: readonly string[] | undefined,
): boolean {
  if (!events?.length) return false
  const lines: string[] = []
  for (const event of events) {
    const line = event.trim()
    if (line) lines.push(line)
  }
  if (lines.length) {
    state.messages.push({
      role: 'user',
      content: `[sub-agent events]\n${lines.join('\n')}`,
    })
    return true
  }
  return false
}

function completionRequest(
  request: TurnRequest,
  messages: readonly ChatMessage[],
  tools: readonly ToolDefinition[] | undefined,
) {
  return {
    model: request.model,
    messages: [...messages],
    ...(tools?.length ? { tools } : {}),
    ...(request.maxTokens !== undefined
      ? { maxTokens: request.maxTokens }
      : {}),
    ...(request.temperature !== undefined
      ? { temperature: request.temperature }
      : {}),
    ...(request.topK !== undefined ? { topK: request.topK } : {}),
    ...(request.topP !== undefined ? { topP: request.topP } : {}),
    // Passthrough, not translation: the resolved per-turn directive travels
    // untouched from the TurnRequest to the CompletionRequest so the owning
    // provider adapter (client.ts addSampling, anthropic.ts) is the single
    // place that maps it onto wire-specific fields.
    ...(request.thinking !== undefined ? { thinking: request.thinking } : {}),
  }
}

function ensureSystemPrompt(
  messages: ChatMessage[],
  systemPrompt: string | undefined,
): void {
  if (!systemPrompt) return
  const index = messages.findIndex((message) => message.role === 'system')
  if (index < 0) {
    messages.unshift({ role: 'system', content: systemPrompt })
    return
  }
  messages[index] = { role: 'system', content: systemPrompt }
}

/** Separate model-visible calls from provider calls outside the configured surface. */
function partitionToolCalls(
  calls: readonly ToolCall[],
  tools: readonly ToolDefinition[] | undefined,
): { readonly exposed: readonly ToolCall[]; readonly unconfigured: readonly ToolCall[] } {
  if (!calls.length) return { exposed: calls, unconfigured: calls }
  const exposedNames = new Set((tools ?? []).map((tool) => tool.function.name))
  const exposed: ToolCall[] = []
  const unconfigured: ToolCall[] = []
  for (const call of calls) {
    if (exposedNames.has(call.function.name)) {
      exposed.push(call)
    } else {
      unconfigured.push(call)
    }
  }
  return { exposed, unconfigured }
}

function processDelta(
  delta: LlmDelta,
  parser: ThinkingStreamParser,
  textParts: string[],
  thinkingParts: string[],
): IncrementalTextEvent[] {
  const events: IncrementalTextEvent[] = []
  if (delta.thinking) {
    thinkingParts.push(delta.thinking)
    events.push({ type: 'thinking', text: delta.thinking })
  }
  if (delta.content) {
    for (const part of parser.process(delta.content)) {
      if (part.type === 'text') {
        textParts.push(part.text)
        events.push({ type: 'text', text: part.text })
      } else {
        thinkingParts.push(part.text)
        events.push({ type: 'thinking', text: part.text })
      }
    }
  }
  return events
}

function accumulateUsage(
  usage: TokenUsage | undefined,
  state: AgentState,
  receive: (usage: TokenUsage) => void,
): void {
  if (!usage) {
    return
  }
  state.totalInputTokens += usage.inputTokens
  state.totalOutputTokens += usage.outputTokens
  state.totalCacheReadTokens += usage.cacheReadTokens ?? 0
  state.totalCacheCreationTokens += usage.cacheCreationTokens ?? 0
  receive(usage)
}

function mergeUsage(
  existing: TokenUsage | undefined,
  incoming: TokenUsage,
): TokenUsage {
  if (!existing) {
    return incoming
  }
  return {
    inputTokens: incoming.inputTokens || existing.inputTokens,
    outputTokens: incoming.outputTokens || existing.outputTokens,
    ...((incoming.cacheReadTokens ?? existing.cacheReadTokens)
      ? {
          cacheReadTokens: incoming.cacheReadTokens ?? existing.cacheReadTokens,
        }
      : {}),
    ...((incoming.cacheCreationTokens ?? existing.cacheCreationTokens)
      ? {
          cacheCreationTokens:
            incoming.cacheCreationTokens ?? existing.cacheCreationTokens,
        }
      : {}),
    ...((incoming.reasoningTokens ?? existing.reasoningTokens) !== undefined
      ? { reasoningTokens: incoming.reasoningTokens ?? existing.reasoningTokens }
      : {}),
  }
}

function createPermissionRequest(call: ToolCall): PermissionRequest {
  return {
    requestId: `${call.id}:${crypto.randomUUID()}`,
    toolCall: call,
    inputs: call.function.arguments,
    description: permissionDescription(call),
  }
}

function appendToolResult(
  state: AgentState,
  result: ToolResult,
  call: ToolCall,
  objectiveToolExecutions: ObjectiveToolExecutionEvidence[],
): void {
  state.messages.push({
    role: 'tool',
    content: result.result,
    name: result.name,
    tool_call_id: result.toolCallId,
  })
  const execution = {
    name: result.name,
    inputs: call.function.arguments,
    result: result.result,
    permitted: result.permitted,
    toolCallId: result.toolCallId,
    durationMs: result.durationMs,
  }
  state.toolExecutions.push(execution)
  objectiveToolExecutions.push(execution)
}

function deniedToolResult(call: ToolCall): ToolResult {
  return {
    name: call.function.name,
    result: deniedResult(call),
    permitted: false,
    toolCallId: call.id,
    durationMs: 0,
  }
}

function cancelledToolResult(call: ToolCall): ToolResult {
  return {
    name: call.function.name,
    result: 'Cancelled before execution.',
    permitted: false,
    toolCallId: call.id,
    durationMs: 0,
  }
}

function failedToolResult(
  call: ToolCall,
  error: unknown,
  durationMs = 0,
): ToolResult {
  return {
    name: call.function.name,
    result: `Tool execution failed: ${errorMessage(error)}`,
    permitted: true,
    toolCallId: call.id,
    durationMs,
  }
}

function unconfiguredToolResult(call: ToolCall): ToolResult {
  return {
    name: call.function.name,
    result: `Tool execution failed: ${call.function.name} was not configured for this turn.`,
    permitted: true,
    toolCallId: call.id,
    durationMs: 0,
  }
}

function defaultDelay(
  milliseconds: number,
  signal?: AbortSignal,
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason)
      return
    }
    const timer = setTimeout(resolve, milliseconds)
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(timer)
        reject(signal.reason)
      },
      { once: true },
    )
  })
}

/** Raised when a provider stream yields no chunk inside the inactivity budget. */
export class StreamInactivityError extends Error {
  readonly timeoutMs: number

  constructor(timeoutMs: number) {
    super(`Provider stream stalled: no chunk received within ${timeoutMs}ms (stream inactivity timeout)`)
    this.name = 'StreamInactivityError'
    this.timeoutMs = timeoutMs
  }
}

interface AttemptSignal {
  readonly controller: AbortController
  readonly release: () => void
}

/** Chain a per-attempt controller to the caller's signal so the watchdog can cancel one attempt only. */
function linkAttemptSignal(signal: AbortSignal | undefined): AttemptSignal {
  const controller = new AbortController()
  if (signal === undefined) {
    return { controller, release: () => undefined }
  }
  if (signal.aborted) {
    controller.abort(signal.reason)
    return { controller, release: () => undefined }
  }
  const forward = () => controller.abort(signal.reason)
  signal.addEventListener('abort', forward, { once: true })
  return { controller, release: () => signal.removeEventListener('abort', forward) }
}

/**
 * Wrap provider iteration with an inactivity watchdog. The timer only exists
 * while waiting for the next chunk, so it cannot fire during tool execution,
 * and it is cleared on completion, error, and consumer-close paths.
 */
async function* watchProviderStream(
  stream: AsyncIterable<LlmDelta>,
  timeoutMs: number,
  attempt: AttemptSignal,
): AsyncGenerator<LlmDelta> {
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
    yield* stream
    return
  }
  const iterator = stream[Symbol.asyncIterator]()
  try {
    while (true) {
      const result = await nextDeltaWithTimeout(iterator, timeoutMs, attempt)
      if (result.done) {
        return
      }
      yield result.value
    }
  } finally {
    // Best-effort release of the abandoned provider stream. A stalled provider
    // may never settle its pending read, so cleanup must not block retry.
    void iterator.return?.()?.catch(() => undefined)
  }
}

async function nextDeltaWithTimeout(
  iterator: AsyncIterator<LlmDelta>,
  timeoutMs: number,
  attempt: AttemptSignal,
): Promise<IteratorResult<LlmDelta>> {
  const signal = attempt.controller.signal
  let timer: ReturnType<typeof setTimeout> | undefined
  let onAbort: (() => void) | undefined
  const pending = iterator.next()
  try {
    return await Promise.race([
      pending,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => {
          const error = new StreamInactivityError(timeoutMs)
          attempt.controller.abort(error)
          reject(error)
        }, timeoutMs)
      }),
      new Promise<never>((_, reject) => {
        if (signal.aborted) {
          reject(signal.reason)
          return
        }
        onAbort = () => reject(signal.reason)
        signal.addEventListener('abort', onAbort, { once: true })
      }),
    ])
  } finally {
    if (timer !== undefined) {
      clearTimeout(timer)
    }
    if (onAbort !== undefined) {
      signal.removeEventListener('abort', onAbort)
    }
    // A timed-out or aborted attempt abandons the pending provider read; keep
    // its late rejection from surfacing as an unhandled rejection.
    void pending.catch(() => undefined)
  }
}

/**
 * Dispatch one plugin hook point without letting plugin code break the turn.
 * HookRunner.run resolves with every hook result (sync and async); a
 * runner-level rejection is tolerated the way per-callback failures are
 * isolated inside the runner.
 */
async function dispatchHook(
  hookRunner: HookRunner | undefined,
  point: HookPoint,
  payload: Record<string, unknown>,
): Promise<unknown> {
  if (hookRunner === undefined) {
    return undefined
  }
  try {
    return await hookRunner.run(point, payload)
  } catch {
    return undefined
  }
}

/** Latest non-empty hook return wins, matching sequential mutation application. */
function hookMutation(result: unknown, current: unknown): unknown {
  if (Array.isArray(result)) {
    for (let index = result.length - 1; index >= 0; index -= 1) {
      if (result[index] !== undefined && result[index] !== null) {
        return result[index]
      }
    }
    return current
  }
  return result ?? current
}

/** Apply a before_tool_call argument mutation only when it is a real JSON object. */
function applyToolArgumentsMutation(call: ToolCall, mutated: unknown): ToolCall {
  if (!isJsonObject(mutated) || mutated === call.function.arguments) {
    return call
  }
  return { ...call, function: { ...call.function, arguments: mutated } }
}
