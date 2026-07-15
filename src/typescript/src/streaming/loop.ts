// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  errorMessage,
  type ToolExecutor,
  type ToolExecutionContext,
} from '../executors/toolRegistry.js'
import type { LlmClient, LlmDelta, TokenUsage } from '../llms/client.js'
import { LoopDetector, ToolLoopError } from '../runtime/loopDetector.js'
import {
  inspectObjectiveResponse,
  objectiveGuardRetryLimit,
} from '../runtime/objectiveGuard.js'
import type { ChatMessage } from '../types/messages.js'
import type { ToolCall, ToolDefinition } from '../types/toolCalls.js'
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

export const DEFAULT_MAX_TOOL_TURNS = 50
export const DEFAULT_RETRY_DELAYS = [1_000, 2_000] as const

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
  readonly topP?: number
  readonly tools?: readonly ToolDefinition[]
  readonly userMessage: string
}

export interface TurnDependencies {
  readonly delay?: (milliseconds: number, signal?: AbortSignal) => Promise<void>
  /** Supplies passive sub-agent status lines at safe provider/tool boundaries. */
  readonly drainAgentEvents?: () => readonly string[]
  /** Supplies steering text at safe provider/tool boundaries for daemon turns. */
  readonly drainSteer?: () => readonly string[]
  readonly llm: LlmClient
  readonly loopDetector?: LoopDetector
  /**
   * Observes provider-requested tools that were not included in the model-visible
   * surface. Returning `stop` ends the current turn without executing or retrying
   * those calls.
   */
  readonly onUnconfiguredToolCalls?: (calls: readonly ToolCall[]) => 'continue' | 'stop'
  readonly permissionBroker?: PermissionBroker
  readonly policy?: ToolPolicy
  readonly retryDelays?: readonly number[]
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
  const loopDetector = dependencies.loopDetector ?? new LoopDetector()
  const toolContext: ToolExecutionContext = {
    ...(request.agentId ? { agentId: request.agentId } : {}),
    ...(request.sessionId ? { sessionId: request.sessionId } : {}),
    metadata: state.metadata,
  }
  ensureSystemPrompt(state.messages, request.systemPrompt)
  state.messages.push({ role: 'user', content: request.userMessage })
  state.metadata.model = request.model
  state.turnCount += 1
  loopDetector.reset()

  let inputTokens = 0
  let outputTokens = 0
  let cacheReadTokens = 0
  let cacheCreationTokens = 0
  let reasoningTokens = 0
  let reasoningUsageComplete = true
  let usageComplete = true
  let apiCallsCount = 0
  let objectiveGuardRetries = 0
  let toolCallsCount = 0
  let latestToolRoundText: string | undefined
  const objectiveGuardLimit = objectiveGuardRetryLimit(
    request.objectiveGuardMaxRetries === undefined
      ? {}
      : { objective_guard_max_retries: request.objectiveGuardMaxRetries },
  )

  for (let toolTurn = 0; toolTurn < maxToolTurns; toolTurn += 1) {
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
    const parser =
      dependencies.thinkingParserFactory?.() ?? new ThinkingParser()
    const textParts: string[] = []
    const thinkingParts: string[] = []
    let thinkingSignature: string | undefined
    let roundToolCalls: readonly ToolCall[] = []
    let lastUsage: TokenUsage | undefined
    let streamCompleted = false
    const textDeduper = new ToolRoundTextDeduper(latestToolRoundText)

    for (let attempt = 0; attempt <= retryDelays.length; attempt += 1) {
      try {
        apiCallsCount += 1
        for await (const delta of dependencies.llm.stream(
          completionRequest(request, state.messages),
          signal,
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
        const final = attempt === retryDelays.length || signal?.aborted === true
        yield {
          type: 'provider_retry',
          error: errorMessage(error),
          attempt: attempt + 1,
          maxAttempts: retryDelays.length + 1,
          delay: final ? 0 : (retryDelays[attempt] ?? 0),
          final,
        }
        if (final) {
          const errorText = `[Error: ${errorMessage(error)}]`
          textParts.push(errorText)
          yield { type: 'text', text: errorText }
          streamCompleted = true
          break
        }
        await (dependencies.delay ?? defaultDelay)(
          retryDelays[attempt] ?? 0,
          signal,
        )
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
    const { exposed, unconfigured } = partitionToolCalls(providerToolCalls, request.tools)
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
        const result = unconfiguredToolResult(call)
        appendToolResult(state, result, call)
        yield { type: 'tool_end', result }
      }
      if (dependencies.onUnconfiguredToolCalls?.(unconfigured) === 'stop') {
        break
      }
      if (!roundToolCalls.length) {
        continue
      }
    }

    if (!roundToolCalls.length) {
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
        mode: request.interactionMode ?? 'code',
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
          const result = cancelledToolResult(cancelled)
          appendToolResult(state, result, cancelled)
          yield { type: 'tool_end', result }
        }
        break
      }
      const loopEvent = loopDetector.recordCall(
        call.function.name,
        call.function.arguments,
      )
      if (loopEvent.severity === 'critical') {
        const result = failedToolResult(call, new ToolLoopError(loopEvent))
        appendToolResult(state, result, call)
        yield { type: 'tool_end', result }
        continue
      }

      const permission = permissionDisposition(
        call,
        permissionMode,
        dependencies.policy,
        request.agentId,
      )
      if (permission === 'deny') {
        const result = deniedToolResult(call)
        appendToolResult(state, result, call)
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
          const result = cancelledToolResult(call)
          appendToolResult(state, result, call)
          yield { type: 'tool_end', result }
          continue
        }
        if (decision === 'reject') {
          const result = deniedToolResult(call)
          appendToolResult(state, result, call)
          yield { type: 'tool_end', result }
          continue
        }
      }

      yield { type: 'tool_start', call }
      const startedAt = performance.now()
      try {
        const output = dependencies.toolExecutor
          ? await dependencies.toolExecutor.execute(call, toolContext, signal)
          : `Tool ${call.function.name} is unavailable.`
        const result: ToolResult = {
          name: call.function.name,
          result: output,
          permitted: true,
          toolCallId: call.id,
          durationMs: performance.now() - startedAt,
        }
        appendToolResult(state, result, call)
        yield { type: 'tool_end', result }
      } catch (error) {
        const result = failedToolResult(
          call,
          error,
          performance.now() - startedAt,
        )
        appendToolResult(state, result, call)
        yield { type: 'tool_end', result }
      }
    }
    if (signal?.aborted) {
      break
    }
  }

  state.totalApiCalls += apiCallsCount
  state.usageComplete &&= usageComplete
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
): void {
  if (!events?.length) return
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
  }
}

function completionRequest(
  request: TurnRequest,
  messages: readonly ChatMessage[],
) {
  return {
    model: request.model,
    messages: [...messages],
    ...(request.tools?.length ? { tools: request.tools } : {}),
    ...(request.maxTokens !== undefined
      ? { maxTokens: request.maxTokens }
      : {}),
    ...(request.temperature !== undefined
      ? { temperature: request.temperature }
      : {}),
    ...(request.topP !== undefined ? { topP: request.topP } : {}),
  }
}

function ensureSystemPrompt(
  messages: ChatMessage[],
  systemPrompt: string | undefined,
): void {
  if (!systemPrompt || messages.some((message) => message.role === 'system')) {
    return
  }
  messages.unshift({ role: 'system', content: systemPrompt })
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
): void {
  state.messages.push({
    role: 'tool',
    content: result.result,
    name: result.name,
    tool_call_id: result.toolCallId,
  })
  state.toolExecutions.push({
    name: result.name,
    inputs: call.function.arguments,
    result: result.result,
    permitted: result.permitted,
    toolCallId: result.toolCallId,
    durationMs: result.durationMs,
  })
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
