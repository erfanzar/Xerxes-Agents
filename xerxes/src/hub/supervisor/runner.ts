// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { completeLlm, type LlmClient } from '../../llms/client.js'
import type { ToolRegistry } from '../../executors/toolRegistry.js'
import type { ChatMessage } from '../../types/messages.js'
import type { ToolCall, ToolDefinition } from '../../types/toolCalls.js'
import { DIGEST_SYSTEM_PROMPT, fenceUntrusted, type DigestSummarizer, type DigestSummaryInput } from './digests.js'
import { SUPERVISOR_TOOLS, assertSupervisorToolCeiling } from './schemas.js'
import { TURN_METADATA_KEY, type SupervisorTurnState } from './tools.js'
import { MAX_MESSAGE_CHARS, messageToWire, type SupervisorThreads, type ThreadMessage, type UserPrincipal } from './threads.js'

/**
 * Runs supervisor turns on an injected, explicitly configured model.
 *
 * - The tool registry is the dedicated supervisor registry. Its ceiling is
 *   asserted at construction, so a registry with a forbidden or extra tool
 *   never runs.
 * - Turns are serialized per thread.
 * - A turn makes at most 8 tool calls and is bounded in model calls, tokens,
 *   cost and wall time. It stops, and tells the thread, on a denial or when a
 *   quota runs out.
 * - Only a person's message starts a turn. The supervisor's own replies,
 *   confirmation echoes and notices never do.
 * - Tool results are task data: they enter the prompt fenced as untrusted.
 */

export const HARD_MAX_TOOL_CALLS_PER_TURN = 8

export interface SupervisorQuotas {
  /** Clamped to 8. */
  readonly maxToolCallsPerTurn: number
  readonly maxModelCallsPerTurn: number
  readonly maxTokensPerTurn: number
  readonly maxOutputTokens: number
  readonly maxTurnMs: number
  readonly windowMs: number
  readonly maxTokensPerWindow?: number
  /** Needs a model port that reports cost. Unknown cost is never counted as zero. */
  readonly maxCostUsdPerWindow?: number
}

export const DEFAULT_SUPERVISOR_QUOTAS: SupervisorQuotas = Object.freeze({
  maxToolCallsPerTurn: HARD_MAX_TOOL_CALLS_PER_TURN,
  maxModelCallsPerTurn: 10,
  maxTokensPerTurn: 60_000,
  maxOutputTokens: 1_200,
  maxTurnMs: 90_000,
  windowMs: 24 * 60 * 60 * 1000,
})

export interface SupervisorModelUsage {
  readonly inputTokens: number
  readonly outputTokens: number
  readonly costUsd?: number
}

export interface SupervisorModelRequest {
  readonly system: string
  readonly messages: readonly ChatMessage[]
  readonly tools: readonly ToolDefinition[]
  readonly maxTokens: number
  readonly signal?: AbortSignal
}

export interface SupervisorModelResponse {
  readonly text: string
  readonly toolCalls: readonly ToolCall[]
  readonly usage?: SupervisorModelUsage
}

/** A cheap, tool-capable model chosen explicitly by the operator. */
export interface SupervisorModelPort {
  readonly id: string
  /** Whether `usage.costUsd` is reported. A cost quota needs it. */
  readonly reportsCost: boolean
  complete(request: SupervisorModelRequest): Promise<SupervisorModelResponse>
}

/** Adapt a provider client. Housekeeping query source: this is never the user's main loop. */
export function createLlmModelPort(client: LlmClient, model: string, options: { readonly timeoutMs?: number } = {}): SupervisorModelPort {
  return {
    id: model,
    reportsCost: false,
    async complete(request) {
      const completion = await completeLlm(client, {
        model,
        messages: [{ role: 'system', content: request.system }, ...request.messages],
        maxTokens: request.maxTokens,
        temperature: 0,
        querySource: 'classification',
        ...(request.tools.length === 0 ? {} : { tools: request.tools }),
      }, request.signal, options.timeoutMs === undefined ? {} : { timeoutMs: options.timeoutMs })
      return {
        text: completion.content,
        toolCalls: completion.toolCalls,
        ...(completion.usage === undefined ? {} : { usage: { inputTokens: completion.usage.inputTokens, outputTokens: completion.usage.outputTokens } }),
      }
    },
  }
}

/** Digest prose from the same cheap model. The version is part of the digest cache key. */
export function createModelDigestSummarizer(model: SupervisorModelPort, promptVersion = 'v1'): DigestSummarizer {
  return {
    // A getter: the model can be swapped at runtime and the digest cache must follow it.
    get version() { return `${model.id}:${promptVersion}` },
    async summarize(input: DigestSummaryInput, signal?: AbortSignal): Promise<string> {
      const parts = [
        fenceUntrusted('task_facts', JSON.stringify(input.facts)),
        input.previousSummary === undefined ? 'There is no previous summary.' : `Previous summary:\n${fenceUntrusted('previous_summary', input.previousSummary)}`,
        `New events${input.skippedEvents > 0 ? ` (${input.skippedEvents} older ones were left out)` : ''}:\n${fenceUntrusted('task_events', input.eventsText)}`,
        'Write the updated digest now.',
      ]
      const response = await model.complete({
        system: DIGEST_SYSTEM_PROMPT,
        messages: [{ role: 'user', content: parts.join('\n\n') }],
        tools: [],
        maxTokens: 700,
        ...(signal === undefined ? {} : { signal }),
      })
      return response.text
    },
  }
}

/**
 * A turn the Hub started by itself, not a person's message: an XERXES Autopilot
 * wake-up (`autopilot`) or a Watch wake-up (`watch`). Both are Hub triggers for
 * the action rules; a `watch` turn only auto-runs where Autopilot is active.
 */
export interface SupervisorTurnTrigger {
  readonly kind: 'autopilot' | 'watch'
  /** Hub-authored wake text. A function is called only when the turn actually starts. */
  readonly text: string | (() => string)
  /** The one task this wake is about. Only actions on it may run without a Confirm tap. */
  readonly target?: { readonly workspaceId: string; readonly sessionId: string }
}

/** Default spend bound applied when the Hub wires the supervisor: tokens per 24-hour window. */
export const DEFAULT_SUPERVISOR_MAX_TOKENS_PER_WINDOW = 2_000_000

export type SupervisorStop = 'completed' | 'tool_cap' | 'denied' | 'quota' | 'timeout' | 'cancelled' | 'model_error' | 'gone'

export interface SupervisorTurnResult {
  readonly stop: SupervisorStop
  readonly toolCalls: number
  readonly modelCalls: number
  readonly reply?: ThreadMessage
  readonly detail?: string
}

export interface SupervisorStoppedInfo {
  readonly threadId: string
  readonly requester: string
  readonly stop: Exclude<SupervisorStop, 'completed' | 'cancelled' | 'gone'>
  readonly detail: string
}

export interface SupervisorRunnerOptions {
  readonly threads: SupervisorThreads
  readonly registry: ToolRegistry
  readonly model: SupervisorModelPort
  /** The XERXES persona (agents/default/supervisor.md). */
  readonly persona: string
  readonly quotas?: Partial<SupervisorQuotas>
  readonly scopeReader: (threadId: string, requester: UserPrincipal) => SupervisorTurnState['reader']
  readonly clock?: () => number
  readonly onStopped?: (info: SupervisorStoppedInfo) => void
  readonly onError?: (error: unknown, context: string) => void
}

const CONTEXT_MESSAGES = 12
const TOOL_RESULT_CHARS = 14_000

function clip(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, Math.max(0, max - 1))}…`
}

function toChat(message: ThreadMessage): ChatMessage {
  // A reply written in a Hub wake turn may quote task text: it comes back fenced.
  if (message.role === 'xerxes') {
    return { role: 'assistant', content: message.origin === 'wake' ? fenceUntrusted('earlier_wake_reply', message.text) : message.text }
  }
  if (message.role === 'system') return { role: 'user', content: fenceUntrusted('thread_notice', message.text) }
  return { role: 'user', content: `${message.authorName ?? 'Someone'}${message.source === 'voice' ? ' (spoken)' : ''}: ${message.text}` }
}

export class SupervisorRunner {
  readonly quotas: SupervisorQuotas
  private readonly clock: () => number
  private readonly tails = new Map<string, Promise<unknown>>()
  private readonly active = new Map<string, AbortController>()
  private usage = { windowStart: 0, tokens: 0, costUsd: 0, calls: 0 }
  private closed = false

  constructor(private readonly options: SupervisorRunnerOptions) {
    this.clock = options.clock ?? Date.now
    this.quotas = {
      ...DEFAULT_SUPERVISOR_QUOTAS,
      ...options.quotas,
      maxToolCallsPerTurn: Math.min(HARD_MAX_TOOL_CALLS_PER_TURN, options.quotas?.maxToolCallsPerTurn ?? HARD_MAX_TOOL_CALLS_PER_TURN),
    }
    // The ceiling is checked here, not left to the caller: a registry with a forbidden or extra tool never runs.
    const names = options.registry.definitions().map(definition => definition.function.name)
    assertSupervisorToolCeiling(names)
    if (names.length !== SUPERVISOR_TOOLS.size) throw new Error('the supervisor registry must hold exactly the supervisor tools')
    if (this.quotas.maxCostUsdPerWindow !== undefined && !options.model.reportsCost) {
      throw new Error('a cost quota needs a model port that reports cost; unknown cost is never counted as zero')
    }
    if (options.persona.trim() === '') throw new Error('the supervisor persona prompt is empty')
    this.usage.windowStart = this.clock()
  }

  usageSnapshot(): { readonly tokens: number; readonly costUsd: number; readonly calls: number; readonly windowStart: string } {
    this.rollWindow()
    return { tokens: this.usage.tokens, costUsd: this.usage.costUsd, calls: this.usage.calls, windowStart: new Date(this.usage.windowStart).toISOString() }
  }

  busy(threadId: string): boolean {
    return this.tails.has(threadId)
  }

  /** Stop for good: queued turns never start and every running turn is cancelled. */
  shutdown(): void {
    this.closed = true
    for (const controller of this.active.values()) controller.abort(new Error('cancelled'))
  }

  /** Cancel the turn running on this thread. Background agents are not touched. */
  cancel(threadId: string): boolean {
    const controller = this.active.get(threadId)
    if (!controller) return false
    controller.abort(new Error('cancelled'))
    return true
  }

  /**
   * Queue one turn behind any turn already running on the thread. Never rejects.
   * `trigger` marks a Hub-started XERXES Autopilot turn: its Hub-authored text is
   * the last message of the turn, and the requester is the enabling owner.
   */
  enqueue(input: { readonly threadId: string; readonly requester: UserPrincipal; readonly trigger?: SupervisorTurnTrigger }): Promise<SupervisorTurnResult> {
    const previous = this.tails.get(input.threadId) ?? Promise.resolve()
    const start = (): Promise<SupervisorTurnResult> => (this.closed
      ? Promise.resolve({ stop: 'cancelled', toolCalls: 0, modelCalls: 0 })
      : this.run(input))
    const next = previous.then(start, start)
    const tail = next.then(() => undefined, () => undefined)
    this.tails.set(input.threadId, tail)
    void tail.then(() => { if (this.tails.get(input.threadId) === tail) this.tails.delete(input.threadId) })
    return next
  }

  private rollWindow(): void {
    const now = this.clock()
    if (now - this.usage.windowStart >= this.quotas.windowMs) this.usage = { windowStart: now, tokens: 0, costUsd: 0, calls: 0 }
  }

  private exhausted(turnTokens: number, modelCalls: number): string | undefined {
    this.rollWindow()
    const q = this.quotas
    if (modelCalls >= q.maxModelCallsPerTurn) return 'it used all the model calls allowed in one turn'
    if (turnTokens >= q.maxTokensPerTurn) return 'it used all the tokens allowed in one turn'
    if (q.maxTokensPerWindow !== undefined && this.usage.tokens >= q.maxTokensPerWindow) return 'the token quota for this period is used up'
    if (q.maxCostUsdPerWindow !== undefined && this.usage.costUsd >= q.maxCostUsdPerWindow) return 'the spending quota for this period is used up'
    return undefined
  }

  private finishWith(
    threadId: string, requester: UserPrincipal, stop: SupervisorStop, counts: { toolCalls: number; modelCalls: number }, detail?: string,
  ): SupervisorTurnResult {
    const { threads } = this.options
    if (stop !== 'completed' && stop !== 'cancelled' && stop !== 'gone' && detail !== undefined) {
      const { message } = threads.append({ threadId, role: 'system', text: `XERXES stopped: ${detail}` })
      threads.emit(threadId, 'supervisor.thread_event', { message: messageToWire(message) })
      try {
        this.options.onStopped?.({ threadId, requester: requester.id, stop, detail })
      } catch (error) {
        this.options.onError?.(error, 'supervisor stop notice')
      }
    }
    return { stop, ...counts, ...(detail === undefined ? {} : { detail }) }
  }

  private async run(input: { readonly threadId: string; readonly requester: UserPrincipal; readonly trigger?: SupervisorTurnTrigger }): Promise<SupervisorTurnResult> {
    const { threads, registry, model } = this.options
    const { threadId, requester } = input
    const counts = { toolCalls: 0, modelCalls: 0 }
    const thread = threads.get(threadId)
    if (!thread) return this.finishWith(threadId, requester, 'gone', counts)

    const controller = new AbortController()
    this.active.set(threadId, controller)
    const timer = setTimeout(() => controller.abort(new Error('timeout')), this.quotas.maxTurnMs)
    threads.setWorking(threadId, true)
    try {
      const stored = threads.messagesFor(threadId, requester.id, CONTEXT_MESSAGES)
      const state: SupervisorTurnState = {
        threadId, requester, reader: this.options.scopeReader(threadId, requester), denial: undefined,
        ...(input.trigger === undefined ? {} : { trigger: input.trigger.kind }),
        ...(input.trigger?.target === undefined ? {} : { target: input.trigger.target }),
        // Autopilot runs the requester's own actions only when nobody else spoke in this context window,
        // and no reply from a wake turn (which may quote task text) is in it either.
        othersInContext: stored.some(message => (message.role === 'user' && message.authorId !== requester.id)
          || (message.role === 'xerxes' && message.origin === 'wake')),
      }
      const messages: ChatMessage[] = stored.map(toChat)
      // Hub-authored wake text; anything copied from a task inside it is already fenced.
      if (input.trigger !== undefined) {
        let text: string
        try {
          text = typeof input.trigger.text === 'function' ? input.trigger.text() : input.trigger.text
        } catch (error) {
          this.options.onError?.(error, 'supervisor wake text')
          return this.finishWith(threadId, requester, 'model_error', counts, 'the Hub could not build the wake-up message')
        }
        messages.push({ role: 'user', content: text })
      }
      // The person's message this turn answers; voice uses it to match a reply to its submission.
      const inReplyTo = stored.filter(message => message.role === 'user').at(-1)?.id
      const tools = registry.definitions()
      let turnTokens = 0

      for (;;) {
        if (controller.signal.aborted) return this.aborted(controller, threadId, requester, counts)
        const reason = this.exhausted(turnTokens, counts.modelCalls)
        if (reason !== undefined) return this.finishWith(threadId, requester, 'quota', counts, reason)

        let response: Awaited<ReturnType<SupervisorModelPort['complete']>>
        counts.modelCalls += 1
        try {
          response = await model.complete({ system: this.options.persona, messages, tools, maxTokens: this.quotas.maxOutputTokens, signal: controller.signal })
        } catch (error) {
          if (controller.signal.aborted) return this.aborted(controller, threadId, requester, counts)
          this.options.onError?.(error, 'supervisor model')
          return this.finishWith(threadId, requester, 'model_error', counts, `it could not reach its model (${clip(error instanceof Error ? error.message : String(error), 160)})`)
        }
        // A model that ignored the abort (or a shutdown during the call) must not act on its answer.
        if (controller.signal.aborted) return this.aborted(controller, threadId, requester, counts)
        this.usage.calls += 1
        if (response.usage !== undefined) {
          const tokens = Math.max(0, response.usage.inputTokens) + Math.max(0, response.usage.outputTokens)
          this.usage.tokens += tokens
          turnTokens += tokens
          if (response.usage.costUsd !== undefined) this.usage.costUsd += Math.max(0, response.usage.costUsd)
        }

        if (response.toolCalls.length === 0) {
          const text = clip(response.text.trim(), MAX_MESSAGE_CHARS)
          const { message } = threads.append({
            threadId, role: 'xerxes', text: text === '' ? 'I have nothing to add.' : text, ...(input.trigger === undefined ? {} : { origin: 'wake' as const }),
          })
          threads.emit(threadId, 'supervisor.thread_event', { message: messageToWire(message) })
          threads.publishReply(threadId, { id: message.id, text: message.text, ...(inReplyTo === undefined ? {} : { inReplyTo }) })
          return { stop: 'completed', ...counts, reply: message }
        }

        messages.push({ role: 'assistant', content: response.text, tool_calls: response.toolCalls })
        for (const call of response.toolCalls) {
          if (counts.toolCalls >= this.quotas.maxToolCallsPerTurn) {
            return this.finishWith(threadId, requester, 'tool_cap', counts, `it reached the limit of ${this.quotas.maxToolCallsPerTurn} tool calls in one turn`)
          }
          counts.toolCalls += 1
          let content: string
          let failed = false
          try {
            content = await registry.execute(call, { metadata: { [TURN_METADATA_KEY]: state } }, controller.signal)
          } catch (error) {
            if (controller.signal.aborted) return this.aborted(controller, threadId, requester, counts)
            failed = true
            content = error instanceof Error ? error.message : String(error)
          }
          messages.push({
            role: 'tool', name: call.function.name, tool_call_id: call.id,
            content: fenceUntrusted(`tool_result.${call.function.name}`, clip(content, TOOL_RESULT_CHARS)),
            ...(failed ? { is_error: true } : {}),
          })
          if (state.denial !== undefined) return this.finishWith(threadId, requester, 'denied', counts, state.denial)
        }
      }
    } finally {
      clearTimeout(timer)
      this.active.delete(threadId)
      // After shutdown the database may already be closed; there is nobody left to tell.
      if (!this.closed) threads.setWorking(threadId, false)
    }
  }

  private aborted(controller: AbortController, threadId: string, requester: UserPrincipal, counts: { toolCalls: number; modelCalls: number }): SupervisorTurnResult {
    const reason = controller.signal.reason
    if (reason instanceof Error && reason.message === 'timeout') {
      return this.finishWith(threadId, requester, 'timeout', counts, `the turn took longer than ${Math.round(this.quotas.maxTurnMs / 1000)} seconds`)
    }
    return this.finishWith(threadId, requester, 'cancelled', counts)
  }
}
