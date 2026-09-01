// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { parseStreamingJson } from '@earendil-works/pi-ai'

import { ProviderError } from '../core/errors.js'
import type { LlmDelta, TokenUsage } from '../llms/client.js'
import { isJsonObject, parseToolArguments, type ToolCall } from '../types/toolCalls.js'

export interface ResponsesUsage extends TokenUsage {
  readonly finishReason: string
  readonly toolCalls: readonly ToolCall[]
}

interface PendingFunctionCall {
  argumentsText: string
  /**
   * Set for OpenAI `custom` (grammar-constrained) tool calls: the raw input
   * text accumulates unmodified and lands under this single string property
   * with no JSON parsing (pi-ai custom-tool semantics).
   */
  customProperty?: string
  readonly id: string
  name: string
}

/**
 * Stateful normalizer for streamed OpenAI Responses API events.
 *
 * It keeps partial function-call arguments out of visible model text and
 * produces the same neutral LlmDelta vocabulary consumed by the agent loop.
 * Terminal provider error events throw instead of passing as silent success.
 */
export class ResponsesEventTranslator {
  usage: ResponsesUsage = {
    inputTokens: 0,
    outputTokens: 0,
    toolCalls: [],
    finishReason: 'stop',
  }

  private readonly pendingCalls = new Map<string, PendingFunctionCall>()

  /** Maps every observed item_id/call_id onto its pending entry's map key. */
  private readonly pendingAliases = new Map<string, string>()

  /** Identities of function calls already committed to usage. */
  private readonly completedCallIds = new Set<string>()

  /** Thinking text already emitted through official Responses reasoning deltas. */
  private thinkingText = ''

  /** True once the provider has sent a terminal completed/incomplete response event. */
  private terminal = false

  /**
   * @param grammarToolInputProperties tool name → grammar input property for
   * the request's custom (grammar-constrained) tools, so streamed
   * `custom_tool_call` items resolve their raw text onto the right argument.
   */
  constructor(private readonly grammarToolInputProperties: ReadonlyMap<string, string> = new Map()) {}

  /** Translate one decoded Responses API event into zero or more neutral deltas. */
  translate(event: Readonly<Record<string, unknown>>): LlmDelta[] {
    const type = stringValue(event.type)
    if (type === 'response.failed' || type === 'error') {
      throw new ProviderError('responses', responsesErrorMessage(event))
    }
    // Some compatible providers keep sending decoded events after their
    // terminal response. They cannot change the already-committed semantic
    // result; accepting them would append text or launch tools after finish.
    if (this.terminal) return []
    if (type === 'response.output_text.delta') {
      const text = stringValue(event.delta) || stringValue(event.text)
      return text ? [{ content: text }] : []
    }
    if (type === 'response.reasoning_summary_text.delta'
      || type === 'response.reasoning_text.delta'
      || type === 'response.reasoning.delta') {
      const thinking = stringValue(event.delta) || stringValue(event.text)
      if (!thinking) return []
      this.thinkingText += thinking
      return [{ thinking }]
    }
    if (type === 'response.reasoning_summary_part.done') {
      this.thinkingText += '\n\n'
      return [{ thinking: '\n\n' }]
    }
    if (type === 'response.refusal.delta') {
      const text = stringValue(event.delta) || stringValue(event.text)
      return text ? [{ content: text }] : []
    }
    if (type === 'response.output_item.added') {
      this.addFunctionCall(recordValue(event.item))
      return []
    }
    if (type === 'response.function_call_arguments.delta') {
      const rawId = stringValue(event.item_id) || stringValue(event.call_id)
      if (!rawId) return []
      const id = this.pendingAliases.get(rawId) ?? rawId
      const pending = this.pendingCalls.get(id) ?? this.createPendingCall(id)
      pending.argumentsText += stringValue(event.delta)
      this.pendingCalls.set(id, pending)
      this.pendingAliases.set(rawId, id)
      return []
    }
    if (type === 'response.custom_tool_call_input.delta'
      || type === 'response.custom_tool_call_input.done') {
      const rawId = stringValue(event.item_id) || stringValue(event.call_id)
      if (!rawId) return []
      const id = this.pendingAliases.get(rawId) ?? rawId
      const pending = this.pendingCalls.get(id) ?? this.createPendingCall(id)
      pending.customProperty ??= 'input'
      if (type === 'response.custom_tool_call_input.done') {
        // The done event carries the authoritative full input; it must extend
        // what the deltas accumulated (pi-ai's monotonicity contract).
        const authoritative = stringValue(event.input)
        if (authoritative && !authoritative.startsWith(pending.argumentsText)) {
          throw new ProviderError(
            'responses',
            `grammar tool input for "${pending.name || rawId}" changed non-monotonically`,
          )
        }
        pending.argumentsText = authoritative || pending.argumentsText
      } else {
        pending.argumentsText += stringValue(event.delta)
      }
      this.pendingCalls.set(id, pending)
      this.pendingAliases.set(rawId, id)
      return []
    }
    if (type === 'response.output_item.done') {
      const item = recordValue(event.item)
      if (stringValue(item.type) === 'reasoning') return this.completeReasoning(item)
      this.completeFunctionCall(item)
      return []
    }
    if (type === 'response.completed') {
      this.completeUsage(recordValue(event.response))
      this.terminal = true
      return [this.completionDelta()]
    }
    if (type === 'response.incomplete') {
      this.completeUsage(recordValue(event.response))
      this.usage = { ...this.usage, finishReason: incompleteFinishReason(recordValue(event.response)) }
      this.terminal = true
      return [this.completionDelta()]
    }
    return []
  }

  /** Translate an ordered event sequence and require a provider terminal event. */
  *translateAll(events: Iterable<Readonly<Record<string, unknown>>>): Generator<LlmDelta> {
    for (const event of events) {
      for (const delta of this.translate(event)) yield delta
    }
    this.finish()
  }

  /** Validate that a finite transport ended after a terminal provider event. */
  finish(): void {
    if (!this.terminal) {
      throw new ProviderError('responses', 'Responses API stream ended before a terminal response event')
    }
  }

  private addFunctionCall(item: Readonly<Record<string, unknown>>): void {
    this.upsertPendingCall(item)
  }

  private completeReasoning(item: Readonly<Record<string, unknown>>): LlmDelta[] {
    const completeText = responseReasoningText(item)
    const delta = completeText && completeText.startsWith(this.thinkingText)
      ? completeText.slice(this.thinkingText.length)
      : this.thinkingText ? '' : completeText
    if (delta) this.thinkingText += delta
    return [{
      ...(delta ? { thinking: delta } : {}),
      thinkingSignature: JSON.stringify(item),
    }]
  }

  private completeFunctionCall(item: Readonly<Record<string, unknown>>): void {
    if (this.isCompletedCall(item)) return
    const upserted = this.upsertPendingCall(item)
    if (!upserted) return
    this.pendingCalls.delete(upserted.id)
    this.recordToolCall(upserted.pending, [stringValue(item.id), stringValue(item.call_id)])
  }

  private isCompletedCall(item: Readonly<Record<string, unknown>>): boolean {
    return [stringValue(item.id), stringValue(item.call_id)]
      .some(id => id && this.completedCallIds.has(id))
  }

  /**
   * Merge an output item into its pending entry, aliasing item_id and call_id.
   *
   * Argument deltas may carry only one of the two identifiers, so both forms
   * must resolve to the same pending entry; otherwise a delta-only stub would
   * flush as a duplicate, nameless tool call at the end of the stream.
   */
  private upsertPendingCall(
    item: Readonly<Record<string, unknown>>,
  ): { id: string; pending: PendingFunctionCall } | undefined {
    if (!isFunctionCallItem(item)) return undefined
    const itemId = stringValue(item.id)
    const callId = stringValue(item.call_id)
    let id = this.findPendingKey(itemId, callId) ?? (itemId || callId)
    if (!id) return undefined
    // Keep item_id as the stream lookup key, but expose call_id as the tool
    // correlation ID so the subsequent function_call_output can reference it.
    let pending = this.pendingCalls.get(id) ?? this.createPendingCall(callId || itemId)
    if (itemId && id !== itemId) {
      this.pendingCalls.delete(id)
      id = itemId
    }
    const correlationId = callId || itemId
    if (correlationId && pending.id !== correlationId) {
      pending = { ...pending, id: correlationId }
    }
    const name = stringValue(item.name)
    const isCustomCall = stringValue(item.type) === 'custom_tool_call'
    if (isCustomCall) {
      pending.customProperty ??= (name ? this.grammarToolInputProperties.get(name) : undefined) ?? 'input'
    }
    const argumentValue = isCustomCall
      ? stringValue(item.input)
      : argumentsText(item.arguments)
    if (name) pending.name = name
    if (argumentValue) pending.argumentsText = argumentValue
    this.pendingCalls.set(id, pending)
    for (const alias of [itemId, callId]) {
      if (alias) this.pendingAliases.set(alias, id)
    }
    return { id, pending }
  }

  private findPendingKey(...ids: string[]): string | undefined {
    for (const id of ids) {
      if (!id) continue
      const resolved = this.pendingAliases.get(id) ?? id
      if (this.pendingCalls.has(resolved)) return resolved
    }
    return undefined
  }

  private completeUsage(response: Readonly<Record<string, unknown>>): void {
    const usage = recordValue(response.usage)
    const inputDetails = recordValue(usage.input_tokens_details)
    const outputDetails = recordValue(usage.output_tokens_details)
    const cacheReadTokens = finiteNumber(usage.cache_read_tokens)
      ?? finiteNumber(inputDetails.cached_tokens)
    const cacheCreationTokens = finiteNumber(usage.cache_creation_tokens)
      ?? finiteNumber(inputDetails.cache_write_tokens)
      ?? finiteNumber(outputDetails.cache_creation_tokens)
    const reasoningTokens = finiteNumber(outputDetails.reasoning_tokens)
    this.completePendingCalls()
    this.usage = {
      // `input_tokens` is the whole prompt here and `cached_tokens` is a subset
      // of it, the opposite of Anthropic where the two are disjoint. Consumers
      // add the pair to size a prompt, so the overlap is removed to match that
      // convention — otherwise a cache hit inflates reported context by however
      // much was cached, and the number tracks cache luck instead of usage.
      inputTokens: freshInputTokens(
        finiteNumber(usage.input_tokens) ?? 0,
        cacheReadTokens,
        cacheCreationTokens,
      ),
      outputTokens: finiteNumber(usage.output_tokens) ?? 0,
      toolCalls: this.usage.toolCalls,
      finishReason: completedFinishReason(stringValue(response.status), this.usage.toolCalls.length > 0),
      ...(cacheReadTokens === undefined ? {} : { cacheReadTokens }),
      ...(cacheCreationTokens === undefined ? {} : { cacheCreationTokens }),
      ...(reasoningTokens === undefined ? {} : { reasoningTokens }),
      ...(stringValue(response.service_tier) ? { serviceTier: stringValue(response.service_tier) } : {}),
    }
  }

  private completePendingCalls(): void {
    for (const [id, pending] of this.pendingCalls) {
      this.pendingCalls.delete(id)
      this.recordToolCall(pending)
    }
    this.pendingAliases.clear()
  }

  private recordToolCall(pending: PendingFunctionCall, aliases: readonly string[] = []): void {
    if (!pending.name) return
    for (const id of [pending.id, ...aliases]) {
      if (id) this.completedCallIds.add(id)
    }
    const call: ToolCall = {
      id: pending.id,
      type: 'function',
      function: {
        name: pending.name,
        arguments: pending.customProperty !== undefined
          ? { [pending.customProperty]: pending.argumentsText }
          : (() => {
            const partial = parseStreamingJson(pending.argumentsText)
            return isJsonObject(partial) ? partial : parseToolArguments(pending.argumentsText)
          })(),
      },
    }
    this.usage = { ...this.usage, toolCalls: [...this.usage.toolCalls, call] }
  }

  private completionDelta(): LlmDelta {
    return {
      finishReason: this.usage.finishReason,
      usage: tokenUsage(this.usage),
      ...(this.usage.toolCalls.length ? { toolCalls: this.usage.toolCalls } : {}),
    }
  }

  private createPendingCall(id: string): PendingFunctionCall {
    return { id, name: '', argumentsText: '' }
  }
}

function responseReasoningText(item: Readonly<Record<string, unknown>>): string {
  for (const key of ['summary', 'content']) {
    const text = Array.isArray(item[key])
      ? (item[key] as unknown[])
        .map(part => stringValue(recordValue(part).text))
        .filter(Boolean)
        .join('\n\n')
      : ''
    if (text) return text
  }
  return ''
}

function isFunctionCallItem(item: Readonly<Record<string, unknown>>): boolean {
  const type = stringValue(item.type)
  return type === 'function_call' || type === 'tool_call' || type === 'custom_tool_call'
}

/**
 * Map a completed response status onto the neutral finish vocabulary.
 *
 * The raw provider status 'completed' is not a valid loop finish reason; it
 * becomes 'tool_calls' when calls were recorded and 'stop' otherwise.
 */
function completedFinishReason(status: string, hasToolCalls: boolean): string {
  if (hasToolCalls) return 'tool_calls'
  return !status || status === 'completed' ? 'stop' : status
}

/** Map an incomplete response's reason onto the neutral finish vocabulary. */
function incompleteFinishReason(response: Readonly<Record<string, unknown>>): string {
  const reason = stringValue(recordValue(response.incomplete_details).reason)
  if (reason === 'max_output_tokens') return 'length'
  return reason || stringValue(response.status) || 'incomplete'
}

/** Format a terminal Responses API error event with its provider-supplied payload. */
function responsesErrorMessage(event: Readonly<Record<string, unknown>>): string {
  const nested = recordValue(recordValue(event.response).error)
  const direct = recordValue(event.error)
  const code = stringValue(nested.code) || stringValue(direct.code) || stringValue(event.code)
  const message = stringValue(nested.message) || stringValue(direct.message) || stringValue(event.message)
  return `stream returned API error${code ? ` (${code})` : ''}: ${message || 'unknown error'}`
}

function argumentsText(value: unknown): string {
  if (typeof value === 'string') return value
  if (isJsonObject(value)) return JSON.stringify(value)
  return ''
}

function tokenUsage(usage: ResponsesUsage): TokenUsage {
  return {
    inputTokens: usage.inputTokens,
    outputTokens: usage.outputTokens,
    ...(usage.cacheReadTokens === undefined ? {} : { cacheReadTokens: usage.cacheReadTokens }),
    ...(usage.cacheCreationTokens === undefined ? {} : { cacheCreationTokens: usage.cacheCreationTokens }),
    ...(usage.reasoningTokens === undefined ? {} : { reasoningTokens: usage.reasoningTokens }),
    ...(usage.serviceTier === undefined ? {} : { serviceTier: usage.serviceTier }),
  }
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

/**
 * Prompt tokens that were not served from cache.
 *
 * Guarded rather than a bare subtraction: a provider reporting a cached count
 * above its own input total would otherwise yield a negative token count that
 * silently corrupts every downstream sum.
 */
function freshInputTokens(
  inputTokens: number,
  cacheReadTokens: number | undefined,
  cacheCreationTokens: number | undefined,
): number {
  return Math.max(0, inputTokens - (cacheReadTokens ?? 0) - (cacheCreationTokens ?? 0))
}

function recordValue(value: unknown): Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value : ''
}
