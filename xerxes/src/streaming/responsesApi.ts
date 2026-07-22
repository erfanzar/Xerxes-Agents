// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ProviderError } from '../core/errors.js'
import type { LlmDelta, TokenUsage } from '../llms/client.js'
import { isJsonObject, parseToolArguments, type ToolCall } from '../types/toolCalls.js'

export interface ResponsesUsage extends TokenUsage {
  readonly finishReason: string
  readonly toolCalls: readonly ToolCall[]
}

interface PendingFunctionCall {
  argumentsText: string
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

  /** Translate one decoded Responses API event into zero or more neutral deltas. */
  translate(event: Readonly<Record<string, unknown>>): LlmDelta[] {
    const type = stringValue(event.type)
    if (type === 'response.output_text.delta') {
      const text = stringValue(event.delta) || stringValue(event.text)
      return text ? [{ content: text }] : []
    }
    if (type === 'response.reasoning.delta') {
      const thinking = stringValue(event.delta) || stringValue(event.text)
      return thinking ? [{ thinking }] : []
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
    if (type === 'response.output_item.done') {
      this.completeFunctionCall(recordValue(event.item))
      return []
    }
    if (type === 'response.completed') {
      this.completeUsage(recordValue(event.response))
      return [this.completionDelta()]
    }
    if (type === 'response.incomplete') {
      this.completeUsage(recordValue(event.response))
      this.usage = { ...this.usage, finishReason: incompleteFinishReason(recordValue(event.response)) }
      return [this.completionDelta()]
    }
    if (type === 'response.failed' || type === 'error') {
      throw new ProviderError('responses', responsesErrorMessage(event))
    }
    return []
  }

  /** Translate an ordered event sequence and flush any unfinished tool calls. */
  *translateAll(events: Iterable<Readonly<Record<string, unknown>>>): Generator<LlmDelta> {
    for (const event of events) {
      for (const delta of this.translate(event)) yield delta
    }
    const final = this.finish()
    if (final) yield final
  }

  /** Flush a truncated transport after it ends without response.completed. */
  finish(): LlmDelta | undefined {
    if (!this.pendingCalls.size) return undefined
    this.completePendingCalls()
    return this.completionDelta()
  }

  private addFunctionCall(item: Readonly<Record<string, unknown>>): void {
    this.upsertPendingCall(item)
  }

  private completeFunctionCall(item: Readonly<Record<string, unknown>>): void {
    const upserted = this.upsertPendingCall(item)
    if (!upserted) return
    this.pendingCalls.delete(upserted.id)
    this.recordToolCall(upserted.pending)
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
    let pending = this.pendingCalls.get(id) ?? this.createPendingCall(id)
    if (itemId && pending.id !== itemId) {
      this.pendingCalls.delete(id)
      pending = { ...pending, id: itemId }
      id = itemId
    }
    const name = stringValue(item.name)
    const argumentValue = argumentsText(item.arguments)
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
      ?? finiteNumber(outputDetails.cache_creation_tokens)
    const reasoningTokens = finiteNumber(outputDetails.reasoning_tokens)
    this.completePendingCalls()
    this.usage = {
      inputTokens: finiteNumber(usage.input_tokens) ?? 0,
      outputTokens: finiteNumber(usage.output_tokens) ?? 0,
      toolCalls: this.usage.toolCalls,
      finishReason: completedFinishReason(stringValue(response.status), this.usage.toolCalls.length > 0),
      ...(cacheReadTokens === undefined ? {} : { cacheReadTokens }),
      ...(cacheCreationTokens === undefined ? {} : { cacheCreationTokens }),
      ...(reasoningTokens === undefined ? {} : { reasoningTokens }),
    }
  }

  private completePendingCalls(): void {
    for (const [id, pending] of this.pendingCalls) {
      this.pendingCalls.delete(id)
      this.recordToolCall(pending)
    }
    this.pendingAliases.clear()
  }

  private recordToolCall(pending: PendingFunctionCall): void {
    if (!pending.name) return
    const call: ToolCall = {
      id: pending.id,
      type: 'function',
      function: {
        name: pending.name,
        arguments: parseToolArguments(pending.argumentsText),
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

function isFunctionCallItem(item: Readonly<Record<string, unknown>>): boolean {
  const type = stringValue(item.type)
  return type === 'function_call' || type === 'tool_call'
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
  }
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function recordValue(value: unknown): Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value : ''
}
