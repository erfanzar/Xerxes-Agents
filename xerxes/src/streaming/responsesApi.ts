// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

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
 */
export class ResponsesEventTranslator {
  usage: ResponsesUsage = {
    inputTokens: 0,
    outputTokens: 0,
    toolCalls: [],
    finishReason: 'stop',
  }

  private readonly pendingCalls = new Map<string, PendingFunctionCall>()

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
      const id = stringValue(event.item_id) || stringValue(event.call_id)
      if (!id) return []
      const pending = this.pendingCalls.get(id) ?? this.createPendingCall(id)
      pending.argumentsText += stringValue(event.delta)
      this.pendingCalls.set(id, pending)
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
    if (type === 'response.failed' || type === 'error') {
      this.usage = { ...this.usage, finishReason: 'error' }
      return []
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
    if (!isFunctionCallItem(item)) return
    const id = stringValue(item.id) || stringValue(item.call_id)
    if (!id) return
    const pending = this.pendingCalls.get(id) ?? this.createPendingCall(id)
    const name = stringValue(item.name)
    const argumentValue = argumentsText(item.arguments)
    if (name) pending.name = name
    if (argumentValue) pending.argumentsText = argumentValue
    this.pendingCalls.set(id, pending)
  }

  private completeFunctionCall(item: Readonly<Record<string, unknown>>): void {
    if (!isFunctionCallItem(item)) return
    const id = stringValue(item.id) || stringValue(item.call_id)
    if (!id) return
    const pending = this.pendingCalls.get(id) ?? this.createPendingCall(id)
    const name = stringValue(item.name)
    const argumentValue = argumentsText(item.arguments)
    if (name) pending.name = name
    if (argumentValue) pending.argumentsText = argumentValue
    this.pendingCalls.delete(id)
    this.recordToolCall(pending)
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
      finishReason: stringValue(response.status) || 'stop',
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
