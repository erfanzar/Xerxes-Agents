// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { SmartTokenCounter } from './tokenCounter.js'

export interface WindowUsageOptions {
  readonly model: string
  readonly systemPrompt?: string
  readonly tokenCounter?: SmartTokenCounter
  readonly toolSchemas?: readonly Readonly<Record<string, unknown>>[]
}

/** Return provider-request messages that sit outside the persisted chat transcript. */
export function requestScaffoldingMessages(
  options: Pick<WindowUsageOptions, 'systemPrompt' | 'toolSchemas'> = {},
): Array<Record<string, unknown>> {
  const messages: Array<Record<string, unknown>> = []
  if (options.systemPrompt) messages.push({ role: 'system', content: options.systemPrompt })
  if (options.toolSchemas?.length) {
    messages.push({
      role: 'system',
      content: '[available tool schemas]\n' + stableJson(options.toolSchemas),
    })
  }
  return messages
}

/** Estimate prompt tokens attributable to system instructions and tool schemas alone. */
export function estimateRequestOverheadTokens(options: WindowUsageOptions): number {
  const scaffolding = requestScaffoldingMessages(options)
  return scaffolding.length ? countMessages(scaffolding, options) : 0
}

/** Estimate the live provider request window, including non-transcript request scaffolding. */
export function estimateContextTokens(
  messages: readonly Readonly<Record<string, unknown>>[],
  options: WindowUsageOptions,
): number {
  const requestMessages = [...requestScaffoldingMessages(options), ...messages]
  return requestMessages.length ? countMessages(requestMessages, options) : 0
}

function countMessages(messages: readonly Readonly<Record<string, unknown>>[], options: WindowUsageOptions): number {
  try {
    const counter = options.tokenCounter ?? new SmartTokenCounter({ model: options.model })
    return Math.max(0, counter.countTokens(messages))
  } catch {
    const text = messages.map(message => String(message.role ?? '') + ': ' + contentText(message.content)).join('\n')
    return Math.max(0, Math.floor(text.length / 4))
  }
}

function contentText(value: unknown): string {
  if (typeof value === 'string') return value
  if (value === undefined || value === null) return ''
  return stableJson(value)
}

function stableJson(value: unknown): string {
  try {
    return JSON.stringify(sortValue(value)) ?? String(value)
  } catch {
    return String(value)
  }
}

function sortValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortValue)
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => [key, sortValue(item)]))
  }
  return value
}
