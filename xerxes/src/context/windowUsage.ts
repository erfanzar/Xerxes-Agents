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

/** Session metadata key holding the latest provider-measured calibration. */
export const CONTEXT_CALIBRATION_METADATA_KEY = 'context_calibration'

const MIN_CALIBRATION_RATIO = 0.5
const MAX_CALIBRATION_RATIO = 4

/**
 * A heuristic count is not what the provider bills. On code-heavy Claude
 * transcripts it ran ~1.8x low: a 1M window the meter put at 54% was
 * rejected as full, and compaction — gated on the same estimate — never ran.
 * The provider's own count of the previous request is the correction.
 */
export interface PromptCalibration {
  /** Provider tokens per estimated token, from the latest measured request. */
  readonly ratio: number
  /**
   * Scale the estimate of the request about to be sent. `observedPromptTokens`
   * is the provider's count of the request estimated by the previous call;
   * omit it when that pairing no longer holds (the history was replaced).
   */
  project(estimate: number, observedPromptTokens?: number): number
}

export function promptCalibration(initialRatio = 1): PromptCalibration {
  let ratio = clampCalibrationRatio(initialRatio)
  let sentEstimate: number | undefined
  return {
    get ratio() { return ratio },
    project(estimate, observedPromptTokens) {
      if (observedPromptTokens !== undefined && observedPromptTokens > 0 && sentEstimate !== undefined && sentEstimate > 0) {
        ratio = clampCalibrationRatio(observedPromptTokens / sentEstimate)
      }
      sentEstimate = estimate
      return Math.ceil(estimate * ratio)
    },
  }
}

/** The persisted ratio for `model`, or 1 when absent, malformed, or measured on another model. */
export function contextCalibrationRatio(metadata: Readonly<Record<string, unknown>>, model: string): number {
  const value = metadata[CONTEXT_CALIBRATION_METADATA_KEY]
  if (!value || typeof value !== 'object' || Array.isArray(value)) return 1
  const { model: measured, ratio } = value as Record<string, unknown>
  if (measured !== model || typeof ratio !== 'number' || !Number.isFinite(ratio) || ratio <= 0) return 1
  return clampCalibrationRatio(ratio)
}

function clampCalibrationRatio(ratio: number): number {
  if (!Number.isFinite(ratio) || ratio <= 0) return 1
  return Math.min(MAX_CALIBRATION_RATIO, Math.max(MIN_CALIBRATION_RATIO, ratio))
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
