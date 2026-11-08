// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The per-reply budget, one of the stuck signals: how long one model reply may
 * reason before it acts. A reply still reasoning past either limit, with no
 * tool call and no answer text yet, is stopped, its reasoning is dropped, and
 * the model is asked again with a note (`streaming/loop.ts`). One runaway reply
 * then costs minutes, not the whole task.
 *
 * `runtime.reply_budget` sets it for any model; unset, the model card decides
 * (Bonsai has one, other models have none).
 */

import { ValidationError } from '../core/errors.js'
import { modelCard, type ReplyBudget } from '../llms/modelCards.js'

export type { ReplyBudget } from '../llms/modelCards.js'

/** Used for a field the setting leaves out when the model card has no budget. */
export const DEFAULT_REPLY_BUDGET: ReplyBudget = Object.freeze({ maxReasoningTokens: 8_192, maxSeconds: 600 })

const KNOWN_KEYS = new Set(['max_reasoning_tokens', 'max_seconds'])

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function integerSetting(source: Record<string, unknown>, key: string, fallback: number, min: number, max: number): number {
  const value = source[key]
  if (value === undefined || value === null) return fallback
  const field = `reply_budget.${key}`
  if (typeof value !== 'number' || !Number.isInteger(value)) throw new ValidationError(field, 'must be an integer', value)
  if (value < min || value > max) throw new ValidationError(field, `must be between ${min} and ${max}`, value)
  return value
}

/**
 * The budget for `model` under `runtime.reply_budget`, or null for none.
 *
 * - unset: the model card's budget, or none;
 * - `false` or `'off'`: none;
 * - `true` or `'on'`: the card's budget, or {@link DEFAULT_REPLY_BUDGET};
 * - `{ max_reasoning_tokens, max_seconds }`: those, each defaulting as for `true`.
 *
 * Anything else, and any other key, throws a ValidationError naming the setting.
 */
export function resolveReplyBudget(runtime: Record<string, unknown>, model: string): ReplyBudget | null {
  const raw = runtime['reply_budget']
  const card = modelCard(model).replyBudget
  if (raw === undefined || raw === null) return card ?? null
  if (raw === false || raw === 'off') return null
  const base = card ?? DEFAULT_REPLY_BUDGET
  if (raw === true || raw === 'on') return base
  if (!isRecord(raw)) {
    throw new ValidationError('reply_budget', "must be 'off', 'on', or an object with max_reasoning_tokens and max_seconds", raw)
  }
  for (const key of Object.keys(raw)) {
    if (!KNOWN_KEYS.has(key)) throw new ValidationError(`reply_budget.${key}`, 'unknown setting', raw[key])
  }
  return Object.freeze({
    maxReasoningTokens: integerSetting(raw, 'max_reasoning_tokens', base.maxReasoningTokens, 256, 1_000_000),
    maxSeconds: integerSetting(raw, 'max_seconds', base.maxSeconds, 10, 86_400),
  })
}
