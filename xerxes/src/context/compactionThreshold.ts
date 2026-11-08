// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * When automatic compaction starts. One number carries both forms people use:
 * a share of the prompt budget (0 < n <= 1, so 0.6 is 60%) or an absolute
 * token count (n >= MIN_THRESHOLD_TOKENS, so 400000 is 400K tokens). 0 is off.
 * Main sessions, mid-turn checks and delegated children all read it here, so
 * none of them can run against a different threshold.
 */

import { mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'

import { ValidationError } from '../core/errors.js'

/** Compact once a task's prompt reaches 60% of the model's prompt budget. */
export const DEFAULT_AUTO_COMPACT_THRESHOLD = 0.6

/** Smallest absolute threshold. Between 1 and this a number is neither a share nor a useful token count. */
export const MIN_THRESHOLD_TOKENS = 1024

/** Clamp a stored threshold into its encoded form; 0 means off. */
export function normalizeCompactionThreshold(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 0
  if (value <= 1) return value
  // An old fraction above 1 meant "only when the window is full".
  if (value < MIN_THRESHOLD_TOKENS) return 1
  return Math.floor(value)
}

/**
 * Token count at which a context is due for compaction, or 0 when compaction
 * is disabled or the budget is unknown. A token count at or above a smaller
 * model's budget would only compact once that window is already full, so
 * such a model uses the default share of its own budget instead.
 */
export function compactionThresholdTokens(promptBudget: number, threshold: number): number {
  const normalized = normalizeCompactionThreshold(threshold)
  if (normalized <= 0 || !Number.isFinite(promptBudget) || promptBudget <= 0) return 0
  const budget = Math.floor(promptBudget)
  if (normalized <= 1) return Math.floor(budget * normalized)
  return normalized < budget ? normalized : Math.floor(budget * DEFAULT_AUTO_COMPACT_THRESHOLD)
}

/**
 * Read what a person typed or saved: `60%`, `60`, `0.6`, `400k`, `1m`,
 * `400000`, or `off` / `0`. Throws a ValidationError naming the setting.
 */
export function parseCompactionThreshold(value: unknown): number {
  const parsed = readThreshold(value)
  if (parsed === undefined) {
    throw new ValidationError(
      'auto_compact_threshold',
      `must be a percentage such as 60%, a fraction such as 0.6, a token count of at least ${MIN_THRESHOLD_TOKENS} such as 400k, or off`,
      value,
    )
  }
  return parsed
}

/** Lenient form for values already on disk: an unreadable value is ignored. */
export function optionalCompactionThreshold(value: unknown): number | undefined {
  return readThreshold(value)
}

/** `60%`, `400K tokens`, or `off`. */
export function describeCompactionThreshold(value: number): string {
  const normalized = normalizeCompactionThreshold(value)
  if (normalized <= 0) return 'off'
  if (normalized <= 1) return `${Math.round(normalized * 1000) / 10}%`
  if (normalized % 1_000_000 === 0) return `${normalized / 1_000_000}M tokens`
  if (normalized % 1_000 === 0) return `${normalized / 1_000}K tokens`
  return `${normalized.toLocaleString('en-US')} tokens`
}

/** Where Settings keeps the threshold a person chose. */
export function compactionSettingsPath(home: string): string {
  return join(home, 'daemon', 'compaction.json')
}

/** The saved threshold, or undefined when nothing readable is saved. */
export function readCompactionSetting(file: string): number | undefined {
  let text: string
  try {
    text = readFileSync(file, 'utf8')
  } catch {
    return undefined
  }
  try {
    const value: unknown = JSON.parse(text)
    return value !== null && typeof value === 'object'
      ? optionalCompactionThreshold((value as Record<string, unknown>).auto_compact_threshold)
      : undefined
  } catch {
    return undefined
  }
}

/** Save a threshold in its parsed form, atomically. Returns what was saved. */
export function saveCompactionSetting(file: string, threshold: unknown): number {
  const parsed = parseCompactionThreshold(threshold)
  mkdirSync(dirname(file), { recursive: true })
  const temp = `${file}.tmp-${process.pid}`
  writeFileSync(temp, `${JSON.stringify({ version: 1, auto_compact_threshold: parsed })}\n`, { encoding: 'utf8', mode: 0o600 })
  renameSync(temp, file)
  return parsed
}

function readThreshold(value: unknown): number | undefined {
  if (typeof value === 'number') return fromNumber(value)
  if (typeof value !== 'string') return undefined
  const text = value.trim().toLowerCase().replace(/[\s,_]/g, '').replace(/tokens?$/, '')
  if (text === 'off') return 0
  const match = /^(\d+(?:\.\d+)?)(%|k|m)?$/.exec(text)
  if (!match) return undefined
  const amount = Number(match[1])
  const unit = match[2]
  if (unit === '%') return amount >= 0 && amount <= 100 ? amount / 100 : undefined
  if (unit === 'k') return tokensOnly(amount * 1_000)
  if (unit === 'm') return tokensOnly(amount * 1_000_000)
  return fromNumber(amount)
}

function fromNumber(value: number): number | undefined {
  if (!Number.isFinite(value) || value < 0) return undefined
  if (value <= 1) return value
  // A bare 60 or 75 is a percentage; nobody compacts at 60 tokens.
  if (value <= 100) return value / 100
  return tokensOnly(value)
}

function tokensOnly(value: number): number | undefined {
  return Number.isFinite(value) && value >= MIN_THRESHOLD_TOKENS ? Math.floor(value) : undefined
}
