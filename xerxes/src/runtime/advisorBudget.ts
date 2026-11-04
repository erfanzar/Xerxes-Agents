// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'

/**
 * Settings for the advisor: one bounded reasoning call from a stronger model.
 * Off by default; with no `profile` the advisor tool is not registered at all.
 */
export interface AdvisorConfig {
  readonly profile?: string
  readonly model?: string
  readonly maxCalls: number
  readonly maxUsd: number
  readonly autoOnCheckFail: boolean
  readonly timeoutMs: number
  readonly maxInputChars: number
}

export const ADVISOR_DEFAULT_MAX_CALLS = 5
export const ADVISOR_DEFAULT_MAX_USD = 1
export const ADVISOR_DEFAULT_TIMEOUT_MS = 120_000
export const ADVISOR_DEFAULT_MAX_INPUT_CHARS = 40_000

const ADVISOR_PROFILE_ENV = 'XERXES_ADVISOR_PROFILE'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function optionalText(source: Record<string, unknown>, key: string): string | undefined {
  const value = source[key]
  if (value === undefined || value === null) return undefined
  const field = `advisor.${key}`
  if (typeof value !== 'string') throw new ValidationError(field, 'must be a string', value)
  const trimmed = value.trim()
  if (!trimmed) throw new ValidationError(field, 'must not be empty', value)
  return trimmed
}

function integerSetting(source: Record<string, unknown>, key: string, fallback: number, min: number, max: number): number {
  const value = source[key]
  if (value === undefined || value === null) return fallback
  const field = `advisor.${key}`
  if (typeof value !== 'number' || !Number.isInteger(value)) throw new ValidationError(field, 'must be an integer', value)
  if (value < min || value > max) throw new ValidationError(field, `must be between ${min} and ${max}`, value)
  return value
}

function usdSetting(source: Record<string, unknown>, key: string, fallback: number): number {
  const value = source[key]
  if (value === undefined || value === null) return fallback
  const field = `advisor.${key}`
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new ValidationError(field, 'must be a number', value)
  if (value <= 0 || value > 1000) throw new ValidationError(field, 'must be greater than 0 and at most 1000', value)
  return value
}

function booleanSetting(source: Record<string, unknown>, key: string, fallback: boolean): boolean {
  const value = source[key]
  if (value === undefined || value === null) return fallback
  if (typeof value !== 'boolean') throw new ValidationError(`advisor.${key}`, 'must be a boolean', value)
  return value
}

/**
 * Parse `runtime.advisor` plus `XERXES_ADVISOR_PROFILE`. The env var wins over
 * the file. Invalid values throw a ValidationError naming the key; nothing
 * falls back to a looser value.
 */
export function parseAdvisorConfig(
  runtime: Record<string, unknown>,
  env: Record<string, string | undefined>,
): AdvisorConfig {
  const raw = runtime['advisor']
  if (raw !== undefined && raw !== null && !isRecord(raw)) {
    throw new ValidationError('advisor', 'must be an object', raw)
  }
  const source = isRecord(raw) ? raw : {}
  const envProfile = env[ADVISOR_PROFILE_ENV]?.trim()
  const profile = envProfile ? envProfile : optionalText(source, 'profile')
  const model = optionalText(source, 'model')
  return {
    ...(profile === undefined ? {} : { profile }),
    ...(model === undefined ? {} : { model }),
    maxCalls: integerSetting(source, 'max_calls', ADVISOR_DEFAULT_MAX_CALLS, 1, 100),
    maxUsd: usdSetting(source, 'max_usd', ADVISOR_DEFAULT_MAX_USD),
    autoOnCheckFail: booleanSetting(source, 'auto_on_check_fail', false),
    timeoutMs: integerSetting(source, 'timeout_ms', ADVISOR_DEFAULT_TIMEOUT_MS, 1000, 600_000),
    maxInputChars: integerSetting(source, 'max_input_chars', ADVISOR_DEFAULT_MAX_INPUT_CHARS, 2000, 200_000),
  }
}

export type AdvisorCallCheck = { readonly ok: true } | { readonly ok: false; readonly reason: string }

export interface AdvisorUsageRecord {
  readonly inputTokens: number
  readonly outputTokens: number
  /** Absent means the price is unknown; it is never estimated. */
  readonly costUsd?: number
  /** The attempt produced no answer (timeout, provider error). It still uses a call. */
  readonly failed?: boolean
}

export interface AdvisorBudgetSnapshot {
  readonly maxCalls: number
  readonly maxUsd: number
  readonly calls: number
  readonly failedCalls: number
  readonly callsLeft: number
  readonly inputTokens: number
  readonly outputTokens: number
  /** Sum of the calls whose price was known. */
  readonly spentUsd: number
  /** Successful calls whose price was unknown; they count against calls only. */
  readonly unknownCostCalls: number
  /** False when `spentUsd` omits at least one call of unknown price. */
  readonly costComplete: boolean
}

/** Per-session ceiling on advisor calls and spend. */
export class AdvisorBudget {
  private calls = 0
  private failedCalls = 0
  private inputTokens = 0
  private outputTokens = 0
  private spentUsd = 0
  private unknownCostCalls = 0

  constructor(private readonly limits: Pick<AdvisorConfig, 'maxCalls' | 'maxUsd'>) {}

  canCall(): AdvisorCallCheck {
    if (this.calls >= this.limits.maxCalls) {
      return { ok: false, reason: `advisor call limit reached (${this.calls} of ${this.limits.maxCalls} used)` }
    }
    if (this.spentUsd >= this.limits.maxUsd) {
      return {
        ok: false,
        reason: `advisor spend limit reached ($${this.spentUsd.toFixed(4)} of $${this.limits.maxUsd} used)`,
      }
    }
    return { ok: true }
  }

  record(usage: AdvisorUsageRecord): void {
    requireCount('inputTokens', usage.inputTokens)
    requireCount('outputTokens', usage.outputTokens)
    if (usage.costUsd !== undefined && (!Number.isFinite(usage.costUsd) || usage.costUsd < 0)) {
      throw new ValidationError('costUsd', 'must be a non-negative finite number or absent', usage.costUsd)
    }
    this.calls += 1
    this.inputTokens += usage.inputTokens
    this.outputTokens += usage.outputTokens
    if (usage.failed) {
      this.failedCalls += 1
      return
    }
    if (usage.costUsd === undefined) this.unknownCostCalls += 1
    else this.spentUsd += usage.costUsd
  }

  /** Calls still allowed: zero as soon as either limit blocks the next call. */
  callsLeft(): number {
    return this.canCall().ok ? this.limits.maxCalls - this.calls : 0
  }

  snapshot(): AdvisorBudgetSnapshot {
    return {
      maxCalls: this.limits.maxCalls,
      maxUsd: this.limits.maxUsd,
      calls: this.calls,
      failedCalls: this.failedCalls,
      callsLeft: this.callsLeft(),
      inputTokens: this.inputTokens,
      outputTokens: this.outputTokens,
      spentUsd: this.spentUsd,
      unknownCostCalls: this.unknownCostCalls,
      costComplete: this.unknownCostCalls === 0,
    }
  }
}

function requireCount(field: string, value: number): void {
  if (!Number.isFinite(value) || value < 0) {
    throw new ValidationError(field, 'must be a non-negative finite number', value)
  }
}
