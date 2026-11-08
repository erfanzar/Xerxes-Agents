// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Escalation policy (M2.3): decide, by rule and from result-check findings,
 * whether a failed attempt is retried locally, moved to another model, sent to
 * the cloud advisor, or stopped. The model never judges its own output here.
 *
 * Off by default. The controller is pure and deterministic: the caller reports
 * each attempt's findings and cost, and acts on the returned step.
 */

import { ValidationError } from '../core/errors.js'
import { modelCard } from '../llms/modelCards.js'
import type { DataBoundaryMode } from '../security/boundaryTypes.js'
import { buildCheckFailureAdvisorRequest } from '../tools/advisorTool.js'
import { parseAdvisorConfig } from './advisorBudget.js'
import { renderFindings, type CheckFinding } from './resultChecks.js'

/** Environment variable that overrides `runtime.escalation.mode`. */
export const ESCALATION_ENV = 'XERXES_ESCALATION'

export const ESCALATION_DEFAULT_MAX_LOCAL_RETRIES = 2
export const ESCALATION_DEFAULT_MAX_MODEL_CHANGES = 1
export const ESCALATION_DEFAULT_MAX_ADVISOR_CALLS = 2
export const ESCALATION_DEFAULT_MAX_USD = 1

export interface EscalationConfig {
  readonly mode: 'off' | 'on'
  readonly maxLocalRetries: number
  readonly maxModelChanges: number
  readonly maxAdvisorCalls: number
  readonly maxUsd: number
}

const KNOWN_KEYS = new Set(['mode', 'max_local_retries', 'max_model_changes', 'max_advisor_calls', 'max_usd'])

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function normalizeMode(value: unknown, field: string): 'off' | 'on' {
  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase()
    if (normalized === 'off' || normalized === 'on') return normalized
  }
  throw new ValidationError(field, "must be 'off' or 'on'", value)
}

function integerSetting(source: Record<string, unknown>, key: string, fallback: number, min: number, max: number): number {
  const value = source[key]
  if (value === undefined || value === null) return fallback
  const field = `escalation.${key}`
  if (typeof value !== 'number' || !Number.isInteger(value)) throw new ValidationError(field, 'must be an integer', value)
  if (value < min || value > max) throw new ValidationError(field, `must be between ${min} and ${max}`, value)
  return value
}

function usdSetting(source: Record<string, unknown>, key: string, fallback: number): number {
  const value = source[key]
  if (value === undefined || value === null) return fallback
  const field = `escalation.${key}`
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new ValidationError(field, 'must be a number', value)
  if (value <= 0 || value > 1000) throw new ValidationError(field, 'must be greater than 0 and at most 1000', value)
  return value
}

/**
 * Parse `runtime.escalation` plus `XERXES_ESCALATION`. The env var wins for
 * `mode`; with neither set, `defaultMode` applies (the host picks it per
 * session). Invalid values and unknown keys throw a ValidationError naming the key.
 */
export function parseEscalationConfig(
  runtime: Record<string, unknown>,
  env: Record<string, string | undefined>,
  defaultMode: 'off' | 'on' = 'off',
): EscalationConfig {
  const raw = runtime['escalation']
  if (raw !== undefined && raw !== null && !isRecord(raw)) {
    throw new ValidationError('escalation', 'must be an object', raw)
  }
  const source = isRecord(raw) ? raw : {}
  for (const key of Object.keys(source)) {
    if (!KNOWN_KEYS.has(key)) throw new ValidationError(`escalation.${key}`, 'unknown setting', source[key])
  }
  const envMode = env[ESCALATION_ENV]?.trim()
  const fileMode = source['mode']
  const mode = envMode
    ? normalizeMode(envMode, ESCALATION_ENV)
    : fileMode === undefined || fileMode === null
      ? defaultMode
      : normalizeMode(fileMode, 'escalation.mode')
  return {
    mode,
    maxLocalRetries: integerSetting(source, 'max_local_retries', ESCALATION_DEFAULT_MAX_LOCAL_RETRIES, 0, 10),
    maxModelChanges: integerSetting(source, 'max_model_changes', ESCALATION_DEFAULT_MAX_MODEL_CHANGES, 0, 5),
    maxAdvisorCalls: integerSetting(source, 'max_advisor_calls', ESCALATION_DEFAULT_MAX_ADVISOR_CALLS, 0, 20),
    maxUsd: usdSetting(source, 'max_usd', ESCALATION_DEFAULT_MAX_USD),
  }
}

export interface EscalationDefaultInput {
  readonly runtime: Record<string, unknown>
  readonly env: Record<string, string | undefined>
  /** The model the turn runs on. */
  readonly model: string
  /** Whether that model runs on this machine; consulted only when it matters. */
  readonly local: () => boolean
}

/**
 * The escalation mode when neither `runtime.escalation.mode` nor
 * XERXES_ESCALATION is set: on for a model whose card says so (Bonsai) or a local one (a failed
 * check is cheap to retry there, and the plan's "escalation by rule" applies),
 * and when `advisor.auto_on_check_fail` asks for it; off otherwise. Every entry
 * point uses this, so a benchmark run gets the same cascade as the daemon.
 */
export function defaultEscalationMode(input: EscalationDefaultInput): 'off' | 'on' {
  if (parseAdvisorConfig(input.runtime, input.env).autoOnCheckFail) return 'on'
  if (modelCard(input.model).escalationByDefault) return 'on'
  try {
    return input.local() ? 'on' : 'off'
  } catch {
    return 'off'
  }
}

export type EscalationStep =
  | { readonly action: 'accept' }
  | { readonly action: 'retry_local'; readonly feedback: string }
  | { readonly action: 'change_model'; readonly feedback: string }
  | { readonly action: 'ask_advisor'; readonly question: string; readonly context: string }
  | { readonly action: 'stop'; readonly reason: string; readonly findings: readonly CheckFinding[] }

export type EscalationAction = EscalationStep['action']

export interface EscalationOptions {
  readonly config: EscalationConfig
  readonly boundaryMode: DataBoundaryMode
  /** The task's context includes private data. */
  readonly touchesPrivate: boolean
  readonly advisorAvailable: boolean
  readonly alternateModelAvailable: boolean
  /** What the agent was doing; goes into the advisor request. */
  readonly taskSummary?: string
}

export interface EscalationAttempt {
  readonly findings: readonly CheckFinding[]
  /** Cost of this attempt in USD. Absent means unknown (counted as 0 and reported). */
  readonly costUsd?: number
}

export interface EscalationSummary {
  readonly attempts: number
  readonly localRetries: number
  readonly modelChanges: number
  readonly advisorCalls: number
  readonly spentUsd: number
  readonly unknownCostAttempts: number
  readonly lastAction: EscalationAction | undefined
}

interface Score {
  readonly errors: number
  readonly warns: number
}

const DEFAULT_TASK_SUMMARY = 'The local agent was working on a task and its result failed automated checks.'

function scoreOf(findings: readonly CheckFinding[]): Score {
  let errors = 0
  let warns = 0
  for (const finding of findings) {
    if (finding.severity === 'error') errors += 1
    else warns += 1
  }
  return { errors, warns }
}

/** Strictly better: fewer errors, or the same errors and fewer warnings. */
function improved(now: Score, before: Score): boolean {
  return now.errors < before.errors || (now.errors === before.errors && now.warns < before.warns)
}

function usd(value: number): string {
  return `$${value.toFixed(4)}`
}

/**
 * One controller per task attempt chain. Call {@link next} with each attempt's
 * check findings; it returns the next step. Once it returns `stop` it keeps
 * returning that same stop.
 */
export class EscalationController {
  private readonly options: EscalationOptions
  private attempts = 0
  private localRetries = 0
  private modelChanges = 0
  private advisorCalls = 0
  private spentUsd = 0
  private unknownCostAttempts = 0
  private lastAction: EscalationAction | undefined
  private previousScore: Score | undefined
  private noImprovementStreak = 0
  private stopped: Extract<EscalationStep, { action: 'stop' }> | undefined

  constructor(options: EscalationOptions) {
    this.options = options
  }

  next(attempt: EscalationAttempt): EscalationStep {
    if (this.stopped !== undefined) return this.stopped
    this.attempts += 1
    this.recordCost(attempt.costUsd)

    const errorFindings = attempt.findings.filter(finding => finding.severity === 'error')
    if (errorFindings.length === 0) return this.finish({ action: 'accept' })

    const { config } = this.options
    if (config.mode === 'off') return this.stop('escalation off', attempt.findings)
    if (this.spentUsd >= config.maxUsd) {
      return this.stop(`spend limit reached (${usd(this.spentUsd)} of ${usd(config.maxUsd)})`, attempt.findings)
    }

    const score = scoreOf(attempt.findings)
    if (this.previousScore !== undefined) {
      this.noImprovementStreak = improved(score, this.previousScore) ? 0 : this.noImprovementStreak + 1
    }
    this.previousScore = score
    if (this.noImprovementStreak >= 2) {
      return this.stop(`no improvement after ${this.attempts} attempts`, attempt.findings)
    }

    const report = renderFindings(attempt.findings)

    if (this.localRetries < config.maxLocalRetries) {
      this.localRetries += 1
      return this.finish({
        action: 'retry_local',
        feedback:
          `${report}\n\nYour result failed these checks. Fix exactly these problems and leave everything else unchanged ` +
          `(local retry ${this.localRetries} of ${config.maxLocalRetries}).`,
      })
    }

    if (this.options.alternateModelAvailable && this.modelChanges < config.maxModelChanges) {
      this.modelChanges += 1
      this.noImprovementStreak = 0
      return this.finish({
        action: 'change_model',
        feedback:
          `${report}\n\nThe previous model could not fix these checks. Redo the task and make sure each failed check ` +
          `above passes (model change ${this.modelChanges} of ${config.maxModelChanges}).`,
      })
    }

    const advisorBlock = this.advisorBlockReason()
    if (advisorBlock === undefined && this.advisorCalls < config.maxAdvisorCalls) {
      this.advisorCalls += 1
      this.noImprovementStreak = 0
      const request = buildCheckFailureAdvisorRequest(report, this.options.taskSummary ?? DEFAULT_TASK_SUMMARY)
      return this.finish({ action: 'ask_advisor', question: request.question, context: request.context })
    }

    return this.stop(this.exhaustedReason(advisorBlock), attempt.findings)
  }

  summary(): EscalationSummary {
    return {
      attempts: this.attempts,
      localRetries: this.localRetries,
      modelChanges: this.modelChanges,
      advisorCalls: this.advisorCalls,
      spentUsd: this.spentUsd,
      unknownCostAttempts: this.unknownCostAttempts,
      lastAction: this.lastAction,
    }
  }

  private recordCost(costUsd: number | undefined): void {
    if (costUsd === undefined) {
      this.unknownCostAttempts += 1
      return
    }
    if (!Number.isFinite(costUsd) || costUsd < 0) {
      throw new ValidationError('costUsd', 'must be a non-negative finite number', costUsd)
    }
    this.spentUsd += costUsd
  }

  /** Why the cloud advisor may not be used, or undefined when it may. */
  private advisorBlockReason(): string | undefined {
    const { advisorAvailable, boundaryMode, touchesPrivate } = this.options
    if (!advisorAvailable) return 'no advisor is configured'
    if (boundaryMode === 'private') return 'the cloud advisor is not allowed because the data boundary is private'
    if (touchesPrivate && boundaryMode !== 'off') {
      return `the cloud advisor is not allowed because the task touches private data (data boundary: ${boundaryMode})`
    }
    return undefined
  }

  private exhaustedReason(advisorBlock: string | undefined): string {
    const { config } = this.options
    const parts = [`local retries used ${this.localRetries}/${config.maxLocalRetries}`]
    if (!this.options.alternateModelAvailable) parts.push('no alternate model is available')
    else parts.push(`model changes used ${this.modelChanges}/${config.maxModelChanges}`)
    parts.push(advisorBlock ?? `advisor calls used ${this.advisorCalls}/${config.maxAdvisorCalls}`)
    return `escalation exhausted after ${this.attempts} attempts: ${parts.join('; ')}`
  }

  private finish<T extends EscalationStep>(step: T): T {
    this.lastAction = step.action
    return step
  }

  private stop(reason: string, findings: readonly CheckFinding[]): EscalationStep {
    const step = this.finish({ action: 'stop' as const, reason, findings })
    this.stopped = step
    return step
  }
}
