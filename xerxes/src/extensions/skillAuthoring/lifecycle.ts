// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'

export interface SkillUsageEvent {
  readonly durationMs?: number
  readonly failureReason?: string
  readonly kind: 'used'
  readonly outcome: string
  readonly skillName: string
  readonly timestamp?: number
  readonly version?: string
}

export interface SkillFeedbackEvent {
  readonly kind: 'feedback'
  readonly rating: 'bad' | 'good'
  readonly skillName: string
}

export interface SkillAuthoredEvent {
  readonly kind: 'authored'
  readonly skillName: string
  readonly timestamp?: number
  readonly version?: string
}

export type SkillTelemetryEvent = SkillAuthoredEvent | SkillFeedbackEvent | SkillUsageEvent

export interface SkillStats {
  readonly authoredAt?: number
  readonly durationsMs: readonly number[]
  readonly failures: number
  readonly feedbackBad: number
  readonly feedbackGood: number
  readonly invocations: number
  readonly lastFailureReason: string
  readonly lastInvoked?: number
  readonly skillName: string
  readonly successes: number
  readonly version: string
}

interface MutableSkillStats {
  authoredAt: number | undefined
  durationsMs: number[]
  failures: number
  feedbackBad: number
  feedbackGood: number
  invocations: number
  lastFailureReason: string
  lastInvoked: number | undefined
  skillName: string
  successes: number
  version: string
}

/** In-memory telemetry aggregation independent from audit or persistence implementations. */
export class SkillTelemetry {
  private readonly entries = new Map<string, MutableSkillStats>()
  private readonly now: () => number

  constructor(options: { readonly now?: () => number } = {}) {
    this.now = options.now ?? Date.now
  }

  allStats(): Record<string, SkillStats> {
    const result: Record<string, SkillStats> = {}
    for (const [name, stats] of this.entries) {
      result[name] = snapshot(stats)
    }
    return result
  }

  candidatesForDeprecation(
    options: { readonly maxSuccessRate?: number; readonly minInvocations?: number } = {},
  ): string[] {
    const minInvocations = nonNegativeInteger(options.minInvocations ?? 5, 'minInvocations')
    const maxSuccessRate = rate(options.maxSuccessRate ?? 0.4, 'maxSuccessRate')
    return [...this.entries.values()]
      .filter(stats => stats.invocations >= minInvocations && successRate(stats) <= maxSuccessRate)
      .sort((left, right) => successRate(left) - successRate(right) || left.skillName.localeCompare(right.skillName))
      .map(stats => stats.skillName)
  }

  record(event: SkillTelemetryEvent): void {
    switch (event.kind) {
      case 'used':
        this.recordUsage(event)
        return
      case 'feedback':
        this.recordFeedback(event)
        return
      case 'authored':
        this.recordAuthored(event)
        return
    }
  }

  recordAuthored(event: Omit<SkillAuthoredEvent, 'kind'>): void {
    const stats = this.entry(event.skillName)
    if (event.version) {
      stats.version = event.version
    }
    stats.authoredAt = event.timestamp ?? this.now()
  }

  recordFeedback(event: Omit<SkillFeedbackEvent, 'kind'>): void {
    const stats = this.entry(event.skillName)
    if (event.rating === 'good') {
      stats.feedbackGood += 1
    } else {
      stats.feedbackBad += 1
    }
  }

  recordUsage(event: Omit<SkillUsageEvent, 'kind'>): void {
    const stats = this.entry(event.skillName)
    stats.invocations += 1
    stats.lastInvoked = event.timestamp ?? this.now()
    if (event.version) {
      stats.version = event.version
    }
    if (event.outcome === 'success') {
      stats.successes += 1
    } else {
      stats.failures += 1
      if (event.failureReason || event.outcome) {
        stats.lastFailureReason = event.failureReason ?? event.outcome
      }
    }
    if (event.durationMs !== undefined) {
      if (!Number.isFinite(event.durationMs) || event.durationMs < 0) {
        throw new RangeError('durationMs must be a finite non-negative number')
      }
      if (event.durationMs > 0) {
        insertSorted(stats.durationsMs, event.durationMs)
      }
    }
  }

  stats(skillName: string): SkillStats | undefined {
    const stats = this.entries.get(skillName)
    return stats === undefined ? undefined : snapshot(stats)
  }

  private entry(skillName: string): MutableSkillStats {
    const normalized = skillName.trim()
    if (!normalized) {
      throw new TypeError('skillName must not be empty')
    }
    let stats = this.entries.get(normalized)
    if (!stats) {
      stats = {
        skillName: normalized,
        version: '',
        invocations: 0,
        successes: 0,
        failures: 0,
        durationsMs: [],
        lastInvoked: undefined,
        lastFailureReason: '',
        feedbackGood: 0,
        feedbackBad: 0,
        authoredAt: undefined,
      }
      this.entries.set(normalized, stats)
    }
    return stats
  }
}

export function feedbackScore(stats: SkillStats): number {
  return stats.feedbackGood - stats.feedbackBad
}

export function percentile(durationsMs: readonly number[], quantile: number): number {
  if (!durationsMs.length) {
    return 0
  }
  if (!Number.isFinite(quantile) || quantile < 0 || quantile > 1) {
    throw new RangeError('quantile must be between 0 and 1')
  }
  const index = Math.max(0, Math.min(durationsMs.length - 1, Math.round(quantile * (durationsMs.length - 1))))
  return durationsMs[index] ?? 0
}

export function successRate(stats: Pick<SkillStats, 'invocations' | 'successes'>): number {
  return stats.invocations === 0 ? 0 : stats.successes / stats.invocations
}

export interface SkillVariantOptions {
  readonly baseName: string
  readonly rollout?: number
  readonly variantName: string
}

/** Deterministic canary rollout definition. */
export class SkillVariant {
  readonly baseName: string
  readonly rollout: number
  readonly variantName: string

  constructor(options: SkillVariantOptions | string, variantName?: string, rollout = 0.5) {
    const resolved = typeof options === 'string'
      ? { baseName: options, variantName: variantName ?? '', rollout }
      : options
    if (!resolved.baseName.trim() || !resolved.variantName.trim()) {
      throw new TypeError('variant baseName and variantName must not be empty')
    }
    this.baseName = resolved.baseName
    this.variantName = resolved.variantName
    this.rollout = clampRollout(resolved.rollout ?? 0.5)
  }
}

/** Routes the same user to the same canary bucket without external state. */
export class SkillVariantPicker {
  private readonly variants = new Map<string, SkillVariant>()

  add(variant: SkillVariant): void {
    this.variants.set(variant.baseName, variant)
  }

  all(): Map<string, SkillVariant> {
    return new Map(this.variants)
  }

  get(baseName: string): SkillVariant | undefined {
    return this.variants.get(baseName)
  }

  pick(baseName: string, userId = ''): string {
    const variant = this.variants.get(baseName)
    if (!variant || variant.rollout <= 0) {
      return baseName
    }
    if (variant.rollout >= 1) {
      return variant.variantName
    }
    const digest = createHash('md5').update(userId + '::' + baseName, 'utf8').digest()
    const bucket = digest.readUInt32BE(0) / 0xffff_ffff
    return bucket < variant.rollout ? variant.variantName : baseName
  }

  remove(baseName: string): boolean {
    return this.variants.delete(baseName)
  }
}

export type DeprecationAction = 'deprecated' | 'kept' | 'missing' | 'proposed' | 'unavailable'

export interface DeprecationDecision {
  readonly action: DeprecationAction
  readonly deprecatedLocation?: string
  readonly reason: string
  readonly skillName: string
}

/** Explicit host boundary for an irreversible retirement action such as a file rename or registry update. */
export interface SkillRetirementPort {
  deprecate(input: { readonly reason: string; readonly skillName: string }): Promise<{
    readonly action: 'deprecated' | 'kept' | 'missing'
    readonly deprecatedLocation?: string
    readonly reason?: string
  }>
}

export interface SkillLifecycleManagerOptions {
  readonly maxSuccessRate?: number
  readonly minInvocations?: number
  readonly retirement?: SkillRetirementPort
}

/**
 * Proposes deprecation from telemetry and applies it only via an explicit host port.
 *
 * It never renames files or mutates a SkillRegistry by itself.
 */
export class SkillLifecycleManager {
  readonly maxSuccessRate: number
  readonly minInvocations: number
  private readonly retirement: SkillRetirementPort | undefined
  private readonly telemetry: SkillTelemetry

  constructor(telemetry: SkillTelemetry, options: SkillLifecycleManagerOptions = {}) {
    this.telemetry = telemetry
    this.minInvocations = nonNegativeInteger(options.minInvocations ?? 10, 'minInvocations')
    this.maxSuccessRate = rate(options.maxSuccessRate ?? 0.4, 'maxSuccessRate')
    this.retirement = options.retirement
  }

  async apply(decisions: readonly DeprecationDecision[] = this.evaluate()): Promise<DeprecationDecision[]> {
    if (!this.retirement) {
      return decisions.map(decision => decision.action === 'proposed'
        ? {
            skillName: decision.skillName,
            action: 'unavailable',
            reason: decision.reason + '; no retirement port configured',
          }
        : decision)
    }
    const applied: DeprecationDecision[] = []
    for (const decision of decisions) {
      if (decision.action !== 'proposed') {
        applied.push(decision)
        continue
      }
      try {
        const outcome = await this.retirement.deprecate({
          skillName: decision.skillName,
          reason: decision.reason,
        })
        applied.push({
          skillName: decision.skillName,
          action: outcome.action,
          reason: outcome.reason ?? decision.reason,
          ...(outcome.deprecatedLocation === undefined ? {} : { deprecatedLocation: outcome.deprecatedLocation }),
        })
      } catch {
        applied.push({
          skillName: decision.skillName,
          action: 'kept',
          reason: decision.reason + '; retirement port failed',
        })
      }
    }
    return applied
  }

  evaluate(): DeprecationDecision[] {
    return this.telemetry.candidatesForDeprecation({
      minInvocations: this.minInvocations,
      maxSuccessRate: this.maxSuccessRate,
    }).map(skillName => {
      const stats = this.telemetry.stats(skillName)
      const reason = stats
        ? 'success_rate=' + Math.round(successRate(stats) * 100) + '% after ' + stats.invocations + ' invocations'
        : 'stats unavailable'
      return { skillName, action: 'proposed', reason }
    })
  }
}

function clampRollout(value: number): number {
  if (!Number.isFinite(value)) {
    throw new RangeError('rollout must be finite')
  }
  return Math.max(0, Math.min(1, value))
}

function insertSorted(values: number[], value: number): void {
  let low = 0
  let high = values.length
  while (low < high) {
    const middle = Math.floor((low + high) / 2)
    if ((values[middle] ?? value) < value) {
      low = middle + 1
    } else {
      high = middle
    }
  }
  values.splice(low, 0, value)
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError(name + ' must be a non-negative integer')
  }
  return value
}

function rate(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new RangeError(name + ' must be between 0 and 1')
  }
  return value
}

function snapshot(stats: MutableSkillStats): SkillStats {
  return {
    skillName: stats.skillName,
    version: stats.version,
    invocations: stats.invocations,
    successes: stats.successes,
    failures: stats.failures,
    durationsMs: Object.freeze([...stats.durationsMs]),
    lastFailureReason: stats.lastFailureReason,
    feedbackGood: stats.feedbackGood,
    feedbackBad: stats.feedbackBad,
    ...(stats.lastInvoked === undefined ? {} : { lastInvoked: stats.lastInvoked }),
    ...(stats.authoredAt === undefined ? {} : { authoredAt: stats.authoredAt }),
  }
}
