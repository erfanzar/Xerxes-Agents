// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { renderIntervention } from './interventions.js'

/**
 * Consecutive refused tool calls tolerated before a turn stops.
 *
 * `DEFAULT_MAX_TOOL_TURNS` is infinite and no daemon caller pins `maxToolTurns`,
 * so a denying policy paired with a model that keeps re-asking spins
 * deny -> retry -> deny for as long as the provider answers. This is the only
 * bound on that shape. It is deliberately high: a legitimate session can be
 * denied a handful of times while the model searches for a permitted route.
 */
export const DEFAULT_MAX_CONSECUTIVE_DENIALS = 25

/**
 * Consecutive static `policy_denied` refusals on the exact same tool tolerated
 * before exhausting early. Unlike user permission rejections or cross-tool
 * route search, a static policy denial depends only on tool identity, so
 * repeating the identical blocked tool cannot succeed.
 */
export const DEFAULT_MAX_SAME_TOOL_POLICY_DENIALS = 5

/** Why a tool call never ran. Kept separate from tool failures, which are real work. */
export type DenialKind = 'cancelled' | 'permission_rejected' | 'policy_denied'

export interface DenialRecord {
  readonly kind: DenialKind
  readonly toolName: string
}

export interface DenialBudgetOptions {
  readonly maxSameToolPolicyDenials?: number | null
}

/**
 * Consecutive-refusal counter for one agent turn.
 *
 * One instance per `runTurn`, which makes it per-subagent for free: a child
 * running under a stricter policy exhausts its own budget without touching the
 * parent's. Mutations are synchronous between awaits, so no locking is needed.
 */
export class DenialBudget {
  readonly maxDenials: number | undefined
  readonly maxSameToolPolicyDenials: number | undefined
  private consecutiveDenials = 0
  private consecutiveSameToolPolicyDenials = 0
  private lastDenialRecord: DenialRecord | undefined

  /**
   * `undefined` means "no opinion" and takes the default; an explicit `null` or
   * a non-positive number is an operator opting out. Defaulting to unbounded
   * would leave the loop exactly as unprotected as it was before this class
   * existed.
   */
  constructor(maxDenials?: number | null, options: DenialBudgetOptions = {}) {
    this.maxDenials = maxDenials === undefined
      ? DEFAULT_MAX_CONSECUTIVE_DENIALS
      : normalizeMaximum(maxDenials)
    if (this.maxDenials === undefined) {
      this.maxSameToolPolicyDenials = undefined
    } else if (options.maxSameToolPolicyDenials === undefined) {
      this.maxSameToolPolicyDenials = Math.min(this.maxDenials, DEFAULT_MAX_SAME_TOOL_POLICY_DENIALS)
    } else {
      const normalized = normalizeMaximum(options.maxSameToolPolicyDenials)
      this.maxSameToolPolicyDenials = normalized === undefined ? undefined : Math.min(this.maxDenials, normalized)
    }
  }

  get used(): number {
    return this.consecutiveDenials
  }

  get sameToolPolicyStreak(): number {
    return this.consecutiveSameToolPolicyDenials
  }

  get remaining(): number | undefined {
    if (this.maxDenials === undefined) return undefined
    if (this.exhausted) return 0
    return Math.max(0, this.maxDenials - this.consecutiveDenials)
  }

  get exhausted(): boolean {
    return (this.maxDenials !== undefined && this.consecutiveDenials >= this.maxDenials)
      || (this.maxSameToolPolicyDenials !== undefined && this.consecutiveSameToolPolicyDenials >= this.maxSameToolPolicyDenials)
  }

  get lastDenial(): DenialRecord | undefined {
    return this.lastDenialRecord
  }

  /** Charge one refusal and return the new consecutive total. */
  record(kind: DenialKind, toolName: string): number {
    if (kind === 'policy_denied') {
      this.consecutiveSameToolPolicyDenials =
        this.lastDenialRecord?.kind === 'policy_denied' && this.lastDenialRecord.toolName === toolName
          ? this.consecutiveSameToolPolicyDenials + 1
          : 1
    } else {
      this.consecutiveSameToolPolicyDenials = 0
    }
    this.consecutiveDenials += 1
    this.lastDenialRecord = { kind, toolName }
    return this.consecutiveDenials
  }

  /**
   * Clear the streak. Called when a tool actually ran: the model found a
   * permitted route, so the refusals before it were search, not a loop.
   */
  reset(): void {
    this.consecutiveDenials = 0
    this.consecutiveSameToolPolicyDenials = 0
    this.lastDenialRecord = undefined
  }
}

/**
 * Terminal wording for an exhausted denial budget.
 *
 * Deliberately not phrased as a request for approval. `permissionDisposition`
 * documents that a policy denial is final, and the daemon frequently runs with
 * no interaction board attached, so turning a denial into a prompt would either
 * hang or quietly re-ask a question nobody can answer. The turn stops and says
 * which rule refused what. Rendering lives in the shared intervention catalog
 * so every loop guard stays byte-consistent.
 */
export function denialBudgetStopText(budget: DenialBudget): string {
  return renderIntervention({
    kind: 'denial-budget',
    ...(budget.lastDenial === undefined ? {} : { lastDenial: budget.lastDenial }),
    used: budget.used,
  })
}

/** Stable audit pattern label for an exhausted denial budget. */
export const DENIAL_LOOP_PATTERN = 'tool_denial_loop'

export interface DenialBudgetConfigOptions {
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly envVar?: string
  readonly key?: string
  readonly sameToolEnvVar?: string
  readonly sameToolKey?: string
}

function isUnsetBudgetValue(value: unknown): boolean {
  return value === undefined || value === null || (typeof value === 'string' && value.trim() === '')
}

/** Build a budget from config first, then an injectable environment fallback. */
export function denialBudgetFromConfig(
  config: Readonly<Record<string, unknown>>,
  options: DenialBudgetConfigOptions = {},
): DenialBudget {
  const key = options.key ?? 'max_consecutive_denials'
  const envVar = options.envVar ?? 'XERXES_MAX_CONSECUTIVE_DENIALS'
  const sameToolKey = options.sameToolKey ?? 'max_same_tool_policy_denials'
  const sameToolEnvVar = options.sameToolEnvVar ?? 'XERXES_MAX_SAME_TOOL_POLICY_DENIALS'
  const environment = options.environment ?? process.env
  const configured = config[key]
  const raw = isUnsetBudgetValue(configured) ? environment[envVar] : configured
  const sameToolConfigured = config[sameToolKey]
  const sameToolRaw = isUnsetBudgetValue(sameToolConfigured) ? environment[sameToolEnvVar] : sameToolConfigured
  const budgetOptions: DenialBudgetOptions = isUnsetBudgetValue(sameToolRaw)
    ? {}
    : { maxSameToolPolicyDenials: parseMaximum(sameToolRaw) }
  if (isUnsetBudgetValue(raw)) return new DenialBudget(undefined, budgetOptions)
  return new DenialBudget(parseMaximum(raw), budgetOptions)
}

/** Non-positive is an explicit opt-out; anything else must be a real integer. */
function normalizeMaximum(value: number | null): number | undefined {
  if (value === null || value <= 0) return undefined
  if (!Number.isSafeInteger(value)) throw new RangeError('maxDenials must be a safe integer')
  return value
}

/**
 * A present-but-invalid ceiling is a configuration error, not a licence to run
 * unbounded: silently reading a typo as "no limit" disables the exact guard the
 * operator asked for.
 */
function parseMaximum(value: unknown): number {
  const parsed = typeof value === 'number'
    ? value
    : typeof value === 'string' && /^[-+]?\d+$/.test(value.trim())
      ? Number(value)
      : Number.NaN
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new RangeError(
      'denial budget maximum must be a positive safe integer; got ' + JSON.stringify(value),
    )
  }
  return parsed
}
