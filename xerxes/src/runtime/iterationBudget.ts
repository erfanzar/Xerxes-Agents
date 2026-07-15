// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Raised when a bounded iteration budget cannot charge the requested amount. */
export class BudgetExhausted extends Error {
  constructor(used: number, maximum: number, requested: number) {
    super('Iteration budget exhausted (used=' + used + ', max=' + maximum + ', asked=' + requested + ')')
    this.name = 'BudgetExhausted'
  }
}

/**
 * Optional iteration counter for agent loops.
 *
 * A missing, zero, or negative maximum means unbounded. JavaScript mutations
 * are synchronous between awaits, so consume/refund transitions remain atomic
 * without exposing the mutable counter.
 */
export class IterationBudget {
  readonly maxIterations: number | undefined
  private usedIterations = 0

  constructor(maxIterations?: number | null) {
    this.maxIterations = normalizeMaximum(maxIterations)
  }

  get used(): number {
    return this.usedIterations
  }

  get remaining(): number | undefined {
    return this.maxIterations === undefined ? undefined : Math.max(0, this.maxIterations - this.usedIterations)
  }

  get exhausted(): boolean {
    return this.maxIterations !== undefined && this.usedIterations >= this.maxIterations
  }

  /** Charge one or more iterations and return the new charged total. */
  consume(count = 1): number {
    const requested = positiveInteger(count, 'count')
    if (this.maxIterations !== undefined && this.usedIterations + requested > this.maxIterations) {
      throw new BudgetExhausted(this.usedIterations, this.maxIterations, requested)
    }
    this.usedIterations += requested
    return this.usedIterations
  }

  /** Return false rather than throwing when a bounded budget is exhausted. */
  tryConsume(count = 1): boolean {
    try {
      this.consume(count)
      return true
    } catch (error) {
      if (error instanceof BudgetExhausted) return false
      throw error
    }
  }

  /** Refund one or more charges while clamping the used count at zero. */
  refund(count = 1): number {
    const requested = positiveInteger(count, 'count')
    this.usedIterations = Math.max(0, this.usedIterations - requested)
    return this.usedIterations
  }

  reset(): void {
    this.usedIterations = 0
  }
}

export interface IterationBudgetConfigOptions {
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly envVar?: string
  readonly key?: string
}

/** Build a budget from config first, then an injectable environment fallback. */
export function iterationBudgetFromConfig(
  config: Readonly<Record<string, unknown>>,
  options: IterationBudgetConfigOptions = {},
): IterationBudget {
  const key = options.key ?? 'max_tool_turns'
  const envVar = options.envVar ?? 'XERXES_MAX_TOOL_TURNS'
  const environment = options.environment ?? process.env
  const configured = config[key]
  const raw = configured === undefined || configured === null || configured === ''
    ? environment[envVar]
    : configured
  const parsed = parsePositiveMaximum(raw)
  return new IterationBudget(parsed)
}

function normalizeMaximum(value: number | null | undefined): number | undefined {
  if (value === undefined || value === null || value <= 0) return undefined
  if (!Number.isSafeInteger(value)) throw new RangeError('maxIterations must be a safe integer')
  return value
}

function parsePositiveMaximum(value: unknown): number | undefined {
  if (value === undefined || value === null || value === '') return undefined
  const parsed = typeof value === 'number'
    ? value
    : typeof value === 'string' && /^[-+]?\d+$/.test(value.trim())
      ? Number(value)
      : Number.NaN
  if (!Number.isSafeInteger(parsed) || parsed <= 0) return undefined
  return parsed
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) throw new RangeError(name + ' must be a positive safe integer')
  return value
}
