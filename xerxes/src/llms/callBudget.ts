// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { AsyncLocalStorage } from 'node:async_hooks'

export type ModelCallReceipt = (usage?: { readonly inputTokens: number; readonly outputTokens: number; readonly cacheReadTokens?: number; readonly cacheCreationTokens?: number }, completed?: boolean) => void
export interface ModelCallUsage {
  readonly input_tokens: number
  readonly output_tokens: number
  readonly measured_calls: number
  readonly settled_calls: number
  readonly pending_calls: number
  readonly complete: boolean
}
export interface TokenAdmissionBudget { readonly maximum: number; readonly priorTokens: number; readonly priorComplete: boolean }

/** Serializable ownership needed to restore a detached model-call scope. */
export type ModelCallBinding =
  | { readonly kind: 'goal'; readonly sessionId: string; readonly goalId: string }
  | { readonly kind: 'unrecoverable' }

export interface ModelCallScope {
  readonly scopeKey?: string
  /** Present only when this scope can be reconstructed after a restart. */
  readonly recoveryBinding?: ModelCallBinding
  readonly maximum?: number | undefined
  readonly used: number
  readonly available: boolean
  readonly exhausted: boolean
  readonly persistenceError?: Error | undefined
  readonly tokenFailure?: Error | undefined
  assertAdmission(): void
  charge(): ModelCallReceipt
  /** Freeze dynamic ownership when work is delegated beyond the current turn. */
  fork?(): ModelCallScope | undefined
}

/** A logical provider-call admission limit, not a token or transport-retry cap. */
export class ModelCallBudget {
  private admitted = 0
  private denied = false
  private closed = false
  private settled = 0
  private measured = 0
  private inputTokens = 0
  private outputTokens = 0
  private checkpointFailure: Error | undefined
  private deniedTokens: ModelTokenBudgetError | undefined
  constructor(readonly maximum?: number, private readonly checkpoint?: (usage: ModelCallUsage) => void, private readonly tokens?: TokenAdmissionBudget) {
    if (maximum !== undefined && (!Number.isSafeInteger(maximum) || maximum < 1 || maximum > 10000)) throw new Error('Model call limit must be an integer from 1 to 10000')
    if (tokens && (![tokens.maximum, tokens.priorTokens].every(n => Number.isSafeInteger(n) && n >= 0) || tokens.maximum < 1 || typeof tokens.priorComplete !== 'boolean')) throw new Error('Invalid token admission budget')
  }
  get used(): number { return this.admitted }
  get exhausted(): boolean { return this.denied }
  get persistenceError(): Error | undefined { return this.checkpointFailure }
  get tokenFailure(): ModelTokenBudgetError | undefined { return this.deniedTokens }
  get available(): boolean { return !this.closed && !this.tokenError() && (this.maximum === undefined || this.admitted < this.maximum) }
  private tokenError(): ModelTokenBudgetError | undefined {
    if (!this.tokens) return
    if (!this.tokens.priorComplete || this.measured < this.settled) return new ModelTokenBudgetError('Token usage is incomplete; further calls are blocked by the total budget')
    if (this.tokens.priorTokens + this.inputTokens + this.outputTokens >= this.tokens.maximum) return new ModelTokenBudgetError(`Total token admission budget exhausted (${this.tokens.priorTokens + this.inputTokens + this.outputTokens}/${this.tokens.maximum})`)
  }
  get usage(): ModelCallUsage {
    return { input_tokens: this.inputTokens, output_tokens: this.outputTokens, measured_calls: this.measured, settled_calls: this.settled, pending_calls: this.admitted - this.settled, complete: this.measured === this.admitted }
  }
  assertAdmission(): void {
    if (this.checkpointFailure) throw this.checkpointFailure
    const tokenError = this.tokenError()
    if (tokenError) { this.deniedTokens = tokenError; throw tokenError }
    if (!this.available) {
      this.denied = true
      throw new ModelCallBudgetError(this.admitted, this.maximum, this.closed)
    }
  }
  charge(): ModelCallReceipt {
    this.assertAdmission()
    this.admitted++
    this.persist()
    let recorded = false
    return (usage, completed = true) => {
      if (recorded) return
      recorded = true
      this.settled++
      const counts = usage ? [usage.inputTokens, usage.outputTokens, usage.cacheReadTokens ?? 0, usage.cacheCreationTokens ?? 0] : []
      if (usage && counts.every(n => Number.isSafeInteger(n) && n >= 0)) {
        const input = this.inputTokens + usage.inputTokens + (usage.cacheReadTokens ?? 0) + (usage.cacheCreationTokens ?? 0)
        const output = this.outputTokens + usage.outputTokens
        if (!Number.isSafeInteger(input) || !Number.isSafeInteger(output) || !Number.isSafeInteger(input + output)) {
          if (!this.closed) {
            try { this.persist() } catch { /* checkpointFailure is recorded on this.checkpointFailure */ }
          }
          this.closed = true
          this.deniedTokens = new ModelTokenBudgetError('Model token accounting overflow')
          throw this.deniedTokens
        }
        this.inputTokens = input
        this.outputTokens = output
        if (completed) this.measured++
      }
      if (!this.closed) this.persist()
    }
  }
  private persist(): void {
    try { this.checkpoint?.(this.usage) }
    catch (cause) {
      this.closed = true
      this.checkpointFailure = new ModelUsageCheckpointError(cause)
      throw this.checkpointFailure
    }
  }
  close(): void { this.closed = true }
}

export class ModelUsageCheckpointError extends Error {
  constructor(cause: unknown) { super('Could not persist model usage checkpoint', { cause }); this.name = 'ModelUsageCheckpointError' }
}
export class ModelTokenBudgetError extends Error {
  constructor(message: string) { super(message); this.name = 'ModelTokenBudgetError' }
}

export class ModelCallBudgetError extends Error {
  constructor(readonly used: number, readonly maximum: number | undefined, closed = false) {
    super(closed ? 'Scheduled model-call scope has ended' : `Model call budget exhausted (${used}/${maximum})`)
    this.name = 'ModelCallBudgetError'
  }
}

const activeBudgets = new AsyncLocalStorage<readonly ModelCallScope[]>()

/** Descendants inherit all enclosing limits; adding a scope cannot replace one. */
export function withModelCallBudget<T>(budget: ModelCallScope, work: () => T): T {
  const parents = activeBudgets.getStore() ?? []
  return parents.includes(budget) ? work() : activeBudgets.run([...parents, budget], work)
}

export function captureModelCallScopes(): readonly ModelCallScope[] {
  return (activeBudgets.getStore() ?? []).flatMap(scope => {
    const captured = scope.fork ? scope.fork() : scope
    return captured ? [captured] : []
  })
}

/** Convert captured scopes to a bounded, serializable recovery description. */
export function serializeModelCallScopes(scopes: readonly ModelCallScope[]): readonly ModelCallBinding[] {
  if (scopes.length > 64) return Object.freeze([{ kind: 'unrecoverable' }])
  return Object.freeze(scopes.map(scope => {
    const binding = scope.recoveryBinding
    if (!binding || binding.kind === 'unrecoverable') return { kind: 'unrecoverable' as const }
    if (binding.kind !== 'goal' || typeof binding.sessionId !== 'string' || !binding.sessionId
      || binding.sessionId.trim() !== binding.sessionId || binding.sessionId.length > 256
      || typeof binding.goalId !== 'string' || !binding.goalId
      || binding.goalId.trim() !== binding.goalId || binding.goalId.length > 256) {
      throw new Error('Invalid model-call recovery binding')
    }
    return Object.freeze({ kind: 'goal' as const, sessionId: binding.sessionId, goalId: binding.goalId })
  }))
}

export function withCapturedModelCallScopes<T>(scopes: readonly ModelCallScope[], work: () => T): T {
  const merged = new Map<unknown, ModelCallScope>()
  for (const scope of [...(activeBudgets.getStore() ?? []), ...scopes]) merged.set(scope.scopeKey ?? scope, scope)
  return activeBudgets.run([...merged.values()], work)
}

/** Host-only boundary for a separately admitted scheduled run or reaction. */
export function withIndependentModelCallBudget<T>(budget: ModelCallBudget, work: () => T): T {
  return activeBudgets.run([budget], work)
}

/** Charge every enclosing scope before starting a stream or auxiliary completion. */
export function chargeModelCall(): ModelCallReceipt | undefined {
  const budgets = activeBudgets.getStore()
  if (!budgets?.length) return undefined
  // Preflight all limits before changing counters. No asynchronous work can
  // enter between these checks and charges on the same event loop.
  for (const budget of budgets) budget.assertAdmission()
  const receipts: ModelCallReceipt[] = []
  try { for (const budget of budgets) receipts.push(budget.charge()) }
  catch (error) {
    // No provider call started. Retain conservative admission accounting, but
    // settle earlier scopes rather than stranding their in-flight counters.
    for (const receipt of receipts) {
      try { receipt(undefined, false) } catch { /* Preserve the admission failure; each failed scope records its own persistenceError. */ }
    }
    throw error
  }
  return (usage, completed) => {
    let failure: { error: unknown } | undefined
    for (const receipt of receipts) {
      try { receipt(usage, completed) } catch (error) { failure ??= { error } }
    }
    if (failure) throw failure.error
  }
}

/** Optional work must fit every enclosing scope. */
export function optionalModelCallAvailable(): boolean {
  return activeBudgets.getStore()?.every(budget => budget.available) ?? true
}

export function assertModelCallBudget(): void {
  for (const budget of activeBudgets.getStore() ?? []) {
    if (budget.persistenceError) throw budget.persistenceError
    if (budget.tokenFailure) throw budget.tokenFailure
    if (budget.exhausted) throw new ModelCallBudgetError(budget.used, budget.maximum)
  }
}
