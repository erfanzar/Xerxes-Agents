// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { OperatorPlanStateSnapshot, OperatorPlanStep } from './types.js'

export interface PlanStateManagerOptions {
  readonly now?: () => Date
}

/** Session-scoped structured plan with monotonic revisions. */
export class PlanStateManager {
  private explanation: string | undefined
  private revision = 0
  private steps: OperatorPlanStep[] = []
  private updatedAt: string
  private readonly now: () => Date

  constructor(options: PlanStateManagerOptions = {}) {
    this.now = options.now ?? (() => new Date())
    this.updatedAt = this.now().toISOString()
  }

  get state(): OperatorPlanStateSnapshot {
    return this.snapshot()
  }

  update(explanation: string | undefined, plan: readonly OperatorPlanStep[]): OperatorPlanStateSnapshot {
    this.explanation = normalizeExplanation(explanation)
    this.steps = plan.map(step => Object.freeze({
      step: requireStep(step.step),
      status: normalizeStatus(step.status),
    }))
    this.revision += 1
    this.updatedAt = this.now().toISOString()
    return this.snapshot()
  }

  summary(): string {
    if (!this.steps.length) return 'No plan'
    return this.steps.slice(0, 3).map(step => `${step.status}:${step.step}`).join(', ')
  }

  private snapshot(): OperatorPlanStateSnapshot {
    return Object.freeze({
      ...(this.explanation === undefined ? {} : { explanation: this.explanation }),
      revision: this.revision,
      updatedAt: this.updatedAt,
      steps: Object.freeze(this.steps.map(step => Object.freeze({ ...step }))),
    })
  }
}

function normalizeExplanation(value: string | undefined): string | undefined {
  if (value === undefined) return undefined
  const trimmed = value.trim()
  return trimmed || undefined
}

function requireStep(value: string): string {
  const trimmed = value.trim()
  if (!trimmed) throw new TypeError('Plan step must not be empty')
  return trimmed
}

function normalizeStatus(value: string): string {
  const trimmed = value.trim()
  return trimmed || 'pending'
}
