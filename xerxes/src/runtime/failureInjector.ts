// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export type FailureMode = 'error' | 'latency' | 'hang' | 'ok'

export interface FailureRule {
  readonly mode: FailureMode
  /** Probability between 0 and 1. */
  readonly probability: number
  readonly latencyMs?: number
  readonly errorMessage?: string
  readonly match?: (operation: string) => boolean
}

export interface FailureInjectorOptions {
  readonly rules?: Readonly<FailureRule[]>
  readonly random?: () => number
}

/** Deterministic failure injector for resiliency tests. */
export class FailureInjector {
  private readonly rules: readonly FailureRule[]
  private readonly random: () => number

  constructor(options: FailureInjectorOptions = {}) {
    this.rules = options.rules ?? []
    this.random = options.random ?? (() => Math.random())
  }

  async inject<T>(operation: string, fn: () => T | Promise<T>): Promise<T> {
    const rule = this.matchingRule(operation)
    if (rule !== undefined) {
      const roll = this.random()
      if (roll < rule.probability) {
        if (rule.mode === 'latency' && rule.latencyMs !== undefined) {
          await sleep(rule.latencyMs)
        } else if (rule.mode === 'hang') {
          await sleep(2_147_483_647)
        } else if (rule.mode === 'error') {
          throw new Error(rule.errorMessage ?? `injected failure for ${operation}`)
        }
      }
    }
    return fn()
  }

  private matchingRule(operation: string): FailureRule | undefined {
    return this.rules.find(rule => (rule.match ?? (() => true))(operation))
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}
