// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export interface BenchmarkRunResult {
  readonly durationMs: number
  readonly error?: string
  readonly success: boolean
}

export interface BenchmarkOptions {
  readonly name: string
  readonly iterations: number
  readonly now?: () => number
}

export interface BenchmarkSummary {
  readonly name: string
  readonly iterations: number
  readonly successes: number
  readonly failures: number
  readonly minMs: number
  readonly maxMs: number
  readonly meanMs: number
  readonly p50Ms: number
  readonly p95Ms: number
  readonly p99Ms: number
}

export class Benchmark {
  readonly name: string
  readonly iterations: number
  private readonly now: () => number
  private readonly runs: BenchmarkRunResult[] = []

  constructor(options: BenchmarkOptions) {
    this.name = options.name
    this.iterations = options.iterations
    this.now = options.now ?? (() => performance.now())
  }

  async run(fn: () => unknown | Promise<unknown>): Promise<BenchmarkSummary> {
    for (let i = 0; i < this.iterations; i += 1) {
      const start = this.now()
      let success = false
      let error: string | undefined
      try {
        await fn()
        success = true
      } catch (err) {
        error = err instanceof Error ? err.message : String(err)
      }
      const result: BenchmarkRunResult = error === undefined
        ? { durationMs: this.now() - start, success }
        : { durationMs: this.now() - start, success, error }
      this.runs.push(result)
    }
    return this.summarize()
  }

  summarize(): BenchmarkSummary {
    const durations = this.runs.map(run => run.durationMs)
    const sorted = [...durations].sort((a, b) => a - b)
    const successes = this.runs.filter(run => run.success).length
    const total = this.runs.length
    const sum = sorted.reduce((acc, value) => acc + value, 0)
    return {
      name: this.name,
      iterations: total,
      successes,
      failures: total - successes,
      minMs: sorted[0] ?? 0,
      maxMs: sorted[sorted.length - 1] ?? 0,
      meanMs: total === 0 ? 0 : sum / total,
      p50Ms: percentile(sorted, 0.5),
      p95Ms: percentile(sorted, 0.95),
      p99Ms: percentile(sorted, 0.99),
    }
  }
}

function percentile(sorted: readonly number[], p: number): number {
  if (sorted.length === 0) return 0
  const index = Math.max(0, Math.min(sorted.length - 1, Math.ceil(p * sorted.length) - 1))
  return sorted[index] ?? 0
}
