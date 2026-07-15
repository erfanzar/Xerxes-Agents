// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { appendFile, mkdir, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'

import { ContextCompressor, type ContextMessage } from '../context/compressor.js'

export interface Trajectory {
  readonly id?: string
  readonly messages: readonly ContextMessage[]
  readonly [key: string]: unknown
}

export interface TrajectoryMetrics {
  readonly compressedCount: number
  readonly prunedToolResults: number
  readonly ratio: number
  readonly strategy: string
  readonly tokensAfter: number
  readonly tokensBefore: number
  readonly trajectoryId: string
}

export interface CompressionRun {
  readonly errors: readonly { readonly error: string; readonly trajectoryId: string }[]
  readonly metrics: readonly TrajectoryMetrics[]
  readonly processed: number
  readonly skipped: number
}

export interface TrajectoryCompressorOptions {
  readonly compressor?: ContextCompressor
  readonly workers?: number
}

export interface TrajectoryRunOptions {
  readonly alreadyDone?: ReadonlySet<string>
  readonly metricsPath?: string
  readonly outPath?: string
}

/** Uses the shared context compactor to make model trajectories replay-budget friendly. */
export class TrajectoryCompressor {
  private readonly compressor: ContextCompressor
  private readonly workers: number

  constructor(options: TrajectoryCompressorOptions = {}) {
    this.compressor = options.compressor ?? new ContextCompressor({ threshold: 0.5, contextWindow: 200_000 })
    this.workers = options.workers ?? 4
  }

  compressOne(trajectory: Trajectory): { readonly metrics: TrajectoryMetrics; readonly trajectory: Record<string, unknown> } {
    const result = this.compressor.compress(trajectory.messages)
    const trajectoryId = trajectory.id ?? trajectoryHash(trajectory)
    return {
      trajectory: {
        ...trajectory,
        messages: result.messages,
        compression: {
          strategy: String(result.metadata.strategy ?? 'unknown'),
          tokens_before: result.tokensBefore,
          tokens_after: result.tokensAfter,
          summary_tokens: result.summaryTokens,
        },
      },
      metrics: {
        trajectoryId,
        tokensBefore: result.tokensBefore,
        tokensAfter: result.tokensAfter,
        ratio: result.tokensBefore === 0 ? 0 : result.tokensAfter / result.tokensBefore,
        compressedCount: result.compressedCount,
        prunedToolResults: result.prunedToolResults,
        strategy: String(result.metadata.strategy ?? 'unknown'),
      },
    }
  }

  async run(trajectories: Iterable<Trajectory> | AsyncIterable<Trajectory>, options: TrajectoryRunOptions = {}): Promise<CompressionRun> {
    const done = new Set(options.alreadyDone ?? [])
    const pending: Array<{ readonly id: string; readonly trajectory: Trajectory }> = []
    let skipped = 0
    for await (const trajectory of trajectories) {
      const id = trajectory.id ?? trajectoryHash(trajectory)
      if (done.has(id)) {
        skipped += 1
        continue
      }
      pending.push({ id, trajectory })
    }
    const results = await mapConcurrent(pending, this.workers, async item => {
      try {
        return { id: item.id, value: this.compressOne(item.trajectory) }
      } catch (error) {
        return { id: item.id, error: error instanceof Error ? error.message : String(error) }
      }
    })
    const metrics: TrajectoryMetrics[] = []
    const outputs: Record<string, unknown>[] = []
    const errors: Array<{ trajectoryId: string; error: string }> = []
    for (const result of results) {
      if ('error' in result) errors.push({ trajectoryId: result.id, error: result.error })
      else {
        metrics.push(result.value.metrics)
        outputs.push(result.value.trajectory)
      }
    }
    if (options.outPath && outputs.length) await appendFile(options.outPath, outputs.map(output => `${JSON.stringify(output)}\n`).join(''), 'utf8')
    if (options.metricsPath) {
      await mkdir(dirname(options.metricsPath), { recursive: true })
      await writeFile(options.metricsPath, `${JSON.stringify(metrics, null, 2)}\n`, 'utf8')
    }
    return { processed: metrics.length, skipped, metrics, errors }
  }
}

export function trajectoryHash(trajectory: Pick<Trajectory, 'messages'>): string {
  return createHash('sha1').update(JSON.stringify(trajectory.messages)).digest('hex').slice(0, 16)
}

async function mapConcurrent<T, R>(items: readonly T[], workers: number, work: (item: T) => Promise<R>): Promise<R[]> {
  const results = Array<R>(items.length)
  let next = 0
  await Promise.all(Array.from({ length: Math.min(Math.max(1, workers), items.length) }, async () => {
    while (true) {
      const index = next
      next += 1
      const item = items[index]
      if (item === undefined) return
      results[index] = await work(item)
    }
  }))
  return results
}
