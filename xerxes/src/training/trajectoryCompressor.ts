// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { type FileHandle, mkdir, open, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'

import { ContextCompressor, type ContextMessage } from '../context/compressor.js'
import { redactPayload } from '../security/redact.js'

/** Output trajectories contain full message/tool content; keep them owner-only. */
const TRAJECTORY_FILE_MODE = 0o600

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

  /**
   * Stream trajectories through a bounded worker pool.
   *
   * Input is consumed lazily and each compressed trajectory is appended (owner-only
   * file, fsync per record) as soon as it finishes, so memory stays bounded and a
   * late failure cannot discard already-processed work. Messages are redacted
   * before they reach the training corpus.
   */
  async run(trajectories: Iterable<Trajectory> | AsyncIterable<Trajectory>, options: TrajectoryRunOptions = {}): Promise<CompressionRun> {
    const done = new Set(options.alreadyDone ?? [])
    const metrics: TrajectoryMetrics[] = []
    const errors: Array<{ trajectoryId: string; error: string }> = []
    let skipped = 0
    let handle: FileHandle | undefined
    if (options.outPath) {
      await mkdir(dirname(options.outPath), { recursive: true })
      handle = await open(options.outPath, 'a', TRAJECTORY_FILE_MODE)
    }
    let writeChain: Promise<unknown> = Promise.resolve()
    const appendOutput = (line: string): void => {
      if (handle === undefined) return
      const target = handle
      writeChain = writeChain.then(async () => {
        await target.write(line)
        await target.sync()
      })
    }
    try {
      const iterator = toAsyncIterator(trajectories)
      let pullChain: Promise<unknown> = Promise.resolve()
      const pull = (): Promise<IteratorResult<Trajectory>> => {
        const result = pullChain.then(() => iterator.next())
        pullChain = result.catch(() => undefined)
        return result
      }
      const worker = async (): Promise<void> => {
        while (true) {
          const next = await pull()
          if (next.done === true) return
          const trajectory = next.value
          const id = trajectory.id ?? trajectoryHash(trajectory)
          if (done.has(id)) {
            skipped += 1
            continue
          }
          try {
            const value = this.compressOne(trajectory)
            metrics.push(value.metrics)
            appendOutput(`${JSON.stringify({ ...value.trajectory, messages: redactPayload(value.trajectory.messages) })}\n`)
          } catch (error) {
            errors.push({ trajectoryId: id, error: error instanceof Error ? error.message : String(error) })
          }
        }
      }
      await Promise.all(Array.from({ length: Math.max(1, this.workers) }, () => worker()))
    } finally {
      await writeChain
      await handle?.close()
    }
    if (options.metricsPath) {
      await mkdir(dirname(options.metricsPath), { recursive: true })
      await writeFile(options.metricsPath, `${JSON.stringify(metrics, null, 2)}\n`, 'utf8')
    }
    return { processed: metrics.length, skipped, metrics, errors }
  }
}

export function trajectoryHash(trajectory: Pick<Trajectory, 'messages'>): string {
  return createHash('sha256').update(JSON.stringify(trajectory.messages)).digest('hex')
}

function toAsyncIterator<T>(source: Iterable<T> | AsyncIterable<T>): AsyncIterator<T> {
  if (Symbol.asyncIterator in source) {
    return (source as AsyncIterable<T>)[Symbol.asyncIterator]()
  }
  const iterator = (source as Iterable<T>)[Symbol.iterator]()
  return { next: () => Promise.resolve(iterator.next()) }
}
