// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { type FileHandle, mkdir, open, readFile } from 'node:fs/promises'
import { dirname } from 'node:path'

/** Batch results may contain prompts and responses with sensitive content. */
const BATCH_OUTPUT_FILE_MODE = 0o600

export interface BatchRecord {
  readonly id: string
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly prompt: string
}

export interface BatchResult {
  readonly costUsd?: number
  readonly error?: string | null
  readonly finishReason?: string
  readonly id: string
  readonly inputTokens?: number
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly outputTokens?: number
  readonly response: string
  readonly toolCalls?: number
}

export interface BatchSummary {
  readonly failed: number
  readonly skipped: number
  readonly succeeded: number
  readonly total: number
  readonly totalCostUsd: number
  readonly totalInputTokens: number
  readonly totalOutputTokens: number
}

export type BatchRunnerFunction = (record: BatchRecord) => BatchResult | Promise<BatchResult>

export interface BatchRunOptions {
  readonly dedupBy?: 'content' | 'id'
  readonly outPath?: string
  readonly resumeIds?: ReadonlySet<string>
}

/** Concurrent, JSONL-resumable offline agent-runner for eval and distillation data. */
export class BatchRunner {
  constructor(private readonly runner: BatchRunnerFunction, readonly workers = 4) {
    if (!Number.isInteger(workers) || workers < 1) throw new Error('workers must be a positive integer')
  }

  /**
   * Stream records through a bounded worker pool.
   *
   * Input is consumed lazily and each result is appended (owner-only file, fsync
   * per record) as soon as it finishes, so memory stays bounded and an
   * interruption cannot discard already-finished work.
   */
  async run(records: Iterable<BatchRecord> | AsyncIterable<BatchRecord>, options: BatchRunOptions = {}): Promise<BatchSummary> {
    const seen = new Set(options.resumeIds ?? [])
    let total = 0
    let skipped = 0
    let failed = 0
    let succeeded = 0
    let totalInputTokens = 0
    let totalOutputTokens = 0
    let totalCostUsd = 0
    let handle: FileHandle | undefined
    if (options.outPath) {
      await mkdir(dirname(options.outPath), { recursive: true })
      handle = await open(options.outPath, 'a', BATCH_OUTPUT_FILE_MODE)
    }
    let writeChain: Promise<unknown> = Promise.resolve()
    const appendResult = (line: string): void => {
      if (handle === undefined) return
      const target = handle
      writeChain = writeChain.then(async () => {
        await target.write(line)
        await target.sync()
      })
    }
    try {
      const iterator = toAsyncIterator(records)
      let pullChain: Promise<unknown> = Promise.resolve()
      const pull = (): Promise<IteratorResult<BatchRecord>> => {
        const result = pullChain.then(() => iterator.next())
        pullChain = result.catch(() => undefined)
        return result
      }
      const worker = async (): Promise<void> => {
        while (true) {
          const next = await pull()
          if (next.done === true) return
          const record = next.value
          total += 1
          const key = options.dedupBy === 'content' ? contentHash(record) : record.id
          if (seen.has(key)) {
            skipped += 1
            continue
          }
          seen.add(key)
          const result = await this.safeRun(record)
          if (result.error) {
            failed += 1
          } else {
            succeeded += 1
            totalInputTokens += result.inputTokens ?? 0
            totalOutputTokens += result.outputTokens ?? 0
            totalCostUsd += result.costUsd ?? 0
          }
          appendResult(`${JSON.stringify(resultToRecord(result))}\n`)
        }
      }
      await Promise.all(Array.from({ length: this.workers }, () => worker()))
    } finally {
      await writeChain
      await handle?.close()
    }
    return { total, skipped, failed, succeeded, totalInputTokens, totalOutputTokens, totalCostUsd }
  }

  private async safeRun(record: BatchRecord): Promise<BatchResult> {
    try {
      return await this.runner(record)
    } catch (error) {
      return { id: record.id, response: '', error: error instanceof Error ? error.message : String(error), finishReason: 'error' }
    }
  }
}

export function contentHash(record: BatchRecord): string {
  const metadata = JSON.stringify(record.metadata ?? {}, Object.keys(record.metadata ?? {}).sort())
  return createHash('sha256').update(`${record.prompt}|${metadata}`).digest('hex')
}

export async function loadCompletedIds(path: string, dedupField = 'id'): Promise<Set<string>> {
  try {
    const content = await readFile(path, 'utf8')
    const ids = content.split(/\r?\n/).flatMap(line => {
      try {
        const value = JSON.parse(line) as Record<string, unknown>
        const field = value[dedupField]
        return field === undefined ? [] : [String(field)]
      } catch {
        return []
      }
    })
    return new Set(ids)
  } catch {
    return new Set()
  }
}

export async function readJsonl(path: string): Promise<BatchRecord[]> {
  try {
    const content = await readFile(path, 'utf8')
    return content.split(/\r?\n/).flatMap(line => {
      try {
        const parsed = JSON.parse(line) as Record<string, unknown>
        return typeof parsed.prompt === 'string'
          ? [{ id: typeof parsed.id === 'string' ? parsed.id : 'auto', prompt: parsed.prompt, ...(isRecord(parsed.metadata) ? { metadata: parsed.metadata } : {}) }]
          : []
      } catch {
        return []
      }
    })
  } catch {
    return []
  }
}

export function resultToRecord(result: BatchResult): Record<string, unknown> {
  return {
    id: result.id,
    response: result.response,
    tool_calls: result.toolCalls ?? 0,
    input_tokens: result.inputTokens ?? 0,
    output_tokens: result.outputTokens ?? 0,
    cost_usd: result.costUsd ?? 0,
    finish_reason: result.finishReason ?? 'stop',
    error: result.error ?? null,
    metadata: { ...(result.metadata ?? {}) },
  }
}

function toAsyncIterator<T>(source: Iterable<T> | AsyncIterable<T>): AsyncIterator<T> {
  if (Symbol.asyncIterator in source) {
    return (source as AsyncIterable<T>)[Symbol.asyncIterator]()
  }
  const iterator = (source as Iterable<T>)[Symbol.iterator]()
  return { next: () => Promise.resolve(iterator.next()) }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
