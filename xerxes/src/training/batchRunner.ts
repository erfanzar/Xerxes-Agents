// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { appendFile, readFile } from 'node:fs/promises'

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

  async run(records: Iterable<BatchRecord> | AsyncIterable<BatchRecord>, options: BatchRunOptions = {}): Promise<BatchSummary> {
    const seen = new Set(options.resumeIds ?? [])
    const pending: BatchRecord[] = []
    let total = 0
    let skipped = 0
    for await (const record of records) {
      total += 1
      const key = options.dedupBy === 'content' ? contentHash(record) : record.id
      if (seen.has(key)) {
        skipped += 1
        continue
      }
      seen.add(key)
      pending.push(record)
    }
    const results = await mapConcurrent(pending, this.workers, record => this.safeRun(record))
    if (options.outPath) {
      await appendFile(options.outPath, results.map(result => `${JSON.stringify(resultToRecord(result))}\n`).join(''), 'utf8')
    }
    return results.reduce<BatchSummary>((summary, result) => ({
      total,
      skipped,
      failed: summary.failed + (result.error ? 1 : 0),
      succeeded: summary.succeeded + (result.error ? 0 : 1),
      totalInputTokens: summary.totalInputTokens + (result.error ? 0 : result.inputTokens ?? 0),
      totalOutputTokens: summary.totalOutputTokens + (result.error ? 0 : result.outputTokens ?? 0),
      totalCostUsd: summary.totalCostUsd + (result.error ? 0 : result.costUsd ?? 0),
    }), { total, skipped, failed: 0, succeeded: 0, totalInputTokens: 0, totalOutputTokens: 0, totalCostUsd: 0 })
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
  return createHash('sha1').update(`${record.prompt}|${metadata}`).digest('hex').slice(0, 16)
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

async function mapConcurrent<T, R>(items: readonly T[], workers: number, work: (item: T) => Promise<R>): Promise<R[]> {
  const results = Array<R>(items.length)
  let next = 0
  const worker = async (): Promise<void> => {
    while (true) {
      const index = next
      next += 1
      const item = items[index]
      if (item === undefined) return
      results[index] = await work(item)
    }
  }
  await Promise.all(Array.from({ length: Math.min(workers, items.length) }, worker))
  return results
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
