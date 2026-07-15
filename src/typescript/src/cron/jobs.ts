// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'

export interface CronJobOptions {
  readonly deliver?: string
  readonly id: string
  readonly lastRunAt?: string
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly nextRunAt?: string
  readonly oneshot?: boolean
  readonly paused?: boolean
  readonly prompt: string
  readonly recipient?: string
  readonly schedule?: string
  readonly workspaceId?: string
}

/** A persisted recurring or one-shot agent prompt. */
export class CronJob {
  deliver: string
  readonly id: string
  lastRunAt: string | undefined
  metadata: Record<string, unknown>
  nextRunAt: string | undefined
  oneshot: boolean
  paused: boolean
  readonly prompt: string
  recipient: string
  schedule: string
  workspaceId: string | undefined

  constructor(options: CronJobOptions) {
    this.id = options.id
    this.prompt = options.prompt
    this.schedule = options.schedule ?? ''
    this.deliver = options.deliver ?? 'none'
    this.recipient = options.recipient ?? ''
    this.paused = options.paused ?? false
    this.oneshot = options.oneshot ?? false
    this.lastRunAt = options.lastRunAt
    this.nextRunAt = options.nextRunAt
    this.workspaceId = options.workspaceId
    this.metadata = { ...(options.metadata ?? {}) }
  }

  toRecord(): Record<string, unknown> {
    return {
      id: this.id,
      prompt: this.prompt,
      schedule: this.schedule,
      deliver: this.deliver,
      recipient: this.recipient,
      paused: this.paused,
      oneshot: this.oneshot,
      last_run_at: this.lastRunAt ?? null,
      next_run_at: this.nextRunAt ?? null,
      workspace_id: this.workspaceId ?? null,
      metadata: { ...this.metadata },
    }
  }

  static fromRecord(value: Record<string, unknown>): CronJob {
    const id = stringValue(value.id)
    const prompt = stringValue(value.prompt)
    if (!id || !prompt)
      throw new Error('Cron job records require id and prompt')
    return new CronJob({
      id,
      prompt,
      schedule: stringValue(value.schedule),
      deliver: stringValue(value.deliver) || 'none',
      recipient: stringValue(value.recipient),
      paused: value.paused === true,
      oneshot: value.oneshot === true,
      ...(nullableString(value.last_run_at)
        ? { lastRunAt: nullableString(value.last_run_at) as string }
        : {}),
      ...(nullableString(value.next_run_at)
        ? { nextRunAt: nullableString(value.next_run_at) as string }
        : {}),
      ...(nullableString(value.workspace_id)
        ? { workspaceId: nullableString(value.workspace_id) as string }
        : {}),
      ...(isRecord(value.metadata) ? { metadata: value.metadata } : {}),
    })
  }
}

/** Threadless, JSON-backed job persistence suitable for a single Bun daemon process. */
export class JobStore {
  constructor(readonly path: string) {
    mkdirSync(dirname(path), { recursive: true })
    try {
      readFileSync(path, 'utf8')
    } catch {
      writeFileSync(path, '[]\n', 'utf8')
    }
  }

  add(job: CronJob): CronJob {
    const records = this.load().filter((record) => record.id !== job.id)
    records.push(job.toRecord())
    this.save(records)
    return job
  }

  get(jobId: string): CronJob | undefined {
    return this.listJobs().find((job) => job.id === jobId)
  }

  listJobs(): CronJob[] {
    return this.load().flatMap((record) => {
      try {
        return [CronJob.fromRecord(record)]
      } catch {
        return []
      }
    })
  }

  newId(): string {
    return crypto.randomUUID().replaceAll('-', '').slice(0, 12)
  }

  remove(jobId: string): boolean {
    const records = this.load()
    const filtered = records.filter((record) => record.id !== jobId)
    if (filtered.length === records.length) return false
    this.save(filtered)
    return true
  }

  update(
    jobId: string,
    changes: Readonly<Record<string, unknown>>,
  ): CronJob | undefined {
    const records = this.load()
    const position = records.findIndex((record) => record.id === jobId)
    if (position < 0) return undefined
    const existing = records[position]
    if (!existing) return undefined
    const updated = { ...existing, ...recordToSnakeCase(changes) }
    records[position] = updated
    this.save(records)
    return CronJob.fromRecord(updated)
  }

  private load(): Record<string, unknown>[] {
    try {
      const parsed = JSON.parse(readFileSync(this.path, 'utf8')) as unknown
      return Array.isArray(parsed) ? parsed.filter(isRecord) : []
    } catch {
      return []
    }
  }

  private save(records: readonly Record<string, unknown>[]): void {
    writeFileSync(this.path, `${JSON.stringify(records, null, 2)}\n`, 'utf8')
  }
}

/** Calculate the first UTC minute strictly after `now` matching a five-field cron expression. */
export function nextFireAt(schedule: string, now = new Date()): Date {
  const parts = schedule.trim().split(/\s+/)
  if (parts.length !== 5)
    throw new Error(
      `expected 5-field cron expression, got ${JSON.stringify(schedule)}`,
    )
  const minute = parseCronField(parts[0] ?? '', 0, 59)
  const hour = parseCronField(parts[1] ?? '', 0, 23)
  const day = parseCronField(parts[2] ?? '', 1, 31)
  const month = parseCronField(parts[3] ?? '', 1, 12)
  const weekDay = parseCronField(parts[4] ?? '', 0, 6)
  const candidate = new Date(now)
  candidate.setUTCSeconds(0, 0)
  candidate.setUTCMinutes(candidate.getUTCMinutes() + 1)
  for (let count = 0; count < 60 * 24 * 366; count += 1) {
    if (
      minute.has(candidate.getUTCMinutes()) &&
      hour.has(candidate.getUTCHours()) &&
      day.has(candidate.getUTCDate()) &&
      month.has(candidate.getUTCMonth() + 1) &&
      weekDay.has(candidate.getUTCDay())
    )
      return candidate
    candidate.setUTCMinutes(candidate.getUTCMinutes() + 1)
  }
  throw new Error(
    `no fire time found within a year for ${JSON.stringify(schedule)}`,
  )
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function nullableString(value: unknown): string | undefined {
  return typeof value === 'string' && value ? value : undefined
}

function parseCronField(spec: string, low: number, high: number): Set<number> {
  if (!spec) throw new Error('empty cron field')
  const values = new Set<number>()
  for (const part of spec.split(',')) {
    const [rangeSpec, stepSpec] = part.split('/', 2)
    const step = stepSpec === undefined ? 1 : integer(stepSpec)
    if (step <= 0) throw new Error(`invalid cron step ${JSON.stringify(part)}`)
    const [start, end] =
      rangeSpec === '*' ? [low, high] : parseRange(rangeSpec ?? '', low, high)
    for (let value = start; value <= end; value += step) values.add(value)
  }
  return values
}

function parseRange(spec: string, low: number, high: number): [number, number] {
  const segments = spec.split('-', 2)
  const start = integer(segments[0] ?? '')
  const end = segments.length === 1 ? start : integer(segments[1] ?? '')
  if (start < low || end > high || start > end)
    throw new Error(`cron value ${JSON.stringify(spec)} outside ${low}-${high}`)
  return [start, end]
}

function integer(value: string): number {
  if (!/^\d+$/.test(value))
    throw new Error(`invalid cron integer ${JSON.stringify(value)}`)
  return Number(value)
}

function recordToSnakeCase(
  changes: Readonly<Record<string, unknown>>,
): Record<string, unknown> {
  const aliases: Record<string, string> = {
    lastRunAt: 'last_run_at',
    nextRunAt: 'next_run_at',
    workspaceId: 'workspace_id',
  }
  return Object.fromEntries(
    Object.entries(changes).map(([key, value]) => [aliases[key] ?? key, value]),
  )
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value : ''
}
