// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { CronJob, JobStore, nextFireAt } from './jobs.js'

export type JobRunner = (job: CronJob) => string | Promise<string>
export type JobCompletion = (
  job: CronJob,
  output: string,
) => void | Promise<void>

export interface CronSchedulerOptions {
  readonly onComplete?: JobCompletion
  readonly pollInterval?: number
  /** Per-job execution timeout in milliseconds; 0 disables it. Defaults to 5 minutes. */
  readonly jobTimeout?: number
  /** Maximum automatic retries for a failed one-shot job before it is paused. */
  readonly maxOneShotRetries?: number
  /** Base delay in milliseconds for one-shot retry backoff (doubles per attempt). */
  readonly oneShotRetryBaseMs?: number
}

const DEFAULT_JOB_TIMEOUT_MS = 5 * 60_000
const DEFAULT_MAX_ONESHOT_RETRIES = 3
const DEFAULT_ONESHOT_RETRY_BASE_MS = 60_000

/** Polling Bun scheduler with deterministic `tick` support for tests and daemon control. */
export class CronScheduler {
  private interval: ReturnType<typeof setInterval> | undefined
  private readonly jobTimeout: number
  private readonly maxOneShotRetries: number
  private readonly onComplete: JobCompletion | undefined
  private readonly oneShotRetryBaseMs: number
  private readonly pollInterval: number
  private ticking = false

  constructor(
    private readonly store: JobStore,
    private readonly runJob: JobRunner,
    options: CronSchedulerOptions = {},
  ) {
    this.onComplete = options.onComplete
    this.pollInterval = options.pollInterval ?? 30_000
    this.jobTimeout = options.jobTimeout ?? DEFAULT_JOB_TIMEOUT_MS
    this.maxOneShotRetries =
      options.maxOneShotRetries ?? DEFAULT_MAX_ONESHOT_RETRIES
    this.oneShotRetryBaseMs =
      options.oneShotRetryBaseMs ?? DEFAULT_ONESHOT_RETRY_BASE_MS
  }

  start(): void {
    if (this.interval) return
    this.runScheduledTick()
    this.interval = setInterval(
      () => this.runScheduledTick(),
      this.pollInterval,
    )
  }

  stop(): void {
    if (!this.interval) return
    clearInterval(this.interval)
    this.interval = undefined
  }

  async tick(now = new Date()): Promise<string[]> {
    if (this.ticking) return []
    this.ticking = true
    try {
      const current = new Date(now)
      current.setUTCMilliseconds(0)
      const due = this.store
        .listJobs()
        .filter((job) => !job.paused && this.isDue(job, current))
      // Due jobs run concurrently with a per-job timeout so one hung or failing
      // job can neither block the queue nor starve later jobs.
      const outcomes = await Promise.all(
        due.map((job) => this.runDue(job, current)),
      )
      return outcomes.flatMap((id) => (id ? [id] : []))
    } finally {
      this.ticking = false
    }
  }

  private async runDue(job: CronJob, now: Date): Promise<string | undefined> {
    let output: string
    try {
      const result = this.runJob(job)
      output =
        typeof result === 'string'
          ? result
          : await this.withTimeout(result, job.id)
    } catch (error) {
      // One failing job must not reject the tick or starve later jobs.
      this.reportError(`job ${job.id} failed`, error)
      this.handleFailure(job, now, error)
      return undefined
    }
    if (this.onComplete) {
      try {
        await this.onComplete(job, output)
      } catch (error) {
        this.reportError(`onComplete failed for job ${job.id}`, error)
      }
    }
    this.handleSuccess(job, now)
    return job.id
  }

  private async withTimeout(
    result: Promise<string>,
    jobId: string,
  ): Promise<string> {
    const timeout = this.jobTimeout
    if (!Number.isFinite(timeout) || timeout <= 0) return await result
    let timer: ReturnType<typeof setTimeout> | undefined
    try {
      return await Promise.race([
        result,
        new Promise<never>((_resolve, reject) => {
          timer = setTimeout(() => {
            reject(new Error(`job ${jobId} timed out after ${timeout}ms`))
          }, timeout)
          timer.unref?.()
        }),
      ])
    } finally {
      if (timer) clearTimeout(timer)
    }
  }

  private handleSuccess(job: CronJob, now: Date): void {
    // One-shot jobs are removed only after a successful run.
    if (job.oneshot) {
      this.store.remove(job.id)
      return
    }
    this.scheduleNext(job, now)
  }

  private handleFailure(job: CronJob, now: Date, error: unknown): void {
    const message = error instanceof Error ? error.message : String(error)
    const metadata = {
      ...job.metadata,
      last_error: message,
      last_error_at: now.toISOString(),
    }
    if (!job.oneshot) {
      // Recurring jobs keep their cadence; the next fire time is the retry.
      this.scheduleNext(job, now, false, metadata)
      return
    }
    const attempts = retryCount(job.metadata) + 1
    if (attempts > this.maxOneShotRetries) {
      // Retries exhausted: keep the job, record the failure, and pause it so an
      // operator can inspect and resume it instead of losing it silently.
      this.store.update(job.id, {
        paused: true,
        metadata: { ...metadata, retry_count: attempts },
      })
      return
    }
    const delay = this.oneShotRetryBaseMs * 2 ** (attempts - 1)
    this.store.update(job.id, {
      nextRunAt: new Date(now.getTime() + delay).toISOString(),
      metadata: { ...metadata, retry_count: attempts },
    })
  }

  private isDue(job: CronJob, now: Date): boolean {
    if (!job.oneshot && !job.schedule) {
      // A recurring job without a schedule can never fire; pause it (once) so a
      // stale past fire time does not make it due on every poll.
      this.scheduleNext(job, now, true)
      return false
    }
    if (!job.nextRunAt) {
      this.scheduleNext(job, now, true)
      return false
    }
    const next = new Date(job.nextRunAt)
    return !Number.isNaN(next.valueOf()) && next <= now
  }

  private scheduleNext(
    job: CronJob,
    now: Date,
    justSeen = false,
    metadata?: Record<string, unknown>,
  ): void {
    if (job.oneshot) {
      if (justSeen) this.store.update(job.id, { nextRunAt: now.toISOString() })
      else this.store.remove(job.id)
      return
    }
    if (!job.schedule) {
      // A recurring job without a schedule can never fire; pause it and clear any
      // stale fire time so it does not look due on every poll.
      this.store.update(job.id, {
        nextRunAt: null,
        paused: true,
        ...(metadata ? { metadata } : {}),
      })
      return
    }
    try {
      this.store.update(job.id, {
        nextRunAt: nextFireAt(job.schedule, now).toISOString(),
        lastRunAt: now.toISOString(),
        ...(metadata ? { metadata } : {}),
      })
    } catch (error) {
      // Invalid schedules remain stored and can be repaired by their owner.
      this.reportWarning(`invalid schedule for job ${job.id}`, error)
    }
  }

  private runScheduledTick(): void {
    void this.tick().catch((error) => this.reportError('tick failed', error))
  }

  private reportError(message: string, error: unknown): void {
    console.error(`CronScheduler ${message}`, error)
  }

  private reportWarning(message: string, error: unknown): void {
    console.warn(`CronScheduler ${message}`, error)
  }
}

function retryCount(metadata: Readonly<Record<string, unknown>>): number {
  const value = metadata.retry_count
  return typeof value === 'number' && Number.isFinite(value) && value > 0
    ? Math.floor(value)
    : 0
}
