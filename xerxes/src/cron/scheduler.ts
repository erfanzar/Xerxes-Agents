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
}

/** Polling Bun scheduler with deterministic `tick` support for tests and daemon control. */
export class CronScheduler {
  private interval: ReturnType<typeof setInterval> | undefined
  private readonly onComplete: JobCompletion | undefined
  private readonly pollInterval: number
  private ticking = false

  constructor(
    private readonly store: JobStore,
    private readonly runJob: JobRunner,
    options: CronSchedulerOptions = {},
  ) {
    this.onComplete = options.onComplete
    this.pollInterval = options.pollInterval ?? 30_000
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
      const ran: string[] = []
      for (const job of this.store.listJobs()) {
        if (job.paused || !this.isDue(job, current)) continue
        const output = await this.runJob(job)
        ran.push(job.id)
        if (this.onComplete) {
          try {
            await this.onComplete(job, output)
          } catch (error) {
            this.reportError(`onComplete failed for job ${job.id}`, error)
          }
        }
        this.scheduleNext(job, current)
      }
      return ran
    } finally {
      this.ticking = false
    }
  }

  private isDue(job: CronJob, now: Date): boolean {
    if (!job.nextRunAt) {
      this.scheduleNext(job, now, true)
      return false
    }
    const next = new Date(job.nextRunAt)
    return !Number.isNaN(next.valueOf()) && next <= now
  }

  private scheduleNext(job: CronJob, now: Date, justSeen = false): void {
    if (job.oneshot) {
      if (justSeen) this.store.update(job.id, { nextRunAt: now.toISOString() })
      else this.store.remove(job.id)
      return
    }
    if (!job.schedule) return
    try {
      this.store.update(job.id, {
        nextRunAt: nextFireAt(job.schedule, now).toISOString(),
        lastRunAt: now.toISOString(),
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
