// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ActivityChanges } from '../runtime/activityChanges.js'
import { CronJob, JobStore, nextFireAt } from './jobs.js'
import { DeliveryError } from './delivery.js'
import { scheduleTokenState } from './tokenUsage.js'

/**
 * Lets a runner keep time it spends queued behind work it does not control
 * (its target conversation's own turn) out of the job timeout. Without it a
 * follow-up in a busy chat timed out, unrun, while still waiting its turn.
 */
export interface JobClock {
  /** Stop the timeout while the run waits for admission. */
  pause(): void
  /** Start a fresh full timeout once the run is admitted and real work begins. */
  resume(): void
}

export type JobRunner = (job: CronJob, signal: AbortSignal, clock: JobClock) => string | Promise<string>

/**
 * The scheduler itself aborted the run: an operator cancel or a scheduler
 * stop. Neither is a failure of the job, so neither earns an automatic retry
 * that would replay the work the operator just stopped.
 */
export class CronRunInterrupted extends Error {
  constructor(readonly interruption: 'operator' | 'shutdown', message: string) {
    super(message)
    this.name = 'CronRunInterrupted'
  }
}
export type JobCompletion = (
  job: CronJob,
  output: string,
) => void | Promise<void>

export interface CronSchedulerOptions {
  /**
   * Ownership gate consulted at the top of every tick. The job store is shared
   * by every project's daemon, so without it each open project fires the same
   * job as its own agent turn. Defaults to always-true, which is the correct
   * answer for a single process.
   */
  readonly holdsLease?: () => boolean
  readonly onComplete?: JobCompletion
  readonly pollInterval?: number
  /** Maximum unsettled runs, including timed-out runners still cancelling. */
  readonly maxConcurrentJobs?: number
  readonly maxConcurrentJobsPerProject?: number
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
  readonly activityChanges = new ActivityChanges()
  private readonly holdsLease: (() => boolean) | undefined
  private interval: ReturnType<typeof setInterval> | undefined
  private readonly jobTimeout: number
  private readonly maxOneShotRetries: number
  private readonly onComplete: JobCompletion | undefined
  private readonly oneShotRetryBaseMs: number
  private readonly pollInterval: number
  private ticking = false
  private readonly maxConcurrentJobs: number
  private readonly maxConcurrentJobsPerProject: number
  private readonly activeProjects = new Map<string, string>()
  private readonly settling = new Set<Promise<unknown>>()
  // Keep ownership until the actual runner settles, even after its timeout.
  private readonly active = new Map<string, AbortController>()

  constructor(
    private readonly store: JobStore,
    private readonly runJob: JobRunner,
    options: CronSchedulerOptions = {},
  ) {
    this.holdsLease = options.holdsLease
    this.maxConcurrentJobs = positiveInteger(options.maxConcurrentJobs ?? 4, 'maxConcurrentJobs')
    this.maxConcurrentJobsPerProject = positiveInteger(options.maxConcurrentJobsPerProject ?? 4, 'maxConcurrentJobsPerProject')
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
    if (this.interval) clearInterval(this.interval)
    this.interval = undefined
    for (const controller of this.active.values()) {
      controller.abort(new CronRunInterrupted('shutdown', 'scheduler stopped'))
    }
  }

  get activeCount(): number { return this.active.size }

  state(jobId: string): 'idle' | 'running' | 'cancelling' {
    const controller = this.active.get(jobId)
    return controller ? controller.signal.aborted ? 'cancelling' : 'running' : 'idle'
  }

  /** Request cancellation without releasing admission until the runner settles. */
  cancel(jobId: string): boolean {
    const controller = this.active.get(jobId)
    if (!controller) return false
    controller.abort(new CronRunInterrupted('operator', `job ${jobId} cancelled by operator`))
    return true
  }

  /** Wait for real execution cleanup, including runners that ignored timeout. */
  async waitForIdle(): Promise<void> {
    while (this.settling.size) await Promise.allSettled([...this.settling])
  }

  /** Manual and scheduled execution share admission, cancellation and limits. */
  async runNow<T>(job: CronJob, runner: (signal: AbortSignal, clock: JobClock) => Promise<T>): Promise<T> {
    if (!this.owns()) throw new Error('Cron scheduling belongs to another daemon; run this job on its owner')
    if (this.active.has(job.id)) throw new Error(`job ${job.id} is already running or cancelling`)
    if (!this.hasCapacity(job)) throw new Error('Cron concurrency limit reached; retry when a running job finishes')
    return this.runOwned(job, runner)
  }

  private withinExpiry(job: CronJob, now: Date): boolean {
    if (job.expiresAt === undefined || now.getTime() < Date.parse(job.expiresAt)) return true
    this.store.update(job.id, { paused: true, metadata: { ...job.metadata, schedule_expired_at: job.expiresAt } })
    return false
  }

  private withinRunLimit(job: CronJob): boolean {
    if (job.maxRuns === undefined || job.runsStarted < job.maxRuns) return true
    this.store.update(job.id, { paused: true, metadata: { ...job.metadata, execution_limit_reached: true } })
    return false
  }
  private withinTokenBudget(job: CronJob): boolean {
    if (job.maxTotalTokens === undefined) return true
    let reason: string | undefined
    try {
      const total = scheduleTokenState(job.metadata.total_token_usage, job.runsStarted)
      if (!total.complete) reason = 'Historical token usage is incomplete; remove the total budget or create a new follow-up'
      else if (total.used >= job.maxTotalTokens) reason = 'Total token admission budget exhausted; increase or remove it before resuming'
    } catch { reason = 'Cumulative token usage is invalid; repair the schedule accounting before resuming' }
    if (!reason) return true
    this.store.update(job.id, { paused: true, metadata: { ...job.metadata, token_budget_blocker: reason } })
    return false
  }

  private hasCapacity(job: CronJob): boolean {
    const project = job.projectRoot ?? ''
    return this.active.size < this.maxConcurrentJobs &&
      [...this.activeProjects.values()].filter(value => value === project).length < this.maxConcurrentJobsPerProject
  }

  async tick(now = new Date()): Promise<string[]> {
    if (this.ticking) return []
    // Nothing at all, not even the bookkeeping in `isDue`: a process without
    // the lease that advanced `next_run_at` would consume the lease holder's
    // fire time and the job would silently never run.
    if (!this.owns()) return []
    const pending: Promise<string | undefined>[] = []
    this.ticking = true
    try {
      const current = new Date(now)
      current.setUTCMilliseconds(0)
      const due = this.store
        .listJobs()
        .filter((job) => !this.active.has(job.id) && !job.paused && this.withinExpiry(job, now) && this.withinRunLimit(job) && this.withinTokenBudget(job) && this.isDue(job, current))
      // Due jobs run concurrently with a per-job timeout so one hung or failing
      // job can neither block the queue nor starve later jobs.
      for (const job of due) {
        if (this.hasCapacity(job)) pending.push(this.runDue(job, current))
      }
    } finally {
      // The guard covers selection and admission only. `runOwned` records each
      // admitted job in `active` synchronously, which already stops a later
      // tick from firing it twice. Holding the guard until runs finish made
      // one long job silence every poll, so other due jobs ran late and
      // `skip` jobs past their grace were dropped.
      this.ticking = false
    }
    const outcomes = await Promise.all(pending)
    return outcomes.flatMap((id) => (id ? [id] : []))
  }

  /**
   * A predicate that throws (an unreadable lease file, say) is treated as "not
   * ours" so a transient filesystem failure skips one tick instead of letting
   * every daemon fall back to running the shared store.
   */
  private owns(): boolean {
    if (!this.holdsLease) return true
    try {
      return this.holdsLease()
    } catch (error) {
      this.reportWarning('lease check failed; skipping tick', error)
      return false
    }
  }

  private async runDue(job: CronJob, now: Date): Promise<string | undefined> {
    let phase: 'admission' | 'execution' | 'completion' = 'admission'
    let runSignal: AbortSignal | undefined
    return this.runOwned(job, async (signal, clock) => {
      runSignal = signal
      // Commit intent before calling the model. An unfinished receipt after
      // restart requires review: external effects cannot safely be replayed.
      const receipt = { state: 'running', occurrence: job.nextRunAt, started_at: now.toISOString() }
      if (!this.store.update(job.id, { metadata: { ...job.metadata, execution_receipt: receipt } })) throw new Error('Schedule removed before execution')
      phase = 'execution'
      const output = await this.runJob(job, signal, clock)
      phase = 'completion'
      signal.throwIfAborted()
      const persisted = this.store.get(job.id)
      if (!persisted || !this.store.update(job.id, { metadata: { ...persisted.metadata, execution_receipt: { ...receipt, state: 'completed' } } })) throw new Error('Schedule removed during execution')
      if (this.onComplete) {
        try {
          await this.onComplete(job, output)
          const current = this.store.get(job.id)
          if (current?.metadata.delivery_state === 'failed') this.store.update(job.id, {
            metadata: { ...current.metadata, delivery_state: 'delivered', delivery_error: null },
          })
        } catch (error) {
          this.reportError(`onComplete failed for job ${job.id}`, error)
          // Execution already succeeded. Preserve delivery evidence without
          // feeding this failure into the model-execution retry policy.
          const current = this.store.get(job.id)
          if (current) this.store.update(job.id, {
            ...(job.oneshot ? { paused: true } : {}),
            lastRunAt: now.toISOString(),
            metadata: { ...current.metadata, delivery_state: 'failed',
              delivery_error: error instanceof Error ? error.message : String(error),
              delivery_failed_at: now.toISOString(),
              ...(error instanceof DeliveryError ? { delivery_archive: error.archivePath, delivery_id: error.deliveryId ?? null } : {}) },
          })
          if (!job.oneshot) this.scheduleNext(job, now)
          return job.id
        }
      }
      signal.throwIfAborted()
      this.handleSuccess(job, now)
      return job.id
    }, now).catch(error => {
      this.reportError(`job ${job.id} failed`, error)
      // Judge by why the signal was aborted, not by what the runner threw: a
      // cancelled turn surfaces as many different errors.
      const interruption = runSignal?.aborted && runSignal.reason instanceof CronRunInterrupted ? runSignal.reason : undefined
      if (phase === 'execution' && interruption) this.handleInterrupted(job, now, interruption)
      else if (phase === 'execution') this.handleFailure(job, now, error)
      else if (phase === 'completion') {
        // Do not convert bookkeeping/delivery failures into model retries.
        // If this write also fails, the intent receipt still fences next tick.
        try { this.requireReview(job.id, error) }
        catch (failure) { this.reportError(`cannot persist recovery state for ${job.id}`, failure) }
      }
      return undefined
    })
  }

  private requireReview(jobId: string, error: unknown): void {
    const current = this.store.get(jobId)
    if (!current) return
    this.store.update(jobId, { paused: true, metadata: { ...current.metadata,
      execution_recovery_required: true,
      last_error: `Execution requires review before resume; work may already have completed. ${error instanceof Error ? error.message : String(error)}`,
    } })
  }

  private async runOwned<T>(job: CronJob, runner: (signal: AbortSignal, clock: JobClock) => Promise<T>, now = new Date()): Promise<T> {
    const current = this.store.get(job.id)
    if (current) {
      if (current.metadata.followup_completion != null) throw new Error('Follow-up condition was reported met; explicitly resume before running again')
      if (!this.withinTokenBudget(current)) throw new Error('Total token budget prevents execution; inspect the schedule for details')
      if (!this.withinExpiry(current, now)) throw new Error('Schedule expired; edit or remove the expiry before running again')
      if (!this.withinRunLimit(current)) throw new Error('Schedule execution limit reached; edit the limit before running again')
      if (!Number.isSafeInteger(current.runsStarted + 1)) throw new Error('Schedule execution counter exhausted')
      if (!this.store.update(job.id, { runsStarted: current.runsStarted + 1 })) throw new Error('Schedule removed before admission')
    } else if (job.maxRuns !== undefined || job.expiresAt !== undefined) throw new Error('Bounded schedules must be persisted before running')
    const controller = new AbortController()
    this.active.set(job.id, controller)
    controller.signal.addEventListener('abort', () => this.activityChanges.notify(), { once: true })
    this.activityChanges.notify()
    this.activeProjects.set(job.id, job.projectRoot ?? '')
    const deadline = this.deadline(job.id, controller, job.timeoutMs ?? this.jobTimeout)
    const result = Promise.resolve().then(() => {
      controller.signal.throwIfAborted()
      return runner(controller.signal, deadline.clock)
    }).finally(() => {
      if (this.active.get(job.id) === controller) {
        this.active.delete(job.id)
        this.activityChanges.notify()
        this.activeProjects.delete(job.id)
      }
      this.settling.delete(result)
    })
    this.settling.add(result)
    try {
      return await Promise.race([result, deadline.expired])
    } finally {
      deadline.settle()
    }
  }

  /** A per-run timeout the runner may pause while it waits for admission. */
  private deadline(jobId: string, controller: AbortController, timeout: number): {
    readonly clock: JobClock
    readonly expired: Promise<never>
    readonly settle: () => void
  } {
    const enabled = Number.isFinite(timeout) && timeout > 0
    let timer: ReturnType<typeof setTimeout> | undefined
    let settled = false
    let expire!: (error: Error) => void
    const expired = new Promise<never>((_resolve, reject) => { expire = reject })
    const disarm = (): void => {
      if (timer) clearTimeout(timer)
      timer = undefined
    }
    const arm = (): void => {
      disarm()
      if (!enabled || settled || controller.signal.aborted) return
      timer = setTimeout(() => {
        timer = undefined
        const error = new Error(`job ${jobId} timed out after ${timeout}ms`)
        controller.abort(error)
        expire(error)
      }, timeout)
      // NOTE: no unref() here. On Windows, an unref'd timer that is the
      // only pending handle never fires (the event loop sleeps), which
      // would let a hung job wedge the scheduler forever. The timer is
      // always cleared once the race settles, so keeping it referenced
      // costs nothing.
    }
    arm()
    return {
      clock: { pause: disarm, resume: arm },
      expired,
      settle: () => { settled = true; disarm() },
    }
  }

  private handleSuccess(job: CronJob, now: Date): void {
    if (this.store.get(job.id)?.metadata.followup_completion != null) {
      this.store.update(job.id, { paused: true, nextRunAt: null, lastRunAt: now.toISOString() })
      return
    }
    // One-shot jobs are removed only after a successful run.
    if (job.oneshot) {
      this.store.remove(job.id)
      return
    }
    this.scheduleNext(job, now)
  }

  private handleFailure(job: CronJob, now: Date, error: unknown): void {
    const message = error instanceof Error ? error.message : String(error)
    // This attempt returned a known failure. Preserve state written while it
    // ran, but not its running receipt, which would fence a legitimate retry.
    const { execution_receipt: _receipt, ...currentMetadata } = this.store.get(job.id)?.metadata ?? job.metadata
    const metadata: Record<string, unknown> = {
      ...currentMetadata,
      last_error: message,
      last_error_at: now.toISOString(),
    }
    if (metadata.followup_completion != null) {
      this.store.update(job.id, { paused: true, nextRunAt: null, metadata })
      return
    }
    if (!job.oneshot) {
      // Recurring jobs keep their cadence; the next fire time is the retry.
      this.scheduleNext(job, now, false, metadata)
      return
    }
    const attempts = retryCount(job.metadata) + 1
    if (attempts > (job.maxRetries ?? this.maxOneShotRetries)) {
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

  private handleInterrupted(job: CronJob, now: Date, interruption: CronRunInterrupted): void {
    if (interruption.interruption === 'shutdown') {
      // A one-shot stopped mid-run may already have had external effects;
      // replaying it automatically after the restart is not safe. A recurring
      // job's next occurrence is new work, so its cadence continues.
      if (job.oneshot) this.requireReview(job.id, interruption)
      else this.handleFailure(job, now, interruption)
      return
    }
    // The operator stopped this run on purpose. Record that, without the
    // running receipt (it would fence a later resume), and never retry it.
    const { execution_receipt: _receipt, ...currentMetadata } = this.store.get(job.id)?.metadata ?? job.metadata
    const metadata: Record<string, unknown> = { ...currentMetadata, last_cancelled_at: now.toISOString() }
    if (metadata.followup_completion != null) this.store.update(job.id, { paused: true, nextRunAt: null, metadata })
    else if (job.oneshot) this.store.update(job.id, { paused: true, metadata })
    else this.scheduleNext(job, now, false, metadata)
  }

  private isDue(job: CronJob, now: Date): boolean {
    if (job.metadata.execution_receipt != null) {
      this.requireReview(job.id, 'An unfinished execution receipt was recovered.')
      return false
    }
    if (!job.oneshot && !job.schedule && job.intervalSeconds === undefined) {
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
    if (job.missedRunPolicy === 'skip' && now.getTime() - next.getTime() > job.misfireGraceSeconds * 1000) {
      const metadata = { ...job.metadata, last_missed_run: { occurrence: next.toISOString(), observed_at: now.toISOString(), policy: 'skip' } }
      if (job.oneshot) this.store.update(job.id, { paused: true, metadata })
      else this.scheduleNext(job, now, true, metadata)
      return false
    }
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
    if (!job.schedule && job.intervalSeconds === undefined) {
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
      const { execution_receipt: _receipt, execution_recovery_required: _recovery, ...remaining } = metadata ?? this.store.get(job.id)?.metadata ?? job.metadata
      this.store.update(job.id, {
        nextRunAt: (job.intervalSeconds === undefined ? nextFireAt(job.schedule, now, job.timezone) : new Date(now.getTime() + job.intervalSeconds * 1000)).toISOString(),
        ...(!justSeen ? { lastRunAt: now.toISOString() } : {}),
        metadata: remaining,
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

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new Error(`${name} must be a positive integer`)
  return value
}
