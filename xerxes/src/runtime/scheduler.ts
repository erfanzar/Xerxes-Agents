// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { appendFile, mkdir, readFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'

import { parseJsonlEventLog, truncateTornTail } from '../core/jsonlEventLog.js'
import type { DurableTaskDefinition } from '../tasks/durableTaskRuntime.js'

export type TriggerKind = 'interval' | 'cron' | 'webhook' | 'event'

export interface IntervalTrigger {
  readonly kind: 'interval'
  readonly intervalSeconds: number
}

export interface CronTrigger {
  readonly kind: 'cron'
  readonly minute?: string
  readonly hour?: string
  readonly day?: string
  readonly dayOfWeek?: string
}

export interface WebhookTrigger {
  readonly kind: 'webhook'
  readonly path: string
}

export interface EventTrigger {
  readonly kind: 'event'
  readonly topic: string
}

export type TriggerSchedule = IntervalTrigger | CronTrigger | WebhookTrigger | EventTrigger

export interface ScheduledTrigger {
  readonly id: string
  readonly owner: string
  readonly enabled: boolean
  readonly schedule: TriggerSchedule
  readonly payload: DurableTaskDefinition
  readonly lastFiredAt?: number
}

export interface SchedulerState {
  readonly triggers: ReadonlyMap<string, ScheduledTrigger>
  readonly deliveries: ReadonlySet<string>
}

export interface SchedulerOptions {
  readonly directory: string
  readonly now?: () => number
}

export interface CreateTriggerOptions {
  readonly id: string
  readonly owner: string
  readonly schedule: TriggerSchedule
  readonly payload: DurableTaskDefinition
  readonly enabled?: boolean
}

export interface FireResult {
  readonly fired: boolean
  readonly taskId?: string
  readonly reason?: string
}

export class Scheduler {
  readonly logPath: string
  readonly now: () => number

  constructor(options: SchedulerOptions) {
    this.logPath = join(options.directory, 'scheduler.jsonl')
    this.now = options.now ?? (() => Date.now())
  }

  async load(): Promise<SchedulerState> {
    const events = await this.readEvents()
    return project(events, this.now)
  }

  async createTrigger(options: CreateTriggerOptions): Promise<ScheduledTrigger> {
    await this.append({ type: 'trigger_created', trigger: {
      id: options.id,
      owner: options.owner,
      enabled: options.enabled ?? true,
      schedule: options.schedule,
      payload: options.payload,
    }})
    const state = await this.load()
    const trigger = state.triggers.get(options.id)
    if (trigger === undefined) throw new Error('trigger creation failed')
    return trigger
  }

  async disableTrigger(id: string): Promise<void> {
    await this.append({ type: 'trigger_disabled', id })
  }

  async enableTrigger(id: string): Promise<void> {
    await this.append({ type: 'trigger_enabled', id })
  }

  async removeTrigger(id: string): Promise<void> {
    await this.append({ type: 'trigger_removed', id })
  }

  /**
   * Fire an event or webhook trigger exactly once per delivery id.
   *
   * Serialized, because "exactly once" was a read-then-append with nothing
   * between the two: concurrent calls for one delivery id both loaded before
   * either appended, both missed the dedupe set, and both returned fired —
   * so a webhook delivered twice produced two tasks.
   */
  async fire(triggerId: string, deliveryId: string, eventPayload?: Record<string, unknown>): Promise<FireResult> {
    return this.mutate(async state => {
      const trigger = state.triggers.get(triggerId)
      if (trigger === undefined) return { fired: false, reason: 'trigger not found' }
      if (!trigger.enabled) return { fired: false, reason: 'trigger disabled' }
      if (!isEventOrWebhook(trigger.schedule)) return { fired: false, reason: 'trigger is not event or webhook' }
      if (state.deliveries.has(deliveryId)) return { fired: false, reason: 'delivery already processed' }

      const taskId = `${triggerId}:${deliveryId}`
      await this.append({
        type: 'delivery_recorded',
        triggerId,
        deliveryId,
        taskId,
        timestamp: this.now(),
      })
      return { fired: true, taskId }
    })
  }

  /**
   * Record that a time-based trigger ran, so it stops being due.
   *
   * Without this an interval trigger was permanently due: `lastFiredAt` is only
   * written by `delivery_recorded`, and `fire()` refuses anything that is not
   * event or webhook — so nothing could ever write it, and
   * `interval:3600` fired on every poll instead of hourly. Cron had the same
   * shape at minute granularity: any cadence faster than a minute reported the
   * same occurrence due over and over.
   */
  async markFired(triggerId: string, at = this.now()): Promise<boolean> {
    return this.mutate(async state => {
      if (!state.triggers.has(triggerId)) return false
      await this.append({ type: 'trigger_fired', triggerId, timestamp: at })
      return true
    })
  }

  /** Serialize read-modify-append so concurrent callers cannot both win. */
  private async mutate<T>(operation: (state: SchedulerState) => Promise<T>): Promise<T> {
    const previous = schedulerWrites.get(this.logPath) ?? Promise.resolve()
    let release!: () => void
    const current = new Promise<void>(resolve => { release = resolve })
    schedulerWrites.set(this.logPath, current)
    await previous.catch(() => undefined)
    try {
      return await operation(await this.load())
    } finally {
      release()
      if (schedulerWrites.get(this.logPath) === current) schedulerWrites.delete(this.logPath)
    }
  }

  /** Return triggers that are due based on their schedule and the current time. */
  async evaluate(now = this.now()): Promise<readonly ScheduledTrigger[]> {
    const state = await this.load()
    const due: ScheduledTrigger[] = []
    for (const trigger of state.triggers.values()) {
      if (!trigger.enabled) continue
      if (trigger.schedule.kind === 'interval') {
        if (trigger.lastFiredAt === undefined || now - trigger.lastFiredAt >= trigger.schedule.intervalSeconds * 1000) {
          due.push(trigger)
        }
      } else if (trigger.schedule.kind === 'cron') {
        // Cron matching is minute-granular, so a poll cadence faster than a
        // minute reported the same occurrence as due on every tick. One firing
        // per matched minute.
        const firedThisMinute = trigger.lastFiredAt !== undefined
          && Math.floor(trigger.lastFiredAt / 60_000) === Math.floor(now / 60_000)
        if (!firedThisMinute && cronMatches(trigger.schedule, now)) due.push(trigger)
      }
    }
    return due
  }

  private async append(event: SchedulerEvent): Promise<void> {
    await mkdir(dirname(this.logPath), { recursive: true })
    // Repair a tail torn by an earlier crash before adding to it; appending
    // onto a partial record fuses the two into one malformed middle line.
    await truncateTornTail(this.logPath)
    await appendFile(this.logPath, JSON.stringify({ ...event, timestamp: this.now() }) + '\n', 'utf8')
  }

  private async readEvents(): Promise<readonly SchedulerEvent[]> {
    let text = ''
    try { text = await readFile(this.logPath, 'utf8') } catch (error) {
      if (!isMissing(error)) throw error
    }
    // Tolerate only a torn final record — see parseJsonlEventLog. A crash
    // during an append otherwise made every later read and write throw.
    const { events } = parseJsonlEventLog(text, {
      label: 'scheduler event log',
      isValid: isSchedulerEvent,
    })
    return events
  }
}

/** One in-flight mutation per log path, so read-modify-append cannot interleave. */
const schedulerWrites = new Map<string, Promise<void>>()

function project(events: readonly SchedulerEvent[], now: () => number): SchedulerState {
  const triggers = new Map<string, ScheduledTrigger>()
  const deliveries = new Set<string>()
  for (const event of events) {
    switch (event.type) {
      case 'trigger_created': {
        const created: ScheduledTrigger = { ...event.trigger }
        triggers.set(event.trigger.id, created)
        break
      }
      case 'trigger_enabled': {
        const t = triggers.get(event.id)
        if (t !== undefined) triggers.set(event.id, { ...t, enabled: true })
        break
      }
      case 'trigger_disabled': {
        const t = triggers.get(event.id)
        if (t !== undefined) triggers.set(event.id, { ...t, enabled: false })
        break
      }
      case 'trigger_removed': {
        triggers.delete(event.id)
        break
      }
      case 'delivery_recorded': {
        deliveries.add(event.deliveryId)
        const t = triggers.get(event.triggerId)
        if (t !== undefined) triggers.set(event.triggerId, { ...t, lastFiredAt: event.timestamp })
        break
      }
      case 'trigger_fired': {
        const t = triggers.get(event.triggerId)
        if (t !== undefined) triggers.set(event.triggerId, { ...t, lastFiredAt: event.timestamp })
        break
      }
    }
  }
  return { triggers, deliveries }
}

type SchedulerEvent =
  | { readonly type: 'trigger_created'; readonly trigger: ScheduledTrigger }
  | { readonly type: 'trigger_enabled'; readonly id: string }
  | { readonly type: 'trigger_disabled'; readonly id: string }
  | { readonly type: 'trigger_removed'; readonly id: string }
  | { readonly type: 'delivery_recorded'; readonly triggerId: string; readonly deliveryId: string; readonly taskId: string; readonly timestamp: number }
  | { readonly type: 'trigger_fired'; readonly triggerId: string; readonly timestamp: number }

function isSchedulerEvent(value: unknown): value is SchedulerEvent {
  if (!isRecord(value) || typeof value.type !== 'string') return false
  switch (value.type) {
    case 'trigger_created':
      return isRecord(value.trigger) && typeof value.trigger.id === 'string'
    case 'trigger_enabled':
    case 'trigger_disabled':
    case 'trigger_removed':
      return typeof value.id === 'string'
    case 'delivery_recorded':
      return typeof value.triggerId === 'string' && typeof value.deliveryId === 'string' && typeof value.taskId === 'string'
    case 'trigger_fired':
      return typeof value.triggerId === 'string'
    default:
      return false
  }
}

function isEventOrWebhook(schedule: TriggerSchedule): boolean {
  return schedule.kind === 'event' || schedule.kind === 'webhook'
}

function cronMatches(schedule: CronTrigger, nowTimestamp: number): boolean {
  const date = new Date(nowTimestamp)
  if (schedule.minute !== undefined && !fieldMatches(date.getUTCMinutes(), schedule.minute, 0, 59)) return false
  if (schedule.hour !== undefined && !fieldMatches(date.getUTCHours(), schedule.hour, 0, 23)) return false
  if (schedule.day !== undefined && !fieldMatches(date.getUTCDate(), schedule.day, 1, 31)) return false
  if (schedule.dayOfWeek !== undefined && !fieldMatches(date.getUTCDay(), schedule.dayOfWeek, 0, 6)) return false
  return true
}

/**
 * Match one cron field.
 *
 * `min`/`max` were accepted and never used, so `minute:99` parsed cleanly and
 * then never fired — a job silently scheduled for a time that does not exist.
 * A blank field was worse: `Number('')` is 0, so `cron:` became "every hour at
 * minute 0" rather than an error. Both now fail the match loudly at parse time
 * via {@link isValidCronField}; here, an out-of-range literal simply cannot
 * match.
 */
function fieldMatches(value: number, pattern: string, min: number, max: number): boolean {
  if (pattern === '*') return true
  if (pattern.startsWith('*/')) {
    const step = Number(pattern.slice(2))
    if (Number.isNaN(step) || step <= 0) return false
    return value % step === 0
  }
  const parts = pattern.split(',')
  for (const part of parts) {
    if (part.includes('-')) {
      const range = part.split('-').map(Number)
      const [start, end] = [range[0], range[1]]
      if (start !== undefined && end !== undefined && !Number.isNaN(start) && !Number.isNaN(end) && value >= start && value <= end) return true
    } else {
      const n = Number(part)
      if (!Number.isNaN(n) && value === n) return true
    }
  }
  return false
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isMissing(error: unknown): boolean {
  return error instanceof Error && 'code' in error && (error as { code: unknown }).code === 'ENOENT'
}
