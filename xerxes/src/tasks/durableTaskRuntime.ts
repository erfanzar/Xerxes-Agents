// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { appendFile, mkdir, readFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'

import { parseJsonlEventLog, truncateTornTail } from '../core/jsonlEventLog.js'

export type DurableTaskStatus = 'cancelled' | 'completed' | 'failed' | 'pending' | 'running'
export type DurableAttemptStatus = 'cancelled' | 'completed' | 'failed' | 'running'
export interface DurableTaskDefinition {
  readonly creatorId: string
  readonly dependencies: readonly string[]
  readonly id: string
  readonly objective: string
  readonly parentId?: string
}
export interface DurableTask extends DurableTaskDefinition {
  readonly error?: string
  readonly result?: string
  readonly status: DurableTaskStatus
}
export interface DurableAttempt {
  readonly deliveryId?: string
  readonly error?: string
  readonly executorId: string
  readonly id: string
  readonly leaseExpiresAt?: number
  readonly leaseId?: string
  readonly output?: string
  readonly retryable?: boolean
  readonly status: DurableAttemptStatus
  readonly taskId: string
}
export interface DurableResultDelivery { readonly acknowledged: boolean; readonly attemptId: string; readonly output: string }
export interface DurableCohort { readonly id: string; readonly taskIds: readonly string[] }
interface EventIdentity { readonly eventId: string; readonly sequence: number }
type DurableTaskEventPayload =
  | { readonly type: 'task_created'; readonly task: DurableTaskDefinition }
  | { readonly type: 'cohort_created'; readonly cohort: DurableCohort }
  | { readonly type: 'attempt_started'; readonly attempt: Omit<DurableAttempt, 'status'> }
  | { readonly type: 'attempt_completed'; readonly attemptId: string; readonly deliveryId: string; readonly output: string }
  | { readonly type: 'attempt_failed'; readonly attemptId: string; readonly error: string; readonly retryable: boolean }
  | { readonly type: 'result_acknowledged'; readonly deliveryId: string }
  | { readonly type: 'task_cancelled'; readonly error: string; readonly taskId: string }
export type DurableTaskEvent = EventIdentity & DurableTaskEventPayload
export interface DurableTaskState {
  readonly attempts: ReadonlyMap<string, DurableAttempt>
  readonly cohorts: ReadonlyMap<string, DurableCohort>
  readonly deliveries: ReadonlyMap<string, DurableResultDelivery>
  readonly lastSequence: number
  readonly tasks: ReadonlyMap<string, DurableTask>
}

const writes = new Map<string, Promise<void>>()

export class DurableTaskRuntime {
  readonly eventLogPath: string
  readonly now: () => number
  constructor(options: { readonly directory: string; readonly now?: () => number }) {
    this.eventLogPath = join(options.directory, 'tasks.jsonl')
    this.now = options.now ?? (() => Date.now())
  }

  async load(): Promise<DurableTaskState> {
    let text = ''
    try { text = await readFile(this.eventLogPath, 'utf8') } catch (error) {
      if (!isMissing(error)) throw error
    }
    // A crash mid-append can only tear the final record; everything before it
    // is intact. Throwing on that tail bricked the log for every later read and
    // write, in the module whose whole purpose is crash durability.
    const { events } = parseJsonlEventLog(text, {
      label: 'durable task event log',
      isValid: validEvent,
    })
    return project(events)
  }

  async createTask(task: DurableTaskDefinition): Promise<DurableTask> {
    return this.mutate(async state => {
      validateTask(task)
      if (state.tasks.has(task.id)) throw new Error(`duplicate task ${task.id}`)
      for (const dependency of task.dependencies) {
        if (!state.tasks.has(dependency)) throw new Error(`unknown dependency ${dependency}`)
      }
      // Parent IDs are compatibility links; a subagent parent may be a daemon
      // session or root identity that is not itself a task in this runtime.
      await this.append({ type: 'task_created', task }, state.lastSequence + 1)
      return { ...task, status: 'pending' }
    })
  }

  async createCohort(cohort: DurableCohort): Promise<DurableCohort> {
    return this.mutate(async state => {
      if (!cohort.id || state.cohorts.has(cohort.id)) throw new Error(`invalid or duplicate cohort ${cohort.id}`)
      for (const taskId of cohort.taskIds) if (!state.tasks.has(taskId)) throw new Error(`unknown task ${taskId}`)
      await this.append({ type: 'cohort_created', cohort }, state.lastSequence + 1)
      return cohort
    })
  }

  async startAttempt(attempt: Omit<DurableAttempt, 'status'>): Promise<DurableAttempt> {
    return this.mutate(async state => {
      if (!attempt.id || !attempt.executorId) throw new Error('invalid attempt')
      if (state.attempts.has(attempt.id)) throw new Error(`duplicate attempt ${attempt.id}`)
      const task = state.tasks.get(attempt.taskId)
      if (!task) throw new Error(`unknown task ${attempt.taskId}`)
      if (terminal(task.status)) throw new Error(`task ${task.id} is terminal`)
      for (const dependency of task.dependencies) {
        if (state.tasks.get(dependency)?.status !== 'completed') throw new Error(`task ${task.id} dependency ${dependency} is incomplete`)
      }
      const activeLease = activeLeaseForTask(state, task.id, this.now)
      if (activeLease !== undefined) throw new Error(`active lease ${activeLease.leaseId} for task ${task.id}`)
      await this.append({ type: 'attempt_started', attempt }, state.lastSequence + 1)
      return { ...attempt, status: 'running' }
    })
  }

  async completeAttempt(attemptId: string, result: { readonly deliveryId: string; readonly output: string }): Promise<void> {
    await this.mutate(async state => {
      const attempt = state.attempts.get(attemptId)
      if (!attempt) throw new Error(`unknown attempt ${attemptId}`)
      if (attempt.status !== 'running') throw new Error(`attempt ${attemptId} is terminal`)
      if (attempt.leaseExpiresAt !== undefined && attempt.leaseExpiresAt <= this.now()) {
        throw new Error(`lease ${attempt.leaseId} expired`)
      }
      await this.append({ type: 'attempt_completed', attemptId, deliveryId: result.deliveryId, output: result.output }, state.lastSequence + 1)
    })
  }

  async failAttempt(attemptId: string, result: { readonly error: string; readonly retryable: boolean }): Promise<void> {
    await this.mutate(async state => {
      const attempt = state.attempts.get(attemptId)
      if (!attempt) throw new Error(`unknown attempt ${attemptId}`)
      if (attempt.status !== 'running') throw new Error(`attempt ${attemptId} is terminal`)
      await this.append({ type: 'attempt_failed', attemptId, error: result.error, retryable: result.retryable }, state.lastSequence + 1)
    })
  }

  async acknowledgeResult(deliveryId: string): Promise<void> {
    await this.mutate(async state => {
      if (!deliveryId) throw new Error('missing delivery id')
      if (!state.deliveries.has(deliveryId)) throw new Error(`unknown delivery ${deliveryId}`)
      if (state.deliveries.get(deliveryId)?.acknowledged) return
      await this.append({ type: 'result_acknowledged', deliveryId }, state.lastSequence + 1)
    })
  }

  async cancelTask(taskId: string, error: string): Promise<void> {
    await this.mutate(async state => {
      const task = state.tasks.get(taskId)
      if (!task) throw new Error(`unknown task ${taskId}`)
      if (terminal(task.status)) throw new Error(`task ${taskId} is terminal`)
      await this.append({ type: 'task_cancelled', taskId, error }, state.lastSequence + 1)
    })
  }

  private async mutate<T>(operation: (state: DurableTaskState) => Promise<T>): Promise<T> {
    const previous = writes.get(this.eventLogPath) ?? Promise.resolve()
    let release!: () => void
    const current = new Promise<void>(resolve => { release = resolve })
    writes.set(this.eventLogPath, current)
    await previous.catch(() => undefined)
    try { return await operation(await this.load()) } finally {
      release(); if (writes.get(this.eventLogPath) === current) writes.delete(this.eventLogPath)
    }
  }

  private async append(event: DurableTaskEventPayload, sequence: number): Promise<void> {
    await mkdir(dirname(this.eventLogPath), { recursive: true })
    // Repair a tail torn by an earlier crash before adding to it; appending
    // onto a partial record fuses the two into one malformed middle line.
    await truncateTornTail(this.eventLogPath)
    await appendFile(this.eventLogPath, `${JSON.stringify({ ...event, eventId: crypto.randomUUID(), sequence })}
`)
  }
}

function activeLeaseForTask(state: DurableTaskState, taskId: string, now: () => number): DurableAttempt | undefined {
  for (const attempt of state.attempts.values()) {
    if (attempt.taskId !== taskId || attempt.status !== 'running') continue
    if (attempt.leaseExpiresAt !== undefined && attempt.leaseExpiresAt > now()) return attempt
    // Attempts without explicit leases are considered unbounded; once we start
    // tracking leases this path should not arise for new attempts.
  }
  return undefined
}

function project(events: readonly DurableTaskEvent[]): DurableTaskState {
  const tasks = new Map<string, DurableTask>(); const attempts = new Map<string, DurableAttempt>(); const cohorts = new Map<string, DurableCohort>()
  const deliveries = new Map<string, DurableResultDelivery>()
  let expected = 1
  for (const event of events) {
    if (event.sequence !== expected) throw new Error(`task event sequence gap: expected ${expected}, received ${event.sequence}`)
    expected += 1
    switch (event.type) {
      case 'task_created': tasks.set(event.task.id, { ...event.task, status: 'pending' }); break
      case 'cohort_created': cohorts.set(event.cohort.id, event.cohort); break
      case 'attempt_started': {
        const task = tasks.get(event.attempt.taskId); if (!task) throw new Error(`unknown task ${event.attempt.taskId}`)
        attempts.set(event.attempt.id, { ...event.attempt, status: 'running' }); tasks.set(task.id, { ...task, status: 'running' }); break
      }
      case 'attempt_completed': {
        const attempt = attempts.get(event.attemptId); if (!attempt) throw new Error(`unknown attempt ${event.attemptId}`)
        attempts.set(attempt.id, { ...attempt, deliveryId: event.deliveryId, output: event.output, status: 'completed' })
        deliveries.set(event.deliveryId, { acknowledged: false, attemptId: attempt.id, output: event.output })
        const task = tasks.get(attempt.taskId)!; tasks.set(task.id, { ...task, result: event.output, status: 'completed' }); break
      }
      case 'attempt_failed': {
        const attempt = attempts.get(event.attemptId); if (!attempt) throw new Error(`unknown attempt ${event.attemptId}`)
        attempts.set(attempt.id, { ...attempt, error: event.error, retryable: event.retryable, status: 'failed' })
        const task = tasks.get(attempt.taskId)!; tasks.set(task.id, { ...task, error: event.error, status: event.retryable ? 'pending' : 'failed' }); break
      }
      case 'result_acknowledged': {
        const existing = deliveries.get(event.deliveryId)
        if (!existing) throw new Error(`unknown delivery ${event.deliveryId}`)
        deliveries.set(event.deliveryId, { ...existing, acknowledged: true }); break
      }
      case 'task_cancelled': {
        const task = tasks.get(event.taskId); if (!task) throw new Error(`unknown task ${event.taskId}`)
        tasks.set(task.id, { ...task, error: event.error, status: 'cancelled' })
        for (const attempt of attempts.values()) if (attempt.taskId === task.id && attempt.status === 'running') {
          attempts.set(attempt.id, { ...attempt, error: event.error, status: 'cancelled' })
        }
        break
      }
    }
  }
  return { attempts, cohorts, deliveries, lastSequence: events.at(-1)?.sequence ?? 0, tasks }
}

function validEvent(raw: unknown): raw is DurableTaskEvent {
  return typeof raw === 'object' && raw !== null && !Array.isArray(raw)
    && typeof (raw as Record<string, unknown>).eventId === 'string'
    && Number.isSafeInteger((raw as Record<string, unknown>).sequence)
    && typeof (raw as Record<string, unknown>).type === 'string'
}
function validateTask(task: DurableTaskDefinition): void {
  if (!task.id || !task.objective || !task.creatorId || task.dependencies.includes(task.id)) throw new Error('invalid task')
}
function terminal(status: DurableTaskStatus): boolean { return ['cancelled', 'completed', 'failed'].includes(status) }
function isMissing(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && (error as { code?: unknown }).code === 'ENOENT'
}
