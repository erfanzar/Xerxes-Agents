// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import { nextFireAt } from '../cron/jobs.js'
import type { ToolRegistry } from '../executors/toolRegistry.js'
import type { SwarmBoardMessage } from '../swarm/index.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'

const MAX_ACTIVE_TASKS_PER_MEMBER = 16
const MAX_RETAINED_TASKS_PER_MEMBER = 64
const MAX_PROMPT_LENGTH = 16_000
const MAX_DURATION_SECONDS = 86_400
const MIN_DURATION_SECONDS = 0.05
const MAX_CRON_ITERATIONS = 1_000
const SENDER_NAME_PATTERN = /^[a-zA-Z0-9_-]{1,64}$/

export type SwarmWorkerScheduleState = 'scheduled' | 'completed' | 'cancelled' | 'killed'

export interface SwarmWorkerTaskSnapshot {
  readonly task_id: string
  readonly session_id: string
  readonly member: string
  readonly kind: 'timer' | 'cron'
  readonly state: SwarmWorkerScheduleState
  readonly prompt: string
  readonly created_at: string
  readonly next_fire_at: string | null
  readonly timer_condition?: string
  readonly duration_seconds?: number
  readonly cron_expression?: string
  readonly exact_timing?: boolean
  readonly is_daemon?: boolean
  readonly runs_completed: number
  readonly max_iterations?: number
  readonly last_triggered_at?: string
  readonly cancel_reason?: string
}

interface InternalWorkerScheduleTask {
  readonly id: string
  readonly sessionId: string
  readonly memberName: string
  readonly kind: 'timer' | 'cron'
  readonly prompt: string
  readonly createdAt: string
  readonly timerCondition?: string
  readonly durationSeconds?: number
  readonly cronExpression?: string
  readonly exactTiming?: boolean
  readonly isDaemon?: boolean
  readonly maxIterations?: number
  readonly wakeCallback: (memberName: string, wakePrompt: string) => Promise<boolean> | boolean
  state: SwarmWorkerScheduleState
  nextFireAt: string | null
  runsCompleted: number
  lastTriggeredAt?: string
  cancelReason?: string
  timeout?: ReturnType<typeof setTimeout>
}

export interface CreateSwarmWorkerScheduleInput {
  readonly sessionId: string
  readonly memberName: string
  readonly prompt: string
  readonly durationSeconds?: number
  readonly timerCondition?: string
  readonly cronExpression?: string
  readonly maxIterations?: number
  readonly exactTiming?: boolean
  readonly isDaemon?: boolean
  readonly wake: (memberName: string, wakePrompt: string) => Promise<boolean> | boolean
}

function memberKey(sessionId: string, memberName: string): string {
  return `${sessionId}\u0000${memberName}`
}

function toTaskSnapshot(task: InternalWorkerScheduleTask): SwarmWorkerTaskSnapshot {
  return {
    task_id: task.id,
    session_id: task.sessionId,
    member: task.memberName,
    kind: task.kind,
    state: task.state,
    prompt: task.prompt,
    created_at: task.createdAt,
    next_fire_at: task.nextFireAt,
    ...(task.timerCondition === undefined ? {} : { timer_condition: task.timerCondition }),
    ...(task.durationSeconds === undefined ? {} : { duration_seconds: task.durationSeconds }),
    ...(task.cronExpression === undefined ? {} : { cron_expression: task.cronExpression }),
    ...(task.exactTiming === undefined ? {} : { exact_timing: task.exactTiming }),
    ...(task.isDaemon === undefined ? {} : { is_daemon: task.isDaemon }),
    runs_completed: task.runsCompleted,
    ...(task.maxIterations === undefined ? {} : { max_iterations: task.maxIterations }),
    ...(task.lastTriggeredAt === undefined ? {} : { last_triggered_at: task.lastTriggeredAt }),
    ...(task.cancelReason === undefined ? {} : { cancel_reason: task.cancelReason }),
  }
}

function normalizeTimerCondition(raw: string | undefined): string {
  if (raw === undefined) return 'never'
  const trimmed = raw.trim()
  if (!trimmed || trimmed === 'never') return 'never'
  if (trimmed === 'any') return 'any'
  const stripped = trimmed.startsWith('@') ? trimmed.slice(1) : trimmed
  if (!SENDER_NAME_PATTERN.test(stripped)) {
    throw new ValidationError(
      'TimerCondition',
      'must be "never", "any", or a valid swarm sender name',
      raw,
    )
  }
  return stripped
}

function formatScheduledWakePrompt(prompt: string): string {
  const trimmed = prompt.trim()
  return trimmed.startsWith('[schedule]') ? trimmed : `[schedule] ${trimmed}`
}

export class SwarmWorkerScheduleRegistry {
  private readonly tasksByMember = new Map<string, Map<string, InternalWorkerScheduleTask>>()
  private nextSequence = 0
  private closed = false

  create(input: CreateSwarmWorkerScheduleInput): SwarmWorkerTaskSnapshot {
    if (this.closed) {
      throw new Error('Swarm worker schedule registry is closed')
    }
    const prompt = input.prompt.trim()
    if (!prompt || prompt.length > MAX_PROMPT_LENGTH) {
      throw new ValidationError('Prompt', `must contain 1–${MAX_PROMPT_LENGTH} characters`)
    }

    const hasDuration = input.durationSeconds !== undefined
    const hasCron = input.cronExpression !== undefined && input.cronExpression.trim().length > 0
    if (hasDuration === hasCron) {
      throw new ValidationError(
        'schedule',
        'provide exactly one of DurationSeconds (one-shot timer) or CronExpression (5-field cron)',
      )
    }

    const scopeKey = memberKey(input.sessionId, input.memberName)
    let bucket = this.tasksByMember.get(scopeKey)
    if (!bucket) {
      bucket = new Map()
      this.tasksByMember.set(scopeKey, bucket)
    }

    const activeCount = [...bucket.values()].filter(item => item.state === 'scheduled').length
    if (activeCount >= MAX_ACTIVE_TASKS_PER_MEMBER) {
      throw new ValidationError(
        'schedule',
        `maximum of ${MAX_ACTIVE_TASKS_PER_MEMBER} active schedules reached for worker "${input.memberName}"; cancel stale tasks with manage_task first`,
      )
    }

    const now = new Date()
    const createdAt = now.toISOString()

    if (hasDuration) {
      const durationSeconds = input.durationSeconds!
      if (!Number.isFinite(durationSeconds) || durationSeconds < MIN_DURATION_SECONDS || durationSeconds > MAX_DURATION_SECONDS) {
        throw new ValidationError(
          'DurationSeconds',
          `must be between ${MIN_DURATION_SECONDS} and ${MAX_DURATION_SECONDS} seconds`,
          durationSeconds,
        )
      }
      if (input.maxIterations !== undefined) {
        throw new ValidationError('MaxIterations', 'is only supported with CronExpression')
      }
      if (input.exactTiming !== undefined) {
        throw new ValidationError('ExactTiming', 'is only supported with CronExpression')
      }
      if (input.isDaemon !== undefined) {
        throw new ValidationError('IsDaemon', 'is only supported with CronExpression')
      }
      const timerCondition = normalizeTimerCondition(input.timerCondition)
      const delayMs = Math.max(1, Math.round(durationSeconds * 1_000))
      const nextFire = new Date(now.getTime() + delayMs).toISOString()
      const id = `sched-${++this.nextSequence}`

      const task: InternalWorkerScheduleTask = {
        id,
        sessionId: input.sessionId,
        memberName: input.memberName,
        kind: 'timer',
        prompt,
        createdAt,
        timerCondition,
        durationSeconds,
        wakeCallback: input.wake,
        state: 'scheduled',
        nextFireAt: nextFire,
        runsCompleted: 0,
      }
      bucket.set(id, task)
      this.pruneRetained(bucket)
      this.armTimeout(task, delayMs)
      return toTaskSnapshot(task)
    }

    const cronExpression = input.cronExpression!.trim()
    if (input.timerCondition !== undefined && input.timerCondition.trim() !== '' && input.timerCondition.trim() !== 'never') {
      throw new ValidationError('TimerCondition', 'is only supported for one-shot DurationSeconds timers')
    }
    if (
      input.maxIterations !== undefined
      && (!Number.isInteger(input.maxIterations) || input.maxIterations < 1 || input.maxIterations > MAX_CRON_ITERATIONS)
    ) {
      throw new ValidationError('MaxIterations', `must be an integer between 1 and ${MAX_CRON_ITERATIONS}`, input.maxIterations)
    }

    const nextFireDate = nextFireAt(cronExpression, now, 'UTC')
    const baseDelayMs = Math.max(1, nextFireDate.getTime() - now.getTime())
    const jitterMs = input.exactTiming === true || baseDelayMs <= 5_000
      ? 0
      : Math.floor(Math.random() * 1_000)
    const id = `sched-${++this.nextSequence}`

    const task: InternalWorkerScheduleTask = {
      id,
      sessionId: input.sessionId,
      memberName: input.memberName,
      kind: 'cron',
      prompt,
      createdAt,
      cronExpression,
      ...(input.exactTiming === undefined ? {} : { exactTiming: input.exactTiming }),
      ...(input.isDaemon === undefined ? {} : { isDaemon: input.isDaemon }),
      ...(input.maxIterations === undefined ? {} : { maxIterations: input.maxIterations }),
      wakeCallback: input.wake,
      state: 'scheduled',
      nextFireAt: new Date(now.getTime() + baseDelayMs + jitterMs).toISOString(),
      runsCompleted: 0,
    }
    bucket.set(id, task)
    this.pruneRetained(bucket)
    this.armTimeout(task, baseDelayMs + jitterMs)
    return toTaskSnapshot(task)
  }

  listForMember(sessionId: string, memberName: string): SwarmWorkerTaskSnapshot[] {
    const bucket = this.tasksByMember.get(memberKey(sessionId, memberName))
    if (!bucket) return []
    return [...bucket.values()]
      .filter(task => task.state === 'scheduled')
      .map(toTaskSnapshot)
  }

  statusForMember(sessionId: string, memberName: string, taskId: string): SwarmWorkerTaskSnapshot {
    const cleanId = taskId.trim()
    const bucket = this.tasksByMember.get(memberKey(sessionId, memberName))
    const task = bucket?.get(cleanId)
    if (!task) {
      throw new ValidationError('TaskId', `unknown schedule task "${cleanId}" for worker "${memberName}"`, taskId)
    }
    return toTaskSnapshot(task)
  }

  killForMember(
    sessionId: string,
    memberName: string,
    taskId: string,
  ): { readonly cancelled: boolean; readonly task: SwarmWorkerTaskSnapshot } {
    const cleanId = taskId.trim()
    const bucket = this.tasksByMember.get(memberKey(sessionId, memberName))
    const task = bucket?.get(cleanId)
    if (!task) {
      throw new ValidationError('TaskId', `unknown schedule task "${cleanId}" for worker "${memberName}"`, taskId)
    }
    const wasScheduled = task.state === 'scheduled'
    if (wasScheduled) {
      this.clearTaskTimer(task)
      task.state = 'killed'
      task.nextFireAt = null
      task.cancelReason = 'killed_by_manage_task'
    }
    return {
      cancelled: wasScheduled,
      task: toTaskSnapshot(task),
    }
  }

  onBoardMessagePosted(sessionId: string, message: SwarmBoardMessage): number {
    let cancelled = 0
    const prefix = `${sessionId}\u0000`
    for (const [key, bucket] of this.tasksByMember.entries()) {
      if (!key.startsWith(prefix)) continue
      for (const task of bucket.values()) {
        if (task.state !== 'scheduled' || task.kind !== 'timer') continue
        const condition = task.timerCondition ?? 'never'
        if (condition === 'never') continue
        if (this.matchesTimerCondition(task, condition, message)) {
          this.clearTaskTimer(task)
          task.state = 'cancelled'
          task.nextFireAt = null
          task.cancelReason = `early_cancel:message_#${message.seq}_from_${message.agent}`
          cancelled += 1
        }
      }
    }
    return cancelled
  }

  cancelMember(sessionId: string, memberName: string, reason = 'member_killed'): number {
    const bucket = this.tasksByMember.get(memberKey(sessionId, memberName))
    if (!bucket) return 0
    let count = 0
    for (const task of bucket.values()) {
      if (task.state === 'scheduled') {
        this.clearTaskTimer(task)
        task.state = 'cancelled'
        task.nextFireAt = null
        task.cancelReason = reason
        count += 1
      }
    }
    return count
  }

  cancelSession(sessionId: string, reason = 'session_disposed'): number {
    let count = 0
    const prefix = `${sessionId}\u0000`
    for (const [key, bucket] of this.tasksByMember.entries()) {
      if (!key.startsWith(prefix)) continue
      for (const task of bucket.values()) {
        if (task.state === 'scheduled') {
          this.clearTaskTimer(task)
          task.state = 'cancelled'
          task.nextFireAt = null
          task.cancelReason = reason
          count += 1
        }
      }
      this.tasksByMember.delete(key)
    }
    return count
  }

  activeCount(sessionId?: string, memberName?: string): number {
    if (sessionId !== undefined && memberName !== undefined) {
      const bucket = this.tasksByMember.get(memberKey(sessionId, memberName))
      if (!bucket) return 0
      return [...bucket.values()].filter(task => task.state === 'scheduled').length
    }
    let count = 0
    const prefix = sessionId !== undefined ? `${sessionId}\u0000` : undefined
    for (const [key, bucket] of this.tasksByMember.entries()) {
      if (prefix !== undefined && !key.startsWith(prefix)) continue
      for (const task of bucket.values()) {
        if (task.state === 'scheduled') count += 1
      }
    }
    return count
  }

  close(): void {
    this.closed = true
    for (const bucket of this.tasksByMember.values()) {
      for (const task of bucket.values()) {
        if (task.state === 'scheduled') {
          this.clearTaskTimer(task)
          task.state = 'cancelled'
          task.nextFireAt = null
          task.cancelReason = 'host_closed'
        }
      }
    }
    this.tasksByMember.clear()
  }

  private matchesTimerCondition(
    task: InternalWorkerScheduleTask,
    condition: string,
    message: SwarmBoardMessage,
  ): boolean {
    if (condition === 'any') {
      // A worker's own outgoing board post does not count as external board activity.
      return message.agent !== task.memberName
    }
    return message.agent === condition || message.agent.startsWith(`${condition}-`)
  }

  private armTimeout(task: InternalWorkerScheduleTask, delayMs: number): void {
    const timer = setTimeout(() => {
      void this.onTimerFired(task)
    }, delayMs)
    if (task.isDaemon === true) {
      timer.unref?.()
    }
    task.timeout = timer
  }

  private async onTimerFired(task: InternalWorkerScheduleTask): Promise<void> {
    delete task.timeout
    if (this.closed || task.state !== 'scheduled') {
      return
    }

    const firedAt = new Date()
    task.runsCompleted += 1
    task.lastTriggeredAt = firedAt.toISOString()

    if (task.kind === 'timer') {
      task.state = 'completed'
      task.nextFireAt = null
    } else if (task.maxIterations !== undefined && task.runsCompleted >= task.maxIterations) {
      task.state = 'completed'
      task.nextFireAt = null
    } else if (task.cronExpression) {
      try {
        const nextDate = nextFireAt(task.cronExpression, firedAt, 'UTC')
        const baseDelayMs = Math.max(1, nextDate.getTime() - Date.now())
        const jitterMs = task.exactTiming === true || baseDelayMs <= 5_000
          ? 0
          : Math.floor(Math.random() * 1_000)
        task.nextFireAt = new Date(Date.now() + baseDelayMs + jitterMs).toISOString()
        this.armTimeout(task, baseDelayMs + jitterMs)
      } catch {
        task.state = 'cancelled'
        task.nextFireAt = null
        task.cancelReason = 'cron_exhausted'
      }
    }

    try {
      const delivered = await task.wakeCallback(task.memberName, formatScheduledWakePrompt(task.prompt))
      if (!delivered && task.state === 'scheduled') {
        this.clearTaskTimer(task)
        task.state = 'cancelled'
        task.nextFireAt = null
        task.cancelReason = 'member_unavailable'
      }
    } catch {
      if (task.state === 'scheduled') {
        this.clearTaskTimer(task)
        task.state = 'cancelled'
        task.nextFireAt = null
        task.cancelReason = 'wake_failed'
      }
    }
  }

  private clearTaskTimer(task: InternalWorkerScheduleTask): void {
    if (task.timeout !== undefined) {
      clearTimeout(task.timeout)
      delete task.timeout
    }
  }

  private pruneRetained(bucket: Map<string, InternalWorkerScheduleTask>): void {
    if (bucket.size <= MAX_RETAINED_TASKS_PER_MEMBER) return
    for (const [id, item] of bucket.entries()) {
      if (bucket.size <= MAX_RETAINED_TASKS_PER_MEMBER) break
      if (item.state !== 'scheduled') {
        bucket.delete(id)
      }
    }
  }
}

export const SWARM_WORKER_SCHEDULE_TOOL_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'schedule',
    description:
      'Schedule a one-shot timer (DurationSeconds) or a recurring 5-field UTC cron job (CronExpression) that wakes you directly with "[schedule] <Prompt>" without posting noise to the swarm board. Set TimerCondition ("any" or a member name) on a one-shot timer to cancel it early as soon as matching board traffic arrives. End your turn or call swarm_wait after scheduling; use manage_task to inspect or cancel active schedules.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        Prompt: {
          type: 'string',
          minLength: 1,
          maxLength: MAX_PROMPT_LENGTH,
          description: 'Wakeup message delivered to you as "[schedule] <Prompt>" when the timer or cron fires.',
        },
        DurationSeconds: {
          type: 'number',
          minimum: MIN_DURATION_SECONDS,
          maximum: MAX_DURATION_SECONDS,
          description: 'Seconds to wait for a one-shot timer. Mutually exclusive with CronExpression.',
        },
        TimerCondition: {
          type: 'string',
          minLength: 1,
          maxLength: 64,
          description:
            'Timer only. Early-cancellation condition when board traffic arrives: "never" (default), "any" (any other member posts), or a sender name.',
        },
        CronExpression: {
          type: 'string',
          minLength: 1,
          maxLength: 256,
          description: 'Standard 5-field UTC cron expression for a recurring wakeup. Mutually exclusive with DurationSeconds.',
        },
        MaxIterations: {
          type: 'integer',
          minimum: 1,
          maximum: MAX_CRON_ITERATIONS,
          description: 'Cron only. Maximum number of triggers before stopping automatically.',
        },
        ExactTiming: {
          type: 'boolean',
          description: 'Cron only. Fire at exact wall-clock minute boundary without jitter.',
        },
        IsDaemon: {
          type: 'boolean',
          description: 'Cron only. Keep firing as a standing background schedule until killed.',
        },
      },
      required: ['Prompt'],
    },
  },
}

export const SWARM_WORKER_MANAGE_TASK_TOOL_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'manage_task',
    description:
      'Manage your active swarm worker schedule tasks: "list" running tasks, inspect a task "status", or "kill" (cancel) a task by TaskId.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        Action: {
          type: 'string',
          enum: ['list', 'status', 'kill'],
          description: '"list" reports active tasks; "status" and "kill" operate on TaskId.',
        },
        TaskId: {
          type: 'string',
          minLength: 1,
          maxLength: 128,
          description: 'Task identifier returned by schedule (required for "status" and "kill").',
        },
      },
      required: ['Action'],
    },
  },
}

export interface SwarmWorkerScheduleBinding {
  readonly sessionId: string
  readonly memberName: string
  readonly wake: (memberName: string, wakePrompt: string) => Promise<boolean> | boolean
}

export function registerSwarmWorkerScheduleTools(
  registry: ToolRegistry,
  schedules: SwarmWorkerScheduleRegistry,
  binding: SwarmWorkerScheduleBinding,
): void {
  registry.replace(
    SWARM_WORKER_SCHEDULE_TOOL_DEFINITION,
    (args: JsonObject) => {
      const prompt = typeof args.Prompt === 'string' ? args.Prompt : ''
      const durationSeconds = typeof args.DurationSeconds === 'number' ? args.DurationSeconds : undefined
      const timerCondition = typeof args.TimerCondition === 'string' ? args.TimerCondition : undefined
      const cronExpression = typeof args.CronExpression === 'string' ? args.CronExpression : undefined
      const maxIterations = typeof args.MaxIterations === 'number' ? args.MaxIterations : undefined
      const exactTiming = typeof args.ExactTiming === 'boolean' ? args.ExactTiming : undefined
      const isDaemon = typeof args.IsDaemon === 'boolean' ? args.IsDaemon : undefined

      const created = schedules.create({
        sessionId: binding.sessionId,
        memberName: binding.memberName,
        prompt,
        ...(durationSeconds === undefined ? {} : { durationSeconds }),
        ...(timerCondition === undefined ? {} : { timerCondition }),
        ...(cronExpression === undefined ? {} : { cronExpression }),
        ...(maxIterations === undefined ? {} : { maxIterations }),
        ...(exactTiming === undefined ? {} : { exactTiming }),
        ...(isDaemon === undefined ? {} : { isDaemon }),
        wake: binding.wake,
      })
      return {
        ok: true,
        status: 'scheduled',
        task_id: created.task_id,
        task: created,
      }
    },
    'default',
    {
      concurrencySafe: true,
      defer: false,
      destructive: false,
      interruptBehavior: 'block',
      openWorld: false,
      readOnly: false,
    },
  )

  registry.replace(
    SWARM_WORKER_MANAGE_TASK_TOOL_DEFINITION,
    (args: JsonObject) => {
      const action = typeof args.Action === 'string' ? args.Action : ''
      if (action === 'list') {
        const tasks = schedules.listForMember(binding.sessionId, binding.memberName)
        return {
          ok: true,
          tasks,
        }
      }
      const taskId = typeof args.TaskId === 'string' ? args.TaskId.trim() : ''
      if (!taskId) {
        throw new ValidationError('TaskId', `is required when Action is "${action}"`)
      }
      if (action === 'status') {
        return {
          ok: true,
          task: schedules.statusForMember(binding.sessionId, binding.memberName, taskId),
        }
      }
      if (action === 'kill') {
        const outcome = schedules.killForMember(binding.sessionId, binding.memberName, taskId)
        return {
          ok: true,
          task_id: outcome.task.task_id,
          status: outcome.task.state,
          cancelled: outcome.cancelled,
          task: outcome.task,
        }
      }
      throw new ValidationError('Action', 'must be "list", "status", or "kill"', action)
    },
    'default',
    {
      concurrencySafe: true,
      defer: false,
      destructive: false,
      interruptBehavior: 'block',
      openWorld: false,
      readOnly: false,
    },
  )
}
