// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir } from 'node:fs/promises'
import { dirname, join } from 'node:path'

import { xerxesHome } from '../daemon/paths.js'
import { Scheduler } from './scheduler.js'

export type ScheduleCommandAction = 'create' | 'disable' | 'enable' | 'remove' | 'fire' | 'list'

export interface ScheduleCommandOptions {
  readonly action: ScheduleCommandAction
  readonly id?: string
  readonly owner?: string
  readonly schedule?: string
  readonly objective?: string
  readonly deliveryId?: string
  readonly directory?: string
}

export interface ScheduleCommandResult {
  readonly ok: boolean
  readonly message?: string
  readonly error?: string
}

export async function runScheduleCommand(options: ScheduleCommandOptions): Promise<ScheduleCommandResult> {
  const directory = options.directory ?? join(xerxesHome(), 'scheduler')
  await mkdir(dirname(directory), { recursive: true })
  const scheduler = new Scheduler({ directory })

  switch (options.action) {
    case 'create': {
      if (!options.id || !options.owner || !options.schedule || !options.objective) {
        return { ok: false, error: 'create requires --id, --owner, --schedule, and --objective' }
      }
      const parsed = parseSchedule(options.schedule)
      if (!parsed.ok) return { ok: false, error: parsed.error }
      const trigger = await scheduler.createTrigger({
        id: options.id,
        owner: options.owner,
        schedule: parsed.schedule,
        payload: { id: options.id, objective: options.objective, creatorId: options.owner, dependencies: [] },
      })
      return { ok: true, message: `created trigger ${trigger.id}` }
    }
    case 'disable': {
      if (!options.id) return { ok: false, error: 'disable requires --id' }
      await scheduler.disableTrigger(options.id)
      return { ok: true, message: `disabled trigger ${options.id}` }
    }
    case 'enable': {
      if (!options.id) return { ok: false, error: 'enable requires --id' }
      await scheduler.enableTrigger(options.id)
      return { ok: true, message: `enabled trigger ${options.id}` }
    }
    case 'remove': {
      if (!options.id) return { ok: false, error: 'remove requires --id' }
      await scheduler.removeTrigger(options.id)
      return { ok: true, message: `removed trigger ${options.id}` }
    }
    case 'fire': {
      if (!options.id || !options.deliveryId) return { ok: false, error: 'fire requires --id and --delivery-id' }
      const result = await scheduler.fire(options.id, options.deliveryId)
      if (!result.fired) return { ok: false, error: result.reason ?? 'fire failed' }
      return { ok: true, message: `fired trigger ${options.id} with task ${result.taskId}` }
    }
    case 'list': {
      const state = await scheduler.load()
      const lines = Array.from(state.triggers.values()).map(t => `${t.id}\t${t.enabled ? 'enabled' : 'disabled'}\t${t.schedule.kind}`)
      return { ok: true, message: lines.join('\n') || 'no triggers' }
    }
  }
}

function parseSchedule(schedule: string): { ok: true; schedule: Parameters<Scheduler['createTrigger']>[0]['schedule'] } | { ok: false; error: string } {
  if (schedule.startsWith('interval:')) {
    const seconds = Number(schedule.slice('interval:'.length))
    if (Number.isNaN(seconds) || seconds <= 0) return { ok: false, error: 'interval requires a positive number of seconds' }
    return { ok: true, schedule: { kind: 'interval', intervalSeconds: seconds } }
  }
  if (schedule.startsWith('webhook:')) {
    return { ok: true, schedule: { kind: 'webhook', path: schedule.slice('webhook:'.length) } }
  }
  if (schedule.startsWith('event:')) {
    return { ok: true, schedule: { kind: 'event', topic: schedule.slice('event:'.length) } }
  }
  if (schedule.startsWith('cron:')) {
    const parts = schedule.slice('cron:'.length).split('/')
    const fields = [
      { name: 'minute', value: parts[0], min: 0, max: 59 },
      { name: 'hour', value: parts[1], min: 0, max: 23 },
      { name: 'day', value: parts[2], min: 1, max: 31 },
      { name: 'day-of-week', value: parts[3], min: 0, max: 6 },
    ] as const
    // Reject here, where the user is still looking. A blank field used to
    // become 0 (`Number('')`), so `cron:` silently created an hourly job, and
    // an out-of-range value parsed fine and then never fired — the two worst
    // outcomes for a scheduler, both silent.
    for (const field of fields) {
      if (field.value === undefined) continue
      if (!isValidCronField(field.value, field.min, field.max)) {
        return { ok: false, error: `cron ${field.name} field ${JSON.stringify(field.value)} is not a valid `
          + `${field.min}-${field.max} value, list, range, or step` }
      }
    }
    // Omit blank trailing fields rather than storing undefined-as-present.
    return {
      ok: true,
      schedule: {
        kind: 'cron',
        ...(parts[0] === undefined ? {} : { minute: parts[0] }),
        ...(parts[1] === undefined ? {} : { hour: parts[1] }),
        ...(parts[2] === undefined ? {} : { day: parts[2] }),
        ...(parts[3] === undefined ? {} : { dayOfWeek: parts[3] }),
      },
    }
  }
  return { ok: false, error: 'schedule must be interval:<sec>, webhook:<path>, event:<topic>, or cron:<min>/<hour>/<day>/<dow>' }
}

/**
 * Whether one cron field is a usable `*`, step, list, range, or literal.
 *
 * Blank and whitespace are rejected because `Number('')` is 0, which turned a
 * malformed schedule into a real but unrequested one; literals outside the
 * field's range are rejected because they parse fine and then match no clock.
 */
export function isValidCronField(pattern: string, min: number, max: number): boolean {
  if (pattern === '*') return true
  if (pattern.startsWith('*/')) {
    const step = Number(pattern.slice(2))
    return Number.isInteger(step) && step > 0 && step <= max
  }
  const parts = pattern.split(',')
  if (!parts.length) return false
  return parts.every(part => {
    if (part.trim() === '') return false
    const bounds = part.includes('-') ? part.split('-') : [part]
    if (bounds.length > 2) return false
    return bounds.every(bound => {
      if (bound.trim() === '' || !/^\d+$/.test(bound.trim())) return false
      const value = Number(bound)
      return Number.isInteger(value) && value >= min && value <= max
    })
  })
}
