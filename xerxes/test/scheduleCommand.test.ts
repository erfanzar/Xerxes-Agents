// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { runScheduleCommand } from '../src/runtime/scheduleCommand.js'

test('schedule command creates, fires, and lists triggers', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-schedule-cmd-'))
  try {
    const createResult = await runScheduleCommand({
      action: 'create',
      id: 'cmd-trigger',
      owner: 'user',
      schedule: 'webhook:/hooks/build',
      objective: 'run build',
      directory,
    })
    expect(createResult.ok).toBeTrue()
    expect(createResult.message).toContain('created trigger cmd-trigger')

    const listResult = await runScheduleCommand({ action: 'list', directory })
    expect(listResult.ok).toBeTrue()
    expect(listResult.message).toContain('cmd-trigger')

    const fireResult = await runScheduleCommand({ action: 'fire', id: 'cmd-trigger', deliveryId: 'delivery-1', directory })
    expect(fireResult.ok).toBeTrue()

    const duplicateFire = await runScheduleCommand({ action: 'fire', id: 'cmd-trigger', deliveryId: 'delivery-1', directory })
    expect(duplicateFire.ok).toBeFalse()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('malformed cron schedules are rejected instead of silently mis-scheduling', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-schedule-cron-validate-'))
  try {
    const create = (schedule: string) => runScheduleCommand({
      action: 'create', directory, id: `t-${schedule}`, owner: 'user', objective: 'run job', schedule,
    })

    // `Number('')` is 0, so a blank field used to become "minute 0" — a real,
    // hourly job the user never asked for.
    expect(await create('cron:')).toMatchObject({ ok: false })
    expect(await create('cron:  ')).toMatchObject({ ok: false })
    // Out of range parses fine and then matches no clock, so the job simply
    // never runs and nothing reports why.
    expect(await create('cron:99')).toMatchObject({ ok: false })
    expect(await create('cron:0/25')).toMatchObject({ ok: false })
    expect(await create('cron:abc')).toMatchObject({ ok: false })

    // Valid shapes still work.
    for (const schedule of ['cron:5/14', 'cron:*/15', 'cron:0,30', 'cron:0-30', 'cron:*']) {
      expect(await create(schedule)).toMatchObject({ ok: true })
    }
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
