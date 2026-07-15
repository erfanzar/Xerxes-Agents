// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  CronJob,
  CronScheduler,
  JobStore,
  nextFireAt,
  routeOutput,
} from '../src/cron/index.js'

function temporaryDirectory(): string {
  return mkdtempSync(join(tmpdir(), 'xerxes-cron-'))
}

function removeDirectory(path: string): void {
  rmSync(path, { recursive: true, force: true })
}

test('cron arithmetic supports every minute, specific times, steps, and Sunday-zero', () => {
  const base = new Date('2026-05-15T12:00:00.000Z')

  expect(nextFireAt('* * * * *', base).toISOString()).toBe(
    '2026-05-15T12:01:00.000Z',
  )
  expect(
    nextFireAt('0 9 * * *', new Date('2026-05-15T08:30:00.000Z')).toISOString(),
  ).toBe('2026-05-15T09:00:00.000Z')
  expect(
    nextFireAt(
      '*/15 * * * *',
      new Date('2026-05-15T12:04:00.000Z'),
    ).toISOString(),
  ).toBe('2026-05-15T12:15:00.000Z')

  const sunday = nextFireAt('0 9 * * 0', new Date('2026-05-16T23:59:00.000Z'))
  expect(sunday.getUTCDay()).toBe(0)
  expect(sunday.getUTCHours()).toBe(9)
})

test('cron arithmetic rejects expressions without five fields', () => {
  expect(() => nextFireAt('not enough fields')).toThrow('5-field')
})

test('job store initializes empty, persists jobs, and updates them', () => {
  const directory = temporaryDirectory()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    expect(store.listJobs()).toEqual([])

    const id = store.newId()
    expect(id).toMatch(/^[a-f0-9]{12}$/)
    const job = store.add(
      new CronJob({
        id,
        prompt: 'say hi',
        schedule: '* * * * *',
        workspaceId: 'workspace',
        metadata: { source: 'test' },
      }),
    )
    expect(store.get(job.id)).toMatchObject({
      id,
      prompt: 'say hi',
      workspaceId: 'workspace',
      metadata: { source: 'test' },
    })
    expect(store.update(job.id, { paused: true })?.paused).toBe(true)
    expect(store.get(job.id)?.paused).toBe(true)
  } finally {
    removeDirectory(directory)
  }
})

test('job store replaces same-id jobs and reports removals', () => {
  const directory = temporaryDirectory()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(new CronJob({ id: 'abc', prompt: 'first' }))
    store.add(new CronJob({ id: 'abc', prompt: 'second' }))

    expect(store.listJobs()).toHaveLength(1)
    expect(store.get('abc')?.prompt).toBe('second')
    expect(store.remove('abc')).toBe(true)
    expect(store.listJobs()).toEqual([])
    expect(store.remove('abc')).toBe(false)
  } finally {
    removeDirectory(directory)
  }
})

test('route output always archives and does not deliver archive-only targets', async () => {
  const directory = temporaryDirectory()
  try {
    const sent: unknown[][] = []
    const path = await routeOutput({ platform: 'none' }, 'content', {
      archiveDirectory: directory,
      jobId: 'job1',
      sender: (...args) => {
        sent.push(args)
      },
    })

    expect(readFileSync(path, 'utf8')).toBe('content')
    expect(sent).toEqual([])

    const workspacePath = await routeOutput(
      { platform: 'workspace' },
      'workspace content',
      {
        archiveDirectory: directory,
        jobId: 'workspace-job',
        sender: (...args) => {
          sent.push(args)
        },
      },
    )
    expect(readFileSync(workspacePath, 'utf8')).toBe('workspace content')
    expect(sent).toEqual([])
  } finally {
    removeDirectory(directory)
  }
})

test('route output forwards channel targets with the configured recipient', async () => {
  const directory = temporaryDirectory()
  try {
    const sent: Array<[string, string, string]> = []
    const path = await routeOutput(
      { platform: 'telegram', recipient: '123' },
      'Daily digest',
      {
        archiveDirectory: directory,
        jobId: 'job1',
        sender: (platform, recipient, content) => {
          sent.push([platform, recipient, content])
        },
      },
    )

    expect(readFileSync(path, 'utf8')).toBe('Daily digest')
    expect(sent).toEqual([['telegram', '123', 'Daily digest']])
  } finally {
    removeDirectory(directory)
  }
})

test('scheduler schedules a recurring job before its first fire', async () => {
  const directory = temporaryDirectory()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    const job = store.add(
      new CronJob({
        id: 'nightly',
        prompt: 'summarize',
        schedule: '* * * * *',
      }),
    )
    const scheduler = new CronScheduler(store, (current) => `ran:${current.id}`)
    const now = new Date('2026-05-15T12:00:00.000Z')

    expect(await scheduler.tick(now)).toEqual([])
    expect(store.get(job.id)?.nextRunAt).toBe('2026-05-15T12:01:00.000Z')
    expect(await scheduler.tick(new Date('2026-05-15T12:02:00.000Z'))).toEqual([
      'nightly',
    ])
    expect(store.get(job.id)?.lastRunAt).toBe('2026-05-15T12:02:00.000Z')
  } finally {
    removeDirectory(directory)
  }
})

test('scheduler skips paused jobs', async () => {
  const directory = temporaryDirectory()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(
      new CronJob({
        id: 'paused',
        prompt: 'hi',
        schedule: '* * * * *',
        paused: true,
        nextRunAt: '2026-05-15T12:00:00.000Z',
      }),
    )
    const runs: string[] = []
    const scheduler = new CronScheduler(store, (job) => {
      runs.push(job.id)
      return ''
    })

    expect(await scheduler.tick(new Date('2026-05-15T13:00:00.000Z'))).toEqual(
      [],
    )
    expect(runs).toEqual([])
  } finally {
    removeDirectory(directory)
  }
})

test('scheduler removes a one-shot job after it runs', async () => {
  const directory = temporaryDirectory()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(new CronJob({ id: 'once', prompt: 'once', oneshot: true }))
    const scheduler = new CronScheduler(store, () => 'out')
    const now = new Date('2026-05-15T12:00:00.000Z')

    expect(await scheduler.tick(now)).toEqual([])
    expect(await scheduler.tick(new Date('2026-05-15T12:01:00.000Z'))).toEqual([
      'once',
    ])
    expect(store.listJobs()).toEqual([])
  } finally {
    removeDirectory(directory)
  }
})

test('scheduler invokes completion after a job runs', async () => {
  const directory = temporaryDirectory()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(
      new CronJob({ id: 'complete', prompt: 'p', schedule: '* * * * *' }),
    )
    const completions: Array<[string, string]> = []
    const scheduler = new CronScheduler(store, () => 'result text', {
      onComplete: (job, output) => {
        completions.push([job.id, output])
      },
    })
    const now = new Date('2026-05-15T12:00:00.000Z')

    await scheduler.tick(now)
    await scheduler.tick(new Date('2026-05-15T12:02:00.000Z'))
    expect(completions).toEqual([['complete', 'result text']])
  } finally {
    removeDirectory(directory)
  }
})

test('scheduler logs completion failures but still advances the job schedule', async () => {
  const directory = temporaryDirectory()
  const previousError = console.error
  const errors: unknown[][] = []
  console.error = (...args: unknown[]) => {
    errors.push(args)
  }
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(
      new CronJob({
        id: 'delivery-failure',
        prompt: 'p',
        schedule: '* * * * *',
        nextRunAt: '2026-05-15T12:00:00.000Z',
      }),
    )
    const scheduler = new CronScheduler(store, () => 'result text', {
      onComplete: () => {
        throw new Error('delivery unavailable')
      },
    })

    expect(await scheduler.tick(new Date('2026-05-15T12:00:00.000Z'))).toEqual([
      'delivery-failure',
    ])
    expect(store.get('delivery-failure')?.nextRunAt).toBe(
      '2026-05-15T12:01:00.000Z',
    )
    expect(errors[0]?.[0]).toBe(
      'CronScheduler onComplete failed for job delivery-failure',
    )
  } finally {
    console.error = previousError
    removeDirectory(directory)
  }
})

test('scheduler starts with an immediate background tick', async () => {
  const directory = temporaryDirectory()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(
      new CronJob({
        id: 'background',
        prompt: 'p',
        schedule: '* * * * *',
        nextRunAt: new Date(Date.now() - 60_000).toISOString(),
      }),
    )
    let resolveRun: (value: string) => void = () => {}
    const ran = new Promise<string>((resolve) => {
      resolveRun = resolve
    })
    const scheduler = new CronScheduler(
      store,
      (job) => {
        resolveRun(job.id)
        return 'out'
      },
      { pollInterval: 60_000 },
    )

    scheduler.start()
    try {
      expect(
        await Promise.race([ran, Bun.sleep(100).then(() => 'timeout')]),
      ).toBe('background')
    } finally {
      scheduler.stop()
    }
  } finally {
    removeDirectory(directory)
  }
})

test('scheduler contains background runner failures at the polling boundary', async () => {
  const directory = temporaryDirectory()
  const previousError = console.error
  let resolveLog: (value: void) => void = () => {}
  const logged = new Promise<void>((resolve) => {
    resolveLog = resolve
  })
  const errors: unknown[][] = []
  console.error = (...args: unknown[]) => {
    errors.push(args)
    resolveLog()
  }
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(
      new CronJob({
        id: 'background-error',
        prompt: 'p',
        schedule: '* * * * *',
        nextRunAt: new Date(Date.now() - 60_000).toISOString(),
      }),
    )
    const scheduler = new CronScheduler(
      store,
      () => {
        throw new Error('runner unavailable')
      },
      { pollInterval: 60_000 },
    )

    scheduler.start()
    try {
      await Promise.race([logged, Bun.sleep(100)])
      expect(errors[0]?.[0]).toBe('CronScheduler tick failed')
    } finally {
      scheduler.stop()
    }
  } finally {
    console.error = previousError
    removeDirectory(directory)
  }
})
