// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import {
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
} from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  archiveOutput,
  CronJob,
  CronScheduler,
  JobStore,
  nextFireAt,
} from '../src/cron/index.js'

function temporaryDirectory(): string {
  return mkdtempSync(join(tmpdir(), 'xerxes-cron-hardening-'))
}

function removeDirectory(path: string): void {
  rmSync(path, { recursive: true, force: true })
}

function silenceConsole(): { errors: unknown[][]; restore: () => void } {
  const errors: unknown[][] = []
  const previousError = console.error
  const previousWarn = console.warn
  console.error = (...args: unknown[]) => {
    errors.push(args)
  }
  console.warn = () => {}
  return {
    errors,
    restore: () => {
      console.error = previousError
      console.warn = previousWarn
    },
  }
}

test('nextFireAt finds Feb-29 schedules across non-leap years', () => {
  // 2026 and 2027 are not leap years; the next valid fire is 2028-02-29.
  expect(
    nextFireAt('0 9 29 2 *', new Date('2026-06-01T00:00:00.000Z')).toISOString(),
  ).toBe('2028-02-29T09:00:00.000Z')
  expect(
    nextFireAt('0 9 29 2 *', new Date('2027-03-01T00:00:00.000Z')).toISOString(),
  ).toBe('2028-02-29T09:00:00.000Z')
  // Inside a leap year the upcoming Feb-29 is used when it is still ahead.
  expect(
    nextFireAt('0 9 29 2 *', new Date('2028-02-01T00:00:00.000Z')).toISOString(),
  ).toBe('2028-02-29T09:00:00.000Z')
})

test('nextFireAt ORs restricted day-of-month and day-of-week fields', () => {
  // POSIX semantics: with both fields restricted, either match fires.
  // From Friday 2026-05-15, "0 9 1 * 1" fires Monday 2026-05-18, not only
  // when the 1st is a Monday (2026-06-01).
  expect(
    nextFireAt('0 9 1 * 1', new Date('2026-05-15T12:00:00.000Z')).toISOString(),
  ).toBe('2026-05-18T09:00:00.000Z')
  // With only one day field restricted, that field alone gates the match.
  expect(
    nextFireAt('0 9 * * 1', new Date('2026-05-15T12:00:00.000Z')).toISOString(),
  ).toBe('2026-05-18T09:00:00.000Z')
  expect(
    nextFireAt('0 9 1 * *', new Date('2026-05-15T12:00:00.000Z')).toISOString(),
  ).toBe('2026-06-01T09:00:00.000Z')
})

test('nextFireAt accepts 7 as a Sunday day-of-week alias', () => {
  // 2026-05-16 is a Saturday; the next Sunday is 2026-05-17.
  const fromSeven = nextFireAt('0 9 * * 7', new Date('2026-05-16T12:00:00.000Z'))
  expect(fromSeven.toISOString()).toBe('2026-05-17T09:00:00.000Z')
  expect(fromSeven.getUTCDay()).toBe(0)
  // Ranges including 7 map onto Sunday too.
  expect(
    nextFireAt('0 9 * * 5-7', new Date('2026-05-16T12:00:00.000Z')).toISOString(),
  ).toBe('2026-05-17T09:00:00.000Z')
})

test('a failed one-shot is retained, retried, then paused after bounded retries', async () => {
  const directory = temporaryDirectory()
  const { restore } = silenceConsole()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(
      new CronJob({
        id: 'flaky-once',
        prompt: 'p',
        oneshot: true,
        nextRunAt: '2026-05-15T12:00:00.000Z',
      }),
    )
    let failures = 0
    const scheduler = new CronScheduler(
      store,
      () => {
        failures += 1
        throw new Error('boom')
      },
      { maxOneShotRetries: 2, oneShotRetryBaseMs: 1_000 },
    )
    const now = new Date('2026-05-15T12:00:00.000Z')

    // First failure: retained with retry_count 1 and a 1s backoff.
    expect(await scheduler.tick(now)).toEqual([])
    let job = store.get('flaky-once')
    expect(job).toBeDefined()
    expect(job?.paused).toBe(false)
    expect(job?.nextRunAt).toBe('2026-05-15T12:00:01.000Z')
    expect(job?.metadata).toMatchObject({
      last_error: 'boom',
      retry_count: 1,
    })

    // Second failure (retry 2): backoff doubles to 2s.
    expect(
      await scheduler.tick(new Date('2026-05-15T12:00:01.500Z')),
    ).toEqual([])
    job = store.get('flaky-once')
    expect(job?.nextRunAt).toBe('2026-05-15T12:00:03.000Z')
    expect(job?.metadata.retry_count).toBe(2)

    // Third failure exceeds the retry budget: the job is paused, not deleted.
    expect(
      await scheduler.tick(new Date('2026-05-15T12:00:04.000Z')),
    ).toEqual([])
    job = store.get('flaky-once')
    expect(job).toBeDefined()
    expect(job?.paused).toBe(true)
    expect(failures).toBe(3)
  } finally {
    restore()
    removeDirectory(directory)
  }
})

test('a retried one-shot that eventually succeeds is removed', async () => {
  const directory = temporaryDirectory()
  const { restore } = silenceConsole()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(
      new CronJob({
        id: 'recovers',
        prompt: 'p',
        oneshot: true,
        nextRunAt: '2026-05-15T12:00:00.000Z',
      }),
    )
    let attempts = 0
    const scheduler = new CronScheduler(
      store,
      () => {
        attempts += 1
        if (attempts === 1) throw new Error('transient')
        return 'done'
      },
      { oneShotRetryBaseMs: 1_000 },
    )

    expect(await scheduler.tick(new Date('2026-05-15T12:00:00.000Z'))).toEqual([])
    expect(store.get('recovers')).toBeDefined()
    expect(
      await scheduler.tick(new Date('2026-05-15T12:00:01.500Z')),
    ).toEqual(['recovers'])
    expect(store.get('recovers')).toBeUndefined()
  } finally {
    restore()
    removeDirectory(directory)
  }
})

test('a recurring job without a schedule is paused instead of hot-looping', async () => {
  const directory = temporaryDirectory()
  const { restore } = silenceConsole()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(
      new CronJob({
        id: 'scheduleless',
        prompt: 'p',
        schedule: '',
        nextRunAt: '2026-05-15T12:00:00.000Z',
      }),
    )
    const runs: string[] = []
    const scheduler = new CronScheduler(store, (job) => {
      runs.push(job.id)
      return ''
    })

    expect(await scheduler.tick(new Date('2026-05-15T12:00:00.000Z'))).toEqual([])
    const job = store.get('scheduleless')
    expect(job?.paused).toBe(true)
    expect(job?.nextRunAt).toBeUndefined()
    // Later polls stay inert: no run, and no repeated rescheduling work.
    expect(await scheduler.tick(new Date('2026-05-15T12:01:00.000Z'))).toEqual([])
    expect(runs).toEqual([])
  } finally {
    restore()
    removeDirectory(directory)
  }
})

test('a hung job times out without starving other due jobs or future ticks', async () => {
  const directory = temporaryDirectory()
  const { errors, restore } = silenceConsole()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(
      new CronJob({
        id: 'hung',
        prompt: 'p',
        schedule: '* * * * *',
        nextRunAt: '2026-05-15T12:00:00.000Z',
      }),
    )
    store.add(
      new CronJob({
        id: 'quick',
        prompt: 'p',
        schedule: '* * * * *',
        nextRunAt: '2026-05-15T12:00:00.000Z',
      }),
    )
    const scheduler = new CronScheduler(
      store,
      (job) =>
        job.id === 'hung'
          ? new Promise<string>(() => {})
          : `ran:${job.id}`,
      { jobTimeout: 25 },
    )
    const now = new Date('2026-05-15T12:00:00.000Z')

    const ran = await scheduler.tick(now)
    expect(ran).toEqual(['quick'])
    // The hung job was reported and rescheduled instead of wedging the queue.
    expect(errors.some((entry) => entry[0] === 'CronScheduler job hung failed')).toBe(true)
    expect(store.get('hung')?.nextRunAt).toBe('2026-05-15T12:01:00.000Z')
    expect(store.get('hung')?.metadata.last_error).toContain('timed out')
    // A subsequent tick is not blocked by the previous hung run.
    expect(
      await scheduler.tick(new Date('2026-05-15T12:01:00.000Z')),
    ).toEqual(['quick'])
  } finally {
    restore()
    removeDirectory(directory)
  }
})

test('archiveOutput never overwrites same-second runs and prunes to the retention cap', async () => {
  const directory = temporaryDirectory()
  try {
    const instant = new Date('2026-05-15T12:00:00.000Z')
    const first = await archiveOutput(directory, 'job1', 'first', instant)
    const second = await archiveOutput(directory, 'job1', 'second', instant)
    expect(first).not.toBe(second)
    expect(readFileSync(first, 'utf8')).toBe('first')
    expect(readFileSync(second, 'utf8')).toBe('second')

    for (let index = 0; index < 12; index += 1) {
      await archiveOutput(
        directory,
        'job1',
        `extra:${index}`,
        new Date(instant.getTime() + (index + 1) * 1_000),
        { retention: 10 },
      )
    }
    const remaining = readdirSync(join(directory, 'job1')).sort()
    expect(remaining).toHaveLength(10)
    // The oldest archives were pruned; the newest are retained.
    const contents = remaining.map((name) =>
      readFileSync(join(directory, 'job1', name), 'utf8'),
    )
    expect(contents).toContain('extra:11')
    expect(contents).not.toContain('first')
    expect(contents).not.toContain('second')
  } finally {
    removeDirectory(directory)
  }
})

test('job store saves atomically without leaving temporary files behind', () => {
  const directory = temporaryDirectory()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(new CronJob({ id: 'atomic', prompt: 'p' }))
    store.update('atomic', { paused: true })
    store.remove('atomic')
    store.add(new CronJob({ id: 'kept', prompt: 'p' }))

    expect(readdirSync(directory)).toEqual(['jobs.json'])
    const persisted = JSON.parse(
      readFileSync(join(directory, 'jobs.json'), 'utf8'),
    ) as Array<Record<string, unknown>>
    expect(persisted).toHaveLength(1)
    expect(persisted[0]?.id).toBe('kept')
    expect(store.get('kept')?.prompt).toBe('p')
  } finally {
    removeDirectory(directory)
  }
})
