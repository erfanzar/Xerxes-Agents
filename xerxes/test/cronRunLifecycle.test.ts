// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, spyOn, test } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { CronJob, CronScheduler, JobStore } from '../src/cron/index.js'

function withStore(run: (store: JobStore) => Promise<void>): Promise<void> {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-cron-lifecycle-'))
  const quiet = spyOn(console, 'error').mockImplementation(() => {})
  return run(new JobStore(join(directory, 'jobs.json'))).finally(() => {
    quiet.mockRestore()
    rmSync(directory, { recursive: true, force: true })
  })
}

test('a long-running job does not stop later ticks from firing other due jobs', () => withStore(async store => {
  store.add(new CronJob({ id: 'slow', prompt: 'p', oneshot: true, projectRoot: '/a', nextRunAt: '2026-05-15T12:00:00Z' }))
  const slow = Promise.withResolvers<string>()
  const fired: string[] = []
  const scheduler = new CronScheduler(store, job => { fired.push(job.id); return job.id === 'slow' ? slow.promise : 'fast' }, { jobTimeout: 0 })
  const first = scheduler.tick(new Date('2026-05-15T12:00:00Z'))
  await Bun.sleep(0)
  expect(fired).toEqual(['slow'])
  // A one-shot 'skip' job becomes due while slow still runs. The next poll
  // must run it inside its grace instead of waiting for slow to finish.
  store.add(new CronJob({ id: 'fast', prompt: 'p', oneshot: true, projectRoot: '/b', missedRunPolicy: 'skip', misfireGraceSeconds: 1, nextRunAt: '2026-05-15T12:00:30Z' }))
  expect(await scheduler.tick(new Date('2026-05-15T12:00:30Z'))).toEqual(['fast'])
  expect(store.get('fast')).toBeUndefined()
  // The running job is not fired a second time by the overlapping tick.
  expect(fired).toEqual(['slow', 'fast'])
  slow.resolve('done')
  expect(await first).toEqual(['slow'])
}))

test('time a runner spends paused waiting for admission does not count toward the job timeout', () => withStore(async store => {
  store.add(new CronJob({ id: 'followup', prompt: 'p', oneshot: true, maxRuns: 3, nextRunAt: '2026-05-15T12:00:00Z' }))
  let worked = false
  const scheduler = new CronScheduler(store, async (_job, signal, clock) => {
    clock.pause()
    await Bun.sleep(150) // queued behind the conversation's own turn
    clock.resume()
    signal.throwIfAborted()
    await Bun.sleep(20)
    signal.throwIfAborted()
    worked = true
    return 'ran'
  }, { jobTimeout: 60 })
  expect(await scheduler.tick(new Date('2026-05-15T12:00:00Z'))).toEqual(['followup'])
  expect(worked).toBe(true)
  expect(store.get('followup')).toBeUndefined()
}))

test('the timeout still applies once the paused runner is admitted', () => withStore(async store => {
  store.add(new CronJob({ id: 'hung', prompt: 'p', intervalSeconds: 60, nextRunAt: '2026-05-15T12:00:00Z' }))
  const scheduler = new CronScheduler(store, async (_job, signal, clock) => {
    clock.pause()
    await Bun.sleep(80)
    clock.resume()
    await new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(signal.reason), { once: true }))
    return 'never'
  }, { jobTimeout: 40 })
  expect(await scheduler.tick(new Date('2026-05-15T12:00:00Z'))).toEqual([])
  expect(store.get('hung')?.metadata.last_error).toContain('timed out after 40ms')
  await scheduler.waitForIdle()
}))

test('cancelling a running one-shot pauses it instead of scheduling a retry', () => withStore(async store => {
  store.add(new CronJob({ id: 'once', prompt: 'send the report', oneshot: true, nextRunAt: '2026-05-15T12:00:00Z' }))
  const started = Promise.withResolvers<void>()
  let runs = 0
  const scheduler = new CronScheduler(store, (_job, signal) => new Promise<string>((_resolve, reject) => {
    runs += 1
    signal.addEventListener('abort', () => reject(signal.reason), { once: true })
    started.resolve()
  }), { jobTimeout: 0 })
  const tick = scheduler.tick(new Date('2026-05-15T12:00:00Z'))
  await started.promise
  expect(scheduler.cancel('once')).toBe(true)
  expect(await tick).toEqual([])
  const job = store.get('once')!
  expect(job.paused).toBe(true)
  expect(job.nextRunAt).toBe('2026-05-15T12:00:00Z')
  expect(job.metadata.retry_count).toBeUndefined()
  expect(job.metadata.execution_receipt).toBeUndefined()
  expect(job.metadata.last_cancelled_at).toBe('2026-05-15T12:00:00.000Z')
  expect(await scheduler.tick(new Date('2026-05-15T12:05:00Z'))).toEqual([])
  expect(runs).toBe(1)
}))

test('cancelling a recurring job keeps its cadence without recording a failure', () => withStore(async store => {
  store.add(new CronJob({ id: 'every', prompt: 'p', intervalSeconds: 60, nextRunAt: '2026-05-15T12:00:00Z' }))
  const started = Promise.withResolvers<void>()
  const scheduler = new CronScheduler(store, (_job, signal) => new Promise<string>((_resolve, reject) => {
    signal.addEventListener('abort', () => reject(signal.reason), { once: true })
    started.resolve()
  }), { jobTimeout: 0 })
  const tick = scheduler.tick(new Date('2026-05-15T12:00:00Z'))
  await started.promise
  scheduler.cancel('every')
  await tick
  const job = store.get('every')!
  expect(job.paused).toBe(false)
  expect(job.nextRunAt).toBe('2026-05-15T12:01:00.000Z')
  expect(job.metadata.last_error).toBeUndefined()
  expect(job.metadata.last_cancelled_at).toBe('2026-05-15T12:00:00.000Z')
}))

test('stopping the scheduler mid-run sends a one-shot to review instead of replaying it', () => withStore(async store => {
  store.add(new CronJob({ id: 'once', prompt: 'p', oneshot: true, nextRunAt: '2026-05-15T12:00:00Z' }))
  const started = Promise.withResolvers<void>()
  const scheduler = new CronScheduler(store, (_job, signal) => new Promise<string>((_resolve, reject) => {
    signal.addEventListener('abort', () => reject(signal.reason), { once: true })
    started.resolve()
  }), { jobTimeout: 0 })
  const tick = scheduler.tick(new Date('2026-05-15T12:00:00Z'))
  await started.promise
  scheduler.stop()
  await tick
  const job = store.get('once')!
  expect(job.paused).toBe(true)
  expect(job.metadata.execution_recovery_required).toBe(true)
  expect(job.metadata.retry_count).toBeUndefined()
}))
