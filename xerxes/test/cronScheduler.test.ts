// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  acquireCronLease,
  classifyCronLeaseHolder,
  CronJob,
  CronScheduler,
  JobStore,
  readCronLease,
  releaseCronLease,
} from '../src/cron/index.js'

function temporaryDirectory(): string {
  return mkdtempSync(join(tmpdir(), 'xerxes-cron-lease-'))
}

function removeDirectory(path: string): void {
  rmSync(path, { recursive: true, force: true })
}

const HOLDER = {
  acquiredAt: '2026-05-15T12:00:00.000Z',
  command: 'bun /repo/src/cli.ts daemon',
  ownerKey: '/repo',
  pid: 4242,
} as const

test('lease classification separates our own lease, a live holder, and an abandoned one', () => {
  const alive = () => true
  const sameCommand = () => HOLDER.command

  expect(
    classifyCronLeaseHolder(HOLDER, {
      commandOf: sameCommand,
      isAlive: alive,
      ownerKey: HOLDER.ownerKey,
      pid: HOLDER.pid,
    }),
  ).toBe('owned')

  expect(
    classifyCronLeaseHolder(HOLDER, {
      commandOf: sameCommand,
      isAlive: alive,
      ownerKey: '/other-repo',
      pid: 99,
    }),
  ).toBe('refused')

  expect(
    classifyCronLeaseHolder(HOLDER, {
      commandOf: () => '',
      isAlive: () => false,
      ownerKey: '/other-repo',
      pid: 99,
    }),
  ).toBe('stale')

  // A recycled pid running something else is abandoned, but a pid we simply
  // cannot inspect must never be treated as free.
  expect(
    classifyCronLeaseHolder(HOLDER, {
      commandOf: () => '/usr/bin/unrelated --thing',
      isAlive: alive,
      ownerKey: '/other-repo',
      pid: 99,
    }),
  ).toBe('stale')
  expect(
    classifyCronLeaseHolder(HOLDER, {
      commandOf: () => '',
      isAlive: alive,
      ownerKey: '/other-repo',
      pid: 99,
    }),
  ).toBe('refused')
})

test('lease acquire is exclusive, re-entrant for its owner, and released only by it', () => {
  const directory = temporaryDirectory()
  const path = join(directory, 'cron.lease')
  try {
    const first = acquireCronLease(path, {
      commandOf: () => 'daemon-a',
      isAlive: () => true,
      ownerKey: '/repo-a',
      pid: 101,
    })
    expect(first.state).toBe('acquired')
    expect(first.held).toBe(true)
    expect(readCronLease(path)?.pid).toBe(101)

    const again = acquireCronLease(path, {
      commandOf: () => 'daemon-a',
      isAlive: () => true,
      ownerKey: '/repo-a',
      pid: 101,
    })
    expect(again.state).toBe('owned')
    expect(again.held).toBe(true)

    const other = acquireCronLease(path, {
      commandOf: () => 'daemon-a',
      isAlive: () => true,
      ownerKey: '/repo-b',
      pid: 202,
    })
    expect(other.state).toBe('refused')
    expect(other.held).toBe(false)
    expect(other.holder?.ownerKey).toBe('/repo-a')

    expect(
      releaseCronLease(path, { ownerKey: '/repo-b', pid: 202 }),
    ).toBe(false)
    expect(existsSync(path)).toBe(true)
    expect(
      releaseCronLease(path, { ownerKey: '/repo-a', pid: 101 }),
    ).toBe(true)
    expect(existsSync(path)).toBe(false)
  } finally {
    removeDirectory(directory)
  }
})

test('a lease abandoned by a dead holder is recovered by exactly one of two recoverers', () => {
  const directory = temporaryDirectory()
  const path = join(directory, 'cron.lease')
  try {
    writeFileSync(
      path,
      `${JSON.stringify({
        acquired_at: HOLDER.acquiredAt,
        command: HOLDER.command,
        owner_key: HOLDER.ownerKey,
        pid: HOLDER.pid,
      })}\n`,
      'utf8',
    )

    // The holder's pid was recycled by an unrelated program, which is the only
    // way a claimant can prove the lease was abandoned rather than busy.
    const commands = new Map<number, string>([
      [HOLDER.pid, '/usr/bin/unrelated --recycled'],
      [301, 'daemon-c'],
      [302, 'daemon-d'],
    ])
    const isAlive = (): boolean => true
    const commandOf = (pid: number): string => commands.get(pid) ?? ''

    // The second recoverer classifies the dead holder and then, before it can
    // unlink, the first recoverer wins. Driving that from inside the injected
    // command lookup reproduces the interleaving deterministically.
    let winner: string | undefined
    const second = acquireCronLease(path, {
      commandOf: (pid) => {
        if (pid === HOLDER.pid && !winner) {
          const first = acquireCronLease(path, {
            commandOf,
            isAlive,
            ownerKey: '/repo-c',
            pid: 301,
          })
          winner = first.state
        }
        return commandOf(pid)
      },
      isAlive,
      ownerKey: '/repo-d',
      pid: 302,
    })

    expect(winner).toBe('acquired')
    expect(second.held).toBe(false)
    expect(second.state).toBe('refused')
    expect(readCronLease(path)?.pid).toBe(301)
  } finally {
    removeDirectory(directory)
  }
})

test('scheduler without the lease runs nothing and leaves the store untouched', async () => {
  const directory = temporaryDirectory()
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(
      new CronJob({
        id: 'shared',
        prompt: 'summarize',
        schedule: '* * * * *',
        nextRunAt: '2026-05-15T12:00:00.000Z',
      }),
    )
    const runs: string[] = []
    let leased = false
    const scheduler = new CronScheduler(
      store,
      (job) => {
        runs.push(job.id)
        return 'done'
      },
      { holdsLease: () => leased },
    )

    expect(await scheduler.tick(new Date('2026-05-15T12:01:00.000Z'))).toEqual([])
    expect(runs).toEqual([])
    // The lease holder's fire time must survive an unleased tick untouched.
    expect(store.get('shared')?.nextRunAt).toBe('2026-05-15T12:00:00.000Z')
    expect(store.get('shared')?.lastRunAt).toBeUndefined()

    leased = true
    expect(await scheduler.tick(new Date('2026-05-15T12:01:00.000Z'))).toEqual([
      'shared',
    ])
    expect(runs).toEqual(['shared'])
  } finally {
    removeDirectory(directory)
  }
})

test('a throwing lease predicate skips the tick instead of running the shared store', async () => {
  const directory = temporaryDirectory()
  const previousWarn = console.warn
  console.warn = () => {}
  try {
    const store = new JobStore(join(directory, 'jobs.json'))
    store.add(
      new CronJob({
        id: 'shared',
        prompt: 'summarize',
        schedule: '* * * * *',
        nextRunAt: '2026-05-15T12:00:00.000Z',
      }),
    )
    const runs: string[] = []
    const scheduler = new CronScheduler(
      store,
      (job) => {
        runs.push(job.id)
        return 'done'
      },
      {
        holdsLease: () => {
          throw new Error('lease file unreadable')
        },
      },
    )

    expect(await scheduler.tick(new Date('2026-05-15T12:01:00.000Z'))).toEqual([])
    expect(runs).toEqual([])
  } finally {
    console.warn = previousWarn
    removeDirectory(directory)
  }
})
