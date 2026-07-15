// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  BackgroundSessionManager,
  BackgroundStatus,
} from '../src/runtime/backgroundSessions.js'

interface Deferred<T> {
  readonly promise: Promise<T>
  resolve(value: T): void
  reject(error: unknown): void
}

function deferred<T>(): Deferred<T> {
  let resolvePromise: ((value: T) => void) | undefined
  let rejectPromise: ((error: unknown) => void) | undefined
  const promise = new Promise<T>((resolve, reject) => {
    resolvePromise = resolve
    rejectPromise = reject
  })
  return {
    promise,
    resolve: value => resolvePromise?.(value),
    reject: error => rejectPromise?.(error),
  }
}

async function tick(): Promise<void> {
  await Promise.resolve()
  await Promise.resolve()
}

test('background sessions detach, preserve injected timestamps, and expose immutable snapshots', async () => {
  let now = 10
  const metadata = { nested: { mode: 'safe' } }
  const manager = new BackgroundSessionManager({
    now: () => now,
    runner: session => 'answered:' + session.prompt,
  })
  const submitted = manager.submit('hello', { sessionId: 'session-a', metadata })
  metadata.nested.mode = 'changed'

  expect(submitted).toMatchObject({
    id: 'session-a',
    prompt: 'hello',
    status: BackgroundStatus.RUNNING,
    startedAt: 10,
    finishedAt: 0,
  })
  expect(Object.isFrozen(submitted)).toBe(true)

  now = 11
  const settled = await manager.wait('session-a', { settled: true })
  expect(settled).toMatchObject({
    status: BackgroundStatus.SUCCEEDED,
    result: 'answered:hello',
    error: '',
    startedAt: 10,
    finishedAt: 11,
    metadata: { nested: { mode: 'safe' } },
  })
  expect(Object.isFrozen(settled?.metadata)).toBe(true)
  expect(manager.listSessions()).toHaveLength(1)
})

test('background sessions capture runner failures and maintain FIFO under the concurrency cap', async () => {
  const gates = new Map<string, Deferred<string>>()
  const started: string[] = []
  const manager = new BackgroundSessionManager({
    maxConcurrent: 1,
    runner: session => {
      started.push(session.prompt)
      const gate = deferred<string>()
      gates.set(session.prompt, gate)
      return gate.promise
    },
  })
  const first = manager.submit('first', { sessionId: 'first' })
  const second = manager.submit('second', { sessionId: 'second' })
  await tick()

  expect(first.status).toBe(BackgroundStatus.RUNNING)
  expect(manager.get(second.id)?.status).toBe(BackgroundStatus.PENDING)
  expect(started).toEqual(['first'])

  gates.get('first')?.resolve('first done')
  expect((await manager.wait(first.id, { settled: true }))?.status).toBe(BackgroundStatus.SUCCEEDED)
  await tick()
  expect(manager.get(second.id)?.status).toBe(BackgroundStatus.RUNNING)
  expect(started).toEqual(['first', 'second'])
  gates.get('second')?.resolve('second done')
  expect((await manager.wait(second.id, { settled: true }))?.result).toBe('second done')

  const failures = new BackgroundSessionManager({
    runner: () => {
      throw new Error('nope')
    },
  })
  const failed = failures.submit('fail', { sessionId: 'failed' })
  const result = await failures.wait(failed.id, { settled: true })
  expect(result).toMatchObject({
    status: BackgroundStatus.FAILED,
    result: '',
    error: 'Error: nope',
  })
})

test('cancellation aborts cooperative runners, retains cancelled state against late success, and skips pending work', async () => {
  const firstGate = deferred<string>()
  let firstSignal: AbortSignal | undefined
  const started: string[] = []
  const manager = new BackgroundSessionManager({
    maxConcurrent: 1,
    runner: (session, signal) => {
      started.push(session.prompt)
      if (session.prompt === 'first') {
        firstSignal = signal
        return firstGate.promise
      }
      return 'should not run'
    },
  })
  const first = manager.submit('first', { sessionId: 'first' })
  const pending = manager.submit('pending', { sessionId: 'pending' })
  await tick()

  expect(manager.cancel(pending.id)).toBe(true)
  expect(manager.get(pending.id)?.status).toBe(BackgroundStatus.CANCELLED)
  expect(manager.cancel(first.id)).toBe(true)
  expect(firstSignal?.aborted).toBe(true)
  expect(manager.get(first.id)?.status).toBe(BackgroundStatus.CANCELLED)

  firstGate.resolve('late success')
  const settled = await manager.wait(first.id, { settled: true })
  expect(settled).toMatchObject({
    status: BackgroundStatus.CANCELLED,
    result: '',
    error: '',
  })
  expect(started).toEqual(['first'])
  expect(manager.cancel(first.id)).toBe(false)
})

test('completed retention prunes only settled terminal sessions and shutdown rejects new submissions', async () => {
  let now = 1
  const manager = new BackgroundSessionManager({
    maxCompleted: 2,
    now: () => now,
    runner: session => session.prompt,
  })

  const first = manager.submit('one', { sessionId: 'one' })
  await manager.wait(first.id, { settled: true })
  now = 2
  const second = manager.submit('two', { sessionId: 'two' })
  await manager.wait(second.id, { settled: true })
  now = 3
  const third = manager.submit('three', { sessionId: 'three' })
  await manager.wait(third.id, { settled: true })

  expect(manager.get('one')).toBeUndefined()
  expect(manager.listSessions().map(session => session.id)).toEqual(['two', 'three'])
  await manager.shutdown()
  expect(() => manager.submit('after shutdown')).toThrow('shutting down')
})
