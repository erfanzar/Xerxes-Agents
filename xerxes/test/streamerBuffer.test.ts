// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { KILL_TAG, StreamerBuffer, StreamerBufferError } from '../src/core/streamerBuffer.js'

test('StreamerBuffer yields queued values in order and drains them before close ends the stream', async () => {
  const buffer = new StreamerBuffer<string>()
  expect(buffer.put(null)).toBe(true)
  expect(buffer.put('first')).toBe(true)
  expect(buffer.put('second')).toBe(true)
  buffer.close()

  const values: string[] = []
  for await (const value of buffer) values.push(value)
  expect(values).toEqual(['first', 'second'])
  expect(buffer.closed).toBe(true)
})

test('StreamerBuffer close releases waiting consumers and drops writes after shutdown', async () => {
  const buffer = new StreamerBuffer<string>()
  const iterator = buffer.stream()
  const pending = iterator.next()
  expect(buffer.close()).toBe(true)
  expect(await pending).toEqual({ done: true, value: undefined })
  expect(buffer.close()).toBe(false)
  expect(buffer.put('dropped')).toBe(false)
  expect(await buffer.push('also dropped')).toBe(false)

  const multiConsumer = new StreamerBuffer<string>()
  const first = multiConsumer.get()
  const second = multiConsumer.get()
  multiConsumer.close()
  expect(await Promise.all([first, second])).toEqual([undefined, undefined])
})

test('StreamerBuffer provides async backpressure for bounded queues', async () => {
  const buffer = new StreamerBuffer<string>({ maxSize: 1 })
  expect(buffer.put('first')).toBe(true)
  expect(buffer.put('would overflow')).toBe(false)

  const waitingPush = buffer.push('second')
  expect(await buffer.get()).toBe('first')
  expect(await waitingPush).toBe(true)
  expect(await buffer.get()).toBe('second')
  buffer.close()
})

test('StreamerBuffer recognizes completion only when streamed and then maybeFinish closes it', async () => {
  type Event = { readonly type: 'completion' | 'text'; readonly value: string }
  const buffer = new StreamerBuffer<Event>({ isCompletion: event => event.type === 'completion' })
  buffer.put({ type: 'completion', value: 'done' })

  const iterator = buffer.stream()
  expect((await iterator.next()).value).toEqual({ type: 'completion', value: 'done' })
  expect(buffer.finishHit).toBe(true)
  buffer.maybeFinish(undefined)
  expect(buffer.closed).toBe(true)
  expect(await iterator.next()).toEqual({ done: true, value: undefined })
})

test('StreamerBuffer supports the legacy kill tag and rejects invalid wait configuration', async () => {
  const buffer = new StreamerBuffer<string>()
  expect(buffer.put('available')).toBe(true)
  expect(buffer.put(KILL_TAG)).toBe(true)
  expect(await buffer.get()).toBe('available')
  expect(await buffer.get({ timeoutMs: 0 })).toBeUndefined()

  const pending = new StreamerBuffer<string>()
  const controller = new AbortController()
  const read = pending.get({ signal: controller.signal })
  controller.abort(new Error('cancelled'))
  await expect(read).rejects.toThrow('cancelled')
  expect(() => new StreamerBuffer({ maxSize: -1 })).toThrow(StreamerBufferError)
  await expect(pending.get({ timeoutMs: -1 })).rejects.toThrow(StreamerBufferError)
})
