// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'

import { ModelAvailability, ModelUnavailableError } from '../src/hub/supervisor/modelAvailability.js'
import type { SupervisorModelPort } from '../src/hub/supervisor/runner.js'
import { disposeAll } from './hubHarness.js'
import { makeRunner, makeWorld } from './hubSupervisorHarness.js'

afterEach(disposeAll)

describe('a turn whose model is unavailable waits instead of failing', () => {
  test('one notice, then the same turn completes when the model is back', async () => {
    const world = makeWorld()
    world.threads.access('shared', world.user('alice'))
    world.threads.append({ threadId: 'shared', role: 'user', text: 'status?', author: { id: 'alice', name: 'ALICE' } })
    let online = false
    let attempts = 0
    const model: SupervisorModelPort = {
      id: 'm', reportsCost: false,
      async complete() {
        attempts += 1
        if (!online) throw new ModelUnavailableError('Mac is offline and no other online machine offers p/m.')
        return { text: 'All good.', toolCalls: [] }
      },
    }
    const availability = new ModelAvailability({ retryMs: 60_000 })
    const pending = makeRunner(world, model, { availability }).enqueue({ threadId: 'shared', requester: world.user('alice') })
    await Bun.sleep(10)
    availability.notify() // still offline: a second attempt, no second notice
    await Bun.sleep(10)
    online = true
    availability.notify()
    const result = await pending
    expect(result.stop).toBe('completed')
    expect(result.reply?.text).toBe('All good.')
    expect(attempts).toBe(3)
    const notices = world.threads.messagesFor('shared', 'alice').filter(m => m.role === 'system' && m.text.startsWith('XERXES is waiting for its model'))
    expect(notices).toHaveLength(1)
    expect(world.threads.messagesFor('shared', 'alice').some(m => m.text.startsWith('XERXES stopped'))).toBe(false)
  })

  test('the owner can stop a waiting turn', async () => {
    const world = makeWorld()
    world.threads.access('shared', world.user('alice'))
    world.threads.append({ threadId: 'shared', role: 'user', text: 'status?', author: { id: 'alice', name: 'ALICE' } })
    const model: SupervisorModelPort = { id: 'm', reportsCost: false, complete: async () => { throw new ModelUnavailableError('offline') } }
    const runner = makeRunner(world, model, { availability: new ModelAvailability({ retryMs: 60_000 }) })
    const pending = runner.enqueue({ threadId: 'shared', requester: world.user('alice') })
    await Bun.sleep(10)
    expect(runner.cancel('shared')).toBe(true)
    expect((await pending).stop).toBe('cancelled')
  })

  test('without an availability port the old model_error stop is kept', async () => {
    const world = makeWorld()
    world.threads.access('shared', world.user('alice'))
    world.threads.append({ threadId: 'shared', role: 'user', text: 'status?', author: { id: 'alice', name: 'ALICE' } })
    const model: SupervisorModelPort = { id: 'm', reportsCost: false, complete: async () => { throw new ModelUnavailableError('offline') } }
    expect((await makeRunner(world, model).enqueue({ threadId: 'shared', requester: world.user('alice') })).stop).toBe('model_error')
  })
})

describe('XERXES model availability wait', () => {
  test('notify() wakes every waiting turn at once', async () => {
    const availability = new ModelAvailability({ retryMs: 60_000 })
    const woke: number[] = []
    const a = availability.wait().then(() => woke.push(1))
    const b = availability.wait().then(() => woke.push(2))
    expect(availability.waiting).toBe(2)
    availability.notify()
    await Promise.all([a, b])
    expect(woke.sort()).toEqual([1, 2])
    expect(availability.waiting).toBe(0)
  })

  test('the retry interval wakes a turn even without a health change', async () => {
    const availability = new ModelAvailability({ retryMs: 10 })
    await availability.wait()
    expect(availability.waiting).toBe(0)
  })

  test('an owner stop ends the wait with its reason and leaves nothing behind', async () => {
    const availability = new ModelAvailability({ retryMs: 60_000 })
    const controller = new AbortController()
    const pending = availability.wait(controller.signal)
    controller.abort(new Error('owner stopped'))
    await expect(pending).rejects.toThrow('owner stopped')
    expect(availability.waiting).toBe(0)
    await expect(availability.wait(controller.signal)).rejects.toThrow('owner stopped')
  })

  test('the error type is distinguishable from a real model failure', () => {
    const error = new ModelUnavailableError('Mac is offline')
    expect(error).toBeInstanceOf(Error)
    expect(error.name).toBe('ModelUnavailableError')
  })
})
