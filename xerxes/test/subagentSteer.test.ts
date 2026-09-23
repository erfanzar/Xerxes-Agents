// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Steering a running child.
 *
 * The mechanism is deliberately the same one main sessions use: text is
 * queued and the streaming loop drains it via `drainSteer` at each
 * provider/tool boundary. These tests pin the queue's behaviour at the
 * manager, because every case where it silently does nothing is a case
 * where the desktop would otherwise tell someone their redirect landed.
 */

import { expect, test } from 'bun:test'

import { SubAgentManager, type SubagentTaskRunRequest } from '../src/agents/subagentManager.js'

/** A manager whose runner blocks until released, so a task stays 'running'. */
function runningManager() {
  let drain: (() => readonly string[]) | undefined
  let release: (() => void) | undefined
  const started = new Promise<void>(resolve => {
    release = resolve
  })
  const finished: Promise<void>[] = []
  const manager = new SubAgentManager({
    idFactory: (() => {
      let n = 0
      return () => `task_${(n += 1)}`
    })(),
    pathResolver: (path: string) => path,
    runner: async (request: SubagentTaskRunRequest) => {
      // Capture the drain the manager handed us; this is exactly what the
      // real runner forwards to `runTurn`'s dependencies.
      drain = request.drainSteer
      await started
      return 'done'
    },
  })
  return { manager, drainOf: () => drain, release: () => release?.(), finished }
}

test('a message queued for a running child is drained exactly once', async () => {
  const { manager, drainOf, release } = runningManager()
  const handle = await manager.spawn({ prompt: 'work on it' })
  // Let the runner start so the task is actually running.
  await Promise.resolve()
  await new Promise(resolve => setTimeout(resolve, 10))

  expect(manager.steer(handle.id, 'actually use the other branch')).toBe(true)

  const drain = drainOf()
  expect(drain).toBeDefined()
  expect(drain!()).toEqual(['actually use the other branch'])
  // Destructive: a second boundary must not replay the same instruction.
  expect(drain!()).toEqual([])
  release()
})

test('steering a child that is not running reports failure', async () => {
  const { manager } = runningManager()
  // Never spawned — the desktop can reach this by sending to a child that
  // finished between the roster rendering and the user pressing send.
  expect(manager.steer('task_does_not_exist', 'too late')).toBe(false)
})

test('empty text is refused rather than queued', async () => {
  const { manager, release } = runningManager()
  const handle = await manager.spawn({ prompt: 'work' })
  await new Promise(resolve => setTimeout(resolve, 10))
  expect(manager.steer(handle.id, '   ')).toBe(false)
  expect(manager.steer(handle.id, '')).toBe(false)
  release()
})

test('the queue is capped so a burst cannot all land at one boundary', async () => {
  const { manager, drainOf, release } = runningManager()
  const handle = await manager.spawn({ prompt: 'work' })
  await new Promise(resolve => setTimeout(resolve, 10))

  const accepted: boolean[] = []
  for (let n = 0; n < 12; n += 1) accepted.push(manager.steer(handle.id, `note ${n}`))

  // Eight through, the rest refused — and refused loudly, so the UI can say
  // so instead of dropping them on the floor.
  expect(accepted.filter(Boolean)).toHaveLength(8)
  expect(accepted.slice(8).every(value => value === false)).toBe(true)
  expect(drainOf()!()).toHaveLength(8)
  release()
})
