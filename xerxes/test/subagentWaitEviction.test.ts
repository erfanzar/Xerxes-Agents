// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { SpawnedAgentManager } from '../src/operators/subagents.js'
import {
  SubAgentManager,
  type SubAgentManagerOptions,
} from '../src/agents/subagentManager.js'

/** Test seam: the concrete native manager owned by a SubAgentManager. */
function handlesOf(manager: SubAgentManager): SpawnedAgentManager {
  return (manager as unknown as { readonly handleManager: SpawnedAgentManager }).handleManager
}

function handleMapOf(manager: SubAgentManager): Map<string, unknown> {
  return (handlesOf(manager) as unknown as { readonly handles: Map<string, unknown> }).handles
}

function delay(milliseconds: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, milliseconds))
}

/** Runner body that settles promptly when cancellation is requested. */
async function cancellableRun(request: { cancelSignal: AbortSignal; prompt: string }): Promise<string> {
  if (request.prompt !== 'block') return `done:${request.prompt}`
  await new Promise<void>(resolve => {
    if (request.cancelSignal.aborted) resolve()
    else request.cancelSignal.addEventListener('abort', () => resolve(), { once: true })
  })
  if (request.cancelSignal.aborted) throw new Error('interrupted by test teardown')
  return 'done:block'
}

test('waits resolve from retained task snapshots when handles were evicted past the native cap', async () => {
  const manager = new SubAgentManager({
    // Tight native bound so old handles are provably gone while every task stays retained.
    maxRetainedTerminalHandles: 1,
    maxRetainedTerminalTasks: 32,
    runner: async request => `done:${request.prompt}`,
  })
  try {
    for (const name of ['w0', 'w1', 'w2', 'w3']) {
      await manager.spawn({ id: name, name, prompt: `task ${name}` })
      await manager.wait(name)
    }
    expect(handlesOf(manager).listHandles()).toHaveLength(1)
    expect(handlesOf(manager).hasHandle('w0')).toBeFalse()
    expect(handlesOf(manager).hasHandle('w3')).toBeTrue()

    // Retained tasks without handles must answer from stored state, never throw.
    const evicted = await manager.wait('w0')
    expect(evicted?.id).toBe('w0')
    expect(evicted?.status).toBe('completed')
    expect(evicted?.result).toBe('done:task w0')
    expect(await manager.wait('w1')).toMatchObject({ id: 'w1', status: 'completed' })

    const batch = await manager.waitAll(['w0', 'w1', 'w3'])
    expect(batch.pending).toEqual([])
    expect(batch.completed.map(snapshot => snapshot.id)).toEqual(['w0', 'w1', 'w3'])
    expect(batch.completed.map(snapshot => snapshot.result)).toEqual([
      'done:task w0',
      'done:task w1',
      'done:task w3',
    ])
  } finally {
    await manager.close()
  }
})

test('a live task whose native handle went missing waits out from its stored snapshot', async () => {
  const manager = new SubAgentManager({
    // Delayed rather than gated so the run settles on its own for clean shutdown.
    runner: async request => {
      await delay(80)
      return `done:${request.prompt}`
    },
  })
  try {
    await manager.spawn({ id: 'orphan', name: 'orphan', prompt: 'work' })
    expect(handlesOf(manager).hasHandle('orphan')).toBeTrue()

    // Simulate the retention-map divergence the auditor could not reproduce
    // naturally: the task is live while its native handle is gone.
    expect(handleMapOf(manager).delete('orphan')).toBeTrue()
    expect(handlesOf(manager).hasHandle('orphan')).toBeFalse()

    const orphan = await manager.wait('orphan', 250)
    expect(orphan).toBeDefined()
    expect(orphan?.status).toBe('running')

    // The batch path answers the same way.
    const batch = await manager.waitAll(['orphan'], 250)
    expect(batch.pending.map(snapshot => snapshot.id)).toEqual(['orphan'])

    await delay(150)
    const settled = await manager.wait('orphan', 1_000)
    expect(settled?.status).toBe('completed')
  } finally {
    await manager.close()
  }
})

test('live tasks with retained handles still block until they settle', async () => {
  let release!: () => void
  const gate = new Promise<void>(resolve => {
    release = resolve
  })
  const manager = new SubAgentManager({
    runner: async request => {
      if (request.prompt !== 'block') return `done:${request.prompt}`
      await new Promise<void>(resolve => {
        if (request.cancelSignal.aborted) resolve()
        else {
          const onAbort = (): void => resolve()
          request.cancelSignal.addEventListener('abort', onAbort, { once: true })
          void gate.then(onAbort)
        }
      })
      if (request.cancelSignal.aborted) throw new Error('interrupted before settling')
      return 'done:block'
    },
  })
  try {
    await manager.spawn({ id: 'busy', name: 'busy', prompt: 'block' })

    let settled = false
    const pending = manager.wait('busy', 2_000).then(task => {
      settled = true
      return task
    })
    await delay(120)
    expect(settled).toBeFalse()
    expect(handlesOf(manager).hasHandle('busy')).toBeTrue()

    release()
    const busy = await pending
    expect(busy?.status).toBe('completed')

    const batch = await manager.waitAll(['busy'])
    expect(batch.completed.map(snapshot => snapshot.id)).toEqual(['busy'])
  } finally {
    await manager.close()
  }
})

test('genuinely unknown ids keep their existing behavior', async () => {
  const manager = new SubAgentManager({ runner: cancellableRun })
  try {
    expect(await manager.wait('ghost')).toBeUndefined()
    expect(manager.findTask('ghost')).toBeUndefined()
    const batch = await manager.waitAll(['ghost'])
    expect(batch.completed).toEqual([])
    expect(batch.pending).toEqual([])

    // The native layer keeps failing closed for ids nothing retains.
    const native = new SpawnedAgentManager({ runner: async () => 'x' })
    expect(native.hasHandle('ghost')).toBeFalse()
    await expect(native.wait(['ghost'], 5)).rejects.toThrow('spawned agent not found')
  } finally {
    await manager.close()
  }
})
