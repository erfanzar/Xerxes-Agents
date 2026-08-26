// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { CortexOrchestrator } from '../src/cortex/orchestrator.js'
import { DurableTaskRuntime } from '../src/tasks/durableTaskRuntime.js'
import { bridgeDurableTaskLifecycle } from '../src/tasks/durableTaskBridge.js'

test('Cortex orchestrator records task execution lifecycle through a durable bridge', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-cortex-durable-'))
  try {
    const runtime = new DurableTaskRuntime({ directory })
    const bridge = bridgeDurableTaskLifecycle(runtime)
    const orchestrator = new CortexOrchestrator({
      durableTaskBridge: bridge,
      executor: async ({ task }) => `output for ${task.id}`,
      tasks: [
        { id: 'a', description: 'first', expectedOutput: 'ok', dependencies: [] },
        { id: 'b', description: 'second', expectedOutput: 'ok', dependencies: ['a'] },
      ],
    })

    const result = await orchestrator.run()
    expect(result.status).toBe('succeeded')

    const state = await runtime.load()
    expect(state.tasks.get('a')).toMatchObject({ status: 'completed', result: 'output for a' })
    expect(state.tasks.get('b')).toMatchObject({ status: 'completed', result: 'output for b' })
    expect(state.attempts.size).toBe(2)
    expect(state.lastSequence).toBe(6)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('Cortex orchestrator records failed durable attempts non-retryably', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-cortex-durable-fail-'))
  try {
    const runtime = new DurableTaskRuntime({ directory })
    const bridge = bridgeDurableTaskLifecycle(runtime)
    const orchestrator = new CortexOrchestrator({
      durableTaskBridge: bridge,
      executor: async () => { throw new Error('boom') },
      tasks: [{ id: 'a', description: 'fails', expectedOutput: 'ok', dependencies: [] }],
    })

    const result = await orchestrator.run()
    expect(result.status).toBe('failed')

    const state = await runtime.load()
    expect(state.tasks.get('a')).toMatchObject({ status: 'failed' })
    expect(state.attempts.get(String([...state.attempts.keys()][0]))).toMatchObject({ status: 'failed', retryable: false })
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
