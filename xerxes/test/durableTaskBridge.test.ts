// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { DurableTaskRuntime } from '../src/tasks/durableTaskRuntime.js'
import { bridgeDurableTaskLifecycle } from '../src/tasks/durableTaskBridge.js'

test('durable bridge records task creation, attempts, completion, failure, and cancellation', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-durable-bridge-'))
  try {
    const runtime = new DurableTaskRuntime({ directory })
    const bridge = bridgeDurableTaskLifecycle(runtime)

    await bridge.taskCreated({
      id: 'task-1',
      objective: 'inspect code',
      creatorId: 'root',
      dependencies: [],
      parentId: 'session-1',
    })
    const attempt = await bridge.attemptStarted({
      id: 'attempt-1', taskId: 'task-1', executorId: 'agent-a',
      leaseId: 'lease-1', leaseExpiresAt: Date.now() + 60_000,
    })
    await bridge.attemptCompleted(attempt.id, { output: 'done', deliveryId: 'delivery-1' })

    const state = await runtime.load()
    expect(state.tasks.get('task-1')).toMatchObject({ status: 'completed', parentId: 'session-1', result: 'done' })
    expect(state.attempts.get('attempt-1')).toMatchObject({ status: 'completed', deliveryId: 'delivery-1' })
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('durable bridge retryable failures leave task pending for retry', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-durable-bridge-retry-'))
  try {
    const runtime = new DurableTaskRuntime({ directory })
    const bridge = bridgeDurableTaskLifecycle(runtime)

    await bridge.taskCreated({ id: 'task-1', objective: 'fix bug', creatorId: 'root', dependencies: [] })
    await bridge.attemptStarted({ id: 'attempt-1', taskId: 'task-1', executorId: 'agent-a' })
    await bridge.attemptFailed('attempt-1', { error: 'provider error', retryable: true })

    const state = await runtime.load()
    expect(state.tasks.get('task-1')).toMatchObject({ status: 'pending' })
    expect(state.attempts.get('attempt-1')).toMatchObject({ status: 'failed', retryable: true })
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('durable bridge cancellation records durable terminal state', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-durable-bridge-cancel-'))
  try {
    const runtime = new DurableTaskRuntime({ directory })
    const bridge = bridgeDurableTaskLifecycle(runtime)

    await bridge.taskCreated({ id: 'task-1', objective: 'long task', creatorId: 'root', dependencies: [] })
    await bridge.attemptStarted({ id: 'attempt-1', taskId: 'task-1', executorId: 'agent-a' })
    await bridge.taskCancelled('task-1', 'user cancelled')

    const state = await runtime.load()
    expect(state.tasks.get('task-1')).toMatchObject({ status: 'cancelled', error: 'user cancelled' })
    expect(state.attempts.get('attempt-1')).toMatchObject({ status: 'cancelled', error: 'user cancelled' })
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
