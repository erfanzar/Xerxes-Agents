// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { DurableTaskRuntime, type DurableTaskEvent } from '../src/tasks/durableTaskRuntime.js'

test('durable tasks, attempts, and cohorts recover from their append-only event log', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-durable-tasks-'))
  try {
    const runtime = new DurableTaskRuntime({ directory })
    await runtime.createTask({ id: 'task-a', objective: 'inspect code', creatorId: 'root', dependencies: [] })
    await runtime.createTask({ id: 'task-b', objective: 'fix code', creatorId: 'root', dependencies: ['task-a'] })
    await runtime.createCohort({ id: 'cohort-1', taskIds: ['task-a', 'task-b'] })
    const attempt = await runtime.startAttempt({ id: 'attempt-a1', taskId: 'task-a', executorId: 'agent-a' })
    await runtime.completeAttempt(attempt.id, { output: 'inspection complete', deliveryId: 'delivery-1' })

    const recovered = await new DurableTaskRuntime({ directory }).load()
    expect(recovered.tasks.get('task-a')).toMatchObject({ status: 'completed', result: 'inspection complete' })
    expect(recovered.tasks.get('task-b')).toMatchObject({ status: 'pending', dependencies: ['task-a'] })
    expect(recovered.attempts.get('attempt-a1')).toMatchObject({ status: 'completed', output: 'inspection complete' })
    expect(recovered.cohorts.get('cohort-1')?.taskIds).toEqual(['task-a', 'task-b'])
    expect(recovered.lastSequence).toBe(5)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('task projection rejects invalid transitions, missing dependencies, and sequence corruption', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-durable-task-invalid-'))
  try {
    const runtime = new DurableTaskRuntime({ directory })
    await expect(runtime.createTask({ id: 'task-b', objective: 'bad', creatorId: 'root', dependencies: ['missing'] }))
      .rejects.toThrow('unknown dependency missing')
    await runtime.createTask({ id: 'task-a', objective: 'work', creatorId: 'root', dependencies: [] })
    await expect(runtime.completeAttempt('missing', { output: 'fabricated', deliveryId: 'delivery-missing' })).rejects.toThrow('unknown attempt missing')

    const forged: DurableTaskEvent = {
      eventId: 'forged', sequence: 3, type: 'task_created',
      task: { id: 'task-c', objective: 'gap', creatorId: 'root', dependencies: [] },
    }
    await Bun.write(runtime.eventLogPath, `${JSON.stringify(forged)}\n`)
    await expect(new DurableTaskRuntime({ directory }).load()).rejects.toThrow('task event sequence gap')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('attempt retries, leases, hierarchy, and result delivery remain durable and exactly once', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-durable-task-retry-'))
  try {
    let now = 1_000
    const runtime = new DurableTaskRuntime({ directory, now: () => now })
    await runtime.createTask({ id: 'parent', objective: 'parent', creatorId: 'root', dependencies: [] })
    await runtime.createTask({
      id: 'child', objective: 'child', creatorId: 'parent-agent', dependencies: [], parentId: 'parent',
    })
    await runtime.startAttempt({
      id: 'attempt-1', taskId: 'child', executorId: 'agent-a', leaseId: 'lease-1', leaseExpiresAt: 1_100,
    })
    await runtime.failAttempt('attempt-1', { error: 'provider failed', retryable: true })
    now = 1_050
    await runtime.startAttempt({
      id: 'attempt-2', taskId: 'child', executorId: 'agent-b', leaseId: 'lease-2', leaseExpiresAt: 1_200,
    })
    await runtime.completeAttempt('attempt-2', { output: 'done', deliveryId: 'delivery-1' })
    await runtime.acknowledgeResult('delivery-1')
    await runtime.acknowledgeResult('delivery-1')

    const state = await new DurableTaskRuntime({ directory }).load()
    expect(state.tasks.get('child')).toMatchObject({ parentId: 'parent', status: 'completed', result: 'done' })
    expect(state.attempts.get('attempt-1')).toMatchObject({ status: 'failed', retryable: true })
    expect(state.attempts.get('attempt-2')).toMatchObject({ status: 'completed', deliveryId: 'delivery-1' })
    expect(state.deliveries.get('delivery-1')).toEqual({ acknowledged: true, attemptId: 'attempt-2', output: 'done' })
    expect(state.lastSequence).toBe(7)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('expired leases cannot complete and active task leases exclude concurrent attempts', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-durable-task-lease-'))
  try {
    let now = 10
    const runtime = new DurableTaskRuntime({ directory, now: () => now })
    await runtime.createTask({ id: 'task-a', objective: 'work', creatorId: 'root', dependencies: [] })
    await runtime.startAttempt({
      id: 'attempt-1', taskId: 'task-a', executorId: 'agent-a', leaseId: 'lease-1', leaseExpiresAt: 20,
    })
    await expect(runtime.startAttempt({
      id: 'attempt-2', taskId: 'task-a', executorId: 'agent-b', leaseId: 'lease-2', leaseExpiresAt: 30,
    })).rejects.toThrow('active lease lease-1')
    now = 21
    await expect(runtime.completeAttempt('attempt-1', { output: 'late', deliveryId: 'delivery-late' }))
      .rejects.toThrow('lease lease-1 expired')
    await runtime.startAttempt({
      id: 'attempt-2', taskId: 'task-a', executorId: 'agent-b', leaseId: 'lease-2', leaseExpiresAt: 30,
    })
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('cancelling a task cancels its running attempt and remains terminal after restart', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-durable-task-cancel-'))
  try {
    const runtime = new DurableTaskRuntime({ directory })
    await runtime.createTask({ id: 'task-a', objective: 'work', creatorId: 'root', dependencies: [] })
    await runtime.startAttempt({ id: 'attempt-a1', taskId: 'task-a', executorId: 'agent-a' })
    await runtime.cancelTask('task-a', 'parent cancelled')

    const state = await new DurableTaskRuntime({ directory }).load()
    expect(state.tasks.get('task-a')).toMatchObject({ status: 'cancelled', error: 'parent cancelled' })
    expect(state.attempts.get('attempt-a1')).toMatchObject({ status: 'cancelled', error: 'parent cancelled' })
    await expect(runtime.startAttempt({ id: 'attempt-a2', taskId: 'task-a', executorId: 'agent-a' }))
      .rejects.toThrow('task task-a is terminal')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('a torn final record does not brick the durable log', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-durable-torn-'))
  try {
    const runtime = new DurableTaskRuntime({ directory })
    await runtime.createTask({ id: 't1', objective: 'first', creatorId: 'root', dependencies: [] })
    await runtime.createTask({ id: 't2', objective: 'second', creatorId: 'root', dependencies: [] })

    // Simulate a crash midway through an append: the final line is partial.
    const intact = await Bun.file(runtime.eventLogPath).text()
    await Bun.write(runtime.eventLogPath, `${intact}{"type":"task_created","task":{"id":"t3"`)

    // Every read AND every write used to throw from here on, permanently —
    // in the module whose entire purpose is surviving a crash.
    const state = await runtime.load()
    expect([...state.tasks.keys()]).toEqual(['t1', 't2'])

    // And the log is still writable, so the runtime recovers rather than
    // needing the file deleted by hand.
    await runtime.createTask({ id: 't4', objective: 'after recovery', creatorId: 'root', dependencies: [] })
    expect([...(await runtime.load()).tasks.keys()]).toContain('t4')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
