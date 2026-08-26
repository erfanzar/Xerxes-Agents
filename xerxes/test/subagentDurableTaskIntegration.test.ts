// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { SubAgentManager } from '../src/agents/subagentManager.js'
import { DurableTaskRuntime } from '../src/tasks/durableTaskRuntime.js'
import { bridgeDurableTaskLifecycle } from '../src/tasks/durableTaskBridge.js'

test('SubAgentManager records task creation and completion through a durable bridge', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-subagent-durable-integration-'))
  try {
    const runtime = new DurableTaskRuntime({ directory })
    const bridge = bridgeDurableTaskLifecycle(runtime)
    const manager = new SubAgentManager({
      durableTaskBridge: bridge,
      runner: async () => 'subagent result',
    })

    const task = await manager.spawn({ prompt: 'do work', creatorId: 'root' })
    await manager.wait(task.id, 1_000)
    expect(task.status).toBe('completed')
    await manager.close()

    const state = await runtime.load()
    expect(state.tasks.get(task.id)).toMatchObject({ status: 'completed', result: 'subagent result' })
    expect(state.attempts.size).toBe(1)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('SubAgentManager records durable setup failure', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-subagent-durable-fail-integration-'))
  try {
    const runtime = new DurableTaskRuntime({ directory })
    const bridge = bridgeDurableTaskLifecycle(runtime)
    const manager = new SubAgentManager({
      durableTaskBridge: bridge,
      runner: async () => 'subagent result',
    })

    const task = await manager.spawn({ prompt: 'work', creatorId: 'root', depth: 0, agentDefinition: {
      allowedTools: null,
      description: '',
      excludeTools: [],
      isolation: '',
      maxDepth: 0,
      model: '',
      name: 'zero-depth',
      source: 'yaml',
      systemPrompt: '',
      tools: [],
    } })
    await manager.wait(task.id, 1_000)
    expect(task.status).toBe('failed')
    await manager.close()

    const state = await runtime.load()
    expect(state.tasks.get(task.id)).toMatchObject({ status: 'failed' })
    expect(state.attempts.size).toBe(1)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})
