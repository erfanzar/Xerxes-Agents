// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import {
  InMemoryRLBackend,
  RL_TOOL_DEFINITIONS,
  registerRlTools,
} from '../src/tools/rlTools.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

test('InMemoryRLBackend keeps deterministic run state and never represents inference as a real model', () => {
  const backend = new InMemoryRLBackend()
  const registered = backend.register(
    'cartpole',
    { nested: { threshold: 200 }, seed: 7 },
    'Balance a pole on a cart.',
  )
  const mutableRegisteredConfig = registered.config as JsonObject
  mutableRegisteredConfig.seed = 99

  expect(backend.listEnvironments()).toEqual([{
    config: { nested: { threshold: 200 }, seed: 7 },
    description: 'Balance a pole on a cart.',
    name: 'cartpole',
  }])
  expect(backend.selectEnvironment('cartpole')).toEqual({
    config: { nested: { threshold: 200 }, seed: 7 },
    description: 'Balance a pole on a cart.',
    name: 'cartpole',
  })
  expect(backend.editConfig({ learning_rate: 0.001 })).toEqual({
    config: { learning_rate: 0.001, nested: { threshold: 200 }, seed: 7 },
    environment: 'cartpole',
  })

  const firstRun = backend.start()
  expect(firstRun).toMatchObject({
    config: { learning_rate: 0.001, nested: { threshold: 200 }, seed: 7 },
    endedAt: 0,
    environment: 'cartpole',
    metrics: { reward: 0, step: 0 },
    runId: 'run-000001',
    startedAt: 0,
    status: 'running',
  })
  expect(backend.results(firstRun.runId)).toEqual({
    partial_metrics: { reward: 0, step: 0 },
    status: 'running',
  })
  expect(backend.stop(firstRun.runId)).toMatchObject({
    endedAt: 1,
    runId: 'run-000001',
    status: 'stopped',
  })
  expect(backend.results(firstRun.runId)).toEqual({
    duration_s: 1,
    metrics: { reward: 0, step: 0 },
    results: {},
    status: 'stopped',
  })
  expect(backend.testInference('What action should I take?', firstRun.runId)).toEqual({
    backend: 'in_memory',
    completion: '[in-memory simulation] What action should I take?',
    prompt: 'What action should I take?',
    run_id: 'run-000001',
    simulated: true,
  })
})

test('rl tools expose every Python-compatible handler through an injected backend', async () => {
  let timestamp = 10
  const backend = new InMemoryRLBackend({ clock: () => timestamp++ })
  backend.register('cartpole', { threshold: 200 }, 'Balance a pole.')
  const registry = new ToolRegistry()
  registerRlTools(registry, { backend })

  expect(RL_TOOL_DEFINITIONS.map(definition => definition.function.name)).toEqual([
    'rl_list_environments',
    'rl_select_environment',
    'rl_get_current_config',
    'rl_edit_config',
    'rl_start_training',
    'rl_stop_training',
    'rl_check_status',
    'rl_get_results',
    'rl_list_runs',
    'rl_test_inference',
  ])
  expect(await execute(registry, 'rl_list_environments', {})).toEqual({
    count: 1,
    environments: [{ config: { threshold: 200 }, description: 'Balance a pole.', name: 'cartpole' }],
  })
  expect(await execute(registry, 'rl_start_training', {})).toEqual({
    error: 'backend_error',
    message: 'no environment selected',
    ok: false,
  })
  expect(await execute(registry, 'rl_select_environment', { name: 'cartpole' })).toEqual({
    config: { threshold: 200 },
    description: 'Balance a pole.',
    name: 'cartpole',
  })
  expect(await execute(registry, 'rl_get_current_config', {})).toEqual({
    config: { threshold: 200 },
    environment: 'cartpole',
  })
  expect(await execute(registry, 'rl_edit_config', { updates: { gamma: 0.99 } })).toEqual({
    config: { gamma: 0.99, threshold: 200 },
    environment: 'cartpole',
  })

  const started = await execute(registry, 'rl_start_training', {})
  expect(started).toEqual({
    ended_at: 0,
    environment: 'cartpole',
    metrics: { reward: 0, step: 0 },
    run_id: 'run-000001',
    started_at: 10,
    status: 'running',
  })
  expect(await execute(registry, 'rl_check_status', { run_id: 'run-000001' })).toEqual(started)
  expect(await execute(registry, 'rl_get_results', { run_id: 'run-000001' })).toEqual({
    partial_metrics: { reward: 0, step: 0 },
    status: 'running',
  })
  expect(await execute(registry, 'rl_list_runs', {})).toEqual({ count: 1, runs: [started] })
  expect(await execute(registry, 'rl_stop_training', { run_id: 'run-000001' })).toEqual({
    ...started,
    ended_at: 11,
    status: 'stopped',
  })
  expect(await execute(registry, 'rl_get_results', { run_id: 'run-000001' })).toEqual({
    duration_s: 1,
    metrics: { reward: 0, step: 0 },
    results: {},
    status: 'stopped',
  })
  expect(await execute(registry, 'rl_test_inference', { prompt: 'observe state' })).toEqual({
    backend: 'in_memory',
    completion: '[in-memory simulation] observe state',
    prompt: 'observe state',
    run_id: 'cartpole',
    simulated: true,
  })
  expect(await execute(registry, 'rl_check_status', { run_id: 'unknown' })).toEqual({
    error: 'not_found',
    run_id: 'unknown',
  })
})

test('a multi-session resolver reports an absent external backend instead of fabricating a result', async () => {
  const registry = new ToolRegistry()
  let seenSessionId: string | undefined
  registerRlTools(registry, {
    resolveBackend: async context => {
      seenSessionId = context.sessionId
      return undefined
    },
  })

  expect(await execute(registry, 'rl_list_environments', {}, { sessionId: 'session-no-backend' })).toEqual({
    error: 'backend_not_configured',
    message: 'No reinforcement-learning backend is configured for this session.',
    ok: false,
  })
  expect(seenSessionId).toBe('session-no-backend')
})

async function execute(
  registry: ToolRegistry,
  name: string,
  arguments_: JsonObject,
  context: { readonly sessionId?: string } = {},
): Promise<JsonObject> {
  const value = await registry.execute({
    id: name,
    type: 'function',
    function: { name, arguments: arguments_ },
  } satisfies ToolCall, {
    metadata: {},
    ...(context.sessionId === undefined ? {} : { sessionId: context.sessionId }),
  })
  return JSON.parse(value) as JsonObject
}
