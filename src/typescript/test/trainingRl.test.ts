// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  builtinEnvironments,
  canTransition,
  mapTinkerStatus,
  RLEnvironment,
  RLEnvironmentRegistry,
  RLRunState,
  RLRunStatus,
  RLRunTracker,
  TinkerClient,
  TinkerRunConfig,
  TinkerTransportUnavailableError,
  WandBHook,
  type RLRunEvent,
  type TinkerRunPayload,
  type TinkerTransport,
} from '../src/training/rl/index.js'

test('RL environment registry keeps immutable metadata and built-in reward ownership', () => {
  const registry = new RLEnvironmentRegistry()
  const environment = new RLEnvironment({
    name: 'zeta',
    description: 'A local reward function.',
    tags: ['test'],
    rewardFn: trajectory => trajectory.ok === true ? 0.5 : 0,
  })
  registry.register(environment)
  registry.register(new RLEnvironment({ name: 'alpha', description: 'First by name.' }))

  expect(registry.listEnvironments().map(value => value.name)).toEqual(['alpha', 'zeta'])
  expect(registry.get('zeta')).toBe(environment)
  expect(environment.reward({ ok: true })).toBe(0.5)
  expect(() => new RLEnvironment({ name: '', description: 'missing id' })).toThrow('name must be non-empty')

  const builtins = builtinEnvironments()
  expect(builtins.listEnvironments().map(value => value.name)).toEqual([
    'agentic-opd',
    'xerxes-swe-bench',
    'xerxes-terminal-test',
  ])
  expect(builtins.get('xerxes-terminal-test')?.reward({ passed_all_tasks: true })).toBe(1)
  expect(builtins.get('xerxes-swe-bench')?.reward({ tests_passed: 3, tests_total: 4 })).toBe(0.75)
  expect(builtins.get('agentic-opd')?.reward({})).toBeUndefined()
})

test('RL run tracker enforces local transitions while preserving external observations in its event ledger', () => {
  let milliseconds = 0
  const emitted: RLRunEvent[] = []
  const tracker = new RLRunTracker(new RLRunState({ runId: 'run-1' }), {
    clock: () => new Date(milliseconds++ * 1000),
    eventSink: { record: event => emitted.push(event) },
  })

  expect(tracker.events[0]).toMatchObject({
    kind: 'created',
    runId: 'run-1',
    state: expect.objectContaining({ status: RLRunStatus.PENDING }),
    timestamp: '1970-01-01T00:00:00.000Z',
  })
  tracker.update({ iteration: 3, reward: 0.2, metadata: { batch: 'first' } })
  tracker.transition(RLRunStatus.RUNNING, { tokensSeen: 12 })
  tracker.transition(RLRunStatus.SUCCEEDED, { loss: 0.1 })

  expect(tracker.state.toRecord()).toEqual({
    run_id: 'run-1',
    status: RLRunStatus.SUCCEEDED,
    iteration: 3,
    reward: 0.2,
    loss: 0.1,
    tokens_seen: 12,
    wandb_url: '',
    error: '',
    metadata: { batch: 'first' },
  })
  expect(() => tracker.transition(RLRunStatus.CANCELLED)).toThrow('cannot transition RL run run-1')
  const observed = tracker.observe(new RLRunState({ runId: 'run-1', status: RLRunStatus.CANCELLED }))
  expect(observed).toMatchObject({
    kind: 'observed',
    previousStatus: RLRunStatus.SUCCEEDED,
    transitionValid: false,
  })
  expect(emitted).toHaveLength(5)
  expect(canTransition(RLRunStatus.PENDING, RLRunStatus.RUNNING)).toBeTrue()
  expect(canTransition(RLRunStatus.PENDING, RLRunStatus.SUCCEEDED)).toBeFalse()
})

test('Tinker client requires transport, maps provider state, and records remote lifecycle events', async () => {
  const payloads: TinkerRunPayload[] = []
  const events: RLRunEvent[] = []
  let milliseconds = 0
  const transport: TinkerTransport = {
    createRun: payload => {
      payloads.push(payload)
      return { id: 'tinker-123' }
    },
    getRun: () => ({
      status: 'active',
      iteration: 2,
      reward: 0.7,
      loss: 0.3,
      tokens_seen: 42,
      wandb_url: 'https://wandb.example/runs/tinker-123',
      metadata: { provider: 'tinker' },
    }),
    cancelRun: () => true,
  }
  const client = new TinkerClient({
    transport,
    clock: () => new Date(milliseconds++ * 1000),
    eventSink: { record: event => events.push(event) },
  })
  const config = new TinkerRunConfig({
    model: 'provider/model',
    env: 'xerxes-terminal-test',
    learningRate: 0.0002,
    batchSize: 4,
    steps: 20,
    extra: { warmup_steps: 2 },
  })

  expect(await client.start(config)).toBe('tinker-123')
  expect(payloads).toEqual([{
    model: 'provider/model',
    env: 'xerxes-terminal-test',
    learning_rate: 0.0002,
    batch_size: 4,
    steps: 20,
    warmup_steps: 2,
  }])
  expect(await client.status('tinker-123')).toMatchObject({
    status: RLRunStatus.RUNNING,
    iteration: 2,
    reward: 0.7,
    tokensSeen: 42,
    metadata: { provider: 'tinker' },
  })
  expect(await client.cancel('tinker-123')).toBeTrue()
  expect(client.trackedState('tinker-123')?.status).toBe(RLRunStatus.CANCELLED)
  expect(client.events('tinker-123').map(event => event.kind)).toEqual(['created', 'observed', 'transitioned'])
  expect(events).toHaveLength(3)
  expect(mapTinkerStatus('completed')).toBe(RLRunStatus.SUCCEEDED)
  expect(mapTinkerStatus('vendor-unknown')).toBe(RLRunStatus.FAILED)

  const unavailable = new TinkerClient()
  await expect(unavailable.start(config)).rejects.toBeInstanceOf(TinkerTransportUnavailableError)
  expect(await unavailable.status('missing')).toMatchObject({
    status: RLRunStatus.FAILED,
    error: 'Tinker transport not configured',
  })
  expect(await unavailable.cancel('missing')).toBeFalse()
})

test('WandB hook stays inert without a port and forwards lifecycle calls through injected telemetry', async () => {
  const disabled = new WandBHook()
  expect(disabled.isAvailable()).toBeFalse()
  expect(await disabled.start({ model: 'none' })).toBe('')
  await disabled.log({ reward: 1 })
  await disabled.finish()

  const starts: Array<Record<string, unknown>> = []
  const metrics: Array<Record<string, unknown>> = []
  let finished = 0
  const hook = new WandBHook({
    project: 'xerxes-tests',
    entity: 'agents',
    telemetry: {
      startRun: input => {
        starts.push({ ...input })
        return {
          url: 'https://wandb.example/runs/one',
          log: values => {
            metrics.push({ ...values })
          },
          finish: () => {
            finished += 1
          },
        }
      },
    },
  })

  expect(hook.isAvailable()).toBeTrue()
  expect(await hook.start({ model: 'provider/model', steps: 20 }, { name: 'run-one' })).toBe(
    'https://wandb.example/runs/one',
  )
  await hook.log({ reward: 0.9 })
  await hook.finish()
  await hook.log({ ignored: true })

  expect(starts).toEqual([{
    project: 'xerxes-tests',
    entity: 'agents',
    name: 'run-one',
    config: { model: 'provider/model', steps: 20 },
  }])
  expect(metrics).toEqual([{ reward: 0.9 }])
  expect(finished).toBe(1)
})
