// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { CortexAgent } from '../src/cortex/agents/agent.js'
import {
  Cortex,
  CortexCancellationError,
} from '../src/cortex/cortex.js'
import { ProcessType } from '../src/cortex/core/enums.js'
import {
  CortexOrchestrator,
  CortexRunAbortedError,
  CortexRunFailedError,
  DEFAULT_MAX_PARALLEL,
} from '../src/cortex/orchestrator.js'
import { TaskGraphError } from '../src/cortex/task.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'

test('CortexAgent serializes concurrent executions so shared conversation history cannot interleave', async () => {
  const client = new GatedTextClient('done')
  const agent = new CortexAgent({
    role: 'Serial Agent',
    goal: 'Run one execution at a time',
    backstory: 'Protects its shared history.',
    model: 'gpt-test',
    llm: client,
  })

  const first = agent.execute('first task')
  const second = agent.execute('second task')
  await delay(10)
  expect(client.active).toBe(1)
  expect(client.maxActive).toBe(1)

  client.releaseAll()
  await expect(first).resolves.toBe('done')
  await delay(10)
  expect(client.active).toBe(1)
  client.releaseAll()
  await expect(second).resolves.toBe('done')
  expect(client.maxActive).toBe(1)
  expect(agent.getExecutionStats().timesExecuted).toBe(2)
})

test('Cortex propagates cancellation into executors so in-flight work aborts with the caller signal', async () => {
  let observed: AbortSignal | undefined
  let executorSettled = false
  const engine = new Cortex({
    process: ProcessType.SEQUENTIAL,
    agents: [{ id: 'worker' }],
    tasks: [{ id: 'wait', description: 'Wait', expectedOutput: 'Never', agentId: 'worker' }],
    executor: context => {
      observed = context.signal
      return new Promise<string>((_resolve, reject) => {
        context.signal?.addEventListener('abort', () => {
          executorSettled = true
          reject(context.signal?.reason)
        }, { once: true })
      })
    },
  })

  const controller = new AbortController()
  const pending = engine.kickoff({ signal: controller.signal })
  await delay(1)
  controller.abort(new Error('stop now'))
  await expect(pending).rejects.toBeInstanceOf(CortexCancellationError)
  expect(observed?.aborted).toBe(true)
  // The detached topology settles promptly because the signal reached the executor.
  await delay(10)
  expect(executorSettled).toBe(true)
})

test('CortexOrchestrator stops scheduling new tasks once the run signal aborts', async () => {
  const executed: string[] = []
  const controller = new AbortController()
  const orchestrator = new CortexOrchestrator({
    executor: context => {
      executed.push(context.task.id)
      controller.abort()
      return 'ok'
    },
    tasks: [
      { id: 'first', description: 'First', expectedOutput: 'one' },
      { id: 'second', description: 'Second', expectedOutput: 'two', dependencies: ['first'] },
    ],
  })

  await expect(orchestrator.run({ signal: controller.signal })).rejects.toBeInstanceOf(CortexRunAbortedError)
  expect(executed).toEqual(['first'])
})

test('CortexOrchestrator caps parallel batches at a finite default and allows explicit unbounded opt-in', async () => {
  const tasks = Array.from({ length: 8 }, (_value, index) => ({
    id: `task-${index}`,
    description: `Task ${index}`,
    expectedOutput: 'done',
  }))

  const capped = trackConcurrency()
  const defaultRunner = new CortexOrchestrator({
    process: 'parallel',
    executor: capped.executor,
    tasks,
  })
  await defaultRunner.run()
  expect(capped.maxActive).toBe(DEFAULT_MAX_PARALLEL)

  const unbounded = trackConcurrency()
  const unboundedRunner = new CortexOrchestrator({
    process: 'parallel',
    maxParallel: Number.POSITIVE_INFINITY,
    executor: unbounded.executor,
    tasks,
  })
  await unboundedRunner.run()
  expect(unbounded.maxActive).toBe(tasks.length)
})

test('Cortex consensus caps candidate fan-out by default and allows explicit unbounded opt-in', async () => {
  const agents = Array.from({ length: 6 }, (_value, index) => ({ id: `candidate-${index}` }))
  const tasks = [{ id: 'decide', description: 'Decide', expectedOutput: 'Decision' }]

  const capped = trackConcurrency()
  const cappedEngine = new Cortex({
    process: ProcessType.CONSENSUS,
    agents,
    tasks,
    executor: capped.executor,
  })
  const cappedOutput = await cappedEngine.kickoff()
  expect(cappedOutput.taskOutputs[0]?.status).toBe('succeeded')
  expect(capped.maxActive).toBe(DEFAULT_MAX_PARALLEL)

  const unbounded = trackConcurrency()
  const unboundedEngine = new Cortex({
    process: ProcessType.CONSENSUS,
    agents,
    tasks,
    consensus: { maxCandidatesParallel: Number.POSITIVE_INFINITY },
    executor: unbounded.executor,
  })
  const unboundedOutput = await unboundedEngine.kickoff()
  expect(unboundedOutput.taskOutputs[0]?.status).toBe('succeeded')
  expect(unbounded.maxActive).toBe(agents.length)
})

test('CortexOrchestrator reports aggregate status and failed counts for fully and partially failed runs', async () => {
  const failing = new CortexOrchestrator({
    executor: () => { throw new Error('boom') },
    tasks: [
      { id: 'a', description: 'A', expectedOutput: 'a' },
      { id: 'b', description: 'B', expectedOutput: 'b' },
    ],
  })
  const failed = await failing.run()
  expect(failed.status).toBe('failed')
  expect(failed.failedCount).toBe(2)
  expect(failed.succeededCount).toBe(0)
  expect(failed.rawOutput).toBe('')

  const mixed = new CortexOrchestrator({
    executor: context => {
      if (context.task.id === 'bad') throw new Error('boom')
      return 'good'
    },
    tasks: [
      { id: 'good', description: 'Good', expectedOutput: 'g' },
      { id: 'bad', description: 'Bad', expectedOutput: 'b' },
    ],
  })
  const partial = await mixed.run()
  expect(partial.status).toBe('partial')
  expect(partial.failedCount).toBe(1)
  expect(partial.succeededCount).toBe(1)
  expect(partial.rawOutput).toBe('good')
})

test('CortexOrchestrator fail-fast mode rejects on the first failure and skips remaining tasks', async () => {
  const executed: string[] = []
  const orchestrator = new CortexOrchestrator({
    failFast: true,
    executor: context => {
      executed.push(context.task.id)
      if (context.task.id === 'first') throw new Error('first failed')
      return 'ok'
    },
    tasks: [
      { id: 'first', description: 'First', expectedOutput: 'one' },
      { id: 'second', description: 'Second', expectedOutput: 'two' },
    ],
  })

  const failure = await orchestrator.run().catch((error: unknown) => error)
  expect(failure).toBeInstanceOf(CortexRunFailedError)
  expect((failure as CortexRunFailedError).failures.map(output => output.taskId)).toEqual(['first'])
  expect(executed).toEqual(['first'])
})

test('CortexOrchestrator rejects context task references outside an earlier dependency layer', async () => {
  expect(() => new CortexOrchestrator({
    executor: () => 'ok',
    tasks: [
      { id: 'a', description: 'A', expectedOutput: 'a' },
      { id: 'b', description: 'B', expectedOutput: 'b', contextTaskIds: ['a'] },
    ],
  })).toThrow(TaskGraphError)

  expect(() => new CortexOrchestrator({
    executor: () => 'ok',
    tasks: [{ id: 'a', description: 'A', expectedOutput: 'a', contextTaskIds: ['ghost'] }],
  })).toThrow(/unknown context task/u)

  const contexts: string[] = []
  const valid = new CortexOrchestrator({
    executor: context => {
      contexts.push(context.context)
      return context.task.id
    },
    tasks: [
      { id: 'a', description: 'A', expectedOutput: 'a' },
      { id: 'b', description: 'B', expectedOutput: 'b', dependencies: ['a'], contextTaskIds: ['a'] },
    ],
  })
  const output = await valid.run()
  expect(output.status).toBe('succeeded')
  expect(contexts[1]).toBe('Output from task a:\na')
})

test('CortexAgent delegated task ids are monotonic across sequential delegations', async () => {
  const ids: string[] = []
  const agent = new CortexAgent({
    role: 'Delegator',
    goal: 'Hand off work',
    backstory: 'Delegates through an explicit port.',
    model: 'gpt-test',
    llm: new StaticTextClient('unused'),
    allowDelegation: true,
    delegation: {
      delegate: request => {
        ids.push(request.task.id)
        return 'delegated output'
      },
    },
  })

  await agent.delegateTask('first delegated task')
  await agent.delegateTask('second delegated task')
  expect(ids).toEqual([`${agent.id}-delegation-1`, `${agent.id}-delegation-2`])
})

test('CortexAgent bounds retained execution timing samples while keeping lifetime totals', async () => {
  let now = 0
  const agent = new CortexAgent({
    role: 'Timed',
    goal: 'Track time',
    backstory: 'Bounded buffers.',
    model: 'gpt-test',
    llm: new StaticTextClient('ok'),
    now: () => (now += 5),
  })

  for (let index = 0; index < 110; index += 1) await agent.execute(`task ${index}`)
  const stats = agent.getExecutionStats()
  expect(stats.timesExecuted).toBe(110)
  expect(stats.recentExecutionTimesMs).toHaveLength(5)
  expect(stats.totalExecutionTimeMs).toBeGreaterThan(0)
  expect(stats.averageExecutionTimeMs).toBeCloseTo(stats.totalExecutionTimeMs / 110, 6)
  expect(stats.minExecutionTimeMs).toBeLessThanOrEqual(stats.maxExecutionTimeMs)
})

function delay(milliseconds: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, milliseconds))
}

function trackConcurrency(): { readonly executor: () => Promise<string>; readonly maxActive: number } {
  const state = { active: 0, maxActive: 0 }
  return {
    get maxActive() {
      return state.maxActive
    },
    executor: async () => {
      state.active += 1
      state.maxActive = Math.max(state.maxActive, state.active)
      await delay(5)
      state.active -= 1
      return 'ok'
    },
  }
}

/** Yields text and then holds the stream open until the test releases it. */
class GatedTextClient implements LlmClient {
  active = 0
  maxActive = 0
  private gates: Array<() => void> = []

  constructor(private readonly text: string) {}

  releaseAll(): void {
    for (const release of this.gates.splice(0)) release()
  }

  async *stream(_request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.active += 1
    this.maxActive = Math.max(this.maxActive, this.active)
    try {
      yield { content: this.text }
      await new Promise<void>(resolve => {
        this.gates.push(resolve)
      })
    } finally {
      this.active -= 1
    }
  }
}

class StaticTextClient implements LlmClient {
  constructor(private readonly text: string) {}

  async *stream(_request: CompletionRequest): AsyncGenerator<LlmDelta> {
    yield { content: this.text }
  }
}
