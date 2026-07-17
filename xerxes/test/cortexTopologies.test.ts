// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  Cortex,
  CortexCancellationError,
  nativeConsensusSynthesis,
} from '../src/cortex/cortex.js'
import { ProcessType } from '../src/cortex/core/enums.js'
import { CortexPlanner } from '../src/cortex/planner.js'

test('Cortex composes sequential and parallel task runners while retaining dependency barriers', async () => {
  const sequentialContexts: string[] = []
  let tick = 0
  const sequential = new Cortex({
    process: ProcessType.SEQUENTIAL,
    now: () => new Date(tick++ * 10),
    agents: [{ id: 'worker' }],
    tasks: [
      { id: 'discover', description: 'Find facts', expectedOutput: 'Facts', agentId: 'worker' },
      {
        id: 'write',
        description: 'Write report',
        expectedOutput: 'Report',
        agentId: 'worker',
        dependencies: ['discover'],
        contextTaskIds: ['discover'],
      },
    ],
    executor: context => {
      if (context.task.id === 'discover') return 'facts'
      sequentialContexts.push(context.context)
      return 'report'
    },
  })

  const sequentialOutput = await sequential.kickoff()
  expect(sequentialOutput.process).toBe(ProcessType.SEQUENTIAL)
  expect(sequentialOutput.rawOutput).toBe('report')
  expect(sequentialOutput.executionTimeMs).toBeGreaterThan(0)
  expect(sequentialContexts).toEqual(['Output from task discover:\nfacts'])

  const transitions: string[] = []
  const parallel = new Cortex({
    process: ProcessType.PARALLEL,
    maxParallel: 2,
    agents: [{ id: 'worker' }],
    tasks: [
      { id: 'left', description: 'Left', expectedOutput: 'left', agentId: 'worker' },
      { id: 'right', description: 'Right', expectedOutput: 'right', agentId: 'worker' },
      { id: 'join', description: 'Join', expectedOutput: 'joined', agentId: 'worker', dependencies: ['left', 'right'] },
    ],
    executor: async context => {
      transitions.push(`start:${context.task.id}`)
      if (context.task.id === 'join') {
        expect(transitions).toContain('end:left')
        expect(transitions).toContain('end:right')
        return `${context.dependencyOutputs.get('left')?.output}/${context.dependencyOutputs.get('right')?.output}`
      }
      await Promise.resolve()
      transitions.push(`end:${context.task.id}`)
      return context.task.id
    },
  })

  const parallelOutput = await parallel.run()
  expect(parallelOutput.process).toBe(ProcessType.PARALLEL)
  expect(parallelOutput.rawOutput).toBe('left/right')
  expect(parallelOutput.taskOutputs.map(output => output.status)).toEqual(['succeeded', 'succeeded', 'succeeded'])
})

test('Cortex hierarchical mode uses manager ports, bounded review, and deterministic safe fallback assignments', async () => {
  const executions: Array<{ readonly agent: string; readonly context: string; readonly task: string }> = []
  const reviews: Array<{ readonly attempt: number; readonly task: string }> = []
  const researcher = {
    id: 'researcher',
    role: 'Researcher',
    execute: async (context: { readonly task: { readonly id: string }; readonly context: string }) => {
      executions.push({ agent: 'researcher', task: context.task.id, context: context.context })
      return 'facts'
    },
  }
  const writer = {
    id: 'writer',
    role: 'Writer',
    execute: async (context: { readonly task: { readonly id: string }; readonly context: string }) => {
      executions.push({ agent: 'writer', task: context.task.id, context: context.context })
      return context.context.includes('Manager review feedback:') ? 'revised report' : 'first report'
    },
  }
  const cortex = new Cortex({
    process: ProcessType.HIERARCHICAL,
    agents: [researcher, writer],
    tasks: [
      { id: 'research', description: 'Collect sources', expectedOutput: 'Sources' },
      { id: 'draft', description: 'Write report', expectedOutput: 'Report', dependencies: ['research'] },
    ],
    hierarchy: {
      plan: () => ({
        assignments: [
          { taskId: 'research', agentId: 'researcher' },
          { taskId: 'draft', agentId: 'unknown-manager-agent', dependencies: ['research'] },
        ],
      }),
      review: request => {
        reviews.push({ task: request.task.id, attempt: request.attempt })
        if (request.task.id === 'draft' && request.attempt === 1) {
          return { approved: false, feedback: 'Cite the collected sources.', improvementsNeeded: ['Add citations'] }
        }
        return { approved: true }
      },
      summarize: request => `manager summary: ${request.taskOutputs.map(output => output.output).join(' | ')}`,
    },
  })

  const output = await cortex.kickoff()
  expect(output.rawOutput).toBe('manager summary: facts | revised report')
  expect(output.taskOutputs.map(item => item.output)).toEqual(['facts', 'revised report'])
  expect(executions.filter(item => item.task === 'draft').map(item => item.agent)).toEqual(['writer', 'writer'])
  expect(executions.find(item => item.task === 'draft')?.context).toContain('Output from task research:\nfacts')
  expect(executions.at(-1)?.context).toContain('Manager review feedback:')
  expect(reviews).toEqual([
    { task: 'research', attempt: 1 },
    { task: 'draft', attempt: 1 },
    { task: 'draft', attempt: 2 },
  ])
  expect(output.diagnostics).toContainEqual(expect.objectContaining({
    code: 'hierarchy_plan_fallback',
    taskId: 'draft',
  }))
})

test('Cortex consensus executes every candidate and delegates synthesis through an explicit port', async () => {
  const candidateCalls: string[] = []
  const synthesisInputs: string[][] = []
  const first = {
    id: 'analyst',
    role: 'Analyst',
    execute: (context: { readonly task: { readonly id: string }; readonly context: string }) => {
      candidateCalls.push(`analyst:${context.task.id}`)
      if (context.task.id === 'publish') expect(context.context).toContain('synthesis:analyst-first|critic-first')
      return `analyst-${context.task.id}`
    },
  }
  const second = {
    id: 'critic',
    role: 'Critic',
    execute: (context: { readonly task: { readonly id: string } }) => {
      candidateCalls.push(`critic:${context.task.id}`)
      return `critic-${context.task.id}`
    },
  }
  const cortex = new Cortex({
    process: ProcessType.CONSENSUS,
    agents: [first, second],
    tasks: [
      { id: 'first', description: 'Assess options', expectedOutput: 'Assessment' },
      { id: 'publish', description: 'Publish result', expectedOutput: 'Result', dependencies: ['first'], contextTaskIds: ['first'] },
    ],
    consensus: {
      maxCandidatesParallel: 2,
      synthesizer: request => {
        const values = request.candidates.map(candidate => candidate.output)
        synthesisInputs.push(values)
        return { output: `synthesis:${values.join('|')}`, metadata: { source: 'test synthesizer' } }
      },
    },
  })

  const output = await cortex.run()
  expect(candidateCalls).toEqual(['analyst:first', 'critic:first', 'analyst:publish', 'critic:publish'])
  expect(synthesisInputs).toEqual([
    ['analyst-first', 'critic-first'],
    ['analyst-publish', 'critic-publish'],
  ])
  expect(output.rawOutput).toBe('synthesis:analyst-publish|critic-publish')
  expect(output.taskOutputs[0]?.metadata).toMatchObject({
    consensus: { strategy: 'injected', completedCandidates: ['analyst', 'critic'] },
  })

  const native = nativeConsensusSynthesis({
    task: { id: 'native', description: 'Aggregate', expectedOutput: 'Aggregate' },
    context: {
      task: { id: 'native', description: 'Aggregate', expectedOutput: 'Aggregate' },
      agent: undefined,
      context: '',
      inputs: {},
      dependencyOutputs: new Map(),
    },
    inputs: {},
    candidates: [
      { agent: first, output: 'one', metadata: {} },
      { agent: second, output: 'two', metadata: {} },
    ],
  })
  expect(native.output).toBe('### Analyst\none\n\n### Critic\ntwo')
})

test('Cortex planned mode uses CortexPlanner execution while adding declared task dependency barriers', async () => {
  const calls: string[] = []
  const researcher = {
    id: 'researcher',
    role: 'Researcher',
    execute: (context: { readonly task: { readonly id: string } }) => {
      calls.push(`research:${context.task.id}`)
      return 'evidence'
    },
  }
  const writer = {
    id: 'writer',
    role: 'Writer',
    execute: (context: {
      readonly context: string
      readonly dependencyOutputs: ReadonlyMap<string, { readonly output: string }>
      readonly task: { readonly id: string }
    }) => {
      calls.push(`write:${context.task.id}`)
      expect(context.context).toContain('Output from planned step step-research:\nevidence')
      expect(context.dependencyOutputs.get('research')?.output).toBe('evidence')
      return 'published report'
    },
  }
  const planner = new CortexPlanner(async () => `<plan>
    <objective>Research and publish</objective>
    <complexity>medium</complexity>
    <estimated_time>2</estimated_time>
    <step id="step-research">
      <agent>Researcher</agent>
      <action>research</action>
      <arguments><topic>TypeScript</topic></arguments>
      <dependencies></dependencies>
      <description>Collect evidence</description>
    </step>
    <step id="step-write">
      <agent>Writer</agent>
      <action>write</action>
      <arguments><source>result_from_step_step-research</source></arguments>
      <dependencies></dependencies>
      <description>Publish a report</description>
    </step>
  </plan>`)
  const cortex = new Cortex({
    process: ProcessType.PLANNED,
    agents: [researcher, writer],
    planner,
    tasks: [
      { id: 'research', description: 'Collect evidence', expectedOutput: 'Evidence', agentId: 'researcher' },
      { id: 'write', description: 'Publish report', expectedOutput: 'Report', agentId: 'writer', dependencies: ['research'] },
    ],
  })

  const output = await cortex.kickoff()
  expect(calls).toEqual(['research:research', 'write:write'])
  expect(output.rawOutput).toBe('published report')
  expect(output.taskOutputs.map(item => item.status)).toEqual(['succeeded', 'succeeded'])
  expect(output.taskOutputs[1]?.metadata).toMatchObject({ planStepId: 'step-write', planAction: 'write' })
})

test('Cortex rejects promptly on caller cancellation and passes the signal to the execution port', async () => {
  const controller = new AbortController()
  let markStarted: (() => void) | undefined
  const started = new Promise<void>(resolve => { markStarted = resolve })
  const cortex = new Cortex({
    process: ProcessType.SEQUENTIAL,
    tasks: [{ id: 'wait', description: 'Wait', expectedOutput: 'Done' }],
    taskRunner: request => new Promise<string>((_resolve, reject) => {
      markStarted?.()
      request.signal?.addEventListener('abort', () => reject(new Error('execution port observed cancellation')), { once: true })
    }),
  })

  const run = cortex.kickoff({ signal: controller.signal })
  await started
  controller.abort('stop now')
  await expect(run).rejects.toBeInstanceOf(CortexCancellationError)
})

test('Cortex consensus observes late sibling rejections instead of leaking unhandled rejections', async () => {
  const unhandled: unknown[] = []
  const onUnhandled = (reason: unknown) => { unhandled.push(reason) }
  process.on('unhandledRejection', onUnhandled)
  try {
    const fast = {
      id: 'fast',
      role: 'Fast',
      execute: () => { throw new CortexCancellationError('fast failure') },
    }
    const slow = {
      id: 'slow',
      role: 'Slow',
      execute: async () => {
        await Bun.sleep(25)
        throw new CortexCancellationError('slow failure')
      },
    }
    const cortex = new Cortex({
      process: ProcessType.CONSENSUS,
      agents: [fast, slow],
      tasks: [{ id: 'boom', description: 'Fail together', expectedOutput: 'Nothing' }],
      consensus: { maxCandidatesParallel: 2 },
    })

    const output = await cortex.run()
    expect(output.taskOutputs[0]?.status).toBe('failed')
    expect(output.taskOutputs[0]?.error).toContain('fast failure')
    await Bun.sleep(50)
    expect(unhandled).toEqual([])
  } finally {
    process.removeListener('unhandledRejection', onUnhandled)
  }
})
