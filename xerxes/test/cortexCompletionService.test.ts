// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  CortexCompletionService,
  CortexProcessType,
  aggregateCortexExecution,
  deriveCortexCompletionConfig,
  latestUserPrompt,
  type CortexExecutionPort,
  type CortexExecutionRequest,
  type CortexStreamEvent,
  type CortexTaskExecutionRequest,
} from '../src/api-server/cortexCompletionService.js'

test('Cortex completion config honors model variants and explicit metadata overrides', () => {
  expect(deriveCortexCompletionConfig({ model: 'cortex-task-parallel' })).toEqual({
    taskMode: true,
    processType: CortexProcessType.PARALLEL,
  })
  expect(deriveCortexCompletionConfig({
    model: 'cortex-task-parallel',
    metadata: {
      task_mode: false,
      process_type: 'HIERARCHICAL',
      background: 'Follow the release process.',
    },
  })).toEqual({
    taskMode: false,
    processType: CortexProcessType.HIERARCHICAL,
    background: 'Follow the release process.',
  })
  expect(deriveCortexCompletionConfig({
    model: 'cortex-hierarchical',
    metadata: { process_type: 'unknown-strategy' },
  }).processType).toBe(CortexProcessType.HIERARCHICAL)
})

test('Cortex prompt extraction uses the most recent user message before transcript fallback', () => {
  expect(latestUserPrompt([
    { role: 'system', content: 'Follow policy.' },
    { role: 'user', content: 'Older request.' },
    { role: 'assistant', content: 'Older answer.' },
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Latest request.' },
        { type: 'image_url', image_url: { url: 'https://example.test/chart.png' } },
      ],
    },
  ])).toBe('Latest request.')
  expect(latestUserPrompt([
    { role: 'system', content: 'System context.' },
    { role: 'assistant', content: 'Prior response.' },
  ])).toBe('System context.\nPrior response.')
})

test('Cortex task completions pass derived config to the execution port and aggregate successful task outputs', async () => {
  const taskRequests: CortexTaskExecutionRequest[] = []
  const instructionRequests: CortexExecutionRequest[] = []
  const service = new CortexCompletionService({
    execution: {
      executeTask: async request => {
        taskRequests.push(request)
        return {
          taskOutputs: [
            { status: 'succeeded', output: 'Research findings' },
            { status: 'failed', output: 'This must not leak' },
            { status: 'succeeded', output: 'Release draft' },
          ],
        }
      },
      executeInstruction: async request => {
        instructionRequests.push(request)
        return { output: 'unused' }
      },
      streamTask: emptyStream,
      streamInstruction: emptyStream,
    },
    now: () => 1_700_000_000_987,
    responseId: () => 'chatcmpl-cortex-task',
  })

  const response = await service.createCompletion({
    model: 'cortex-task-parallel',
    metadata: { process_type: 'hierarchical', background: 'Use approved sources.' },
    messages: [
      { role: 'user', content: 'Ignore this older request.' },
      { role: 'assistant', content: 'Acknowledged.' },
      { role: 'user', content: 'Prepare release notes.' },
    ],
  })

  expect(taskRequests).toEqual([{
    model: 'cortex-task-parallel',
    prompt: 'Prepare release notes.',
    processType: CortexProcessType.HIERARCHICAL,
    background: 'Use approved sources.',
  }])
  expect(instructionRequests).toEqual([])
  expect(response).toEqual({
    id: 'chatcmpl-cortex-task',
    object: 'chat.completion',
    created: 1_700_000_000,
    model: 'cortex-task-parallel',
    choices: [{
      index: 0,
      message: { role: 'assistant', content: 'Research findings\n\nRelease draft' },
      finish_reason: 'stop',
    }],
    usage: { prompt_tokens: 3, completion_tokens: 4, total_tokens: 7 },
  })
  expect(aggregateCortexExecution({
    rawOutput: 'Terminal answer',
    taskOutputs: [{ status: 'succeeded', output: 'Earlier task output' }],
  })).toBe('Terminal answer')
})

test('Cortex instruction completions preserve the model-derived hierarchical process', async () => {
  const instructionRequests: CortexExecutionRequest[] = []
  const service = new CortexCompletionService({
    execution: {
      executeTask: async () => ({ output: 'unused' }),
      executeInstruction: async request => {
        instructionRequests.push(request)
        return { rawOutput: 'Direct answer' }
      },
      streamTask: emptyStream,
      streamInstruction: emptyStream,
    },
    responseId: () => 'chatcmpl-cortex-instruction',
  })

  const response = await service.createCompletion({
    model: 'cortex-hierarchical',
    messages: [{ role: 'user', content: 'Coordinate reviewers.' }],
  })

  expect(instructionRequests).toEqual([{
    model: 'cortex-hierarchical',
    prompt: 'Coordinate reviewers.',
    processType: CortexProcessType.HIERARCHICAL,
  }])
  expect(response.choices[0].message.content).toBe('Direct answer')
})

test('Cortex streaming completions frame native events as OpenAI SSE with Cortex metadata', async () => {
  const taskRequests: CortexTaskExecutionRequest[] = []
  const events: CortexStreamEvent[] = [
    {
      type: 'stream_chunk',
      toolCalls: [{ name: 'Search', arguments: { query: 'release process' } }],
    },
    { type: 'stream_chunk', content: 'Drafting the release note.' },
    { type: 'function_detection', message: 'search and write' },
    { type: 'functions_extracted', functions: ['Search', 'Write'] },
    { type: 'function_start', functionName: 'Search', progress: 0.5 },
    { type: 'function_complete', functionName: 'Search', status: 'ok', result: 'x'.repeat(101) },
    { type: 'agent_switch', fromAgent: 'researcher', toAgent: 'writer', reason: 'sources collected' },
    { type: 'reinvoke', message: 'validate citations' },
    { type: 'completion', functionCallsExecuted: 2 },
  ]
  const service = new CortexCompletionService({
    execution: {
      executeTask: async () => ({ output: 'unused' }),
      executeInstruction: async () => ({ output: 'unused' }),
      streamTask: request => {
        taskRequests.push(request)
        return streamEvents(events)
      },
      streamInstruction: emptyStream,
    },
    now: () => 1_700_000_000_987,
    responseId: () => 'chatcmpl-cortex-stream',
  })

  const frames = await collect(service.createStreamingCompletion({
    model: 'cortex-task-parallel',
    metadata: { background: 'Only use verified release notes.' },
    messages: [{ role: 'user', content: 'Write a release note.' }],
  }))
  const payloads = frames
    .filter(frame => frame.startsWith('data: {'))
    .map(frame => JSON.parse(frame.slice('data: '.length)) as Record<string, unknown>)

  expect(taskRequests).toEqual([{
    model: 'cortex-task-parallel',
    prompt: 'Write a release note.',
    processType: CortexProcessType.PARALLEL,
    background: 'Only use verified release notes.',
  }])
  expect(payloads).toHaveLength(10)
  expect(payloads[0]).toMatchObject({
    id: 'chatcmpl-cortex-stream',
    object: 'chat.completion.chunk',
    created: 1_700_000_000,
    model: 'cortex-task-parallel',
    choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }],
    metadata: { tool_calls: [{ name: 'Search', arguments: { query: 'release process' } }] },
  })
  expect(payloads[1]).toMatchObject({
    choices: [{ index: 0, delta: { content: 'Drafting the release note.' }, finish_reason: null }],
  })
  expect(payloads.slice(2, 9).map(payload => (payload.metadata as { event?: string }).event)).toEqual([
    'function_detection',
    'functions_extracted',
    'function_start',
    'function_complete',
    'agent_switch',
    'reinvoke',
    'completion',
  ])
  expect(payloads[4]).toMatchObject({ metadata: { event: 'function_start', function: 'Search', progress: 0.5 } })
  expect(payloads[5]).toMatchObject({
    metadata: { event: 'function_complete', function: 'Search', status: 'ok', has_result: true },
    choices: [{ delta: { content: `\n*Search completed*\n   Result: ${'x'.repeat(100)}...\n` } }],
  })
  expect(payloads[6]).toMatchObject({
    metadata: { event: 'agent_switch', from_agent: 'researcher', to_agent: 'writer', reason: 'sources collected' },
  })
  expect(payloads[8]).toMatchObject({ metadata: { event: 'completion', functions_executed: 2 } })
  expect(payloads[9]).toMatchObject({
    choices: [{ index: 0, delta: { content: '' }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
  })
  expect(frames.at(-1)).toBe('data: [DONE]\n\n')
})

test('Cortex execution failures propagate instead of producing a fabricated completion', async () => {
  const service = new CortexCompletionService({
    execution: {
      executeTask: async () => { throw new Error('task executor unavailable') },
      executeInstruction: async () => { throw new Error('instruction executor unavailable') },
      streamTask: emptyStream,
      streamInstruction: emptyStream,
    },
  })

  await expect(service.createCompletion({
    model: 'cortex-task',
    messages: [{ role: 'user', content: 'Do not pretend this completed.' }],
  })).rejects.toThrow('task executor unavailable')
})

async function* emptyStream(): AsyncGenerator<CortexStreamEvent> {}

async function* streamEvents(events: readonly CortexStreamEvent[]): AsyncGenerator<CortexStreamEvent> {
  for (const event of events) yield event
}

async function collect(values: AsyncIterable<string>): Promise<string[]> {
  const result: string[] = []
  for await (const value of values) result.push(value)
  return result
}
