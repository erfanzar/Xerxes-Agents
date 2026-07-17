// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { getEventListeners } from 'node:events'

import { RateLimitError, XerxesTimeoutError } from '../src/core/errors.js'
import {
  CortexAgent,
  abortableDelay,
  createSchemaExample,
  formatOutputGuidance,
} from '../src/cortex/agents/agent.js'
import { CortexMemory } from '../src/cortex/agents/memoryIntegration.js'
import { CortexOrchestrator } from '../src/cortex/orchestrator.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import type { CortexTask } from '../src/cortex/task.js'
import type { JsonSchema, ToolDefinition } from '../src/types/toolCalls.js'

const LOOKUP_TOOL: ToolDefinition = {
  type: 'function',
  function: {
    name: 'lookup_evidence',
    description: 'Look up native execution evidence.',
    parameters: {
      type: 'object',
      properties: { topic: { type: 'string' } },
      required: ['topic'],
    },
  },
}

test('CortexAgent executes a Cortex task through injected LLM/tool ports with memory, knowledge, and stream events', async () => {
  const client = new ToolThenTextClient()
  const memory = new CortexMemory()
  const steps: string[] = []
  const streamEvents: string[] = []
  const registry = new ToolRegistry()
  const agent = new CortexAgent({
    role: 'Native Reviewer',
    goal: 'Verify the native task result',
    backstory: 'A precise Bun-native reviewer.',
    model: 'gpt-test',
    llm: client,
    memory,
    knowledge: { runtime: 'Use only injected native ports.' },
    knowledgeSources: ['acceptance evidence'],
    tools: [LOOKUP_TOOL],
    toolExecutor: registry,
    permissionMode: 'accept-all',
    stepCallback: step => { steps.push(step.step) },
  })
  registry.register(LOOKUP_TOOL, inputs => `evidence:${String(inputs.topic)}`, agent.id)

  const task: CortexTask = {
    id: 'review-native-port',
    description: 'Review the native port',
    expectedOutput: 'A verified structured result',
    importance: 0.8,
  }
  const result = await agent.execute({
    agent,
    task,
    context: 'The task must remain Bun-native.',
    inputs: { branch: 'main' },
    dependencyOutputs: new Map(),
  }, {
    outputSchema: outputSchema(),
    streamCallback: event => { streamEvents.push(event.type) },
  })

  expect(result.output).toBe('Verified completion: the injected evidence proves the Bun-native result.')
  expect(result.metadata).toMatchObject({ attempts: 1, model: 'gpt-test', toolCalls: 1, toolFailures: 0 })
  expect(client.requests).toHaveLength(2)
  expect(client.requests[0]?.messages[0]).toMatchObject({ role: 'system', content: expect.stringContaining('Role: Native Reviewer') })
  const firstPrompt = textContent(client.requests[0]?.messages.at(-1)?.content)
  expect(firstPrompt).toContain('Available knowledge:')
  expect(firstPrompt).toContain('acceptance evidence')
  expect(firstPrompt).toContain('The task must remain Bun-native.')
  expect(firstPrompt).toContain('<json>')
  expect(firstPrompt).toContain('"summary"')
  expect(streamEvents).toEqual(expect.arrayContaining(['tool_start', 'tool_end', 'text', 'turn_done']))
  expect(steps).toEqual(['execution_start', 'execution_complete'])
  expect(memory.getAgentHistory(agent.role).join('\n')).toContain('Verified completion: the injected evidence')
  expect(agent.getExecutionStats()).toMatchObject({ timesExecuted: 1 })
})

test('CortexAgent retries only terminal provider failures and surfaces a real successful second attempt', async () => {
  const client = new FailsOnceClient()
  const callbacks: string[] = []
  const agent = new CortexAgent({
    role: 'Retry Agent',
    goal: 'Recover provider execution',
    backstory: 'Uses bounded retries.',
    model: 'gpt-test',
    llm: client,
    maxIterations: 2,
    executionRetryDelays: [0],
    stepCallback: step => { callbacks.push(step.step) },
  })

  await expect(agent.execute('Retry the provider safely')).resolves.toBe('Verified completion after retry.')
  expect(client.requests).toHaveLength(2)
  expect(callbacks).toEqual(['execution_start', 'retry', 'execution_complete'])
  expect(agent.getExecutionStats()).toMatchObject({ timesExecuted: 1 })
})

test('abortableDelay removes its abort listener whether it resolves or rejects', async () => {
  const resolved = new AbortController()
  await abortableDelay(5, resolved.signal)
  expect(getEventListeners(resolved.signal, 'abort')).toHaveLength(0)

  const aborted = new AbortController()
  const pending = abortableDelay(10_000, aborted.signal)
  aborted.abort(new Error('stop waiting'))
  await expect(pending).rejects.toThrow('stop waiting')
  expect(getEventListeners(aborted.signal, 'abort')).toHaveLength(0)
})

test('CortexAgent propagates timeout/cancellation through the injected LLM signal and never reports success', async () => {
  const agent = new CortexAgent({
    role: 'Timed Agent',
    goal: 'Respect the task deadline',
    backstory: 'Stops when the host cancels it.',
    model: 'gpt-test',
    llm: new BlockingClient(),
    maxExecutionTime: 0.01,
    maxIterations: 1,
  })

  await expect(agent.execute('Wait forever')).rejects.toBeInstanceOf(XerxesTimeoutError)
  expect(agent.getExecutionStats().timesExecuted).toBe(1)

  const controller = new AbortController()
  controller.abort(new Error('caller cancelled'))
  await expect(agent.execute('Do not start', { signal: controller.signal })).rejects.toThrow('caller cancelled')
})

test('CortexAgent tracks RPM in a caller-controlled clock and exposes explicit delegation only', async () => {
  let now = 1_000
  const delegated: string[] = []
  const agent = new CortexAgent({
    role: 'Rate Agent',
    goal: 'Respect host rate limits',
    backstory: 'Delegates only when asked.',
    model: 'gpt-test',
    llm: new StaticTextClient('Verified completion.'),
    maxRpm: 1,
    now: () => now,
    allowDelegation: true,
    delegation: {
      delegate: request => {
        delegated.push(`${request.sourceAgentId}:${request.targetAgentId}:${request.task.id}`)
        return { output: 'A specialist completed the delegated review.' }
      },
    },
  })

  await expect(agent.execute('First request')).resolves.toBe('Verified completion.')
  expect(agent.getRateLimitStatus()).toEqual({ currentRequests: 1, maxRpm: 1, rateLimited: true, requestsRemaining: 0 })
  await expect(agent.execute('Second request')).rejects.toBeInstanceOf(RateLimitError)
  now += 60_001
  await expect(agent.execute('Request after the next minute')).resolves.toBe('Verified completion.')

  const delegatedResult = await agent.delegateTask('Review a specialist-only concern', {
    targetAgentId: 'specialist',
    inputs: { branch: 'main' },
  })
  expect(delegatedResult).toBe('A specialist completed the delegated review.')
  expect(delegated).toEqual([`${agent.id}:specialist:${agent.id}-delegation-1`])

  const noDelegate = new CortexAgent({
    role: 'No Delegate',
    goal: 'Remain local',
    backstory: 'Does not delegate.',
    model: 'gpt-test',
    llm: new StaticTextClient('unused'),
  })
  await expect(noDelegate.delegateTask('Cannot hand off')).rejects.toThrow('delegation is disabled')
})

test('CortexAgent interpolates its prompt fields, generates schema guidance, and plugs into CortexOrchestrator', async () => {
  const agent = new CortexAgent({
    role: '{domain} Reviewer',
    goal: 'Review {domain} changes',
    backstory: 'A specialist for {domain}.',
    model: 'gpt-test',
    llm: new StaticTextClient('Verified completion.'),
  })
  agent.interpolateInputs({ domain: 'TypeScript' })
  expect(agent.role).toBe('TypeScript Reviewer')
  expect(agent.goal).toBe('Review TypeScript changes')
  expect(agent.instructions).toContain('TypeScript Reviewer')

  const schema: JsonSchema = {
    title: 'ReviewOutput',
    type: 'object',
    properties: {
      findings: {
        type: 'array',
        minItems: 2,
        items: { $ref: '#/$defs/finding' },
      },
    },
    $defs: {
      finding: {
        type: 'object',
        properties: { title: { type: 'string' }, severity: { enum: ['low', 'high'] } },
      },
    },
  }
  expect(createSchemaExample(schema)).toEqual({
    findings: [
      { title: 'Example Title', severity: 'low' },
      { title: 'Example Title', severity: 'low' },
    ],
  })
  expect(createSchemaExample({
    allOf: [
      { type: 'object', properties: { title: { type: 'string' } } },
      { type: 'object', properties: { verified: { type: 'boolean' } } },
    ],
  })).toEqual({ title: 'Example Title', verified: true })
  expect(formatOutputGuidance(schema, 'json')).toContain('ReviewOutput')

  const orchestrator = new CortexOrchestrator({
    agents: [agent],
    tasks: [{
      id: 'orchestrated',
      agentId: agent.id,
      description: 'Review the task contract',
      expectedOutput: 'Verified output',
    }],
  })
  const output = await orchestrator.run()
  expect(output.rawOutput).toBe('Verified completion.')
})

function outputSchema(): JsonSchema {
  return {
    title: 'ReviewResult',
    type: 'object',
    properties: {
      summary: { type: 'string', description: 'concise review summary' },
      verified: { type: 'boolean' },
    },
  }
}

function textContent(content: unknown): string {
  if (typeof content === 'string') return content
  return Array.isArray(content)
    ? content.flatMap(part => typeof part === 'object' && part !== null && 'text' in part && typeof part.text === 'string'
      ? [part.text]
      : []).join('')
    : ''
}

class ToolThenTextClient implements LlmClient {
  readonly requests: CompletionRequest[] = []

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    if (request.messages.some(message => message.role === 'tool')) {
      yield { content: 'Verified completion: the injected evidence proves the Bun-native result.' }
      return
    }
    yield {
      toolCalls: [{
        id: 'evidence-1',
        type: 'function',
        function: { name: 'lookup_evidence', arguments: { topic: 'native port' } },
      }],
    }
  }
}

class FailsOnceClient implements LlmClient {
  readonly requests: CompletionRequest[] = []

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    if (this.requests.length === 1) throw new Error('temporary provider outage')
    yield { content: 'Verified completion after retry.' }
  }
}

class BlockingClient implements LlmClient {
  async *stream(_request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
    await new Promise<void>((_resolve, reject) => {
      if (signal?.aborted) {
        reject(signal.reason)
        return
      }
      signal?.addEventListener('abort', () => reject(signal.reason), { once: true })
    })
    yield { content: 'This must not be emitted.' }
  }
}

class StaticTextClient implements LlmClient {
  constructor(private readonly text: string) {}

  async *stream(_request: CompletionRequest): AsyncGenerator<LlmDelta> {
    yield { content: this.text }
  }
}
