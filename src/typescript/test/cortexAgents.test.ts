// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import {
  CortexMemory,
  UniversalAgent,
  UniversalTaskCreator,
} from '../src/cortex/agents/index.js'
import { TaskCreator } from '../src/cortex/taskCreator.js'
import type { CortexTask, TaskExecutionContext } from '../src/cortex/task.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'

const LOOKUP_TOOL: ToolDefinition = {
  type: 'function',
  function: {
    name: 'lookup_fact',
    description: 'Look up a fact through an explicitly injected handler.',
    parameters: {
      type: 'object',
      properties: { topic: { type: 'string' } },
      required: ['topic'],
    },
  },
}

test('CortexMemory composes native tiers without ambient persistence and provides bounded task context', () => {
  const memory = new CortexMemory({ enableUser: true, shortTermCapacity: 8 })
  memory.saveTaskResult({
    taskDescription: 'Review Ada implementation details',
    result: 'Ada resolved the implementation issue with a native Bun change.',
    agentRole: 'reviewer',
    importance: 0.9,
    metadata: { source: 'acceptance' },
  })
  memory.saveAgentInteraction({
    agentRole: 'reviewer',
    action: 'verified',
    content: 'the focused Bun test suite',
    importance: 0.7,
  })
  memory.userMemory?.saveMemory('user-1', 'The user prefers concise reports.', {}, { importance: 0.8 })

  const context = memory.buildContextForTask('Review Ada implementation details', {
    agentRole: 'reviewer',
    additionalContext: 'Keep the implementation Bun-native.',
  })

  expect(context).toContain('Background:\nKeep the implementation Bun-native.')
  expect(context).toContain('Recent context:')
  expect(context).toContain('Relevant knowledge:')
  expect(context).toContain('Related memories:')
  expect(memory.getAgentHistory('reviewer')).toContain('Ada resolved the implementation issue with a native Bun change.')
  expect(memory.getUserContext('user-1')).toContain('The user prefers concise reports.')
  expect(memory.getSummary()).toContain('Recent activity:')

  memory.resetAll()
  expect(memory.getAgentHistory('reviewer')).toEqual([])
})

test('UniversalAgent uses only injected LLM and tool ports, then saves an actual result', async () => {
  const client = new ToolThenTextClient()
  const registry = new ToolRegistry()
  const handled: string[] = []
  registry.register(LOOKUP_TOOL, inputs => {
    handled.push(String(inputs.topic))
    return 'Bun-native proof from the injected handler.'
  }, 'universal')
  const memory = new CortexMemory()
  const agent = new UniversalAgent({
    llm: client,
    model: 'gpt-4o',
    tools: [LOOKUP_TOOL],
    toolExecutor: registry,
    permissionMode: 'accept-all',
    memory,
  })

  const result = await agent.execute(executionContext(agent, {
    id: 'task-1',
    description: 'Find the runtime evidence',
    expectedOutput: 'A verified answer',
    importance: 0.8,
  }, 'Output from task setup:\nNative ports are required.'))

  expect(result.output).toBe('The verified answer is Bun-native.')
  expect(result.metadata).toMatchObject({ model: 'gpt-4o', toolCalls: 1, toolFailures: 0 })
  expect(handled).toEqual(['runtime evidence'])
  expect(client.requests).toHaveLength(2)
  expect(client.requests[0]?.tools?.map(tool => tool.function.name)).toEqual(['lookup_fact'])
  expect(client.requests[1]?.messages.some(message => message.role === 'tool')).toBeTrue()
  expect(memory.getAgentHistory(agent.role).join('\n')).toContain('The verified answer is Bun-native.')
  expect(agent.describeCapabilities()).toContain('Total Tools Available: 1')
})

test('UniversalAgent rejects provider failures and does not turn them into task success', async () => {
  const agent = new UniversalAgent({ llm: new FailingClient(), model: 'gpt-4o' })

  await expect(agent.execute(executionContext(agent, {
    id: 'task-2',
    description: 'Do work',
    expectedOutput: 'A result',
  }))).rejects.toThrow('LLM execution failed: provider unavailable')
})

test('UniversalAgent rejects a model request for a tool outside its configured port surface', async () => {
  const agent = new UniversalAgent({ llm: new UnconfiguredToolClient(), model: 'gpt-4o' })

  await expect(agent.execute(executionContext(agent, {
    id: 'task-unsafe-tool',
    description: 'Inspect the workspace',
    expectedOutput: 'A verified inspection',
  }))).rejects.toThrow('LLM requested unconfigured tool: hidden_tool')
})

test('UniversalAgent delegation is an explicit port and never an automatic fallback', async () => {
  const delegated: string[] = []
  const agent = new UniversalAgent({
    llm: new TextClient('Unused direct execution.'),
    model: 'gpt-4o',
    delegation: {
      delegate: async request => {
        delegated.push(`${request.sourceAgentId}:${request.targetAgent}:${request.task.id}`)
        return { output: 'Delegate completed real work.', metadata: { route: 'specialist' } }
      },
    },
  })
  const task: CortexTask = { id: 'delegate-1', description: 'Specialized review', expectedOutput: 'Review result' }
  const result = await agent.delegateTask(task, 'Prior task context', { branch: 'main' }, { targetAgent: 'reviewer' })

  expect(result.output).toBe('Delegate completed real work.')
  expect(delegated).toEqual(['universal:reviewer:delegate-1'])
  await expect(new UniversalAgent({
    llm: new TextClient('unused'),
    model: 'gpt-4o',
    allowDelegation: false,
    delegation: { delegate: async () => 'unexpected' },
  }).delegateTask(task, '', {})).rejects.toThrow('delegation is disabled')
})

test('UniversalTaskCreator assigns matching specialized roles and retains the universal fallback', async () => {
  const universal = new UniversalAgent({ llm: new TextClient('unused'), model: 'gpt-4o' })
  const taskCreator = new TaskCreator({
    generator: async () => `<task_plan>
      <objective>Ship the native port</objective>
      <approach>Research then implement</approach>
      <complexity>medium</complexity>
      <sequential>true</sequential>
      <task id="research">
        <description>Gather evidence</description>
        <expected_output>Evidence</expected_output>
        <agent_role>Research Specialist</agent_role>
        <dependencies></dependencies>
        <context_needed>false</context_needed>
        <tools_needed></tools_needed>
        <importance>0.7</importance>
        <validation_required>false</validation_required>
        <human_feedback>false</human_feedback>
      </task>
      <task id="implement">
        <description>Implement native code</description>
        <expected_output>Native implementation</expected_output>
        <dependencies>research</dependencies>
        <context_needed>true</context_needed>
        <tools_needed></tools_needed>
        <importance>1</importance>
        <validation_required>true</validation_required>
        <human_feedback>false</human_feedback>
      </task>
    </task_plan>`,
  })
  const creator = new UniversalTaskCreator({ taskCreator, universalAgent: universal })

  const result = await creator.createAndAssignTasks({
    prompt: 'Ship the native port',
    specializedAgents: [{ id: 'researcher', role: 'Research Specialist' }],
  })

  expect(result.usedFallback).toBeFalse()
  expect(result.tasks.map(task => task.agentId)).toEqual(['researcher', universal.id])
  expect(result.tasks[1]).toMatchObject({ dependencies: [`${result.plan.id}:research`] })
})

test('UniversalAgent refuses to advertise tools without a real execution port', () => {
  expect(() => new UniversalAgent({
    llm: new TextClient('unused'),
    model: 'gpt-4o',
    tools: [LOOKUP_TOOL],
  })).toThrow('requires toolExecutor')
})

function executionContext(agent: UniversalAgent, task: CortexTask, context = ''): TaskExecutionContext {
  return {
    agent,
    task,
    context,
    inputs: { workspace: '/workspace/xerxes' },
    dependencyOutputs: new Map(),
  }
}

class TextClient implements LlmClient {
  constructor(private readonly text: string) {}

  async *stream(_request: CompletionRequest): AsyncGenerator<LlmDelta> {
    yield { content: this.text }
  }
}

class FailingClient implements LlmClient {
  async *stream(_request: CompletionRequest): AsyncGenerator<LlmDelta> {
    throw new Error('provider unavailable')
  }
}

class ToolThenTextClient implements LlmClient {
  readonly requests: CompletionRequest[] = []

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    if (request.messages.some(message => message.role === 'tool')) {
      yield { content: 'The verified answer is Bun-native.' }
      return
    }
    yield {
      toolCalls: [{
        id: 'lookup-1',
        type: 'function',
        function: { name: 'lookup_fact', arguments: { topic: 'runtime evidence' } },
      }],
    }
  }
}

class UnconfiguredToolClient implements LlmClient {
  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    if (request.messages.some(message => message.role === 'tool')) {
      yield { content: 'I completed the inspection.' }
      return
    }
    yield {
      toolCalls: [{
        id: 'hidden-1',
        type: 'function',
        function: { name: 'hidden_tool', arguments: {} },
      }],
    }
  }
}
