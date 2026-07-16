// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import type { ToolExecutor } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { LoopDetector } from '../src/runtime/loopDetector.js'
import { PolicyEngine, ToolPolicy } from '../src/security/policy.js'
import { createAgentState, type StreamEvent } from '../src/streaming/events.js'
import { runTurn } from '../src/streaming/loop.js'
import type { ToolCall, ToolDefinition } from '../src/types/toolCalls.js'

const READ_FILE: ToolDefinition = {
  type: 'function',
  function: {
    name: 'ReadFile',
    description: 'Read a file.',
    parameters: { type: 'object', properties: { path: { type: 'string' } } },
  },
}

function readFileCall(id: string, path = 'README.md'): ToolCall {
  return {
    id,
    type: 'function',
    function: { name: 'ReadFile', arguments: { path } },
  }
}

async function collect(events: AsyncIterable<StreamEvent>): Promise<StreamEvent[]> {
  const result: StreamEvent[] = []
  for await (const event of events) result.push(event)
  return result
}

class ToolThenTextClient implements LlmClient {
  readonly requests: CompletionRequest[] = []

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    if (this.requests.length === 1) {
      yield {
        toolCalls: [readFileCall('call-read')],
        usage: { inputTokens: 5, outputTokens: 6, cacheReadTokens: 3, cacheCreationTokens: 2 },
      }
      return
    }
    yield { content: 'done', usage: { inputTokens: 7, outputTokens: 8 } }
  }
}

const successfulExecutor: ToolExecutor = {
  async execute(): Promise<string> {
    return 'tool result'
  },
}

test('streaming loop inserts drained steering before the following provider request', async () => {
  const client = new ToolThenTextClient()
  const batches = [[], ['please reconsider the approach'], []] as string[][]
  const drained: string[][] = []
  const state = createAgentState()

  await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: [READ_FILE],
    userMessage: 'inspect the project',
  }, {
    drainSteer: () => {
      const batch = batches.shift() ?? []
      drained.push(batch)
      return batch
    },
    llm: client,
    toolExecutor: successfulExecutor,
  }))

  expect(drained).toEqual([[], ['please reconsider the approach'], []])
  expect(client.requests).toHaveLength(2)
  expect(client.requests[0]?.messages.some(message => String(message.content).includes('please reconsider'))).toBe(false)
  expect(client.requests[1]?.messages).toContainEqual({
    role: 'user',
    content: '[steer from user]\nplease reconsider the approach',
  })
})

test('streaming loop saves a steer that arrives with a terminal text response for the next turn', async () => {
  const state = createAgentState()
  const batches = [[], ['make a follow-up todo']] as string[][]
  const events = await collect(runTurn({
    model: 'gpt-4o',
    state,
    userMessage: 'finish the task',
  }, {
    drainSteer: () => batches.shift() ?? [],
    llm: {
      async *stream(): AsyncGenerator<LlmDelta> {
        yield { content: 'finished' }
      },
    },
  }))

  expect(events).toContainEqual({ type: 'text', text: '\n[Steer saved for next turn: make a follow-up todo]' })
  expect(state.messages).toContainEqual({
    role: 'user',
    content: '[steer from user saved for next turn]\nmake a follow-up todo',
  })
})

test('streaming loop injects passive sub-agent events through the explicit native dependency', async () => {
  const client = new ToolThenTextClient()
  const batches = [[], ['[agent researcher] completed source scan']] as string[][]
  const state = createAgentState()

  await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: [READ_FILE],
    userMessage: 'research the project',
  }, {
    drainAgentEvents: () => batches.shift() ?? [],
    llm: client,
    toolExecutor: successfulExecutor,
  }))

  expect(client.requests[1]?.messages).toContainEqual({
    role: 'user',
    content: '[sub-agent events]\n[agent researcher] completed source scan',
  })
})

test('streaming loop waits for detached subagents and resumes the same parent turn with their results', async () => {
  const requests: CompletionRequest[] = []
  let waits = 0
  const state = createAgentState()

  const events = await collect(runTurn({
    model: 'gpt-4o',
    state,
    userMessage: 'run the parallel review',
  }, {
    awaitAgentEvents: async () => waits++ === 0
      ? ['[agent result title="Review" status=completed]\nreview findings\n[/agent result]']
      : [],
    llm: {
      async *stream(request): AsyncGenerator<LlmDelta> {
        requests.push(request)
        yield { content: requests.length === 1 ? 'The review is running.' : 'Integrated review findings.' }
      },
    },
  }))

  expect(requests).toHaveLength(2)
  expect(requests[1]?.messages).toContainEqual({
    role: 'user',
    content:
      '[sub-agent events]\n' +
      '[agent result title="Review" status=completed]\nreview findings\n[/agent result]',
  })
  expect(state.turnCount).toBe(1)
  expect(state.messages.filter(message => message.role === 'user')).toHaveLength(2)
  expect(events.filter(event => event.type === 'text').map(event => event.text)).toEqual([
    'The review is running.',
    'Integrated review findings.',
  ])
})

test('streaming loop joins detached subagents before a final tool-turn boundary can end the parent', async () => {
  const requests: CompletionRequest[] = []
  let waits = 0
  const state = createAgentState()

  const events = await collect(runTurn({
    maxToolTurns: 1,
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: [READ_FILE],
    userMessage: 'delegate at the turn boundary',
  }, {
    awaitAgentEvents: async () => waits++ === 0
      ? ['[agent result title="Boundary child" status=completed]\nchild report\n[/agent result]']
      : [],
    llm: {
      async *stream(request): AsyncGenerator<LlmDelta> {
        requests.push(request)
        if (requests.length === 1) {
          yield { toolCalls: [readFileCall('spawn-at-boundary')] }
          return
        }
        yield { content: 'Integrated boundary child report.' }
      },
    },
    toolExecutor: successfulExecutor,
  }))

  expect(waits).toBe(2)
  expect(requests).toHaveLength(2)
  expect(requests[1]?.tools).toBeUndefined()
  expect(requests[1]?.messages).toContainEqual({
    role: 'user',
    content:
      '[sub-agent events]\n' +
      '[agent result title="Boundary child" status=completed]\nchild report\n[/agent result]',
  })
  expect(events.filter(event => event.type === 'text').map(event => event.text)).toEqual([
    'Integrated boundary child report.',
  ])
  expect(events.at(-1)).toMatchObject({ type: 'turn_done' })
})

test('default streaming loop executes more than twenty-five distinct tool calls', async () => {
  const requests: CompletionRequest[] = []
  const executed: string[] = []
  const events = await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state: createAgentState(),
    tools: [READ_FILE],
    userMessage: 'read every distinct file needed',
  }, {
    llm: {
      async *stream(request): AsyncGenerator<LlmDelta> {
        requests.push(request)
        if (requests.length <= 30) {
          yield { toolCalls: [readFileCall(`default-${requests.length}`, `file-${requests.length}.ts`)] }
          return
        }
        yield { content: 'Completed thirty distinct reads.' }
      },
    },
    toolExecutor: {
      async execute(call): Promise<string> {
        executed.push(call.id)
        return 'tool result'
      },
    },
  }))

  expect(requests).toHaveLength(31)
  expect(executed).toEqual(Array.from({ length: 30 }, (_, index) => `default-${index + 1}`))
  expect(events.filter(event => event.type === 'tool_start')).toHaveLength(30)
  expect(events.filter(event => event.type === 'tool_end')).toHaveLength(30)
  expect(events).toContainEqual({ type: 'text', text: 'Completed thirty distinct reads.' })
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', toolCallsCount: 30 })
})

test('default tool turns can exceed fifty without a hidden round ceiling', async () => {
  class RepeatingToolClient implements LlmClient {
    calls = 0

    async *stream(): AsyncGenerator<LlmDelta> {
      this.calls += 1
      if (this.calls <= 52) {
        yield { toolCalls: [readFileCall(`call-${this.calls}`, `file-${this.calls}.ts`)] }
        return
      }
      yield { content: 'Finished after fifty-two productive tool rounds.' }
    }
  }

  const client = new RepeatingToolClient()
  const executed: string[] = []
  const events = await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state: createAgentState(),
    tools: [READ_FILE],
    userMessage: 'continue reading',
  }, {
    llm: client,
    toolExecutor: {
      async execute(call): Promise<string> {
        executed.push(call.id)
        return 'tool result'
      },
    },
  }))

  expect(client.calls).toBe(53)
  expect(executed).toEqual(Array.from({ length: 52 }, (_, index) => `call-${index + 1}`))
  expect(events.filter(event => event.type === 'tool_start')).toHaveLength(52)
  expect(events.filter(event => event.type === 'tool_end')).toHaveLength(52)
  expect(events).toContainEqual({ type: 'text', text: 'Finished after fifty-two productive tool rounds.' })
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', toolCallsCount: 52 })
})

test('retained same-call protection stops repeated work and forces a tool-free summary', async () => {
  const requests: CompletionRequest[] = []
  const executed: string[] = []
  const events = await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state: createAgentState(),
    tools: [READ_FILE],
    userMessage: 'do not repeat the same work forever',
  }, {
    llm: {
      async *stream(request): AsyncGenerator<LlmDelta> {
        requests.push(request)
        if (requests.length <= 5) {
          yield { toolCalls: [readFileCall(`repeat-${requests.length}`)] }
          return
        }
        yield { content: 'Summarized after stopping the repeated call.' }
      },
    },
    toolExecutor: {
      async execute(call): Promise<string> {
        executed.push(call.id)
        return 'same evidence'
      },
    },
  }))

  expect(executed).toEqual(['repeat-1', 'repeat-2', 'repeat-3', 'repeat-4'])
  expect(requests).toHaveLength(6)
  expect(requests[5]?.tools).toBeUndefined()
  expect(requests[5]?.messages).toContainEqual(expect.objectContaining({
    role: 'user',
    content: expect.stringContaining('Tool loop detected (same_call)'),
  }))
  expect(events.filter(event => event.type === 'tool_start')).toHaveLength(4)
  expect(events.filter(event => event.type === 'tool_end')).toHaveLength(5)
  expect(events).toContainEqual({ type: 'text', text: 'Summarized after stopping the repeated call.' })
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', toolCallsCount: 5 })
})

test('tool-call budget exhaustion allows the configured maximum then forces one tool-free summary', async () => {
  const requests: CompletionRequest[] = []
  const executed: string[] = []
  const events = await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state: createAgentState(),
    tools: [READ_FILE],
    userMessage: 'inspect without looping forever',
  }, {
    llm: {
      async *stream(request): AsyncGenerator<LlmDelta> {
        requests.push(request)
        if (requests.length < 3) {
          yield { toolCalls: [readFileCall(`budget-${requests.length}`)] }
          return
        }
        yield { content: 'Final summary from completed work.' }
      },
    },
    loopDetector: new LoopDetector({ maxToolCallsPerTurn: 1 }),
    toolExecutor: {
      async execute(call): Promise<string> {
        executed.push(call.id)
        return 'evidence'
      },
    },
  }))

  expect(executed).toEqual(['budget-1'])
  expect(requests).toHaveLength(3)
  expect(requests[0]?.tools?.map(tool => tool.function.name)).toEqual(['ReadFile'])
  expect(requests[1]?.tools?.map(tool => tool.function.name)).toEqual(['ReadFile'])
  expect(requests[2]?.tools).toBeUndefined()
  expect(requests[2]?.messages).toContainEqual(expect.objectContaining({
    role: 'user',
    content: expect.stringContaining('[Tool loop stopped]'),
  }))
  expect(events.filter(event => event.type === 'tool_end')).toHaveLength(2)
  expect(events).toContainEqual({ type: 'text', text: 'Final summary from completed work.' })
})

test('streaming loop keeps the full tool result, timing, and cache usage in its native state', async () => {
  const fullResult = 'x'.repeat(8_000)
  const state = createAgentState()
  const events = await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: [READ_FILE],
    userMessage: 'read the large file',
  }, {
    llm: new ToolThenTextClient(),
    toolExecutor: {
      async execute(): Promise<string> {
        return fullResult
      },
    },
  }))

  const toolEnd = events.find(event => event.type === 'tool_end')
  expect(toolEnd).toMatchObject({
    type: 'tool_end',
    result: { result: fullResult, toolCallId: 'call-read', permitted: true },
  })
  if (!toolEnd || toolEnd.type !== 'tool_end') throw new Error('expected one tool result')
  expect(toolEnd.result.durationMs).toBeGreaterThanOrEqual(0)
  expect(state.messages).toContainEqual({
    role: 'tool',
    name: 'ReadFile',
    tool_call_id: 'call-read',
    content: fullResult,
  })
  expect(state.totalInputTokens).toBe(12)
  expect(state.totalOutputTokens).toBe(14)
  expect(state.totalCacheReadTokens).toBe(3)
  expect(state.totalCacheCreationTokens).toBe(2)
})

test('unconfigured model tool calls emit an explicit error and let the model recover', async () => {
  const state = createAgentState()
  const requests: CompletionRequest[] = []
  const hiddenCall: ToolCall = {
    id: 'call-hidden',
    type: 'function',
    function: { name: 'hidden_tool', arguments: { value: 'unexpected' } },
  }
  const events = await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: [READ_FILE],
    userMessage: 'finish without hidden tools',
  }, {
    llm: {
      async *stream(request): AsyncGenerator<LlmDelta> {
        requests.push(request)
        if (requests.length === 1) {
          yield { toolCalls: [hiddenCall] }
          return
        }
        yield { content: 'recovered without the unavailable tool' }
      },
    },
  }))

  expect(requests).toHaveLength(2)
  expect(events).toContainEqual(expect.objectContaining({
    type: 'tool_end',
    result: expect.objectContaining({
      name: 'hidden_tool',
      permitted: true,
      result: 'Tool execution failed: hidden_tool was not configured for this turn.',
      toolCallId: 'call-hidden',
    }),
  }))
  expect(requests[1]?.messages).toContainEqual({
    role: 'tool',
    name: 'hidden_tool',
    tool_call_id: 'call-hidden',
    content: 'Tool execution failed: hidden_tool was not configured for this turn.',
  })
  expect(state.messages).toContainEqual(expect.objectContaining({
    role: 'assistant',
    tool_calls: [hiddenCall],
  }))
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', toolCallsCount: 1 })
})

test('unconfigured-tool stop hooks still publish the failure before terminating the turn', async () => {
  const events = await collect(runTurn({
    model: 'gpt-4o',
    state: createAgentState(),
    tools: [READ_FILE],
    userMessage: 'do not use hidden tools',
  }, {
    llm: {
      async *stream(): AsyncGenerator<LlmDelta> {
        yield {
          toolCalls: [{
            id: 'call-hidden-stop',
            type: 'function',
            function: { name: 'hidden_tool', arguments: {} },
          }],
        }
      },
    },
    onUnconfiguredToolCalls: () => 'stop',
  }))

  expect(events).toContainEqual(expect.objectContaining({
    type: 'tool_end',
    result: expect.objectContaining({
      result: 'Tool execution failed: hidden_tool was not configured for this turn.',
      toolCallId: 'call-hidden-stop',
    }),
  }))
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', toolCallsCount: 1 })
})

test('streaming loop backfills unrun tool calls when cancellation lands between tools', async () => {
  const controller = new AbortController()
  const executed: string[] = []
  const state = createAgentState()
  const events = await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: [READ_FILE],
    userMessage: 'read both files',
  }, {
    llm: {
      async *stream(): AsyncGenerator<LlmDelta> {
        yield { toolCalls: [readFileCall('call-first'), readFileCall('call-second')] }
      },
    },
    toolExecutor: {
      async execute(call): Promise<string> {
        executed.push(call.id)
        controller.abort('cancel after first tool')
        return 'first result'
      },
    },
  }, controller.signal))

  expect(executed).toEqual(['call-first'])
  expect(events.filter(event => event.type === 'tool_end')).toEqual([
    expect.objectContaining({ result: expect.objectContaining({ toolCallId: 'call-first', result: 'first result' }) }),
    expect.objectContaining({ result: expect.objectContaining({ toolCallId: 'call-second', result: 'Cancelled before execution.' }) }),
  ])
  expect(state.messages.filter(message => message.role === 'tool')).toHaveLength(2)
})

test('streaming loop retries transient provider failures through its injected retry policy', async () => {
  class FlakyClient implements LlmClient {
    calls = 0

    async *stream(): AsyncGenerator<LlmDelta> {
      this.calls += 1
      if (this.calls < 3) throw new Error(`transient-${this.calls}`)
      yield { content: 'recovered response', usage: { inputTokens: 9, outputTokens: 4, reasoningTokens: 2 } }
    }
  }

  const client = new FlakyClient()
  const delays: number[] = []
  const state = createAgentState()
  const events = await collect(runTurn({
    model: 'gpt-4o',
    state,
    userMessage: 'retry the request',
  }, {
    delay: async milliseconds => {
      delays.push(milliseconds)
    },
    llm: client,
    retryDelays: [0, 0],
  }))

  expect(client.calls).toBe(3)
  expect(delays).toEqual([0, 0])
  expect(events.filter(event => event.type === 'provider_retry')).toEqual([
    expect.objectContaining({ attempt: 1, final: false, error: 'transient-1' }),
    expect.objectContaining({ attempt: 2, final: false, error: 'transient-2' }),
  ])
  expect(events.at(-1)).toMatchObject({
    type: 'turn_done',
    apiCallsCount: 3,
    usageComplete: false,
  })
  expect(state.totalApiCalls).toBe(3)
  expect(state.apiCallsComplete).toBe(true)
  expect(state.usageComplete).toBe(false)
  expect(state.messages.at(-1)).toMatchObject({ role: 'assistant', content: 'recovered response' })
})

test('manual permission denial emits a structured result and does not execute the tool', async () => {
  let executions = 0
  const events = await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'manual',
    state: createAgentState(),
    tools: [READ_FILE],
    userMessage: 'read the file',
  }, {
    llm: new ToolThenTextClient(),
    toolExecutor: {
      async execute(): Promise<string> {
        executions += 1
        return 'unreachable'
      },
    },
  }))

  expect(executions).toBe(0)
  expect(events.some(event => event.type === 'permission_request')).toBe(true)
  expect(events).toContainEqual(expect.objectContaining({
    type: 'tool_end',
    result: expect.objectContaining({ toolCallId: 'call-read', permitted: false }),
  }))
})

test('static policy denial cannot be overridden by the permission broker', async () => {
  let approvals = 0
  let executions = 0
  const events = await collect(runTurn({
    agentId: 'restricted',
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state: createAgentState(),
    tools: [READ_FILE],
    userMessage: 'read the file',
  }, {
    llm: new ToolThenTextClient(),
    permissionBroker: {
      async request() {
        approvals += 1
        return 'approve'
      },
    },
    policy: new PolicyEngine({
      agentPolicies: { restricted: new ToolPolicy({ deny: ['ReadFile'] }) },
    }),
    toolExecutor: {
      async execute(): Promise<string> {
        executions += 1
        return 'unreachable'
      },
    },
  }))

  expect(approvals).toBe(0)
  expect(executions).toBe(0)
  expect(events.some(event => event.type === 'permission_request')).toBe(false)
  expect(events).toContainEqual(expect.objectContaining({
    type: 'tool_end',
    result: expect.objectContaining({ toolCallId: 'call-read', permitted: false }),
  }))
  expect(events.at(-1)).toEqual(expect.objectContaining({ type: 'turn_done' }))
})

test('cancellation while awaiting an injected approval broker never starts the tool', async () => {
  const controller = new AbortController()
  let executions = 0
  const events = await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'manual',
    state: createAgentState(),
    tools: [READ_FILE],
    userMessage: 'read the file',
  }, {
    llm: new ToolThenTextClient(),
    permissionBroker: {
      async request() {
        controller.abort('cancel while approval is open')
        return 'approve'
      },
    },
    toolExecutor: {
      async execute(): Promise<string> {
        executions += 1
        return 'unreachable'
      },
    },
  }, controller.signal))

  expect(executions).toBe(0)
  expect(events.some(event => event.type === 'permission_request')).toBe(true)
  expect(events.some(event => event.type === 'tool_start')).toBe(false)
  expect(events).toContainEqual(expect.objectContaining({
    type: 'tool_end',
    result: expect.objectContaining({
      permitted: false,
      result: 'Cancelled before execution.',
      toolCallId: 'call-read',
    }),
  }))
  expect(events.at(-1)).toEqual(expect.objectContaining({ type: 'turn_done' }))
})
