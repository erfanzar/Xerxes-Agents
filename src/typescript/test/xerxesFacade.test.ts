// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  ShortTermMemory,
  SandboxMode,
  SandboxRouter,
  ToolRegistry,
  Xerxes,
  type AgentDefinition,
  type CompletionRequest,
  type LlmClient,
  type LlmDelta,
  type ToolDefinition,
} from '../src/index.js'

const ECHO_TOOL: ToolDefinition = {
  type: 'function',
  function: {
    name: 'Echo',
    description: 'Echo supplied text.',
    parameters: { type: 'object' },
  },
}

class ToolUsingClient implements LlmClient {
  readonly requests: CompletionRequest[] = []

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push({
      ...request,
      messages: request.messages.map(message => ({ ...message })),
    })
    if (request.messages.at(-1)?.role === 'tool') {
      yield { content: 'The native tool completed successfully.', usage: { inputTokens: 7, outputTokens: 5 } }
      return
    }
    yield {
      toolCalls: [{
        id: 'echo_1',
        type: 'function',
        function: { name: 'Echo', arguments: { text: 'hello' } },
      }],
      usage: { inputTokens: 3, outputTokens: 1 },
    }
  }
}

function definition(name: string, options: Partial<AgentDefinition> = {}): AgentDefinition {
  return {
    name,
    description: name + ' agent',
    systemPrompt: 'System prompt for ' + name,
    model: 'gpt-4o',
    tools: [],
    allowedTools: null,
    excludeTools: [],
    source: 'test',
    maxDepth: 5,
    isolation: 'shared',
    ...options,
  }
}

test('Xerxes facade routes an agent through native tools, memory, and a reusable QueryEngine', async () => {
  const client = new ToolUsingClient()
  const registry = new ToolRegistry()
  registry.register(ECHO_TOOL, inputs => 'echo:' + inputs.text)
  const memory = new ShortTermMemory()
  memory.save('Remember the project uses Bun-native tool execution.')
  const xerxes = new Xerxes({
    llm: client,
    model: 'gpt-4o',
    toolRegistry: registry,
    coreTools: false,
    memory,
    memoryMinChars: 1,
    permissionMode: 'accept-all',
    systemPrompt: 'System prompt for default',
  })

  const result = await xerxes.run('Use the project tool now.')
  expect(result.output).toBe('The native tool completed successfully.')
  expect(result.toolCalls).toEqual(['Echo'])
  expect(memory.size).toBe(2)
  expect(client.requests[0]?.tools?.map(tool => tool.function.name)).toEqual(['Echo'])
  expect(client.requests[0]?.messages).toEqual(expect.arrayContaining([
    expect.objectContaining({ role: 'system', content: 'Relevant retained memory for this turn:\n- Remember the project uses Bun-native tool execution.' }),
    expect.objectContaining({ role: 'system', content: 'System prompt for default' }),
  ]))

  await xerxes.run('Use the project tool again.')
  expect(client.requests).toHaveLength(4)
})

test('Xerxes facade exposes capability and error-recovery agent routing', () => {
  const client: LlmClient = {
    async *stream(): AsyncGenerator<LlmDelta> {
      yield { content: 'unused' }
    },
  }
  const xerxes = new Xerxes({ llm: client, coreTools: false })
  xerxes.registerAgent(definition('primary'), {
    capabilities: [{ name: 'build', description: 'build work', performanceScore: 0.7 }],
    fallbackAgentId: 'recovery',
  })
  xerxes.registerAgent(definition('recovery'), {
    capabilities: [{ name: 'repair', description: 'repair work', performanceScore: 0.9 }],
  })

  xerxes.selectAgent('primary')
  expect(xerxes.evaluateAgentSwitch({ required_capability: 'repair' })).toBe('recovery')
  expect(xerxes.currentAgentId).toBe('recovery')
  xerxes.selectAgent('primary')
  expect(xerxes.evaluateAgentSwitch({ execution_error: 'tool failed' })).toBe('recovery')
})

test('fresh facade streams expose the shared event vocabulary and do not retain the temporary engine', async () => {
  const client: LlmClient = {
    async *stream(): AsyncGenerator<LlmDelta> {
      yield { content: 'streaming output' }
    },
  }
  const xerxes = new Xerxes({ llm: client, coreTools: false })
  const stream = xerxes.runStream('stream this', { freshSession: true })
  const events = []
  while (true) {
    const next = await stream.next()
    if (next.done) {
      expect(next.value.output).toBe('streaming output')
      break
    }
    events.push(next.value)
  }
  expect(events.map(event => event.type)).toEqual(['text', 'turn_done'])
})

test('Xerxes facade routes tool calls through an attached sandbox router', async () => {
  const client = new ToolUsingClient()
  const registry = new ToolRegistry()
  let hostCalled = false
  registry.register(ECHO_TOOL, () => {
    hostCalled = true
    return 'host echo'
  })
  const xerxes = new Xerxes({
    llm: client,
    toolRegistry: registry,
    coreTools: false,
    permissionMode: 'accept-all',
    sandboxRouter: new SandboxRouter({
      config: { mode: SandboxMode.STRICT, sandboxedTools: ['Echo'] },
      backend: { execute: async request => 'sandbox:' + request.arguments.text },
    }),
  })

  await xerxes.run('route it')
  expect(hostCalled).toBeFalse()
  expect(client.requests[1]?.messages.at(-1)).toMatchObject({
    role: 'tool',
    content: 'sandbox:hello',
  })
})
