// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { BUILTIN_AGENTS, type AgentDefinition } from '../src/agents/definitions.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { createNativeSubagentHost } from '../src/daemon/subagentHost.js'
import type { DaemonEvent, DaemonSession } from '../src/daemon/runtime.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { AGENT_MEMORY_WRITE_DEFINITION } from '../src/tools/agentMemoryTools.js'
import { registerClaudeAgentTools } from '../src/tools/claudeTools/agentOps.js'
import type { JsonObject, ToolCall, ToolDefinition } from '../src/types/toolCalls.js'

function toolCall(name: string, arguments_: JsonObject): ToolCall {
  return { id: crypto.randomUUID(), type: 'function', function: { name, arguments: arguments_ } }
}

function agentDefinition(name: string): AgentDefinition {
  return {
    allowedTools: null,
    description: `${name} test agent`,
    excludeTools: [],
    isolation: '',
    maxDepth: 3,
    model: '',
    name,
    source: 'test',
    systemPrompt: `You are the ${name} test agent.`,
    tools: [],
  }
}

function creatorDefinition(...children: readonly string[]): AgentDefinition {
  return {
    ...agentDefinition('default'),
    subagents: Object.freeze(Object.fromEntries(children.map(name => [
      name,
      Object.freeze({ path: `${name}.yaml`, description: `${name} child` }),
    ]))),
  }
}

function session(id = 'parent-session'): DaemonSession {
  return {
    activeTurnId: 'parent-turn',
    agentId: 'default',
    cancelRequested: false,
    cwd: process.cwd(),
    extra: {},
    id,
    interactionMode: 'code',
    lastActive: 0,
    messages: [],
    metadata: {},
    model: 'test-model',
    planMode: false,
    sessionKey: id,
    status: 'working',
    thinkingContent: [],
    toolExecutions: [],
    totalInputTokens: 0,
    totalOutputTokens: 0,
    turnCount: 0,
    workspace: `/tmp/agents/${id}`,
  }
}

async function waitFor(predicate: () => boolean, timeoutMs = 1_000): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error(`condition was not met within ${timeoutMs}ms`)
    await Bun.sleep(2)
  }
}

test('daemon subagent event bus isolates sessions and removes unsubscribed listeners', () => {
  const bus = new DaemonSubagentEventBus()
  const first: DaemonEvent[] = []
  const second: DaemonEvent[] = []
  const releaseFirst = bus.subscribe('session-a', event => first.push(event))
  bus.subscribe('session-b', event => second.push(event))
  const event: DaemonEvent = { type: 'subagent_event', payload: { agent_id: 'child-a' } }

  bus.publish('session-a', event)
  expect(first).toEqual([event])
  expect(second).toEqual([])

  releaseFirst()
  bus.publish('session-a', { type: 'subagent_event', payload: { agent_id: 'late-child' } })
  bus.publish('session-b', { type: 'subagent_event', payload: { agent_id: 'child-b' } })
  expect(first).toEqual([event])
  expect(second).toEqual([{ type: 'subagent_event', payload: { agent_id: 'child-b' } }])
})

test('daemon subagent event bus replays events emitted between parent turns exactly once', () => {
  const bus = new DaemonSubagentEventBus()
  const event: DaemonEvent = {
    type: 'subagent_event',
    payload: { agent_id: 'background-child', event: { type: 'turn_end', payload: { status: 'completed' } } },
  }
  bus.publish('session-a', event)

  const replayed: DaemonEvent[] = []
  const release = bus.subscribe('session-a', item => replayed.push(item))
  expect(replayed).toEqual([event])
  release()

  const duplicate: DaemonEvent[] = []
  bus.subscribe('session-a', item => duplicate.push(item))
  expect(duplicate).toEqual([])
})

class ToolCapturingParentClient implements LlmClient {
  requests: CompletionRequest[] = []

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    yield { content: 'ready' }
  }
}

test('agent turn runner exposes registered delegation tools to the provider', async () => {
  const client = new ToolCapturingParentClient()
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: BUILTIN_AGENTS,
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: client,
    model: 'test-model',
    permissionMode: 'auto',
    toolExecutor: registry,
    tools: registry.definitions(),
  })
  registerClaudeAgentTools(registry, { manager: host.managerPort })
  const runner = new AgentTurnRunner({
    agentDefinitions: BUILTIN_AGENTS,
    llm: client,
    model: 'test-model',
    permissionMode: 'auto',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    for await (const _event of runner.run(session(), 'delegate if useful', new AbortController().signal)) {
      // Consume the complete turn so the captured request reflects production iteration.
    }
    const names = client.requests[0]?.tools?.map(tool => tool.function.name) ?? []
    expect(names).toContain('AgentTool')
    expect(names).toContain('SpawnAgents')
    expect(names).toContain('AwaitAgents')
    expect(names).toContain('SendMessageTool')
  } finally {
    await host.manager.shutdown()
  }
})

test('agent turn runner refreshes the model-visible mode overlay on an existing session', async () => {
  const client = new ToolCapturingParentClient()
  const activeSession = session('mode-session')
  const modeTool: ToolDefinition = {
    type: 'function',
    function: { name: 'SetInteractionModeTool', description: 'switch modes', parameters: {} },
  }
  const runner = new AgentTurnRunner({
    agentDefinitions: BUILTIN_AGENTS,
    llm: client,
    model: 'test-model',
    tools: [modeTool],
  })

  for await (const _event of runner.run(activeSession, 'inspect the implementation', new AbortController().signal)) {
    // Consume the first turn.
  }
  activeSession.interactionMode = 'researcher'
  activeSession.planMode = false
  for await (const _event of runner.run(activeSession, 'continue in research mode', new AbortController().signal)) {
    // Consume the second turn after the mode change.
  }

  const firstSystem = client.requests[0]?.messages.filter(message => message.role === 'system') ?? []
  const secondSystem = client.requests[1]?.messages.filter(message => message.role === 'system') ?? []
  expect(firstSystem).toHaveLength(1)
  expect(secondSystem).toHaveLength(1)
  expect(String(firstSystem[0]?.content)).toContain('Use code mode for normal implementation.')
  expect(String(firstSystem[0]?.content)).not.toContain('You are in researcher mode.')
  expect(String(secondSystem[0]?.content)).toContain('You are in researcher mode.')
  expect(String(secondSystem[0]?.content)).not.toContain('Use code mode for normal implementation.')
})

test('native subagent host rejects unknown profiles instead of granting a generic child surface', async () => {
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([['coder', agentDefinition('coder')]]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: new ToolCapturingParentClient(),
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    await expect(host.managerPort.spawn({
      message: 'review the change',
      promptProfile: 'review',
      title: 'Review change',
    })).rejects.toThrow('is not a registered agent profile')
    expect(host.managerPort.listHandles()).toEqual([])
  } finally {
    await host.manager.shutdown()
  }
})

test('native subagent host enforces the creator profile child catalog after alias resolution', async () => {
  const registry = new ToolRegistry()
  const researcher = agentDefinition('researcher')
  const creator: AgentDefinition = {
    ...agentDefinition('default'),
    subagents: Object.freeze({
      researcher: Object.freeze({ path: 'researcher.yaml', description: 'read-only research' }),
    }),
  }
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([
      ['coder', agentDefinition('coder')],
      ['default', creator],
      ['researcher', researcher],
    ]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: new ToolCapturingParentClient(),
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    await expect(host.managerPort.spawn({
      creatorAgentId: 'default',
      message: 'implement it',
      promptProfile: 'general-purpose',
      title: 'Implement change',
    })).rejects.toThrow("is not allowed by agent 'default'")
    expect(host.managerPort.listHandles()).toEqual([])
  } finally {
    await host.manager.shutdown()
  }
})

test('native subagent host rejects unknown creators and creators without a child catalog', async () => {
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([
      ['coder', agentDefinition('coder')],
      ['default', agentDefinition('default')],
    ]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: new ToolCapturingParentClient(),
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    await expect(host.managerPort.spawn({
      creatorAgentId: 'missing-parent',
      message: 'implement it',
      promptProfile: 'coder',
      title: 'Implement change',
    })).rejects.toThrow('is not a registered agent profile')
    await expect(host.managerPort.spawn({
      creatorAgentId: 'default',
      message: 'implement it',
      promptProfile: 'coder',
      title: 'Implement change',
    })).rejects.toThrow("allowed profiles: (none)")
    expect(host.managerPort.listHandles()).toEqual([])
  } finally {
    await host.manager.shutdown()
  }
})

class ConcurrentChildClient implements LlmClient {
  active = 0
  calls = 0
  maxActive = 0
  private readonly bothStarted = Promise.withResolvers<void>()

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.calls += 1
    this.active += 1
    this.maxActive = Math.max(this.maxActive, this.active)
    if (this.calls === 2) this.bothStarted.resolve()
    try {
      await Promise.race([
        this.bothStarted.promise,
        Bun.sleep(500).then(() => {
          throw new Error('second subagent did not start concurrently')
        }),
      ])
      const prompt = request.messages.findLast(message => message.role === 'user')?.content
      const text = typeof prompt === 'string' ? prompt : '(missing prompt)'
      yield { thinking: `checking ${text}` }
      await Bun.sleep(2)
      yield {
        content: `finished:${text}`,
        usage: { inputTokens: 11, outputTokens: 7, reasoningTokens: 3 },
      }
    } finally {
      this.active -= 1
    }
  }
}

class ReloadGenerationChildClient implements LlmClient {
  readonly models: string[] = []
  readonly prompts: string[] = []
  readonly started = Promise.withResolvers<void>()
  private readonly released = Promise.withResolvers<void>()

  constructor(
    private readonly label: string,
    blocked = false,
  ) {
    if (!blocked) this.released.resolve()
  }

  release(): void {
    this.released.resolve()
  }

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    const prompt = request.messages.findLast(message => message.role === 'user')?.content
    const text = typeof prompt === 'string' ? prompt : '(missing prompt)'
    this.models.push(request.model)
    this.prompts.push(text)
    this.started.resolve()
    await this.released.promise
    yield { content: `${this.label}:${request.model}:${text}` }
  }
}

test('native child spawning honors the creator-local resolved catalog profile', async () => {
  const client = new ReloadGenerationChildClient('catalog-provider')
  const registry = new ToolRegistry()
  const globalCoder = { ...agentDefinition('coder'), model: 'global-model' }
  const boundCoder = { ...agentDefinition('coder'), model: 'bound-model' }
  const creator: AgentDefinition = {
    ...agentDefinition('default'),
    subagents: {
      coder: {
        description: 'creator-local coder',
        path: '/profiles/readonly-coder.yaml',
        resolvedProfile: '@catalog:coder:/profiles/readonly-coder.yaml',
      },
    },
  }
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([
      ['@catalog:coder:/profiles/readonly-coder.yaml', boundCoder],
      ['coder', globalCoder],
      ['default', creator],
    ]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: client,
    model: 'connection-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })
  registerClaudeAgentTools(registry, { manager: host.managerPort })

  try {
    await registry.execute(toolCall('AgentTool', {
      prompt: 'use the creator-local profile',
      subagent_type: 'coder',
      title: 'Bound catalog',
    }), { agentId: 'default', metadata: {}, sessionId: 'catalog-parent' })
    expect(client.models).toEqual(['bound-model'])
  } finally {
    await host.manager.shutdown()
  }
})

test('native child models follow explicit, profile, parent-session, then connection precedence', async () => {
  const client = new ReloadGenerationChildClient('model-provider')
  const registry = new ToolRegistry()
  const profiled: AgentDefinition = {
    ...agentDefinition('profiled'),
    model: 'profile-model',
  }
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([
      ['default', creatorDefinition('plain', 'profiled')],
      ['plain', agentDefinition('plain')],
      ['profiled', profiled],
    ]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: client,
    model: 'connection-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })
  registerClaudeAgentTools(registry, { manager: host.managerPort })
  const parentContext = {
    agentId: 'default',
    metadata: { model: 'parent-session-model', permission_mode: 'plan' },
    sessionId: 'parent-session',
  }

  try {
    await registry.execute(toolCall('AgentTool', {
      model: 'explicit-model',
      prompt: 'use the explicit model',
      subagent_type: 'profiled',
      title: 'Explicit model',
    }), parentContext)
    await registry.execute(toolCall('AgentTool', {
      prompt: 'use the profile model',
      subagent_type: 'profiled',
      title: 'Profile model',
    }), parentContext)
    await registry.execute(toolCall('AgentTool', {
      prompt: 'inherit the parent model',
      subagent_type: 'plain',
      title: 'Parent model',
    }), parentContext)
    await registry.execute(toolCall('AgentTool', {
      prompt: 'use the connection fallback',
      subagent_type: 'plain',
      title: 'Connection model',
    }), { agentId: 'default', metadata: {}, sessionId: 'fallback-session' })

    expect(client.models).toEqual([
      'explicit-model',
      'profile-model',
      'parent-session-model',
      'connection-model',
    ])
    expect(host.managerPort.listHandles().map(snapshot => snapshot.rules?.[0])).toEqual([
      'permission:plan',
      'permission:plan',
      'permission:plan',
      'permission:accept-all',
    ])
  } finally {
    await host.manager.shutdown()
  }
})

test('native subagent handles survive runtime reloads while new spawns use the latest generation', async () => {
  const oldClient = new ReloadGenerationChildClient('old-provider', true)
  const newClient = new ReloadGenerationChildClient('new-provider')
  const eventBus = new DaemonSubagentEventBus()
  const definitions = new Map([['coder', agentDefinition('coder')]])
  const oldRegistry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: definitions,
    cwd: process.cwd(),
    eventBus,
    llm: oldClient,
    model: 'old-model',
    permissionMode: 'manual',
    toolExecutor: oldRegistry,
    tools: oldRegistry.definitions(),
  })
  const stableManagerPort = host.managerPort

  try {
    const oldTask = await host.managerPort.spawn({
      message: 'finish the old generation',
      nickname: 'old-worker',
      promptProfile: 'coder',
      title: 'Old worker',
    })
    await oldClient.started.promise

    const newRegistry = new ToolRegistry()
    host.reconfigure({
      agentDefinitions: definitions,
      cwd: process.cwd(),
      eventBus,
      llm: newClient,
      model: 'new-model',
      permissionMode: 'accept-all',
      toolExecutor: newRegistry,
      tools: newRegistry.definitions(),
    })

    expect(host.managerPort).toBe(stableManagerPort)
    expect(host.managerPort.listHandles()).toContainEqual(expect.objectContaining({
      id: oldTask.id,
      model: 'old-model',
      rules: ['permission:manual', 'delegation:blocked'],
      status: 'running',
      title: 'Old worker',
    }))

    const newTask = await host.managerPort.spawn({
      message: 'run on the new generation',
      nickname: 'new-worker',
      promptProfile: 'coder',
      title: 'New worker',
    })
    const newResult = await host.managerPort.wait([newTask.id], 1_000)
    expect(newResult.completed[0]).toMatchObject({
      lastOutput: 'new-provider:new-model:run on the new generation',
      model: 'new-model',
      rules: ['permission:accept-all', 'delegation:blocked'],
      status: 'completed',
      title: 'New worker',
    })

    oldClient.release()
    const oldResult = await host.managerPort.wait([oldTask.id], 1_000)
    expect(oldResult.completed[0]).toMatchObject({
      lastOutput: 'old-provider:old-model:finish the old generation',
      model: 'old-model',
      rules: ['permission:manual', 'delegation:blocked'],
      status: 'completed',
      title: 'Old worker',
    })

    host.managerPort.resume(oldTask.id)
    const continuedTask = await host.managerPort.sendInput(oldTask.id, {
      message: 'follow up after reload',
    })
    const continuedResult = await host.managerPort.wait([continuedTask.id], 1_000)
    expect(continuedResult.completed[0]?.lastOutput).toBe('old-provider:old-model:follow up after reload')
    expect(oldClient.models).toEqual(['old-model', 'old-model'])
    expect(newClient.models).toEqual(['new-model'])
  } finally {
    await host.manager.shutdown()
  }
})

test('tightening YOLO permissions cancels active children and prevents old-policy resume', async () => {
  const client = new ReloadGenerationChildClient('yolo-provider', true)
  const eventBus = new DaemonSubagentEventBus()
  const definitions = new Map([['coder', agentDefinition('coder')]])
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: definitions,
    cwd: process.cwd(),
    eventBus,
    llm: client,
    model: 'yolo-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    const task = await host.managerPort.spawn({
      message: 'keep working under yolo',
      nickname: 'yolo-worker',
      promptProfile: 'coder',
      title: 'YOLO worker',
    })
    await client.started.promise

    host.reconfigure({
      agentDefinitions: definitions,
      cwd: process.cwd(),
      eventBus,
      llm: client,
      model: 'safe-model',
      permissionMode: 'auto',
      toolExecutor: registry,
      tools: registry.definitions(),
    })

    expect(host.managerPort.listHandles()).toContainEqual(expect.objectContaining({
      closed: true,
      id: task.id,
      status: 'cancelled',
    }))
    expect(() => host.managerPort.resume(task.id)).toThrow(
      'was invalidated when permissions were tightened',
    )
  } finally {
    client.release()
    await host.manager.shutdown()
  }
})

test('stale parent metadata cannot spawn a child above the current permission ceiling', async () => {
  const client = new ReloadGenerationChildClient('ceiling-provider')
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([['coder', agentDefinition('coder')]]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: client,
    model: 'safe-model',
    permissionMode: 'auto',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    const task = await host.managerPort.spawn({
      message: 'continue from a stale yolo parent turn',
      permissionMode: 'accept-all',
      promptProfile: 'coder',
      title: 'Ceiling worker',
    })
    expect(task.rules).toContain('permission:auto')
    expect(task.rules).not.toContain('permission:accept-all')
  } finally {
    client.release()
    await host.manager.shutdown()
  }
})

test('agent catalog changes invalidate old handles before they can resume', async () => {
  const client = new ReloadGenerationChildClient('catalog-version-provider')
  const registry = new ToolRegistry()
  const eventBus = new DaemonSubagentEventBus()
  const coder = agentDefinition('coder')
  const initialDefinitions = new Map([
    ['coder', coder],
    ['default', creatorDefinition('coder')],
  ])
  const host = createNativeSubagentHost({
    agentDefinitions: initialDefinitions,
    cwd: process.cwd(),
    eventBus,
    llm: client,
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    const task = await host.managerPort.spawn({
      creatorAgentId: 'default',
      message: 'finish before catalog reload',
      promptProfile: 'coder',
      title: 'Catalog worker',
    })
    await host.managerPort.wait([task.id], 1_000)
    host.reconfigure({
      agentDefinitions: new Map([
        ['coder', coder],
        ['default', agentDefinition('default')],
      ]),
      cwd: process.cwd(),
      eventBus,
      llm: client,
      model: 'test-model',
      permissionMode: 'accept-all',
      toolExecutor: registry,
      tools: registry.definitions(),
    })
    expect(() => host.managerPort.resume(task.id)).toThrow('invalidated')
  } finally {
    client.release()
    await host.manager.shutdown()
  }
})

test('removing a parent session cancels and invalidates its background children', async () => {
  const client = new ReloadGenerationChildClient('background-provider', true)
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([['coder', agentDefinition('coder')]]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: client,
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    const task = await host.managerPort.spawn({
      message: 'keep working in the background',
      promptProfile: 'coder',
      sourceAgentId: 'session-a',
      title: 'Background worker',
    })
    await client.started.promise

    expect(host.cancelSource('session-b')).toBe(0)
    expect(host.cancelSource('session-a')).toBe(1)
    expect(host.managerPort.listHandles()).toContainEqual(expect.objectContaining({
      closed: true,
      id: task.id,
      status: 'cancelled',
    }))
    expect(() => host.managerPort.resume(task.id)).toThrow('invalidated')
  } finally {
    client.release()
    await host.manager.shutdown()
  }
})

test('tightening auto permissions to plan also cancels broader-policy children', async () => {
  const client = new ReloadGenerationChildClient('auto-provider', true)
  const eventBus = new DaemonSubagentEventBus()
  const definitions = new Map([['coder', agentDefinition('coder')]])
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: definitions,
    cwd: process.cwd(),
    eventBus,
    llm: client,
    model: 'auto-model',
    permissionMode: 'auto',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    const task = await host.managerPort.spawn({
      message: 'keep working under auto policy',
      promptProfile: 'coder',
      title: 'Auto worker',
    })
    await client.started.promise

    host.reconfigure({
      agentDefinitions: definitions,
      cwd: process.cwd(),
      eventBus,
      llm: client,
      model: 'plan-model',
      permissionMode: 'plan',
      toolExecutor: registry,
      tools: registry.definitions(),
    })

    expect(host.managerPort.listHandles()).toContainEqual(expect.objectContaining({
      closed: true,
      id: task.id,
      status: 'cancelled',
    }))
  } finally {
    client.release()
    await host.manager.shutdown()
  }
})

test('switching delegated permissions from plan to manual cancels children that retained automatic read access', async () => {
  const client = new ReloadGenerationChildClient('plan-provider', true)
  const eventBus = new DaemonSubagentEventBus()
  const definitions = new Map([['coder', agentDefinition('coder')]])
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: definitions,
    cwd: process.cwd(),
    eventBus,
    llm: client,
    model: 'plan-model',
    permissionMode: 'plan',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    const task = await host.managerPort.spawn({
      message: 'keep reading under plan policy',
      promptProfile: 'coder',
      title: 'Plan worker',
    })
    await client.started.promise

    host.reconfigure({
      agentDefinitions: definitions,
      cwd: process.cwd(),
      eventBus,
      llm: client,
      model: 'manual-model',
      permissionMode: 'manual',
      toolExecutor: registry,
      tools: registry.definitions(),
    })

    expect(host.managerPort.listHandles()).toContainEqual(expect.objectContaining({
      closed: true,
      id: task.id,
      status: 'cancelled',
    }))
    expect(() => host.managerPort.resume(task.id)).toThrow('invalidated')
  } finally {
    client.release()
    await host.manager.shutdown()
  }
})

test('invalidating a native host cancels every child and prevents later resume', async () => {
  const client = new ReloadGenerationChildClient('retired-provider', true)
  const registry = new ToolRegistry()
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([['coder', agentDefinition('coder')]]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: client,
    model: 'retired-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })

  try {
    const task = await host.managerPort.spawn({
      message: 'continue until the runtime disappears',
      promptProfile: 'coder',
      title: 'Retired worker',
    })
    await client.started.promise

    expect(host.invalidateAll()).toBe(1)
    expect(host.managerPort.listHandles()).toContainEqual(expect.objectContaining({
      closed: true,
      id: task.id,
      status: 'cancelled',
    }))
    expect(() => host.managerPort.resume(task.id)).toThrow('invalidated')
  } finally {
    client.release()
    await host.manager.shutdown()
  }
})

test('native SpawnAgents runs two real child turns concurrently and routes their lifecycle to the parent session', async () => {
  const client = new ConcurrentChildClient()
  const eventBus = new DaemonSubagentEventBus()
  const registry = new ToolRegistry()
  const events: DaemonEvent[] = []
  const release = eventBus.subscribe('session-a', event => events.push(event))
  const definitions = new Map([
    ['coder', agentDefinition('coder')],
    ['default', creatorDefinition('coder', 'researcher')],
    ['researcher', agentDefinition('researcher')],
  ])
  const host = createNativeSubagentHost({
    agentDefinitions: definitions,
    cwd: process.cwd(),
    eventBus,
    llm: client,
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })
  registerClaudeAgentTools(registry, { manager: host.managerPort })

  try {
    const response = JSON.parse(await registry.execute(toolCall('SpawnAgents', {
      agents: [
        { name: 'implementation', prompt: 'inspect runtime', subagent_type: 'coder', title: 'Inspect runtime' },
        { name: 'research', prompt: 'inspect tests', subagent_type: 'researcher', title: 'Inspect tests' },
      ],
      timeout: 2,
      wait: true,
    }), { agentId: 'default', metadata: {}, sessionId: 'session-a' })) as Array<Record<string, unknown>>

    expect(client.calls).toBe(2)
    expect(client.maxActive).toBe(2)
    expect(response).toHaveLength(2)
    expect(response.map(entry => entry.status)).toEqual(['completed', 'completed'])
    expect(response.map(entry => entry.source_agent_id)).toEqual(['session-a', 'session-a'])
    expect(response.map(entry => entry.creator_id)).toEqual(['default', 'default'])
    expect(response.map(entry => entry.parent_id)).toEqual(['default', 'default'])
    expect(response.map(entry => entry.title)).toEqual(['Inspect runtime', 'Inspect tests'])
    expect(response.map(entry => entry.last_output)).toEqual([
      'finished:inspect runtime',
      'finished:inspect tests',
    ])

    await waitFor(() => events.filter(event => nestedEventType(event) === 'turn_end').length === 2)
    const starts = events.filter(event => nestedEventType(event) === 'turn_begin')
    const completions = events.filter(event => nestedEventType(event) === 'turn_end')
    expect(starts).toHaveLength(2)
    expect(completions).toHaveLength(2)
    expect(new Set(starts.map(event => event.payload.agent_id)).size).toBe(2)
    expect(starts.map(event => event.payload.agent_name).sort()).toEqual(['implementation', 'research'])
    expect(starts.map(event => event.payload.subagent_type).sort()).toEqual(['coder', 'researcher'])
    expect(starts.map(event => event.payload.title).sort()).toEqual(['Inspect runtime', 'Inspect tests'])
    expect(starts.every(event => event.payload.creator_id === 'default' && event.payload.parent_id === 'default')).toBeTrue()
    expect(starts.every(event => event.payload.model === 'test-model')).toBeTrue()
    expect(starts.every(event => Array.isArray(event.payload.rules))).toBeTrue()
    expect(starts.every(event => Array.isArray(event.payload.toolsets))).toBeTrue()
    expect(events.filter(event => nestedEventType(event) === 'think_part')).toHaveLength(2)
    expect(completions.map(event => event.payload.api_calls)).toEqual([1, 1])
    expect(completions.map(event => event.payload.input_tokens)).toEqual([11, 11])
    expect(completions.map(event => event.payload.output_tokens)).toEqual([7, 7])
    expect(completions.map(event => event.payload.reasoning_tokens)).toEqual([3, 3])
    expect(completions.map(event => event.payload.tool_count)).toEqual([0, 0])
    expect(completions.map(event => nestedPayload(event).summary).sort()).toEqual([
      'finished:inspect runtime',
      'finished:inspect tests',
    ])
  } finally {
    release()
    await host.manager.shutdown()
  }
})

class ProjectMemoryChildClient implements LlmClient {
  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    const results = request.messages.filter(message => message.role === 'tool')
    if (results.length === 0) {
      yield {
        toolCalls: [toolCall('agent_memory_write', {
          body: 'project findings',
          path: 'deepscan/findings/researcher.md',
          scope: 'project',
        })],
      }
      return
    }
    if (results.length === 1) {
      yield {
        toolCalls: [toolCall('agent_memory_write', {
          body: 'must stay denied',
          path: 'global.md',
          scope: 'global',
        })],
      }
      return
    }
    yield { content: results.map(message => String(message.content)).join('\n') }
  }
}

test('auto-mode native subagents can persist project memory without gaining global-memory writes', async () => {
  const registry = new ToolRegistry()
  const writes: JsonObject[] = []
  registry.register(AGENT_MEMORY_WRITE_DEFINITION, inputs => {
    writes.push(inputs)
    return { ok: true, path: inputs.path, scope: inputs.scope }
  })
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([['researcher', agentDefinition('researcher')]]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: new ProjectMemoryChildClient(),
    model: 'test-model',
    permissionMode: 'auto',
    toolExecutor: registry,
    tools: registry.definitions(),
  })
  registerClaudeAgentTools(registry, { manager: host.managerPort })

  try {
    const response = JSON.parse(await registry.execute(toolCall('SpawnAgents', {
      agents: [{
        name: 'memory-writer',
        prompt: 'persist findings',
        subagent_type: 'researcher',
        title: 'Persist findings',
      }],
      timeout: 2,
      wait: true,
    }), { metadata: {}, sessionId: 'session-memory' })) as Array<Record<string, unknown>>

    expect(writes).toEqual([{
      body: 'project findings',
      path: 'deepscan/findings/researcher.md',
      scope: 'project',
    }])
    expect(response[0]?.status).toBe('completed')
    expect(response[0]?.last_output).toContain('"ok":true')
    expect(response[0]?.last_output).toContain('Permission denied for agent_memory_write.')
  } finally {
    await host.manager.shutdown()
  }
})

class MetadataChildClient implements LlmClient {
  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    const results = request.messages.filter(message => message.role === 'tool')
    if (results.length === 0) {
      yield {
        toolCalls: [toolCall('ReadFile', { file_path: 'src/input.ts' })],
        usage: { inputTokens: 5, outputTokens: 2, reasoningTokens: 1 },
      }
      return
    }
    if (results.length === 1) {
      yield {
        toolCalls: [toolCall('WriteFile', { content: 'done', file_path: 'src/output.ts' })],
        usage: { inputTokens: 7, outputTokens: 2, reasoningTokens: 1 },
      }
      return
    }
    yield {
      content: 'metadata task complete',
      usage: { inputTokens: 11, outputTokens: 5, reasoningTokens: 2 },
    }
  }
}

test('native completion events expose exact title, hierarchy, policy, file, tool, and usage metadata', async () => {
  const eventBus = new DaemonSubagentEventBus()
  const registry = new ToolRegistry()
  const toolDefinition = (name: string): ToolDefinition => ({
    type: 'function',
    function: {
      description: `${name} test tool`,
      name,
      parameters: { type: 'object', additionalProperties: true, properties: {} },
    },
  })
  registry.register(toolDefinition('ReadFile'), () => 'input body')
  registry.register(toolDefinition('WriteFile'), () => 'written')
  const events: DaemonEvent[] = []
  const release = eventBus.subscribe('metadata-session', event => events.push(event))
  const host = createNativeSubagentHost({
    agentDefinitions: new Map([
      ['coder', agentDefinition('coder')],
      ['default', creatorDefinition('coder')],
    ]),
    cwd: process.cwd(),
    eventBus,
    llm: new MetadataChildClient(),
    model: 'metadata-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
  })
  registerClaudeAgentTools(registry, { manager: host.managerPort })

  try {
    await registry.execute(toolCall('AgentTool', {
      name: 'metadata-worker',
      prompt: 'inspect and update metadata fixtures',
      subagent_type: 'coder',
      timeout: 2,
      title: 'Inspect metadata fixtures',
      wait: true,
    }), { agentId: 'default', metadata: {}, sessionId: 'metadata-session' })

    await waitFor(() => events.some(event => nestedEventType(event) === 'turn_end'))
    const completion = events.find(event => nestedEventType(event) === 'turn_end')
    expect(completion?.payload).toMatchObject({
      agent_name: 'metadata-worker',
      api_calls: 3,
      creator_id: 'default',
      files_read: ['src/input.ts'],
      files_written: ['src/output.ts'],
      input_tokens: 23,
      model: 'metadata-model',
      output_tokens: 9,
      parent_id: 'default',
      reasoning_tokens: 4,
      rules: ['permission:accept-all', 'delegation:blocked'],
      summary: 'metadata task complete',
      title: 'Inspect metadata fixtures',
      tool_count: 2,
      toolsets: ['ReadFile', 'WriteFile'],
    })
    expect(nestedPayload(completion!).summary).toBe('metadata task complete')
  } finally {
    release()
    await host.manager.shutdown()
  }
})

class ToolCallingParentClient implements LlmClient {
  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    if (request.messages.some(message => message.role === 'tool')) {
      yield { content: 'parent finished' }
      return
    }
    yield {
      toolCalls: [{
        id: 'delegate-1',
        type: 'function',
        function: { name: 'DelegateForTest', arguments: {} },
      }],
    }
  }
}

test('agent turn runner multiplexes a child event between parent tool start and result with turn correlation', async () => {
  const eventBus = new DaemonSubagentEventBus()
  const registry = new ToolRegistry()
  let delegatedPermissionMode: unknown
  const definition: ToolDefinition = {
    type: 'function',
    function: {
      description: 'emit one delegated lifecycle event',
      name: 'DelegateForTest',
      parameters: { type: 'object', additionalProperties: false, properties: {} },
    },
  }
  registry.register(definition, async (_inputs, context) => {
    delegatedPermissionMode = context.metadata.permission_mode
    eventBus.publish(context.sessionId ?? '', {
      type: 'subagent_event',
      payload: {
        agent_id: 'child-1',
        event: { type: 'turn_begin', payload: { status: 'running' } },
        goal: 'verify multiplexing',
      },
    })
    await Bun.sleep(5)
    return 'delegation complete'
  })
  const runner = new AgentTurnRunner({
    llm: new ToolCallingParentClient(),
    model: 'test-model',
    permissionMode: 'accept-all',
    subagentEvents: eventBus,
    toolExecutor: registry,
    tools: [definition],
  })
  const events: DaemonEvent[] = []
  for await (const event of runner.run(session(), 'delegate this', new AbortController().signal)) {
    events.push(event)
  }

  const toolCallIndex = events.findIndex(event => event.type === 'tool_call')
  const childIndex = events.findIndex(event => event.type === 'subagent_event')
  const toolResultIndex = events.findIndex(event => event.type === 'tool_result')
  expect(toolCallIndex).toBeGreaterThanOrEqual(0)
  expect(childIndex).toBeGreaterThan(toolCallIndex)
  expect(toolResultIndex).toBeGreaterThan(childIndex)
  expect(events[childIndex]).toEqual({
    type: 'subagent_event',
    payload: {
      agent_id: 'child-1',
      event: { type: 'turn_begin', payload: { status: 'running' } },
      goal: 'verify multiplexing',
      session_id: 'parent-session',
      turn_id: 'parent-turn',
    },
  })
  expect(events).toContainEqual({ type: 'text_part', payload: { text: 'parent finished' } })
  expect(delegatedPermissionMode).toBe('accept-all')
})

function nestedEventType(event: DaemonEvent): unknown {
  return nestedEvent(event)?.type
}

function nestedPayload(event: DaemonEvent): Record<string, unknown> {
  const payload = nestedEvent(event)?.payload
  return isRecord(payload) ? payload : {}
}

function nestedEvent(event: DaemonEvent): Record<string, unknown> | undefined {
  const nested = event.payload.event
  return isRecord(nested) ? nested : undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
