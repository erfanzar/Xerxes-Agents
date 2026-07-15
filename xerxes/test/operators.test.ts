// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ValidationError } from '../src/core/errors.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import {
  BrowserManager,
  MAX_AGENT_TITLE_LENGTH,
  OperatorState,
  PlanStateManager,
  PtySessionManager,
  SpawnedAgentManager,
  UserPromptManager,
  createOperatorRuntimeConfig,
  type BrowserAdapter,
} from '../src/operators/index.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

function toolCall(name: string, arguments_: JsonObject): ToolCall {
  return { id: crypto.randomUUID(), type: 'function', function: { name, arguments: arguments_ } }
}

test('plan manager snapshots revisions and retains a concise summary', () => {
  const manager = new PlanStateManager({ now: () => new Date('2026-07-13T10:00:00.000Z') })
  expect(manager.summary()).toBe('No plan')
  expect(manager.update('Ship the port', [
    { step: 'inventory', status: 'completed' },
    { step: 'implement', status: 'in_progress' },
  ])).toEqual({
    explanation: 'Ship the port',
    revision: 1,
    updatedAt: '2026-07-13T10:00:00.000Z',
    steps: [
      { step: 'inventory', status: 'completed' },
      { step: 'implement', status: 'in_progress' },
    ],
  })
  expect(manager.summary()).toBe('completed:inventory, in_progress:implement')
})

test('user prompt manager resolves numbered choices and rejects disallowed freeform input', async () => {
  const manager = new UserPromptManager({ idFactory: () => 'question-1' })
  const response = manager.request({ question: 'Pick one', options: ['first', 'second'], allowFreeform: false })
  expect(manager.getPending()).toMatchObject({ requestId: 'question-1', question: 'Pick one' })
  expect(() => manager.answer('other')).toThrow('Choose one of the listed options')
  expect(manager.answer('2')).toEqual({
    requestId: 'question-1',
    question: 'Pick one',
    answer: 'second',
    rawInput: '2',
    selectedOption: { label: 'second', value: 'second' },
    usedFreeform: false,
  })
  await expect(response).resolves.toMatchObject({ answer: 'second', selectedOption: { value: 'second' } })
  expect(manager.hasPending()).toBeFalse()
})

test('spawned agents execute queued input sequentially and expose a stable lifecycle snapshot', async () => {
  const received: string[] = []
  const manager = new SpawnedAgentManager({
    idFactory: () => 'worker',
    runner: async request => {
      received.push(request.input)
      await Bun.sleep(5)
      return { content: request.input.toUpperCase() }
    },
  })
  const spawned = await manager.spawn({ message: 'first' })
  expect(spawned.status).toBe('running')
  await manager.sendInput('worker', { message: 'second' })
  const settled = await manager.wait(['worker'], 1_000)
  expect(received).toEqual(['first', 'second'])
  expect(settled.pending).toEqual([])
  expect(settled.completed[0]).toMatchObject({
    id: 'worker',
    status: 'completed',
    lastInput: 'second',
    lastOutput: 'SECOND',
    queueSize: 0,
  })
})

test('spawned-agent titles normalize to one concise line and reject oversized input', async () => {
  let nextId = 0
  const seenTitles: Array<string | undefined> = []
  const manager = new SpawnedAgentManager({
    idFactory: () => `worker-${++nextId}`,
    runner: async request => {
      seenTitles.push(request.agent.title)
      return { content: 'done' }
    },
  })

  const spawned = await manager.spawn({ message: 'inspect runtime', title: '  Inspect\n  runtime  ' })
  await manager.wait([spawned.id], 1_000)
  expect(spawned.title).toBe('Inspect runtime')
  expect(seenTitles).toEqual(['Inspect runtime'])

  await expect(manager.spawn({
    message: 'too long',
    title: 'x'.repeat(MAX_AGENT_TITLE_LENGTH + 1),
  })).rejects.toBeInstanceOf(ValidationError)
})

test('Bun PTY sessions preserve terminal output across a bounded first read', async () => {
  const manager = new PtySessionManager()
  const shell = Bun.which('sh') ?? '/bin/sh'
  const result = await manager.createSession('printf terminal-ready', {
    shell,
    login: false,
    yieldTimeMs: 1_000,
    maxOutputChars: 1_000,
  })
  try {
    expect(result.stdout).toContain('terminal-ready')
    expect(manager.listSessions()).toHaveLength(1)
  } finally {
    await manager.close(result.sessionId)
  }
})

test('browser manager delegates only public navigation to its adapter and tracks links', async () => {
  const adapter: BrowserAdapter = {
    open: async request => ({
      ...(request.refId === undefined ? {} : { refId: request.refId }),
      url: request.url ?? 'https://example.com/next',
      title: 'Example',
      contentPreview: 'example body',
      links: [{ url: 'https://example.com/next' }],
    }),
    click: async request => ({ refId: request.refId, url: 'https://example.com/clicked', title: 'Clicked' }),
    find: async (refId, pattern) => ({ refId, pattern, matchCount: 1, matches: [pattern] }),
    screenshot: async (refId, options) => ({ refId, path: options.path ?? '/tmp/page.png', fullPage: options.fullPage }),
  }
  const manager = new BrowserManager({ adapter })
  const first = await manager.open({ url: 'https://example.com' })
  expect(first.links).toEqual([{ id: 0, url: 'https://example.com/next' }])
  const next = await manager.click(first.refId, { linkId: 0 })
  expect(next.url).toBe('https://example.com/next')
  await expect(manager.open({ url: 'http://127.0.0.1/private' })).rejects.toBeInstanceOf(ValidationError)
})

test('operator state replaces provisional process tooling and exposes only configured capabilities', async () => {
  const registry = new ToolRegistry()
  registry.register({
    type: 'function',
    function: { name: 'exec_command', description: 'provisional', parameters: { type: 'object' } },
  }, () => 'provisional')
  const state = new OperatorState({
    config: createOperatorRuntimeConfig({ enabled: true, powerToolsEnabled: true }),
  })
  state.registerTools(registry)
  const plan = JSON.parse(await registry.execute(toolCall('update_plan', {
    explanation: 'Port operators',
    plan: [{ step: 'implement', status: 'in_progress' }],
  }), { metadata: {} })) as {
    explanation: string
    revision: number
    steps: Array<{ status: string; step: string }>
    updated_at: string
  }
  expect(plan).toEqual({
    explanation: 'Port operators',
    revision: 1,
    updated_at: expect.any(String),
    steps: [{ step: 'implement', status: 'in_progress' }],
  })
  expect(registry.definitions().find(tool => tool.function.name === 'exec_command')?.function.description)
    .toContain('persistent PTY')
  state.setPowerToolsEnabled(false)
  expect(state.toolDefinitions().map(tool => tool.function.name)).not.toContain('exec_command')
  expect(state.toolDefinitions().map(tool => tool.function.name)).toContain('update_plan')
  expect(registry.definitions().map(tool => tool.function.name)).not.toContain('exec_command')
  state.setPowerToolsEnabled(true)
  expect(registry.definitions().map(tool => tool.function.name)).toContain('exec_command')
  await state.close()
})
