// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  EntryKind,
  ExecutionRegistry,
  ExecutionStatus,
} from '../src/runtime/executionRegistry.js'

interface Deferred<T> {
  readonly promise: Promise<T>
  resolve(value: T): void
}

function deferred<T>(): Deferred<T> {
  let resolvePromise: ((value: T) => void) | undefined
  const promise = new Promise<T>(resolve => {
    resolvePromise = resolve
  })
  return {
    promise,
    resolve: value => resolvePromise?.(value),
  }
}

async function tick(): Promise<void> {
  await Promise.resolve()
  await Promise.resolve()
}

test('execution registry preserves command/tool lookup, routing, schemas, and immediate outcomes', async () => {
  let monotonic = 10
  const registry = new ExecutionRegistry({ monotonicNow: () => monotonic })
  registry.registerCommand('Help', inputs => {
    monotonic = 17
    return 'help:' + String(inputs.topic)
  }, {
    description: 'Show command and tool help',
    category: 'core',
    sourceHint: 'builtin',
  })
  registry.registerTool('ReadFile', inputs => 'read:' + String(inputs.path), {
    description: 'Read files from the workspace',
    category: 'filesystem',
    safe: true,
  })
  registry.registerTool('Custom', undefined, {
    description: 'Custom schema',
    schema: { name: 'Custom', input_schema: { type: 'object', properties: { key: { type: 'string' } } } },
  })

  expect(registry.commandCount).toBe(1)
  expect(registry.toolCount).toBe(2)
  expect(registry.getCommand('help')).toMatchObject({ name: 'Help', kind: EntryKind.COMMAND })
  expect(registry.getTool('readfile')).toBeUndefined()
  expect(registry.listTools({ safeOnly: true }).map(entry => entry.name)).toEqual(['ReadFile'])
  expect(registry.route('read files from filesystem', 2)[0]).toMatchObject({
    name: 'ReadFile',
    kind: EntryKind.TOOL,
    sourceHint: '',
  })

  const command = await registry.executeCommand('HELP', { topic: 'tools' })
  expect(command).toEqual({
    name: 'Help',
    kind: EntryKind.COMMAND,
    handled: true,
    result: 'help:tools',
    durationMs: 7,
    error: '',
  })
  expect(await registry.executeTool('ReadFile', { path: 'README.md' })).toMatchObject({
    handled: true,
    result: 'read:README.md',
  })
  expect(await registry.executeTool('missing')).toEqual({
    name: 'missing',
    kind: EntryKind.TOOL,
    handled: false,
    result: 'Unknown tool: missing',
    durationMs: 0,
    error: '',
  })
  expect(registry.toolSchemas()).toEqual([
    {
      name: 'ReadFile',
      description: 'Read files from the workspace',
      input_schema: { type: 'object', properties: {} },
    },
    {
      name: 'Custom',
      input_schema: { type: 'object', properties: { key: { type: 'string' } } },
    },
  ])
  const code = String.fromCharCode(96)
  expect(registry.summary()).toContain(code + '/Help' + code)
  expect(registry.summary()).toContain(code + 'ReadFile' + code + ' [safe]')
})

test('execution registry accepts both agent function objects and OpenAI function definitions', async () => {
  const registry = new ExecutionRegistry()
  registry.registerFromAgentFunctions([
    {
      name: 'LocalTool',
      description: 'locally callable',
      callable_func: (inputs: Readonly<Record<string, unknown>>) => 'local:' + String(inputs.value),
    },
    {
      type: 'function',
      function: { name: 'WireTool', description: 'wire-only definition', parameters: { type: 'object' } },
    },
    { function: { description: 'invalid without a name' } },
  ])

  expect(registry.listTools().map(tool => tool.name)).toEqual(['LocalTool', 'WireTool'])
  expect(registry.getTool('WireTool')).toMatchObject({ description: 'wire-only definition', handler: undefined })
  await expect(registry.executeTool('LocalTool', { value: 'ok' })).resolves.toMatchObject({ result: 'local:ok' })
})

test('registered handlers run as retained execution records with status, result, error, duration, and immutable data', async () => {
  let now = 1
  let monotonic = 100
  const metadata = { nested: { owner: 'runtime' } }
  const registry = new ExecutionRegistry({
    now: () => now,
    monotonicNow: () => monotonic,
  })
  registry.registerCommand('echo', inputs => {
    monotonic = 106
    return String(inputs.message)
  })
  registry.registerTool('fail', () => {
    monotonic = 112
    throw new Error('nope')
  })

  const successful = registry.submitCommand('ECHO', { message: 'hello' }, {
    executionId: 'success',
    metadata,
  })
  metadata.nested.owner = 'changed'
  expect(successful).toMatchObject({
    status: ExecutionStatus.RUNNING,
    entryName: 'echo',
    createdAt: 1,
    startedAt: 1,
  })

  now = 2
  const complete = await registry.waitExecution('success', { settled: true })
  expect(complete).toMatchObject({
    status: ExecutionStatus.SUCCESS,
    result: 'hello',
    error: '',
    durationMs: 6,
    finishedAt: 2,
    metadata: { nested: { owner: 'runtime' } },
  })
  expect(Object.isFrozen(complete?.metadata)).toBe(true)

  now = 3
  const failed = registry.submitTool('fail', {}, { executionId: 'failure' })
  const failure = await registry.waitExecution(failed.id, { settled: true })
  expect(failure).toMatchObject({
    status: ExecutionStatus.FAILURE,
    result: '',
    error: 'Error: nope',
    durationMs: 6,
  })
})

test('detached executions honor FIFO capacity and cancellation cannot be overwritten by late success', async () => {
  const gates = new Map<string, Deferred<string>>()
  const started: string[] = []
  let secondSignal: AbortSignal | undefined
  const registry = new ExecutionRegistry({
    maxConcurrent: 1,
    runner: (record, signal) => {
      started.push(record.entryName)
      const gate = deferred<string>()
      gates.set(record.entryName, gate)
      if (record.entryName === 'second') secondSignal = signal
      return gate.promise
    },
  })
  const first = registry.submit(EntryKind.TOOL, 'first', {}, { executionId: 'first' })
  const second = registry.submit(EntryKind.TOOL, 'second', {}, { executionId: 'second' })
  const third = registry.submit(EntryKind.TOOL, 'third', {}, { executionId: 'third' })
  await tick()

  expect(first.status).toBe(ExecutionStatus.RUNNING)
  expect(registry.getExecution(second.id)?.status).toBe(ExecutionStatus.PENDING)
  expect(started).toEqual(['first'])
  expect(registry.cancelExecution(third.id)).toBe(true)
  expect(registry.getExecution(third.id)?.status).toBe(ExecutionStatus.CANCELLED)

  gates.get('first')?.resolve('first done')
  expect((await registry.waitExecution(first.id, { settled: true }))?.status).toBe(ExecutionStatus.SUCCESS)
  await tick()
  expect(registry.getExecution(second.id)?.status).toBe(ExecutionStatus.RUNNING)
  expect(started).toEqual(['first', 'second'])

  expect(registry.cancelExecution(second.id)).toBe(true)
  expect(secondSignal?.aborted).toBe(true)
  gates.get('second')?.resolve('late result')
  const cancelled = await registry.waitExecution(second.id, { settled: true })
  expect(cancelled).toMatchObject({
    status: ExecutionStatus.CANCELLED,
    result: '',
    error: '',
  })
  expect(registry.cancelExecution(second.id)).toBe(false)
})

test('retention prunes only settled terminal records and shutdown cancels pending work then rejects submissions', async () => {
  let now = 1
  const pending = deferred<string>()
  const registry = new ExecutionRegistry({
    maxCompleted: 1,
    maxConcurrent: 1,
    now: () => now,
    runner: record => record.entryName === 'first' ? pending.promise : record.entryName,
  })
  const first = registry.submit(EntryKind.TOOL, 'first', {}, { executionId: 'first' })
  const queued = registry.submit(EntryKind.TOOL, 'queued', {}, { executionId: 'queued' })
  await tick()

  await registry.shutdown({ timeoutMs: 0 })
  expect(registry.getExecution(queued.id)?.status).toBe(ExecutionStatus.CANCELLED)
  expect(() => registry.submit(EntryKind.TOOL, 'after')).toThrow('shutting down')

  now = 2
  pending.resolve('done')
  expect((await registry.waitExecution(first.id, { settled: true }))?.status).toBe(ExecutionStatus.SUCCESS)
  expect(registry.getExecution(queued.id)).toBeUndefined()
  expect(registry.listExecutions().map(record => record.id)).toEqual(['first'])
})
