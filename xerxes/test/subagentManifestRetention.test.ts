// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import type { AgentDefinition } from '../src/agents/definitions.js'
import { DEFAULT_MAX_RETAINED_TERMINAL_TASKS } from '../src/agents/subagentManager.js'
import { persistedSubagentSnapshotValues } from '../src/agents/subagentPersistence.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { createNativeSubagentHost } from '../src/daemon/subagentHost.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import type { SpawnedAgentManagerPort, SpawnedAgentSnapshot } from '../src/operators/subagents.js'
import { AgentEventMailbox, ClaudeAgentTools } from '../src/tools/claudeTools/agentOps.js'

const worker: AgentDefinition = { name: 'worker', description: 'retention test', source: 'test', model: '',
  systemPrompt: 'Answer.', allowedTools: null, excludeTools: [], tools: [], isolation: '', maxDepth: 3 }

class EchoClient implements LlmClient {
  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    const prompt = request.messages.findLast(message => message.role === 'user')?.content
    yield { content: `done:${typeof prompt === 'string' ? prompt : ''}` }
  }
}

function makeHost() {
  const tools = new ToolRegistry()
  return createNativeSubagentHost({
    agentDefinitions: new Map([['worker', worker]]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: new EchoClient(),
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: tools,
    tools: tools.definitions(),
  })
}

function snapshot(id: string, status: SpawnedAgentSnapshot['status'] = 'completed'): SpawnedAgentSnapshot {
  return {
    agentId: 'coder', closed: false, createdAt: '2026-01-01T00:00:00.000Z', id, lastInput: `task:${id}`,
    lastOutput: `output:${id}`, name: id, promptProfile: 'coder', queueSize: 0, sourceAgentId: 'session-1',
    status, title: `Task ${id}`, updatedAt: '2026-01-01T00:00:00.000Z',
  }
}

async function burst(host: ReturnType<typeof makeHost>, count: number): Promise<void> {
  // Waves stay under the live-agent cap; only the terminal history piles up.
  for (let start = 0; start < count; start += 50) {
    const ids: string[] = []
    for (let index = start; index < Math.min(count, start + 50); index += 1) {
      const task = await host.managerPort.spawn({ message: `filler ${index}`, promptProfile: 'worker', sourceAgentId: 'other-session' })
      ids.push(task.id)
    }
    await host.managerPort.wait(ids, 30_000)
  }
}

test('an agent tool call keeps manifest rows for agents the live view no longer lists', async () => {
  let handles = [snapshot('a'), snapshot('b'), snapshot('c')]
  const manager: SpawnedAgentManagerPort = {
    close: () => { throw new Error('unused') },
    listHandles: () => handles,
    resume: () => { throw new Error('unused') },
    sendInput: async () => { throw new Error('unused') },
    spawn: async () => { throw new Error('unused') },
    wait: async () => ({ completed: [], pending: [] }),
  }
  const tools = new ClaudeAgentTools({ manager })
  const metadata: Record<string, unknown> = {}
  await tools.execute('TaskListTool', {}, { metadata, sessionId: 'session-1' })
  // The shared manager evicted the two oldest terminal tasks.
  handles = [snapshot('c')]
  await tools.execute('TaskListTool', {}, { metadata, sessionId: 'session-1' })
  expect(persistedSubagentSnapshotValues(metadata).map(row => [row.id, row.last_output])).toEqual([
    ['a', 'output:a'], ['b', 'output:b'], ['c', 'output:c'],
  ])
})

test('a rolled-back spawn batch still leaves no manifest rows behind', async () => {
  const live: SpawnedAgentSnapshot[] = []
  let spawned = 0
  const manager: SpawnedAgentManagerPort = {
    close: id => {
      const index = live.findIndex(candidate => candidate.id === id)
      const closed = { ...live[index]!, closed: true, status: 'closed' as const }
      live[index] = closed
      return { ...closed, previousStatus: 'running' }
    },
    listHandles: () => live,
    resume: () => { throw new Error('unused') },
    sendInput: async () => { throw new Error('unused') },
    spawn: async () => {
      spawned += 1
      if (spawned === 2) throw new Error('second spawn refused')
      const created = snapshot(`spawned-${spawned}`, 'running')
      live.push(created)
      return created
    },
    wait: async () => ({ completed: [], pending: [] }),
  }
  const tools = new ClaudeAgentTools({ manager, spawnConcurrency: 1 })
  const metadata: Record<string, unknown> = {}
  await expect(tools.execute('SpawnAgents', {
    agents: [{ title: 'One', prompt: 'one' }, { title: 'Two', prompt: 'two' }],
    wait: false,
  }, { metadata, sessionId: 'session-1' })).rejects.toThrow('second spawn refused')
  expect(persistedSubagentSnapshotValues(metadata)).toEqual([])
})

test('a result a parent turn is waiting for survives a workspace-wide burst of completions', async () => {
  const host = makeHost()
  try {
    const cohort = host.turnCoordinator.begin('parent')
    const awaited = await host.managerPort.spawn({ message: 'background work', promptProfile: 'worker', sourceAgentId: 'parent' })
    host.turnCoordinator.track([awaited])
    await host.managerPort.wait([awaited.id], 5_000)
    // Another session's workflow finishes more agents than the shared
    // manager retains, pushing the awaited result out of the live set.
    await burst(host, DEFAULT_MAX_RETAINED_TERMINAL_TASKS + 4)
    const results = await cohort.waitForResults()
    expect(results.map(result => [result.id, result.lastOutput])).toEqual([[awaited.id, 'done:background work']])
    cohort.close()
  } finally { await host.manager.shutdown() }
}, 60_000)

test('per-task handle state is released once the manager forgets a task', async () => {
  const host = makeHost()
  try {
    await burst(host, DEFAULT_MAX_RETAINED_TERMINAL_TASKS + 300)
    const retained = new Set(host.manager.listRetryTasks().map(task => task.id))
    const live = Reflect.get(host.managerPort, 'live') as object
    const handles = Reflect.get(live, 'handles') as Map<string, unknown>
    expect(handles.size).toBe(retained.size)
    expect([...handles.keys()].every(id => retained.has(id))).toBeTrue()
  } finally { await host.manager.shutdown() }
}, 60_000)

test('the event mailbox forgets agents that left the manager view', () => {
  const mailbox = new AgentEventMailbox()
  mailbox.capture([snapshot('a'), snapshot('b')])
  mailbox.capture([snapshot('b')])
  const observed = Reflect.get(mailbox, 'observed') as Map<string, unknown>
  expect([...observed.keys()]).toEqual(['b'])
  // A known agent with no change still records nothing new.
  const before = mailbox.latestSeq()
  mailbox.capture([snapshot('b')])
  expect(mailbox.latestSeq()).toBe(before)
})
