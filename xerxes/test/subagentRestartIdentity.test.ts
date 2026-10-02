// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import type { AgentDefinition } from '../src/agents/definitions.js'
import {
  persistedSubagentDeliveryValues,
  persistedSubagentSnapshotValues,
  replacePersistedSubagentDeliveries,
  replacePersistedSubagentSnapshots,
} from '../src/agents/subagentPersistence.js'
import { recoverSubagentSnapshots } from '../src/daemon/subagentCoordinator.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { createNativeSubagentHost } from '../src/daemon/subagentHost.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'

const worker: AgentDefinition = { name: 'worker', description: 'restart identity test', source: 'test', model: '',
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

test('unnamed agents each survive a restart instead of collapsing onto the shared default name', async () => {
  const first = makeHost()
  const metadata: Record<string, unknown> = {}
  const ids: string[] = []
  try {
    for (const message of ['scan a', 'scan b', 'scan c']) {
      const task = await first.managerPort.spawn({ message, promptProfile: 'worker', sourceAgentId: 'parent' })
      ids.push(task.id)
    }
    await first.managerPort.wait(ids, 5_000)
    // Every daemon id starts `subagent_`, so the id-derived default name is
    // shared by all three; that is the shape the restart path has to survive.
    expect(new Set(first.managerPort.listHandles().map(snapshot => snapshot.name)).size).toBe(1)
    replacePersistedSubagentSnapshots(metadata, first.managerPort.listHandles())
  } finally { await first.manager.shutdown() }

  const recovered = recoverSubagentSnapshots([], 'parent', persistedSubagentSnapshotValues(metadata))
  expect(recovered.map(snapshot => snapshot.id).sort()).toEqual([...ids].sort())

  const next = makeHost()
  try {
    expect(next.turnCoordinator.restore?.('parent', recovered)).toBe(3)
    for (const id of ids) {
      const retried = await next.retry(id, { sourceAgentId: 'parent', message: `again ${id}` })
      expect(retried.id).toBe(id)
    }
  } finally { await next.manager.shutdown() }
})

test('an explicitly named agent still restores only its newest generation', () => {
  const row = (id: string, createdAt: string) => ({
    id, name: 'reviewer', title: 'Reviewer', agent_id: 'worker', prompt_profile: 'worker',
    status: 'completed', source_agent_id: 'parent', created_at: createdAt, updated_at: createdAt,
  })
  const recovered = recoverSubagentSnapshots([], 'parent', [
    row('subagent_000000000001', '2026-01-01T00:00:00.000Z'),
    row('subagent_000000000002', '2026-01-02T00:00:00.000Z'),
  ])
  expect(recovered.map(snapshot => snapshot.id)).toEqual(['subagent_000000000002'])
})

test('retrying an agent recovered after a restart hands its new result to the parent turn', async () => {
  const first = makeHost()
  const metadata: Record<string, unknown> = {}
  let id = ''
  try {
    const task = await first.managerPort.spawn({ message: 'first pass', promptProfile: 'worker', sourceAgentId: 'parent' })
    id = task.id
    const cohort = first.turnCoordinator.begin('parent')
    first.turnCoordinator.track([task])
    const delivered = await cohort.waitForResults()
    expect(delivered.map(snapshot => snapshot.id)).toEqual([id])
    first.turnCoordinator.consume(delivered)
    cohort.close()
    replacePersistedSubagentSnapshots(metadata, first.managerPort.listHandles())
    replacePersistedSubagentDeliveries(metadata, first.turnCoordinator.deliveredState?.() ?? [])
  } finally { await first.manager.shutdown() }

  const next = makeHost()
  try {
    next.turnCoordinator.hydrateDelivered?.(persistedSubagentDeliveryValues(metadata))
    next.turnCoordinator.restore?.('parent', recoverSubagentSnapshots([], 'parent', persistedSubagentSnapshotValues(metadata)))
    const cohort = next.turnCoordinator.begin('parent')
    // The delivered first pass must not come back on its own.
    expect(next.turnCoordinator.trackedIds('parent')).toEqual([])
    const retried = await next.retry(id, { sourceAgentId: 'parent', message: 'second pass' })
    expect(retried.attempt).toBe(1)
    next.turnCoordinator.track([retried])
    expect(next.turnCoordinator.trackedIds('parent')).toEqual([id])
    const results = await cohort.waitForResults()
    expect(results.map(snapshot => snapshot.lastOutput)).toEqual(['done:second pass'])
    cohort.close()
  } finally { await next.manager.shutdown() }
})
