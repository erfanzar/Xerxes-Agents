// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, spyOn, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { AutoCompactAgent } from '../src/agents/autoCompactAgent.js'
import { CompactionAgent } from '../src/agents/compactionAgent.js'
import {
  filterSubagentTools,
  SubAgentManager,
  type SubagentTaskRunRequest,
} from '../src/agents/subagentManager.js'
import { loadAgentDefinitions, type AgentDefinition } from '../src/agents/definitions.js'
import { UserProfileStore } from '../src/memory/userProfile.js'
import { ProfileAgent } from '../src/agents/profileAgent.js'

const definition: AgentDefinition = {
  name: 'researcher',
  description: 'Inspect a bounded implementation question.',
  systemPrompt: 'Inspect the task and report concrete findings.',
  model: 'gpt-test',
  tools: ['ReadFile', 'WriteFile'],
  allowedTools: ['ReadFile', 'WriteFile'],
  excludeTools: ['WriteFile'],
  source: 'test',
  maxDepth: 3,
  isolation: '',
}

test('compaction and automatic compaction use caller-owned model and compactor ports', async () => {
  const requests: string[] = []
  const agent = new CompactionAgent({
    model: 'gpt-test',
    completion: request => {
      requests.push(request.prompt)
      return { choices: [{ message: { content: 'durable summary of the resolved request' } }] }
    },
  })
  const messages = [
    { role: 'system', content: 'Remain factual.' },
    { role: 'user', content: 'old request '.repeat(100) },
    { role: 'assistant', content: 'old result '.repeat(100) },
    { role: 'user', content: 'latest request' },
  ]
  const compacted = await agent.summarizeMessages(messages)

  expect(requests).toHaveLength(1)
  expect(requests[0]).toContain('CONTEXT TO SUMMARIZE:')
  expect(compacted[0]).toEqual(messages[0])
  expect(compacted.at(-1)).toEqual(messages.at(-1))
  expect(compacted.some(message => String(message.content).includes('durable summary'))).toBeTrue()

  const automatic = new AutoCompactAgent({
    maxContextTokens: 20,
    compactThreshold: 0.5,
    compactor: agent,
  })
  const result = await automatic.compact(messages)
  expect(result.metadata.summaryCreated).toBeTrue()
  expect(automatic.getStatistics()).toMatchObject({ compactionCount: 1 })

  const unavailable = await new AutoCompactAgent({ maxContextTokens: 20, compactThreshold: 0.5 }).compact(messages)
  expect(unavailable.metadata).toMatchObject({ summaryCreated: false, reason: 'no_summary_agent' })
})

test('profile agent persists heuristic domains, preferences, feedback, and injected summaries', async () => {
  const store = new UserProfileStore()
  const agent = new ProfileAgent({
    store,
    llmSummarizer: () => 'the user prefers concise technical answers',
    nerExtractor: () => ({ technologies: ['TypeScript'] }),
  })
  const result = await agent.update('alice', {
    userPrompt: 'Please use TypeScript and pytest, I prefer concise responses.',
    agentResponse: 'I will keep it concise.',
    signals: ['correction'],
  })
  const profile = store.get('alice')

  expect(result.domainsAdded).toEqual(expect.arrayContaining(['python', 'javascript']))
  expect(result.prefsAdded.some(preference => preference.toLowerCase().includes('prefer'))).toBeTrue()
  expect(profile?.feedbackHistory.map(entry => entry.signal)).toContain('correction')
  expect(profile?.notes).toContain('the user prefers concise technical answers')
  expect(agent.extractEntities('TypeScript')).toEqual({ technologies: ['TypeScript'] })
})

test('subagent manager layers definitions, queues native runner input, reports tools, and requires a worktree port', async () => {
  const calls: SubagentTaskRunRequest[] = []
  const ids = ['task-1', 'task-2']
  const manager = new SubAgentManager({
    idFactory: () => ids.shift() ?? `task-${crypto.randomUUID()}`,
    maxConcurrent: 1,
    pathResolver: path => `/workspace/${path}`,
    runner: async request => {
      calls.push(request)
      request.report.text(`working:${request.prompt}`)
      request.report.toolStart({
        toolCallId: `read-${calls.length}`,
        name: 'ReadFile',
        inputs: { file_path: 'src/service.ts' },
      })
      request.report.toolEnd({
        toolCallId: `read-${calls.length}`,
        name: 'ReadFile',
        permitted: true,
        result: 'source',
      })
      await Bun.sleep(5)
      return { content: `done:${request.prompt}` }
    },
  })

  const task = await manager.spawn({
    prompt: 'inspect the service',
    systemPrompt: 'base system',
    agentDefinition: definition,
    name: 'research',
  })
  expect(await manager.sendMessage('research', 'inspect tests too')).toBeTrue()
  const settled = await manager.wait(task.id, 1_000)

  expect(settled?.status).toBe('completed')
  expect(calls.map(call => call.prompt)).toEqual(['inspect the service', 'inspect tests too'])
  expect(calls[0]?.systemPrompt).toContain('You are now running as a subagent')
  expect(calls[0]?.systemPrompt).toContain('The filesystem is shared with the parent and other agents.')
  expect(calls[0]?.systemPrompt).toContain('Return a distilled final summary')
  expect(calls[0]?.config).toMatchObject({ model: 'gpt-test', _toolsAllowed: ['ReadFile', 'WriteFile'] })
  expect(task.readFiles).toEqual(new Set(['/workspace/src/service.ts']))
  expect(manager.peekMailbox().map(event => event.type)).toEqual(expect.arrayContaining(['spawn', 'tool_start', 'tool_end', 'done']))

  const isolated = await manager.spawn({ prompt: 'isolated task', isolation: 'worktree' })
  expect(isolated).toMatchObject({ status: 'failed', error: "isolation='worktree' requires a configured worktree port" })
  await manager.close()
})

test('subagent manager coalesces thinking bursts and flushes the latest bounded preview at lifecycle boundaries', async () => {
  const manager = new SubAgentManager({
    idFactory: () => 'thinking-task',
    thinkingFlushIntervalMs: 10,
    runner: async request => {
      for (let index = 0; index < 80; index += 1) request.report.thinking(`first-${index}|`)
      request.report.toolStart({ toolCallId: 'read-1', name: 'ReadFile', inputs: { file_path: 'src/a.ts' } })
      for (let index = 0; index < 80; index += 1) request.report.thinking(`second-${index}|`)
      request.report.toolEnd({
        toolCallId: 'read-1',
        name: 'ReadFile',
        permitted: true,
        result: 'ok',
      })
      for (let index = 0; index < 80; index += 1) request.report.thinking(`final-${index}|`)
      return { content: 'finished' }
    },
  })

  const task = await manager.spawn({ prompt: 'exercise thinking coalescing' })
  await manager.wait(task.id, 1_000)
  const events = manager.peekMailbox()
  const thinking = events.filter(event => event.type === 'thinking')
  const previews = thinking.map(event => String(event.data.preview ?? ''))

  expect(thinking).toHaveLength(3)
  expect(previews.every(preview => preview.length <= 400)).toBeTrue()
  expect(previews).toEqual([
    expect.stringContaining('first-79|'),
    expect.stringContaining('second-79|'),
    expect.stringContaining('final-79|'),
  ])
  expect(events.findIndex(event => event.type === 'thinking')).toBeLessThan(
    events.findIndex(event => event.type === 'tool_start'),
  )
  expect(events.findLastIndex(event => event.type === 'thinking')).toBeLessThan(
    events.findIndex(event => event.type === 'done'),
  )
  await manager.close()
})

test('subagent manager limits free-running thinking to one event per cadence and clears cancellation timers', async () => {
  let releaseSecondBurst: (() => void) | undefined
  const secondBurst = new Promise<void>(resolve => {
    releaseSecondBurst = resolve
  })
  let started: (() => void) | undefined
  const running = new Promise<void>(resolve => {
    started = resolve
  })
  const ids = ['paced-task', 'cancelled-task']
  const manager = new SubAgentManager({
    idFactory: () => ids.shift() ?? crypto.randomUUID(),
    thinkingFlushIntervalMs: 15,
    runner: async request => {
      if (request.task.id === 'cancelled-task') {
        request.report.thinking('cancelled preview')
        started?.()
        await new Promise<void>((_resolve, reject) => {
          request.cancelSignal.addEventListener('abort', () => reject(request.cancelSignal.reason), { once: true })
        })
      }
      for (let index = 0; index < 100; index += 1) request.report.thinking(`a${index}|`)
      await Bun.sleep(35)
      for (let index = 0; index < 100; index += 1) request.report.thinking(`b${index}|`)
      releaseSecondBurst?.()
      await Bun.sleep(35)
      return { content: 'paced' }
    },
  })

  const paced = await manager.spawn({ prompt: 'pace live reasoning' })
  await secondBurst
  await manager.wait(paced.id, 1_000)
  const pacedEvents = manager.peekMailbox().filter(event => event.taskId === paced.id && event.type === 'thinking')
  expect(pacedEvents).toHaveLength(2)
  expect(String(pacedEvents[0]?.data.preview)).toContain('a99|')
  expect(String(pacedEvents[1]?.data.preview)).toContain('b99|')

  const cancelled = await manager.spawn({ prompt: 'cancel live reasoning' })
  await running
  expect(manager.cancel(cancelled.id)).toBeTrue()
  await Bun.sleep(40)
  const cancelledEvents = manager.peekMailbox().filter(event => event.taskId === cancelled.id)
  expect(cancelledEvents.filter(event => event.type === 'thinking')).toHaveLength(1)
  expect(cancelledEvents.findIndex(event => event.type === 'thinking')).toBeLessThan(
    cancelledEvents.findIndex(event => event.type === 'cancelled'),
  )
  await manager.close()
})

test('subagent manager flushes pending thinking before failure events', async () => {
  const manager = new SubAgentManager({
    idFactory: () => 'failed-task',
    thinkingFlushIntervalMs: 1_000,
    runner: request => {
      request.report.thinking('last reasoning before failure')
      throw new Error('runner failed')
    },
  })

  const task = await manager.spawn({ prompt: 'fail after reasoning' })
  await manager.wait(task.id, 1_000)
  const types = manager.peekMailbox().filter(event => event.taskId === task.id).map(event => event.type)

  expect(task.status).toBe('failed')
  expect(types.indexOf('thinking')).toBeLessThan(types.indexOf('error'))
  expect(types.indexOf('error')).toBeLessThan(types.indexOf('done'))
  await manager.close()
})

test('subagent manager reports an empty final response as a failed task', async () => {
  const manager = new SubAgentManager({
    idFactory: () => 'empty-response-task',
    runner: () => ({ content: '   \n\t' }),
  })

  const task = await manager.spawn({ prompt: 'return a useful result' })
  await manager.wait(task.id, 1_000)

  expect(task.status).toBe('failed')
  expect(task.error).toBe('Subagent completed without a final response')
  expect(task.result).toBe('Error: Subagent completed without a final response')
  expect(manager.peekMailbox().filter(event => event.taskId === task.id).map(event => event.type)).toEqual(
    expect.arrayContaining(['error', 'done']),
  )
  await manager.close()
})

test('subagent tool filtering prevents recursive delegation and parent-mode mutation', async () => {
  const calls: string[] = []
  const filtered = filterSubagentTools({
    isSubagent: true,
    toolSchemas: [
      { name: 'ReadFile' },
      { name: 'SetInteractionModeTool' },
      { name: 'SpawnAgents' },
      { name: 'WriteFile' },
    ],
    config: { _toolsAllowed: ['ReadFile', 'SetInteractionModeTool', 'SpawnAgents'] },
    toolExecutor: toolName => {
      calls.push(toolName)
      return 'ok'
    },
  })

  expect(filtered.toolSchemas.map(schema => schema.name)).toEqual(['ReadFile'])
  expect(await filtered.execute?.('ReadFile', {})).toBe('ok')
  expect(await filtered.execute?.('SetInteractionModeTool', {})).toBe(
    "Error: tool 'SetInteractionModeTool' is not allowed for this agent.",
  )
  expect(await filtered.execute?.('SpawnAgents', {})).toBe("Error: tool 'SpawnAgents' is not allowed for this agent.")
  expect(calls).toEqual(['ReadFile'])
})

test('subagent manager enforces explicit and definition depth limits with explicit errors', async () => {
  const depths: number[] = []
  const manager = new SubAgentManager({
    maxDepth: 5,
    runner: request => {
      depths.push(request.depth)
      return { content: 'ok' }
    },
  })

  const atDefaultLimit = await manager.spawn({ prompt: 'too deep', depth: 5 })
  expect(atDefaultLimit.status).toBe('failed')
  expect(atDefaultLimit.error).toBe('Max depth (5, from manager default) exceeded: cannot spawn at depth 5')

  const shallow = await manager.spawn({ prompt: 'shallow child', depth: 4 })
  await manager.wait(shallow.id, 1_000)
  expect(shallow.status).toBe('completed')
  // Child runs originating inside the task see the next depth level.
  expect(depths).toEqual([5])

  const strict: AgentDefinition = { ...definition, maxDepth: 1 }
  const blocked = await manager.spawn({ prompt: 'blocked child', depth: 1, agentDefinition: strict })
  expect(blocked.status).toBe('failed')
  expect(blocked.error).toBe("Max depth (1, from agent definition 'researcher') exceeded: cannot spawn at depth 1")

  const allowed = await manager.spawn({ prompt: 'allowed child', depth: 2, agentDefinition: definition })
  await manager.wait(allowed.id, 1_000)
  expect(allowed.status).toBe('completed')
  await manager.close()
})

test('subagent manager defaults to unbounded depth and concurrency', async () => {
  const manager = new SubAgentManager({ runner: () => ({ content: 'ok' }) })

  expect(manager.maxDepth).toBe(Number.POSITIVE_INFINITY)
  expect(manager.maxConcurrent).toBe(Number.POSITIVE_INFINITY)

  const deep = await manager.spawn({ prompt: 'deep spawn', depth: 25 })
  await manager.wait(deep.id, 1_000)
  expect(deep.status).toBe('completed')
  await manager.close()
})

test('subagent manager enforces YAML max_depth loaded through the definition catalog', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-subagent-depth-'))
  try {
    const projectAgents = join(root, '.xerxes', 'agents')
    await mkdir(projectAgents, { recursive: true })
    await writeFile(join(projectAgents, 'nested.yaml'), `version: 1
agent:
  name: nested
  system_prompt: nested worker
  max_depth: 2
`, 'utf8')
    const definitions = loadAgentDefinitions({
      builtinDefinitions: new Map(),
      cwd: root,
      userDirectory: join(root, 'user'),
      projectDirectory: projectAgents,
    })
    const nested = definitions.get('nested')
    if (nested === undefined) throw new Error('nested definition missing')
    expect(nested.maxDepth).toBe(2)

    const manager = new SubAgentManager({ runner: () => ({ content: 'ok' }) })
    const blocked = await manager.spawn({ prompt: 'too deep', depth: 2, agentDefinition: nested })
    expect(blocked.status).toBe('failed')
    expect(blocked.error).toBe("Max depth (2, from agent definition 'nested') exceeded: cannot spawn at depth 2")
    const allowed = await manager.spawn({ prompt: 'shallow', depth: 1, agentDefinition: nested })
    await manager.wait(allowed.id, 1_000)
    expect(allowed.status).toBe('completed')
    await manager.close()
  } finally {
    await rm(root, { force: true, recursive: true })
  }
})

test('subagent manager contains event sink failures so healthy turns still complete', async () => {
  const errors: unknown[][] = []
  const spy = spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
    errors.push(args)
  })
  try {
    const manager = new SubAgentManager({
      runner: async request => {
        request.report.toolStart({ toolCallId: 't1', name: 'ReadFile', inputs: {} })
        request.report.toolEnd({ toolCallId: 't1', name: 'ReadFile', permitted: true, result: 'ok' })
        return { content: 'survived' }
      },
      onEvent: () => { throw new Error('sink exploded') },
    })
    const task = await manager.spawn({ prompt: 'resilient turn' })
    await manager.wait(task.id, 1_000)

    expect(task.status).toBe('completed')
    expect(task.result).toBe('survived')
    expect(errors.length).toBeGreaterThan(0)
    expect(String(errors[0]?.[0])).toContain('sink exploded')
    await manager.close()
  } finally {
    spy.mockRestore()
  }
})

test('subagent manager drainMailbox drops only events at or below the drained cursor', async () => {
  const manager = new SubAgentManager({ runner: () => ({ content: 'ok' }) })
  const task = await manager.spawn({ prompt: 'mailbox' })
  await manager.wait(task.id, 1_000)
  const all = manager.peekMailbox()
  expect(all.length).toBeGreaterThan(0)

  // Draining returns newer events while retaining them for lower-cursor consumers.
  const first = manager.drainMailbox(0)
  expect(first.map(event => event.sequence)).toEqual(all.map(event => event.sequence))
  expect(manager.peekMailbox(0)).toHaveLength(all.length)
  const second = manager.drainMailbox(0)
  expect(second.map(event => event.sequence)).toEqual(first.map(event => event.sequence))

  // Advancing the cursor past the drained events finally drops them.
  expect(manager.drainMailbox(manager.latestSequence())).toEqual([])
  expect(manager.peekMailbox(0)).toHaveLength(0)
  await manager.close()
})
