// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { AutoCompactAgent } from '../src/agents/autoCompactAgent.js'
import { CompactionAgent } from '../src/agents/compactionAgent.js'
import {
  filterSubagentTools,
  SubAgentManager,
  type SubagentTaskRunRequest,
} from '../src/agents/subagentManager.js'
import type { AgentDefinition } from '../src/agents/definitions.js'
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
