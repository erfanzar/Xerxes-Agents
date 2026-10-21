// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { SpawnedAgentManager } from '../src/operators/subagents.js'
import { ClaudeAgentTools } from '../src/tools/claudeTools/agentOps.js'
import { runWorkflowScript, type WorkflowAgentPort } from '../src/tools/claudeTools/workflowScript.js'

function port(): WorkflowAgentPort {
  return {
    async run(request) {
      return { id: request.prompt, status: request.prompt === 'fail' ? 'failed' : 'completed', output: request.prompt, tokens: 1 }
    },
    async correct() { throw new Error('Unexpected correction') },
    closeAll() {},
  }
}

test('duplicate labels fail the ledger even when the expected count matches, without losing results', async () => {
  const result = await runWorkflowScript({
    name: 'Duplicates', concurrency: 2, maxAgents: 10, expectAgents: 2, port: port(),
    script: `return await parallel([() => agent('first', { label: 'same' }), () => agent('second', { label: 'same' })])`,
  })
  expect(result.status).toBe('completed')
  expect(result.result).toEqual(['first', 'second'])
  expect(result.ledger).toMatchObject({ expected: 2, started: 2, completed: 2, failed: 0, resumed: 0, duplicateLabels: ['same'], ok: false })
  expect(result.ledger.problems).toEqual(['Duplicate agent labels: "same". Give every agent() call a unique label.'])
})

test('repeated titles are display text, not identities, so they never count as duplicates', async () => {
  const result = await runWorkflowScript({
    name: 'Titles', concurrency: 3, maxAgents: 10, expectAgents: 3, port: port(),
    script: `return await parallel(['a', 'b', 'c'].map(p => () => agent(p, { title: 'Review slice' })))`,
  })
  expect(result.result).toEqual(['a', 'b', 'c'])
  expect(result.ledger).toMatchObject({ expected: 3, started: 3, completed: 3, failed: 0, duplicateLabels: [], ok: true })
  expect(result.ledger.problems).toEqual([])
})

test('duplicate-label problems precede exact-count problems', async () => {
  const result = await runWorkflowScript({
    name: 'Mismatch', concurrency: 2, maxAgents: 10, expectAgents: 3, port: port(),
    script: `return await parallel([() => agent('first', { label: 'same' }), () => agent('second', { label: 'same' })])`,
  })
  expect(result.result).toEqual(['first', 'second'])
  expect(result.ledger.problems[0]).toStartWith('Duplicate agent labels:')
  expect(result.ledger.problems[1]).toBe('Expected 3 agents, but started 2.')
  expect(result.ledger.ok).toBe(false)
})

test('unlabelled calls get distinct identities, and failures returned as null remain observable', async () => {
  const result = await runWorkflowScript({
    name: 'Mixed', concurrency: 3, maxAgents: 10, expectAgents: 3, port: port(),
    script: `return await parallel([() => agent('first'), () => agent('fail'), () => agent('third')])`,
  })
  expect(result.result).toEqual(['first', null, 'third'])
  expect(result.ledger).toMatchObject({ started: 3, completed: 2, failed: 1, resumed: 0, duplicateLabels: [], ok: false })
})

test('refused agent() calls are tracked even when the runaway guard prevents a start', async () => {
  const result = await runWorkflowScript({
    name: 'Refused', concurrency: 2, maxAgents: 1, expectAgents: 2, port: port(),
    script: `return await parallel([() => agent('first', { label: 'same' }), () => agent('second', { label: 'same' })])`,
  })
  expect(result.result).toEqual(['first', null])
  expect(result.ledger).toMatchObject({ started: 1, completed: 1, failed: 1, duplicateLabels: ['same'], ok: false })
  expect(result.ledger.problems[1]).toBe('Expected 2 agents, but started 1.')
})

test('long distinct labels and large result content are preserved, with a size warning rather than truncation', async () => {
  const result = await runWorkflowScript({
    name: 'Preserve', concurrency: 2, maxAgents: 10, expectAgents: 2, port: port(),
    args: { prefix: 'x'.repeat(130), content: 'z'.repeat(200_100) },
    script: `const outputs = await parallel([() => agent('first', { label: args.prefix + 'a' }), () => agent('second', { label: args.prefix + 'b' })]); return { outputs, content: args.content }`,
  })
  expect(result.ledger.ok).toBe(true)
  expect(result.result).toEqual({ outputs: ['first', 'second'], content: 'z'.repeat(200_100) })
  expect(result.logs.some(line => line.includes('200000') && line.includes('preserved'))).toBe(true)
})

test('the tool returns the ledger and cached-on-resume results are counted separately without charging them again', async () => {
  let runs = 0
  const manager = new SpawnedAgentManager({
    runner: async request => { runs++; return { content: request.input } },
  })
  const tools = new ClaudeAgentTools({ manager })
  const context = { metadata: {}, sessionId: 'ledger-test' }
  const first = await tools.execute('Workflow', { expect_agents: 1, script: `return await agent('one', { label: 'one' })` }, context) as Record<string, unknown>
  const second = await tools.execute('Workflow', { resume: String(first.workflow_id) }, context) as Record<string, unknown>
  expect(runs).toBe(1)
  expect(second.result).toBe('one')
  expect(second.ledger).toMatchObject({ expected: 1, started: 1, completed: 1, failed: 0, resumed: 1, ok: true })
  expect(second.tokens).toBe(0)
})

test('duplicates are reported without an expected count, and a count-only mismatch preserves the result', async () => {
  const duplicate = await runWorkflowScript({ name: 'No expectation', concurrency: 2, maxAgents: 2, port: port(), script: `return await parallel([() => agent('first', { label: 'same' }), () => agent('second', { label: 'same' })])` })
  expect(duplicate.ledger).toMatchObject({ duplicateLabels: ['same'], ok: false })
  expect(duplicate.ledger.expected).toBeUndefined()
  const mismatch = await runWorkflowScript({ name: 'Count', concurrency: 1, maxAgents: 2, expectAgents: 2, port: port(), script: `return await agent('only')` })
  expect(mismatch.result).toBe('only')
  expect(mismatch.ledger.problems).toEqual(['Expected 2 agents, but started 1.'])
})

test('cached outcomes do not consume the new run token or cost budget', async () => {
  const cachedPort: WorkflowAgentPort = {
    ...port(),
    async run(request) { return { id: 'cached', status: 'completed', output: request.prompt, tokens: 1000, costUsd: 2, cached: true } },
  }
  const result = await runWorkflowScript({ name: 'Cached', concurrency: 1, maxAgents: 2, expectAgents: 2, tokenBudget: 1, port: cachedPort, script: `const first = await agent('first'); return [first, await agent('second')]` })
  expect(result.result).toEqual(['first', 'second'])
  expect(result.tokens).toBe(0)
  expect(result.cost).toEqual({ usd: 0, unpriced: 0 })
  expect(result.ledger).toMatchObject({ resumed: 2, completed: 2, ok: true })
})

test('invalid prompts remain ledger failures without ever starting an agent', async () => {
  const result = await runWorkflowScript({ name: 'Invalid', concurrency: 1, maxAgents: 1, expectAgents: 1, port: port(), script: `return await parallel([() => agent('', { label: 'empty' })])` })
  expect(result.result).toEqual([null])
  expect(result.ledger).toMatchObject({ started: 0, failed: 1, ok: false })
})

test('cancellation exposes unfinished calls instead of a successful ledger', async () => {
  const controller = new AbortController()
  const waitingPort: WorkflowAgentPort = {
    ...port(),
    async run(_request, signal) {
      setTimeout(() => controller.abort(new Error('Stop test run')), 1)
      return new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(signal.reason), { once: true }))
    },
  }
  const result = await runWorkflowScript({ name: 'Cancelled', concurrency: 1, maxAgents: 1, expectAgents: 1, signal: controller.signal, port: waitingPort, script: `return await agent('slow')` })
  expect(result.status).toBe('cancelled')
  expect(result.ledger).toMatchObject({ started: 1, completed: 0, failed: 1, ok: false })
})

test('expect_agents is validated before a workflow starts', async () => {
  const tools = new ClaudeAgentTools({ manager: new SpawnedAgentManager({ runner: async () => ({ content: 'unused' }) }) })
  await expect(tools.execute('Workflow', { expect_agents: 0, script: `return 'unused'` }, { metadata: {} })).rejects.toThrow('positive integer')
  await expect(runWorkflowScript({ name: 'Bad count', concurrency: 1, maxAgents: 1, expectAgents: 1.5, script: `return null`, port: port() })).rejects.toThrow('positive integer')
})
