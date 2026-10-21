// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { SpawnedAgentManager } from '../src/operators/subagents.js'
import { preflightWorkflow, workflowPreflightPlan } from '../src/runtime/preflight.js'
import { ClaudeAgentTools } from '../src/tools/claudeTools/agentOps.js'

test('preflight collects actionable problems for worktree, model/profile and quota checks', async () => {
  const checked: Array<readonly [string | undefined, string | undefined]> = []
  const result = await preflightWorkflow({
    expected: 12, requests: [{ isolation: 'worktree', profile: 'missing', model: 'fast' }],
  }, {
    worktreeAvailable: () => false,
    resolveModel: (profile, model) => { checked.push([profile, model]); return 'Profile is not configured' },
    quota: () => ({ exhausted: true, detail: 'Hourly quota resets in 30 minutes' }),
  })
  expect(result.ok).toBe(false)
  expect(result.problems).toHaveLength(3)
  expect(result.problems[0]).toContain('worktree adapter')
  expect(result.problems[1]).toContain('missing')
  expect(result.problems[1]).toContain('Profile is not configured')
  expect(result.problems[2]).toContain('Hourly quota resets in 30 minutes')
  expect(checked).toEqual([['missing', 'fast']])
})

test('successful checks are real injected observations; duplicate model/profile pairs are resolved once', async () => {
  let resolutions = 0
  const result = await preflightWorkflow({ requests: [{ model: 'fast' }, { model: 'fast', isolation: 'worktree' }] }, {
    worktreeAvailable: () => true,
    resolveModel: async () => { resolutions++; return 'ok' },
    quota: () => ({ exhausted: false, detail: 'Available' }),
  })
  expect(result).toMatchObject({ ok: true, problems: [], notes: [] })
  expect(resolutions).toBe(1)
})

test('unavailable ports are explicitly noted rather than fabricated as verified checks', async () => {
  const result = await preflightWorkflow({ requests: [{ isolation: 'worktree', model: 'fast' }] }, {})
  expect(result.ok).toBe(true)
  // ok with nothing checked must say so, not read as a verified pass.
  expect(result.checked).toBe(0)
  expect(result.notes).toHaveLength(4)
  expect(result.notes.join(' ')).toContain('worktree')
  expect(result.notes.join(' ')).toContain('model/profile')
  expect(result.notes.join(' ')).toContain('quota')
  expect(result.notes.join(' ')).toContain('nothing was verified')
  const verified = await preflightWorkflow({ requests: [{ isolation: 'worktree' }] }, { worktreeAvailable: () => true, resolveModel: () => 'ok' })
  expect(verified).toMatchObject({ ok: true, checked: 2 })
})

test('a throwing port is an actionable preflight failure', async () => {
  const result = await preflightWorkflow({ requests: [{ model: 'fast' }] }, {
    resolveModel: () => { throw new Error('Cannot read configured credentials') },
    quota: () => { throw new Error('Usage reporter unavailable') },
  })
  expect(result.ok).toBe(false)
  expect(result.problems.join(' ')).toContain('Cannot read configured credentials')
  expect(result.problems.join(' ')).toContain('Usage reporter unavailable')
})

test('static plans discover literal options and nested args without executing a script', () => {
  const plan = workflowPreflightPlan({
    expected: 11,
    script: `throw new Error('must never execute'); return agent('read', { isolation: 'worktree', model: 'fast', profile: 'local' })`,
    args: { tasks: [{ opts: { model: 'strong', provider_profile: 'remote' } }] },
  })
  expect(plan.expected).toBe(11)
  expect(plan.requests).toContainEqual({ isolation: 'worktree', model: 'fast', profile: 'local' })
  expect(plan.requests).toContainEqual({ model: 'strong', profile: 'remote' })
})

test('script comments and prompt strings do not invent worktree or provider requests', () => {
  const plan = workflowPreflightPlan({ script: `// { isolation: 'worktree', model: 'fake' }\nreturn agent("Explain { isolation: 'worktree' }", { model: 'real' })` })
  expect(plan.requests).toEqual([{ model: 'real' }])
})

test('static args references are read, but dynamic expressions are not guessed as literal model names', () => {
  const plan = workflowPreflightPlan({
    script: `const a = { model: args.config.model, profile: args.config.profile, isolation: args.config.isolation }; const b = { model: 'prefix-' + suffix }; const c = { model: args.config.model + suffix }`,
    args: { config: { model: 'fast', profile: 'local', isolation: 'worktree' } },
  })
  expect(plan.requests).toEqual([
    { model: 'fast', profile: 'local', isolation: 'worktree' },
    { model: 'fast', profile: 'local', isolation: 'worktree' },
  ])
})

test('a large workflow refused by preflight starts zero agents, even if ordinary calls precede isolation', async () => {
  let starts = 0
  const manager = new SpawnedAgentManager({ runner: async () => { starts++; return { content: 'should not run' } } })
  const tools = new ClaudeAgentTools({ manager, workflowPreflight: { worktreeAvailable: () => false } })
  const result = await tools.execute('Workflow', {
    expect_agents: 11,
    script: `await agent('spend money first'); return await agent('isolated', { isolation: 'worktree' })`,
  }, { metadata: {} })
  expect(result).toMatchObject({ status: 'failed', error: expect.stringContaining('preflight'), preflight: { ok: false }, agents: { started: 0 } })
  expect(starts).toBe(0)
  expect(manager.listHandles()).toEqual([])
})

test('explicit max_agents above ten also preflights model/profile failures from args', async () => {
  let starts = 0
  const tools = new ClaudeAgentTools({
    manager: new SpawnedAgentManager({ runner: async () => { starts++; return { content: 'unused' } } }),
    workflowPreflight: { resolveModel: () => 'Configure a provider key for this profile' },
  })
  const result = await tools.execute('Workflow', {
    max_agents: 11, args: { opts: { model: 'fast', profile: 'unconfigured' } },
    script: `return await agent('read', args.opts)`,
  }, { metadata: {} })
  expect(result).toMatchObject({ status: 'failed', preflight: { ok: false } })
  expect(starts).toBe(0)
})

test('the default runaway backstop alone does not trigger preflight; explicit large runs expose skipped checks', async () => {
  let checks = 0
  const manager = new SpawnedAgentManager({ runner: async request => ({ content: request.input }) })
  const tools = new ClaudeAgentTools({ manager, workflowPreflight: { resolveModel: () => { checks++; return 'ok' } } })
  const small = await tools.execute('Workflow', { script: `return await agent('small')` }, { metadata: {} }) as Record<string, unknown>
  expect(small.status).toBe('completed')
  expect(checks).toBe(0)
  const large = await tools.execute('Workflow', { max_agents: 11, script: `return await agent('large')` }, { metadata: {} }) as Record<string, unknown>
  expect(large.status).toBe('completed')
  expect(checks).toBe(1)
  expect(large.preflight).toMatchObject({ ok: true, notes: expect.arrayContaining([expect.stringContaining('quota')]) })
})
