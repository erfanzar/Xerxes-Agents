// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, beforeEach, expect, test } from 'bun:test'
import { appendFile, mkdir, mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { renderTrajectory } from '../src/context/trajectory.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import {
  buildReflectionPrompt, MAX_REFLECTION_PROMPT_CHARS, parseReflectionProposal, proposeReflection,
  ReflectionError, ReflectionProposalStore, reflectionTrigger,
  type ReflectionApplier, type ReflectionPromptInput, type ReflectionProposal,
} from '../src/runtime/reflection.js'
import { SkillBundleStore } from '../src/tools/agentMetaTools.js'
import { registerReflectionTools } from '../src/tools/reflectionTools.js'
import type { JsonObject } from '../src/types/toolCalls.js'

const messages = [
  { role: 'user', content: 'Make fixture creation reliable.' },
  { role: 'assistant', tool_calls: [{ id: 'check-1', function: { name: 'exec_command', arguments: { cmd: 'bun', args: ['test'] } } }] },
  { role: 'tool', tool_call_id: 'check-1', content: 'ENOENT: fixture directory missing', is_error: true },
  { role: 'assistant', tool_calls: [{ id: 'check-2', function: { name: 'exec_command', arguments: { cmd: 'bun', args: ['test'] } } }] },
  { role: 'tool', tool_call_id: 'check-2', content: '4 pass, 0 fail after mkdir' },
]
const evidence = '[ERROR] call_id="check-1" ENOENT: fixture directory missing'
const proposal: ReflectionProposal = {
  kind: 'skill_create', target: 'fixture-setup',
  rationale: 'The [ERROR] shows that fixture setup must create its directory first.',
  evidence: [evidence],
  skill: { description: 'Set up reliable temporary test fixtures.', instructions: 'Create a temporary directory before writing fixture files. Clean it up after the test.' },
}

function promptInput(overrides: Partial<ReflectionPromptInput> = {}): ReflectionPromptInput {
  return {
    objective: 'Make fixture creation reliable.', trigger: 'goal_complete', messages,
    criteria: [{ id: 'tests', description: 'Tests pass', evidence: { toolCallId: 'check-2', summary: '4 pass, 0 fail', recordedAt: 1 } }],
    existingSkillNames: [], existingAgentNames: ['reviewer'], ...overrides,
  }
}

let directory: string
let workspace: string
let store: ReflectionProposalStore
let calls: unknown[]
let applier: ReflectionApplier

beforeEach(async () => {
  directory = await mkdtemp(join(tmpdir(), 'xerxes-reflection-'))
  workspace = join(directory, 'workspace')
  await mkdir(workspace)
  store = new ReflectionProposalStore({ directory, now: () => 1_000 })
  calls = []
  applier = {
    applySkill: request => { calls.push(request); return { ok: true, name: request.name } },
    applyAgent: (name, yaml) => { calls.push({ name, yaml }); return { ok: true, name } },
  }
})

afterEach(async () => { await rm(directory, { recursive: true, force: true }) })

test('trigger defaults are inclusive, terminal goals take precedence, and ordinary turns do not reflect', () => {
  expect(reflectionTrigger({ toolCallCount: 24, turnDurationMs: 1_199_999 })).toBeUndefined()
  expect(reflectionTrigger({ toolCallCount: 25, turnDurationMs: 0 })).toBe('long_turn')
  expect(reflectionTrigger({ toolCallCount: 0, turnDurationMs: 1_200_000 })).toBe('long_turn')
  expect(reflectionTrigger({ goalPhase: 'complete', toolCallCount: 25, turnDurationMs: 1_200_000 })).toBe('goal_complete')
  expect(reflectionTrigger({ goalPhase: 'blocked', toolCallCount: 0, turnDurationMs: 0 })).toBe('goal_blocked')
  expect(reflectionTrigger({ goalPhase: 'active', toolCallCount: 0, turnDurationMs: 0 })).toBeUndefined()
})

test('trigger accepts explicit thresholds and refuses non-finite or negative measurements', () => {
  expect(reflectionTrigger({ toolCallCount: 3, turnDurationMs: 0, thresholds: { toolCallCount: 3, turnDurationMs: 100 } })).toBe('long_turn')
  expect(reflectionTrigger({ toolCallCount: 0, turnDurationMs: 100, thresholds: { toolCallCount: 3, turnDurationMs: 100 } })).toBe('long_turn')
  for (const value of [Number.NaN, Number.POSITIVE_INFINITY, -1]) {
    expect(reflectionTrigger({ toolCallCount: value, turnDurationMs: value })).toBeUndefined()
  }
  expect(() => reflectionTrigger({ toolCallCount: 1, turnDurationMs: 0, thresholds: { toolCallCount: 0 } })).toThrow()
})

test('reflection prompt uses the shared trajectory grammar, evidence, catalog names and explicit human-review rules', () => {
  const prompt = buildReflectionPrompt(promptInput({ existingSkillNames: ['existing-skill'] }))
  expect(prompt).toContain(renderTrajectory(messages, { maxChars: 24_000, maxObservationChars: 2_000, maxArgumentChars: 500 }))
  expect(prompt).toContain('[ACTION] exec_command(')
  expect(prompt).toContain(evidence)
  expect(prompt).toContain('[OBSERVATION] call_id="check-2" 4 pass, 0 fail after mkdir')
  expect(prompt).toContain('existing-skill')
  expect(prompt).toContain('reviewer')
  expect(prompt).toContain('4 pass, 0 fail')
  expect(prompt).toContain('at most ONE')
  expect(prompt).toContain('human approval')
  expect(prompt).toContain('none')
  expect(prompt).toContain('not evidence of a tool failure')
})

test('reflection prompt bounds trajectory, objective, criteria and both catalogs, and discloses clipping', () => {
  const prompt = buildReflectionPrompt(promptInput({
    objective: 'o'.repeat(100_000), evidenceSummary: 'e'.repeat(100_000),
    criteria: [{ id: 'large', description: 'd'.repeat(100_000) }],
    existingSkillNames: Array.from({ length: 10_000 }, (_, index) => `skill-${index}`),
    existingAgentNames: Array.from({ length: 10_000 }, (_, index) => `agent-${index}`),
    messages: [{ role: 'user', content: 'head' + 'x'.repeat(100_000) + 'tail' }],
  }))
  expect(prompt.length).toBeLessThanOrEqual(MAX_REFLECTION_PROMPT_CHARS)
  expect(prompt).toContain('transcript truncated by the renderer')
  expect(prompt).toContain('[context clipped]')
  expect(prompt).toContain('tail')
})

test('parser accepts each concrete proposal kind and an explicit none', () => {
  const candidates: ReflectionProposal[] = [proposal, { ...proposal, kind: 'skill_update' }, {
    kind: 'agent_update', target: 'reviewer', rationale: proposal.rationale, evidence: [evidence],
    agent: { yaml: 'name: reviewer\nsystem_prompt: Check fixture setup first.\n' },
  }, {
    kind: 'plugin_idea', target: 'fixture-helper', rationale: proposal.rationale, evidence: [evidence],
    plugin: { description: 'A tool that prepares isolated fixtures and cleans them up.' },
  }]
  for (const candidate of candidates) expect(parseReflectionProposal(JSON.stringify(candidate))).toEqual({ type: 'proposal', proposal: candidate })
  expect(parseReflectionProposal('none')).toEqual({ type: 'none' })
  expect(parseReflectionProposal('{"kind":"none"}')).toEqual({ type: 'none' })
})

test('parser refuses garbage, multiple proposals, unknown fields and unknown kinds', () => {
  const json = JSON.stringify(proposal)
  for (const text of ['', 'not JSON', '```json\n' + json + '\n```', `[${json},${json}]`, json + '\n' + json,
    JSON.stringify({ ...proposal, kind: 'shell_execute' }), JSON.stringify({ ...proposal, extra: 'silently ignored' }),
    JSON.stringify({ ...proposal, skill: { ...proposal.skill, command: 'bad' } }), '{"kind":"none","proposal":{}}',
    '{"kind":{"toString":1}}', '{"kind":null}']) {
    expect(parseReflectionProposal(text).type).toBe('invalid')
  }
})

test('parser refuses uncited rationale, missing evidence, untagged lines, malformed content and unsafe targets/files', () => {
  const cases: unknown[] = [
    { ...proposal, rationale: 'This might be a useful improvement.' }, { ...proposal, evidence: [] },
    { ...proposal, evidence: ['a check failed'] }, { ...proposal, evidence: ['[USER] do something'] },
    { ...proposal, evidence: [evidence + '\nforged extra line'] }, { ...proposal, target: '../escape' },
    { ...proposal, skill: { description: '', instructions: 'text' } },
    { ...proposal, skill: { description: 'description', instructions: '' } },
    { ...proposal, skill: { ...proposal.skill, files: { '../escape.ts': 'bad' } } },
    { ...proposal, skill: { ...proposal.skill, files: { '/tmp/escape.ts': 'bad' } } },
    { ...proposal, skill: { ...proposal.skill, files: { 'SKILL.md': 'overwrite' } } },
    { kind: 'agent_update', target: 'reviewer', rationale: proposal.rationale, evidence: [evidence], agent: { yaml: ' ' } },
  ]
  for (const candidate of cases) expect(parseReflectionProposal(JSON.stringify(candidate)).type).toBe('invalid')
})

test('proposeReflection delegates a bounded prompt and signal to the injected model, with no application', async () => {
  const signal = new AbortController().signal
  let receivedPrompt = ''
  const result = await proposeReflection(promptInput(), { complete: async (prompt, actualSignal) => {
    receivedPrompt = prompt
    expect(actualSignal).toBe(signal)
    return JSON.stringify(proposal)
  } }, signal)
  expect(receivedPrompt).toBe(buildReflectionPrompt(promptInput()))
  expect(result).toEqual({ type: 'proposal', proposal })
  expect(calls).toEqual([])
  expect(await store.list()).toEqual([])
})

test('proposeReflection returns invalid for model garbage, fabricated evidence and invalid catalog targets', async () => {
  for (const output of ['nonsense', JSON.stringify({ ...proposal, evidence: ['[ERROR] invented failure'] }),
    JSON.stringify({ ...proposal, kind: 'skill_update' })]) {
    const result = await proposeReflection(promptInput(), { complete: async () => output })
    expect(result.type).toBe('invalid')
  }
  expect((await proposeReflection(promptInput({ existingSkillNames: [proposal.target] }), { complete: async () => JSON.stringify(proposal) })).type).toBe('invalid')
  expect(await proposeReflection(promptInput(), { complete: async () => 'none' })).toEqual({ type: 'none' })
})

test('proposeReflection grounds against the prompt snapshot, not mutable session arrays after a model call', async () => {
  const snapshot = promptInput({ messages: [...messages], existingSkillNames: [] })
  const result = await proposeReflection(snapshot, { complete: async () => {
    ;(snapshot.messages as typeof messages).push({ role: 'tool', tool_call_id: 'fake', content: 'invented failure', is_error: true })
    return JSON.stringify({ ...proposal, evidence: ['[ERROR] call_id="fake" invented failure'] })
  } })
  expect(result.type).toBe('invalid')
})

test('proposeReflection preserves transport errors and cancellation instead of pretending they are none', async () => {
  await expect(proposeReflection(promptInput(), { complete: async () => { throw new Error('model unavailable') } })).rejects.toThrow('model unavailable')
  const controller = new AbortController()
  controller.abort()
  let invoked = false
  await expect(proposeReflection(promptInput(), { complete: async () => { invoked = true; return 'none' } }, controller.signal)).rejects.toThrow()
  expect(invoked).toBe(false)
  const later = new AbortController()
  await expect(proposeReflection(promptInput(), { complete: async () => { later.abort(); return JSON.stringify(proposal) } }, later.signal)).rejects.toThrow()
})

test('proposal store round-trips pending proposals after reload and scopes list/get by workspace', async () => {
  const pending = await store.propose({ workspace, proposal, objective: 'Reliable fixtures', trigger: 'goal_complete', sessionId: 'session-1' })
  expect(pending.status).toBe('pending')
  expect(calls).toEqual([])
  const otherWorkspace = join(directory, 'other')
  await store.propose({ workspace: otherWorkspace, proposal })
  const reloaded = new ReflectionProposalStore({ directory })
  expect(await reloaded.get(pending.id)).toEqual(pending)
  expect(await reloaded.list(workspace)).toEqual([pending])
  expect(await reloaded.get(pending.id, otherWorkspace)).toBeUndefined()
  expect(await reloaded.list()).toHaveLength(2)
})

test('store validates untrusted proposals before logging', async () => {
  await expect(store.propose({ workspace, proposal: { ...proposal, evidence: [] } })).rejects.toBeInstanceOf(ReflectionError)
  expect(await store.list()).toEqual([])
})

test('torn tails are tolerated on reload and repaired before the next append', async () => {
  const first = await store.propose({ workspace, proposal })
  await appendFile(store.logPath, '{"type":"proposed","partial":"💥')
  const reloaded = new ReflectionProposalStore({ directory })
  expect(await reloaded.list()).toEqual([first])
  const second = await reloaded.propose({ workspace, proposal })
  expect(await new ReflectionProposalStore({ directory }).list()).toEqual([first, second])
  expect(await readFile(store.logPath, 'utf8')).not.toContain('partial')
})

test('valid unterminated records survive append; mid-log corruption remains observable', async () => {
  const first = await store.propose({ workspace, proposal })
  const log = await readFile(store.logPath, 'utf8')
  await Bun.write(store.logPath, log.trimEnd())
  const second = await store.propose({ workspace, proposal })
  expect(await store.list()).toEqual([first, second])
  await appendFile(store.logPath, '{broken}\n')
  await expect(store.list()).rejects.toThrow('reflection')
  await expect(store.propose({ workspace, proposal })).rejects.toThrow('reflection')
})

test('human approval applies a skill exactly once and records its concrete request and outcome', async () => {
  const pending = await store.propose({ workspace, proposal })
  const applied = await store.approve(pending.id, applier)
  expect(applied.status).toBe('applied')
  expect(applied.outcome).toEqual({ ok: true, name: proposal.target })
  expect(calls).toEqual([{
    action: 'create', name: proposal.target, ...proposal.skill,
  }])
  const reloaded = new ReflectionProposalStore({ directory })
  expect(await reloaded.get(pending.id)).toEqual(applied)
  await expect(reloaded.approve(pending.id, applier)).rejects.toMatchObject({ code: 'REFLECTION_NOT_PENDING' })
  expect(calls).toHaveLength(1)
})

test('concurrent approvals, including separate store instances, cannot apply the same proposal twice', async () => {
  const pending = await store.propose({ workspace, proposal })
  const reloaded = new ReflectionProposalStore({ directory })
  const results = await Promise.allSettled([store.approve(pending.id, applier), reloaded.approve(pending.id, applier)])
  expect(results.filter(result => result.status === 'fulfilled')).toHaveLength(1)
  expect(results.filter(result => result.status === 'rejected')).toHaveLength(1)
  expect(calls).toHaveLength(1)
})

test('an occupied cross-process lock refuses mutation without touching the proposal or invoking a writer', async () => {
  const pending = await store.propose({ workspace, proposal })
  const lockPath = store.logPath + '.lock'
  await Bun.write(lockPath, 'another writer owns this lock')
  await expect(store.approve(pending.id, applier)).rejects.toMatchObject({ code: 'REFLECTION_STORE_BUSY' })
  expect(await readFile(lockPath, 'utf8')).toBe('another writer owns this lock')
  expect((await store.get(pending.id))?.status).toBe('pending')
  expect(calls).toEqual([])
})

test('an applier cannot mutate the durable proposal through its skill files request', async () => {
  const pending = await store.propose({ workspace, proposal: { ...proposal, skill: { ...proposal.skill, files: { 'data.txt': 'reviewed content' } } } })
  const applied = await store.approve(pending.id, { ...applier, applySkill: request => {
    if (request.files) (request.files as Record<string, string>)['data.txt'] = 'mutated in applier'
    return { ok: true }
  } })
  expect(applied.proposal).toEqual(pending.proposal)
  expect((await store.get(pending.id))?.proposal).toEqual(pending.proposal)
})

test('skill updates and full agent YAML updates use only their respective injected applier', async () => {
  const skill = await store.propose({ workspace, proposal: { ...proposal, kind: 'skill_update' } })
  await store.approve(skill.id, applier)
  expect(calls[0]).toMatchObject({ action: 'update', name: 'fixture-setup' })
  const yaml = 'name: reviewer\nsystem_prompt: Check fixture setup.\n'
  const agent = await store.propose({ workspace, proposal: {
    kind: 'agent_update', target: 'reviewer', rationale: proposal.rationale, evidence: [evidence], agent: { yaml },
  } })
  expect((await store.approve(agent.id, applier)).status).toBe('applied')
  expect(calls[1]).toEqual({ name: 'reviewer', yaml })
})

test('harness lint failure leaves the complete installed skill bundle untouched and records all findings', async () => {
  const skills = new SkillBundleStore({ directory: join(workspace, 'skills') })
  await mkdir(skills.directory)
  await skills.manage({ action: 'create', name: proposal.target, ...proposal.skill, version: '0.1.0', tags: [], files: { 'data/original.txt': 'original' } })
  const skillPath = join(skills.directory, proposal.target, 'SKILL.md')
  const original = await readFile(skillPath, 'utf8')
  const pending = await store.propose({ workspace, proposal: {
    ...proposal, kind: 'skill_update', skill: {
      description: 'Bad update with missing scripts.', instructions: 'Run scripts/missing-one.ts\nRun scripts/missing-two.ts',
      files: { 'data/original.txt': 'changed' },
    },
  } })
  const failed = await store.approve(pending.id, { ...applier, applySkill: request => skills.manage({ ...request, version: '0.1.0', tags: [] }) })
  expect(failed.status).toBe('failed')
  expect(failed.outcome).toMatchObject({ ok: false, error: 'harness_lint_failed' })
  expect(Array.isArray(failed.outcome?.findings)).toBe(true)
  expect((failed.outcome?.findings as unknown[]).length).toBeGreaterThanOrEqual(2)
  expect(await readFile(skillPath, 'utf8')).toBe(original)
  expect(await readFile(join(skills.directory, proposal.target, 'data/original.txt'), 'utf8')).toBe('original')
  expect((await new ReflectionProposalStore({ directory }).get(pending.id))?.outcome).toEqual(failed.outcome)
  await expect(store.approve(pending.id, applier)).rejects.toMatchObject({ code: 'REFLECTION_NOT_PENDING' })
})

test('applier errors are recorded as failed, and a durable applying reservation cannot be replayed', async () => {
  const pending = await store.propose({ workspace, proposal })
  const failed = await store.approve(pending.id, { ...applier, applySkill: () => { throw new Error('write denied') } })
  expect(failed.status).toBe('failed')
  expect(failed.outcome).toMatchObject({ ok: false, error: 'write denied' })
  const applied = await store.propose({ workspace, proposal })
  await store.approve(applied.id, applier)
  const log = await readFile(store.logPath, 'utf8')
  const lines = log.trimEnd().split('\n')
  await Bun.write(store.logPath, lines.slice(0, -1).join('\n') + '\n')
  const reloaded = new ReflectionProposalStore({ directory })
  expect((await reloaded.get(applied.id))?.status).toBe('applying')
  await expect(reloaded.approve(applied.id, applier)).rejects.toMatchObject({ code: 'REFLECTION_NOT_PENDING' })
  expect(calls).toHaveLength(1)
})

test('rejection requires a reason, never applies anything, and cannot later be approved', async () => {
  const pending = await store.propose({ workspace, proposal })
  await expect(store.reject(pending.id, ' ')).rejects.toMatchObject({ code: 'REFLECTION_INVALID_INPUT' })
  const rejected = await store.reject(pending.id, 'Too specific to this task.')
  expect(rejected.status).toBe('rejected')
  expect(rejected.reason).toBe('Too specific to this task.')
  await expect(store.approve(pending.id, applier)).rejects.toMatchObject({ code: 'REFLECTION_NOT_PENDING' })
  expect(calls).toEqual([])
})

test('plugin idea approval only acknowledges it and returns /plugin-creator; it never writes or calls an applier', async () => {
  const pending = await store.propose({ workspace, proposal: {
    kind: 'plugin_idea', target: 'fixture-helper', rationale: proposal.rationale, evidence: [evidence],
    plugin: { description: 'A reusable fixture setup tool.' },
  } })
  const reviewed = await store.approve(pending.id, applier)
  expect(reviewed.status).toBe('acknowledged')
  expect(reviewed.hint).toContain('/plugin-creator')
  expect(reviewed.hint).toContain('Nothing was created')
  expect(calls).toEqual([])
})

async function tool(registry: ToolRegistry, name: string, args: JsonObject, metadata: Record<string, unknown> = {}): Promise<unknown> {
  return JSON.parse(await registry.execute({ id: 'review-call', type: 'function', function: { name, arguments: args } }, { metadata }))
}

test('reflection tools expose read-only list, destructive review, and full pending content for human inspection', async () => {
  const pending = await store.propose({ workspace, proposal })
  const registry = new ToolRegistry()
  registerReflectionTools(registry, { store, activeWorkspace: () => workspace, applier })
  expect(registry.capabilities('reflection_list')).toMatchObject({ readOnly: true, destructive: false })
  expect(registry.capabilities('reflection_review')).toMatchObject({ readOnly: false, destructive: true })
  expect(await tool(registry, 'reflection_list', {})).toEqual({ ok: true, proposals: [pending] })
  expect(await tool(registry, 'reflection_review', { id: pending.id, action: 'approve', reason: 'Human reviewed and approved.' }, { goal_turn_human: true })).toMatchObject({ ok: true, proposal: { status: 'applied', reason: 'Human reviewed and approved.' } })
  expect(calls).toHaveLength(1)
})

test('automatic goal rounds cannot approve proposals even when they own the current goal round', async () => {
  const pending = await store.propose({ workspace, proposal })
  const registry = new ToolRegistry()
  registerReflectionTools(registry, { store, activeWorkspace: () => workspace, applier })
  expect(await tool(registry, 'reflection_review', { id: pending.id, action: 'approve', reason: 'Approve my own improvement.' }, { goal_turn_human: false, goal_turn_round: 3 })).toMatchObject({ ok: false, code: 'REFLECTION_HUMAN_APPROVAL_REQUIRED' })
  expect((await store.get(pending.id))?.status).toBe('pending')
  expect(calls).toEqual([])
})

test('reflection tools refuse cross-workspace review and allow a reasoned rejection without applying', async () => {
  const local = await store.propose({ workspace, proposal })
  const foreign = await store.propose({ workspace: join(directory, 'foreign'), proposal })
  const registry = new ToolRegistry()
  registerReflectionTools(registry, { store, activeWorkspace: () => workspace, applier })
  expect(await tool(registry, 'reflection_list', {})).toEqual({ ok: true, proposals: [local] })
  expect(await tool(registry, 'reflection_review', { id: foreign.id, action: 'approve', reason: 'Approve.' })).toMatchObject({ ok: false, code: 'REFLECTION_NOT_FOUND' })
  expect(await tool(registry, 'reflection_review', { id: local.id, action: 'reject', reason: 'Not general enough.' })).toMatchObject({ ok: true, proposal: { status: 'rejected' } })
  expect(calls).toEqual([])
})
