// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, mkdir, readFile, rm, symlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import {
  formatImprovementPlan,
  formatImprovementReport,
  planImprovement,
  runImprovement,
  type CandidateWorkspacePort,
  type HarnessCandidate,
  type HarnessEdit,
  type ImproveBudget,
} from '../playground/improve.js'
import { runImproveCli } from '../playground/improveCli.js'
import type { NativeHardTask } from '../playground/hard.js'
import type { EvaluationAgent, EvaluationSessionPortFactory, EvaluationTurnResult } from '../playground/types.js'

const baseline: HarnessCandidate = { id: 'baseline', edits: [] }
const budget: ImproveBudget = { maxCandidates: 10, maxTaskRuns: 30 }
const tasks = ['stable', 'inv', 'dom'].map(tinyTask)
const inv = edit('inv', 'INV')
const dom = edit('dom', 'DOM')

test('planning declares all runs, including a conditional composition slot, before any turn', async () => {
  const fixture = await fixtureFor()
  try {
    expect(() => planImprovement({ baseline, proposals: [inv, dom], tasks, budget: { maxCandidates: 2, maxTaskRuns: 30 } })).toThrow('3 candidates')
    expect(() => planImprovement({ baseline, proposals: [inv, dom], tasks, budget: { maxCandidates: 3, maxTaskRuns: 11 } })).toThrow('12 task runs')
    expect(fixture.turns).toHaveLength(0)
    const plan = planImprovement({ baseline, proposals: [inv, dom], tasks, budget })
    expect(plan.candidateCount).toBe(3)
    expect(plan.taskRuns).toBe(12)
    expect(plan.taskNames).toEqual(['stable', 'inv', 'dom'])
    expect(formatImprovementPlan(plan)).toContain('(3 candidates + 1 baseline) × 3 tasks = 12 task runs')
  } finally { await fixture.cleanup() }
})

test('invalid budgets, duplicate names/ids and unsafe edit paths are refused', () => {
  for (const value of [0, -1, 1.5, Infinity, NaN]) {
    expect(() => planImprovement({ baseline, proposals: [], tasks, budget: { ...budget, maxTaskRuns: value } })).toThrow('positive integer')
  }
  expect(() => planImprovement({ baseline, proposals: [], tasks, budget: { ...budget, maxTokens: 0 } })).toThrow('maxTokens')
  expect(() => planImprovement({ baseline, proposals: [], tasks, budget: { ...budget, maxWallMs: -1 } })).toThrow('maxWallMs')
  expect(() => planImprovement({ baseline, proposals: [inv, inv], tasks, budget })).toThrow('duplicate')
  expect(() => planImprovement({ baseline, proposals: [], tasks: [tasks[0]!, tasks[0]!], budget })).toThrow('duplicate task')
  expect(() => planImprovement({ baseline, proposals: [{ ...inv, apply: { path: '../escape.md', content: 'bad' } }], tasks, budget })).toThrow('inside')
})

test('baseline runs first, scores the same tasks, and composes improving INV before DOM edits', async () => {
  const fixture = await fixtureFor()
  try {
    const plan = planImprovement({ baseline, proposals: [dom, inv], tasks, budget })
    const result = await runImprovement(plan, { ...fixture.ports, tasks })
    expect(result.status).toBe('completed')
    expect(fixture.created.map(item => item.id)).toEqual(['baseline', 'dom', 'inv', 'composed'])
    expect(fixture.created.at(-1)?.edits.map(item => item.scope)).toEqual(['INV', 'DOM'])
    expect(fixture.turns.map(item => item.split(':')[0])).toEqual([
      'baseline', 'baseline', 'baseline', 'dom', 'dom', 'dom',
      'inv', 'inv', 'inv', 'composed', 'composed', 'composed',
    ])
    expect(result.candidates.map(item => item.passed)).toEqual([1, 2, 2, 3])
    expect(result.recommendation.candidate.id).toBe('composed')
    expect(result.recommendation.candidate.edits.map(item => item.id)).toEqual(['inv', 'dom'])
    expect(result.budgetUsed.taskRuns).toBe(12)
    expect(fixture.disposed).toEqual(['baseline', 'dom', 'inv', 'composed'])
    expect(new Set(fixture.directories).size).toBe(4)
    const report = formatImprovementReport(result)
    expect(report).toContain('| Candidate | stable | inv | dom | Passes | Regressions |')
    expect(report).toContain('| composed | PASS | PASS | PASS | 3/3 | none |')
    expect(report).toContain('Task runs: 12/30')
    expect(report).toContain('Recommendation: composed')
    expect(report).toContain('No edits were applied to the source repository')
  } finally { await fixture.cleanup() }
})

test('a higher score with a regression and a tie both keep baseline; incomplete composition is skipped', async () => {
  const fixture = await fixtureFor((ids, task) => ids.includes('regress') ? task !== 'stable' : task === 'stable')
  try {
    const result = await runImprovement(planImprovement({ baseline, proposals: [edit('regress'), edit('tie')], tasks, budget }), { ...fixture.ports, tasks })
    expect(result.candidates[1]).toMatchObject({ passed: 2, improves: false, regressions: ['stable'] })
    expect(result.candidates[2]).toMatchObject({ passed: 1, improves: false })
    expect(result.recommendation.candidate.id).toBe('baseline')
    expect(result.recommendation.reason).toContain('no candidate beat baseline')
    expect(fixture.created).toHaveLength(3)
    expect(result.budgetUsed.taskRuns).toBe(9)
    expect(formatImprovementReport(result)).toContain('stable')
  } finally { await fixture.cleanup() }
})

test('a regressing composition cannot replace an individually improving candidate', async () => {
  const fixture = await fixtureFor((ids, task) => ids.length === 2 ? task !== 'stable' : task === 'stable' || ids.includes(task))
  try {
    const result = await runImprovement(planImprovement({ baseline, proposals: [inv, dom], tasks, budget }), { ...fixture.ports, tasks })
    expect(result.candidates.at(-1)?.regressions).toEqual(['stable'])
    expect(result.recommendation.candidate.id).toBe('inv')
  } finally { await fixture.cleanup() }
})

test('token budget exhaustion retains completed baseline rows and disposes the workspace', async () => {
  const fixture = await fixtureFor()
  try {
    const result = await runImprovement(planImprovement({ baseline, proposals: [inv], tasks, budget: { ...budget, maxTokens: 2 } }), { ...fixture.ports, tasks })
    expect(result.status).toBe('budget_exhausted')
    expect(result.budgetUsed.taskRuns).toBe(2)
    expect(result.budgetUsed.tokens).toBe(2)
    expect(result.candidates[0]?.report.rows).toHaveLength(2)
    expect(result.candidates[0]?.complete).toBeFalse()
    expect(result.recommendation.candidate.id).toBe('baseline')
    expect(fixture.disposed).toEqual(['baseline'])
  } finally { await fixture.cleanup() }
})

test('abort retains completed rows and never starts the next task or candidate', async () => {
  const fixture = await fixtureFor()
  const controller = new AbortController()
  try {
    const result = await runImprovement(planImprovement({ baseline, proposals: [inv], tasks, budget }), {
      ...fixture.ports, tasks, signal: controller.signal,
      onProgress: progress => { if (progress.type === 'task_completed') controller.abort() },
    })
    expect(result.status).toBe('cancelled')
    expect(result.candidates[0]?.report.rows).toHaveLength(1)
    expect(fixture.turns).toHaveLength(1)
    expect(fixture.disposed).toEqual(['baseline'])
    expect(fixture.closed).toEqual(['baseline'])
  } finally { await fixture.cleanup() }
})

test('pre-aborted runs allocate nothing, and task changes cannot bypass the declared plan', async () => {
  const fixture = await fixtureFor()
  try {
    const controller = new AbortController()
    controller.abort()
    const plan = planImprovement({ baseline, proposals: [inv], tasks, budget })
    const result = await runImprovement(plan, { ...fixture.ports, tasks, signal: controller.signal })
    expect(result.status).toBe('cancelled')
    expect(fixture.created).toHaveLength(0)
    await expect(runImprovement(plan, { ...fixture.ports, tasks: [tinyTask('other')] })).rejects.toThrow('declared tasks')
    expect(fixture.turns).toHaveLength(0)
  } finally { await fixture.cleanup() }
})

test('wall deadlines interrupt an in-flight turn and still close and dispose', async () => {
  const fixture = await fixtureFor()
  try {
    const result = await runImprovement(planImprovement({ baseline, proposals: [], tasks, budget: { ...budget, maxWallMs: 20 } }), {
      ...fixture.ports, tasks,
      agentFor: directory => ({
        ...fixture.ports.agentFor(directory),
        turn: async () => new Promise<EvaluationTurnResult>(() => {}),
      }),
    })
    expect(result.status).toBe('budget_exhausted')
    expect(result.candidates[0]?.report.rows).toHaveLength(0)
    expect(fixture.closed).toEqual(['baseline'])
    expect(fixture.disposed).toEqual(['baseline'])
  } finally { await fixture.cleanup() }
})

test('agent errors close the agent and always dispose the workspace', async () => {
  const fixture = await fixtureFor()
  try {
    await expect(runImprovement(planImprovement({ baseline, proposals: [inv], tasks, budget }), {
      ...fixture.ports, tasks,
      agentFor: directory => ({ ...fixture.ports.agentFor(directory), start: async () => { throw new Error('transport failed') } }),
    })).rejects.toThrow('transport failed')
    expect(fixture.closed).toEqual(['baseline'])
    expect(fixture.disposed).toEqual(['baseline'])
  } finally { await fixture.cleanup() }
})

test('cancelling at candidate announcement never constructs an agent and still disposes', async () => {
  const fixture = await fixtureFor()
  const controller = new AbortController()
  let constructions = 0
  try {
    const result = await runImprovement(planImprovement({ baseline, proposals: [inv], tasks, budget }), {
      ...fixture.ports, tasks, signal: controller.signal,
      agentFor: directory => { constructions += 1; return fixture.ports.agentFor(directory) },
      onProgress: progress => { if (progress.type === 'candidate_started') controller.abort() },
    })
    expect(result.status).toBe('cancelled')
    expect(constructions).toBe(0)
    expect(fixture.disposed).toEqual(['baseline'])
  } finally { await fixture.cleanup() }
})

test('multi-step tasks count once and abort in a grader preserves only prior completed rows', async () => {
  const fixture = await fixtureFor()
  const controller = new AbortController()
  const selected = [
    { ...tinyTask('stable'), steps: [{ freshSession: true, prompt: 'Task stable' }, { freshSession: false, prompt: 'Task stable' }] },
    { ...tinyTask('inv'), check: () => { controller.abort(); return { ok: true, detail: 'cancelled during grader' } } },
  ]
  try {
    const result = await runImprovement(planImprovement({ baseline, proposals: [inv], tasks: selected, budget }), { ...fixture.ports, tasks: selected, signal: controller.signal })
    expect(result.status).toBe('cancelled')
    expect(result.budgetUsed.taskRuns).toBe(2)
    expect(fixture.turns).toHaveLength(3)
    expect(result.candidates[0]?.report.rows.map(row => row.name)).toEqual(['stable'])
    expect(fixture.disposed).toEqual(['baseline'])
  } finally { await fixture.cleanup() }
})

test('an injected proposer runs after baseline within declared slots and receives baseline evidence', async () => {
  const fixture = await fixtureFor()
  try {
    const result = await runImprovement(planImprovement({ baseline, proposals: [edit('placeholder')], tasks, budget }), {
      ...fixture.ports, tasks,
      proposer: { propose: async context => {
        expect(fixture.turns).toHaveLength(3)
        expect(context.baselineReport.rows).toHaveLength(3)
        expect(context.taskNames).toEqual(['stable', 'inv', 'dom'])
        return [inv]
      } },
    })
    expect(result.recommendation.candidate.id).toBe('inv')
    expect(fixture.disposed).toEqual(['baseline', 'inv'])
  } finally { await fixture.cleanup() }
})

test('a proposer cannot expand the approved slots and returns partial baseline evidence', async () => {
  const fixture = await fixtureFor()
  try {
    const result = await runImprovement(planImprovement({ baseline, proposals: [inv], tasks, budget }), {
      ...fixture.ports, tasks, proposer: { propose: async () => [inv, dom] },
    })
    expect(result.status).toBe('budget_exhausted')
    expect(result.stopReason).toContain('2 edits for 1 declared')
    expect(fixture.turns).toHaveLength(3)
    expect(result.candidates[0]?.complete).toBeTrue()
    expect(fixture.disposed).toEqual(['baseline'])
  } finally { await fixture.cleanup() }
})

test('CLI dry-run declares its budget without loading a transport or creating isolation', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-improve-cli-'))
  const output: string[] = []
  let loads = 0
  try {
    await writeFile(join(root, 'proposals.json'), JSON.stringify([{ id: 'general', edits: [inv] }, { id: 'domain', edits: [dom] }]))
    const code = await runImproveCli(['--proposals', 'proposals.json', '--max-candidates', '3', '--max-task-runs', '12', '--transport', 'does-not-exist.ts', '--dry-run'], {
      cwd: root, tasks, write: text => { output.push(text) },
      createSessionPort: () => { loads += 1; throw new Error('must not load') },
      createIsolation: async () => { throw new Error('must not allocate') },
    })
    expect(code).toBe(0)
    expect(loads).toBe(0)
    expect(output.join('')).toContain('(3 candidates + 1 baseline) × 3 tasks = 12 task runs')
    expect(output.join('')).toContain('composition slot')
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('CLI requires explicit transport, validates JSON and budgets before a transport call, and supports keywords', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-improve-cli-'))
  const errors: string[] = []
  let loads = 0
  const args = ['--proposals', 'proposals.json', '--max-candidates', '1', '--max-task-runs', '6']
  try {
    await writeFile(join(root, 'proposals.json'), JSON.stringify([inv]))
    expect(await runImproveCli(args, { cwd: root, tasks, writeError: text => { errors.push(text) } })).toBe(2)
    expect(errors.join('')).toContain('--transport <module> is required')
    const dependency = { cwd: root, tasks, write: () => {}, writeError: (text: string) => { errors.push(text) }, createSessionPort: () => { loads += 1; throw new Error('must not load') } }
    expect(await runImproveCli([...args, '--max-task-runs', '1'], dependency)).not.toBe(0)
    expect(loads).toBe(0)
    expect(await runImproveCli([...args, '--keyword', 'inv', '--max-task-runs', '2', '--dry-run'], dependency)).toBe(0)
    await writeFile(join(root, 'proposals.json'), JSON.stringify([{ ...inv, scope: 'unknown' }]))
    expect(await runImproveCli([...args, '--dry-run'], dependency)).not.toBe(0)
    expect(loads).toBe(0)
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('CLI copies baseline files into private candidates, applies replacements, and leaves source untouched', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-improve-cli-'))
  const output: string[] = []
  const starts: string[] = []
  const privateHomes: string[] = []
  const createSessionPort: EvaluationSessionPortFactory = context => ({
    approve: async () => {}, close: async () => {}, reset: async () => {},
    start: async request => { starts.push(request.workspaceDirectory); privateHomes.push(request.homeDirectory); return { model: 'offline' } },
    submit: async function* () {
      const content = await readFile(join(context.workspaceDirectory, 'inv.md'), 'utf8')
      yield { type: 'text', text: content.trim() }
      yield { type: 'turn_end' }
    },
  })
  try {
    await writeFile(join(root, 'inv.md'), 'original')
    await writeFile(join(root, 'proposals.json'), JSON.stringify([inv]))
    const code = await runImproveCli(['--proposals', 'proposals.json', '--max-candidates', '1', '--max-task-runs', '2'], {
      cwd: root, createSessionPort, tasks: [tinyTask('inv')], write: text => { output.push(text) }, writeError: text => { output.push(text) },
    })
    expect(code).toBe(0)
    expect(output.join('')).toContain('Recommendation: inv')
    expect(await readFile(join(root, 'inv.md'), 'utf8')).toBe('original')
    expect(starts).toHaveLength(2)
    expect(new Set(starts).size).toBe(2)
    expect(new Set(privateHomes).size).toBe(2)
    for (const directory of starts) expect(await Bun.file(join(directory, 'inv.md')).exists()).toBeFalse()
  } finally { await rm(root, { recursive: true, force: true }) }
})

test('CLI rejects symlink source escapes before creating any transport', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-improve-cli-'))
  const errors: string[] = []
  let loads = 0
  try {
    await mkdir(join(root, 'source'))
    await writeFile(join(root, 'outside.md'), 'private')
    await symlink(join(root, 'outside.md'), join(root, 'source', 'inv.md'))
    await writeFile(join(root, 'source', 'proposals.json'), JSON.stringify([inv]))
    const code = await runImproveCli(['--proposals', 'proposals.json', '--max-candidates', '1', '--max-task-runs', '2'], {
      cwd: join(root, 'source'), tasks: [tinyTask('inv')], write: () => {}, writeError: text => { errors.push(text) },
      createSessionPort: () => { loads += 1; throw new Error('must not load') },
    })
    expect(code).not.toBe(0)
    expect(errors.join('')).toContain('symlink')
    expect(loads).toBe(0)
  } finally { await rm(root, { recursive: true, force: true }) }
})

function edit(id: string, scope: HarnessEdit['scope'] = 'INV'): HarnessEdit {
  return { id, scope, target: 'prompt_addendum', description: `Improve ${id}`, apply: { path: `${id}.md`, content: id } }
}

function tinyTask(name: string): NativeHardTask {
  return {
    name, category: 'tiny', difficulty: 1, files: [], whyHard: 'deterministic',
    steps: [{ prompt: `Task {dir}`, freshSession: true }],
    check: context => ({ ok: context.results.at(-1)?.text === name, detail: `expected ${name}` }),
  }
}

async function fixtureFor(pass = (ids: readonly string[], task: string) => task === 'stable' || ids.includes(task)) {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-improve-'))
  const created: { id: string; edits: readonly HarnessEdit[] }[] = []
  const disposed: string[] = []
  const closed: string[] = []
  const turns: string[] = []
  const directories: string[] = []
  const workspaces: CandidateWorkspacePort = {
    create: async (id, edits) => {
      const directory = join(root, id)
      await mkdir(directory)
      for (const item of edits) await writeFile(join(directory, item.apply.path), item.apply.content)
      created.push({ id, edits })
      directories.push(directory)
      return { directory, dispose: async () => { disposed.push(id); await rm(directory, { recursive: true, force: true }) } }
    },
  }
  const agentFor = (directory: string): EvaluationAgent => {
    const id = directory.split('/').at(-1)!
    return {
      model: 'offline', start: async () => {}, freshSession: async () => {}, close: async () => { closed.push(id) },
      turn: async prompt => {
        const task = prompt.split('/').at(-1)!
        const edits = created.find(item => item.id === id)!.edits
        const ids: string[] = []
        for (const item of edits) ids.push(await readFile(join(directory, item.apply.path), 'utf8'))
        turns.push(`${id}:${task}`)
        return { text: pass(ids, task) ? task : 'fail', contextTokens: 1, latencyMs: 1, error: undefined, retries: 0, tools: [] }
      },
    }
  }
  return { root, created, disposed, closed, turns, directories, ports: { workspaces, agentFor }, cleanup: () => rm(root, { force: true, recursive: true }) }
}
