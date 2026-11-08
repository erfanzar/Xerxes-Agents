// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, beforeEach, expect, test } from 'bun:test'
import { mkdtemp, readdir, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { ValidationError } from '../src/core/errors.js'
import {
  HYBRID_UNSPLITTABLE_REASON,
  analyzeSplit,
  assignPlacements,
  parseHybridPlansConfig,
  runHybridPlan,
  validateHybridPlan,
  type HybridPlan,
  type HybridStep,
} from '../src/cortex/hybridPlan.js'
import { PlanStore } from '../src/cortex/planStore.js'
import type { ExecutionPlan, PlanStep } from '../src/cortex/planner.js'
import { DataBoundaryError } from '../src/security/boundaryTypes.js'

function step(id: string, action: string, dependencies: string[] = [], inputs = ''): PlanStep {
  return { id, action, description: action, arguments: inputs ? { inputs } : {}, dependencies }
}

function plan(steps: PlanStep[], id = 'plan-test'): ExecutionPlan {
  return { id, objective: 'test objective', complexity: 'low', estimatedMinutes: 0, steps }
}

const sensitivityByName = (ref: string) => (ref.startsWith('private:') ? 'private' as const : 'shareable' as const)

function chain(boundaryMode: 'off' | 'private' | 'hybrid' | 'cloud' = 'hybrid'): HybridPlan {
  return assignPlacements(
    plan([
      step('s1', 'read_file', [], 'private:secret.csv'),
      step('s2', 'summarize', ['s1']),
      step('s3', 'reason', ['s2']),
    ]),
    { boundaryMode, inputSensitivity: sensitivityByName },
  ).plan
}

function mixed(): HybridPlan {
  return assignPlacements(
    plan([
      step('p1', 'read_file', [], 'private:secret.csv'),
      step('p2', 'summarize', ['p1']),
      step('c1', 'research', [], 'public:topic'),
      step('c2', 'reason', ['c1']),
    ]),
    { boundaryMode: 'hybrid', inputSensitivity: sensitivityByName },
  ).plan
}

let dir: string

beforeEach(async () => {
  dir = await mkdtemp(join(tmpdir(), 'xerxes-hybrid-plan-'))
})

afterEach(async () => {
  await rm(dir, { recursive: true, force: true })
})

test('parseHybridPlansConfig: default off, env wins, bad values throw', () => {
  expect(parseHybridPlansConfig({}, {})).toBe(false)
  expect(parseHybridPlansConfig({ hybrid_plans: 'on' }, {})).toBe(true)
  expect(parseHybridPlansConfig({ hybrid_plans: 'on' }, { XERXES_HYBRID_PLANS: 'off' })).toBe(false)
  expect(parseHybridPlansConfig({}, { XERXES_HYBRID_PLANS: '1' })).toBe(true)
  expect(() => parseHybridPlansConfig({ hybrid_plans: 'maybe' }, {})).toThrow(ValidationError)
  expect(() => parseHybridPlansConfig({}, { XERXES_HYBRID_PLANS: 'maybe' })).toThrow('XERXES_HYBRID_PLANS')
})

test('sensitivity propagates through a three-step chain and keeps every step local', () => {
  const hybrid = chain()
  expect(hybrid.version).toBe(1)
  for (const item of hybrid.steps) {
    expect(item.produces.sensitivity).toBe('private')
    expect(item.placement).toBe('local')
  }
  expect(hybrid.steps[2]!.inputs).toEqual([{ ref: 'out:s2', sensitivity: 'private' }])
  expect(validateHybridPlan(hybrid, 'hybrid')).toEqual({ ok: true })
})

test('shareable reasoning is either; tool use stays local; private boundary makes everything local', () => {
  const open = assignPlacements(
    plan([step('a', 'research', [], 'public:x'), step('b', 'reason', ['a']), step('c', 'run_tool', ['b'])]),
    { boundaryMode: 'hybrid', inputSensitivity: sensitivityByName },
  )
  expect(open.plan.steps.map(item => item.placement)).toEqual(['either', 'either', 'local'])

  const locked = assignPlacements(
    plan([step('a', 'research', [], 'public:x'), step('b', 'reason', ['a'])]),
    { boundaryMode: 'private', inputSensitivity: sensitivityByName },
  )
  expect(locked.plan.steps.every(item => item.placement === 'local')).toBe(true)
  expect(locked.notes.join(' ')).toContain('private')
})

test('analyzeSplit: unsplittable when every strong step depends on private data', () => {
  const verdict = analyzeSplit(chain())
  expect(verdict.splittable).toBe(false)
  expect(verdict.cloudSteps).toEqual([])
  expect(verdict.reason).toBe(HYBRID_UNSPLITTABLE_REASON)
  expect(verdict.reason).toContain('Finish locally or explain the limitation')

  const split = analyzeSplit(mixed())
  expect(split.splittable).toBe(true)
  expect(split.cloudSteps).toEqual(['c1', 'c2'])
})

test('validateHybridPlan refuses a cloud step that consumes a private ref', () => {
  const base = chain()
  const tampered: HybridPlan = {
    ...base,
    steps: base.steps.map((item): HybridStep => (item.id === 's2' ? { ...item, placement: 'cloud' } : item)),
  }
  const verdict = validateHybridPlan(tampered, 'hybrid')
  expect(verdict.ok).toBe(false)
  if (!verdict.ok) expect(verdict.errors.join(' ')).toContain('s2')

  const eitherTampered: HybridPlan = {
    ...base,
    steps: base.steps.map((item): HybridStep => (item.id === 's3' ? { ...item, placement: 'either' } : item)),
  }
  expect(validateHybridPlan(eitherTampered, 'hybrid').ok).toBe(false)

  const privateMode = mixed()
  expect(validateHybridPlan(privateMode, 'private').ok).toBe(false)
})

test('validateHybridPlan reuses graph validation (cycle)', () => {
  const base = chain()
  const cyclic: HybridPlan = {
    ...base,
    steps: base.steps.map((item): HybridStep => (item.id === 's1' ? { ...item, dependencies: ['s3'] } : item)),
  }
  expect(validateHybridPlan(cyclic, 'hybrid').ok).toBe(false)
})

test('runHybridPlan never hands a private value to a cloud execution', async () => {
  const calls: { id: string; location: 'local' | 'cloud'; inputs: string[] }[] = []
  const result = await runHybridPlan(
    mixed(),
    {
      externalInputs: { 'private:secret.csv': 'TOP SECRET', 'public:topic': 'open topic' },
      chooseLocation: () => 'cloud',
      async execute(item, location, inputs) {
        calls.push({ id: item.id, location, inputs: [...inputs.keys()] })
        return { output: `out-${item.id}`, accepted: true }
      },
    },
  )
  expect(result.accepted).toBe(true)
  const byId = new Map(calls.map(call => [call.id, call]))
  expect(byId.get('p1')!.location).toBe('local')
  expect(byId.get('p2')!.location).toBe('local')
  expect(byId.get('c1')!.location).toBe('cloud')
  expect(byId.get('c2')!.location).toBe('cloud')
  // Only declared inputs are passed; the cloud steps never see a private ref.
  expect(byId.get('c1')!.inputs).toEqual(['public:topic'])
  expect(byId.get('c2')!.inputs).toEqual(['out:c1'])
  for (const call of calls) {
    if (call.location === 'cloud') expect(call.inputs.some(ref => ref.startsWith('private:') || ref === 'out:p1' || ref === 'out:p2')).toBe(false)
  }
})

test('runHybridPlan refuses a tampered plan before running anything', async () => {
  const base = chain()
  const tampered: HybridPlan = {
    ...base,
    steps: base.steps.map((item): HybridStep => (item.id === 's2' ? { ...item, placement: 'cloud' } : item)),
  }
  let executed = 0
  const run = runHybridPlan(tampered, {
    externalInputs: { 'private:secret.csv': 'TOP SECRET' },
    async execute() {
      executed += 1
      return { output: 'x', accepted: true }
    },
  })
  await expect(run).rejects.toBeInstanceOf(DataBoundaryError)
  await expect(run).rejects.toMatchObject({ code: 'DATA_BOUNDARY_DENIED' })
  expect(executed).toBe(0)
})

test('runHybridPlan skips dependents of a rejected or failed step and runs independent branches', async () => {
  const hybrid = assignPlacements(
    plan([
      step('a', 'research', [], 'public:x'),
      step('b', 'reason', ['a']),
      step('c', 'summarize', ['b']),
      step('d', 'research', [], 'public:y'),
      step('e', 'research', [], 'public:z'),
      step('f', 'reason', ['e']),
    ]),
    { boundaryMode: 'hybrid', inputSensitivity: sensitivityByName },
  ).plan
  const ran: string[] = []
  const result = await runHybridPlan(hybrid, {
    externalInputs: { 'public:x': 'x', 'public:y': 'y', 'public:z': 'z' },
    async execute(item) {
      ran.push(item.id)
      if (item.id === 'b') return { output: 'bad', accepted: false, detail: 'check failed' }
      if (item.id === 'e') throw new Error('boom')
      return { output: item.id, accepted: true }
    },
  })
  const status = Object.fromEntries(result.results.map(item => [item.stepId, item.status]))
  expect(status).toEqual({ a: 'succeeded', b: 'rejected', c: 'skipped', d: 'succeeded', e: 'failed', f: 'skipped' })
  expect(ran.sort()).toEqual(['a', 'b', 'd', 'e'])
  expect(result.accepted).toBe(false)
  expect(result.results.find(item => item.stepId === 'b')!.detail).toBe('check failed')
  expect(result.results.find(item => item.stepId === 'e')!.detail).toBe('boom')
})

test('runHybridPlan fails a step whose input value is missing', async () => {
  const hybrid = assignPlacements(plan([step('a', 'research', [], 'public:x')]), {
    boundaryMode: 'hybrid',
    inputSensitivity: sensitivityByName,
  }).plan
  let executed = 0
  const result = await runHybridPlan(hybrid, {
    async execute() {
      executed += 1
      return { output: 'x', accepted: true }
    },
  })
  expect(executed).toBe(0)
  expect(result.results[0]!.status).toBe('failed')
  expect(result.results[0]!.detail).toContain('public:x')
})

test('runHybridPlan honors abort between layers', async () => {
  const controller = new AbortController()
  const hybrid = assignPlacements(
    plan([step('a', 'research', [], 'public:x'), step('b', 'reason', ['a']), step('c', 'summarize', ['b'])]),
    { boundaryMode: 'hybrid', inputSensitivity: sensitivityByName },
  ).plan
  const ran: string[] = []
  const result = await runHybridPlan(
    hybrid,
    {
      externalInputs: { 'public:x': 'x' },
      async execute(item) {
        ran.push(item.id)
        controller.abort()
        return { output: item.id, accepted: true }
      },
    },
    controller.signal,
  )
  expect(ran).toEqual(['a'])
  expect(result.aborted).toBe(true)
  expect(result.accepted).toBe(false)
  expect(result.results.map(item => item.status)).toEqual(['succeeded', 'skipped', 'skipped'])
})

test('PlanStore round trip with 0600 file and 0700 directory', async () => {
  const store = new PlanStore({ dir: join(dir, 'plans'), now: () => new Date('2026-11-03T10:00:00Z') })
  const hybrid = mixed()
  const saved = await store.save(hybrid, { reviewed: true, name: 'My plan' })
  expect(saved.savedAt).toBe('2026-11-03T10:00:00.000Z')
  const loaded = await store.load(hybrid.id)
  expect(loaded?.plan).toEqual(hybrid)
  expect(loaded?.reviewed).toBe(true)
  expect(loaded?.name).toBe('My plan')
  expect(await store.load('missing-plan')).toBeUndefined()

  const listing = await store.list()
  expect(listing).toEqual([
    { id: 'plan-test', name: 'My plan', objective: 'test objective', reviewed: true, savedAt: '2026-11-03T10:00:00.000Z', steps: 4 },
  ])
  expect((await stat(join(dir, 'plans', 'plan-test.json'))).mode & 0o777).toBe(0o600)
  expect((await stat(join(dir, 'plans'))).mode & 0o777).toBe(0o700)
  expect(await readdir(join(dir, 'plans'))).toEqual(['plan-test.json'])

  expect(await store.remove('plan-test')).toBe(true)
  expect(await store.remove('plan-test')).toBe(false)
  expect(await store.list()).toEqual([])
})

test('PlanStore refuses to save a plan that breaks the boundary rules', async () => {
  const store = new PlanStore({ dir })
  const base = chain()
  const tampered: HybridPlan = {
    ...base,
    steps: base.steps.map((item): HybridStep => (item.id === 's2' ? { ...item, placement: 'cloud' } : item)),
  }
  await expect(store.save(tampered, { reviewed: true })).rejects.toThrow(ValidationError)
  expect(await store.list()).toEqual([])
})

test('prepareRerun needs a reviewed plan and every input', async () => {
  const store = new PlanStore({ dir })
  const hybrid = assignPlacements(
    plan([step('a', 'research', [], 'public:old'), step('b', 'reason', ['a'])], 'plan-rerun'),
    { boundaryMode: 'hybrid', inputSensitivity: sensitivityByName },
  ).plan

  await store.save(hybrid, { reviewed: false })
  const unreviewed = await store.prepareRerun('plan-rerun', { 'public:old': 'public:new' })
  expect(unreviewed.ok).toBe(false)
  if (!unreviewed.ok) expect(unreviewed.code).toBe('not_reviewed')

  await store.save(hybrid, { reviewed: true })
  const missing = await store.prepareRerun('plan-rerun', {})
  expect(missing.ok).toBe(false)
  if (!missing.ok) {
    expect(missing.code).toBe('missing_input')
    expect(missing.refs).toEqual(['public:old'])
  }
  const unknown = await store.prepareRerun('plan-rerun', { 'public:old': 'public:new', 'other': 'x' })
  expect(unknown.ok).toBe(false)
  if (!unknown.ok) expect(unknown.code).toBe('unknown_input')

  const ready = await store.prepareRerun('plan-rerun', { 'public:old': 'public:new' })
  expect(ready.ok).toBe(true)
  if (ready.ok) {
    expect(ready.plan.steps[0]!.inputs).toEqual([{ ref: 'public:new', sensitivity: 'shareable' }])
    expect(ready.rebound).toEqual({ 'public:old': 'public:new' })
    expect(validateHybridPlan(ready.plan, 'hybrid')).toEqual({ ok: true })
  }

  const absent = await store.prepareRerun('no-such-plan', {})
  expect(absent.ok).toBe(false)
  if (!absent.ok) expect(absent.code).toBe('not_found')
})

test('prepareRerun never loosens sensitivity of a private input', async () => {
  const store = new PlanStore({ dir })
  await store.save(chain(), { reviewed: true })
  const ready = await store.prepareRerun('plan-test', { 'private:secret.csv': 'private:other.csv' })
  expect(ready.ok).toBe(true)
  if (ready.ok) {
    expect(ready.plan.steps.every(item => item.placement === 'local' && item.produces.sensitivity === 'private')).toBe(true)
    expect(ready.plan.steps[0]!.inputs).toEqual([{ ref: 'private:other.csv', sensitivity: 'private' }])
  }
})

test('PlanStore refuses path traversal and bad ids', async () => {
  const store = new PlanStore({ dir })
  for (const id of ['../evil', 'a/b', '..', 'UPPER', '', 'x'.repeat(65), 'a.b']) {
    await expect(store.load(id)).rejects.toThrow(ValidationError)
    await expect(store.remove(id)).rejects.toThrow(ValidationError)
    const rerun = await store.prepareRerun(id, {})
    expect(rerun.ok).toBe(false)
    if (!rerun.ok) expect(rerun.code).toBe('invalid')
  }
  const badId: HybridPlan = { ...chain(), id: '../escape' }
  await expect(store.save(badId, { reviewed: true })).rejects.toThrow(ValidationError)
  expect(await readdir(dir)).toEqual([])
})

test('PlanStore refuses malformed, mismatched and unknown-version files', async () => {
  const store = new PlanStore({ dir })
  await store.save(chain(), { reviewed: true })
  const good = JSON.parse(await Bun.file(join(dir, 'plan-test.json')).text()) as Record<string, unknown>

  await writeFile(join(dir, 'broken.json'), '{ not json')
  await writeFile(join(dir, 'future.json'), JSON.stringify({ ...good, id: 'future', version: 2 }))
  await writeFile(join(dir, 'wrong-name.json'), JSON.stringify(good))
  await writeFile(join(dir, 'no-steps.json'), JSON.stringify({ ...good, id: 'no-steps', plan: { ...(good.plan as object), steps: [] } }))

  await expect(store.load('broken')).rejects.toThrow(ValidationError)
  await expect(store.load('future')).rejects.toThrow('version')
  await expect(store.load('wrong-name')).rejects.toThrow(ValidationError)
  await expect(store.load('no-steps')).rejects.toThrow(ValidationError)

  const rerun = await store.prepareRerun('broken', {})
  expect(rerun.ok).toBe(false)
  if (!rerun.ok) expect(rerun.code).toBe('invalid')

  expect((await store.list()).map(item => item.id)).toEqual(['plan-test'])
  expect((await store.listInvalid()).map(item => item.id).sort()).toEqual(['broken', 'future', 'no-steps', 'wrong-name'])
})
