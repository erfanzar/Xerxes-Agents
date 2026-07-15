// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  SkillAuthoringConfig,
  SkillAuthoringPipeline,
  SkillAuthoringTrigger,
  SkillCandidate,
  SkillLifecycleManager,
  SkillMatcher,
  SkillProposalDrafter,
  SkillTelemetry,
  SkillVariant,
  SkillVariantPicker,
  SkillVerifier,
  ToolSequenceTracker,
  type SkillCatalogEntry,
} from '../src/extensions/skillAuthoring/index.js'

function skill(name: string, description: string, tags: readonly string[]): SkillCatalogEntry {
  return {
    instructions: description + ' instructions',
    metadata: {
      name,
      description,
      tags,
      requiredTools: [],
      version: '1.0.0',
    },
  }
}

function candidate(): SkillCandidate {
  const tracker = new ToolSequenceTracker()
  tracker.beginTurn({ agentId: 'coder', turnId: 'turn-1', userPrompt: 'set up continuous integration' })
  tracker.recordCall({ toolName: 'Read', arguments: { path: 'ci.yml' } })
  tracker.recordCall({ toolName: 'Edit', arguments: { path: 'ci.yml', apiKey: 'must-not-appear' } })
  tracker.recordCall({ toolName: 'Bash', arguments: { cmd: 'bun test' } })
  return tracker.endTurn('CI configured')
}

test('observation tracking detects stable retries and trigger decisions preserve recovered failures', () => {
  let monotonic = 100
  const tracker = new ToolSequenceTracker({
    now: () => 1_000,
    monotonicNow: () => {
      monotonic += 8
      return monotonic
    },
  })
  tracker.beginTurn({ agentId: 'coder', turnId: 'turn-1', userPrompt: 'repair build' })
  tracker.markCallStart()
  tracker.recordCall({ toolName: 'Bash', arguments: { b: 2, a: 1 }, status: 'failure', errorType: 'ExitError' })
  const retry = tracker.recordCall({ toolName: 'Bash', arguments: { a: 1, b: 2 } })
  const observed = tracker.endTurn('fixed')

  expect(retry.retryOf).toBe(0)
  expect(observed).toMatchObject({
    agentId: 'coder',
    turnId: 'turn-1',
    userPrompt: 'repair build',
    finalResponse: 'fixed',
  })
  expect(observed.retries).toBe(1)
  expect(observed.totalDurationMs).toBe(8)
  expect(observed.signature()).toBe('Bash>Bash')

  const permissive = new SkillAuthoringTrigger({
    config: { minToolCalls: 2, minUniqueTools: 1, maxRetriesAllowed: 1, skipIfSkillSignatureExists: false },
  })
  expect(permissive.evaluate(observed)).toMatchObject({ eligible: true, reason: 'skill-worthy' })

  const terminal = new SkillCandidate({
    events: [
      { toolName: 'Read', arguments: {}, status: 'success', durationMs: 0, timestamp: 1 },
      { toolName: 'Bash', arguments: {}, status: 'failure', durationMs: 0, timestamp: 2 },
    ],
  })
  const terminalTrigger = new SkillAuthoringTrigger({
    config: { minToolCalls: 2, minUniqueTools: 1, skipIfSkillSignatureExists: false },
  })
  expect(terminalTrigger.evaluate(terminal)).toMatchObject({
    eligible: false,
    reason: 'candidate has unrecovered failures',
    terminalFailureIndexes: [1],
  })

  const duplicate = new SkillAuthoringTrigger({
    config: new SkillAuthoringConfig({ minToolCalls: 2, minUniqueTools: 1 }),
    catalog: { all: () => [skill('build-repair', 'repair builds', ['Bash'])] },
  })
  expect(duplicate.reason(observed)).toBe('an existing skill already covers this tool combination')
})

test('semantic matcher only uses its injected embedding port and caches skill vectors', async () => {
  let calls = 0
  const catalog = {
    all: () => [
      skill('ci', 'set up continuous integration', ['ci']),
      skill('docs', 'write documentation', ['docs']),
    ],
  }
  const matcher = new SkillMatcher({
    embed: text => {
      calls += 1
      return text.includes('continuous') || text.includes('ci') ? [1, 0] : [0, 1]
    },
  }, { catalog, minScore: 0 })

  expect((await matcher.match('continuous integration', { limit: 1 }))[0]?.skill.metadata.name).toBe('ci')
  expect((await matcher.best('continuous integration'))?.skill.metadata.name).toBe('ci')
  expect(calls).toBe(4)
  matcher.invalidate()
  await matcher.match('continuous integration')
  expect(calls).toBe(7)
})

test(
  'proposal drafting and verification are deterministic, redact secrets, and accept only explicit refiners',
  async () => {
  const observed = candidate()
  const verifier = new SkillVerifier()
  const recipe = verifier.generate(observed)
  expect(verifier.verify(recipe, observed)).toEqual({
    passed: true,
    passedSteps: recipe.map((_, index) => index),
    failedSteps: [],
  })

  const draft = new SkillProposalDrafter().create(observed)
  expect(draft.markdown).toContain('# When to use')
  expect(draft.markdown).toContain('# Procedure')
  expect(draft.markdown).toContain('# Verification')
  expect(draft.markdown).not.toContain('must-not-appear')
  expect(draft.refinement).toBe('none')

  let refinementCalls = 0
  const rejected = new SkillProposalDrafter({
    refiner: {
      refine: () => {
        refinementCalls += 1
        return 'not a valid skill document'
      },
    },
  })
  expect((await rejected.refine(draft)).refinement).toBe('rejected')
  expect(refinementCalls).toBe(1)

  const mismatch = new SkillCandidate({
    events: [
      { toolName: 'Read', arguments: { path: 'ci.yml' }, status: 'success', durationMs: 0, timestamp: 1 },
      { toolName: 'Bash', arguments: { cmd: 'bun test' }, status: 'success', durationMs: 0, timestamp: 2 },
    ],
  })
  expect(verifier.verify(recipe, mismatch).passed).toBeFalse()
  },
)

test('pipeline exposes a proposal without a store and reports authored only after explicit persistence', async () => {
  const proposed = new SkillAuthoringPipeline({
    config: { minToolCalls: 3, minUniqueTools: 2, skipIfSkillSignatureExists: false },
  })
  proposed.beginTurn({ userPrompt: 'set up CI' })
  proposed.recordCall({ toolName: 'Read', arguments: {} })
  proposed.recordCall({ toolName: 'Edit', arguments: {} })
  proposed.recordCall({ toolName: 'Bash', arguments: {} })
  const proposalResult = await proposed.onTurnEnd()
  expect(proposalResult).toMatchObject({ status: 'proposed', authored: false, reason: '' })
  expect(proposalResult.proposal?.markdown).toContain('# Procedure')

  const saved: string[] = []
  const telemetry = new SkillTelemetry()
  const persisted = new SkillAuthoringPipeline({
    config: { minToolCalls: 3, minUniqueTools: 2, skipIfSkillSignatureExists: false },
    telemetry,
    proposalStore: {
      persist: async input => {
        saved.push(input.proposal.name)
        return { id: 'proposal-1', location: 'memory://proposal-1' }
      },
    },
  })
  persisted.beginTurn({ userPrompt: 'set up CI' })
  persisted.recordCall({ toolName: 'Read', arguments: {} })
  persisted.recordCall({ toolName: 'Edit', arguments: {} })
  persisted.recordCall({ toolName: 'Bash', arguments: {} })
  const persistedResult = await persisted.endTurn('done')
  expect(persistedResult).toMatchObject({
    status: 'persisted',
    authored: true,
    persistence: { id: 'proposal-1', location: 'memory://proposal-1' },
  })
  expect(saved).toHaveLength(1)
  expect(telemetry.stats(persistedResult.proposal?.name ?? '')?.version).toBe('0.1.0')
})

test('telemetry lifecycle decisions require an explicit retirement port and variants are deterministic', async () => {
  const telemetry = new SkillTelemetry()
  for (let index = 0; index < 2; index += 1) {
    telemetry.recordUsage({ skillName: 'bad-ci', outcome: 'success', durationMs: 10 })
  }
  for (let index = 0; index < 3; index += 1) {
    telemetry.recordUsage({ skillName: 'bad-ci', outcome: 'failure', durationMs: 20 })
  }
  const manager = new SkillLifecycleManager(telemetry, { minInvocations: 5, maxSuccessRate: 0.5 })
  expect(manager.evaluate()).toEqual([{
    skillName: 'bad-ci',
    action: 'proposed',
    reason: 'success_rate=40% after 5 invocations',
  }])
  expect(await manager.apply()).toEqual([{
    skillName: 'bad-ci',
    action: 'unavailable',
    reason: 'success_rate=40% after 5 invocations; no retirement port configured',
  }])

  const retired: string[] = []
  const applicator = new SkillLifecycleManager(telemetry, {
    minInvocations: 5,
    maxSuccessRate: 0.5,
    retirement: {
      deprecate: async input => {
        retired.push(input.skillName)
        return { action: 'deprecated', deprecatedLocation: 'memory://deprecated/bad-ci' }
      },
    },
  })
  expect(await applicator.apply()).toEqual([{
    skillName: 'bad-ci',
    action: 'deprecated',
    reason: 'success_rate=40% after 5 invocations',
    deprecatedLocation: 'memory://deprecated/bad-ci',
  }])
  expect(retired).toEqual(['bad-ci'])

  const variants = new SkillVariantPicker()
  variants.add(new SkillVariant('ci', 'ci-v2', 0.5))
  const first = variants.pick('ci', 'alice')
  expect(variants.pick('ci', 'alice')).toBe(first)
  variants.add(new SkillVariant('full', 'full-v2', 2))
  expect(variants.pick('full', 'someone')).toBe('full-v2')
})
