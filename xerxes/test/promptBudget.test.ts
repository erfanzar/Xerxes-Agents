// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  assembleContextLayers,
  layerDigests,
  readAssemblyProvenance,
  recordAssemblyProvenance,
} from '../src/context/assembly.js'
import {
  DEFAULT_PROMPT_BUDGETS,
  evaluateBudget,
  measureLayers,
} from '../src/context/promptBudget.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import type { DaemonEvent, DaemonSession } from '../src/daemon/runtime.js'
import type { CompletionRequest, LlmClient } from '../src/llms/client.js'
import { joinSystemSegments } from '../src/streaming/promptCaching.js'

function sessionFor(cwd: string, id = 'prompt-budget'): DaemonSession {
  return {
    activeTurnId: '', agentId: 'default', cancelRequested: false, cwd, extra: {}, id,
    interactionMode: 'code', sessionKey: id, lastActive: 0, messages: [], metadata: {},
    model: 'gpt-4o', planMode: false, status: 'working', thinkingContent: [],
    toolExecutions: [], totalInputTokens: 0, totalOutputTokens: 0, turnCount: 0,
    workspace: cwd,
  }
}

async function run(runner: AgentTurnRunner, session: DaemonSession, signal = new AbortController().signal): Promise<DaemonEvent[]> {
  const events: DaemonEvent[] = []
  for await (const event of runner.run(session, 'Continue with the original content.', signal)) events.push(event)
  return events
}

function budgetWarnings(events: readonly DaemonEvent[]): DaemonEvent[] {
  return events.filter(event => event.type === 'notification' && event.payload.prompt_budget !== undefined)
}

describe('always-on prompt budget', () => {
  test('measures exact, untrimmed UTF-16 characters and chars/4 token estimates rounded up, including join separators', () => {
    const measure = measureLayers([
      { name: 'bootstrap', text: '  hi\n' },
      { name: 'memory', text: '😀' },
    ])
    expect(measure.layers).toEqual([
      { name: 'bootstrap', chars: 5, estimatedTokens: 2 },
      { name: 'memory', chars: 2, estimatedTokens: 1 },
    ])
    expect(measure.total).toEqual({ chars: 9, estimatedTokens: 3 })
    expect(measureLayers([])).toEqual({ layers: [], total: { chars: 0, estimatedTokens: 0 } })
  })

  test('defaults report every overage with sizes, token caps and human reasons', () => {
    const names = ['bootstrap', 'tool_guidance', 'memory', 'deferred_catalog']
    const measure = measureLayers(names.map(name => ({
      name, text: 'x'.repeat((DEFAULT_PROMPT_BUDGETS.layers[name]! + 1) * 4),
    })))
    const report = evaluateBudget(measure)
    expect(report.ok).toBe(false)
    expect(report.total).toEqual(measure.total)
    expect(report.layers).toEqual(measure.layers)
    expect(report.overages.map(overage => overage.layer).sort()).toEqual([...names, 'total'].sort())
    for (const overage of report.overages) {
      expect(overage.chars).toBeGreaterThan(overage.budget * 4)
      expect(overage.estimatedTokens).toBe(Math.ceil(overage.chars / 4))
      expect(overage.reason.length).toBeGreaterThan(30)
      expect(overage.reason).toMatch(/[.!]$/)
    }
    expect(report.overages.find(overage => overage.layer === 'total')?.budget).toBe(24_000)
    expect(report.overages.find(overage => overage.layer === 'memory')?.reason).toContain('memory')
  })

  test('options override total and individual layer caps without losing defaults or measurements', () => {
    const measure = measureLayers([
      { name: 'bootstrap', text: 'x'.repeat(16) },
      { name: 'custom_layer', text: 'y'.repeat(8) },
      { name: 'memory', text: 'z'.repeat((DEFAULT_PROMPT_BUDGETS.layers.memory! + 1) * 4) },
    ])
    const report = evaluateBudget(measure, { total: 100_000, layers: { bootstrap: 3, custom_layer: 1 } })
    expect(report.overages.map(overage => overage.layer)).toEqual(['bootstrap', 'custom_layer', 'memory'])
    expect(report.overages[0]).toMatchObject({ chars: 16, estimatedTokens: 4, budget: 3 })
    expect(report.overages[1]?.reason).toContain('custom_layer')
    expect(report.total).toEqual(measure.total)
  })

  test('custom layer names cannot inherit a non-string reason from the object prototype', () => {
    const measure = measureLayers([{ name: 'toString', text: '12345678' }])
    const report = evaluateBudget(measure, { layers: { toString: 1 } })
    expect(report.overages[0]?.reason).toContain('toString')
    expect(typeof report.overages[0]?.reason).toBe('string')
  })

  test('an exact cap is accepted, zero caps work, and invalid budgets fail explicitly', () => {
    const measure = measureLayers([{ name: 'bootstrap', text: '1234' }])
    expect(evaluateBudget(measure, { total: 1, layers: { bootstrap: 1 } })).toMatchObject({ ok: true, overages: [] })
    expect(evaluateBudget(measure, { total: 0, layers: { bootstrap: 0 } }).overages).toHaveLength(2)
    for (const invalid of [-1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => evaluateBudget(measure, { total: invalid })).toThrow(RangeError)
      expect(() => evaluateBudget(measure, { layers: { bootstrap: invalid } })).toThrow(/bootstrap/)
    }
  })

  test('measuring and evaluating never trim, truncate or mutate assembled prompt content', () => {
    const text = `  ${'user content\n'.repeat(20_000)}終わり  `
    const segments = Object.freeze(assembleContextLayers({
      bootstrap: text, agentPrompt: 'persona', toolGuidance: '', modeHint: '', subagentJoin: '',
    }).map(segment => Object.freeze(segment)))
    const before = joinSystemSegments(segments)
    const report = evaluateBudget(measureLayers(segments))
    expect(report.ok).toBe(false)
    expect(joinSystemSegments(segments)).toBe(before)
    expect(segments[0]?.text).toBe(text)
    expect(report.layers[0]?.chars).toBe(text.length)
    expect(report.total.chars).toBe(before.length)
  })

  test('provenance carries the full size report and remains compatible with legacy entries', () => {
    const segments = [{ name: 'bootstrap', text: 'unchanged text' }]
    const budget = evaluateBudget(measureLayers(segments))
    const metadata: Record<string, unknown> = {}
    const legacy = { layers: layerDigests(segments), recordedAt: 1 }
    recordAssemblyProvenance(metadata, legacy)
    recordAssemblyProvenance(metadata, { layers: layerDigests(segments), recordedAt: 2, budget })
    const persisted: Record<string, unknown> = JSON.parse(JSON.stringify(metadata))
    const records = readAssemblyProvenance(persisted)
    expect(records).toHaveLength(2)
    expect(records[0]).toEqual(legacy)
    expect(records[1]?.budget).toEqual(budget)
    expect(records[1]?.budget?.layers[0]).toMatchObject({ chars: 14, estimatedTokens: 4 })
  })

  test('provenance rejects malformed optional size reports rather than trusting persisted data', () => {
    const valid = { layers: [], recordedAt: 1, budget: evaluateBudget(measureLayers([])) }
    const metadata = { context_assembly: [
      valid,
      { ...valid, budget: { ...valid.budget, total: { chars: -1, estimatedTokens: 0 } } },
      { ...valid, budget: { ...valid.budget, layers: [{ name: 'memory', chars: 10, estimatedTokens: 'wrong' }] } },
      { ...valid, budget: { ...valid.budget, overages: [{ layer: 'memory', chars: 10, estimatedTokens: 2.5, budget: 1 }] } },
    ] }
    expect(readAssemblyProvenance(metadata)).toEqual([valid])
  })

  test('runner warns once per session and distinct overage set, even after reload, without changing provider bytes', async () => {
    const root = await mkdtemp(join(tmpdir(), 'xerxes-prompt-budget-'))
    try {
      const requests: CompletionRequest[] = []
      const llm: LlmClient = { async *stream(request) { requests.push(request); yield { content: 'done' } } }
      const bootstrap = 'BEGIN\n' + 'unchanged always-on content\n'.repeat(4_000) + 'END'
      const makeRunner = () => new AgentTurnRunner({ llm, model: 'gpt-4o', bootstrapSystemPrompt: () => bootstrap })
      const session = sessionFor(root)
      const runner = makeRunner()
      const first = budgetWarnings(await run(runner, session))
      expect(first).toHaveLength(1)
      expect(first[0]?.payload.level).toBe('warning')
      expect(String(first[0]?.payload.message)).toContain('bootstrap')
      expect(String(first[0]?.payload.message)).toContain(`${bootstrap.length} chars`)
      expect(String(first[0]?.payload.message)).toContain('tokens')
      expect(budgetWarnings(await run(runner, session))).toEqual([])

      const rebuilt = makeRunner()
      const restored = sessionFor(root, session.id)
      restored.metadata = JSON.parse(JSON.stringify(session.metadata))
      expect(budgetWarnings(await run(rebuilt, restored))).toEqual([])
      expect(budgetWarnings(await run(rebuilt, sessionFor(root, 'different-session')))).toHaveLength(1)
      const systemOf = (request: CompletionRequest | undefined) => request?.messages.filter(message => message.role === 'system').map(message => message.content)
      const originalSegments = session.requestScaffold?.systemSegments ?? []
      expect(systemOf(requests[0])).toEqual([joinSystemSegments(originalSegments)])
      expect(systemOf(requests[1])).toEqual(systemOf(requests[0]))
      expect(systemOf(requests[2])).toEqual(systemOf(requests[0]))
      expect(String(systemOf(requests[0]))).toContain(bootstrap)
      const records = readAssemblyProvenance(session.metadata)
      expect(records).toHaveLength(2)
      expect(records[0]?.budget).toEqual(evaluateBudget(measureLayers(originalSegments)))
      expect(records[1]?.budget).toEqual(records[0]?.budget)

      // A genuinely new set (bootstrap only, not total) warns once. Going back
      // to an already seen set is still quiet; differing sizes alone do not spam.
      const smaller = new AgentTurnRunner({ llm, model: 'gpt-4o', bootstrapSystemPrompt: () => 'x'.repeat(52_000) })
      expect(budgetWarnings(await run(smaller, session))).toHaveLength(1)
      expect(budgetWarnings(await run(smaller, session))).toEqual([])
      expect(budgetWarnings(await run(makeRunner(), session))).toEqual([])
      const grown = new AgentTurnRunner({ llm, model: 'gpt-4o', bootstrapSystemPrompt: () => bootstrap + 'more'.repeat(4_000) })
      expect(budgetWarnings(await run(grown, session))).toEqual([])
    } finally { await rm(root, { recursive: true, force: true }) }
  })

  test('runner does not warn within budget and still records exact sizes', async () => {
    const root = await mkdtemp(join(tmpdir(), 'xerxes-prompt-budget-small-'))
    try {
      const runner = new AgentTurnRunner({ model: 'gpt-4o', bootstrapSystemPrompt: () => 'short prompt', llm: {
        async *stream() { yield { content: 'done' } },
      } })
      const session = sessionFor(root)
      expect(budgetWarnings(await run(runner, session))).toEqual([])
      expect(readAssemblyProvenance(session.metadata)[0]?.budget).toMatchObject({
        ok: true, layers: [{ name: 'bootstrap', chars: 12, estimatedTokens: 3 }], overages: [],
      })
    } finally { await rm(root, { recursive: true, force: true }) }
  })

  test('cancellation after a warning preserves provenance and does not repeat the warning on retry', async () => {
    const root = await mkdtemp(join(tmpdir(), 'xerxes-prompt-budget-cancel-'))
    try {
      const bootstrap = 'x'.repeat(52_000)
      const controller = new AbortController()
      const cancelled = new AgentTurnRunner({ model: 'gpt-4o', bootstrapSystemPrompt: () => bootstrap, llm: {
        async *stream() { controller.abort(); throw new Error('fixture cancellation') },
      } })
      const session = sessionFor(root)
      const events = await run(cancelled, session, controller.signal)
      expect(budgetWarnings(events)).toHaveLength(1)
      expect(events.some(event => event.payload.stop_reason === 'aborted')).toBe(true)
      expect(readAssemblyProvenance(session.metadata)[0]?.budget?.ok).toBe(false)
      const recovered = new AgentTurnRunner({ model: 'gpt-4o', bootstrapSystemPrompt: () => bootstrap, llm: {
        async *stream() { yield { content: 'recovered' } },
      } })
      expect(budgetWarnings(await run(recovered, session))).toEqual([])
    } finally { await rm(root, { recursive: true, force: true }) }
  })

  test('provider failure after a warning preserves provenance and warning deduplication', async () => {
    const root = await mkdtemp(join(tmpdir(), 'xerxes-prompt-budget-failure-'))
    try {
      const bootstrap = 'x'.repeat(52_000)
      const failing = new AgentTurnRunner({ model: 'gpt-4o', bootstrapSystemPrompt: () => bootstrap, llm: {
        async *stream() { throw new Error('invalid request: fixture provider rejected the turn') },
      } })
      const session = sessionFor(root)
      const events = await run(failing, session)
      expect(events.some(event => event.type === 'notification' && event.payload.level === 'error'
        && String(event.payload.message).includes('fixture provider rejected'))).toBe(true)
      expect(events.some(event => event.payload.stop_reason === 'provider_failed')).toBe(true)
      expect(budgetWarnings(events)).toHaveLength(1)
      expect(readAssemblyProvenance(session.metadata)[0]?.budget?.ok).toBe(false)
      const recovered = new AgentTurnRunner({ model: 'gpt-4o', bootstrapSystemPrompt: () => bootstrap, llm: {
        async *stream() { yield { content: 'recovered' } },
      } })
      expect(budgetWarnings(await run(recovered, session))).toEqual([])
    } finally { await rm(root, { recursive: true, force: true }) }
  })
})
