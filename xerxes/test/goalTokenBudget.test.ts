// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { captureModelCallScopes, chargeModelCall, withCapturedModelCallScopes, withModelCallBudget } from '../src/llms/callBudget.js'
import { createGoal, completeGoal, getGoal } from '../src/runtime/goalDomain.js'
import { GoalTokenLedger } from '../src/runtime/goalTokenLedger.js'
import { GoalTokenBudget } from '../src/runtime/goalTokenBudget.js'
import { SubAgentManager } from '../src/agents/subagentManager.js'
import { completeLlm, type LlmClient } from '../src/llms/client.js'
import { ProviderError } from '../src/core/errors.js'
import { createAgentState, type StreamEvent } from '../src/streaming/events.js'
import { runTurn } from '../src/streaming/loop.js'

function fixture(work: (ledger: GoalTokenLedger) => void) {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-goal-token-scope-'))
  const ledger = new GoalTokenLedger(join(directory, 'ledger.sqlite'))
  try { work(ledger) } finally { ledger.close(); rmSync(directory, { recursive: true, force: true }) }
}

test('detached scopes retain goal ownership and nested scopes charge once', () => fixture(ledger => {
  const session = { id: 'scope-parent', metadata: {} }
  const goal = createGoal(session.metadata, session.id, { objective: 'work', maxTotalTokens: 20 }, 1000)
  ledger.initialize(session.id, goal.id, true)
  const budget = new GoalTokenBudget(() => session, ledger, 'owner', session.id)
  const child = withModelCallBudget(budget, () => {
    chargeModelCall()!({ inputTokens: 5, outputTokens: 5 })
    return captureModelCallScopes()
  })
  withModelCallBudget(budget, () => withCapturedModelCallScopes(child, () => chargeModelCall()!({ inputTokens: 5, outputTokens: 5 })))
  expect(ledger.inspect(session.id, goal.id)).toMatchObject({ inputTokens: 10, outputTokens: 10, settledCalls: 2 })
  expect(() => withCapturedModelCallScopes(child, () => chargeModelCall())).toThrow()
  expect(getGoal(session.metadata, session.id)?.blockedReason?.code).toBe('token-budget')
  expect(() => withModelCallBudget(budget, () => chargeModelCall())).toThrow()
}))

test('a scope activates for a model-created goal before the next provider call', () => fixture(ledger => {
  const session = { id: 'scope-new', metadata: {} }
  const budget = new GoalTokenBudget(() => session, ledger, 'owner', session.id)
  withModelCallBudget(budget, () => {
    const beforeGoal = chargeModelCall()!
    const goal = createGoal(session.metadata, session.id, { objective: 'new goal', maxTotalTokens: 10 }, 1000)
    ledger.initialize(session.id, goal.id, true)
    beforeGoal({ inputTokens: 100, outputTokens: 100 })
    chargeModelCall()!({ inputTokens: 8, outputTokens: 2 })
    expect(ledger.inspect(session.id, goal.id)).toMatchObject({ inputTokens: 8, outputTokens: 2, settledCalls: 1 })
    expect(() => chargeModelCall()).toThrow()
  })
}))

test('a captured child cannot silently charge a replacement goal', () => fixture(ledger => {
  const session = { id: 'scope-replace', metadata: {} }
  const goal = createGoal(session.metadata, session.id, { objective: 'old', maxTotalTokens: 10 }, 1000)
  ledger.initialize(session.id, goal.id, true)
  const budget = new GoalTokenBudget(() => session, ledger, 'owner', session.id)
  const captured = withModelCallBudget(budget, captureModelCallScopes)
  completeGoal(session.metadata, session.id, goal, 2000)
  const replacement = createGoal(session.metadata, session.id, { objective: 'new', maxTotalTokens: 10 }, 3000)
  ledger.initialize(session.id, replacement.id, true)
  expect(() => withCapturedModelCallScopes(captured, () => chargeModelCall())).toThrow()
  expect(getGoal(session.metadata, session.id)?.phase).toBe('active')
  expect(ledger.inspect(session.id, replacement.id)?.settledCalls).toBe(0)
}))

test('an existing capped goal without a ledger baseline fails closed', () => fixture(ledger => {
  const session = { id: 'scope-legacy', metadata: {} }
  createGoal(session.metadata, session.id, { objective: 'legacy', maxTotalTokens: 10 }, 1000)
  const budget = new GoalTokenBudget(() => session, ledger, 'owner', session.id)
  expect(() => withModelCallBudget(budget, () => chargeModelCall())).toThrow()
  expect(getGoal(session.metadata, session.id)?.phase).toBe('blocked')
}))

test('subagent retries outside the parent async context retain the goal cap', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-goal-child-budget-'))
  const ledger = new GoalTokenLedger(join(directory, 'ledger.sqlite'))
  const session = { id: 'child-owner', metadata: {} }
  const goal = createGoal(session.metadata, session.id, { objective: 'child work', maxTotalTokens: 10 }, 1000)
  ledger.initialize(session.id, goal.id, true)
  const budget = new GoalTokenBudget(() => session, ledger, 'owner', session.id)
  const manager = new SubAgentManager({ runner: async () => {
    chargeModelCall()!({ inputTokens: 4, outputTokens: 2 })
    return 'done'
  } })
  try {
    const task = await withModelCallBudget(budget, () => manager.spawn({ prompt: 'work', name: 'worker' }))
    await manager.wait(task.id, 2000)
    await manager.retry(task.id)
    await manager.wait(task.id, 2000)
    expect(ledger.inspect(session.id, goal.id)).toMatchObject({ inputTokens: 8, outputTokens: 4, settledCalls: 2 })
    await manager.retry(task.id)
    await manager.wait(task.id, 2000)
    expect(ledger.inspect(session.id, goal.id)?.settledCalls).toBe(2)
    expect(getGoal(session.metadata, session.id)?.blockedReason?.code).toBe('token-budget')
  } finally { ledger.close(); rmSync(directory, { recursive: true, force: true }) }
})

test('provider completions account for reported cache tokens before admitting another call', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-goal-provider-budget-'))
  const ledger = new GoalTokenLedger(join(directory, 'ledger.sqlite'))
  const session = { id: 'provider-owner', metadata: {} }
  const goal = createGoal(session.metadata, session.id, { objective: 'provider work', maxTotalTokens: 10 }, 1000)
  ledger.initialize(session.id, goal.id, true)
  let calls = 0
  const llm: LlmClient = { async *stream() {
    calls++
    yield { content: 'done', usage: { inputTokens: 2, outputTokens: 2, cacheReadTokens: 3, cacheCreationTokens: 3 } }
  } }
  const budget = new GoalTokenBudget(() => session, ledger, 'owner', session.id)
  try {
    await withModelCallBudget(budget, async () => {
      await completeLlm(llm, { model: 'test', messages: [{ role: 'user', content: 'auxiliary' }] })
      await expect(completeLlm(llm, { model: 'test', messages: [{ role: 'user', content: 'next' }] })).rejects.toThrow()
    })
    expect(calls).toBe(1)
    expect(ledger.inspect(session.id, goal.id)).toMatchObject({ inputTokens: 8, outputTokens: 2, measuredCalls: 1, complete: true })
  } finally { ledger.close(); rmSync(directory, { recursive: true, force: true }) }
})

test('a request the provider rejected before streaming does not block a capped goal', async () => {
  // Regression: the loop settled a 429 attempt as unknown spend, so the retry's
  // admission found an unmeasured receipt and blocked the goal for good.
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-goal-rejected-'))
  const ledger = new GoalTokenLedger(join(directory, 'ledger.sqlite'))
  const session = { id: 'rejected-owner', metadata: {} }
  const goal = createGoal(session.metadata, session.id, { objective: 'retry work', maxTotalTokens: 1_000 }, 1000)
  ledger.initialize(session.id, goal.id, true)
  let calls = 0
  const llm: LlmClient = { async *stream() {
    calls++
    if (calls === 1) throw new ProviderError('anthropic', 'stream request failed (429): rate limited', undefined, { status: 429 })
    yield { content: 'done', usage: { inputTokens: 3, outputTokens: 2 } }
  } }
  const budget = new GoalTokenBudget(() => session, ledger, 'owner', session.id)
  try {
    const events: StreamEvent[] = []
    await withModelCallBudget(budget, async () => {
      for await (const event of runTurn({ model: 'test', state: createAgentState(), userMessage: 'go' }, { delay: async () => undefined, llm, retryDelays: [0] })) events.push(event)
    })
    expect(calls).toBe(2)
    expect(events.some(event => event.type === 'text' && event.text === 'done')).toBe(true)
    expect(getGoal(session.metadata, session.id)?.phase).toBe('active')
    expect(ledger.inspect(session.id, goal.id)).toMatchObject({ inputTokens: 3, outputTokens: 2, settledCalls: 2, complete: true })
    // A later round (or /goal resume) is still admitted.
    expect(() => ledger.assertAdmission(session.id, goal.id, 'owner', 1_000)).not.toThrow()
  } finally { ledger.close(); rmSync(directory, { recursive: true, force: true }) }
})

test('a stream that fails after output still settles as unknown spend', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-goal-dropped-'))
  const ledger = new GoalTokenLedger(join(directory, 'ledger.sqlite'))
  const session = { id: 'dropped-owner', metadata: {} }
  const goal = createGoal(session.metadata, session.id, { objective: 'dropped work', maxTotalTokens: 1_000 }, 1000)
  ledger.initialize(session.id, goal.id, true)
  const llm: LlmClient = { async *stream() {
    yield { content: 'partial' }
    throw new ProviderError('anthropic', 'stream failed (500) mid-response', undefined, { status: 500 })
  } }
  const budget = new GoalTokenBudget(() => session, ledger, 'owner', session.id)
  try {
    await withModelCallBudget(budget, async () => {
      for await (const _event of runTurn({ model: 'test', state: createAgentState(), userMessage: 'go' }, { delay: async () => undefined, llm, retryDelays: [0] })) { /* drain */ }
    })
    expect(ledger.inspect(session.id, goal.id)).toMatchObject({ complete: false })
    expect(getGoal(session.metadata, session.id)?.phase).toBe('blocked')
  } finally { ledger.close(); rmSync(directory, { recursive: true, force: true }) }
})
