// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act, useState } from 'react'
import { afterEach, expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import { GoalOverlay } from '../opentui/goalOverlay.js'
import { SessionFollowups } from '../opentui/sessionFollowups.js'
import { DARK_THEME } from '../theme.js'
const job = { id: 'check', target_session_id: 'owner', prompt: 'Check deployment health', paused: false, schedule: '',
  execution_state: 'idle', runs_started: 3, max_runs: 10, expires_at: '2099-02-01T00:00:00Z', next_run_at: '2099-01-01T00:00:00Z', last_run_at: null }
const response = (jobs: unknown[], owner = 'owner') => ({ ok: true, owner_session_id: owner, jobs })

const inspectedGoal = {
  ok: true,
  session_id: 'owner',
  goal: {
    id: 'goal-1', revision: 3, objective: 'Deploy safely', phase: 'blocked', roundsStarted: 2, maxGoalRounds: 5,
    blockedReason: { code: 'missing-check', message: 'Health check is unavailable.' },
    criteria: [
      { id: 'tests', description: 'All deployment tests pass', evidence: { toolCallId: 'call-7', summary: 'bun test passed', recordedAt: 1_000 } },
      { id: 'health', description: 'The health endpoint responds', },
    ],
  },
}
afterEach(() => { resetOverlayState(); resetUiState() })
it.each([[220, 65], [40, 18]])('renders populated follow-ups beside a goal and opens controls at %ix%i', async (width, height) => {
  patchUiState({ sid: 'owner', info: { model: 'test', skills: {}, tools: {}, goal: 'Deploy safely', goal_phase: 'active' } })
  patchOverlayState({ goal: true })
  const rpc = vi.fn(async () => response([{ ...job, latest_attempt: { state: 'failed' } }]))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width, height })
  try {
    await screen.flush()
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('3/10 attempts') })
    expect(screen.captureCharFrame()).toContain('Deploy safely')
    expect(screen.captureCharFrame()).toContain('Next')
    if (width >= 100) expect(screen.captureCharFrame()).toContain('Latest: failed')
    expect(rpc).toHaveBeenCalledWith('schedule.list', { scope: 'session', owner_session_id: 'owner', summary: true }, { reportError: false })
    act(() => screen.mockInput.pressKey('l'))
    await screen.flush()
    expect(getOverlayState()).toMatchObject({ goal: false, loops: true })
  } finally { act(() => screen.renderer.destroy()) }
})

it.each([[220, 65], [40, 18]])('renders inspected criteria, evidence, rounds, and blockers at %ix%i', async (width, height) => {
  patchUiState({ sid: 'owner', info: { model: 'test', skills: {}, tools: {}, goal: 'fallback', goal_phase: 'active' } })
  const rpc = vi.fn(async (method: string) => method === 'goal.inspect' ? inspectedGoal : response([job]))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width, height })
  try {
    await screen.flush(); await screen.flush()
    if (width < 60) { act(() => screen.mockInput.pressKey('end')); await screen.flush() }
    const frame = screen.captureCharFrame()
    expect(frame).toContain('rounds 2/5')
    expect(frame).toContain('Health check is unavailable.')
    expect(frame).toContain('ACCEPTANCE CRITERIA')
    expect(frame).not.toContain('TIME LIMIT')
    if (width >= 60) {
      expect(frame).toContain('All deployment tests pass')
      expect(frame).toContain('bun test passed')
      expect(frame).toContain('tool call-7')
      expect(frame).toContain('The health endpoint responds')
      expect(frame).toContain('Pending evidence')
      expect(frame).toContain('model-assessed relevance')
    }
    expect(rpc).toHaveBeenCalledWith('goal.inspect', {})
  } finally { act(() => screen.renderer.destroy()); resetUiState() }
})

it('renders a configured wall-clock goal limit with elapsed and remaining time', async () => {
  const createdAt = Date.now() - 30_000
  const timedGoal = { ...inspectedGoal, goal: { ...inspectedGoal.goal, maxDurationMs: 120_000, createdAt } }
  const rpc = vi.fn(async (method: string) => method === 'goal.inspect' ? timedGoal : response([job]))
  patchUiState({ sid: 'owner', info: { model: 'test', skills: {}, tools: {}, goal: 'fallback', goal_phase: 'active' } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try {
    await screen.flush(); await screen.flush()
    const frame = screen.captureCharFrame()
    expect(frame).toContain('TIME LIMIT')
    expect(frame).toContain('elapsed')
    expect(frame).toContain('remaining')
    expect(frame).not.toContain('token')
  } finally { act(() => screen.renderer.destroy()); resetUiState() }
})

it('renders an expired wall-clock goal limit explicitly', async () => {
  const timedGoal = { ...inspectedGoal, goal: { ...inspectedGoal.goal, maxDurationMs: 1_000, createdAt: Date.now() - 2_000 } }
  const rpc = vi.fn(async (method: string) => method === 'goal.inspect' ? timedGoal : response([job]))
  patchUiState({ sid: 'owner', info: { model: 'test', skills: {}, tools: {}, goal: 'fallback', goal_phase: 'active' } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try {
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('TIME LIMIT')
    expect(screen.captureCharFrame()).toContain('remaining 0s')
    expect(screen.captureCharFrame()).toContain('EXPIRED')
  } finally { act(() => screen.renderer.destroy()); resetUiState() }
})

it('renders counted goal tokens against the configured cap and pending calls', async () => {
  const budgetGoal = {
    ...inspectedGoal,
    goal: { ...inspectedGoal.goal, maxTotalTokens: 500 },
    token_usage: { inputTokens: 120, outputTokens: 30, measuredCalls: 2, settledCalls: 1, pendingCalls: 1, complete: false },
  }
  const rpc = vi.fn(async (method: string) => method === 'goal.inspect' ? budgetGoal : response([job]))
  patchUiState({ sid: 'owner', info: { model: 'test', skills: {}, tools: {}, goal: 'fallback', goal_phase: 'active' } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width: 140, height: 30 })
  try {
    await screen.flush(); await screen.flush()
    const frame = screen.captureCharFrame()
    expect(frame).toContain('TOKEN BUDGET')
    expect(frame).toContain('150 / 500 tokens')
    expect(frame).toContain('1 pending calls')
    expect(frame).toContain('usage unknown')
  } finally { act(() => screen.renderer.destroy()); resetUiState() }
})

it('marks a fully measured goal token ledger complete', async () => {
  const budgetGoal = {
    ...inspectedGoal,
    goal: { ...inspectedGoal.goal, maxTotalTokens: 100 },
    token_usage: { inputTokens: 12, outputTokens: 8, measuredCalls: 1, settledCalls: 1, pendingCalls: 0, complete: true },
  }
  const rpc = vi.fn(async (method: string) => method === 'goal.inspect' ? budgetGoal : response([job]))
  patchUiState({ sid: 'owner', info: { model: 'test', skills: {}, tools: {}, goal: 'fallback', goal_phase: 'active' } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width: 140, height: 30 })
  try {
    await screen.flush(); await screen.flush()
    const frame = screen.captureCharFrame()
    expect(frame).toContain('20 / 100 tokens')
    expect(frame).toContain('0 pending calls')
    expect(frame).toContain('usage complete')
    expect(frame).not.toContain('usage unknown')
  } finally { act(() => screen.renderer.destroy()); resetUiState() }
})

it('does not infer zero tokens when the goal ledger is absent', async () => {
  const budgetGoal = { ...inspectedGoal, goal: { ...inspectedGoal.goal, maxTotalTokens: 500 } }
  const rpc = vi.fn(async (method: string) => method === 'goal.inspect' ? budgetGoal : response([job]))
  patchUiState({ sid: 'owner', info: { model: 'test', skills: {}, tools: {}, goal: 'fallback', goal_phase: 'active' } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width: 140, height: 30 })
  try {
    await screen.flush(); await screen.flush()
    const frame = screen.captureCharFrame()
    expect(frame).toContain('TOKEN BUDGET · No usage recorded · cap 500')
    expect(frame).not.toContain('0 / 500 tokens')
  } finally { act(() => screen.renderer.destroy()); resetUiState() }
})

it('keeps the last valid inspection visible when refresh fails', async () => {
  patchUiState({ sid: 'owner', info: { model: 'test', skills: {}, tools: {}, goal: 'fallback', goal_phase: 'active' } })
  let inspections = 0
  const rpc = vi.fn(async (method: string) => {
    if (method === 'goal.inspect') {
      inspections += 1
      return inspections === 1 ? inspectedGoal : { ok: false, error: 'temporary goal service failure' }
    }
    return response([job])
  })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try {
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('r'))
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('end'))
    await screen.flush()
    const frame = screen.captureCharFrame()
    expect(frame).toContain('All deployment tests pass')
    expect(frame).toContain('Goal details unavailable: temporary goal service failure')
    expect(rpc).toHaveBeenCalledTimes(3)
    expect(rpc).toHaveBeenCalledWith('goal.inspect', {})
    expect(rpc.mock.calls.filter(call => call[0] === 'goal.inspect')).toHaveLength(2)
  } finally { act(() => screen.renderer.destroy()); resetUiState() }
})

it('ignores an inspection response from a previous session after switching sessions', async () => {
  let resolveFirst!: (value: unknown) => void
  let resolveSecond!: (value: unknown) => void
  const first = new Promise(resolve => { resolveFirst = resolve })
  const second = new Promise(resolve => { resolveSecond = resolve })
  let goalRequests = 0
  const rpc = vi.fn((method: string) => {
    if (method === 'goal.inspect') return goalRequests++ === 0 ? first : second
    return Promise.resolve(response([job]))
  })
  patchUiState({ sid: 'owner', info: { model: 'test', skills: {}, tools: {}, goal: 'old fallback', goal_phase: 'active' } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try {
    await screen.flush()
    act(() => patchUiState({ sid: 'new-owner', info: { model: 'test', skills: {}, tools: {}, goal: 'new fallback', goal_phase: 'active' } }))
    for (let index = 0; index < 3; index += 1) { await screen.flush(); await Promise.resolve() }
    await act(async () => { resolveSecond({ ok: true, session_id: 'new-owner', goal: { ...inspectedGoal.goal, objective: 'New session objective', criteria: [] } }); await Promise.resolve(); await screen.flush() })
    await act(async () => { resolveFirst({ ...inspectedGoal, goal: { ...inspectedGoal.goal, currentMilestone: 'Old session milestone' } }); await Promise.resolve(); await screen.flush() })
    const frame = screen.captureCharFrame()
    expect(frame).toContain('New session objective')
    expect(frame).not.toContain('Deploy safely')
    expect(frame).not.toContain('All deployment tests pass')
    expect(frame).not.toContain('Old session milestone')
  } finally { act(() => screen.renderer.destroy()); resetUiState() }
})

const decisionGoal = (overrides: Record<string, unknown> = {}) => ({
  ok: true, session_id: 'owner',
  goal: { id: 'goal-1', revision: 3, objective: 'Deploy safely', phase: 'active', roundsStarted: 1, maxGoalRounds: 5,
    criteria: [{ id: 'tests', description: 'All deployment tests pass' }], ...overrides },
})

it('accepts a selected criterion with an explicit human note', async () => {
  patchUiState({ sid: 'owner', info: { model: 'test', skills: {}, tools: {}, goal: 'Deploy safely', goal_phase: 'active' } })
  const accepted = decisionGoal({ revision: 4, criteria: [{ id: 'tests', description: 'All deployment tests pass', evidence: { kind: 'user-decision', decisionId: 'decision-1', summary: 'Verified the deployment checks.', recordedAt: 2_000 } }] })
  let current = decisionGoal()
  const rpc = vi.fn(async (method: string) => {
    if (method === 'goal.decision') { current = accepted; return accepted }
    return method === 'goal.inspect' ? current : response([job])
  })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try {
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('a')); await screen.flush()
    await act(async () => { await screen.mockInput.typeText('Verified the deployment checks.'); await screen.flush() })
    await act(async () => { await screen.mockInput.pressKey('RETURN'); await screen.flush(); await screen.flush() })
    expect(rpc).toHaveBeenCalledWith('goal.decision', { session_id: 'owner', goal_id: 'goal-1', revision: 3, criterion_id: 'tests', summary: 'Verified the deployment checks.' })
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Human decision (user confirmed)') })
  } finally { act(() => screen.renderer.destroy()) }
})

it('cancels decision editing without changing the inspector', async () => {
  patchUiState({ sid: 'owner', info: { model: 'test', skills: {}, tools: {}, goal: 'Deploy safely', goal_phase: 'active' } })
  const rpc = vi.fn(async (method: string) => method === 'goal.inspect' ? decisionGoal() : response([job]))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width: 100, height: 26 })
  try {
    await screen.flush(); await screen.flush(); act(() => screen.mockInput.pressKey('a')); await screen.flush()
    await act(async () => { await screen.mockInput.typeText('draft note'); await screen.flush() })
    await act(async () => { await screen.mockInput.pressKey('ESCAPE'); await screen.flush(); await screen.flush() })
    expect(screen.captureCharFrame()).toContain('All deployment tests pass')
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).not.toContain('ACCEPT CRITERION · HUMAN DECISION') })
    expect(rpc.mock.calls.some(call => call[0] === 'goal.decision')).toBe(false)
  } finally { act(() => screen.renderer.destroy()) }
})

it('keeps a draft after a stale revision error and offers refresh review', async () => {
  patchUiState({ sid: 'owner', info: { model: 'test', skills: {}, tools: {}, goal: 'Deploy safely', goal_phase: 'active' } })
  let revision = 3
  let attempts = 0
  const rpc = vi.fn(async (method: string) => {
    if (method === 'goal.decision') {
      attempts++
      if (attempts === 1) { revision = 4; return { ok: false, error: 'stale goal revision' } }
      revision = 5
    }
    return method === 'goal.inspect' || method === 'goal.decision' ? decisionGoal({ revision }) : response([job])
  })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try {
    await screen.flush(); await screen.flush(); act(() => screen.mockInput.pressKey('a')); await screen.flush()
    await act(async () => { await screen.mockInput.typeText('keep this explanation'); await screen.flush() })
    await act(async () => { await screen.mockInput.pressKey('RETURN'); await screen.flush(); await screen.flush() })
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Draft kept') })
    await act(async () => { screen.mockInput.pressKey('r', { ctrl: true }); await screen.flush() })
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).not.toContain('ACCEPT CRITERION · HUMAN DECISION') })
    await act(async () => { screen.mockInput.pressKey('a'); await screen.flush() })
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('keep this explanation') })
    await act(async () => { screen.mockInput.pressKey('RETURN'); await screen.flush() })
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('goal.decision', {
      session_id: 'owner', goal_id: 'goal-1', revision: 4, criterion_id: 'tests', summary: 'keep this explanation',
    }))
  } finally { act(() => screen.renderer.destroy()) }
})

it('typing shortcuts and pasting multiline notes never accepts a criterion', async () => {
  patchUiState({ sid: 'owner' })
  const rpc = vi.fn(async (method: string) => method === 'goal.inspect' ? decisionGoal() : response([job]))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width: 48, height: 20 })
  try {
    await screen.flush(); await screen.flush()
    await act(async () => { screen.mockInput.pressKey('a'); await screen.flush() })
    await act(async () => { await screen.mockInput.pasteBracketedText('review layout\nsecond line'); await screen.flush() })
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('review layout') })
    expect(screen.captureCharFrame()).toContain('second line')
    expect(screen.captureCharFrame()).toContain('Enter accept')
    expect(rpc.mock.calls.some(call => call[0] === 'goal.decision')).toBe(false)
  } finally { act(() => screen.renderer.destroy()) }
})

it('ignores a late decision response after the user switches sessions', async () => {
  patchUiState({ sid: 'owner' })
  let resolveDecision!: (value: unknown) => void
  const pending = new Promise(resolve => { resolveDecision = resolve })
  let switched = false
  const rpc = vi.fn((method: string) => {
    if (method === 'goal.decision') return pending
    return Promise.resolve(method === 'goal.inspect'
      ? switched ? { ...decisionGoal({ id: 'goal-2', objective: 'New session objective', criteria: [] }), session_id: 'new-owner' } : decisionGoal()
      : response([job]))
  })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width: 100, height: 28 })
  try {
    await screen.flush(); await screen.flush()
    await act(async () => { screen.mockInput.pressKey('a'); await screen.flush() })
    await act(async () => { await screen.mockInput.typeText('I accept'); screen.mockInput.pressKey('RETURN'); await screen.flush() })
    await vi.waitFor(() => expect(rpc.mock.calls.some(call => call[0] === 'goal.decision')).toBe(true))
    switched = true
    act(() => patchUiState({ sid: 'new-owner' }))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('New session objective') })
    await act(async () => { resolveDecision(decisionGoal({ revision: 4 })); await screen.flush() })
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('New session objective')
    expect(screen.captureCharFrame()).not.toContain('Deploy safely')
    expect(screen.captureCharFrame()).not.toContain('Draft kept')
  } finally { act(() => screen.renderer.destroy()) }
})

it('accepts an uncapped goal carrying createdAt and renders human and model evidence separately', async () => {
  const goal = decisionGoal({ createdAt: 123, criteria: [
    { id: 'tests', description: 'All deployment tests pass', evidence: { kind: 'tool-result', toolCallId: 'call-7', summary: 'bun test passed', recordedAt: 1_000 } },
    { id: 'review', description: 'Human review completed', evidence: { kind: 'user-decision', decisionId: 'decision-2', summary: 'Reviewed manually.', recordedAt: 2_000 } },
  ] })
  patchUiState({ sid: 'owner', info: { model: 'test', skills: {}, tools: {}, goal: 'Deploy safely', goal_phase: 'active' } })
  const rpc = vi.fn(async (method: string) => method === 'goal.inspect' ? goal : response([job]))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try {
    await screen.flush(); await screen.flush(); act(() => screen.mockInput.pressKey('end')); await screen.flush()
    const frame = screen.captureCharFrame()
    expect(frame).toContain('model-assessed relevance')
    expect(frame).toContain('Human decision (user confirmed)')
    expect(frame).not.toContain('TIME LIMIT')
  } finally { act(() => screen.renderer.destroy()) }
})

it.each([
  [{ state: 'queued', round: 1 }, 'CONTINUATION · queued · waiting'],
  [{ state: 'running', round: 2 }, 'CONTINUATION · running round 2'],
  [{ state: 'settled', round: 2 }, 'CONTINUATION · settled'],
  [{ state: 'interrupted', round: 2, reason: 'provider stopped' }, 'last round not rerun automatically'],
  [{ state: 'cancelled', reason: 'user cancelled' }, 'CONTINUATION · cancelled · user cancelled'],
] as const)('renders continuation metadata (%j)', async (continuation, expected) => {
  patchUiState({ sid: 'owner', info: { model: 'test', skills: {}, tools: {}, goal: 'Deploy safely', goal_phase: 'active' } })
  const goal = { ...decisionGoal(), continuation: { version: 1, id: 'wake-1', sessionId: 'owner', goalId: 'goal-1', revision: 3, queuedAt: 1_000, ...continuation } }
  const rpc = vi.fn(async (method: string) => method === 'goal.inspect' ? goal : response([job]))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try { await screen.flush(); await screen.flush(); expect(screen.captureCharFrame()).toContain(expected) } finally { act(() => screen.renderer.destroy()) }
})

it('shows resume required when a queued continuation is disarmed', async () => {
  patchUiState({ sid: 'owner', info: { model: 'test', skills: {}, tools: {}, goal: 'Deploy safely', goal_phase: 'active' } })
  const goal = { ...decisionGoal(), goal: { ...decisionGoal().goal, activation: 'disarmed' }, continuation: { version: 1, id: 'wake-1', sessionId: 'owner', goalId: 'goal-1', revision: 3, state: 'queued', queuedAt: 1_000 } }
  const rpc = vi.fn(async (method: string) => method === 'goal.inspect' ? goal : response([job]))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try { await screen.flush(); await screen.flush(); expect(screen.captureCharFrame()).toContain('/goal resume required') } finally { act(() => screen.renderer.destroy()) }
})

it.each([[120, 30], [40, 16]])('renders a multiline current milestone at %ix%i', async (width, height) => {
  patchUiState({ sid: 'owner', info: { model: 'test', skills: {}, tools: {}, goal: 'Deploy safely', goal_phase: 'active' } })
  const goal = { ...decisionGoal(), goal: { ...decisionGoal().goal, currentMilestone: 'Validate the health endpoint\nthen review the rollout evidence.' } }
  const rpc = vi.fn(async (method: string) => method === 'goal.inspect' ? goal : response([job]))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width, height })
  try { await screen.flush(); await screen.flush(); expect(screen.captureCharFrame()).toContain('CURRENT MILESTONE'); expect(screen.captureCharFrame()).toContain('Validate the health endpoint') } finally { act(() => screen.renderer.destroy()) }
})

it('labels a completed goal milestone as the last milestone and leaves legacy goals blank', async () => {
  patchUiState({ sid: 'owner', info: { model: 'test', skills: {}, tools: {}, goal: 'Deploy safely', goal_phase: 'active' } })
  let current = { ...decisionGoal(), goal: { ...decisionGoal().goal, phase: 'complete', currentMilestone: 'Release verified' } }
  const rpc = vi.fn(async (method: string) => method === 'goal.inspect' ? current : response([job]))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width: 100, height: 26 })
  try { await screen.flush(); await screen.flush(); expect(screen.captureCharFrame()).toContain('LAST MILESTONE'); expect(screen.captureCharFrame()).toContain('Release verified'); current = { ...decisionGoal(), goal: { ...decisionGoal().goal } }; act(() => screen.mockInput.pressKey('r')); await screen.flush(); await screen.flush(); expect(screen.captureCharFrame()).not.toContain('Release verified') } finally { act(() => screen.renderer.destroy()) }
})

it('rejects a malformed current milestone projection', async () => {
  patchUiState({ sid: 'owner', info: { model: 'test', skills: {}, tools: {}, goal: 'Deploy safely', goal_phase: 'active' } })
  const rpc = vi.fn(async (method: string) => method === 'goal.inspect' ? { ...decisionGoal(), goal: { ...decisionGoal().goal, currentMilestone: '' } } : response([job]))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width: 100, height: 26 })
  try { await screen.flush(); await screen.flush(); expect(screen.captureCharFrame()).toContain('Goal details unavailable: Invalid goal inspection milestone') } finally { act(() => screen.renderer.destroy()) }
})

it('polls goal inspection every three seconds without overlapping requests', async () => {
  let active = 0
  let maxActive = 0
  let calls = 0
  let intervalCallback: (() => void) | undefined
  let intervalDelay: number | undefined
  const setIntervalSpy = vi.spyOn(globalThis, 'setInterval').mockImplementation(((callback: TimerHandler, delay?: number) => {
    intervalCallback = callback as () => void
    intervalDelay = delay
    return 1 as ReturnType<typeof setInterval>
  }) as typeof setInterval)
  const finishers: Array<() => void> = []
  const rpc = vi.fn((method: string) => {
    if (method !== 'goal.inspect') return Promise.resolve(response([job]))
    calls += 1
    active += 1
    maxActive = Math.max(maxActive, active)
    return new Promise(resolve => {
      finishers.push(() => { active -= 1; resolve(inspectedGoal) })
    })
  })
  patchUiState({ sid: 'owner', info: { model: 'test', skills: {}, tools: {}, goal: 'fallback', goal_phase: 'active' } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><GoalOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try {
    await screen.flush()
    expect(calls).toBe(1)
    expect(intervalDelay).toBe(3_000)
    expect(intervalCallback).toBeDefined()
    await act(async () => { intervalCallback?.(); await Promise.resolve(); await screen.flush() })
    expect(calls).toBe(1)
    await act(async () => { finishers.shift()?.(); await Promise.resolve(); await screen.flush() })
    await act(async () => { intervalCallback?.(); await Promise.resolve(); await screen.flush() })
    expect(calls).toBe(2)
    expect(maxActive).toBe(1)
  } finally {
    act(() => screen.renderer.destroy())
    setIntervalSpy.mockRestore()
    resetUiState()
  }
})

it.each([
  [{ expires_at: '2000-01-01T00:00:00Z' }, 'Expired'],
  [{ runs_started: 10 }, 'Attempt limit reached'],
  [{ paused: true }, 'Paused'],
  [{ paused: true, metadata: { followup_completion: { source: 'model_reported' } } }, 'Condition met · model reported'],
  [{ metadata: { execution_recovery_required: true } }, 'Needs review'],
  [{ execution_state: 'cancelling', runs_started: 10 }, 'Cancelling'],
  [{ execution_state: 'running', runs_started: 10 }, 'Running or queued'],
])('renders the actual eligibility without promising another wake: %j', async (changes, expected) => {
  const rpc = vi.fn(async () => response([{ ...job, ...changes }]))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><SessionFollowups t={DARK_THEME} sessionId="owner" /></GatewayProvider>, { width: 150, height: 20 })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain(expected) })
    expect(screen.captureCharFrame()).not.toContain('Next')
  } finally { act(() => screen.renderer.destroy()) }
})
it('discards an old conversation response after switching and shows errors instead of stale wake promises', async () => {
  let finish!: (result: unknown) => void
  let switchOwner!: (owner: string) => void
  const rpc = vi.fn(async (_method: string, params: { owner_session_id: string }) => params.owner_session_id === 'owner'
    ? new Promise(resolve => { finish = resolve }) : { ok: false, error: 'Disconnected' })
  function Harness() {
    const [owner, setOwner] = useState('owner'); switchOwner = setOwner
    return <GatewayProvider value={{ rpc } as unknown as GatewayServices}><SessionFollowups t={DARK_THEME} sessionId={owner} /></GatewayProvider>
  }
  const screen = await testRender(<Harness />, { width: 150, height: 20 })
  try {
    await screen.flush()
    act(() => switchOwner('second'))
    await screen.flush()
    finish(response([job]))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Follow-ups unavailable') })
    expect(screen.captureCharFrame()).not.toContain('3/10 attempts')
  } finally { act(() => screen.renderer.destroy()) }
})
it('refuses a mismatched owner returned by the daemon', async () => {
  const rpc = vi.fn(async () => response([job], 'other'))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><SessionFollowups t={DARK_THEME} sessionId="owner" /></GatewayProvider>, { width: 80, height: 20 })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Follow-ups unavailable') })
    expect(screen.captureCharFrame()).not.toContain('3/10 attempts')
  } finally { act(() => screen.renderer.destroy()) }
})

it.each([true, false])('shows blocked lifetime tokens instead of promising a wake (complete=%s)', async complete => {
  const rpc = vi.fn(async () => response([{ ...job, token_budget: { used: complete ? 20 : 5, maximum: 20, complete, blocked: true } }]))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><SessionFollowups t={DARK_THEME} sessionId="owner" expanded /></GatewayProvider>, { width: 100, height: 20 })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain(complete ? 'Token threshold reached' : 'Token usage incomplete') })
    expect(screen.captureCharFrame()).toContain(`Lifetime tokens: ${complete ? 20 : 5} / 20`)
    expect(screen.captureCharFrame()).not.toContain('Next 2099')
  } finally { act(() => screen.renderer.destroy()) }
})

it('ignores a rejected refresh after changing conversations without reporting it globally', async () => {
  let rejectOld!: (error: Error) => void
  let switchOwner!: () => void
  const rpc = vi.fn(async (_method: string, params: { owner_session_id: string }, options?: { reportError?: boolean }) => {
    expect(options?.reportError).toBe(false)
    if (params.owner_session_id === 'owner') return await new Promise((_resolve, reject) => { rejectOld = reject })
    return response([], 'second')
  })
  function Harness() {
    const [owner, setOwner] = useState('owner'); switchOwner = () => setOwner('second')
    return <GatewayProvider value={{ rpc } as unknown as GatewayServices}><SessionFollowups t={DARK_THEME} sessionId={owner} expanded /></GatewayProvider>
  }
  const screen = await testRender(<Harness />, { width: 80, height: 18 })
  try {
    await screen.flush()
    act(() => switchOwner())
    await screen.flush()
    rejectOld(new Error('The active conversation changed; refresh its follow-ups before acting'))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('No follow-ups') })
    expect(screen.captureCharFrame()).not.toContain('conversation changed')
    expect(screen.captureCharFrame()).not.toContain('unavailable')
  } finally { act(() => screen.renderer.destroy()) }
})
