// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { registerInteractionModeTool } from '../src/runtime/interactionModeTool.js'

test('interaction mode tool delegates the canonical mode to its live-session host', async () => {
  const registry = new ToolRegistry()
  const requests: Array<Record<string, unknown>> = []
  const metadata: Record<string, unknown> = {}
  registerInteractionModeTool(registry, {
    setMode(request) {
      requests.push(request)
      return { mode: request.mode, planMode: request.mode === 'plan' }
    },
  })

  const result = JSON.parse(await registry.execute({
    id: 'mode-1',
    type: 'function',
    function: { name: 'SetInteractionModeTool', arguments: { mode: 'plan', reason: 'design first' } },
  }, {
    agentId: 'default',
    metadata,
    sessionId: 'session-1',
  }))

  expect(requests).toEqual([{
    context: { agentId: 'default', metadata, sessionId: 'session-1' },
    mode: 'plan',
    reason: 'design first',
  }])
  // The host commits the change immediately, so the result says so. It used to
  // claim the transition was "scheduled for the next turn" while the session had
  // already moved — and the enforced tool policy really does lag by a turn, so
  // the two facts are stated separately rather than collapsed into a wrong one.
  expect(result).toEqual({
    mode: 'plan',
    plan_mode: true,
    reason: 'design first',
    message: 'Interaction mode is now plan. Reason: design first',
    guidance: expect.stringContaining('tool policy applies from the next turn'),
  })
  expect(metadata).toMatchObject({
    context_deltas: [expect.objectContaining({ layer: 'interaction-mode', value: 'plan' })],
    pending_interaction_mode: 'plan',
  })
  expect(registry.capabilities('SetInteractionModeTool')).toMatchObject({
    defer: false,
    destructive: false,
    openWorld: false,
    readOnly: false,
  })
})

test('interaction mode tool does not duplicate a context delta already recorded by its host', async () => {
  const registry = new ToolRegistry()
  const metadata: Record<string, unknown> = {
    interaction_mode: 'code',
    context_deltas: [{ at: 1, layer: 'interaction-mode', value: 'plan' }],
  }
  registerInteractionModeTool(registry, {
    setMode(request) {
      return {
        mode: request.mode,
        planMode: request.mode === 'plan',
        contextDeltaRecorded: true,
      }
    },
  })

  await registry.execute({
    id: 'mode-host-delta',
    type: 'function',
    function: { name: 'SetInteractionModeTool', arguments: { mode: 'plan' } },
  }, { agentId: 'default', metadata, sessionId: 'session-1' })

  expect(metadata.context_deltas).toEqual([{ at: 1, layer: 'interaction-mode', value: 'plan' }])
  expect(metadata.pending_interaction_mode).toBe('plan')
})

test('interaction mode tool rejects unknown modes before invoking the host', async () => {
  const registry = new ToolRegistry()
  let called = false
  registerInteractionModeTool(registry, {
    setMode() {
      called = true
      return { mode: 'code', planMode: false }
    },
  })

  await expect(registry.execute({
    id: 'mode-2',
    type: 'function',
    function: { name: 'SetInteractionModeTool', arguments: { mode: 'turbo' } },
  }, { metadata: {} })).rejects.toThrow('must be code, researcher, plan, or objective')
  expect(called).toBeFalse()
})

test('authorized main agent can schedule transitions from every guarded mode', async () => {
  for (const [currentMode, nextMode] of [
    ['plan', 'researcher'],
    ['researcher', 'objective'],
    ['objective', 'code'],
  ] as const) {
    const registry = new ToolRegistry()
    const requests: Array<Record<string, unknown>> = []
    const metadata: Record<string, unknown> = { interaction_mode: currentMode, session_kind: 'main' }
    registerInteractionModeTool(registry, {
      setMode(request) {
        requests.push(request)
        return { mode: request.mode, planMode: request.mode === 'plan' }
      },
    })

    const result = JSON.parse(await registry.execute({
      id: `mode-${currentMode}-to-${nextMode}`,
      type: 'function',
      function: { name: 'SetInteractionModeTool', arguments: { mode: nextMode, reason: 'next policy' } },
    }, { agentId: 'main-agent', metadata, sessionId: 'session-main' }))

    expect(requests).toEqual([{
      context: { agentId: 'main-agent', metadata, sessionId: 'session-main' },
      mode: nextMode,
      reason: 'next policy',
    }])
    expect(result.mode).toBe(nextMode)
    expect(result.guidance).toContain('tool policy applies from the next turn')
    expect(metadata.pending_interaction_mode).toBe(nextMode)
  }
})

test('plan mode is entered and left through the same live session as the mode tool', async () => {
  const registry = new ToolRegistry()
  const applied: Array<{ mode: string }> = []
  const metadata: Record<string, unknown> = { interaction_mode: 'code', session_kind: 'main' }
  registerInteractionModeTool(registry, {
    setMode(request) {
      applied.push({ mode: request.mode })
      return { mode: request.mode, planMode: request.mode === 'plan' }
    },
  })

  const enter = JSON.parse(await registry.execute({
    id: 'enter', type: 'function', function: { name: 'EnterPlanModeTool', arguments: {} },
  }, { agentId: 'main-agent', metadata, sessionId: 'session-plan' }))
  expect(enter).toMatchObject({ mode: 'plan', plan_mode: true })
  expect(metadata).toMatchObject({ pending_interaction_mode: 'plan' })

  const exit = JSON.parse(await registry.execute({
    id: 'exit', type: 'function', function: { name: 'ExitPlanModeTool', arguments: {} },
  }, { agentId: 'main-agent', metadata, sessionId: 'session-plan' }))
  expect(exit).toMatchObject({ mode: 'code', plan_mode: false })
  expect(metadata).toMatchObject({ pending_interaction_mode: 'code' })

  // The whole point: the exit reached the SESSION. The inert WorkflowState copy
  // of this tool reported the same success while changing nothing, which is why
  // plan mode kept refusing to write after the model said it had left.
  expect(applied).toEqual([{ mode: 'plan' }, { mode: 'code' }])
})

test('plan mode tools refuse subagents, exactly like the mode tool', async () => {
  const registry = new ToolRegistry()
  let called = false
  registerInteractionModeTool(registry, {
    setMode() {
      called = true
      return { mode: 'code', planMode: false }
    },
  })

  await expect(registry.execute({
    id: 'child-exit', type: 'function', function: { name: 'ExitPlanModeTool', arguments: {} },
  }, { metadata: { session_kind: 'subagent' } })).rejects.toThrow('only the main agent')
  expect(called).toBeFalse()
})

test('interaction mode tool rejects subagent scheduling even if exposed by a host', async () => {
  for (const metadata of [
    { session_kind: 'subagent' },
    { subagent_id: 'child-1' },
  ]) {
    const registry = new ToolRegistry()
    let called = false
    registerInteractionModeTool(registry, {
      setMode() {
        called = true
        return { mode: 'plan', planMode: true }
      },
    })

    await expect(registry.execute({
      id: 'mode-subagent',
      type: 'function',
      function: { name: 'SetInteractionModeTool', arguments: { mode: 'plan' } },
    }, { agentId: 'child-1', metadata, sessionId: 'session-child' })).rejects.toThrow(
      'only the main agent may schedule interaction-mode transitions',
    )
    expect(called).toBeFalse()
    expect(metadata).not.toHaveProperty('pending_interaction_mode')
  }
})
