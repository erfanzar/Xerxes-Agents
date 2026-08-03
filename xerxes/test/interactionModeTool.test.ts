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
  expect(result).toEqual({
    mode: 'plan',
    plan_mode: true,
    reason: 'design first',
    message: 'Interaction mode plan is scheduled for the next turn. Reason: design first',
    guidance: expect.stringContaining('apply on the next user turn'),
  })
  expect(metadata).toEqual({ pending_interaction_mode: 'plan' })
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
    expect(result.guidance).toContain('next user turn')
    expect(metadata.pending_interaction_mode).toBe(nextMode)
  }
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
