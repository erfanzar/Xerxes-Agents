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

test('interaction mode tool cannot let the model escape an active guarded mode', async () => {
  const registry = new ToolRegistry()
  let called = false
  registerInteractionModeTool(registry, {
    setMode() {
      called = true
      return { mode: 'code', planMode: false }
    },
  })

  await expect(registry.execute({
    id: 'mode-guarded-exit',
    type: 'function',
    function: { name: 'SetInteractionModeTool', arguments: { mode: 'code' } },
  }, { metadata: { interaction_mode: 'objective' } })).rejects.toThrow(
    'user or session host must switch modes',
  )
  expect(called).toBeFalse()
})
