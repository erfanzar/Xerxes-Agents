// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { EventEmitter } from 'node:events'
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { getOverlayState, patchOverlayState, resetFlowOverlays, resetOverlayState } from '../app/overlayStore.js'
import { getUiState, patchUiState, resetUiState } from '../app/uiStore.js'
import { useLearningSync } from '../app/useLearningSync.js'
import type { GatewayClient } from '../gatewayClient.js'
import type { LearningListResponse } from '../gatewayTypes.js'
import { LearningOverlay } from '../opentui/learningOverlay.js'
import { SessionHeader } from '../opentui/appChrome.js'
import { DEFAULT_THEME } from '../theme.js'

const catalog = (): Extract<LearningListResponse, { ok: true }> => ({
  ok: true, workspace: '/repo', counts: { pending_proposals: 1, pending_lessons: 1 },
  proposals: [{ id: 'p1', workspace: '/repo', status: 'pending', createdAt: 1, updatedAt: 1,
    proposal: { kind: 'skill_create', target: 'safe-build', rationale: 'Use the verified [OBSERVATION]', evidence: ['[OBSERVATION] build passed'], skill: { description: 'Build safely', instructions: 'Run the focused checks.', files: { 'examples/test.ts': 'exact file content' } } } }],
  lessons: [
    { id: 'l1', workspace: '/repo', cue: 'before build', lesson: 'Check the runtime', confidence: 0.9, reviewStatus: 'pending', agentId: 'main', createdAt: 1, evidence: { kind: 'test', command: 'bun test', exitCode: 0, summary: 'tests pass' } },
    { id: 'l2', workspace: '/repo', cue: 'approved cue', lesson: 'Keep native tools', confidence: 1, reviewStatus: 'approved', agentId: 'main', createdAt: 2, evidence: { kind: 'tool_call', toolCallId: 't1', toolName: 'ReadFile', summary: 'checked' } }
  ]
})

afterEach(() => { resetOverlayState(); resetUiState() })

const setup = async (width = 80, height = 24, reject = false) => {
  const data = catalog()
  const rpc = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'learning.list') { await Bun.sleep(0); return data }
    if (method === 'learning.review') {
      if (reject) return { ok: false, report: 'Lint: missing description', error: 'Cannot apply proposal' }
      if (params?.target === 'proposal') data.proposals = data.proposals.map(p => ({ ...p, status: 'applied' }))
      data.counts = { pending_proposals: 0, pending_lessons: 0 }
      return { ok: true, report: 'Human review saved' }
    }
    throw new Error('Unexpected RPC ' + method)
  })
  patchUiState({ sid: 's1', info: { model: 'model', tools: {}, skills: {}, cwd: '/repo' } })
  patchOverlayState({ learning: true })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><LearningOverlay /></GatewayProvider>, { width, height, kittyKeyboard: true })
  await act(async () => { await Bun.sleep(0) })
  await screen.flush()
  const key = async (name: string) => {
    await act(async () => {
      if (name === 'UP' || name === 'DOWN') screen.mockInput.pressArrow(name === 'UP' ? 'up' : 'down')
      else screen.mockInput.pressKey(name)
      await Bun.sleep(0)
    })
    await screen.flush()
  }
  return { screen, key, rpc, data }
}

it.each([[80, 24], [40, 18]])('lists proposals and lessons and keeps navigation visible at %ix%i', async (width, height) => {
  const { screen, key, rpc } = await setup(width, height)
  try {
    expect(rpc).toHaveBeenCalledWith('learning.list', { session_key: 's1' }, { reportError: false })
    expect(screen.captureCharFrame()).toContain('Proposals')
    expect(screen.captureCharFrame()).toContain('Lessons')
    expect(screen.captureCharFrame()).toContain('safe-build')
    expect(screen.captureCharFrame()).toContain('Esc')
    await key('RETURN')
    expect(screen.captureCharFrame()).toContain('skill_create')
    await key('ESCAPE')
    expect(getOverlayState().learning).toBe(true)
    await key('TAB')
    expect(screen.captureCharFrame()).toContain('before build')
    act(() => resetFlowOverlays())
    expect(getOverlayState().learning).toBe(true)
    await key('ESCAPE')
    expect(getOverlayState().learning).toBe(false)
  } finally { act(() => screen.renderer.destroy()) }
})

it('shows the full proposed content and quoted evidence with keyboard scrolling', async () => {
  const { screen, key } = await setup()
  try {
    await key('RETURN')
    expect(screen.captureCharFrame()).toContain('[OBSERVATION] build passed')
    await key('END')
    expect(screen.captureCharFrame()).toContain('exact file content')
  } finally { act(() => screen.renderer.destroy()) }
})

it('approves once, refreshes counts, keeps the result and disallows further actions on outcomes', async () => {
  const { screen, key, rpc } = await setup()
  try {
    await act(async () => { screen.mockInput.pressKey('a'); screen.mockInput.pressKey('a'); await Bun.sleep(5) })
    await screen.flush()
    const reviews = rpc.mock.calls.filter(([method]) => method === 'learning.review')
    expect(reviews).toHaveLength(1)
    expect(reviews[0]).toEqual(['learning.review', { session_key: 's1', target: 'proposal', id: 'p1', action: 'approve' }, { reportError: false }])
    expect(screen.captureCharFrame()).toContain('Human review saved')
    expect(screen.captureCharFrame()).toContain('applied')
    expect(getUiState().pendingProposals + getUiState().pendingLessons).toBe(0)
    await key('a'); await key('r'); await key('b')
    expect(rpc.mock.calls.filter(([method]) => method === 'learning.review')).toHaveLength(1)
  } finally { act(() => screen.renderer.destroy()) }
})

it('requires a reason to reject and retains failures inline', async () => {
  const { screen, key, rpc } = await setup(80, 24, true)
  try {
    await key('r'); await key('RETURN')
    expect(rpc.mock.calls.filter(([method]) => method === 'learning.review')).toHaveLength(0)
    expect(screen.captureCharFrame()).toContain('Reason required')
    await act(async () => { await screen.mockInput.typeText('Not reusable') })
    await screen.flush(); await key('RETURN')
    expect(rpc).toHaveBeenCalledWith('learning.review', { session_key: 's1', target: 'proposal', id: 'p1', action: 'reject', reason: 'Not reusable' }, { reportError: false })
    expect(screen.captureCharFrame()).toContain('Cannot apply proposal')
    expect(screen.captureCharFrame()).toContain('Lint: missing description')
    expect(getOverlayState().learning).toBe(true)
  } finally { act(() => screen.renderer.destroy()) }
})

it('allows rollback only for approved lessons with a reason', async () => {
  const { screen, key, rpc } = await setup()
  try {
    await key('b'); await key('TAB'); await key('b')
    expect(screen.captureCharFrame()).not.toContain('Rollback reason')
    await key('DOWN'); await key('b'); await key('RETURN')
    expect(rpc.mock.calls.filter(([method]) => method === 'learning.review')).toHaveLength(0)
    await act(async () => { await screen.mockInput.typeText('Outdated') })
    await screen.flush(); await key('RETURN')
    expect(rpc).toHaveBeenCalledWith('learning.review', { session_key: 's1', target: 'lesson', id: 'l2', action: 'rollback', reason: 'Outdated' }, { reportError: false })
  } finally { act(() => screen.renderer.destroy()) }
})

it('keeps a list failure inline and remains cancellable without approval', async () => {
  patchUiState({ sid: 's1' })
  patchOverlayState({ learning: true })
  const rpc = vi.fn(async () => { await Bun.sleep(0); throw new Error('Learning store unavailable') })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><LearningOverlay /></GatewayProvider>, { width: 40, height: 18, kittyKeyboard: true })
  try {
    await act(async () => { await Bun.sleep(5) }); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Learning store unavailable')
    expect(screen.captureCharFrame()).toContain('Ctrl+R retry')
    act(() => screen.mockInput.pressKey('a'))
    expect(rpc).toHaveBeenCalledTimes(1)
    act(() => screen.mockInput.pressKey('ESCAPE'))
    expect(getOverlayState().learning).toBe(false)
  } finally { act(() => screen.renderer.destroy()) }
})

it('does not regress count broadcasts with a stale list or leak responses across session switches', async () => {
  const events = new EventEmitter()
  const initial = Promise.withResolvers<LearningListResponse>()
  const replacement = Promise.withResolvers<LearningListResponse>()
  const request = vi.fn().mockReturnValueOnce(initial.promise).mockReturnValueOnce(replacement.promise)
  const gw = Object.assign(events, { request }) as unknown as GatewayClient
  const Probe = () => { useLearningSync(gw); return null }
  patchUiState({ sid: 's1', info: { model: '', skills: {}, tools: {}, cwd: '/repo' } })
  const screen = await testRender(<Probe />, { width: 40, height: 18 })
  try {
    await screen.flush()
    act(() => events.emit('event', { type: 'learning.changed', payload: { workspace: '/repo', counts: { pending_proposals: 4, pending_lessons: 3 } } }))
    await act(async () => { initial.resolve(catalog()) })
    expect(getUiState().pendingProposals + getUiState().pendingLessons).toBe(7)
    await act(async () => { patchUiState({ sid: 's2' }) }); await screen.flush()
    act(() => patchUiState({ sid: 's3' }))
    await act(async () => { replacement.resolve(catalog()) })
    expect(getUiState().pendingProposals + getUiState().pendingLessons).toBe(7)
  } finally { act(() => screen.renderer.destroy()) }
  expect(events.listenerCount('event')).toBe(0)
})

it.each([[80, 24], [40, 18]])('fetches on connect/session switch, uses broadcasts without polling and hides a zero chip at %ix%i', async (width, height) => {
  const events = new EventEmitter()
  const request = vi.fn(async () => { await Bun.sleep(0); return catalog() })
  const gw = Object.assign(events, { request }) as unknown as GatewayClient
  const Probe = () => { useLearningSync(gw); return <LearningChip /> }
  const LearningChip = () => <SessionHeader t={DEFAULT_THEME} sessionTitle="Existing chat" />
  patchUiState({ sid: 's1', info: { model: '', skills: {}, tools: {}, cwd: '/repo' } })
  const screen = await testRender(<Probe />, { width, height })
  try {
    await act(async () => { await Bun.sleep(5) })
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('2 to review')
    await act(async () => { events.emit('event', { type: 'gateway.ready' }); await Bun.sleep(5) })
    await screen.flush()
    expect(request).toHaveBeenCalledTimes(2)
    await act(async () => { patchUiState({ sid: 's2' }) })
    await act(async () => { await Bun.sleep(5) })
    await screen.flush()
    expect(request).toHaveBeenLastCalledWith('learning.list', { session_key: 's2' })
    const before = request.mock.calls.length
    act(() => events.emit('event', { type: 'learning.changed', payload: { workspace: '/other', counts: { pending_proposals: 99, pending_lessons: 99 } } }))
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('2 to review')
    act(() => events.emit('event', { type: 'learning.changed', payload: { workspace: '/repo', counts: { pending_proposals: 3, pending_lessons: 1 } } }))
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('4 to review')
    const state = getUiState()
    act(() => events.emit('event', { type: 'learning.changed', payload: { workspace: '/repo', counts: { pending_proposals: 3, pending_lessons: 1 } } }))
    expect(getUiState()).toBe(state)
    act(() => events.emit('event', { type: 'learning.changed', payload: { workspace: '/repo', counts: { pending_proposals: 0, pending_lessons: 0 } } }))
    await screen.flush()
    expect(screen.captureCharFrame()).not.toContain('to review')
    expect(request).toHaveBeenCalledTimes(before)
  } finally { act(() => screen.renderer.destroy()) }
})
