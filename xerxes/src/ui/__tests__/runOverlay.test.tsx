// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { getOverlayState, patchOverlayState, resetFlowOverlays, resetOverlayState } from '../app/overlayStore.js'
import { RunOverlay } from '../opentui/runOverlay.js'
import { DARK_THEME } from '../theme.js'

const row = { id: 'run-1', title: 'Build application', kind: 'terminal', state: 'failed', revision: 2, unread: true, startedAt: 1, endedAt: 2000, exitCode: 1, terminalKind: 'background', ownerSessionId: 'session-1', workspace: '/repo', sourceId: 'process-1' }
afterEach(() => resetOverlayState())
it.each([[220, 65], [110, 35], [60, 24], [40, 18]])('renders populated runs with readable controls at %ix%i', async (width, height) => {
  const rpc = vi.fn(async (method: string) => method === 'run.list' ? { ok: true, runs: [row] } : { ok: true, run: { ...row, output: 'Typecheck failed at app.ts:12', error: 'exit 1' } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><RunOverlay t={DARK_THEME} /></GatewayProvider>, { width, height })
  try {
    await screen.flush()
    await screen.flush()
    const text = screen.captureCharFrame()
    expect(text).toContain('Runs')
    expect(text).toContain('Build application')
    expect(text).toContain('Esc close')
    expect(text).toContain('Typecheck failed')
    if (width >= 110) {
      expect(text).toContain('source process-1')
      expect(text).toContain('background · exit 1')
      expect(text).toContain('revision 2')
    }
  } finally { act(() => screen.renderer.destroy()) }
})
it('preserves user-opened runs across turn completion and acknowledges the inspected revision', async () => {
  patchOverlayState({ runs: true })
  resetFlowOverlays()
  expect(getOverlayState().runs).toBe(true)
  const rpc = vi.fn(async (method: string) => method === 'run.list' ? { ok: true, runs: [row] } : { ok: true, run: { ...row, output: 'failure details' } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><RunOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try {
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('a'))
    await screen.flush()
    expect(rpc).toHaveBeenCalledWith('run.acknowledge', { run_id: row.id, revision: 2, scope: 'workspace' })
    act(() => screen.mockInput.pressKey('ESCAPE'))
    await screen.flush()
    await vi.waitFor(() => expect(getOverlayState().runs).toBe(false))
  } finally { act(() => screen.renderer.destroy()) }
})
it('shows a daemon failure instead of fabricating an empty successful inbox', async () => {
  const rpc = vi.fn(async () => ({ ok: false, error: 'History storage unavailable' }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><RunOverlay t={DARK_THEME} /></GatewayProvider>, { width: 100, height: 25 })
  try { await screen.flush(); expect(screen.captureCharFrame()).toContain('History storage unavailable') }
  finally { act(() => screen.renderer.destroy()) }
})
it('shows unresolved reaction cleanup and its limits separately from the watch outcome', async () => {
  const watch = { ...row, kind: 'monitor', state: 'succeeded' }
  const rpc = vi.fn(async (method: string) => method === 'run.list' ? { ok: true, runs: [watch] } : { ok: true, run: {
    ...watch, output: 'error: compiler failed', reaction_health: { state: 'awaiting-cleanup', attempts: 2, maxReactions: 3, pendingEvents: 4, lastError: 'provider unavailable', usage: { inputTokens: 120, outputTokens: 30, complete: false } },
  } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><RunOverlay t={DARK_THEME} /></GatewayProvider>, { width: 160, height: 40 })
  try {
    await screen.flush(); await screen.flush()
    const text = screen.captureCharFrame()
    expect(text).toContain('awaiting-cleanup')
    expect(text).toContain('2/3 attempts')
    expect(text).toContain('4 queued events')
    expect(text).toContain('provider unavailable')
    expect(text).toContain('120 input')
    expect(text).toContain('incomplete usage')
  } finally { act(() => screen.renderer.destroy()) }
})

it('pages older results and resets the cursor when the scope changes', async () => {
  const first = Array.from({ length: 100 }, (_, index) => ({ ...row, id: `run-${index}`, startedAt: 200 - index }))
  const rpc = vi.fn(async (method: string, params: Record<string, unknown>) => method === 'run.list'
    ? { ok: true, runs: params.before_id ? [{ ...row, id: 'older', title: 'Older result' }] : first, has_more: !params.before_id }
    : { ok: true, run: { ...row, id: params.run_id, output: 'Output' } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><RunOverlay t={DARK_THEME} /></GatewayProvider>, { width: 150, height: 40 })
  try {
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('n'))
    await screen.flush(); await screen.flush()
    expect(rpc).toHaveBeenCalledWith('run.list', expect.objectContaining({ before_id: 'run-99', before_started_at: 101 }))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Older result') })
    expect(screen.captureCharFrame()).toContain('Page 2')
    act(() => screen.mockInput.pressKey('w'))
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Page 1')
    expect(rpc.mock.calls.filter(([method]) => method === 'run.list').at(-1)?.[1]).not.toHaveProperty('before_id')
  } finally { await act(async () => { screen.renderer.destroy() }) }
})

it('passes kind and state filters to the daemon rather than filtering only the current page', async () => {
  const rpc = vi.fn(async (method: string, params: Record<string, unknown>) => method === 'run.list'
    ? { ok: true, runs: [{ ...row, title: params.state === 'failed' ? 'Old failure' : 'Recent result' }], has_more: false }
    : { ok: true, run: { ...row, output: 'Output' } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><RunOverlay t={DARK_THEME} /></GatewayProvider>, { width: 150, height: 40 })
  try {
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('k'))
    act(() => screen.mockInput.pressKey('s'))
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('s'))
    await screen.flush(); await screen.flush()
    expect(rpc).toHaveBeenLastCalledWith(expect.any(String), expect.any(Object))
    const request = rpc.mock.calls.filter(([method]) => method === 'run.list').at(-1)?.[1]
    expect(request).toMatchObject({ kind: 'agent', state: 'failed' })
    expect(request).not.toHaveProperty('before_id')
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Old failure') })
    expect(screen.captureCharFrame()).toContain('S failed')
  } finally { await act(async () => { screen.renderer.destroy() }) }
})

it('shows daemon-provided cancellation and submits the inspected revision', async () => {
  const active = { ...row, state: 'running', cancel_label: 'Stop process', output: 'Building' }
  const rpc = vi.fn(async (method: string) => method === 'run.list' ? { ok: true, runs: [active] } : method === 'run.cancel' ? { ok: false, error: 'Run changed; refresh before cancelling' } : { ok: true, run: active })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><RunOverlay t={DARK_THEME} /></GatewayProvider>, { width: 150, height: 40 })
  try {
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('X · Stop process')
    act(() => screen.mockInput.pressKey('x'))
    await screen.flush(); await screen.flush()
    expect(rpc).toHaveBeenCalledWith('run.cancel', { run_id: row.id, revision: row.revision, scope: 'workspace' })
    expect(screen.captureCharFrame()).toContain('Run changed')
  } finally { await act(async () => { screen.renderer.destroy() }) }
})

it('keeps an inspection failure visible through successful list polling and clears it after retry', async () => {
  let available = false
  let lists = 0
  const rpc = vi.fn(async (method: string) => {
    if (method === 'run.list') { lists++; return { ok: true, runs: [row] } }
    if (!available) throw new Error('Result storage unavailable')
    return { ok: true, run: { ...row, output: 'Recovered result' } }
  })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><RunOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 30 })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Result storage unavailable') })
    await vi.waitFor(() => expect(lists).toBeGreaterThan(1), { timeout: 4000 })
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('Result storage unavailable')
    available = true
    await act(async () => screen.mockInput.pressKey('r'))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Recovered result') })
    expect(screen.captureCharFrame()).not.toContain('Result storage unavailable')
  } finally { act(() => screen.renderer.destroy()) }
})

it.each([true, false])('renders historical schedule usage with completeness %s', async complete => {
  const schedule = { ...row, kind: 'schedule' }
  const rpc = vi.fn(async (method: string) => method === 'run.list' ? { ok: true, runs: [schedule] } : { ok: true, run: { ...schedule, output: 'Saved evidence', tokenUsage: { input_tokens: 123, output_tokens: 45, complete } } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><RunOverlay t={DARK_THEME} /></GatewayProvider>, { width: 160, height: 40 })
  try {
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('123 input')
    expect(screen.captureCharFrame().includes('some usage unavailable')).toBe(!complete)
  } finally { act(() => screen.renderer.destroy()) }
})

it.each([[220, 65], [40, 18]])('shows upcoming workspace jobs and opens schedule management at %ix%i', async (width, height) => {
  const rpc = vi.fn(async () => ({ ok: true, runs: [], upcoming_total: 1,
    upcoming: [{ id: 'job', title: 'Review builds', next_run_at: '2099-01-01T09:00:00Z', timezone: 'UTC', execution_state: 'idle' }] }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><RunOverlay t={DARK_THEME} /></GatewayProvider>, { width, height })
  try {
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Scheduled next')
    expect(screen.captureCharFrame()).toContain('2099-01-01')
    act(() => screen.mockInput.pressKey('t')); await screen.flush()
    expect(getOverlayState().schedules).toBe(true)
    expect(getOverlayState().runs).toBe(false)
  } finally { act(() => screen.renderer.destroy()) }
})

it.each([[220, 65], [40, 18]])('shows session attention and returns to chat without approving at %ix%i', async (width, height) => {
  const close = vi.fn()
  const rpc = vi.fn(async () => ({ ok: true, runs: [], attention_total: 1,
    attention: [{ id: 'approval', kind: 'approval', title: 'Write report' }] }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><RunOverlay t={DARK_THEME} onClose={close} /></GatewayProvider>, { width, height })
  try {
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Session attention')
    expect(screen.captureCharFrame()).toContain('Write report')
    act(() => screen.mockInput.pressKey('e')); await screen.flush()
    expect(close).toHaveBeenCalledOnce()
    expect(rpc.mock.calls.every(call => call[0] === 'run.list')).toBe(true)
  } finally { act(() => screen.renderer.destroy()) }
})
