// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act, useState } from 'react'
import { expect, it, vi } from 'vitest'
import type { GatewayRpc, GatewayServices } from '../app/interfaces.js'
import { GatewayProvider } from '../app/gatewayContext.js'
import { readRunEvents } from '../lib/runEvents.js'
import { RunEvents } from '../opentui/runEvents.js'
import { RunOverlay } from '../opentui/runOverlay.js'
import { DARK_THEME } from '../theme.js'

const event = (sequence: number) => ({ sequence, text: `Compiler evidence ${sequence}\nsource/path.ts:42`, at: 1789473976000 + sequence })
it.each([
  { events: [event(2), event(1)], next_cursor: 1, has_more: false },
  { events: [event(1)], next_cursor: 9, has_more: true },
  { events: [], next_cursor: 0, has_more: true },
  { events: [{ ...event(1), text: 3 }], next_cursor: 1, has_more: false },
])('rejects invalid evidence pages without silently advancing (%j)', async page => {
  const rpc = vi.fn(async () => ({ ok: true, ...page })) as unknown as GatewayRpc
  await expect(readRunEvents(rpc, 'run', 'workspace')).rejects.toThrow('Invalid run event')
})

it.each([[140, 40], [80, 28], [40, 18]])('pages events and returns after errors at %ix%i', async (width, height) => {
  let fail = true
  const rpc = vi.fn(async (_method: string, params: Record<string, unknown>) => {
    if (params.after_sequence === 20 && fail) return { ok: false, error: 'Disconnected. Reconnect and retry.' }
    const start = Number(params.after_sequence)
    return { ok: true, events: start ? [event(21)] : Array.from({ length: 20 }, (_, i) => event(i + 1)), next_cursor: start ? 21 : 20, has_more: !start }
  })
  const close = vi.fn()
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><RunEvents t={DARK_THEME} runId="run" title="Compile project" scope="workspace" onClose={close} /></GatewayProvider>, { width, height })
  const press = async (key: string) => { await act(async () => screen.mockInput.pressKey(key)); await screen.flush() }
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Compiler evidence 1') })
    expect(screen.captureCharFrame()).toContain('Esc back')
    await press('n')
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Disconnected') })
    expect(screen.captureCharFrame()).toContain('Page 1')
    expect(screen.captureCharFrame()).toContain('Compiler evidence 1')
    fail = false
    await press('r')
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Compiler evidence 21') })
    expect(screen.captureCharFrame()).toContain('Page 2')
    await press('p')
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Page 1') })
    await press('r')
    expect(rpc).toHaveBeenLastCalledWith('run.events', { run_id: 'run', scope: 'workspace', after_sequence: 0, limit: 20 })
    await press('ESCAPE')
    await vi.waitFor(() => expect(close).toHaveBeenCalledOnce())
    expect(rpc.mock.calls.every(([method]) => method === 'run.events')).toBe(true)
  } finally { await act(async () => screen.renderer.destroy()) }
})

it('ignores an old request after switching runs and shows the genuine empty state', async () => {
  let resolveOld!: (value: unknown) => void
  const rpc = vi.fn((_: string, params: Record<string, unknown>) => params.run_id === 'old'
    ? new Promise(resolve => { resolveOld = resolve })
    : Promise.resolve({ ok: true, events: [], next_cursor: 0, has_more: false }))
  const view = (id: string) => <GatewayProvider value={{ rpc } as unknown as GatewayServices}><RunEvents t={DARK_THEME} runId={id} title={id} scope="workspace" onClose={() => {}} /></GatewayProvider>
  let switchRun!: (id: string) => void
  function Harness() { const [id, setId] = useState('old'); switchRun = setId; return view(id) }
  const screen = await testRender(<Harness />, { width: 80, height: 24 })
  try {
    await screen.flush()
    await act(async () => switchRun('new'))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('No saved events.') })
    await act(async () => resolveOld({ ok: true, events: [event(1)], next_cursor: 1, has_more: false }))
    await screen.flush()
    expect(screen.captureCharFrame()).not.toContain('Compiler evidence')
  } finally { await act(async () => screen.renderer.destroy()) }
})

it('opens events from Runs and Escape returns to the selected result without acknowledging', async () => {
  const row = { id: 'run', title: 'Selected build', ownerSessionId: 'owner', kind: 'monitor', state: 'succeeded', revision: 2, unread: true, startedAt: 1, workspace: '/repo', sourceId: 'watch' }
  const rpc = vi.fn(async (method: string) => method === 'run.list' ? { ok: true, runs: [row] }
    : method === 'run.events' ? { ok: true, events: [event(1)], next_cursor: 1, has_more: false }
    : { ok: true, run: { ...row, output: 'Output tail' } })
  const close = vi.fn()
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><RunOverlay t={DARK_THEME} onClose={close} /></GatewayProvider>, { width: 110, height: 35 })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Output tail') })
    await act(async () => screen.mockInput.pressKey('v'))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Compiler evidence 1') })
    await act(async () => screen.mockInput.pressKey('ESCAPE'))
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Output tail') })
    expect(close).not.toHaveBeenCalled()
    expect(rpc.mock.calls.some(([method]) => method === 'run.acknowledge')).toBe(false)
  } finally { await act(async () => screen.renderer.destroy()) }
})
