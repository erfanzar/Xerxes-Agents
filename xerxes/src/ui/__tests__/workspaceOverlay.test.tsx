// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { getOverlayState, patchOverlayState, resetFlowOverlays, resetOverlayState } from '../app/overlayStore.js'
import { WorkspaceOverlay } from '../opentui/workspaceOverlay.js'
import { DARK_THEME } from '../theme.js'
import { opsCommands } from '../app/slash/commands/ops.js'
import type { SlashRunCtx } from '../app/slash/types.js'

const row = { id: 'one', taskId: 'Fix runtime', path: '/repo/agent-one', branch: 'agent-one' }
const detail = { ...row, base: 'abc123', head: 'abc123', status: ' M file.ts', diff: 'diff --git a/file.ts b/file.ts\n--- a/file.ts\n+++ b/file.ts\n@@ -1 +1 @@\n-old code\n+new code\n' }
afterEach(() => resetOverlayState())
it.each([[220, 65], [110, 35], [60, 24], [40, 18]])('renders workspace review at %ix%i', async (width, height) => {
  const rpc = vi.fn(async (method: string) => method === 'workspace.list' ? { ok: true, inventory: { records: [row] } } : { ok: true, review: detail })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><WorkspaceOverlay t={DARK_THEME} /></GatewayProvider>, { width, height })
  try {
    await screen.flush(); await screen.flush()
    const text = screen.captureCharFrame()
    expect(text).toContain('Agent workspaces')
    expect(text).toContain('Fix runtime')
    expect(text).toContain('Esc close')
    expect(rpc).toHaveBeenCalledWith('workspace.inspect', { workspace_id: 'one' })
    if (height >= 24) expect(text).toContain('new code')
  } finally { act(() => screen.renderer.destroy()) }
})
it('opens from /workspaces and preserves the panel across turn completion', () => {
  opsCommands.find(command => command.name === 'workspaces')!.run('', {} as SlashRunCtx, 'workspaces')
  expect(getOverlayState().workspaces).toBe(true)
  resetFlowOverlays()
  expect(getOverlayState().workspaces).toBe(true)
})
it('ignores stale reviews, handles paging and closes on Escape', async () => {
  const old = Promise.withResolvers<unknown>()
  const rpc = vi.fn(async (method: string, params: Record<string, unknown>) => {
    if (method === 'workspace.list') return { ok: true, inventory: { records: params.after ? [{ ...row, id: 'three', taskId: 'Third task' }] : [row, { ...row, id: 'two', taskId: 'Second task' }], ...(params.after ? {} : { next: 'two' }) } }
    if (params.workspace_id === 'one') return old.promise
    return { ok: true, review: { ...detail, id: params.workspace_id, taskId: params.workspace_id === 'two' ? 'Second task' : 'Third task', diff: '+current result' } }
  })
  patchOverlayState({ workspaces: true })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><WorkspaceOverlay t={DARK_THEME} /></GatewayProvider>, { width: 180, height: 45 })
  try {
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressArrow('down'))
    await screen.flush(); await screen.flush()
    old.resolve({ ok: true, review: { ...detail, diff: '+STALE RESULT' } })
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).not.toContain('STALE RESULT')
    expect(screen.captureCharFrame()).toContain('current result')
    act(() => screen.mockInput.pressKey('n'))
    await vi.waitFor(async () => {
      await screen.flush()
      expect(rpc).toHaveBeenCalledWith('workspace.list', { after: 'two' })
      expect(screen.captureCharFrame()).toContain('Third task')
    })
    act(() => screen.mockInput.pressKey('p'))
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Page 1')
    act(() => screen.mockInput.pressEscape())
    await vi.waitFor(() => expect(getOverlayState().workspaces).toBe(false))
  } finally { old.resolve({}); act(() => screen.renderer.destroy()) }
})
it('shows inspection errors and recovers on refresh without retaining the old diff', async () => {
  let failed = true
  const rpc = vi.fn(async (method: string) => method === 'workspace.list' ? { ok: true, inventory: { records: [row] } } : failed ? { ok: false, error: 'Workspace identity changed' } : { ok: true, review: detail })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><WorkspaceOverlay t={DARK_THEME} /></GatewayProvider>, { width: 160, height: 40 })
  try {
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Workspace identity changed')
    failed = false
    act(() => screen.mockInput.pressKey('r'))
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).not.toContain('Workspace identity changed')
    expect(screen.captureCharFrame()).toContain('new code')
  } finally { act(() => screen.renderer.destroy()) }
})

it('lets users scroll horizontally to inspect the end of a long changed line', async () => {
  const long = { ...detail, diff: 'diff --git a/file.ts b/file.ts\n--- a/file.ts\n+++ b/file.ts\n@@ -1 +1 @@\n-old\n+' + 'x'.repeat(220) + 'TAIL_VISIBLE\n' }
  const rpc = vi.fn(async (method: string) => method === 'workspace.list' ? { ok: true, inventory: { records: [row] } } : { ok: true, review: long })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><WorkspaceOverlay t={DARK_THEME} /></GatewayProvider>, { width: 110, height: 35 })
  try {
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).not.toContain('TAIL_VISIBLE')
    for (let i = 0; i < 20; i++) act(() => screen.mockInput.pressArrow('right'))
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('TAIL_VISIBLE')
  } finally { act(() => screen.renderer.destroy()) }
})

it('checks only the reviewed content and discards the check after selecting another workspace', async () => {
  const check = Promise.withResolvers<unknown>()
  const reviewId = 'a'.repeat(64)
  const rpc = vi.fn(async (method: string, params: Record<string, unknown>) => {
    if (method === 'workspace.list') return { ok: true, inventory: { records: [row, { ...row, id: 'two' }] } }
    if (method === 'workspace.checkApply') return check.promise
    return { ok: true, review: { ...detail, id: params.workspace_id, reviewId } }
  })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><WorkspaceOverlay t={DARK_THEME} /></GatewayProvider>, { width: 160, height: 40 })
  try {
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('c'))
    await screen.flush()
    expect(rpc).toHaveBeenCalledWith('workspace.checkApply', { workspace_id: 'one', review_id: reviewId })
    act(() => screen.mockInput.pressArrow('down'))
    await screen.flush(); await screen.flush()
    act(() => check.resolve({ ok: true, check: { reviewId, canApply: true, destination: '/OLD_DESTINATION' } }))
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).not.toContain('OLD_DESTINATION')
    expect(rpc).not.toHaveBeenCalledWith('workspace.apply', expect.anything())
  } finally { check.resolve({}); act(() => screen.renderer.destroy()) }
})

it.each([40, 160])('requires check and cancellable confirmation before applying at width %i', async width => {
  const pending = Promise.withResolvers<unknown>()
  const reviewId = 'a'.repeat(64), destinationState = 'b'.repeat(64)
  const rpc = vi.fn(async (method: string) => {
    if (method === 'workspace.list') return { ok: true, inventory: { records: [row] } }
    if (method === 'workspace.checkApply') return { ok: true, check: { reviewId, destinationState, destination: '/parent', canApply: true } }
    if (method === 'workspace.apply') return pending.promise
    return { ok: true, review: { ...detail, reviewId } }
  })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><WorkspaceOverlay t={DARK_THEME} /></GatewayProvider>, { width, height: 30 })
  try {
    await screen.flush(); await screen.flush()
    act(() => { screen.mockInput.pressKey('a'); screen.mockInput.pressKey('y') })
    expect(rpc).not.toHaveBeenCalledWith('workspace.apply', expect.anything())
    act(() => screen.mockInput.pressKey('c'))
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('a'))
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('Apply to /parent?')
    expect(screen.captureCharFrame()).toContain('Y confirm')
    act(() => screen.mockInput.pressEscape())
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).not.toContain('Y confirm') })
    expect(rpc).not.toHaveBeenCalledWith('workspace.apply', expect.anything())
    act(() => screen.mockInput.pressKey('a'))
    await screen.flush()
    act(() => { screen.mockInput.pressKey('y'); screen.mockInput.pressKey('y') })
    await screen.flush()
    expect(rpc.mock.calls.filter(([method]) => method === 'workspace.apply')).toHaveLength(1)
    expect(rpc).toHaveBeenCalledWith('workspace.apply', { workspace_id: 'one', review_id: reviewId, destination_state: destinationState, confirm: true })
    await act(async () => { pending.resolve({ ok: false, error: 'Destination changed; check again' }); await pending.promise })
    await screen.flush(); await screen.flush()
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Destination changed') })
    act(() => screen.mockInput.pressKey('a'))
    await screen.flush()
    expect(screen.captureCharFrame()).not.toContain('Y confirm')
  } finally { pending.resolve({}); act(() => screen.renderer.destroy()) }
})

it.each([[40, 'prepared'], [160, 'prepared'], [40, 'preparing'], [160, 'preparing']] as const)('recovers an interrupted integration at width %i (%s)', async (width, status) => {
  const pending = Promise.withResolvers<unknown>()
  let restored = false
  const rpc = vi.fn(async (method: string) => {
    if (method === 'workspace.list') return { ok: true, inventory: { records: [row] } }
    if (method === 'workspace.integrations') return { ok: true, inventory: { records: [{ id: 'apply-one', status: restored ? (status === 'preparing' ? 'abandoned' : 'rolled-back') : status, backupPath: '/backups/one', destination: '/parent', paths: status === 'preparing' ? [] : ['file.ts'] }] } }
    if (method === 'workspace.recover') return pending.promise
    return { ok: true, review: detail }
  })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><WorkspaceOverlay t={DARK_THEME} /></GatewayProvider>, { width, height: 30 })
  try {
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('i'))
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Integration recovery')
    act(() => screen.mockInput.pressKey('b'))
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('Y confirm')
    act(() => screen.mockInput.pressEscape())
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).not.toContain('Y confirm') })
    expect(rpc).not.toHaveBeenCalledWith('workspace.recover', expect.anything())
    act(() => screen.mockInput.pressKey('b'))
    await screen.flush()
    act(() => { screen.mockInput.pressKey('y'); screen.mockInput.pressKey('y') })
    await screen.flush()
    expect(rpc.mock.calls.filter(([method]) => method === 'workspace.recover')).toHaveLength(1)
    expect(rpc).toHaveBeenCalledWith('workspace.recover', { integration_id: 'apply-one', confirm: true })
    restored = true
    await act(async () => { pending.resolve({ ok: true, recovery: { id: 'apply-one', status: status === 'preparing' ? 'abandoned' : 'rolled-back', conflicts: [] } }); await pending.promise })
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain(status === 'preparing' ? 'abandoned.' : 'Original files restored.') })
    expect(screen.captureCharFrame()).not.toContain('B · Restore')
    act(() => screen.mockInput.pressEscape())
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Agent workspaces') })
  } finally { pending.resolve({}); act(() => screen.renderer.destroy()) }
})

it('shows per-file recovery actions and ignores an old selection inspection', async () => {
  const old = Promise.withResolvers<unknown>()
  const rpc = vi.fn(async (method: string, params: Record<string, unknown>) => {
    if (method === 'workspace.list') return { ok: true, inventory: { records: [row] } }
    if (method === 'workspace.integrations') return { ok: true, inventory: { records: ['one', 'two'].map(id => ({ id, status: 'prepared', backupPath: '/backup/' + id, destination: '/parent', paths: ['file.ts'] })) } }
    if (method === 'workspace.integration.inspect') return params.integration_id === 'one' ? old.promise : { ok: true, inspection: { id: 'two', files: [{ path: 'newer.ts', action: 'conflict', reason: 'Contains newer content' }] } }
    return { ok: true, review: detail }
  })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><WorkspaceOverlay t={DARK_THEME} /></GatewayProvider>, { width: 160, height: 40 })
  try {
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('i'))
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressArrow('down'))
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('conflict · newer.ts')
    await act(async () => { old.resolve({ ok: true, inspection: { id: 'one', files: [{ path: 'OLD_FILE.ts', action: 'restore', reason: 'Matches prepared result' }] } }); await old.promise })
    await screen.flush()
    expect(screen.captureCharFrame()).not.toContain('OLD_FILE')
    expect(rpc).not.toHaveBeenCalledWith('workspace.recover', expect.anything())
  } finally { old.resolve({}); act(() => screen.renderer.destroy()) }
})
