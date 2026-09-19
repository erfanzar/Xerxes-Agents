// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { getOverlayState, resetFlowOverlays, resetOverlayState } from '../app/overlayStore.js'
import { opsCommands } from '../app/slash/commands/ops.js'
import type { SlashRunCtx } from '../app/slash/types.js'
import { SnapshotOverlay } from '../opentui/snapshotOverlay.js'
import { DARK_THEME } from '../theme.js'
const row = { id: 'one', label: 'Before runtime edit', created_at: '2026-09-05', turn_index: 3 }
const preview = { ok: true, snapshot_id: 'one', revision: 'a'.repeat(64), diff: 'diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-current\n+restored\n', truncated: false }
afterEach(() => resetOverlayState())
it.each([[220,65], [100,30], [40,18]])('renders snapshot timeline and guarded confirmation at %ix%i', async (width, height) => {
  const rpc = vi.fn(async (method: string) => method === 'snapshot.list' ? { ok: true, snapshots: [row] } : method === 'snapshot.preview' ? preview : { ok: true })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><SnapshotOverlay t={DARK_THEME} /></GatewayProvider>, { width, height })
  try {
    // Listing and preview are separate asynchronous effects. Flushing twice
    // does not guarantee their commits under the full-suite renderer load.
    await vi.waitFor(async () => {
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('Snapshot timeline')
      expect(screen.captureCharFrame()).toContain('Before turn 3')
      expect(screen.captureCharFrame()).toContain('restored')
    }, { timeout: 2000, interval: 20 })
    act(() => screen.mockInput.pressKey('a')); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Y restore')
    expect(rpc.mock.calls.some(([method]) => method === 'slash.exec')).toBe(false)
    act(() => screen.mockInput.pressEscape()); await new Promise(resolve => setTimeout(resolve, 60)); await screen.flush()
    expect(rpc.mock.calls.some(([method]) => method === 'slash.exec')).toBe(false)
    act(() => screen.mockInput.pressKey('a')); await screen.flush()
    act(() => screen.mockInput.pressKey('y')); await screen.flush(); await screen.flush()
    expect(rpc).toHaveBeenCalledWith('slash.exec', { command: `rollback apply one ${preview.revision}` })
    expect(screen.captureCharFrame()).toContain('Files restored')
  } finally { act(() => screen.renderer.destroy()) }
})
it('opens the timeline without erasing chat and keeps it across turn completion', () => {
  opsCommands.find(command => command.name === 'snapshots')!.run('', {} as SlashRunCtx, 'snapshots')
  expect(getOverlayState().snapshots).toBe(true)
  resetFlowOverlays()
  expect(getOverlayState().snapshots).toBe(true)
})
it('ignores a stale preview and requires refresh after restore rejection', async () => {
  const pending = Promise.withResolvers<unknown>()
  const rpc = vi.fn(async (method: string, params: Record<string, unknown>) => {
    if (method === 'snapshot.list') return { ok: true, snapshots: [{ ...row, id: 'two' }, row] }
    if (method === 'snapshot.preview') return params.snapshot_id === 'one' ? pending.promise : { ...preview, snapshot_id: 'two', diff: '+CURRENT' }
    return { ok: false, error: 'Snapshot preview is stale' }
  })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><SnapshotOverlay t={DARK_THEME} /></GatewayProvider>, { width: 220, height: 65 })
  try {
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressArrow('down')); await screen.flush(); await screen.flush()
    pending.resolve({ ...preview, diff: '+STALE' }); await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).not.toContain('STALE')
    act(() => screen.mockInput.pressKey('a')); await screen.flush()
    act(() => screen.mockInput.pressKey('y')); await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Snapshot preview is stale')
    act(() => screen.mockInput.pressKey('a')); await screen.flush()
    expect(screen.captureCharFrame()).not.toContain('Y restore')
  } finally { pending.resolve(preview); act(() => screen.renderer.destroy()) }
})
it('keeps an empty or failed timeline usable and prevents duplicate restores', async () => {
  let listFailed = true
  const completion = Promise.withResolvers<unknown>()
  const rpc = vi.fn(async (method: string) => {
    if (method === 'snapshot.list') return listFailed ? { ok: false, error: 'Capture store unavailable' } : { ok: true, snapshots: [row] }
    if (method === 'snapshot.preview') return preview
    return completion.promise
  })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><SnapshotOverlay t={DARK_THEME} /></GatewayProvider>, { width: 100, height: 30 })
  try {
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Capture store unavailable')
    listFailed = false
    act(() => screen.mockInput.pressKey('r')); await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('a')); await screen.flush()
    act(() => { screen.mockInput.pressKey('y'); screen.mockInput.pressKey('y') }); await screen.flush()
    expect(rpc.mock.calls.filter(([method]) => method === 'slash.exec')).toHaveLength(1)
    expect(screen.captureCharFrame()).toContain('Restoring')
    completion.resolve({ ok: true }); await screen.flush(); await screen.flush()
  } finally { completion.resolve({ ok: true }); act(() => screen.renderer.destroy()) }
})
it.each([[220,65], [40,18]])('selects a file and confirms removal explicitly at %ix%i', async (width, height) => {
  const rpc = vi.fn(async (method: string, params: Record<string, unknown>) => {
    if (method === 'snapshot.list') return { ok: true, snapshots: [row] }
    if (method === 'snapshot.preview') return { ...preview, files: ['new file.txt'], ...(params.path ? { action: 'remove' } : {}) }
    return { ok: true }
  })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><SnapshotOverlay t={DARK_THEME} /></GatewayProvider>, { width, height })
  try {
    await screen.flush(); await screen.flush()
    act(() => screen.mockInput.pressKey('f')); await screen.flush(); await screen.flush()
    expect(rpc).toHaveBeenCalledWith('snapshot.preview', { snapshot_id: 'one', path: 'new file.txt' })
    expect(screen.captureCharFrame()).toContain('Remove: new file.txt')
    act(() => screen.mockInput.pressKey('a')); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Remove new file.txt?')
    act(() => screen.mockInput.pressKey('y')); await screen.flush(); await screen.flush()
    expect(rpc).toHaveBeenCalledWith('snapshot.restoreFile', { snapshot_id: 'one', path: 'new file.txt', revision: preview.revision })
    expect(rpc.mock.calls.some(([method]) => method === 'slash.exec')).toBe(false)
  } finally { act(() => screen.renderer.destroy()) }
})
it('shows unfinished restore recovery and opens its backup without restoring', async () => {
  const rpc = vi.fn(async (method: string, params: Record<string, unknown>) => method === 'snapshot.list'
    ? { ok: true, snapshots: [{ ...row, id: 'backup', label: 'Recovery backup' }, row], restore_attempts: [{ backupId: 'backup', phase: 'failed' }] }
    : { ...preview, snapshot_id: params.snapshot_id })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><SnapshotOverlay t={DARK_THEME} /></GatewayProvider>, { width: 220, height: 65 })
  try {
    await screen.flush(); await screen.flush()
    expect(screen.captureCharFrame()).toContain('unfinished restore')
    act(() => screen.mockInput.pressKey('b')); await screen.flush(); await screen.flush()
    expect(rpc).toHaveBeenCalledWith('snapshot.preview', { snapshot_id: 'backup' })
    expect(rpc.mock.calls.some(([method]) => method === 'slash.exec' || method === 'snapshot.restoreFile')).toBe(false)
  } finally { act(() => screen.renderer.destroy()) }
})
