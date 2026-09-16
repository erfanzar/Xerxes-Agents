// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { PresetEditor } from '../opentui/presetEditor.js'
import { DARK_THEME } from '../theme.js'
import { findSlashCommand } from '../app/slash/registry.js'
import { getOverlayState, resetFlowOverlays, resetOverlayState } from '../app/overlayStore.js'

const preset = { id: 'my-agent', name: 'My agent', description: 'Research and coding', manageable: true }
const original = 'version: 1\nagent:\n  name: my-agent\n'
it('exposes composition management without replacing the transcript', () => {
  resetOverlayState()
  findSlashCommand('preset')!.run('manage', {} as never, 'preset')
  expect(getOverlayState().presetEditor).toBe(true)
  resetFlowOverlays()
  expect(getOverlayState().presetEditor).toBe(true)
  resetOverlayState()
})
it.each([[140, 40], [60, 24]])('retains rejected drafts and saves the exact original guard at %ix%i', async (width, height) => {
  let reject = true
  const rpc = vi.fn(async (method: string) => method === 'agentPreset.list' ? { ok: true, presets: [preset] }
    : method === 'agentPreset.read' ? { ok: true, content: original, guarded_write: true }
    : method === 'agentPreset.openDocument' ? { ok: true, opened: false, path: '/home/agents/my-agent' }
    : reject ? { ok: false, error: 'Composition changed on disk' } : { ok: true, preset })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><PresetEditor t={DARK_THEME} sessionId={`preset-${width}`} onClose={() => {}} /></GatewayProvider>, { width, height })
  const press = async (key: string, ctrl = false) => { await act(async () => screen.mockInput.pressKey(key, { ctrl })); await screen.flush() }
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('version: 1') })
    await press('e')
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Edit composition') })
    await act(async () => screen.mockInput.typeText('# retained draft\n'))
    await press('s', true)
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Composition changed') })
    expect(rpc).toHaveBeenCalledWith('agentPreset.write', { agent_preset: preset.id, content: expect.stringContaining('retained draft'), expected_content: original })
    await press('ESCAPE')
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Agent compositions') })
    await press('e')
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('retained draft') })
    reject = false
    await press('s', true)
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Agent compositions') })
    await press('o')
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('/home/agents/my-agent') })
  } finally { await act(async () => screen.renderer.destroy()) }
})
it.each([false, true])('keeps shipped or unguarded compositions read-only (guard %s)', async guarded => {
  const rpc = vi.fn(async (method: string) => method === 'agentPreset.list' ? { ok: true, presets: [{ ...preset, manageable: !guarded }] }
    : { ok: true, content: original, guarded_write: guarded })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><PresetEditor t={DARK_THEME} sessionId="readonly" onClose={() => {}} /></GatewayProvider>, { width: 100, height: 30 })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('version: 1') })
    await act(async () => screen.mockInput.pressKey('e'))
    await screen.flush()
    expect(screen.captureCharFrame()).not.toContain('Edit composition')
    expect(rpc.mock.calls.some(([method]) => method === 'agentPreset.write')).toBe(false)
  } finally { await act(async () => screen.renderer.destroy()) }
})
