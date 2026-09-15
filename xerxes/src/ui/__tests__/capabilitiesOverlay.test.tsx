// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { afterEach, expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { getOverlayState, patchOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { CapabilitiesOverlay, toolGroup } from '../opentui/capabilitiesOverlay.js'
import { DARK_THEME } from '../theme.js'
import { scheduleDescription } from '../lib/scheduleTemplates.js'
afterEach(() => resetOverlayState())
it.each([[140, 42], [40, 18]])('searches and previews capabilities at %ix%i without activating a skill', async (width, height) => {
  const rpc = vi.fn(async (method: string) => method === 'capabilities.list' ? { ok: true, skills: [{ name: 'review', description: 'Review code', uses: 3 }, { name: 'testing', description: 'Test code', uses: 0 }], tools: [] } : { ok: true, instructions: 'Review instructions here' })
  patchOverlayState({ capabilities: true })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><CapabilitiesOverlay t={DARK_THEME} /></GatewayProvider>, { width, height })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('review') })
    expect(rpc).toHaveBeenCalledWith('capabilities.inspect', { name: 'review' })
    expect(screen.captureCharFrame()).toContain('Esc close')
    if (width > 100) await vi.waitFor(async () => {
      await screen.flush()
      expect(screen.captureCharFrame()).toContain('Review instructions here')
    })
    await act(async () => screen.mockInput.typeText('testing'))
    await vi.waitFor(async () => { await screen.flush(); expect(rpc).toHaveBeenCalledWith('capabilities.inspect', { name: 'testing' }) })
    act(() => screen.mockInput.pressKey('ESCAPE'))
    await vi.waitFor(() => expect(getOverlayState().capabilities).toBe(false))
    expect(rpc.mock.calls.every(([method]) => method.startsWith('capabilities.'))).toBe(true)
  } finally { act(() => screen.renderer.destroy()) }
})
it('shows retry on inventory failure and opens controls with the keyboard', async () => {
  const rpc = vi.fn(async () => ({ ok: false, error: 'Daemon unavailable' }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><CapabilitiesOverlay t={DARK_THEME} /></GatewayProvider>, { width: 120, height: 35 })
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Ctrl+R retry') })
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush()
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush()
    await act(async () => screen.mockInput.typeText('Models'))
    await screen.flush(); act(() => screen.mockInput.pressKey('RETURN'))
    expect(getOverlayState().modelPicker).toBe(true)
    expect(getOverlayState().capabilities).toBe(false)
  } finally { act(() => screen.renderer.destroy()) }
})
it('describes common schedules while preserving complex cron expressions', () => {
  expect(scheduleDescription('0 9 * * 1-5', 'Europe/Istanbul')).toBe('Weekdays at 09:00 (Europe/Istanbul)')
  expect(scheduleDescription('0 16 * * 5', 'UTC')).toBe('Every Friday at 16:00 (UTC)')
  expect(scheduleDescription('0 9 1 * *', 'UTC')).toBe('Custom cron: 0 9 1 * * (UTC)')
  expect(scheduleDescription('', '', 0)).toBe('Enter a positive interval')
  expect(toolGroup('ExecCommand')).toBe('Terminal & processes')
})
