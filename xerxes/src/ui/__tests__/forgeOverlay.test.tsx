// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { ForgeOverlay } from '../opentui/forgeOverlay.js'
import { DARK_THEME } from '../theme.js'
import { forgePackage } from '../lib/forge.js'
import { findSlashCommand } from '../app/slash/registry.js'
import { getOverlayState, resetFlowOverlays, resetOverlayState } from '../app/overlayStore.js'

const pkg = { name: 'report', version: '1.0.0', description: 'Reusable report', created_at: '2026-09-15', template: 'Report: {{topic}}', parameters: [{ name: 'topic', description: 'Report topic', required: true, default: 'Build' }] }
it('discovers Forge as a local panel and retains it across turn completion', () => {
  resetOverlayState()
  const command = findSlashCommand('forge')!
  expect(command.group).toBe('tools')
  command.run('', {} as never, 'forge')
  resetFlowOverlays()
  expect(getOverlayState().forge).toBe(true)
  resetOverlayState()
  expect(() => forgePackage({ ...pkg, parameters: [{}] })).toThrow('Invalid Forge parameter')
})
it.each([[140, 40], [80, 28], [40, 18]])('runs declared inputs and confirms removal at %ix%i', async (width, height) => {
  const rpc = vi.fn(async (method: string) => method === 'forge.list' ? { ok: true, packages: [pkg] }
    : method === 'forge.run' ? { ok: true, output: 'Report: Build' }
    : { ok: true, package: pkg })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ForgeOverlay t={DARK_THEME} sessionId={`forge-${width}`} onClose={() => {}} /></GatewayProvider>, { width, height })
  const press = async (key: string, ctrl = false) => { await act(async () => screen.mockInput.pressKey(key, { ctrl })); await screen.flush() }
  const see = async (text: string) => { await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain(text) }) }
  try {
    await see('Reusable report')
    await press('RETURN')
    await see('topic')
    await press('s', true)
    await see('Report: Build')
    expect(rpc).toHaveBeenCalledWith('forge.run', { name: 'report', version: '1.0.0', input: { topic: 'Build' } })
    await press('ESCAPE'); await see('Forge packages')
    await press('d'); await see('Remove package?')
    await press('n'); await see('Forge packages')
    expect(rpc.mock.calls.some(([method]) => method === 'forge.undefine')).toBe(false)
    await press('d'); await see('Remove package?')
    await press('y')
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('forge.undefine', { name: 'report', version: '1.0.0', confirm: true }))
  } finally { await act(async () => screen.renderer.destroy()) }
})
it('keeps a rejected definition for review and requires explicit save confirmation', async () => {
  const rpc = vi.fn(async (method: string) => method === 'forge.list' ? { ok: true, packages: [] } : { ok: false, error: 'Version already exists' })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><ForgeOverlay t={DARK_THEME} sessionId="new-forge" onClose={() => {}} /></GatewayProvider>, { width: 100, height: 32 })
  const press = async (key: string, ctrl = false) => { await act(async () => screen.mockInput.pressKey(key, { ctrl })); await screen.flush() }
  const type = async (text: string) => { await act(async () => screen.mockInput.typeText(text)); await screen.flush() }
  const see = async (text: string) => { await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain(text) }) }
  try {
    await see('No Forge packages.')
    await press('n'); await see('Package name')
    await type('release-note'); await press('TAB'); await see('Version')
    await press('TAB'); await see('Description'); await type('Release summary')
    await press('TAB'); await see('Template'); await type('Hello from template')
    await press('s', true); await see('Review definition')
    expect(rpc.mock.calls.some(([method]) => method === 'forge.define')).toBe(false)
    await press('s', true); await see('Version already exists')
    expect(screen.captureCharFrame()).toContain('Hello from template')
    expect(rpc).toHaveBeenCalledWith('forge.define', expect.objectContaining({ name: 'release-note', version: '1.0.0', description: 'Release summary', template: 'Hello from template', parameters: [], confirm: true }))
    await press('ESCAPE'); await see('Forge packages')
    await press('n'); await see('release-note')
  } finally { await act(async () => screen.renderer.destroy()) }
})
