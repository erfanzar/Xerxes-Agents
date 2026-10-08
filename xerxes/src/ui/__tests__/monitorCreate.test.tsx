// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { MonitorCreate } from '../opentui/monitorCreate.js'
import { DARK_THEME } from '../theme.js'

it.each([[150, 40], [40, 18]])('creates a notification watch from the form at %ix%i', async (width, height) => {
  const created = vi.fn()
  const rpc = vi.fn(async (method: string) => method === 'terminal.list'
    ? { ok: true, terminals: [{ id: 'build', label: 'Build server', running: true, kind: 'background' }] }
    : { ok: true, monitor: { id: 'new', terminalId: 'build', state: 'watching', match: 'error', expiresAt: 1000 } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorCreate t={DARK_THEME} onCreated={created} onClose={() => {}} /></GatewayProvider>, { width, height })
  try {
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('Build server')
    act(() => screen.mockInput.pressKey('TAB'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('TAB'))
    await screen.flush()
    await act(async () => screen.mockInput.typeText('error'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN'))
    await screen.flush()
    await vi.waitFor(() => expect(created).toHaveBeenCalledWith('new'))
    expect(rpc).toHaveBeenCalledWith('monitor.create', { terminal_id: 'build', trigger: 'output', match: 'error', duration_seconds: 3600, react: false, max_reactions: 3 })
  } finally { act(() => screen.renderer.destroy()) }
})
it('does not send a create request without a live source', async () => {
  const rpc = vi.fn(async () => ({ ok: true, terminals: [] }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorCreate t={DARK_THEME} onCreated={() => {}} onClose={() => {}} /></GatewayProvider>, { width: 80, height: 24 })
  try {
    await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN'))
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('Start a terminal command first')
    expect(rpc).toHaveBeenCalledTimes(1)
  } finally { act(() => screen.renderer.destroy()) }
})

it('preserves settings after a rejected reaction request and permits retry', async () => {
  const created = vi.fn()
  let requests = 0
  const rpc = vi.fn(async (method: string) => {
    if (method === 'terminal.list') return { ok: true, terminals: [{ id: 'build', label: 'Build server', running: true, kind: 'background' }] }
    if (++requests === 1) return { ok: false, error: 'Reaction host unavailable' }
    return { ok: true, monitor: { id: 'new', terminalId: 'build', state: 'watching', match: 'error', expiresAt: 1000 } }
  })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorCreate t={DARK_THEME} onCreated={created} onClose={() => {}} /></GatewayProvider>, { width: 110, height: 35 })
  try {
    await screen.flush()
    act(() => screen.mockInput.pressKey('TAB'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('TAB'))
    await screen.flush()
    await act(async () => screen.mockInput.typeText('error'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('TAB'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('TAB'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('ARROW_RIGHT'))
    await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN'))
    await screen.flush()
    await vi.waitFor(() => expect(screen.captureCharFrame()).toContain('Reaction host unavailable'))
    expect(created).not.toHaveBeenCalled()
    expect(screen.captureCharFrame()).toContain('Match: error')
    expect(screen.captureCharFrame()).toContain('Automatic reactions: Enabled')
    act(() => screen.mockInput.pressKey('RETURN'))
    await screen.flush()
    await vi.waitFor(() => expect(created).toHaveBeenCalledWith('new'))
    expect(rpc).toHaveBeenLastCalledWith('monitor.create', { terminal_id: 'build', trigger: 'output', match: 'error', duration_seconds: 3600, react: true, max_reactions: 3 })
  } finally { act(() => screen.renderer.destroy()) }
})

it('creates a completion watch without match text for a finished command', async () => {
  const created = vi.fn()
  const rpc = vi.fn(async (method: string) => method === 'terminal.list'
    ? { ok: true, terminals: [{ id: 'done', label: 'Finished build', running: false, kind: 'background' }] }
    : { ok: true, monitor: { id: 'watch', terminalId: 'done', state: 'source-ended', match: 'Command completion' } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorCreate t={DARK_THEME} onCreated={created} onClose={() => {}} /></GatewayProvider>, { width: 110, height: 35 })
  try {
    await screen.flush()
    for (let i = 0; i < 7; i++) { act(() => screen.mockInput.pressKey('TAB')); await screen.flush() }
    act(() => screen.mockInput.pressKey('ARROW_RIGHT')); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Trigger: Command completion')
    act(() => screen.mockInput.pressKey('RETURN')); await screen.flush()
    await vi.waitFor(() => expect(created).toHaveBeenCalledWith('watch'))
    expect(rpc).toHaveBeenCalledWith('monitor.create', expect.objectContaining({ terminal_id: 'done', trigger: 'completion', match: '' }))
  } finally { act(() => screen.renderer.destroy()) }
})

it.each([[220, 65], [40, 18]])('configures a reaction token threshold at %ix%i', async (width, height) => {
  const rpc = vi.fn(async (method: string) => method === 'terminal.list'
    ? { ok: true, terminals: [{ id: 'build', label: 'Build', running: true, kind: 'background' }] }
    : { ok: true, monitor: { id: 'watch', terminalId: 'build', state: 'watching', match: 'error' } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorCreate t={DARK_THEME} onCreated={() => {}} onClose={() => {}} /></GatewayProvider>, { width, height })
  try {
    await screen.flush()
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush()
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush()
    await act(async () => screen.mockInput.typeText('error')); await screen.flush()
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush()
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush()
    act(() => screen.mockInput.pressKey('ARROW_RIGHT')); await screen.flush()
    for (let i = 0; i < 4; i++) { act(() => screen.mockInput.pressKey('TAB')); await screen.flush() }
    await act(async () => screen.mockInput.typeText('2000')); await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN')); await screen.flush()
    await vi.waitFor(() => expect(rpc).toHaveBeenCalledWith('monitor.create', expect.objectContaining({ react: true, max_total_tokens: 2000 })))
  } finally { act(() => screen.renderer.destroy()) }
})

it('creates a file-change monitor without terminal-only fields', async () => {
  const created = vi.fn()
  const rpc = vi.fn(async (method: string) => method === 'terminal.list'
    ? { ok: true, terminals: [] }
    : { ok: true, monitor: { id: 'file-watch', terminalId: '', source: { kind: 'file', path: 'src/app.ts', workspace: '/repo' }, trigger: 'change', match: '', state: 'watching', expiresAt: 1000 } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorCreate t={DARK_THEME} onCreated={created} onClose={() => {}} /></GatewayProvider>, { width: 110, height: 35 })
  try {
    await screen.flush()
    act(() => screen.mockInput.pressKey('ARROW_RIGHT')); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Source: File changes')
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush()
    await act(async () => screen.mockInput.typeText('src/app.ts')); await screen.flush()
    expect(screen.captureCharFrame()).toContain('File path: src/app.ts')
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN')); await screen.flush()
    await vi.waitFor(() => expect(created).toHaveBeenCalledWith('file-watch'))
    expect(rpc).toHaveBeenCalledWith('monitor.create', { source_kind: 'file', file_path: 'src/app.ts', trigger: 'change', duration_seconds: 3600, react: false, max_reactions: 3 })
  } finally { act(() => screen.renderer.destroy()) }
})

it('creates a websocket monitor with text matching and a token threshold', async () => {
  const created = vi.fn()
  const rpc = vi.fn(async (method: string) => method === 'terminal.list'
    ? { ok: true, terminals: [] }
    : { ok: true, monitor: { id: 'socket-watch', terminalId: '', source: { kind: 'websocket', url: 'wss://events.example/ws' }, trigger: 'output', match: 'failure', state: 'watching', expiresAt: 1000 } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorCreate t={DARK_THEME} onCreated={created} onClose={() => {}} /></GatewayProvider>, { width: 120, height: 40 })
  try {
    await screen.flush()
    act(() => screen.mockInput.pressKey('ARROW_RIGHT')); await screen.flush()
    act(() => screen.mockInput.pressKey('ARROW_RIGHT')); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Source: Websocket server push')
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush()
    await act(async () => screen.mockInput.typeText('wss://events.example/ws')); await screen.flush()
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush()
    await act(async () => screen.mockInput.typeText('failure')); await screen.flush()
    for (let i = 0; i < 2; i++) { act(() => screen.mockInput.pressKey('TAB')); await screen.flush() }
    act(() => screen.mockInput.pressKey('ARROW_RIGHT')); await screen.flush()
    for (let i = 0; i < 3; i++) { act(() => screen.mockInput.pressKey('TAB')); await screen.flush() }
    await act(async () => screen.mockInput.typeText('2000')); await screen.flush()
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN')); await screen.flush()
    await vi.waitFor(() => expect(created).toHaveBeenCalledWith('socket-watch'))
    expect(rpc).toHaveBeenCalledWith('monitor.create', { source_kind: 'websocket', websocket_url: 'wss://events.example/ws', trigger: 'output', match: 'failure', duration_seconds: 3600, react: true, max_reactions: 3, max_total_tokens: 2000 })
  } finally { act(() => screen.renderer.destroy()) }
})

it('keeps websocket match required after switching from terminal completion and cycles source backwards', async () => {
  const rpc = vi.fn(async (method: string) => method === 'terminal.list'
    ? { ok: true, terminals: [{ id: 'done', label: 'Finished build', running: false, kind: 'background' }] }
    : { ok: true, monitor: { id: 'socket-watch', terminalId: '', source: { kind: 'websocket', url: 'wss://events.example/ws' }, trigger: 'output', match: 'failure', state: 'watching', expiresAt: 1000 } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorCreate t={DARK_THEME} onCreated={() => {}} onClose={() => {}} /></GatewayProvider>, { width: 120, height: 40 })
  try {
    await screen.flush()
    for (let i = 0; i < 7; i++) { act(() => screen.mockInput.pressKey('TAB')); await screen.flush() }
    act(() => screen.mockInput.pressKey('ARROW_RIGHT')); await screen.flush()
    for (let i = 0; i < 2; i++) { act(() => screen.mockInput.pressKey('TAB')); await screen.flush() }
    act(() => screen.mockInput.pressKey('ARROW_RIGHT')); await screen.flush()
    act(() => screen.mockInput.pressKey('ARROW_RIGHT')); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Source: Websocket server push')
    expect(screen.captureCharFrame()).toContain('Match: (required)')
    act(() => screen.mockInput.pressKey('ARROW_LEFT')); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Source: File changes')
    act(() => screen.mockInput.pressKey('ARROW_LEFT')); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Source: Terminal output')
  } finally { act(() => screen.renderer.destroy()) }
})

it('discovers a configured webhook and submits its name without secrets', async () => {
  const created = vi.fn()
  const rpc = vi.fn(async (method: string) => method === 'terminal.list'
    ? { ok: true, terminals: [] }
    : method === 'monitor.sources'
      ? { ok: true, webhooks: [{ name: 'deploy-events' }] }
      : { ok: true, monitor: { id: 'webhook-watch', terminalId: '', source: { kind: 'webhook', name: 'deploy-events' }, trigger: 'output', match: 'failure', state: 'watching', expiresAt: 1000 } })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorCreate t={DARK_THEME} onCreated={created} onClose={() => {}} /></GatewayProvider>, { width: 120, height: 40 })
  try {
    await screen.flush()
    for (let i = 0; i < 3; i++) { act(() => screen.mockInput.pressKey('ARROW_RIGHT')); await screen.flush() }
    await vi.waitFor(() => expect(screen.captureCharFrame()).toContain('Webhook name: deploy-events'))
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush()
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush()
    await act(async () => screen.mockInput.typeText('failure')); await screen.flush()
    act(() => screen.mockInput.pressKey('RETURN')); await screen.flush()
    await vi.waitFor(() => expect(created).toHaveBeenCalledWith('webhook-watch'))
    expect(rpc).toHaveBeenCalledWith('monitor.create', { source_kind: 'webhook', webhook_name: 'deploy-events', trigger: 'output', match: 'failure', duration_seconds: 3600, react: false, max_reactions: 3 })
  } finally { act(() => screen.renderer.destroy()) }
})

it('reports missing webhook configuration without sending a create request', async () => {
  const rpc = vi.fn(async (method: string) => method === 'terminal.list'
    ? { ok: true, terminals: [] }
    : method === 'monitor.sources' ? { ok: true, webhooks: [] } : { ok: true })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorCreate t={DARK_THEME} onCreated={() => {}} onClose={() => {}} /></GatewayProvider>, { width: 110, height: 35 })
  try {
    await screen.flush()
    for (let i = 0; i < 3; i++) { act(() => screen.mockInput.pressKey('ARROW_RIGHT')); await screen.flush() }
    await vi.waitFor(() => expect(screen.captureCharFrame()).toContain('(no configured webhooks)'))
    act(() => screen.mockInput.pressKey('RETURN')); await screen.flush()
    expect(screen.captureCharFrame()).toContain('No configured webhook sources are available')
    expect(rpc).not.toHaveBeenCalledWith('monitor.create', expect.anything())
  } finally { act(() => screen.renderer.destroy()) }
})

it('keeps the webhook form usable after source discovery fails', async () => {
  const rpc = vi.fn(async (method: string) => method === 'terminal.list'
    ? { ok: true, terminals: [] }
    : method === 'monitor.sources' ? { ok: false, error: 'Webhook host unavailable' } : { ok: true })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorCreate t={DARK_THEME} onCreated={() => {}} onClose={() => {}} /></GatewayProvider>, { width: 110, height: 35 })
  try {
    await screen.flush()
    for (let i = 0; i < 3; i++) { act(() => screen.mockInput.pressKey('ARROW_RIGHT')); await screen.flush() }
    await vi.waitFor(() => expect(screen.captureCharFrame()).toContain('Webhook host unavailable'))
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush()
    act(() => screen.mockInput.pressKey('TAB')); await screen.flush()
    await act(async () => screen.mockInput.typeText('failure')); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Match: failure')
    expect(screen.captureCharFrame()).toContain('Webhook host unavailable')
  } finally { act(() => screen.renderer.destroy()) }
})

it('ignores a late webhook discovery response after leaving the source form', async () => {
  let resolveSources!: (value: unknown) => void
  const sources = new Promise(resolve => { resolveSources = resolve })
  const rpc = vi.fn((method: string) => method === 'terminal.list'
    ? Promise.resolve({ ok: true, terminals: [{ id: 'build', label: 'Build', running: true, kind: 'background' }] })
    : method === 'monitor.sources' ? sources : Promise.resolve({ ok: true }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MonitorCreate t={DARK_THEME} onCreated={() => {}} onClose={() => {}} /></GatewayProvider>, { width: 110, height: 35 })
  try {
    await screen.flush()
    for (let i = 0; i < 3; i++) { act(() => screen.mockInput.pressKey('ARROW_RIGHT')); await screen.flush() }
    act(() => screen.mockInput.pressKey('ARROW_LEFT')); await screen.flush()
    await act(async () => { resolveSources({ ok: true, webhooks: [{ name: 'late-secret-looking-name' }] }); await sources })
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('Source: Websocket server push')
    expect(screen.captureCharFrame()).not.toContain('late-secret-looking-name')
  } finally { act(() => screen.renderer.destroy()) }
})
