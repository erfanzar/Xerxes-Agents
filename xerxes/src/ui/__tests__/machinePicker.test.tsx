// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import { MachinePicker } from '../opentui/machinePicker.js'
import { DEFAULT_THEME } from '../theme.js'

describe('machine picker', () => {
  it.each([[220, 65], [50, 18]])('creates a workspace through the form at %sx%s and retains failed drafts', async (width, height) => {
    let fail = true
    const machine = { alias: 'gpu', target: 'me@host', workspacePath: '/work/my repo' }
    const rpc = vi.fn(async (_method: string, args: { command: string }) => args.command === 'machine list' ? { ok: true, machines: [] } : fail ? { ok: false, error: 'Name already exists' } : { ok: true, machines: [machine] })
    const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MachinePicker t={DEFAULT_THEME} onCancel={() => undefined} /></GatewayProvider>, { width, height })
    const press = async (key: string) => { await act(async () => { screen.mockInput.pressKey(key); await Bun.sleep(key === 'ESCAPE' ? 80 : 0) }); await screen.flush() }
    try {
      await act(async () => { await Bun.sleep(0) }); await screen.flush()
      expect(screen.captureCharFrame()).toContain('Add workspace')
      await press('RETURN')
      for (const [i, value] of ['gpu', 'me@host', '/work/my repo'].entries()) {
        await act(async () => screen.mockInput.typeText(value)); await screen.flush()
        if (i < 2) await press('TAB')
      }
      await press('RETURN')
      expect(rpc).toHaveBeenLastCalledWith('slash.exec', { command: 'machine add "gpu" "me@host" "/work/my repo"' })
      expect(screen.captureCharFrame()).toContain('Name already exists')
      expect(screen.captureCharFrame()).toContain('/work/my repo')
      fail = false
      await press('RETURN')
      expect(screen.captureCharFrame()).toContain('gpu · me@host')
      expect(screen.captureCharFrame()).not.toContain('Name already exists')
    } finally { act(() => screen.renderer.destroy()) }
  })
  it('connects through the saved-machine RPC and keeps failures in the picker', async () => {
    const machine = { alias: 'gpu', target: 'host', workspacePath: '/work/repo' }
    const rpc = vi.fn(async (_method: string, args: { command: string }) => args.command === 'machine list' ? { ok: true, machines: [machine] } : { ok: true, machine })
    const connect = vi.fn(async () => { throw new Error('SSH unavailable') })
    const onCancel = vi.fn()
    const setup = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MachinePicker t={DEFAULT_THEME} onCancel={onCancel} connect={connect} /></GatewayProvider>, { width: 140, height: 40 })
    try {
      await act(async () => { await Bun.sleep(0) })
      await setup.flush()
      expect(setup.captureCharFrame()).toContain('gpu · host')
      await act(async () => { setup.renderer.keyInput.processParsedKey({ name: 'return', raw: '\r', sequence: '\r', ctrl: false, shift: false, meta: false, option: false, eventType: 'press', source: 'raw' }); await Bun.sleep(0) })
      await setup.flush()
      expect(setup.captureCharFrame()).toContain('Remote setup')
      expect(connect).not.toHaveBeenCalled()
      await act(async () => { setup.mockInput.pressKey('RETURN'); await Bun.sleep(0) }); await setup.flush()
      expect(rpc).toHaveBeenCalledWith('slash.exec', { command: 'machine connect gpu' })
      expect(connect).toHaveBeenCalledOnce()
      expect(setup.captureCharFrame()).toContain('SSH unavailable')
      expect(setup.captureCharFrame()).toContain('Enter prepare task')
      await act(async () => { setup.mockInput.pressKey('RETURN'); await Bun.sleep(0) })
      await setup.flush()
      expect(connect).toHaveBeenCalledTimes(2)
      expect(onCancel).not.toHaveBeenCalled()
    } finally { act(() => setup.renderer.destroy()) }
  })
})

it('shows setup guidance and closes with Escape in a narrow terminal', async () => {
  const onCancel = vi.fn()
  const setup = await testRender(<GatewayProvider value={{ rpc: async () => ({ ok: true, machines: [] }) } as unknown as GatewayServices}><MachinePicker t={DEFAULT_THEME} onCancel={onCancel} /></GatewayProvider>, { width: 50, height: 18 })
  try {
    await act(async () => { await Bun.sleep(0) })
    await setup.flush()
    expect(setup.captureCharFrame()).toContain('No remote workspaces')
    act(() => setup.renderer.keyInput.processParsedKey({ name: 'escape', raw: '\u001b', sequence: '\u001b', ctrl: false, shift: false, meta: false, option: false, eventType: 'press', source: 'raw' }))
    expect(onCancel).toHaveBeenCalledOnce()
  } finally { act(() => setup.renderer.destroy()) }
})

it.each([[220, 65], [50, 18]])('chooses SSH hosts and remote folders at %sx%s', async (width, height) => {
  const rpc = vi.fn(async (_method: string, args: { command: string }) => {
    if (args.command === 'machine list') return { ok: true, machines: [] }
    if (args.command === 'machine hosts') return { ok: true, hosts: ['gpu'] }
    if (args.command === 'machine browse gpu') return { ok: true, path: '/home/me', directories: ['project folder'] }
    if (args.command.startsWith('machine browse')) return { ok: true, path: '/home/me/project folder', directories: [] }
    return { ok: true, machines: [{ alias: 'compute', target: 'gpu', workspacePath: '/home/me/project folder' }] }
  })
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MachinePicker t={DEFAULT_THEME} onCancel={() => undefined} /></GatewayProvider>, { width, height })
  const press = async (key: string) => { await act(async () => { screen.mockInput.pressKey(key); await Bun.sleep(key === 'ESCAPE' ? 80 : 0) }); await screen.flush() }
  try {
    await act(async () => { await Bun.sleep(0) }); await screen.flush()
    await press('RETURN')
    await act(async () => screen.mockInput.typeText('compute')); await screen.flush()
    await press('TAB'); await press('F2')
    expect(screen.captureCharFrame()).toContain('Choose SSH host')
    expect(screen.captureCharFrame()).toContain('gpu')
    await press('RETURN')
    expect(screen.captureCharFrame()).toContain('compute')
    await press('TAB'); await press('F2')
    expect(screen.captureCharFrame()).toContain('/home/me')
    await press('RETURN')
    expect(screen.captureCharFrame()).toContain('No subfolders')
    await act(async () => { screen.renderer.keyInput.processParsedKey({ name: 'space', raw: ' ', sequence: ' ', ctrl: false, shift: false, meta: false, option: false, eventType: 'press', source: 'raw' }) }); await screen.flush()
    expect(screen.captureCharFrame()).toContain('/home/me/project folder')
    await press('F2'); await press('ESCAPE')
    expect(screen.captureCharFrame()).toContain('/home/me/project folder')
    await press('RETURN')
    expect(rpc).toHaveBeenLastCalledWith('slash.exec', { command: 'machine add "compute" "gpu" "/home/me/project folder"' })
  } finally { act(() => screen.renderer.destroy()) }
})

it('leaves a pending browse and ignores its late response', async () => {
  let complete!: (result: unknown) => void
  const rpc = vi.fn(async (_method: string, args: { command: string }) => args.command === 'machine list' ? { ok: true, machines: [] } : new Promise(resolve => { complete = resolve }))
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MachinePicker t={DEFAULT_THEME} onCancel={() => undefined} /></GatewayProvider>, { width: 100, height: 30 })
  const press = async (key: string) => { await act(async () => { screen.mockInput.pressKey(key); await Bun.sleep(key === 'ESCAPE' ? 80 : 0) }); await screen.flush() }
  try {
    await act(async () => { await Bun.sleep(0) }); await screen.flush()
    await press('RETURN'); await press('TAB')
    await act(async () => screen.mockInput.typeText('gpu')); await screen.flush()
    await press('TAB'); await press('F2')
    expect(screen.captureCharFrame()).toContain('Loading')
    await press('ESCAPE')
    await act(async () => { complete({ ok: true, path: '/late', directories: [] }); await Bun.sleep(0) }); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Add workspace')
    expect(screen.captureCharFrame()).not.toContain('/late')
    expect(screen.captureCharFrame()).toContain('gpu')
  } finally { act(() => screen.renderer.destroy()) }
})

it.each([[80, 24], [40, 18]])('reviews every integration before remote setup at %sx%s without reading credentials', async (width, height) => {
  const machine = { alias: 'audit', target: 'review-host', workspacePath: '/workspace/review' }
  const rpc = vi.fn(async () => ({ ok: true, machines: [machine] }))
  const connect = vi.fn()
  const onCancel = vi.fn()
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MachinePicker t={DEFAULT_THEME} connect={connect} onCancel={onCancel} /></GatewayProvider>, { width, height })
  const press = async (key: string) => { await act(async () => { screen.mockInput.pressKey(key); await Bun.sleep(key === 'ESCAPE' ? 80 : 0) }); await screen.flush() }
  try {
    await screen.flush(); await press('RETURN')
    for (const title of ['Overview', 'Providers and API credentials', 'Subscription authentication', 'Models and reasoning', 'Search', 'MCP connections', 'Skills and assets', 'Agent definitions', 'Plugins, hooks and LSP', 'Browser', 'Channels and webhooks', 'SSH and persistence']) {
      for (const word of title.split(' ')) expect(screen.captureCharFrame()).toContain(word)
      expect(screen.captureCharFrame()).toContain('review-host')
      await press('END')
      // Paging must keep the action reachable in a narrow terminal.
      expect(screen.captureCharFrame()).toContain('Enter prepare task')
      await press('TAB')
    }
    expect(rpc).toHaveBeenCalledTimes(1)
    expect(connect).not.toHaveBeenCalled()
    await press('ESCAPE')
    expect(screen.captureCharFrame()).toContain('audit · review-host')
    expect(onCancel).not.toHaveBeenCalled()
  } finally { act(() => screen.renderer.destroy()) }
})

it('rejects a destination changed since review instead of connecting to it', async () => {
  const machine = { alias: 'audit', target: 'review-host', workspacePath: '/workspace' }
  const rpc = vi.fn(async (_method: string, args: { command: string }) => args.command === 'machine list'
    ? { ok: true, machines: [machine] } : { ok: true, machine: { ...machine, target: 'different-host' } })
  const connect = vi.fn()
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MachinePicker t={DEFAULT_THEME} connect={connect} onCancel={() => {}} /></GatewayProvider>, { width: 100, height: 30 })
  try {
    await screen.flush()
    await act(async () => { screen.mockInput.pressKey('RETURN'); await Bun.sleep(0) }); await screen.flush()
    await act(async () => { screen.mockInput.pressKey('RETURN'); await Bun.sleep(0) }); await screen.flush()
    expect(screen.captureCharFrame()).toContain('Saved destination changed')
    expect(connect).not.toHaveBeenCalled()
  } finally { act(() => screen.renderer.destroy()) }
})

it('cancels a pending reviewed connection and does not adopt its late response', async () => {
  const machine = { alias: 'audit', target: 'review-host', workspacePath: '/workspace' }
  let complete!: (result: unknown) => void
  const rpc = vi.fn(async (_method: string, args: { command: string }) => args.command === 'machine list'
    ? { ok: true, machines: [machine] } : new Promise(resolve => { complete = resolve }))
  const connect = vi.fn()
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MachinePicker t={DEFAULT_THEME} connect={connect} onCancel={() => {}} /></GatewayProvider>, { width: 100, height: 30 })
  try {
    await screen.flush()
    await act(async () => { screen.mockInput.pressKey('RETURN'); await Bun.sleep(0) }); await screen.flush()
    await act(async () => { screen.mockInput.pressKey('RETURN'); screen.mockInput.pressKey('RETURN'); await Bun.sleep(0) }); await screen.flush()
    expect(rpc.mock.calls.filter(call => call[1].command === 'machine connect audit')).toHaveLength(1)
    await act(async () => { screen.mockInput.pressKey('ESCAPE'); await Bun.sleep(80) }); await screen.flush()
    await act(async () => { complete({ ok: true, machine }); await Bun.sleep(0) }); await screen.flush()
    expect(screen.captureCharFrame()).toContain('audit · review-host')
    expect(connect).not.toHaveBeenCalled()
    expect(screen.captureCharFrame()).not.toContain('Remote session closed')
  } finally { act(() => screen.renderer.destroy()) }
})

it('opens an explicitly named workspace at review without connecting', async () => {
  const machines = [{ alias: 'first', target: 'first-host', workspacePath: '/first' }, { alias: 'chosen', target: 'chosen-host', workspacePath: '/chosen' }]
  const rpc = vi.fn(async () => ({ ok: true, machines }))
  const connect = vi.fn()
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><MachinePicker initialAlias="chosen" t={DEFAULT_THEME} connect={connect} onCancel={() => {}} /></GatewayProvider>, { width: 80, height: 24 })
  try {
    await screen.flush()
    expect(screen.captureCharFrame()).toContain('Remote setup')
    expect(screen.captureCharFrame()).toContain('chosen-host')
    expect(screen.captureCharFrame()).not.toContain('first-host')
    expect(connect).not.toHaveBeenCalled()
  } finally { act(() => screen.renderer.destroy()) }
})
