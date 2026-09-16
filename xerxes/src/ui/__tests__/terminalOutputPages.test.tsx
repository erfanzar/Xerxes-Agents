// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { expect, it, vi } from 'vitest'
import type { GatewayServices } from '../app/interfaces.js'
import { GatewayProvider } from '../app/gatewayContext.js'
import { TerminalOutputPages } from '../opentui/terminalOutputPages.js'
import { DARK_THEME } from '../theme.js'

it.each([[140, 40], [80, 28], [40, 18]])('retains output on page failure and retries the same cursor at %ix%i', async (width, height) => {
  let fail = true
  const rpc = vi.fn(async (_: string, params: Record<string, unknown>) => {
    if (params.cursor && fail) return { ok: false, error: 'Disconnected' }
    return { ok: true, page: { text: params.cursor ? 'Next output\nsecond line' : 'First output', cursor: { streamId: 'stream', offset: params.cursor ? 40 : 20 }, droppedChars: params.cursor ? 10 : 0, running: false, hasMore: !params.cursor } }
  })
  const close = vi.fn()
  const screen = await testRender(<GatewayProvider value={{ rpc } as unknown as GatewayServices}><TerminalOutputPages t={DARK_THEME} terminalId="terminal" onClose={close} /></GatewayProvider>, { width, height })
  const press = async (key: string) => { await act(async () => screen.mockInput.pressKey(key)); await screen.flush() }
  try {
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('First output') })
    await press('n')
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Disconnected') })
    expect(screen.captureCharFrame()).toContain('First output')
    fail = false
    await press('r')
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('Next output') })
    expect(screen.captureCharFrame()).toContain('10 earlier characters')
    expect(rpc).toHaveBeenLastCalledWith('terminal.output', { terminal_id: 'terminal', max_output_chars: 8000, cursor: { streamId: 'stream', offset: 20 } })
    await press('p')
    await vi.waitFor(async () => { await screen.flush(); expect(screen.captureCharFrame()).toContain('First output') })
    await press('ESCAPE')
    await vi.waitFor(() => expect(close).toHaveBeenCalledOnce())
  } finally { await act(async () => screen.renderer.destroy()) }
})
