// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */

import { testRender } from '@opentui/react/test-utils'
import { act } from 'react'
import { describe, expect, it, vi } from 'vitest'

import { GatewayProvider } from '../app/gatewayContext.js'
import type { GatewayServices } from '../app/interfaces.js'
import type { GatewayClient } from '../gatewayClient.js'
import { listTerminals, terminalAge, terminalState, type TerminalSummary } from '../lib/terminals.js'
import { TerminalPanelOverlay } from '../opentui/terminalPanel.js'
import { DEFAULT_THEME } from '../theme.js'

const NOW = 1_800_000_000_000

const wireTerminal = (overrides: Record<string, unknown> = {}) => ({
  canInterrupt: false,
  canKill: true,
  canWrite: false,
  command: 'bun test ./test',
  cwd: '/repo',
  exitCode: null,
  id: 'proc-1',
  kind: 'background',
  label: 'bun test',
  outputChars: 2_400,
  pid: 4242,
  running: true,
  startedAt: NOW - 74_000,
  ...overrides
})

const servicesWith = (rpc: GatewayServices['rpc']): GatewayServices =>
  ({ gw: {} as GatewayClient, rpc }) as unknown as GatewayServices

const settle = async (setup: { flush: () => Promise<void> }) => {
  await act(async () => {
    await Bun.sleep(0)
    await Bun.sleep(0)
  })
  await setup.flush()
}

describe('terminal list model', () => {
  it('puts running terminals first and drops rows it cannot address', async () => {
    const rpc = vi.fn(async () => ({
      ok: true,
      terminals: [
        wireTerminal({ endedAt: NOW - 1_000, exitCode: 0, id: 'old', running: false, startedAt: NOW - 9_000 }),
        wireTerminal({ id: 'live' }),
        // No id: every action against this row would fail, so it is not a row.
        wireTerminal({ id: '   ' })
      ]
    }))

    const rows = await listTerminals(rpc as unknown as GatewayServices['rpc'])

    expect(rows.map(row => row.id)).toEqual(['live', 'old'])
  })

  it('reports state from the exit code rather than calling every exit a success', () => {
    expect(terminalState(wireTerminal() as unknown as TerminalSummary)).toBe('running')
    expect(
      terminalState(wireTerminal({ exitCode: 0, running: false }) as unknown as TerminalSummary)
    ).toBe('exited')
    expect(
      terminalState(wireTerminal({ exitCode: 137, running: false }) as unknown as TerminalSummary)
    ).toBe('failed')
  })

  it('ages a finished terminal from when it ended, not from now', () => {
    const finished = wireTerminal({
      endedAt: NOW - 60_000,
      exitCode: 0,
      running: false,
      startedAt: NOW - 90_000
    }) as unknown as TerminalSummary

    expect(terminalAge(finished, NOW)).toBe('30s')
    expect(terminalAge(wireTerminal() as unknown as TerminalSummary, NOW)).toBe('1m 14s')
  })
})

describe('OpenTUI terminal panel', () => {
  it('lists what each terminal is and how long it has been running', async () => {
    const rpc = vi.fn(async () => ({ ok: true, terminals: [wireTerminal()] }))
    const setup = await testRender(
      <GatewayProvider value={servicesWith(rpc as unknown as GatewayServices['rpc'])}>
        <TerminalPanelOverlay onClose={() => undefined} t={DEFAULT_THEME} />
      </GatewayProvider>,
      { height: 24, width: 90 }
    )

    try {
      await settle(setup)
      const frame = setup.captureCharFrame()

      expect(frame).toContain('Terminals')
      expect(frame).toContain('1 running')
      expect(frame).toContain('bg bun test')
      expect(frame).toContain('pid 4242')
      expect(rpc).toHaveBeenCalledWith('terminal.list', {})
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('opens one terminal on Enter and shows its output tail', async () => {
    const rpc = vi.fn(async (method: string) =>
      method === 'terminal.list'
        ? { ok: true, terminals: [wireTerminal()] }
        : { ok: true, terminal: { ...wireTerminal(), output: 'compiling…\n42 tests passed\n', outputTruncated: false } }
    )
    const setup = await testRender(
      <GatewayProvider value={servicesWith(rpc as unknown as GatewayServices['rpc'])}>
        <TerminalPanelOverlay onClose={() => undefined} t={DEFAULT_THEME} />
      </GatewayProvider>,
      { height: 24, width: 90 }
    )

    try {
      await settle(setup)
      act(() => setup.mockInput.pressEnter())
      await settle(setup)
      const frame = setup.captureCharFrame()

      expect(frame).toContain('42 tests passed')
      expect(frame).toContain('/repo')
      expect(rpc).toHaveBeenCalledWith('terminal.inspect', {
        max_output_chars: 60_000,
        terminal_id: 'proc-1'
      })
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('keeps bracketed paste intact while composing PTY input', async () => {
    const terminal = wireTerminal({ canInterrupt: true, canWrite: true, kind: 'pty' })
    const rpc = vi.fn(async (method: string) =>
      method === 'terminal.list'
        ? { ok: true, terminals: [terminal] }
        : method === 'terminal.inspect'
          ? { ok: true, terminal: { ...terminal, output: '', outputTruncated: false } }
          : { ok: true }
    )
    const setup = await testRender(
      <GatewayProvider value={servicesWith(rpc as unknown as GatewayServices['rpc'])}>
        <TerminalPanelOverlay onClose={() => undefined} t={DEFAULT_THEME} />
      </GatewayProvider>,
      { height: 24, width: 90 }
    )

    try {
      await settle(setup)
      act(() => setup.mockInput.pressEnter())
      await settle(setup)
      act(() => setup.mockInput.pressKey('i'))
      await setup.flush()
      act(() => setup.renderer.keyInput.processPaste(new TextEncoder().encode('echo alpha\nsecond line')))
      await setup.flush()

      expect(setup.captureCharFrame()).toContain('echo alpha')

      act(() => setup.mockInput.pressEnter())
      await settle(setup)
      expect(rpc).toHaveBeenCalledWith('terminal.control', {
        action: 'write',
        chars: 'echo alpha\nsecond line\n',
        terminal_id: 'proc-1'
      })
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('kills the selected terminal and refuses input on one that has no stdin', async () => {
    const rpc = vi.fn(async (method: string) =>
      method === 'terminal.list'
        ? { ok: true, terminals: [wireTerminal()] }
        : { ok: true, terminal: { ...wireTerminal(), output: '', outputTruncated: false } }
    )
    const setup = await testRender(
      <GatewayProvider value={servicesWith(rpc as unknown as GatewayServices['rpc'])}>
        <TerminalPanelOverlay onClose={() => undefined} t={DEFAULT_THEME} />
      </GatewayProvider>,
      { height: 24, width: 90 }
    )

    try {
      await settle(setup)
      act(() => setup.mockInput.pressKey('k'))
      await settle(setup)

      expect(rpc).toHaveBeenCalledWith('terminal.control', { action: 'kill', terminal_id: 'proc-1' })

      // `i` on a background process is refused with the reason rather than
      // opening an input line that could never deliver anything.
      act(() => setup.mockInput.pressKey('i'))
      await settle(setup)
      expect(setup.captureCharFrame()).toContain('open a terminal first')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('says so plainly when no daemon is connected', async () => {
    const setup = await testRender(
      <TerminalPanelOverlay onClose={() => undefined} t={DEFAULT_THEME} />,
      { height: 20, width: 90 }
    )

    try {
      await settle(setup)
      expect(setup.captureCharFrame()).toContain('not connected to a daemon')
    } finally {
      act(() => setup.renderer.destroy())
    }
  })
})
