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
  it('does not turn a daemon error into an empty terminal list', async () => {
    const rpc = vi.fn(async () => ({ ok: false, error: 'connection unavailable' }))
    await expect(listTerminals(rpc as unknown as GatewayServices['rpc'])).rejects.toThrow('connection unavailable')
  })
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
  it('retains rejected input and blocks duplicate writes while the request is pending', async () => {
    const terminal = wireTerminal({ id: 'write-retry', canWrite: true, kind: 'pty' })
    let finish: (value: unknown) => void = () => undefined
    const rpc = vi.fn(async (method: string) => method === 'terminal.list'
      ? { ok: true, terminals: [terminal] }
      : method === 'terminal.inspect'
        ? { ok: true, terminal: { ...terminal, output: '', outputTruncated: false } }
        : new Promise(resolve => { finish = resolve }))
    const setup = await testRender(<GatewayProvider value={servicesWith(rpc as unknown as GatewayServices['rpc'])}>
      <TerminalPanelOverlay onClose={() => undefined} t={DEFAULT_THEME} />
    </GatewayProvider>, { width: 90, height: 24 })
    try {
      await settle(setup)
      act(() => setup.mockInput.pressEnter()); await settle(setup)
      act(() => setup.mockInput.pressKey('i')); await settle(setup)
      await act(async () => setup.mockInput.typeText('echo retained')); await settle(setup)
      act(() => setup.mockInput.pressEnter()); await settle(setup)
      act(() => setup.mockInput.pressEnter()); await settle(setup)
      expect(rpc.mock.calls.filter(call => call[0] === 'terminal.control')).toHaveLength(1)
      await act(async () => finish({ ok: false, error: 'write rejected' })); await settle(setup)
      expect(setup.captureCharFrame()).toContain('write rejected')
      expect(setup.captureCharFrame()).toContain('echo retained')
      act(() => setup.mockInput.pressKey('ESCAPE')); await settle(setup)
      act(() => setup.mockInput.pressKey('i')); await settle(setup)
      expect(setup.captureCharFrame()).toContain('echo retained')
      act(() => setup.mockInput.pressEnter()); await settle(setup)
      expect(rpc.mock.calls.filter(call => call[0] === 'terminal.control')).toHaveLength(2)
      await act(async () => finish({ ok: true })); await settle(setup)
    } finally { act(() => setup.renderer.destroy()) }
  })
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
      // Header budget + the mockup's single-row entry: dot, bold label,
      // muted command, right-aligned state budget.
      expect(frame).toContain('1 tracked · 1 running')
      expect(frame).toContain('bun test')
      expect(frame).toContain('bun test ./test')
      expect(frame).toContain('running ·')
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

  it('arms a two-step kill and only signals on the confirming repeat', async () => {
    const rpc = vi.fn(async (method: string) =>
      method === 'terminal.list'
        ? { ok: true, terminals: [wireTerminal()] }
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
      const controlCalls = () => rpc.mock.calls.filter(([method]) => method === 'terminal.control')

      // Mockup 06: destructive keys are two-step. The first k only arms — the
      // row shows the warn confirm line and no signal leaves the process.
      act(() => setup.mockInput.pressKey('k'))
      await settle(setup)

      expect(setup.captureCharFrame()).toContain('kill bun test? k again to confirm · esc cancel')
      expect(controlCalls()).toHaveLength(0)

      // The repeat press executes.
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

  it('makes K (force) take the same two steps before sending SIGKILL', async () => {
    const rpc = vi.fn(async (method: string) =>
      method === 'terminal.list' ? { ok: true, terminals: [wireTerminal()] } : { ok: true }
    )
    const setup = await testRender(
      <GatewayProvider value={servicesWith(rpc as unknown as GatewayServices['rpc'])}>
        <TerminalPanelOverlay onClose={() => undefined} t={DEFAULT_THEME} />
      </GatewayProvider>,
      { height: 24, width: 90 }
    )

    try {
      await settle(setup)

      // Force is a different signal, not a different level of caution: K
      // arms exactly like k, and only the repeat sends SIGKILL.
      act(() => setup.mockInput.pressKey('K'))
      await settle(setup)

      expect(setup.captureCharFrame()).toContain('force kill bun test? K again to confirm · esc cancel')
      expect(rpc).not.toHaveBeenCalledWith(
        'terminal.control',
        expect.objectContaining({ action: 'kill' })
      )

      act(() => setup.mockInput.pressKey('K'))
      await settle(setup)

      expect(rpc).toHaveBeenCalledWith('terminal.control', {
        action: 'kill',
        signal: 'SIGKILL',
        terminal_id: 'proc-1'
      })
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('steps back cleanly: Esc cancels an armed kill without closing the panel', async () => {
    const rpc = vi.fn(async (method: string) =>
      method === 'terminal.list'
        ? { ok: true, terminals: [wireTerminal()] }
        : method === 'terminal.inspect'
          ? { ok: true, terminal: { ...wireTerminal(), output: 'tail', outputTruncated: false } }
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

      act(() => setup.mockInput.pressKey('k'))
      await settle(setup)

      act(() => setup.mockInput.pressEscape())
      // The renderer holds a bare ESC briefly to disambiguate escape sequences.
      await act(async () => {
        await Bun.sleep(60)
      })
      await setup.flush()

      // The cancel consumed this Esc: the confirm line is gone and the
      // overlay is still up — the next Enter opens the detail view.
      expect(setup.captureCharFrame()).not.toContain('again to confirm')

      act(() => setup.mockInput.pressEnter())
      await settle(setup)

      expect(setup.captureCharFrame()).toContain('OUTPUT —')
      expect(rpc).not.toHaveBeenCalledWith(
        'terminal.control',
        expect.objectContaining({ action: 'kill' })
      )
    } finally {
      act(() => setup.renderer.destroy())
    }
  })

  it('disarms the pending kill when the selection moves to another terminal', async () => {
    const rpc = vi.fn(async (method: string) =>
      method === 'terminal.list'
        ? {
            ok: true,
            terminals: [
              wireTerminal({ id: 'live-a', label: 'alpha' }),
              wireTerminal({ id: 'live-b', label: 'beta', startedAt: NOW - 200_000 })
            ]
          }
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

      act(() => setup.mockInput.pressKey('k'))
      await settle(setup)
      expect(setup.captureCharFrame()).toContain('kill alpha?')

      // Moving the selection abandons the arm; killing now needs two fresh
      // presses aimed at the newly selected row.
      act(() => setup.mockInput.pressArrow('down'))
      await settle(setup)
      expect(setup.captureCharFrame()).not.toContain('again to confirm')

      act(() => setup.mockInput.pressKey('k'))
      await settle(setup)
      act(() => setup.mockInput.pressKey('k'))
      await settle(setup)

      expect(rpc).toHaveBeenCalledWith('terminal.control', { action: 'kill', terminal_id: 'live-b' })
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

it.each([[220, 65], [60, 24]])('filters terminals and searches retained output at %ix%i', async (width, height) => {
  const onClose = vi.fn()
  const live = wireTerminal({ label: 'Active build', command: 'compile' })
  const done = wireTerminal({ id: 'done', label: 'Finished tests', command: 'tests', running: false, exitCode: 0 })
  const rpc = vi.fn(async (method: string) => method === 'terminal.list'
    ? { ok: true, terminals: [live, done] }
    : { ok: true, terminal: { ...done, output: 'healthy line\nERROR missing file\nhealthy tail', outputTruncated: false } })
  const setup = await testRender(<GatewayProvider value={servicesWith(rpc as unknown as GatewayServices['rpc'])}><TerminalPanelOverlay onClose={onClose} t={DEFAULT_THEME} /></GatewayProvider>, { width, height })
  try {
    await settle(setup)
    act(() => setup.mockInput.pressKey('f')); await settle(setup)
    expect(setup.captureCharFrame()).toContain('Active build')
    expect(setup.captureCharFrame()).not.toContain('Finished tests')
    act(() => setup.mockInput.pressKey('f')); await settle(setup)
    expect(setup.captureCharFrame()).toContain('Finished tests')
    expect(setup.captureCharFrame()).not.toContain('Active build')
    act(() => setup.mockInput.pressEnter()); await settle(setup)
    expect(setup.captureCharFrame()).toContain('ended')
    act(() => setup.mockInput.pressKey('/')); await settle(setup)
    await act(async () => setup.mockInput.typeText('error')); await settle(setup)
    act(() => setup.mockInput.pressEnter()); await settle(setup)
    expect(setup.captureCharFrame()).toContain('2: ERROR missing file')
    expect(setup.captureCharFrame()).not.toContain('healthy line')
    expect(rpc.mock.calls.some(call => call[0] === 'terminal.control')).toBe(false)
    act(() => setup.mockInput.pressEscape()); await act(async () => { await Bun.sleep(60) }); await settle(setup)
    expect(setup.captureCharFrame()).toContain('healthy line')
    expect(onClose).not.toHaveBeenCalled()
  } finally { act(() => setup.renderer.destroy()) }
})
