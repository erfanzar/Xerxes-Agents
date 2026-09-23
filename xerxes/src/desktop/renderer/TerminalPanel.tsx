// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Interactive terminal in the task rail.
 *
 * The shell is a real PTY owned by the workspace runtime (`terminal.open`),
 * so it outlives this view: switching rail tabs, closing the rail or
 * reloading the window re-attaches and replays the retained output. Output is
 * pushed as `terminal_output` events after `terminal.attach` — no polling —
 * and keystrokes go back through `terminal.control` in order.
 *
 * Only shells you open here are listed as tabs. The agent's own PTYs stay in
 * the Terminals list, where they are shown read-mostly with their command.
 */

import { useCallback, useEffect, useRef, useState, type ReactElement } from 'react'
import { Terminal as XTerm, type ITheme } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'

import { desktopCall, text, type RpcRecord } from './desktopRpc.js'
import { Icon } from './Icon.js'
import { userShells, type ShellRow } from './terminalShells.js'
import type { Snapshot } from './store.js'

const DARK_ANSI = {
  black: '#1b1f27', red: '#f28b82', green: '#81c995', yellow: '#fdd663', blue: '#6ea8fe', magenta: '#c58af9', cyan: '#78d9ec', white: '#d6dce8',
  brightBlack: '#5c6373', brightRed: '#f6aea9', brightGreen: '#a8dab5', brightYellow: '#fde293', brightBlue: '#9ec5fe', brightMagenta: '#d7aefb', brightCyan: '#a1e4f2', brightWhite: '#ffffff',
}
const LIGHT_ANSI = {
  black: '#1c1f26', red: '#c5221f', green: '#137333', yellow: '#946200', blue: '#1a56c4', magenta: '#8430ce', cyan: '#007b83', white: '#5c6373',
  brightBlack: '#80868b', brightRed: '#d93025', brightGreen: '#188038', brightYellow: '#b06000', brightBlue: '#2c6ecb', brightMagenta: '#9334e6', brightCyan: '#12848e', brightWhite: '#1c1f26',
}

/** The terminal follows the app's own tokens, so it never looks bolted on. */
function terminalTheme(host: HTMLElement): ITheme {
  const style = getComputedStyle(host)
  const token = (name: string, fallback: string): string => style.getPropertyValue(name).trim() || fallback
  const light = document.documentElement.getAttribute('data-theme') === 'light'
  return {
    ...(light ? LIGHT_ANSI : DARK_ANSI),
    background: token('--x-screen', light ? '#ffffff' : '#11141a'),
    foreground: token('--x-prose', light ? '#333843' : '#d6dce8'),
    cursor: token('--x-accent', '#6ea8fe'),
    cursorAccent: token('--x-screen', '#11141a'),
    selectionBackground: light ? 'rgba(26,86,196,0.22)' : 'rgba(110,168,254,0.28)',
  }
}

export function TerminalPanel({ snap }: { snap: Snapshot }): ReactElement {
  const [shells, setShells] = useState<ShellRow[] | null>(null)
  const [active, setActive] = useState('')
  const [error, setError] = useState('')
  const [opening, setOpening] = useState(false)
  const call = useCallback((method: string, params: RpcRecord = {}) => desktopCall(window.xerxes, snap.sessionKey, method, params), [snap.sessionKey])
  // Guards against two automatic opens racing (e.g. a reconnect during the first open).
  const started = useRef(false)

  const open = useCallback(async (): Promise<void> => {
    setOpening(true); setError('')
    try {
      const size = { cols: 100, rows: 30 }
      const result = await call('terminal.open', size)
      const id = text(result.terminal_id)
      if (!id) throw new Error('The runtime did not return a terminal')
      setShells(current => [...(current ?? []), { id, running: true }])
      setActive(id)
    } catch (failure) {
      const message = failure instanceof Error ? failure.message : String(failure)
      setError(/Unknown method: terminal\.open/.test(message) ? 'The running workspace runtime is older than this app. Update it from the runtime status in the sidebar to use terminals here.' : message)
    } finally { setOpening(false) }
  }, [call])

  // Pick up shells that already exist (they outlive this view); open one if
  // none. Runs again whenever the runtime connection comes back — a restarted
  // runtime (an app update) must not leave an old error on screen.
  const [attempt, setAttempt] = useState(0)
  const online = snap.connection === 'online'
  useEffect(() => {
    if (!online) return
    let current = true
    setError('')
    void call('terminal.list').then(result => {
      if (!current) return
      const rows = userShells(Array.isArray(result.terminals) ? result.terminals as RpcRecord[] : [])
      setShells(rows)
      if (rows.length) setActive(id => rows.some(row => row.id === id) ? id : rows[0]!.id)
      // No shell (first visit, or the runtime restarted and its shells died): open one.
      else if (!started.current) { started.current = true; void open().finally(() => { started.current = false }) }
    }).catch(failure => { if (current) { setShells([]); setError(failure instanceof Error ? failure.message : String(failure)) } })
    return () => { current = false }
  }, [call, open, online, attempt])

  // The tab goes at once; the shell shuts down in the background. Waiting on
  // the kill made closing feel laggy.
  const close = (id: string): void => {
    const next = (shells ?? []).filter(row => row.id !== id)
    setShells(next)
    if (active === id) setActive(next.at(-1)?.id ?? '')
    void call('terminal.control', { terminal_id: id, action: 'kill' }).catch(() => { /* already gone */ })
  }

  const markExited = useCallback((id: string) => {
    setShells(current => (current ?? []).map(row => row.id === id ? { ...row, running: false } : row))
  }, [])

  return (
    <div className="term">
      <div className="term__bar" role="tablist" aria-label="Terminals">
        {(shells ?? []).map((row, index) => (
          <span key={row.id} className={`term-tab${row.id === active ? ' is-active' : ''}${row.running ? '' : ' is-exited'}`}>
            <button role="tab" aria-selected={row.id === active} onClick={() => setActive(row.id)} title={row.running ? 'Shell' : 'Exited'}>
              <Icon name="terminal" size={12} /><span>Shell {index + 1}</span>
            </button>
            <button className="term-tab__close" aria-label={`Close shell ${index + 1}`} title="Close shell" onClick={() => close(row.id)}><Icon name="close" size={10} /></button>
          </span>
        ))}
        <button className="term__new" aria-label="New shell" title="New shell" disabled={opening} onClick={() => void open()}><Icon name="plus" size={13} /></button>
      </div>
      {error && <div className="term__error" role="alert"><span>{error}</span><button className="btn" onClick={() => { started.current = false; setAttempt(value => value + 1) }}>Try again</button></div>}
      {shells !== null && shells.length === 0 && !opening && !error && (
        <div className="term__empty"><Icon name="terminal" size={22} /><p>No shell open.</p><button className="btn" onClick={() => void open()}>Open a shell</button></div>
      )}
      {(shells ?? []).map(row => (
        <XtermView key={row.id} snap={snap} terminalId={row.id} visible={row.id === active} onExit={markExited} />
      ))}
    </div>
  )
}

function XtermView({ snap, terminalId, visible, onExit }: { snap: Snapshot; terminalId: string; visible: boolean; onExit: (id: string) => void }): ReactElement {
  const host = useRef<HTMLDivElement>(null)
  const term = useRef<XTerm | null>(null)
  const fit = useRef<FitAddon | null>(null)
  const call = useCallback((method: string, params: RpcRecord = {}) => desktopCall(window.xerxes, snap.sessionKey, method, { terminal_id: terminalId, ...params }), [snap.sessionKey, terminalId])

  useEffect(() => {
    const element = host.current
    if (!element) return
    const xterm = new XTerm({
      allowProposedApi: false,
      convertEol: false,
      cursorBlink: true,
      fontFamily: getComputedStyle(element).getPropertyValue('--mono').trim() || 'ui-monospace, SFMono-Regular, Menlo, monospace',
      fontSize: 12.5,
      lineHeight: 1.2,
      scrollback: 5000,
      macOptionIsMeta: true,
      theme: terminalTheme(element),
    })
    const fitter = new FitAddon()
    xterm.loadAddon(fitter)
    xterm.open(element)
    term.current = xterm
    fit.current = fitter
    let alive = true
    let exited = false

    // Subscribe before attaching: an event that races the attach reply is
    // buffered and flushed after the replay, never lost and never duplicated
    // (the daemon subscribes and snapshots in one synchronous step).
    const pending: string[] = []
    let attached = false
    const unsubscribe = window.xerxes.onEvent(event => {
      if (event.type !== 'terminal_output' || event.payload.terminal_id !== terminalId) return
      const data = text(event.payload.data)
      if (!attached) { if (data) pending.push(data); return }
      if (data) xterm.write(data)
      if (event.payload.closed === true && !exited) {
        exited = true
        xterm.write('\r\n\x1b[2m[process exited]\x1b[0m\r\n')
        onExit(terminalId)
      }
    })
    // Fit before replaying: xterm starts at a 2-column minimum until its font
    // is measured, and history written at that width never re-wraps.
    const fitNow = (): boolean => {
      if (!element.offsetWidth || !element.offsetHeight) return false
      try { fitter.fit(); return xterm.cols > 2 } catch { return false }
    }
    const attach = async (): Promise<void> => {
      await document.fonts?.ready
      for (let tries = 0; tries < 20 && alive && !fitNow(); tries += 1) await new Promise(resolve => requestAnimationFrame(resolve))
      if (!alive) return
      void call('terminal.resize', { cols: xterm.cols, rows: xterm.rows }).catch(() => {})
      const result = await call('terminal.attach')
      if (!alive) return
      xterm.write(text(result.data))
      attached = true
      for (const chunk of pending.splice(0)) xterm.write(chunk)
      if (result.running === false) { exited = true; xterm.write('\r\n\x1b[2m[process exited]\x1b[0m\r\n'); onExit(terminalId) }
    }
    void attach().catch(failure => { if (alive) xterm.write(`\r\n\x1b[31m${failure instanceof Error ? failure.message : String(failure)}\x1b[0m\r\n`) })

    const input = xterm.onData(data => { if (!exited) void call('terminal.control', { action: 'write', chars: data }).catch(() => {}) })

    let resizeTimer: ReturnType<typeof setTimeout> | undefined
    const syncSize = (): void => {
      if (!element.offsetWidth || !element.offsetHeight) return
      try { fitter.fit() } catch { return }
      clearTimeout(resizeTimer)
      resizeTimer = setTimeout(() => { if (!exited) void call('terminal.resize', { cols: xterm.cols, rows: xterm.rows }).catch(() => {}) }, 120)
    }
    const observer = new ResizeObserver(syncSize)
    observer.observe(element)
    requestAnimationFrame(syncSize)

    // Follow the app's light/dark switch.
    const themeWatch = new MutationObserver(() => { xterm.options.theme = terminalTheme(element) })
    themeWatch.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] })

    return () => {
      alive = false
      clearTimeout(resizeTimer)
      observer.disconnect()
      themeWatch.disconnect()
      input.dispose()
      unsubscribe()
      void call('terminal.detach').catch(() => {})
      xterm.dispose()
      term.current = null
      fit.current = null
    }
  }, [call, terminalId, onExit])

  // A hidden tab has no size; refit and focus when it becomes visible.
  useEffect(() => {
    if (!visible) return
    requestAnimationFrame(() => {
      try { fit.current?.fit() } catch { /* not laid out yet */ }
      term.current?.focus()
    })
  }, [visible])

  return <div className="term__view" hidden={!visible} ref={host} onClick={() => term.current?.focus()} />
}
