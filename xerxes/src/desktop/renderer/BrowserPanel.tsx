// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The right panel's Browser tab. The page itself is a sandboxed native view
 * the main process lays over the `.browser-panel__page` box; this component
 * owns the address bar, the "traffic leaves from" label and the Ports list.
 * On an SSH workspace `localhost:8080` is the remote machine's port 8080.
 */

import { useEffect, useLayoutEffect, useRef, useState, type ReactElement } from 'react'

import { desktopError } from './desktopRpc.js'
import { Icon } from './Icon.js'

export interface BrowserStateView {
  url: string
  title: string
  loading: boolean
  canGoBack: boolean
  canGoForward: boolean
  via: string | null
  error?: string
}
interface PortRow { remote: number; local: number; state: 'starting' | 'ready' | 'failed'; error?: string }

const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value)

export function browserStateOf(value: unknown): BrowserStateView | null {
  if (!record(value) || typeof value.url !== 'string') return null
  return {
    url: value.url, title: typeof value.title === 'string' ? value.title : '', loading: value.loading === true,
    canGoBack: value.canGoBack === true, canGoForward: value.canGoForward === true,
    via: typeof value.via === 'string' ? value.via : null,
    ...(typeof value.error === 'string' ? { error: value.error } : {}),
  }
}

/** The address as the person typed it: the tunnel is shown separately, never in the URL. */
export function displayUrl(url: string): string {
  return url.replace(/^http:\/\//, '').replace(/\/$/, '')
}

export function BrowserPanel(): ReactElement {
  const bridge = window.xerxes
  const page = useRef<HTMLDivElement>(null)
  const [state, setState] = useState<BrowserStateView | null>(null)
  const [address, setAddress] = useState('')
  const [editing, setEditing] = useState(false)
  const [error, setError] = useState('')
  const [ports, setPorts] = useState<{ via: string | null; ports: PortRow[] }>({ via: null, ports: [] })
  const [portInput, setPortInput] = useState('')
  const [showPorts, setShowPorts] = useState(false)
  const available = typeof bridge.browser === 'function'

  const loadPorts = async (): Promise<void> => {
    if (!bridge.browser) return
    const reply = await bridge.browser('ports') as { via?: unknown; ports?: unknown }
    setPorts({ via: typeof reply.via === 'string' ? reply.via : null, ports: Array.isArray(reply.ports) ? reply.ports as PortRow[] : [] })
  }

  useEffect(() => {
    if (!bridge.browser) return
    let alive = true
    void bridge.browser('state').then(value => { const next = browserStateOf(value); if (alive && next) { setState(next); if (!editing) setAddress(displayUrl(next.url)) } }).catch(() => {})
    void loadPorts().catch(() => {})
    const off = bridge.onBrowserState?.(value => { const next = browserStateOf(value); if (next) { setState(next); setAddress(current => editing ? current : displayUrl(next.url)) } })
    return () => { alive = false; off?.() }
  }, [])

  // Keep the native page exactly over our box; hide it when the tab goes away.
  useLayoutEffect(() => {
    const box = page.current
    if (!box || !bridge.browser) return
    const send = () => {
      const rect = box.getBoundingClientRect()
      void bridge.browser!('bounds', { rect: { x: rect.left, y: rect.top, width: rect.width, height: rect.height } }).catch(() => {})
    }
    send()
    const observer = new ResizeObserver(send)
    observer.observe(box)
    window.addEventListener('resize', send)
    // Covered by a dialog or another window view: hide the page so it cannot paint over them.
    const offOccluded = bridge.onOccluded?.(occluded => { if (occluded) void bridge.browser!('bounds', { rect: null }).catch(() => {}); else send() })
    return () => {
      observer.disconnect()
      window.removeEventListener('resize', send)
      offOccluded?.()
      void bridge.browser!('bounds', { rect: null }).catch(() => {})
    }
  }, [])

  const run = async (work: () => Promise<unknown>): Promise<void> => {
    setError('')
    try { await work() } catch (failure) { setError(desktopError(failure)) }
  }
  const go = () => run(async () => {
    setEditing(false)
    const next = browserStateOf(await bridge.browser!('open', { url: address }))
    if (next) setState(next)
  })
  const forward = () => run(async () => {
    const port = portInput.trim()
    if (!port) return
    await bridge.browser!('port-forward', { remote: Number(port) })
    setPortInput('')
    await loadPorts()
  })

  if (!available) return <p className="studio-muted browser-panel__empty">The browser is available in the desktop app.</p>
  const via = state?.via ?? ports.via
  return <div className="browser-panel">
    <form className="browser-panel__bar" onSubmit={event => { event.preventDefault(); void go() }}>
      <button type="button" className="browser-panel__back" aria-label="Back" title="Back" disabled={!state?.canGoBack} onClick={() => void run(() => bridge.browser!('back'))}><Icon name="arrow" size={13} /></button>
      <button type="button" aria-label="Forward" title="Forward" disabled={!state?.canGoForward} onClick={() => void run(() => bridge.browser!('forward'))}><Icon name="arrow" size={13} /></button>
      <button type="button" aria-label={state?.loading ? 'Stop' : 'Reload'} title={state?.loading ? 'Stop' : 'Reload'} disabled={!state?.url} onClick={() => void run(() => bridge.browser!(state?.loading ? 'stop' : 'reload'))}><Icon name={state?.loading ? 'close' : 'retry'} size={13} /></button>
      <input className="browser-panel__address" aria-label="Address" spellCheck={false} autoCapitalize="off" value={address}
        placeholder={via ? `localhost:8080 on ${via}` : 'localhost:8080'}
        onFocus={event => { setEditing(true); event.currentTarget.select() }} onBlur={() => setEditing(false)} onChange={event => setAddress(event.target.value)} />
      <button type="button" aria-label="Ports" title="Forwarded ports" aria-pressed={showPorts} onClick={() => { setShowPorts(value => !value); void loadPorts().catch(() => {}) }}><Icon name="tree" size={13} /></button>
      <button type="button" aria-label="Open in your browser" title="Open in your browser" disabled={!state?.url} onClick={() => void run(() => bridge.browser!('external'))}><Icon name="expand" size={13} /></button>
    </form>
    <p className={`browser-panel__via${via ? ' is-remote' : ''}`} role="status">
      <Icon name={via ? 'cloud' : 'laptop'} size={12} />
      {via ? <>Traffic leaves from <strong>{via}</strong>. localhost here is that machine.</> : <>Traffic leaves from this computer.</>}
    </p>
    {error && <p className="browser-panel__error" role="alert">{error}</p>}
    {showPorts && <section className="browser-panel__ports" aria-label="Forwarded ports">
      {via ? <form onSubmit={event => { event.preventDefault(); void forward() }}>
        <input aria-label="Remote port" inputMode="numeric" placeholder={`Port on ${via}`} value={portInput} onChange={event => setPortInput(event.target.value.replace(/[^0-9]/g, '').slice(0, 5))} />
        <button type="submit" disabled={!portInput}>Forward</button>
      </form> : <p className="studio-muted">Local workspace: ports are already on this computer.</p>}
      {ports.ports.map(row => <div key={row.remote} className="browser-panel__port" data-state={row.state}>
        <span>{row.remote} → <code>localhost:{row.local}</code></span>
        <span className="browser-panel__portstate">{row.state === 'failed' ? row.error ?? 'failed' : row.state}</span>
        <button type="button" disabled={row.state !== 'ready'} onClick={() => void run(async () => { setAddress(`localhost:${row.remote}`); await bridge.browser!('open', { url: `localhost:${row.remote}` }) })}>Open</button>
        <button type="button" onClick={() => void navigator.clipboard?.writeText(`http://localhost:${row.local}/`).catch(() => {})}>Copy URL</button>
        <button type="button" onClick={() => void run(async () => { await bridge.browser!('unforward', { remote: row.remote }); await loadPorts() })}>Stop</button>
      </div>)}
    </section>}
    <div ref={page} className="browser-panel__page" aria-label={state?.title || 'Page'}>
      {!state?.url && <p className="studio-muted">Type an address, or just a port such as <code>8080</code>.</p>}
      {state?.url && state.error && <div className="browser-panel__failed" role="alert">
        <strong>Couldn’t open {displayUrl(state.url)}</strong>
        <p>{state.error}</p>
        <button type="button" className="btn" onClick={() => void run(() => bridge.browser!('reload'))}>Try again</button>
      </div>}
    </div>
  </div>
}
