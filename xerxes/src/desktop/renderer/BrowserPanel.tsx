// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The right panel's Browser tab. The page itself is a sandboxed native view
 * the main process lays over the `.browser-panel__page` box; this component
 * owns the address bar, the "traffic leaves from" label and the Ports list.
 * On an SSH workspace `localhost:8080` is the remote machine's port 8080.
 */

import { useEffect, useLayoutEffect, useRef, useState, type ReactElement } from 'react'

import { CopyButton } from './CopyButton.js'
import { desktopError } from './desktopRpc.js'
import { Icon } from './Icon.js'
import { localMachineLabel } from './runModeView.js'

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

export interface BrowserBoundsRect {
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
}

/** True when a native WebContentsView rect intersects any overlapping DOM overlay/menu/dialog. */
export function rectOverlapsAny(
  target: BrowserBoundsRect,
  obstacles: readonly BrowserBoundsRect[],
): boolean {
  if (target.width <= 0 || target.height <= 0) return false
  const tRight = target.x + target.width
  const tBottom = target.y + target.height
  for (const obs of obstacles) {
    if (obs.width <= 0 || obs.height <= 0) continue
    if (
      obs.x < tRight
      && obs.x + obs.width > target.x
      && obs.y < tBottom
      && obs.y + obs.height > target.y
    ) {
      return true
    }
  }
  return false
}

export const BROWSER_OVERLAY_SELECTOR = [
  '[role="menu"]',
  '[role="dialog"]',
  '[role="listbox"]',
  '.rail-add__menu',
  '.palette',
  '.modelpop',
  '.hints',
  '.t3-popover',
  '.acp-picker__pop',
  '.acp-pop',
  '.combo-pop',
  '.btw-model-pop',
  '.swarm-model-picker',
  '.decision-dock',
  '.findbar',
  '.artifact-toast',
  '.notify-stack',
  '.t3-details-card',
  '.notice-stack',
  '.workspace-notice',
  '.modal',
].join(',')

function overlayCoversBox(box: HTMLElement, rect: DOMRect): boolean {
  if (typeof document === 'undefined' || rect.width <= 0 || rect.height <= 0) return false
  const target: BrowserBoundsRect = { x: rect.left, y: rect.top, width: rect.width, height: rect.height }
  const obstacles: BrowserBoundsRect[] = []
  for (const el of document.querySelectorAll(BROWSER_OVERLAY_SELECTOR)) {
    if (el === box || box.contains(el) || el.contains(box)) continue
    if (el instanceof HTMLElement && el.hidden) continue
    if (typeof window !== 'undefined' && typeof window.getComputedStyle === 'function') {
      const style = window.getComputedStyle(el)
      if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0' || style.pointerEvents === 'none') continue
    }
    const r = el.getBoundingClientRect()
    if (r.width > 0 && r.height > 0) {
      obstacles.push({ x: r.left, y: r.top, width: r.width, height: r.height })
    }
  }
  return rectOverlapsAny(target, obstacles)
}

export function BrowserPanel(): ReactElement {
  const bridge = window.xerxes
  const page = useRef<HTMLDivElement>(null)
  const [state, setState] = useState<BrowserStateView | null>(null)
  const [address, setAddress] = useState('')
  const [, setEditing] = useState(false)
  const editingRef = useRef(false)
  const markEditing = (next: boolean): void => {
    editingRef.current = next
    setEditing(next)
  }
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
    void bridge.browser('state').then(value => { const next = browserStateOf(value); if (alive && next) { setState(next); if (!editingRef.current) setAddress(displayUrl(next.url)) } }).catch(() => {})
    void loadPorts().catch(() => {})
    const off = bridge.onBrowserState?.(value => { const next = browserStateOf(value); if (next) { setState(next); setAddress(current => editingRef.current ? current : displayUrl(next.url)) } })
    return () => { alive = false; off?.() }
  }, [])

  // Keep the native page exactly over our box; hide it when occluded by another
  // window view OR any overlapping in-page menu/popover/dialog (Electron's native
  // WebContentsView paints above all DOM layers and would otherwise clip menus).
  useLayoutEffect(() => {
    const box = page.current
    if (!box || !bridge.browser) return
    let occluded = typeof document !== 'undefined' && document.documentElement.hasAttribute('data-occluded')
    let lastSent = ''
    let raf = 0
    const sync = (force = false) => {
      raf = 0
      const rect = box.getBoundingClientRect()
      const hidden = occluded || rect.width <= 0 || rect.height <= 0 || overlayCoversBox(box, rect)
      const payload: BrowserBoundsRect | null = hidden
        ? null
        : {
            x: Math.round(rect.left),
            y: Math.round(rect.top),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
          }
      const key = payload ? `${payload.x},${payload.y},${payload.width},${payload.height}` : 'null'
      if (!force && key === lastSent) return
      lastSent = key
      void bridge.browser!('bounds', { rect: payload }).catch(() => {})
    }
    const schedule = () => {
      if (raf) return
      raf = requestAnimationFrame(() => sync(false))
    }
    const onWake = () => {
      if (typeof document !== 'undefined' && document.hidden) return
      sync(true)
    }
    sync(true)
    // Re-check once after initial layout/tab-transition frame settles.
    schedule()
    const observer = new ResizeObserver(schedule)
    observer.observe(box)
    const mutationObserver = typeof MutationObserver !== 'undefined'
      ? new MutationObserver(schedule)
      : null
    if (mutationObserver && document.body) {
      mutationObserver.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['open', 'hidden', 'data-context-full', 'data-review'],
      })
    }
    window.addEventListener('resize', schedule)
    window.addEventListener('focus', onWake)
    document.addEventListener('visibilitychange', onWake)
    const offOccluded = bridge.onOccluded?.(next => {
      occluded = next
      sync(!next)
    })
    return () => {
      if (raf) cancelAnimationFrame(raf)
      observer.disconnect()
      mutationObserver?.disconnect()
      window.removeEventListener('resize', schedule)
      window.removeEventListener('focus', onWake)
      document.removeEventListener('visibilitychange', onWake)
      offOccluded?.()
      lastSent = 'null'
      void bridge.browser!('bounds', { rect: null }).catch(() => {})
    }
  }, [])

  const run = async (work: () => Promise<unknown>): Promise<void> => {
    setError('')
    try { await work() } catch (failure) { setError(desktopError(failure)) }
  }
  const go = () => run(async () => {
    markEditing(false)
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
        onFocus={event => { markEditing(true); event.currentTarget.select() }} onBlur={() => markEditing(false)} onChange={event => setAddress(event.target.value)} />
      <button type="button" aria-label="Ports" title="Forwarded ports" aria-pressed={showPorts} onClick={() => { setShowPorts(value => !value); void loadPorts().catch(() => {}) }}><Icon name="tree" size={13} /></button>
      <button type="button" aria-label="Open in your browser" title="Open in your browser" disabled={!state?.url} onClick={() => void run(() => bridge.browser!('external'))}><Icon name="expand" size={13} /></button>
    </form>
    <p className={`browser-panel__via${via ? ' is-remote' : ''}`} role="status">
      <Icon name={via ? 'cloud' : 'laptop'} size={12} />
      {via ? <>Traffic leaves from <strong>{via}</strong>. localhost here is that machine.</> : <>Traffic leaves from {localMachineLabel()}.</>}
    </p>
    {error && <p className="browser-panel__error" role="alert">{error}</p>}
    {showPorts && <section className="browser-panel__ports" aria-label="Forwarded ports">
      {via ? <form onSubmit={event => { event.preventDefault(); void forward() }}>
        <input aria-label="Remote port" inputMode="numeric" placeholder={`Port on ${via}`} value={portInput} onChange={event => setPortInput(event.target.value.replace(/[^0-9]/g, '').slice(0, 5))} />
        <button type="submit" className="btn btn--solid" disabled={!portInput}>Forward</button>
      </form> : <p className="studio-muted">Local workspace: ports are already on {localMachineLabel()}.</p>}
      {ports.ports.map(row => <div key={row.remote} className="browser-panel__port" data-state={row.state}>
        <span>{row.remote} → <code>localhost:{row.local}</code></span>
        <span className="browser-panel__portstate">{row.state === 'failed' ? row.error ?? 'Failed' : row.state === 'ready' ? 'Ready' : 'Starting'}</span>
        <button type="button" className="btn" disabled={row.state !== 'ready'} onClick={() => void run(async () => { setAddress(`localhost:${row.remote}`); await bridge.browser!('open', { url: `localhost:${row.remote}` }) })}>Open</button>
        <CopyButton text={`http://localhost:${row.local}/`} label="Copy URL" />
        <button type="button" className="btn btn--ghost" onClick={() => void run(async () => { await bridge.browser!('unforward', { remote: row.remote }); await loadPorts() })}>Stop</button>
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
