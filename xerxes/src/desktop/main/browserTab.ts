// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The right panel's Browser tab: a native WebContentsView laid over the
 * panel's page area. On an SSH workspace its own in-memory session sends all
 * traffic (DNS included) through that workspace's SOCKS listener, so
 * `localhost:8080` is the remote machine's port. Nothing else in the app uses
 * that session, and no system proxy is set.
 *
 * The page is untrusted: sandboxed, no Node, no preload, no Xerxes IPC, no
 * permissions, no downloads, no popups.
 */

import { session as electronSession, shell, WebContentsView, type BrowserWindow, type Session } from 'electron'

import { browserUrl, type RemoteNetwork } from './remoteNetwork.js'

export interface BrowserState {
  readonly url: string
  readonly title: string
  readonly loading: boolean
  readonly canGoBack: boolean
  readonly canGoForward: boolean
  /** Where the page's traffic leaves from: the SSH alias, or null for this computer. */
  readonly via: string | null
  readonly error?: string
}
export interface BrowserBounds { readonly x: number; readonly y: number; readonly width: number; readonly height: number }

let partitions = 0

export class BrowserTab {
  private view: WebContentsView | null = null
  private bounds: BrowserBounds | null = null
  private shown = false
  private error: string | undefined
  private closed = false
  private generation = 0
  private routeSetup: Promise<WebContentsView> | undefined
  private proxyPort: number | undefined
  private retryUrl = ''

  constructor(
    private readonly window: BrowserWindow,
    private readonly network: () => RemoteNetwork | null,
    private readonly publish: (state: BrowserState) => void,
  ) {}

  private async ensure(): Promise<WebContentsView> {
    if (this.closed || this.window.isDestroyed()) throw new Error('The window is closed')
    if (this.routeSetup) return this.routeSetup
    const setup = this.prepare(this.generation)
    this.routeSetup = setup
    try { return await setup }
    finally { if (this.routeSetup === setup) this.routeSetup = undefined }
  }

  private async prepare(generation: number): Promise<WebContentsView> {
    const network = this.network()
    const stillCurrent = () => {
      if (generation !== this.generation || network !== this.network() || this.closed || this.window.isDestroyed()) {
        throw new Error('The Browser workspace changed; open the address again')
      }
    }
    const existing = this.view && !this.view.webContents.isDestroyed() ? this.view : null
    // The SSH process can die while the native view is still alive. Its owner
    // replaces a failed listener on demand; refresh the route before navigating.
    const port = network ? await network.socksPort() : undefined
    stillCurrent()
    // In-memory partition (no "persist:"): cookies and storage die with the tab.
    const ses: Session = existing?.webContents.session ?? electronSession.fromPartition(`xerxes-browser-${process.pid}-${++partitions}`)
    if (port !== undefined && (!existing || this.proxyPort !== port)) {
      // A single fixed SOCKS proxy, with Chromium's implicit localhost bypass
      // removed. An unavailable proxy must never make the tab connect directly.
      await ses.setProxy({ mode: 'fixed_servers', proxyRules: `socks5://127.0.0.1:${port}`, proxyBypassRules: '<-loopback>' })
      stillCurrent()
      if (existing) await ses.closeAllConnections()
      stillCurrent()
    }
    this.proxyPort = port
    if (existing) return existing
    ses.setPermissionRequestHandler((_contents, _permission, callback) => callback(false))
    ses.setPermissionCheckHandler(() => false)
    ses.on('will-download', event => event.preventDefault())
    const view = new WebContentsView({ webPreferences: {
      session: ses, sandbox: true, contextIsolation: true, nodeIntegration: false, nodeIntegrationInSubFrames: false,
      webviewTag: false, safeDialogs: true, spellcheck: false,
    } })
    const contents = view.webContents
    // WebRTC UDP isn't carried by the workspace SOCKS route.
    if (network) contents.setWebRTCIPHandlingPolicy('disable_non_proxied_udp')
    contents.setWindowOpenHandler(({ url }) => {
      // Popups open in this tab, through the same route; never in a new window.
      if (/^https?:/i.test(url)) void contents.loadURL(url)
      return { action: 'deny' }
    })
    contents.on('will-navigate', (event, url) => { if (!/^https?:/i.test(url)) event.preventDefault() })
    const active = () => this.view === view && !this.closed
    const send = () => { if (active()) this.emit() }
    contents.on('did-start-loading', () => { if (active()) { this.error = undefined; send() } })
    contents.on('did-stop-loading', send)
    contents.on('did-navigate', (_event, url) => {
      if (active() && /^https?:\/\//i.test(url)) this.retryUrl = ''
      send()
    })
    contents.on('did-navigate-in-page', send)
    contents.on('page-title-updated', send)
    contents.on('did-fail-load', (_event, code, description, url, isMainFrame) => {
      if (active() && isMainFrame && code !== -3) {
        if (/^https?:\/\//i.test(url)) this.retryUrl = url
        this.error = `${description} (${code})`
        send()
      }
    })
    contents.on('render-process-gone', () => { this.error = 'The page crashed; reload it'; send() })
    this.view = view
    this.place()
    return view
  }

  state(): BrowserState {
    const contents = this.view && !this.view.webContents.isDestroyed() ? this.view.webContents : null
    return {
      url: this.retryUrl || contents?.getURL() || '', title: contents?.getTitle() ?? '', loading: contents?.isLoading() ?? false,
      canGoBack: contents?.navigationHistory.canGoBack() ?? false, canGoForward: contents?.navigationHistory.canGoForward() ?? false,
      via: this.network()?.label ?? null, ...(this.error ? { error: this.error } : {}),
    }
  }

  private emit(): void { if (!this.closed) this.publish(this.state()) }

  async open(input: unknown): Promise<BrowserState> {
    const url = browserUrl(String(input ?? ''))
    const generation = this.generation
    this.retryUrl = url
    this.error = undefined
    try {
      const view = await this.ensure()
      if (view !== this.view || generation !== this.generation) throw new Error('The Browser workspace changed; open the address again')
      void view.webContents.loadURL(url).catch(() => { /* did-fail-load reports it */ })
    } finally {
      if (generation === this.generation) this.emit()
    }
    return this.state()
  }

  async navigate(action: 'back' | 'forward' | 'reload' | 'stop'): Promise<BrowserState> {
    const contents = this.view?.webContents
    if (action === 'reload' && this.retryUrl) return this.open(this.retryUrl)
    if (!contents || contents.isDestroyed()) return this.state()
    if (action === 'stop') { contents.stop(); return this.state() }
    if (action === 'back' && !contents.navigationHistory.canGoBack()) return this.state()
    if (action === 'forward' && !contents.navigationHistory.canGoForward()) return this.state()
    const view = await this.ensure()
    if (view !== this.view || view.webContents !== contents) return this.state()
    this.retryUrl = ''
    if (action === 'reload') contents.reload()
    else if (action === 'back') contents.navigationHistory.goBack()
    else contents.navigationHistory.goForward()
    return this.state()
  }

  /** The panel's page rectangle in window coordinates; null hides the page. */
  setBounds(value: unknown): void {
    const rect = value && typeof value === 'object' ? value as Record<string, unknown> : null
    const ok = rect && ['x', 'y', 'width', 'height'].every(key => typeof rect[key] === 'number' && Number.isFinite(rect[key] as number))
    this.bounds = ok ? { x: Math.round(rect.x as number), y: Math.round(rect.y as number), width: Math.max(0, Math.round(rect.width as number)), height: Math.max(0, Math.round(rect.height as number)) } : null
    this.shown = Boolean(this.bounds && this.bounds.width > 0 && this.bounds.height > 0)
    this.place()
  }

  private place(): void {
    const view = this.view
    if (!view || this.window.isDestroyed()) return
    if (this.shown && this.bounds) {
      // Re-adding keeps it above the workspace view that the window may have restacked.
      this.window.contentView.addChildView(view)
      view.setBounds(this.bounds)
      view.setVisible(true)
    } else view.setVisible(false)
  }

  /** Public pages can open in the system browser; a tunnelled localhost would not resolve there. */
  async openExternal(): Promise<void> {
    const url = this.view?.webContents.getURL() ?? ''
    if (!/^https?:/i.test(url)) return
    const host = new URL(url).hostname
    if (this.network() && /^(localhost|127\.|0\.0\.0\.0|\[::1\]|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/.test(host)) {
      throw new Error('This address only exists on the remote machine. Forward its port from the Ports list to open it in your browser.')
    }
    await shell.openExternal(url)
  }

  /** Drop the page and its session (used when the workspace changes or closes). */
  reset(): void {
    const view = this.view
    this.generation++
    this.routeSetup = undefined
    this.proxyPort = undefined
    this.retryUrl = ''
    this.view = null
    this.error = undefined
    if (view) {
      if (!this.window.isDestroyed()) this.window.contentView.removeChildView(view)
      if (!view.webContents.isDestroyed()) view.webContents.close()
    }
    this.emit()
  }

  close(): void {
    this.reset()
    this.closed = true
  }
}
