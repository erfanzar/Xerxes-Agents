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

  constructor(
    private readonly window: BrowserWindow,
    private readonly network: () => RemoteNetwork | null,
    private readonly publish: (state: BrowserState) => void,
  ) {}

  private async ensure(): Promise<WebContentsView> {
    if (this.closed || this.window.isDestroyed()) throw new Error('The window is closed')
    if (this.view && !this.view.webContents.isDestroyed()) return this.view
    // In-memory partition (no "persist:"): cookies and storage die with the tab.
    const ses: Session = electronSession.fromPartition(`xerxes-browser-${process.pid}-${++partitions}`)
    const network = this.network()
    if (network) {
      const port = await network.socksPort()
      // "<-loopback>" removes Chromium's implicit localhost bypass, so localhost
      // goes through the tunnel too; socks5 resolves names on the remote side.
      await ses.setProxy({ proxyRules: `socks5://127.0.0.1:${port}`, proxyBypassRules: '<-loopback>' })
    }
    ses.setPermissionRequestHandler((_contents, _permission, callback) => callback(false))
    ses.setPermissionCheckHandler(() => false)
    ses.on('will-download', event => event.preventDefault())
    const view = new WebContentsView({ webPreferences: {
      session: ses, sandbox: true, contextIsolation: true, nodeIntegration: false, nodeIntegrationInSubFrames: false,
      webviewTag: false, safeDialogs: true, spellcheck: false,
    } })
    const contents = view.webContents
    contents.setWindowOpenHandler(({ url }) => {
      // Popups open in this tab, through the same route; never in a new window.
      if (/^https?:/i.test(url)) void contents.loadURL(url)
      return { action: 'deny' }
    })
    contents.on('will-navigate', (event, url) => { if (!/^https?:/i.test(url)) event.preventDefault() })
    const send = () => this.emit()
    contents.on('did-start-loading', () => { this.error = undefined; send() })
    contents.on('did-stop-loading', send)
    contents.on('did-navigate', send)
    contents.on('did-navigate-in-page', send)
    contents.on('page-title-updated', send)
    contents.on('did-fail-load', (_event, code, description, _url, isMainFrame) => {
      if (isMainFrame && code !== -3) { this.error = `${description} (${code})`; send() }
    })
    contents.on('render-process-gone', () => { this.error = 'The page crashed; reload it'; send() })
    this.view = view
    this.place()
    return view
  }

  state(): BrowserState {
    const contents = this.view && !this.view.webContents.isDestroyed() ? this.view.webContents : null
    return {
      url: contents?.getURL() ?? '', title: contents?.getTitle() ?? '', loading: contents?.isLoading() ?? false,
      canGoBack: contents?.navigationHistory.canGoBack() ?? false, canGoForward: contents?.navigationHistory.canGoForward() ?? false,
      via: this.network()?.label ?? null, ...(this.error ? { error: this.error } : {}),
    }
  }

  private emit(): void { if (!this.closed) this.publish(this.state()) }

  async open(input: unknown): Promise<BrowserState> {
    const url = browserUrl(String(input ?? ''))
    const view = await this.ensure()
    this.error = undefined
    view.webContents.loadURL(url).catch(() => { /* did-fail-load reports it */ })
    this.emit()
    return this.state()
  }

  navigate(action: 'back' | 'forward' | 'reload' | 'stop'): BrowserState {
    const contents = this.view?.webContents
    if (contents && !contents.isDestroyed()) {
      if (action === 'back' && contents.navigationHistory.canGoBack()) contents.navigationHistory.goBack()
      else if (action === 'forward' && contents.navigationHistory.canGoForward()) contents.navigationHistory.goForward()
      else if (action === 'reload') contents.reload()
      else if (action === 'stop') contents.stop()
    }
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
    const url = this.state().url
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
