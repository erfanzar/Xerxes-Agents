// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The extension-host side of one Xerxes panel: answers the bridge channels
 * the desktop's Electron main process answers (see desktop/bridgeApi.ts),
 * against the VS Code workspace. Pure of the `vscode` module so it is tested
 * headlessly; extension.ts supplies the ports.
 */

import { notificationFor } from '../desktop/main/notify.js'
import { invokeFrame, type PushFrame, type ResultFrame } from './messages.js'

/** The runtime connection (desktop/main/daemon.ts DaemonRpc satisfies it). */
export interface PanelDaemon {
  call(method: string, params?: Record<string, unknown>): Promise<Record<string, unknown>>
  onEvent(handler: (type: string, payload: Record<string, unknown>) => void): void
  offEvent(handler: (type: string, payload: Record<string, unknown>) => void): void
  restartRuntime?(allowLegacy?: boolean, force?: boolean): Promise<Record<string, unknown>>
  dispose(): void
}

/** What the panel needs from VS Code. */
export interface PanelPorts {
  /** The folder this panel works in (the VS Code workspace folder), or null when none is open. */
  readonly workspace: string | null
  readonly workspaceFolders: () => readonly string[]
  /** `vscode.env.remoteName`: set when VS Code runs this extension on a remote host (Remote-SSH, WSL…). */
  readonly remoteName: string | undefined
  chooseFolder(): Promise<string | null>
  /** Open another folder in VS Code (a new window); the runtime follows the window's folder. */
  openFolder(dir: string): Promise<void>
  /** Open another Xerxes panel, on a session or fresh. */
  openPanel(options: { readonly sessionId?: string; readonly fresh?: boolean }): void
  /** Show a daemon-resolved file or folder in VS Code. */
  revealPath(path: string): Promise<boolean>
  /** A bundled background (renderer/backgrounds/…) as a data URL. */
  backgroundData(name: string): Promise<string>
  /** Persisted per workspace: the session this panel last showed. */
  readonly memory: { get(key: string): string | undefined; set(key: string, value: string | undefined): void }
  notify(message: string): void
  /** Open a workspace file (absolute or relative) in the editor. */
  openFile(path: string, line?: number): Promise<boolean>
  /** Show a file's uncommitted changes in VS Code's diff editor. */
  openDiff(path: string): Promise<boolean>
  /** Quick-pick workspace files; workspace-relative paths. */
  pickFiles(): Promise<string[]>
  showSourceControl(): Promise<boolean>
  /** Whether the panel is visible now (notifications only fire while it is not). */
  visible(): boolean
}

export interface PanelOptions {
  readonly sessionId?: string
  readonly fresh?: boolean
}

const RESUME_KEY = 'xerxes.resumeSession'
const SESSION_ID = /^[a-zA-Z0-9_-]{1,256}$/

export class PanelHost {
  private daemon: PanelDaemon | null = null
  private session: string | null
  private fresh: boolean
  private notifications = true
  private readonly forward = (type: string, payload: Record<string, unknown>): void => {
    this.post({ kind: 'push', channel: 'daemon:event', value: { type, payload } })
    // The same moments the desktop pings for, only while the panel is out of sight.
    const note = this.notifications && !this.ports.visible() ? notificationFor({ type, payload }) : null
    if (note) this.ports.notify(`Xerxes — ${note.title}: ${note.body}`)
  }

  constructor(
    private readonly ports: PanelPorts,
    private readonly post: (frame: ResultFrame | PushFrame) => void,
    private readonly connect: (projectDir: string) => PanelDaemon,
    options: PanelOptions = {},
  ) {
    this.session = options.sessionId ?? (options.fresh ? null : ports.memory.get(RESUME_KEY) ?? null)
    this.fresh = options.fresh === true
  }

  /** One frame from the webview. Invalid frames are ignored; every request gets an answer. */
  async receive(value: unknown): Promise<void> {
    const frame = invokeFrame(value)
    if (!frame) return
    try {
      this.post({ kind: 'result', id: frame.id, ok: true, value: await this.answer(frame.channel, frame.args) })
    } catch (error) {
      this.post({ kind: 'result', id: frame.id, ok: false, error: error instanceof Error ? error.message : String(error) })
    }
  }

  dispose(): void {
    if (this.daemon) { this.daemon.offEvent(this.forward); this.daemon.dispose(); this.daemon = null }
  }

  private runtime(): PanelDaemon {
    if (this.daemon) return this.daemon
    if (!this.ports.workspace) throw new Error('Open a folder in VS Code to use Xerxes.')
    this.daemon = this.connect(this.ports.workspace)
    this.daemon.onEvent(this.forward)
    return this.daemon
  }

  private async answer(channel: string, args: readonly unknown[]): Promise<unknown> {
    switch (channel) {
      case 'daemon:call': return this.call(args[0], args[1])
      case 'desktop:workspace': return this.ports.workspace
      case 'desktop:workspaces': return [...this.ports.workspaceFolders()]
      case 'desktop:resume': return this.session
      case 'desktop:starts-fresh': { const fresh = this.fresh; this.fresh = false; return fresh }
      // The window's folder is the workspace; a remote one runs under Remote-SSH,
      // so to the renderer this is always the local scope of that host.
      case 'desktop:context-scope': return 'local'
      case 'desktop:contexts': return []
      case 'desktop:activate-context': throw new Error('Switch folders with File → Open Folder in VS Code.')
      case 'desktop:choose-workspace': {
        const dir = await this.ports.chooseFolder()
        if (dir) await this.ports.openFolder(dir)
        return dir
      }
      case 'desktop:use-workspace': {
        const dir = typeof args[0] === 'string' ? args[0] : ''
        if (!dir) throw new Error('Choose a folder.')
        if (dir === this.ports.workspace) return dir
        await this.ports.openFolder(dir)
        return null
      }
      case 'desktop:new-window': {
        const [dir, sessionId, fresh, existingOnly] = args
        if (existingOnly === true) return null
        if (typeof dir === 'string' && dir !== this.ports.workspace) { await this.ports.openFolder(dir); return dir }
        this.ports.openPanel({ ...(typeof sessionId === 'string' && SESSION_ID.test(sessionId) ? { sessionId } : {}), ...(fresh === true ? { fresh: true } : {}) })
        return this.ports.workspace
      }
      case 'desktop:window-chrome': return { trafficLights: false }
      case 'desktop:occluded': return false
      case 'desktop:window-blur': return null
      case 'desktop:background-data': {
        const name = args[0]
        if (typeof name !== 'string' || !/^backgrounds\/background-\d+\.jpg$/.test(name)) throw new Error('invalid background')
        return this.ports.backgroundData(name)
      }
      case 'native:notifications:set': this.notifications = args[0] === true; return this.notifications
      // VS Code itself starts with the system; extensions do not register login items.
      case 'native:login-item:get': return false
      case 'native:login-item:set': return false
      case 'native:preset:open-path': {
        const path = args[0]
        if (typeof path !== 'string' || !path) throw new Error('invalid preset path')
        return this.ports.revealPath(path)
      }
      // The extension updates through VS Code (Marketplace or a new .vsix).
      case 'desktop:app-update': return { phase: 'idle' }
      case 'desktop:voice': throw new Error('Dictation is not available in VS Code yet.')
      case 'desktop:remote': return this.remote(args[0])
      case 'host:open-file': return this.ports.openFile(String(args[0]), typeof args[1] === 'number' ? args[1] : undefined)
      case 'host:open-diff': return this.ports.openDiff(String(args[0]))
      case 'host:pick-files': return this.ports.pickFiles()
      case 'host:source-control': return this.ports.showSourceControl()
      default: throw new Error(`Unknown channel ${channel}`)
    }
  }

  private async call(method: unknown, params: unknown): Promise<unknown> {
    if (typeof method !== 'string') throw new Error('invalid rpc method')
    const runtime = this.runtime()
    const clean = params && typeof params === 'object' && !Array.isArray(params) ? params as Record<string, unknown> : {}
    if (method === 'desktop.restartRuntime') {
      if (!runtime.restartRuntime) throw new Error('This runtime connection cannot restart.')
      return runtime.restartRuntime(clean.allow_legacy === true, clean.force === true)
    }
    const result = await runtime.call(method, clean)
    // Remember the session this panel shows, so reopening VS Code returns to it.
    if ((method === 'initialize' || method === 'session.open') && result && typeof result === 'object') {
      const session = (result as { session?: { id?: unknown } }).session
      if (session && typeof session.id === 'string' && SESSION_ID.test(session.id)) {
        this.session = session.id
        this.ports.memory.set(RESUME_KEY, session.id)
      }
    }
    return result
  }

  /** Xerxes' own SSH workspaces are replaced by Remote-SSH in VS Code. */
  private remote(action: unknown): unknown {
    if (action === 'status') return { ok: true, machine: null, connected: false, connecting: false, error: '' }
    if (action === 'list') return { ok: true, machines: [] }
    if (action === 'provider-activity') return { ok: true, bound: false }
    throw new Error(this.ports.remoteName
      ? 'This VS Code window already runs on the remote host; Xerxes works there directly.'
      : 'To work on another machine, open its folder with VS Code Remote-SSH; Xerxes runs there.')
  }
}
