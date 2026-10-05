// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The `window.xerxes` bridge, independent of how it reaches its host. The
 * desktop's preload runs it over Electron IPC; the VS Code extension runs the
 * same object over the webview message channel, so both shells expose one
 * API with the same validation. Everything crossing is validated and cloned.
 */

/** How the bridge reaches its host: request/response by channel, and pushed events. */
export interface BridgeTransport {
  invoke(channel: string, ...args: unknown[]): Promise<unknown>
  /** Subscribe to pushed values on a channel; returns the unsubscribe function. */
  on(channel: string, listener: (value: unknown) => void): () => void
  setZoomFactor(factor: number): void
}

const METHOD = /^[A-Za-z0-9_.]{1,128}$/

const EVENT_CHANNEL = 'daemon:event'
const CALL_CHANNEL = 'daemon:call'
const WORKSPACE_CHANNEL = 'desktop:choose-workspace'
const USE_WORKSPACE_CHANNEL = 'desktop:use-workspace'
const WORKSPACE_STATE_CHANNEL = 'desktop:workspace'
const NOTIFICATIONS_CHANNEL = 'native:notifications:set'
const LOGIN_ITEM_CHANNEL = 'native:login-item:get'
const LOGIN_ITEM_SET_CHANNEL = 'native:login-item:set'
const OPEN_PRESET_PATH_CHANNEL = 'native:preset:open-path'

function cleanParams(params: unknown): Record<string, unknown> {
  if (params === undefined || params === null) return {}
  if (typeof params !== 'object' || Array.isArray(params)) {
    throw new TypeError('params must be an object')
  }
  const out: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(params as Record<string, unknown>)) {
    if (!key || key.length > 256) throw new TypeError('invalid param key')
    out[key] = value === undefined ? null : structuredClone(value)
  }
  return out
}

function cleanEvent(frame: unknown): { type: string; payload: Record<string, unknown> } | null {
  if (!frame || typeof frame !== 'object') return null
  const { type, payload } = frame as { type?: unknown; payload?: unknown }
  if (typeof type !== 'string' || !type || type.length > 64) return null
  try {
    return {
      type,
      payload:
        payload && typeof payload === 'object'
          ? Object.freeze(structuredClone(payload) as Record<string, unknown>)
          : Object.freeze({}),
    }
  } catch {
    return null
  }
}

export function createBridge(transport: BridgeTransport) {
  return {
  /** UI scale: zoom this window's page (0.5–3). */
  setZoomFactor(factor: unknown): void {
    if (typeof factor !== 'number' || !Number.isFinite(factor) || factor < 0.5 || factor > 3) throw new TypeError('invalid zoom factor')
    transport.setZoomFactor(factor)
  },
  getWindowChrome(): Promise<{ trafficLights: boolean }> { return transport.invoke('desktop:window-chrome') as Promise<{ trafficLights: boolean }> },
  onWindowChrome(handler: (state: { trafficLights: boolean }) => void): () => void {
    const listener = (state: unknown) => {
      if (state && typeof state === 'object' && 'trafficLights' in state && typeof state.trafficLights === 'boolean') handler({ trafficLights: state.trafficLights })
    }
    return transport.on('desktop:window-chrome', listener)
  },
  /** Menu commands the main process can only express through the shell. */
  onMenuCommand(handler: (command: string) => void): () => void {
    const listener = (command: unknown) => {
      if (typeof command === 'string' && command && command.length <= 64) handler(command)
    }
    return transport.on('desktop:menu', listener)
  },
  getResumeSession(): Promise<string | null> {
    return transport.invoke('desktop:resume') as Promise<string | null>
  },
  getContextScope(): Promise<string> { return transport.invoke('desktop:context-scope') as Promise<string> },
  getContexts(): Promise<unknown[]> { return transport.invoke('desktop:contexts') as Promise<unknown[]> },
  activateContext(id: number, sessionId?: string): Promise<void> {
    if (!Number.isSafeInteger(id) || sessionId !== undefined && (typeof sessionId !== 'string' || !/^[a-zA-Z0-9_-]{1,256}$/.test(sessionId))) return Promise.reject(new Error('Invalid workspace context'))
    return transport.invoke('desktop:activate-context', id, sessionId) as Promise<void>
  },
  voice(action: string, params: unknown = {}): Promise<unknown> {
    if (!['check', 'transcribe', 'cancel'].includes(action))
      return Promise.reject(new Error('Unknown dictation action'))
    return transport.invoke('desktop:voice', action, cleanParams(params))
  },
  /** The app's own update: check, install (after the person agrees), skip, dismiss, open-release, state. */
  appUpdate(action: string): Promise<unknown> {
    if (!['state', 'check', 'install', 'skip', 'dismiss', 'open-release'].includes(action)) return Promise.reject(new Error('Unknown update action'))
    return transport.invoke('desktop:app-update', action)
  },
  onAppUpdate(handler: unknown): () => void {
    if (typeof handler !== 'function') throw new TypeError('handler must be a function')
    const listener = (state: unknown): void => {
      try { (handler as (state: unknown) => void)(state) } catch { /* a renderer fault must not break the bridge */ }
    }
    return transport.on('desktop:app-update-state', listener)
  },
  /** Fires when an SSH conversation's prompts start or stop running through this computer. */
  onProviderRelay(handler: unknown): () => void {
    if (typeof handler !== 'function') throw new TypeError('handler must be a function')
    const listener = (): void => {
      try { (handler as () => void)() } catch { /* a renderer fault must not break the bridge */ }
    }
    return transport.on('desktop:provider-relay', listener)
  },
  remote(action: string, params: unknown = {}): Promise<unknown> {
    if (
      !['list', 'hosts', 'browse', 'save', 'remove', 'connect', 'cancel', 'status', 'provider-activity', 'provider-use-local', 'provider-review', 'provider-share', 'provider-revoke'].includes(action)
    )
      return Promise.reject(new Error('Unknown remote action'))
    return transport.invoke('desktop:remote', action, cleanParams(params))
  },
  call<T = Record<string, unknown>>(method: unknown, params?: unknown): Promise<T> {
    if (typeof method !== 'string' || !METHOD.test(method)) {
      return Promise.reject(new TypeError(`invalid rpc method: ${String(method).slice(0, 32)}`))
    }
    let clean: Record<string, unknown>
    try {
      clean = cleanParams(params)
    } catch (error) {
      return Promise.reject(error instanceof Error ? error : new Error(String(error)))
    }
    return transport.invoke(CALL_CHANNEL, method, clean) as Promise<T>
  },

  onEvent(handler: unknown): () => void {
    if (typeof handler !== 'function') throw new TypeError('handler must be a function')
    const listener = (frame: unknown): void => {
      const clean = cleanEvent(frame)
      if (clean) {
        try {
          ;(handler as (event: unknown) => void)(clean)
        } catch {
          // Renderer handler faults must not break the bridge listener.
        }
      }
    }
    return transport.on(EVENT_CHANNEL, listener)
  },

  /** Open a folder independently, preserving the current workspace connection. */
  chooseWorkspace(): Promise<unknown> {
    return transport.invoke(WORKSPACE_CHANNEL) as Promise<unknown>
  },

  getWorkspaceDirectories(): Promise<string[]> {
    return transport.invoke('desktop:workspaces') as Promise<string[]>
  },

  openWorkspaceWindow(dir?: unknown, resumeSessionId?: unknown, options?: unknown): Promise<unknown> {
    if (dir !== undefined && (typeof dir !== 'string' || !dir || /[\x00-\x1f]/.test(dir))) return Promise.reject(new TypeError('invalid workspace dir'))
    if (resumeSessionId !== undefined && (typeof resumeSessionId !== 'string' || !resumeSessionId || resumeSessionId.length > 256 || /[\x00-\x1f]/.test(resumeSessionId))) return Promise.reject(new TypeError('invalid session identity'))
    const fresh = options !== null && typeof options === 'object' && (options as { fresh?: unknown }).fresh === true
    const existingOnly = options !== null && typeof options === 'object' && (options as { existingOnly?: unknown }).existingOnly === true
    return transport.invoke('desktop:new-window', dir, resumeSessionId, fresh || undefined, existingOnly || undefined) as Promise<unknown>
  },
  /** macOS: blur what is behind the window (native material) or show it clear. */
  setWindowBlur(on: unknown): void {
    if (typeof on !== 'boolean') throw new TypeError('invalid window blur')
    void transport.invoke('desktop:window-blur', on)
  },
  /** A bundled background as a data URL (so its blur can be baked). */
  backgroundData(name: unknown): Promise<string> {
    if (typeof name !== 'string' || !/^backgrounds\/background-\d+\.jpg$/.test(name)) return Promise.reject(new TypeError('invalid background'))
    return transport.invoke('desktop:background-data', name) as Promise<string>
  },
  /** Asked once as the page starts: is another workspace view covering it? */
  isOccluded(): Promise<boolean> {
    return transport.invoke('desktop:occluded') as Promise<boolean>
  },
  /** Whether another workspace view covers this page (the window's base page only). */
  onOccluded(handler: (occluded: boolean) => void): () => void {
    const listener = (occluded: unknown) => { if (typeof occluded === 'boolean') handler(occluded) }
    return transport.on('desktop:occluded', listener)
  },
  /** True once, for a window opened to start a fresh task. */
  startsFresh(): Promise<boolean> {
    return transport.invoke('desktop:starts-fresh') as Promise<boolean>
  },

  /** Enter a workspace by absolute folder path (sidebar header click). */
  useWorkspace(dir: unknown, resumeSessionId?: unknown): Promise<unknown> {
    if (typeof dir !== 'string' || !dir)
      return Promise.reject(new TypeError('invalid workspace dir'))
    if (resumeSessionId !== undefined && (typeof resumeSessionId !== 'string' || !resumeSessionId || resumeSessionId.length > 256 || /[\x00-\x1f]/.test(resumeSessionId)))
      return Promise.reject(new TypeError('invalid resume session id'))
    return transport.invoke(USE_WORKSPACE_CHANNEL, dir, resumeSessionId) as Promise<unknown>
  },

  /** The saved workspace folder, or null while the gate is showing. */
  getWorkspace(): Promise<string | null> {
    return transport.invoke(WORKSPACE_STATE_CHANNEL) as Promise<string | null>
  },

  /** Whether the shell pings for needs-input / task-finished moments. */
  setNotifications(on: unknown): Promise<boolean> {
    if (typeof on !== 'boolean')
      return Promise.reject(new TypeError('notifications expects a boolean'))
    return transport.invoke(NOTIFICATIONS_CHANNEL, on) as Promise<boolean>
  },

  /** Current launch-at-login registration. */
  getLoginItem(): Promise<boolean> {
    return transport.invoke(LOGIN_ITEM_CHANNEL) as Promise<boolean>
  },

  /** Register or unregister the app as a login item; returns the new state. */
  setLoginItem(on: unknown): Promise<boolean> {
    if (typeof on !== 'boolean')
      return Promise.reject(new TypeError('login item expects a boolean'))
    return transport.invoke(LOGIN_ITEM_SET_CHANNEL, on) as Promise<boolean>
  },

  /** VS Code: open a workspace file in the editor, at a line when given. */
  openInEditor(path: unknown, line?: unknown): Promise<boolean> {
    if (typeof path !== 'string' || !path || path.length > 4096 || /[\x00-\x1f]/.test(path)) return Promise.reject(new TypeError('invalid file path'))
    if (line !== undefined && (!Number.isSafeInteger(line) || (line as number) < 1)) return Promise.reject(new TypeError('invalid line'))
    return transport.invoke('host:open-file', path, line) as Promise<boolean>
  },
  /** VS Code: show a file's uncommitted changes in the diff editor. */
  openDiff(path: unknown): Promise<boolean> {
    if (typeof path !== 'string' || !path || path.length > 4096 || /[\x00-\x1f]/.test(path)) return Promise.reject(new TypeError('invalid file path'))
    return transport.invoke('host:open-diff', path) as Promise<boolean>
  },
  /** VS Code: pick workspace files to add as context; workspace-relative paths. */
  pickFiles(): Promise<string[]> { return transport.invoke('host:pick-files') as Promise<string[]> },
  /** VS Code: show the Source Control view. */
  showSourceControl(): Promise<boolean> { return transport.invoke('host:source-control') as Promise<boolean> },

  /** Reveal one daemon-resolved user preset directory. Main re-checks containment. */
  openPath(path: unknown): Promise<boolean> {
    if (typeof path !== 'string' || !path)
      return Promise.reject(new TypeError('invalid preset path'))
    return transport.invoke(OPEN_PRESET_PATH_CHANNEL, path) as Promise<boolean>
  },
  }
}

export type XerxesBridgeApi = ReturnType<typeof createBridge>
