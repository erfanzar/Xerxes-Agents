// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Preload bridge. Two capabilities cross: `call(method, params)` and
 * `onEvent(handler)`. Everything is validated here and cloned across the
 * context boundary; `ipcRenderer` never leaves this file.
 *
 * Bundled as CommonJS (`format: 'cjs'` in buildDesktop.ts) because the
 * package is `"type": "module"` and sandboxed preloads only load CJS.
 */

import { contextBridge, ipcRenderer } from 'electron'

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

const bridge = {
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
    return ipcRenderer.invoke(CALL_CHANNEL, method, clean) as Promise<T>
  },

  onEvent(handler: unknown): () => void {
    if (typeof handler !== 'function') throw new TypeError('handler must be a function')
    const listener = (_event: Electron.IpcRendererEvent, frame: unknown): void => {
      const clean = cleanEvent(frame)
      if (clean) {
        try {
          ;(handler as (event: unknown) => void)(clean)
        } catch {
          // Renderer handler faults must not break the bridge listener.
        }
      }
    }
    ipcRenderer.on(EVENT_CHANNEL, listener)
    return () => ipcRenderer.removeListener(EVENT_CHANNEL, listener)
  },

  /** Pick a workspace folder; the shell switches daemons and reloads. */
  chooseWorkspace(): Promise<unknown> {
    return ipcRenderer.invoke(WORKSPACE_CHANNEL) as Promise<unknown>
  },

  /** Enter a workspace by absolute folder path (sidebar header click). */
  useWorkspace(dir: unknown): Promise<unknown> {
    if (typeof dir !== 'string' || !dir) return Promise.reject(new TypeError('invalid workspace dir'))
    return ipcRenderer.invoke(USE_WORKSPACE_CHANNEL, dir) as Promise<unknown>
  },

  /** The saved workspace folder, or null while the gate is showing. */
  getWorkspace(): Promise<string | null> {
    return ipcRenderer.invoke(WORKSPACE_STATE_CHANNEL) as Promise<string | null>
  },

  /** Whether the shell pings for needs-input / task-finished moments. */
  setNotifications(on: unknown): Promise<boolean> {
    if (typeof on !== 'boolean') return Promise.reject(new TypeError('notifications expects a boolean'))
    return ipcRenderer.invoke(NOTIFICATIONS_CHANNEL, on) as Promise<boolean>
  },

  /** Current launch-at-login registration. */
  getLoginItem(): Promise<boolean> {
    return ipcRenderer.invoke(LOGIN_ITEM_CHANNEL) as Promise<boolean>
  },

  /** Register or unregister the app as a login item; returns the new state. */
  setLoginItem(on: unknown): Promise<boolean> {
    if (typeof on !== 'boolean') return Promise.reject(new TypeError('login item expects a boolean'))
    return ipcRenderer.invoke(LOGIN_ITEM_SET_CHANNEL, on) as Promise<boolean>
  },

  /** Reveal one daemon-resolved user preset directory. Main re-checks containment. */
  openPath(path: unknown): Promise<boolean> {
    if (typeof path !== 'string' || !path) return Promise.reject(new TypeError('invalid preset path'))
    return ipcRenderer.invoke(OPEN_PRESET_PATH_CHANNEL, path) as Promise<boolean>
  },
}

contextBridge.exposeInMainWorld('xerxes', bridge)
