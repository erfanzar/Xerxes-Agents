// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { BrowserWindow, ipcMain } from 'electron'

import type { DaemonRpc } from './daemon.js'

const METHOD_PATTERN = /^[A-Za-z0-9_.]{1,128}$/

/**
 * The live daemon behind the one `daemon:call` channel. Swappable so the
 * shell can retarget to another workspace's daemon without re-registering
 * ipc handlers (a second `handle()` on the same channel throws).
 */
let active: DaemonRpc | null = null
let registered = false
let forward: ((type: string, payload: Record<string, unknown>) => void) | null = null
let eventObserver: ((type: string, payload: Record<string, unknown>) => void) | null = null

/**
 * Main-process-side hook on the daemon event stream (native notifications
 * ride it). At most one observer; the pipe stays the single forward path.
 */
export function setDaemonEventObserver(
  observer: ((type: string, payload: Record<string, unknown>) => void) | null,
): void {
  eventObserver = observer
}

/** Point the bridge at `next`, moving the event pipe off the previous one. */
export function attachDaemon(next: DaemonRpc): void {
  if (forward && active) active.offEvent(forward)
  active = next
  forward = (type, payload) => {
    eventObserver?.(type, payload)
    for (const window of BrowserWindow.getAllWindows()) {
      if (!window.isDestroyed()) window.webContents.send('daemon:event', { type, payload })
    }
  }
  next.onEvent(forward)
}

/**
 * The whole renderer↔daemon seam. The renderer may only send a method name
 * plus a plain-data params object on one channel; events flow out on another.
 * `ipcRenderer` itself never crosses the bridge.
 */
export function registerDaemonBridge(daemon: DaemonRpc): void {
  if (!registered) {
    registered = true
    ipcMain.handle('daemon:call', (_event, method: unknown, params: unknown) => {
      const name = typeof method === 'string' && METHOD_PATTERN.test(method) ? method : ''
      if (!name) return Promise.reject(new TypeError(`invalid rpc method: ${String(method).slice(0, 32)}`))
      if (params !== undefined && params !== null && (typeof params !== 'object' || Array.isArray(params))) {
        return Promise.reject(new TypeError('params must be an object'))
      }
      const current = active
      if (!current) return Promise.reject(new Error('daemon bridge has no active connection'))
      return current.call(name, (params ?? {}) as Record<string, unknown>)
    })
  }
  attachDaemon(daemon)
}
