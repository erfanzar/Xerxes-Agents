// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ipcMain, type WebContents } from 'electron'
import { WindowConnections } from "./windowRoutes.js"
import type { DaemonRpc } from './daemon.js'

const METHOD_PATTERN = /^[A-Za-z0-9_.]{1,128}$/
const connections = new WindowConnections<DaemonRpc>()
const results = new Map<number, (method: string, result: Record<string, unknown>) => void>()
let registered = false

/** Remove only this renderer's event subscription. The caller owns its connection. */
export function detachDaemon(id: number): void {
  connections.detach(id)
  results.delete(id)
}

/** Calls and events stay inside the workspace window that owns the connection. */
export function registerDaemonBridge(target: WebContents, daemon?: DaemonRpc, observer?: (type: string, payload: Record<string, unknown>) => void, resultObserver?: (method: string, result: Record<string, unknown>) => void): void {
  if (!registered) {
    registered = true
    ipcMain.handle('daemon:call', (event, method: unknown, params: unknown) => {
      const name = typeof method === 'string' && METHOD_PATTERN.test(method) ? method : ''
      if (!name) throw new TypeError(`invalid rpc method: ${String(method).slice(0, 32)}`)
      if (params !== undefined && params !== null && (typeof params !== 'object' || Array.isArray(params))) throw new TypeError('params must be an object')
      const current = connections.get(event.sender.id)
      if (!current) throw new Error('Choose a workspace folder before using runtime features')
      if (name === 'desktop.restartRuntime') return current.restartRuntime((params as Record<string, unknown> | undefined)?.allow_legacy === true)
      return current.call(name, (params ?? {}) as Record<string, unknown>).then(result => {
        if (connections.get(event.sender.id) === current) results.get(event.sender.id)?.(name, result)
        return result
      })
    })
  }
  detachDaemon(target.id)
  if (!daemon) return
  if (resultObserver) results.set(target.id, resultObserver)
  const forward = (type: string, payload: Record<string, unknown>) => {
    if (target.isDestroyed()) return
    observer?.(type, payload)
    target.send('daemon:event', { type, payload })
  }
  connections.attach(target.id, daemon, forward)
}
