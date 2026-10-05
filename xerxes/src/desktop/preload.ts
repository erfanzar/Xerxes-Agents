// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Preload bridge: the shared `window.xerxes` API (bridgeApi.ts) over Electron
 * IPC. `ipcRenderer` never leaves this file.
 *
 * Bundled as CommonJS (`format: 'cjs'` in buildDesktop.ts) because the
 * package is `"type": "module"` and sandboxed preloads only load CJS.
 */

import { contextBridge, ipcRenderer, webFrame } from 'electron'

import { createBridge } from './bridgeApi.js'

contextBridge.exposeInMainWorld('xerxes', createBridge({
  invoke: (channel, ...args) => ipcRenderer.invoke(channel, ...args),
  on(channel, listener) {
    const forward = (_event: Electron.IpcRendererEvent, value: unknown): void => listener(value)
    ipcRenderer.on(channel, forward)
    return () => { ipcRenderer.removeListener(channel, forward) }
  },
  setZoomFactor: factor => webFrame.setZoomFactor(factor),
}))
