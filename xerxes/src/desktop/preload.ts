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
import { createChannelDispatcher } from './main/preloadDispatcher.js'

const dispatcher = createChannelDispatcher({
  on(channel, listener) { ipcRenderer.on(channel, listener) },
  removeListener(channel, listener) { ipcRenderer.removeListener(channel, listener) },
})

contextBridge.exposeInMainWorld('xerxes', createBridge({
  invoke: (channel, ...args) => ipcRenderer.invoke(channel, ...args),
  on: (channel, listener) => dispatcher.on(channel, listener),
  setZoomFactor: factor => webFrame.setZoomFactor(factor),
}))
