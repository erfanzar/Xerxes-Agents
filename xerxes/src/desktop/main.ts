// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Electron main process for the Xerxes desktop app.
 *
 * The renderer is a static bundle: it ships every asset it needs and reaches no
 * remote origin, which is why the window can run with `nodeIntegration` off,
 * context isolation on, and a CSP that permits only `'self'`. Wiring it to the
 * daemon is a separate pass; when it lands it goes through a preload bridge
 * rather than by relaxing any of these.
 */

import { app, BrowserWindow, shell } from 'electron'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))

/** Below this the layout has nothing left to drop; the design degrades to a single pane. */
const MINIMUM_WIDTH = 720
const MINIMUM_HEIGHT = 520

function createWindow(): BrowserWindow {
  const window = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: MINIMUM_WIDTH,
    minHeight: MINIMUM_HEIGHT,
    // The design paints its own title row, so the frame is hidden — but the
    // traffic lights stay REAL and inset into that row. Painted ones look
    // pressable and are not.
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    trafficLightPosition: { x: 14, y: 13 },
    backgroundColor: '#0a0a0d',
    // Nothing is shown until the first paint, so the window never flashes an
    // empty white rectangle before the theme applies.
    show: false,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  })

  window.once('ready-to-show', () => window.show())
  void window.loadFile(join(here, 'renderer', 'index.html'))

  // A link in agent output must not navigate the app shell out of itself.
  window.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url)
    return { action: 'deny' }
  })
  window.webContents.on('will-navigate', (event) => event.preventDefault())

  return window
}

void app.whenReady().then(() => {
  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
