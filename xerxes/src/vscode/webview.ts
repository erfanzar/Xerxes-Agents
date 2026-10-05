// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Runs first in the Xerxes webview: installs `window.xerxes` over the webview
 * channel, then the desktop renderer (loaded next) starts exactly as it does
 * in the app.
 */

import { createBridge } from '../desktop/bridgeApi.js'
import { webviewTransport, type WebviewApi } from './webviewTransport.js'

declare function acquireVsCodeApi(): WebviewApi

// VS Code scales webviews with its own zoom (View → Appearance → Zoom); CSS
// zoom on the page shrank the layout inside the view instead of rescaling it.
const transport = webviewTransport(acquireVsCodeApi(), window, () => {})
;(window as unknown as { xerxes: ReturnType<typeof createBridge> }).xerxes = createBridge(transport)
// "Send selection to Xerxes": text from the editor lands in the composer the
// same way the welcome screen's prompts do.
transport.on('desktop:compose', text => {
  if (typeof text === 'string' && text) window.dispatchEvent(new CustomEvent('xerxes:add-context', { detail: text }))
})
