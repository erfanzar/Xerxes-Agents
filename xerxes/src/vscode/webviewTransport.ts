// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { BridgeTransport } from '../desktop/bridgeApi.js'
import { hostFrame, type InvokeFrame } from './messages.js'

/** The part of the webview API the transport needs (acquireVsCodeApi()). */
export interface WebviewApi { postMessage(message: unknown): void }

/**
 * The bridge transport inside the VS Code webview: requests go to the
 * extension host by postMessage and resolve on its answer; pushed values
 * reach the channel's listeners.
 */
export function webviewTransport(api: WebviewApi, events: Pick<Window, 'addEventListener'>, zoom: (factor: number) => void): BridgeTransport {
  let next = 1
  const pending = new Map<number, { resolve(value: unknown): void; reject(error: Error): void }>()
  const listeners = new Map<string, Set<(value: unknown) => void>>()
  events.addEventListener('message', (event: Event) => {
    const frame = hostFrame((event as MessageEvent).data)
    if (!frame) return
    if (frame.kind === 'result') {
      const waiting = pending.get(frame.id)
      if (!waiting) return
      pending.delete(frame.id)
      if (frame.ok) waiting.resolve(frame.value)
      else waiting.reject(new Error(frame.error))
      return
    }
    for (const listener of listeners.get(frame.channel) ?? []) {
      try { listener(frame.value) } catch { /* a renderer fault must not break the bridge */ }
    }
  })
  return {
    invoke(channel, ...args) {
      const id = next++
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject })
        const frame: InvokeFrame = { kind: 'invoke', id, channel, args }
        api.postMessage(frame)
      })
    },
    on(channel, listener) {
      let set = listeners.get(channel)
      if (!set) { set = new Set(); listeners.set(channel, set) }
      set.add(listener)
      return () => { set.delete(listener) }
    },
    setZoomFactor: zoom,
  }
}
