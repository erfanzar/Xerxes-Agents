// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Shared IPC channel multiplexer for the Electron preload bridge.
 *
 * Registers at most one `ipcRenderer.on(channel, ...)` handler per channel no
 * matter how many renderer components subscribe (avoiding duplicate Electron IPC
 * listeners and MaxListenersExceeded warnings), automatically removes the
 * underlying listener when the last subscriber unsubscribes, and transparently
 * unpacks `daemon:event-batch` micro-batches for `daemon:event` subscribers.
 */

import { DAEMON_EVENT_BATCH_CHANNEL, DAEMON_EVENT_CHANNEL } from './ipcEventBatcher.js'

export interface IpcRendererLike {
  on(channel: string, listener: (_event: unknown, value: unknown) => void): void
  removeListener(channel: string, listener: (_event: unknown, value: unknown) => void): void
}

export interface ChannelDispatcher {
  on(channel: string, listener: (value: unknown) => void): () => void
  /** Number of active underlying ipcRenderer channel bindings (useful for leak tests). */
  bindingCount(): number
}

export function createChannelDispatcher(ipc: IpcRendererLike): ChannelDispatcher {
  const listenersByChannel = new Map<string, Set<(value: unknown) => void>>()
  const bindings = new Map<string, (_event: unknown, value: unknown) => void>()

  const dispatch = (channel: string, value: unknown): void => {
    const set = listenersByChannel.get(channel)
    if (!set || set.size === 0) return
    for (const listener of [...set]) {
      try {
        listener(value)
      } catch {
        // One subscriber fault must not drop the event for sibling subscribers.
      }
    }
  }

  const ensureBinding = (channel: string): void => {
    if (!bindings.has(channel)) {
      const handler = (_event: unknown, value: unknown): void => dispatch(channel, value)
      bindings.set(channel, handler)
      ipc.on(channel, handler)
    }
    if (channel === DAEMON_EVENT_CHANNEL && !bindings.has(DAEMON_EVENT_BATCH_CHANNEL)) {
      const batchHandler = (_event: unknown, value: unknown): void => {
        if (!Array.isArray(value)) return
        for (const frame of value) dispatch(DAEMON_EVENT_CHANNEL, frame)
      }
      bindings.set(DAEMON_EVENT_BATCH_CHANNEL, batchHandler)
      ipc.on(DAEMON_EVENT_BATCH_CHANNEL, batchHandler)
    }
  }

  const releaseBindingIfEmpty = (channel: string): void => {
    const set = listenersByChannel.get(channel)
    if (set && set.size > 0) return
    listenersByChannel.delete(channel)
    const handler = bindings.get(channel)
    if (handler) {
      bindings.delete(channel)
      ipc.removeListener(channel, handler)
    }
    if (channel === DAEMON_EVENT_CHANNEL) {
      const batchHandler = bindings.get(DAEMON_EVENT_BATCH_CHANNEL)
      if (batchHandler) {
        bindings.delete(DAEMON_EVENT_BATCH_CHANNEL)
        ipc.removeListener(DAEMON_EVENT_BATCH_CHANNEL, batchHandler)
      }
    }
  }

  return {
    on(channel, listener) {
      let set = listenersByChannel.get(channel)
      if (!set) {
        set = new Set()
        listenersByChannel.set(channel, set)
      }
      set.add(listener)
      ensureBinding(channel)
      let active = true
      return () => {
        if (!active) return
        active = false
        listenersByChannel.get(channel)?.delete(listener)
        releaseBindingIfEmpty(channel)
      }
    },
    bindingCount: () => bindings.size,
  }
}
