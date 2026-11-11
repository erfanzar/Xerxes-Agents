// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ObservationHosts } from '../hub/supervisor/abilities/observations.js'
import { observeWorkspace } from './superagentObservation.js'

/** Bounded, connection-owned reads only; no task ownership or execution authority. */
export class WorkspaceObservationCalls {
  private readonly active = new Map<object, Set<AbortController>>()
  private readonly disconnected = new WeakSet<object>()
  private count = 0
  private closed = false
  constructor(private readonly hosts: (root: string) => ObservationHosts = () => ({}), private readonly timeoutMs = 10_000) {
    if (!Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 10_000) throw new Error('Invalid observation deadline')
  }
  async read(connection: object, root: string, input: Readonly<Record<string, unknown>>): Promise<Readonly<Record<string, unknown>>> {
    if (this.closed || this.disconnected.has(connection)) throw new Error('Observation service or connection closed')
    const controllers = this.active.get(connection) ?? new Set<AbortController>()
    if (controllers.size >= 2 || this.count >= 16) throw new Error('Observation capacity reached; retry the read later')
    const controller = new AbortController()
    controllers.add(controller); this.active.set(connection, controllers); this.count += 1
    const timer = setTimeout(() => controller.abort(), this.timeoutMs)
    let cancel = (): void => {}
    const cancelled = new Promise<never>((_resolve, reject) => {
      cancel = () => reject(new Error('Observation cancelled or deadline exceeded'))
      controller.signal.addEventListener('abort', cancel, { once: true })
    })
    // Cancellation ends the caller's wait, not the host's work. Keep the slot
    // until actual settlement so an abort-ignoring host cannot accumulate reads.
    const operation = Promise.resolve().then(() => {
      controller.signal.throwIfAborted()
      return observeWorkspace(root, input, this.hosts(root), controller.signal)
    }).finally(() => {
      controllers.delete(controller); this.count -= 1
      if (controllers.size === 0 && this.active.get(connection) === controllers) this.active.delete(connection)
    })
    try {
      const result = await Promise.race([operation, cancelled])
      controller.signal.throwIfAborted()
      return result
    } finally {
      clearTimeout(timer); controller.signal.removeEventListener('abort', cancel)
    }
  }
  cancel(connection: object): void { this.disconnected.add(connection); for (const controller of this.active.get(connection) ?? []) controller.abort() }
  close(): void { this.closed = true; for (const connection of this.active.keys()) this.cancel(connection) }
}
