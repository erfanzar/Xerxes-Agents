// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { cleanFolder, sameRemote } from './contextNavigation.js'
import type { RemoteConnection, RemoteTarget } from './remote.js'

export type RemoteTunnelOpener = (
  machine: RemoteTarget,
  signal: AbortSignal,
  onFailure: (error: Error) => void,
) => Promise<RemoteConnection>

interface PendingReconnect {
  readonly controller: AbortController
  waiters: number
  readonly promise: Promise<void>
}

interface SharedTunnelEntry {
  readonly target: string
  machine: RemoteTarget
  connection: RemoteConnection
  refs: number
  failed: boolean
  closing: boolean
  readonly failureListeners: Set<(error: Error) => void>
  reconnecting: PendingReconnect | null
  generation: number
}

interface PendingOpen {
  readonly controller: AbortController
  waiters: number
  readonly promise: Promise<SharedTunnelEntry>
}

/**
 * Ref-counted SSH tunnel pool keyed by SSH host (`machine.target`).
 *
 * Every Xerxes workspace on one SSH host shares the host's single global daemon
 * socket, and OpenSSH `-L <localSock>:<remoteSock>` multiplexes unlimited local
 * client connections over one authenticated SSH transport. Pooling avoids
 * spawning duplicate `ssh -L` processes and eliminates 3–5 sequential SSH
 * handshakes whenever a second view or folder opens on an already-connected host.
 */
export class SharedRemoteTunnelPool {
  private readonly entries = new Map<string, SharedTunnelEntry>()
  private readonly openings = new Map<string, PendingOpen>()

  /** Number of active pooled hosts (for diagnostics & tests). */
  get activeHosts(): number {
    return this.entries.size
  }

  /** Borrow a ref-counted handle to the host's shared SSH tunnel, opening one if needed. */
  async acquire(
    machine: RemoteTarget,
    signal: AbortSignal,
    onFailure: (error: Error) => void,
    opener: RemoteTunnelOpener,
  ): Promise<RemoteConnection> {
    signal.throwIfAborted()
    const target = machine.target

    const existing = this.entries.get(target)
    if (existing && !existing.failed && !existing.closing) {
      return this.leaseFrom(existing, machine, onFailure, opener)
    }

    let pending = this.openings.get(target)
    if (!pending) {
      const controller = new AbortController()
      const created: PendingOpen = {
        controller,
        waiters: 0,
        promise: (async () => {
          const failureListeners = new Set<(error: Error) => void>()
          let entryRef: SharedTunnelEntry | undefined
          const fanOutFailure = (error: Error): void => {
            if (!entryRef || entryRef.failed || entryRef.closing) return
            entryRef.failed = true
            if (this.entries.get(target) === entryRef) this.entries.delete(target)
            for (const listener of [...entryRef.failureListeners]) {
              try { listener(error) } catch { /* ignore listener errors */ }
            }
          }
          const connection = await opener(machine, controller.signal, fanOutFailure)
          if (controller.signal.aborted) {
            await connection.close().catch(() => undefined)
            throw new Error('Remote connection cancelled')
          }
          const entry: SharedTunnelEntry = {
            target,
            machine,
            connection,
            refs: 0,
            failed: false,
            closing: false,
            failureListeners,
            reconnecting: null,
            generation: 1,
          }
          entryRef = entry
          this.entries.set(target, entry)
          return entry
        })().finally(() => {
          if (this.openings.get(target) === created) this.openings.delete(target)
        }),
      }
      this.openings.set(target, created)
      pending = created
    }

    pending.waiters += 1
    let detached = false
    const onAbort = (): void => {
      if (detached) return
      detached = true
      pending!.waiters = Math.max(0, pending!.waiters - 1)
      if (pending!.waiters === 0) pending!.controller.abort()
    }
    signal.addEventListener('abort', onAbort, { once: true })
    let resolvedEntry: SharedTunnelEntry | null = null
    try {
      const entry = await Promise.race([
        pending.promise,
        new Promise<never>((_resolve, reject) => {
          if (signal.aborted) {
            reject(new Error('Remote connection cancelled'))
            return
          }
          signal.addEventListener('abort', () => reject(new Error('Remote connection cancelled')), { once: true })
        }),
      ])
      resolvedEntry = entry
      signal.throwIfAborted()
      return this.leaseFrom(entry, machine, onFailure, opener)
    } finally {
      signal.removeEventListener('abort', onAbort)
      if (!detached) {
        detached = true
        pending.waiters = Math.max(0, pending.waiters - 1)
      }
      // If every waiter aborted right after opener finished before leasing, don't leak an unreferenced tunnel.
      if (resolvedEntry && resolvedEntry.refs === 0 && pending.waiters === 0 && !resolvedEntry.closing) {
        resolvedEntry.closing = true
        if (this.entries.get(target) === resolvedEntry) this.entries.delete(target)
        void resolvedEntry.connection.close().catch(() => undefined)
      }
    }
  }

  private leaseFrom(
    entry: SharedTunnelEntry,
    machine: RemoteTarget,
    onFailure: (error: Error) => void,
    opener: RemoteTunnelOpener,
  ): RemoteConnection {
    entry.refs += 1
    entry.failureListeners.add(onFailure)
    const leasedGeneration = entry.generation
    let released = false

    const resolvedProjectDir = sameRemote(entry.machine, machine)
      ? entry.connection.projectDir
      : cleanFolder(machine.workspacePath)

    const handle: RemoteConnection = {
      get socketPath() {
        return entry.connection.socketPath
      },
      get projectDir() {
        return sameRemote(entry.machine, machine) ? entry.connection.projectDir : resolvedProjectDir
      },
      get expectedBuildId() {
        return entry.connection.expectedBuildId
      },
      set expectedBuildId(value: string | undefined) {
        entry.connection.expectedBuildId = value
      },
      update(force = false) {
        return entry.connection.update(force)
      },
      reconnect: async (retrySignal, retryFailure) => {
        retrySignal.throwIfAborted()
        // If another borrower on the same host already replaced the failed tunnel
        // with a healthy one, attach immediately to the live replacement!
        const current = this.entries.get(entry.target)
        if (current && current !== entry && !current.failed && !current.closing) {
          return this.leaseFrom(current, machine, retryFailure, opener)
        }
        if (current === entry && !entry.failed && !entry.closing && entry.generation > leasedGeneration) {
          return this.leaseFrom(entry, machine, retryFailure, opener)
        }
        // Single-flight shared reconnect for this host so multiple views never
        // race to spawn duplicate replacement tunnels on the same drop, and one
        // waiter cancelling does not abort another borrower's shared reconnect.
        let pendingReconnect = entry.reconnecting
        if (!pendingReconnect) {
          const controller = new AbortController()
          const created: PendingReconnect = {
            controller,
            waiters: 0,
            promise: (async () => {
              const previousConnection = entry.connection
              const replacement = await previousConnection.reconnect(controller.signal, error => {
                if (entry.connection !== replacement || entry.failed || entry.closing) return
                entry.failed = true
                if (this.entries.get(entry.target) === entry) this.entries.delete(entry.target)
                for (const listener of [...entry.failureListeners]) {
                  try { listener(error) } catch { /* ignore */ }
                }
              })
              if (controller.signal.aborted) {
                await replacement.close().catch(() => undefined)
                throw new Error('Remote connection cancelled')
              }
              entry.connection = replacement
              entry.failed = false
              entry.generation += 1
              if (!entry.closing) this.entries.set(entry.target, entry)
              await previousConnection.close().catch(() => undefined)
            })().finally(() => {
              if (entry.reconnecting === created) entry.reconnecting = null
            }),
          }
          entry.reconnecting = created
          pendingReconnect = created
        }
        pendingReconnect.waiters += 1
        let detached = false
        const onAbort = (): void => {
          if (detached) return
          detached = true
          pendingReconnect!.waiters = Math.max(0, pendingReconnect!.waiters - 1)
          if (pendingReconnect!.waiters === 0) pendingReconnect!.controller.abort()
        }
        retrySignal.addEventListener('abort', onAbort, { once: true })
        try {
          await Promise.race([
            pendingReconnect.promise,
            new Promise<never>((_resolve, reject) => {
              if (retrySignal.aborted) {
                reject(new Error('Remote connection cancelled'))
                return
              }
              retrySignal.addEventListener('abort', () => reject(new Error('Remote connection cancelled')), { once: true })
            }),
          ])
        } finally {
          retrySignal.removeEventListener('abort', onAbort)
          if (!detached) {
            detached = true
            pendingReconnect.waiters = Math.max(0, pendingReconnect.waiters - 1)
          }
        }
        retrySignal.throwIfAborted()
        const active = this.entries.get(entry.target)
        if (active && !active.failed && !active.closing) {
          return this.leaseFrom(active, machine, retryFailure, opener)
        }
        return this.acquire(machine, retrySignal, retryFailure, opener)
      },
      close: async () => {
        if (released) return
        released = true
        entry.failureListeners.delete(onFailure)
        entry.refs = Math.max(0, entry.refs - 1)
        if (entry.refs === 0 && !entry.closing) {
          entry.closing = true
          if (this.entries.get(entry.target) === entry) this.entries.delete(entry.target)
          await entry.connection.close().catch(() => undefined)
        }
      },
    }
    return handle
  }

  /** Abort in-flight opens and close every pooled SSH tunnel on app quit. */
  async closeAll(): Promise<void> {
    for (const pending of this.openings.values()) pending.controller.abort()
    this.openings.clear()
    const current = [...this.entries.values()]
    this.entries.clear()
    await Promise.all(
      current.map(entry => {
        entry.closing = true
        entry.reconnecting?.controller.abort()
        entry.failureListeners.clear()
        return entry.connection.close().catch(() => undefined)
      }),
    )
  }
}
