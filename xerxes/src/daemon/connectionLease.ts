// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomBytes } from 'node:crypto'
import type { DaemonTransportConnection } from './transport.js'

interface Lease {
  token: string
  owner: DaemonTransportConnection
  transport: DaemonTransportConnection | null
  restoring: boolean
  frames: object[]
  bytes: number
  timer?: ReturnType<typeof setTimeout>
}

/** Stable ownership across a short transport interruption. A bounded event
 * journal preserves streaming deltas; overflow expires the lease safely.
 */
export class ConnectionLeases {
  private readonly tokens = new Map<string, Lease>()
  private readonly transports = new WeakMap<DaemonTransportConnection, Lease>()
  private readonly owners = new WeakMap<DaemonTransportConnection, Lease>()

  constructor(
    private readonly expire: (owner: DaemonTransportConnection) => void,
    readonly graceMs = 30_000,
  ) {}

  owner(transport: DaemonTransportConnection): DaemonTransportConnection {
    return this.transports.get(transport)?.owner ?? transport
  }

  has(token: string): boolean { return this.tokens.has(token) }

  /** Private provider requests contain context, not UI events. Never journal
   * them across a disconnect or restoration boundary. */
  sendPrivate(owner: DaemonTransportConnection, frame: object): boolean {
    const lease = this.owners.get(owner)
    if (!lease) { owner.send(frame); return true }
    if (!lease.transport || lease.restoring) return false
    lease.transport.send(frame)
    return true
  }

  enable(transport: DaemonTransportConnection): string {
    const existing = this.transports.get(transport)
    if (existing) return existing.token
    const lease: Lease = {
      token: randomBytes(32).toString('hex'), transport, restoring: false, frames: [], bytes: 0,
      owner: { activeSessionKey: transport.activeSessionKey, send: frame => {
        if (lease.transport && !lease.restoring) { lease.transport.send(frame); return }
        if (!this.tokens.has(lease.token)) return
        const encoded = JSON.stringify(frame)
        const bytes = Buffer.byteLength(encoded)
        if (lease.bytes + bytes > 1024 * 1024) { this.release(lease); return }
        lease.frames.push(JSON.parse(encoded) as object); lease.bytes += bytes
      } },
    }
    this.tokens.set(lease.token, lease)
    this.owners.set(lease.owner, lease)
    this.bind(lease, transport)
    return lease.token
  }

  resume(transport: DaemonTransportConnection, token: string, accepts: (owner: DaemonTransportConnection) => boolean = () => true): DaemonTransportConnection {
    const lease = this.tokens.get(token)
    if (!lease) throw new Error('Connection lease expired or invalid. Reopen the saved session.')
    if (!accepts(lease.owner)) throw new Error('Connection lease does not belong to this session.')
    if (lease.transport === transport) return lease.owner
    if (lease.transport) throw new Error('Connection lease is already attached to another transport.')
    if (this.transports.has(transport)) throw new Error('Transport already owns a connection lease.')
    lease.restoring = true
    this.bind(lease, transport)
    return lease.owner
  }

  /** Finish restoring atomically with a bounded journal of missed events. */
  takeReplay(owner: DaemonTransportConnection): object[] | undefined {
    const lease = this.owners.get(owner)
    if (!lease?.restoring) return undefined
    lease.restoring = false
    if (lease.timer) clearTimeout(lease.timer)
    delete lease.timer
    const frames = lease.frames
    lease.frames = []; lease.bytes = 0
    return frames
  }

  /** Returns true when the lease, rather than the caller, owns cleanup. */
  disconnect(transport: DaemonTransportConnection): boolean {
    const lease = this.transports.get(transport)
    if (!lease) return false
    this.transports.delete(transport)
    if (lease.transport !== transport) return true
    lease.transport = null
    if (lease.timer) clearTimeout(lease.timer)
    lease.timer = setTimeout(() => this.release(lease), this.graceMs)
    lease.timer.unref?.()
    return true
  }

  close(): void {
    for (const lease of [...this.tokens.values()]) this.release(lease)
  }

  private bind(lease: Lease, transport: DaemonTransportConnection): void {
    lease.transport = transport
    this.transports.set(transport, lease)
    // Broadcast routing and session liveness still inspect raw transports.
    Object.defineProperty(transport, 'activeSessionKey', {
      configurable: true,
      get: () => lease.owner.activeSessionKey,
      set: (value: string) => { lease.owner.activeSessionKey = value },
    })
  }

  private release(lease: Lease): void {
    if (!this.tokens.delete(lease.token)) return
    if (lease.timer) clearTimeout(lease.timer)
    if (lease.transport) this.transports.delete(lease.transport)
    this.owners.delete(lease.owner)
    lease.frames = []; lease.bytes = 0
    lease.transport = null
    this.expire(lease.owner)
  }
}
