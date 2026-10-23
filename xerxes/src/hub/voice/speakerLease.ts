// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * One renewable active-speaker lease per conversation. The lease belongs to an
 * account and a device (never to a voice print): the same user on the same
 * device may take it back after a reconnect, anyone else must wait for it to
 * be released or to expire. All operations are synchronous, so two racing
 * acquirers are serialized by the event loop.
 */

export interface LeaseHolder {
  readonly userId: string
  readonly deviceId: string
  /** Socket that currently exercises the lease. */
  readonly connectionId: string
}

export type LeaseOutcome =
  | { readonly ok: true; readonly expiresAt: number }
  | { readonly ok: false; readonly heldBy: { readonly userId: string; readonly deviceId: string }; readonly expiresAt: number }

interface Lease {
  holder: LeaseHolder
  expiresAt: number
}

export interface SpeakerLeaseOptions {
  readonly ttlMs: number
  readonly clock?: () => number
  readonly maxConversations?: number
}

export class SpeakerLeases {
  private readonly leases = new Map<string, Lease>()
  private readonly ttlMs: number
  private readonly clock: () => number
  private readonly maxConversations: number

  constructor(options: SpeakerLeaseOptions) {
    if (!Number.isInteger(options.ttlMs) || options.ttlMs < 1) throw new RangeError('ttlMs must be a positive integer')
    this.ttlMs = options.ttlMs
    this.clock = options.clock ?? Date.now
    this.maxConversations = options.maxConversations ?? 10_000
  }

  acquire(conversationId: string, holder: LeaseHolder): LeaseOutcome {
    const now = this.clock()
    this.sweep(now)
    const current = this.leases.get(conversationId)
    if (current !== undefined && !sameSpeaker(current.holder, holder)) {
      return { ok: false, heldBy: { userId: current.holder.userId, deviceId: current.holder.deviceId }, expiresAt: current.expiresAt }
    }
    if (current === undefined && this.leases.size >= this.maxConversations) {
      return { ok: false, heldBy: { userId: '', deviceId: '' }, expiresAt: now + this.ttlMs }
    }
    const expiresAt = now + this.ttlMs
    this.leases.set(conversationId, { holder, expiresAt })
    return { ok: true, expiresAt }
  }

  /** Extend a lease the caller still holds. A lease that already expired is not revived. */
  renew(conversationId: string, holder: LeaseHolder): LeaseOutcome {
    const now = this.clock()
    const current = this.leases.get(conversationId)
    if (current === undefined || current.expiresAt <= now) {
      if (current !== undefined) this.leases.delete(conversationId)
      return { ok: false, heldBy: { userId: '', deviceId: '' }, expiresAt: now }
    }
    if (!sameSpeaker(current.holder, holder)) {
      return { ok: false, heldBy: { userId: current.holder.userId, deviceId: current.holder.deviceId }, expiresAt: current.expiresAt }
    }
    current.holder = holder
    current.expiresAt = now + this.ttlMs
    return { ok: true, expiresAt: current.expiresAt }
  }

  isHolder(conversationId: string, holder: LeaseHolder): boolean {
    const current = this.leases.get(conversationId)
    return current !== undefined && current.expiresAt > this.clock() && sameSpeaker(current.holder, holder) && current.holder.connectionId === holder.connectionId
  }

  holder(conversationId: string): { readonly userId: string; readonly deviceId: string; readonly expiresAt: number } | undefined {
    const current = this.leases.get(conversationId)
    if (current === undefined || current.expiresAt <= this.clock()) return undefined
    return { userId: current.holder.userId, deviceId: current.holder.deviceId, expiresAt: current.expiresAt }
  }

  release(conversationId: string, holder: LeaseHolder): boolean {
    const current = this.leases.get(conversationId)
    if (current === undefined || !sameSpeaker(current.holder, holder)) return false
    this.leases.delete(conversationId)
    return true
  }

  /** Drop every lease a closed socket still holds. */
  releaseConnection(connectionId: string): void {
    for (const [conversationId, lease] of this.leases) if (lease.holder.connectionId === connectionId) this.leases.delete(conversationId)
  }

  get size(): number {
    this.sweep(this.clock())
    return this.leases.size
  }

  private sweep(now: number): void {
    for (const [conversationId, lease] of this.leases) if (lease.expiresAt <= now) this.leases.delete(conversationId)
  }
}

function sameSpeaker(left: LeaseHolder, right: LeaseHolder): boolean {
  return left.userId === right.userId && left.deviceId === right.deviceId
}
