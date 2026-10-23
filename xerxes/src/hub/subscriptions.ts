// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { canSeeWorkspace, grantedSessionIds, hasScope, type HubGrant, type HubPrincipal } from '../security/principal.js'
import { cursorScopeKey, encodeCursor, resolveClientCursor, type CursorScope } from './cursor.js'
import { HubEventStore } from './eventStore.js'
import {
  HUB_LIMITS, HubProtocolError, encodeHubFrame, hubEvent, readIdList, utf8ByteLength,
  type HubEvent, type HubEventFrame, type HubGapReason, type HubStreamGap,
} from './protocol.js'

export interface SubscriptionSink {
  /** Resolve only when the transport has accepted/drained this frame, not just buffered it. */
  send(frame: HubEventFrame): void | Promise<void>
  close(code: number, reason?: string): void
}

export interface SubscriptionRequest extends CursorScope {
  readonly afterCursor?: string
}

export interface SubscriptionResult {
  readonly subscriptionId: string
  readonly watermark: string
  /** Exclusive replay anchor immediately before the first retained event. */
  readonly oldest: string
  readonly resetRequired: boolean
}

export interface SubscriptionManagerOptions {
  readonly clock?: () => number
  /** Includes queued AND in-flight serialized frames. Default 1 MiB per phone. */
  readonly maxQueueBytes?: number
  readonly maxBatchBytes?: number
  readonly pageEventLimit?: number
  /** Resolve current grants on every delivery. Otherwise call updateGrants after ACL changes. */
  readonly grantsForPrincipal?: (principalId: string) => readonly HubGrant[]
}

interface Outbound {
  readonly frame: HubEventFrame
  readonly bytes: number
  readonly throughSequence?: number
}

interface Subscription {
  readonly id: string
  readonly principal: HubPrincipal
  grants: readonly HubGrant[]
  readonly scope: CursorScope
  readonly scopeKey: string
  readonly initialAccess: string
  readonly sink: SubscriptionSink
  readonly watermark: number
  readonly oldest: number
  readonly queue: Outbound[]
  scanSequence: number
  backfillThrough: number | null
  lastSentSequence: number
  ackedSequence: number
  ackedCursor: string | undefined
  queuedBytes: number
  sending: boolean
  scheduled: boolean
  active: boolean
}

/**
 * Persisted events are the outbox; this class holds only bounded transport queues.
 * Acks are in-memory and connection-local. The phone must persist its last applied
 * cursor and supply it on reconnect; an uncertain/unacked frame is replayed.
 */
export class SubscriptionManager {
  private readonly subscriptions = new Map<string, Subscription>()
  private readonly clock: () => number
  private readonly maxQueueBytes: number
  private readonly maxBatchBytes: number
  private readonly pageEventLimit: number
  private readonly detach: () => void
  private closed = false

  constructor(private readonly store: HubEventStore, private readonly options: SubscriptionManagerOptions = {}) {
    this.clock = options.clock ?? Date.now
    this.maxQueueBytes = options.maxQueueBytes ?? HUB_LIMITS.maxFrameBytes
    this.maxBatchBytes = options.maxBatchBytes ?? HUB_LIMITS.normalFrameBytes
    this.pageEventLimit = options.pageEventLimit ?? HUB_LIMITS.maxBatchEvents
    if (!Number.isSafeInteger(this.maxQueueBytes) || this.maxQueueBytes < 1) throw new RangeError('maxQueueBytes must be positive')
    if (!Number.isSafeInteger(this.maxBatchBytes) || this.maxBatchBytes < 2048 || this.maxBatchBytes > HUB_LIMITS.maxFrameBytes) {
      throw new RangeError('maxBatchBytes must be between 2048 and the Hub frame cap')
    }
    if (!Number.isSafeInteger(this.pageEventLimit) || this.pageEventLimit < 1 || this.pageEventLimit > HUB_LIMITS.maxBatchEvents) {
      throw new RangeError('invalid pageEventLimit')
    }
    this.detach = store.onEvents(() => {
      for (const subscription of this.subscriptions.values()) this.schedule(subscription)
    })
  }

  subscribe(principal: HubPrincipal, grants: readonly HubGrant[], request: SubscriptionRequest, sink: SubscriptionSink): SubscriptionResult {
    if (this.closed) throw new HubProtocolError('unavailable', 'subscription manager is closed')
    const workspaceIds = readIdList(request.workspaceIds, 'workspaceIds', { required: true })
    if (workspaceIds.length === 0) throw new HubProtocolError('invalid_request', 'at least one workspace is required')
    const sessionIds = request.sessionIds === undefined ? undefined : readIdList(request.sessionIds, 'sessionIds')
    const scope: CursorScope = { workspaceIds, ...(sessionIds === undefined ? {} : { sessionIds }) }
    const currentGrants = this.options.grantsForPrincipal?.(principal.id) ?? grants
    this.assertAccess(principal, currentGrants, scope)
    const scopeKey = cursorScopeKey(scope)
    const resolved = resolveClientCursor(request.afterCursor, { streamId: this.store.streamId, scopeKey })
    // Both reads and registration are synchronous on this writer's event loop;
    // delivery is deferred so the caller can send its snapshot/watermark first.
    const watermark = this.store.watermark()
    const oldest = this.store.oldestSequence()
    let sequence = resolved.kind === 'after' ? resolved.sequence : 0
    if (sequence > watermark) throw new HubProtocolError('invalid_request', 'cursor is ahead of this Hub stream')
    let reason: HubGapReason | undefined
    if (resolved.kind === 'stream_replaced') reason = 'stream_replaced'
    else if (sequence < oldest - 1) reason = 'retention'
    if (reason) sequence = watermark
    const subscription: Subscription = {
      id: crypto.randomUUID(), principal, grants, scope, scopeKey, initialAccess: this.accessFingerprint(principal, currentGrants, scope), sink,
      watermark, oldest, queue: [], scanSequence: sequence, backfillThrough: watermark,
      lastSentSequence: sequence, ackedSequence: sequence, ackedCursor: request.afterCursor,
      queuedBytes: 0, sending: false, scheduled: false, active: true,
    }
    this.subscriptions.set(subscription.id, subscription)
    this.schedule(subscription)
    if (reason) {
      const frame = this.gapFrame(subscription, reason, request.afterCursor)
      this.add(subscription, frame, undefined, true)
    }
    return {
      subscriptionId: subscription.id,
      watermark: this.cursor(subscription, watermark), oldest: this.cursor(subscription, oldest - 1), resetRequired: reason !== undefined,
    }
  }

  /** Ack only applied frames. Regressing duplicate acks are harmless; future acks are refused. */
  ack(subscriptionId: string, cursor: string): void {
    const subscription = this.get(subscriptionId)
    const resolved = resolveClientCursor(cursor, { streamId: this.store.streamId, scopeKey: subscription.scopeKey })
    if (resolved.kind !== 'after' || resolved.sequence > subscription.lastSentSequence) {
      throw new HubProtocolError('invalid_request', 'ack does not name a delivered position in this subscription')
    }
    if (resolved.sequence < subscription.ackedSequence) return
    subscription.ackedSequence = resolved.sequence
    subscription.ackedCursor = cursor
  }

  lastAckedCursor(subscriptionId: string): string | undefined { return this.get(subscriptionId).ackedCursor }

  unsubscribe(subscriptionId: string): void {
    const subscription = this.subscriptions.get(subscriptionId)
    if (subscription) this.remove(subscription)
  }

  /** Use this when no grantsForPrincipal resolver was supplied. Queued events are rechecked. */
  updateGrants(principalId: string, grants: readonly HubGrant[]): void {
    for (const subscription of this.subscriptions.values()) {
      if (subscription.principal.id !== principalId) continue
      subscription.grants = grants
      this.schedule(subscription)
    }
  }

  revokePrincipal(principalId: string): void {
    for (const subscription of [...this.subscriptions.values()]) {
      if (subscription.principal.id !== principalId) continue
      this.terminate(subscription, 1008, 'principal revoked', hubEvent('hub.revoked', { principal_id: principalId }))
    }
  }

  close(): void {
    if (this.closed) return
    this.closed = true
    this.detach()
    for (const subscription of [...this.subscriptions.values()]) this.terminate(subscription, 1001, 'Hub subscriptions closed')
  }

  private assertAccess(principal: HubPrincipal, grants: readonly HubGrant[], scope: CursorScope): void {
    const now = this.clock()
    for (const workspaceId of scope.workspaceIds) {
      // Session-only invites may subscribe to those sessions, but never the
      // workspace's inventory. canSeeWorkspace alone intentionally excludes them.
      const broad = canSeeWorkspace(principal, grants, workspaceId, now) && hasScope(principal, grants, { workspaceId }, 'session.read', now)
      if (broad) continue
      const allowed = grantedSessionIds(principal, grants, workspaceId, now)
      if (allowed.length === 0 || (scope.sessionIds !== undefined && !scope.sessionIds.some(id => allowed.includes(id)))) {
        throw new HubProtocolError('forbidden', `no readable sessions in workspace ${workspaceId}`)
      }
    }
  }

  private grants(subscription: Subscription): readonly HubGrant[] {
    return this.options.grantsForPrincipal?.(subscription.principal.id) ?? subscription.grants
  }

  private accessFingerprint(principal: HubPrincipal, grants: readonly HubGrant[], scope: CursorScope): string {
    const now = this.clock()
    const relevant = grants.filter(grant => grant.principalId === principal.id
      && (grant.resource.type === 'hub' || scope.workspaceIds.includes(grant.resource.workspaceId))
      && (grant.expiresAt === undefined || Date.parse(grant.expiresAt) > now))
      .map(grant => JSON.stringify(grant)).sort()
    const effective = scope.workspaceIds.map(workspaceId => [
      workspaceId, hasScope(principal, grants, { workspaceId }, 'session.read', now),
      grantedSessionIds(principal, grants, workspaceId, now),
    ])
    return JSON.stringify([relevant, effective, principal.kind === 'user' ? [...(principal.deviceScopeCeiling ?? [])].sort() : 'agent'])
  }

  private checkAccess(subscription: Subscription): boolean {
    const current = this.grants(subscription)
    if (this.accessFingerprint(subscription.principal, current, subscription.scope) === subscription.initialAccess) return true
    this.terminate(subscription, 1008, 'subscription access changed', this.gapFrame(subscription, 'scope_changed'))
    return false
  }

  private visible(subscription: Subscription, event: HubEvent): boolean {
    return subscription.scope.workspaceIds.includes(event.workspace_id)
      && (subscription.scope.sessionIds === undefined || (event.session_id !== undefined && subscription.scope.sessionIds.includes(event.session_id)))
      && hasScope(subscription.principal, this.grants(subscription), {
        workspaceId: event.workspace_id, ...(event.session_id === undefined ? {} : { sessionId: event.session_id }),
      }, 'session.read', this.clock())
  }

  private schedule(subscription: Subscription): void {
    if (!subscription.active || subscription.scheduled) return
    subscription.scheduled = true
    queueMicrotask(() => {
      subscription.scheduled = false
      if (!subscription.active) return
      try {
        this.pump(subscription)
        this.drain(subscription)
      } catch {
        this.terminate(subscription, 1011, 'Hub journal unavailable', this.gapFrame(subscription, 'journal_unavailable'))
      }
    })
  }

  private pump(subscription: Subscription): void {
    while (subscription.active) {
      if (!this.checkAccess(subscription)) return
      const through = subscription.backfillThrough ?? this.store.watermark()
      if (subscription.scanSequence >= through) {
        if (subscription.backfillThrough !== null) { subscription.backfillThrough = null; continue }
        return
      }
      const page = this.store.read(subscription.scanSequence, subscription.scope, this.pageEventLimit, this.maxBatchBytes - 1024, through)
      if (page.gap) {
        this.terminate(subscription, 1008, 'Hub replay expired', this.gapFrame(subscription, 'retention'))
        return
      }
      const events = page.events.filter(event => this.visible(subscription, event))
      const frame = hubEvent('hub.event_batch', {
        subscription_id: subscription.id, events, through_cursor: this.cursor(subscription, page.throughSequence),
      })
      subscription.scanSequence = page.throughSequence
      if (!this.add(subscription, frame, page.throughSequence)) return
    }
  }

  private add(subscription: Subscription, frame: HubEventFrame, throughSequence?: number, defer = false): boolean {
    const bytes = utf8ByteLength(encodeHubFrame(frame, this.maxBatchBytes))
    if (subscription.queuedBytes + bytes > this.maxQueueBytes) {
      this.terminate(subscription, 1013, 'slow client; resume from last applied cursor')
      return false
    }
    subscription.queue.push({ frame, bytes, ...(throughSequence === undefined ? {} : { throughSequence }) })
    subscription.queuedBytes += bytes
    // Do not send anything from subscribe itself: snapshot + watermark must be
    // observable before backfill. pump/drain run in the next microtask.
    if (defer) return true
    this.drain(subscription)
    return subscription.active
  }

  private drain(subscription: Subscription): void {
    if (subscription.sending || !subscription.active) return
    while (subscription.queue.length > 0 && subscription.active) {
      if (!this.checkAccess(subscription)) return
      const item = subscription.queue.shift()!
      if (item.frame.type === 'hub.event_batch') {
        const events = item.frame.payload.events as readonly HubEvent[]
        // No stale queued ACL decision can leak an event to its transport.
        if (events.some(event => !this.visible(subscription, event))) {
          this.terminate(subscription, 1008, 'subscription access changed', this.gapFrame(subscription, 'scope_changed'))
          return
        }
      }
      if (item.throughSequence !== undefined) subscription.lastSentSequence = item.throughSequence
      subscription.sending = true
      let pending: void | Promise<void>
      try { pending = subscription.sink.send(item.frame) } catch {
        this.terminate(subscription, 1011, 'client transport failed')
        return
      }
      if (pending !== undefined) {
        Promise.resolve(pending).then(() => {
          if (!subscription.active) return
          subscription.queuedBytes -= item.bytes
          subscription.sending = false
          try { this.drain(subscription) } catch {
            this.terminate(subscription, 1011, 'client delivery failed')
          }
        }, () => { this.terminate(subscription, 1011, 'client transport failed') })
        return
      }
      subscription.queuedBytes -= item.bytes
      subscription.sending = false
    }
  }

  private gapFrame(subscription: Subscription, reason: HubGapReason, requestedCursor?: string): HubEventFrame {
    let watermark = subscription.watermark
    let oldest = subscription.oldest
    try { watermark = this.store.watermark(); oldest = this.store.oldestSequence() } catch { /* Preserve the last known positions when the journal is unavailable. */ }
    const gap: HubStreamGap = {
      subscription_id: subscription.id, reason,
      ...(requestedCursor === undefined ? {} : { requested_cursor: requestedCursor }),
      oldest_cursor: this.cursor(subscription, oldest - 1), watermark: this.cursor(subscription, watermark),
    }
    return hubEvent('hub.stream_gap', { ...gap })
  }

  private cursor(subscription: Subscription, sequence: number): string {
    return encodeCursor({ streamId: this.store.streamId, sequence, scopeKey: subscription.scopeKey })
  }

  private get(id: string): Subscription {
    const value = this.subscriptions.get(id)
    if (!value) throw new HubProtocolError('not_found', 'subscription not found')
    return value
  }

  private remove(subscription: Subscription): void {
    subscription.active = false
    subscription.queue.length = 0
    subscription.queuedBytes = 0
    this.subscriptions.delete(subscription.id)
  }

  private terminate(subscription: Subscription, code: number, reason: string, finalFrame?: HubEventFrame): void {
    if (!subscription.active) return
    this.remove(subscription)
    // Revocation/control frames bypass a blocked data queue. Closing is immediate;
    // do not wait for a stalled phone's outstanding send promise to settle.
    try {
      if (finalFrame) {
        const pending = subscription.sink.send(finalFrame)
        if (pending !== undefined) Promise.resolve(pending).catch(() => { /* Socket is already being closed. */ })
      }
    } catch { /* Closing remains mandatory even if the control frame cannot be sent. */ }
    finally { subscription.sink.close(code, reason) }
  }
}
