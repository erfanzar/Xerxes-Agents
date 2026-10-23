// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto';
import { isAbsolute } from 'node:path';
import type { JsonRpcPayload } from '../protocol/jsonRpc.js';
import { resolveProjectDirectory } from './paths.js';
import { EVENT_BATCH_BYTES, EVENT_HARD_BYTES, eventBytes, TransportEventJournal, type EventGapReason, type TransportEvent } from './transportEventJournal.js';

interface Subscription {
  readonly id: string;
  readonly owner: object;
  readonly projects: Set<string>;
  readonly sessions: Set<string> | undefined;
  readonly detail: 'summary' | 'conversation';
  readonly maxBytes: number;
  readonly send: (type: string, payload: JsonRpcPayload) => void;
  cursor: string;
  watermark: number;
  queuedBytes: number;
  paused: boolean;
  timer: ReturnType<typeof setTimeout> | undefined;
}

/** Read-only transport state. Nothing here registers an interaction owner or session observer. */
export class EventSubscriptions {
  private readonly subscriptions = new Map<string, Subscription>();
  private readonly unlisten: () => void;
  constructor(private readonly journal: TransportEventJournal, private readonly maxQueueBytes = EVENT_HARD_BYTES) {
    this.unlisten = journal.listen(events => {
      for (const subscription of this.subscriptions.values()) {
        if (!events) { this.gap(subscription, 'journal_unavailable'); continue; }
        if (subscription.paused) continue;
        for (const event of events) if (event.sequence > subscription.watermark && this.accepts(subscription, event)) subscription.queuedBytes += Math.min(eventBytes(event), subscription.maxBytes);
        // The queue is represented by cursors, not duplicate payloads. Bound the backlog anyway.
        if (subscription.queuedBytes > this.maxQueueBytes) this.gap(subscription, 'retention');
        else this.schedule(subscription);
      }
    });
  }
  subscribe(owner: object, params: JsonRpcPayload, send: Subscription['send']): JsonRpcPayload {
    if ([...this.subscriptions.values()].filter(subscription => subscription.owner === owner).length >= 32) throw new Error('Too many event subscriptions');
    const projects = strings(params.project_dirs, 'project_dirs');
    if (!projects.length || projects.some(project => !isAbsolute(project))) throw new Error('project_dirs must contain absolute paths');
    if (params.detail !== 'summary' && params.detail !== 'conversation') throw new Error('detail must be summary or conversation');
    const after = optionalCursor(params.after_cursor);
    const checked = this.journal.checkCursor(after);
    const subscription: Subscription = {
      id: randomUUID(), owner, projects: new Set(projects.map(project => resolveProjectDirectory(project))),
      sessions: params.session_ids === undefined ? undefined : new Set(strings(params.session_ids, 'session_ids')),
      detail: params.detail, maxBytes: byteLimit(params.max_batch_bytes), send,
      cursor: after ?? this.journal.watermark, watermark: this.journal.sequence,
      queuedBytes: 0, paused: Boolean(checked.reason), timer: undefined,
    };
    // Synchronous install + watermark capture; no commit can sneak between these operations.
    this.subscriptions.set(subscription.id, subscription);
    if (checked.reason) subscription.timer = setTimeout(() => { subscription.timer = undefined; this.gap(subscription, checked.reason!); }, 0);
    else this.schedule(subscription);
    return { ok: true, subscription_id: subscription.id, stream_id: this.journal.streamId, watermark: this.journal.watermark,
      oldest_cursor: this.journal.oldestCursor, reset_required: Boolean(checked.reason) };
  }
  read(owner: object, params: JsonRpcPayload): JsonRpcPayload {
    const subscription = this.owned(owner, params.subscription_id);
    const after = optionalCursor(params.after_cursor) ?? subscription.cursor;
    const page = this.journal.read(after, {
      limit: integer(params.limit, 500, 1, 500), maxBytes: Math.min(subscription.maxBytes, byteLimit(params.max_bytes)),
      accepts: event => this.accepts(subscription, event),
    });
    if (page.reason) this.gap(subscription, page.reason, after);
    else if (subscription.paused) {
      // Explicit read is recovery after a live overflow. Its page precedes resumed automatic delivery.
      subscription.paused = false;
      subscription.cursor = page.next_cursor;
      subscription.watermark = this.journal.sequence;
      subscription.queuedBytes = 0;
      this.schedule(subscription);
    }
    return { ok: true, events: page.events, next_cursor: page.next_cursor, has_more: page.has_more,
      watermark: page.watermark, reset_required: page.reset_required };
  }
  unsubscribe(owner: object, params: JsonRpcPayload): JsonRpcPayload {
    const subscription = this.owned(owner, params.subscription_id);
    this.remove(subscription);
    return { ok: true };
  }
  disconnect(owner: object): void {
    for (const subscription of this.subscriptions.values()) if (subscription.owner === owner) this.remove(subscription);
  }
  close(): void {
    this.unlisten();
    for (const subscription of this.subscriptions.values()) this.remove(subscription);
  }
  private owned(owner: object, id: unknown): Subscription {
    const subscription = typeof id === 'string' ? this.subscriptions.get(id) : undefined;
    if (!subscription || subscription.owner !== owner) throw new Error('Unknown event subscription on this connection');
    return subscription;
  }
  private accepts(subscription: Subscription, event: TransportEvent): boolean {
    const type = detailEventType(event.type, event.payload);
    return subscription.projects.has(event.project_dir) && (!subscription.sessions || (event.session_id !== undefined && subscription.sessions.has(event.session_id)))
      && (subscription.detail === 'conversation' || !['text_part', 'think_part', 'thinking_part', 'text_delta', 'thinking_delta'].includes(type));
  }
  private schedule(subscription: Subscription): void {
    if (subscription.timer || subscription.paused || !this.subscriptions.has(subscription.id)) return;
    subscription.timer = setTimeout(() => { subscription.timer = undefined; this.pump(subscription); }, 0);
  }
  private pump(subscription: Subscription): void {
    if (subscription.paused || !this.subscriptions.has(subscription.id)) return;
    // Finish the original watermark first. Live commits remain in the journal until the next page.
    const checked = this.journal.checkCursor(subscription.cursor);
    if (checked.reason) { this.gap(subscription, checked.reason); return; }
    const through = checked.sequence < subscription.watermark ? subscription.watermark : this.journal.sequence;
    const page = this.journal.read(subscription.cursor, { maxBytes: subscription.maxBytes, through, accepts: event => this.accepts(subscription, event) });
    if (page.reason) { this.gap(subscription, page.reason); return; }
    const drainedLiveBytes = page.events.reduce((sum, event) => {
      const size = event.payload.truncated === true && typeof event.payload.original_bytes === 'number'
        ? event.payload.original_bytes : eventBytes(event);
      return event.sequence > subscription.watermark ? sum + Math.min(size, subscription.maxBytes) : sum;
    }, 0);
    subscription.queuedBytes = Math.max(0, subscription.queuedBytes - drainedLiveBytes);
    subscription.cursor = page.next_cursor;
    if (page.events.length) this.send(subscription, 'event_batch', { subscription_id: subscription.id, events: page.events, through_cursor: page.next_cursor });
    if (page.has_more || subscription.cursor !== this.journal.watermark) this.schedule(subscription);
    else { subscription.watermark = this.journal.sequence; subscription.queuedBytes = 0; }
  }
  private gap(subscription: Subscription, reason: EventGapReason, requested = subscription.cursor): void {
    if (subscription.timer) clearTimeout(subscription.timer);
    subscription.timer = undefined;
    subscription.paused = true;
    subscription.queuedBytes = 0;
    this.send(subscription, 'event_gap', { subscription_id: subscription.id, requested_cursor: requested,
      oldest_cursor: this.journal.oldestCursor, watermark: this.journal.watermark, reason });
  }
  private send(subscription: Subscription, type: string, payload: JsonRpcPayload): void {
    try { subscription.send(type, payload); }
    catch (error) {
      this.remove(subscription);
      console.error('Daemon event subscriber disconnected after delivery failure:', error);
    }
  }
  private remove(subscription: Subscription): void {
    if (subscription.timer) clearTimeout(subscription.timer);
    this.subscriptions.delete(subscription.id);
  }
}
function detailEventType(type: string, payload: JsonRpcPayload, depth = 0): string {
  if (depth >= 4) return type;
  if (type === 'cron_event' && typeof payload.event_type === 'string') {
    const nested = record(payload.payload);
    return nested ? detailEventType(payload.event_type, nested, depth + 1) : payload.event_type;
  }
  const nested = type === 'subagent_event' ? record(payload.event) : undefined;
  return nested && typeof nested.type === 'string' ? detailEventType(nested.type, record(nested.payload) ?? {}, depth + 1) : type;
}
function record(value: unknown): JsonRpcPayload | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as JsonRpcPayload : undefined;
}
function strings(value: unknown, name: string): string[] {
  if (!Array.isArray(value) || value.length > 200 || value.some(item => typeof item !== 'string' || !item || item.length > 4096)) throw new Error(`${name} must be a bounded string array`);
  return value as string[];
}
function optionalCursor(value: unknown): string | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== 'string' || !value || value.length > 200) throw new Error('Invalid event cursor');
  return value;
}
function byteLimit(value: unknown): number { return integer(value, EVENT_BATCH_BYTES, 1024, EVENT_HARD_BYTES); }
function integer(value: unknown, fallback: number, minimum: number, maximum: number): number {
  if (value === undefined) return fallback;
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < minimum || value > maximum) throw new Error(`Expected integer between ${minimum} and ${maximum}`);
  return value;
}
