// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { randomUUID } from 'node:crypto';
import { mkdir, open, readdir, rename, rm, type FileHandle } from 'node:fs/promises';
import { join } from 'node:path';
import { redactPayload } from '../security/redact.js';
import type { JsonRpcPayload } from '../protocol/jsonRpc.js';
import { resolveProjectDirectory } from './paths.js';

export interface TransportEvent {
  readonly event_id: string;
  readonly sequence: number;
  readonly cursor: string;
  readonly runtime_id: string;
  readonly instance_id: string;
  readonly project_dir: string;
  readonly session_id?: string;
  readonly type: string;
  readonly payload: JsonRpcPayload;
  readonly occurred_at: string;
}
export interface CompactTransportEvent {
  readonly event_id: string;
  readonly sequence: number;
  readonly cursor: string;
  readonly runtime_id: string;
  readonly instance_id: string;
  readonly type: string;
  readonly payload: JsonRpcPayload;
  readonly truncated: true;
  readonly original_bytes: number;
}
export type WireTransportEvent = TransportEvent | CompactTransportEvent;
export type EventGapReason = 'retention' | 'stream_replaced' | 'journal_unavailable';
export interface JournalInput {
  readonly project_dir: string;
  readonly session_id?: string;
  readonly type: string;
  readonly payload: JsonRpcPayload;
}
export interface TransportEventJournalOptions {
  readonly directory: string;
  readonly maxBytes?: number;
  readonly maxAgeMs?: number;
  readonly segmentBytes?: number;
  readonly maxPendingBytes?: number;
  /** Default: none (OS write completion). batch additionally fsyncs before delivery. */
  readonly fsync?: 'none' | 'batch';
  readonly now?: () => number;
  /** Fault-injection seam, before the native append; never a replacement storage backend. */
  readonly beforeWrite?: () => void | Promise<void>;
}
interface Segment { path: string; bytes: number; events: TransportEvent[] }
interface Pending { event: TransportEvent; line: string; deliver: () => void }
export interface JournalPage {
  events: WireTransportEvent[];
  next_cursor: string;
  has_more: boolean;
  watermark: string;
  oldest_cursor: string;
  reset_required: boolean;
  reason?: EventGapReason;
}

export const EVENT_BATCH_BYTES = 256 * 1024;
export const EVENT_HARD_BYTES = 1024 * 1024;
export const eventBytes = (value: unknown): number => Buffer.byteLength(JSON.stringify(value));

/** One writer per daemon state root. All I/O is asynchronous, and token deltas batch per tick. */
export class TransportEventJournal {
  streamId: string = randomUUID();
  runtimeId: string = randomUUID();
  readonly instanceId = randomUUID();
  available = false;
  private started = false;
  private closed = false;
  private committed = 0;
  private assigned = 0;
  private lastOccurredAt = 0;
  private segments: Segment[] = [];
  private file: FileHandle | undefined;
  private pending: Pending[] = [];
  private pendingBytes = 0;
  private readonly projectPaths = new Map<string, string>();
  private scheduled = false;
  private tail: Promise<void> = Promise.resolve();
  private readonly listeners = new Set<(events: readonly TransportEvent[] | undefined) => void>();
  private readonly maxBytes: number;
  private readonly maxAgeMs: number;
  private readonly segmentBytes: number;
  private readonly now: () => number;
  private maintenance: ReturnType<typeof setInterval> | undefined;

  constructor(private readonly options: TransportEventJournalOptions) {
    this.maxBytes = positive(options.maxBytes, 64 * 1024 * 1024);
    this.maxAgeMs = positive(options.maxAgeMs, 7 * 24 * 60 * 60 * 1000);
    this.segmentBytes = Math.min(this.maxBytes, positive(options.segmentBytes, EVENT_HARD_BYTES));
    this.now = options.now ?? Date.now;
  }
  get sequence(): number { return this.committed; }
  cursor(sequence = this.committed): string { return `${this.streamId}:${sequence}`; }
  get watermark(): string { return this.cursor(); }
  get oldestCursor(): string {
    const first = this.retainedEvents().next().value;
    return this.cursor(first ? first.sequence - 1 : this.committed);
  }
  listen(listener: (events: readonly TransportEvent[] | undefined) => void): () => void {
    this.listeners.add(listener);
    return () => { this.listeners.delete(listener); };
  }
  async start(): Promise<void> {
    if (this.started) return;
    this.started = true;
    try {
      await mkdir(this.options.directory, { recursive: true, mode: 0o700 });
      const identityPath = join(this.options.directory, 'identity.json');
      const paths = (await readdir(this.options.directory)).filter(path => /^events-\d+-[\w-]+\.jsonl$/.test(path)).sort();
      if (await Bun.file(identityPath).exists()) {
        const identity: unknown = await Bun.file(identityPath).json();
        if (!isRecord(identity) || !isUuid(identity.stream_id) || !isUuid(identity.runtime_id)
          || !Number.isSafeInteger(identity.sequence) || Number(identity.sequence) < 0) throw new Error('Invalid event stream identity');
        this.streamId = identity.stream_id;
        this.runtimeId = identity.runtime_id;
        this.committed = Number(identity.sequence);
      } else {
        if (paths.length) throw new Error('Event journal identity missing; refusing an invented replay');
        await this.saveIdentity();
      }
      let previous = 0;
      for (const name of paths) {
        const path = join(this.options.directory, name);
        const text = await Bun.file(path).text();
        const lines = text.split('\n');
        lines.pop(); // An incomplete final append was never acknowledged; abandon that tail.
        const events: TransportEvent[] = [];
        for (const line of lines) {
          const value: unknown = JSON.parse(line);
          if (!validEvent(value) || value.runtime_id !== this.runtimeId || value.cursor !== this.cursor(value.sequence)
            || value.event_id !== value.cursor || (previous !== 0 && value.sequence !== previous + 1)) throw new Error('Invalid or non-contiguous event journal');
          previous = value.sequence;
          events.push(value);
          this.lastOccurredAt = Math.max(this.lastOccurredAt, Date.parse(value.occurred_at));
          this.committed = Math.max(this.committed, value.sequence);
        }
        this.segments.push({ path, bytes: Buffer.byteLength(text), events });
      }
      if (previous !== 0 && previous < this.committed) throw new Error('Acknowledged event journal tail is missing');
      this.assigned = this.committed;
      await this.prune();
      this.available = true;
      this.maintenance = setInterval(() => {
        this.tail = this.tail.then(() => this.prune()).catch(error => this.fail(error));
      }, Math.min(this.maxAgeMs, 60_000));
      this.maintenance.unref();
    } catch (error) { this.fail(error); }
  }
  /** Ordered legacy-only delivery for credential-bearing UI responses. Never persisted. */
  deliverPrivate(deliver: () => void): void {
    if (this.scheduled) this.scheduleFlush();
    this.tail = this.tail.then(() => { deliver(); }).catch(error => { console.error('Daemon private event delivery failed:', error); });
  }
  append(input: JournalInput, deliver: () => void = () => undefined): void {
    // Private desktop/provider transport requests have separate authority and never enter replay.
    if (input.type.startsWith('provider.') || input.type === 'event_batch' || input.type === 'event_gap') { deliver(); return; }
    if (!this.available || this.closed) {
      if (this.scheduled) this.scheduleFlush();
      this.tail = this.tail.then(() => { deliver(); }).catch(error => { console.error('Daemon event delivery failed:', error); });
      return;
    }
    if (this.assigned >= Number.MAX_SAFE_INTEGER) {
      this.fail(new Error('Event stream sequence exhausted'));
      this.deliverPrivate(deliver);
      return;
    }
    const redacted: unknown = redactPayload(input.payload);
    let project = this.projectPaths.get(input.project_dir);
    if (!project) {
      project = resolveProjectDirectory(input.project_dir);
      if (this.projectPaths.size >= 1000) this.projectPaths.clear();
      this.projectPaths.set(input.project_dir, project);
    }
    this.lastOccurredAt = Math.max(this.lastOccurredAt, this.now());
    const event: TransportEvent = {
      event_id: `${this.streamId}:${++this.assigned}`, sequence: this.assigned, cursor: this.cursor(this.assigned),
      runtime_id: this.runtimeId, instance_id: this.instanceId, project_dir: project,
      ...(input.session_id === undefined ? {} : { session_id: input.session_id }),
      type: input.type, payload: isRecord(redacted) ? redacted : {}, occurred_at: new Date(this.lastOccurredAt).toISOString(),
    };
    // A pathological tool result must not defeat the total disk quota.
    const bounded = eventBytes(event) > Math.min(this.maxBytes, EVENT_HARD_BYTES) - 128 ? eventStub(event) : event;
    const line = JSON.stringify(bounded) + '\n';
    this.pending.push({ event: bounded, line, deliver });
    this.pendingBytes += Buffer.byteLength(line);
    if (this.pendingBytes > positive(this.options.maxPendingBytes, 4 * EVENT_HARD_BYTES)) this.fail(new Error('Event journal ingestion queue limit reached'));
    if (!this.scheduled) {
      this.scheduled = true;
      queueMicrotask(() => this.scheduleFlush());
    }
  }
  private scheduleFlush(): void {
    this.scheduled = false;
    const pending = this.pending.splice(0);
    if (!pending.length) return;
    this.tail = this.tail.then(async () => {
      try {
        if (this.available) {
          let index = 0;
          while (index < pending.length) {
            const first = pending[index]!;
            let segment = this.segments.at(-1);
            if (!this.file || !segment || segment.bytes + Buffer.byteLength(first.line) > this.segmentBytes) {
              if (this.options.fsync === 'batch') await this.file?.sync();
              await this.file?.close();
              const path = join(this.options.directory, `events-${String(first.event.sequence).padStart(16, '0')}-${this.instanceId}.jsonl`);
              this.file = await open(path, 'ax', 0o600);
              segment = { path, bytes: 0, events: [] };
              this.segments.push(segment);
            }
            const group: Pending[] = [];
            let bytes = 0;
            do {
              const item = pending[index]!;
              if (group.length && segment.bytes + bytes + Buffer.byteLength(item.line) > this.segmentBytes) break;
              group.push(item);
              bytes += Buffer.byteLength(item.line);
              index++;
            } while (index < pending.length);
            await this.options.beforeWrite?.();
            // The fd is O_APPEND. One asynchronous native write per batch/segment, not per token.
            const written = await Bun.write(Bun.file(this.file.fd), group.map(item => item.line).join(''));
            if (written !== bytes) throw new Error('Short event journal append');
            segment.bytes += bytes;
            segment.events.push(...group.map(item => item.event));
          }
          if (this.options.fsync === 'batch') await this.file?.sync();
          this.committed = pending.at(-1)!.event.sequence;
          await this.saveIdentity();
          await this.prune();
          this.notify(pending.map(item => item.event));
        }
      } catch (error) { this.fail(error); }
      finally {
        this.pendingBytes -= pending.reduce((sum, item) => sum + Buffer.byteLength(item.line), 0);
        for (const item of pending) {
          try { item.deliver(); }
          catch (error) { console.error('Daemon event delivery failed:', error); }
        }
      }
    });
  }
  async flush(): Promise<void> {
    if (this.scheduled) this.scheduleFlush();
    await this.tail;
  }
  async close(): Promise<void> {
    await this.flush();
    this.closed = true;
    if (this.maintenance) clearInterval(this.maintenance);
    await this.file?.close();
    this.file = undefined;
  }
  checkCursor(cursor?: string): { sequence: number; reason?: EventGapReason } {
    if (!this.available) return { sequence: this.committed, reason: 'journal_unavailable' };
    if (cursor === undefined) return { sequence: this.committed };
    const split = cursor.lastIndexOf(':');
    if (cursor.slice(0, split) !== this.streamId) return { sequence: this.committed, reason: 'stream_replaced' };
    const raw = cursor.slice(split + 1);
    const sequence = Number(raw);
    if (!/^\d+$/.test(raw) || !Number.isSafeInteger(sequence) || sequence > this.committed) return { sequence: this.committed, reason: 'stream_replaced' };
    const oldest = this.retainedEvents().next().value?.sequence ?? this.committed + 1;
    return sequence < oldest - 1 ? { sequence, reason: 'retention' } : { sequence };
  }
  read(afterCursor?: string, options: { limit?: number; maxBytes?: number; through?: number; accepts?: (event: TransportEvent) => boolean } = {}): JournalPage {
    const checked = this.checkCursor(afterCursor);
    const base = { watermark: this.watermark, oldest_cursor: this.oldestCursor, reset_required: Boolean(checked.reason) };
    if (checked.reason) return { ...base, events: [], next_cursor: afterCursor ?? this.watermark, has_more: false, reason: checked.reason };
    const events: WireTransportEvent[] = [];
    let sequence = checked.sequence;
    const through = options.through ?? this.committed;
    const maxBytes = Math.min(EVENT_HARD_BYTES, options.maxBytes ?? EVENT_BATCH_BYTES);
    const limit = options.limit ?? 500;
    let bytes = 2;
    let hasMore = false;
    for (const original of this.retainedEvents(checked.sequence, through)) {
      if (options.accepts && !options.accepts(original)) { sequence = original.sequence; continue; }
      let event: WireTransportEvent = eventBytes(original) + 320 > maxBytes ? eventStub(original) : original;
      if (eventBytes(event) + 322 > maxBytes) {
        // Very long paths/metadata must not turn a too-small page into an endless empty page.
        event = { event_id: original.event_id, sequence: original.sequence, cursor: original.cursor,
          runtime_id: original.runtime_id, instance_id: original.instance_id, type: original.type.slice(0, 32),
          payload: { truncated: true, original_bytes: eventBytes(original), type: original.type.slice(0, 32) },
          truncated: true, original_bytes: eventBytes(original) };
      }
      const size = eventBytes(event) + (events.length ? 1 : 0);
      if (events.length >= limit || bytes + size + 320 > maxBytes) { hasMore = true; break; }
      events.push(event);
      bytes += size;
      sequence = original.sequence;
    }
    if (!hasMore) sequence = through;
    return { ...base, events, next_cursor: this.cursor(sequence), has_more: hasMore };
  }
  private *retainedEvents(after = 0, through = this.committed): Generator<TransportEvent, undefined> {
    const cutoff = this.now() - this.maxAgeMs;
    for (const segment of this.segments) {
      if ((segment.events.at(-1)?.sequence ?? 0) <= after) continue;
      for (const event of segment.events) {
        if (event.sequence > through) return;
        if (event.sequence > after && Date.parse(event.occurred_at) > cutoff) yield event;
      }
    }
  }
  private async prune(): Promise<void> {
    let bytes = this.segments.reduce((sum, segment) => sum + segment.bytes, 0);
    const cutoff = this.now() - this.maxAgeMs;
    while (this.segments.length) {
      const first = this.segments[0]!;
      if (bytes <= this.maxBytes && first.events.some(event => Date.parse(event.occurred_at) > cutoff)) break;
      // Save the high watermark before deleting even the final retained segment.
      await this.saveIdentity();
      if (this.segments.length === 1) { await this.file?.close(); this.file = undefined; }
      await rm(first.path);
      bytes -= first.bytes;
      this.segments.shift();
    }
  }
  private async saveIdentity(): Promise<void> {
    const target = join(this.options.directory, 'identity.json');
    const temp = `${target}.${this.instanceId}.tmp`;
    const fd = await open(temp, 'w', 0o600);
    try {
      await Bun.write(Bun.file(fd.fd), JSON.stringify({ stream_id: this.streamId, runtime_id: this.runtimeId, sequence: this.committed }));
      if (this.options.fsync === 'batch') await fd.sync();
    } finally { await fd.close(); }
    await rename(temp, target);
  }
  private fail(error: unknown): void {
    this.available = false;
    console.error('Daemon event journal unavailable:', error instanceof Error ? error.message : String(error));
    this.notify(undefined);
  }
  private notify(events: readonly TransportEvent[] | undefined): void {
    for (const listener of this.listeners) {
      try { listener(events); }
      catch (error) { console.error('Daemon event subscription delivery failed:', error); }
    }
  }
}
export function eventStub(event: TransportEvent): TransportEvent {
  return { ...event, payload: { truncated: true, original_bytes: eventBytes(event), type: event.type } };
}
function positive(value: number | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error('Event journal quotas must be positive integers');
  return value;
}
function isRecord(value: unknown): value is Record<string, unknown> { return typeof value === 'object' && value !== null && !Array.isArray(value); }
function isUuid(value: unknown): value is string { return typeof value === 'string' && /^[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$/i.test(value); }
function validEvent(value: unknown): value is TransportEvent {
  return isRecord(value) && typeof value.event_id === 'string' && Number.isSafeInteger(value.sequence) && Number(value.sequence) > 0
    && typeof value.cursor === 'string' && isUuid(value.runtime_id) && isUuid(value.instance_id)
    && typeof value.project_dir === 'string' && (value.session_id === undefined || typeof value.session_id === 'string')
    && typeof value.type === 'string' && isRecord(value.payload) && typeof value.occurred_at === 'string' && Number.isFinite(Date.parse(value.occurred_at));
}
