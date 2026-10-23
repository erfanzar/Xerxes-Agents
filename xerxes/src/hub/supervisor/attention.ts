// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { HubWorkspaceView } from '../server.js'

/**
 * Deterministic attention watchers.
 *
 * The queue is filled from projections and events only, never from a model:
 * failed or blocked tasks, blocked goals, and people waiting on a pending
 * interaction. Items are deduplicated per target and kind, carry a severity,
 * and can be acknowledged. Subscribers are told once per activation, with
 * minimal text.
 *
 * A watcher never resolves an interaction. It has no backend, no daemon call
 * and no model: it only reads what the Hub already knows.
 */

export type AttentionKind = 'task_failed' | 'task_blocked' | 'goal_blocked' | 'human_waiting'
export type AttentionSeverity = 'info' | 'attention' | 'critical'
export type AttentionState = 'open' | 'acknowledged' | 'resolved'

export interface AttentionItem {
  readonly id: string
  readonly key: string
  readonly kind: AttentionKind
  readonly severity: AttentionSeverity
  readonly workspaceId: string
  readonly sessionId: string
  readonly title: string
  readonly since: string
  readonly state: AttentionState
  readonly acknowledgedBy: readonly string[]
  readonly resolvedAt?: string
}

export interface AttentionEvent {
  readonly workspace_id: string
  readonly session_id?: string
  readonly type: string
  readonly payload: Readonly<Record<string, unknown>>
}

export interface AttentionPush {
  readonly title: string
  readonly body: string
  readonly url: string
  readonly dedupeKey: string
}

export interface AttentionNotifier {
  /** People who may be told about this item (they can read its task). */
  recipients(item: AttentionItem): readonly string[]
  push(userId: string, message: AttentionPush): Promise<unknown>
}

export interface AttentionQueueOptions {
  readonly workspaces: Pick<HubWorkspaceView, 'list' | 'projection'>
  readonly clock?: () => number
  readonly notifier?: AttentionNotifier
  /** Fires when an item opens, is acknowledged or resolves. */
  readonly onChange?: (item: AttentionItem) => void
  readonly onError?: (error: unknown, context: string) => void
  /** Resolved items are kept this long for display. Default 1 hour. */
  readonly retainResolvedMs?: number
}

const SEVERITY: Readonly<Record<AttentionKind, AttentionSeverity>> = {
  task_failed: 'critical',
  task_blocked: 'attention',
  goal_blocked: 'attention',
  human_waiting: 'attention',
}

/** Lock-screen text: no task title, no transcript, no arguments. */
const PUSH_BODY: Readonly<Record<AttentionKind, string>> = {
  task_failed: 'A task failed.',
  task_blocked: 'A task is blocked.',
  goal_blocked: 'A goal is blocked.',
  human_waiting: 'A task is waiting for you.',
}

export function attentionToWire(item: AttentionItem): Record<string, unknown> {
  return {
    id: item.id,
    kind: item.kind,
    severity: item.severity,
    workspace_id: item.workspaceId,
    session_id: item.sessionId,
    title: item.title,
    since: item.since,
    state: item.state,
    ...(item.resolvedAt === undefined ? {} : { resolved_at: item.resolvedAt }),
  }
}

export class AttentionQueue {
  private readonly items = new Map<string, AttentionItem>()
  private readonly blockedGoals = new Set<string>()
  private readonly clock: () => number
  private sequence = 0
  private timer: ReturnType<typeof setInterval> | undefined

  constructor(private readonly options: AttentionQueueOptions) {
    this.clock = options.clock ?? Date.now
  }

  start(intervalMs = 5000): void {
    if (this.timer !== undefined) return
    this.timer = setInterval(() => { void this.scan() }, intervalMs)
    this.timer.unref?.()
  }

  stop(): void {
    if (this.timer !== undefined) clearInterval(this.timer)
    this.timer = undefined
  }

  private static key(kind: AttentionKind, workspaceId: string, sessionId: string): string {
    return `${kind}\u0000${workspaceId}\u0000${sessionId}`
  }

  /** Goal-blocked is carried by a status event, not by the projection. */
  observeEvents(events: readonly AttentionEvent[]): void {
    for (const event of events) {
      if (event.session_id === undefined || event.type !== 'status_update') continue
      const phase = event.payload.goal_phase
      if (typeof phase !== 'string') continue
      const key = AttentionQueue.key('goal_blocked', event.workspace_id, event.session_id)
      if (phase === 'blocked') this.blockedGoals.add(key)
      else this.blockedGoals.delete(key)
    }
  }

  /** Evaluate every projection once. Returns the items that opened during this scan. */
  async scan(): Promise<AttentionItem[]> {
    const active = new Set<string>()
    const opened: AttentionItem[] = []
    for (const workspace of this.options.workspaces.list()) {
      const projection = this.options.workspaces.projection(workspace.id)
      if (!projection) continue
      for (const row of Object.values(projection.sessions)) {
        const kinds: AttentionKind[] = []
        if (row.pending_interaction_count > 0) kinds.push('human_waiting')
        if (row.status === 'failed') kinds.push('task_failed')
        if (row.interrupted === true) kinds.push('task_blocked')
        if (this.blockedGoals.has(AttentionQueue.key('goal_blocked', workspace.id, row.id))) kinds.push('goal_blocked')
        if (row.stale) {
          // Contact is lost: keep what is known, neither raise nor clear.
          for (const kind of ['human_waiting', 'task_failed', 'task_blocked', 'goal_blocked'] as const) {
            const key = AttentionQueue.key(kind, workspace.id, row.id)
            if (this.items.get(key)?.state !== undefined && this.items.get(key)!.state !== 'resolved') active.add(key)
          }
          if (kinds.length === 0) continue
        }
        for (const kind of kinds) {
          const key = AttentionQueue.key(kind, workspace.id, row.id)
          active.add(key)
          const existing = this.items.get(key)
          if (existing && existing.state !== 'resolved') continue
          const item: AttentionItem = {
            id: `att-${(this.sequence += 1)}`,
            key,
            kind,
            severity: SEVERITY[kind],
            workspaceId: workspace.id,
            sessionId: row.id,
            title: row.title,
            since: new Date(this.clock()).toISOString(),
            state: 'open',
            acknowledgedBy: [],
          }
          this.items.set(key, item)
          opened.push(item)
        }
      }
    }
    for (const [key, item] of this.items) {
      if (item.state === 'resolved' || active.has(key)) continue
      const resolved: AttentionItem = { ...item, state: 'resolved', resolvedAt: new Date(this.clock()).toISOString() }
      this.items.set(key, resolved)
      this.changed(resolved)
    }
    this.prune()
    for (const item of opened) this.changed(item)
    await Promise.all(opened.map(item => this.notify(item)))
    return opened
  }

  list(filter: { readonly includeResolved?: boolean } = {}): AttentionItem[] {
    return [...this.items.values()]
      .filter(item => filter.includeResolved === true || item.state !== 'resolved')
      .sort((a, b) => (a.severity === b.severity ? a.since.localeCompare(b.since) : a.severity === 'critical' ? -1 : 1))
  }

  /** Acknowledge an item for one person. Returns the item, or undefined when it does not exist. */
  acknowledge(itemId: string, userId: string): AttentionItem | undefined {
    for (const [key, item] of this.items) {
      if (item.id !== itemId) continue
      if (item.state === 'resolved') return item
      const next: AttentionItem = {
        ...item,
        state: 'acknowledged',
        acknowledgedBy: item.acknowledgedBy.includes(userId) ? item.acknowledgedBy : [...item.acknowledgedBy, userId],
      }
      this.items.set(key, next)
      this.changed(next)
      return next
    }
    return undefined
  }

  private changed(item: AttentionItem): void {
    try { this.options.onChange?.(item) } catch (error) { this.options.onError?.(error, 'attention change') }
  }

  private async notify(item: AttentionItem): Promise<void> {
    const notifier = this.options.notifier
    if (!notifier) return
    try {
      for (const userId of notifier.recipients(item)) {
        if (item.acknowledgedBy.includes(userId)) continue
        try {
          await notifier.push(userId, {
            title: 'XERXES',
            body: PUSH_BODY[item.kind],
            url: `/w/${encodeURIComponent(item.workspaceId)}/s/${encodeURIComponent(item.sessionId)}`,
            dedupeKey: `${item.key}@${item.since}`,
          })
        } catch (error) {
          this.options.onError?.(error, 'attention push')
        }
      }
    } catch (error) {
      this.options.onError?.(error, 'attention recipients')
    }
  }

  private prune(): void {
    const cutoff = this.clock() - (this.options.retainResolvedMs ?? 60 * 60 * 1000)
    for (const [key, item] of this.items) {
      if (item.state === 'resolved' && item.resolvedAt !== undefined && Date.parse(item.resolvedAt) < cutoff) this.items.delete(key)
    }
  }
}
