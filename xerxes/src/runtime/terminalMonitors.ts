// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { ActivityChanges } from './activityChanges.js'
import type { ReactionMailbox, ReactionHealth, ReactionPolicyEdit } from './reactionMailbox.js'
import type { RunHistory, MonitorConfiguration } from './runHistory.js'
import type { FileMonitorSource } from './fileMonitorSource.js'
import type { WebSocketMonitorSource, WebSocketMonitorEvent } from './websocketMonitorSource.js'
import type { WebhookMonitorSource } from './webhookMonitorSource.js'
import type { TerminalInspection, TerminalRegistry } from './terminalRegistry.js'

export interface MonitorReaction { readonly maxTotalTokens?: number; readonly maxReactions: number; readonly maxDurationMs: number }
export interface MonitorEvent { readonly sequence: number; readonly text: string; readonly at: number }
export interface MonitorSummary {
  readonly source?: NonNullable<MonitorConfiguration['source']>
  readonly stopAction?: 'stop-watch' | 'cancel-reactions' | null
  readonly sourceStatus?: string
  readonly deliveryError?: string
  readonly trigger: 'output' | 'completion' | 'change'
  readonly id: string; readonly terminalId: string; readonly owner: string; readonly match: string
  readonly state: 'watching' | 'stopped' | 'expired' | 'source-ended' | 'limit-reached' | 'failed' | 'archived' | 'interrupted' | 'detached'
  readonly reactionHealth?: ReactionHealth; readonly reaction?: MonitorReaction; readonly error?: string; readonly events: readonly MonitorEvent[]; readonly expiresAt: number; readonly droppedEvents: number
}
interface Watch {
  source?: NonNullable<MonitorConfiguration['source']>;
  lastFileIdentity?: string;
  sourceStatus?: string;
  deliveryError?: string;
  trigger: 'output' | 'completion' | 'change'; id: string; terminalId: string; owner: string; match: string; state: MonitorSummary['state'];
  reaction?: MonitorReaction; error?: string; events: MonitorEvent[]; expiresAt: number; droppedEvents: number; sequence: number;
  partial: string; matchedLine?: string; prefixOmitted: boolean; suffixOmitted: boolean; seen: Set<string>; maxEvents: number; unsubscribe: () => void; timer?: ReturnType<typeof setTimeout>
}
/** Event-driven source watches. No model polling and no ownership of source processes. */
export class TerminalMonitors {
  readonly activityChanges = new ActivityChanges()
  private readonly watches = new Map<string, Watch>()
  private readonly opening = new Set<{ owner: string; controller: AbortController }>()
  constructor(private readonly terminals: TerminalRegistry, private readonly history: RunHistory,
    private readonly onEvent?: (monitor: MonitorSummary, event: MonitorEvent) => void,
    private readonly onError: (error: unknown) => void = error => console.error('Monitor failed:', error),
    private readonly reactionMailbox?: ReactionMailbox,
    private readonly files?: { readonly source: FileMonitorSource; readonly resolveWorkspace: (owner: string) => string },
    private readonly websockets?: { readonly source: WebSocketMonitorSource; readonly resolveWorkspace: (owner: string) => string },
    private readonly webhooks?: { readonly source: WebhookMonitorSource; readonly resolveWorkspace: (owner: string) => string }) {}

  webhookSources(owner: string): readonly { readonly name: string }[] {
    if (!owner.trim()) throw new Error('Monitor requires a session owner')
    return this.webhooks?.source.list() ?? []
  }

  async startWebhook(owner: string, options: { name: string; match: string; durationMs?: number; maxEvents?: number; reaction?: MonitorReaction; signal?: AbortSignal }): Promise<MonitorSummary> {
    if (!this.webhooks) throw new Error('Webhook monitoring is unavailable on this host')
    if (!owner.trim()) throw new Error('Monitor requires a session owner')
    if (options.reaction && !this.reactionMailbox) throw new Error('Automatic monitor reactions are unavailable on this host')
    const name = options.name
    if (!/^[a-zA-Z0-9_-]{1,64}$/.test(name)) throw new Error('Webhook name must match [a-zA-Z0-9_-]{1,64}')
    const match = options.match.trim(), duration = options.durationMs ?? 3_600_000, maxEvents = options.maxEvents ?? 50
    if (!match || match.length > 1024) throw new Error('Monitor match must contain 1–1024 characters')
    if (!Number.isSafeInteger(duration) || duration < 100 || duration > 86_400_000) throw new Error('Monitor duration must be 100–86400000ms')
    if (!Number.isSafeInteger(maxEvents) || maxEvents < 1 || maxEvents > 1000) throw new Error('Monitor maxEvents must be 1–1000')
    this.assertCapacity(owner)
    const pending = { owner, controller: new AbortController() }
    this.opening.add(pending)
    const abort = () => pending.controller.abort(options.signal?.reason)
    options.signal?.addEventListener('abort', abort, { once: true })
    if (options.signal?.aborted) abort()
    let watch: Watch | undefined
    let subscription: Awaited<ReturnType<WebhookMonitorSource['open']>> | undefined
    const queued: Array<{ text: string; identity: string }> = []
    let openingError: unknown
    const consume = (event: { text: string; identity: string }) => {
      if (!watch) { if (queued.length < 32) queued.push(event); else openingError = new Error('Webhook source exceeded attachment event capacity'); return }
      if (watch.state !== 'watching' || !event.text.toLowerCase().includes(match.toLowerCase()) || watch.seen.has(event.identity)) return
      try {
        watch.seen.add(event.identity)
        if (watch.seen.size > 256) watch.seen.delete(watch.seen.values().next().value!)
        const text = `[Webhook message] ${event.text}`
        this.recordEvent(watch, text.length > 8192 ? `${text.slice(0, 8160)} [message truncated]` : text)
      } catch (error) { this.fail(watch, error) }
    }
    try {
      const workspace = this.webhooks.resolveWorkspace(owner)
      subscription = await this.webhooks.source.open(name, consume, error => {
        if (watch) { if (watch.state === 'watching') this.fail(watch, error) } else openingError = error
      }, pending.controller.signal)
      pending.controller.signal.throwIfAborted()
      if (openingError !== undefined) throw openingError
      const source = { kind: 'webhook' as const, name: subscription.name }, expiresAt = Date.now() + duration
      const run = this.history.startMonitor({ ownerSessionId: owner, workspace, kind: 'monitor', sourceId: source.name, title: `Watch Webhook ${source.name}` }, { trigger: 'output', match, expiresAt, source })
      watch = { source, trigger: 'output', id: run.id, terminalId: '', owner, match, state: 'watching', events: [], expiresAt,
        ...(options.reaction ? { reaction: { ...options.reaction } } : {}), droppedEvents: 0, sequence: 0, partial: '', prefixOmitted: false, suffixOmitted: false, seen: new Set(), maxEvents, unsubscribe: subscription.close, sourceStatus: 'Watching webhook messages' }
      this.watches.set(watch.id, watch)
    this.activityChanges.notify()
      if (options.reaction) this.reactionMailbox!.configure({ owner, runId: watch.id, expiresAt, ...options.reaction })
      const active = watch
      watch.timer = setTimeout(() => this.finish(active, 'expired'), duration)
      watch.timer.unref?.()
      for (const event of queued) consume(event)
      return this.snapshot(watch)
    } catch (error) {
      subscription?.close()
      if (watch) this.fail(watch, error)
      throw error
    } finally { this.opening.delete(pending); options.signal?.removeEventListener('abort', abort) }
  }

  async startWebSocket(owner: string, options: { url: string; match: string; durationMs?: number; maxEvents?: number; reaction?: MonitorReaction; signal?: AbortSignal }): Promise<MonitorSummary> {
    if (!this.websockets) throw new Error('WebSocket monitoring is unavailable on this host')
    if (!owner.trim()) throw new Error('Monitor requires a session owner')
    if (options.reaction && !this.reactionMailbox) throw new Error('Automatic monitor reactions are unavailable on this host')
    const match = options.match.trim(), duration = options.durationMs ?? 3_600_000, maxEvents = options.maxEvents ?? 50
    if (!match || match.length > 1024) throw new Error('Monitor match must contain 1–1024 characters')
    if (!Number.isSafeInteger(duration) || duration < 100 || duration > 86_400_000) throw new Error('Monitor duration must be 100–86400000ms')
    if (!Number.isSafeInteger(maxEvents) || maxEvents < 1 || maxEvents > 1000) throw new Error('Monitor maxEvents must be 1–1000')
    this.assertCapacity(owner)
    const pending = { owner, controller: new AbortController() }
    this.opening.add(pending)
    const abort = () => pending.controller.abort(options.signal?.reason)
    options.signal?.addEventListener('abort', abort, { once: true })
    if (options.signal?.aborted) abort()
    let watch: Watch | undefined
    let subscription: Awaited<ReturnType<WebSocketMonitorSource['open']>> | undefined
    const queued: WebSocketMonitorEvent[] = []
    let openingError: unknown, status = 'Connecting to WebSocket source'
    const consume = (event: WebSocketMonitorEvent) => {
      if (!watch) { if (queued.length < 32) queued.push(event); else openingError = new Error('WebSocket source exceeded attachment event capacity'); return }
      if (watch.state !== 'watching') return
      if (event.kind === 'message' && !event.text.toLowerCase().includes(match.toLowerCase())) return
      if (watch.seen.has(event.identity)) return
      try {
        watch.seen.add(event.identity)
        if (watch.seen.size > 256) watch.seen.delete(watch.seen.values().next().value!)
        const text = event.kind === 'gap' ? `[Observation gap] ${event.text}` : `[WebSocket message] ${event.text}`
        this.recordEvent(watch, text.length > 8192 ? `${text.slice(0, 8160)} [message truncated]` : text)
      } catch (error) { this.fail(watch, error) }
    }
    try {
      const workspace = this.websockets.resolveWorkspace(owner)
      subscription = await this.websockets.source.open(options.url, consume, next => {
        status = next.slice(0, 8192)
        if (watch?.state === 'watching') watch.sourceStatus = status
      }, error => {
        if (watch) { if (watch.state === 'watching') this.fail(watch, error) } else openingError = error
      }, pending.controller.signal)
      pending.controller.signal.throwIfAborted()
      if (openingError !== undefined) throw openingError
      const source = { kind: 'websocket' as const, url: subscription.url }, expiresAt = Date.now() + duration
      const run = this.history.startMonitor({ ownerSessionId: owner, workspace, kind: 'monitor', sourceId: source.url, title: `Watch WebSocket ${source.url}` }, { trigger: 'output', match, expiresAt, source })
      watch = { source, trigger: 'output', id: run.id, terminalId: '', owner, match, state: 'watching', events: [], expiresAt,
        ...(options.reaction ? { reaction: { ...options.reaction } } : {}), droppedEvents: 0, sequence: 0, partial: '', prefixOmitted: false, suffixOmitted: false, seen: new Set(), maxEvents, unsubscribe: subscription.close, sourceStatus: status }
      this.watches.set(watch.id, watch)
    this.activityChanges.notify()
      if (options.reaction) this.reactionMailbox!.configure({ owner, runId: watch.id, expiresAt, ...options.reaction })
      const active = watch
      watch.timer = setTimeout(() => this.finish(active, 'expired'), duration)
      watch.timer.unref?.()
      for (const event of queued) consume(event)
      return this.snapshot(watch)
    } catch (error) {
      subscription?.close()
      if (watch) this.fail(watch, error)
      throw error
    } finally { this.opening.delete(pending); options.signal?.removeEventListener('abort', abort) }
  }

  async startFile(owner: string, options: { path: string; durationMs?: number; maxEvents?: number; reaction?: MonitorReaction; signal?: AbortSignal }): Promise<MonitorSummary> {
    if (!this.files) throw new Error('File monitoring is unavailable on this host')
    if (!owner.trim()) throw new Error('Monitor requires a session owner')
    if (options.reaction && !this.reactionMailbox) throw new Error('Automatic monitor reactions are unavailable on this host')
    const duration = options.durationMs ?? 3_600_000, maxEvents = options.maxEvents ?? 50
    if (!Number.isSafeInteger(duration) || duration < 100 || duration > 86_400_000) throw new Error('Monitor duration must be 100–86400000ms')
    if (!Number.isSafeInteger(maxEvents) || maxEvents < 1 || maxEvents > 1000) throw new Error('Monitor maxEvents must be 1–1000')
    this.assertCapacity(owner)
    const pending = { owner, controller: new AbortController() }
    this.opening.add(pending)
    const abort = () => pending.controller.abort(options.signal?.reason)
    options.signal?.addEventListener('abort', abort, { once: true })
    if (options.signal?.aborted) abort()
    let watch: Watch | undefined
    let subscription: Awaited<ReturnType<FileMonitorSource['open']>> | undefined
    const queued: Array<{ text: string; identity: string }> = []
    let openingError: unknown
    const consume = (event: { text: string; identity: string }) => {
      if (!watch) { if (queued.length < 32) queued.push(event); else openingError = new Error('File changed too often during attachment; create the watch again'); return }
      if (watch.state !== 'watching' || event.identity === watch.lastFileIdentity) return
      try {
        watch.lastFileIdentity = event.identity
        this.recordEvent(watch, event.text.slice(0, 8192))
      } catch (error) { this.fail(watch, error) }
    }
    try {
      subscription = await this.files.source.open(this.files.resolveWorkspace(owner), options.path, consume, error => {
        if (watch) { if (watch.state === 'watching') this.fail(watch, error) } else openingError = error
      }, pending.controller.signal)
      pending.controller.signal.throwIfAborted()
      if (openingError !== undefined) throw openingError
      const source = { kind: 'file' as const, path: subscription.path, workspace: subscription.workspace }
      const expiresAt = Date.now() + duration
      const run = this.history.startMonitor({ ownerSessionId: owner, workspace: source.workspace, kind: 'monitor', sourceId: source.path, title: `Watch file ${source.path}` }, { trigger: 'change', match: 'File metadata changes', expiresAt, source })
      watch = { source, trigger: 'change', id: run.id, terminalId: '', owner, match: 'File metadata changes', state: 'watching', events: [], expiresAt,
        ...(options.reaction ? { reaction: { ...options.reaction } } : {}), droppedEvents: 0, sequence: 0, partial: '', prefixOmitted: false, suffixOmitted: false, seen: new Set(), maxEvents, unsubscribe: subscription.close,
        sourceStatus: 'Watching file metadata changes. Rapid changes may coalesce; file contents are not read.' }
      this.watches.set(watch.id, watch)
    this.activityChanges.notify()
      if (options.reaction) this.reactionMailbox!.configure({ owner, runId: watch.id, expiresAt, ...options.reaction })
      const active = watch
      watch.timer = setTimeout(() => this.finish(active, 'expired'), duration)
      watch.timer.unref?.()
      for (const event of queued) consume(event)
      return this.snapshot(watch)
    } catch (error) {
      subscription?.close()
      if (watch) this.fail(watch, error)
      throw error
    } finally { this.opening.delete(pending); options.signal?.removeEventListener('abort', abort) }
  }

  private assertCapacity(owner: string): void {
    const active = [...this.watches.values()].filter(w => w.state === 'watching')
    if (active.filter(w => w.owner === owner).length + [...this.opening].filter(w => w.owner === owner).length >= 16) throw new Error('Session monitor limit reached')
    if (active.length + this.opening.size >= 128) throw new Error('Host monitor limit reached')
  }

  start(owner: string, options: { terminalId: string; match?: string; trigger?: 'output' | 'completion'; durationMs?: number; maxEvents?: number; reaction?: MonitorReaction }): MonitorSummary {
    if (options.reaction && !this.reactionMailbox) throw new Error('Automatic monitor reactions are unavailable on this host')
    if (!owner.trim()) throw new Error('Monitor requires a session owner')
    const trigger = options.trigger ?? 'output'
    if (trigger !== 'output' && trigger !== 'completion') throw new Error('Invalid monitor trigger')
    const match = trigger === 'completion' ? 'Command completion' : (options.match ?? '').trim()
    if (!match || match.length > 1024) throw new Error('Monitor match must contain 1–1024 characters')
    const duration = options.durationMs ?? 3_600_000
    const maxEvents = options.maxEvents ?? 50
    if (!Number.isSafeInteger(duration) || duration < 100 || duration > 86_400_000) throw new Error('Monitor duration must be 100–86400000ms')
    if (!Number.isSafeInteger(maxEvents) || maxEvents < 1 || maxEvents > 1000) throw new Error('Monitor maxEvents must be 1–1000')
    this.assertCapacity(owner)
    const terminal = this.terminals.inspect(owner, options.terminalId)
    if (!terminal || (trigger === 'output' && !terminal.running)) throw new Error('Monitor requires a live terminal owned by this session')
    const expiresAt = Date.now() + duration
    const run = this.history.startMonitor({ ownerSessionId: owner, workspace: terminal.cwd, kind: 'monitor', sourceId: terminal.id, title: `Watch ${terminal.label}: ${match}` }, { trigger, match, expiresAt })
    const watch: Watch = { trigger, id: run.id, terminalId: terminal.id, owner, match, state: 'watching', events: [], expiresAt,
      ...(options.reaction ? { reaction: { ...options.reaction } } : {}), droppedEvents: 0, sequence: 0, partial: '', prefixOmitted: false, suffixOmitted: false, seen: new Set(), maxEvents, unsubscribe: () => {} }
    this.watches.set(watch.id, watch)
    this.activityChanges.notify()
    try {
      if (options.reaction) this.reactionMailbox!.configure({ owner, runId: watch.id, expiresAt: watch.expiresAt, ...options.reaction })
      if (trigger === 'completion' && !terminal.running) {
        this.completeCommand(watch, terminal)
        return this.snapshot(watch)
      }
      watch.unsubscribe = this.terminals.subscribe(owner, terminal.id, event => {
        try {
          if (trigger === 'completion') {
            if (event.closed) {
              const finished = this.terminals.inspect(owner, terminal.id, 8192)
              if (!finished) throw new Error('Completed terminal evidence unavailable')
              this.completeCommand(watch, finished)
            }
          } else this.consume(watch, event.text, event.closed)
          if (event.closed && watch.state === 'watching') this.finish(watch, 'source-ended')
        } catch (error) { this.fail(watch, error) }
      })
      watch.timer = setTimeout(() => {
        try { this.finish(watch, 'expired') }
        catch (error) { console.error('Monitor expiry persistence failed:', error) }
      }, duration)
      watch.timer.unref?.()
      // Only future output is watched. Existing terminal tails may begin in
      // the middle of a line and must not manufacture a new event at attach.
      return this.snapshot(watch)
    } catch (error) { this.reactionMailbox?.cancel(owner, watch.id); this.finish(watch, 'stopped'); throw error }
  }

  list(owner: string): MonitorSummary[] {
    const live = [...this.watches.values()].filter(w => w.owner === owner).map(w => this.snapshot(w))
    const ids = new Set(live.map(w => w.id))
    try { return [...live, ...this.history.monitorRuns(owner).filter(run => !ids.has(run.id)).map(run => this.archived(owner, run.id)!)] }
    catch (error) {
      if (!live.length) throw error
      const unavailable = 'Archived watches unavailable: ' + (error instanceof Error ? error.message : String(error))
      return live.map(watch => ({ ...watch, error: [watch.error, unavailable].filter(Boolean).join('\n') }))
    }
  }
  private archived(owner: string, id: string): MonitorSummary | undefined {
    const configuration = this.history.monitorConfiguration(owner, id)
    const run = this.history.inspect(owner, id)
    if (!configuration || !run) return undefined
    const last = this.history.eventCursor(owner, id)
    const events = this.history.events(owner, id, Math.max(0, last - 20), 20).events
    const reactionHealth = this.reactionMailbox?.inspect(owner, id)
    const state = run.state === 'running' ? 'detached' : run.state === 'interrupted' ? 'interrupted' : 'archived'
    return { ...configuration, id, owner, terminalId: configuration.source && configuration.source.kind !== 'terminal' ? '' : run.sourceId, state, stopAction: stopAction(state, reactionHealth),
      events, droppedEvents: Math.max(0, last - events.length),
      sourceStatus: run.state === 'running' ? 'Watch is not attached to this daemon. Its owning process may still be running.' : configuration.source?.kind === 'websocket' ? 'WebSocket watch is no longer attached. Messages during downtime were not observed; create a new watch.' : configuration.source?.kind === 'file' ? 'File watch is no longer attached. Changes during downtime were not observed; create a new watch to establish a new baseline.' : configuration.source?.kind === 'webhook' ? 'Webhook watch is no longer attached. Deliveries during downtime were not observed; create a new watch.' : 'Stored watch outcome: ' + run.state + '. No live source is attached.',
      ...(run.error ? { error: run.error } : {}),
      ...(reactionHealth ? { reactionHealth } : {}),
    }
  }
  inspect(owner: string, id: string): MonitorSummary {
    const watch = this.watches.get(id)
    if (!watch) { const archived = this.archived(owner, id); if (archived) return archived }
    if (!watch || watch.owner !== owner) throw new Error('Unknown monitor')
    return this.snapshot(watch)
  }
  stop(owner: string, id: string): MonitorSummary {
    const watch = this.watches.get(id)
    if (!watch) {
      const archived = this.archived(owner, id)
      if (!archived) throw new Error('Unknown monitor')
      if (archived.state === 'detached') throw new Error('Watch belongs to another daemon; stop it through its owner')
      this.reactionMailbox?.cancel(owner, id)
      return this.archived(owner, id)!
    }
    if (!watch || watch.owner !== owner) throw new Error('Unknown monitor')
    this.reactionMailbox?.cancel(owner, id)
    this.finish(watch, 'stopped')
    return this.snapshot(watch)
  }
  updateReaction(owner: string, id: string, edit: ReactionPolicyEdit): MonitorSummary {
    const current = this.inspect(owner, id)
    if (current.state === 'detached') throw new Error('Watch belongs to another daemon; edit through its owner')
    if (!this.reactionMailbox) throw new Error('Reaction host unavailable')
    this.reactionMailbox.updatePolicy(owner, id, edit)
    const live = this.watches.get(id)
    if (live) live.reaction = { maxReactions: edit.maxReactions, maxDurationMs: edit.maxDurationMs, ...(edit.maxTotalTokens === null ? {} : { maxTotalTokens: edit.maxTotalTokens }) }
    return this.inspect(owner, id)
  }
  disposeOwner(owner: string): void { for (const item of this.opening) if (item.owner === owner) item.controller.abort(); this.reactionMailbox?.cancel(owner); for (const watch of this.watches.values()) if (watch.owner === owner) this.finish(watch, 'stopped') }
  close(): void { for (const item of this.opening) item.controller.abort(); for (const watch of this.watches.values()) this.finish(watch, watch.source && watch.source.kind !== 'terminal' ? 'interrupted' : 'stopped') }

  private recordEvent(watch: Watch, text: string): void {
    const event = { sequence: ++watch.sequence, text, at: Date.now() }
    watch.events.push(event)
    if (watch.events.length > 100) { watch.events.shift(); watch.droppedEvents++ }
    this.history.appendEvent(watch.owner, watch.id, event, this.output(watch), watch.droppedEvents > 0)
    this.notify(watch, event)
    if (watch.sequence >= watch.maxEvents) this.finish(watch, 'limit-reached')
  }

  private consume(watch: Watch, chunk: string, flush: boolean): void {
    if (watch.state !== 'watching') return
    // Scan bounded segments so an arbitrarily large write cannot allocate a
    // second unbounded string/line array. Preserve the first match independently
    // of the rolling search tail until the logical line ends.
    let offset = 0
    while (offset < chunk.length && watch.state === 'watching') {
      const newline = chunk.indexOf('\n', offset)
      const end = newline < 0 ? chunk.length : newline
      while (offset < end) {
        const next = Math.min(end, offset + 4096)
        this.appendSegment(watch, chunk.slice(offset, next))
        offset = next
      }
      if (newline < 0) break
      this.completeLine(watch)
      offset = newline + 1
    }
    if (flush && watch.state === 'watching') this.completeLine(watch)
  }

  private appendSegment(watch: Watch, segment: string): void {
    if (watch.matchedLine !== undefined) {
      const available = 8192 - watch.matchedLine.length
      watch.matchedLine += segment.slice(0, available)
      if (segment.length > available) watch.suffixOmitted = true
      return
    }
    const text = watch.partial + segment
    const matchedAt = text.toLowerCase().indexOf(watch.match.toLowerCase())
    if (matchedAt >= 0) {
      const from = Math.max(0, matchedAt - 256)
      watch.prefixOmitted ||= from > 0
      watch.matchedLine = text.slice(from, from + 8192)
      watch.suffixOmitted = text.length > from + 8192
      watch.partial = ''
    } else {
      watch.prefixOmitted ||= text.length > 8192
      watch.partial = text.slice(-8192)
    }
  }

  private completeLine(watch: Watch): void {
    const raw = watch.matchedLine
    const prefix = watch.prefixOmitted
    const suffix = watch.suffixOmitted
    watch.partial = ''
    delete watch.matchedLine
    watch.prefixOmitted = false
    watch.suffixOmitted = false
    if (raw !== undefined) {
      const line = `${prefix ? '[prefix omitted] ' : ''}${raw.replace(/\r$/, '')}${suffix ? ' [suffix omitted]' : ''}`
      if (watch.seen.has(line)) return
      watch.seen.add(line)
      if (watch.seen.size > 256) watch.seen.delete(watch.seen.values().next().value!)
      const event = { sequence: ++watch.sequence, text: line, at: Date.now() }
      watch.events.push(event)
      if (watch.events.length > 100) { watch.events.shift(); watch.droppedEvents++ }
      this.history.appendEvent(watch.owner, watch.id, event, this.output(watch), watch.droppedEvents > 0)
      this.notify(watch, event)
      if (watch.sequence >= watch.maxEvents) this.finish(watch, 'limit-reached')
    }
  }
  private completeCommand(watch: Watch, terminal: TerminalInspection): void {
    if (watch.state !== 'watching') return
    const details = {
      terminal_id: terminal.id.slice(0, 128), command: terminal.command.slice(0, 256), cwd: terminal.cwd.slice(0, 256),
      metadata_truncated: terminal.id.length > 128 || terminal.command.length > 256 || terminal.cwd.length > 256,
      exit_code: terminal.exitCode, output: terminal.output.slice(-8192),
      output_truncated: terminal.outputTruncated || terminal.output.length > 8192,
    }
    let text = JSON.stringify(details)
    while (text.length > 8300) {
      details.output = details.output.slice(Math.ceil(details.output.length / 2))
      details.output_truncated = true
      text = JSON.stringify(details)
    }
    const event = { sequence: ++watch.sequence, at: Date.now(), text }
    watch.events.push(event)
    this.history.appendEvent(watch.owner, watch.id, event, this.output(watch), false)
    this.finish(watch, 'source-ended')
    const summary = this.snapshot(watch)
    if (summary.state !== 'failed') this.notify(watch, event)
  }
  private notify(watch: Watch, event: MonitorEvent): void {
    try {
      this.onEvent?.(this.snapshot(watch), event)
      delete watch.deliveryError
    } catch (error) {
      // The durable event is already committed. Notification failure must not
      // rewrite execution history or revoke an already-enqueued reaction.
      watch.deliveryError = (error instanceof Error ? error.message : String(error)).slice(0, 8192)
      try { this.onError(error) } catch (observerError) {
        console.error('Monitor notification error observer failed:', observerError)
      }
    }
  }
  private finish(watch: Watch, state: MonitorSummary['state']): void {
    if (watch.state !== 'watching') return
    watch.unsubscribe()
    if (watch.timer) clearTimeout(watch.timer)
    watch.state = state
    if (watch.source && state !== 'interrupted') watch.sourceStatus = `Source closed (${state}).`
    watch.partial = ''
    delete watch.matchedLine
    if (state === 'interrupted' && watch.source?.kind === 'file') watch.sourceStatus = 'File watch interrupted. Changes during downtime were not observed; create a new watch.'
    if (state === 'interrupted' && watch.source?.kind === 'websocket') watch.sourceStatus = 'WebSocket watch interrupted. Messages during downtime were not observed; create a new watch.'
    if (state === 'interrupted' && watch.source?.kind === 'webhook') watch.sourceStatus = 'Webhook watch interrupted. Deliveries during downtime were not observed; create a new watch.'
    try { this.history.finish(watch.owner, watch.id, state === 'stopped' ? 'cancelled' : state === 'interrupted' ? 'interrupted' : 'succeeded', {
      output: `${state} · ${watch.sequence} matches · ${watch.droppedEvents} older events omitted\n${this.output(watch)}`,
      outputTruncated: watch.droppedEvents > 0, notify: watch.sequence > 0,
    }) } catch (error) { this.fail(watch, error) }
    this.activityChanges.notify()
    this.pruneCompleted()
  }
  private pruneCompleted(): void {
    // Failed watches count too: repeated storage failures must not grow memory.
    const completed = [...this.watches.values()].filter(w => w.state !== 'watching')
    for (const old of completed.slice(0, Math.max(0, completed.length - 100))) this.watches.delete(old.id)
  }
  private fail(watch: Watch, error: unknown): void {
    watch.unsubscribe()
    if (watch.timer) clearTimeout(watch.timer)
    this.reactionMailbox?.cancel(watch.owner, watch.id)
    watch.state = 'failed'
    this.activityChanges.notify()
    watch.partial = ''
    delete watch.matchedLine
    watch.error = error instanceof Error ? error.message : String(error)
    try {
      this.history.finish(watch.owner, watch.id, 'failed', { error: watch.error, output: this.output(watch) })
    } catch (persistenceError) {
      // A broken store cannot record its own failure. Preserve visible in-memory
      // health and report both errors; never let it leave the observer attached.
      this.onError(persistenceError)
    }
    this.pruneCompleted()
    this.onError(error)
  }
  private snapshot(watch: Watch): MonitorSummary {
    const reactionHealth = this.reactionMailbox?.inspect(watch.owner, watch.id)
    return { ...(watch.source ? { source: watch.source } : {}), ...(watch.sourceStatus ? { sourceStatus: watch.sourceStatus } : {}), stopAction: stopAction(watch.state, reactionHealth), ...(watch.deliveryError ? { deliveryError: watch.deliveryError } : {}), trigger: watch.trigger, id: watch.id, terminalId: watch.terminalId, owner: watch.owner, match: watch.match,
      state: watch.state, ...(reactionHealth ? { reactionHealth } : {}), ...(watch.reaction ? { reaction: { ...watch.reaction } } : {}), ...(watch.error ? { error: watch.error } : {}), events: [...watch.events], expiresAt: watch.expiresAt, droppedEvents: watch.droppedEvents }
  }
  private output(watch: Watch): string {
    return watch.events.map(event => `${event.sequence}. ${new Date(event.at).toISOString()}\n${event.text}`).join('\n\n')
  }
}

function stopAction(state: MonitorSummary['state'], health?: ReactionHealth): Exclude<MonitorSummary['stopAction'], undefined> {
  if (state === 'watching') return 'stop-watch'
  if (state === 'detached') return null
  return health && ['waiting', 'queued', 'running', 'cancelling', 'awaiting-cleanup'].includes(health.state) ? 'cancel-reactions' : null
}
