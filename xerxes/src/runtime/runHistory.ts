// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { terminalOutputPage, type TerminalOutputCursor, type TerminalOutputPage } from './terminalOutput.js'
import { Database } from 'bun:sqlite'
import { chmodSync, mkdirSync } from 'node:fs'
import { dirname } from 'node:path'
import { processCommand, processIsAlive, processStartIdentity } from '../core/processLiveness.js'
import type { ModelCallUsage } from '../llms/callBudget.js'

export type RunKind = 'schedule' | 'terminal' | 'agent' | 'monitor'
export type RunState = 'running' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted'
export interface RunRecord {
  readonly tokenUsage?: ModelCallUsage | null
  readonly id: string
  readonly ownerSessionId: string
  readonly workspace: string
  readonly kind: RunKind
  readonly sourceId: string
  readonly title: string
  readonly state: RunState
  readonly startedAt: number
  readonly endedAt: number | null
  readonly output: string
  readonly outputTruncated: boolean
  readonly error: string | null
  readonly revision: number
  readonly unread: boolean
  readonly terminalKind: 'background' | 'foreground' | 'pty' | null
  readonly exitCode: number | null
}
export interface RunEvent {
  readonly sequence: number
  readonly text: string
  readonly at: number
}
export type RunOutcome = Pick<RunRecord, 'id' | 'state' | 'startedAt' | 'endedAt'>
export interface RunEventPage {
  readonly events: readonly RunEvent[]
  readonly nextCursor: number
  readonly hasMore: boolean
}
export interface MonitorConfiguration {
  readonly trigger: 'output' | 'completion' | 'change'
  readonly match: string
  readonly expiresAt: number
  readonly source?: MonitorSource
}
export type MonitorSource =
  | { readonly kind: 'terminal'; readonly terminalId: string }
  | { readonly kind: 'file'; readonly path: string; readonly workspace: string }
  | { readonly kind: 'websocket'; readonly url: string }
  | { readonly kind: 'webhook'; readonly name: string }
export interface RunStart {
  readonly ownerSessionId: string
  readonly workspace: string
  readonly kind: RunKind
  readonly sourceId: string
  readonly title: string
  readonly terminalKind?: 'background' | 'foreground' | 'pty'
}
interface Row {
  id: string; owner: string; workspace: string; kind: RunKind; source: string; title: string;
  state: RunState; started: number; ended: number | null; output: string; truncated: number;
  error: string | null; revision: number; acknowledged: number; pid: number
  terminal_kind: RunRecord['terminalKind']; exit_code: number | null
  owner_command: string | null
  owner_start: string | null
  token_usage: string | null
}
const OUTPUT_LIMIT = 64_000
const TERMINAL_STATES: readonly RunState[] = ['succeeded', 'failed', 'cancelled', 'interrupted']

function validateMonitorConfiguration(configuration: MonitorConfiguration): void {
  if (!['completion', 'output', 'change'].includes(configuration.trigger)
    || typeof configuration.match !== 'string' || !configuration.match.trim() || configuration.match.length > 1024
    || !Number.isSafeInteger(configuration.expiresAt) || configuration.expiresAt <= 0) throw new Error('Invalid monitor configuration')
  if (configuration.source !== undefined) {
    if (configuration.source.kind === 'terminal') {
      if (!validMonitorSourceText(configuration.source.terminalId) || configuration.trigger === 'change') throw new Error('Invalid terminal monitor source')
    } else if (configuration.source.kind === 'file') {
      if (!validMonitorSourceText(configuration.source.path) || !validMonitorSourceText(configuration.source.workspace) || configuration.trigger !== 'change') throw new Error('Invalid file monitor source')
    } else if (configuration.source.kind === 'websocket') {
      if (!validWebsocketUrl(configuration.source.url) || configuration.trigger !== 'output') throw new Error('Invalid websocket monitor source')
    } else if (configuration.source.kind === 'webhook') {
      if (!validWebhookName(configuration.source.name) || configuration.trigger !== 'output') throw new Error('Invalid webhook monitor source')
    } else throw new Error('Invalid monitor source')
  } else if (configuration.trigger === 'change') throw new Error('File change monitors require a source')
}

function validMonitorSourceText(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0 && value.length <= 8192 && !/[\0\r\n]/.test(value)
}

function parseMonitorSource(value: unknown): MonitorSource {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) throw new Error('invalid source')
  const source = value as Record<string, unknown>
  if (source.kind === 'terminal' && validMonitorSourceText(source.terminalId)) return { kind: 'terminal', terminalId: source.terminalId }
  if (source.kind === 'file' && validMonitorSourceText(source.path) && validMonitorSourceText(source.workspace)) return { kind: 'file', path: source.path, workspace: source.workspace }
  if (source.kind === 'websocket' && validWebsocketUrl(source.url)) return { kind: 'websocket', url: source.url }
  if (source.kind === 'webhook' && validWebhookName(source.name)) return { kind: 'webhook', name: source.name }
  throw new Error('invalid source')
}

function validWebhookName(value: unknown): value is string {
  return typeof value === 'string' && /^[a-zA-Z0-9_-]{1,64}$/.test(value)
}

function validWebsocketUrl(value: unknown): value is string {
  if (!validMonitorSourceText(value) || value.length > 4096) return false
  try {
    const url = new URL(value)
    return (url.protocol === 'ws:' || url.protocol === 'wss:') && !url.username && !url.password && !url.search && !url.hash
  } catch { return false }
}

/** Durable read model only. Execution and cancellation stay with each run's owner. */
export class RunHistory {
  private readonly db: Database
  private readonly now: () => number
  private readonly pid: number
  private readonly ownerStart: string
  private readonly ownerCommand: string
  private readonly listeners = new Set<(run: RunRecord) => void>()

  constructor(path: string, options: { now?: () => number; pid?: number; isAlive?: (pid: number) => boolean; commandOf?: (pid: number) => string; startOf?: (pid: number) => string } = {}) {
    this.now = options.now ?? Date.now
    this.pid = options.pid ?? process.pid
    const commandOf = options.commandOf ?? processCommand
    const commands = new Map<number, string>()
    const command = (pid: number): string => {
      if (!commands.has(pid)) commands.set(pid, commandOf(pid).trim())
      return commands.get(pid)!
    }
    const startOf = options.startOf ?? processStartIdentity
    const starts = new Map<number, string>()
    const start = (pid: number): string => {
      if (!starts.has(pid)) starts.set(pid, startOf(pid).trim())
      return starts.get(pid)!
    }
    this.ownerStart = start(this.pid)
    this.ownerCommand = command(this.pid)
    if (path !== ':memory:') mkdirSync(dirname(path), { recursive: true, mode: 0o700 })
    this.db = new Database(path, { create: true, strict: true })
    if (path !== ':memory:') chmodSync(path, 0o600)
    this.db.exec(`PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;
      CREATE TABLE IF NOT EXISTS run_history (
        id TEXT PRIMARY KEY, owner TEXT NOT NULL, workspace TEXT NOT NULL,
        kind TEXT NOT NULL, source TEXT NOT NULL, title TEXT NOT NULL,
        state TEXT NOT NULL, started INTEGER NOT NULL, ended INTEGER,
        output TEXT NOT NULL DEFAULT '', truncated INTEGER NOT NULL DEFAULT 0,
        error TEXT, revision INTEGER NOT NULL DEFAULT 1, acknowledged INTEGER NOT NULL DEFAULT 1,
        pid INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS run_history_owner ON run_history(owner, started DESC);
      CREATE INDEX IF NOT EXISTS run_history_source_outcome ON run_history(owner, kind, source, started DESC, id DESC);
      CREATE TABLE IF NOT EXISTS terminal_output (
        run_id TEXT PRIMARY KEY, tail TEXT NOT NULL, total INTEGER NOT NULL
      );
      CREATE TABLE IF NOT EXISTS monitor_configurations (
        run_id TEXT PRIMARY KEY, trigger TEXT NOT NULL, match TEXT NOT NULL, expires INTEGER NOT NULL, source_json TEXT
      );
      CREATE TABLE IF NOT EXISTS run_events (
        run_id TEXT NOT NULL, sequence INTEGER NOT NULL, text TEXT NOT NULL, at INTEGER NOT NULL,
        PRIMARY KEY(run_id, sequence)
      );`)
    this.db.transaction(() => {
      const columns = new Set(this.db.query<{ name: string }, []>('PRAGMA table_info(run_history)').all().map(column => column.name))
      if (!columns.has('owner_start')) this.db.exec('ALTER TABLE run_history ADD COLUMN owner_start TEXT')
      if (!columns.has('owner_command')) this.db.exec('ALTER TABLE run_history ADD COLUMN owner_command TEXT')
      if (!columns.has('terminal_kind')) this.db.exec('ALTER TABLE run_history ADD COLUMN terminal_kind TEXT')
      if (!columns.has('exit_code')) this.db.exec('ALTER TABLE run_history ADD COLUMN exit_code INTEGER')
      if (!columns.has('token_usage')) this.db.exec('ALTER TABLE run_history ADD COLUMN token_usage TEXT')
    }).immediate()
    this.db.transaction(() => {
      const columns = new Set(this.db.query<{ name: string }, []>('PRAGMA table_info(monitor_configurations)').all().map(column => column.name))
      if (!columns.has('source_json')) this.db.exec('ALTER TABLE monitor_configurations ADD COLUMN source_json TEXT')
    }).immediate()
    // Recover records, never process handles. A surviving daemon keeps ownership.
    const isAlive = options.isAlive ?? processIsAlive
    for (const row of this.db.query<Row, []>("SELECT * FROM run_history WHERE state = 'running'").all()) {
      if (!isAlive(row.pid)) this.finish(row.owner, row.id, 'interrupted', { error: 'Owning process exited before reporting completion.' })
      else if (row.owner_start && start(row.pid) && start(row.pid) !== row.owner_start) {
        this.finish(row.owner, row.id, 'interrupted', { error: 'Owning process identity changed; the recorded PID has a different start time.' })
      } else if (row.owner_command) {
        const observed = command(row.pid)
        if (observed && observed !== row.owner_command) this.finish(row.owner, row.id, 'interrupted', { error: 'Owning process identity changed; the recorded PID now belongs to a different command.' })
      }
    }
  }

  start(input: RunStart): RunRecord {
    for (const [key, value] of Object.entries(input)) {
      if (typeof value !== 'string' || !value.trim() || (key !== 'title' && value.length > 8192)) throw new Error(`Invalid run ${key}`)
    }
    if (!['schedule', 'terminal', 'agent', 'monitor'].includes(input.kind)) throw new Error('Invalid run kind')
    const id = crypto.randomUUID()
    if (input.terminalKind && !['background', 'foreground', 'pty'].includes(input.terminalKind)) throw new Error('Invalid terminal kind')
    this.db.query(`INSERT INTO run_history(id,owner,workspace,kind,source,title,state,started,pid,terminal_kind,owner_command,owner_start)
      VALUES(?,?,?,?,?,?,'running',?,?,?,?,?)`).run(id, input.ownerSessionId, input.workspace, input.kind, input.sourceId, input.title.slice(0, 8192), this.now(), this.pid, input.terminalKind ?? null, this.ownerCommand || null, this.ownerStart || null)
    return this.inspect(input.ownerSessionId, id)!
  }

  startMonitor(input: RunStart, configuration: MonitorConfiguration): RunRecord {
    validateMonitorConfiguration(configuration)
    if (input.kind !== 'monitor'
      || typeof configuration.match !== 'string' || !configuration.match.trim() || configuration.match.length > 1024
      || !Number.isSafeInteger(configuration.expiresAt) || configuration.expiresAt <= 0) throw new Error('Invalid monitor configuration')
    return this.db.transaction(() => {
      const run = this.start(input)
      this.db.query('INSERT INTO monitor_configurations(run_id,trigger,match,expires,source_json) VALUES(?,?,?,?,?)')
        .run(run.id, configuration.trigger, configuration.match, configuration.expiresAt, configuration.source === undefined ? null : JSON.stringify(configuration.source))
      return run
    }).immediate()
  }

  monitorConfiguration(owner: string, id: string): MonitorConfiguration | undefined {
    const row = this.db.query<{ trigger: string; match: string; expires: number; source_json: string | null }, [string, string]>(
      `SELECT m.trigger,m.match,m.expires,m.source_json FROM monitor_configurations m JOIN run_history r ON r.id=m.run_id WHERE r.owner=? AND r.id=? AND r.kind='monitor'`)
      .get(owner, id)
    if (!row) return undefined
    let source: MonitorSource | undefined
    if (row.source_json !== null) {
      try { source = parseMonitorSource(JSON.parse(row.source_json) as unknown) }
      catch { throw new Error('Invalid stored monitor source') }
    }
    const configuration = { trigger: row.trigger as MonitorConfiguration['trigger'], match: row.match, expiresAt: row.expires, ...(source === undefined ? {} : { source }) }
    try { validateMonitorConfiguration(configuration) } catch { throw new Error('Invalid stored monitor configuration') }
    return configuration
  }

  monitorRuns(owner: string): RunRecord[] {
    return this.db.query<Row, [string]>(
      `SELECT r.* FROM run_history r JOIN monitor_configurations m ON m.run_id=r.id WHERE r.owner=? AND r.kind='monitor' ORDER BY r.started DESC,r.id DESC LIMIT 100`)
      .all(owner).map(record)
  }

  checkpointOutput(owner: string, id: string, output: string, truncated = false): void {
    const result = this.db.query(`UPDATE run_history SET output=?,truncated=? WHERE owner=? AND id=? AND state='running'`)
      .run(output.slice(-OUTPUT_LIMIT), Number(truncated || output.length > OUTPUT_LIMIT), owner, id)
    if (!result.changes) throw new Error('Unknown or completed run')
  }

  checkpointTerminalOutput(owner: string, id: string, tail: string, total: number): void {
    if (!Number.isSafeInteger(total) || total < tail.length) throw new Error('Invalid terminal output length')
    this.db.transaction(() => {
      const run = this.inspect(owner, id)
      if (!run || run.kind !== 'terminal' || run.state !== 'running') throw new Error('Unknown or completed terminal')
      const previous = this.db.query<{ total: number }, [string]>('SELECT total FROM terminal_output WHERE run_id=?').get(id)
      if (previous && total < previous.total) throw new Error('Terminal output cursor moved backwards')
      const retained = tail.slice(-OUTPUT_LIMIT)
      this.checkpointOutput(owner, id, retained, total > retained.length)
      this.db.query('INSERT INTO terminal_output(run_id,tail,total) VALUES(?,?,?) ON CONFLICT(run_id) DO UPDATE SET tail=excluded.tail,total=excluded.total')
        .run(id, retained, total)
    }).immediate()
  }

  terminalOutput(owner: string, id: string, cursor?: TerminalOutputCursor, limit?: number): TerminalOutputPage {
    const run = this.inspect(owner, id)
    if (!run || run.kind !== 'terminal') throw new Error('Unknown terminal')
    const row = this.db.query<{ tail: string; total: number }, [string]>('SELECT tail,total FROM terminal_output WHERE run_id=?').get(id)
    if (!row) throw new Error('Incremental output unavailable for this legacy terminal; use terminal.inspect')
    return terminalOutputPage(id, row.tail, row.total, run.state === 'running', cursor, limit)
  }

  /** Persist admission before provider execution, and observed usage after settlement. */
  checkpointUsage(owner: string, id: string, usage: ModelCallUsage): void {
    const valid = validateUsage(usage)
    const result = this.db.query("UPDATE run_history SET token_usage=? WHERE owner=? AND id=? AND state='running'")
      .run(JSON.stringify(valid), owner, id)
    if (!result.changes) throw new Error('Unknown or completed run')
  }

  /** Commit evidence and the displayed tail together, before external delivery. */
  appendEvent(owner: string, id: string, event: RunEvent, output: string, truncated = false): boolean {
    if (!Number.isSafeInteger(event.sequence) || event.sequence < 1 || event.sequence > 1000 ||
        !Number.isSafeInteger(event.at) || event.at < 0 || typeof event.text !== 'string' || event.text.length > 8300) {
      throw new Error('Invalid run event')
    }
    return this.db.transaction(() => {
      const run = this.inspect(owner, id)
      if (!run) throw new Error('Unknown run')
      const existing = this.db.query<RunEvent, [string, number]>(
        'SELECT sequence,text,at FROM run_events WHERE run_id=? AND sequence=?').get(id, event.sequence)
      if (existing) {
        if (existing.text !== event.text || existing.at !== event.at) throw new Error('Conflicting run event sequence')
        return false
      }
      const last = this.db.query<{ sequence: number | null }, [string]>(
        'SELECT MAX(sequence) AS sequence FROM run_events WHERE run_id=?').get(id)?.sequence ?? 0
      if (event.sequence !== last + 1) throw new Error('Run event sequence gap')
      this.checkpointOutput(owner, id, output, truncated)
      this.db.query('INSERT INTO run_events(run_id,sequence,text,at) VALUES(?,?,?,?)')
        .run(id, event.sequence, event.text, event.at)
      // A live match is actionable before its watch expires. Acknowledging the
      // current result must not hide events that arrive after that revision.
      this.db.query('UPDATE run_history SET revision=revision+1 WHERE id=? AND owner=?').run(id, owner)
      return true
    }).immediate()
  }

  events(owner: string, id: string, after = 0, limit = 20): RunEventPage {
    if (!Number.isSafeInteger(after) || after < 0 || !Number.isSafeInteger(limit) || limit < 1 || limit > 50) {
      throw new Error('Invalid event cursor or page limit')
    }
    if (!this.inspect(owner, id)) throw new Error('Unknown run')
    const rows = this.db.query<RunEvent, [string, number, number]>(
      'SELECT sequence,text,at FROM run_events WHERE run_id=? AND sequence>? ORDER BY sequence LIMIT ?')
      .all(id, after, limit + 1)
    const events = rows.slice(0, limit)
    return { events, nextCursor: events.at(-1)?.sequence ?? after, hasMore: rows.length > limit }
  }

  eventCursor(owner: string, id: string): number {
    if (!this.inspect(owner, id)) throw new Error('Unknown run')
    return this.db.query<{ sequence: number | null }, [string]>('SELECT MAX(sequence) AS sequence FROM run_events WHERE run_id=?').get(id)?.sequence ?? 0
  }

  finish(owner: string, id: string, state: Exclude<RunState, 'running'>, detail: { tokenUsage?: ModelCallUsage; output?: string; error?: string; outputTruncated?: boolean; notify?: boolean; exitCode?: number | null } = {}): RunRecord {
    if (!TERMINAL_STATES.includes(state)) throw new Error('Invalid terminal run state')
    const previous = this.inspect(owner, id)
    if (!previous) throw new Error('Unknown run')
    const tokenUsage = detail.tokenUsage === undefined ? previous.tokenUsage : validateUsage(detail.tokenUsage)
    const output = detail.output ?? previous.output
    const truncated = detail.outputTruncated ?? (detail.output === undefined && previous.outputTruncated)
    if (detail.exitCode != null && !Number.isSafeInteger(detail.exitCode)) throw new Error('Invalid exit code')
    const changed = this.db.query(`UPDATE run_history SET state=?,ended=?,output=?,truncated=?,error=?,
      acknowledged=CASE WHEN ?=1 THEN revision+1 ELSE acknowledged END,revision=revision+1,exit_code=?,token_usage=?
      WHERE id=? AND owner=? AND state='running'`).run(state, this.now(), output.slice(-OUTPUT_LIMIT), Number(truncated || output.length > OUTPUT_LIMIT), detail.error?.slice(0, 8192) ?? null, Number(detail.notify === false), detail.exitCode ?? null, tokenUsage ? JSON.stringify(tokenUsage) : null, id, owner)
    const record = this.inspect(owner, id)
    if (!record) throw new Error('Unknown run')
    if (changed.changes === 0 && record.state !== state) throw new Error('Run already completed with a different outcome')
    if (changed.changes > 0 && record.unread) {
      for (const listener of this.listeners) {
        try { listener(record) }
        catch (error) { console.error('Run completion listener failed:', error) }
      }
    }
    return record
  }

  /**
   * Forget a running record whose work continues under another run, without
   * completing it: a foreground command adopted as a background job is still
   * running, so finishing its record in any state would report an outcome that
   * never happened.
   */
  discard(owner: string, id: string): void {
    this.db.transaction(() => {
      const changed = this.db.query("DELETE FROM run_history WHERE id=? AND owner=? AND state='running'").run(id, owner)
      if (!changed.changes) throw new Error('Unknown or completed run')
      this.db.query('DELETE FROM terminal_output WHERE run_id=?').run(id)
    }).immediate()
  }

  /** Small status projection: dashboard polling must not load archived output. */
  latestOutcome(owner: string, sourceId: string, kind: RunKind): RunOutcome | null {
    const row = this.db.query('SELECT id,state,started,ended FROM run_history WHERE owner=? AND source=? AND kind=? ORDER BY started DESC,id DESC LIMIT 1')
      .get(owner, sourceId, kind) as { id: string; state: RunState; started: number; ended: number | null } | null
    return row ? { id: row.id, state: row.state, startedAt: row.started, endedAt: row.ended } : null
  }

  list(owner: string, options: { unreadOnly?: boolean; limit?: number; sourceId?: string; kind?: RunKind; state?: RunState; before?: { startedAt: number; id: string } } = {}): RunRecord[] {
    return this.listScope('owner', owner, options)
  }

  listWorkspace(workspace: string, options: { unreadOnly?: boolean; limit?: number; sourceId?: string; kind?: RunKind; state?: RunState; before?: { startedAt: number; id: string } } = {}): RunRecord[] {
    return this.listScope('workspace', workspace, options)
  }

  private listScope(column: 'owner' | 'workspace', scope: string, options: { unreadOnly?: boolean; limit?: number; sourceId?: string; kind?: RunKind; state?: RunState; before?: { startedAt: number; id: string } }): RunRecord[] {
    const limit = options.limit ?? 100
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > 500) throw new Error('Run limit must be an integer from 1 to 500')
    if (options.state && !['running', 'succeeded', 'failed', 'cancelled', 'interrupted'].includes(options.state)) throw new Error('Unknown run state')
    const before = options.before
    if (before && (!Number.isSafeInteger(before.startedAt) || before.startedAt < 0 || !before.id || before.id.length > 8192)) throw new Error('Invalid run page cursor')
    return this.db.query<Row, [string, number, string, string, string, string, string, string, number, number, number, string, number]>(`SELECT * FROM run_history WHERE ${column}=?
      AND (?=0 OR revision>acknowledged) AND (?= '' OR source=?) AND (?= '' OR kind=?) AND (?= '' OR state=?) AND (?=0 OR started<? OR (started=? AND id<?)) ORDER BY started DESC,id DESC LIMIT ?`).all(scope, Number(options.unreadOnly === true), options.sourceId ?? '', options.sourceId ?? '', options.kind ?? '', options.kind ?? '', options.state ?? '', options.state ?? '', Number(Boolean(before)), before?.startedAt ?? 0, before?.startedAt ?? 0, before?.id ?? '', limit).map(record)
  }

  inspectWorkspace(workspace: string, id: string): RunRecord | undefined {
    const row = this.db.query<Row, [string, string]>('SELECT * FROM run_history WHERE workspace=? AND id=?').get(workspace, id)
    return row ? record(row) : undefined
  }

  inspect(owner: string, id: string): RunRecord | undefined {
    const row = this.db.query<Row, [string, string]>('SELECT * FROM run_history WHERE owner=? AND id=?').get(owner, id)
    return row ? record(row) : undefined
  }

  acknowledge(owner: string, id: string, revision: number): RunRecord {
    if (!Number.isSafeInteger(revision) || revision < 1) throw new Error('Invalid run revision')
    const changed = this.db.query('UPDATE run_history SET acknowledged=? WHERE owner=? AND id=? AND revision=?').run(revision, owner, id, revision)
    if (!changed.changes) throw new Error('Run missing or changed; refresh before acknowledging')
    return this.inspect(owner, id)!
  }

  subscribe(listener: (run: RunRecord) => void): () => void {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }

  close(): void { this.listeners.clear(); this.db.close() }
}
function record(row: Row): RunRecord {
  return { id: row.id, ownerSessionId: row.owner, workspace: row.workspace, kind: row.kind, sourceId: row.source,
    title: row.title, state: row.state, startedAt: row.started, endedAt: row.ended, output: row.output,
    outputTruncated: Boolean(row.truncated), error: row.error, revision: row.revision, unread: row.revision > row.acknowledged,
    terminalKind: row.terminal_kind, exitCode: row.exit_code, tokenUsage: row.token_usage == null ? null : validateUsage(JSON.parse(row.token_usage)) }
}

function validateUsage(value: unknown): ModelCallUsage {
  if (!value || typeof value !== 'object') throw new Error('Invalid run token usage')
  const usage = value as Record<string, unknown>
  const fields = ['input_tokens', 'output_tokens', 'measured_calls', 'settled_calls', 'pending_calls'] as const
  for (const key of fields) if (typeof usage[key] !== 'number' || !Number.isSafeInteger(usage[key]) || usage[key] < 0) throw new Error('Invalid run token usage')
  const parsed = usage as unknown as ModelCallUsage
  if (typeof parsed.complete !== 'boolean' || parsed.measured_calls > parsed.settled_calls || (parsed.complete && (parsed.pending_calls !== 0 || parsed.measured_calls !== parsed.settled_calls))) throw new Error('Invalid run token usage completeness')
  return { input_tokens: parsed.input_tokens, output_tokens: parsed.output_tokens, measured_calls: parsed.measured_calls, settled_calls: parsed.settled_calls, pending_calls: parsed.pending_calls, complete: parsed.complete }
}
