// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// GatewayClient — the TS side of the Xerxes TUI ⇄ daemon seam. It connects to
// the per-project Unix domain socket published by `python -m xerxes.daemon`
// (spawning the daemon if none is reachable), speaks newline-delimited
// JSON-RPC 2.0, and demuxes responses (carry `id`) from streaming events
// (`method === "event"`). See `src/ui-tui/PROTOCOL.md` for the frozen contract.
//
// The transport is a Unix socket (Node `net`) rather than child stdio.

import { type ChildProcess, execFileSync, spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { EventEmitter } from 'node:events'
import { existsSync, readFileSync, realpathSync } from 'node:fs'
import { connect, type Socket } from 'node:net'
import { homedir } from 'node:os'
import { join, resolve } from 'node:path'

import {
  adaptDaemonEvent,
  sessionInfoFromInit,
  transcriptFromStoredMessages,
  usageFromStatus
} from './gatewayAdapter.js'
import type { AnyEvent } from './gatewayTypes.js'
import type { SessionInfo, Usage } from './types.js'

const MAX_GATEWAY_LOG_LINES = 200
const MAX_LOG_LINE_BYTES = 4096
const STARTUP_TIMEOUT_MS = Math.max(
  5000,
  Number.parseInt(process.env.XERXES_TUI_STARTUP_TIMEOUT_MS ?? '15000', 10) || 15000
)
const REQUEST_TIMEOUT_MS = Math.max(
  30000,
  Number.parseInt(process.env.XERXES_TUI_RPC_TIMEOUT_MS ?? '120000', 10) || 120000
)
const CONNECT_RETRY_MS = 150

// ── Path resolution (mirrors core/paths.py + tui/engine.py) ─────────────

/** `$XERXES_HOME` or `~/.xerxes` — matches `core/paths.py::xerxes_home`. */
function xerxesHome(): string {
  const override = (process.env.XERXES_HOME ?? '').trim()
  return override ? resolve(override) : join(homedir(), '.xerxes')
}

/** Canonical project dir: nearest git root when available, otherwise cwd. */
export function resolveProjectDir(projectDir?: string): string {
  const raw = resolve(projectDir ?? process.cwd())
  try {
    const root = execFileSync('git', ['-C', raw, 'rev-parse', '--show-toplevel'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore']
    }).trim()
    if (root) {
      return realpathSync(root)
    }
  } catch {
    // Fall through to raw path canonicalization for non-git directories.
  }
  try {
    return realpathSync(raw)
  } catch {
    return raw
  }
}

/**
 * Per-project socket + pid paths, mirroring `tui/engine.py`. `XERXES_DAEMON_SOCKET`
 * overrides the socket (the daemon's default config path is then used for pid).
 */
export function daemonPaths(projectDir: string): { socketPath: string; pidPath: string } {
  const digest = createHash('sha256').update(projectDir, 'utf8').digest('hex').slice(0, 16)
  const base = join(xerxesHome(), 'daemon', 'projects')
  const override = (process.env.XERXES_DAEMON_SOCKET ?? '').trim()
  return {
    socketPath: override || join(base, `${digest}.sock`),
    pidPath: join(base, `${digest}.pid`)
  }
}

/** Interpreter resolution: XERXES_PYTHON → PYTHON → $VIRTUAL_ENV → .venv → venv → python3. */
function resolvePython(projectDir: string): string {
  const configured = (process.env.XERXES_PYTHON ?? process.env.PYTHON ?? '').trim()
  if (configured) {
    return configured
  }
  const venv = (process.env.VIRTUAL_ENV ?? '').trim()
  const candidates = [
    venv && join(venv, 'bin/python'),
    venv && join(venv, 'Scripts/python.exe'),
    join(projectDir, '.venv/bin/python'),
    join(projectDir, '.venv/bin/python3'),
    join(projectDir, 'venv/bin/python'),
    join(projectDir, 'venv/bin/python3')
  ].filter(Boolean) as string[]
  return candidates.find(p => existsSync(p)) ?? (process.platform === 'win32' ? 'python' : 'python3')
}

// ── Client ──────────────────────────────────────────────────────────────

interface Pending {
  resolve: (value: unknown) => void
  reject: (err: Error) => void
  timer: NodeJS.Timeout
}

type RpcObject = Record<string, any>

export interface GatewayClientOptions {
  projectDir?: string
  /** Connection-local session key; defaults to `tui:<uuid12>`. */
  sessionKey?: string
}

/**
 * Emits:
 *   - `event` (AnyEvent)        every decoded gateway/client event
 *   - any specific event `type` likewise (e.g. `text_part`, `init_done`)
 *   - `close`                   the socket closed
 */
export class GatewayClient extends EventEmitter {
  readonly sessionKey: string
  private readonly projectDir: string
  private socket: Socket | null = null
  private proc: ChildProcess | null = null
  private nextId = 1
  private readonly pending = new Map<number, Pending>()
  private buffer = ''
  private readonly stderrRing: string[] = []
  private spawnedDaemon = false
  private closed = false
  private activeSessionKey: string
  private readonly sessionKeys = new Map<string, string>()
  private lastApprovalRequestId = ''

  constructor(opts: GatewayClientOptions = {}) {
    super()
    this.setMaxListeners(100)
    this.projectDir = resolveProjectDir(opts.projectDir)
    this.sessionKey = opts.sessionKey ?? `tui:${randomKey()}`
    this.activeSessionKey = this.sessionKey
  }

  /** Connect, launching the daemon if none is reachable. Idempotent once connected. */
  async start(): Promise<void> {
    if (this.socket) {
      return
    }
    const { socketPath, pidPath } = daemonPaths(this.projectDir)

    if (await this.tryConnect(socketPath)) {
      this.emitClient('gateway.ready', { socketPath, spawned: false })
      return
    }

    // No daemon reachable — spawn one and poll until the socket appears.
    this.spawnDaemon(socketPath, pidPath)
    const deadline = Date.now() + STARTUP_TIMEOUT_MS
    while (Date.now() < deadline) {
      if (this.proc && this.proc.exitCode !== null) {
        throw new Error(`daemon exited (code ${this.proc.exitCode}) before becoming ready:\n${this.stderrSnapshot()}`)
      }
      if (await this.tryConnect(socketPath)) {
        this.emitClient('gateway.ready', { socketPath, spawned: true })
        return
      }
      await delay(CONNECT_RETRY_MS)
    }
    throw new Error(`daemon did not become ready within ${STARTUP_TIMEOUT_MS}ms:\n${this.stderrSnapshot()}`)
  }

  private tryConnect(socketPath: string): Promise<boolean> {
    return new Promise<boolean>(res => {
      const sock = connect({ path: socketPath })
      const onError = () => {
        sock.destroy()
        res(false)
      }
      sock.once('error', onError)
      sock.once('connect', () => {
        sock.removeListener('error', onError)
        this.attachSocket(sock)
        res(true)
      })
    })
  }

  private attachSocket(sock: Socket): void {
    this.socket = sock
    sock.setEncoding('utf8')
    sock.on('data', (chunk: string) => this.onData(chunk))
    sock.on('error', err => this.emitClient('gateway.error', { message: String((err as Error).message ?? err) }))
    sock.on('close', () => {
      this.socket = null
      if (!this.closed) {
        this.emitClient('gateway.closed', {})
        this.emit('close')
      }
      for (const [, p] of this.pending) {
        clearTimeout(p.timer)
        p.reject(new Error('gateway socket closed'))
      }
      this.pending.clear()
    })
  }

  private spawnDaemon(socketPath: string, pidPath: string): void {
    const python = resolvePython(this.projectDir)
    const argv = [
      '-m',
      'xerxes.daemon',
      '--project-dir',
      this.projectDir,
      '--socket',
      socketPath,
      '--pid-file',
      pidPath
    ]
    this.proc = spawn(python, argv, {
      stdio: ['ignore', 'ignore', 'pipe'],
      detached: true,
      env: process.env
    })
    this.spawnedDaemon = true
    this.proc.stderr?.setEncoding('utf8')
    this.proc.stderr?.on('data', (chunk: string) => {
      for (const line of chunk.split('\n')) {
        if (!line) {
          continue
        }
        this.pushStderr(line)
        this.emitClient('gateway.stderr', { line: truncate(line) })
      }
    })
    this.proc.unref()
  }

  // ── Line framing ────────────────────────────────────────────────────

  private onData(chunk: string): void {
    this.buffer += chunk
    let nl = this.buffer.indexOf('\n')
    while (nl !== -1) {
      const line = this.buffer.slice(0, nl)
      this.buffer = this.buffer.slice(nl + 1)
      if (line.trim()) {
        this.onLine(line)
      }
      nl = this.buffer.indexOf('\n')
    }
  }

  private onLine(line: string): void {
    let frame: { id?: unknown; method?: unknown; result?: unknown; error?: unknown; params?: unknown }
    try {
      frame = JSON.parse(line)
    } catch {
      this.emitClient('gateway.protocol_error', { line: truncate(line) })
      return
    }

    // Response/error: carries an `id`.
    if (frame.id !== undefined && frame.id !== null) {
      const pending = this.pending.get(frame.id as number)
      if (!pending) {
        return
      }
      this.pending.delete(frame.id as number)
      clearTimeout(pending.timer)
      if (frame.error) {
        const e = frame.error as { code: number; message: string }
        pending.reject(new Error(`rpc ${e.code}: ${e.message}`))
      } else {
        pending.resolve(frame.result)
      }
      return
    }

    // Event notification: `{ method: "event", params: { type, payload } }`.
    if (frame.method === 'event' && frame.params && typeof frame.params === 'object') {
      const params = frame.params as { type?: string; payload?: Record<string, unknown> }
      const type = String(params.type ?? '')
      const payload = (params.payload ?? {}) as Record<string, unknown>
      if (type === 'approval_request') {
        this.lastApprovalRequestId = String(payload.id ?? '')
      }
      for (const evt of adaptDaemonEvent(type, payload)) {
        this.emitEvent(evt)
      }
      return
    }

    this.emitClient('gateway.protocol_error', { line: truncate(line) })
  }

  // ── Requests ────────────────────────────────────────────────────────

  /** Send a JSON-RPC request and await its result. */
  request<T = unknown>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    return this.requestCompat<T>(method, params)
  }

  private async requestCompat<T>(method: string, params: Record<string, unknown>): Promise<T> {
    switch (method) {
      case 'setup.status':
        return { provider_configured: true } as T

      case 'commands.catalog':
        return this.commandsCatalog() as Promise<T>

      case 'config.get':
        return this.configGet(params) as T

      case 'config.set':
        return this.configSet(params) as Promise<T>

      case 'session.create':
        return this.sessionCreate(params) as Promise<T>

      case 'session.resume':
      case 'session.activate':
        return this.sessionResume(params) as Promise<T>

      case 'session.active_list':
      case 'session.list':
        return this.sessionList(params, method === 'session.active_list') as Promise<T>

      case 'session.close':
        return { closed: true, ok: true } as T

      case 'session.delete':
        return { deleted: String(params.session_id ?? '') } as T

      case 'session.most_recent':
        return {} as T

      case 'session.title':
        return { title: String(params.title ?? '') } as T

      case 'session.status':
        return this.sessionStatus(params) as Promise<T>

      case 'session.compress':
        await this.rawRequest('slash', { command: '/compact' }).catch(() => null)
        return { messages: [], summary: { noop: false }, usage: {} } as T

      case 'session.usage':
        return this.sessionUsage(params) as Promise<T>

      case 'session.save':
        return { file: '' } as T

      case 'session.undo':
        return { removed: 0 } as T

      case 'session.interrupt':
        return this.rawRequest<T>('cancel', { session_key: this.keyFor(params.session_id) })

      case 'session.steer':
        return this.rawRequest<T>('steer', {
          content: String(params.text ?? ''),
          session_key: this.keyFor(params.session_id)
        })

      case 'prompt.submit':
        return this.rawRequest<T>('turn.submit', {
          session_key: this.keyFor(params.session_id),
          text: String(params.text ?? '')
        })

      case 'slash.exec':
        return this.slashExec(params) as Promise<T>

      case 'command.dispatch':
        return this.slashExec({
          command: `/${String(params.name ?? '')} ${String(params.arg ?? '')}`.trim(),
          session_id: params.session_id
        }) as Promise<T>

      case 'shell.exec':
        return this.shellExec(params) as Promise<T>

      case 'approval.respond':
        return this.approvalRespond(params) as Promise<T>

      case 'clarify.respond':
        return this.clarifyRespond(params) as Promise<T>

      case 'complete.path':
      case 'complete.slash':
        return this.complete(method, params) as Promise<T>

      case 'terminal.resize':
      case 'image.attach':
      case 'clipboard.paste':
      case 'paste.collapse':
      case 'input.detect_drop':
      case 'voice.toggle':
      case 'voice.record':
      case 'plugins.manage':
      case 'skills.reload':
      case 'skills.manage':
      case 'delegation.status':
      case 'delegation.pause':
      case 'subagent.interrupt':
      case 'spawn_tree.save':
      case 'spawn_tree.list':
      case 'spawn_tree.load':
      case 'process.stop':
      case 'reload.mcp':
      case 'reload.env':
      case 'browser.manage':
      case 'rollback.list':
      case 'rollback.diff':
      case 'rollback.restore':
      case 'tools.configure':
      case 'model.disconnect':
      case 'model.save_key':
        return this.stub(method, params) as T

      case 'model.options':
        return this.modelOptions() as Promise<T>

      default:
        return this.rawRequest<T>(method, params)
    }
  }

  private rawRequest<T = unknown>(method: string, params: Record<string, unknown> = {}): Promise<T> {
    const sock = this.socket
    if (!sock) {
      return Promise.reject(new Error('gateway not connected'))
    }
    const id = this.nextId++
    const frame = JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n'
    return new Promise<T>((res, rej) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        rej(new Error(`rpc timeout: ${method} (${REQUEST_TIMEOUT_MS}ms)`))
      }, REQUEST_TIMEOUT_MS)
      this.pending.set(id, { resolve: res as (v: unknown) => void, reject: rej, timer })
      sock.write(frame, err => {
        if (err) {
          clearTimeout(timer)
          this.pending.delete(id)
          rej(err)
        }
      })
    })
  }

  /** Fire-and-forget notification (no id, no response expected). */
  notify(method: string, params: Record<string, unknown> = {}): void {
    this.socket?.write(JSON.stringify({ jsonrpc: '2.0', method, params }) + '\n')
  }

  close(): void {
    this.closed = true
    this.socket?.end()
    this.socket = null
  }

  kill(_reason = ''): void {
    this.close()
    if (this.proc && this.proc.exitCode === null) {
      this.proc.kill('SIGTERM')
    }
    this.emit('exit')
  }

  drain(): void {}

  getLogTail(lines = 20): string {
    return this.stderrRing.slice(-Math.max(1, lines)).join('\n')
  }

  /** Last lines captured from a daemon we spawned (empty if we attached). */
  stderrSnapshot(): string {
    return this.stderrRing.join('\n')
  }

  get didSpawnDaemon(): boolean {
    return this.spawnedDaemon
  }

  /** True once the socket is connected (before any events have been emitted). */
  get connected(): boolean {
    return this.socket !== null
  }

  // ── helpers ─────────────────────────────────────────────────────────

  private async commandsCatalog(): Promise<RpcObject> {
    const raw = await this.rawRequest<RpcObject>('commands.catalog', {}).catch(() => null)
    if (raw && typeof raw === 'object' && Array.isArray(raw.pairs)) {
      return raw
    }
    return this.fallbackCommandsCatalog()
  }

  private fallbackCommandsCatalog(): RpcObject {
    const pairs: [string, string][] = [
      ['/help', 'show help'],
      ['/new', 'start a new session'],
      ['/resume', 'resume a session'],
      ['/model', 'switch model'],
      ['/provider', 'manage providers'],
      ['/skills', 'list skills'],
      ['/compact', 'compact context'],
      ['/steer', 'steer the active turn'],
      ['/quit', 'quit']
    ]
    return {
      canon: Object.fromEntries(pairs.map(([name]) => [name, name])),
      categories: [{ name: 'core', pairs }],
      pairs,
      skill_count: 0,
      sub: {}
    }
  }

  private configGet(params: Record<string, unknown>): RpcObject {
    const key = String(params.key ?? '')
    if (key === 'full') {
      return {
        config: {
          display: {
            mouse_tracking: 'all',
            show_reasoning: true,
            tui_agents_nudge: true,
            tui_auto_resume_recent: false
          },
          paste_collapse_char_threshold: 12000,
          paste_collapse_threshold: 20,
          voice: {}
        }
      }
    }
    if (key === 'mtime') {
      return { mtime: Date.now() / 1000 }
    }
    return { value: '' }
  }

  private async configSet(params: Record<string, unknown>): Promise<RpcObject> {
    const key = String(params.key ?? '')
    const value = String(params.value ?? '')
    if (key === 'model' && value) {
      await this.rawRequest('runtime.reload', { model: value }).catch(() => null)
      return { value }
    }
    if (key === 'reasoning') {
      const raw = (await this.rawRequest('runtime.reload', { reasoning_effort: value })) as RpcObject
      const effort = String(raw?.reasoning_effort ?? value)
      return {
        info: { reasoning_effort: effort },
        value: effort
      }
    }
    if (key === 'mode') {
      await this.rawRequest('set_mode', { mode: value }).catch(() => null)
    }
    return { value }
  }

  private async sessionCreate(_params: Record<string, unknown>): Promise<RpcObject> {
    this.activeSessionKey = `tui:${randomKey()}`
    const capture = this.captureInitializeInfo()
    const raw = (await this.rawRequest('initialize', {
      project_dir: this.projectDir,
      session_key: this.activeSessionKey
    })) as RpcObject
    const captured = capture()
    const session = (raw.session ?? {}) as RpcObject
    const sessionId = String(session.id ?? randomKey())
    this.sessionKeys.set(sessionId, this.activeSessionKey)
    return {
      info: this.sessionInfoFromInitialize(raw, session, captured),
      session_id: sessionId
    }
  }

  private async sessionResume(params: Record<string, unknown>): Promise<RpcObject> {
    const id = String(params.session_id ?? '')
    this.activeSessionKey = id || this.sessionKey
    const capture = this.captureInitializeInfo()
    const raw = (await this.rawRequest('initialize', {
      project_dir: this.projectDir,
      resume_session_id: id,
      session_key: this.activeSessionKey
    })) as RpcObject
    const captured = capture()
    const session = (raw.session ?? {}) as RpcObject
    const sessionId = String(session.id ?? id)
    this.sessionKeys.set(sessionId, this.activeSessionKey)
    return {
      info: this.sessionInfoFromInitialize(raw, session, captured),
      messages: transcriptFromStoredMessages(session.messages),
      resumed: sessionId,
      running: Boolean(session.active_turn_id),
      session_id: sessionId,
      status: session.active_turn_id ? 'working' : 'idle'
    }
  }

  private async sessionList(_params: Record<string, unknown>, _active: boolean): Promise<RpcObject> {
    const raw = (await this.rawRequest('session.list', {})) as RpcObject
    const rows = Array.isArray(raw.sessions) ? raw.sessions : []
    const sessions = rows.map((row: RpcObject) => {
      const id = String(row.id ?? row.session_id ?? row.key ?? '')
      if (id && row.key) {
        this.sessionKeys.set(id, String(row.key))
      }
      return {
        current: this.keyFor(id) === this.activeSessionKey,
        id,
        last_active: Date.now() / 1000,
        message_count: Number(row.messages ?? 0),
        model: String(row.model ?? ''),
        preview: String(row.title ?? row.key ?? id),
        started_at: Date.now() / 1000,
        status: row.active_turn_id ? 'working' : 'idle',
        title: String(row.title ?? row.key ?? id)
      }
    })
    return { sessions }
  }

  private async sessionStatus(params: Record<string, unknown>): Promise<RpcObject> {
    const raw = (await this.rawRequest('session.status', { session_key: this.keyFor(params.session_id) })) as RpcObject
    return { output: JSON.stringify(raw.session ?? raw, null, 2) }
  }

  private async sessionUsage(params: Record<string, unknown>): Promise<RpcObject> {
    const raw = (await this.rawRequest('session.status', { session_key: this.keyFor(params.session_id) })) as RpcObject
    return usageFromStatus((raw.session ?? raw) as Record<string, unknown>)
  }

  private async slashExec(params: Record<string, unknown>): Promise<RpcObject> {
    const raw = String(params.command ?? '').trim()
    const command = raw.startsWith('/') || raw.startsWith('!') ? raw : `/${raw}`
    await this.rawRequest('slash', { command })
    return { output: '' }
  }

  private async shellExec(params: Record<string, unknown>): Promise<RpcObject> {
    await this.rawRequest('slash', { command: `!${String(params.command ?? '')}` })
    return { code: 0, stderr: '', stdout: '' }
  }

  private approvalRespond(params: Record<string, unknown>): Promise<unknown> {
    const choice = String(params.choice ?? '')
    const response =
      choice === 'always' || choice === 'approve_for_session'
        ? 'approve_for_session'
        : choice === 'deny' || choice === 'reject'
          ? 'reject'
          : 'approve'
    return this.rawRequest('permission_response', { request_id: this.lastApprovalRequestId, response })
  }

  private clarifyRespond(params: Record<string, unknown>): Promise<unknown> {
    const requestId = String(params.request_id ?? '')
    const [daemonRequestId, questionId = 'q'] = requestId.split(':', 2)
    return this.rawRequest('question_response', {
      answers: { [questionId]: String(params.answer ?? '') },
      request_id: daemonRequestId
    })
  }

  private async complete(method: string, params: Record<string, unknown>): Promise<RpcObject> {
    const text = method === 'complete.path' ? String(params.word ?? '') : String(params.text ?? '')
    const raw = (await this.rawRequest('complete', { text })) as RpcObject
    const items = Array.isArray(raw.completions)
      ? raw.completions.map((item: RpcObject) => ({
          display: String(item.label ?? item.value ?? ''),
          meta: item.meta ? String(item.meta) : undefined,
          text: String(item.value ?? '')
        }))
      : []
    return { items, replace_from: method === 'complete.slash' ? 1 : undefined }
  }

  private async modelOptions(): Promise<RpcObject> {
    const raw = (await this.rawRequest('provider_list', {}).catch(() => ({ profiles: [] }))) as RpcObject
    const profiles = Array.isArray(raw.profiles) ? raw.profiles : []
    return {
      providers: profiles.map((profile: RpcObject) => ({
        authenticated: true,
        is_current: Boolean(profile.active),
        models: [String(profile.model ?? '')].filter(Boolean),
        name: String(profile.name ?? profile.provider ?? 'provider'),
        slug: String(profile.name ?? profile.provider ?? 'provider'),
        total_models: 1
      }))
    }
  }

  private stub(_method: string, params: Record<string, unknown>): RpcObject {
    return {
      attached: false,
      available: false,
      changed: [],
      disconnected: false,
      dropped: false,
      entries: [],
      found: false,
      installed: false,
      matched: false,
      messages: [],
      ok: true,
      paused: false,
      skills: {},
      value: String(params.value ?? '')
    }
  }

  private keyFor(id: unknown): string {
    const sid = String(id ?? '').trim()
    return sid ? (this.sessionKeys.get(sid) ?? sid) : this.activeSessionKey
  }

  private captureInitializeInfo(): () => { info: null | SessionInfo; usage: null | Usage } {
    let info: null | SessionInfo = null
    let usage: null | Usage = null
    const onInfo = (ev: AnyEvent) => {
      const incoming = ev.payload as SessionInfo | undefined
      if (!incoming) {
        return
      }
      if (info) {
        info = mergeSessionInfo(info, incoming)
      } else if (hasRenderableSessionInfo(incoming)) {
        info = incoming
      }
    }
    const onStatus = (ev: AnyEvent) => {
      const incoming = (ev.payload as { usage?: Usage } | undefined)?.usage
      if (incoming) {
        usage = mergeUsage(usage ?? {}, incoming)
      }
    }
    this.on('session.info', onInfo)
    this.on('status.update', onStatus)
    return () => {
      this.off('session.info', onInfo)
      this.off('status.update', onStatus)
      return { info, usage }
    }
  }

  private sessionInfoFromInitialize(
    raw: RpcObject,
    session: RpcObject,
    captured: { info: null | SessionInfo; usage: null | Usage }
  ): SessionInfo {
    const rawInfo = sessionInfoFromInit({
      ...raw,
      cwd: raw.cwd ?? session.cwd,
      mode: raw.mode ?? session.mode,
      model: raw.model ?? session.model,
      session_id: raw.session_id ?? session.id
    })
    const withEvent = mergeSessionInfo(rawInfo, captured.info ?? undefined)
    const withUsage = captured.usage
      ? { ...withEvent, usage: mergeUsage(withEvent.usage ?? {}, captured.usage) }
      : withEvent
    const cwd = withUsage.cwd || this.projectDir
    return {
      ...withUsage,
      cwd,
      head_hash: withUsage.head_hash || localGitHead(cwd),
      version: withUsage.version || localProjectVersion(cwd)
    }
  }

  private emitClient(type: string, payload: Record<string, unknown>): void {
    const evt = { type, payload } as AnyEvent
    this.emitEvent(evt)
  }

  private emitEvent(evt: AnyEvent): void {
    this.emit('event', evt)
    if (evt.type) {
      this.emit(evt.type, evt)
    }
  }

  private pushStderr(line: string): void {
    this.stderrRing.push(line)
    if (this.stderrRing.length > MAX_GATEWAY_LOG_LINES) {
      this.stderrRing.shift()
    }
  }
}

function truncate(line: string): string {
  return line.length > MAX_LOG_LINE_BYTES
    ? `${line.slice(0, MAX_LOG_LINE_BYTES)}… [truncated ${line.length} bytes]`
    : line
}

function delay(ms: number): Promise<void> {
  return new Promise(res => setTimeout(res, ms))
}

function randomKey(): string {
  // Avoid Math.random for nothing security-sensitive; crypto is already imported.
  return createHash('sha256').update(`${process.pid}:${process.hrtime.bigint()}`).digest('hex').slice(0, 12)
}

function hasStringGroups(record?: Record<string, string[]>): boolean {
  return Boolean(record && Object.values(record).some(values => Array.isArray(values) && values.length > 0))
}

function hasStringRecord(record?: Record<string, string>): boolean {
  return Boolean(record && Object.keys(record).length > 0)
}

function hasRenderableSessionInfo(info: SessionInfo): boolean {
  return Boolean(
    info.cwd ||
    info.version ||
    info.head_hash ||
    info.model ||
    hasStringGroups(info.skills) ||
    hasStringGroups(info.tools)
  )
}

function mergeUsage(base: Partial<Usage>, incoming?: null | Partial<Usage>): Usage {
  const merged = { ...base, ...(incoming ?? {}) }
  return {
    calls: merged.calls ?? 0,
    compressions: merged.compressions,
    context_max: incoming?.context_max || base.context_max,
    context_percent: merged.context_percent,
    context_used: incoming?.context_used ?? base.context_used,
    cost_status: merged.cost_status,
    cost_usd: merged.cost_usd,
    dev_credits_spent_micros: merged.dev_credits_spent_micros,
    input: merged.input ?? 0,
    output: merged.output ?? 0,
    reasoning: merged.reasoning,
    total: merged.total ?? 0
  }
}

function mergeSessionInfo(base: SessionInfo, incoming?: null | Partial<SessionInfo>): SessionInfo {
  if (!incoming) {
    return base
  }

  return {
    ...base,
    ...incoming,
    cwd: incoming.cwd || base.cwd,
    head_hash: incoming.head_hash || base.head_hash,
    model: incoming.model || base.model,
    mode: incoming.mode || base.mode,
    profile_name: incoming.profile_name || base.profile_name,
    reasoning_effort: incoming.reasoning_effort || base.reasoning_effort,
    skillDescriptions: hasStringRecord(incoming.skillDescriptions)
      ? incoming.skillDescriptions
      : base.skillDescriptions,
    skills: incoming.skills && hasStringGroups(incoming.skills) ? incoming.skills : base.skills,
    tools: incoming.tools && hasStringGroups(incoming.tools) ? incoming.tools : base.tools,
    usage: mergeUsage(base.usage ?? {}, incoming.usage),
    version: incoming.version || base.version
  }
}

function localGitHead(projectDir: string): string {
  try {
    return execFileSync('git', ['-C', projectDir, 'rev-parse', '--short=12', 'HEAD'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore']
    }).trim()
  } catch {
    return ''
  }
}

function localProjectVersion(projectDir: string): string {
  let inProject = false

  try {
    for (const line of readFileSync(join(projectDir, 'pyproject.toml'), 'utf8').split(/\r?\n/)) {
      const section = /^\s*\[([^\]]+)]\s*$/.exec(line)
      if (section) {
        inProject = section[1] === 'project'
        continue
      }

      if (!inProject) {
        continue
      }

      const version = /^\s*version\s*=\s*["']([^"']+)["']\s*$/.exec(line)
      if (version?.[1]) {
        return version[1]
      }
    }
  } catch {
    return ''
  }

  return ''
}
