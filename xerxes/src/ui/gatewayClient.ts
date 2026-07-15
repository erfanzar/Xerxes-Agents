// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// GatewayClient — the TS side of the Xerxes TUI ⇄ daemon seam. It connects to
// the per-project Unix domain socket published by the Bun TypeScript daemon
// (spawning that daemon if none is reachable), speaks newline-delimited
// JSON-RPC 2.0, and demuxes responses (carry `id`) from streaming events
// (`method === "event"`). See `xerxes/src/ui/PROTOCOL.md` for the frozen contract.
//
// The transport is a Unix socket (Node `net`) rather than child stdio.

import { type ChildProcess, execFileSync, spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { EventEmitter } from 'node:events'
import { existsSync, readFileSync, realpathSync } from 'node:fs'
import { connect, type Socket } from 'node:net'
import { homedir } from 'node:os'
import { dirname, isAbsolute, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  adaptDaemonEvent,
  sessionInfoFromInit,
  transcriptFromStoredMessages,
  usageFromStatus
} from './gatewayAdapter.js'
import type { AnyEvent, GatewayTranscriptMessage } from './gatewayTypes.js'
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
const DAEMON_IDENTITY_TIMEOUT_MS = Math.min(5000, STARTUP_TIMEOUT_MS)
// A bundled daemon is normally ready in 85-105 ms. A 25 ms cadence reaches
// it within one short interval without a busy loop or the old 150 ms stall.
export const DAEMON_CONNECT_RETRY_MS = 25

// ── Path resolution (v35 daemon path contract) ───────────────────────────

/** `$XERXES_HOME` or `~/.xerxes`. */
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
 * Per-project socket + pid paths. `XERXES_DAEMON_SOCKET` overrides the socket
 * while retaining the deterministic per-project pid path.
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

export interface BunDaemonLaunch {
  readonly args: readonly string[]
  readonly binary: string
  readonly entryPath: string
}

export type DaemonBuildDecision = 'current' | 'reject' | 'restart'

export interface DaemonBuildDecisionInput {
  readonly activeTurns: boolean
  readonly actualBuildId: string
  readonly commandMatches: boolean
  readonly daemonPid: number | undefined
  readonly daemonProtocol: number | undefined
  readonly daemonRuntime: string
  readonly expectedBuildId: string
  readonly explicitSocket: boolean
  readonly pidFilePid: number | undefined
}

/** Decide whether one connected daemon can be reused without touching an unrelated process. */
export function daemonBuildDecision(input: DaemonBuildDecisionInput): DaemonBuildDecision {
  if (input.actualBuildId === input.expectedBuildId) {
    return input.daemonRuntime === 'bun-typescript' && input.daemonProtocol === 35 ? 'current' : 'reject'
  }
  if (
    input.explicitSocket ||
    input.daemonRuntime !== 'bun-typescript' ||
    input.daemonProtocol !== 35 ||
    input.daemonPid === undefined ||
    input.pidFilePid !== input.daemonPid ||
    !input.commandMatches ||
    input.activeTurns
  ) {
    return 'reject'
  }
  return 'restart'
}

/** Exact source-checkout daemon signature required before automatic restart. */
export function daemonCommandMatches(command: string, launch: BunDaemonLaunch): boolean {
  const [entryPath, daemon, projectFlag, projectDir, socketFlag, socketPath, pidFlag, pidPath] = launch.args
  if (
    daemon !== 'daemon' ||
    projectFlag !== '--project-dir' ||
    socketFlag !== '--socket' ||
    pidFlag !== '--pid-file' ||
    !entryPath ||
    !projectDir ||
    !socketPath ||
    !pidPath
  ) {
    return false
  }
  // Refuse ambiguous command displays. A checkout path containing whitespace
  // remains usable, but the user must restart that daemon explicitly.
  if ([entryPath, projectDir, socketPath, pidPath].some(value => /\s/.test(value))) {
    return false
  }
  return command.includes(
    `${entryPath} daemon --project-dir ${projectDir} --socket ${socketPath} --pid-file ${pidPath}`
  )
}

type Environment = Readonly<Record<string, string | undefined>>

export function bunDaemonEnvironment(expectedBuildId: string, environment: Environment = process.env): NodeJS.ProcessEnv {
  return {
    ...environment,
    ...(expectedBuildId ? { XERXES_DAEMON_BUILD_ID: expectedBuildId } : {})
  }
}

const NATIVE_UNSUPPORTED_RPC_GUIDANCE: Readonly<Record<string, string>> = Object.freeze({
  'delegation.pause': 'Native subagent delegation controls are not configured in this daemon.',
  'delegation.status': 'Native subagent delegation status is not configured in this daemon.',
  'model.disconnect': 'Use the native /provider flow to change or remove a provider profile.',
  'model.save_key': 'Use the native /provider flow to save provider credentials.',
  'plugins.manage': 'Native plugin management is not configured in this daemon.',
  'process.stop': 'Use /stop to cancel the active native turn; this daemon has no background-process registry.',
  'reload.env': 'Restart the Bun daemon after changing environment values; live .env reload is unavailable.',
  'reload.mcp': 'Restart the Bun daemon after changing MCP configuration; live MCP reload is unavailable.',
  'rollback.diff': 'Use /snapshots and /rollback <snapshot-id> for the native snapshot workflow.',
  'rollback.list': 'Use /snapshots for the native snapshot workflow.',
  'rollback.restore': 'Use /rollback <snapshot-id> for the native snapshot workflow.',
  'session.close': 'Native sessions are persistent; use /new or the session switcher instead.',
  'skills.manage': 'Use `bun run xerxes skill <name>` for bundled native skills.',
  'skills.reload': 'Restart the Bun daemon to reload bundled skill content.',
  'spawn_tree.list': 'Native spawn-tree persistence is not configured in this daemon.',
  'spawn_tree.load': 'Native spawn-tree persistence is not configured in this daemon.',
  'spawn_tree.save': 'Native spawn-tree persistence is not configured in this daemon.',
  'subagent.interrupt': 'Native subagent lifecycle control is not configured in this daemon.',
  'tools.configure': 'Native runtime tool configuration is not available through the daemon.',
  'voice.record': 'Native voice capture is not configured in this daemon.',
  'voice.toggle': 'Native voice capture is not configured in this daemon.'
})

/** Explicit error returned when an old UI RPC has no Bun daemon implementation. */
export class NativeDaemonUnsupportedError extends Error {
  readonly method: string

  constructor(method: string) {
    const guidance = NATIVE_UNSUPPORTED_RPC_GUIDANCE[method] ?? 'Use /help to see supported native commands.'
    super(`Native Bun daemon does not implement ${method}. ${guidance}`)
    this.name = 'NativeDaemonUnsupportedError'
    this.method = method
  }
}

/** Render the user-facing startup failure consistently across initial launch and recovery. */
export function formatBunDaemonStartupFailure(error: unknown): string {
  const detail = error instanceof Error ? error.message : String(error)
  return `Bun daemon startup failed: ${detail || 'unknown error'}`
}

/**
 * Resolve the Bun executable and TypeScript runtime entry used for a daemon
 * launched by the UI. Explicit settings win; the unified TypeScript package
 * works from source, build output, or a staged release without ambient setup.
 */
export function bunDaemonLaunch(
  projectDir: string,
  socketPath: string,
  pidPath: string,
  environment: Environment = process.env
): BunDaemonLaunch {
  const binary = firstEnvironmentValue(environment, 'XERXES_TUI_BUN', 'XERXES_BUN') || 'bun'
  const entryPath = resolveBunDaemonEntry(projectDir, environment)
  return {
    binary,
    entryPath,
    args: [entryPath, 'daemon', '--project-dir', projectDir, '--socket', socketPath, '--pid-file', pidPath]
  }
}

/** Resolve a configured or colocated Bun daemon entry point. */
export function resolveBunDaemonEntry(projectDir: string, environment: Environment = process.env): string {
  const configured = firstEnvironmentValue(environment, 'XERXES_TUI_BUN_DAEMON', 'XERXES_BUN_DAEMON')
  if (configured) {
    const entryPath = isAbsolute(configured) ? configured : resolve(projectDir, configured)
    if (!existsSync(entryPath)) {
      throw new Error(`Configured Bun daemon entry does not exist: ${entryPath}`)
    }
    return entryPath
  }

  const uiDirectory = dirname(fileURLToPath(import.meta.url))
  const packageRoot = resolve(uiDirectory, '..', '..')
  const candidates = [
    join(projectDir, 'xerxes', 'src', 'cli.ts'),
    join(projectDir, 'xerxes', 'dist', 'cli.js'),
    join(packageRoot, 'src', 'cli.ts'),
    join(packageRoot, 'dist', 'cli.js'),
    resolve(uiDirectory, '..', 'bin', 'xerxes.js')
  ]
  const entryPath = [...new Set(candidates)].find(candidate => existsSync(candidate))
  if (!entryPath) {
    throw new Error(
      'Could not locate the Bun daemon entry. Set XERXES_TUI_BUN_DAEMON (or XERXES_BUN_DAEMON) to the runtime cli path.'
    )
  }
  return entryPath
}

function firstEnvironmentValue(environment: Environment, ...names: readonly string[]): string {
  for (const name of names) {
    const value = environment[name]?.trim()
    if (value) {
      return value
    }
  }
  return ''
}

/** Convert a daemon slash response into the UI's shell-result shape without masking failures. */
export function shellResultFromSlashResponse(response: Record<string, unknown>): {
  readonly code: number
  readonly stderr: string
  readonly stdout: string
} {
  if (response.ok === false) {
    return {
      code: 127,
      stdout: '',
      stderr: String(response.error ?? 'Bang-command execution is unavailable in the Bun daemon.')
    }
  }
  return {
    code: 0,
    stdout: typeof response.output === 'string' ? response.output : '',
    stderr: typeof response.stderr === 'string' ? response.stderr : ''
  }
}

// ── Client ──────────────────────────────────────────────────────────────

interface Pending {
  resolve: (value: unknown) => void
  reject: (err: Error) => void
  timer: NodeJS.Timeout
}

type RpcObject = Record<string, any>

export interface GatewayClientOptions {
  /** Bun executable used when the client must launch a daemon. */
  bunBinary?: string
  /** Bun TypeScript CLI entry used when the client must launch a daemon. */
  bunDaemonPath?: string
  /** Expected source/release build identity supplied by the launching CLI. */
  expectedDaemonBuildId?: string
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
  private readonly bunBinary: string | undefined
  private readonly bunDaemonPath: string | undefined
  private readonly expectedDaemonBuildId: string
  private socket: Socket | null = null
  private proc: ChildProcess | null = null
  private nextId = 1
  private readonly pending = new Map<number, Pending>()
  private buffer = ''
  private initializeTranscriptCapture: GatewayTranscriptMessage[] | null = null
  private readonly stderrRing: string[] = []
  private spawnError: Error | null = null
  private spawnedDaemon = false
  private closed = false
  private activeSessionKey: string
  private readonly sessionKeys = new Map<string, string>()
  private lastApprovalRequestId = ''
  private readonly silentSockets = new WeakSet<Socket>()

  constructor(opts: GatewayClientOptions = {}) {
    super()
    this.setMaxListeners(100)
    this.bunBinary = opts.bunBinary?.trim() || undefined
    this.bunDaemonPath = opts.bunDaemonPath?.trim() || undefined
    this.expectedDaemonBuildId =
      opts.expectedDaemonBuildId?.trim() || process.env.XERXES_EXPECTED_DAEMON_BUILD_ID?.trim() || ''
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
      if (await this.ensureConnectedDaemonCurrent(socketPath, pidPath)) {
        this.emitClient('gateway.ready', { socketPath, spawned: false })
        return
      }
    }

    // No daemon reachable — launch the Bun runtime and poll until the socket appears.
    this.spawnBunDaemon(socketPath, pidPath)
    const deadline = Date.now() + STARTUP_TIMEOUT_MS
    while (Date.now() < deadline) {
      if (this.spawnError) {
        throw new Error(`could not start Bun daemon: ${this.spawnError.message}`)
      }
      if (this.proc && this.proc.exitCode !== null) {
        throw new Error(`daemon exited (code ${this.proc.exitCode}) before becoming ready:\n${this.stderrSnapshot()}`)
      }
      if (await this.tryConnect(socketPath)) {
        if (!(await this.ensureConnectedDaemonCurrent(socketPath, pidPath))) {
          throw new Error('newly spawned Bun daemon reported an unexpected build identity')
        }
        this.emitClient('gateway.ready', { socketPath, spawned: true })
        return
      }
      await delay(DAEMON_CONNECT_RETRY_MS)
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

  /**
   * Validate a source-checkout build before exposing gateway.ready. A stale
   * daemon is restarted only when its RPC PID, pid file, and exact launch
   * command all prove that it is this project's default local daemon.
   */
  private async ensureConnectedDaemonCurrent(socketPath: string, pidPath: string): Promise<boolean> {
    const expectedBuildId = this.expectedDaemonBuildId
    if (!expectedBuildId) {
      return true
    }

    let status: RpcObject
    try {
      status = await this.rawRequest<RpcObject>('runtime.status', {}, DAEMON_IDENTITY_TIMEOUT_MS)
    } catch (error) {
      await this.detachSocketSilently()
      throw new Error(
        `could not verify Bun daemon build ${expectedBuildId}: ${error instanceof Error ? error.message : String(error)}`
      )
    }

    const actualBuildId = String(status.daemon_build_id ?? '').trim()
    if (
      actualBuildId === expectedBuildId &&
      String(status.runtime ?? '') === 'bun-typescript' &&
      positiveInteger(status.daemon_protocol) === 35
    ) {
      return true
    }
    const daemonPid = positiveInteger(status.pid)
    const pidFilePid = pidFromFile(pidPath)
    const activity = await this.rawRequest<RpcObject>(
      'session.active_list',
      {},
      DAEMON_IDENTITY_TIMEOUT_MS
    ).catch(() => null)
    const sessions = activity && Array.isArray(activity.sessions) ? activity.sessions : null
    // An unverified inventory is treated as busy: source refresh must never
    // risk interrupting another TUI's in-flight turn.
    const activeTurns =
      sessions === null ||
      sessions.some(row => {
        if (!row || typeof row !== 'object') return true
        const session = row as RpcObject
        return Boolean(String(session.active_turn_id ?? '').trim()) || String(session.status ?? '') === 'working'
      })
    const launch = bunDaemonLaunch(this.projectDir, socketPath, pidPath, {
      ...process.env,
      ...(this.bunBinary ? { XERXES_TUI_BUN: this.bunBinary } : {}),
      ...(this.bunDaemonPath ? { XERXES_TUI_BUN_DAEMON: this.bunDaemonPath } : {})
    })
    const command = daemonPid === undefined ? '' : daemonProcessCommand(daemonPid)
    const decision = daemonBuildDecision({
      activeTurns,
      actualBuildId,
      commandMatches: daemonCommandMatches(command, launch),
      daemonPid,
      daemonProtocol: positiveInteger(status.daemon_protocol),
      daemonRuntime: String(status.runtime ?? ''),
      expectedBuildId,
      explicitSocket: Boolean(process.env.XERXES_DAEMON_SOCKET?.trim()),
      pidFilePid
    })

    if (decision === 'current') {
      return true
    }

    await this.detachSocketSilently()
    const mismatch = `Bun daemon build mismatch (running ${actualBuildId || 'unknown'}, expected ${expectedBuildId})`
    if (decision === 'reject' || daemonPid === undefined) {
      const reason = activeTurns
        ? 'Its active-session state is busy or could not be verified'
        : 'The connected process is custom or could not be proven local'
      throw new Error(
        `${mismatch}. ${reason}, so Xerxes left it running; restart it explicitly when idle.`
      )
    }

    try {
      process.kill(daemonPid, 'SIGTERM')
    } catch (error) {
      if (!isMissingProcessError(error)) {
        throw new Error(`${mismatch}. Could not stop the stale local daemon: ${String(error)}`)
      }
    }
    const deadline = Date.now() + 5000
    while (processIsAlive(daemonPid) && Date.now() < deadline) {
      await delay(DAEMON_CONNECT_RETRY_MS)
    }
    if (processIsAlive(daemonPid)) {
      throw new Error(`${mismatch}. The stale local daemon did not stop after SIGTERM.`)
    }
    return false
  }

  private async detachSocketSilently(): Promise<void> {
    const socket = this.socket
    if (!socket) {
      return
    }
    this.silentSockets.add(socket)
    await new Promise<void>(resolve => {
      socket.once('close', resolve)
      socket.destroy()
    })
  }

  private attachSocket(sock: Socket): void {
    this.socket = sock
    sock.setEncoding('utf8')
    sock.on('data', (chunk: string) => this.onData(chunk))
    sock.on('error', err => this.emitClient('gateway.error', { message: String((err as Error).message ?? err) }))
    sock.on('close', () => {
      const active = this.socket === sock
      if (active) {
        this.socket = null
      }
      if (active && !this.closed && !this.silentSockets.has(sock)) {
        this.emitClient('gateway.closed', {})
        this.emit('close')
      }
      if (active) {
        for (const [, p] of this.pending) {
          clearTimeout(p.timer)
          p.reject(new Error('gateway socket closed'))
        }
        this.pending.clear()
      }
    })
  }

  private spawnBunDaemon(socketPath: string, pidPath: string): void {
    const launch = bunDaemonLaunch(this.projectDir, socketPath, pidPath, {
      ...process.env,
      ...(this.bunBinary ? { XERXES_TUI_BUN: this.bunBinary } : {}),
      ...(this.bunDaemonPath ? { XERXES_TUI_BUN_DAEMON: this.bunDaemonPath } : {})
    })
    this.spawnError = null
    this.proc = spawn(launch.binary, launch.args, {
      stdio: ['ignore', 'ignore', 'pipe'],
      detached: true,
      env: bunDaemonEnvironment(this.expectedDaemonBuildId)
    })
    this.spawnedDaemon = true
    this.proc.once('error', error => {
      this.spawnError = error
      this.pushStderr(`Bun daemon launch failed: ${error.message}`)
    })
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
        this.lastApprovalRequestId = String(payload.id ?? payload.request_id ?? '')
      }
      for (const evt of adaptDaemonEvent(type, payload)) {
        const sessionId = typeof payload.session_id === 'string' ? payload.session_id : ''
        this.emitEvent(sessionId ? ({ ...evt, session_id: sessionId } as AnyEvent) : evt)
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
        return this.setupStatus() as Promise<T>

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
        return this.sessionActiveList(params) as Promise<T>

      case 'session.list':
        return this.sessionHistoryList(params) as Promise<T>

      case 'session.close':
        throw new NativeDaemonUnsupportedError(method)

      case 'session.delete':
        return this.sessionDelete(params) as Promise<T>

      case 'session.most_recent':
        return this.sessionMostRecent() as Promise<T>

      case 'session.title':
        return this.sessionTitle(params) as Promise<T>

      case 'session.status':
        return this.sessionStatus(params) as Promise<T>

      case 'session.compress':
        return this.sessionCompress(params) as Promise<T>

      case 'session.usage':
        return this.sessionUsage(params) as Promise<T>

      case 'session.save':
        return this.sessionSave(params) as Promise<T>

      case 'session.undo':
        return this.sessionUndo(params) as Promise<T>

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
          text: String(params.text ?? ''),
          ...(typeof params.display_text === 'string' ? { display_text: params.display_text } : {})
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
      case 'rollback.list':
      case 'rollback.diff':
      case 'rollback.restore':
      case 'tools.configure':
      case 'model.disconnect':
      case 'model.save_key':
        throw new NativeDaemonUnsupportedError(method)

      case 'browser.manage':
        return this.browserManage(params) as Promise<T>

      case 'model.options':
        return this.modelOptions() as Promise<T>

      default:
        return this.rawRequest<T>(method, params)
    }
  }

  private rawRequest<T = unknown>(
    method: string,
    params: Record<string, unknown> = {},
    timeoutMs = REQUEST_TIMEOUT_MS
  ): Promise<T> {
    const sock = this.socket
    if (!sock) {
      return Promise.reject(new Error('gateway not connected'))
    }
    const id = this.nextId++
    const frame = JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n'
    return new Promise<T>((res, rej) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        rej(new Error(`rpc timeout: ${method} (${timeoutMs}ms)`))
      }, timeoutMs)
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

  /**
   * Native daemon operations may return an application-level `{ ok: false }`
   * inside an otherwise valid JSON-RPC response. Convert that to a rejected
   * request so UI callers never render a fabricated success state.
   */
  private async nativeSuccess(method: string, params: Record<string, unknown>): Promise<RpcObject> {
    const raw = (await this.rawRequest<RpcObject>(method, params)) as RpcObject

    if (raw.ok === false) {
      throw new Error(String(raw.error ?? `native daemon rejected ${method}`))
    }

    return raw
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

  private async setupStatus(): Promise<RpcObject> {
    const raw = (await this.rawRequest('provider_list', {})) as RpcObject
    const profiles = Array.isArray(raw.profiles) ? raw.profiles : []
    return { provider_configured: profiles.length > 0 }
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
      // The native daemon does not expose the retired file-backed config
      // mtime contract.  A synthetic `Date.now()` made the poller report a
      // configuration change every five seconds even when nothing changed.
      // Zero intentionally disables that compatibility poll until the daemon
      // has a real revision source.
      return { mtime: 0 }
    }
    return { value: '' }
  }

  private async configSet(params: Record<string, unknown>): Promise<RpcObject> {
    const key = String(params.key ?? '')
    const value = String(params.value ?? '')
    if (key === 'model' && value) {
      const scopedValue = value.replace(/\s+--(?:global|tui-session)\s*$/i, '').trim()
      const selection = scopedValue.match(/^(.*?)\s+--provider\s+(.+)$/i)
      const model = (selection?.[1] ?? scopedValue).trim()
      const providerProfile = selection?.[2]?.trim()

      if (!model) {
        throw new Error('model id is required')
      }

      // The picker lists provider *profiles*, not just vendor labels. Select
      // that profile first so its base URL and credential travel with the
      // chosen model; treating `--provider` as part of the model id silently
      // left the previous provider active.
      if (providerProfile) {
        await this.nativeSuccess('provider_select', { name: providerProfile })
      }

      await this.nativeSuccess('runtime.reload', { model })
      return { value: model }
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
      await this.rawRequest('set_mode', { mode: value })
    }
    return { value }
  }

  private async sessionCreate(_params: Record<string, unknown>): Promise<RpcObject> {
    this.activeSessionKey = `tui:${randomKey()}`
    const finishCapture = this.captureInitializeInfo()

    try {
      const raw = await this.nativeSuccess('initialize', {
        project_dir: this.projectDir,
        session_key: this.activeSessionKey
      })
      const captured = finishCapture()
      const session = (raw.session ?? {}) as RpcObject
      const sessionId = String(session.id ?? '').trim()

      if (!sessionId) {
        throw new Error('native daemon initialize returned no session id')
      }

      this.sessionKeys.set(sessionId, this.activeSessionKey)
      return {
        info: this.sessionInfoFromInitialize(raw, session, captured),
        session_id: sessionId
      }
    } catch (error) {
      finishCapture()
      throw error
    }
  }

  private async sessionResume(params: Record<string, unknown>): Promise<RpcObject> {
    const id = String(params.session_id ?? '')
    this.activeSessionKey = id || this.sessionKey
    // `initialize` replays persisted history as notifications before its RPC
    // response. Capture those rows at the transport boundary and hydrate the
    // React transcript once: the v35 response intentionally exposes only a
    // numeric `session.messages` count, and forwarding every replay event
    // would otherwise cause one render per historical message.
    const finishCapture = this.captureInitializeInfo(true)

    try {
      const raw = await this.nativeSuccess('initialize', {
        project_dir: this.projectDir,
        resume_session_id: id,
        session_key: this.activeSessionKey
      })
      const captured = finishCapture()
      const session = (raw.session ?? {}) as RpcObject
      const sessionId = String(session.id ?? '').trim()

      if (!sessionId) {
        throw new Error('native daemon resume returned no session id')
      }

      const responseMessages = transcriptFromStoredMessages(session.messages)
      const messages = responseMessages.length ? responseMessages : captured.transcript
      const messageCount =
        typeof session.message_count === 'number'
          ? session.message_count
          : typeof session.messages === 'number'
            ? session.messages
            : messages.length

      this.sessionKeys.set(sessionId, this.activeSessionKey)
      return {
        info: this.sessionInfoFromInitialize(raw, session, captured),
        message_count: messageCount,
        messages,
        resumed: sessionId,
        running: Boolean(session.active_turn_id),
        session_id: sessionId,
        status: session.active_turn_id ? 'working' : 'idle'
      }
    } catch (error) {
      finishCapture()
      throw error
    }
  }

  private async sessionActiveList(params: Record<string, unknown>): Promise<RpcObject> {
    const raw = await this.nativeSuccess('session.active_list', params)
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

  private async sessionHistoryList(params: Record<string, unknown>): Promise<RpcObject> {
    const raw = await this.nativeSuccess('session.list', params)
    const rows = Array.isArray(raw.sessions) ? raw.sessions : []
    const sessions = rows.map((row: RpcObject) => {
      const id = String(row.session_id ?? row.id ?? row.key ?? '')
      if (id && row.key) {
        this.sessionKeys.set(id, String(row.key))
      }
      const updatedAt = Date.parse(String(row.updated_at ?? '')) / 1000
      const title = String(row.title ?? row.key ?? id)
      return {
        id,
        message_count: Number(row.message_count ?? row.messages ?? 0),
        preview: title,
        source: 'saved',
        started_at: Number.isFinite(updatedAt) ? updatedAt : Date.now() / 1000,
        title
      }
    })
    return { sessions }
  }

  private async sessionStatus(params: Record<string, unknown>): Promise<RpcObject> {
    const raw = await this.nativeSuccess('session.status', { session_key: this.keyFor(params.session_id) })
    return { output: JSON.stringify(raw.session ?? raw, null, 2) }
  }

  private async sessionMostRecent(): Promise<RpcObject> {
    const raw = await this.nativeSuccess('session.most_recent', {})
    const session = raw.session as RpcObject | null | undefined

    if (!session || typeof session !== 'object') {
      return {}
    }

    const sessionId = String(session.session_id ?? session.id ?? '').trim()
    return {
      ...(sessionId ? { session_id: sessionId } : {}),
      source: 'saved',
      title: String(session.title ?? '')
    }
  }

  private async sessionDelete(params: Record<string, unknown>): Promise<RpcObject> {
    const requested = String(params.session_id ?? '').trim()

    if (!requested) {
      throw new Error('session id is required')
    }

    const raw = await this.nativeSuccess('session.delete', { session_id: requested })
    const deleted = String(raw.session_id ?? '').trim()

    if (!deleted) {
      throw new Error('native session deletion returned no session id')
    }

    return { deleted }
  }

  private async sessionTitle(params: Record<string, unknown>): Promise<RpcObject> {
    const raw = await this.nativeSuccess('session.title', {
      session_key: this.keyFor(params.session_id),
      ...(typeof params.title === 'string' ? { title: params.title } : {})
    })
    return { title: String(raw.title ?? '') }
  }

  private async sessionCompress(params: Record<string, unknown>): Promise<RpcObject> {
    const raw = await this.nativeSuccess('session.compress', { session_key: this.keyFor(params.session_id) })
    const compacted = raw.compacted === true
    const before = finiteNumber(raw.tokens_before)
    const after = finiteNumber(raw.tokens_after)
    const tokenLine = before !== undefined && after !== undefined ? `${before} → ${after} tokens` : undefined

    return {
      ...(before === undefined ? {} : { before_tokens: before }),
      ...(after === undefined ? {} : { after_tokens: after }),
      summary: {
        headline: compacted ? 'context compacted' : 'nothing to compress',
        noop: !compacted,
        ...(tokenLine ? { token_line: tokenLine } : {})
      }
    }
  }

  private async sessionSave(params: Record<string, unknown>): Promise<RpcObject> {
    const raw = await this.nativeSuccess('session.save', {
      session_key: this.keyFor(params.session_id),
      ...(typeof params.title === 'string' ? { title: params.title } : {})
    })
    const session = raw.session as RpcObject | null | undefined
    const file = typeof session?.path === 'string' ? session.path : ''

    if (!file) {
      throw new Error('native session save returned no transcript path')
    }

    return { file }
  }

  private async sessionUndo(params: Record<string, unknown>): Promise<RpcObject> {
    const raw = await this.nativeSuccess('session.undo', { session_key: this.keyFor(params.session_id) })
    return { removed: finiteNumber(raw.dropped) ?? 0 }
  }

  private async browserManage(params: Record<string, unknown>): Promise<RpcObject> {
    const action = String(params.action ?? 'status')
      .trim()
      .toLowerCase()
    const url = typeof params.url === 'string' ? params.url.trim() : ''
    const raw = await this.nativeSuccess('browser.manage', {
      action,
      ...(url ? { url } : {})
    })
    const status = raw.status as RpcObject | undefined

    if (!status || typeof status.connected !== 'boolean') {
      throw new Error('native browser manager returned no connection status')
    }

    return {
      connected: status.connected,
      kind: typeof status.kind === 'string' ? status.kind : 'none',
      pages: Array.isArray(raw.pages) ? raw.pages : [],
      ...(typeof status.endpoint === 'string' ? { url: status.endpoint } : {})
    }
  }

  private async sessionUsage(params: Record<string, unknown>): Promise<RpcObject> {
    const raw = await this.nativeSuccess('session.status', { session_key: this.keyFor(params.session_id) })
    return usageFromStatus((raw.session ?? raw) as Record<string, unknown>)
  }

  private async slashExec(params: Record<string, unknown>): Promise<RpcObject> {
    const raw = String(params.command ?? '').trim()
    const command = raw.startsWith('/') || raw.startsWith('!') ? raw : `/${raw}`
    const result = (await this.rawRequest('slash', { command })) as RpcObject
    if (result.ok === false) {
      return { output: 'error: ' + String(result.error ?? 'command was rejected') }
    }
    return { output: typeof result.output === 'string' ? result.output : '' }
  }

  private async shellExec(params: Record<string, unknown>): Promise<RpcObject> {
    const result = await this.rawRequest('slash', { command: `!${String(params.command ?? '')}` })
    return shellResultFromSlashResponse(result as Record<string, unknown>)
  }

  private approvalRespond(params: Record<string, unknown>): Promise<unknown> {
    const choice = String(params.choice ?? '')
    const response =
      choice === 'always'
        ? 'always'
        : choice === 'session' || choice === 'approve_for_session'
          ? 'approve_for_session'
        : choice === 'deny' || choice === 'reject'
          ? 'reject'
          : 'approve'
    const requestId = String(params.request_id ?? this.lastApprovalRequestId)

    return this.rawRequest('permission_response', { request_id: requestId, response })
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
    const raw = (await this.rawRequest('provider_list', {})) as RpcObject
    const profiles = Array.isArray(raw.profiles) ? raw.profiles : []
    const active = profiles.find((profile: RpcObject) => Boolean(profile.active)) as RpcObject | undefined

    return {
      model: String(active?.model ?? ''),
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

  private keyFor(id: unknown): string {
    const sid = String(id ?? '').trim()
    return sid ? (this.sessionKeys.get(sid) ?? sid) : this.activeSessionKey
  }

  private captureInitializeInfo(
    captureTranscript = false
  ): () => { info: null | SessionInfo; transcript: GatewayTranscriptMessage[]; usage: null | Usage } {
    let info: null | SessionInfo = null
    let usage: null | Usage = null
    let stopped = false
    const transcript: GatewayTranscriptMessage[] = []

    if (captureTranscript) {
      if (this.initializeTranscriptCapture) {
        throw new Error('cannot initialize two resumed sessions concurrently')
      }
      this.initializeTranscriptCapture = transcript
    }
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
      if (!stopped) {
        stopped = true
        this.off('session.info', onInfo)
        this.off('status.update', onStatus)
        if (this.initializeTranscriptCapture === transcript) {
          this.initializeTranscriptCapture = null
        }
      }
      return { info, transcript, usage }
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
    if (evt.type === 'transcript.append' && this.initializeTranscriptCapture) {
      this.initializeTranscriptCapture.push({ ...(evt.payload as GatewayTranscriptMessage) })
      return
    }

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

function positiveInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0 ? value : undefined
}

function pidFromFile(path: string): number | undefined {
  try {
    const value = readFileSync(path, 'utf8').trim()
    return /^[1-9]\d*$/.test(value) ? positiveInteger(Number(value)) : undefined
  } catch {
    return undefined
  }
}

function daemonProcessCommand(pid: number): string {
  try {
    return execFileSync('ps', ['-p', String(pid), '-o', 'command='], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore']
    }).trim()
  } catch {
    return ''
  }
}

function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)
    return true
  } catch (error) {
    return !isMissingProcessError(error)
  }
}

function isMissingProcessError(error: unknown): boolean {
  return error instanceof Error && 'code' in error && error.code === 'ESRCH'
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

function finiteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
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
  try {
    const packageJson = JSON.parse(readFileSync(join(projectDir, 'package.json'), 'utf8')) as { version?: unknown }
    return typeof packageJson.version === 'string' ? packageJson.version.trim() : ''
  } catch {
    return ''
  }
}
