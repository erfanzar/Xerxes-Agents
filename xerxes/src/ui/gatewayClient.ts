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

import { type ChildProcess, execFile, execFileSync, spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { EventEmitter } from 'node:events'
import { existsSync, readFileSync, realpathSync } from 'node:fs'
import { promisify } from 'node:util'
import { connect, type Socket } from 'node:net'
import { homedir } from 'node:os'
import { dirname, isAbsolute, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import {
  adaptDaemonEvent,
  looksLikeInternalUserPrompt,
  sessionInfoFromInit,
  transcriptFromStoredMessages,
  usageFromStatus
} from './gatewayAdapter.js'
import type {
  AnyEvent,
  GatewayTranscriptMessage,
  LiveSessionStatus,
  SessionInflightTurn,
  SessionInflightTool,
  SubagentSnapshotPayload
} from './gatewayTypes.js'
import { controlChannelPath, isWindows } from './lib/hostPlatform.js'
import { ImageAttachmentError, loadImageAttachment, resolveAttachmentPath } from './lib/imageAttachment.js'
import type { SessionInfo, Usage } from './types.js'
import { compact } from './lib/compact.js'

const MAX_GATEWAY_LOG_LINES = 200
const MAX_LOG_LINE_BYTES = 4096
/** Matches the daemon socket protocol cap for one newline-delimited JSON-RPC frame. */
export const MAX_GATEWAY_FRAME_BYTES = 16 * 1024 * 1024
const STARTUP_TIMEOUT_MS = Math.max(
  5000,
  Number.parseInt(process.env.XERXES_TUI_STARTUP_TIMEOUT_MS ?? '15000', 10) || 15000
)
const REQUEST_TIMEOUT_MS = Math.max(
  30000,
  Number.parseInt(process.env.XERXES_TUI_RPC_TIMEOUT_MS ?? '120000', 10) || 120000
)
const DAEMON_IDENTITY_TIMEOUT_MS = Math.min(5000, STARTUP_TIMEOUT_MS)
/** Bound a single socket connect/detach attempt so a wedged listener cannot stall startup. */
const SOCKET_OPERATION_TIMEOUT_MS = 2000
/** One extra identity probe absorbs a daemon that is briefly busy during startup. */
const DAEMON_IDENTITY_ATTEMPTS = 2
// A bundled daemon is normally ready in 85-105 ms. A 25 ms cadence reaches
// it within one short interval without a busy loop or the old 150 ms stall.
export const DAEMON_CONNECT_RETRY_MS = 25
// Bound the session-id → session-key map: a long-lived TUI switching through
// many sessions must not grow it without limit. Insertion order doubles as
// recency, so eviction simply drops the oldest entry (simple LRU).
const MAX_SESSION_KEYS = 200

// ── Path resolution (v35 daemon path contract) ───────────────────────────

/** `$XERXES_HOME` or `~/.xerxes`. */
function xerxesHome(): string {
  const override = (process.env.XERXES_HOME ?? '').trim()
  return override ? resolve(override) : join(homedir(), '.xerxes')
}

/** Canonical project dir: nearest git root when available, otherwise cwd. */
export function resolveProjectDir(projectDir?: string): string {
  // Synchronous by design: this runs exactly once per process, in the
  // GatewayClient constructor before the renderer mounts, so the ~10ms git
  // call can never freeze a live frame loop. Every mid-session subprocess
  // (git head, ps identity probe) uses the async helpers instead.
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
 * Per-project control-channel + pid paths. `XERXES_DAEMON_SOCKET` overrides the
 * channel address while retaining the deterministic per-project pid path.
 *
 * Windows gets a named pipe rather than a Unix socket; `node:net` connects to
 * either through the same `path` option. This must derive the identical address
 * to `daemon/paths.ts`, or the TUI looks for the daemon somewhere it never bound
 * and starts a second one.
 */
export function daemonPaths(
  projectDir: string,
  platform: NodeJS.Platform = process.platform
): { socketPath: string; pidPath: string } {
  const digest = createHash('sha256').update(projectDir, 'utf8').digest('hex').slice(0, 16)
  const base = join(xerxesHome(), 'daemon', 'projects')
  const override = (process.env.XERXES_DAEMON_SOCKET ?? '').trim()
  return {
    socketPath: override || controlChannelPath(base, digest, platform),
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
  /** Undefined means the connected daemon cannot prove that no child work is active. */
  readonly activeSubagents: number | undefined
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
    input.activeSubagents === undefined ||
    input.activeSubagents > 0 ||
    input.activeTurns
  ) {
    return 'reject'
  }
  return 'restart'
}

/** Exact source-checkout daemon signature required before automatic restart. */
export function daemonCommandMatches(
  command: string,
  launch: BunDaemonLaunch,
  platform: NodeJS.Platform = process.platform
): boolean {
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
  // Windows reports a single quoted command line — `"C:\...\bun.exe"
  // "C:\...\cli.ts" daemon --project-dir ...` — so the argv-joined expectation
  // never appears verbatim and every Windows daemon looked unrecognized, which
  // downgraded a routine build-change restart into "restart it explicitly".
  // Dropping the quotes is safe here: a Windows path cannot contain `"`, and the
  // whitespace guard above already rejected anything whose tokens could merge.
  // POSIX matching is left byte-identical.
  const observed = isWindows(platform) ? command.replaceAll('"', '') : command
  return observed.includes(
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
  private readonly stderrRing: string[] = []
  private spawnError: Error | null = null
  private spawnedDaemon = false
  private closed = false
  private startPromise: Promise<void> | null = null
  private activeSessionKey: string
  private readonly sessionKeys = new Map<string, string>()
  /** Most recent approval request per session id; '' key holds the last untagged request. */
  private readonly approvalRequestIds = new Map<string, string>()
  private lastApprovalRequestId = ''
  private readonly silentSockets = new WeakSet<Socket>()
  /** Serializes socket writes so a full kernel buffer applies backpressure instead of growing userland memory. */
  private writeChain: Promise<void> = Promise.resolve()
  private initializeTranscriptCapture: { rows: GatewayTranscriptMessage[]; sessionId: string | null } | null = null

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

  /** Connect, launching the daemon if none is reachable. Concurrent callers share one cold-start attempt. */
  start(): Promise<void> {
    if (this.socket) return Promise.resolve()
    if (this.startPromise) return this.startPromise

    const attempt = this.startOnce()
    this.startPromise = attempt
    void attempt.then(
      () => {
        if (this.startPromise === attempt) this.startPromise = null
      },
      () => {
        if (this.startPromise === attempt) this.startPromise = null
      }
    )
    return attempt
  }

  private async startOnce(): Promise<void> {
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
      let settled = false
      const finish = (connected: boolean) => {
        if (settled) {
          return
        }
        settled = true
        clearTimeout(timer)
        if (!connected) {
          sock.destroy()
        }
        res(connected)
      }
      // A socket path that exists but never answers the connect handshake
      // would otherwise hang startup forever.
      const timer = setTimeout(() => finish(false), SOCKET_OPERATION_TIMEOUT_MS)
      const onError = () => finish(false)
      sock.once('error', onError)
      sock.once('connect', () => {
        sock.removeListener('error', onError)
        this.attachSocket(sock)
        finish(true)
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
      status = await this.probeDaemonIdentity()
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
    // Unverified parent or child activity is treated as busy: source refresh
    // must never risk interrupting another TUI's in-flight work.
    const activeTurns =
      sessions === null ||
      sessions.some(row => {
        if (!row || typeof row !== 'object') return true
        const session = row as RpcObject
        return Boolean(String(session.active_turn_id ?? '').trim()) || String(session.status ?? '') === 'working'
      })
    const activeSubagents = nonNegativeInteger(status.active_subagents)
    const launch = bunDaemonLaunch(this.projectDir, socketPath, pidPath, {
      ...process.env,
      ...(this.bunBinary ? { XERXES_TUI_BUN: this.bunBinary } : {}),
      ...(this.bunDaemonPath ? { XERXES_TUI_BUN_DAEMON: this.bunDaemonPath } : {})
    })
    const command = daemonPid === undefined ? '' : await daemonProcessCommand(daemonPid)
    const decision = daemonBuildDecision({
      activeSubagents,
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
      // Name the remedy, and name the right one. This used to end every
      // rejection with "restart it explicitly when idle" regardless of why
      // it was rejected — which is actively misleading for a daemon that is
      // already idle and was only refused because its provenance could not
      // be proven. It also never said *how*, leaving you to find the pid.
      const busy = activeTurns || activeSubagents === undefined || activeSubagents > 0
      const stop = daemonPid === undefined
        ? `find its pid in ${pidPath} and stop that process`
        : `stop it with: kill ${daemonPid}`

      throw new Error(
        busy
          ? `${mismatch}. A session or subagent is still working, so Xerxes left the daemon running. Retry once it goes idle, or ${stop}, then relaunch.`
          : `${mismatch}. The daemon was not started by this Xerxes install, so Xerxes will not stop it for you. To continue, ${stop}, then relaunch.`
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

  /**
   * Probe `runtime.status` with one retry. A daemon mid-compaction or serving a
   * large replay can miss the first short deadline without being unhealthy, and
   * failing startup on that single timeout forced a pointless manual restart.
   */
  private async probeDaemonIdentity(): Promise<RpcObject> {
    let lastError: unknown = null
    for (let attempt = 0; attempt < DAEMON_IDENTITY_ATTEMPTS; attempt++) {
      try {
        return await this.rawRequest<RpcObject>('runtime.status', {}, DAEMON_IDENTITY_TIMEOUT_MS)
      } catch (error) {
        lastError = error
      }
    }
    throw lastError instanceof Error ? lastError : new Error(String(lastError))
  }

  private async detachSocketSilently(): Promise<void> {
    const socket = this.socket
    if (!socket) {
      return
    }
    this.silentSockets.add(socket)
    await new Promise<void>(resolve => {
      // 'close' always follows destroy() on a healthy stream, but never let a
      // misbehaving transport wedge startup behind a listener that never fires.
      const timer = setTimeout(resolve, SOCKET_OPERATION_TIMEOUT_MS)
      socket.once('close', () => {
        clearTimeout(timer)
        resolve()
      })
      socket.destroy()
      // destroy() sets the flag synchronously even when 'close' is deferred to
      // a later tick; a socket already gone needs no wait at all.
      if (socket.destroyed) {
        clearTimeout(timer)
        resolve()
      }
    })
  }

  private attachSocket(sock: Socket): void {
    this.socket = sock
    // A partial line buffered from a dead socket must not prefix the first
    // frame decoded on its replacement.
    this.buffer = ''
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
      if (Buffer.byteLength(line, 'utf8') > MAX_GATEWAY_FRAME_BYTES) {
        this.closeForOversizedFrame()
        return
      }
      if (line.trim()) {
        this.onLine(line)
      }
      nl = this.buffer.indexOf('\n')
    }
    if (Buffer.byteLength(this.buffer, 'utf8') > MAX_GATEWAY_FRAME_BYTES) {
      this.closeForOversizedFrame()
    }
  }

  private closeForOversizedFrame(): void {
    this.buffer = ''
    this.emitClient('gateway.protocol_error', {
      message: `gateway frame exceeds maximum size of ${MAX_GATEWAY_FRAME_BYTES} bytes`
    })
    this.socket?.destroy()
  }

  private onLine(line: string): void {
    let parsed: unknown
    try {
      parsed = JSON.parse(line)
    } catch {
      this.emitClient('gateway.protocol_error', { line: truncate(line) })
      return
    }

    if (!isRecord(parsed)) {
      this.emitClient('gateway.protocol_error', { line: truncate(line) })
      return
    }
    const frame = parsed as { id?: unknown; method?: unknown; result?: unknown; error?: unknown; params?: unknown }

    // Response/error: carries an `id`.
    if (frame.id !== undefined && frame.id !== null) {
      const pending = this.pending.get(frame.id as number)
      if (!pending) {
        return
      }
      this.pending.delete(frame.id as number)
      clearTimeout(pending.timer)
      if (frame.error) {
        const error = isRecord(frame.error) ? frame.error : {}
        const code = typeof error.code === 'number' ? ` ${error.code}` : ''
        const message = typeof error.message === 'string' && error.message ? error.message : 'unknown error'
        pending.reject(new Error(`rpc${code}: ${message}`))
      } else {
        pending.resolve(frame.result)
      }
      return
    }

    // Event notification: `{ method: "event", params: { type, payload } }`.
    if (frame.method === 'event' && frame.params && typeof frame.params === 'object') {
      const params = frame.params as { type?: string; payload?: Record<string, unknown> }
      const type = typeof params.type === 'string' ? params.type : ''
      if (!type || (params.payload !== undefined && !isRecord(params.payload))) {
        this.emitClient('gateway.protocol_error', { line: truncate(line) })
        return
      }
      const payload = params.payload ?? {}
      if (type === 'approval_request') {
        const requestId = String(payload.id ?? payload.request_id ?? '')
        const sessionId = typeof payload.session_id === 'string' ? payload.session_id : ''
        // Track per session: with several live sessions, a connection-global
        // "last approval" can answer a prompt belonging to a different tab.
        this.lastApprovalRequestId = requestId
        if (sessionId && requestId) {
          // delete+set keeps insertion order as recency for the LRU bound.
          this.approvalRequestIds.delete(sessionId)
          this.approvalRequestIds.set(sessionId, requestId)
          if (this.approvalRequestIds.size > MAX_SESSION_KEYS) {
            const oldest = this.approvalRequestIds.keys().next().value
            if (oldest !== undefined) {
              this.approvalRequestIds.delete(oldest)
            }
          }
        }
      }
      for (const evt of adaptDaemonEvent(type, payload)) {
        const sessionId = typeof payload.session_id === 'string'
          ? payload.session_id
          : typeof payload.background_task_id === 'string'
            ? payload.background_task_id
            : ''
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
        return this.sessionResume(params) as Promise<T>

      case 'session.activate':
        return this.sessionActivate(params) as Promise<T>

      case 'session.active_list':
        return this.sessionActiveList(params) as Promise<T>

      case 'session.peek':
        return this.sessionPeek(params) as Promise<T>

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
        return this.nativeSuccess('cancel', { session_key: this.keyFor(params.session_id) }) as Promise<T>

      case 'session.steer':
        return this.rawRequest<T>('steer', {
          content: String(params.text ?? ''),
          session_key: this.keyFor(params.session_id)
        })

      case 'prompt.submit':
        return this.nativeSuccess('turn.submit', {
          session_key: this.keyFor(params.session_id),
          ...(typeof params.submission_id === 'string' && params.submission_id.trim()
            ? { submission_id: params.submission_id.trim() }
            : {}),
          text: String(params.text ?? ''),
          ...(typeof params.display_text === 'string' ? { display_text: params.display_text } : {}),
          // Validated PendingAttachment entries from the /image command. The
          // daemon re-validates at the turn.submit boundary; a missing or
          // empty list keeps the frame identical to a plain-text submit.
          ...(Array.isArray(params.images) && params.images.length ? { images: params.images } : {})
        }) as Promise<T>

      case 'prompt.background':
        return this.nativeSuccess('turn.background', {
          session_key: this.keyFor(params.session_id),
          text: String(params.text ?? '')
        }) as Promise<T>

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

      case 'image.attach':
        return this.imageAttach(params) as Promise<T>

      case 'terminal.resize':
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
        return this.modelOptions(params) as Promise<T>

      case 'model.models':
        return this.modelModels(params) as Promise<T>

      case 'reasoning.levels':
        return this.reasoningLevels() as Promise<T>

      default:
        return this.rawRequest<T>(method, params)
    }
  }

  private rawRequest<T = unknown>(
    method: string,
    params: Record<string, unknown> = {},
    timeoutMs = REQUEST_TIMEOUT_MS
  ): Promise<T> {
    if (!this.socket) {
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
      this.enqueueWrite(frame).catch(error => {
        clearTimeout(timer)
        // The response may already have settled this id; only reject when the
        // entry is still ours.
        if (this.pending.delete(id)) {
          rej(error instanceof Error ? error : new Error(String(error)))
        }
      })
    })
  }

  /**
   * Serialize frames through a chain gated on the socket's write callback.
   * The callback fires only once the chunk is flushed to the kernel, so when
   * a 20MB image frame fills the buffer, later frames wait here instead of
   * piling up in userland memory.
   */
  private enqueueWrite(frame: string): Promise<void> {
    const next = this.writeChain.then(() => this.writeFrameNow(frame))
    // A failed frame must not poison the chain for everything after it.
    this.writeChain = next.catch(() => {})
    return next
  }

  private writeFrameNow(frame: string): Promise<void> {
    const sock = this.socket
    if (!sock) {
      return Promise.reject(new Error('gateway not connected'))
    }
    return new Promise<void>((resolve, reject) => {
      sock.write(frame, error => (error ? reject(error) : resolve()))
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
    if (!this.socket) {
      return
    }
    void this.enqueueWrite(JSON.stringify({ jsonrpc: '2.0', method, params }) + '\n').catch(() => {})
  }

  close(): void {
    this.closed = true
    this.startPromise = null
    const socket = this.socket
    this.socket = null
    // Reject in-flight requests immediately: nulling this.socket first makes
    // the socket 'close' handler skip this map, so without an explicit sweep
    // callers would hang until their 120s timeout fires (and those timers
    // keep the process alive the whole time).
    for (const [, p] of this.pending) {
      clearTimeout(p.timer)
      p.reject(new Error('gateway closed'))
    }
    this.pending.clear()
    socket?.end()
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
      const raw = await this.nativeSuccess('set_mode', {
        mode: value,
        session_key: this.keyFor(params.session_id)
      })
      const mode = String(raw.mode ?? value)
      const planMode = raw.plan_mode === true
      return { info: { mode, plan_mode: planMode }, value: mode }
    }
    return { value }
  }

  private async sessionCreate(params: Record<string, unknown>): Promise<RpcObject> {
    // Commit the new key only after the daemon confirms it; a failed
    // initialize must not strand the client on a dead session key.
    const nextSessionKey = `tui:${randomKey()}`
    const finishCapture = this.captureInitializeInfo()

    try {
      const agentId = typeof params.agent_id === 'string' ? params.agent_id.trim() : ''
      const raw = await this.nativeSuccess('initialize', {
        project_dir: this.projectDir,
        session_key: nextSessionKey,
        ...(agentId ? { agent_id: agentId } : {})
      })
      const captured = finishCapture()
      const session = (raw.session ?? {}) as RpcObject
      const sessionId = String(session.id ?? '').trim()

      if (!sessionId) {
        throw new Error('native daemon initialize returned no session id')
      }

      this.activeSessionKey = nextSessionKey
      this.rememberSessionKey(sessionId, nextSessionKey)
      return {
        info: await this.sessionInfoFromInitialize(raw, session, captured),
        session_id: sessionId
      }
    } catch (error) {
      finishCapture()
      throw error
    }
  }

  private async sessionResume(params: Record<string, unknown>): Promise<RpcObject> {
    const id = String(params.session_id ?? '')
    // Never bind the raw session id as this connection's session key: another
    // connection resuming the same session would derive the identical key and
    // the daemon would alias both connections onto one session. Reuse the key
    // this connection already owns for the session; otherwise mint a fresh
    // connection-scoped key, exactly like sessionCreate.
    const nextSessionKey = id ? (this.sessionKeys.get(id) ?? `tui:${randomKey()}`) : this.sessionKey
    // `initialize` replays persisted history as notifications before its RPC
    // response. Capture those rows at the transport boundary and hydrate the
    // React transcript once: the v35 response intentionally exposes only a
    // numeric `session.messages` count, and forwarding every replay event
    // would otherwise cause one render per historical message. The capture is
    // scoped to the resumed session so tagged rows from other live sessions
    // keep flowing to their own tabs instead of being swallowed.
    const finishCapture = this.captureInitializeInfo(true, id || null)

    try {
      const raw = await this.nativeSuccess('initialize', {
        project_dir: this.projectDir,
        resume_session_id: id,
        session_key: nextSessionKey
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

      // Same commit-after-confirm rule as sessionCreate: only adopt the key
      // once the daemon accepted the resume.
      this.activeSessionKey = nextSessionKey
      this.rememberSessionKey(sessionId, nextSessionKey)
      const status = liveSessionStatus(session)
      const subagentSnapshots = subagentSnapshotsFromSession(session)
      return {
        info: await this.sessionInfoFromInitialize(raw, session, captured),
        // A resumed session can still be mid-turn (reattach to live work);
        // forward the inflight snapshot exactly like session.activate.
        inflight: inflightFromSession(session),
        message_count: messageCount,
        messages,
        resumed: sessionId,
        running: status !== 'idle',
        session_id: sessionId,
        status,
        ...(subagentSnapshots ? { subagent_snapshots: subagentSnapshots } : {})
      }
    } catch (error) {
      finishCapture()
      throw error
    }
  }

  private async sessionActivate(params: Record<string, unknown>): Promise<RpcObject> {
    const id = String(params.session_id ?? '')
    const nextSessionKey = this.keyFor(id)
    const statusRaw = await this.nativeSuccess('session.status', { session_key: nextSessionKey })
    const statusSession = (statusRaw.session ?? {}) as RpcObject
    const targetCwd = optionalTrimmedText(statusSession.cwd) || this.projectDir
    // session.open is an idempotent attach for an already-live session and,
    // unlike session.status, also moves the daemon connection's active key.
    // Daemon-owned slash commands omit an explicit key, so a status-only
    // activation made them act on the previously selected tab. Pass the
    // target's own cwd because session.open also refreshes an existing
    // session's cwd; using the project root would silently undo a tab-local
    // directory change.
    const raw = await this.nativeSuccess('session.open', {
      project_dir: targetCwd,
      session_key: nextSessionKey
    })
    const session = (raw.session ?? {}) as RpcObject
    const sessionId = String(session.id ?? '').trim()

    if (!sessionId) {
      throw new Error('native daemon activation returned no session id')
    }

    this.activeSessionKey = nextSessionKey
    this.rememberSessionKey(sessionId, nextSessionKey)
    const inflight = inflightFromSession(session)
    const subagentSnapshots = subagentSnapshotsFromSession(session)
    const messages = transcriptFromStoredMessages(session.transcript)
    const status = liveSessionStatus(session)
    return {
      info: await this.sessionInfoFromInitialize(raw, session, { info: null, usage: null }),
      inflight,
      message_count: Number(session.message_count ?? session.messages ?? 0),
      // session.open returns the already-live transcript without competing
      // with its running turn.
      messages,
      running: status !== 'idle',
      session_id: sessionId,
      session_key: nextSessionKey,
      status,
      ...(subagentSnapshots ? { subagent_snapshots: subagentSnapshots } : {})
    }
  }

  private async sessionActiveList(params: Record<string, unknown>): Promise<RpcObject> {
    const raw = await this.nativeSuccess('session.active_list', params)
    const rows = Array.isArray(raw.sessions) ? raw.sessions : []
    const sessions = rows.map((row: RpcObject) => {
      const id = String(row.id ?? row.session_id ?? row.key ?? '')
      if (id && row.key) {
        this.rememberSessionKey(id, String(row.key))
      }
      const lastActive = Number(row.last_active)
      const status = liveSessionStatus(row)
      const inflight = isRecord(row.inflight) ? row.inflight : undefined
      const activityText = optionalTrimmedText(inflight?.user)
      // Internal prompts (skill activation, compaction, steers) are runtime
      // scaffolding; they must never become the card's visible activity line.
      const activity = activityText && !looksLikeInternalUserPrompt(activityText) ? activityText : undefined
      const title = optionalTrimmedText(row.title) ?? ''
      return {
        ...optionalSessionLinkFields(row),
        ...(activity ? { activity } : {}),
        current: this.keyFor(id) === this.activeSessionKey,
        id,
        ...(Number.isFinite(lastActive) ? { last_active: lastActive } : {}),
        message_count: Number(row.messages ?? 0),
        model: String(row.model ?? ''),
        preview: title,
        status,
        title
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
        this.rememberSessionKey(id, String(row.key))
      }
      const updatedAt = Date.parse(String(row.updated_at ?? '')) / 1000
      const title = optionalTrimmedText(row.title) ?? ''
      return {
        ...optionalSessionLinkFields(row),
        id,
        message_count: Number(row.message_count ?? row.messages ?? 0),
        preview: title,
        source: 'saved',
        ...(Number.isFinite(updatedAt) ? { last_message_at: updatedAt } : {}),
        // Compatibility for existing UI callers; never fabricate `now`, which
        // makes an unreadable/missing timestamp look like a new conversation.
        started_at: Number.isFinite(updatedAt) ? updatedAt : 0,
        title
      }
    })
    return { sessions }
  }

  private async sessionStatus(params: Record<string, unknown>): Promise<RpcObject> {
    const raw = await this.nativeSuccess('session.status', { session_key: this.keyFor(params.session_id) })
    return { output: JSON.stringify(raw.session ?? raw, null, 2) }
  }

  /** Inspect one live session without changing the daemon connection's active key. */
  private async sessionPeek(params: Record<string, unknown>): Promise<RpcObject> {
    const sessionId = String(params.session_id ?? '').trim()
    if (!sessionId) {
      throw new Error('session id is required')
    }

    const raw = await this.nativeSuccess('session.status', { session_key: this.keyFor(sessionId) })
    const session = isRecord(raw.session) ? raw.session : undefined
    if (!session) {
      throw new Error(`live session not found: ${sessionId}`)
    }
    const subagentSnapshots = subagentSnapshotsFromSession(session)

    return {
      inflight: inflightFromSession(session),
      messages: transcriptFromStoredMessages(session.transcript),
      session_id: String(session.id ?? sessionId),
      status: liveSessionStatus(session),
      ...(subagentSnapshots ? { subagent_snapshots: subagentSnapshots } : {})
    }
  }

  private async sessionMostRecent(): Promise<RpcObject> {
    const raw = await this.nativeSuccess('session.most_recent', { project_dir: this.projectDir })
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
    const command = raw.startsWith('/') || raw.startsWith('!') || raw.startsWith('#') ? raw : `/${raw}`
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

  /**
   * `image.attach` is a local TUI operation, not a daemon RPC: the file is
   * read, magic-byte-sniffed, and capped here, and the resulting base64 rides
   * the `images` param of the next `turn.submit` (where the daemon validates
   * it again). Params: `{ path, cwd? }`.
   */
  private async imageAttach(params: Record<string, unknown>): Promise<RpcObject> {
    const rawPath = String(params.path ?? '').trim()
    if (!rawPath) {
      throw new ImageAttachmentError('image.attach requires a path')
    }
    const cwd = typeof params.cwd === 'string' && params.cwd.trim() ? params.cwd : process.cwd()
    const loaded = await loadImageAttachment(resolveAttachmentPath(rawPath, cwd))
    return {
      attached: true,
      data: loaded.data,
      media_type: loaded.mediaType,
      name: loaded.name,
      path: loaded.path,
      size: loaded.size
    }
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
    const sessionId = String(params.session_id ?? '').trim()
    const requestId = String(
      params.request_id || (sessionId ? this.approvalRequestIds.get(sessionId) : '') || this.lastApprovalRequestId
    )
    if (sessionId) {
      this.approvalRequestIds.delete(sessionId)
    }

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
          group: item.category ? String(item.category) : undefined,
          meta: item.meta ? String(item.meta) : undefined,
          text: String(item.value ?? '')
        }))
      : []
    return { items, replace_from: method === 'complete.slash' ? 1 : undefined }
  }

  private async modelOptions(params: Record<string, unknown>): Promise<RpcObject> {
    const [raw, status] = (await Promise.all([
      this.rawRequest('provider_list', {}),
      this.rawRequest('session.status', { session_key: this.keyFor(params.session_id) })
    ])) as [RpcObject, RpcObject]
    const profiles = Array.isArray(raw.profiles) ? raw.profiles : []
    const storedActive = profiles.find((profile: RpcObject) => Boolean(profile.active)) as RpcObject | undefined
    const session = status.session && typeof status.session === 'object' ? (status.session as RpcObject) : undefined
    const hasRuntimeProfileIdentity = Boolean(
      session && Object.prototype.hasOwnProperty.call(session, 'profile_name')
    )
    const runtimeProfileName = String(session?.profile_name ?? '').trim()
    const current = hasRuntimeProfileIdentity
      ? (profiles.find(
          (profile: RpcObject) =>
            String(profile.name ?? profile.provider ?? '').trim() === runtimeProfileName
        ) as RpcObject | undefined)
      : storedActive
    const currentName = String(current?.name ?? current?.provider ?? '').trim()
    const liveModel = String(session?.model ?? '').trim()

    return {
      model: liveModel || String(current?.model ?? ''),
      provider: currentName,
      providers: profiles.map((profile: RpcObject) => {
        const profileName = String(profile.name ?? profile.provider ?? 'provider')
        return {
          configured_model: String(profile.model ?? ''),
          is_current: Boolean(currentName && profileName === currentName),
          name: profileName,
          provider_type: String(profile.provider ?? ''),
          slug: profileName
        }
      })
    }
  }

  private async modelModels(params: Record<string, unknown>): Promise<RpcObject> {
    const profileName = String(params.profile_name ?? params.profile ?? '').trim()
    if (!profileName) {
      throw new Error('provider profile name is required')
    }
    const raw = await this.nativeSuccess('fetch_models', { profile_name: profileName })
    const models = Array.isArray(raw.models)
      ? [...new Set(raw.models.map(model => String(model).trim()).filter(Boolean))]
      : []

    const catalog = Array.isArray(raw.catalog)
      ? raw.catalog
          .filter((entry): entry is RpcObject => typeof entry === 'object' && entry !== null)
          .map(entry => ({
            id: String(entry.id ?? '').trim(),
            ...(typeof entry.context_limit === 'number' ? { context_limit: entry.context_limit } : {}),
            ...(entry.context_source ? { context_source: String(entry.context_source) } : {}),
            ...(typeof entry.max_output_tokens === 'number'
              ? { max_output_tokens: entry.max_output_tokens }
              : {}),
            ...(entry.output_source ? { output_source: String(entry.output_source) } : {}),
            ...(entry.overridden === true ? { overridden: true } : {})
          }))
          .filter(entry => Boolean(entry.id))
      : []

    return {
      models,
      ...(catalog.length ? { catalog } : {}),
      ...(raw.source ? { source: String(raw.source) } : {}),
      ...(raw.warning ? { warning: String(raw.warning) } : {})
    }
  }

  /**
   * Reasoning efforts the active model accepts, as the provider reports them.
   *
   * The list is model-scoped rather than fixed, so the picker asks each time
   * it opens instead of rendering a menu that may not match the model.
   */
  private async reasoningLevels(): Promise<RpcObject> {
    const raw = (await this.nativeSuccess('reasoning_levels', {})) as RpcObject
    const levels = Array.isArray(raw.levels) ? raw.levels : []
    return {
      current: String(raw.current ?? ''),
      default: raw.default ? String(raw.default) : '',
      levels: levels
        .map((entry: RpcObject) => ({
          description: entry.description ? String(entry.description) : '',
          effort: String(entry.effort ?? '').trim()
        }))
        .filter((entry: { effort: string }) => Boolean(entry.effort)),
      source: String(raw.source ?? '')
    }
  }

  private keyFor(id: unknown): string {
    const sid = String(id ?? '').trim()
    return sid ? (this.sessionKeys.get(sid) ?? sid) : this.activeSessionKey
  }

  private rememberSessionKey(id: string, key: string): void {
    this.sessionKeys.delete(id)
    this.sessionKeys.set(id, key)
    if (this.sessionKeys.size > MAX_SESSION_KEYS) {
      const oldest = this.sessionKeys.keys().next().value
      if (oldest !== undefined) {
        this.sessionKeys.delete(oldest)
      }
    }
  }

  private captureInitializeInfo(
    captureTranscript = false,
    transcriptSessionId: string | null = null
  ): () => { info: null | SessionInfo; transcript: GatewayTranscriptMessage[]; usage: null | Usage } {
    let info: null | SessionInfo = null
    let usage: null | Usage = null
    let stopped = false
    const transcript: GatewayTranscriptMessage[] = []

    if (captureTranscript) {
      if (this.initializeTranscriptCapture) {
        throw new Error('cannot initialize two resumed sessions concurrently')
      }
      this.initializeTranscriptCapture = { rows: transcript, sessionId: transcriptSessionId }
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
    const capture = this.initializeTranscriptCapture
    return () => {
      if (!stopped) {
        stopped = true
        this.off('session.info', onInfo)
        this.off('status.update', onStatus)
        if (capture !== null && this.initializeTranscriptCapture === capture) {
          this.initializeTranscriptCapture = null
        }
      }
      return { info, transcript, usage }
    }
  }

  private async sessionInfoFromInitialize(
    raw: RpcObject,
    session: RpcObject,
    captured: { info: null | SessionInfo; usage: null | Usage }
  ): Promise<SessionInfo> {
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
      head_hash: withUsage.head_hash || (await localGitHead(cwd)),
      version: withUsage.version || localProjectVersion(cwd)
    }
  }

  private emitClient(type: string, payload: Record<string, unknown>): void {
    const evt = { type, payload } as AnyEvent
    this.emitEvent(evt)
  }

  private emitEvent(evt: AnyEvent): void {
    const capture = this.initializeTranscriptCapture
    if (evt.type === 'transcript.append' && capture) {
      const eventSessionId = (evt as { session_id?: unknown }).session_id
      // Rows tagged for a different live session belong to that session's
      // stream; swallowing them into the resume capture would lose them.
      const foreign =
        capture.sessionId !== null && typeof eventSessionId === 'string' && eventSessionId !== capture.sessionId
      if (!foreign) {
        capture.rows.push({ ...(evt.payload as GatewayTranscriptMessage) })
        return
      }
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

/**
 * Map the daemon's inflight turn snapshot. The user line is filtered like the
 * transcript: an internal prompt (skill activation, compaction request, …)
 * is runtime scaffolding, never the chat's visible activity.
 */
function inflightFromSession(session: RpcObject): null | SessionInflightTurn {
  const inflight = isRecord(session.inflight) ? session.inflight : undefined
  if (!inflight) {
    return null
  }
  const user = String(inflight.user ?? '')
  const tools = Array.isArray(inflight.tools)
    ? inflight.tools.flatMap((row): SessionInflightTool[] => {
        if (!isRecord(row)) {
          return []
        }
        const name = optionalTrimmedText(row.name) ?? 'tool'
        const id = optionalTrimmedText(row.id)
        const args = optionalTrimmedText(row.arguments)
        const error = optionalTrimmedText(row.error)
        const durationMs = typeof row.duration_ms === 'number' && Number.isFinite(row.duration_ms)
          ? row.duration_ms
          : undefined

        return [{
          ...(args ? { arguments: args } : {}),
          ...(durationMs === undefined ? {} : { duration_ms: durationMs }),
          ...(error ? { error } : {}),
          ...(id ? { id } : {}),
          name,
          ...(typeof row.ok === 'boolean' ? { ok: row.ok } : {})
        }]
      })
    : undefined
  const startedAt = typeof inflight.started_at === 'number' && Number.isFinite(inflight.started_at)
    ? inflight.started_at
    : undefined
  const thinking = optionalTrimmedText(inflight.thinking)

  return {
    assistant: String(inflight.assistant ?? ''),
    ...(startedAt === undefined ? {} : { started_at: startedAt }),
    streaming: Boolean(inflight.streaming),
    ...(thinking ? { thinking } : {}),
    ...(tools?.length ? { tools } : {}),
    user: looksLikeInternalUserPrompt(user) ? '' : user
  }
}

/** Forward the daemon's persisted subagent manifest rows, dropping malformed entries. */
function subagentSnapshotsFromSession(session: RpcObject): SubagentSnapshotPayload[] | undefined {
  const rows = Array.isArray(session.subagent_snapshots) ? session.subagent_snapshots : undefined
  if (!rows?.length) {
    return undefined
  }
  const snapshots = rows.flatMap((row): SubagentSnapshotPayload[] => {
    if (!isRecord(row)) {
      return []
    }
    const id = optionalTrimmedText(row.id)
    const status = optionalTrimmedText(row.status)
    if (!id || !status) {
      return []
    }

    return [{ ...(row as Record<string, unknown>), id, status } as SubagentSnapshotPayload]
  })

  return snapshots.length ? snapshots : undefined
}

function optionalSessionLinkFields(row: RpcObject): RpcObject {
  const agentId = optionalTrimmedText(row.agent_id)
  const rawKind = optionalTrimmedText(row.kind ?? row.session_kind)
  const kind = rawKind === 'main' || rawKind === 'subagent' ? rawKind : undefined
  const model = optionalTrimmedText(row.model)
  const parentSessionId = nullableTrimmedText(row.parent_session_id)
  const resumable = typeof row.resumable === 'boolean' ? row.resumable : undefined
  const rootSessionId = nullableTrimmedText(row.root_session_id)
  const status = optionalTrimmedText(row.status)
  const subagentId = nullableTrimmedText(row.subagent_id)

  return {
    ...(agentId ? { agent_id: agentId } : {}),
    ...(kind ? { kind } : {}),
    ...(model ? { model } : {}),
    ...(parentSessionId !== undefined ? { parent_session_id: parentSessionId } : {}),
    ...(resumable === undefined ? {} : { resumable }),
    ...(rootSessionId !== undefined ? { root_session_id: rootSessionId } : {}),
    ...(status ? { status } : {}),
    ...(subagentId !== undefined ? { subagent_id: subagentId } : {})
  }
}

function liveSessionStatus(row: RpcObject): LiveSessionStatus {
  const status = optionalTrimmedText(row.status)
  if (status === 'idle' || status === 'starting' || status === 'waiting' || status === 'working') {
    return status
  }

  return row.active_turn_id ? 'working' : 'idle'
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

function optionalTrimmedText(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined
  const normalized = value.trim()

  return normalized || undefined
}

function nullableTrimmedText(value: unknown): null | string | undefined {
  if (value === null) return null

  return optionalTrimmedText(value)
}

function positiveInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0 ? value : undefined
}

function nonNegativeInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined
}

function pidFromFile(path: string): number | undefined {
  try {
    const value = readFileSync(path, 'utf8').trim()
    return /^[1-9]\d*$/.test(value) ? positiveInteger(Number(value)) : undefined
  } catch {
    return undefined
  }
}

const execFileAsync = promisify(execFile)

/**
 * The daemon's command line, or '' when it cannot be read.
 *
 * Mirrors `core/processLiveness.ts`; see the note at the top of that file for
 * why the TUI keeps its own copy. Windows has no `ps`, so the equivalent is a
 * pid-filtered CIM query. An unreadable command line means "identity not
 * proven", which the caller already treats as a reason not to kill anything.
 * Async because the identity probe runs while the UI is already up: a sync
 * `ps`/`powershell` spawn would freeze the frame loop for its full duration.
 */
async function daemonProcessCommand(pid: number, platform: NodeJS.Platform = process.platform): Promise<string> {
  const target = Number.isFinite(pid) ? Math.trunc(pid) : -1
  const [command, args] = isWindows(platform)
    ? ([
        'powershell.exe',
        [
          '-NoProfile',
          '-NonInteractive',
          '-Command',
          `(Get-CimInstance Win32_Process -Filter "ProcessId=${target}").CommandLine`
        ]
      ] as const)
    : (['ps', ['-p', String(target), '-o', 'command=']] as const)
  try {
    const { stdout } = await execFileAsync(command, [...args], {
      encoding: 'utf8',
      ...(isWindows(platform) ? { windowsHide: true } : {})
    })
    return stdout.trim()
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
  return compact<Usage>({
    calls: merged.calls ?? 0,
    compressions: merged.compressions,
    // Zero is the protocol's explicit unknown-capacity sentinel; do not revive
    // a previous model's window after a profile switch.
    context_max: incoming?.context_max ?? base.context_max,
    context_percent: merged.context_percent,
    context_used: incoming?.context_used ?? base.context_used,
    cost_status: merged.cost_status,
    cost_usd: merged.cost_usd,
    dev_credits_spent_micros: merged.dev_credits_spent_micros,
    input: merged.input ?? 0,
    output: merged.output ?? 0,
    reasoning: merged.reasoning,
    total: merged.total ?? 0
  })
}

function mergeSessionInfo(base: SessionInfo, incoming?: null | Partial<SessionInfo>): SessionInfo {
  if (!incoming) {
    return base
  }

  return compact<SessionInfo>({
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
  })
}

async function localGitHead(projectDir: string): Promise<string> {
  try {
    const { stdout } = await execFileAsync('git', ['-C', projectDir, 'rev-parse', '--short=12', 'HEAD'], {
      encoding: 'utf8'
    })
    return stdout.trim()
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
