// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * DaemonRpc — one NDJSON JSON-RPC 2.0 connection to the shared daemon
 * (ui/PROTOCOL.md is the frozen contract).
 *
 * Deliberately small: `call` auto-connects (reusing a listening daemon or
 * launching one), requests are id-correlated with deadlines, no-id frames
 * named `event` fan out to subscribers, and the connection retries in the
 * background with capped backoff so a daemon restart heals itself instead of
 * stranding the UI offline.
 */

import { releaseIdleProjectDaemon } from '../../ui/lib/daemonMigration.js'
import { existsSync } from 'node:fs'
import { EventEmitter } from 'node:events'
import type { ChildProcess } from 'node:child_process'
import { Socket } from 'node:net'

import {
  canonicalProjectDir,
  daemonAddress,
  legacyProjectDaemonPaths,
  launchDaemon,
  type Env,
} from './spawn.js'

/** Matches the daemon's newline-delimited frame cap. */
export const MAX_FRAME_BYTES = 16 * 1024 * 1024

const CONNECT_TIMEOUT_MS = 2_000
const STARTUP_TIMEOUT_MS = 15_000
const DEFAULT_DEADLINE_MS = 120_000
const POLL_MS = 25
const RETRY_BASE_MS = 250
const RETRY_MAX_MS = 5_000

interface Waiter {
  method: string
  resolve: (value: Record<string, unknown>) => void
  reject: (error: Error) => void
  timer: NodeJS.Timeout
}

export interface DaemonRpcOptions {
  projectDir?: string
  env?: Env
  deadlineMs?: number
  socketPath?: string
  /** Injectable startup boundary for deterministic desktop transport tests. */
  launch?: typeof launchDaemon
  startupTimeoutMs?: number
  remoteUpdate?: () => Promise<Record<string, unknown>>
  reconnectRemote?: (signal: AbortSignal) => Promise<string>
  expectedRemoteBuildId?: () => string | undefined
}

export class DaemonRpc extends EventEmitter {
  readonly projectDir: string
  private externalSocket: string | undefined
  private readonly reconnectRemote: DaemonRpcOptions['reconnectRemote']
  private readonly lifecycle = new AbortController()
  private readonly env: Env
  private readonly deadlineMs: number
  private socket: Socket | null = null
  private buffer = ''
  private seq = 1
  private readonly waiters = new Map<number, Waiter>()
  private connecting: Promise<void> | null = null
  private retryTimer: NodeJS.Timeout | null = null
  private retries = 0
  private previouslyConnected = false
  private connectionLeaseToken: string | undefined
  private connectionLeaseAttached = false
  private stopped = false
  private stderrRing: string[] = []
  private writeTail: Promise<void> = Promise.resolve()
  private launchedProcess: ChildProcess | null = null
  private readonly launch: typeof launchDaemon
  private readonly startupTimeoutMs: number
  private readonly remoteUpdate: DaemonRpcOptions['remoteUpdate']
  private readonly expectedRemoteBuildId: DaemonRpcOptions['expectedRemoteBuildId']

  constructor(options: DaemonRpcOptions = {}) {
    super()
    this.externalSocket = options.socketPath
    this.reconnectRemote = options.reconnectRemote
    this.remoteUpdate = options.remoteUpdate
    this.expectedRemoteBuildId = options.expectedRemoteBuildId
    this.projectDir = options.socketPath ? options.projectDir ?? "/" : canonicalProjectDir(options.projectDir)
    this.env = options.env ?? process.env
    this.deadlineMs = options.deadlineMs ?? DEFAULT_DEADLINE_MS
    this.launch = options.launch ?? launchDaemon
    this.startupTimeoutMs = Math.max(1, options.startupTimeoutMs ?? STARTUP_TIMEOUT_MS)
  }

  get online(): boolean {
    return this.socket !== null
  }

  onEvent(handler: (type: string, payload: Record<string, unknown>) => void): void {
    this.on('event', handler)
  }

  offEvent(handler: (type: string, payload: Record<string, unknown>) => void): void {
    this.off('event', handler)
  }

  onConnection(handler: (online: boolean) => void): void {
    this.on('connection', handler)
  }

  /**
   * JSON-RPC call; resolves with the result object (RPC-level `{ok:false}`
   * payloads are results the caller reads, only transport faults reject).
   */
  async call<T = Record<string, unknown>>(
    method: string,
    params: Record<string, unknown> = {},
  ): Promise<T> {
    await this.ensure()
    if (method === 'initialize' && this.connectionLeaseToken && !this.connectionLeaseAttached) {
      try {
        const lease = await this.send<Record<string, unknown>>('connection.lease', { token: this.connectionLeaseToken, project_dir: this.projectDir })
        if (lease.ok === true) this.connectionLeaseAttached = true
        else this.connectionLeaseToken = undefined
      } catch (error) {
        // Only a daemon ANSWER says the token is unusable: send() formats
        // JSON-RPC failures as `rpc -<code>: <message>` ("already attached to
        // another transport" is the normal case after a remote tunnel drop,
        // because the remote daemon has not observed the dead socket's EOF) —
        // and wedging on it replays the same failing reclaim on every future
        // initialize while the heartbeat refuses to retry (rpc errors
        // classify as configuration, not transport). Drop the token on an
        // answer and fall through to the plain initialize. A transport fault
        // ("connection closed", "rpc timeout: ...") says nothing about the
        // token: keep it so the server-side grace can still re-bind the next
        // connection, and let the fault surface from the send below.
        if (error instanceof Error && /^rpc( -\d+|:)/.test(error.message)) {
          this.connectionLeaseToken = undefined
        } else {
          throw error
        }
      }
    }
    // A shared daemon's launch directory is not this window's workspace.
    // Bind every opening handshake to the host-owned window target, including
    // resumes and reconnects, rather than trusting a renderer-supplied path.
    const result = await this.send<T>(method, method === 'initialize' || method === 'session.open'
      ? { ...params, project_dir: this.projectDir }
      : params)
    if (method === 'initialize' && result && typeof result === 'object'
      && 'connection_lease_supported' in result && result.connection_lease_supported === true && !this.connectionLeaseAttached) {
      // Best effort: the handshake above already succeeded, so a failure to
      // establish lease ownership (losing the 30s reconnect grace) must not
      // discard it.
      try {
        const lease = await this.send<Record<string, unknown>>('connection.lease', {})
        if (lease.ok === true && typeof lease.token === 'string') {
          this.connectionLeaseToken = lease.token
          this.connectionLeaseAttached = true
        }
      } catch {
        // Keep the successful initialize result; the next clean initialize
        // can try the lease again.
      }
    }
    if (method === 'initialize' && this.externalSocket && this.expectedRemoteBuildId?.()) {
      return { ...result, desktop_expected_daemon_build_id: this.expectedRemoteBuildId() }
    }
    return result
  }

  /** Replace only an idle local runtime, then wait for a fresh connection. */
  async restartRuntime(allowLegacy = false): Promise<Record<string, unknown>> {
    if (this.externalSocket && this.remoteUpdate) return this.remoteUpdate()
    if (this.externalSocket) return { ok: false, error: 'Update the runtime on the remote machine, then reconnect this workspace.' }
    await this.ensure()
    const previous = this.socket
    let result = await this.send<Record<string, unknown>>('runtime.restart_if_idle', {})
    if (result.ok !== true && typeof result.error === 'string' && result.error.startsWith('Unknown method') && allowLegacy) {
      // Only an explicit click may migrate a runtime without atomic idle restart.
      // Read every session, including work owned by other connected clients.
      const status = await this.send<Record<string, unknown>>('runtime.status', {})
      const list = await this.send<Record<string, unknown>>('session.active_list', {})
      if (status.ok !== true || list.ok !== true || !Array.isArray(list.sessions)) return { ok: false, error: 'Could not check running work. Runtime was left connected.' }
      if (status.active_subagents !== 0 || status.channels_configured !== false) return { ok: false, busy: true }
      for (const value of list.sessions) {
        if (!value || typeof value !== 'object') return { ok: false, error: 'Could not check a session. Runtime was left connected.' }
        const session = value as Record<string, unknown>
        if (session.status !== 'idle' || session.active_turn_id) return { ok: false, busy: true }
        if (typeof session.key !== 'string') return { ok: false, error: 'Session identity unavailable. Runtime was left connected.' }
        const terminals = await this.send<Record<string, unknown>>('terminal.list', { session_key: session.key })
        const monitors = await this.send<Record<string, unknown>>('monitor.list', { session_key: session.key })
        if (terminals.ok !== true || !Array.isArray(terminals.terminals) || monitors.ok !== true || !Array.isArray(monitors.monitors)) return { ok: false, error: 'Could not check background work. Runtime was left connected.' }
        if (terminals.terminals.some(row => row?.running) || monitors.monitors.some(row => row?.state === 'watching')) return { ok: false, busy: true }
      }
      result = await this.send<Record<string, unknown>>('shutdown', {})
    }
    if (result.ok !== true) {
      if (result.busy === true) return result
      if (typeof result.error === 'string' && !result.error.startsWith('Unknown method')) return result
      return { ok: false, error: 'This older runtime needs a one-time restart. Click Restart workspace runtime to check running work and reconnect with the bundled version.' }
    }
    const deadline = Date.now() + STARTUP_TIMEOUT_MS
    while (this.socket === previous) {
      if (this.stopped) throw new Error('Runtime update cancelled: workspace closed')
      if (Date.now() >= deadline) throw new Error('The runtime did not shut down. Retry when it has finished stopping.')
      await new Promise<void>(resolve => setTimeout(resolve, POLL_MS))
    }
    await this.ensure()
    return { ok: true }
  }

  /** Stop reconnecting and drop the socket; a launched daemon keeps running. */
  dispose(): void {
    this.lifecycle.abort()
    this.stopped = true
    if (this.retryTimer) {
      clearTimeout(this.retryTimer)
      this.retryTimer = null
    }
    const socket = this.socket
    this.socket = null
    socket?.destroy()
    this.failWaiters(new Error('connection disposed'))
  }

  // ── Connection ───────────────────────────────────────────────────────

  private ensure(): Promise<void> {
    if (this.socket) return Promise.resolve()
    if (this.stopped) return Promise.reject(new Error('daemon rpc disposed'))
    if (this.connecting) return this.connecting
    const attempt = this.open().then(
      () => {
        if (this.connecting === attempt) this.connecting = null
      },
      error => {
        if (this.connecting === attempt) this.connecting = null
        throw error
      },
    )
    this.connecting = attempt
    return attempt
  }

  private async open(): Promise<void> {
    if (this.externalSocket) {
      if (!await this.tryAttach(this.externalSocket)) {
        if (!this.reconnectRemote) throw new Error('SSH transport unavailable. Reconnect from Workspace.')
        const replacement = await this.reconnectRemote(this.lifecycle.signal)
        this.lifecycle.signal.throwIfAborted()
        this.externalSocket = replacement
        if (!await this.tryAttach(replacement)) throw new Error('SSH reconnected but the remote runtime is unavailable. Retry the connection.')
      }
      this.announce(true); return
    }
    // Keep existing project work attached until its old runtime exits.
    const legacy = legacyProjectDaemonPaths(this.projectDir, this.env).socketPath
    if (!this.env.XERXES_DAEMON_SOCKET && (process.platform === 'win32' || existsSync(legacy)) && await this.tryAttach(legacy)) {
      if (!await releaseIdleProjectDaemon(method => this.send(method, {}))) {
        this.announce(true)
        return
      }
    }
    const { socketPath, pidPath } = daemonAddress(this.projectDir, this.env)
    if (await this.tryAttach(socketPath)) {
      this.announce(true)
      return
    }
    try {
      // A slow startup must not create a new process on every retry. Continue
      // probing the existing child until it exits or its socket becomes ready.
      const child = this.launchedProcess
      if (!child || child.exitCode !== null || child.signalCode !== null) {
        const launched = this.launch(this.projectDir, socketPath, pidPath, this.env, line => {
          this.stderrRing.push(line.slice(0, 512))
          if (this.stderrRing.length > 200) this.stderrRing.shift()
        })
        this.launchedProcess = launched
        launched.once('error', error => {
          this.stderrRing.push(error.message)
          if (this.launchedProcess === launched) this.launchedProcess = null
        })
      }
    } catch (error) {
      throw new Error(
        `could not launch daemon: ${error instanceof Error ? error.message : String(error)}`,
      )
    }
    const deadline = Date.now() + this.startupTimeoutMs
    while (Date.now() < deadline) {
      if (this.stopped) throw new Error('daemon rpc disposed')
      if (await this.tryAttach(socketPath)) {
        this.retries = 0
        this.announce(true)
        return
      }
      await new Promise<void>(r => setTimeout(r, POLL_MS))
    }
    throw new Error(
      `daemon not ready within ${this.startupTimeoutMs}ms${this.launchedProcess?.pid ? ` (process ${this.launchedProcess.pid} is still starting)` : ''}:\n${this.stderrRing.slice(-8).join('\n')}`,
    )
  }

  private tryAttach(socketPath: string): Promise<boolean> {
    return new Promise<boolean>(resolveAttach => {
      const sock = new Socket()
      let done = false
      const settle = (outcome: boolean): void => {
        if (done) return
        done = true
        clearTimeout(guard)
        if (outcome) this.attach(sock)
        else sock.destroy()
        // Resolve on EVERY path; an unresolved promise once parked startup
        // forever when a socket file existed but never answered.
        resolveAttach(outcome)
      }
      const guard = setTimeout(() => settle(false), CONNECT_TIMEOUT_MS)
      sock.on('error', () => settle(false))
      sock.once('connect', () => settle(true))
      try { sock.connect({ path: socketPath }) } catch { settle(false) }
    })
  }

  private attach(sock: Socket): void {
    this.socket = sock
    this.connectionLeaseAttached = false
    this.buffer = ''
    sock.setEncoding('utf8')
    sock.on('data', (chunk: string) => this.onData(chunk))
    sock.on('error', error => this.emit('protocol_error', { message: String((error as Error).message ?? error) }))
    sock.on('close', () => {
      if (this.socket !== sock) return
      this.socket = null
      this.failWaiters(new Error('connection closed'))
      if (this.stopped) return
      this.announce(false)
      this.scheduleRetry()
    })
  }

  private scheduleRetry(): void {
    if (this.stopped || this.retryTimer || this.connecting) return
    const wait = Math.min(RETRY_BASE_MS * 2 ** Math.min(this.retries, 16), RETRY_MAX_MS)
    this.retries += 1
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null
      this.ensure().catch(() => this.scheduleRetry())
    }, wait)
    this.retryTimer.unref?.()
  }

  private announce(online: boolean): void {
    if (online) this.retries = 0
    this.emit('connection', online)
    if (this.previouslyConnected) this.emit('event', 'desktop_connection', { online })
    if (online) this.previouslyConnected = true
  }

  // ── Framing ──────────────────────────────────────────────────────────

  private onData(chunk: string): void {
    this.buffer += chunk
    let nl = this.buffer.indexOf('\n')
    while (nl !== -1) {
      const line = this.buffer.slice(0, nl)
      this.buffer = this.buffer.slice(nl + 1)
      if (Buffer.byteLength(line, 'utf8') > MAX_FRAME_BYTES) {
        this.breakOversized()
        return
      }
      if (line.trim()) this.onFrame(line)
      nl = this.buffer.indexOf('\n')
    }
    if (Buffer.byteLength(this.buffer, 'utf8') > MAX_FRAME_BYTES) this.breakOversized()
  }

  private breakOversized(): void {
    this.buffer = ''
    this.emit('protocol_error', {
      message: `gateway frame exceeds maximum size of ${MAX_FRAME_BYTES} bytes`,
    })
    this.failWaiters(new Error('frame exceeded maximum size'))
    this.socket?.destroy()
  }

  private onFrame(line: string): void {
    let parsed: unknown
    try {
      parsed = JSON.parse(line)
    } catch {
      this.emit('protocol_error', { message: `unparseable frame: ${line.slice(0, 160)}` })
      return
    }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      this.emit('protocol_error', { message: 'non-object frame' })
      return
    }
    const frame = parsed as {
      id?: unknown
      method?: unknown
      result?: unknown
      error?: unknown
      params?: unknown
    }
    if (frame.id !== undefined && frame.id !== null) {
      const waiter = this.waiters.get(Number(frame.id))
      if (!waiter) return
      this.waiters.delete(Number(frame.id))
      clearTimeout(waiter.timer)
      if (frame.error && typeof frame.error === 'object') {
        const err = frame.error as Record<string, unknown>
        const code = typeof err.code === 'number' ? ` ${err.code}` : ''
        const message = typeof err.message === 'string' && err.message ? err.message : 'unknown error'
        waiter.reject(new Error(`rpc${code}: ${message}`))
      } else {
        waiter.resolve(
          (frame.result && typeof frame.result === 'object' && !Array.isArray(frame.result)
            ? frame.result
            : {}) as Record<string, unknown>,
        )
      }
      return
    }
    if (frame.method === 'event' && frame.params && typeof frame.params === 'object') {
      const params = frame.params as Record<string, unknown>
      const type = typeof params.type === 'string' ? params.type : ''
      if (!type) {
        this.emit('protocol_error', { message: 'event frame without type' })
        return
      }
      const payload =
        params.payload && typeof params.payload === 'object' && !Array.isArray(params.payload)
          ? (params.payload as Record<string, unknown>)
          : {}
      // initialize returns the authoritative structured transcript. Its legacy
      // history notifications travel over a separate Electron IPC channel and
      // can arrive after that response, duplicating already hydrated messages.
      // Slash-command replay outside initialization remains available.
      if (type === 'notification' && payload.category === 'history'
        && [...this.waiters.values()].some(waiter => waiter.method === 'initialize')) return
      this.emit('event', type, payload)
      return
    }
    this.emit('protocol_error', { message: `unrecognized frame: ${line.slice(0, 160)}` })
  }

  // ── Requests ─────────────────────────────────────────────────────────

  private send<T>(method: string, params: Record<string, unknown>): Promise<T> {
    const sock = this.socket
    if (!sock) return Promise.reject(new Error('daemon not connected'))
    const id = this.seq++
    const frame = `${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`
    return new Promise<T>((resolveCall, rejectCall) => {
      const timer = setTimeout(() => {
        this.waiters.delete(id)
        rejectCall(new Error(`rpc timeout: ${method} (${this.deadlineMs}ms)`))
      }, this.deadlineMs)
      this.waiters.set(id, {
        method,
        resolve: resolveCall as (value: Record<string, unknown>) => void,
        reject: rejectCall,
        timer,
      })
      this.write(sock, frame).catch(error => {
        clearTimeout(timer)
        if (this.waiters.delete(id)) {
          rejectCall(error instanceof Error ? error : new Error(String(error)))
        }
      })
    })
  }

  /** Serialized writes so a full kernel buffer becomes backpressure, not memory. */
  private write(sock: Socket, frame: string): Promise<void> {
    const next = this.writeTail.then(
      () =>
        new Promise<void>((done, fail) => {
          if (this.socket !== sock) {
            fail(new Error('daemon not connected'))
            return
          }
          sock.write(frame, error => (error ? fail(error) : done()))
        }),
    )
    this.writeTail = next.catch(() => {})
    return next
  }

  private failWaiters(error: Error): void {
    for (const [, waiter] of this.waiters) {
      clearTimeout(waiter.timer)
      waiter.reject(error)
    }
    this.waiters.clear()
  }
}
