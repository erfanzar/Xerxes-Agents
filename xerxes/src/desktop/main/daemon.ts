// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * DaemonRpc — one NDJSON JSON-RPC 2.0 connection to the per-project daemon
 * (ui/PROTOCOL.md is the frozen contract).
 *
 * Deliberately small: `call` auto-connects (reusing a listening daemon or
 * launching one), requests are id-correlated with deadlines, no-id frames
 * named `event` fan out to subscribers, and the connection retries in the
 * background with capped backoff so a daemon restart heals itself instead of
 * stranding the UI offline.
 */

import { EventEmitter } from 'node:events'
import { connect, type Socket } from 'node:net'

import {
  canonicalProjectDir,
  daemonAddress,
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
  resolve: (value: Record<string, unknown>) => void
  reject: (error: Error) => void
  timer: NodeJS.Timeout
}

export interface DaemonRpcOptions {
  projectDir?: string
  env?: Env
  deadlineMs?: number
}

export class DaemonRpc extends EventEmitter {
  readonly projectDir: string
  private readonly env: Env
  private readonly deadlineMs: number
  private socket: Socket | null = null
  private buffer = ''
  private seq = 1
  private readonly waiters = new Map<number, Waiter>()
  private connecting: Promise<void> | null = null
  private retryTimer: NodeJS.Timeout | null = null
  private retries = 0
  private stopped = false
  private stderrRing: string[] = []
  private writeTail: Promise<void> = Promise.resolve()

  constructor(options: DaemonRpcOptions = {}) {
    super()
    this.projectDir = canonicalProjectDir(options.projectDir)
    this.env = options.env ?? process.env
    this.deadlineMs = options.deadlineMs ?? DEFAULT_DEADLINE_MS
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
    return this.send<T>(method, params)
  }

  /** Stop reconnecting and drop the socket; a launched daemon keeps running. */
  dispose(): void {
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
    const { socketPath, pidPath } = daemonAddress(this.projectDir, this.env)
    if (await this.tryAttach(socketPath)) {
      this.announce(true)
      return
    }
    try {
      launchDaemon(this.projectDir, socketPath, pidPath, this.env, line => {
        this.stderrRing.push(line.slice(0, 512))
        if (this.stderrRing.length > 200) this.stderrRing.shift()
      })
    } catch (error) {
      throw new Error(
        `could not launch daemon: ${error instanceof Error ? error.message : String(error)}`,
      )
    }
    const deadline = Date.now() + STARTUP_TIMEOUT_MS
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
      `daemon not ready within ${STARTUP_TIMEOUT_MS}ms:\n${this.stderrRing.slice(-8).join('\n')}`,
    )
  }

  private tryAttach(socketPath: string): Promise<boolean> {
    return new Promise<boolean>(resolveAttach => {
      const sock = connect({ path: socketPath })
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
      sock.once('error', () => settle(false))
      sock.once('connect', () => settle(true))
    })
  }

  private attach(sock: Socket): void {
    this.socket = sock
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
