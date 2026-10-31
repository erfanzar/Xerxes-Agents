// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { HubOperatorError, isPrivateProviderType } from './ports.js'
import { isRecord } from './protocol.js'

export const DAEMON_MAX_FRAME_BYTES = 16 * 1024 * 1024

/** A daemon conversation id (the form a resumed chat is bound under), as the daemon's own `looksLikeSessionId`. */
function looksLikeSessionId(value: string): boolean {
  return /^[0-9a-fA-F]{8,32}$/.test(value)
}
export type DaemonConnectionState = 'connecting' | 'online' | 'disconnected' | 'needs_operator' | 'closed'
export interface DaemonHandshake {
  readonly initialized: Readonly<Record<string, unknown>>
  readonly status: Readonly<Record<string, unknown>>
}
export interface DaemonCallOptions { readonly signal?: AbortSignal; readonly timeoutMs?: number }
export interface HubDaemonClientOptions {
  readonly socketPath?: string
  /** Called afresh for every attempt, so SSH can recreate a dead forward. */
  readonly resolveSocketPath?: (signal: AbortSignal) => Promise<string>
  readonly projectDir: string
  readonly sessionKey?: string
  readonly requestTimeoutMs?: number
  readonly connectTimeoutMs?: number
  readonly retryBaseMs?: number
  readonly retryMaxMs?: number
  readonly random?: () => number
  readonly maxFrameBytes?: number
}
export class DaemonTransportError extends Error {
  constructor(message: string, readonly written: boolean, readonly reason: 'closed' | 'timeout' | 'aborted' | 'protocol' | 'unavailable') {
    super(message); this.name = 'DaemonTransportError'
  }
}
export class DaemonRpcError extends Error {
  constructor(message: string, readonly code: number) { super(message); this.name = 'DaemonRpcError' }
}
interface Pending {
  readonly resolve: (result: Record<string, unknown>) => void
  readonly reject: (error: Error) => void
  readonly cleanup: () => void
  written: boolean
}
interface Write { readonly id: number; readonly bytes: Uint8Array; offset: number }

/** Bun-only, connect-only v35 transport. Never launches or shuts down a daemon. */
export class HubDaemonClient {
  private socket: Bun.Socket<undefined> | undefined
  private opening: Promise<DaemonHandshake> | undefined
  private readonly lifetime = new AbortController()
  private readonly events = new Set<(type: string, payload: Record<string, unknown>) => void>()
  private readonly states = new Set<(state: DaemonConnectionState, error?: Error) => void>()
  private readonly handshakes = new Set<(handshake: DaemonHandshake) => void>()
  private readonly pending = new Map<number, Pending>()
  private readonly writes: Write[] = []
  private queuedBytes = 0
  private buffer = new Uint8Array(0)
  private sequence = 1
  private retry: ReturnType<typeof setTimeout> | undefined
  private retries = 0
  private stopped = false
  private operatorError: HubOperatorError | undefined
  private leaseToken: string | undefined
  private readonly sessionKey: string
  private currentState: DaemonConnectionState = 'disconnected'
  private handshake: DaemonHandshake | undefined

  constructor(private readonly options: HubDaemonClientOptions) {
    this.sessionKey = options.sessionKey ?? `hub:${crypto.randomUUID()}`
    if (!options.socketPath && !options.resolveSocketPath) throw new Error('A daemon socket or socket resolver is required')
  }
  get online(): boolean { return this.currentState === 'online' }
  get state(): DaemonConnectionState { return this.currentState }
  onEvent(listener: (type: string, payload: Record<string, unknown>) => void): () => void { this.events.add(listener); return () => this.events.delete(listener) }
  onState(listener: (state: DaemonConnectionState, error?: Error) => void): () => void { this.states.add(listener); return () => this.states.delete(listener) }
  onHandshake(listener: (handshake: DaemonHandshake) => void): () => void { this.handshakes.add(listener); return () => this.handshakes.delete(listener) }

  connect(): Promise<DaemonHandshake> {
    if (this.stopped) return Promise.reject(new DaemonTransportError('Daemon client is closed', false, 'closed'))
    if (this.operatorError) return Promise.reject(this.operatorError)
    if (this.opening) return this.opening
    if (this.online && this.handshake) return Promise.resolve(this.handshake)
    clearTimeout(this.retry); this.retry = undefined
    const attempt = this.open().catch((error: unknown) => {
      const failure = this.operatorError ?? (error instanceof Error ? error : new Error(String(error)))
      if (failure instanceof HubOperatorError) this.operatorError = failure
      this.disconnect(failure)
      if (!this.stopped) this.announce(this.operatorError ? 'needs_operator' : 'disconnected', failure)
      throw failure
    }).finally(() => { if (this.opening === attempt) this.opening = undefined; this.scheduleRetry() })
    this.opening = attempt
    return attempt
  }

  /** A call never reconnects/replays itself. Admission belongs to the pool. */
  call(method: string, params: Readonly<Record<string, unknown>> = {}, options: DaemonCallOptions = {}): Promise<Record<string, unknown>> {
    if (!this.online) return Promise.reject(new DaemonTransportError('Daemon is not online', false, 'unavailable'))
    if (isPrivateProviderType(method)) return Promise.reject(new DaemonRpcError('Provider relay is not available through the Hub', -32601))
    return this.send(method, params, options)
  }

  /** Force a transport retry (for example a dead SSH master), without cancelling daemon work. */
  interrupt(error: Error = new Error('Daemon transport interrupted')): void {
    if (this.stopped) return
    if (error instanceof HubOperatorError) this.operatorError = error
    this.disconnect(error)
    this.announce(this.operatorError ? 'needs_operator' : 'disconnected', this.operatorError ?? error)
    this.scheduleRetry()
  }
  /** Explicit operator retry after fixing authentication/host keys. Never retries a command. */
  reconnect(): Promise<DaemonHandshake> {
    this.operatorError = undefined
    return this.connect()
  }
  close(): void {
    if (this.stopped) return
    this.stopped = true; this.lifetime.abort(); clearTimeout(this.retry); this.retry = undefined
    this.disconnect(new Error('Daemon client closed')); this.announce('closed')
  }

  private announce(state: DaemonConnectionState, error?: Error): void {
    this.currentState = state
    for (const listener of this.states) listener(state, error)
  }
  private async open(): Promise<DaemonHandshake> {
    this.announce('connecting')
    const socketPath = this.options.resolveSocketPath ? await this.options.resolveSocketPath(this.lifetime.signal) : this.options.socketPath!
    this.lifetime.signal.throwIfAborted()
    const timeout = AbortSignal.timeout(this.options.connectTimeoutMs ?? 2000)
    const signal = AbortSignal.any([this.lifetime.signal, timeout])
    await new Promise<void>((resolve, reject) => {
      const abort = () => { this.disconnect(new Error('Daemon connection cancelled')); reject(new DaemonTransportError('Daemon socket connect timed out or was cancelled', false, 'timeout')) }
      signal.addEventListener('abort', abort, { once: true })
      void Bun.connect<undefined>({ unix: socketPath, socket: {
        open: socket => {
          if (signal.aborted || this.stopped) { socket.terminate(); return }
          this.socket = socket; this.buffer = new Uint8Array(0); resolve()
        },
        data: (socket, bytes) => { if (this.socket === socket) this.receive(bytes) },
        drain: socket => { if (this.socket === socket) this.flush() },
        close: socket => {
          if (this.socket !== socket) return
          this.disconnect(new Error('Daemon socket closed'))
          if (!this.stopped) { this.announce('disconnected'); this.scheduleRetry() }
        },
        error: (socket, error) => { if (this.socket === socket) this.interrupt(error) },
        connectError: (_socket, error) => reject(error),
      } }).catch(reject).finally(() => signal.removeEventListener('abort', abort))
      if (signal.aborted) abort()
    })
    if (this.leaseToken) {
      try {
        const reclaimed = await this.send('connection.lease', { token: this.leaseToken, project_dir: this.options.projectDir })
        if (reclaimed.ok !== true) this.leaseToken = undefined
      } catch (error) { if (error instanceof DaemonRpcError) this.leaseToken = undefined; else throw error }
    }
    // A per-task lane's key is often the conversation id itself (a resumed chat
    // is bound under its id). A bare initialize on that key makes a daemon evict
    // the chat and open an empty session under the same id, so the task's
    // history vanished from every client. Attach by id instead.
    const byId = looksLikeSessionId(this.sessionKey) ? { resume_session_id: this.sessionKey } : {}
    const initialized = await this.send('initialize', { session_key: this.sessionKey, ...byId, project_dir: this.options.projectDir, session_owned_turns: true, history_limit: 0 })
    if (initialized.ok !== true || initialized.daemon_protocol !== 35) throw new HubOperatorError('The remote socket is not a compatible Xerxes daemon (protocol v35 required). Inspect or update the runtime on its host.')
    if (initialized.connection_lease_supported === true && !this.leaseToken) {
      const lease = await this.send('connection.lease', {})
      if (lease.ok !== true || typeof lease.token !== 'string' || !lease.token) throw new Error('Could not establish daemon reconnect lease')
      this.leaseToken = lease.token
    }
    const status = await this.send('runtime.status', {})
    if (status.ok !== true || status.daemon_protocol !== 35) throw new HubOperatorError('The daemon did not return a valid runtime.status handshake. Inspect the runtime on its host.')
    this.lifetime.signal.throwIfAborted()
    if (!this.socket) throw new DaemonTransportError('Daemon disconnected during handshake', false, 'closed')
    const handshake = { initialized, status }
    this.handshake = handshake; this.retries = 0
    this.announce('online')
    for (const listener of this.handshakes) listener(handshake)
    return handshake
  }
  private scheduleRetry(): void {
    if (this.stopped || this.operatorError || this.online || this.retry || this.opening) return
    const base = Math.min((this.options.retryBaseMs ?? 250) * 2 ** Math.min(this.retries++, 16), this.options.retryMaxMs ?? 5000)
    const jitter = 0.75 + Math.max(0, Math.min(1, (this.options.random ?? Math.random)())) * 0.25
    this.retry = setTimeout(() => { this.retry = undefined; void this.connect().catch(() => { /* connect reports state and schedules the next bounded attempt */ }) }, base * jitter)
    this.retry.unref?.()
  }
  private send(method: string, params: Readonly<Record<string, unknown>>, options: DaemonCallOptions = {}): Promise<Record<string, unknown>> {
    if (options.signal?.aborted) return Promise.reject(new DaemonTransportError('Daemon request cancelled', false, 'aborted'))
    if (!this.socket) return Promise.reject(new DaemonTransportError('Daemon socket unavailable', false, 'unavailable'))
    const id = this.sequence++
    const bytes = new TextEncoder().encode(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
    if (bytes.length > DAEMON_MAX_FRAME_BYTES || this.queuedBytes + bytes.length > 2 * DAEMON_MAX_FRAME_BYTES || this.pending.size >= 256) return Promise.reject(new DaemonTransportError('Daemon request queue or frame limit exceeded', false, 'unavailable'))
    return new Promise((resolve, reject) => {
      const fail = (reason: 'timeout' | 'aborted') => {
        const pending = this.pending.get(id); if (!pending) return
        this.pending.delete(id); pending.cleanup()
        reject(new DaemonTransportError(reason === 'timeout' ? `Daemon request timed out: ${method}` : 'Daemon request cancelled', pending.written, reason))
        const unsent = this.writes.findIndex(write => write.id === id && write.offset === 0)
        if (unsent >= 0) { this.queuedBytes -= this.writes[unsent]!.bytes.length; this.writes.splice(unsent, 1) }
      }
      const abort = () => fail('aborted')
      const timer = setTimeout(() => fail('timeout'), options.timeoutMs ?? this.options.requestTimeoutMs ?? 30_000)
      this.pending.set(id, { resolve, reject, written: false, cleanup: () => { clearTimeout(timer); options.signal?.removeEventListener('abort', abort) } })
      options.signal?.addEventListener('abort', abort, { once: true })
      this.writes.push({ id, bytes, offset: 0 }); this.queuedBytes += bytes.length; this.flush()
    })
  }
  private flush(): void {
    const socket = this.socket; if (!socket) return
    while (this.writes.length) {
      const write = this.writes[0]!
      let written: number
      try { written = socket.write(write.bytes.subarray(write.offset)) } catch (error) { this.interrupt(error instanceof Error ? error : new Error(String(error))); return }
      if (written < 0) { this.interrupt(new Error('Daemon socket write failed')); return }
      if (written === 0) return
      const pending = this.pending.get(write.id); if (pending) pending.written = true
      write.offset += written; this.queuedBytes -= written
      if (write.offset < write.bytes.length) return
      this.writes.shift()
    }
  }
  private receive(bytes: Uint8Array): void {
    // Process chunks incrementally: the cap is per frame, not per read containing many frames.
    let start = 0
    for (let index = 0; index <= bytes.length; index++) {
      if (index !== bytes.length && bytes[index] !== 10) continue
      const part = bytes.subarray(start, index)
      if (this.buffer.length + part.length > Math.min(this.options.maxFrameBytes ?? DAEMON_MAX_FRAME_BYTES, DAEMON_MAX_FRAME_BYTES)) { this.interrupt(new DaemonTransportError('Daemon frame exceeds inbound frame limit (at most 16 MiB)', true, 'protocol')); return }
      const joined = new Uint8Array(this.buffer.length + part.length); joined.set(this.buffer); joined.set(part, this.buffer.length)
      this.buffer = joined
      if (index < bytes.length) {
        const line = new TextDecoder().decode(this.buffer); this.buffer = new Uint8Array(0)
        if (line.trim()) this.frame(line)
        if (!this.socket) return
      }
      start = index + 1
    }
  }
  private frame(line: string): void {
    let frame: unknown
    try { frame = JSON.parse(line) as unknown } catch { this.interrupt(new DaemonTransportError('Malformed daemon JSON', true, 'protocol')); return }
    if (!isRecord(frame) || frame.jsonrpc !== '2.0') { this.interrupt(new DaemonTransportError('Invalid daemon JSON-RPC frame', true, 'protocol')); return }
    // Check method before id correlation: a private request may itself have an id.
    if (typeof frame.method === 'string' && isPrivateProviderType(frame.method)) return
    if (frame.method === 'event' && isRecord(frame.params)) {
      const type = frame.params.type
      if (typeof type !== 'string' || isPrivateProviderType(type)) return
      const payload = isRecord(frame.params.payload) ? frame.params.payload : {}
      for (const listener of this.events) listener(type, payload)
      return
    }
    const id = typeof frame.id === 'number' ? frame.id : typeof frame.id === 'string' && /^\d+$/.test(frame.id) ? Number(frame.id) : undefined
    if (id === undefined) return
    const pending = this.pending.get(id); if (!pending) return
    this.pending.delete(id); pending.cleanup()
    if (isRecord(frame.error)) pending.reject(new DaemonRpcError(typeof frame.error.message === 'string' ? frame.error.message : 'Daemon RPC rejected', typeof frame.error.code === 'number' ? frame.error.code : -32000))
    else if (isRecord(frame.result)) pending.resolve(frame.result)
    else pending.reject(new DaemonTransportError('Daemon response result is not an object', pending.written, 'protocol'))
  }
  private disconnect(error: Error): void {
    const socket = this.socket; this.socket = undefined; socket?.terminate()
    this.buffer = new Uint8Array(0); this.writes.length = 0; this.queuedBytes = 0
    for (const pending of this.pending.values()) { pending.cleanup(); pending.reject(new DaemonTransportError(error.message, pending.written, 'closed')) }
    this.pending.clear()
  }
}
