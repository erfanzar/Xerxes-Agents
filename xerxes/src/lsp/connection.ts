// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { planSpawn } from '../core/windowsSpawn.js'
import { encodeLspMessage, LspMessageDecoder } from './framing.js'

interface Pending { resolve(value: unknown): void; reject(error: Error): void; cleanup(): void }
export interface LspConnectionOptions {
  readonly command: string
  readonly args?: readonly string[]
  readonly cwd: string
  readonly env?: Record<string, string>
  readonly timeoutMs?: number
  readonly onNotification?: (method: string, params: unknown) => void
}
const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value)

/** Owned stdio transport. Initialization and document semantics belong to the host.
 * Stderr is drained but never echoed, since servers may print launch credentials.
 */
export class LspConnection {
  private readonly child: Bun.PipedSubprocess
  private readonly stdout: ReadableStreamDefaultReader<Uint8Array>
  private readonly stderr: ReadableStreamDefaultReader<Uint8Array>
  private readonly pending = new Map<number, Pending>()
  private nextId = 0
  private closed: Error | undefined
  private writes: Promise<void> = Promise.resolve()
  private queuedBytes = 0
  private queuedMessages = 0
  private termination: Promise<void> | undefined
  private closing: Promise<void> | undefined
  private readonly timeoutMs: number
  constructor(private readonly options: LspConnectionOptions) {
    this.timeoutMs = options.timeoutMs ?? 30_000
    if (!Number.isSafeInteger(this.timeoutMs) || this.timeoutMs < 1 || this.timeoutMs > 120_000) throw new Error('LSP timeout must be 1–120000 milliseconds')
    const plan = planSpawn(options.command, options.args ?? [])
    try {
      this.child = Bun.spawn(plan.argv, { cwd: options.cwd, env: { ...process.env, ...options.env }, stdin: 'pipe', stdout: 'pipe', stderr: 'pipe',
        windowsHide: true,
        ...(plan.windowsVerbatimArguments ? { windowsVerbatimArguments: true } : {}) })
    } catch { throw new Error('Cannot launch configured language server; verify its executable, arguments and workspace directory') }
    this.stdout = this.child.stdout.getReader(); this.stderr = this.child.stderr.getReader()
    void this.read()
    void this.drainErrors()
    void this.child.exited.then(code => this.fail(new Error(`Language server exited (${code})`)), () => this.fail(new Error('Language server process failed')))
  }

  get pid(): number { return this.child.pid }
  get connected(): boolean { return !this.closed }

  request(method: string, params: unknown = null, signal?: AbortSignal, timeoutMs = this.timeoutMs): Promise<unknown> {
    if (this.closed) return Promise.reject(this.closed)
    if (this.closing && method !== 'shutdown') return Promise.reject(new Error('Language server connection is closing'))
    if (signal?.aborted) return Promise.reject(new DOMException('LSP request cancelled', 'AbortError'))
    if (!method || method.length > 256 || !Number.isSafeInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 120_000) return Promise.reject(new Error('Invalid LSP request options'))
    if (this.pending.size >= 128) return Promise.reject(new Error('Language server request limit reached'))
    const id = ++this.nextId
    return new Promise((resolve, reject) => {
      const cancel = (error: Error) => {
        if (!this.settle(id, undefined, error)) return
        void this.notify('$/cancelRequest', { id }).catch(() => { /* A failed transport already rejects pending calls. */ })
      }
      const abort = () => cancel(new DOMException('LSP request cancelled', 'AbortError'))
      const timer = setTimeout(() => cancel(new Error('Language server request timed out')), timeoutMs)
      const cleanup = () => { clearTimeout(timer); signal?.removeEventListener('abort', abort) }
      this.pending.set(id, { resolve, reject, cleanup })
      signal?.addEventListener('abort', abort, { once: true })
      if (signal?.aborted) { abort(); return }
      void this.send({ jsonrpc: '2.0', id, method, params }, () => this.pending.has(id)).catch(error => this.settle(id, undefined, error instanceof Error ? error : new Error('LSP write failed')))
    })
  }

  notify(method: string, params: unknown = null): Promise<void> {
    if (!method || method.length > 256) return Promise.reject(new Error('Invalid LSP notification method'))
    if (this.closing && method !== 'exit') return Promise.reject(new Error('Language server connection is closing'))
    return this.send({ jsonrpc: '2.0', method, params })
  }

  close(): Promise<void> {
    return this.closing ??= (async () => {
      if (!this.closed) {
        try { await Promise.race([(async () => { await this.request('shutdown', null, undefined, 500); await this.notify('exit') })(), Bun.sleep(500)]) }
        catch { /* Graceful shutdown failed; bounded process termination follows. */ }
      }
      this.fail(new Error('Language server connection closed'))
      await this.terminate()
      if (this.child.exitCode === null && this.child.signalCode === null) throw new Error(`Language server process ${this.child.pid} did not exit after termination`)
    })()
  }

  private send(message: unknown, stillNeeded: () => boolean = () => true): Promise<void> {
    if (this.closed) return Promise.reject(this.closed)
    let frame: Uint8Array
    try { frame = encodeLspMessage(message) } catch { return Promise.reject(new Error('Invalid or oversized outbound LSP message')) }
    if (this.queuedMessages >= 128 || this.queuedBytes + frame.length > 8 * 1024 * 1024) return Promise.reject(new Error('Language server output queue is full'))
    this.queuedBytes += frame.length
    this.queuedMessages++
    const writing = this.writes.then(async () => {
      if (!stillNeeded()) return
      if (this.closed) throw this.closed
      try { this.child.stdin.write(frame); await this.child.stdin.flush() }
      catch { this.fail(new Error('Language server stdin failed')); throw this.closed }
    }).finally(() => { this.queuedBytes -= frame.length; this.queuedMessages-- })
    this.writes = writing.catch(() => {})
    return writing
  }

  private async read(): Promise<void> {
    const decoder = new LspMessageDecoder(message => this.receive(message))
    try {
      while (!this.closed) {
        const chunk = await this.stdout.read()
        if (chunk.done) { decoder.end(); this.fail(new Error('Language server stdout closed')); return }
        decoder.push(chunk.value)
      }
    } catch { this.fail(new Error('Language server sent malformed protocol data or closed its output')) }
  }
  private async drainErrors(): Promise<void> {
    try { while (!(await this.stderr.read()).done) { /* Drain without retaining server diagnostics/credentials. */ } }
    catch { if (!this.closed) this.fail(new Error('Language server stderr failed')) }
  }
  private receive(message: unknown): void {
    if (!record(message) || message.jsonrpc !== '2.0') throw new Error('Invalid JSON-RPC envelope')
    if (typeof message.method === 'string') {
      if ('result' in message || 'error' in message) throw new Error('Invalid JSON-RPC request')
      if ('id' in message) {
        if (typeof message.id === 'string' ? message.id.length > 128 : !Number.isSafeInteger(message.id)) throw new Error('Invalid server request ID')
        void this.send({ jsonrpc: '2.0', id: message.id, error: { code: -32601, message: 'Client method not supported' } }).catch(() => this.fail(new Error('Cannot respond to language server request')))
      } else this.options.onNotification?.(message.method, message.params)
      return
    }
    if (!Number.isSafeInteger(message.id) || (('result' in message) === ('error' in message))) throw new Error('Invalid JSON-RPC response')
    if ('error' in message) {
      if (!record(message.error) || !Number.isInteger(message.error.code) || typeof message.error.message !== 'string') throw new Error('Invalid JSON-RPC error')
      this.settle(message.id as number, undefined, new Error(`Language server request failed (${message.error.code})`))
    } else this.settle(message.id as number, message.result)
  }
  private settle(id: number, result?: unknown, error?: Error): boolean {
    const pending = this.pending.get(id)
    if (!pending) return false // Late response after cancellation or timeout.
    this.pending.delete(id); pending.cleanup()
    if (error) pending.reject(error); else pending.resolve(result)
    return true
  }
  private fail(error: Error): void {
    if (this.closed) return
    this.closed = error
    for (const id of this.pending.keys()) this.settle(id, undefined, error)
    void this.terminate()
  }
  private terminate(): Promise<void> {
    return this.termination ??= (async () => {
      if (this.child.exitCode === null && this.child.signalCode === null) {
        try { this.child.kill('SIGTERM') } catch { /* Process may have exited concurrently. */ }
        await Promise.race([this.child.exited, Bun.sleep(250)])
      }
      if (this.child.exitCode === null && this.child.signalCode === null) {
        try { this.child.kill('SIGKILL') } catch { /* Process may have exited concurrently. */ }
        await Promise.race([this.child.exited, Bun.sleep(250)])
      }
      await Promise.race([Promise.allSettled([this.stdout.cancel(), this.stderr.cancel()]), Bun.sleep(250)])
    })()
  }
}
