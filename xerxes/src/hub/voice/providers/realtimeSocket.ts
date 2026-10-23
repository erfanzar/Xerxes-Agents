// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { VoiceError } from '../protocol.js'
import type { RealtimeEvent } from '../ports.js'

/**
 * Provider WebSocket seam. The Hub injects a factory so tests use fake
 * sockets and the keys in `headers` never leave the Hub process.
 */
export interface RealtimeSocketHandlers {
  open(): void
  message(data: string): void
  close(code: number, reason: string): void
  error(error: unknown): void
}

export interface RealtimeSocket {
  /** False when the socket is closed or refused the frame. */
  send(data: string): boolean
  bufferedAmount(): number
  close(code?: number, reason?: string): void
}

export interface RealtimeSocketRequest {
  readonly url: string
  readonly headers: Readonly<Record<string, string>>
}

export type RealtimeSocketFactory = (request: RealtimeSocketRequest, handlers: RealtimeSocketHandlers) => RealtimeSocket

/** Bun's WebSocket accepts request headers; the DOM lib typing does not declare them. */
const BunWebSocket = WebSocket as unknown as new (url: string, options: { readonly headers: Record<string, string> }) => WebSocket

/** Production factory over Bun's WebSocket (which accepts request headers). */
export const bunRealtimeSocketFactory: RealtimeSocketFactory = (request, handlers) => {
  const socket = new BunWebSocket(request.url, { headers: { ...request.headers } })
  socket.addEventListener('open', () => handlers.open())
  socket.addEventListener('message', event => {
    const data: unknown = event.data
    if (typeof data === 'string') handlers.message(data)
    // Some providers (Gemini Live) send JSON in binary frames.
    else if (data instanceof ArrayBuffer) handlers.message(new TextDecoder().decode(data))
    else if (data instanceof Blob) void data.text().then(handlers.message, () => undefined)
  })
  socket.addEventListener('close', event => handlers.close(event.code, event.reason))
  socket.addEventListener('error', event => handlers.error(event))
  return {
    send(data) {
      if (socket.readyState !== WebSocket.OPEN) return false
      socket.send(data)
      return true
    },
    bufferedAmount: () => socket.bufferedAmount,
    close: (code, reason) => socket.close(code, reason),
  }
}

/** Fan-out with a small buffer so events that arrive before the first listener are not lost. */
export class RealtimeEmitter {
  private readonly listeners = new Set<(event: RealtimeEvent) => void>()
  private backlog: RealtimeEvent[] | undefined = []

  on(listener: (event: RealtimeEvent) => void): () => void {
    this.listeners.add(listener)
    const pending = this.backlog
    this.backlog = undefined
    if (pending !== undefined) for (const event of pending) listener(event)
    return () => { this.listeners.delete(listener) }
  }

  emit(event: RealtimeEvent): void {
    if (this.backlog !== undefined) {
      if (this.backlog.length < 64) this.backlog.push(event)
      return
    }
    for (const listener of [...this.listeners]) listener(event)
  }
}

export interface ConnectedSocket {
  readonly socket: RealtimeSocket
  readonly emitter: RealtimeEmitter
  /** Resolves once the socket is open; rejects with a typed VoiceError. */
  readonly opened: Promise<void>
}

/**
 * Open a provider socket and route its frames to `onMessage`. Rejection is
 * typed: cancelled, timed out, or refused. Provider close reasons are not
 * forwarded to callers.
 */
export function openRealtimeSocket(
  factory: RealtimeSocketFactory,
  request: RealtimeSocketRequest,
  signal: AbortSignal,
  connectTimeoutMs: number,
  onMessage: (data: string, emitter: RealtimeEmitter) => void,
): ConnectedSocket {
  const emitter = new RealtimeEmitter()
  let settle: { resolve: () => void; reject: (error: VoiceError) => void } | undefined
  const opened = new Promise<void>((resolve, reject) => { settle = { resolve, reject } })
  let isOpen = false
  const timer = setTimeout(() => fail(new VoiceError('VOICE_PROVIDER', 'the realtime provider did not answer in time')), connectTimeoutMs)
  const fail = (error: VoiceError): void => {
    clearTimeout(timer)
    if (!isOpen) settle?.reject(error)
    try { socket.close(1000, 'connect failed') } catch { /* already closed */ }
  }
  const socket = factory(request, {
    open() { isOpen = true; clearTimeout(timer); settle?.resolve() },
    message(data) { onMessage(data, emitter) },
    close(_code, reason) {
      clearTimeout(timer)
      if (!isOpen) settle?.reject(new VoiceError('VOICE_PROVIDER', 'the realtime provider refused the connection'))
      else emitter.emit({ type: 'closed', reason: reason === '' ? 'provider closed' : 'provider closed the session' })
    },
    error() {
      if (!isOpen) fail(new VoiceError('VOICE_PROVIDER', 'the realtime provider could not be reached'))
      else emitter.emit({ type: 'error', error: new VoiceError('VOICE_PROVIDER', 'the realtime provider connection failed') })
    },
  })
  const onAbort = (): void => fail(new VoiceError('VOICE_CANCELLED', 'the realtime connection was cancelled'))
  if (signal.aborted) onAbort()
  else signal.addEventListener('abort', onAbort, { once: true })
  void opened.then(() => signal.removeEventListener('abort', onAbort), () => signal.removeEventListener('abort', onAbort))
  return { socket, emitter, opened }
}

export function decodeBase64Audio(value: unknown, maxBytes = 1024 * 1024): Uint8Array | undefined {
  if (typeof value !== 'string' || value === '' || value.length > Math.ceil((maxBytes * 4) / 3) + 4) return undefined
  try {
    return new Uint8Array(Buffer.from(value, 'base64'))
  } catch {
    return undefined
  }
}

export function encodeBase64Audio(bytes: Uint8Array): string {
  return Buffer.from(bytes).toString('base64')
}

export function parseJsonObject(data: string): Record<string, unknown> | undefined {
  try {
    const value: unknown = JSON.parse(data)
    return typeof value === 'object' && value !== null && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined
  } catch {
    return undefined
  }
}
