// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  VOICE_LIMITS,
  decodeVoiceFrame,
  encodeVoiceFrame,
  parseVoiceServerMessage,
  type VoiceClientMessage,
  type VoiceFrame,
  type VoiceFrameHeader,
  type VoiceServerMessage,
} from '../hub/voice/protocol.js'

/**
 * One WebSocket to the Hub voice endpoint. It frames and parses; it never
 * reconnects by itself and never resends anything: a dropped socket ends the
 * voice session, and the next press opens a new one (so audio is never replayed).
 */

export interface VoiceSocketHandlers {
  open(): void
  text(data: string): void
  binary(data: Uint8Array): void
  close(code: number, reason: string): void
  error(): void
}

export interface VoiceSocketLike {
  send(data: string | Uint8Array): void
  close(code?: number, reason?: string): void
  bufferedAmount(): number
}

export type VoiceSocketFactory = (url: string, handlers: VoiceSocketHandlers) => VoiceSocketLike

export type VoiceLinkEvent =
  | { readonly kind: 'control'; readonly message: VoiceServerMessage }
  | { readonly kind: 'frame'; readonly frame: VoiceFrame }
  | { readonly kind: 'closed'; readonly code: number; readonly reason: string }

export function hubVoiceUrl(location: { readonly protocol: string; readonly host: string }): string {
  return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/hub/v1/voice`
}

export const browserVoiceSocketFactory: VoiceSocketFactory = (url, handlers) => {
  const socket = new WebSocket(url)
  socket.binaryType = 'arraybuffer'
  socket.addEventListener('open', () => handlers.open())
  socket.addEventListener('message', event => {
    const data: unknown = event.data
    if (typeof data === 'string') handlers.text(data)
    else if (data instanceof ArrayBuffer) handlers.binary(new Uint8Array(data))
  })
  socket.addEventListener('close', event => handlers.close(event.code, event.reason))
  socket.addEventListener('error', () => handlers.error())
  return {
    send: data => { socket.send(data as string | ArrayBufferView<ArrayBuffer>) },
    close: (code, reason) => socket.close(code, reason),
    bufferedAmount: () => socket.bufferedAmount,
  }
}

export interface VoiceLinkOptions {
  readonly url: string
  readonly socketFactory?: VoiceSocketFactory
}

export class VoiceLink {
  private socket: VoiceSocketLike | undefined
  private opening: Promise<void> | undefined
  private isOpen = false
  private isClosed = false
  private readonly listeners = new Set<(event: VoiceLinkEvent) => void>()

  constructor(private readonly options: VoiceLinkOptions) {}

  get open(): boolean { return this.isOpen && !this.isClosed }
  get closed(): boolean { return this.isClosed }

  /** Open the socket once. Rejects when it closes or errors before opening. */
  connect(): Promise<void> {
    if (this.isClosed) return Promise.reject(new Error('The voice connection is closed.'))
    if (this.opening !== undefined) return this.opening
    this.opening = new Promise<void>((resolve, reject) => {
      const factory = this.options.socketFactory ?? browserVoiceSocketFactory
      try {
        this.socket = factory(this.options.url, {
          open: () => { this.isOpen = true; resolve() },
          text: data => this.onText(data),
          binary: data => this.onBinary(data),
          close: (code, reason) => this.onClose(code, reason, reject),
          error: () => { if (!this.isOpen) this.onClose(1006, 'connection failed', reject) },
        })
      } catch (error) {
        this.isClosed = true
        reject(error instanceof Error ? error : new Error('The voice connection could not be opened.'))
      }
    })
    return this.opening
  }

  onEvent(listener: (event: VoiceLinkEvent) => void): () => void {
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener) }
  }

  sendControl(message: VoiceClientMessage): boolean {
    if (!this.open || this.socket === undefined) return false
    this.socket.send(JSON.stringify({ v: 1, ...message }))
    return true
  }

  /** Send one microphone frame. False when the link is down or backed up; the caller ends the recording. */
  sendFrame(header: VoiceFrameHeader, payload: Uint8Array): boolean {
    if (!this.open || this.socket === undefined) return false
    if (this.socket.bufferedAmount() + payload.byteLength > VOICE_LIMITS.maxBufferedBytes) return false
    this.socket.send(encodeVoiceFrame(header, payload))
    return true
  }

  close(code = 1000, reason = 'done'): void {
    if (this.isClosed) return
    this.isClosed = true
    this.isOpen = false
    try { this.socket?.close(code, reason) } catch { /* already closed */ }
    this.emit({ kind: 'closed', code, reason })
  }

  private onText(data: string): void {
    const message = parseVoiceServerMessage(data)
    if (message !== undefined) this.emit({ kind: 'control', message })
  }

  private onBinary(data: Uint8Array): void {
    try {
      this.emit({ kind: 'frame', frame: decodeVoiceFrame(data) })
    } catch {
      // A malformed frame is dropped; the Hub is the authority on stream validity.
    }
  }

  private onClose(code: number, reason: string, reject: (error: Error) => void): void {
    if (this.isClosed) return
    const wasOpen = this.isOpen
    this.isClosed = true
    this.isOpen = false
    if (!wasOpen) reject(new Error('The voice connection could not be opened.'))
    this.emit({ kind: 'closed', code, reason })
  }

  private emit(event: VoiceLinkEvent): void {
    for (const listener of [...this.listeners]) listener(event)
  }
}
