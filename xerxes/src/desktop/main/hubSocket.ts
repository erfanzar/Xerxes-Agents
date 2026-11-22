// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * A small RFC 6455 WebSocket client for the Electron main process.
 *
 * The Hub accepts a socket only when its `Origin` is the Hub's own origin and
 * the session cookie rides along. A browser renderer can set neither, and
 * Node's global `WebSocket` offers no portable way to send `Origin`, so the
 * main process opens the upgrade itself over `node:http(s)` and speaks the
 * framing here. Text frames only (the Hub protocol is JSON text); a binary
 * frame is reported, never delivered.
 */

import { createHash, randomBytes } from 'node:crypto'
import http from 'node:http'
import https from 'node:https'
import type { Duplex } from 'node:stream'

const GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
const EMPTY = Buffer.alloc(0)

export interface HubSocketHandlers {
  open(): void
  message(text: string): void
  /** A frame that was not text. The payload is dropped. */
  binary(): void
  error(message: string): void
  close(code: number, reason: string): void
}

export interface HubSocket {
  send(text: string): void
  close(code?: number, reason?: string): void
}

export interface HubSocketTarget {
  /** `ws://` or `wss://` URL. */
  readonly url: string
  /** Sent as `Origin`. */
  readonly origin: string
  /** Sent as `Cookie`. */
  readonly cookie: string
}

export type HubSocketConnector = (target: HubSocketTarget, handlers: HubSocketHandlers) => HubSocket

export interface RawFrame {
  readonly fin: boolean
  readonly opcode: number
  readonly payload: Buffer
}

/** One masked client frame. */
export function encodeClientFrame(opcode: number, payload: Uint8Array, mask: Buffer = randomBytes(4)): Buffer {
  const length = payload.length
  const head = length < 126 ? 2 : length < 65_536 ? 4 : 10
  const out = Buffer.alloc(head + 4 + length)
  out[0] = 0x80 | opcode
  if (length < 126) {
    out[1] = 0x80 | length
  } else if (length < 65_536) {
    out[1] = 0x80 | 126
    out.writeUInt16BE(length, 2)
  } else {
    out[1] = 0x80 | 127
    out.writeBigUInt64BE(BigInt(length), 2)
  }
  mask.copy(out, head)
  for (let index = 0; index < length; index += 1) out[head + 4 + index] = payload[index]! ^ mask[index & 3]!
  return out
}

/** Incremental decoder for unmasked server frames. Throws on a protocol violation. */
export class FrameDecoder {
  private buffer: Buffer = EMPTY
  private readonly maxPayload: number

  constructor(maxPayload: number) {
    this.maxPayload = maxPayload
  }

  push(chunk: Uint8Array): RawFrame[] {
    this.buffer = this.buffer.length === 0 ? Buffer.from(chunk) : Buffer.concat([this.buffer, chunk])
    const frames: RawFrame[] = []
    for (;;) {
      const bytes = this.buffer
      if (bytes.length < 2) break
      const first = bytes[0]!
      const second = bytes[1]!
      if ((first & 0x70) !== 0) throw new Error('reserved bits are set')
      if ((second & 0x80) !== 0) throw new Error('a server frame must not be masked')
      let length = second & 0x7f
      let offset = 2
      if (length === 126) {
        if (bytes.length < 4) break
        length = bytes.readUInt16BE(2)
        offset = 4
      } else if (length === 127) {
        if (bytes.length < 10) break
        const big = bytes.readBigUInt64BE(2)
        if (big > BigInt(this.maxPayload)) throw new Error('frame is too large')
        length = Number(big)
        offset = 10
      }
      if (length > this.maxPayload) throw new Error('frame is too large')
      const opcode = first & 0x0f
      const fin = (first & 0x80) !== 0
      if (opcode >= 8 && (length > 125 || !fin)) throw new Error('invalid control frame')
      if (bytes.length < offset + length) break
      frames.push({ fin, opcode, payload: Buffer.from(bytes.subarray(offset, offset + length)) })
      const nextOffset = offset + length
      this.buffer = nextOffset >= bytes.length ? EMPTY : bytes.subarray(nextOffset)
    }
    return frames
  }
}

/** Build a valid RFC 6455 close payload, truncating at a UTF-8 code-point boundary within 123 bytes. */
export function closePayload(code: number, reason: string): Buffer {
  const full = Buffer.from(reason, 'utf8')
  let end = Math.min(full.length, 123)
  while (end > 0 && end < full.length && (full[end]! & 0xc0) === 0x80) end -= 1
  const out = Buffer.alloc(2 + end)
  out.writeUInt16BE(code, 0)
  full.copy(out, 2, 0, end)
  return out
}

export interface WrapOptions {
  /** Largest text message accepted, in bytes. */
  readonly maxMessageBytes: number
  /** Bytes the HTTP layer read past the upgrade response. */
  readonly head?: Uint8Array
}

/** Frame an already-upgraded duplex stream. */
export function wrapHubSocket(socket: Duplex, handlers: HubSocketHandlers, options: WrapOptions): HubSocket {
  const decoder = new FrameDecoder(options.maxMessageBytes)
  const textDecoder = new TextDecoder('utf-8', { fatal: true })
  let fragments: Buffer[] = []
  let fragmentBytes = 0
  let fragmentMode: 0 | 1 | 2 = 0
  let finished = false
  let closing = false

  const finish = (code: number, reason: string): void => {
    if (finished) return
    finished = true
    handlers.close(code, reason)
  }
  const write = (opcode: number, payload: Uint8Array): void => {
    if (socket.destroyed || !socket.writable) return
    socket.write(encodeClientFrame(opcode, payload))
  }
  const fail = (code: number, reason: string): void => {
    if (!closing) {
      closing = true
      write(0x8, closePayload(code, reason))
    }
    finish(code, reason)
    socket.destroy()
  }

  const onFrame = (frame: RawFrame): void => {
    switch (frame.opcode) {
      case 0x1: {
        if (fragmentMode !== 0) { fail(1002, 'unexpected text frame'); return }
        if (frame.payload.length > options.maxMessageBytes) { fail(1009, 'message too large'); return }
        if (!frame.fin) {
          fragmentMode = 1
          fragmentBytes = frame.payload.length
          fragments = [frame.payload]
          return
        }
        let text: string
        try {
          text = textDecoder.decode(frame.payload)
        } catch {
          fail(1007, 'invalid text')
          return
        }
        handlers.message(text)
        return
      }
      case 0x2: {
        if (fragmentMode !== 0) { fail(1002, 'unexpected binary frame'); return }
        if (frame.payload.length > options.maxMessageBytes) { fail(1009, 'message too large'); return }
        if (!frame.fin) {
          fragmentMode = 2
          fragmentBytes = frame.payload.length
          return
        }
        handlers.binary()
        return
      }
      case 0x0: {
        if (fragmentMode === 0) { fail(1002, 'unexpected continuation'); return }
        fragmentBytes += frame.payload.length
        if (fragmentBytes > options.maxMessageBytes) { fail(1009, 'message too large'); return }
        if (fragmentMode === 2) {
          if (frame.fin) {
            fragmentMode = 0
            fragmentBytes = 0
            handlers.binary()
          }
          return
        }
        fragments.push(frame.payload)
        if (!frame.fin) return
        const whole = fragments.length === 1 ? fragments[0]! : Buffer.concat(fragments)
        fragments = []
        fragmentBytes = 0
        fragmentMode = 0
        let text: string
        try {
          text = textDecoder.decode(whole)
        } catch {
          fail(1007, 'invalid text')
          return
        }
        handlers.message(text)
        return
      }
      case 0x8: {
        if (frame.payload.length === 1) { fail(1002, 'invalid close frame'); return }
        const code = frame.payload.length >= 2 ? frame.payload.readUInt16BE(0) : 1005
        let reason = ''
        if (frame.payload.length > 2) {
          try {
            reason = textDecoder.decode(frame.payload.subarray(2))
          } catch {
            fail(1007, 'invalid close reason')
            return
          }
        }
        if (!closing) {
          closing = true
          write(0x8, frame.payload.length >= 2 ? frame.payload.subarray(0, 2) : EMPTY)
        }
        finish(code, reason)
        socket.end()
        return
      }
      case 0x9:
        write(0xa, frame.payload)
        return
      case 0xa:
        return
      default:
        fail(1002, 'unknown opcode')
    }
  }

  const feed = (chunk: Uint8Array): void => {
    if (finished) return
    let frames: RawFrame[]
    try {
      frames = decoder.push(chunk)
    } catch (error) {
      fail(1002, error instanceof Error ? error.message : 'protocol error')
      return
    }
    for (const frame of frames) {
      if (finished) return
      onFrame(frame)
    }
  }

  socket.on('data', (chunk: Buffer) => feed(chunk))
  socket.on('error', error => { handlers.error(error.message) })
  socket.on('close', () => finish(1006, 'connection lost'))
  if (options.head !== undefined && options.head.length > 0) queueMicrotask(() => feed(options.head!))

  return {
    send(text) {
      if (finished || closing) throw new Error('The Hub link is closed.')
      write(0x1, Buffer.from(text, 'utf8'))
    },
    close(code = 1000, reason = '') {
      if (closing || finished) return
      closing = true
      write(0x8, closePayload(code, reason))
      const timer = setTimeout(() => { finish(code, reason); socket.destroy() }, 1000)
      timer.unref?.()
      socket.once('close', () => clearTimeout(timer))
    },
  }
}

export interface ConnectOptions {
  readonly maxMessageBytes?: number
  readonly connectTimeoutMs?: number
}

/** Open the Hub socket: HTTP upgrade with `Origin` and `Cookie`, then framing. */
export const connectHubSocket = (target: HubSocketTarget, handlers: HubSocketHandlers, options: ConnectOptions = {}): HubSocket => {
  const url = new URL(target.url)
  if (url.protocol !== 'ws:' && url.protocol !== 'wss:') throw new Error('The Hub socket must be ws:// or wss://')
  const secure = url.protocol === 'wss:'
  const key = randomBytes(16).toString('base64')
  const expected = createHash('sha1').update(key + GUID).digest('base64')
  let inner: HubSocket | undefined
  let ended = false
  const end = (code: number, reason: string): void => {
    if (ended) return
    ended = true
    handlers.close(code, reason)
  }
  const failBeforeOpen = (message: string): void => {
    if (ended) return
    handlers.error(message)
    end(1006, message)
  }
  const request = (secure ? https : http).request(`${secure ? 'https' : 'http'}://${url.host}${url.pathname}${url.search}`, {
    agent: false,
    timeout: options.connectTimeoutMs ?? 15_000,
    headers: {
      Connection: 'Upgrade',
      Upgrade: 'websocket',
      'Sec-WebSocket-Version': '13',
      'Sec-WebSocket-Key': key,
      Origin: target.origin,
      Cookie: target.cookie,
    },
  })
  request.on('upgrade', (response, socket, head) => {
    if (ended) {
      socket.destroy()
      return
    }
    if (response.statusCode !== 101 || response.headers['sec-websocket-accept'] !== expected) {
      socket.destroy()
      failBeforeOpen('The Hub did not accept the connection.')
      return
    }
    socket.setTimeout(0)
    inner = wrapHubSocket(socket, {
      open: () => handlers.open(),
      message: text => handlers.message(text),
      binary: () => handlers.binary(),
      error: message => handlers.error(message),
      close: (code, reason) => end(code, reason),
    }, { maxMessageBytes: options.maxMessageBytes ?? 1024 * 1024, head })
    handlers.open()
  })
  request.on('response', response => {
    response.resume()
    failBeforeOpen(response.statusCode === 401 ? 'The Hub did not accept this sign-in.' : response.statusCode === 403 ? 'The Hub refused this connection.' : `The Hub answered ${response.statusCode ?? 'with an error'}.`)
  })
  request.on('timeout', () => { request.destroy(); failBeforeOpen('The Hub did not answer in time.') })
  request.on('error', error => failBeforeOpen(error.message))
  request.end()
  return {
    send(text) {
      if (inner === undefined) throw new Error('The Hub link is not open yet.')
      inner.send(text)
    },
    close(code, reason) {
      if (inner !== undefined) { inner.close(code, reason); return }
      end(code ?? 1000, reason ?? '')
      request.destroy()
    },
  }
}
