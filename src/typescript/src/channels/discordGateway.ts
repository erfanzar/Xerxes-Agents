// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ChannelHttpError, providerUrl, type ChannelFetch } from './http.js'

const DISCORD_GATEWAY_VERSION = '10'
const DEFAULT_RECONNECT_DELAY = 1_000
const MAX_RECONNECT_DELAY = 30_000
const INVALID_SESSION_MIN_DELAY = 1_000
const INVALID_SESSION_DELAY_SPREAD = 4_000
const FATAL_CLOSE_CODES = new Set([4004, 4010, 4011, 4012, 4013, 4014])

/** Intents required to receive guild/direct messages without privileged content access. */
export const DISCORD_GATEWAY_BASE_INTENTS = (1 << 0) | (1 << 9) | (1 << 12)
/** Discord's privileged message-content intent. */
export const DISCORD_GATEWAY_MESSAGE_CONTENT_INTENT = 1 << 15

export interface DiscordGatewayBot {
  /** Gateway endpoint returned by Discord's authenticated `GET /gateway/bot` API. */
  readonly url: string
}

/** Explicit REST boundary used only to discover a Gateway endpoint. */
export interface DiscordGatewayRestPort {
  gatewayBot(botToken: string): Promise<DiscordGatewayBot>
}

/** WebSocket operations the Gateway state machine needs after a connection opens. */
export interface DiscordGatewaySocket {
  close(code?: number, reason?: string): void | Promise<void>
  send(payload: string): void | Promise<void>
}

export interface DiscordGatewayCloseEvent {
  readonly code: number
  readonly reason: string
}

export interface DiscordGatewayConnectionHandlers {
  onClose(socket: DiscordGatewaySocket, event: DiscordGatewayCloseEvent): void
  onError(socket: DiscordGatewaySocket, error: unknown): void
  onMessage(socket: DiscordGatewaySocket, payload: string): void
  onOpen(socket: DiscordGatewaySocket): void
}

/** Explicit WebSocket boundary for live Discord Gateway frames. */
export interface DiscordGatewayWebSocketPort {
  connect(url: string, handlers: DiscordGatewayConnectionHandlers): Promise<DiscordGatewaySocket>
}

/** Injectable clock used for heartbeats and reconnect scheduling. */
export interface DiscordGatewayClock {
  clearTimeout(timer: unknown): void
  setTimeout(callback: () => void, milliseconds: number): unknown
}

/** Host-owned ports for a Discord Gateway connection. No credentials are discovered here. */
export interface DiscordGatewayPorts {
  readonly clock?: DiscordGatewayClock
  readonly random?: () => number
  readonly rest: DiscordGatewayRestPort
  readonly webSocket: DiscordGatewayWebSocketPort
}

export interface DiscordGatewayDispatch {
  readonly data: Readonly<Record<string, unknown>>
  readonly sequence: number | undefined
  readonly type: string
}

export interface DiscordGatewayState {
  readonly connected: boolean
  readonly lastError: string
  readonly sequence: number | undefined
  readonly sessionId: string | undefined
}

export interface DiscordGatewayTransportOptions {
  /** Explicit token supplied by the configured adapter; never read from the environment. */
  readonly botToken: string
  readonly messageContentIntent?: boolean
  readonly onDispatch: (dispatch: DiscordGatewayDispatch) => void | Promise<void>
  readonly onError?: (error: unknown) => void
  readonly ports: DiscordGatewayPorts
  /** Base retry delay. Each consecutive reconnect is bounded exponential backoff. */
  readonly reconnectDelay?: number
}

/**
 * Direct Discord Gateway protocol implementation.
 *
 * The transport owns protocol state only: endpoint discovery, heartbeats,
 * identify/resume frames, and reconnect scheduling. A host supplies both
 * network ports, so this layer never discovers credentials or silently starts
 * an external client.
 */
export class DiscordGatewayTransport {
  private active = false
  private awaitingHeartbeatAck = false
  private readonly botToken: string
  private readonly clock: DiscordGatewayClock
  private connecting = false
  private generation = 0
  private gatewayUrl = ''
  private heartbeatInterval = 0
  private heartbeatTimer: unknown
  private lastErrorText = ''
  private readonly messageContentIntent: boolean
  private readonly onDispatch: (dispatch: DiscordGatewayDispatch) => void | Promise<void>
  private readonly onError: ((error: unknown) => void) | undefined
  private readonly ports: DiscordGatewayPorts
  private readonly random: () => number
  private readonly reconnectDelay: number
  private reconnectAttempts = 0
  private reconnectTimer: unknown
  private resumeGatewayUrl = ''
  private sequence: number | undefined
  private sessionId: string | undefined
  private socket: DiscordGatewaySocket | undefined

  constructor(options: DiscordGatewayTransportOptions) {
    this.botToken = requiredText(options.botToken, 'Discord bot token')
    this.clock = options.ports.clock ?? systemClock
    this.messageContentIntent = options.messageContentIntent ?? true
    this.onDispatch = options.onDispatch
    this.onError = options.onError
    this.ports = options.ports
    this.random = options.ports.random ?? Math.random
    this.reconnectDelay = nonNegativeInteger(options.reconnectDelay ?? DEFAULT_RECONNECT_DELAY, 'reconnectDelay')
  }

  get state(): DiscordGatewayState {
    return {
      connected: this.socket !== undefined,
      lastError: this.lastErrorText,
      sequence: this.sequence,
      sessionId: this.sessionId,
    }
  }

  /** Discover and open the first Gateway connection. Connection failures are surfaced to the caller. */
  async start(): Promise<void> {
    if (this.active) return
    this.active = true
    this.lastErrorText = ''
    try {
      await this.open(true)
    } catch (error) {
      this.active = false
      this.clearTimers()
      this.socket = undefined
      throw error
    }
  }

  /** Stop heartbeats/retries and close the currently active socket. */
  async stop(): Promise<void> {
    this.active = false
    this.generation += 1
    this.clearTimers()
    const socket = this.socket
    this.socket = undefined
    if (socket) {
      await socket.close(1000, 'Xerxes Discord gateway stopped')
    }
  }

  private async open(initial: boolean): Promise<void> {
    if (!this.active || this.connecting) return
    this.connecting = true
    const generation = ++this.generation
    try {
      const socket = await this.ports.webSocket.connect(await this.endpoint(), {
        onOpen: candidate => this.acceptSocket(generation, candidate),
        onMessage: (candidate, payload) => { void this.handleMessage(generation, candidate, payload) },
        onClose: (candidate, event) => this.handleClose(generation, candidate, event),
        onError: (candidate, error) => this.handleSocketError(generation, candidate, error),
      })
      this.acceptSocket(generation, socket)
    } catch (error) {
      if (initial) {
        this.report(error)
        throw error
      }
      this.report(error)
      this.scheduleReconnect(this.canResume())
    } finally {
      this.connecting = false
    }
  }

  private async endpoint(): Promise<string> {
    if (this.resumeGatewayUrl) {
      return discordGatewayUrl(this.resumeGatewayUrl)
    }
    if (!this.gatewayUrl) {
      const discovered = await this.ports.rest.gatewayBot(this.botToken)
      this.gatewayUrl = requiredText(discovered.url, 'Discord Gateway URL')
    }
    return discordGatewayUrl(this.gatewayUrl)
  }

  private acceptSocket(generation: number, socket: DiscordGatewaySocket): boolean {
    if (!this.active || generation !== this.generation) {
      void socket.close(1000, 'stale Discord gateway connection')
      return false
    }
    this.socket = socket
    return true
  }

  private async handleMessage(generation: number, socket: DiscordGatewaySocket, payload: string): Promise<void> {
    if (!this.acceptSocket(generation, socket)) return
    const frame = parseFrame(payload)
    if (!frame) {
      this.report(new TypeError('Discord Gateway sent an invalid JSON frame'))
      return
    }
    const sequence = gatewaySequence(frame.s)
    if (sequence !== undefined) {
      this.sequence = sequence
    }
    const opcode = gatewayOpcode(frame.op)
    if (opcode === undefined) {
      this.report(new TypeError('Discord Gateway frame omitted a valid opcode'))
      return
    }
    switch (opcode) {
      case 0:
        await this.handleDispatch(frame, sequence)
        return
      case 1:
        await this.sendRequestedHeartbeat(socket)
        return
      case 7:
        this.scheduleReconnect(this.canResume())
        return
      case 9:
        this.handleInvalidSession(frame.d === true)
        return
      case 10:
        await this.handleHello(socket, frame.d)
        return
      case 11:
        this.awaitingHeartbeatAck = false
        return
      default:
        return
    }
  }

  private async handleDispatch(
    frame: Readonly<Record<string, unknown>>,
    sequence: number | undefined,
  ): Promise<void> {
    const type = textValue(frame.t)
    if (!type) return
    const data = recordValue(frame.d)
    if (type === 'READY') {
      this.recordReady(data)
      this.reconnectAttempts = 0
    }
    try {
      await this.onDispatch({ data, sequence, type })
    } catch (error) {
      // An inbound application failure must not discard a healthy Gateway session.
      this.report(error)
    }
  }

  private recordReady(data: Readonly<Record<string, unknown>>): void {
    const sessionId = textValue(data.session_id)
    if (sessionId) this.sessionId = sessionId
    const resumeUrl = textValue(data.resume_gateway_url)
    if (resumeUrl) this.resumeGatewayUrl = resumeUrl
  }

  private async handleHello(socket: DiscordGatewaySocket, value: unknown): Promise<void> {
    const interval = positiveInteger(recordValue(value).heartbeat_interval)
    if (interval === undefined) {
      const error = new TypeError('Discord Gateway HELLO omitted heartbeat_interval')
      this.report(error)
      this.scheduleReconnect(false)
      return
    }
    this.heartbeatInterval = interval
    this.awaitingHeartbeatAck = false
    this.clearHeartbeatTimer()
    await this.heartbeat(socket)
    if (!this.active || socket !== this.socket) return
    await this.send(socket, this.canResume()
      ? { op: 6, d: { seq: this.sequence, session_id: this.sessionId, token: this.botToken } }
      : {
          op: 2,
          d: {
            intents: discordGatewayIntents(this.messageContentIntent),
            properties: { $browser: 'xerxes', $device: 'xerxes', $os: 'bun' },
            token: this.botToken,
          },
        })
  }

  private async heartbeat(socket: DiscordGatewaySocket): Promise<void> {
    if (!this.active || socket !== this.socket) return
    if (this.awaitingHeartbeatAck) {
      this.scheduleReconnect(this.canResume())
      return
    }
    this.awaitingHeartbeatAck = true
    await this.send(socket, { op: 1, d: this.sequence ?? null })
    if (!this.active || socket !== this.socket || !this.heartbeatInterval) return
    this.heartbeatTimer = this.clock.setTimeout(() => {
      this.heartbeatTimer = undefined
      void this.heartbeat(socket)
    }, this.heartbeatInterval)
  }

  private async sendRequestedHeartbeat(socket: DiscordGatewaySocket): Promise<void> {
    if (!this.active || socket !== this.socket) return
    this.awaitingHeartbeatAck = true
    await this.send(socket, { op: 1, d: this.sequence ?? null })
  }

  private async send(socket: DiscordGatewaySocket, frame: object): Promise<void> {
    try {
      await socket.send(JSON.stringify(frame))
    } catch (error) {
      this.report(error)
      this.scheduleReconnect(this.canResume())
    }
  }

  private handleInvalidSession(resumable: boolean): void {
    if (!resumable) {
      this.clearSession()
    }
    const random = Math.min(1, Math.max(0, this.random()))
    this.scheduleReconnect(resumable && this.canResume(), INVALID_SESSION_MIN_DELAY + Math.floor(random * INVALID_SESSION_DELAY_SPREAD))
  }

  private handleClose(generation: number, socket: DiscordGatewaySocket, event: DiscordGatewayCloseEvent): void {
    if (generation !== this.generation || !this.active) return
    if (socket === this.socket) this.socket = undefined
    this.clearHeartbeatTimer()
    if (FATAL_CLOSE_CODES.has(event.code)) {
      this.report(new Error(`Discord Gateway closed permanently (${event.code})`))
      this.active = false
      this.clearTimers()
      return
    }
    this.scheduleReconnect(this.canResume())
  }

  private handleSocketError(generation: number, socket: DiscordGatewaySocket, error: unknown): void {
    if (generation !== this.generation || !this.active) return
    this.report(error)
    if (socket === this.socket) this.socket = undefined
    this.clearHeartbeatTimer()
    this.scheduleReconnect(this.canResume())
    void socket.close(4000, 'Discord Gateway reconnecting after socket error')
  }

  private scheduleReconnect(resumable: boolean, explicitDelay?: number): void {
    if (!this.active || this.reconnectTimer !== undefined) return
    this.clearHeartbeatTimer()
    if (!resumable) this.clearSession()
    const socket = this.socket
    this.socket = undefined
    this.generation += 1
    if (socket) void socket.close(4000, 'Discord Gateway reconnecting')
    const delay = explicitDelay ?? this.backoffDelay()
    this.reconnectTimer = this.clock.setTimeout(() => {
      this.reconnectTimer = undefined
      void this.open(false)
    }, delay)
  }

  private backoffDelay(): number {
    const exponent = Math.min(this.reconnectAttempts, 5)
    this.reconnectAttempts += 1
    return Math.min(MAX_RECONNECT_DELAY, this.reconnectDelay * (2 ** exponent))
  }

  private canResume(): boolean {
    return this.sessionId !== undefined && this.sequence !== undefined
  }

  private clearSession(): void {
    this.sessionId = undefined
    this.sequence = undefined
    this.resumeGatewayUrl = ''
  }

  private clearHeartbeatTimer(): void {
    if (this.heartbeatTimer === undefined) return
    this.clock.clearTimeout(this.heartbeatTimer)
    this.heartbeatTimer = undefined
    this.awaitingHeartbeatAck = false
  }

  private clearTimers(): void {
    this.clearHeartbeatTimer()
    if (this.reconnectTimer !== undefined) {
      this.clock.clearTimeout(this.reconnectTimer)
      this.reconnectTimer = undefined
    }
  }

  private report(error: unknown): void {
    this.lastErrorText = errorMessage(error)
    try {
      this.onError?.(error)
    } catch {
      // Diagnostic hooks must not stop the Gateway state machine.
    }
  }
}

/** HTTP implementation that a host may explicitly place in {@link DiscordGatewayPorts}. */
export class FetchDiscordGatewayRestPort implements DiscordGatewayRestPort {
  private readonly apiBaseUrl: string
  private readonly fetchImplementation: ChannelFetch

  constructor(options: { readonly apiBaseUrl?: string; readonly fetchImplementation?: ChannelFetch } = {}) {
    this.apiBaseUrl = options.apiBaseUrl ?? 'https://discord.com/api/v10/'
    this.fetchImplementation = options.fetchImplementation ?? fetch
  }

  async gatewayBot(botToken: string): Promise<DiscordGatewayBot> {
    const token = requiredText(botToken, 'Discord bot token')
    const response = await this.fetchImplementation(providerUrl(this.apiBaseUrl, 'gateway/bot'), {
      headers: { Authorization: `Bot ${token}`, Accept: 'application/json' },
      method: 'GET',
    })
    if (!response.ok) throw new ChannelHttpError(response.status)
    const payload = recordValue(await response.json().catch(() => undefined))
    const url = textValue(payload.url)
    return { url: discordGatewayUrl(requiredText(url, 'Discord Gateway URL')) }
  }
}

/** Bun's real WebSocket client exposed behind the injectable Gateway port. */
export class BunDiscordGatewayWebSocketPort implements DiscordGatewayWebSocketPort {
  connect(url: string, handlers: DiscordGatewayConnectionHandlers): Promise<DiscordGatewaySocket> {
    const endpoint = discordGatewayUrl(url)
    return new Promise((resolve, reject) => {
      let settled = false
      let raw: WebSocket
      try {
        raw = new WebSocket(endpoint)
      } catch (error) {
        reject(error)
        return
      }
      const socket = new BunDiscordGatewaySocket(raw)
      raw.addEventListener('open', () => {
        handlers.onOpen(socket)
        if (!settled) {
          settled = true
          resolve(socket)
        }
      })
      raw.addEventListener('message', event => {
        void messageText(event.data).then(
          payload => {
            if (payload !== undefined) handlers.onMessage(socket, payload)
          },
          error => handlers.onError(socket, error),
        )
      })
      raw.addEventListener('error', event => {
        const error = new Error('Discord Gateway WebSocket error', { cause: event })
        handlers.onError(socket, error)
        if (!settled) {
          settled = true
          reject(error)
        }
      })
      raw.addEventListener('close', event => {
        handlers.onClose(socket, { code: event.code, reason: event.reason })
        if (!settled) {
          settled = true
          reject(new Error(`Discord Gateway closed before opening (${event.code})`))
        }
      })
    })
  }
}

/** Normalize an endpoint to Discord Gateway v10 JSON framing. */
export function discordGatewayUrl(value: string): string {
  let url: URL
  try {
    url = new URL(value)
  } catch (error) {
    throw new TypeError('Discord Gateway URL must be absolute', { cause: error })
  }
  if (url.protocol !== 'ws:' && url.protocol !== 'wss:') {
    throw new TypeError('Discord Gateway URL must use WS or WSS')
  }
  url.searchParams.set('v', DISCORD_GATEWAY_VERSION)
  url.searchParams.set('encoding', 'json')
  return url.toString()
}

/** Return Discord's required message intents with optional privileged content access. */
export function discordGatewayIntents(messageContentIntent = true): number {
  return DISCORD_GATEWAY_BASE_INTENTS | (messageContentIntent ? DISCORD_GATEWAY_MESSAGE_CONTENT_INTENT : 0)
}

class BunDiscordGatewaySocket implements DiscordGatewaySocket {
  constructor(private readonly socket: WebSocket) {}

  close(code?: number, reason?: string): void {
    this.socket.close(code, reason)
  }

  send(payload: string): void {
    this.socket.send(payload)
  }
}

const systemClock: DiscordGatewayClock = {
  clearTimeout: timer => globalThis.clearTimeout(timer as number),
  setTimeout: (callback, milliseconds) => globalThis.setTimeout(callback, milliseconds),
}

function parseFrame(payload: string): Readonly<Record<string, unknown>> | undefined {
  try {
    return recordValue(JSON.parse(payload))
  } catch {
    return undefined
  }
}

function recordValue(value: unknown): Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function textValue(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function gatewayOpcode(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : undefined
}

function gatewaySequence(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : undefined
}

function positiveInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0 ? value : undefined
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(name + ' must be a non-negative safe integer')
  }
  return value
}

function requiredText(value: string, name: string): string {
  const normalized = value.trim()
  if (!normalized) throw new TypeError(name + ' is required')
  return normalized
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

async function messageText(value: unknown): Promise<string | undefined> {
  if (typeof value === 'string') return value
  if (value instanceof ArrayBuffer) return new TextDecoder().decode(value)
  if (ArrayBuffer.isView(value)) return new TextDecoder().decode(value)
  if (value instanceof Blob) return value.text()
  return undefined
}
