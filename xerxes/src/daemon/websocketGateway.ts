// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { timingSafeEqual } from 'node:crypto'
import type { ServerWebSocket } from 'bun'

import { daemonEvent, jsonRpcFailure, type JsonRpcPayload } from '../protocol/jsonRpc.js'
import type { DaemonTransportConnection } from './transport.js'

const DEFAULT_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
const DEFAULT_MAX_BUFFERED_BYTES = 8 * 1024 * 1024
const DEFAULT_IDLE_TIMEOUT = 120
const SERVER_STOP_GRACE = 50
const WEBSOCKET_OPEN = 1

export interface DaemonWebSocketGatewayOptions {
  /** Token accepted from `Authorization: Bearer` or the `token` query parameter. */
  readonly authToken?: string
  /** Address to listen on. A loopback default prevents accidental remote exposure. */
  readonly host?: string
  /** Max queued outbound bytes allowed per client before it is disconnected. */
  readonly maxBufferedBytes?: number
  /** Max inbound or outbound JSON frame size. */
  readonly maxMessageBytes?: number
  /** WebSocket endpoint path. */
  readonly path?: string
  /** Seconds without traffic before Bun closes an idle client. */
  readonly idleTimeout?: number
  /** TCP port, or `0` to let the operating system choose one. */
  readonly port: number
}

export type DaemonWebSocketRequestHandler = (
  connection: DaemonTransportConnection,
  rawRequest: string,
) => Promise<void>

export type DaemonWebSocketDisconnectHandler = (
  connection: DaemonTransportConnection,
) => void

/**
 * Bun-native WebSocket transport for the daemon's JSON-RPC endpoint.
 *
 * It deliberately delegates request parsing and method dispatch to the daemon
 * server. That keeps the Unix NDJSON and WebSocket surfaces protocol-identical
 * while this layer handles upgrade authentication, client lifecycle, frame
 * limits, and backpressure.
 */
export class DaemonWebSocketGateway {
  private readonly authToken: string | undefined
  private readonly clients = new Set<GatewayConnection>()
  private readonly handler: DaemonWebSocketRequestHandler
  private readonly host: string
  private readonly idleTimeout: number
  private readonly maxBufferedBytes: number
  private readonly maxMessageBytes: number
  private readonly onDisconnect: DaemonWebSocketDisconnectHandler
  private readonly path: string
  private readonly port: number
  private server: Bun.Server<GatewayConnection> | undefined

  constructor(
    options: DaemonWebSocketGatewayOptions,
    handler: DaemonWebSocketRequestHandler,
    onDisconnect: DaemonWebSocketDisconnectHandler = () => undefined,
  ) {
    this.authToken = nonBlank(options.authToken)
    this.handler = handler
    this.host = nonBlank(options.host) ?? '127.0.0.1'
    this.idleTimeout = boundedInteger(options.idleTimeout, DEFAULT_IDLE_TIMEOUT, 'idleTimeout', 1, 960)
    this.maxBufferedBytes = boundedInteger(
      options.maxBufferedBytes,
      DEFAULT_MAX_BUFFERED_BYTES,
      'maxBufferedBytes',
      1024,
      DEFAULT_MAX_MESSAGE_BYTES,
    )
    this.maxMessageBytes = boundedInteger(
      options.maxMessageBytes,
      DEFAULT_MAX_MESSAGE_BYTES,
      'maxMessageBytes',
      1024,
      DEFAULT_MAX_MESSAGE_BYTES,
    )
    this.onDisconnect = onDisconnect
    this.path = endpointPath(options.path)
    this.port = boundedInteger(options.port, 0, 'port', 0, 65_535)
  }

  /** The public WebSocket URL once the server has been started. */
  get url(): URL | undefined {
    const serverUrl = this.server?.url
    if (!serverUrl) {
      return undefined
    }
    const url = new URL(serverUrl)
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:'
    url.pathname = this.path
    return url
  }

  /** Actual listening port, useful when the configured port was `0`. */
  get listeningPort(): number | undefined {
    return this.server?.port
  }

  /** Number of successfully upgraded, live clients. */
  get clientCount(): number {
    return this.clients.size
  }

  start(): void {
    if (this.server) {
      return
    }
    this.server = Bun.serve<GatewayConnection>({
      hostname: this.host,
      port: this.port,
      fetch: (request, server) => this.upgrade(request, server),
      websocket: {
        open: websocket => this.open(websocket),
        message: (websocket, message) => this.message(websocket, message),
        close: websocket => this.close(websocket),
        // Bun protects the parser at the hard 16 MiB ceiling. The message
        // callback below enforces a caller's lower configured limit and can
        // return a standards-compliant 1009 close frame instead of a reset.
        maxPayloadLength: DEFAULT_MAX_MESSAGE_BYTES,
        backpressureLimit: this.maxBufferedBytes,
        closeOnBackpressureLimit: true,
        idleTimeout: this.idleTimeout,
      },
    })
  }

  async stop(): Promise<void> {
    const server = this.server
    if (!server) {
      return
    }
    this.server = undefined
    for (const client of this.clients) {
      client.close(1001, 'daemon shutting down')
    }
    this.clients.clear()
    // Bun stops accepting connections synchronously, but its completion
    // promise can wait on a peer's close handshake. A stalled remote peer must
    // not keep daemon shutdown blocked indefinitely.
    const stopped = server.stop(true).catch(() => undefined)
    await Promise.race([stopped, Bun.sleep(SERVER_STOP_GRACE)])
  }

  /** Fan a daemon event to all currently connected remote clients. */
  broadcast(type: string, payload: JsonRpcPayload): void {
    const serialized = this.serialize(daemonEvent(type, payload))
    if (!serialized) {
      return
    }
    for (const client of this.clients) {
      this.sendSerialized(client, serialized)
    }
  }

  private upgrade(request: Request, server: Bun.Server<GatewayConnection>): Response | undefined {
    const url = new URL(request.url)
    if (url.pathname !== this.path) {
      return new Response('Not Found', { status: 404 })
    }
    if (request.headers.get('upgrade')?.toLowerCase() !== 'websocket') {
      return new Response('WebSocket Upgrade Required', {
        status: 426,
        headers: { Upgrade: 'websocket' },
      })
    }
    if (!websocketRequestAuthorized(request, this.authToken)) {
      return new Response(null, {
        status: 401,
        headers: { 'WWW-Authenticate': 'Bearer realm="xerxes"' },
      })
    }
    const connection = new GatewayConnection(this)
    if (!server.upgrade(request, { data: connection })) {
      return new Response('WebSocket upgrade failed', { status: 400 })
    }
    return undefined
  }

  private open(websocket: ServerWebSocket<GatewayConnection>): void {
    const connection = websocket.data
    connection.attach(websocket)
    this.clients.add(connection)
  }

  private message(websocket: ServerWebSocket<GatewayConnection>, message: string | Buffer<ArrayBuffer>): void {
    if (typeof message !== 'string') {
      websocket.close(1003, 'text JSON-RPC messages only')
      return
    }
    if (utf8Length(message) > this.maxMessageBytes) {
      websocket.close(1009, 'message exceeds limit')
      return
    }
    const connection = websocket.data
    connection.enqueue(async () => {
      try {
        await this.handler(connection, message)
      } catch {
        this.send(connection, jsonRpcFailure(null, -32000, 'Internal daemon error'))
      }
    })
  }

  private close(websocket: ServerWebSocket<GatewayConnection>): void {
    const connection = websocket.data
    this.clients.delete(connection)
    connection.detach()
    this.onDisconnect(connection)
  }

  send(connection: GatewayConnection, frame: object): void {
    const serialized = this.serialize(frame)
    if (!serialized) {
      connection.close(1009, 'outbound message exceeds limit')
      return
    }
    this.sendSerialized(connection, serialized)
  }

  private sendSerialized(connection: GatewayConnection, serialized: string): void {
    const websocket = connection.websocket
    if (!websocket || websocket.readyState !== WEBSOCKET_OPEN) {
      this.clients.delete(connection)
      return
    }
    if (websocket.getBufferedAmount() >= this.maxBufferedBytes) {
      this.dropSlowClient(connection)
      return
    }
    const sent = websocket.sendText(serialized)
    if (sent <= 0 || websocket.getBufferedAmount() >= this.maxBufferedBytes) {
      this.dropSlowClient(connection)
    }
  }

  private dropSlowClient(connection: GatewayConnection): void {
    this.clients.delete(connection)
    connection.close(1013, 'client backpressure limit reached')
  }

  private serialize(frame: object): string | undefined {
    let serialized: string | undefined
    try {
      serialized = JSON.stringify(frame)
    } catch {
      return undefined
    }
    if (!serialized || utf8Length(serialized) > this.maxMessageBytes) {
      return undefined
    }
    return serialized
  }
}

/** Authenticate an upgrade request without exposing a timing oracle for tokens. */
export function websocketRequestAuthorized(request: Request, expectedToken?: string): boolean {
  const expected = nonBlank(expectedToken)
  if (!expected) {
    return true
  }
  const bearer = bearerToken(request.headers.get('authorization'))
  if (bearer && tokensMatch(bearer, expected)) {
    return true
  }
  const queryToken = new URL(request.url).searchParams.get('token')
  return queryToken !== null && tokensMatch(queryToken, expected)
}

class GatewayConnection implements DaemonTransportConnection {
  activeSessionKey = `ws:${connectionKey()}`
  private pending = Promise.resolve()
  websocket: ServerWebSocket<GatewayConnection> | undefined

  constructor(private readonly gateway: DaemonWebSocketGateway) {}

  attach(websocket: ServerWebSocket<GatewayConnection>): void {
    this.websocket = websocket
  }

  detach(): void {
    this.websocket = undefined
  }

  enqueue(task: () => Promise<void>): void {
    this.pending = this.pending.then(task, task)
  }

  send(frame: object): void {
    this.gateway.send(this, frame)
  }

  close(code: number, reason: string): void {
    const websocket = this.websocket
    if (websocket?.readyState === WEBSOCKET_OPEN) {
      websocket.close(code, reason)
    }
  }
}

function endpointPath(value: string | undefined): string {
  const path = nonBlank(value) ?? '/'
  if (!path.startsWith('/') || path.includes('?') || path.includes('#')) {
    throw new Error('path must be an absolute path without a query or fragment')
  }
  return path
}

function boundedInteger(
  value: number | undefined,
  fallback: number,
  name: string,
  minimum: number,
  maximum: number,
): number {
  const candidate = value ?? fallback
  if (!Number.isInteger(candidate) || candidate < minimum || candidate > maximum) {
    throw new Error(`${name} must be an integer between ${minimum} and ${maximum}`)
  }
  return candidate
}

function nonBlank(value: string | undefined): string | undefined {
  const trimmed = value?.trim()
  return trimmed || undefined
}

function bearerToken(value: string | null): string | undefined {
  if (!value) {
    return undefined
  }
  const match = /^Bearer\s+(.+)$/i.exec(value)
  return match?.[1]?.trim() || undefined
}

function tokensMatch(actual: string, expected: string): boolean {
  const actualBytes = Buffer.from(actual)
  const expectedBytes = Buffer.from(expected)
  return actualBytes.byteLength === expectedBytes.byteLength && timingSafeEqual(actualBytes, expectedBytes)
}

function utf8Length(value: string): number {
  return Buffer.byteLength(value, 'utf8')
}

function connectionKey(): string {
  return crypto.randomUUID().replaceAll('-', '').slice(0, 12)
}
