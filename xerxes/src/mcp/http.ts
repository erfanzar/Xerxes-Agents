// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { SSEParser, type SSEEvent } from '../streaming/sse.js'
import {
  MCPConnectionError,
  MCPProtocolError,
  type MCPClientOptions,
  type MCPFetch,
  type MCPHttpTransportKind,
  type MCPJsonRpcNotification,
  type MCPJsonRpcRequest,
  type MCPJsonRpcResponse,
  type MCPServerConfig,
} from './types.js'

const DEFAULT_TIMEOUT_MS = 30_000
const JSON_CONTENT_TYPE = 'application/json'
const SSE_CONTENT_TYPE = 'text/event-stream'
const STREAMABLE_HTTP_ACCEPT = `${JSON_CONTENT_TYPE}, ${SSE_CONTENT_TYPE}`
const HEADER_NAME = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/
const VISIBLE_ASCII = /^[\x21-\x7E]+$/
const PROTOCOL_CONTROLLED_HEADERS = new Set([
  'accept',
  'content-length',
  'content-type',
  'host',
  'last-event-id',
  'mcp-protocol-version',
  'mcp-session-id',
])

type MCPOutboundFrame = MCPJsonRpcNotification | MCPJsonRpcRequest | MCPJsonRpcResponse

export type MCPHttpMessageHandler = (frame: unknown) => void
export type MCPHttpCloseHandler = (error: Error) => void

/**
 * HTTP wire transport for MCP. `sse` implements the deprecated two-endpoint
 * transport; `streamable_http` implements POST responses in JSON or SSE form.
 */
export class MCPHttpClientTransport {
  readonly expectsResponseInBody: boolean
  readonly kind: MCPHttpTransportKind

  private closed = false
  private legacyAbortController: AbortController | undefined
  private legacyEndpoint: URL | undefined
  private legacyEndpointReject: ((reason: Error) => void) | undefined
  private legacyEndpointResolve: ((endpoint: URL) => void) | undefined
  private legacyReader: ReadableStreamDefaultReader<Uint8Array> | undefined
  private protocolVersion: string | undefined
  private sessionId: string | undefined

  private readonly endpoint: URL
  private readonly fetchImpl: MCPFetch
  private readonly onClosed: MCPHttpCloseHandler | undefined
  private readonly onMessage: MCPHttpMessageHandler
  private readonly requestHeaders: Headers
  private readonly serverName: string
  private readonly timeoutMs: number

  constructor(
    config: MCPServerConfig,
    kind: MCPHttpTransportKind,
    onMessage: MCPHttpMessageHandler,
    onClosed: MCPHttpCloseHandler,
    options: MCPClientOptions = {},
  ) {
    this.kind = kind
    this.expectsResponseInBody = kind === 'streamable_http'
    this.endpoint = parseHttpUrl(config.url, `MCP ${kind} server ${config.name}`)
    this.fetchImpl = options.fetch ?? fetch
    this.onMessage = onMessage
    this.onClosed = onClosed
    this.requestHeaders = parseRequestHeaders(config.headers)
    this.serverName = config.name
    this.timeoutMs = parseTimeout(config.timeoutMs)
  }

  /** Open the legacy event stream. Streamable HTTP has no persistent connection to open. */
  async connect(): Promise<void> {
    if (this.kind === 'sse') {
      await this.openLegacySseStream()
    }
  }

  /** Send one JSON-RPC frame through the configured transport. */
  async send(frame: MCPOutboundFrame): Promise<void> {
    if (this.closed) {
      throw new MCPConnectionError(`MCP HTTP transport for ${this.serverName} is closed`)
    }
    if (this.kind === 'sse') {
      await this.sendLegacyFrame(frame)
      return
    }
    await this.sendStreamableFrame(frame)
  }

  /** Record the protocol version negotiated by initialize for subsequent HTTP requests. */
  setProtocolVersion(version: string): void {
    if (!version.trim()) {
      throw new MCPProtocolError('MCP negotiated an empty protocol version')
    }
    this.protocolVersion = version
  }

  /** Return the optional session identifier issued during Streamable HTTP initialization. */
  getSessionId(): string | undefined {
    return this.sessionId
  }

  /** Close the persistent SSE reader and best-effort terminate a Streamable HTTP session. */
  async disconnect(): Promise<void> {
    if (this.closed) {
      return
    }
    this.closed = true

    this.legacyEndpointReject?.(new MCPConnectionError(`Disconnected from MCP server ${this.serverName}`))
    this.legacyEndpointReject = undefined
    this.legacyEndpointResolve = undefined
    this.legacyAbortController?.abort()
    this.legacyAbortController = undefined

    try {
      await this.legacyReader?.cancel()
    } catch {
      // The HTTP stream may already be closed.
    }
    this.legacyReader = undefined

    const sessionId = this.sessionId
    this.sessionId = undefined
    if (this.kind === 'streamable_http' && sessionId !== undefined) {
      await this.terminateSession(sessionId)
    }
  }

  private async openLegacySseStream(): Promise<void> {
    const controller = new AbortController()
    this.legacyAbortController = controller
    let timedOut = false
    const timeout = setTimeout(() => {
      timedOut = true
      controller.abort()
    }, this.timeoutMs)

    let resolveEndpoint: (endpoint: URL) => void = () => undefined
    let rejectEndpoint: (reason: Error) => void = () => undefined
    const endpointReady = new Promise<URL>((resolve, reject) => {
      resolveEndpoint = resolve
      rejectEndpoint = reject
    })
    this.legacyEndpointResolve = resolveEndpoint
    this.legacyEndpointReject = rejectEndpoint

    try {
      const response = await this.fetchImpl(this.endpoint, {
        headers: this.headersForLegacySseGet(),
        method: 'GET',
        signal: controller.signal,
      })
      if (!response.ok) {
        throw httpStatusError(this.serverName, response, 'opening the legacy SSE stream')
      }
      if (!isContentType(response, SSE_CONTENT_TYPE)) {
        throw new MCPProtocolError(
          `MCP legacy SSE endpoint for ${this.serverName} returned ${displayContentType(response)} instead of ${SSE_CONTENT_TYPE}`,
        )
      }
      if (!response.body) {
        throw new MCPProtocolError(`MCP legacy SSE endpoint for ${this.serverName} returned no response body`)
      }

      const reader = response.body.getReader()
      this.legacyReader = reader
      void this.consumeLegacySse(reader)
      await endpointReady
    } catch (error) {
      controller.abort()
      const failure = timedOut
        ? new MCPConnectionError(`MCP legacy SSE connection to ${this.serverName} timed out after ${this.timeoutMs}ms`)
        : asHttpTransportError(error)
      throw failure
    } finally {
      clearTimeout(timeout)
      this.legacyEndpointReject = undefined
      this.legacyEndpointResolve = undefined
    }
  }

  private async consumeLegacySse(reader: ReadableStreamDefaultReader<Uint8Array>): Promise<void> {
    try {
      await consumeSse(reader, event => this.handleLegacySseEvent(event))
      if (!this.closed) {
        this.handleLegacyStreamFailure(new MCPConnectionError(`MCP legacy SSE stream for ${this.serverName} closed`))
      }
    } catch (error) {
      if (!this.closed) {
        this.handleLegacyStreamFailure(asHttpTransportError(error))
      }
    } finally {
      if (this.legacyReader === reader) {
        this.legacyReader = undefined
      }
      reader.releaseLock()
    }
  }

  private handleLegacySseEvent(event: SSEEvent): void {
    if (!this.legacyEndpoint) {
      if (event.event !== 'endpoint') {
        throw new MCPProtocolError(`MCP legacy SSE server ${this.serverName} did not send endpoint as its first event`)
      }
      const endpoint = parseLegacyMessageEndpoint(event.data, this.endpoint, this.serverName)
      this.legacyEndpoint = endpoint
      this.legacyEndpointResolve?.(endpoint)
      return
    }

    if (event.event !== 'message') {
      throw new MCPProtocolError(`MCP legacy SSE server ${this.serverName} emitted unexpected ${event.event} event`)
    }
    this.onMessage(parseJsonFrame(event.data, `MCP legacy SSE server ${this.serverName}`))
  }

  private handleLegacyStreamFailure(error: Error): void {
    this.legacyEndpointReject?.(error)
    this.legacyEndpointReject = undefined
    this.legacyEndpointResolve = undefined
    this.onClosed?.(error)
  }

  private async sendLegacyFrame(frame: MCPOutboundFrame): Promise<void> {
    const endpoint = this.legacyEndpoint
    if (!endpoint) {
      throw new MCPConnectionError(`MCP legacy SSE server ${this.serverName} has not sent a message endpoint`)
    }

    await this.runWithTimeout(`MCP legacy SSE request to ${this.serverName}`, async signal => {
      const response = await this.fetchImpl(endpoint, {
        body: JSON.stringify(frame),
        headers: this.headersForLegacyPost(),
        method: 'POST',
        signal,
      })
      if (!response.ok) {
        throw httpStatusError(this.serverName, response, 'sending a legacy SSE message')
      }
      await cancelBody(response)
    })
  }

  private async sendStreamableFrame(frame: MCPOutboundFrame): Promise<void> {
    await this.runWithTimeout(`MCP Streamable HTTP request to ${this.serverName}`, async signal => {
      const response = await this.fetchImpl(this.endpoint, {
        body: JSON.stringify(frame),
        headers: this.headersForStreamablePost(frame),
        method: 'POST',
        signal,
      })
      if (!response.ok) {
        if (response.status === 404 && this.sessionId !== undefined) {
          this.sessionId = undefined
        }
        throw httpStatusError(this.serverName, response, 'sending a Streamable HTTP message')
      }

      if (!isRequest(frame)) {
        if (response.status !== 202) {
          throw new MCPProtocolError(
            `MCP Streamable HTTP server ${this.serverName} returned HTTP ${response.status} for a notification; expected 202`,
          )
        }
        await cancelBody(response)
        return
      }

      this.captureSessionId(frame, response)
      if (isContentType(response, JSON_CONTENT_TYPE)) {
        this.onMessage(await parseJsonResponse(response, `MCP Streamable HTTP server ${this.serverName}`))
        return
      }
      if (isContentType(response, SSE_CONTENT_TYPE)) {
        if (!response.body) {
          throw new MCPProtocolError(`MCP Streamable HTTP server ${this.serverName} returned an empty SSE response`)
        }
        const reader = response.body.getReader()
        try {
          await consumeSse(reader, event => this.handleStreamableSseEvent(event))
        } finally {
          reader.releaseLock()
        }
        return
      }
      throw new MCPProtocolError(
        `MCP Streamable HTTP server ${this.serverName} returned unsupported content type ${displayContentType(response)}`,
      )
    })
  }

  private handleStreamableSseEvent(event: SSEEvent): void {
    if (event.event !== 'message') {
      throw new MCPProtocolError(`MCP Streamable HTTP server ${this.serverName} emitted unexpected ${event.event} event`)
    }
    this.onMessage(parseJsonFrame(event.data, `MCP Streamable HTTP server ${this.serverName}`))
  }

  private captureSessionId(frame: MCPJsonRpcRequest, response: Response): void {
    if (frame.method !== 'initialize') {
      return
    }
    const sessionId = response.headers.get('mcp-session-id')
    if (sessionId === null) {
      return
    }
    if (!VISIBLE_ASCII.test(sessionId)) {
      throw new MCPProtocolError(`MCP Streamable HTTP server ${this.serverName} returned an invalid Mcp-Session-Id`)
    }
    this.sessionId = sessionId
  }

  private headersForLegacySseGet(): Headers {
    const headers = new Headers(this.requestHeaders)
    headers.set('Accept', SSE_CONTENT_TYPE)
    return headers
  }

  private headersForLegacyPost(): Headers {
    const headers = new Headers(this.requestHeaders)
    headers.set('Accept', JSON_CONTENT_TYPE)
    headers.set('Content-Type', JSON_CONTENT_TYPE)
    return headers
  }

  private headersForStreamablePost(frame: MCPOutboundFrame): Headers {
    const headers = new Headers(this.requestHeaders)
    headers.set('Accept', STREAMABLE_HTTP_ACCEPT)
    headers.set('Content-Type', JSON_CONTENT_TYPE)
    if (this.protocolVersion !== undefined && !isInitializeRequest(frame)) {
      headers.set('MCP-Protocol-Version', this.protocolVersion)
    }
    if (this.sessionId !== undefined && !isInitializeRequest(frame)) {
      headers.set('Mcp-Session-Id', this.sessionId)
    }
    return headers
  }

  private async terminateSession(sessionId: string): Promise<void> {
    try {
      await this.runWithTimeout(`terminating MCP session for ${this.serverName}`, async signal => {
        const headers = new Headers(this.requestHeaders)
        headers.set('Accept', STREAMABLE_HTTP_ACCEPT)
        headers.set('Mcp-Session-Id', sessionId)
        if (this.protocolVersion !== undefined) {
          headers.set('MCP-Protocol-Version', this.protocolVersion)
        }
        const response = await this.fetchImpl(this.endpoint, { headers, method: 'DELETE', signal })
        await cancelBody(response)
      })
    } catch {
      // Disconnect remains best-effort when a remote session is already gone.
    }
  }

  private async runWithTimeout<T>(label: string, operation: (signal: AbortSignal) => Promise<T>): Promise<T> {
    const controller = new AbortController()
    let timedOut = false
    const timeout = setTimeout(() => {
      timedOut = true
      controller.abort()
    }, this.timeoutMs)
    try {
      return await operation(controller.signal)
    } catch (error) {
      if (timedOut) {
        throw new MCPConnectionError(`${label} timed out after ${this.timeoutMs}ms`)
      }
      throw asHttpTransportError(error)
    } finally {
      clearTimeout(timeout)
    }
  }
}

async function consumeSse(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onEvent: (event: SSEEvent) => void,
): Promise<void> {
  const decoder = new TextDecoder()
  const parser = new SSEParser()
  const emitCompleted = (): void => {
    for (const event of parser.drain()) {
      if (event.data || event.event === 'message') {
        onEvent(event)
      }
    }
  }
  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      parser.feed(decoder.decode())
      parser.feed('\n\n')
      emitCompleted()
      return
    }
    parser.feed(decoder.decode(value, { stream: true }))
    emitCompleted()
  }
}

function parseHttpUrl(value: string | undefined, label: string): URL {
  if (!value) {
    throw new MCPConnectionError(`${label} requires a URL`)
  }
  let url: URL
  try {
    url = new URL(value)
  } catch {
    throw new MCPConnectionError(`${label} URL is invalid`)
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') {
    throw new MCPConnectionError(`${label} URL scheme must be http or https`)
  }
  if (!url.hostname) {
    throw new MCPConnectionError(`${label} URL must include a host`)
  }
  if (url.username || url.password) {
    throw new MCPConnectionError(`${label} URL must not include credentials; use headers instead`)
  }
  if (url.hash) {
    throw new MCPConnectionError(`${label} URL must not include a fragment`)
  }
  return url
}

function parseLegacyMessageEndpoint(value: string, origin: URL, serverName: string): URL {
  let endpoint: URL
  try {
    endpoint = new URL(value, origin)
  } catch {
    throw new MCPProtocolError(`MCP legacy SSE server ${serverName} sent an invalid message endpoint`)
  }
  if ((endpoint.protocol !== 'http:' && endpoint.protocol !== 'https:') || endpoint.origin !== origin.origin) {
    throw new MCPProtocolError(`MCP legacy SSE server ${serverName} sent a cross-origin message endpoint`)
  }
  return endpoint
}

function parseRequestHeaders(headers: Readonly<Record<string, string>> | undefined): Headers {
  const parsed = new Headers()
  for (const [name, value] of Object.entries(headers ?? {})) {
    if (!HEADER_NAME.test(name)) {
      throw new MCPConnectionError(`MCP HTTP header name ${name} is invalid`)
    }
    if (PROTOCOL_CONTROLLED_HEADERS.has(name.toLowerCase())) {
      throw new MCPConnectionError(`MCP HTTP header ${name} is controlled by the transport`)
    }
    if (typeof value !== 'string' || value.includes('\r') || value.includes('\n')) {
      throw new MCPConnectionError(`MCP HTTP header ${name} has an invalid value`)
    }
    parsed.set(name, value)
  }
  return parsed
}

function parseTimeout(value: number | undefined): number {
  const timeout = value ?? DEFAULT_TIMEOUT_MS
  if (!Number.isFinite(timeout) || timeout <= 0) {
    throw new MCPConnectionError('MCP timeoutMs must be a positive finite number')
  }
  return timeout
}

function isRequest(frame: MCPOutboundFrame): frame is MCPJsonRpcRequest {
  return 'id' in frame && 'method' in frame
}

function isInitializeRequest(frame: MCPOutboundFrame): boolean {
  return isRequest(frame) && frame.method === 'initialize'
}

function isContentType(response: Response, expected: string): boolean {
  return response.headers.get('content-type')?.split(';', 1)[0]?.trim().toLowerCase() === expected
}

function displayContentType(response: Response): string {
  return response.headers.get('content-type') ?? 'no content type'
}

function parseJsonFrame(data: string, source: string): unknown {
  try {
    return JSON.parse(data) as unknown
  } catch {
    throw new MCPProtocolError(`${source} emitted invalid JSON-RPC JSON`)
  }
}

async function parseJsonResponse(response: Response, source: string): Promise<unknown> {
  return parseJsonFrame(await response.text(), source)
}

function httpStatusError(serverName: string, response: Response, operation: string): MCPConnectionError {
  return new MCPConnectionError(`MCP HTTP server ${serverName} returned HTTP ${response.status} while ${operation}`)
}

function asHttpTransportError(error: unknown): Error {
  if (error instanceof MCPConnectionError || error instanceof MCPProtocolError) {
    return error
  }
  return new MCPConnectionError(error instanceof Error ? error.message : String(error))
}

async function cancelBody(response: Response): Promise<void> {
  try {
    await response.body?.cancel()
  } catch {
    // A response can be closed before the caller drains it.
  }
}
