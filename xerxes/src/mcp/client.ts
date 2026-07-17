// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { JsonObject, JsonSchema } from '../types/toolCalls.js'
import { MCPHttpClientTransport } from './http.js'
import {
  MCP_PROTOCOL_VERSION,
  MCP_STREAMABLE_HTTP_PROTOCOL_VERSION,
  MCPConnectionError,
  MCPProtocolError,
  MCPRemoteError,
  type MCPCapabilities,
  type MCPClientOptions,
  type MCPContent,
  type MCPHttpTransportKind,
  type MCPImplementation,
  type MCPInitializeResult,
  type MCPJsonRpcErrorData,
  type MCPJsonRpcId,
  type MCPJsonRpcNotification,
  type MCPJsonRpcRequest,
  type MCPJsonRpcResult,
  type MCPJsonRpcResponse,
  type MCPPrompt,
  type MCPPromptArgument,
  type MCPPromptResult,
  type MCPResource,
  type MCPResourceContentsResult,
  type MCPServerCapabilities,
  type MCPServerConfig,
  type MCPTool,
  type MCPToolAnnotations,
  type MCPToolCallResult,
  isMCPJsonObject,
  isMCPJsonRpcId,
  isMCPRecord,
  mcpJsonRpcFailure,
} from './types.js'

const DEFAULT_TIMEOUT_MS = 30_000
const MAX_STDERR_CHARS = 16_384

interface PendingRequest {
  readonly detach?: () => void
  readonly reject: (reason: unknown) => void
  readonly resolve: (value: MCPJsonRpcResult) => void
  readonly timeout: ReturnType<typeof setTimeout>
}

export type MCPNotificationHandler = (notification: MCPJsonRpcNotification) => void

/** Options for one MCP tool invocation. */
export interface MCPToolCallOptions {
  /** Rejects the call and discards its pending request when aborted. */
  readonly signal?: AbortSignal
}

/**
 * An MCP client that owns a stdio subprocess or an HTTP transport.
 *
 * The client starts the MCP handshake on {@link connect}, discovers tools, and keeps a
 * pending-request table so responses can arrive in any order.
 */
export class MCPClient {
  private static nextRequestId = 0

  readonly config: MCPServerConfig
  readonly prompts: MCPPrompt[] = []
  readonly resources: MCPResource[] = []
  readonly tools: MCPTool[] = []

  connected = false
  initializeResult: MCPInitializeResult | undefined
  sessionId: string | undefined
  serverCapabilities: MCPServerCapabilities = {}
  serverInfo: MCPImplementation | undefined

  private closing = false
  private readonly notificationHandlers = new Set<MCPNotificationHandler>()
  private readonly pending = new Map<MCPJsonRpcId, PendingRequest>()
  private httpTransport: MCPHttpClientTransport | undefined
  private process: Bun.PipedSubprocess | undefined
  private stderrOutput = ''
  private stdoutReader: ReadableStreamDefaultReader<Uint8Array> | undefined

  constructor(config: MCPServerConfig, private readonly options: MCPClientOptions = {}) {
    this.config = config
  }

  /** Open the configured transport, initialize it, and discover its capabilities. */
  async connect(): Promise<void> {
    if (this.connected) {
      return
    }
    if (this.config.enabled === false) {
      throw new MCPConnectionError(`MCP server ${this.config.name} is disabled`)
    }
    const transport = this.config.transport ?? 'stdio'
    this.closing = false
    this.stderrOutput = ''
    try {
      if (transport === 'stdio') {
        await this.openStdioTransport()
      } else if (transport === 'sse' || transport === 'streamable_http') {
        await this.openHttpTransport(transport)
      } else {
        throw new MCPConnectionError(`MCP transport ${String(transport)} is not supported`)
      }

      const result = await this.request('initialize', {
        capabilities: this.config.clientCapabilities ?? {},
        clientInfo: this.config.clientInfo ?? { name: 'xerxes', version: '0.3.0' },
        protocolVersion: this.config.protocolVersion ?? defaultProtocolVersion(transport),
      }, true)
      this.initializeResult = parseInitializeResult(result)
      this.serverCapabilities = this.initializeResult.capabilities
      this.serverInfo = this.initializeResult.serverInfo
      this.httpTransport?.setProtocolVersion(this.initializeResult.protocolVersion)
      this.sessionId = this.httpTransport?.getSessionId()
      this.connected = true
      await this.notify('notifications/initialized')
      await this.refreshCapabilities()
    } catch (error) {
      await this.disconnect()
      throw error
    }
  }

  /** Close the active transport and reject every request still awaiting a response. */
  async disconnect(): Promise<void> {
    const process = this.process
    const httpTransport = this.httpTransport
    this.closing = true
    this.connected = false
    this.process = undefined
    this.httpTransport = undefined
    this.sessionId = undefined
    this.rejectPending(new MCPConnectionError(`Disconnected from MCP server ${this.config.name}`))

    try {
      await this.stdoutReader?.cancel()
    } catch {
      // The subprocess may already have closed stdout.
    }
    this.stdoutReader = undefined

    await httpTransport?.disconnect()

    if (process) {
      try {
        process.stdin.end()
      } catch {
        // A process that exited during connection setup has no writable stdin.
      }
      if (process.exitCode === null) {
        process.kill('SIGTERM')
      }
      await Promise.race([process.exited, sleep(250)])
      if (process.exitCode === null) {
        process.kill('SIGKILL')
        await process.exited
      }
    }
    this.closing = false
  }

  private async openStdioTransport(): Promise<void> {
    if (!this.config.command) {
      throw new MCPConnectionError(`MCP stdio server ${this.config.name} requires a command`)
    }
    let child: Bun.PipedSubprocess
    try {
      child = Bun.spawn([this.config.command, ...(this.config.args ?? [])], {
        env: { ...process.env, ...this.config.env },
        stderr: 'pipe',
        stdin: 'pipe',
        stdout: 'pipe',
      })
    } catch (error) {
      throw asConnectionError(error)
    }
    this.process = child
    void this.readStdout(child, child.stdout)
    void this.captureStderr(child.stderr)
    void child.exited.then(exitCode => this.handleProcessExit(child, exitCode))
  }

  private async openHttpTransport(kind: MCPHttpTransportKind): Promise<void> {
    const transport = new MCPHttpClientTransport(
      this.config,
      kind,
      value => this.handleInboundValue(value),
      error => this.handleHttpTransportClosed(error),
      this.options,
    )
    this.httpTransport = transport
    await transport.connect()
  }

  /** Subscribe to server notifications. Returns an unsubscribe callback. */
  onNotification(handler: MCPNotificationHandler): () => void {
    this.notificationHandlers.add(handler)
    return () => this.notificationHandlers.delete(handler)
  }

  /** Fetch and cache the server's published tool list. */
  async listTools(): Promise<readonly MCPTool[]> {
    const result = await this.request('tools/list', {})
    const rawTools = result.tools
    if (!Array.isArray(rawTools)) {
      throw new MCPProtocolError('MCP tools/list response did not include a tools array')
    }
    this.tools.splice(0, this.tools.length, ...rawTools.map(value => parseTool(value, this.config.name)))
    return this.tools
  }

  /** Fetch and cache the server's published resource list. */
  async listResources(): Promise<readonly MCPResource[]> {
    const result = await this.request('resources/list', {})
    const rawResources = result.resources
    if (!Array.isArray(rawResources)) {
      throw new MCPProtocolError('MCP resources/list response did not include a resources array')
    }
    this.resources.splice(0, this.resources.length, ...rawResources.map(value => parseResource(value, this.config.name)))
    return this.resources
  }

  /** Fetch and cache the server's published prompt list. */
  async listPrompts(): Promise<readonly MCPPrompt[]> {
    const result = await this.request('prompts/list', {})
    const rawPrompts = result.prompts
    if (!Array.isArray(rawPrompts)) {
      throw new MCPProtocolError('MCP prompts/list response did not include a prompts array')
    }
    this.prompts.splice(0, this.prompts.length, ...rawPrompts.map(value => parsePrompt(value, this.config.name)))
    return this.prompts
  }

  /** Invoke one published MCP tool. */
  async callTool(name: string, arguments_: JsonObject = {}, options: MCPToolCallOptions = {}): Promise<MCPToolCallResult> {
    const result = await this.request('tools/call', { name, arguments: arguments_ }, false, options.signal)
    const content = result.content
    if (!Array.isArray(content)) {
      throw new MCPProtocolError('MCP tools/call response did not include a content array')
    }
    const isError = result.isError
    if (isError !== undefined && typeof isError !== 'boolean') {
      throw new MCPProtocolError('MCP tools/call response has an invalid isError flag')
    }
    const structuredContent = result.structuredContent
    if (structuredContent !== undefined && !isMCPJsonObject(structuredContent)) {
      throw new MCPProtocolError('MCP tools/call response has invalid structuredContent')
    }
    return {
      content: content as MCPContent[],
      ...(isError === undefined ? {} : { isError }),
      ...(structuredContent === undefined ? {} : { structuredContent }),
    }
  }

  /** Read a resource by URI. */
  async readResource(uri: string): Promise<MCPResourceContentsResult> {
    const result = await this.request('resources/read', { uri })
    const contents = result.contents
    if (!Array.isArray(contents)) {
      throw new MCPProtocolError('MCP resources/read response did not include a contents array')
    }
    return { contents: contents as MCPResourceContentsResult['contents'] }
  }

  /** Resolve a prompt template with optional arguments. */
  async getPrompt(name: string, arguments_: JsonObject = {}): Promise<MCPPromptResult> {
    const result = await this.request('prompts/get', { name, arguments: arguments_ })
    const messages = result.messages
    if (!Array.isArray(messages)) {
      throw new MCPProtocolError('MCP prompts/get response did not include a messages array')
    }
    const description = result.description
    if (description !== undefined && typeof description !== 'string') {
      throw new MCPProtocolError('MCP prompts/get response has an invalid description')
    }
    return {
      messages: messages as MCPPromptResult['messages'],
      ...(description === undefined ? {} : { description }),
    }
  }

  /** Return recent stderr output captured from the subprocess for diagnostics. */
  getStderr(): string {
    return this.stderrOutput
  }

  private async refreshCapabilities(): Promise<void> {
    await this.listTools()
    await Promise.all([
      this.loadOptionalCapability(() => this.listResources(), this.resources),
      this.loadOptionalCapability(() => this.listPrompts(), this.prompts),
    ])
  }

  private async loadOptionalCapability<T>(load: () => Promise<readonly T[]>, target: T[]): Promise<void> {
    try {
      await load()
    } catch {
      target.splice(0, target.length)
    }
  }

  private request(
    method: string,
    params: MCPJsonRpcResult,
    allowBeforeConnected = false,
    signal?: AbortSignal,
  ): Promise<MCPJsonRpcResult> {
    if ((!this.process && !this.httpTransport) || (!this.connected && !allowBeforeConnected)) {
      throw new MCPConnectionError(`Not connected to MCP server ${this.config.name}`)
    }

    const id = ++MCPClient.nextRequestId
    const frame: MCPJsonRpcRequest = { jsonrpc: '2.0', id, method, params }
    const httpTransport = this.httpTransport
    return new Promise<MCPJsonRpcResult>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.rejectPendingRequest(id, new MCPConnectionError(`MCP request ${method} timed out after ${this.timeoutMs}ms`))
      }, this.timeoutMs)
      const detach = signal === undefined
        ? undefined
        : onAbort(signal, () => {
          this.rejectPendingRequest(id, new MCPConnectionError(`MCP request ${method} was aborted`))
        })
      this.pending.set(id, { resolve, reject, timeout, ...(detach === undefined ? {} : { detach }) })
      if (signal?.aborted) {
        this.rejectPendingRequest(id, new MCPConnectionError(`MCP request ${method} was aborted`))
        return
      }
      void this.send(frame).then(
        () => {
          if (httpTransport?.expectsResponseInBody && this.pending.has(id)) {
            this.rejectPendingRequest(
              id,
              new MCPProtocolError(`MCP Streamable HTTP response for ${method} ended before its JSON-RPC result`),
            )
          }
        },
        error => this.rejectPendingRequest(id, asMcpError(error)),
      )
    })
  }

  private async notify(method: string, params?: MCPJsonRpcResult): Promise<void> {
    const notification: MCPJsonRpcNotification = params === undefined
      ? { jsonrpc: '2.0', method }
      : { jsonrpc: '2.0', method, params }
    await this.send(notification)
  }

  private async send(frame: MCPJsonRpcNotification | MCPJsonRpcRequest | MCPJsonRpcResponse): Promise<void> {
    const stdin = this.process?.stdin
    if (stdin) {
      stdin.write(`${JSON.stringify(frame)}\n`)
      stdin.flush()
      return
    }
    const httpTransport = this.httpTransport
    if (httpTransport) {
      await httpTransport.send(frame)
      return
    }
    throw new MCPConnectionError(`MCP server ${this.config.name} has no writable transport`)
  }

  private async readStdout(process: Bun.PipedSubprocess, stdout: ReadableStream<Uint8Array>): Promise<void> {
    const decoder = new TextDecoder()
    let buffer = ''
    const reader = stdout.getReader()
    this.stdoutReader = reader
    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          break
        }
        buffer += decoder.decode(value, { stream: true })
        buffer = this.consumeLines(buffer)
      }
      buffer += decoder.decode()
      if (buffer.trim()) {
        this.handleLine(buffer)
      }
    } catch (error) {
      if (!this.closing) {
        this.rejectPending(asConnectionError(error))
      }
    } finally {
      if (this.stdoutReader === reader) {
        this.stdoutReader = undefined
      }
      if (this.process === process && !this.closing) {
        this.connected = false
        this.rejectPending(new MCPConnectionError(`MCP server ${this.config.name} closed stdout`))
      }
    }
  }

  private async captureStderr(stderr: ReadableStream<Uint8Array>): Promise<void> {
    const reader = stderr.getReader()
    const decoder = new TextDecoder()
    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          return
        }
        this.stderrOutput = `${this.stderrOutput}${decoder.decode(value, { stream: true })}`.slice(-MAX_STDERR_CHARS)
      }
    } finally {
      this.stderrOutput = `${this.stderrOutput}${decoder.decode()}`.slice(-MAX_STDERR_CHARS)
    }
  }

  private consumeLines(buffer: string): string {
    let newline = buffer.indexOf('\n')
    while (newline >= 0) {
      const line = buffer.slice(0, newline).trim()
      buffer = buffer.slice(newline + 1)
      if (line) {
        this.handleLine(line)
      }
      newline = buffer.indexOf('\n')
    }
    return buffer
  }

  private handleLine(line: string): void {
    let value: unknown
    try {
      value = JSON.parse(line) as unknown
    } catch {
      // A subprocess may print non-JSON diagnostics on stdout; one bad line
      // must not fail every request currently in flight.
      this.debug(`skipping non-JSON stdout line from MCP server ${this.config.name}`)
      return
    }
    this.handleInboundValue(value)
  }

  private handleInboundValue(value: unknown): void {
    if (!isMCPRecord(value) || value.jsonrpc !== '2.0') {
      const id = isMCPRecord(value) && isMCPJsonRpcId(value.id) ? value.id : undefined
      if (id !== undefined && this.pending.has(id)) {
        this.rejectPendingRequest(id, new MCPProtocolError('MCP server emitted an invalid JSON-RPC frame'))
        return
      }
      this.debug(`skipping invalid JSON-RPC frame from MCP server ${this.config.name}`)
      return
    }

    if (typeof value.method === 'string') {
      if (Object.hasOwn(value, 'id') && !isMCPJsonRpcId(value.id)) {
        this.debug(`skipping MCP server request with an invalid request id from ${this.config.name}`)
        return
      }
      this.handleInboundRequestOrNotification(value)
      return
    }
    if (!isMCPJsonRpcId(value.id)) {
      this.debug(`skipping MCP server response without a valid request id from ${this.config.name}`)
      return
    }
    const pending = this.dropPendingRequest(value.id)
    if (!pending) {
      return
    }
    const hasError = Object.hasOwn(value, 'error')
    const hasResult = Object.hasOwn(value, 'result')
    if (hasError === hasResult) {
      pending.reject(new MCPProtocolError('MCP server response must include exactly one of result or error'))
      return
    }
    if (hasError) {
      if (!isMCPRecord(value.error)) {
        pending.reject(new MCPProtocolError('MCP server response has an invalid error object'))
        return
      }
      const error = parseRemoteError(value.error)
      pending.reject(error)
      return
    }
    if (!isMCPRecord(value.result)) {
      pending.reject(new MCPProtocolError('MCP server response has no result object'))
      return
    }
    pending.resolve(value.result)
  }

  private handleInboundRequestOrNotification(value: Record<string, unknown>): void {
    const params = isMCPRecord(value.params) ? value.params : undefined
    if (!isMCPJsonRpcId(value.id)) {
      const notification: MCPJsonRpcNotification = params === undefined
        ? { jsonrpc: '2.0', method: String(value.method) }
        : { jsonrpc: '2.0', method: String(value.method), params }
      for (const handler of this.notificationHandlers) {
        try {
          handler(notification)
        } catch (error) {
          // A throwing subscriber must not take down the transport read loop.
          this.debug(`MCP notification handler for ${this.config.name} threw: ${errorMessage(error)}`)
        }
      }
      return
    }
    void this.send(mcpJsonRpcFailure(value.id, -32601, `Client does not implement ${String(value.method)}`)).catch(() => {
      // The transport has already closed, so the inbound request cannot be answered.
    })
  }

  private handleProcessExit(process: Bun.PipedSubprocess, exitCode: number): void {
    if (this.process !== process || this.closing) {
      return
    }
    this.connected = false
    const stderr = this.stderrOutput.trim()
    const detail = stderr ? `: ${stderr}` : ''
    this.rejectPending(new MCPConnectionError(`MCP server ${this.config.name} exited with code ${exitCode}${detail}`))
  }

  private rejectPending(error: Error): void {
    for (const id of [...this.pending.keys()]) {
      this.dropPendingRequest(id)?.reject(error)
    }
  }

  private rejectPendingRequest(id: MCPJsonRpcId, error: Error): void {
    this.dropPendingRequest(id)?.reject(error)
  }

  private dropPendingRequest(id: MCPJsonRpcId): PendingRequest | undefined {
    const pending = this.pending.get(id)
    if (!pending) {
      return undefined
    }
    this.pending.delete(id)
    clearTimeout(pending.timeout)
    pending.detach?.()
    return pending
  }

  private debug(message: string): void {
    try {
      this.options.debug?.(message)
    } catch {
      // Diagnostics must never alter transport behavior.
    }
  }

  private handleHttpTransportClosed(error: Error): void {
    if (this.closing) {
      return
    }
    this.connected = false
    this.rejectPending(error)
  }

  private get timeoutMs(): number {
    const timeout = this.config.timeoutMs ?? DEFAULT_TIMEOUT_MS
    if (!Number.isFinite(timeout) || timeout <= 0) {
      throw new MCPConnectionError('MCP timeoutMs must be a positive finite number')
    }
    return timeout
  }
}

function parseInitializeResult(value: MCPJsonRpcResult): MCPInitializeResult {
  if (typeof value.protocolVersion !== 'string' || !value.protocolVersion.trim()) {
    throw new MCPProtocolError('MCP initialize response has no protocolVersion')
  }
  if (!isMCPRecord(value.capabilities)) {
    throw new MCPProtocolError('MCP initialize response has no capabilities object')
  }
  if (!isMCPRecord(value.serverInfo)) {
    throw new MCPProtocolError('MCP initialize response has no serverInfo object')
  }
  const serverInfo = parseImplementation(value.serverInfo, 'serverInfo')
  const instructions = value.instructions
  if (instructions !== undefined && typeof instructions !== 'string') {
    throw new MCPProtocolError('MCP initialize response has invalid instructions')
  }
  return {
    protocolVersion: value.protocolVersion,
    capabilities: value.capabilities,
    serverInfo,
    ...(instructions === undefined ? {} : { instructions }),
  }
}

function defaultProtocolVersion(transport: 'stdio' | MCPHttpTransportKind): string {
  return transport === 'streamable_http' ? MCP_STREAMABLE_HTTP_PROTOCOL_VERSION : MCP_PROTOCOL_VERSION
}

function parseTool(value: unknown, serverName: string): MCPTool {
  if (!isMCPRecord(value) || typeof value.name !== 'string') {
    throw new MCPProtocolError('MCP tools/list includes an invalid tool')
  }
  const description = value.description
  if (description !== undefined && typeof description !== 'string') {
    throw new MCPProtocolError(`MCP tool ${value.name} has an invalid description`)
  }
  const inputSchema = value.inputSchema
  if (inputSchema !== undefined && !isMCPRecord(inputSchema)) {
    throw new MCPProtocolError(`MCP tool ${value.name} has an invalid inputSchema`)
  }
  const annotations = value.annotations
  if (annotations !== undefined && !isMCPRecord(annotations)) {
    throw new MCPProtocolError(`MCP tool ${value.name} has invalid annotations`)
  }
  return {
    name: value.name,
    inputSchema: (inputSchema ?? {}) as JsonSchema,
    serverName,
    ...(description === undefined ? {} : { description }),
    ...(annotations === undefined ? {} : { annotations: annotations as MCPToolAnnotations }),
  }
}

function parseResource(value: unknown, serverName: string): MCPResource {
  if (!isMCPRecord(value) || typeof value.uri !== 'string' || typeof value.name !== 'string') {
    throw new MCPProtocolError('MCP resources/list includes an invalid resource')
  }
  const description = value.description
  const mimeType = value.mimeType
  if (description !== undefined && typeof description !== 'string') {
    throw new MCPProtocolError(`MCP resource ${value.uri} has an invalid description`)
  }
  if (mimeType !== undefined && typeof mimeType !== 'string') {
    throw new MCPProtocolError(`MCP resource ${value.uri} has an invalid mimeType`)
  }
  return {
    uri: value.uri,
    name: value.name,
    serverName,
    ...(description === undefined ? {} : { description }),
    ...(mimeType === undefined ? {} : { mimeType }),
  }
}

function parsePrompt(value: unknown, serverName: string): MCPPrompt {
  if (!isMCPRecord(value) || typeof value.name !== 'string') {
    throw new MCPProtocolError('MCP prompts/list includes an invalid prompt')
  }
  const description = value.description
  const arguments_ = value.arguments
  if (description !== undefined && typeof description !== 'string') {
    throw new MCPProtocolError(`MCP prompt ${value.name} has an invalid description`)
  }
  if (arguments_ !== undefined && !Array.isArray(arguments_)) {
    throw new MCPProtocolError(`MCP prompt ${value.name} has invalid arguments`)
  }
  return {
    name: value.name,
    serverName,
    ...(description === undefined ? {} : { description }),
    ...(arguments_ === undefined ? {} : { arguments: arguments_.map(parsePromptArgument) }),
  }
}

function parsePromptArgument(value: unknown): MCPPromptArgument {
  if (!isMCPRecord(value) || typeof value.name !== 'string') {
    throw new MCPProtocolError('MCP prompt has an invalid argument')
  }
  const description = value.description
  const required = value.required
  if (description !== undefined && typeof description !== 'string') {
    throw new MCPProtocolError(`MCP prompt argument ${value.name} has an invalid description`)
  }
  if (required !== undefined && typeof required !== 'boolean') {
    throw new MCPProtocolError(`MCP prompt argument ${value.name} has an invalid required flag`)
  }
  return {
    name: value.name,
    ...(description === undefined ? {} : { description }),
    ...(required === undefined ? {} : { required }),
  }
}

function parseImplementation(value: Record<string, unknown>, field: string): MCPImplementation {
  if (typeof value.name !== 'string' || typeof value.version !== 'string') {
    throw new MCPProtocolError(`MCP ${field} must include string name and version fields`)
  }
  return { name: value.name, version: value.version }
}

function parseRemoteError(value: Record<string, unknown>): MCPRemoteError {
  if (typeof value.code !== 'number' || typeof value.message !== 'string') {
    return new MCPRemoteError({ code: -32603, message: 'MCP server returned a malformed error object' })
  }
  const error: MCPJsonRpcErrorData = value.data === undefined
    ? { code: value.code, message: value.message }
    : { code: value.code, message: value.message, data: value.data }
  return new MCPRemoteError(error)
}

function asConnectionError(error: unknown): MCPConnectionError {
  return error instanceof MCPConnectionError
    ? error
    : new MCPConnectionError(error instanceof Error ? error.message : String(error))
}

function asMcpError(error: unknown): Error {
  if (error instanceof MCPConnectionError || error instanceof MCPProtocolError) {
    return error
  }
  return asConnectionError(error)
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

/** Register a one-shot abort listener and return a detach callback. */
function onAbort(signal: AbortSignal, abort: () => void): () => void {
  signal.addEventListener('abort', abort, { once: true })
  return () => signal.removeEventListener('abort', abort)
}

function sleep(milliseconds: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, milliseconds))
}
