// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { JsonObject, JsonSchema, JsonValue } from '../types/toolCalls.js'

/** MCP revision used by the Python runtime's stdio client. */
export const MCP_PROTOCOL_VERSION = '2024-11-05'

/** Default revision for the Streamable HTTP transport introduced after 2024-11-05. */
export const MCP_STREAMABLE_HTTP_PROTOCOL_VERSION = '2025-06-18'

/** JSON-RPC errors reserved by the JSON-RPC 2.0 specification. */
export const MCP_JSON_RPC_ERRORS = {
  internalError: -32603,
  invalidParams: -32602,
  invalidRequest: -32600,
  methodNotFound: -32601,
  parseError: -32700,
} as const

export type MCPTransport = 'stdio' | 'sse' | 'streamable_http'
export type MCPHttpTransportKind = Exclude<MCPTransport, 'stdio'>
export type MCPJsonRpcId = number | string
export type MCPJsonRpcResult = Record<string, unknown>

/** Injectable HTTP primitive for MCP HTTP transports. */
export type MCPFetch = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>

/** Optional runtime dependencies for one MCP client. */
export interface MCPClientOptions {
  readonly fetch?: MCPFetch
  /** Receives diagnostics for skipped malformed server output. Defaults to silent. */
  readonly debug?: (message: string) => void
}

export interface MCPImplementation {
  readonly name: string
  readonly version: string
}

/** Launch and connection settings for a single MCP server. */
export interface MCPServerConfig {
  readonly name: string
  readonly command?: string
  readonly args?: readonly string[]
  readonly env?: Readonly<Record<string, string>>
  /** `sse` is the legacy 2024-11-05 HTTP+SSE transport; prefer `streamable_http` for remote servers. */
  readonly transport?: MCPTransport
  /** Absolute HTTP(S) endpoint required by the HTTP transports. */
  readonly url?: string
  /** Additional HTTP headers, such as Authorization. Protocol-controlled headers are rejected. */
  readonly headers?: Readonly<Record<string, string>>
  readonly enabled?: boolean
  readonly timeoutMs?: number
  readonly protocolVersion?: string
  readonly clientInfo?: MCPImplementation
  readonly clientCapabilities?: MCPCapabilities
}

/** Server capability data deliberately remains extensible across MCP revisions. */
export type MCPServerCapabilities = Readonly<Record<string, unknown>>
export type MCPCapabilities = Readonly<Record<string, unknown>>

export interface MCPToolAnnotations {
  readonly destructiveHint?: boolean
  readonly idempotentHint?: boolean
  readonly openWorldHint?: boolean
  readonly readOnlyHint?: boolean
  readonly title?: string
}

/** A callable capability published by an MCP server. */
export interface MCPTool {
  readonly name: string
  readonly description?: string
  readonly inputSchema: JsonSchema
  readonly annotations?: MCPToolAnnotations
  readonly serverName?: string
}

/** A resource capability published by an MCP server. */
export interface MCPResource {
  readonly uri: string
  readonly name: string
  readonly description?: string
  readonly mimeType?: string
  readonly serverName?: string
}

export interface MCPPromptArgument {
  readonly description?: string
  readonly name: string
  readonly required?: boolean
}

/** A prompt capability published by an MCP server. */
export interface MCPPrompt {
  readonly name: string
  readonly description?: string
  readonly arguments?: readonly MCPPromptArgument[]
  readonly serverName?: string
}

export interface MCPTextContent {
  readonly type: 'text'
  readonly text: string
}

export interface MCPImageContent {
  readonly data: string
  readonly mimeType: string
  readonly type: 'image'
}

export interface MCPEmbeddedResourceContent {
  readonly resource: {
    readonly mimeType?: string
    readonly text?: string
    readonly uri: string
  }
  readonly type: 'resource'
}

export type MCPContent = MCPEmbeddedResourceContent | MCPImageContent | MCPTextContent

export interface MCPToolCallResult {
  readonly content: readonly MCPContent[]
  readonly isError?: boolean
  readonly structuredContent?: JsonObject
}

export interface MCPResourceContentsResult {
  readonly contents: readonly MCPEmbeddedResourceContent['resource'][]
}

export interface MCPPromptMessage {
  readonly content: MCPContent
  readonly role: 'assistant' | 'user'
}

export interface MCPPromptResult {
  readonly description?: string
  readonly messages: readonly MCPPromptMessage[]
}

export interface MCPInitializeResult {
  readonly capabilities: MCPServerCapabilities
  readonly instructions?: string
  readonly protocolVersion: string
  readonly serverInfo: MCPImplementation
}

export interface MCPJsonRpcRequest {
  readonly id: MCPJsonRpcId
  readonly jsonrpc: '2.0'
  readonly method: string
  readonly params?: MCPJsonRpcResult
}

export interface MCPJsonRpcNotification {
  readonly jsonrpc: '2.0'
  readonly method: string
  readonly params?: MCPJsonRpcResult
}

export interface MCPJsonRpcSuccess {
  readonly id: MCPJsonRpcId
  readonly jsonrpc: '2.0'
  readonly result: MCPJsonRpcResult
}

export interface MCPJsonRpcErrorData {
  readonly code: number
  readonly data?: unknown
  readonly message: string
}

export interface MCPJsonRpcFailure {
  readonly error: MCPJsonRpcErrorData
  readonly id: MCPJsonRpcId | null
  readonly jsonrpc: '2.0'
}

export type MCPJsonRpcInbound = MCPJsonRpcFailure | MCPJsonRpcNotification | MCPJsonRpcRequest | MCPJsonRpcSuccess
export type MCPJsonRpcResponse = MCPJsonRpcFailure | MCPJsonRpcSuccess

export class MCPConnectionError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'MCPConnectionError'
  }
}

export class MCPProtocolError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'MCPProtocolError'
  }
}

export class MCPRemoteError extends MCPProtocolError {
  readonly code: number
  readonly data: unknown

  constructor(error: MCPJsonRpcErrorData) {
    super(`MCP server error ${error.code}: ${error.message}`)
    this.name = 'MCPRemoteError'
    this.code = error.code
    this.data = error.data
  }
}

export function isMCPRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function isMCPJsonRpcId(value: unknown): value is MCPJsonRpcId {
  return typeof value === 'number' || typeof value === 'string'
}

export function isMCPJsonObject(value: unknown): value is JsonObject {
  return isMCPRecord(value) && Object.values(value).every(isJsonValue)
}

export function mcpJsonRpcFailure(
  id: MCPJsonRpcId | null,
  code: number,
  message: string,
  data?: unknown,
): MCPJsonRpcFailure {
  const error = data === undefined ? { code, message } : { code, message, data }
  return { jsonrpc: '2.0', id, error }
}

export function mcpJsonRpcSuccess(id: MCPJsonRpcId, result: MCPJsonRpcResult): MCPJsonRpcSuccess {
  return { jsonrpc: '2.0', id, result }
}

function isJsonValue(value: unknown): value is JsonValue {
  if (value === null || typeof value === 'boolean' || typeof value === 'number' || typeof value === 'string') {
    return true
  }
  if (Array.isArray(value)) {
    return value.every(isJsonValue)
  }
  return isMCPRecord(value) && Object.values(value).every(isJsonValue)
}
