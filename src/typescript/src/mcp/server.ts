// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { errorMessage, ToolRegistry, type ToolExecutionContext } from '../executors/toolRegistry.js'
import type { JsonObject } from '../types/toolCalls.js'
import {
  MCP_JSON_RPC_ERRORS,
  MCP_PROTOCOL_VERSION,
  type MCPImplementation,
  type MCPJsonRpcResponse,
  type MCPJsonRpcResult,
  type MCPServerCapabilities,
  type MCPTool,
  isMCPJsonObject,
  isMCPJsonRpcId,
  isMCPRecord,
  mcpJsonRpcFailure,
  mcpJsonRpcSuccess,
} from './types.js'

export interface MCPToolServerOptions {
  readonly agentId?: string
  readonly contextFactory?: (toolName: string, arguments_: JsonObject) => ToolExecutionContext
  readonly instructions?: string
  readonly protocolVersion?: string
  readonly serverInfo?: MCPImplementation
}

export type MCPStdioWriter = (line: string) => void | Promise<void>

const DEFAULT_SERVER_INFO: MCPImplementation = { name: 'xerxes', version: '0.3.0' }
const TOOL_SERVER_CAPABILITIES: MCPServerCapabilities = { tools: { listChanged: false } }

/**
 * Minimal MCP server facade backed by the runtime's ToolRegistry.
 *
 * It exposes the registry's OpenAI-style function definitions through MCP's tools/list and
 * converts tools/call output into standard text content. A failed tool call is returned as an
 * MCP tool result with `isError`, rather than a protocol error, so an MCP client can let its
 * model correct the invocation on a follow-up turn.
 */
export class MCPToolServer {
  private readonly options: MCPToolServerOptions
  private readonly registry: ToolRegistry

  constructor(registry: ToolRegistry, options: MCPToolServerOptions = {}) {
    this.registry = registry
    this.options = options
  }

  /** Map registered Xerxes tools into MCP tool definitions. */
  listTools(): readonly MCPTool[] {
    return this.registry.definitions(this.options.agentId).map(definition => ({
      name: definition.function.name,
      description: definition.function.description,
      inputSchema: definition.function.parameters,
    }))
  }

  /** Process one parsed JSON-RPC frame. Notifications intentionally return no frame. */
  async handle(value: unknown): Promise<MCPJsonRpcResponse | undefined> {
    if (!isMCPRecord(value) || value.jsonrpc !== '2.0' || typeof value.method !== 'string') {
      return mcpJsonRpcFailure(null, MCP_JSON_RPC_ERRORS.invalidRequest, 'Invalid JSON-RPC request')
    }

    const hasId = Object.hasOwn(value, 'id')
    if (hasId && !isMCPJsonRpcId(value.id)) {
      return mcpJsonRpcFailure(null, MCP_JSON_RPC_ERRORS.invalidRequest, 'JSON-RPC request id must be a string or number')
    }
    if (value.params !== undefined && !isMCPRecord(value.params)) {
      return hasId
        ? mcpJsonRpcFailure(value.id as string | number, MCP_JSON_RPC_ERRORS.invalidParams, 'params must be an object')
        : undefined
    }

    if (!hasId) {
      return undefined
    }
    const id = value.id as string | number
    const params = (value.params ?? {}) as MCPJsonRpcResult
    switch (value.method) {
      case 'initialize':
        return mcpJsonRpcSuccess(id, this.initializeResult())
      case 'tools/list':
        return mcpJsonRpcSuccess(id, { tools: this.listTools() })
      case 'tools/call':
        return this.callTool(id, params)
      default:
        return mcpJsonRpcFailure(id, MCP_JSON_RPC_ERRORS.methodNotFound, `Method not found: ${value.method}`)
    }
  }

  /** Parse and process one newline-delimited JSON-RPC frame. */
  async handleLine(line: string): Promise<string | undefined> {
    let value: unknown
    try {
      value = JSON.parse(line) as unknown
    } catch {
      return JSON.stringify(mcpJsonRpcFailure(null, MCP_JSON_RPC_ERRORS.parseError, 'Invalid JSON'))
    }
    try {
      const response = await this.handle(value)
      return response ? JSON.stringify(response) : undefined
    } catch (error) {
      const id = isMCPRecord(value) && isMCPJsonRpcId(value.id) ? value.id : null
      return JSON.stringify(mcpJsonRpcFailure(id, MCP_JSON_RPC_ERRORS.internalError, errorMessage(error)))
    }
  }

  private initializeResult(): MCPJsonRpcResult {
    const protocolVersion = this.options.protocolVersion ?? MCP_PROTOCOL_VERSION
    const serverInfo = this.options.serverInfo ?? DEFAULT_SERVER_INFO
    return {
      protocolVersion,
      capabilities: TOOL_SERVER_CAPABILITIES,
      serverInfo,
      ...(this.options.instructions === undefined ? {} : { instructions: this.options.instructions }),
    }
  }

  private async callTool(id: string | number, params: MCPJsonRpcResult): Promise<MCPJsonRpcResponse> {
    if (typeof params.name !== 'string' || !params.name) {
      return mcpJsonRpcFailure(id, MCP_JSON_RPC_ERRORS.invalidParams, 'tools/call requires a non-empty name')
    }
    const arguments_ = params.arguments ?? {}
    if (!isMCPJsonObject(arguments_)) {
      return mcpJsonRpcFailure(id, MCP_JSON_RPC_ERRORS.invalidParams, 'tools/call arguments must be an object')
    }

    try {
      const result = await this.registry.execute(
        {
          id: crypto.randomUUID(),
          type: 'function',
          function: { name: params.name, arguments: arguments_ },
        },
        this.executionContext(params.name, arguments_),
      )
      return mcpJsonRpcSuccess(id, { content: [{ type: 'text', text: result }] })
    } catch (error) {
      return mcpJsonRpcSuccess(id, {
        content: [{ type: 'text', text: errorMessage(error) }],
        isError: true,
      })
    }
  }

  private executionContext(toolName: string, arguments_: JsonObject): ToolExecutionContext {
    const fromFactory = this.options.contextFactory?.(toolName, arguments_)
    if (fromFactory) {
      return fromFactory
    }
    return this.options.agentId === undefined
      ? { metadata: {} }
      : { agentId: this.options.agentId, metadata: {} }
  }
}

/**
 * Run a tool-backed MCP server over newline-delimited stdio streams.
 *
 * Keeping the writer injected makes this usable from a Bun executable, tests, or another
 * transport adapter without embedding process-global stdin/stdout inside the protocol facade.
 */
export async function serveMCPStdio(
  server: MCPToolServer,
  input: ReadableStream<Uint8Array>,
  write: MCPStdioWriter,
): Promise<void> {
  const reader = input.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) {
        break
      }
      buffer += decoder.decode(value, { stream: true })
      buffer = await respondToLines(server, buffer, write)
    }
    buffer += decoder.decode()
    if (buffer.trim()) {
      await respond(server, buffer.trim(), write)
    }
  } finally {
    reader.releaseLock()
  }
}

async function respondToLines(server: MCPToolServer, buffer: string, write: MCPStdioWriter): Promise<string> {
  let newline = buffer.indexOf('\n')
  while (newline >= 0) {
    const line = buffer.slice(0, newline).trim()
    buffer = buffer.slice(newline + 1)
    if (line) {
      await respond(server, line, write)
    }
    newline = buffer.indexOf('\n')
  }
  return buffer
}

async function respond(server: MCPToolServer, line: string, write: MCPStdioWriter): Promise<void> {
  const response = await server.handleLine(line)
  if (response) {
    await write(`${response}\n`)
  }
}
