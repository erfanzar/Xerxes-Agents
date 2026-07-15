// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../../core/errors.js'
import { ToolRegistry, type ToolExecutionContext } from '../../executors/toolRegistry.js'
import { MCPClient } from '../../mcp/client.js'
import type { MCPContent } from '../../mcp/types.js'
import type { JsonObject, JsonValue, ToolDefinition } from '../../types/toolCalls.js'
import { optionalString, requiredString } from '../inputs.js'

/** Named MCP client collection shared by Claude-compatible MCP tool calls. */
export class MCPClientRegistry {
  private readonly clients = new Map<string, MCPClient>()

  register(client: MCPClient): void {
    const name = client.config.name.trim()
    if (!name) throw new ValidationError('mcp.server_name', 'must not be empty')
    this.clients.set(name, client)
  }

  get(name: string): MCPClient | undefined {
    return this.clients.get(name)
  }

  entries(): readonly [string, MCPClient][] {
    return Object.freeze([...this.clients.entries()].sort(([left], [right]) => left.localeCompare(right)))
  }
}

export interface MCPClientLookup {
  entries(): readonly [string, MCPClient][]
  get(name: string): MCPClient | undefined
}

export interface ClaudeMcpToolsOptions {
  readonly clients: MCPClientLookup
}

export const CLAUDE_MCP_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  definition('MCPTool', 'Call one tool exposed by a connected Model Context Protocol server.', {
    server_name: stringSchema('Configured MCP server name.'),
    tool_name: stringSchema('Published MCP tool name.'),
    arguments: { description: 'Tool arguments object or JSON string.', type: ['object', 'string'] },
  }, ['server_name', 'tool_name']),
  definition('ListMcpResourcesTool', 'List resources from one connected MCP server or all connected servers.', {
    server_name: stringSchema('Optional configured MCP server name.'),
  }),
  definition('ReadMcpResourceTool', 'Read one resource URI from a connected MCP server.', {
    server_name: stringSchema('Configured MCP server name.'),
    uri: stringSchema('Published MCP resource URI.'),
  }, ['server_name', 'uri']),
]

/** Register functional MCP tool and resource calls, replacing Python's placeholder implementation. */
export function registerClaudeMcpTools(
  registry: ToolRegistry,
  options: ClaudeMcpToolsOptions,
  agentId = 'default',
): readonly ToolDefinition[] {
  const adapter = new ClaudeMcpTools(options)
  for (const tool of CLAUDE_MCP_TOOL_DEFINITIONS) {
    registry.replace(tool, (inputs, context, signal) => adapter.execute(tool.function.name, inputs, context, signal), agentId)
  }
  return CLAUDE_MCP_TOOL_DEFINITIONS
}

/** Claude-shaped facade over the native asynchronous Bun MCP client. */
export class ClaudeMcpTools {
  constructor(private readonly options: ClaudeMcpToolsOptions) {}

  async execute(
    name: string,
    inputs: JsonObject,
    _context: ToolExecutionContext,
    signal?: AbortSignal,
  ): Promise<unknown> {
    if (signal?.aborted) throw signal.reason ?? new Error('MCP tool call cancelled')
    switch (name) {
      case 'MCPTool': return this.callTool(inputs, signal)
      case 'ListMcpResourcesTool': return this.listResources(inputs, signal)
      case 'ReadMcpResourceTool': return this.readResource(inputs, signal)
      default: throw new ValidationError('tool', 'is not handled by ClaudeMcpTools', name)
    }
  }

  private async callTool(inputs: JsonObject, signal?: AbortSignal): Promise<Record<string, unknown>> {
    const serverName = requiredString(inputs, 'server_name')
    const toolName = requiredString(inputs, 'tool_name')
    const client = this.requireClient(serverName)
    const result = await abortable(client.callTool(toolName, parseArguments(inputs.arguments)), signal)
    return {
      server_name: serverName,
      tool_name: toolName,
      content: result.content.map(contentWire),
      is_error: result.isError ?? false,
      structured_content: result.structuredContent ?? null,
    }
  }

  private async listResources(inputs: JsonObject, signal?: AbortSignal): Promise<readonly Record<string, unknown>[]> {
    const requested = optionalString(inputs, 'server_name')?.trim()
    const entries = requested ? [[requested, this.requireClient(requested)] as const] : this.options.clients.entries()
    const resources: Record<string, unknown>[] = []
    for (const [serverName, client] of entries) {
      const listed = await abortable(client.listResources(), signal)
      for (const resource of listed) {
        resources.push({
          server_name: serverName,
          uri: resource.uri,
          name: resource.name,
          description: resource.description ?? null,
          mime_type: resource.mimeType ?? null,
        })
      }
    }
    return resources
  }

  private async readResource(inputs: JsonObject, signal?: AbortSignal): Promise<Record<string, unknown>> {
    const serverName = requiredString(inputs, 'server_name')
    const uri = requiredString(inputs, 'uri')
    const result = await abortable(this.requireClient(serverName).readResource(uri), signal)
    return {
      server_name: serverName,
      uri,
      contents: result.contents.map(content => ({
        uri: content.uri,
        mime_type: content.mimeType ?? null,
        text: content.text ?? null,
      })),
    }
  }

  private requireClient(serverName: string): MCPClient {
    const client = this.options.clients.get(serverName)
    if (client === undefined) {
      const known = this.options.clients.entries().map(([name]) => name)
      throw new ValidationError(
        'server_name',
        known.length ? `is not configured; available servers: ${known.join(', ')}` : 'is not configured and no MCP clients are attached',
        serverName,
      )
    }
    return client
  }
}

function definition(
  name: string,
  description: string,
  properties: Record<string, unknown>,
  required: readonly string[] = [],
): ToolDefinition {
  return {
    type: 'function',
    function: {
      name,
      description,
      parameters: { type: 'object', additionalProperties: false, properties, ...(required.length ? { required } : {}) },
    },
  }
}

function stringSchema(description: string): Record<string, unknown> {
  return { type: 'string', description }
}

function parseArguments(value: JsonValue | undefined): JsonObject {
  if (value === undefined || value === null) return {}
  if (isJsonObject(value)) return value
  if (typeof value === 'string') {
    try {
      const parsed = JSON.parse(value) as unknown
      if (isJsonObject(parsed)) return parsed
    } catch {
      // The validation error below is more useful to the model.
    }
  }
  throw new ValidationError('arguments', 'must be an object or JSON-encoded object', value)
}

function contentWire(content: MCPContent): Record<string, unknown> {
  switch (content.type) {
    case 'text': return { type: content.type, text: content.text }
    case 'image': return { type: content.type, data: content.data, mime_type: content.mimeType }
    case 'resource': return {
      type: content.type,
      resource: {
        uri: content.resource.uri,
        mime_type: content.resource.mimeType ?? null,
        text: content.resource.text ?? null,
      },
    }
  }
}

function isJsonObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

async function abortable<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (signal?.aborted) throw signal.reason ?? new Error('MCP tool call cancelled')
  if (signal === undefined) return promise
  return new Promise<T>((resolve, reject) => {
    const abort = () => reject(signal.reason ?? new Error('MCP tool call cancelled'))
    signal.addEventListener('abort', abort, { once: true })
    void promise.then(
      value => {
        signal.removeEventListener('abort', abort)
        resolve(value)
      },
      error => {
        signal.removeEventListener('abort', abort)
        reject(error)
      },
    )
  })
}
