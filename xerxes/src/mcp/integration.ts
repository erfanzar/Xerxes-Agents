// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ToolRegistry, type ToolHandler } from '../executors/toolRegistry.js'
import type { JsonObject, JsonSchema, ToolDefinition } from '../types/toolCalls.js'
import type { MCPTool, MCPToolCallResult } from './types.js'

/** The portion of MCPManager needed to discover and asynchronously route published tools. */
export interface MCPToolManagerPort {
  callTool(name: string, arguments_?: JsonObject): Promise<MCPToolCallResult>
  getAllTools(): readonly MCPTool[]
}

/** One native ToolRegistry definition and handler derived from an MCP tool record. */
export interface MCPToolAdapter {
  readonly definition: ToolDefinition
  readonly handler: ToolHandler
  readonly serverName: string | undefined
}

export interface RegisterMcpToolsOptions {
  /** Target one agent-specific ToolRegistry namespace. */
  readonly agentId?: string
  /** Restrict registration to these connected MCP server names. An empty list keeps all servers. */
  readonly serverNames?: readonly string[]
}

export interface McpToolRegistrationSkip {
  readonly name: string
  readonly reason: 'duplicate-registry-tool' | 'duplicate-source-tool'
  readonly serverName?: string
}

export interface McpToolRegistrationResult {
  /** Definitions that were newly registered in discovery order. */
  readonly registered: readonly ToolDefinition[]
  /** Tools deliberately left unchanged to preserve the existing registry implementation. */
  readonly skipped: readonly McpToolRegistrationSkip[]
}

/** Raised when an MCP record cannot safely map to a native tool or cannot be routed at runtime. */
export class MCPToolIntegrationError extends Error {
  readonly serverName: string | undefined
  readonly toolName: string

  constructor(toolName: string, message: string, serverName?: string) {
    const source = serverName === undefined ? '' : ` from server ${JSON.stringify(serverName)}`
    super(`MCP tool ${JSON.stringify(toolName)}${source}: ${message}`)
    this.name = 'MCPToolIntegrationError'
    this.toolName = toolName
    this.serverName = serverName
  }
}

/** Map an MCP tool's JSON Schema directly onto Xerxes' OpenAI-style function definition. */
export function mcpToolToToolDefinition(tool: MCPTool): ToolDefinition {
  const identity = toolIdentity(tool)
  const description = descriptionFor(tool, identity)
  return {
    type: 'function',
    function: {
      name: identity.name,
      description,
      parameters: { ...inputSchema(tool, identity) },
    },
  }
}

/** Build a native async ToolHandler that routes arguments through MCPManager.callTool. */
export function mcpToolToToolHandler(tool: MCPTool, manager: MCPToolManagerPort): ToolHandler {
  const identity = toolIdentity(tool)
  return async (inputs, _context, signal) => {
    if (signal?.aborted) {
      throw new MCPToolIntegrationError(identity.name, 'call was cancelled before routing', identity.serverName)
    }
    try {
      const pending = Promise.resolve().then(() => manager.callTool(identity.name, inputs))
      return await awaitAbortable(pending, signal)
    } catch (error) {
      if (error instanceof MCPToolIntegrationError) throw error
      if (signal?.aborted) {
        throw new MCPToolIntegrationError(identity.name, 'call was cancelled while routing', identity.serverName)
      }
      throw new MCPToolIntegrationError(
        identity.name,
        'could not route call: ' + errorMessage(error),
        identity.serverName,
      )
    }
  }
}

/** Build a definition/handler pair without Python-style function reflection or synchronous wrappers. */
export function adaptMcpTool(tool: MCPTool, manager: MCPToolManagerPort): MCPToolAdapter {
  const identity = toolIdentity(tool)
  return {
    definition: mcpToolToToolDefinition(tool),
    handler: mcpToolToToolHandler(tool, manager),
    serverName: identity.serverName,
  }
}

/**
 * Register currently discovered MCP tools with one ToolRegistry agent namespace.
 *
 * Existing resolved names are never replaced, so repeated registration and a collision with a
 * native tool leave the prior implementation reachable. MCPManager already resolves duplicate
 * remote names first-server-wins; this guard also protects custom manager ports that do not.
 */
export function registerMcpTools(
  registry: ToolRegistry,
  manager: MCPToolManagerPort,
  options: RegisterMcpToolsOptions = {},
): McpToolRegistrationResult {
  const agentId = options.agentId ?? 'default'
  const serverFilter = normalizeServerFilter(options.serverNames)
  const existingNames = new Set(registry.definitions(agentId).map(definition => definition.function.name))
  const sourceNames = new Set<string>()
  const candidates: MCPToolAdapter[] = []
  const skipped: McpToolRegistrationSkip[] = []

  for (const tool of manager.getAllTools()) {
    const identity = toolIdentity(tool)
    if (serverFilter !== undefined && (identity.serverName === undefined || !serverFilter.has(identity.serverName))) {
      continue
    }
    const definition = mcpToolToToolDefinition(tool)
    if (sourceNames.has(identity.name)) {
      skipped.push(skip(identity, 'duplicate-source-tool'))
      continue
    }
    sourceNames.add(identity.name)
    if (existingNames.has(identity.name)) {
      skipped.push(skip(identity, 'duplicate-registry-tool'))
      continue
    }
    existingNames.add(identity.name)
    candidates.push({
      definition,
      handler: mcpToolToToolHandler(tool, manager),
      serverName: identity.serverName,
    })
  }

  for (const candidate of candidates) {
    registry.register(candidate.definition, candidate.handler, agentId)
  }
  return {
    registered: candidates.map(candidate => candidate.definition),
    skipped,
  }
}

interface ToolIdentity {
  readonly name: string
  readonly serverName: string | undefined
}

function toolIdentity(tool: MCPTool): ToolIdentity {
  if (typeof tool !== 'object' || tool === null || typeof tool.name !== 'string') {
    throw new MCPToolIntegrationError('<unknown>', 'record must include a string name')
  }
  if (!tool.name || tool.name !== tool.name.trim() || tool.name.includes('\0')) {
    throw new MCPToolIntegrationError(String(tool.name), 'name must be non-empty, trimmed, and contain no null bytes')
  }
  const serverName = tool.serverName
  const hasInvalidServerName = serverName !== undefined
    && (typeof serverName !== 'string' || !serverName.trim() || serverName !== serverName.trim())
  if (hasInvalidServerName) {
    throw new MCPToolIntegrationError(tool.name, 'server name must be a non-empty trimmed string')
  }
  return { name: tool.name, serverName }
}

function inputSchema(tool: MCPTool, identity: ToolIdentity): JsonSchema {
  const schema: unknown = tool.inputSchema
  if (typeof schema !== 'object' || schema === null || Array.isArray(schema)) {
    throw new MCPToolIntegrationError(identity.name, 'input schema must be a JSON object', identity.serverName)
  }
  return schema as JsonSchema
}

function descriptionFor(tool: MCPTool, identity: ToolIdentity): string {
  if (tool.description !== undefined && typeof tool.description !== 'string') {
    throw new MCPToolIntegrationError(identity.name, 'description must be a string', identity.serverName)
  }
  const description = tool.description?.trim()
  if (description) return description
  const source = identity.serverName === undefined ? 'an MCP server' : `MCP server ${identity.serverName}`
  return `Tool ${identity.name} supplied by ${source}.`
}

function normalizeServerFilter(serverNames: readonly string[] | undefined): ReadonlySet<string> | undefined {
  if (serverNames === undefined || serverNames.length === 0) return undefined
  const names = new Set<string>()
  for (const serverName of serverNames) {
    if (typeof serverName !== 'string' || !serverName.trim()) {
      throw new MCPToolIntegrationError('<server filter>', 'server names must be non-empty strings')
    }
    names.add(serverName.trim())
  }
  return names
}

function skip(identity: ToolIdentity, reason: McpToolRegistrationSkip['reason']): McpToolRegistrationSkip {
  return {
    name: identity.name,
    reason,
    ...(identity.serverName === undefined ? {} : { serverName: identity.serverName }),
  }
}

async function awaitAbortable<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
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

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
