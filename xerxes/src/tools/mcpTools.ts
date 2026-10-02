// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ToolRegistry } from '../executors/toolRegistry.js'
import type { MCPManager } from '../mcp/manager.js'
import type { AgentDefinition } from '../agents/definitions.js'

/** Stable provider-safe names retain exact server/tool identity even after sanitizing. */
export function mcpRuntimeToolName(server: string, tool: string): string {
  const clean = (value: string, length: number) => value.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, length)
  const hash = new Bun.CryptoHasher('sha256').update(JSON.stringify([server, tool])).digest('hex').slice(0, 12)
  return `mcp__${clean(server, 18)}__${clean(tool, 22)}_${hash}`
}

/** Whether a tool name was minted by {@link mcpRuntimeToolName}. */
export function isMcpRuntimeToolName(name: string): boolean {
  return name.startsWith('mcp__')
}

/** Publish connected server schemas through the ordinary validation, policy and execution path. */
export function registerMcpTools(registry: ToolRegistry, manager: MCPManager): string[] {
  const names: string[] = []
  for (const server of manager.listServers()) {
    const client = manager.getServer(server)
    if (!client || client.connected === false) continue
    for (const tool of client.tools) {
      const name = mcpRuntimeToolName(server, tool.name)
      names.push(name)
      registry.register({ type: 'function', function: {
        name,
        description: `MCP ${server} / ${tool.name}. ${tool.description ?? ''}`,
        parameters: tool.inputSchema,
      } }, async (args, context, signal) => {
        if (!context.sessionId?.trim()) throw new Error('MCP tools require an authenticated session context')
        const result = await manager.callServerTool(server, client, tool.name, args, signal ? { signal } : {})
        if (result.isError) throw new Error(`MCP tool failed: ${JSON.stringify(result)}`)
        return JSON.stringify(result)
      })
      // Server annotations are hints, not authority: retain conservative native
      // defaults for writes, external access, concurrency and cancellation.
    }
  }
  return names
}

/** Extend shipped full-access profiles, never user profiles or restricted-mode ceilings. */
export function addMcpToolsToBuiltinAgents(definitions: Map<string, AgentDefinition>, names: readonly string[]): void {
  for (const name of ['default', 'creator']) {
    const definition = definitions.get(name)
    if (definition?.source === 'built-in') definitions.set(name, {
      ...definition, tools: [...new Set([...definition.tools, ...names])],
    })
  }
}
