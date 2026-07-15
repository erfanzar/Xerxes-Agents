// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import {
  MCPToolIntegrationError,
  mcpToolToToolDefinition,
  registerMcpTools,
  type MCPToolManagerPort,
} from '../src/mcp/integration.js'
import { MCPManager, type MCPClientPort } from '../src/mcp/manager.js'
import type {
  MCPPrompt,
  MCPPromptResult,
  MCPResource,
  MCPResourceContentsResult,
  MCPServerConfig,
  MCPTool,
  MCPToolCallResult,
} from '../src/mcp/types.js'
import type { JsonObject, JsonSchema, ToolCall } from '../src/types/toolCalls.js'

interface ClientFixture {
  readonly failure?: string
  readonly tools: readonly MCPTool[]
}

class IntegrationClient implements MCPClientPort {
  readonly calls: Array<{ readonly arguments_: JsonObject; readonly name: string }> = []
  readonly config: MCPServerConfig
  readonly prompts: readonly MCPPrompt[] = []
  readonly resources: readonly MCPResource[] = []
  readonly tools: readonly MCPTool[]
  connected = false

  constructor(config: MCPServerConfig, private readonly fixture: ClientFixture) {
    this.config = config
    this.tools = fixture.tools
  }

  async connect(): Promise<void> {
    this.connected = true
  }

  async disconnect(): Promise<void> {
    this.connected = false
  }

  async callTool(name: string, arguments_: JsonObject = {}): Promise<MCPToolCallResult> {
    this.calls.push({ name, arguments_ })
    if (this.fixture.failure) throw new Error(this.fixture.failure)
    return {
      content: [{ type: 'text', text: `${this.config.name}:${name}` }],
      structuredContent: { server: this.config.name, tool: name, arguments: arguments_ },
    }
  }

  async readResource(uri: string): Promise<MCPResourceContentsResult> {
    return { contents: [{ uri, text: '' }] }
  }

  async getPrompt(name: string): Promise<MCPPromptResult> {
    return { messages: [{ role: 'user', content: { type: 'text', text: name } }] }
  }
}

const WEATHER_SCHEMA: JsonSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    city: { type: 'string', description: 'City to inspect.' },
    days: { type: 'integer', minimum: 1 },
  },
  required: ['city'],
}

const ALPHA_TOOLS: readonly MCPTool[] = [
  { name: 'weather.lookup', description: 'Look up weather.', inputSchema: WEATHER_SCHEMA },
  { name: 'shared', inputSchema: { type: 'object' } },
  { name: 'native_collision', inputSchema: { type: 'object' } },
]

const BETA_TOOLS: readonly MCPTool[] = [
  { name: 'beta_only', description: 'Only beta has this tool.', inputSchema: { type: 'object' } },
  { name: 'shared', description: 'Second shared implementation.', inputSchema: { type: 'object' } },
]

function call(name: string, arguments_: JsonObject = {}): ToolCall {
  return { id: 'call-' + name, type: 'function', function: { name, arguments: arguments_ } }
}

test('MCP schema maps directly to native ToolRegistry definitions without reflection or name rewriting', () => {
  expect(mcpToolToToolDefinition({
    name: 'weather.lookup',
    description: 'Look up weather.',
    inputSchema: WEATHER_SCHEMA,
    serverName: 'alpha',
  })).toEqual({
    type: 'function',
    function: {
      name: 'weather.lookup',
      description: 'Look up weather.',
      parameters: WEATHER_SCHEMA,
    },
  })
})

test(
  'MCP integration filters servers, keeps first manager tools, and avoids duplicate registry registration',
  async () => {
    const clients = new Map<string, IntegrationClient>()
    const manager = new MCPManager({
      clientFactory: config => {
        const fixture = config.name === 'alpha' ? { tools: ALPHA_TOOLS } : { tools: BETA_TOOLS }
        const client = new IntegrationClient(config, fixture)
        clients.set(config.name, client)
        return client
      },
    })
    await manager.addServer({ name: 'alpha' })
    await manager.addServer({ name: 'beta' })
    try {
      const registry = new ToolRegistry()
      registry.register(
        {
          type: 'function',
          function: {
            name: 'native_collision',
            description: 'Native implementation.',
            parameters: { type: 'object' },
          },
        },
        () => 'native result',
      )

      const betaOnly = registerMcpTools(registry, manager, { serverNames: ['beta'] })
      expect(betaOnly.registered.map(tool => tool.function.name)).toEqual(['beta_only'])
      expect(registry.definitions().map(tool => tool.function.name)).toContain('beta_only')
      expect(registry.definitions().map(tool => tool.function.name)).not.toContain('weather.lookup')

      const allServers = registerMcpTools(registry, manager)
      expect(allServers.registered.map(tool => tool.function.name)).toEqual(['weather.lookup', 'shared'])
      expect(allServers.skipped).toEqual([
        { name: 'native_collision', reason: 'duplicate-registry-tool', serverName: 'alpha' },
        { name: 'beta_only', reason: 'duplicate-registry-tool', serverName: 'beta' },
      ])
      expect(manager.getAllTools().map(tool => tool.name)).toEqual([
        'weather.lookup',
        'shared',
        'native_collision',
        'beta_only',
      ])

      const repeated = registerMcpTools(registry, manager)
      expect(repeated.registered).toEqual([])
      expect(repeated.skipped).toHaveLength(4)
      expect(await registry.execute(call('native_collision'), { metadata: {} })).toBe('native result')

      const weather = await registry.execute(
        call('weather.lookup', { city: 'Istanbul', days: 2 }),
        { metadata: {} },
      )
      expect(JSON.parse(weather)).toEqual({
        content: [{ type: 'text', text: 'alpha:weather.lookup' }],
        structuredContent: {
          server: 'alpha',
          tool: 'weather.lookup',
          arguments: { city: 'Istanbul', days: 2 },
        },
      })
      expect(clients.get('alpha')?.calls).toEqual([
        { name: 'weather.lookup', arguments_: { city: 'Istanbul', days: 2 } },
      ])
    } finally {
      await manager.disconnectAll()
    }
  },
)

test('MCP tool failures and malformed published records expose clear integration errors', async () => {
  const failingManager: MCPToolManagerPort = {
    getAllTools: () => [{ name: 'failing_tool', inputSchema: { type: 'object' }, serverName: 'broken' }],
    callTool: async () => {
      throw new Error('remote service is unavailable')
    },
  }
  const registry = new ToolRegistry()
  registerMcpTools(registry, failingManager)
  await expect(registry.execute(call('failing_tool'), { metadata: {} })).rejects.toThrow(
    'MCP tool "failing_tool" from server "broken": could not route call: remote service is unavailable',
  )

  expect(() => mcpToolToToolDefinition({ name: '', inputSchema: {} })).toThrow(MCPToolIntegrationError)
  expect(() => mcpToolToToolDefinition({
    name: 'bad_schema',
    inputSchema: [] as unknown as JsonSchema,
    serverName: 'broken',
  })).toThrow('input schema must be a JSON object')
})
