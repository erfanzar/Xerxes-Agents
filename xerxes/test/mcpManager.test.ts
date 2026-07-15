// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  MCPCapabilityNotFoundError,
  MCPManager,
  type MCPClientPort,
} from '../src/mcp/manager.js'
import {
  MCPReconnectError,
  ReconnectPolicy,
  reconnectWithBackoff,
  scrubCredentials,
} from '../src/mcp/reconnect.js'
import type {
  MCPPrompt,
  MCPPromptResult,
  MCPResource,
  MCPResourceContentsResult,
  MCPServerConfig,
  MCPTool,
  MCPToolCallResult,
} from '../src/mcp/types.js'
import type { JsonObject } from '../src/types/toolCalls.js'

interface ClientFixture {
  readonly prompts?: readonly MCPPrompt[]
  readonly resources?: readonly MCPResource[]
  readonly tools?: readonly MCPTool[]
  readonly onConnect?: () => void | Promise<void>
  readonly onDisconnect?: () => void | Promise<void>
}

class FakeMCPClient implements MCPClientPort {
  readonly calls: Array<{ readonly arguments_: JsonObject; readonly name: string }> = []
  readonly config: MCPServerConfig
  readonly prompts: readonly MCPPrompt[]
  readonly resources: readonly MCPResource[]
  readonly tools: readonly MCPTool[]
  connected = false
  disconnects = 0

  constructor(config: MCPServerConfig, private readonly fixture: ClientFixture = {}) {
    this.config = config
    this.prompts = fixture.prompts ?? []
    this.resources = fixture.resources ?? []
    this.tools = fixture.tools ?? []
  }

  async connect(): Promise<void> {
    await this.fixture.onConnect?.()
    this.connected = true
  }

  async disconnect(): Promise<void> {
    this.disconnects += 1
    await this.fixture.onDisconnect?.()
    this.connected = false
  }

  async callTool(name: string, arguments_: JsonObject = {}): Promise<MCPToolCallResult> {
    this.calls.push({ name, arguments_ })
    return { content: [{ type: 'text', text: this.config.name + ':' + name }] }
  }

  async readResource(uri: string): Promise<MCPResourceContentsResult> {
    return { contents: [{ uri, text: this.config.name + ':resource' }] }
  }

  async getPrompt(name: string, arguments_: JsonObject = {}): Promise<MCPPromptResult> {
    return {
      messages: [{
        role: 'user',
        content: { type: 'text', text: this.config.name + ':' + name + ':' + String(arguments_.name) },
      }],
    }
  }
}

const ALPHA_TOOLS: readonly MCPTool[] = [
  { name: 'echo', description: 'First echo', inputSchema: { type: 'object' } },
  { name: 'alpha_only', inputSchema: { type: 'object' } },
]
const BETA_TOOLS: readonly MCPTool[] = [
  { name: 'echo', description: 'Second echo', inputSchema: { type: 'object' } },
  { name: 'beta_only', inputSchema: { type: 'object' } },
]

test('MCPManager owns multiple live clients, deduplicates tools, and routes capabilities', async () => {
  const clients: FakeMCPClient[] = []
  const manager = new MCPManager({
    clientFactory: config => {
      const client = new FakeMCPClient(config, config.name === 'alpha'
        ? {
            tools: ALPHA_TOOLS,
            resources: [{ uri: 'memo://alpha', name: 'Alpha' }],
            prompts: [{ name: 'brief' }],
          }
        : {
            tools: BETA_TOOLS,
            resources: [{ uri: 'memo://beta', name: 'Beta' }],
            prompts: [{ name: 'brief' }, { name: 'beta_prompt' }],
          })
      clients.push(client)
      return client
    },
  })

  expect(await manager.addServer({ name: 'alpha' })).toBeTrue()
  expect(await manager.start({ name: 'beta' })).toBeTrue()
  expect(await manager.addServer({ name: 'alpha' })).toBeFalse()
  expect(await manager.addServer({ name: 'off', enabled: false })).toBeFalse()
  expect(clients).toHaveLength(2)
  expect(manager.listServers()).toEqual(['alpha', 'beta'])
  expect(manager.getAllTools()).toEqual([
    { name: 'echo', description: 'First echo', inputSchema: { type: 'object' }, serverName: 'alpha' },
    { name: 'alpha_only', inputSchema: { type: 'object' }, serverName: 'alpha' },
    { name: 'beta_only', inputSchema: { type: 'object' }, serverName: 'beta' },
  ])
  expect(manager.getAllResources()).toEqual([
    { uri: 'memo://alpha', name: 'Alpha', serverName: 'alpha' },
    { uri: 'memo://beta', name: 'Beta', serverName: 'beta' },
  ])
  expect(manager.getAllPrompts()).toEqual([
    { name: 'brief', serverName: 'alpha' },
    { name: 'brief', serverName: 'beta' },
    { name: 'beta_prompt', serverName: 'beta' },
  ])
  expect(manager.getCapabilitiesSummary()).toEqual({
    alpha: { tools: 2, resources: 1, prompts: 1 },
    beta: { tools: 2, resources: 1, prompts: 2 },
  })
  expect(manager.status('beta')).toEqual({
    name: 'beta',
    connected: true,
    tools: 2,
    resources: 1,
    prompts: 2,
  })

  await expect(manager.callTool('echo', { message: 'hello' })).resolves.toEqual({
    content: [{ type: 'text', text: 'alpha:echo' }],
  })
  await expect(manager.readResource('memo://beta')).resolves.toEqual({
    contents: [{ uri: 'memo://beta', text: 'beta:resource' }],
  })
  await expect(manager.getPrompt('beta_prompt', { name: 'Ada' })).resolves.toEqual({
    messages: [{ role: 'user', content: { type: 'text', text: 'beta:beta_prompt:Ada' } }],
  })
  expect(clients[0]?.calls).toEqual([{ name: 'echo', arguments_: { message: 'hello' } }])
  await expect(manager.callTool('missing')).rejects.toBeInstanceOf(MCPCapabilityNotFoundError)

  expect(await manager.stop('alpha')).toBeTrue()
  expect(manager.getServer('alpha')).toBeUndefined()
  expect(clients[0]?.disconnects).toBe(1)
  await manager.stopAll()
  expect(manager.listServers()).toEqual([])
  expect(clients[1]?.disconnects).toBe(1)
})

test('MCPManager reconnects through fresh factory clients with injected backoff', async () => {
  const clients: FakeMCPClient[] = []
  const sleeps: number[] = []
  let connection = 0
  const manager = new MCPManager({
    clientFactory: config => {
      connection += 1
      const attempt = connection
      const client = new FakeMCPClient(config, {
        tools: [{ name: 'echo', inputSchema: { type: 'object' } }],
        onConnect: () => {
          if (attempt === 2 || attempt === 3) {
            throw new Error('authorization: bearer secret-token-value-' + attempt)
          }
        },
      })
      clients.push(client)
      return client
    },
    reconnect: {
      policy: new ReconnectPolicy({ maxAttempts: 4, baseSeconds: 1, factor: 2, maxSeconds: 8 }),
      sleep: seconds => {
        sleeps.push(seconds)
      },
    },
  })

  expect(await manager.addServer({ name: 'alpha' })).toBeTrue()
  expect(await manager.reconnect('alpha')).toBeTrue()
  expect(sleeps).toEqual([1, 2])
  expect(clients).toHaveLength(4)
  expect(clients[0]?.disconnects).toBe(1)
  expect(manager.lastFailure('alpha')).toBeUndefined()
  expect(manager.getServer('alpha')).toBe(clients[3])
})

test('MCPManager removes an exhausted reconnect candidate and retains only a redacted failure', async () => {
  const sleeps: number[] = []
  let factoryCalls = 0
  const manager = new MCPManager({
    clientFactory: config => {
      factoryCalls += 1
      return new FakeMCPClient(config, {
        onConnect: () => {
          if (factoryCalls > 1) {
            throw new Error('api_key=super-secret-key-12345678')
          }
        },
      })
    },
    reconnect: {
      policy: { maxAttempts: 3, baseSeconds: 0.5, factor: 2, maxSeconds: 5 },
      sleep: seconds => {
        sleeps.push(seconds)
      },
    },
  })

  expect(await manager.addServer({ name: 'alpha' })).toBeTrue()
  expect(await manager.reconnect('alpha')).toBeFalse()
  expect(manager.getServer('alpha')).toBeUndefined()
  expect(manager.listServers()).toEqual([])
  expect(sleeps).toEqual([0.5, 1])
  expect(manager.lastFailure('alpha')).toEqual({
    name: 'alpha',
    operation: 'reconnect',
    attempt: 3,
    error: 'api_key=[redacted]',
  })
})

test('reconnectWithBackoff uses seconds, validates policy values, and scrubs terminal errors', async () => {
  const delays: number[] = []
  let attempts = 0
  await expect(reconnectWithBackoff(
    () => {
      attempts += 1
      if (attempts < 3) {
        throw new Error('token=very-secret-token-value-123')
      }
      return 'connected'
    },
    {
      policy: { maxAttempts: 4, baseSeconds: 2, factor: 3, maxSeconds: 5 },
      sleep: seconds => {
        delays.push(seconds)
      },
    },
  )).resolves.toBe('connected')
  expect(delays).toEqual([2, 5])
  expect(scrubCredentials('password hunter2 sk-abcdefghijklmnop')).toBe('password=[redacted] [redacted]')
  expect(() => new ReconnectPolicy({ maxAttempts: 0 })).toThrow(RangeError)

  await expect(reconnectWithBackoff(
    () => {
      throw new Error('authorization: bearer secret-token-value-123')
    },
    { policy: { maxAttempts: 1 } },
  )).rejects.toMatchObject({
    name: 'MCPReconnectError',
    attempts: 1,
    message: 'authorization: bearer=[redacted]',
  } satisfies Partial<MCPReconnectError>)
})
