// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { MCPClient } from '../src/mcp/client.js'
import { MCPManager, type MCPClientPort } from '../src/mcp/manager.js'
import { OAuthToken, refreshToken, type OAuthConfig, type OAuthFetch } from '../src/mcp/oauth.js'
import { MCPConnectionError } from '../src/mcp/types.js'
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

class FakeMCPClient implements MCPClientPort {
  readonly config: MCPServerConfig
  readonly prompts: readonly MCPPrompt[] = []
  readonly resources: readonly MCPResource[] = []
  readonly tools: readonly MCPTool[]
  connected = false
  disconnects = 0

  constructor(
    config: MCPServerConfig,
    private readonly onConnect?: () => void,
  ) {
    this.config = config
    this.tools = [{ name: 'echo', inputSchema: { type: 'object' } }]
  }

  async connect(): Promise<void> {
    this.onConnect?.()
    this.connected = true
  }

  async disconnect(): Promise<void> {
    this.disconnects += 1
    this.connected = false
  }

  async callTool(name: string): Promise<MCPToolCallResult> {
    return { content: [{ type: 'text', text: `${this.config.name}:${name}` }] }
  }

  async readResource(uri: string): Promise<MCPResourceContentsResult> {
    return { contents: [{ uri, text: '' }] }
  }

  async getPrompt(name: string): Promise<MCPPromptResult> {
    return { messages: [{ role: 'user', content: { type: 'text', text: name } }] }
  }
}

function sleep(milliseconds: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, milliseconds))
}

async function withinMs<T>(promise: Promise<T>, milliseconds: number): Promise<T | 'timed out'> {
  return Promise.race([
    promise,
    sleep(milliseconds).then(() => 'timed out' as const),
  ])
}

test('MCPManager reconnect backoff runs outside the lifecycle queue', async () => {
  const clients: FakeMCPClient[] = []
  let alphaConnections = 0
  let enteredSleep!: () => void
  let releaseSleep!: () => void
  const sleepStarted = new Promise<void>(resolve => {
    enteredSleep = resolve
  })
  const sleepGate = new Promise<void>(resolve => {
    releaseSleep = resolve
  })

  const manager = new MCPManager({
    clientFactory: config => {
      const client = new FakeMCPClient(config, () => {
        if (config.name === 'alpha') {
          alphaConnections += 1
          if (alphaConnections === 2) {
            throw new Error('transient connect failure')
          }
        }
      })
      clients.push(client)
      return client
    },
    reconnect: {
      policy: { maxAttempts: 3, baseSeconds: 30, factor: 1, maxSeconds: 30 },
      sleep: () => {
        enteredSleep()
        return sleepGate
      },
    },
  })

  expect(await manager.addServer({ name: 'alpha' })).toBeTrue()
  const reconnecting = manager.reconnect('alpha')
  await sleepStarted

  // While the backoff sleep is pending, the slot is empty and unrelated
  // lifecycle operations proceed without waiting for the retry schedule.
  expect(manager.getServer('alpha')).toBeUndefined()
  expect(await withinMs(manager.addServer({ name: 'beta' }), 250)).toBeTrue()
  expect(await withinMs(manager.removeServer('beta'), 250)).toBeTrue()

  releaseSleep()
  await expect(reconnecting).resolves.toBeTrue()
  const recovered = clients.at(-1)
  expect(recovered?.config.name).toBe('alpha')
  expect(manager.getServer('alpha')).toBe(recovered)
  expect(manager.lastFailure('alpha')).toBeUndefined()
  await manager.disconnectAll()
})

test('MCPManager reconnect keeps a concurrent registration that claimed the slot during backoff', async () => {
  const clients: FakeMCPClient[] = []
  let alphaConnections = 0
  let enteredSleep!: () => void
  let releaseSleep!: () => void
  const sleepStarted = new Promise<void>(resolve => {
    enteredSleep = resolve
  })
  const sleepGate = new Promise<void>(resolve => {
    releaseSleep = resolve
  })

  const manager = new MCPManager({
    clientFactory: config => {
      const client = new FakeMCPClient(config, () => {
        if (config.name === 'alpha') {
          alphaConnections += 1
          if (alphaConnections === 2) {
            throw new Error('transient connect failure')
          }
        }
      })
      clients.push(client)
      return client
    },
    reconnect: {
      policy: { maxAttempts: 3, baseSeconds: 30, factor: 1, maxSeconds: 30 },
      sleep: () => {
        enteredSleep()
        return sleepGate
      },
    },
  })

  expect(await manager.addServer({ name: 'alpha' })).toBeTrue()
  const reconnecting = manager.reconnect('alpha')
  await sleepStarted

  // A fresh registration wins the slot while the retry schedule is sleeping.
  expect(await manager.addServer({ name: 'alpha' })).toBeTrue()
  const registered = manager.getServer('alpha')

  releaseSleep()
  await expect(reconnecting).resolves.toBeTrue()

  // The superseded reconnect candidate is torn down, not swapped over the
  // newer registration.
  expect(manager.getServer('alpha')).toBe(registered)
  const superseded = clients.at(-1)
  expect(superseded).not.toBe(registered)
  expect(superseded?.disconnects).toBe(1)
  await manager.disconnectAll()
})

const STDOUT_FLOOD_SERVER = String.raw`
process.stdout.write('x'.repeat(4 * 1024 * 1024));
setInterval(() => {}, 1000);
`

test('MCPClient caps the stdio stdout line buffer and fails pending requests on overflow', async () => {
  const client = new MCPClient({
    name: 'flood',
    command: process.execPath,
    args: ['-e', STDOUT_FLOOD_SERVER],
    timeoutMs: 10_000,
  })
  await expect(client.connect()).rejects.toThrow(/stdout buffer/)
  expect(client.connected).toBeFalse()
  await client.disconnect()
})

test('MCP HTTP transports reject private and link-local URL literals unless explicitly allowed', async () => {
  for (const url of [
    'http://169.254.169.254/latest/meta-data',
    'http://127.0.0.1:9/mcp',
    'http://[::1]:9/mcp',
    'http://10.0.0.8/mcp',
    'http://localhost:9/mcp',
  ]) {
    const client = new MCPClient({ name: 'ssrf', timeoutMs: 500, transport: 'streamable_http', url })
    await expect(client.connect()).rejects.toThrow(/network safety policy/)
  }

  // The explicit operator opt-in skips the literal-address policy; the
  // remaining failure is the unreachable endpoint, not the URL guard.
  const allowed = new MCPClient({
    allowPrivateNetwork: true,
    name: 'allowed',
    timeoutMs: 1_000,
    transport: 'streamable_http',
    url: 'http://127.0.0.1:9/mcp',
  })
  const failure = await allowed.connect().then(
    () => undefined,
    (error: unknown) => error,
  )
  expect(failure).toBeInstanceOf(MCPConnectionError)
  expect((failure as Error).message).not.toContain('network safety policy')
})

test('MCPClient re-initializes an expired Streamable HTTP session and retries the request', async () => {
  const requests: Array<{ readonly method: string; readonly sessionId: string | null }> = []
  let initializeCount = 0
  const server = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    async fetch(request) {
      if (request.method === 'DELETE') {
        return new Response(null, { status: 204 })
      }
      const message = (await request.json()) as { id?: unknown; method?: string }
      const sessionId = request.headers.get('mcp-session-id')
      requests.push({ method: String(message.method), sessionId })
      if (message.method === 'initialize') {
        initializeCount += 1
        return jsonRpc(message.id, {
          capabilities: { tools: {} },
          protocolVersion: '2025-06-18',
          serverInfo: { name: 'session-fixture', version: '1.0.0' },
        }, { 'Mcp-Session-Id': initializeCount === 1 ? 'session-one' : 'session-two' })
      }
      if (message.method === 'notifications/initialized') {
        return new Response(null, { status: 202 })
      }
      if (sessionId !== 'session-two') {
        // Any pre-expiry request (including the capability refresh during the
        // first connect) is served; tools/call on the stale session 404s.
        if (message.method === 'tools/call') {
          return new Response('unknown session', { status: 404 })
        }
      }
      if (message.method === 'tools/list') {
        return jsonRpc(message.id, { tools: [{ name: 'echo', inputSchema: { type: 'object' } }] })
      }
      if (message.method === 'resources/list') {
        return jsonRpc(message.id, { resources: [] })
      }
      if (message.method === 'prompts/list') {
        return jsonRpc(message.id, { prompts: [] })
      }
      if (message.method === 'tools/call') {
        return jsonRpc(message.id, { content: [{ text: 'recovered', type: 'text' }] })
      }
      return new Response('not found', { status: 404 })
    },
  })

  const client = new MCPClient({
    allowPrivateNetwork: true,
    name: 'recovering',
    timeoutMs: 5_000,
    transport: 'streamable_http',
    url: `http://127.0.0.1:${server.port ?? 0}/mcp`,
  })
  try {
    await client.connect()
    expect(client.sessionId).toBe('session-one')

    await expect(client.callTool('echo')).resolves.toEqual({
      content: [{ text: 'recovered', type: 'text' }],
    })
    expect(client.sessionId).toBe('session-two')
    expect(initializeCount).toBe(2)

    const retriedCall = requests.filter(request => request.method === 'tools/call')
    expect(retriedCall.map(request => request.sessionId)).toEqual(['session-one', 'session-two'])
  } finally {
    await client.disconnect()
    server.stop(true)
  }
})

test('OAuth refreshToken falls back to the original scopes when the response omits scope', async () => {
  const config: OAuthConfig = {
    authorizeUrl: 'https://oauth.example.test/authorize',
    clientId: 'client-id',
    tokenUrl: 'https://oauth.example.test/token',
  }
  const original = new OAuthToken({ accessToken: 'old', refreshToken: 'refresh-1', scopes: ['read', 'write'] })

  const withoutScopeFetch: OAuthFetch = async () => new Response(
    JSON.stringify({ access_token: 'new-access', expires_in: 60 }),
    { headers: { 'Content-Type': 'application/json' } },
  )
  const refreshed = await refreshToken(config, original, { fetchImplementation: withoutScopeFetch, now: () => 1_000 })
  expect(refreshed.accessToken).toBe('new-access')
  expect(refreshed.refreshToken).toBe('refresh-1')
  expect(refreshed.scopes).toEqual(['read', 'write'])
  expect(refreshed.expiresAt).toBe(1_060)

  const withScopeFetch: OAuthFetch = async () => new Response(
    JSON.stringify({ access_token: 'narrowed', refresh_token: 'refresh-2', scope: 'read' }),
    { headers: { 'Content-Type': 'application/json' } },
  )
  const narrowed = await refreshToken(config, original, { fetchImplementation: withScopeFetch, now: () => 1_000 })
  expect(narrowed.refreshToken).toBe('refresh-2')
  expect(narrowed.scopes).toEqual(['read'])
})

function jsonRpc(id: unknown, result: Record<string, unknown>, headers: HeadersInit = {}): Response {
  return new Response(JSON.stringify({ id, jsonrpc: '2.0', result }), {
    headers: { 'Content-Type': 'application/json', ...headers },
  })
}
