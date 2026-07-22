// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { MCPClient } from '../src/mcp/client.js'
import { MCPConnectionError, MCPProtocolError } from '../src/mcp/types.js'

interface CapturedRequest {
  readonly headers: Readonly<Record<string, string | null>>
  readonly message: Record<string, unknown> | undefined
  readonly method: string
  readonly path: string
}

const encoder = new TextEncoder()

test('MCPClient uses Streamable HTTP JSON and SSE responses with negotiated session headers', async () => {
  const requests: CapturedRequest[] = []
  const server = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    async fetch(request) {
      const url = new URL(request.url)
      if (request.method === 'DELETE') {
        requests.push(captureRequest(request, url, undefined))
        return new Response(null, { status: 204 })
      }
      const message = asRecord(await request.json())
      requests.push(captureRequest(request, url, message))
      if (message.method === 'initialize') {
        return jsonRpcResponse(message.id, {
          capabilities: { tools: {} },
          protocolVersion: '2025-06-18',
          serverInfo: { name: 'streamable-fixture', version: '1.0.0' },
        }, { 'Mcp-Session-Id': 'session-abc' })
      }
      if (message.method === 'notifications/initialized') {
        return new Response(null, { status: 202 })
      }
      if (message.method === 'tools/list') {
        return sseResponse([
          { jsonrpc: '2.0', method: 'notifications/progress', params: { progress: 1, total: 1 } },
          {
            id: message.id,
            jsonrpc: '2.0',
            result: { tools: [{ name: 'echo', inputSchema: { type: 'object' } }] },
          },
        ])
      }
      if (message.method === 'resources/list') {
        return jsonRpcResponse(message.id, { resources: [] })
      }
      if (message.method === 'prompts/list') {
        return jsonRpcResponse(message.id, { prompts: [] })
      }
      if (message.method === 'tools/call') {
        return jsonRpcResponse(message.id, { content: [{ text: 'ok', type: 'text' }] })
      }
      return new Response(JSON.stringify({
        error: { code: -32601, message: 'method not found' },
        id: message.id,
        jsonrpc: '2.0',
      }), { headers: { 'Content-Type': 'application/json' } })
    },
  })

  const client = new MCPClient({
    allowPrivateNetwork: true,
    headers: { Authorization: 'Bearer fixture-token' },
    name: 'streamable',
    timeoutMs: 5_000,
    transport: 'streamable_http',
    url: serverUrl(server, '/mcp'),
  })
  const notifications: string[] = []
  const unsubscribe = client.onNotification(notification => notifications.push(notification.method))

  try {
    await client.connect()
    expect(client.connected).toBeTrue()
    expect(client.sessionId).toBe('session-abc')
    expect(client.tools).toEqual([{ name: 'echo', inputSchema: { type: 'object' }, serverName: 'streamable' }])
    expect(notifications).toEqual(['notifications/progress'])
    await expect(client.callTool('echo')).resolves.toEqual({ content: [{ text: 'ok', type: 'text' }] })

    const initialize = requestFor(requests, 'initialize')
    expect(initialize.headers.accept).toBe('application/json, text/event-stream')
    expect(initialize.headers.authorization).toBe('Bearer fixture-token')
    expect(initialize.headers['content-type']).toBe('application/json')
    expect(initialize.headers['mcp-protocol-version']).toBeNull()
    expect(initialize.headers['mcp-session-id']).toBeNull()
    expect(protocolVersionFor(initialize)).toBe('2025-06-18')

    const listedTools = requestFor(requests, 'tools/list')
    expect(listedTools.headers['mcp-protocol-version']).toBe('2025-06-18')
    expect(listedTools.headers['mcp-session-id']).toBe('session-abc')
  } finally {
    unsubscribe()
    await client.disconnect()
    server.stop(true)
  }

  const shutdown = requests.find(request => request.method === 'DELETE')
  expect(shutdown).toMatchObject({
    headers: {
      'mcp-protocol-version': '2025-06-18',
      'mcp-session-id': 'session-abc',
    },
    path: '/mcp',
  })
})

test('MCPClient completes the legacy SSE endpoint handshake and receives split message events', async () => {
  const requests: CapturedRequest[] = []
  let controller: ReadableStreamDefaultController<Uint8Array> | undefined
  const server = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    async fetch(request) {
      const url = new URL(request.url)
      if (request.method === 'GET' && url.pathname === '/sse') {
        const body = new ReadableStream<Uint8Array>({
          start(stream) {
            controller = stream
            stream.enqueue(encoder.encode('event: end'))
            stream.enqueue(encoder.encode('point\ndata: /message?connection=fixture\n\n'))
          },
        })
        requests.push(captureRequest(request, url, undefined))
        return new Response(body, { headers: { 'Content-Type': 'text/event-stream; charset=utf-8' } })
      }
      if (request.method !== 'POST' || url.pathname !== '/message') {
        return new Response('not found', { status: 404 })
      }
      const message = asRecord(await request.json())
      requests.push(captureRequest(request, url, message))
      if (message.method === 'notifications/initialized') {
        return new Response(null, { status: 202 })
      }
      writeLegacyMessage(controller, responseForLegacy(message))
      return new Response(null, { status: 202 })
    },
  })
  const client = new MCPClient({
    allowPrivateNetwork: true,
    headers: { 'X-Fixture': 'legacy' },
    name: 'legacy',
    timeoutMs: 5_000,
    transport: 'sse',
    url: serverUrl(server, '/sse'),
  })

  try {
    await client.connect()
    expect(client.connected).toBeTrue()
    expect(client.tools).toEqual([{ name: 'legacy_echo', inputSchema: { type: 'object' }, serverName: 'legacy' }])
    await expect(client.callTool('legacy_echo')).resolves.toEqual({
      content: [{ text: 'legacy ok', type: 'text' }],
    })

    const stream = requests.find(request => request.method === 'GET')
    expect(stream).toMatchObject({
      headers: { accept: 'text/event-stream', 'x-fixture': 'legacy' },
      path: '/sse',
    })
    const initialize = requestFor(requests, 'initialize')
    expect(initialize.path).toBe('/message')
    expect(initialize.headers['content-type']).toBe('application/json')
    expect(initialize.headers['mcp-protocol-version']).toBeNull()
    expect(initialize.headers['mcp-session-id']).toBeNull()
    expect(protocolVersionFor(initialize)).toBe('2024-11-05')
  } finally {
    await client.disconnect()
    server.stop(true)
  }
})

test('MCP HTTP configuration rejects unsafe protocol header overrides and malformed endpoints', async () => {
  await expect(new MCPClient({
    headers: { 'Mcp-Session-Id': 'not-allowed' },
    name: 'invalid-header',
    transport: 'streamable_http',
    url: 'https://mcp.example.test/mcp',
  }).connect()).rejects.toBeInstanceOf(MCPConnectionError)

  await expect(new MCPClient({
    name: 'invalid-url',
    transport: 'streamable_http',
    url: 'ftp://mcp.example.test/mcp',
  }).connect()).rejects.toBeInstanceOf(MCPConnectionError)

  const crossOriginServer = Bun.serve({
    hostname: '127.0.0.1',
    port: 0,
    fetch() {
      return new Response('event: endpoint\ndata: https://example.invalid/message\n\n', {
        headers: { 'Content-Type': 'text/event-stream' },
      })
    },
  })
  const client = new MCPClient({
    allowPrivateNetwork: true,
    name: 'cross-origin',
    transport: 'sse',
    url: serverUrl(crossOriginServer, '/sse'),
  })
  try {
    await expect(client.connect()).rejects.toBeInstanceOf(MCPProtocolError)
  } finally {
    await client.disconnect()
    crossOriginServer.stop(true)
  }
})

function captureRequest(request: Request, url: URL, message: Record<string, unknown> | undefined): CapturedRequest {
  return {
    headers: {
      accept: request.headers.get('accept'),
      authorization: request.headers.get('authorization'),
      'content-type': request.headers.get('content-type'),
      'mcp-protocol-version': request.headers.get('mcp-protocol-version'),
      'mcp-session-id': request.headers.get('mcp-session-id'),
      'x-fixture': request.headers.get('x-fixture'),
    },
    message,
    method: request.method,
    path: url.pathname,
  }
}

function jsonRpcResponse(id: unknown, result: Record<string, unknown>, headers: HeadersInit = {}): Response {
  return new Response(JSON.stringify({ id, jsonrpc: '2.0', result }), {
    headers: { 'Content-Type': 'application/json', ...headers },
  })
}

function sseResponse(frames: readonly Record<string, unknown>[]): Response {
  const text = frames.map(frame => `event: message\ndata: ${JSON.stringify(frame)}\n\n`).join('')
  const encoded = encoder.encode(text)
  const splitAt = Math.floor(encoded.length / 2)
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoded.slice(0, splitAt))
      controller.enqueue(encoded.slice(splitAt))
      controller.close()
    },
  })
  return new Response(body, { headers: { 'Content-Type': 'text/event-stream' } })
}

function writeLegacyMessage(
  controller: ReadableStreamDefaultController<Uint8Array> | undefined,
  message: Record<string, unknown>,
): void {
  if (!controller) {
    throw new Error('Legacy SSE controller was not established')
  }
  const text = `event: message\ndata: ${JSON.stringify(message)}\n\n`
  const midpoint = Math.floor(text.length / 2)
  controller.enqueue(encoder.encode(text.slice(0, midpoint)))
  controller.enqueue(encoder.encode(text.slice(midpoint)))
}

function responseForLegacy(message: Record<string, unknown>): Record<string, unknown> {
  const id = message.id
  switch (message.method) {
    case 'initialize':
      return {
        id,
        jsonrpc: '2.0',
        result: {
          capabilities: { tools: {} },
          protocolVersion: '2024-11-05',
          serverInfo: { name: 'legacy-fixture', version: '1.0.0' },
        },
      }
    case 'tools/list':
      return {
        id,
        jsonrpc: '2.0',
        result: { tools: [{ name: 'legacy_echo', inputSchema: { type: 'object' } }] },
      }
    case 'resources/list':
      return { id, jsonrpc: '2.0', result: { resources: [] } }
    case 'prompts/list':
      return { id, jsonrpc: '2.0', result: { prompts: [] } }
    case 'tools/call':
      return { id, jsonrpc: '2.0', result: { content: [{ text: 'legacy ok', type: 'text' }] } }
    default:
      return { error: { code: -32601, message: 'method not found' }, id, jsonrpc: '2.0' }
  }
}

function requestFor(requests: readonly CapturedRequest[], method: string): CapturedRequest {
  const request = requests.find(candidate => candidate.message?.method === method)
  if (!request) {
    throw new Error(`No ${method} request was captured`)
  }
  return request
}

function asRecord(value: unknown): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error('Expected a JSON object')
  }
  return value as Record<string, unknown>
}

function protocolVersionFor(request: CapturedRequest): unknown {
  return asRecord(request.message?.params).protocolVersion
}

function serverUrl(server: ReturnType<typeof Bun.serve>, path: string): string {
  if (server.port === undefined) {
    throw new Error('Bun did not report the server port')
  }
  return `http://127.0.0.1:${server.port}${path}`
}
