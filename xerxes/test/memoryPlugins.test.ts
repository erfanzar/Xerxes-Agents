// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  ByteRoverProvider,
  ExternalMemoryProviderBase,
  HindsightProvider,
  HolographicProvider,
  HonchoProvider,
  Mem0Provider,
  OpenVikingProvider,
  RetainDBProvider,
  SupermemoryProvider,
  type ExternalMemoryAction,
  type ExternalMemoryUpstream,
  type MemoryPluginEnvironment,
  type MemoryPluginHttpRequest,
  type MemoryPluginHttpResponse,
  type MemoryPluginHttpTransport,
} from '../src/memory/plugins/index.js'
import type { JsonObject } from '../src/types/toolCalls.js'

class MapEnvironment implements MemoryPluginEnvironment {
  private readonly values = new Map<string, string>()

  constructor(values: Readonly<Record<string, string>> = {}) {
    for (const [name, value] of Object.entries(values)) this.values.set(name, value)
  }

  get(name: string): string | undefined {
    return this.values.get(name)
  }

  set(name: string, value: string): void {
    this.values.set(name, value)
  }
}

class RecordingUpstream implements ExternalMemoryUpstream {
  readonly calls: Array<{ action: ExternalMemoryAction; arguments_: JsonObject }> = []
  available = true
  failWith: Error | undefined
  initialized = 0
  shutdowns = 0

  call(action: ExternalMemoryAction, arguments_: JsonObject): JsonObject {
    if (this.failWith !== undefined) throw this.failWith
    this.calls.push({ action, arguments_: { ...arguments_ } })
    return { entry: `${action}-1` }
  }

  initialize(): void {
    this.initialized += 1
  }

  isAvailable(): boolean {
    return this.available
  }

  shutdown(): void {
    this.shutdowns += 1
  }
}

class RecordingHttpTransport implements MemoryPluginHttpTransport {
  readonly requests: MemoryPluginHttpRequest[] = []
  response: MemoryPluginHttpResponse = { ok: true, status: 200, body: { provider: 'accepted' } }

  async request(request: MemoryPluginHttpRequest): Promise<MemoryPluginHttpResponse> {
    this.requests.push(request)
    return this.response
  }
}

test('external base returns explicit unavailable, validation, and upstream errors', async () => {
  const environment = new MapEnvironment()
  const upstream = new RecordingUpstream()
  const provider = new ExternalMemoryProviderBase({
    name: 'test-provider',
    namespaceLabel: 'test',
    requiredEnvironment: ['TEST_MEMORY_KEY'],
    environment,
    upstream,
    clock: () => 123.5,
  })

  expect(provider.getToolSchemas().map(schema => schema.name)).toEqual([
    'test_add',
    'test_search',
    'test_list',
    'test_remove',
  ])
  expect(await provider.handleToolCall({ name: 'test_add', arguments: { content: 'tea' } })).toEqual({
    ok: false,
    error: 'test-provider backend not available',
  })
  expect(upstream.calls).toEqual([])

  environment.set('TEST_MEMORY_KEY', 'configured')
  const added = await provider.handleToolCall({ name: 'test_add', arguments: { content: 'tea', tags: ['preference'] } })
  expect(added).toEqual({
    ok: true,
    action: 'add',
    result: { entry: 'add-1' },
    ts: 123.5,
  })
  expect(upstream.initialized).toBe(1)
  expect(await provider.handleToolCall({ name: 'test_list', arguments: { limit: -1 } })).toEqual({
    ok: false,
    action: 'list',
    error: 'limit must be a non-negative safe integer',
  })
  expect(await provider.handleToolCall({ name: 'test_rewrite', arguments: {} })).toEqual({
    ok: false,
    error: 'unknown action: test_rewrite',
  })

  upstream.failWith = new Error('upstream rejected request')
  expect(await provider.handleToolCall({ name: 'test_search', arguments: { query: 'tea' } })).toEqual({
    ok: false,
    action: 'search',
    error: 'upstream rejected request',
  })
  await provider.shutdown()
  expect(upstream.shutdowns).toBe(1)
})

test('HTTP adapters produce the configured vendor requests through an injected transport', async () => {
  const environment = new MapEnvironment({
    MEM0_API_KEY: 'mem0-key',
    MEM0_USER_ID: 'user-7',
    HINDSIGHT_API_KEY: 'hindsight-key',
    HINDSIGHT_BANK_ID: 'bank id',
    HINDSIGHT_BUDGET: 'high',
    OPENVIKING_ENDPOINT: 'https://viking.example/api',
    OPENVIKING_API_KEY: 'viking-key',
    SUPERMEMORY_API_KEY: 'super-key',
    BRV_API_KEY: 'brv-key',
  })
  const transport = new RecordingHttpTransport()
  const mem0 = new Mem0Provider({ environment, transport, clock: () => 1 })
  const hindsight = new HindsightProvider({ environment, transport, clock: () => 2 })
  const openViking = new OpenVikingProvider({ environment, transport, clock: () => 3 })
  const supermemory = new SupermemoryProvider({ environment, transport, clock: () => 4 })
  const byteRover = new ByteRoverProvider({ environment, transport, clock: () => 5 })

  for (const [provider, call] of [
    [mem0, { name: 'mem0_add', arguments: { content: 'remember this' } }],
    [hindsight, { name: 'hindsight_search', arguments: { query: 'context' } }],
    [openViking, { name: 'viking_list', arguments: { limit: 7 } }],
    [supermemory, { name: 'super_remove', arguments: { entry_id: 'entry/id' } }],
    [byteRover, { name: 'brv_add', arguments: { content: 'child', parent: 'root' } }],
  ] as const) {
    const result = await provider.handleToolCall(call)
    expect(result.ok).toBeTrue()
  }

  expect(transport.requests).toEqual([
    {
      method: 'POST',
      url: 'https://api.mem0.ai/v1/memories',
      headers: { Authorization: 'Bearer mem0-key' },
      body: { messages: [{ role: 'user', content: 'remember this' }], user_id: 'user-7' },
    },
    {
      method: 'POST',
      url: 'https://api.hindsight.ai/v1/banks/bank%20id/search',
      headers: { 'X-Api-Key': 'hindsight-key' },
      body: { query: 'context', budget: 'high' },
    },
    {
      method: 'GET',
      url: 'https://viking.example/api/v1/contexts?limit=7',
      headers: { Authorization: 'Bearer viking-key' },
    },
    {
      method: 'DELETE',
      url: 'https://api.supermemory.ai/v1/memories/entry%2Fid',
      headers: { Authorization: 'Bearer super-key' },
    },
    {
      method: 'POST',
      url: 'https://api.byterover.dev/v1/nodes',
      headers: { Authorization: 'Bearer brv-key' },
      body: { content: 'child', parent: 'root' },
    },
  ])
})

test('a rejected remote response is surfaced as an explicit result rather than an invented success', async () => {
  const transport = new RecordingHttpTransport()
  transport.response = { ok: false, status: 401 }
  const provider = new Mem0Provider({
    environment: new MapEnvironment({ MEM0_API_KEY: 'mem0-key' }),
    transport,
    clock: () => 7,
  })

  expect(await provider.handleToolCall({ name: 'mem0_search', arguments: { query: 'secret' } })).toEqual({
    ok: false,
    action: 'search',
    error: 'mem0 request failed with HTTP 401',
  })
})

test('host-owned upstream adapters cover Honcho, Holographic, and RetainDB without fallback', async () => {
  const honchoUpstream = new RecordingUpstream()
  const localUpstream = new RecordingUpstream()
  const retainUpstream = new RecordingUpstream()
  const honcho = new HonchoProvider({
    environment: new MapEnvironment({ HONCHO_API_KEY: 'honcho-key' }),
    upstream: honchoUpstream,
  })
  const holographic = new HolographicProvider({ environment: new MapEnvironment(), upstream: localUpstream })
  const retainEnvironment = new MapEnvironment()
  const retain = new RetainDBProvider({ environment: retainEnvironment, upstream: retainUpstream })

  expect(await honcho.handleToolCall({ name: 'honcho_list', arguments: {} })).toMatchObject({ ok: true, action: 'list' })
  expect(await holographic.handleToolCall({ name: 'holo_add', arguments: { content: 'fact' } })).toMatchObject({
    ok: true,
    action: 'add',
  })
  expect(await retain.handleToolCall({ name: 'retain_add', arguments: { content: 'queued' } })).toEqual({
    ok: false,
    error: 'retaindb backend not available',
  })
  retainEnvironment.set('RETAINDB_API_KEY', 'retain-key')
  expect(await retain.handleToolCall({ name: 'retain_add', arguments: { content: 'queued' } })).toMatchObject({
    ok: true,
    action: 'add',
  })
  expect(honchoUpstream.calls.map(call => call.action)).toEqual(['list'])
  expect(localUpstream.calls.map(call => call.action)).toEqual(['add'])
  expect(retainUpstream.calls.map(call => call.action)).toEqual(['add'])
})
