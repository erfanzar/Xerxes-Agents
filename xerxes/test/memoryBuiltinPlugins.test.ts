// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  MemoryProviderRegistry,
  registerBuiltinMemoryProviders as registerBuiltinMemoryProvidersFromMemory,
} from '../src/memory/index.js'
import {
  BUILTIN_MEMORY_PROVIDER_NAMES,
  makeSimpleMemoryProvider,
  registerBuiltinMemoryProviders,
  type ExternalMemoryAction,
  type ExternalMemoryUpstream,
  type MemoryPluginEnvironment,
  type MemoryPluginHttpRequest,
  type MemoryPluginHttpResponse,
  type MemoryPluginHttpTransport,
} from '../src/memory/plugins/index.js'
import type { JsonObject, JsonValue } from '../src/types/toolCalls.js'

class FixtureEnvironment implements MemoryPluginEnvironment {
  constructor(private readonly values: Readonly<Record<string, string>>) {}

  get(name: string): string | undefined {
    return this.values[name]
  }
}

class FixtureUpstream implements ExternalMemoryUpstream {
  readonly actions: ExternalMemoryAction[] = []
  initialized = 0
  shutdowns = 0

  call(action: ExternalMemoryAction, _arguments: JsonObject): JsonValue {
    this.actions.push(action)
    return { action }
  }

  initialize(): void {
    this.initialized += 1
  }

  isAvailable(): boolean {
    return true
  }

  shutdown(): void {
    this.shutdowns += 1
  }
}

class FixtureTransport implements MemoryPluginHttpTransport {
  readonly requests: MemoryPluginHttpRequest[] = []

  async request(request: MemoryPluginHttpRequest): Promise<MemoryPluginHttpResponse> {
    this.requests.push(request)
    return { ok: true, status: 200, body: { accepted: true } }
  }
}

test('explicit built-in registration exposes all legacy providers without selecting or initializing one', async () => {
  const registry = new MemoryProviderRegistry()
  const transport = new FixtureTransport()
  const honcho = new FixtureUpstream()
  const holographic = new FixtureUpstream()
  const retainDb = new FixtureUpstream()
  const environment = new FixtureEnvironment({
    HONCHO_API_KEY: 'honcho-key',
    MEM0_API_KEY: 'mem0-key',
    HINDSIGHT_API_KEY: 'hindsight-key',
    HINDSIGHT_BANK_ID: 'bank-1',
    RETAINDB_API_KEY: 'retain-key',
    OPENVIKING_ENDPOINT: 'https://viking.example',
    OPENVIKING_API_KEY: 'viking-key',
    SUPERMEMORY_API_KEY: 'super-key',
    BRV_API_KEY: 'brv-key',
  })
  const dependencies = {
    environment,
    httpTransport: transport,
    honchoUpstream: honcho,
    holographicUpstream: holographic,
    retainDbUpstream: retainDb,
    clock: () => 42,
  }

  const providers = registerBuiltinMemoryProviders(dependencies, registry)
  expect(providers.map(provider => provider.name)).toEqual([...BUILTIN_MEMORY_PROVIDER_NAMES])
  expect(registry.listNames()).toEqual([...BUILTIN_MEMORY_PROVIDER_NAMES].sort())
  expect(registry.active()).toBeUndefined()
  expect(honcho.initialized).toBe(0)
  expect(holographic.initialized).toBe(0)
  expect(retainDb.initialized).toBe(0)

  const active = registry.setActive('holographic')
  expect(await active?.handleToolCall({ name: 'holo_add', arguments: { content: 'fact' } })).toEqual({
    ok: true,
    action: 'add',
    result: { action: 'add' },
    ts: 42,
  })
  expect(holographic.initialized).toBe(1)
  expect(holographic.actions).toEqual(['add'])

  const mem0 = registry.get('mem0')
  expect(await mem0?.handleToolCall({ name: 'mem0_search', arguments: { query: 'tea' } })).toMatchObject({
    ok: true,
    action: 'search',
  })
  expect(transport.requests).toEqual([{
    method: 'POST',
    url: 'https://api.mem0.ai/v1/memories/search',
    headers: { Authorization: 'Bearer mem0-key' },
    body: { query: 'tea', user_id: 'xerxes' },
  }])

  const replaced = registerBuiltinMemoryProviders(dependencies, registry)
  expect(registry.active()).toBe(replaced.find(provider => provider.name === 'holographic'))
})

test('the public memory entry exports explicit built-in registration without import-time activation', () => {
  expect(registerBuiltinMemoryProvidersFromMemory).toBe(registerBuiltinMemoryProviders)
})

test('simple provider ports the legacy in-memory add, list, search, and remove lifecycle', async () => {
  const provider = makeSimpleMemoryProvider('testmem', {
    environment: new FixtureEnvironment({}),
    clock: () => 7,
  })

  expect(await provider.isAvailable()).toBeTrue()
  const added = await provider.handleToolCall({
    name: 'testmem_add',
    arguments: { content: 'user likes tea', tags: ['preference'] },
  })
  expect(added).toEqual({
    ok: true,
    action: 'add',
    result: { id: 'mem_0001', content: 'user likes tea', tags: ['preference'] },
    ts: 7,
  })
  expect(await provider.handleToolCall({ name: 'testmem_search', arguments: { query: 'tea' } })).toMatchObject({
    ok: true,
    action: 'search',
    result: [{ id: 'mem_0001', content: 'user likes tea', tags: ['preference'] }],
  })
  expect(await provider.handleToolCall({ name: 'testmem_remove', arguments: { entry_id: 'mem_0001' } })).toMatchObject({
    ok: true,
    action: 'remove',
    result: { removed: true },
  })
  expect(await provider.handleToolCall({ name: 'testmem_list', arguments: {} })).toMatchObject({
    ok: true,
    action: 'list',
    result: [],
  })
})
