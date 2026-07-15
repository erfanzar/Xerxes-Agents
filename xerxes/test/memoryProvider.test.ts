// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  MemoryProviderRegistry,
  MemoryProviderRegistryError,
  UnknownMemoryProviderError,
  activeMemoryProvider,
  createMemoryToolCall,
  memoryProviderRegistry,
  registerMemoryProvider,
  setActiveMemoryProvider,
  type MemoryProvider,
  type MemoryToolCall,
} from '../src/memory/index.js'

class RecordingMemoryProvider implements MemoryProvider {
  readonly calls: MemoryToolCall[] = []
  readonly lifecycle: string[] = []
  readonly name: string

  constructor(name: string) {
    this.name = name
  }

  async getToolSchemas() {
    return [{ name: `${this.name}_search` }]
  }

  async handleToolCall(call: MemoryToolCall) {
    this.calls.push(call)
    return { ok: true }
  }

  async initialize(): Promise<void> {
    this.lifecycle.push('initialize')
  }

  async isAvailable(): Promise<boolean> {
    return true
  }

  async onDelegation(parentSessionId: string, childSessionId: string): Promise<void> {
    this.lifecycle.push(`delegation:${parentSessionId}:${childSessionId}`)
  }

  async onMemoryWrite(reference: string, content: string): Promise<void> {
    this.lifecycle.push(`write:${reference}:${content}`)
  }

  async onPreCompress(_state: unknown): Promise<void> {
    this.lifecycle.push('pre-compress')
  }

  async onSessionEnd(_state: unknown): Promise<void> {
    this.lifecycle.push('session-end')
  }

  async onTurnStart(_state: unknown): Promise<void> {
    this.lifecycle.push('turn-start')
  }

  async shutdown(): Promise<void> {
    this.lifecycle.push('shutdown')
  }
}

test('memory provider registry lists, selects, replaces, and removes one active provider', () => {
  const registry = new MemoryProviderRegistry()
  const alpha = new RecordingMemoryProvider('alpha')
  const beta = new RecordingMemoryProvider('beta')

  registry.register(beta)
  registry.register(alpha)
  expect(registry.listNames()).toEqual(['alpha', 'beta'])
  expect(registry.active()).toBeUndefined()
  expect(registry.setActive('alpha')).toBe(alpha)
  expect(registry.active()).toBe(alpha)

  const replacement = new RecordingMemoryProvider('alpha')
  registry.register(replacement)
  expect(registry.active()).toBe(replacement)
  expect(registry.unregister('alpha')).toBeTrue()
  expect(registry.active()).toBeUndefined()
  expect(registry.unregister('alpha')).toBeFalse()
  expect(() => registry.setActive('missing')).toThrow(UnknownMemoryProviderError)
  expect(() => registry.register(new RecordingMemoryProvider(' '))).toThrow(MemoryProviderRegistryError)
})

test('memory provider tool calls and optional lifecycle hooks remain provider-owned', async () => {
  const provider = new RecordingMemoryProvider('local-test')
  const call = createMemoryToolCall(' local-test_search ', { query: 'tea' })

  expect(call).toEqual({ arguments: { query: 'tea' }, name: 'local-test_search' })
  await provider.initialize()
  expect(await provider.isAvailable()).toBeTrue()
  expect(await provider.getToolSchemas()).toEqual([{ name: 'local-test_search' }])
  expect(await provider.handleToolCall(call)).toEqual({ ok: true })
  await provider.onTurnStart?.({ turn: 1 })
  await provider.onPreCompress?.({ messages: [] })
  await provider.onMemoryWrite?.('memory:1', 'tea preference')
  await provider.onDelegation?.('parent', 'child')
  await provider.onSessionEnd?.({})
  await provider.shutdown?.()

  expect(provider.calls).toEqual([call])
  expect(provider.lifecycle).toEqual([
    'initialize',
    'turn-start',
    'pre-compress',
    'write:memory:1:tea preference',
    'delegation:parent:child',
    'session-end',
    'shutdown',
  ])
  expect(() => createMemoryToolCall('')).toThrow(MemoryProviderRegistryError)
})

test('default memory-provider helpers share one active registry slot', () => {
  const registry = memoryProviderRegistry()
  const previous = registry.active()
  const provider = new RecordingMemoryProvider(`default-test-${crypto.randomUUID()}`)
  try {
    registerMemoryProvider(provider)
    expect(setActiveMemoryProvider(provider.name)).toBe(provider)
    expect(activeMemoryProvider()).toBe(provider)
    expect(setActiveMemoryProvider(null)).toBeUndefined()
  } finally {
    registry.unregister(provider.name)
    if (previous) {
      registry.setActive(previous.name)
    }
  }
})
