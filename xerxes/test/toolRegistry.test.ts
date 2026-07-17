// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { JsonObject, ToolCall, ToolDefinition } from '../src/types/toolCalls.js'

function definition(name: string): ToolDefinition {
  return {
    type: 'function',
    function: { name, description: name + ' test double', parameters: { properties: {}, type: 'object' } },
  }
}

function call(name: string, arguments_: JsonObject = {}): ToolCall {
  return { id: crypto.randomUUID(), type: 'function', function: { name, arguments: arguments_ } }
}

test('agent-specific tool variants stay isolated from other agents and anonymous callers', async () => {
  const registry = new ToolRegistry()
  registry.register(definition('shared'), () => 'default-handler')
  registry.register(definition('variant'), () => 'agent-a-handler', 'agent-a')

  // The matching agent gets its own variant.
  expect(registry.get('variant', 'agent-a')?.({}, { metadata: {} })).toBe('agent-a-handler')
  // Other agents and anonymous callers cannot see the agent-only variant at all.
  expect(registry.get('variant', 'agent-b')).toBeUndefined()
  expect(registry.get('variant')).toBeUndefined()
  expect(registry.definitions('agent-b').map(entry => entry.function.name)).toEqual(['shared'])
  expect(registry.definitions().map(entry => entry.function.name)).toEqual(['shared'])
  await expect(registry.execute(call('variant'), { agentId: 'agent-b', metadata: {} }))
    .rejects.toThrow('is not registered')

  // A default entry remains the fallback for every other agent.
  registry.register(definition('mixed'), () => 'mixed-default')
  registry.register(definition('mixed'), () => 'mixed-a', 'agent-a')
  expect(registry.get('mixed', 'agent-a')?.({}, { metadata: {} })).toBe('mixed-a')
  expect(registry.get('mixed', 'agent-b')?.({}, { metadata: {} })).toBe('mixed-default')
  expect(registry.get('mixed')?.({}, { metadata: {} })).toBe('mixed-default')
  expect(await registry.execute(call('mixed'), { agentId: 'agent-b', metadata: {} })).toBe('mixed-default')
})
