// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { ContextualMemory } from '../src/memory/contextualMemory.js'
import { LongTermMemory } from '../src/memory/longTermMemory.js'
import { SimpleStorage } from '../src/memory/storage.js'
import {
  MEMORY_TOOL_DEFINITIONS,
  deleteMemory,
  getMemoryStatistics,
  getMemoryTagsAndTerms,
  registerMemoryTools,
  saveMemory,
} from '../src/tools/memoryTools.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

function call(name: string, arguments_: JsonObject): ToolCall {
  return {
    id: crypto.randomUUID(),
    type: 'function',
    function: { name, arguments: arguments_ },
  }
}

function contextualMemory(): ContextualMemory {
  return new ContextualMemory({
    longTerm: new LongTermMemory({ storage: new SimpleStorage() }),
  })
}

test('memory tools require an injected store and register every legacy public name', async () => {
  const registry = new ToolRegistry()
  registerMemoryTools(registry)
  expect(registry.definitions().map(definition => definition.function.name)).toEqual(
    MEMORY_TOOL_DEFINITIONS.map(definition => definition.function.name),
  )

  const result = JSON.parse(await registry.execute(
    call('save_memory', { content: 'unavailable' }),
    { metadata: {} },
  )) as JsonObject
  expect(result).toEqual({ status: 'error', message: 'Memory store not available' })
})

test('memory tools preserve requested types, route durable values to long-term memory, and honor scoped search', async () => {
  const memory = contextualMemory()
  const registry = new ToolRegistry()
  registerMemoryTools(registry, {
    resolveContext: () => ({
      memory,
      now: () => new Date('2026-07-13T10:00:00.000Z'),
    }),
  })
  const execution = { agentId: 'planner', metadata: {} }

  const saved = JSON.parse(await registry.execute(call('save_memory', {
    content: 'Bun should own the daemon runtime.',
    memory_type: 'semantic',
    tags: ['architecture', 'runtime'],
    metadata: { note_source: 'migration' },
  }), execution)) as JsonObject
  expect(saved.status).toBe('success')
  expect(memory.longTerm.size).toBe(1)
  expect(memory.shortTerm.size).toBe(0)

  const search = JSON.parse(await registry.execute(call('search_memory', {
    query: 'daemon',
    memory_types: ['semantic'],
    tags: ['architecture'],
    time_range: { start: '2026-07-13T09:00:00.000Z', end: '2026-07-13T11:00:00.000Z' },
  }), execution)) as {
    count: number
    memories: Array<{ memory_type: string; metadata: JsonObject; tags: string[] }>
    status: string
  }
  expect(search).toMatchObject({ status: 'success', count: 1 })
  expect(search.memories[0]).toMatchObject({
    memory_type: 'semantic',
    tags: ['architecture', 'runtime'],
    metadata: {
      created_by: 'planner',
      note_source: 'migration',
      requested_memory_type: 'semantic',
      timestamp: '2026-07-13T10:00:00.000Z',
    },
  })

  const outsideRange = JSON.parse(await registry.execute(call('search_memory', {
    query: 'daemon',
    time_range: { start: '2026-07-13T10:00:01.000Z' },
  }), execution)) as { count: number; memories: unknown[]; query: string; status: string }
  expect(outsideRange).toEqual({ status: 'success', count: 0, memories: [], query: 'daemon' })
})

test('memory tools consolidate, count tags, report statistics, and delete filtered long-term entries', async () => {
  const memory = contextualMemory()
  const context = {
    agentId: 'assistant',
    memory,
    now: () => new Date('2026-07-13T10:00:00.000Z'),
  }
  const durable = saveMemory({
    content: 'Keep the JSON RPC daemon protocol at version 35.',
    memoryType: 'long_term',
    tags: ['protocol', 'temporary'],
  }, context) as { memory_id: string; status: string }
  saveMemory({
    content: 'The project prefers Bun for the TypeScript runtime.',
    memoryType: 'working',
    tags: ['runtime', 'protocol'],
  }, context)
  saveMemory({
    content: 'Another protocol note.',
    memoryType: 'episodic',
    tags: ['protocol'],
  }, context)

  const tags = getMemoryTagsAndTerms({}, context) as {
    all_tags: string[]
    tag_frequency: Record<string, number>
    tags_by_type: Record<string, string[]>
    total_unique_tags: number
  }
  expect(tags.tags_by_type).toEqual({
    episodic: ['protocol'],
    long_term: ['protocol', 'temporary'],
    working: ['protocol', 'runtime'],
  })
  expect(tags.tag_frequency).toEqual({ protocol: 3, runtime: 1, temporary: 1 })
  expect(tags.all_tags).toEqual(['protocol', 'runtime', 'temporary'])
  expect(tags.total_unique_tags).toBe(3)

  const registry = new ToolRegistry()
  registerMemoryTools(registry, { context })
  const consolidated = JSON.parse(await awaitRegistry(registry, call('consolidate_agent_memories', {
    agent_id: 'assistant',
    max_items: 10,
  }))) as { statistics: JsonObject; summary: string; status: string }
  expect(consolidated.status).toBe('success')
  expect(consolidated.summary).toContain('Total memories: 3')
  expect(consolidated.summary).toContain('PROTOCOL:')
  expect(consolidated.statistics).toMatchObject({
    total_items: 3,
    total_memories: 3,
    memory_types: { long_term: 1, working: 1, episodic: 1 },
  })

  const stats = getMemoryStatistics({ agentId: 'assistant' }, context) as {
    statistics: { agent_memory_count: number; total_items: number }
  }
  expect(stats.statistics).toMatchObject({ total_items: 3, agent_memory_count: 3 })

  const removed = deleteMemory({ tags: ['temporary'] }, context) as {
    deleted_count: number
    message: string
    status: string
  }
  expect(removed).toEqual({
    status: 'success',
    message: 'Successfully deleted 1 memories',
    deleted_count: 1,
  })
  expect(memory.longTerm.retrieve(durable.memory_id)).toBeUndefined()
})

async function awaitRegistry(registry: ToolRegistry, tool: ToolCall): Promise<string> {
  return registry.execute(tool, { metadata: {} })
}
