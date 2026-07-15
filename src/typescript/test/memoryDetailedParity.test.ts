// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { MemoryItem, ShortTermMemory } from '../src/memory/index.js'

test('memory-item parity creates JSON-safe records and restores identifiers and timestamps', () => {
  const timestamp = new Date('2025-01-01T00:00:00.000Z')
  const lastAccessed = new Date('2025-01-02T00:00:00.000Z')
  const original = new MemoryItem({ agentId: 'a1', content: 'original', timestamp })
  const restored = MemoryItem.fromRecord(original.toRecord())
  const explicit = MemoryItem.fromRecord({
    content: 'test',
    last_accessed: lastAccessed.toISOString(),
    memory_id: 'abc',
    timestamp: timestamp.toISOString(),
  })

  expect(original.memoryType).toBe('general')
  expect(original.memoryId).toBeString()
  expect(original.toRecord()).toMatchObject({
    agent_id: 'a1',
    content: 'original',
    memory_id: original.memoryId,
    timestamp: timestamp.toISOString(),
  })
  expect(restored.content).toBe('original')
  expect(restored.agentId).toBe('a1')
  expect(explicit.memoryId).toBe('abc')
  expect(explicit.timestamp).toEqual(timestamp)
  expect(explicit.lastAccessed).toEqual(lastAccessed)
})

test('short-term-memory parity bounds entries, finds text, filters owners, and supports retrieval lifecycle operations', () => {
  const bounded = new ShortTermMemory({ capacity: 3 })
  for (let index = 0; index < 5; index += 1) bounded.save(`msg ${index}`)
  expect(bounded.size).toBe(3)
  expect(bounded.getRecent(10).map(item => item.content)).toEqual(['msg 2', 'msg 3', 'msg 4'])

  const memory = new ShortTermMemory({ capacity: 20 })
  const python = memory.save('python programming', {}, { agentId: 'a1' })
  memory.save('java development', {}, { agentId: 'a2' })
  memory.save('the quick brown fox', {}, { agentId: 'a1' })
  memory.save('msg 1', {}, { agentId: 'a1' })
  memory.save('msg 2', {}, { agentId: 'a1' })
  memory.save('msg 3', {}, { agentId: 'a2' })

  expect(memory.search('python').at(0)?.content).toBe('python programming')
  expect(memory.search('quick fox').some(item => item.content === 'the quick brown fox')).toBeTrue()
  expect(memory.search('msg', 5, { agentId: 'a1' }).every(item => item.agentId === 'a1')).toBeTrue()
  expect(memory.search('msg', 2)).toHaveLength(2)
  expect(Array.isArray(memory.retrieve(undefined, { agentId: 'a1' }))).toBeTrue()
  expect(Array.isArray(memory.retrieve('missing')) ? undefined : memory.retrieve('missing')).toBeUndefined()

  expect(memory.update(python.memoryId, { content: 'updated' })).toBeTrue()
  const updated = memory.retrieve(python.memoryId)
  expect(Array.isArray(updated) ? undefined : updated?.content).toBe('updated')
  expect(memory.update('missing', { content: 'x' })).toBeFalse()
  expect(memory.delete(python.memoryId)).toBe(1)
  expect(memory.delete(undefined, { agentId: 'a1' })).toBe(3)
  expect(memory.size).toBe(2)
  memory.clear()
  expect(memory.size).toBe(0)
})

test('short-term-memory parity exposes textual, JSON, and markdown context plus summary and statistics', () => {
  const memory = new ShortTermMemory({ capacity: 10 })
  expect(memory.summarize()).toBe('No recent memories.')

  memory.save('hello', {}, { agentId: 'agent-1', userId: 'u1' })
  memory.save('world', {}, { agentId: 'agent-2', conversationId: 'c1' })

  expect(memory.summarize()).toContain('hello')
  expect(memory.getContext(10, 'text')).toContain('[agent-1]: hello')
  expect(memory.getContext(10, 'markdown')).toContain('**agent-1**: hello')
  expect(JSON.parse(memory.getContext(10, 'json'))).toHaveLength(2)
  expect(memory.getStatistics()).toEqual(expect.objectContaining({
    maxItems: 10,
    totalItems: 2,
    uniqueAgents: 2,
    uniqueConversations: 1,
    uniqueUsers: 1,
  }))
})
