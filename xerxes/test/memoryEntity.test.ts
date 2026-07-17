// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { EntityMemory } from '../src/memory/entityMemory.js'
import { SimpleStorage } from '../src/memory/storage.js'

test('entity memory extracts, tracks, and relates named entities', () => {
  const memory = new EntityMemory()
  const entities = memory.extractEntities('John Smith visited New York with "SearchEngine".')
  expect(entities).toEqual(expect.arrayContaining(['John Smith', 'New York', 'SearchEngine']))
  expect(memory.extractEntities('The system remembers This only')).not.toContain('The')

  const item = memory.save('Alice knows Bob')
  expect(item.memoryType).toBe('entity')
  expect(memory.size).toBe(1)
  expect(memory.entities.Alice?.frequency).toBe(1)
  expect(memory.getEntityInfo('Alice').relationships).toEqual([{ relation: 'knows', target: 'Bob' }])
  expect(memory.getEntityInfo('Bob').relationships).toEqual([{ relation: 'inverse_knows', target: 'Alice' }])
})

test('entity memory searches by entity overlap and traverses the relation graph', () => {
  const memory = new EntityMemory()
  memory.save('Alice knows Bob')
  memory.save('Bob knows Charlie')
  memory.save('Dana knows Erin')

  const results = memory.search('Alice', 10, undefined, { entityFilter: ['Alice'] })
  expect(results).toHaveLength(1)
  expect(results[0]?.metadata.entities).toContain('Alice')
  expect(memory.getRelatedEntities('Alice', 2)).toEqual(new Set(['Bob', 'Charlie']))
})

test('entity memory persists records and removes mentions with deletion', () => {
  const storage = new SimpleStorage()
  const memory = new EntityMemory({ storage })
  const item = memory.save('Alice created Widget', {}, { entities: ['Alice', 'Widget'] })

  expect(storage.exists(`entity_${item.memoryId}`)).toBeTrue()
  expect(storage.exists('_entity_entities')).toBeTrue()
  expect(memory.update(item.memoryId, { content: 'Carol knows Dave' })).toBeTrue()
  expect(memory.retrieve(item.memoryId)).toBe(item)
  expect(memory.delete(item.memoryId)).toBe(1)
  expect(memory.entityMentions.Carol).toEqual([])
  expect(storage.exists(`entity_${item.memoryId}`)).toBeFalse()
})

test('entity memory enforces maxItems and caps per-entity contexts', () => {
  const memory = new EntityMemory({ maxItems: 3 })
  for (let index = 0; index < 5; index += 1) memory.save(`Alice mention ${index}`, {}, { entities: ['Alice'] })

  expect(memory.size).toBe(3)
  const retained = memory.retrieve(undefined, undefined, 10)
  expect(Array.isArray(retained) ? retained.map(item => item.content) : []).toEqual([
    'Alice mention 2',
    'Alice mention 3',
    'Alice mention 4',
  ])
  // The graph still tracks every mention, but contexts stay bounded.
  expect(memory.entities.Alice?.frequency).toBe(5)
  for (let index = 0; index < 30; index += 1) memory.save(`Alice context ${index}`, {}, { entities: ['Alice'] })
  expect(memory.entities.Alice?.contexts.length).toBeLessThanOrEqual(20)
})

test('entity memory hydrates records and graph snapshots back from storage', () => {
  const storage = new SimpleStorage()
  const first = new EntityMemory({ storage })
  first.save('Alice knows Bob')
  const item = first.save('Bob created Widget', {}, { entities: ['Bob', 'Widget'] })

  const restored = new EntityMemory({ storage })
  expect(restored.size).toBe(2)
  expect(restored.retrieve(item.memoryId)).toBeDefined()
  expect(restored.entities.Alice?.frequency).toBe(1)
  expect(restored.entities.Alice?.firstSeen).toBeInstanceOf(Date)
  expect(restored.getRelatedEntities('Alice', 1)).toEqual(new Set(['Bob', 'Widget']))
  expect(restored.entityMentions.Bob).toHaveLength(2)
})
