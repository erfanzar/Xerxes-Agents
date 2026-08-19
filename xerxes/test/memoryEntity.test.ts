// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { EntityMemory } from '../src/memory/entityMemory.js'
import { ShortTermMemory } from '../src/memory/shortTermMemory.js'
import { SimpleStorage } from '../src/memory/storage.js'

class RejectingStorage extends SimpleStorage {
  rejectDeletes = false
  rejectSaves = false

  override delete(key: string): boolean {
    return this.rejectDeletes ? false : super.delete(key)
  }

  override save(key: string, data: unknown): boolean {
    return this.rejectSaves ? false : super.save(key, data)
  }
}

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

test('entity memory rebuilds records and relationships after update and deletion', () => {
  const storage = new SimpleStorage()
  const memory = new EntityMemory({ storage })
  const item = memory.save('Alice knows Bob')

  expect(memory.update(item.memoryId, { content: 'Carol knows Dave' })).toBeTrue()
  expect(memory.getRelatedEntities('Alice', 1)).toEqual(new Set())
  expect(memory.entities.Alice).toBeUndefined()
  expect(memory.entityMentions.Alice).toBeUndefined()
  expect(memory.getRelatedEntities('Carol', 1)).toEqual(new Set(['Dave']))
  expect(memory.entities.Carol?.frequency).toBe(1)

  expect(memory.delete(item.memoryId)).toBe(1)
  expect(memory.getRelatedEntities('Carol', 1)).toEqual(new Set())
  expect(memory.entities.Carol).toBeUndefined()
  expect(memory.entityMentions.Carol).toBeUndefined()
  expect(storage.exists(`entity_${item.memoryId}`)).toBeFalse()

  const restored = new EntityMemory({ storage })
  expect(restored.size).toBe(0)
  expect(Object.keys(restored.entities)).toEqual([])
  expect(restored.getRelatedEntities('Alice', 1)).toEqual(new Set())
})

test('entity memory capacity eviction rebuilds graph from retained items', () => {
  const memory = new EntityMemory({ maxItems: 1 })
  memory.save('Alice knows Bob')
  memory.save('Carol knows Dave')

  expect(memory.entities.Alice).toBeUndefined()
  expect(memory.entityMentions.Alice).toBeUndefined()
  expect(memory.getRelatedEntities('Alice', 1)).toEqual(new Set())
  expect(memory.getRelatedEntities('Carol', 1)).toEqual(new Set(['Dave']))
})

test('short-term and entity writes reject false persistence and roll back updates', () => {
  const shortStorage = new RejectingStorage()
  const short = new ShortTermMemory({ storage: shortStorage })
  shortStorage.rejectSaves = true
  expect(() => short.save('not durable')).toThrow('Failed to persist short-term memory')
  expect(short.size).toBe(0)

  shortStorage.rejectSaves = false
  const shortItem = short.save('durable')
  shortStorage.rejectSaves = true
  expect(() => short.update(shortItem.memoryId, { content: 'rejected' })).toThrow('Failed to persist short-term memory')
  expect(shortItem.content).toBe('durable')

  const entityStorage = new RejectingStorage()
  const entity = new EntityMemory({ storage: entityStorage })
  entityStorage.rejectSaves = true
  expect(() => entity.save('Alice knows Bob')).toThrow('Failed to persist entity memory')
  expect(entity.size).toBe(0)
  expect(Object.keys(entity.entities)).toEqual([])

  entityStorage.rejectSaves = false
  const entityItem = entity.save('Alice knows Bob')
  entityStorage.rejectSaves = true
  expect(() => entity.update(entityItem.memoryId, { content: 'Carol knows Dave' })).toThrow('Failed to persist entity memory')
  expect(entityItem.content).toBe('Alice knows Bob')
  expect(entity.getRelatedEntities('Alice', 1)).toEqual(new Set(['Bob']))
  expect(entity.entities.Carol).toBeUndefined()
  expect(() => entity.delete(entityItem.memoryId)).toThrow('Failed to persist entity memory snapshots')
  expect(entity.size).toBe(1)
  expect(entity.getRelatedEntities('Alice', 1)).toEqual(new Set(['Bob']))

  const boundedStorage = new RejectingStorage()
  const bounded = new EntityMemory({ storage: boundedStorage, maxItems: 1 })
  const retained = bounded.save('Alice knows Bob')
  boundedStorage.rejectSaves = true
  expect(() => bounded.save('Carol knows Dave')).toThrow('Failed to persist entity memory')
  expect(bounded.retrieve(retained.memoryId)).toBe(retained)
  expect(bounded.getRelatedEntities('Alice', 1)).toEqual(new Set(['Bob']))
  expect(bounded.entities.Carol).toBeUndefined()
})

test('entity delete and clear false returns leave live and durable records intact', () => {
  const deleteStorage = new RejectingStorage()
  const deleting = new EntityMemory({ storage: deleteStorage })
  const deleted = deleting.save('Alice knows Bob')
  deleteStorage.rejectDeletes = true
  expect(() => deleting.delete(deleted.memoryId)).toThrow('Failed to delete entity memory')
  expect(deleting.retrieve(deleted.memoryId)).toBe(deleted)
  expect(deleting.getRelatedEntities('Alice', 1)).toEqual(new Set(['Bob']))

  const clearStorage = new RejectingStorage()
  const clearing = new EntityMemory({ storage: clearStorage })
  const first = clearing.save('Alice knows Bob')
  const second = clearing.save('Carol knows Dave')
  clearStorage.rejectDeletes = true
  expect(() => clearing.clear()).toThrow('Failed to clear entity memory')
  expect(clearing.retrieve(first.memoryId)).toBe(first)
  expect(clearing.retrieve(second.memoryId)).toBe(second)
  expect(clearing.getRelatedEntities('Alice', 1)).toEqual(new Set(['Bob']))
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
  expect(memory.entities.Alice?.frequency).toBe(3)
  for (let index = 0; index < 30; index += 1) memory.save(`Alice context ${index}`, {}, { entities: ['Alice'] })
  expect(memory.entities.Alice?.frequency).toBe(3)
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
