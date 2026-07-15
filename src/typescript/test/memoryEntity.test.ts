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
