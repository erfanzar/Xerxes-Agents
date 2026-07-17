// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { SimpleStorage } from '../src/memory/storage.js'
import { UserMemory } from '../src/memory/userMemory.js'

test('user memory keeps contextual and entity memory isolated by user', () => {
  const memory = new UserMemory()
  const aliceItem = memory.saveMemory('alice', 'Alice knows Bob')
  memory.saveMemory('bob', 'Carol knows Dave')

  expect(aliceItem.userId).toBe('alice')
  expect(memory.searchUserMemory('alice', 'Alice')).toHaveLength(1)
  expect(memory.searchUserMemory('bob', 'Alice', 10, { minRelevance: 0.1 })).toHaveLength(0)
  expect(memory.userEntities.get('alice')?.entities.Alice).toBeDefined()
  expect(memory.userEntities.get('bob')?.entities.Alice).toBeUndefined()

  const statistics = memory.getUserStatistics('alice')
  expect(statistics.totalMemories).toBe(1)
  expect(statistics.entitiesKnown).toBeGreaterThan(0)
  expect(memory.getUserContext('alice')).toContain('Known entities: Alice, Bob')
})

test('user preferences have defaults, merge updates, and survive rehydration', () => {
  const storage = new SimpleStorage()
  const first = new UserMemory(storage)
  expect(first.getUserPreferences('unknown')).toMatchObject({
    language: 'en',
    response_style: 'balanced',
  })

  first.updateUserPreferences('alice', { language: 'fr', verbosity: 'terse' })
  const restored = new UserMemory(storage)
  expect(restored.getUserPreferences('alice')).toMatchObject({ language: 'fr', verbosity: 'terse' })

  restored.saveMemory('alice', 'Alice knows Bob')
  restored.clearUserMemory('alice')
  expect(restored.userMemories.has('alice')).toBeFalse()
  expect(restored.userEntities.has('alice')).toBeFalse()
  expect(restored.getUserPreferences('alice')).toMatchObject({ language: 'en' })
})

test('per-user tiers over one shared backend stay isolated across instances', () => {
  const storage = new SimpleStorage()
  const first = new UserMemory(storage)
  first.saveMemory('alice', 'Alice prefers Bun', {}, { toLongTerm: true })
  first.saveMemory('bob', 'Bob prefers Node', {}, { toLongTerm: true })

  // A fresh facade hydrating from the same backend must only see each user's own items.
  const restored = new UserMemory(storage)
  expect(restored.searchUserMemory('alice', 'prefers').map(item => item.content)).toEqual(['Alice prefers Bun'])
  expect(restored.searchUserMemory('bob', 'prefers').map(item => item.content)).toEqual(['Bob prefers Node'])
  expect(restored.userEntities.get('alice')?.entities.Alice).toBeDefined()
  expect(restored.userEntities.get('alice')?.entities.Bob).toBeUndefined()
  expect(restored.userEntities.get('bob')?.entities.Alice).toBeUndefined()

  // Clearing one user must leave the other user's persisted records intact.
  restored.clearUserMemory('alice')
  const third = new UserMemory(storage)
  expect(third.searchUserMemory('alice', 'prefers')).toEqual([])
  expect(third.searchUserMemory('bob', 'prefers').map(item => item.content)).toEqual(['Bob prefers Node'])
})
