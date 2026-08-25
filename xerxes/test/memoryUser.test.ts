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

test('per-user preference records survive concurrent writers across instances', () => {
  const storage = new SimpleStorage()
  const first = new UserMemory(storage)
  const second = new UserMemory(storage)

  // The shared wholesale blob made the second save rewrite the whole map from
  // its own stale view, erasing the first user's update.
  first.updateUserPreferences('alice', { language: 'fr' })
  second.updateUserPreferences('bob', { language: 'de' })

  const verifier = new UserMemory(storage)
  expect(verifier.getUserPreferences('alice').language).toBe('fr')
  expect(verifier.getUserPreferences('bob').language).toBe('de')
})

test('legacy preference blobs migrate to per-user keys and stop being authoritative', () => {
  const storage = new SimpleStorage()
  storage.save('_user_preferences', { carol: { language: 'es', verbosity: 'terse' } })

  const migrated = new UserMemory(storage)
  expect(migrated.getUserPreferences('carol')).toMatchObject({ language: 'es', verbosity: 'terse' })
  // Entries distribute once into namespaced per-user keys...
  expect(storage.listKeys('user_preferences').filter(key => /^user_[0-9a-f]{16}_user_preferences$/.test(key)))
    .toHaveLength(1)

  // ...and are then ignored: a stale legacy blob cannot overwrite newer records.
  storage.save('_user_preferences', { carol: { language: 'ru' } })
  const reloaded = new UserMemory(storage)
  expect(reloaded.getUserPreferences('carol').language).toBe('es')

  // Users known only from a legacy blob still resolve.
  storage.save('_user_preferences', { carol: { language: 'ru' }, dave: { language: 'it' } })
  expect(new UserMemory(storage).getUserPreferences('dave').language).toBe('it')
})

test('cleared users stay cleared across restart even with a retained legacy blob', () => {
  const storage = new SimpleStorage()
  storage.save('_user_preferences', { carol: { language: 'es', verbosity: 'terse' } })

  const first = new UserMemory(storage)
  expect(first.getUserPreferences('carol').language).toBe('es')
  first.clearUserMemory('carol')

  // The blob must not resurrect carol on the next construction.
  const second = new UserMemory(storage)
  expect(second.getUserPreferences('carol').language).toBe('en')
  expect(second.getUserPreferences('carol')).toMatchObject({ language: 'en', response_style: 'balanced' })

  // Migration completed: the blob no longer advertises any distributed user.
  const leftoverBlob = storage.load('_user_preferences') as Record<string, unknown>
  expect(Object.keys(leftoverBlob)).toEqual([])
})

test('clearing a user scrubs their not-yet-migrated legacy entry too', () => {
  const storage = new SimpleStorage()
  const memory = new UserMemory(storage)
  // Simulate an old process writing the wholesale blob after this facade
  // already knows the user through namespaced records.
  memory.updateUserPreferences('erin', { language: 'ja' })
  storage.save('_user_preferences', { erin: { language: 'de' }, frank: { language: 'pt' } })

  memory.clearUserMemory('erin')

  const restored = new UserMemory(storage)
  // Erin was scrubbed from the blob alongside her namespaced record...
  expect(restored.getUserPreferences('erin').language).toBe('en')
  // ...while frank, untouched by the clear, still migrates from the blob.
  expect(restored.getUserPreferences('frank').language).toBe('pt')
})
