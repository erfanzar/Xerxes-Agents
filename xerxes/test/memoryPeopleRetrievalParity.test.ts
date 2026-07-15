// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  ConfidentValue,
  EntityMemory,
  HashEmbedder,
  HybridRetriever,
  MemoryItem,
  RetrievalWeights,
  SimpleStorage,
  UserMemory,
  UserProfile,
  UserProfileStore,
} from '../src/memory/index.js'

test('user-memory parity creates user-local stores idempotently, tracks statistics, and clears only the requested user', () => {
  const memory = new UserMemory()
  const first = memory.getOrCreateUserMemory('user-1')

  expect(memory.userMemories.size).toBe(1)
  expect(memory.userEntities.has('user-1')).toBeTrue()
  expect(memory.userPreferences.has('user-1')).toBeTrue()
  expect(memory.getOrCreateUserMemory('user-1')).toBe(first)
  expect(memory.getUserPreferences('unknown')).toMatchObject({ language: 'en', response_style: 'balanced' })
  expect(memory.getUserStatistics('unknown')).toMatchObject({
    entitiesKnown: 0,
    totalMemories: 0,
    userId: 'unknown',
  })

  const alice = memory.saveMemory('user-1', 'Python programming basics')
  memory.saveMemory('user-2', 'Java programming basics')
  expect(alice.content).toBe('Python programming basics')
  expect(memory.searchUserMemory('user-1', 'Python')).toHaveLength(1)
  expect(memory.getUserContext('user-1')).toContain('Python programming basics')
  expect(memory.getUserStatistics('user-1')).toMatchObject({ userId: 'user-1', totalMemories: 1 })
  expect(memory.userMemories.size).toBe(2)

  memory.clearUserMemory('user-1')
  memory.clearUserMemory('missing')
  expect(memory.userMemories.has('user-1')).toBeFalse()
  expect(memory.userEntities.has('user-1')).toBeFalse()
  expect(memory.userMemories.has('user-2')).toBeTrue()
})

test('entity-memory parity accepts explicit entities, maintains frequencies, filters, and clears indexes', () => {
  const memory = new EntityMemory()
  const first = memory.save('Alice met Bob', {}, { entities: ['Alice', 'Bob'] })
  memory.save('Alice visited New York', {}, { entities: ['Alice', 'New York'] })

  expect(first.metadata.entities).toEqual(['Alice', 'Bob'])
  expect(memory.extractEntities('John Smith visited New York with "SearchEngine".'))
    .toEqual(expect.arrayContaining(['John Smith', 'New York', 'SearchEngine']))
  expect(memory.entities.Alice?.frequency).toBe(2)
  expect(memory.search('Alice', 10, undefined, { entityFilter: ['Alice'] }).map(item => item.memoryId))
    .toContain(first.memoryId)
  expect(Array.isArray(memory.retrieve(undefined, { memoryType: 'entity' }))).toBeTrue()
  expect(memory.retrieve('missing')).toBeUndefined()
  expect(memory.update(first.memoryId, { content: 'Carol knows Dave' })).toBeTrue()
  expect(memory.update('missing', {})).toBeFalse()
  expect(memory.entityMentions.Carol).toEqual([first.memoryId])
  expect(memory.delete(first.memoryId)).toBe(1)
  expect(memory.size).toBe(1)

  memory.clear()
  expect(memory.size).toBe(0)
  expect(memory.entities).toEqual({})
  expect(memory.entityMentions).toEqual({})
  expect(memory.relationships).toEqual({})
})

test('profile parity clamps confidence, bounds visible feedback, and persists profile lifecycle state', () => {
  const confidence = new ConfidentValue('terse', { confidence: 0.8 })
  confidence.reinforce(0.5)
  expect(confidence.confidence).toBe(1)
  confidence.demote(1.5)
  expect(confidence.confidence).toBe(0)

  const profile = new UserProfile({ userId: 'u1', notes: Array.from({ length: 50 }, (_, index) => `note ${index}`) })
  profile.expertise.set('python', new ConfidentValue('expert', { confidence: 0.9 }))
  profile.expertise.set('rust', new ConfidentValue('novice', { confidence: 0.1 }))
  expect(profile.render({ maxLines: 5 }).split('\n')).toHaveLength(5)
  expect(profile.render({ minConfidence: 0.5 })).toContain('Expertise in python')
  expect(profile.render({ minConfidence: 0.5 })).not.toContain('rust')
  const lowConfidence = new UserProfile({ userId: 'low' })
  lowConfidence.expertise.set('python', new ConfidentValue('novice', { confidence: 0.05 }))
  expect(lowConfidence.render({ minConfidence: 0.5 })).toBe('')

  profile.recordFeedback('correction', { target: 'response_style' })
  expect(profile.feedbackHistory.at(-1)).toMatchObject({ signal: 'correction', target: 'response_style' })
  for (let index = 0; index < 300; index += 1) profile.recordFeedback('ping')
  expect(profile.feedbackHistory.length).toBeLessThanOrEqual(256)

  const storage = new SimpleStorage()
  const profiles = new UserProfileStore(storage)
  const persisted = profiles.getOrCreate('u1')
  persisted.domains.push('python')
  persisted.tone = new ConfidentValue('terse', { confidence: 0.8 })
  profiles.save(persisted)
  expect(storage.exists('_profile_u1')).toBeTrue()
  expect(new UserProfileStore(storage).get('u1')?.tone?.value).toBe('terse')
  profiles.getOrCreate('u2')
  expect(profiles.allUserIds().sort()).toEqual(['u1', 'u2'])
  expect(profiles.renderFor('unknown')).toBe('')
  expect(profiles.delete('u1')).toBeTrue()
  expect(storage.exists('_profile_u1')).toBeFalse()
})

test('hybrid-retrieval parity normalizes weights and independently scores semantic, lexical, and recency signals', () => {
  const normalized = new RetrievalWeights({ bm25: 0.3, recency: 0.1, semantic: 0.6 }).normalized()
  const fallback = new RetrievalWeights({ bm25: 0, recency: 0, semantic: 0 }).normalized()
  expect(normalized.semantic + normalized.bm25 + normalized.recency).toBeCloseTo(1)
  expect(fallback.semantic + fallback.bm25 + fallback.recency).toBeCloseTo(1)

  const now = new Date('2026-07-13T12:00:00.000Z')
  const deadline = new MemoryItem({ content: 'the project deadline is march 15', timestamp: now })
  const party = new MemoryItem({ content: 'birthday party planning', timestamp: now })
  const grocery = new MemoryItem({ content: 'grocery list potatoes onions', timestamp: now })
  const embedder = new HashEmbedder(64)
  const defaultRanker = new HybridRetriever(embedder)
  expect(defaultRanker.rank('project deadline', [party, grocery, deadline], 3, now).at(0)?.item).toBe(deadline)
  expect(defaultRanker.rank('query', [], 5, now)).toEqual([])

  const old = new MemoryItem({ content: 'anything goes here', timestamp: new Date('2026-04-04T12:00:00.000Z') })
  const recent = new MemoryItem({ content: 'anything goes here', timestamp: new Date('2026-07-12T12:00:00.000Z') })
  const recencyOnly = new HybridRetriever(embedder, new RetrievalWeights({ bm25: 0, recency: 1, semantic: 0 }))
  expect(recencyOnly.rank('query', [old, recent], 2, now).map(result => result.item)).toEqual([recent, old])

  const lexicalOnly = new HybridRetriever(embedder, new RetrievalWeights({ bm25: 1, recency: 0, semantic: 0 }))
  const lexical = lexicalOnly.rank('alpha', [
    new MemoryItem({ content: 'alpha beta gamma', timestamp: now }),
    new MemoryItem({ content: 'delta epsilon zeta', timestamp: now }),
    new MemoryItem({ content: 'alpha alpha alpha', timestamp: now }),
  ], 3, now)
  expect(lexical.at(0)?.item.content).toStartWith('alpha')
  expect(lexical.every(result => result.score >= 0 && result.score <= 1)).toBeTrue()

  const embedded = new MemoryItem({ content: 'the quick brown fox', embedding: embedder.embed('the quick brown fox'), timestamp: now })
  expect(defaultRanker.rank('brown fox', [embedded], 1, now).at(0)?.semanticScore).toBeGreaterThan(0)
})
