// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { SimpleStorage } from '../src/memory/storage.js'
import { ConfidentValue, UserProfile, UserProfileStore } from '../src/memory/userProfile.js'

test('confident values reinforce, demote, and user profiles render only confident beliefs', () => {
  const value = new ConfidentValue('terse', { confidence: 0.3 })
  value.reinforce(0.4)
  expect(value.confidence).toBeCloseTo(0.7)
  expect(value.evidenceCount).toBe(1)
  value.demote(0.2)
  expect(value.confidence).toBeCloseTo(0.5)

  const profile = new UserProfile({ userId: 'u1', domains: ['typescript'] })
  profile.expertise.set('typescript', new ConfidentValue('expert', { confidence: 0.9 }))
  profile.expertise.set('rust', new ConfidentValue('novice', { confidence: 0.1 }))
  profile.notes.push('prefers a direct answer')
  const rendered = profile.render({ minConfidence: 0.5 })
  expect(rendered).toContain('Expertise in typescript')
  expect(rendered).not.toContain('rust')
  expect(rendered).toContain('prefers a direct answer')
})

test('profile feedback stays bounded and profiles round trip through persistent storage', () => {
  const storage = new SimpleStorage()
  const store = new UserProfileStore(storage)
  const profile = store.getOrCreate('alice')
  profile.domains.push('python')
  profile.tone = new ConfidentValue('terse', { confidence: 0.8 })
  profile.expertise.set('python', new ConfidentValue('expert', { confidence: 0.9 }))
  for (let index = 0; index < 300; index += 1) profile.recordFeedback('ping')
  store.save(profile)

  expect(profile.feedbackHistory.length).toBeLessThanOrEqual(256)
  const restored = new UserProfileStore(storage).get('alice')
  expect(restored?.domains).toEqual(['python'])
  expect(restored?.tone?.value).toBe('terse')
  expect(restored?.expertise.get('python')?.value).toBe('expert')
})

test('profile decay prunes stale facts and persists the pruned profile', () => {
  const storage = new SimpleStorage()
  const store = new UserProfileStore(storage)
  const profile = store.getOrCreate('alice')
  const stale = new ConfidentValue('expert', { confidence: 1 })
  stale.lastUpdated = new Date(Date.now() - 300 * 86_400_000)
  profile.expertise.set('obsolete', stale)
  store.save(profile)

  expect(store.decayAll({ halfLifeDays: 30, pruneThreshold: 0.05 }).alice).toBe(1)
  expect(store.get('alice')?.expertise.has('obsolete')).toBeFalse()
  expect(new UserProfileStore(storage).get('alice')?.expertise.has('obsolete')).toBeFalse()
})
