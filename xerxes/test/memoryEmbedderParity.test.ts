// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  HashEmbedder,
  cosineSimilarity,
  getDefaultEmbedder,
  resetDefaultEmbedder,
} from '../src/memory/index.js'

test('hash-embedder parity provides deterministic normalized vectors at the requested dimensions', () => {
  const defaultEmbedder = new HashEmbedder()
  const customEmbedder = new HashEmbedder(64)
  const empty = defaultEmbedder.embed('')
  const sentence = 'the quick brown fox jumps over the lazy dog'
  const vector = defaultEmbedder.embed(sentence)

  expect(defaultEmbedder.embed('hello world')).toHaveLength(256)
  expect(customEmbedder.embed('hello')).toHaveLength(64)
  expect(empty).toEqual(Array<number>(256).fill(0))
  expect(Math.sqrt(vector.reduce((total, value) => total + value * value, 0))).toBeCloseTo(1, 10)
  expect(defaultEmbedder.embed('hello world')).toEqual(defaultEmbedder.embed('hello world'))
  expect(defaultEmbedder.embed('the cat sat on the mat')).not.toEqual(defaultEmbedder.embed('a completely unrelated sentence'))
  expect(defaultEmbedder.embedBatch(['hello', 'world', 'foo bar']))
    .toEqual(['hello', 'world', 'foo bar'].map(text => defaultEmbedder.embed(text)))
  expect(defaultEmbedder.embedBatch([])).toEqual([])
})

test('cosine-similarity parity handles aligned, orthogonal, opposite, empty, and mismatched vectors', () => {
  expect(cosineSimilarity([1, 2, 3], [1, 2, 3])).toBeCloseTo(1)
  expect(cosineSimilarity([1, 0], [0, 1])).toBe(0)
  expect(cosineSimilarity([1, 0], [-1, 0])).toBe(-1)
  expect(cosineSimilarity([0, 0], [1, 1])).toBe(0)
  expect(cosineSimilarity([1, 2], [1, 2, 3])).toBe(0)
})

test('default embedder parity caches the dependency-free fallback until explicitly reset', () => {
  resetDefaultEmbedder()
  try {
    const first = getDefaultEmbedder()
    const second = getDefaultEmbedder()
    expect(first).toBeInstanceOf(HashEmbedder)
    expect(second).toBe(first)
    resetDefaultEmbedder()
    expect(getDefaultEmbedder()).not.toBe(first)
  } finally {
    resetDefaultEmbedder()
  }
})
