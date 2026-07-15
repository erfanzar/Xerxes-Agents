// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'

export type Vector = number[]

export interface Embedder {
  readonly dimension: number
  readonly name: string
  embed(text: string): Vector
  embedBatch(texts: readonly string[]): Vector[]
}

/** Dependency-free, deterministic hashed bag-of-words embedder. */
export class HashEmbedder implements Embedder {
  readonly name = 'hash'

  constructor(readonly dimension = 256) {
    if (!Number.isInteger(dimension) || dimension <= 0) {
      throw new Error('HashEmbedder dimension must be a positive integer')
    }
  }

  embed(text: string): Vector {
    const tokens = text.toLowerCase().split(/\s+/).filter(Boolean)
    const vector = Array<number>(this.dimension).fill(0)
    if (tokens.length === 0) return vector
    for (const token of tokens) {
      const digest = createHash('sha256').update(token).digest()
      const bucket = digest.readUInt32LE(0) % this.dimension
      vector[bucket] = (vector[bucket] ?? 0) + 1 / tokens.length
    }
    return normalize(vector)
  }

  embedBatch(texts: readonly string[]): Vector[] {
    return texts.map(text => this.embed(text))
  }
}

let defaultEmbedder: Embedder | undefined

/** Return the process-wide dependency-free default embedder. */
export function getDefaultEmbedder(): Embedder {
  defaultEmbedder ??= new HashEmbedder()
  return defaultEmbedder
}

/** Clear the cached default, primarily for isolated tests. */
export function resetDefaultEmbedder(): void {
  defaultEmbedder = undefined
}

export function cosineSimilarity(left: readonly number[], right: readonly number[]): number {
  if (left.length !== right.length || left.length === 0) return 0
  let dot = 0
  let leftNorm = 0
  let rightNorm = 0
  for (let index = 0; index < left.length; index += 1) {
    const a = left[index] ?? 0
    const b = right[index] ?? 0
    dot += a * b
    leftNorm += a * a
    rightNorm += b * b
  }
  if (leftNorm === 0 || rightNorm === 0) return 0
  return dot / Math.sqrt(leftNorm * rightNorm)
}

function normalize(vector: Vector): Vector {
  const norm = Math.sqrt(vector.reduce((total, value) => total + value * value, 0))
  return norm === 0 ? vector : vector.map(value => value / norm)
}
