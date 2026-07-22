// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { cosineSimilarity, getDefaultEmbedder, type Embedder } from './embedders.js'
import type { MemoryItem } from './base.js'

export interface RetrievalWeightsOptions {
  readonly bm25?: number
  readonly recency?: number
  readonly semantic?: number
}

export class RetrievalWeights {
  readonly bm25: number
  readonly recency: number
  readonly semantic: number

  constructor(options: RetrievalWeightsOptions = {}) {
    this.semantic = options.semantic ?? 0.55
    this.bm25 = options.bm25 ?? 0.3
    this.recency = options.recency ?? 0.15
  }

  normalized(): RetrievalWeights {
    const total = this.semantic + this.bm25 + this.recency
    return total === 0
      ? new RetrievalWeights()
      : new RetrievalWeights({ semantic: this.semantic / total, bm25: this.bm25 / total, recency: this.recency / total })
  }
}

export interface RetrievalResult {
  readonly bm25Score: number
  readonly item: MemoryItem
  readonly recencyScore: number
  readonly score: number
  readonly semanticScore: number
}

/** Batch-local hybrid ranking that blends hash semantic, BM25, and recency signals. */
export class HybridRetriever {
  readonly weights: RetrievalWeights
  /**
   * Per-item embedding cache keyed by object identity. Ranking used to
   * re-embed every candidate's content on every query — O(n) hashing per
   * search. Content is mutable, so entries are validated against the exact
   * content string they were computed from. The WeakMap lets collected
   * items drop their cached vectors.
   */
  private readonly embeddingCache = new WeakMap<MemoryItem, { readonly content: string; readonly embedding: number[] }>()

  constructor(
    readonly embedder: Embedder = getDefaultEmbedder(),
    weights: RetrievalWeights = new RetrievalWeights(),
    readonly recencyHalfLifeDays = 14,
    readonly bm25K1 = 1.5,
    readonly bm25B = 0.75,
  ) {
    this.weights = weights.normalized()
  }

  rank(query: string, items: readonly MemoryItem[], limit = 10, now = new Date()): RetrievalResult[] {
    if (items.length === 0) return []
    const queryEmbedding = this.embedder.embed(query)
    const rawBm25 = this.bm25(query, items)
    const maxBm25 = Math.max(0, ...rawBm25)
    const results = items.map((item, index) => {
      const itemEmbedding = item.embedding ?? this.embeddingFor(item)
      const semanticScore = Math.max(0, cosineSimilarity(queryEmbedding, itemEmbedding))
      const bm25Score = maxBm25 === 0 ? 0 : (rawBm25[index] ?? 0) / maxBm25
      const recencyScore = this.recency(item.timestamp, now)
      const score = this.weights.semantic * semanticScore
        + this.weights.bm25 * bm25Score
        + this.weights.recency * recencyScore
      return { item, score, semanticScore, bm25Score, recencyScore }
    })
    return results.sort((left, right) => right.score - left.score).slice(0, limit)
  }

  /** Embed one item's content once per distinct content value. */
  private embeddingFor(item: MemoryItem): number[] {
    const cached = this.embeddingCache.get(item)
    if (cached && cached.content === item.content) return cached.embedding
    const embedding = this.embedder.embed(item.content)
    this.embeddingCache.set(item, { content: item.content, embedding })
    return embedding
  }

  private bm25(query: string, items: readonly MemoryItem[]): number[] {
    const queryTerms = tokenize(query)
    if (queryTerms.length === 0) return items.map(() => 0)
    const documents = items.map(item => tokenize(item.content))
    const averageLength = documents.reduce((total, document) => total + document.length, 0) / documents.length || 1
    const documentFrequency = new Map<string, number>()
    for (const document of documents) {
      for (const term of new Set(document)) {
        documentFrequency.set(term, (documentFrequency.get(term) ?? 0) + 1)
      }
    }
    return documents.map(document => {
      const frequencies = new Map<string, number>()
      for (const term of document) frequencies.set(term, (frequencies.get(term) ?? 0) + 1)
      return [...new Set(queryTerms)].reduce((score, term) => {
        const frequency = frequencies.get(term)
        if (!frequency) return score
        const documentCount = documentFrequency.get(term) ?? 0
        const inverseFrequency = Math.log(1 + (documents.length - documentCount + 0.5) / (documentCount + 0.5))
        const numerator = frequency * (this.bm25K1 + 1)
        const denominator = frequency + this.bm25K1 * (1 - this.bm25B + this.bm25B * document.length / averageLength)
        return score + inverseFrequency * numerator / Math.max(denominator, 1e-9)
      }, 0)
    })
  }

  private recency(timestamp: Date, now: Date): number {
    const ageDays = Math.max(0, (now.valueOf() - timestamp.valueOf()) / 86_400_000)
    return 2 ** (-ageDays / Math.max(this.recencyHalfLifeDays, 0.001))
  }
}

function tokenize(text: string): string[] {
  return text.toLowerCase().split(/\s+/).map(token => token.replace(/[^\p{L}\p{N}]/gu, '')).filter(Boolean)
}
