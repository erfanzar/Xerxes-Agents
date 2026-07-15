// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SkillCatalogEntry, SkillCatalogPort } from './trigger.js'

/** Explicit embedding boundary. Hosts choose whether embeddings are local, remote, or unavailable. */
export interface SkillEmbeddingPort {
  embed(text: string): readonly number[] | Promise<readonly number[]>
}

export interface SkillMatch {
  readonly score: number
  readonly skill: SkillCatalogEntry
}

export interface SkillMatchOptions {
  readonly limit?: number
  readonly skills?: readonly SkillCatalogEntry[]
}

/** Semantic recommender with only injected embeddings; it never chooses or calls a model itself. */
export class SkillMatcher {
  private readonly cache = new Map<string, readonly number[]>()
  private readonly catalog: SkillCatalogPort | undefined
  private readonly embedder: SkillEmbeddingPort
  readonly minScore: number

  constructor(
    embedder: SkillEmbeddingPort,
    options: { readonly catalog?: SkillCatalogPort; readonly minScore?: number } = {},
  ) {
    if (!Number.isFinite(options.minScore ?? 0.15)) {
      throw new RangeError('minScore must be finite')
    }
    this.embedder = embedder
    this.catalog = options.catalog
    this.minScore = options.minScore ?? 0.15
  }

  async best(query: string): Promise<SkillMatch | undefined> {
    return (await this.match(query, { limit: 1 }))[0]
  }

  invalidate(): void {
    this.cache.clear()
  }

  async match(query: string, options: SkillMatchOptions = {}): Promise<SkillMatch[]> {
    if (!query.trim()) {
      return []
    }
    const limit = options.limit ?? 5
    if (!Number.isInteger(limit) || limit < 0) {
      throw new RangeError('limit must be a non-negative integer')
    }
    const skills = options.skills ?? this.catalog?.all() ?? []
    if (!skills.length || limit === 0) {
      return []
    }
    const queryVector = await this.embedder.embed(query)
    const matches: SkillMatch[] = []
    for (const skill of skills) {
      const score = cosineSimilarity(queryVector, await this.embedSkill(skill))
      if (score >= this.minScore) {
        matches.push({ skill, score })
      }
    }
    return matches.sort((left, right) => (
      right.score - left.score || left.skill.metadata.name.localeCompare(right.skill.metadata.name)
    )).slice(0, limit)
  }

  private async embedSkill(skill: SkillCatalogEntry): Promise<readonly number[]> {
    const key = skill.metadata.name + ':' + skill.metadata.version
    const cached = this.cache.get(key)
    if (cached) {
      return cached
    }
    const vector = Object.freeze([...(await this.embedder.embed(skillText(skill)))])
    this.cache.set(key, vector)
    return vector
  }
}

/** Cosine similarity for compatible non-empty numeric vectors. */
export function cosineSimilarity(left: readonly number[], right: readonly number[]): number {
  if (!left.length || left.length !== right.length) {
    throw new RangeError('embedding vectors must be non-empty and have equal dimensions')
  }
  let dot = 0
  let leftMagnitude = 0
  let rightMagnitude = 0
  for (let index = 0; index < left.length; index += 1) {
    const a = left[index]
    const b = right[index]
    if (a === undefined || b === undefined || !Number.isFinite(a) || !Number.isFinite(b)) {
      throw new TypeError('embedding vectors must contain finite numbers')
    }
    dot += a * b
    leftMagnitude += a * a
    rightMagnitude += b * b
  }
  if (leftMagnitude === 0 || rightMagnitude === 0) {
    return 0
  }
  return dot / Math.sqrt(leftMagnitude * rightMagnitude)
}

export function skillText(skill: SkillCatalogEntry): string {
  return [
    skill.metadata.name,
    skill.metadata.description,
    skill.metadata.tags.join(' '),
    skill.instructions.slice(0, 1_000),
  ].filter(Boolean).join(' ')
}
