// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  normalizeSkillSearchLimit,
  requireSkillIdentifier,
  requireSkillSearchQuery,
  SkillSourceConfigurationError,
  SkillSourceError,
  type SkillBundle,
  type SkillSearchHit,
  type SkillSource,
} from './base.js'

export const DEFAULT_GITHUB_SKILL_REPOSITORY = 'agentskills/community'

export interface GitHubSkillFetchRequest {
  readonly identifier: string
  readonly repository: string
}

export interface GitHubSkillSearchRequest {
  readonly limit: number
  readonly query: string
  readonly repository: string
}

export interface GitHubSkillSearchRow {
  readonly description?: string
  readonly name: string
  readonly tags?: readonly string[]
  readonly version?: string
}

/** Host-owned GitHub integration. It may use an API client, a cache, or an approved local mirror. */
export interface GitHubSkillSourcePorts {
  readonly fetchSkillMarkdown?: (request: GitHubSkillFetchRequest) => string | Promise<string>
  readonly searchSkills?: (
    request: GitHubSkillSearchRequest,
  ) => readonly GitHubSkillSearchRow[] | Promise<readonly GitHubSkillSearchRow[]>
}

export interface GitHubSkillSourceOptions {
  readonly ports?: GitHubSkillSourcePorts
  readonly repository?: string
}

/** GitHub-backed source that performs no HTTP, authentication, or shell work itself. */
export class GitHubSkillSource implements SkillSource {
  readonly name = 'github'
  readonly repository: string
  private readonly ports: GitHubSkillSourcePorts

  constructor(options: GitHubSkillSourceOptions = {}) {
    const repository = options.repository ?? DEFAULT_GITHUB_SKILL_REPOSITORY
    if (typeof repository !== 'string' || !repository.trim()) {
      throw new SkillSourceError('GitHub skill repository must be a non-empty string')
    }
    this.repository = repository
    this.ports = options.ports ?? {}
  }

  async search(query: string, limit = 20): Promise<readonly SkillSearchHit[]> {
    const normalizedQuery = requireSkillSearchQuery(query)
    const normalizedLimit = normalizeSkillSearchLimit(limit)
    if (normalizedLimit === 0 || this.ports.searchSkills === undefined) return []
    const rows = await this.ports.searchSkills({
      query: normalizedQuery,
      limit: normalizedLimit,
      repository: this.repository,
    })
    return rows.slice(0, normalizedLimit).map(row => {
      if (typeof row.name !== 'string' || !row.name.trim()) {
        throw new SkillSourceError('GitHub search returned a row without a skill name')
      }
      return {
        name: row.name,
        description: row.description ?? '',
        sourceName: this.name,
        version: row.version ?? '',
        tags: [...(row.tags ?? [])],
      }
    })
  }

  async fetch(identifier: string): Promise<SkillBundle> {
    const id = requireSkillIdentifier(identifier)
    if (this.ports.fetchSkillMarkdown === undefined) {
      throw new SkillSourceConfigurationError('GitHubSkillSource was not configured with a fetchSkillMarkdown port')
    }
    const bodyMarkdown = await this.ports.fetchSkillMarkdown({ identifier: id, repository: this.repository })
    if (typeof bodyMarkdown !== 'string') {
      throw new SkillSourceError('GitHub fetchSkillMarkdown port must return a markdown string')
    }
    return {
      name: id,
      version: 'github',
      bodyMarkdown,
      metadata: { repository: this.repository },
      sourceName: this.name,
    }
  }
}
