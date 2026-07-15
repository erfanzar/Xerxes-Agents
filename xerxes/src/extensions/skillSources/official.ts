// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  normalizeSkillSearchLimit,
  requireSkillIdentifier,
  requireSkillSearchQuery,
  SkillSourceError,
  type SkillBundle,
  type SkillSearchHit,
  type SkillSource,
} from './base.js'

export const DEFAULT_OFFICIAL_SKILL_REGISTRY_URL = 'https://skills.xerxes-agent.dev'

/** The minimal response shape the source needs from a host-approved fetch implementation. */
export interface SkillRegistryFetchResponse {
  readonly ok: boolean
  readonly status: number
  text(): Promise<string>
}

/** Explicit remote boundary. The source never falls back to global `fetch`. */
export interface SkillRegistryFetchTransport {
  fetch(url: string, init: Readonly<{ method: 'GET' }>): Promise<SkillRegistryFetchResponse>
}

export interface OfficialSkillSourceOptions {
  readonly baseUrl?: string
  readonly transport: SkillRegistryFetchTransport
}

/** Raised for a non-successful registry download rather than manufacturing an empty bundle. */
export class SkillRegistryHttpError extends SkillSourceError {
  readonly status: number

  constructor(url: string, status: number) {
    super(`skill registry request failed with HTTP ${status}: ${url}`)
    this.name = 'SkillRegistryHttpError'
    this.status = status
  }
}

/** HTTP-backed official catalogue whose transport is supplied by the embedding application. */
export class OfficialSkillSource implements SkillSource {
  readonly baseUrl: string
  readonly name: string = 'official'
  protected readonly transport: SkillRegistryFetchTransport

  constructor(options: OfficialSkillSourceOptions) {
    this.baseUrl = normalizeRegistryBaseUrl(options.baseUrl ?? DEFAULT_OFFICIAL_SKILL_REGISTRY_URL)
    this.transport = options.transport
  }

  async search(query: string, limit = 20): Promise<readonly SkillSearchHit[]> {
    const normalizedQuery = requireSkillSearchQuery(query)
    const normalizedLimit = normalizeSkillSearchLimit(limit)
    if (normalizedLimit === 0) return []
    const url = registryUrl(this.baseUrl, 'search', { q: normalizedQuery, limit: String(normalizedLimit) })
    try {
      const response = await this.transport.fetch(url, { method: 'GET' })
      if (!response.ok) return []
      const rows = parseSearchRows(await response.text())
      return rows.slice(0, normalizedLimit).map(row => ({
        name: row.name,
        description: row.description,
        sourceName: this.name,
        version: row.version,
        tags: row.tags,
      }))
    } catch {
      // Search is best-effort by design, matching the Python registry source.
      return []
    }
  }

  async fetch(identifier: string): Promise<SkillBundle> {
    const id = requireSkillIdentifier(identifier)
    const url = registryUrl(this.baseUrl, `skills/${encodeURIComponent(id)}/SKILL.md`)
    const response = await this.transport.fetch(url, { method: 'GET' })
    if (!response.ok) throw new SkillRegistryHttpError(url, response.status)
    return {
      name: id,
      version: 'official',
      bodyMarkdown: await response.text(),
      metadata: {},
      sourceName: this.name,
    }
  }
}

interface RegistrySearchRow {
  readonly description: string
  readonly name: string
  readonly tags: readonly string[]
  readonly version: string
}

function normalizeRegistryBaseUrl(value: string): string {
  try {
    const url = new URL(value)
    if (url.protocol !== 'http:' && url.protocol !== 'https:') {
      throw new SkillSourceError('skill registry baseUrl must use http or https')
    }
    url.hash = ''
    url.search = ''
    url.pathname = `${url.pathname.replace(/\/+$/, '')}/`
    return url.toString()
  } catch (error) {
    if (error instanceof SkillSourceError) throw error
    throw new SkillSourceError(`invalid skill registry baseUrl: ${value}`, { cause: error })
  }
}

function registryUrl(baseUrl: string, path: string, query: Readonly<Record<string, string>> = {}): string {
  const url = new URL(path, baseUrl)
  for (const [key, value] of Object.entries(query)) url.searchParams.set(key, value)
  return url.toString()
}

function parseSearchRows(body: string): readonly RegistrySearchRow[] {
  const value: unknown = JSON.parse(body)
  if (!Array.isArray(value)) throw new SkillSourceError('skill registry search response must be a JSON array')
  return value.map(row => {
    if (!isRecord(row) || typeof row.name !== 'string' || !row.name.trim()) {
      throw new SkillSourceError('skill registry search row must include a skill name')
    }
    return {
      name: row.name,
      description: typeof row.description === 'string' ? row.description : '',
      version: typeof row.version === 'string' ? row.version : '',
      tags: Array.isArray(row.tags) && row.tags.every(tag => typeof tag === 'string') ? [...row.tags] : [],
    }
  })
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
