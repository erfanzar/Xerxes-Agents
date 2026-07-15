// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ConfigurationError, ValidationError } from '../core/errors.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, JsonValue, ToolDefinition } from '../types/toolCalls.js'
import { optionalInteger, optionalString, requiredString } from './inputs.js'
import { PublicWebClient } from './webTools.js'

const GOOGLE_CUSTOM_SEARCH_ENDPOINT = 'https://www.googleapis.com/customsearch/v1'
const GOOGLE_RESULT_LIMIT = 10

export interface GoogleCustomSearchOptions {
  readonly apiKey: string
  readonly endpoint?: string
  readonly safeSearch?: 'active' | 'off'
  readonly searchEngineId: string
  readonly webClient?: PublicWebClient
}

export interface GoogleSearchRequest {
  readonly nResults?: number
  readonly query: string
  readonly site?: string
  readonly timeRange?: string
}

export interface GoogleSearchResult {
  readonly displayedUrl: string
  readonly snippet: string
  readonly title: string
  readonly url: string
}

export interface GoogleSearchResponse {
  readonly count?: number
  readonly engine: 'google_api'
  readonly error?: string
  readonly query: string
  readonly results: readonly GoogleSearchResult[]
  readonly searchInformation?: JsonObject
}

/** Google Custom Search JSON API client. It intentionally has no anonymous HTML-scraping fallback. */
export class GoogleCustomSearchClient {
  private readonly apiKey: string
  private readonly endpoint: string
  private readonly safeSearch: 'active' | 'off'
  private readonly searchEngineId: string
  private readonly webClient: PublicWebClient

  constructor(options: GoogleCustomSearchOptions) {
    this.apiKey = requiredConfiguration(options.apiKey, 'apiKey')
    this.searchEngineId = requiredConfiguration(options.searchEngineId, 'searchEngineId')
    this.endpoint = options.endpoint ?? GOOGLE_CUSTOM_SEARCH_ENDPOINT
    this.safeSearch = options.safeSearch ?? 'off'
    this.webClient = options.webClient ?? new PublicWebClient()
  }

  async search(request: GoogleSearchRequest, signal?: AbortSignal): Promise<GoogleSearchResponse> {
    const query = request.query.trim()
    if (!query) throw new ValidationError('query', 'must be non-empty', request.query)
    const nResults = requestedResultCount(request.nResults ?? 5)
    const url = this.requestUrl(query, nResults, request.site, request.timeRange)
    const fetched = await this.webClient.fetch(url, { method: 'GET' }, signal === undefined ? {} : { signal })
    if (!fetched.response.ok) {
      return { engine: 'google_api', error: `HTTP ${fetched.response.status}`, query, results: [] }
    }

    const responseText = await this.webClient.text(fetched.response)
    let payload: JsonObject
    try {
      payload = parseObject(responseText)
    } catch {
      return { engine: 'google_api', error: 'Google Custom Search returned invalid JSON', query, results: [] }
    }
    const results = googleResults(payload.items, nResults)
    const response: {
      count: number
      engine: 'google_api'
      query: string
      results: readonly GoogleSearchResult[]
      searchInformation?: JsonObject
    } = {
      count: results.length,
      engine: 'google_api',
      query,
      results,
    }
    const searchInformation = payload.searchInformation
    if (isObject(searchInformation)) response.searchInformation = searchInformation
    return response
  }

  private requestUrl(query: string, nResults: number, site: string | undefined, timeRange: string | undefined): string {
    let endpoint: URL
    try {
      endpoint = new URL(this.endpoint)
    } catch {
      throw new ConfigurationError('endpoint', 'must be an absolute Google Custom Search API URL')
    }
    if (endpoint.protocol !== 'http:' && endpoint.protocol !== 'https:') {
      throw new ConfigurationError('endpoint', 'must use HTTP or HTTPS')
    }
    const sitePrefix = site?.trim()
    endpoint.searchParams.set('key', this.apiKey)
    endpoint.searchParams.set('cx', this.searchEngineId)
    endpoint.searchParams.set('q', sitePrefix ? `site:${sitePrefix} ${query}` : query)
    endpoint.searchParams.set('num', String(nResults))
    endpoint.searchParams.set('safe', this.safeSearch)
    if (timeRange !== undefined) {
      if (!/^[dwmy]\d*$/i.test(timeRange)) {
        throw new ValidationError('time_range', 'must use Google dateRestrict syntax such as d, w, m, y, or m6', timeRange)
      }
      endpoint.searchParams.set('dateRestrict', timeRange)
    }
    return endpoint.toString()
  }
}

export const GOOGLE_SEARCH_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'GoogleSearch',
    description: 'Search through a configured Google Custom Search JSON API key and search-engine id.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        query: { type: 'string' },
        n_results: { type: 'integer', minimum: 1, maximum: GOOGLE_RESULT_LIMIT, default: 5 },
        site: { type: 'string' },
        time_range: { type: 'string', description: 'Google dateRestrict value such as d, w, m, y, or m6.' },
      },
      required: ['query'],
    },
  },
}

/** Register GoogleSearch only after the host explicitly provides API credentials. */
export function registerGoogleSearchTool(
  registry: ToolRegistry,
  search: GoogleCustomSearchClient,
  agentId = 'default',
): void {
  registry.register(GOOGLE_SEARCH_DEFINITION, (inputs, _context, signal) => search.search(googleRequestFromInputs(inputs), signal), agentId)
}

export function googleSearchLimitations(): readonly string[] {
  return Object.freeze([
    'GoogleSearch requires an explicit Google Custom Search API key and search-engine id.',
    'Anonymous Google HTML scraping is deliberately not implemented.',
  ])
}

function requiredConfiguration(value: string, name: string): string {
  const normalized = value.trim()
  if (!normalized) throw new ConfigurationError(name, 'must be explicitly configured')
  return normalized
}

function requestedResultCount(value: number): number {
  if (!Number.isInteger(value) || value < 1) {
    throw new ValidationError('n_results', 'must be a positive integer', value)
  }
  return Math.min(value, GOOGLE_RESULT_LIMIT)
}

function googleRequestFromInputs(inputs: JsonObject): GoogleSearchRequest {
  const request: { nResults: number; query: string; site?: string; timeRange?: string } = {
    nResults: optionalInteger(inputs, 'n_results', 5),
    query: requiredString(inputs, 'query'),
  }
  const site = optionalString(inputs, 'site')
  const timeRange = optionalString(inputs, 'time_range')
  if (site !== undefined) request.site = site
  if (timeRange !== undefined) request.timeRange = timeRange
  return request
}

function googleResults(value: JsonValue | undefined, maximum: number): GoogleSearchResult[] {
  if (!Array.isArray(value)) return []
  const results: GoogleSearchResult[] = []
  for (const item of value) {
    if (!isObject(item)) continue
    const url = stringValue(item.link)
    if (!url) continue
    results.push({
      displayedUrl: stringValue(item.displayLink),
      snippet: stringValue(item.snippet),
      title: stringValue(item.title),
      url,
    })
    if (results.length >= maximum) break
  }
  return results
}

function parseObject(value: string): JsonObject {
  const parsed = JSON.parse(value) as JsonValue
  if (!isObject(parsed)) throw new Error('expected object')
  return parsed
}

function isObject(value: JsonValue | undefined): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringValue(value: JsonValue | undefined): string {
  return typeof value === 'string' ? value : ''
}
