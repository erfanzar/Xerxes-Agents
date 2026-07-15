// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ClientError, ConfigurationError, ValidationError } from '../core/errors.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, JsonValue, ToolDefinition } from '../types/toolCalls.js'
import { optionalBoolean, optionalInteger, optionalString, optionalStringArray, requiredString } from './inputs.js'
import { PublicWebClient } from './webTools.js'

const DUCKDUCKGO_INSTANT_ANSWER_ENDPOINT = 'https://api.duckduckgo.com/'
const MAX_SEARCH_RESULTS = 30
const DEFAULT_TITLE_LIMIT = 200
const DEFAULT_SNIPPET_LIMIT = 1_000

export type DuckDuckGoSearchType = 'images' | 'maps' | 'news' | 'text' | 'videos'
export type DuckDuckGoSafeSearch = 'moderate' | 'off' | 'strict'
export type DuckDuckGoTimeLimit = 'day' | 'month' | 'week' | 'year'

export interface DuckDuckGoSearchRequest {
  readonly allowedDomains?: readonly string[]
  readonly excludeKeywords?: readonly string[]
  readonly excludedDomains?: readonly string[]
  readonly fileType?: string
  readonly mustIncludeKeywords?: readonly string[]
  readonly nResults?: number
  readonly query: string
  readonly region?: string
  readonly safeSearch?: DuckDuckGoSafeSearch
  readonly searchType?: DuckDuckGoSearchType
  readonly snippetLengthLimit?: number
  readonly timeLimit?: DuckDuckGoTimeLimit
  readonly titleLengthLimit?: number
}

export interface DuckDuckGoSearchResult {
  readonly description?: string
  readonly imageUrl?: string
  readonly snippet?: string
  readonly source?: string
  readonly thumbnail?: string
  readonly title: string
  readonly url: string
}

export interface DuckDuckGoSearchMetadata {
  readonly effectiveSearchType: DuckDuckGoSearchType
  readonly filtersApplied: Readonly<Record<string, JsonValue>>
  readonly query: string
  readonly searchType: DuckDuckGoSearchType
  readonly timestamp: string
  readonly totalResults: number
}

export interface DuckDuckGoSearchResponse {
  readonly metadata: DuckDuckGoSearchMetadata
  readonly results: readonly DuckDuckGoSearchResult[]
}

/** Host port for a permitted and configured full DuckDuckGo search provider. */
export interface DuckDuckGoSearchProvider {
  search(request: RequiredSearchRequest, signal?: AbortSignal): Promise<readonly DuckDuckGoSearchResult[]>
  suggestions?(query: string, region: string, signal?: AbortSignal): Promise<readonly string[]>
  translate?(query: string, toLanguage: string, signal?: AbortSignal): Promise<string>
}

interface RequiredSearchRequest {
  readonly allowedDomains: readonly string[]
  readonly excludeKeywords: readonly string[]
  readonly excludedDomains: readonly string[]
  readonly fileType?: string
  readonly mustIncludeKeywords: readonly string[]
  readonly nResults: number
  readonly query: string
  readonly region: string
  readonly safeSearch: DuckDuckGoSafeSearch
  readonly searchType: DuckDuckGoSearchType
  readonly snippetLengthLimit?: number
  readonly timeLimit?: DuckDuckGoTimeLimit
  readonly titleLengthLimit?: number
}

/** Provider-neutral search facade; it does not scrape DuckDuckGo HTML. */
export class DuckDuckGoSearch {
  constructor(private readonly provider: DuckDuckGoSearchProvider) {}

  async search(request: DuckDuckGoSearchRequest, signal?: AbortSignal): Promise<DuckDuckGoSearchResponse> {
    const normalized = normalizeRequest(request)
    const providerResults = await this.provider.search(normalized, signal)
    const results = filterResults(providerResults, normalized)
    return {
      metadata: {
        effectiveSearchType: normalized.searchType,
        filtersApplied: {
          allowed_domains: [...normalized.allowedDomains],
          excluded_domains: [...normalized.excludedDomains],
          file_type: normalized.fileType ?? null,
          keyword_filters: {
            exclude: [...normalized.excludeKeywords],
            must_include: [...normalized.mustIncludeKeywords],
          },
          region: normalized.region,
          safesearch: normalized.safeSearch,
          timelimit: normalized.timeLimit ?? null,
        },
        query: normalized.query,
        searchType: normalized.searchType,
        timestamp: new Date().toISOString(),
        totalResults: results.length,
      },
      results,
    }
  }

  async searchMultipleSources(
    query: string,
    sources: readonly DuckDuckGoSearchType[] = ['text', 'news'],
    nResultsPerSource = 3,
    signal?: AbortSignal,
  ): Promise<Readonly<Record<string, DuckDuckGoSearchResponse | { readonly error: string }>>> {
    const output: Record<string, DuckDuckGoSearchResponse | { readonly error: string }> = {}
    for (const searchType of sources) {
      try {
        output[searchType] = await this.search({ nResults: nResultsPerSource, query, searchType }, signal)
      } catch (error) {
        output[searchType] = { error: errorMessage(error) }
      }
    }
    return output
  }

  async suggestions(query: string, region = 'us-en', signal?: AbortSignal): Promise<readonly string[]> {
    if (this.provider.suggestions === undefined) {
      throw new ConfigurationError('DuckDuckGoSearch', 'configured provider does not implement suggestions')
    }
    return this.provider.suggestions(query, region, signal)
  }

  async translate(query: string, toLanguage = 'en', signal?: AbortSignal): Promise<string> {
    if (this.provider.translate === undefined) {
      throw new ConfigurationError('DuckDuckGoSearch', 'configured provider does not implement translation')
    }
    return this.provider.translate(query, toLanguage, signal)
  }
}

export interface DuckDuckGoInstantAnswerOptions {
  readonly endpoint?: string
  readonly webClient?: PublicWebClient
}

/** Public no-key Instant Answer adapter. It is intentionally limited to text answers and related topics. */
export class DuckDuckGoInstantAnswerProvider implements DuckDuckGoSearchProvider {
  private readonly endpoint: string
  private readonly webClient: PublicWebClient

  constructor(options: DuckDuckGoInstantAnswerOptions = {}) {
    this.endpoint = options.endpoint ?? DUCKDUCKGO_INSTANT_ANSWER_ENDPOINT
    this.webClient = options.webClient ?? new PublicWebClient()
  }

  async search(request: RequiredSearchRequest, signal?: AbortSignal): Promise<readonly DuckDuckGoSearchResult[]> {
    if (request.searchType !== 'text') {
      throw new ConfigurationError(
        'DuckDuckGoSearch',
        'the public Instant Answer API supports text answers only; configure a licensed provider for images, videos, news, or maps',
      )
    }
    let endpoint: URL
    try {
      endpoint = new URL(this.endpoint)
    } catch {
      throw new ConfigurationError('endpoint', 'must be an absolute DuckDuckGo Instant Answer API URL')
    }
    endpoint.searchParams.set('q', request.query)
    endpoint.searchParams.set('format', 'json')
    endpoint.searchParams.set('no_html', '1')
    endpoint.searchParams.set('skip_disambig', '1')
    endpoint.searchParams.set('kl', request.region)
    const fetched = await this.webClient.fetch(endpoint.toString(), { method: 'GET' }, signal === undefined ? {} : { signal })
    if (!fetched.response.ok) {
      throw new ClientError('duckduckgo', `Instant Answer API returned HTTP ${fetched.response.status}`)
    }
    const payload = parsePayload(await this.webClient.text(fetched.response))
    return instantAnswerResults(payload, request.nResults)
  }
}

export const DUCKDUCKGO_SEARCH_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'DuckDuckGoSearch',
    description: 'Search through an explicitly configured DuckDuckGo provider; no browser or HTML scraping is used.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        query: { type: 'string' },
        search_type: { type: 'string', enum: ['text', 'images', 'videos', 'news', 'maps'], default: 'text' },
        n_results: { type: 'integer', minimum: 1, maximum: MAX_SEARCH_RESULTS, default: 5 },
        title_length_limit: { type: 'integer', minimum: 0, default: DEFAULT_TITLE_LIMIT },
        snippet_length_limit: { type: 'integer', minimum: 0, default: DEFAULT_SNIPPET_LIMIT },
        region: { type: 'string', default: 'us-en' },
        safesearch: { type: 'string', enum: ['strict', 'moderate', 'off'], default: 'moderate' },
        timelimit: { type: 'string', enum: ['day', 'week', 'month', 'year'] },
        allowed_domains: { type: 'array', items: { type: 'string' } },
        excluded_domains: { type: 'array', items: { type: 'string' } },
        must_include_keywords: { type: 'array', items: { type: 'string' } },
        exclude_keywords: { type: 'array', items: { type: 'string' } },
        file_type: { type: 'string' },
        return_metadata: { type: 'boolean', default: false },
      },
      required: ['query'],
    },
  },
}

/** Register DuckDuckGoSearch only when the host supplies a public API or licensed provider adapter. */
export function registerDuckDuckGoSearchTool(
  registry: ToolRegistry,
  search: DuckDuckGoSearch,
  agentId = 'default',
): void {
  registry.register(DUCKDUCKGO_SEARCH_DEFINITION, async (inputs, _context, signal) => {
    const result = await search.search(searchRequestFromInputs(inputs), signal)
    return optionalBoolean(inputs, 'return_metadata', false) ? result : result.results
  }, agentId)
}

export function duckDuckGoSearchLimitations(): readonly string[] {
  return Object.freeze([
    'The public DuckDuckGo Instant Answer API has no API key but does not provide general web, image, video, news, or map search.',
    'Full DuckDuckGo search requires a host-provided provider adapter with its own authorized API or service agreement.',
    'DuckDuckGo HTML scraping and browser automation are deliberately not implemented.',
  ])
}

function normalizeRequest(request: DuckDuckGoSearchRequest): RequiredSearchRequest {
  const query = request.query.trim()
  if (!query) throw new ValidationError('query', 'must be non-empty', request.query)
  const nResults = requestedResultCount(request.nResults ?? 5)
  const searchType = request.searchType ?? 'text'
  const titleLengthLimit = optionalLengthLimit(request.titleLengthLimit, 'title_length_limit')
  const snippetLengthLimit = optionalLengthLimit(request.snippetLengthLimit, 'snippet_length_limit')
  const fileType = request.fileType?.trim()
  const allowedDomains = normalizedStrings(request.allowedDomains ?? [])
  const excludedDomains = normalizedStrings(request.excludedDomains ?? [])
  const normalizedQuery = [
    query,
    fileType ? `filetype:${fileType}` : '',
    allowedDomains.length ? `(${allowedDomains.map(domain => `site:${domain}`).join(' OR ')})` : '',
    ...excludedDomains.map(domain => `-site:${domain}`),
  ].filter(Boolean).join(' ')
  return {
    allowedDomains,
    excludeKeywords: normalizedStrings(request.excludeKeywords ?? []),
    excludedDomains,
    ...(fileType ? { fileType } : {}),
    mustIncludeKeywords: normalizedStrings(request.mustIncludeKeywords ?? []),
    nResults,
    query: normalizedQuery,
    region: request.region?.trim() || 'us-en',
    safeSearch: request.safeSearch ?? 'moderate',
    searchType,
    ...(snippetLengthLimit === undefined ? {} : { snippetLengthLimit }),
    ...(request.timeLimit === undefined ? {} : { timeLimit: request.timeLimit }),
    ...(titleLengthLimit === undefined ? {} : { titleLengthLimit }),
  }
}

function filterResults(results: readonly DuckDuckGoSearchResult[], request: RequiredSearchRequest): DuckDuckGoSearchResult[] {
  const output: DuckDuckGoSearchResult[] = []
  for (const result of results) {
    if (!isHttpUrl(result.url)) continue
    if (!matchesDomains(result.url, request.allowedDomains, request.excludedDomains)) continue
    const searchable = `${result.title} ${result.snippet ?? result.description ?? ''}`.toLowerCase()
    if (request.mustIncludeKeywords.some(keyword => !searchable.includes(keyword.toLowerCase()))) continue
    if (request.excludeKeywords.some(keyword => searchable.includes(keyword.toLowerCase()))) continue
    output.push({
      ...(result.description === undefined ? {} : { description: truncate(result.description, request.snippetLengthLimit) }),
      ...(result.imageUrl === undefined ? {} : { imageUrl: result.imageUrl }),
      ...(result.snippet === undefined ? {} : { snippet: truncate(result.snippet, request.snippetLengthLimit) }),
      ...(result.source === undefined ? {} : { source: result.source }),
      ...(result.thumbnail === undefined ? {} : { thumbnail: result.thumbnail }),
      title: truncate(result.title, request.titleLengthLimit),
      url: result.url,
    })
    if (output.length >= request.nResults) break
  }
  return output
}

function parsePayload(value: string): JsonObject {
  const parsed = JSON.parse(value) as JsonValue
  if (!isObject(parsed)) throw new ClientError('duckduckgo', 'Instant Answer API returned a non-object JSON payload')
  return parsed
}

function instantAnswerResults(payload: JsonObject, maximum: number): DuckDuckGoSearchResult[] {
  const results: DuckDuckGoSearchResult[] = []
  const abstractUrl = stringValue(payload.AbstractURL)
  const abstractText = stringValue(payload.AbstractText)
  if (abstractUrl && abstractText && isHttpUrl(abstractUrl)) {
    results.push({
      snippet: abstractText,
      source: stringValue(payload.AbstractSource) || 'DuckDuckGo Instant Answer',
      title: stringValue(payload.Heading) || abstractUrl,
      url: abstractUrl,
    })
  }
  for (const topic of flattenedTopics(payload.RelatedTopics)) {
    const url = stringValue(topic.FirstURL)
    const text = stringValue(topic.Text)
    if (!url || !text || !isHttpUrl(url)) continue
    results.push({ source: 'DuckDuckGo Instant Answer', snippet: text, title: text, url })
    if (results.length >= maximum) break
  }
  return results.slice(0, maximum)
}

function flattenedTopics(value: JsonValue | undefined): JsonObject[] {
  if (!Array.isArray(value)) return []
  const output: JsonObject[] = []
  for (const entry of value) {
    if (!isObject(entry)) continue
    const nested = entry.Topics
    if (nested !== undefined) output.push(...flattenedTopics(nested))
    else output.push(entry)
  }
  return output
}

function searchRequestFromInputs(inputs: JsonObject): DuckDuckGoSearchRequest {
  const request: {
    allowedDomains: readonly string[]
    excludeKeywords: readonly string[]
    excludedDomains: readonly string[]
    fileType?: string
    mustIncludeKeywords: readonly string[]
    nResults: number
    query: string
    region: string
    safeSearch: DuckDuckGoSafeSearch
    searchType: DuckDuckGoSearchType
    snippetLengthLimit: number
    timeLimit?: DuckDuckGoTimeLimit
    titleLengthLimit: number
  } = {
    allowedDomains: optionalStringArray(inputs, 'allowed_domains'),
    excludeKeywords: optionalStringArray(inputs, 'exclude_keywords'),
    excludedDomains: optionalStringArray(inputs, 'excluded_domains'),
    mustIncludeKeywords: optionalStringArray(inputs, 'must_include_keywords'),
    nResults: optionalInteger(inputs, 'n_results', 5),
    query: requiredString(inputs, 'query'),
    region: optionalString(inputs, 'region') ?? 'us-en',
    safeSearch: enumValue(inputs, 'safesearch', ['strict', 'moderate', 'off'], 'moderate'),
    searchType: enumValue(inputs, 'search_type', ['text', 'images', 'videos', 'news', 'maps'], 'text'),
    snippetLengthLimit: optionalInteger(inputs, 'snippet_length_limit', DEFAULT_SNIPPET_LIMIT),
    titleLengthLimit: optionalInteger(inputs, 'title_length_limit', DEFAULT_TITLE_LIMIT),
  }
  const fileType = optionalString(inputs, 'file_type')
  const timeLimit = optionalString(inputs, 'timelimit')
  if (fileType !== undefined) request.fileType = fileType
  if (timeLimit !== undefined) {
    if (!['day', 'week', 'month', 'year'].includes(timeLimit)) {
      throw new ValidationError('timelimit', 'must be day, week, month, or year', timeLimit)
    }
    request.timeLimit = timeLimit as DuckDuckGoTimeLimit
  }
  return request
}

function enumValue<T extends string>(inputs: JsonObject, name: string, values: readonly T[], defaultValue: T): T {
  const value = optionalString(inputs, name) ?? defaultValue
  if (!values.includes(value as T)) throw new ValidationError(name, `must be one of ${values.join(', ')}`, value)
  return value as T
}

function requestedResultCount(value: number): number {
  if (!Number.isInteger(value) || value < 1 || value > MAX_SEARCH_RESULTS) {
    throw new ValidationError('n_results', `must be an integer between 1 and ${MAX_SEARCH_RESULTS}`, value)
  }
  return value
}

function optionalLengthLimit(value: number | undefined, name: string): number | undefined {
  if (value === undefined) return undefined
  if (!Number.isInteger(value) || value < 0) throw new ValidationError(name, 'must be a non-negative integer', value)
  return value
}

function normalizedStrings(values: readonly string[]): string[] {
  return values.map(value => value.trim()).filter(Boolean)
}

function matchesDomains(url: string, allowed: readonly string[], excluded: readonly string[]): boolean {
  let hostname: string
  try {
    hostname = new URL(url).hostname.toLowerCase()
  } catch {
    return false
  }
  const matches = (domain: string) => hostname === cleanDomain(domain) || hostname.endsWith(`.${cleanDomain(domain)}`)
  if (allowed.length && !allowed.some(matches)) return false
  return !excluded.some(matches)
}

function cleanDomain(value: string): string {
  try {
    return new URL(value.includes('://') ? value : `https://${value}`).hostname.toLowerCase()
  } catch {
    return value.toLowerCase().replace(/^\.+|\.+$/g, '')
  }
}

function isHttpUrl(value: string): boolean {
  try {
    const parsed = new URL(value)
    return parsed.protocol === 'http:' || parsed.protocol === 'https:'
  } catch {
    return false
  }
}

function truncate(value: string, limit: number | undefined): string {
  return limit === undefined ? value : value.slice(0, limit)
}

function isObject(value: JsonValue | undefined): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringValue(value: JsonValue | undefined): string {
  return typeof value === 'string' ? value : ''
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
