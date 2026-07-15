// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { requireSkillText, skillFetchText, type SkillFetch } from './http.js'

/** arXiv's public Atom query endpoint. */
export const ARXIV_QUERY_URL = 'https://export.arxiv.org/api/query'

export interface ArxivSearchOptions {
  /** Free-text query. Ignored when `ids` is supplied. */
  readonly query?: string
  /** Restrict the search to an author. */
  readonly author?: string
  /** Restrict the search to an arXiv category such as `cs.AI`. */
  readonly category?: string
  /** Comma-separated arXiv identifiers. Takes precedence over all filters. */
  readonly ids?: string
  /** Maximum result count accepted by arXiv. Defaults to five. */
  readonly maxResults?: number
  /** `relevance`, `date`, or `updated`; other arXiv sort values pass through unchanged. */
  readonly sort?: string
}

export interface ArxivPaper {
  readonly abstract: string
  readonly arxivId: string
  readonly authors: readonly string[]
  readonly categories: readonly string[]
  readonly pdfUrl: string
  readonly published: string
  readonly title: string
  readonly updated: string
  readonly url: string
  readonly version: string
}

export interface ArxivSearchResult {
  readonly papers: readonly ArxivPaper[]
  readonly totalResults?: number
}

export interface ArxivClientOptions {
  readonly endpoint?: string
  readonly fetchImplementation?: SkillFetch
}

/** Build an arXiv Atom endpoint using the filters accepted by the bundled research skill. */
export function arxivSearchUrl(options: ArxivSearchOptions = {}, endpoint = ARXIV_QUERY_URL): string {
  const ids = optionalText(options.ids)
  const query = optionalText(options.query)
  const author = optionalText(options.author)
  const category = optionalText(options.category)
  const maxResults = options.maxResults ?? 5
  if (!Number.isSafeInteger(maxResults) || maxResults < 0) {
    throw new RangeError('maxResults must be a non-negative safe integer')
  }

  const params: string[] = []
  if (ids) {
    params.push(`id_list=${ids}`)
  } else {
    const parts: string[] = []
    if (query) parts.push(`all:${arxivQuote(query)}`)
    if (author) parts.push(`au:${arxivQuote(author)}`)
    if (category) parts.push(`cat:${category}`)
    if (!parts.length) {
      throw new TypeError('provide an arXiv query, author, category, or id list')
    }
    params.push(`search_query=${parts.join('+AND+')}`)
  }

  const sort = options.sort ?? 'relevance'
  const sortBy = sort === 'date' ? 'submittedDate' : sort === 'updated' ? 'lastUpdatedDate' : sort
  params.push(`max_results=${maxResults}`)
  params.push(`sortBy=${sortBy}`)
  params.push('sortOrder=descending')
  return `${requireSkillText(endpoint, 'endpoint').replace(/\?+$/, '')}?${params.join('&')}`
}

/** Parse an arXiv Atom feed into the native, data-first result model. */
export function parseArxivAtom(xml: string): ArxivSearchResult {
  const papers = atomElements(xml, 'entry').map(parseArxivEntry)
  const totalText = atomFirst(xml, 'totalResults')
  const totalNumber = totalText === undefined ? undefined : Number(totalText)
  const totalResults = typeof totalNumber === 'number' && Number.isFinite(totalNumber) ? totalNumber : undefined
  return {
    papers,
    ...(totalResults === undefined ? {} : { totalResults }),
  }
}

/** Render a result in the human-readable shape used by the original CLI helper. */
export function formatArxivResult(result: ArxivSearchResult): string {
  if (!result.papers.length) return 'No results found.'
  const lines: string[] = []
  if (result.totalResults !== undefined) {
    lines.push(`Found ${result.totalResults} results (showing ${result.papers.length})`, '')
  }
  for (const [index, paper] of result.papers.entries()) {
    const abstract = paper.abstract.length > 300 ? `${paper.abstract.slice(0, 300)}...` : paper.abstract
    lines.push(
      `${index + 1}. ${paper.title}`,
      `   ID: ${paper.arxivId}${paper.version} | Published: ${paper.published} | Updated: ${paper.updated}`,
      `   Authors: ${paper.authors.join(', ')}`,
      `   Categories: ${paper.categories.join(', ')}`,
      `   Abstract: ${abstract}`,
      `   Links: ${paper.url} | ${paper.pdfUrl}`,
      '',
    )
  }
  return lines.join('\n').trimEnd()
}

/** Native fetch client for arXiv's Atom API. It has no Python or SDK dependency. */
export class ArxivClient {
  private readonly endpoint: string
  private readonly fetchImplementation: SkillFetch

  constructor(options: ArxivClientOptions = {}) {
    this.endpoint = options.endpoint ?? ARXIV_QUERY_URL
    this.fetchImplementation = options.fetchImplementation ?? fetch
  }

  async search(options: ArxivSearchOptions = {}, signal?: AbortSignal): Promise<ArxivSearchResult> {
    const endpoint = arxivSearchUrl(options, this.endpoint)
    const xml = await skillFetchText(this.fetchImplementation, endpoint, {
      headers: { Accept: 'application/atom+xml, application/xml;q=0.9', 'User-Agent': 'Xerxes/1.0' },
      ...(signal === undefined ? {} : { signal }),
    })
    return parseArxivAtom(xml)
  }
}

function optionalText(value: string | undefined): string | undefined {
  if (value === undefined || !value.trim()) return undefined
  return value.trim()
}

function arxivQuote(value: string): string {
  return encodeURIComponent(value)
    .replace(/[!'()*]/g, character => `%${character.codePointAt(0)?.toString(16).toUpperCase()}`)
    .replaceAll('%2F', '/')
}

function parseArxivEntry(entry: string): ArxivPaper {
  const title = normalizedAtomText(atomFirst(entry, 'title') ?? '')
  const rawId = normalizedAtomText(atomFirst(entry, 'id') ?? '')
  const fullId = rawId.includes('/abs/') ? rawId.split('/abs/').at(-1) ?? rawId : rawId
  const versionMatch = /^(.*?)(v\d+)$/.exec(fullId)
  const arxivId = versionMatch?.[1] ?? fullId
  const version = versionMatch?.[2] ?? ''
  const published = normalizedAtomText(atomFirst(entry, 'published') ?? '').slice(0, 10)
  const updated = normalizedAtomText(atomFirst(entry, 'updated') ?? '').slice(0, 10)
  const authors = atomElements(entry, 'author')
    .map(author => normalizedAtomText(atomFirst(author, 'name') ?? ''))
    .filter(Boolean)
  const abstract = normalizedAtomText(atomFirst(entry, 'summary') ?? '')
  const categories = atomAttributes(entry, 'category', 'term')
  return {
    abstract,
    arxivId,
    authors,
    categories,
    pdfUrl: `https://arxiv.org/pdf/${arxivId}`,
    published,
    title,
    updated,
    url: `https://arxiv.org/abs/${arxivId}`,
    version,
  }
}

function atomFirst(xml: string, tag: string): string | undefined {
  return atomElements(xml, tag)[0]
}

function atomElements(xml: string, tag: string): string[] {
  const expression = new RegExp(`<(?:(?:[\\w-]+):)?${tag}\\b[^>]*>([\\s\\S]*?)<\\/(?:(?:[\\w-]+):)?${tag}\\s*>`, 'gi')
  return [...xml.matchAll(expression)].map(match => match[1] ?? '')
}

function atomAttributes(xml: string, tag: string, attribute: string): string[] {
  const expression = new RegExp(`<(?:(?:[\\w-]+):)?${tag}\\b([^>]*)\\/?\\s*>`, 'gi')
  const attributeExpression = new RegExp(`\\b${attribute}\\s*=\\s*(["'])(.*?)\\1`, 'i')
  return [...xml.matchAll(expression)]
    .map(match => attributeExpression.exec(match[1] ?? '')?.[2])
    .filter((value): value is string => value !== undefined)
    .map(decodeXml)
}

function normalizedAtomText(value: string): string {
  return decodeXml(value.replace(/^<!\[CDATA\[([\s\S]*)\]\]>$/, '$1')).replace(/\s+/g, ' ').trim()
}

function decodeXml(value: string): string {
  return value
    .replaceAll('&lt;', '<')
    .replaceAll('&gt;', '>')
    .replaceAll('&quot;', '"')
    .replaceAll('&apos;', "'")
    .replaceAll('&#39;', "'")
    .replaceAll('&amp;', '&')
    .replace(/&#(\d+);/g, (_match, decimal: string) => String.fromCodePoint(Number(decimal)))
    .replace(/&#x([\da-f]+);/gi, (_match, hexadecimal: string) => String.fromCodePoint(Number.parseInt(hexadecimal, 16)))
}
