// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ClientError, ConfigurationError, ValidationError, XerxesTimeoutError } from '../core/errors.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import { checkUrl, type UrlSafetyDecision, type UrlSafetyOptions } from '../security/urlSafety.js'
import type { JsonObject, JsonValue, ToolDefinition } from '../types/toolCalls.js'
import { optionalBoolean, optionalInteger, optionalString, requiredString } from './inputs.js'

const DEFAULT_TIMEOUT_MS = 30_000
const MAX_TIMEOUT_MS = 300_000
const DEFAULT_MAX_REDIRECTS = 5
const MAX_REDIRECTS = 10
const DEFAULT_MAX_RESPONSE_BYTES = 1_000_000
const MAX_RESPONSE_BYTES = 10_000_000
const MAX_SCRAPED_CONTENT_CHARS = 10_000
const MAX_RSS_CONTENT_CHARS = 5_000
const MAX_LINKS = 100
const MAX_IMAGES = 50
const MAX_RSS_ITEMS = 100
const SENSITIVE_RESPONSE_HEADERS = new Set(['set-cookie', 'set-cookie2', 'proxy-authenticate'])
const RESTRICTED_REQUEST_HEADERS = new Set(['connection', 'content-length', 'host', 'transfer-encoding', 'upgrade'])
const REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308])

export type WebFetch = (url: string, init: RequestInit) => Promise<Response>

export interface PublicWebClientOptions {
  readonly fetcher?: WebFetch
  readonly maxRedirects?: number
  readonly maxResponseBytes?: number
  readonly timeoutMs?: number
  readonly urlSafety?: UrlSafetyOptions
}

export interface PublicFetchOptions {
  readonly signal?: AbortSignal
  readonly timeoutMs?: number
}

export interface PublicFetchResponse {
  readonly response: Response
  readonly url: string
}

/** HTTP-only client that blocks literal private URLs and rechecks every redirect target. */
export class PublicWebClient {
  private readonly fetcher: WebFetch
  private readonly maxRedirects: number
  private readonly maxResponseBytes: number
  private readonly timeoutMs: number
  private readonly urlSafety: UrlSafetyOptions

  constructor(options: PublicWebClientOptions = {}) {
    this.fetcher = options.fetcher ?? nativeFetch
    this.maxRedirects = boundedInteger(options.maxRedirects ?? DEFAULT_MAX_REDIRECTS, 'maxRedirects', 0, MAX_REDIRECTS)
    this.maxResponseBytes = boundedInteger(
      options.maxResponseBytes ?? DEFAULT_MAX_RESPONSE_BYTES,
      'maxResponseBytes',
      1,
      MAX_RESPONSE_BYTES,
    )
    this.timeoutMs = boundedInteger(options.timeoutMs ?? DEFAULT_TIMEOUT_MS, 'timeoutMs', 1, MAX_TIMEOUT_MS)
    this.urlSafety = options.urlSafety ?? {}
  }

  async fetch(url: string, init: RequestInit = {}, options: PublicFetchOptions = {}): Promise<PublicFetchResponse> {
    let currentUrl = assertPublicHttpUrl(url, this.urlSafety)
    const { signal: initSignal, ...requestBase } = init
    const signal = options.signal ?? initSignal ?? undefined
    const timeoutMs = boundedInteger(options.timeoutMs ?? this.timeoutMs, 'timeoutMs', 1, MAX_TIMEOUT_MS)
    const method = (requestBase.method ?? 'GET').toUpperCase()
    let redirectCount = 0

    while (true) {
      const requestInit: RequestInit = { ...requestBase, redirect: 'manual' }
      const response = await runWithTimeout(
        requestSignal => this.fetcher(currentUrl, withSignal(requestInit, requestSignal)),
        timeoutMs,
        signal,
      )
      if (!REDIRECT_STATUSES.has(response.status) || (method !== 'GET' && method !== 'HEAD')) {
        return { response, url: currentUrl }
      }

      const location = response.headers.get('location')
      if (!location) {
        return { response, url: currentUrl }
      }
      if (redirectCount >= this.maxRedirects) {
        throw new ClientError('web', `redirect limit of ${this.maxRedirects} exceeded`)
      }
      currentUrl = assertPublicHttpUrl(new URL(location, currentUrl).toString(), this.urlSafety)
      redirectCount += 1
    }
  }

  async text(response: Response): Promise<string> {
    return boundedResponseText(response, this.maxResponseBytes)
  }

  urlSafetyDecision(url: string): UrlSafetyDecision {
    return checkUrl(url, this.urlSafety)
  }
}

export interface ApiRequest {
  readonly data?: string
  readonly headers?: Readonly<Record<string, string>>
  readonly jsonData?: JsonObject
  readonly method?: string
  readonly params?: Readonly<Record<string, JsonValue>>
  readonly timeoutMs?: number
  readonly url: string
}

export interface ApiResponse {
  readonly headers: Readonly<Record<string, string>>
  readonly json?: JsonValue
  readonly statusCode: number
  readonly text?: string
  readonly url: string
}

/** Make a bounded public HTTP API call without exposing sensitive response headers. */
export async function apiRequest(
  request: ApiRequest,
  client: PublicWebClient = new PublicWebClient(),
  signal?: AbortSignal,
): Promise<ApiResponse> {
  const method = validMethod(request.method ?? 'GET')
  const headers = requestHeaders(request.headers)
  if (request.data !== undefined && request.jsonData !== undefined) {
    throw new ValidationError('data', 'cannot be combined with json_data')
  }

  const url = addQueryParameters(request.url, request.params)
  const init: RequestInit = { headers, method }
  if (request.jsonData !== undefined) {
    if (!hasHeader(headers, 'content-type')) headers['content-type'] = 'application/json'
    init.body = JSON.stringify(request.jsonData)
  } else if (request.data !== undefined) {
    init.body = request.data
  }

  const fetched = await client.fetch(url, init, fetchOptions(signal, request.timeoutMs))
  const text = await client.text(fetched.response)
  const result: {
    headers: Readonly<Record<string, string>>
    json?: JsonValue
    statusCode: number
    text?: string
    url: string
  } = {
    headers: safeResponseHeaders(fetched.response.headers),
    statusCode: fetched.response.status,
    url: fetched.url,
  }
  try {
    result.json = JSON.parse(text) as JsonValue
  } catch {
    result.text = text.slice(0, MAX_SCRAPED_CONTENT_CHARS)
  }
  return result
}

export interface WebScrapeRequest {
  readonly cleanText?: boolean
  readonly extractImages?: boolean
  readonly extractLinks?: boolean
  readonly selector?: string
  readonly timeoutMs?: number
  readonly url: string
}

export interface ScrapedLink {
  readonly text: string
  readonly url: string
}

export interface ScrapedImage {
  readonly alt: string
  readonly src: string
}

export interface ScrapedWebPage {
  readonly content?: string
  readonly images?: readonly ScrapedImage[]
  readonly links?: readonly ScrapedLink[]
  readonly meta: Readonly<Record<string, string>>
  readonly selectedContent?: readonly string[]
  readonly statusCode: number
  readonly title: string | null
  readonly url: string
}

/** Fetch static HTML and extract bounded text, document metadata, and optional public links/images. */
export async function scrapeWebPage(
  request: WebScrapeRequest,
  client: PublicWebClient = new PublicWebClient(),
  signal?: AbortSignal,
): Promise<ScrapedWebPage> {
  const fetched = await client.fetch(request.url, { method: 'GET' }, fetchOptions(signal, request.timeoutMs))
  if (!fetched.response.ok) {
    throw new ClientError('web', `GET ${fetched.url} returned HTTP ${fetched.response.status}`)
  }
  const html = await client.text(fetched.response)
  const extracted = await extractHtml(html, fetched.url, request.selector)
  const result: {
    content?: string
    images?: readonly ScrapedImage[]
    links?: readonly ScrapedLink[]
    meta: Readonly<Record<string, string>>
    selectedContent?: readonly string[]
    statusCode: number
    title: string | null
    url: string
  } = {
    meta: extracted.meta,
    statusCode: fetched.response.status,
    title: extracted.title,
    url: fetched.url,
  }
  if (request.selector !== undefined) {
    result.selectedContent = extracted.selectedContent
  } else {
    result.content = (request.cleanText ?? true)
      ? extracted.text.slice(0, MAX_SCRAPED_CONTENT_CHARS)
      : html.slice(0, MAX_SCRAPED_CONTENT_CHARS)
  }
  if (request.extractLinks ?? false) result.links = extracted.links.slice(0, MAX_LINKS)
  if (request.extractImages ?? false) result.images = extracted.images.slice(0, MAX_IMAGES)
  return result
}

export interface RssFeedItem {
  readonly author: string
  readonly content?: string
  readonly link: string
  readonly published: string
  readonly tags: readonly string[]
  readonly title: string
}

export interface RssFeed {
  readonly description: string
  readonly items: readonly RssFeedItem[]
  readonly link: string
  readonly title: string
  readonly updated: string
}

export interface RssRequest {
  readonly feedUrl: string
  readonly includeContent?: boolean
  readonly maxItems?: number
  readonly timeoutMs?: number
}

/** Read a public RSS 2.0 or Atom feed using a dependency-free, non-DTD XML subset parser. */
export async function readRssFeed(
  request: RssRequest,
  client: PublicWebClient = new PublicWebClient(),
  signal?: AbortSignal,
): Promise<RssFeed> {
  const maxItems = boundedInteger(request.maxItems ?? 10, 'max_items', 1, MAX_RSS_ITEMS)
  const fetched = await client.fetch(request.feedUrl, { method: 'GET' }, fetchOptions(signal, request.timeoutMs))
  if (!fetched.response.ok) {
    throw new ClientError('rss', `GET ${fetched.url} returned HTTP ${fetched.response.status}`)
  }
  const xml = await client.text(fetched.response)
  return parseRssFeed(xml, request.includeContent ?? true, maxItems)
}

export interface UrlAnalysis {
  readonly availabilityError?: string
  readonly description?: string
  readonly domain: string
  readonly domainName?: string
  readonly finalUrl?: string
  readonly fragment: string
  readonly isAvailable?: boolean
  readonly isFetchable: boolean
  readonly isValid: boolean
  readonly openGraph?: Readonly<Record<string, string>>
  readonly params: string
  readonly path: string
  readonly query: string
  readonly safetyReason?: string
  readonly scheme: string
  readonly statusCode?: number
  readonly subdomain?: string
  readonly title?: string | null
  readonly tld?: string
  readonly url: string
}

export interface UrlAnalysisRequest {
  readonly checkAvailability?: boolean
  readonly extractMetadata?: boolean
  readonly timeoutMs?: number
  readonly url: string
}

/** Parse a URL locally and only perform public HEAD/GET checks when explicitly requested. */
export async function analyzeUrl(
  request: UrlAnalysisRequest,
  client: PublicWebClient = new PublicWebClient(),
  signal?: AbortSignal,
): Promise<UrlAnalysis> {
  const local = localUrlAnalysis(request.url, client)
  if (!(request.checkAvailability ?? false) || !local.isValid || !local.isFetchable) return local

  try {
    let fetched = await client.fetch(request.url, { method: 'HEAD' }, fetchOptions(signal, request.timeoutMs))
    if (fetched.response.status === 405 || fetched.response.status === 501) {
      fetched = await client.fetch(request.url, { method: 'GET' }, fetchOptions(signal, request.timeoutMs))
    }
    const available = fetched.response.status < 400
    const result: {
      availabilityError?: string
      description?: string
      domain: string
      domainName?: string
      finalUrl?: string
      fragment: string
      isAvailable?: boolean
      isFetchable: boolean
      isValid: boolean
      openGraph?: Readonly<Record<string, string>>
      params: string
      path: string
      query: string
      safetyReason?: string
      scheme: string
      statusCode?: number
      subdomain?: string
      title?: string | null
      tld?: string
      url: string
    } = {
      ...local,
      finalUrl: fetched.url,
      isAvailable: available,
      statusCode: fetched.response.status,
    }
    if ((request.extractMetadata ?? true) && available) {
      const page = await scrapeWebPage({
        url: request.url,
        ...(request.timeoutMs === undefined ? {} : { timeoutMs: request.timeoutMs }),
      }, client, signal)
      result.title = page.title
      if (page.meta.description !== undefined) result.description = page.meta.description
      const openGraph = Object.fromEntries(Object.entries(page.meta).filter(([key]) => key.startsWith('og:')))
      if (Object.keys(openGraph).length) result.openGraph = openGraph
    }
    return result
  } catch (error) {
    return { ...local, availabilityError: errorMessage(error), isAvailable: false }
  }
}

export const WEB_SCRAPER_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'WebScraper',
    description: 'Fetch a public static HTML page and extract bounded text, metadata, links, or images.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        url: { type: 'string' },
        selector: { type: 'string', description: 'Optional Bun HTMLRewriter CSS selector.' },
        extract_links: { type: 'boolean', default: false },
        extract_images: { type: 'boolean', default: false },
        clean_text: { type: 'boolean', default: true },
        timeout: { type: 'integer', default: 30, minimum: 1, maximum: 300 },
      },
      required: ['url'],
    },
  },
}

export const API_CLIENT_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'APIClient',
    description: 'Make a public HTTP API request with bounded responses and no private-network access.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        url: { type: 'string' },
        method: { type: 'string', enum: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'], default: 'GET' },
        headers: { type: 'object', additionalProperties: { type: 'string' } },
        params: { type: 'object', additionalProperties: true },
        json_data: { type: 'object', additionalProperties: true },
        data: { type: 'string' },
        timeout: { type: 'integer', default: 30, minimum: 1, maximum: 300 },
      },
      required: ['url'],
    },
  },
}

export const RSS_READER_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'RSSReader',
    description: 'Read a public RSS 2.0 or Atom feed without executing feed content.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        feed_url: { type: 'string' },
        max_items: { type: 'integer', default: 10, minimum: 1, maximum: MAX_RSS_ITEMS },
        include_content: { type: 'boolean', default: true },
        timeout: { type: 'integer', default: 30, minimum: 1, maximum: 300 },
      },
      required: ['feed_url'],
    },
  },
}

export const URL_ANALYZER_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'URLAnalyzer',
    description: 'Parse a URL and, when requested, check only its public HTTP availability and metadata.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        url: { type: 'string' },
        check_availability: { type: 'boolean', default: false },
        extract_metadata: { type: 'boolean', default: true },
        timeout: { type: 'integer', default: 30, minimum: 1, maximum: 300 },
      },
      required: ['url'],
    },
  },
}

export const WEB_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  WEB_SCRAPER_DEFINITION,
  API_CLIENT_DEFINITION,
  RSS_READER_DEFINITION,
  URL_ANALYZER_DEFINITION,
]

/** Register safe HTTP-only web tools. Search providers are registered separately and require explicit configuration. */
export function registerWebTools(registry: ToolRegistry, client: PublicWebClient = new PublicWebClient(), agentId = 'default'): void {
  registry.register(WEB_SCRAPER_DEFINITION, (inputs, _context, signal) => scrapeWebPage(webScrapeRequestFromInputs(inputs), client, signal), agentId)
  registry.register(API_CLIENT_DEFINITION, (inputs, _context, signal) => apiRequest(apiRequestFromInputs(inputs), client, signal), agentId)
  registry.register(RSS_READER_DEFINITION, (inputs, _context, signal) => readRssFeed({
    feedUrl: requiredString(inputs, 'feed_url'),
    includeContent: optionalBoolean(inputs, 'include_content', true),
    maxItems: optionalInteger(inputs, 'max_items', 10),
    timeoutMs: timeoutMilliseconds(inputs),
  }, client, signal), agentId)
  registry.register(URL_ANALYZER_DEFINITION, (inputs, _context, signal) => analyzeUrl({
    checkAvailability: optionalBoolean(inputs, 'check_availability', false),
    extractMetadata: optionalBoolean(inputs, 'extract_metadata', true),
    timeoutMs: timeoutMilliseconds(inputs),
    url: requiredString(inputs, 'url'),
  }, client, signal), agentId)
}

function assertPublicHttpUrl(value: string, safetyOptions: UrlSafetyOptions): string {
  let parsed: URL
  try {
    parsed = new URL(value)
  } catch {
    throw new ValidationError('url', 'must be an absolute HTTP or HTTPS URL', value)
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new ValidationError('url', 'must use HTTP or HTTPS', value)
  }
  const safety = checkUrl(parsed.toString(), safetyOptions)
  if (!safety.allowed) {
    throw new ValidationError('url', `is not allowed: ${safety.reason}`, value)
  }
  return parsed.toString()
}

function localUrlAnalysis(url: string, client: PublicWebClient): UrlAnalysis {
  let parsed: URL
  try {
    parsed = new URL(url)
  } catch {
    return {
      domain: '',
      fragment: '',
      isFetchable: false,
      isValid: false,
      params: '',
      path: '',
      query: '',
      scheme: '',
      url,
    }
  }
  const isHttp = parsed.protocol === 'http:' || parsed.protocol === 'https:'
  const safety = isHttp ? client.urlSafetyDecision(parsed.toString()) : undefined
  const hostname = parsed.hostname.toLowerCase()
  const hostParts = hostname.split('.').filter(Boolean)
  const pathParts = parsed.pathname.split(';')
  const path = pathParts.shift() ?? ''
  const params = pathParts.join(';')
  const result: {
    domain: string
    domainName?: string
    fragment: string
    isFetchable: boolean
    isValid: boolean
    params: string
    path: string
    query: string
    safetyReason?: string
    scheme: string
    subdomain?: string
    tld?: string
    url: string
  } = {
    domain: parsed.host,
    fragment: parsed.hash.slice(1),
    isFetchable: isHttp && (safety?.allowed ?? false),
    isValid: Boolean(parsed.protocol && parsed.host),
    params,
    path,
    query: parsed.search.slice(1),
    scheme: parsed.protocol.slice(0, -1),
    url,
  }
  if (safety !== undefined && !safety.allowed) result.safetyReason = safety.reason
  if (hostParts.length >= 2) {
    const last = hostParts[hostParts.length - 1]
    const secondLast = hostParts[hostParts.length - 2]
    if (last !== undefined && secondLast !== undefined) {
      result.tld = last
      result.domainName = `${secondLast}.${last}`
      if (hostParts.length > 2) result.subdomain = hostParts.slice(0, -2).join('.')
    }
  }
  return result
}

async function extractHtml(html: string, baseUrl: string, selector: string | undefined): Promise<{
  readonly images: readonly ScrapedImage[]
  readonly links: readonly ScrapedLink[]
  readonly meta: Readonly<Record<string, string>>
  readonly selectedContent: readonly string[]
  readonly text: string
  readonly title: string | null
}> {
  const titleParts: string[] = []
  const meta = Object.create(null) as Record<string, string>
  const links: ScrapedLink[] = []
  const images: ScrapedImage[] = []
  const linkStack: Array<{ text: string; url: string }> = []
  const selectedStack: Array<{ text: string }> = []
  const selectedContent: string[] = []
  const rewriter = new HTMLRewriter()
  rewriter.on('title', {
    text(chunk) {
      titleParts.push(chunk.text)
    },
  })
  rewriter.on('meta', {
    element(element) {
      const key = element.getAttribute('name') ?? element.getAttribute('property')
      const content = element.getAttribute('content')
      if (key && content !== null && !isPrototypeKey(key)) meta[key] = content
    },
  })
  rewriter.on('a[href]', {
    element(element) {
      const href = element.getAttribute('href')
      const url = href === null ? undefined : resolveHttpUrl(baseUrl, href)
      if (url === undefined) return
      const link = { text: '', url }
      linkStack.push(link)
      element.onEndTag(() => {
        const completed = linkStack.pop()
        if (completed !== undefined) links.push({ text: normalizeText(completed.text), url: completed.url })
      })
    },
    text(chunk) {
      const current = linkStack[linkStack.length - 1]
      if (current !== undefined) current.text += chunk.text
    },
  })
  rewriter.on('img[src]', {
    element(element) {
      const src = element.getAttribute('src')
      const resolved = src === null ? undefined : resolveHttpUrl(baseUrl, src)
      if (resolved !== undefined) images.push({ alt: element.getAttribute('alt') ?? '', src: resolved })
    },
  })
  if (selector !== undefined) {
    try {
      rewriter.on(selector, {
        element(element) {
          const entry = { text: '' }
          selectedStack.push(entry)
          element.onEndTag(() => {
            const completed = selectedStack.pop()
            if (completed !== undefined) selectedContent.push(normalizeText(completed.text))
          })
        },
        text(chunk) {
          const current = selectedStack[selectedStack.length - 1]
          if (current !== undefined) current.text += chunk.text
        },
      })
    } catch (error) {
      throw new ValidationError('selector', `is not supported: ${errorMessage(error)}`, selector)
    }
  }
  try {
    // HTMLRewriter handlers execute as the transformed body is consumed.
    await rewriter.transform(new Response(html)).text()
  } catch (error) {
    if (selector !== undefined) throw new ValidationError('selector', `is not supported: ${errorMessage(error)}`, selector)
    throw new ClientError('html', errorMessage(error), error)
  }
  return {
    images,
    links,
    meta,
    selectedContent: selectedContent.filter(Boolean),
    text: normalizeText(stripHtml(html)),
    title: titleParts.length ? normalizeText(titleParts.join('')) : null,
  }
}

function parseRssFeed(xml: string, includeContent: boolean, maxItems: number): RssFeed {
  const isAtom = /<feed(?:\s|>)/i.test(xml) && !/<rss(?:\s|>)/i.test(xml)
  const feedBody = firstTag(xml, isAtom ? 'feed' : 'channel') ?? xml
  const itemBodies = allTags(xml, isAtom ? 'entry' : 'item').slice(0, maxItems)
  const items = itemBodies.map(body => rssItem(body, isAtom, includeContent))
  const feedMetadata = feedBody.replace(/<(item|entry)(?:\s[^>]*)?>[\s\S]*?<\/\1\s*>/gi, '')
  return {
    description: textFromTag(feedMetadata, isAtom ? 'subtitle' : 'description'),
    items,
    link: linkFromXml(feedMetadata, isAtom),
    title: textFromTag(feedMetadata, 'title'),
    updated: textFromTag(feedMetadata, isAtom ? 'updated' : 'lastBuildDate'),
  }
}

function rssItem(body: string, isAtom: boolean, includeContent: boolean): RssFeedItem {
  const item: {
    author: string
    content?: string
    link: string
    published: string
    tags: readonly string[]
    title: string
  } = {
    author: textFromTag(body, isAtom ? 'author' : 'author'),
    link: linkFromXml(body, isAtom),
    published: textFromTag(body, isAtom ? 'published' : 'pubDate') || textFromTag(body, isAtom ? 'updated' : 'date'),
    tags: tagsFromXml(body),
    title: textFromTag(body, 'title'),
  }
  if (includeContent) {
    item.content = (textFromTag(body, 'encoded') || textFromTag(body, 'content') || textFromTag(body, 'summary')
      || textFromTag(body, 'description')).slice(0, MAX_RSS_CONTENT_CHARS)
  }
  return item
}

function allTags(xml: string, name: string): string[] {
  const expression = new RegExp(`<${name}(?:\\s[^>]*)?>([\\s\\S]*?)<\\/${name}\\s*>`, 'gi')
  const bodies: string[] = []
  for (const match of xml.matchAll(expression)) {
    const body = match[1]
    if (body !== undefined) bodies.push(body)
  }
  return bodies
}

function firstTag(xml: string, name: string): string | undefined {
  return allTags(xml, name)[0]
}

function textFromTag(xml: string, name: string): string {
  const qualifiedName = `(?:[\\w.-]+:)?${escapeRegex(name)}`
  const expression = new RegExp(`<${qualifiedName}(?:\\s[^>]*)?>([\\s\\S]*?)<\\/${qualifiedName}\\s*>`, 'i')
  const value = expression.exec(xml)?.[1]
  return value === undefined ? '' : normalizeText(stripXml(value))
}

function linkFromXml(xml: string, isAtom: boolean): string {
  if (isAtom) {
    const alternate = /<link\b[^>]*\bhref=["']([^"']+)["'][^>]*>/i.exec(xml)?.[1]
    return alternate === undefined ? '' : decodeEntities(alternate)
  }
  return textFromTag(xml, 'link')
}

function tagsFromXml(xml: string): readonly string[] {
  const values: string[] = []
  const expression = /<(?:[\w.-]+:)?category\b([^>]*)>([\s\S]*?)<\/(?:[\w.-]+:)?category\s*>|<(?:[\w.-]+:)?category\b([^>]*)\/>/gi
  for (const match of xml.matchAll(expression)) {
    const attributes = match[1] ?? match[3] ?? ''
    const term = /\bterm=["']([^"']+)["']/i.exec(attributes)?.[1]
    const text = term === undefined ? normalizeText(stripXml(match[2] ?? '')) : decodeEntities(term)
    if (text) values.push(text)
  }
  return values
}

function stripHtml(value: string): string {
  return value
    .replace(/<!--[\s\S]*?-->/g, ' ')
    .replace(/<(script|style|noscript|template)\b[^>]*>[\s\S]*?<\/\1\s*>/gi, ' ')
    .replace(/<(br|hr)\b[^>]*>/gi, '\n')
    .replace(/<\/(address|article|blockquote|div|h[1-6]|li|p|section|table|tr)\s*>/gi, '\n')
    .replace(/<[^>]+>/g, ' ')
}

function stripXml(value: string): string {
  return value
    .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, '$1')
    .replace(/<[^>]+>/g, ' ')
}

function normalizeText(value: string): string {
  return decodeEntities(value).replace(/\s+/g, ' ').trim()
}

function decodeEntities(value: string): string {
  return value
    .replace(/&#x([\da-f]+);/gi, (match, code: string) => decodedCodePoint(match, code, 16))
    .replace(/&#(\d+);/g, (match, code: string) => decodedCodePoint(match, code, 10))
    .replace(/&(amp|apos|gt|lt|nbsp|quot);/gi, (_match, name: string) => {
      const entities: Record<string, string> = { amp: '&', apos: "'", gt: '>', lt: '<', nbsp: ' ', quot: '"' }
      return entities[name.toLowerCase()] ?? _match
    })
}

function decodedCodePoint(fallback: string, value: string, radix: number): string {
  const codePoint = Number.parseInt(value, radix)
  if (!Number.isSafeInteger(codePoint) || codePoint < 0 || codePoint > 0x10ffff) return fallback
  try {
    return String.fromCodePoint(codePoint)
  } catch {
    return fallback
  }
}

function resolveHttpUrl(baseUrl: string, value: string): string | undefined {
  try {
    const parsed = new URL(value, baseUrl)
    return parsed.protocol === 'http:' || parsed.protocol === 'https:' ? parsed.toString() : undefined
  } catch {
    return undefined
  }
}

function addQueryParameters(url: string, params: Readonly<Record<string, JsonValue>> | undefined): string {
  if (params === undefined) return url
  let parsed: URL
  try {
    parsed = new URL(url)
  } catch {
    throw new ValidationError('url', 'must be an absolute HTTP or HTTPS URL', url)
  }
  for (const [key, value] of Object.entries(params)) {
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
      parsed.searchParams.set(key, String(value))
    } else if (value === null) {
      parsed.searchParams.delete(key)
    } else {
      throw new ValidationError('params', 'values must be strings, numbers, booleans, or null', value)
    }
  }
  return parsed.toString()
}

function requestHeaders(headers: Readonly<Record<string, string>> | undefined): Record<string, string> {
  const result = Object.create(null) as Record<string, string>
  for (const [name, value] of Object.entries(headers ?? {})) {
    const normalized = name.trim().toLowerCase()
    if (!normalized || /[\r\n]/.test(name) || /[\r\n]/.test(value)) {
      throw new ValidationError('headers', 'must not contain empty names or line breaks', name)
    }
    if (RESTRICTED_REQUEST_HEADERS.has(normalized)) {
      throw new ValidationError('headers', `cannot set restricted header ${name}`, name)
    }
    result[name] = value
  }
  return result
}

function safeResponseHeaders(headers: Headers): Record<string, string> {
  const result = Object.create(null) as Record<string, string>
  for (const [name, value] of headers) {
    result[name] = SENSITIVE_RESPONSE_HEADERS.has(name.toLowerCase()) ? '[REDACTED]' : value
  }
  return result
}

function validMethod(value: string): string {
  const method = value.toUpperCase()
  if (!['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS'].includes(method)) {
    throw new ValidationError('method', 'must be GET, POST, PUT, DELETE, PATCH, HEAD, or OPTIONS', value)
  }
  return method
}

function withSignal(init: RequestInit, signal: AbortSignal | undefined): RequestInit {
  if (signal === undefined) return init
  return { ...init, signal }
}

async function runWithTimeout<T>(
  operation: (signal: AbortSignal) => Promise<T>,
  timeoutMs: number,
  signal: AbortSignal | undefined,
): Promise<T> {
  if (signal?.aborted) throw new ClientError('web', 'request cancelled before execution')
  const controller = new AbortController()
  const abort = () => controller.abort(signal?.reason)
  signal?.addEventListener('abort', abort, { once: true })
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    return await Promise.race([
      operation(controller.signal),
      new Promise<T>((_resolve, reject) => {
        controller.signal.addEventListener('abort', () => reject(new Error('request aborted')), { once: true })
      }),
    ])
  } catch (error) {
    if (controller.signal.aborted && !signal?.aborted) {
      throw new XerxesTimeoutError('web request', Math.ceil(timeoutMs / 1000))
    }
    throw error
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', abort)
  }
}

async function boundedResponseText(response: Response, maxBytes: number): Promise<string> {
  const declaredLength = response.headers.get('content-length')
  if (declaredLength !== null) {
    const length = Number.parseInt(declaredLength, 10)
    if (Number.isFinite(length) && length > maxBytes) {
      throw new ClientError('web', `response exceeds ${maxBytes} byte limit`)
    }
  }
  if (response.body === null) return ''
  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let length = 0
  try {
    while (true) {
      const chunk = await reader.read()
      if (chunk.done) break
      length += chunk.value.byteLength
      if (length > maxBytes) {
        await reader.cancel()
        throw new ClientError('web', `response exceeds ${maxBytes} byte limit`)
      }
      chunks.push(chunk.value)
    }
  } finally {
    reader.releaseLock()
  }
  const combined = new Uint8Array(length)
  let offset = 0
  for (const chunk of chunks) {
    combined.set(chunk, offset)
    offset += chunk.byteLength
  }
  return new TextDecoder().decode(combined)
}

function optionalStringRecord(inputs: JsonObject, name: string): Readonly<Record<string, string>> | undefined {
  const value = inputs[name]
  if (value === undefined) return undefined
  if (!isObject(value)) throw new ValidationError(name, 'must be an object of strings', value)
  const result: Record<string, string> = {}
  for (const [key, entry] of Object.entries(value)) {
    if (typeof entry !== 'string') throw new ValidationError(name, 'must be an object of strings', value)
    result[key] = entry
  }
  return result
}

function optionalJsonObject(inputs: JsonObject, name: string): JsonObject | undefined {
  const value = inputs[name]
  if (value === undefined) return undefined
  if (!isObject(value)) throw new ValidationError(name, 'must be a JSON object', value)
  return value
}

function webScrapeRequestFromInputs(inputs: JsonObject): WebScrapeRequest {
  const request: {
    cleanText: boolean
    extractImages: boolean
    extractLinks: boolean
    selector?: string
    timeoutMs: number
    url: string
  } = {
    cleanText: optionalBoolean(inputs, 'clean_text', true),
    extractImages: optionalBoolean(inputs, 'extract_images', false),
    extractLinks: optionalBoolean(inputs, 'extract_links', false),
    timeoutMs: timeoutMilliseconds(inputs),
    url: requiredString(inputs, 'url'),
  }
  const selector = optionalString(inputs, 'selector')
  if (selector !== undefined) request.selector = selector
  return request
}

function apiRequestFromInputs(inputs: JsonObject): ApiRequest {
  const request: {
    data?: string
    headers?: Readonly<Record<string, string>>
    jsonData?: JsonObject
    method?: string
    params?: Readonly<Record<string, JsonValue>>
    timeoutMs: number
    url: string
  } = {
    timeoutMs: timeoutMilliseconds(inputs),
    url: requiredString(inputs, 'url'),
  }
  const data = optionalString(inputs, 'data')
  const headers = optionalStringRecord(inputs, 'headers')
  const jsonData = optionalJsonObject(inputs, 'json_data')
  const method = optionalString(inputs, 'method')
  const params = optionalJsonObject(inputs, 'params')
  if (data !== undefined) request.data = data
  if (headers !== undefined) request.headers = headers
  if (jsonData !== undefined) request.jsonData = jsonData
  if (method !== undefined) request.method = method
  if (params !== undefined) request.params = params
  return request
}

function isObject(value: JsonValue): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function timeoutMilliseconds(inputs: JsonObject): number {
  const seconds = optionalInteger(inputs, 'timeout', 30)
  return boundedInteger(seconds, 'timeout', 1, Math.floor(MAX_TIMEOUT_MS / 1000)) * 1000
}

function fetchOptions(signal: AbortSignal | undefined, timeoutMs: number | undefined): PublicFetchOptions {
  const options: { signal?: AbortSignal; timeoutMs?: number } = {}
  if (signal !== undefined) options.signal = signal
  if (timeoutMs !== undefined) options.timeoutMs = timeoutMs
  return options
}

function boundedInteger(value: number, name: string, minimum: number, maximum: number): number {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new ConfigurationError(name, `must be an integer between ${minimum} and ${maximum}`)
  }
  return value
}

function hasHeader(headers: Readonly<Record<string, string>>, expected: string): boolean {
  return Object.keys(headers).some(name => name.toLowerCase() === expected)
}

function isPrototypeKey(value: string): boolean {
  return value === '__proto__' || value === 'constructor' || value === 'prototype'
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

async function nativeFetch(url: string, init: RequestInit): Promise<Response> {
  return fetch(url, init)
}
