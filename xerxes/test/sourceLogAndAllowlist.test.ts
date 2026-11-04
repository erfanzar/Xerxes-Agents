// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, beforeEach, expect, test } from 'bun:test'

import { ValidationError } from '../src/core/errors.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import {
  DuckDuckGoSearch,
  registerDuckDuckGoSearchTool,
  type DuckDuckGoSearchProvider,
} from '../src/tools/duckduckgoEngine.js'
import {
  DEFAULT_SOURCE_SESSION,
  MAX_SOURCES_PER_SESSION,
  SourceLog,
  renderSources,
  setSourceLog,
} from '../src/tools/sourceLog.js'
import {
  PublicWebClient,
  analyzeUrl,
  parseAllowedDomains,
  registerWebTools,
  scrapeWebPage,
  type WebFetch,
} from '../src/tools/webTools.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

let log: SourceLog
let previous: SourceLog

beforeEach(() => {
  log = new SourceLog()
  previous = setSourceLog(log)
})

afterEach(() => {
  setSourceLog(previous)
})

function client(fetcher: WebFetch): PublicWebClient {
  return new PublicWebClient({ fetcher, urlSafety: { dnsLookup: async () => ['93.184.216.34'] } })
}

function call(name: string, args: JsonObject): ToolCall {
  return { id: crypto.randomUUID(), type: 'function', function: { name, arguments: args } }
}

const PAGE = '<html><head><title>Hello Page</title></head><body><p>text</p></body></html>'

test('SourceLog dedupes by url, fills a missing title, and keeps sessions apart', () => {
  const sources = new SourceLog()
  sources.record({ sessionId: 's1', tool: 'DuckDuckGoSearch', url: 'https://example.com/a#frag' })
  sources.record({ sessionId: 's1', tool: 'WebScraper', title: 'Page A', url: 'https://example.com/a' })
  sources.record({ sessionId: 's1', tool: 'WebScraper', title: 'Other title', url: 'https://example.com/a' })
  sources.record({ sessionId: 's2', tool: 'WebScraper', url: 'https://example.com/a' })

  const first = sources.list('s1')
  expect(first).toHaveLength(1)
  expect(first[0]?.tool).toBe('DuckDuckGoSearch')
  expect(first[0]?.title).toBe('Page A')
  expect(sources.list('s2')).toHaveLength(1)
  expect(sources.list('missing')).toEqual([])
})

test('SourceLog ignores non-http urls and files them under the default session without an id', () => {
  const sources = new SourceLog()
  expect(sources.record({ tool: 'x', url: 'file:///etc/passwd' })).toBeUndefined()
  expect(sources.record({ tool: 'x', url: 'not a url' })).toBeUndefined()
  sources.record({ tool: 'x', url: 'https://example.com/' })
  expect(sources.list(DEFAULT_SOURCE_SESSION)).toHaveLength(1)
})

test('SourceLog keeps only the newest 500 sources per session', () => {
  const sources = new SourceLog()
  for (let index = 0; index < MAX_SOURCES_PER_SESSION + 25; index += 1) {
    sources.record({ sessionId: 's', tool: 'x', url: `https://example.com/${index}` })
  }
  const kept = sources.list('s')
  expect(kept).toHaveLength(MAX_SOURCES_PER_SESSION)
  expect(kept[0]?.url).toBe('https://example.com/25')
  expect(kept.at(-1)?.url).toBe(`https://example.com/${MAX_SOURCES_PER_SESSION + 24}`)
})

test('renderSources produces a markdown list and a clear empty message', () => {
  expect(renderSources([])).toContain('No sources')
  const sources = new SourceLog()
  sources.record({ at: '2026-01-01T00:00:00.000Z', sessionId: 's', tool: 'WebScraper', title: 'A [x]', url: 'https://example.com/a' })
  const text = renderSources(sources.list('s'))
  expect(text).toContain('1. [A x](https://example.com/a) — WebScraper, 2026-01-01T00:00:00.000Z')
})

test('WebScraper records the fetched URL and title in the session source log', async () => {
  await scrapeWebPage({ sessionId: 'sess', url: 'https://docs.example.com/page' }, client(async () => new Response(PAGE, { status: 200 })))
  expect(log.list('sess')).toEqual([
    expect.objectContaining({ title: 'Hello Page', tool: 'WebScraper', url: 'https://docs.example.com/page' }),
  ])
})

test('parseAllowedDomains normalizes a comma list and treats unset or blank as unrestricted', () => {
  expect(parseAllowedDomains(undefined)).toEqual([])
  expect(parseAllowedDomains('  ')).toEqual([])
  expect(parseAllowedDomains('Example.com, https://docs.rust-lang.org/book, *.arxiv.org, example.com')).toEqual([
    'example.com',
    'docs.rust-lang.org',
    'arxiv.org',
  ])
})

test('the allowlist admits the domain and its subdomains and refuses everything else before fetching', async () => {
  let fetches = 0
  const web = client(async () => {
    fetches += 1
    return new Response(PAGE, { status: 200 })
  })
  const allowedDomains = ['example.com']
  await scrapeWebPage({ allowedDomains, url: 'https://example.com/' }, web)
  await scrapeWebPage({ allowedDomains, url: 'https://docs.example.com/x' }, web)
  expect(fetches).toBe(2)

  for (const url of ['https://evil.com/', 'https://notexample.com/', 'https://example.com.evil.com/']) {
    await expect(scrapeWebPage({ allowedDomains, url }, web)).rejects.toBeInstanceOf(ValidationError)
  }
  await expect(scrapeWebPage({ allowedDomains, url: 'https://evil.com/' }, web)).rejects.toThrow(/allowlist/)
  expect(fetches).toBe(2)
})

test('the allowlist also constrains redirect hops', async () => {
  const web = client(async url => url.includes('example.com')
    ? new Response(null, { headers: { location: 'https://elsewhere.net/' }, status: 302 })
    : new Response(PAGE, { status: 200 }))
  await expect(scrapeWebPage({ allowedDomains: ['example.com'], url: 'https://example.com/go' }, web))
    .rejects.toThrow(/allowlist/)
})

test('the allowlist does not weaken the private-network checks', async () => {
  const web = client(async () => new Response(PAGE, { status: 200 }))
  await expect(scrapeWebPage({ allowedDomains: ['127.0.0.1'], url: 'http://127.0.0.1/' }, web))
    .rejects.toBeInstanceOf(ValidationError)
})

test('URLAnalyzer reports a disallowed domain as not fetchable and never requests it', async () => {
  let fetches = 0
  const web = client(async () => {
    fetches += 1
    return new Response(PAGE, { status: 200 })
  })
  const denied = await analyzeUrl({ allowedDomains: ['example.com'], checkAvailability: true, url: 'https://evil.com/' }, web)
  expect(denied.isFetchable).toBe(false)
  expect(denied.safetyReason).toContain('allowlist')
  expect(fetches).toBe(0)

  const allowed = await analyzeUrl({
    allowedDomains: ['example.com'], checkAvailability: true, sessionId: 'u', url: 'https://www.example.com/',
  }, web)
  expect(allowed.isAvailable).toBe(true)
  expect(log.list('u').map(entry => entry.url)).toEqual(['https://www.example.com/'])
})

test('with no allowlist, WebScraper registered as a tool behaves as before and records under the session', async () => {
  const saved = process.env.XERXES_RESEARCH_ALLOWED_DOMAINS
  delete process.env.XERXES_RESEARCH_ALLOWED_DOMAINS
  try {
    const registry = new ToolRegistry()
    registerWebTools(registry, client(async () => new Response(PAGE, { status: 200 })))
    const result = await registry.execute(call('WebScraper', { url: 'https://anything.example.org/p' }), {
      metadata: {},
      sessionId: 'tool-session',
    })
    expect(result).toContain('Hello Page')
    expect(log.list('tool-session').map(entry => entry.url)).toEqual(['https://anything.example.org/p'])
  } finally {
    if (saved !== undefined) process.env.XERXES_RESEARCH_ALLOWED_DOMAINS = saved
  }
})

test('the env allowlist is honored by the registered WebScraper and refuses with a clear message', async () => {
  const saved = process.env.XERXES_RESEARCH_ALLOWED_DOMAINS
  process.env.XERXES_RESEARCH_ALLOWED_DOMAINS = 'example.com'
  try {
    let fetches = 0
    const registry = new ToolRegistry()
    registerWebTools(registry, client(async () => {
      fetches += 1
      return new Response(PAGE, { status: 200 })
    }))
    const context = { metadata: {}, sessionId: 'env-session' }
    let refusal = ''
    try {
      refusal = await registry.execute(call('WebScraper', { url: 'https://evil.com/' }), context)
    } catch (error) {
      refusal = error instanceof Error ? error.message : String(error)
    }
    expect(refusal).toContain('allowlist')
    expect(fetches).toBe(0)
    expect(log.list('env-session')).toEqual([])

    const ok = await registry.execute(call('WebScraper', { url: 'https://sub.example.com/' }), context)
    expect(ok).toContain('Hello Page')
  } finally {
    if (saved === undefined) delete process.env.XERXES_RESEARCH_ALLOWED_DOMAINS
    else process.env.XERXES_RESEARCH_ALLOWED_DOMAINS = saved
  }
})

test('DuckDuckGoSearch records every returned result under the session, and under default without one', async () => {
  const provider: DuckDuckGoSearchProvider = {
    search: async () => [
      { snippet: 's', title: 'One', url: 'https://example.com/one' },
      { snippet: 's', title: 'Two', url: 'https://example.com/two' },
      { snippet: 's', title: 'One again', url: 'https://example.com/one' },
    ],
  }
  const registry = new ToolRegistry()
  registerDuckDuckGoSearchTool(registry, new DuckDuckGoSearch(provider))

  await registry.execute(call('DuckDuckGoSearch', { query: 'xerxes' }), { metadata: {}, sessionId: 'search-session' })
  expect(log.list('search-session').map(entry => [entry.url, entry.title, entry.tool])).toEqual([
    ['https://example.com/one', 'One', 'DuckDuckGoSearch'],
    ['https://example.com/two', 'Two', 'DuckDuckGoSearch'],
  ])

  await registry.execute(call('DuckDuckGoSearch', { query: 'xerxes' }), { metadata: {} })
  expect(log.list(DEFAULT_SOURCE_SESSION)).toHaveLength(2)
})
