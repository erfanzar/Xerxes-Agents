// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ConfigurationError, ValidationError } from '../src/core/errors.js'
import {
  DuckDuckGoInstantAnswerProvider,
  DuckDuckGoSearch,
  type DuckDuckGoSearchProvider,
} from '../src/tools/duckduckgoEngine.js'
import { GoogleCustomSearchClient, googleSearchLimitations } from '../src/tools/googleSearch.js'
import {
  PublicWebClient,
  analyzeUrl,
  apiRequest,
  readRssFeed,
  scrapeWebPage,
  type WebFetch,
} from '../src/tools/webTools.js'

function client(fetcher: WebFetch): PublicWebClient {
  return new PublicWebClient({ fetcher })
}

test('PublicWebClient blocks literal private targets and rechecks public redirects', async () => {
  const calls: string[] = []
  const redirecting = client(async url => {
    calls.push(url)
    if (url.endsWith('/start')) {
      return new Response(null, { headers: { location: '/end' }, status: 302 })
    }
    return new Response('ok', { status: 200 })
  })
  const result = await redirecting.fetch('https://example.com/start')
  expect(result.url).toBe('https://example.com/end')
  expect(calls).toEqual(['https://example.com/start', 'https://example.com/end'])

  let privateFetches = 0
  const blocked = client(async () => {
    privateFetches += 1
    return new Response('unexpected')
  })
  await expect(blocked.fetch('http://127.0.0.1:8080/private')).rejects.toBeInstanceOf(ValidationError)
  expect(privateFetches).toBe(0)

  const unsafeRedirect = client(async () => new Response(null, {
    headers: { location: 'http://127.0.0.1/admin' },
    status: 302,
  }))
  await expect(unsafeRedirect.fetch('https://example.com/start')).rejects.toBeInstanceOf(ValidationError)
})

test('web scraper extracts static HTML without browser automation', async () => {
  const html = `
    <html><head><title>Xerxes &amp; Bun</title><meta name="description" content="Runtime"></head>
    <body><article class="lead">Hello <strong>world</strong><a href="/guide">Guide</a></article>
    <img src="/logo.png" alt="Xerxes"></body></html>`
  const page = await scrapeWebPage({
    extractImages: true,
    extractLinks: true,
    selector: 'article.lead',
    url: 'https://docs.example.com/page',
  }, client(async () => new Response(html, { status: 200 })))

  expect(page.title).toBe('Xerxes & Bun')
  expect(page.meta.description).toBe('Runtime')
  expect(page.selectedContent).toEqual(['Hello worldGuide'])
  expect(page.links).toEqual([{ text: 'Guide', url: 'https://docs.example.com/guide' }])
  expect(page.images).toEqual([{ alt: 'Xerxes', src: 'https://docs.example.com/logo.png' }])
})

test('API client serializes JSON, preserves safe query parameters, and redacts cookie headers', async () => {
  let requestUrl = ''
  let requestInit: RequestInit | undefined
  const api = await apiRequest({
    headers: { authorization: 'Bearer configured-token' },
    jsonData: { enabled: true },
    method: 'POST',
    params: { active: true, page: 2, remove_me: null },
    url: 'https://api.example.com/v1/items?existing=yes&remove_me=present',
  }, client(async (url, init) => {
    requestUrl = url
    requestInit = init
    return new Response(JSON.stringify({ data: [{ id: 'one' }] }), {
      headers: { 'content-type': 'application/json', 'set-cookie': 'secret=session' },
      status: 201,
    })
  }))

  const parsedUrl = new URL(requestUrl)
  expect(parsedUrl.searchParams.get('active')).toBe('true')
  expect(parsedUrl.searchParams.get('page')).toBe('2')
  expect(parsedUrl.searchParams.has('remove_me')).toBeFalse()
  expect(requestInit?.body).toBe('{"enabled":true}')
  expect(api.statusCode).toBe(201)
  expect(api.json).toEqual({ data: [{ id: 'one' }] })
  expect(api.headers['set-cookie']).toBe('[REDACTED]')

  await expect(apiRequest({
    headers: { Host: 'internal.example' },
    url: 'https://api.example.com/v1/items',
  }, client(async () => new Response('unused')))).rejects.toBeInstanceOf(ValidationError)
})

test('RSS and Atom feeds are parsed without external XML entities or feed parser dependencies', async () => {
  const rss = `<?xml version="1.0"?><rss><channel><title>Runtime feed</title><description>Updates</description>
    <link>https://example.com/feed</link><item><title>First</title><link>https://example.com/one</link>
    <description><![CDATA[<p>First update</p>]]></description><category>release</category></item></channel></rss>`
  const feed = await readRssFeed({ feedUrl: 'https://example.com/rss' }, client(async () => new Response(rss, { status: 200 })))
  expect(feed).toMatchObject({ description: 'Updates', link: 'https://example.com/feed', title: 'Runtime feed' })
  expect(feed.items).toEqual([{
    author: '', content: 'First update', link: 'https://example.com/one', published: '', tags: ['release'], title: 'First',
  }])

  const atom = `<feed xmlns="http://www.w3.org/2005/Atom"><title>Atom feed</title><link href="https://example.com/atom"/>
    <entry><title>Atom entry</title><link href="https://example.com/atom/one"/><updated>2026-07-13T00:00:00Z</updated>
    <summary>Atom summary</summary></entry></feed>`
  const atomFeed = await readRssFeed({ feedUrl: 'https://example.com/atom' }, client(async () => new Response(atom, { status: 200 })))
  expect(atomFeed.title).toBe('Atom feed')
  expect(atomFeed.items[0]).toMatchObject({ content: 'Atom summary', link: 'https://example.com/atom/one', title: 'Atom entry' })
})

test('URL analysis performs optional public availability and metadata checks only', async () => {
  const methods: string[] = []
  const analysis = await analyzeUrl({
    checkAvailability: true,
    extractMetadata: true,
    url: 'https://example.com/docs;version?tab=api#top',
  }, client(async (_url, init) => {
    methods.push(init.method ?? 'GET')
    if (init.method === 'HEAD') return new Response(null, { status: 200 })
    return new Response('<title>Docs</title><meta property="og:type" content="article">', { status: 200 })
  }))
  expect(analysis).toMatchObject({
    domain: 'example.com', domainName: 'example.com', isAvailable: true, isFetchable: true,
    params: 'version', path: '/docs', query: 'tab=api', title: 'Docs', tld: 'com',
  })
  expect(analysis.openGraph).toEqual({ 'og:type': 'article' })
  expect(methods).toEqual(['HEAD', 'GET'])

  let privateCalls = 0
  const privateAnalysis = await analyzeUrl({ checkAvailability: true, url: 'http://192.168.1.5/admin' }, client(async () => {
    privateCalls += 1
    return new Response('unexpected')
  }))
  expect(privateAnalysis.isValid).toBeTrue()
  expect(privateAnalysis.isFetchable).toBeFalse()
  expect(privateCalls).toBe(0)
})

test('Google Custom Search uses explicit credentials and has no scraping fallback', async () => {
  expect(() => new GoogleCustomSearchClient({ apiKey: '', searchEngineId: 'engine' })).toThrow(ConfigurationError)
  let requestUrl = ''
  const search = new GoogleCustomSearchClient({
    apiKey: 'google-api-key',
    searchEngineId: 'search-engine-id',
    webClient: client(async (url) => {
      requestUrl = url
      return new Response(JSON.stringify({
        items: [{ displayLink: 'github.com', link: 'https://github.com/erfanzar/Xerxes', snippet: 'Agent runtime', title: 'Xerxes' }],
        searchInformation: { totalResults: '1' },
      }), { status: 200 })
    }),
  })
  const result = await search.search({ nResults: 99, query: 'xerxes', site: 'github.com', timeRange: 'm6' })
  const endpoint = new URL(requestUrl)
  expect(endpoint.searchParams.get('key')).toBe('google-api-key')
  expect(endpoint.searchParams.get('cx')).toBe('search-engine-id')
  expect(endpoint.searchParams.get('num')).toBe('10')
  expect(endpoint.searchParams.get('q')).toBe('site:github.com xerxes')
  expect(result).toMatchObject({ count: 1, engine: 'google_api', results: [{ title: 'Xerxes', url: 'https://github.com/erfanzar/Xerxes' }] })
  expect(JSON.stringify(result)).not.toContain('google-api-key')
  expect(googleSearchLimitations()).toContain('Anonymous Google HTML scraping is deliberately not implemented.')
})

test('DuckDuckGo facade filters a host-provided provider and keeps Instant Answer limitations explicit', async () => {
  let providerRequest = ''
  const provider: DuckDuckGoSearchProvider = {
    search: async request => {
      providerRequest = request.query
      return [
        { snippet: 'runtime guide', title: 'Xerxes guide', url: 'https://docs.example.com/guide' },
        { snippet: 'runtime guide', title: 'Other', url: 'https://outside.example.net/guide' },
      ]
    },
  }
  const search = new DuckDuckGoSearch(provider)
  const filtered = await search.search({
    allowedDomains: ['example.com'], fileType: 'pdf', nResults: 5, query: 'xerxes', titleLengthLimit: 8,
  })
  expect(providerRequest).toContain('filetype:pdf')
  expect(providerRequest).toContain('site:example.com')
  expect(filtered.results).toEqual([{ snippet: 'runtime guide', title: 'Xerxes g', url: 'https://docs.example.com/guide' }])
  expect(filtered.metadata.totalResults).toBe(1)

  const instantProvider = new DuckDuckGoInstantAnswerProvider({
    webClient: client(async () => new Response(JSON.stringify({
      AbstractSource: 'Wikipedia', AbstractText: 'A framework', AbstractURL: 'https://example.com/xerxes', Heading: 'Xerxes',
      RelatedTopics: [{ FirstURL: 'https://example.com/topic', Text: 'Related topic' }],
    }), { status: 200 })),
  })
  const instant = new DuckDuckGoSearch(instantProvider)
  expect((await instant.search({ query: 'xerxes' })).results.map(result => result.url)).toEqual([
    'https://example.com/xerxes', 'https://example.com/topic',
  ])
  await expect(instant.search({ query: 'xerxes', searchType: 'news' })).rejects.toBeInstanceOf(ConfigurationError)
})
