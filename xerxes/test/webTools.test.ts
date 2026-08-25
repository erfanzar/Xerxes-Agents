// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ConfigurationError, ValidationError, XerxesTimeoutError } from '../src/core/errors.js'
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
  // Deterministic public DNS answers keep these tests offline; the literal-IP
  // checks under test never consult the resolver.
  return new PublicWebClient({ fetcher, urlSafety: { dnsLookup: async () => ['93.184.216.34'] } })
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

test('PublicWebClient blocks hosts resolving to private addresses, including on redirect hops', async () => {
  let fetches = 0
  const blocked = new PublicWebClient({
    fetcher: async () => {
      fetches += 1
      return new Response('unexpected')
    },
    urlSafety: {
      dnsLookup: async hostname => hostname === 'internal.example' ? ['192.168.1.10'] : ['93.184.216.34'],
    },
  })
  await expect(blocked.fetch('https://internal.example/data')).rejects.toBeInstanceOf(ValidationError)
  expect(fetches).toBe(0)

  // A public start URL whose redirect target resolves privately is blocked on the hop.
  const requested: string[] = []
  const redirecting = new PublicWebClient({
    fetcher: async url => {
      requested.push(url)
      return new Response(null, { headers: { location: 'https://internal.example/data' }, status: 302 })
    },
    urlSafety: {
      dnsLookup: async hostname => hostname === 'internal.example' ? ['10.1.2.3'] : ['93.184.216.34'],
    },
  })
  await expect(redirecting.fetch('https://example.com/start')).rejects.toBeInstanceOf(ValidationError)
  expect(requested).toEqual(['https://example.com/start'])

  // Public DNS answers proceed normally.
  const open = new PublicWebClient({
    fetcher: async () => new Response('ok', { status: 200 }),
    urlSafety: { dnsLookup: async () => ['93.184.216.34'] },
  })
  await expect(open.fetch('https://example.com/start')).resolves.toMatchObject({ url: 'https://example.com/start' })
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

/**
 * A body that delivers one byte every `byteDelayMs` and never ends, plus the
 * number of pulls the stream served.
 *
 * The delay MUST be awaited inside an async pull: a synchronous pull that
 * schedules a delayed enqueue is re-invoked at microtask rate by the stream
 * machinery (per spec the next pull starts once the previous returns), which
 * floods megabytes per second and trips the byte cap long before any deadline —
 * a fixture bug this suite shipped once. An async pull serializes invocations,
 * so pull count is genuine wall-clock pacing.
 */
function slowTrickle(byteDelayMs = 25): {
  readonly stream: ReadableStream<Uint8Array>
  readonly pullCount: () => number
} {
  const encoder = new TextEncoder()
  let pulls = 0
  const stream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      pulls += 1
      await Bun.sleep(byteDelayMs)
      try {
        controller.enqueue(encoder.encode('x'))
      } catch {
        // The reader was cancelled while we slept; nothing to deliver to.
      }
    },
  })
  return { stream, pullCount: () => pulls }
}

test('a response body that trickles forever is cut off by the deadline instead of stalling the call', async () => {
  // The bug this pins: runWithTimeout bounded only fetch() (the headers); the
  // body was then read unbounded in text(), so a server that never finished the
  // response held the tool call open past every configured limit.
  const trickle = slowTrickle(25)
  const slow = new PublicWebClient({
    fetcher: async () => new Response(trickle.stream, { status: 200 }),
    timeoutMs: 500,
    urlSafety: { dnsLookup: async () => ['93.184.216.34'] },
  })
  const fetched = await slow.fetch('https://example.com/trickle')

  const started = Date.now()
  await expect(slow.text(fetched.response)).rejects.toBeInstanceOf(XerxesTimeoutError)
  // Well under the old behavior, which never resolved at all.
  expect(Date.now() - started).toBeLessThan(10_000)
  // Proof the fixture is genuinely slow: ~500ms at ≥25ms/pull is ~20 pulls.
  // A flood (the fixture bug) would show six-figure pull counts here.
  expect(trickle.pullCount()).toBeLessThan(1_000)
})

test('an aborted caller signal cancels an in-flight response body promptly', async () => {
  const aborts = new AbortController()
  const fetched = await client(async () => new Response(slowTrickle(25).stream, { status: 200 }))
    .fetch('https://example.com/trickle', {}, { signal: aborts.signal })

  const pending = client(async () => new Response(slowTrickle(25).stream, { status: 200 }))
    .text(fetched.response, { signal: aborts.signal })
  setTimeout(() => aborts.abort(new Error('caller gave up')), 100)
  const started = Date.now()
  await expect(pending).rejects.toThrow('caller gave up')
  expect(Date.now() - started).toBeLessThan(5_000)
})

test('a redirect hop body is cancelled instead of left pinned unread', async () => {
  // The hop's body is never read — only its Location mattered. Leaving it
  // un-consumed pinned each hop connection until GC noticed.
  let hopCancelled = false
  const hopStream = new ReadableStream<Uint8Array>({
    cancel() {
      hopCancelled = true
    },
  })
  const redirecting = client(url => {
    if (url.endsWith('/hop')) {
      return Promise.resolve(new Response(hopStream, { headers: { location: '/end' }, status: 302 }))
    }
    return Promise.resolve(new Response('done', { status: 200 }))
  })
  const fetched = await redirecting.fetch('https://example.com/hop')
  expect(fetched.url).toBe('https://example.com/end')
  const text = await redirecting.text(fetched.response)
  expect(text).toBe('done')

  for (let attempt = 0; attempt < 50 && !hopCancelled; attempt += 1) {
    await Bun.sleep(10)
  }
  expect(hopCancelled).toBe(true)
})

test('the byte cap still rejects oversized bodies rather than truncating them', async () => {
  const capped = new PublicWebClient({
    fetcher: async () => new Response('y'.repeat(2_000), { status: 200 }),
    maxResponseBytes: 1_000,
    urlSafety: { dnsLookup: async () => ['93.184.216.34'] },
  })
  const fetched = await capped.fetch('https://example.com/big')
  // Declared length over the cap.
  await expect(capped.text(fetched.response)).rejects.toThrow('response exceeds 1000 byte limit')

  // Undeclared length over the cap, caught mid-stream.
  const undeclared = new PublicWebClient({
    fetcher: async () => new Response(slowTrickle(5).stream, {
      headers: { 'content-type': 'text/plain' },
      status: 200,
    }),
    maxResponseBytes: 50,
    urlSafety: { dnsLookup: async () => ['93.184.216.34'] },
  })
  const streamed = await undeclared.fetch('https://example.com/big-stream')
  await expect(undeclared.text(streamed.response)).rejects.toThrow('response exceeds 50 byte limit')
})

test('a body read under an explicit fresh budget times out on its own', async () => {
  const fresh = new PublicWebClient({
    fetcher: async () => new Response(slowTrickle(25).stream, { status: 200 }),
    urlSafety: { dnsLookup: async () => ['93.184.216.34'] },
  })
  // A Response that did not come from this client's fetch carries no recorded
  // deadline; an explicit timeoutMs must still bound the read.
  const trickle = slowTrickle(25)
  const started = Date.now()
  await expect(fresh.text(await Promise.resolve(new Response(trickle.stream, { status: 200 })), { timeoutMs: 400 }))
    .rejects.toBeInstanceOf(XerxesTimeoutError)
  expect(Date.now() - started).toBeLessThan(10_000)
  expect(trickle.pullCount()).toBeLessThan(1_000)
})
