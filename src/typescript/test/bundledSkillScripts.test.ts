// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ArxivClient, arxivSearchUrl, formatArxivResult, parseArxivAtom } from '../src/skills/arxiv.js'
import {
  concatExcalidrawBuffers,
  createExcalidrawEncryptedPayload,
  parseExcalidrawDocument,
  uploadExcalidrawDocument,
} from '../src/skills/excalidraw.js'
import { FindNearbyClient, formatNearbyPlaces, haversineMeters, overpassNearbyQuery } from '../src/skills/nearby.js'
import {
  PolymarketClient,
  formatPolymarketMarket,
  formatPolymarketPercentage,
  formatPolymarketVolume,
} from '../src/skills/polymarket.js'
import {
  YoutubeTranscriptClient,
  extractYoutubeVideoId,
  formatYoutubeTimestamp,
  parseYoutubeTimedText,
  summarizeYoutubeTranscript,
} from '../src/skills/youtubeTranscript.js'

test('arXiv builds native Atom requests and parses results without an XML SDK', async () => {
  const atom = `<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"><opensearch:totalResults>42</opensearch:totalResults><entry><id>http://arxiv.org/abs/2501.12345v2</id><title>  Native\n  Bun &amp; TypeScript  </title><summary>First\nparagraph &amp; details.</summary><published>2025-01-02T03:04:05Z</published><updated>2025-02-03T04:05:06Z</updated><author><name>Ada Lovelace</name></author><author><name>Grace Hopper</name></author><category term="cs.AI" /><category term="cs.SE" /></entry></feed>`
  let requested = ''
  const client = new ArxivClient({
    fetchImplementation: async input => {
      requested = String(input)
      return new Response(atom, { headers: { 'content-type': 'application/atom+xml' } })
    },
  })
  const result = await client.search({ query: 'Bun TypeScript', author: 'Ada Lovelace', maxResults: 2, sort: 'date' })
  expect(requested).toContain('search_query=all:Bun%20TypeScript+AND+au:Ada%20Lovelace')
  expect(requested).toContain('sortBy=submittedDate')
  expect(result).toEqual({
    totalResults: 42,
    papers: [{
      abstract: 'First paragraph & details.',
      arxivId: '2501.12345',
      authors: ['Ada Lovelace', 'Grace Hopper'],
      categories: ['cs.AI', 'cs.SE'],
      pdfUrl: 'https://arxiv.org/pdf/2501.12345',
      published: '2025-01-02',
      title: 'Native Bun & TypeScript',
      updated: '2025-02-03',
      url: 'https://arxiv.org/abs/2501.12345',
      version: 'v2',
    }],
  })
  expect(formatArxivResult(result)).toContain('Found 42 results (showing 1)')
  expect(parseArxivAtom('<feed />').papers).toEqual([])
  expect(arxivSearchUrl({ ids: '2501.12345', sort: 'updated' })).toContain('id_list=2501.12345')
})

test('Polymarket client covers public search, market data, order books, and trade history natively', async () => {
  const calls: string[] = []
  const client = new PolymarketClient({
    gammaUrl: 'https://gamma.test',
    clobUrl: 'https://clob.test',
    dataUrl: 'https://data.test',
    fetchImplementation: async input => {
      const url = String(input)
      calls.push(url)
      if (url.includes('/public-search')) return Response.json({
        events: [{ title: 'Election', slug: 'election', volume: '1200', markets: [{
          question: 'Will it ship?', slug: 'ship', outcomePrices: '["0.75", "0.25"]', outcomes: '["Yes", "No"]', volume: '2500',
        }] }],
        pagination: { totalResults: 7 },
      })
      if (url.includes('/price?')) return Response.json({ price: '0.61' })
      if (url.includes('/midpoint')) return Response.json({ mid: '0.60' })
      if (url.includes('/spread')) return Response.json({ spread: '0.02' })
      if (url.includes('/book?')) return Response.json({
        bids: [{ price: '0.4', size: '4' }, { price: '0.7', size: '2' }],
        asks: [{ price: '0.9', size: '1' }, { price: '0.8', size: '3' }],
        last_trade_price: '0.5', tick_size: '0.01',
      })
      if (url.includes('/prices-history')) return Response.json({ history: [{ t: 1_000, p: '0.4' }] })
      if (url.includes('/trades')) return Response.json([{ side: 'BUY', price: '0.4', size: '5', outcome: 'Yes', title: 'Election' }])
      throw new Error(`unexpected endpoint ${url}`)
    },
  })
  const search = await client.search('election 2026')
  expect(search.totalResults).toBe(7)
  expect(search.events[0]?.markets[0]?.outcomePrices).toEqual(['0.75', '0.25'])
  expect(formatPolymarketMarket(search.events[0]?.markets[0] ?? fail('missing market'))).toContain('Yes: 75.0%')
  expect(await client.price('token id')).toEqual({ tokenId: 'token id', buy: '0.61', midpoint: '0.60', spread: '0.02' })
  const book = await client.book('token id')
  expect(book.bids.map(order => order.price)).toEqual(['0.7', '0.4'])
  expect(book.asks.map(order => order.price)).toEqual(['0.8', '0.9'])
  expect(await client.history('condition')).toEqual([{ timestamp: 1_000, price: 0.4 }])
  expect(await client.trades({ market: 'condition' })).toMatchObject([{ title: 'Election', side: 'BUY' }])
  expect(calls.some(call => call.includes('q=election%202026'))).toBeTrue()
  expect(formatPolymarketPercentage('0.375')).toBe('37.5%')
  expect(formatPolymarketVolume('1200')).toBe('$1.2K')
})

test('YouTube transcript client discovers a native caption track and keeps transcript formatting behavior', async () => {
  const player = JSON.stringify({
    captions: {
      playerCaptionsTracklistRenderer: {
        captionTracks: [
          { baseUrl: 'https://captions.test/api?lang=en', languageCode: 'en' },
          { baseUrl: 'https://captions.test/api?lang=tr', languageCode: 'tr' },
        ],
      },
    },
  })
  const calls: string[] = []
  const client = new YoutubeTranscriptClient({
    watchUrl: 'https://youtube.test/watch',
    fetchImplementation: async input => {
      const url = String(input)
      calls.push(url)
      if (url.startsWith('https://youtube.test/watch')) return new Response(`<script>var ytInitialPlayerResponse = ${player};</script>`)
      return Response.json({ events: [
        { tStartMs: 0, dDurationMs: 1_500, segs: [{ utf8: 'Hello &amp; ' }, { utf8: 'world' }] },
        { tStartMs: 1_500, dDurationMs: 500, segs: [{ utf8: 'again' }] },
      ] })
    },
  })
  const summary = await client.summarize('https://youtu.be/abcdefghijk', { languages: ['tr'], timestamps: true })
  expect(summary).toEqual({
    duration: '0:02', fullText: 'Hello & world again', segmentCount: 2,
    timestampedText: '0:00 Hello & world\n0:01 again', videoId: 'abcdefghijk',
  })
  expect(calls[1]).toContain('lang=tr')
  expect(calls[1]).toContain('fmt=json3')
  expect(extractYoutubeVideoId('https://www.youtube.com/shorts/abcdefghijk?feature=share')).toBe('abcdefghijk')
  expect(formatYoutubeTimestamp(3_661)).toBe('1:01:01')
  expect(parseYoutubeTimedText('<transcript><text start="2" dur="1.5">A &amp; B</text></transcript>')).toEqual([
    { start: 2, duration: 1.5, text: 'A & B' },
  ])
  expect(summarizeYoutubeTranscript('abcdefghijk', [], { timestamps: true }).duration).toBe('0:00')
})

test('find-nearby client geocodes then fails over between Overpass mirrors while preserving zero coordinates', async () => {
  const calls: string[] = []
  const client = new FindNearbyClient({
    geocodeUrl: 'https://nominatim.test/search',
    overpassUrls: ['https://overpass.one/api', 'https://overpass.two/api'],
    fetchImplementation: async input => {
      const url = String(input)
      calls.push(url)
      if (url.startsWith('https://nominatim.test')) return Response.json([{ lat: '0', lon: '0' }])
      if (url.startsWith('https://overpass.one')) return new Response('unavailable', { status: 503 })
      return Response.json({ elements: [
        { lat: 0, lon: 0, tags: { name: 'Zero Cafe', amenity: 'cafe', 'addr:street': 'Main', 'addr:housenumber': '1' } },
        { center: { lat: 0.01, lon: 0 }, tags: { name: 'Far Cafe', amenity: 'cafe', cuisine: 'coffee' } },
      ] })
    },
  })
  const origin = await client.geocode('Null Island')
  const places = await client.findNearby(origin, { types: ['cafe'], radius: 2_000 })
  expect(origin).toEqual({ lat: 0, lon: 0 })
  expect(places.map(place => place.name)).toEqual(['Zero Cafe', 'Far Cafe'])
  expect(places[0]).toMatchObject({ distanceMeters: 0, address: '1 Main' })
  expect(calls).toContain('https://overpass.one/api')
  expect(calls).toContain('https://overpass.two/api')
  expect(haversineMeters({ lat: 0, lon: 0 }, { lat: 0, lon: 0 })).toBe(0)
  expect(overpassNearbyQuery({ lat: 1, lon: 2 }, ['cafe'], 500)).toContain('around:500,1,2')
  expect(formatNearbyPlaces(places, ['cafe'], 2_000)).toContain('Zero Cafe')
})

test('Excalidraw upload uses native AES-GCM payload construction and returns a share URL from the injected transport', async () => {
  expect([...concatExcalidrawBuffers(new Uint8Array([1, 2]), new Uint8Array([3]))]).toEqual([
    0, 0, 0, 1, 0, 0, 0, 2, 1, 2, 0, 0, 0, 1, 3,
  ])
  expect(parseExcalidrawDocument('{"elements":[]}')).toEqual({ elements: [] })
  const encrypted = await createExcalidrawEncryptedPayload('{"elements":[]}', {
    randomBytes: length => new Uint8Array(length).fill(length),
  })
  expect(encrypted.encryptionKey).toEqual(new Uint8Array(16).fill(16))
  expect(encrypted.iv).toEqual(new Uint8Array(12).fill(12))
  expect(encrypted.payload.byteLength).toBeGreaterThan(40)

  let requestBody: BodyInit | null | undefined
  const url = await uploadExcalidrawDocument('{"elements":[]}', {
    compressor: async value => value,
    crypto: { encrypt: async (_key, _iv, plaintext) => new Uint8Array([...plaintext, 9]) },
    randomBytes: length => new Uint8Array(length).fill(7),
    uploadUrl: 'https://excalidraw.test/upload',
    fetchImplementation: async (_input, init) => {
      requestBody = init?.body
      return Response.json({ id: 'file-123' })
    },
  })
  expect(requestBody).toBeInstanceOf(ArrayBuffer)
  expect(url).toBe('https://excalidraw.com/#json=file-123,BwcHBwcHBwcHBwcHBwcHBw')
})

function fail(message: string): never {
  throw new Error(message)
}
