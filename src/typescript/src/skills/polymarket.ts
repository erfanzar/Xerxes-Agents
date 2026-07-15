// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  requireSkillText,
  skillFetchJson,
  skillJsonArray,
  skillJsonObject,
  type SkillFetch,
  type SkillJsonObject,
} from './http.js'

/** Public Polymarket REST endpoints used by the research skill. */
export const POLYMARKET_GAMMA_URL = 'https://gamma-api.polymarket.com'
export const POLYMARKET_CLOB_URL = 'https://clob.polymarket.com'
export const POLYMARKET_DATA_URL = 'https://data-api.polymarket.com'

export interface PolymarketClientOptions {
  readonly clobUrl?: string
  readonly dataUrl?: string
  readonly fetchImplementation?: SkillFetch
  readonly gammaUrl?: string
}

export interface PolymarketMarket {
  readonly clobTokenIds: readonly string[]
  readonly closed: boolean
  readonly conditionId?: string
  readonly description?: string
  readonly outcomePrices: readonly string[]
  readonly outcomes: readonly string[]
  readonly question: string
  readonly raw: SkillJsonObject
  readonly slug: string
  readonly volume: unknown
}

export interface PolymarketEvent {
  readonly closed: boolean
  readonly markets: readonly PolymarketMarket[]
  readonly raw: SkillJsonObject
  readonly slug: string
  readonly title: string
  readonly volume: unknown
}

export interface PolymarketSearchResult {
  readonly events: readonly PolymarketEvent[]
  readonly totalResults: number
}

export interface PolymarketPrice {
  readonly buy: string
  readonly midpoint: string
  readonly spread: string
  readonly tokenId: string
}

export interface PolymarketOrder {
  readonly price: string
  readonly raw: SkillJsonObject
  readonly size: string
}

export interface PolymarketOrderBook {
  readonly asks: readonly PolymarketOrder[]
  readonly bids: readonly PolymarketOrder[]
  readonly lastTradePrice: string
  readonly tickSize: string
  readonly tokenId: string
}

export interface PolymarketHistoryPoint {
  readonly price: number
  readonly timestamp: number
}

export interface PolymarketTrade {
  readonly market?: string
  readonly outcome: string
  readonly price: string
  readonly raw: SkillJsonObject
  readonly side: string
  readonly size: string
  readonly timestamp?: string
  readonly title: string
}

/** Format a decimal probability (`0..1`) as the CLI's one-decimal percentage string. */
export function formatPolymarketPercentage(value: unknown): string {
  const numeric = finiteNumber(value)
  return numeric === undefined ? String(value) : `${(numeric * 100).toFixed(1)}%`
}

/** Format volume as the compact USD representation used in the bundled skill. */
export function formatPolymarketVolume(value: unknown): string {
  const numeric = finiteNumber(value)
  if (numeric === undefined) return String(value)
  if (numeric >= 1_000_000) return `$${(numeric / 1_000_000).toFixed(1)}M`
  if (numeric >= 1_000) return `$${(numeric / 1_000).toFixed(1)}K`
  return `$${numeric.toFixed(0)}`
}

/** Decode stringified array fields returned by parts of the Gamma API. */
export function parsePolymarketJsonField(value: unknown): unknown {
  if (typeof value !== 'string') return value
  try {
    return JSON.parse(value) as unknown
  } catch {
    return value
  }
}

/** Render a market summary in the original skill's human-readable layout. */
export function formatPolymarketMarket(market: PolymarketMarket, indent = ''): string {
  const status = market.closed ? ' [CLOSED]' : ''
  const lines: string[] = []
  if (market.outcomePrices.length >= 2) {
    const labels = market.outcomes.length ? market.outcomes : ['Yes', 'No']
    const prices = market.outcomePrices
      .slice(0, labels.length)
      .map((price, index) => `${labels[index] ?? `Outcome ${index}`}: ${formatPolymarketPercentage(price)}`)
      .join(' / ')
    lines.push(`${indent}${market.question}${status}`, `${indent}  ${prices}  |  Volume: ${formatPolymarketVolume(market.volume)}`)
  } else {
    lines.push(`${indent}${market.question}${status}  |  Volume: ${formatPolymarketVolume(market.volume)}`)
  }
  if (market.slug) lines.push(`${indent}  slug: ${market.slug}`)
  return lines.join('\n')
}

/** Native client for Polymarket's public Gamma, CLOB, and Data APIs. */
export class PolymarketClient {
  private readonly clobUrl: string
  private readonly dataUrl: string
  private readonly fetchImplementation: SkillFetch
  private readonly gammaUrl: string

  constructor(options: PolymarketClientOptions = {}) {
    this.gammaUrl = normalizedBaseUrl(options.gammaUrl ?? POLYMARKET_GAMMA_URL, 'gammaUrl')
    this.clobUrl = normalizedBaseUrl(options.clobUrl ?? POLYMARKET_CLOB_URL, 'clobUrl')
    this.dataUrl = normalizedBaseUrl(options.dataUrl ?? POLYMARKET_DATA_URL, 'dataUrl')
    this.fetchImplementation = options.fetchImplementation ?? fetch
  }

  async search(query: string, signal?: AbortSignal): Promise<PolymarketSearchResult> {
    const data = await this.getJson(this.gamma(`/public-search?q=${encodeURIComponent(requireSkillText(query, 'query'))}`), signal)
    const response = skillJsonObject(data, 'Polymarket search response')
    const events = records(response.events, 'Polymarket search events').map(toEvent)
    const pagination = optionalRecord(response.pagination)
    const totalResults = finiteNumber(pagination?.totalResults) ?? events.length
    return { events, totalResults }
  }

  async trending(limit = 10, signal?: AbortSignal): Promise<readonly PolymarketEvent[]> {
    assertPositiveLimit(limit, 'limit')
    const data = await this.getJson(
      this.gamma(`/events?limit=${limit}&active=true&closed=false&order=volume&ascending=false`),
      signal,
    )
    return skillJsonArray(data, 'Polymarket trending response')
      .map((item, index) => skillJsonObject(item, `Polymarket trending event ${index}`))
      .map(toEvent)
  }

  async market(slug: string, signal?: AbortSignal): Promise<PolymarketMarket | undefined> {
    const data = await this.getJson(this.gamma(`/markets?slug=${encodeURIComponent(requireSkillText(slug, 'slug'))}`), signal)
    const values = skillJsonArray(data, 'Polymarket markets response')
    const first = values[0]
    return first === undefined ? undefined : toMarket(skillJsonObject(first, 'Polymarket market'))
  }

  async event(slug: string, signal?: AbortSignal): Promise<PolymarketEvent | undefined> {
    const data = await this.getJson(this.gamma(`/events?slug=${encodeURIComponent(requireSkillText(slug, 'slug'))}`), signal)
    const values = skillJsonArray(data, 'Polymarket events response')
    const first = values[0]
    return first === undefined ? undefined : toEvent(skillJsonObject(first, 'Polymarket event'))
  }

  async price(tokenId: string, signal?: AbortSignal): Promise<PolymarketPrice> {
    const normalizedTokenId = requireSkillText(tokenId, 'tokenId')
    const encoded = encodeURIComponent(normalizedTokenId)
    const [buyData, midpointData, spreadData] = await Promise.all([
      this.getJson(this.clob(`/price?token_id=${encoded}&side=buy`), signal),
      this.getJson(this.clob(`/midpoint?token_id=${encoded}`), signal),
      this.getJson(this.clob(`/spread?token_id=${encoded}`), signal),
    ])
    const buy = skillJsonObject(buyData, 'Polymarket buy price')
    const midpoint = skillJsonObject(midpointData, 'Polymarket midpoint')
    const spread = skillJsonObject(spreadData, 'Polymarket spread')
    return {
      buy: stringValue(buy.price, '?'),
      midpoint: stringValue(midpoint.mid, '?'),
      spread: stringValue(spread.spread, '?'),
      tokenId: normalizedTokenId,
    }
  }

  async book(tokenId: string, signal?: AbortSignal): Promise<PolymarketOrderBook> {
    const normalizedTokenId = requireSkillText(tokenId, 'tokenId')
    const data = await this.getJson(this.clob(`/book?token_id=${encodeURIComponent(normalizedTokenId)}`), signal)
    const book = skillJsonObject(data, 'Polymarket order book')
    const bids = records(book.bids, 'Polymarket bids').map(toOrder).sort((left, right) => orderPrice(right) - orderPrice(left))
    const asks = records(book.asks, 'Polymarket asks').map(toOrder).sort((left, right) => orderPrice(left) - orderPrice(right))
    return {
      asks,
      bids,
      lastTradePrice: stringValue(book.last_trade_price, '?'),
      tickSize: stringValue(book.tick_size, '?'),
      tokenId: normalizedTokenId,
    }
  }

  async history(
    conditionId: string,
    options: { readonly fidelity?: number; readonly interval?: string; readonly signal?: AbortSignal } = {},
  ): Promise<readonly PolymarketHistoryPoint[]> {
    const market = requireSkillText(conditionId, 'conditionId')
    const fidelity = options.fidelity ?? 50
    if (!Number.isSafeInteger(fidelity) || fidelity <= 0) throw new RangeError('fidelity must be a positive safe integer')
    const interval = options.interval ?? 'all'
    const data = await this.getJson(
      this.clob(`/prices-history?market=${encodeURIComponent(market)}&interval=${encodeURIComponent(interval)}&fidelity=${fidelity}`),
      options.signal,
    )
    const response = skillJsonObject(data, 'Polymarket price history')
    return records(response.history, 'Polymarket history').map((point, index) => {
      const timestamp = finiteNumber(point.t)
      const price = finiteNumber(point.p)
      if (timestamp === undefined || price === undefined) {
        throw new TypeError(`Polymarket history point ${index} requires numeric t and p fields`)
      }
      return { price, timestamp }
    })
  }

  async trades(
    options: { readonly limit?: number; readonly market?: string; readonly signal?: AbortSignal } = {},
  ): Promise<readonly PolymarketTrade[]> {
    const limit = options.limit ?? 10
    assertPositiveLimit(limit, 'limit')
    const market = options.market?.trim()
    const suffix = market ? `&market=${encodeURIComponent(market)}` : ''
    const data = await this.getJson(this.data(`/trades?limit=${limit}${suffix}`), options.signal)
    return skillJsonArray(data, 'Polymarket trades response')
      .map((item, index) => skillJsonObject(item, `Polymarket trade ${index}`))
      .map(toTrade)
  }

  private async getJson(url: string, signal?: AbortSignal): Promise<unknown> {
    return skillFetchJson(this.fetchImplementation, url, {
      headers: { Accept: 'application/json', 'User-Agent': 'xerxes/1.0' },
      ...(signal === undefined ? {} : { signal }),
    })
  }

  private gamma(path: string): string {
    return `${this.gammaUrl}${path}`
  }

  private clob(path: string): string {
    return `${this.clobUrl}${path}`
  }

  private data(path: string): string {
    return `${this.dataUrl}${path}`
  }
}

function normalizedBaseUrl(value: string, name: string): string {
  return requireSkillText(value, name).replace(/\/+$/, '')
}

function records(value: unknown, context: string): readonly SkillJsonObject[] {
  return skillJsonArray(value ?? [], context).map((item, index) => skillJsonObject(item, `${context} ${index}`))
}

function optionalRecord(value: unknown): SkillJsonObject | undefined {
  return value === null || typeof value !== 'object' || Array.isArray(value) ? undefined : value as SkillJsonObject
}

function stringArray(value: unknown): readonly string[] {
  const parsed = parsePolymarketJsonField(value)
  return Array.isArray(parsed) ? parsed.map(item => String(item)) : []
}

function stringValue(value: unknown, fallback = ''): string {
  return value === undefined || value === null ? fallback : String(value)
}

function finiteNumber(value: unknown): number | undefined {
  if (typeof value !== 'number' && typeof value !== 'string') return undefined
  if (typeof value === 'string' && !value.trim()) return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

function toMarket(raw: SkillJsonObject): PolymarketMarket {
  const conditionId = stringValue(raw.conditionId)
  const description = stringValue(raw.description)
  return {
    clobTokenIds: stringArray(raw.clobTokenIds),
    closed: raw.closed === true,
    ...(conditionId ? { conditionId } : {}),
    ...(description ? { description } : {}),
    outcomePrices: stringArray(raw.outcomePrices),
    outcomes: stringArray(raw.outcomes),
    question: stringValue(raw.question, '?'),
    raw,
    slug: stringValue(raw.slug),
    volume: raw.volume ?? 0,
  }
}

function toEvent(raw: SkillJsonObject): PolymarketEvent {
  return {
    closed: raw.closed === true,
    markets: records(raw.markets, 'Polymarket event markets').map(toMarket),
    raw,
    slug: stringValue(raw.slug),
    title: stringValue(raw.title, '?'),
    volume: raw.volume ?? 0,
  }
}

function toOrder(raw: SkillJsonObject): PolymarketOrder {
  return { price: stringValue(raw.price, '0'), raw, size: stringValue(raw.size, '0') }
}

function toTrade(raw: SkillJsonObject): PolymarketTrade {
  const market = stringValue(raw.market)
  const timestamp = stringValue(raw.timestamp)
  return {
    ...(market ? { market } : {}),
    outcome: stringValue(raw.outcome, '?'),
    price: stringValue(raw.price, '?'),
    raw,
    side: stringValue(raw.side, '?'),
    size: stringValue(raw.size, '?'),
    ...(timestamp ? { timestamp } : {}),
    title: stringValue(raw.title, '?'),
  }
}

function orderPrice(order: PolymarketOrder): number {
  return finiteNumber(order.price) ?? 0
}

function assertPositiveLimit(value: number, name: string): void {
  if (!Number.isSafeInteger(value) || value <= 0) throw new RangeError(`${name} must be a positive safe integer`)
}
