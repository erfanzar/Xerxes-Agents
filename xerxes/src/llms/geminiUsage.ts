// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * What Xerxes can know about a Gemini API key's limits. Google publishes no
 * quota endpoint for API keys and sends no rate-limit headers; limits are per
 * Google Cloud project and shown only in AI Studio. So this records two
 * observed facts per key, in this process only: how many requests Xerxes sent
 * since midnight Pacific (when the daily quota resets), and the last 429 with
 * the quota Google named and when it said to retry. Nothing is estimated.
 */

import { createHash } from 'node:crypto'

export const GEMINI_LIMITS_URL = 'https://aistudio.google.com/rate-limit'

export type GeminiLimitKind = 'per-minute' | 'per-day' | 'other'

export interface GeminiLimitHit {
  readonly at: number
  readonly kind: GeminiLimitKind
  readonly quotaId?: string
  readonly limit?: string
  readonly model?: string
  /** Epoch ms Google said a retry may succeed (RetryInfo.retryDelay). */
  readonly retryAt?: number
}

/** Same short hash the usage report keys API-key credentials by. */
export function geminiKeyId(apiKey: string): string {
  return createHash('sha256').update(apiKey.trim()).digest('hex').slice(0, 16)
}

/** The calendar day in Pacific time, when Gemini's requests-per-day quota resets. */
export function pacificDay(now: number): string {
  return new Intl.DateTimeFormat('en-CA', { timeZone: 'America/Los_Angeles', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date(now))
}

const record = (value: unknown): Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}

/** Read Google's QuotaFailure and RetryInfo details out of a 429 body; kind 'other' when it carries none. */
export function parseGeminiQuotaError(body: string, now: number, model?: string): GeminiLimitHit {
  let details: unknown[] = []
  try { const parsed = record(record(JSON.parse(body)).error).details; if (Array.isArray(parsed)) details = parsed } catch { /* not JSON */ }
  let quotaId: string | undefined, limit: string | undefined, quotaModel: string | undefined, retryAt: number | undefined
  for (const raw of details) {
    const detail = record(raw)
    const type = typeof detail['@type'] === 'string' ? detail['@type'] : ''
    if (type.endsWith('QuotaFailure') && Array.isArray(detail.violations)) {
      const violation = record(detail.violations[0])
      if (typeof violation.quotaId === 'string') quotaId = violation.quotaId
      if (typeof violation.quotaValue === 'string' || typeof violation.quotaValue === 'number') limit = String(violation.quotaValue)
      const dimensions = record(violation.quotaDimensions)
      if (typeof dimensions.model === 'string') quotaModel = dimensions.model
    }
    if (type.endsWith('RetryInfo') && typeof detail.retryDelay === 'string') {
      const seconds = Number(/^(\d+(?:\.\d+)?)s$/.exec(detail.retryDelay.trim())?.[1])
      if (Number.isFinite(seconds)) retryAt = now + seconds * 1000
    }
  }
  const kind: GeminiLimitKind = /perday/i.test(quotaId ?? '') ? 'per-day' : /perminute/i.test(quotaId ?? '') ? 'per-minute' : 'other'
  const named = quotaModel ?? model
  return { at: now, kind, ...(quotaId ? { quotaId } : {}), ...(limit ? { limit } : {}), ...(named ? { model: named } : {}), ...(retryAt ? { retryAt } : {}) }
}

export class GeminiUsageLedger {
  private readonly calls = new Map<string, { day: string; count: number }>()
  private readonly hits = new Map<string, GeminiLimitHit>()

  constructor(private readonly now: () => number = Date.now) {}

  recordCall(keyId: string): void {
    const day = pacificDay(this.now())
    const current = this.calls.get(keyId)
    this.calls.set(keyId, current?.day === day ? { day, count: current.count + 1 } : { day, count: 1 })
  }

  recordLimit(keyId: string, body: string, model?: string): void {
    this.hits.set(keyId, parseGeminiQuotaError(body, this.now(), model))
  }

  /** Label/value rows for the Usage tab. */
  facts(keyId: string): { label: string; value: string }[] {
    const now = this.now(), today = pacificDay(now)
    const calls = this.calls.get(keyId)
    const rows = [{ label: 'Requests today (Xerxes)', value: `${calls?.day === today ? calls.count : 0} since midnight Pacific` }]
    const hit = this.hits.get(keyId)
    // A daily limit stops applying at the Pacific day boundary; an older hit is history, not state.
    if (hit && pacificDay(hit.at) === today) {
      const what = hit.kind === 'per-day' ? 'Daily limit' : hit.kind === 'per-minute' ? 'Per-minute limit' : 'Rate limit'
      const scope = [hit.model, hit.limit ? `limit ${hit.limit}` : ''].filter(Boolean).join(', ')
      const time = new Date(hit.at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      const when = hit.kind === 'per-day' ? 'resets at midnight Pacific'
        : hit.retryAt && hit.retryAt > now ? `retry in ${Math.ceil((hit.retryAt - now) / 1000)}s` : 'cleared'
      rows.push({ label: 'Last limit hit', value: `${what}${scope ? ` (${scope})` : ''} at ${time} · ${when}` })
    }
    rows.push({ label: 'Your limits', value: GEMINI_LIMITS_URL })
    return rows
  }
}

/** One ledger per process: the daemon's Gemini clients write it, its usage report reads it. */
export const geminiUsageLedger = new GeminiUsageLedger()
