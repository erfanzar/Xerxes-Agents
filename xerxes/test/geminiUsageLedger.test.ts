// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { buildUsageReport } from '../src/auth/usageReport.js'
import { GEMINI_LIMITS_URL, GeminiUsageLedger, geminiKeyId, parseGeminiQuotaError } from '../src/llms/geminiUsage.js'

const quota429 = JSON.stringify({ error: { code: 429, status: 'RESOURCE_EXHAUSTED', details: [
  { '@type': 'type.googleapis.com/google.rpc.QuotaFailure', violations: [{ quotaId: 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', quotaValue: '250', quotaDimensions: { model: 'gemini-2.5-flash' } }] },
  { '@type': 'type.googleapis.com/google.rpc.RetryInfo', retryDelay: '37s' },
] } })

test('a 429 body names the daily quota, its limit, model and retry time', () => {
  const hit = parseGeminiQuotaError(quota429, 1_000)
  expect(hit).toEqual({ at: 1_000, kind: 'per-day', quotaId: 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', limit: '250', model: 'gemini-2.5-flash', retryAt: 38_000 })
  expect(parseGeminiQuotaError('not json', 5, 'gemini-x')).toEqual({ at: 5, kind: 'other', model: 'gemini-x' })
})

test('the ledger counts requests per key per Pacific day and reports the last limit hit', () => {
  let now = Date.parse('2026-11-16T18:00:00Z')
  const ledger = new GeminiUsageLedger(() => now)
  const key = geminiKeyId('test-key')
  ledger.recordCall(key); ledger.recordCall(key); ledger.recordCall(geminiKeyId('other'))
  ledger.recordLimit(key, quota429)
  const facts = ledger.facts(key)
  expect(facts[0]).toEqual({ label: 'Requests today (Xerxes)', value: '2 since midnight Pacific' })
  expect(facts[1]?.value).toContain('Daily limit (gemini-2.5-flash, limit 250)')
  expect(facts[1]?.value).toContain('resets at midnight Pacific')
  expect(facts.at(-1)).toEqual({ label: 'Your limits', value: GEMINI_LIMITS_URL })
  // The next Pacific day: the count resets and yesterday's limit is no longer shown.
  now += 24 * 3_600_000
  expect(ledger.facts(key).map(fact => fact.label)).toEqual(['Requests today (Xerxes)', 'Your limits'])
  expect(ledger.facts(key)[0]?.value).toBe('0 since midnight Pacific')
})

test('the usage report shows a Gemini key from the ledger instead of "unsupported"', async () => {
  const ledger = new GeminiUsageLedger()
  ledger.recordCall(geminiKeyId('k'))
  const [entry] = await buildUsageReport([{ name: 'gemini-Work', provider: 'gemini', api_key: 'k', base_url: '', model: 'gemini-2.5-flash' }], { geminiLedger: ledger })
  expect(entry?.status).toBe('ok')
  expect(entry?.facts?.[0]?.value).toBe('1 since midnight Pacific')
})
