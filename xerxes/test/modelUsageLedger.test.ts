// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { formatUsageText, parseUsageReport } from '../src/auth/usageView.js'
import {
  closeModelUsageLedger,
  modelUsageLedger,
  ModelUsageLedger,
  openModelUsageLedger,
  recordModelUsage,
} from '../src/runtime/modelUsageLedger.js'

afterEach(() => closeModelUsageLedger())

test('rounds are totalled per model and profile inside the window', () => {
  const ledger = new ModelUsageLedger(':memory:')
  ledger.record({ model: 'gpt-6-sol', profile: 'codex', inputTokens: 100, outputTokens: 20, cacheReadTokens: 900, at: 1_000 })
  ledger.record({ model: 'gpt-6-sol', profile: 'codex', inputTokens: 50, outputTokens: 10, at: 2_000 })
  ledger.record({ model: 'claude-code/opus', inputTokens: 10, outputTokens: 5, at: 3_000 })
  // Too old for the window, and an empty round: neither counts.
  ledger.record({ model: 'gpt-6-sol', profile: 'codex', inputTokens: 999, outputTokens: 999, at: 10 })
  ledger.record({ model: 'gpt-6-sol', profile: 'codex', inputTokens: 0, outputTokens: 0, at: 4_000 })
  expect(ledger.totals(500)).toEqual([
    { model: 'gpt-6-sol', profile: 'codex', rounds: 2, inputTokens: 150, outputTokens: 30, cacheReadTokens: 900, cacheWriteTokens: 0, lastAt: 2_000 },
    { model: 'claude-code/opus', profile: '', rounds: 1, inputTokens: 10, outputTokens: 5, cacheReadTokens: 0, cacheWriteTokens: 0, lastAt: 3_000 },
  ])
  ledger.close()
})

test('recording is a no-op until the daemon opens the ledger, then persists to its file', async () => {
  // Tests and embedded runtimes must never write into a user's home.
  expect(modelUsageLedger()).toBeNull()
  expect(() => recordModelUsage({ model: 'm', inputTokens: 1, outputTokens: 1 })).not.toThrow()

  const directory = await mkdtemp(join(tmpdir(), 'xerxes-model-usage-'))
  try {
    const path = join(directory, 'usage', 'model-usage.sqlite')
    openModelUsageLedger(path)
    recordModelUsage({ model: 'kimi-k3', profile: 'kimi', inputTokens: 7, outputTokens: 3 })
    closeModelUsageLedger()
    const reopened = new ModelUsageLedger(path)
    expect(reopened.totals(0)).toMatchObject([{ model: 'kimi-k3', profile: 'kimi', rounds: 1, inputTokens: 7, outputTokens: 3 }])
    reopened.close()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('the report view reads provider facts and per-model spend, and prints both', () => {
  const report = parseUsageReport({
    fetched_at: 1,
    profiles: [{ profile: 'or', label: 'openrouter', provider: 'openai', model: 'x', status: 'ok', source: 'openrouter', windows: [], balance: '$28.03 left',
      facts: [{ label: 'Credit left', value: '$28.03 of $115.00' }, { label: 'Spent today', value: '$0.02' }, { label: 7 }] }],
    models_since: 0,
    models: [
      { model: 'gpt-6-sol', profile: 'codex', calls: 412, input_tokens: 1_200_000, output_tokens: 88_000, cache_read_tokens: 9_000_000, cost_usd: 3.25 },
      { model: 'mystery-model', calls: 1, input_tokens: 10, output_tokens: 2 },
      { calls: 3 },
    ],
  })
  expect(report.profiles[0]!.facts).toEqual([{ label: 'Credit left', value: '$28.03 of $115.00' }, { label: 'Spent today', value: '$0.02' }])
  expect(report.models).toEqual([
    { model: 'gpt-6-sol', profile: 'codex', calls: 412, input: 1_200_000, output: 88_000, cacheRead: 9_000_000, cacheWrite: 0, costUsd: 3.25 },
    { model: 'mystery-model', calls: 1, input: 10, output: 2, cacheRead: 0, cacheWrite: 0 },
  ])
  const text = formatUsageText(report)
  expect(text).toContain('Credit left       $28.03 of $115.00')
  expect(text).toContain('By model · last 30 days')
  expect(text).toMatch(/gpt-6-sol\s+\$3\.25\s+412 calls · 1\.2M in · 88K out · 9\.0M cached/)
  // A model nobody prices shows its tokens, never an invented $0.
  expect(text).toMatch(/mystery-model\s+price unknown/)
})
