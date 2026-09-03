// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import {
  fetchClaudeUsage,
  fetchCodexUsage,
  fetchKimiUsage,
  fetchZaiUsage,
  formatUsageReport,
  type UsageFetch,
} from '../src/auth/usage.js'

const jsonFetch = (body: unknown, capture?: { url?: string; headers?: unknown }): UsageFetch =>
  async (url, init) => {
    if (capture) {
      capture.url = url
      capture.headers = init?.headers
    }
    return new Response(JSON.stringify(body), { status: 200 })
  }

describe('subscription usage fetchers', () => {
  test('claude maps five-hour and weekly windows with reset timestamps', async () => {
    const capture: { url?: string; headers?: unknown } = {}
    const report = await fetchClaudeUsage('tok', {
      fetchImplementation: jsonFetch({
        five_hour: { utilization: 42.4, resets_at: '2026-09-03T20:00:00Z' },
        seven_day: { utilization: 12 },
        seven_day_opus: { utilization: 99.6, resets_at: 1_788_480_000_000 },
      }, capture),
    })
    expect(capture.url).toContain('api/oauth/usage')
    expect(capture.headers).toMatchObject({ 'anthropic-beta': 'oauth-2025-04-20' })
    expect(report.provider).toBe('claude')
    expect(report.windows.map(w => [w.label, Math.round(w.usedPercent)])).toEqual([
      ['5-hour', 42],
      ['weekly', 12],
      ['weekly', 100],
    ])
    expect(report.windows[0]?.resetsAt).toBe(Date.parse('2026-09-03T20:00:00Z'))
  })

  test('codex labels windows by limit_window_seconds and keeps the plan', async () => {
    const report = await fetchCodexUsage({ Authorization: 'Bearer tok' }, {
      fetchImplementation: jsonFetch({
        plan_type: 'plus',
        rate_limit: {
          primary_window: { used_percent: 63, limit_window_seconds: 18_000, reset_after_seconds: 900 },
          secondary_window: { used_percent: 20, limit_window_seconds: 604_800, reset_at: 1_788_500_000 },
        },
      }),
    })
    expect(report.planType).toBe('plus')
    expect(report.windows.map(w => w.label)).toEqual(['5-hour', 'weekly'])
    expect(report.windows[0]?.resetAfterSeconds).toBe(900)
    expect(report.windows[1]?.resetsAt).toBe(1_788_500_000 * 1_000)
  })

  test('codex includes additional model limits and tolerates null windows', async () => {
    const report = await fetchCodexUsage({ Authorization: 'Bearer tok' }, {
      fetchImplementation: jsonFetch({
        plan_type: 'pro',
        rate_limit: {
          allowed: true,
          primary_window: { used_percent: 9, limit_window_seconds: 604_800, reset_at: 1_788_748_099 },
          secondary_window: null,
        },
        additional_rate_limits: [{
          limit_name: 'GPT-5.3-Codex-Spark',
          rate_limit: {
            primary_window: { used_percent: 0, limit_window_seconds: 18_000, reset_at: 1_788_472_032 },
            secondary_window: { used_percent: 0, limit_window_seconds: 604_800, reset_at: 1_789_058_832 },
          },
        }],
      }),
    })
    expect(report.windows.map(w => [w.label, w.detail ?? null])).toEqual([
      ['weekly', null],
      ['5-hour', 'GPT-5.3-Codex-Spark'],
      ['weekly', 'GPT-5.3-Codex-Spark'],
    ])
  })

  test('kimi accepts a scope-tagged usage list', async () => {
    const report = await fetchKimiUsage('tok', {
      fetchImplementation: jsonFetch({
        usages: [
          { scope: 'LIMIT_5H', used_percent: 30, reset_at: '2026-09-03T21:00:00Z' },
          { scope: 'LIMIT_WEEKLY', percentage: 0.5 },
        ],
      }),
    })
    expect(report.windows.map(w => [w.label, Math.round(w.usedPercent)])).toEqual([
      ['5-hour', 30],
      ['weekly', 50],
    ])
  })

  test('zai maps quota limits into 5-hour and weekly windows', async () => {
    const report = await fetchZaiUsage('key', {
      fetchImplementation: jsonFetch({
        code: 200,
        data: {
          limits: [
            { type: 'TIME_LIMIT', unit: 5, percentage: 88, nextResetTime: 1_788_480_000_000 },
            { type: 'TOKENS_LIMIT', unit: 7, percentage: 3, currentValue: 1_000, remaining: 33_000 },
          ],
        },
      }),
    })
    expect(report.windows.map(w => [w.label, Math.round(w.usedPercent)])).toEqual([
      ['5-hour', 88],
      ['weekly', 3],
    ])
    expect(report.windows[1]?.detail).toBe('33000 remaining')
  })

  test('zai swaps the host for the CN plan', async () => {
    const capture: { url?: string } = {}
    await fetchZaiUsage('key', {
      host: 'cn',
      fetchImplementation: jsonFetch({ data: { limits: [{ type: 'TIME_LIMIT', unit: 5, percentage: 1 }] } }, capture),
    })
    expect(capture.url).toContain('open.bigmodel.cn')
  })

  test('HTTP failures and unrecognized shapes raise actionable errors', async () => {
    await expect(fetchClaudeUsage('tok', {
      fetchImplementation: async () => new Response('nope', { status: 401, statusText: 'Unauthorized' }),
    })).rejects.toThrow('claude usage request failed (401)')
    await expect(fetchCodexUsage({}, {
      fetchImplementation: jsonFetch({ rate_limit: {} }),
    })).rejects.toThrow('XERXES_CODEX_USAGE_URL')
    await expect(fetchKimiUsage('tok', {
      fetchImplementation: jsonFetch({ surprise: true }),
    })).rejects.toThrow('XERXES_KIMI_USAGE_URL')
  })

  test('endpoint overrides come from the environment', async () => {
    const capture: { url?: string } = {}
    await fetchClaudeUsage('tok', {
      environment: { XERXES_CLAUDE_USAGE_URL: 'https://proxy.test/usage' },
      fetchImplementation: jsonFetch({ five_hour: { utilization: 1 } }, capture),
    })
    expect(capture.url).toBe('https://proxy.test/usage')
  })

  test('formatUsageReport renders a compact provider line', () => {
    const line = formatUsageReport({
      provider: 'codex',
      planType: 'pro',
      fetchedAt: 1_000,
      windows: [
        { label: '5-hour', usedPercent: 63, resetAfterSeconds: 900 },
        { label: 'weekly', usedPercent: 20 },
      ],
    }, 1_000)
    expect(line).toBe('Codex (pro) — 5-hour 63% (resets in 15m), weekly 20%')
  })
})
