// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { parseUsageReport } from '../../auth/usageView.js'
import { usagePanelSections } from '../domain/usagePanel.js'
import { panelLineCount } from '../lib/panelLayout.js'

const now = 1_800_000_000_000

describe('usagePanelSections', () => {
  const report = parseUsageReport({
    fetched_at: now,
    session: { model: 'gpt-5.3-codex', turn_count: 4, llm_steps: 9, tool_steps: 12, llm_duration_ms: 61_000, tool_duration_ms: 8_000, ttft_avg_ms: 820, tokens_per_second: 71.25, cache_hit_rate: 0.83, input: 48_200, output: 6_100, context_used: 44_000, context_max: 128_000, cost_usd: 0.42 },
    profiles: [
      { profile: 'codex', provider: 'openai-codex', model: 'gpt-5.3-codex', active: true, source: 'codex', status: 'ok', plan: 'pro', fetchedAt: now, windows: [{ label: '5-hour', usedPercent: 18, resetAfterSeconds: 11_520 }, { label: 'weekly', usedPercent: 31, resetsAt: now + (3 * 1_440 + 23 * 60) * 60_000 }] },
      { profile: 'zai', provider: 'zhipu', model: 'glm-4.6', active: false, source: 'zai', status: 'error', windows: [], message: 'The provider rejected this key (401).' },
      { profile: 'router', provider: 'openrouter', model: 'x', active: false, source: 'openrouter', status: 'ok', balance: '$7.50 left', windows: [{ label: 'Key limit', usedPercent: 25, detail: '$2.50 of $10.00' }] },
      { profile: 'local', provider: 'ollama', model: 'qwen', active: false, status: 'unsupported', windows: [] }
    ]
  })

  it('leads with this session, then every profile with its windows, balance or reason', () => {
    const sections = usagePanelSections(report, now)
    expect(sections[0]).toMatchObject({
      heading: { label: 'This session', notes: ['gpt-5.3-codex'], right: '$0.42', state: 'active' },
      meters: [{ label: 'Context', note: '44K / 128K' }]
    })
    expect(sections[0]!.rows).toEqual([
      ['Turns', '4'], ['Steps', '21'], ['Model time', '1m 1s'], ['Tool time', '8s'], ['First response', '820ms'],
      ['Generation', '71.3 tok/s'], ['Cache hit', '83%'], ['Tokens', '48K in · 6.1K out']
    ])
    expect(sections[1]).toEqual({ title: 'Plans & keys', count: '4' })
    expect(sections[2]).toMatchObject({
      heading: { label: 'codex', notes: ['ChatGPT · pro'], state: 'active' },
      meters: [{ label: '5-hour', percent: 18, right: 'resets in 3h 12m' }, { label: 'Weekly', percent: 31, right: 'resets in 3d 23h' }]
    })
    expect(sections[3]).toMatchObject({ heading: { label: 'zai', right: 'unavailable', state: 'failed' }, text: 'The provider rejected this key (401).' })
    expect(sections[4]).toMatchObject({ heading: { label: 'router', notes: ['OpenRouter'], right: '$7.50 left' }, meters: [{ label: 'Key limit', note: '$2.50 of $10.00' }] })
    expect(sections[5]).toMatchObject({ heading: { label: 'local', notes: ['ollama'], right: 'no limits published', state: 'idle' } })
  })

  it('says how to add a provider when none are imported', () => {
    expect(usagePanelSections(parseUsageReport({ profiles: [] }), now)).toEqual([
      { title: 'Plans & keys', text: 'No provider profiles imported yet. Add one with /provider.' }
    ])
  })

  it('reserves exactly the rows the panel paints', () => {
    const sections = usagePanelSections(report, now)
    // caption + session(heading, 8 rows, 1 meter) + gap + caption
    // + codex(heading, 2 meters) + zai(heading, text) + router(heading, 1 meter) + local(heading) + bottom margin
    expect(panelLineCount({ title: 'Usage', sections }, 120)).toBe(1 + 10 + 1 + 1 + 3 + 2 + 2 + 1 + 1)
  })
})
