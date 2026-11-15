// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import type { UsageProfileView } from '../src/auth/usageView.js'
import { groupUsageProfiles, usageGroupOf } from '../src/desktop/renderer/UsagePanel.js'

const row = (profile: string, provider: string, used: readonly number[], active = false): UsageProfileView => ({
  profile, label: profile, provider, model: 'm', active, status: 'ok',
  windows: used.map((usedPercent, index) => ({ label: `w${index}`, usedPercent })),
})

test('usage profiles group by family in a fixed order, empty groups dropped', () => {
  expect(usageGroupOf('anthropic')).toBe('claude')
  expect(usageGroupOf('claude-code')).toBe('claude')
  expect(usageGroupOf('openai-codex')).toBe('codex')
  expect(usageGroupOf('gemini')).toBe('gemini')
  expect(usageGroupOf('kimi-code')).toBe('other')
  const groups = groupUsageProfiles([
    row('kimi', 'kimi-code', []), row('codex-Omid', 'openai-codex', [4]), row('claude-Prism', 'claude-code', [100, 11]),
    row('claude-Personal', 'claude-code', [15, 44]), row('Claude Code', 'claude-code', [], true),
  ])
  expect(groups.map(group => group.kind)).toEqual(['claude', 'codex', 'other'])
  // Active first, then the most headroom (lowest peak window), unknown usage last.
  expect(groups[0]!.profiles.map(profile => profile.profile)).toEqual(['Claude Code', 'claude-Personal', 'claude-Prism'])
})
