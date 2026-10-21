// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { costLabel, formatUsageText, parseUsageReport, sessionUsageRows } from '../src/auth/usageView.js'
import { providerBilledBySubscription } from '../src/llms/providerRegistry.js'

test('subscription providers are recognised; per-token providers and API-key Anthropic are not', () => {
  expect(providerBilledBySubscription('claude-code')).toBe(true)
  expect(providerBilledBySubscription('openai-codex')).toBe(true)
  expect(providerBilledBySubscription('github-copilot')).toBe(true)
  // Anthropic without a key signs in with a Claude Pro/Max session.
  expect(providerBilledBySubscription('anthropic')).toBe(true)
  expect(providerBilledBySubscription('anthropic', { apiKey: 'sk-ant-test' })).toBe(false)
  expect(providerBilledBySubscription('openai')).toBe(false)
  expect(providerBilledBySubscription('openrouter')).toBe(false)
  expect(providerBilledBySubscription('not-a-provider')).toBe(false)
  expect(providerBilledBySubscription(undefined)).toBe(false)
})

test('one cost label: included in plan, a published estimate, or unknown — never an invented $0', () => {
  expect(costLabel(1.5, 'subscription')).toBe('Included in plan')
  expect(costLabel(undefined, 'subscription')).toBe('Included in plan')
  expect(costLabel(undefined, undefined)).toBe('price unknown')
  expect(costLabel(0.42, null)).not.toContain('plan')
})

test('session rows and the plain-text report say included in plan for a subscription session', () => {
  const report = parseUsageReport({
    session: { model: 'opus', turn_count: 1, input: 10, output: 5, cost_usd: 2, cost_basis: 'subscription' },
    profiles: [],
    models: [{ model: 'opus', calls: 1, input_tokens: 10, output_tokens: 5, cost_usd: 2, cost_basis: 'subscription' }],
  })
  expect(report.session?.costBasis).toBe('subscription')
  expect(sessionUsageRows(report.session!)).toContainEqual(['Cost', 'Included in plan'])
  const text = formatUsageText(report)
  expect(text).toContain('Included in plan')
  expect(text).not.toContain('$2')
  // Without a basis, the published-price estimate is shown as before.
  const perToken = parseUsageReport({ session: { model: 'gpt-4o', turn_count: 1, input: 10, output: 5, cost_usd: 0.42 }, profiles: [] })
  expect(sessionUsageRows(perToken.session!)).toContainEqual(['Cost', '$0.42'])
})
