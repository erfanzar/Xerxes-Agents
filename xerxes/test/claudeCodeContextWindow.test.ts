// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from 'bun:test'

import { resolvedProfileModelCapabilities, type ProviderProfile } from '../src/bridge/profiles.js'
import { ClaudeCodeClient, claudeCodeModelLimits, type ClaudeCodeLauncher } from '../src/llms/claudeCode.js'
import { clearReportedCapabilities } from '../src/llms/modelsDev.js'

/**
 * Claude Code states each model's window on every result
 * (`modelUsage[model].contextWindow`). A plain alias such as `opus` has no
 * other source, and without a known window nothing compacts ahead of time:
 * the first sign was a "Prompt is too long" stop.
 */

afterEach(() => clearReportedCapabilities())

const usage = (entries: Record<string, Record<string, number>>) => entries

test('the window comes from the model that read the request, not a side call on a smaller model', () => {
  expect(claudeCodeModelLimits(usage({
    'claude-haiku-4-5': { inputTokens: 40, cacheReadInputTokens: 0, contextWindow: 200_000, maxOutputTokens: 32_000 },
    'claude-opus-5-5': { inputTokens: 900, cacheReadInputTokens: 120_000, cacheCreationInputTokens: 3_000, contextWindow: 1_000_000, maxOutputTokens: 64_000 },
  }))).toEqual({ contextLimit: 1_000_000, maxOutputTokens: 64_000 })
})

test('missing or malformed usage reports nothing rather than a guess', () => {
  expect(claudeCodeModelLimits(undefined)).toBeUndefined()
  expect(claudeCodeModelLimits('200000')).toBeUndefined()
  expect(claudeCodeModelLimits({ 'claude-opus-5-5': { inputTokens: 10 } })).toBeUndefined()
  expect(claudeCodeModelLimits({ 'claude-opus-5-5': { inputTokens: 10, contextWindow: -1 } })).toBeUndefined()
})

test('after one reply, a Claude Code profile knows the window of its plain alias and compaction can plan for it', async () => {
  const profile = { name: 'Claude Code', provider: 'claude-code', base_url: 'claude-code://', api_key: '', model: 'claude-code/opus' } as unknown as ProviderProfile
  expect(resolvedProfileModelCapabilities(profile, 'claude-code/opus').contextSource).toBe('unknown')

  const lines = [
    JSON.stringify({ type: 'stream_event', event: { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: 'ok' } } }),
    JSON.stringify({
      type: 'result', subtype: 'success', is_error: false,
      usage: { input_tokens: 10, output_tokens: 2 },
      modelUsage: { 'claude-opus-5-5': { inputTokens: 10, outputTokens: 2, cacheReadInputTokens: 30_000, contextWindow: 200_000, maxOutputTokens: 32_000 } },
    }),
  ]
  const launch: ClaudeCodeLauncher = () => ({
    lines: (async function* () { for (const line of lines) yield line })(),
    exited: Promise.resolve(0),
    stderr: Promise.resolve(''),
    kill: () => {},
  })
  const client = new ClaudeCodeClient({ executable: '/bin/claude', launch, workingDirectory: '/tmp', environment: { PATH: '/bin' } })
  for await (const _ of client.stream({ model: 'claude-code/opus', messages: [{ role: 'user', content: 'hi' }] })) { /* drain */ }

  const known = resolvedProfileModelCapabilities(profile, 'claude-code/opus')
  expect(known).toMatchObject({ contextLimit: 200_000, contextSource: 'provider', maxOutputTokens: 32_000, outputSource: 'provider' })
})

test('a size the person set still wins over what Claude Code reports', async () => {
  const profile = { name: 'Claude Code', provider: 'claude-code', base_url: 'claude-code://', api_key: '', model: 'claude-code/opus',
    model_overrides: { 'claude-code/opus': { context_limit: 150_000 } } } as unknown as ProviderProfile
  const { reportModelCapability } = await import('../src/llms/modelsDev.js')
  reportModelCapability('claude-code', 'claude-code/opus', { contextLimit: 200_000 })
  expect(resolvedProfileModelCapabilities(profile, 'claude-code/opus')).toMatchObject({ contextLimit: 150_000, contextSource: 'override' })
})
