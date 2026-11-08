// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { resolvedProfileModelCapabilities, type ProviderProfile } from '../src/bridge/profiles.js'
import { DEFAULT_CARD, MODEL_CARDS, modelCard } from '../src/llms/modelCards.js'
import { nativeSamplingDefaults, publishedSamplingProfile } from '../src/llms/samplingDefaults.js'
import { BUILTIN_MODELS, parseLocalConfig } from '../src/local/catalog.js'
import { defaultEscalationMode } from '../src/runtime/escalationPolicy.js'
import { automaticPromptProfile, parsePromptProfileConfig } from '../src/runtime/leanPrompt.js'
import { formatModelCardReport, modelCardReport } from '../src/runtime/modelCardReport.js'

/** Every place that used to match Bonsai names on its own, asked about one id. */
function seenEverywhere(model: string) {
  return {
    card: modelCard(model).id,
    sampling: publishedSamplingProfile(model)?.source,
    output: resolvedProfileModelCapabilities(undefined, model).maxOutputTokens,
    prompt: automaticPromptProfile({ model }),
    escalation: defaultEscalationMode({ runtime: {}, env: {}, model, local: () => false }),
    nativeDefaults: nativeSamplingDefaults(model),
  }
}

test('each Bonsai id resolves to one card, and every consumer reads that card', () => {
  const cases: Array<[string, string]> = [
    ['ternary-bonsai-2-27b-mlx', 'ternary-bonsai-2-27b'],
    ['prism-ml/Ternary-Bonsai-2-27B-mlx-2bit', 'ternary-bonsai-2-27b'],
    ['OsaurusAI/Bonsai-2-27B-Ternary-JANG', 'ternary-bonsai-2-27b'],
    ['mannix/bonsai-2-27b:latest', 'ternary-bonsai-2-27b'],
    ['bonsai-9b', 'bonsai-9b'],
    ['local/bonsai-9b', 'bonsai-9b'],
    ['prism-ml/Bonsai-2-9B-gguf', 'bonsai-9b'],
    ['bonsai-2-4b', 'bonsai-2-4b'],
    ['Ternary-Bonsai-2-2B-mlx-2bit', 'bonsai-2-2b'],
    ['bonsai-8b', 'bonsai'],
    ['ternary-bonsai-27b-mlx', 'bonsai'],
  ]
  for (const [model, id] of cases) {
    const card = modelCard(model)
    expect(card.id).toBe(id)
    expect(seenEverywhere(model)).toEqual({
      card: id,
      sampling: card.sampling?.source,
      output: card.maxOutputTokens,
      prompt: 'lean',
      escalation: 'on',
      nativeDefaults: card.sampling ? {} : { temperature: 0.6, top_k: 64 },
    })
  }
})

test('a model without a card is unchanged: no sampling, no output cap, the window decides the prompt', () => {
  for (const model of ['qwen3-27b', 'gpt-4o-mini', 'claude-sonnet-5-5', 'llama-3.1-8b']) {
    expect(modelCard(model)).toBe(DEFAULT_CARD)
    expect(seenEverywhere(model)).toEqual({
      card: 'default',
      sampling: undefined,
      output: undefined,
      prompt: 'full',
      escalation: 'off',
      nativeDefaults: { temperature: 0.6, top_k: 64 },
    })
    expect(automaticPromptProfile({ model, contextLimit: 16_384 })).toBe('lean')
  }
  expect(modelCard(undefined)).toBe(DEFAULT_CARD)
  expect(modelCard('')).toBe(DEFAULT_CARD)
})

test('the most specific card wins: the 27B is never read as the 2B or the family', () => {
  expect(modelCard('bonsai-2-27b').id).toBe('ternary-bonsai-2-27b')
  expect(modelCard('bonsai-2-2b').id).toBe('bonsai-2-2b')
  // Only the last card is a catch-all.
  expect(MODEL_CARDS.filter(card => card.match?.test('bonsai-anything')).map(card => card.id)).toEqual(['bonsai'])
})

test('the local catalog is built from the cards that have a llama-server build', () => {
  expect(BUILTIN_MODELS.map(entry => entry.id)).toEqual(['bonsai-9b', 'ternary-bonsai-2-27b'])
  const big = BUILTIN_MODELS.find(entry => entry.id === 'ternary-bonsai-2-27b')
  const card = modelCard('ternary-bonsai-2-27b')
  expect(big).toMatchObject({
    label: card.label,
    fileName: card.llamaServer?.fileName,
    minMemoryBytes: card.llamaServer?.minMemoryBytes,
    ctxSize: card.llamaServer?.ctxSize,
    nativeContext: card.nativeContext,
    lmstudio: card.lmstudio,
  })
  // A configured download source still has to name a card with a build.
  expect(() => parseLocalConfig({ local: { models: { 'bonsai-2-4b': { url: 'https://example.com/a.gguf', sha256: 'a'.repeat(64) } } } }, {}))
    .toThrow(/unknown model id \(known: bonsai-9b, ternary-bonsai-2-27b\)/)
})

test('settings still win over the card', () => {
  const profile = {
    name: 'bonsai-local',
    provider: 'custom',
    base_url: 'http://127.0.0.1:1234/v1',
    model: 'ternary-bonsai-2-27b-mlx',
    model_overrides: { 'ternary-bonsai-2-27b-mlx': { max_output_tokens: 8_192 } },
  } as unknown as ProviderProfile
  expect(resolvedProfileModelCapabilities(profile, 'ternary-bonsai-2-27b-mlx').maxOutputTokens).toBe(8_192)
  expect(parsePromptProfileConfig({ prompt_profile: 'full' }, {}, undefined, { model: 'ternary-bonsai-2-27b-mlx' })).toBe('full')
  expect(defaultEscalationMode({ runtime: {}, env: {}, model: 'gpt-4o-mini', local: () => true })).toBe('on')
})

test('the card report names what set each value', () => {
  const profile = {
    name: 'bonsai-local',
    provider: 'custom',
    base_url: 'http://127.0.0.1:1234/v1',
    model: 'ternary-bonsai-2-27b-mlx',
    model_overrides: { 'ternary-bonsai-2-27b-mlx': { max_output_tokens: 16_384 } },
  } as unknown as ProviderProfile
  const report = modelCardReport({
    model: 'ternary-bonsai-2-27b-mlx',
    runtime: {},
    env: { XERXES_PROMPT_PROFILE: 'full', XERXES_ESCALATION: 'off' },
    profile,
    loadedWindow: 42_752,
  })
  const byName = Object.fromEntries(report.settings.map(setting => [setting.name, setting]))
  expect(byName['Sampling']).toMatchObject({ effective: 'card', from: "Prism's model card" })
  expect(byName['Output limit']).toMatchObject({ card: '32,768 tokens', effective: '16,384 tokens', from: "profile's max_output_tokens" })
  expect(byName['Context window']).toMatchObject({ card: 'reads 262,144 tokens', effective: '42,752 tokens', from: 'loaded by the engine' })
  expect(byName['Prompt and tools']).toMatchObject({ card: 'lean', effective: 'full', from: 'XERXES_PROMPT_PROFILE' })
  expect(byName['Result checks and escalation']).toMatchObject({ card: 'on', effective: 'off', from: 'XERXES_ESCALATION' })
  expect(byName['Local builds']?.card).toContain('LM Studio ternary-bonsai-2-27b-mlx (needs 16 GiB)')
})

test('a model without a card reports Xerxes defaults, not the values the daemon materialized', () => {
  const report = modelCardReport({ model: 'gpt-4o-mini', runtime: {}, env: {}, samplingSet: { temperature: 0.6, top_k: 64 } })
  expect(formatModelCardReport(report)).toStartWith("No model card for `gpt-4o-mini`: Xerxes's general defaults apply.")
  expect(report.settings.find(setting => setting.name === 'Sampling')).toMatchObject({
    card: '—', effective: 'temperature 0.6, top_k 64', from: 'Xerxes default',
  })
  expect(report.settings.some(setting => setting.name === 'Local builds')).toBe(false)
})

test('xerxes bonsai prints the card for a named model without a daemon', async () => {
  const home = await mkdtemp(join(tmpdir(), 'xerxes-bonsai-cli-'))
  try {
    const child = Bun.spawn([process.execPath, join(import.meta.dir, '../src/cli.ts'), 'bonsai', 'bonsai-2-4b'], {
      env: { ...process.env, XERXES_HOME: home, NO_COLOR: '1' },
      stdout: 'pipe',
      stderr: 'pipe',
    })
    const [stdout, code] = await Promise.all([new Response(child.stdout).text(), child.exited])
    expect(code).toBe(0)
    expect(stdout).toContain('Model card for `bonsai-2-4b`: Bonsai 2 4B (`bonsai-2-4b`)')
    expect(stdout).toContain("| Sampling | temperature 1, top_p 0.95, top_k 20, min_p 0, presence_penalty 1")
    const usage = Bun.spawn([process.execPath, join(import.meta.dir, '../src/cli.ts'), 'bonsai', 'a', 'b'], {
      env: { ...process.env, XERXES_HOME: home }, stdout: 'pipe', stderr: 'pipe',
    })
    const [usageError, usageCode] = await Promise.all([new Response(usage.stderr).text(), usage.exited])
    expect(usageCode).toBe(2)
    expect(usageError).toContain('Usage: xerxes bonsai [model]')
  } finally {
    await rm(home, { recursive: true, force: true })
  }
}, 30_000)
