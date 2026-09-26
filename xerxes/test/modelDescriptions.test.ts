// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { describedModel } from '../src/daemon/server.js'
import { ClaudeCodeCatalog } from '../src/llms/claudeCodeCatalog.js'
import { ModelsDevCatalog } from '../src/llms/modelsDev.js'

// A slice of models.dev: the vendor lists the model, and so do resellers
// with other limits (which made an id-only match ambiguous).
const modelsDev = new ModelsDevCatalog()
modelsDev.seed({
  openai: { id: 'openai', api: 'https://api.openai.com/v1', models: { 'gpt-6-sol': { id: 'gpt-6-sol', name: 'GPT-6 Sol', description: 'Frontier reasoning', limit: { context: 1050000, output: 128000 } } } },
  reseller: { id: 'reseller', api: 'https://reseller.example/v1', models: { 'gpt-6-sol': { id: 'gpt-6-sol', name: 'GPT-6 Sol (resold)', limit: { context: 400000, output: 64000 } } } },
  anthropic: { id: 'anthropic', api: 'https://api.anthropic.com', models: { 'claude-opus-5-5': { id: 'claude-opus-5-5', name: 'Claude Opus 5.5', description: 'Most capable Claude', limit: { context: 1000000, output: 128000 } } } },
  zai: { id: 'zai', api: 'https://api.z.ai/api/coding/paas/v4', models: { 'glm-5.3': { id: 'glm-5.3', name: 'GLM-5.3', description: 'Zhipu flagship', limit: { context: 1000000, output: 128000 } } } },
})

test('a Codex model is described by the vendor models.dev lists it under, like GLM is by its own provider', () => {
  expect(describedModel({ provider: 'openai-codex', base_url: 'https://chatgpt.com/backend-api/codex' }, 'gpt-6-sol', { modelsDev, claudeCode: new ClaudeCodeCatalog(async () => []) }))
    .toEqual({ displayName: 'GPT-6 Sol', description: 'Frontier reasoning' })
  expect(describedModel({ provider: 'zhipu', base_url: 'https://api.z.ai/api/coding/paas/v4' }, 'glm-5.3', { modelsDev, claudeCode: new ClaudeCodeCatalog(async () => []) }))
    .toEqual({ displayName: 'GLM-5.3', description: 'Zhipu flagship' })
})

test('a Claude Code alias takes the CLI\'s own name, and models.dev is asked about the model it resolves to', async () => {
  const claudeCode = new ClaudeCodeCatalog(async () => [
    { value: 'opus[1m]', resolvedModel: 'claude-opus-5-5[1m]', displayName: 'Opus (1M context)', description: 'Opus 5.5 for complex work', effortLevels: [], adaptiveThinking: true, contextLimit: 1_000_000 },
    { value: 'sonnet', resolvedModel: 'claude-opus-5-5', displayName: 'Sonnet', effortLevels: [], adaptiveThinking: false },
  ])
  await claudeCode.load()
  expect(describedModel({ provider: 'claude-code', base_url: 'claude-code://local' }, 'claude-code/opus[1m]', { modelsDev, claudeCode }))
    .toEqual({ displayName: 'Opus (1M context)', description: 'Opus 5.5 for complex work' })
  // No CLI description: the resolved model's models.dev description fills in.
  expect(describedModel({ provider: 'claude-code', base_url: 'claude-code://local' }, 'claude-code/sonnet', { modelsDev, claudeCode }))
    .toEqual({ displayName: 'Sonnet', description: 'Most capable Claude' })
})
