// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * A trimmed models.dev document for tests (entries copied from the live
 * https://models.dev/api.json on 2026-09-24). Tests never reach the network,
 * so anything that reads model capabilities from models.dev seeds this.
 */

import { modelsDev } from '../../src/llms/modelsDev.js'

export const MODELS_DEV_FIXTURE = {
  openai: {
    id: 'openai',
    name: 'OpenAI',
    models: {
      'gpt-4o': { id: 'gpt-4o', name: 'GPT-4o', reasoning: false, temperature: true, limit: { context: 128000, output: 16384 }, cost: { input: 2.5, output: 10, cache_read: 1.25 } },
      'gpt-5': { id: 'gpt-5', name: 'GPT-5', reasoning: true, reasoning_options: [{ type: 'effort', values: ['minimal', 'low', 'medium', 'high'] }], temperature: false, limit: { context: 400000, input: 272000, output: 128000 }, cost: { input: 1.25, output: 10, cache_read: 0.125 } },
      'gpt-5.1': { id: 'gpt-5.1', name: 'GPT-5.1', reasoning: true, reasoning_options: [{ type: 'effort', values: ['none', 'low', 'medium', 'high'] }], temperature: false, limit: { context: 400000, input: 272000, output: 128000 }, cost: { input: 1.25, output: 10, cache_read: 0.125 } },
      // A model name the tests use for "a current reasoning model".
      'gpt-6-astra': { id: 'gpt-6-astra', name: 'GPT-6 Astra', reasoning: true, reasoning_options: [{ type: 'effort', values: ['none', 'low', 'medium', 'high', 'xhigh'] }], temperature: false, limit: { context: 1000000, output: 128000 }, cost: { input: 2, output: 16 } },
    },
  },
  anthropic: {
    id: 'anthropic',
    name: 'Anthropic',
    models: {
      'claude-sonnet-4-5': { id: 'claude-sonnet-4-5', name: 'Claude Sonnet 4.5', reasoning: true, reasoning_options: [{ type: 'budget_tokens', min: 1024 }], temperature: true, limit: { context: 1000000, output: 64000 }, cost: { input: 3, output: 15, cache_read: 0.3, cache_write: 3.75 } },
      'claude-haiku-4-5': { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5', reasoning: true, reasoning_options: [{ type: 'budget_tokens', min: 1024 }], temperature: true, limit: { context: 200000, output: 64000 }, cost: { input: 1, output: 5, cache_read: 0.1, cache_write: 1.25 } },
    },
  },
  'amazon-bedrock': {
    id: 'amazon-bedrock',
    name: 'Amazon Bedrock',
    models: {
      'anthropic.claude-opus-4-1-20250805-v1:0': { id: 'anthropic.claude-opus-4-1-20250805-v1:0', name: 'Claude Opus 4.1', reasoning: true, reasoning_options: [{ type: 'budget_tokens', min: 1024 }], temperature: true, limit: { context: 200000, output: 32000 }, cost: { input: 15, output: 75, cache_read: 1.5, cache_write: 18.75 } },
      // Effort levels and no token budget: adaptive thinking.
      'anthropic.claude-opus-4-6-v1': { id: 'anthropic.claude-opus-4-6-v1', name: 'Claude Opus 4.6', reasoning: true, reasoning_options: [{ type: 'effort', values: ['low', 'medium', 'high', 'max'] }], temperature: true, limit: { context: 1000000, output: 128000 }, cost: { input: 5, output: 25, cache_read: 0.5, cache_write: 6.25 } },
      'amazon.nova-lite-v1:0': { id: 'amazon.nova-lite-v1:0', name: 'Nova Lite', reasoning: false, temperature: true, limit: { context: 300000, output: 8192 }, cost: { input: 0.06, output: 0.24 } },
    },
  },
} as const

/** Load the fixture into the shared catalog (what a live fetch would have done). */
export function seedModelsDev(document: unknown = MODELS_DEV_FIXTURE): void {
  modelsDev.seed(document)
}

/** Forget it again, so a test that expects "unknown" is not polluted. */
export function clearModelsDev(): void {
  modelsDev.seed({})
}
