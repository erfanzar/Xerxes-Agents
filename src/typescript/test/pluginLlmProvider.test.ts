// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ConfigurationError } from '../src/core/errors.js'
import {
  PluginRegistry,
  type PluginLlmProviderFactory,
  type PluginLlmProviderRequest,
} from '../src/extensions/plugins.js'
import {
  OpenAiCompatibleClient,
  createLlmClient,
  type CompletionRequest,
  type LlmClient,
  type LlmDelta,
} from '../src/llms/client.js'

const PLUGIN_CLIENT: LlmClient = {
  async *stream(_request: CompletionRequest): AsyncGenerator<LlmDelta> {
    yield { content: 'plugin response' }
  },
}

test('createLlmClient selects a registered plugin provider and passes normalized factory input', () => {
  const registry = new PluginRegistry()
  let received: PluginLlmProviderRequest | undefined
  const fetchImplementation = async (): Promise<Response> => new Response()
  registry.registerProvider('fixture', {
    createClient(request) {
      received = request
      return PLUGIN_CLIENT
    },
  })

  const client = createLlmClient('fixture/code-model', { top_k: 12 }, {
    apiKey: 'plugin-key',
    baseUrl: 'https://provider.example/v1',
    fetchImplementation,
    pluginRegistry: registry,
    promptCaching: false,
    responsesApi: true,
  })

  expect(client).toBe(PLUGIN_CLIENT)
  if (!received) throw new Error('Plugin LLM provider factory was not called')
  expect(received).toEqual({
    model: 'code-model',
    options: {
      apiKey: 'plugin-key',
      baseUrl: 'https://provider.example/v1',
      fetchImplementation,
      promptCaching: false,
      responsesApi: true,
    },
    overrides: { top_k: 12 },
    providerName: 'fixture',
    requestedModel: 'fixture/code-model',
  })
})

test('createLlmClient selects a registered plugin provider from the provider override', () => {
  const registry = new PluginRegistry()
  registry.registerProvider('fixture', { createClient: () => PLUGIN_CLIENT })

  const client = createLlmClient('model-selected-by-profile', { provider: 'fixture' }, { pluginRegistry: registry })

  expect(client).toBe(PLUGIN_CLIENT)
})

test('unregistered and built-in provider names retain native fallback behavior', () => {
  const registry = new PluginRegistry()
  let builtInFactoryCalled = false
  registry.registerProvider('openai', {
    createClient: () => {
      builtInFactoryCalled = true
      return PLUGIN_CLIENT
    },
  })

  const unknown = createLlmClient('missing-provider/gpt-4o', {}, { pluginRegistry: registry })
  const builtIn = createLlmClient('openai/gpt-4o', {}, { pluginRegistry: registry })

  expect(unknown).toBeInstanceOf(OpenAiCompatibleClient)
  expect(builtIn).toBeInstanceOf(OpenAiCompatibleClient)
  expect(builtInFactoryCalled).toBe(false)
})

test('plugin providers reject malformed registrations and invalid LLM client results', () => {
  const registry = new PluginRegistry()
  expect(() => registry.registerProvider('malformed', {} as PluginLlmProviderFactory)).toThrow(
    "Provider 'malformed' must expose createClient(request)",
  )

  registry.registerProvider('invalid-client', {
    createClient: () => ({}) as LlmClient,
  })
  expect(() => createLlmClient('invalid-client/model', {}, { pluginRegistry: registry })).toThrow(ConfigurationError)
})
