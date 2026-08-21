// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ResponsesApiClient } from '../../../xerxes/src/llms/client.js'

test('non-stream Responses API maps max_output_tokens incomplete status to length', async () => {
  const client = new ResponsesApiClient({
    providerName: 'openai',
    apiKey: 'test-key',
    baseUrl: 'https://example.invalid/v1',
    fetchImplementation: async () => Response.json({
      status: 'incomplete',
      incomplete_details: { reason: 'max_output_tokens' },
      output: [{ type: 'message', content: [{ type: 'output_text', text: 'cut off' }] }],
      usage: { input_tokens: 5, output_tokens: 9 },
    }),
  })

  await expect(client.complete({
    model: 'gpt-4o',
    messages: [{ role: 'user', content: 'Continue until truncated.' }],
  })).resolves.toEqual({
    content: 'cut off',
    toolCalls: [],
    finishReason: 'length',
    usage: { inputTokens: 5, outputTokens: 9 },
  })
})
