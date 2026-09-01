// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import type { CompletionRequest } from '../src/llms/client.js'
import { GeminiClient } from '../src/llms/gemini.js'

test('Gemini maps a neutral thinking budget to native thinkingConfig for completion', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new GeminiClient({
    apiKey: 'test-key',
    baseUrl: 'https://gemini.test/v1beta',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return Response.json({
        candidates: [{ content: { parts: [{ text: 'answer' }] }, finishReason: 'STOP' }],
      })
    },
  })

  await client.complete({
    ...simpleRequest(),
    thinking: { budgetTokens: 4_000, effort: 'medium' },
  })

  expect(payload).toEqual({
    contents: [{ role: 'user', parts: [{ text: 'hello' }] }],
    generationConfig: {
      thinkingConfig: {
        thinkingBudget: 4_000,
        includeThoughts: true,
      },
    },
  })
})

test('Gemini gives an effort-only neutral thinking request the project default budget', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new GeminiClient({
    apiKey: 'test-key',
    baseUrl: 'https://gemini.test/v1beta',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return sseResponse({
        candidates: [{ content: { parts: [{ text: 'answer' }] }, finishReason: 'STOP' }],
      })
    },
  })

  for await (const _event of client.stream({ ...simpleRequest(), thinking: { effort: 'high' } })) {
    // Drain the response so the request and stream lifecycle are both exercised.
  }

  expect(payload).toEqual({
    contents: [{ role: 'user', parts: [{ text: 'hello' }] }],
    generationConfig: {
      thinkingConfig: {
        thinkingBudget: 10_000,
        includeThoughts: true,
      },
    },
  })
})

test('Gemini 3 uses thinkingLevel instead of the legacy token budget', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new GeminiClient({
    apiKey: 'test-key',
    baseUrl: 'https://gemini.test/v1beta',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return Response.json({ candidates: [{ content: { parts: [{ text: 'answer' }] }, finishReason: 'STOP' }] })
    },
  })

  await client.complete({ ...simpleRequest(), model: 'gemini-3-pro', thinking: { effort: 'medium', budgetTokens: 4_000 } })

  expect(payload).toEqual({
    contents: [{ role: 'user', parts: [{ text: 'hello' }] }],
    generationConfig: { thinkingConfig: { thinkingLevel: 'MEDIUM', includeThoughts: true } },
  })
})

test('Gemini omits thinkingConfig when neutral thinking is not requested', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new GeminiClient({
    apiKey: 'test-key',
    baseUrl: 'https://gemini.test/v1beta',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return Response.json({
        candidates: [{ content: { parts: [{ text: 'answer' }] }, finishReason: 'STOP' }],
      })
    },
  })

  await client.complete(simpleRequest())

  expect(payload).toEqual({ contents: [{ role: 'user', parts: [{ text: 'hello' }] }] })
})

test('Gemini legacy family maps an explicit off directive to a zero thinking budget', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new GeminiClient({
    apiKey: 'test-key',
    baseUrl: 'https://gemini.test/v1beta',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return Response.json({ candidates: [{ content: { parts: [{ text: 'answer' }] }, finishReason: 'STOP' }] })
    },
  })

  await client.complete({ ...simpleRequest(), thinking: { effort: 'off' } })

  expect(payload).toEqual({
    contents: [{ role: 'user', parts: [{ text: 'hello' }] }],
    generationConfig: { thinkingConfig: { thinkingBudget: 0, includeThoughts: false } },
  })
})

test('Gemini blocked finish reasons fail the turn instead of returning an empty success', async () => {
  const client = new GeminiClient({
    apiKey: 'test-key',
    baseUrl: 'https://gemini.test/v1beta',
    fetchImplementation: async () => Response.json({
      candidates: [{ content: { parts: [] }, finishReason: 'SAFETY' }],
    }),
  })

  await expect(client.complete(simpleRequest())).rejects.toThrow('provider stopped with: SAFETY')
})

function simpleRequest(): CompletionRequest {
  return { model: 'gemini-2.5-flash', messages: [{ role: 'user', content: 'hello' }] }
}

function sseResponse(event: Record<string, unknown>): Response {
  return new Response(`data: ${JSON.stringify(event)}\n\ndata: [DONE]\n\n`, {
    headers: { 'Content-Type': 'text/event-stream' },
  })
}
