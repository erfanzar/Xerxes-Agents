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

test.each(['SAFETY', 'RECITATION', 'BLOCKLIST', 'PROHIBITED_CONTENT', 'SPII', 'MALFORMED_FUNCTION_CALL', 'UNEXPECTED_TOOL_CALL', 'TOO_MANY_TOOL_CALLS', 'IMAGE_OTHER', 'UNRECOGNIZED_PROVIDER_FINISH'])(
  'Gemini blocked or unknown finish reason %s fails both completion and stream', async finishReason => {
    const event = { candidates: [{ content: { parts: [] }, finishReason }] }
    const complete = new GeminiClient({ apiKey: 'test-key', fetchImplementation: async () => Response.json(event) })
    await expect(complete.complete(simpleRequest())).rejects.toThrow(`provider stopped with: ${finishReason}`)
    const stream = new GeminiClient({ apiKey: 'test-key', fetchImplementation: async () => sseResponse(event) })
    await expect((async () => { for await (const _event of stream.stream(simpleRequest())) { /* drain */ } })())
      .rejects.toThrow(`provider stopped with: ${finishReason}`)
  },
)

test('Gemini output-limit finishes remain successful length deltas', async () => {
  const event = { candidates: [{ content: { parts: [{ text: 'partial' }] }, finishReason: 'MAX_TOKENS' }] }
  const client = new GeminiClient({ apiKey: 'test-key', fetchImplementation: async () => Response.json(event) })
  expect(await client.complete(simpleRequest())).toMatchObject({ content: 'partial', finishReason: 'length' })
  const streamed = new GeminiClient({ apiKey: 'test-key', fetchImplementation: async () => sseResponse(event) })
  const events = []
  for await (const delta of streamed.stream(simpleRequest())) events.push(delta)
  expect(events).toEqual([{ content: 'partial', finishReason: 'length' }])
})

test('Gemini blocked prompt is refused even if a candidate is also present', async () => {
  const client = new GeminiClient({
    apiKey: 'test-key',
    fetchImplementation: async () => Response.json({
      promptFeedback: { blockReason: 'PROHIBITED_CONTENT' },
      candidates: [{ content: { parts: [{ text: 'blocked' }] }, finishReason: 'STOP' }],
    }),
  })
  await expect(client.complete(simpleRequest())).rejects.toThrow('provider blocked prompt with: PROHIBITED_CONTENT')
})

function simpleRequest(): CompletionRequest {
  return { model: 'gemini-2.5-flash', messages: [{ role: 'user', content: 'hello' }] }
}

function sseResponse(event: Record<string, unknown>): Response {
  return new Response(`data: ${JSON.stringify(event)}\n\ndata: [DONE]\n\n`, {
    headers: { 'Content-Type': 'text/event-stream' },
  })
}
