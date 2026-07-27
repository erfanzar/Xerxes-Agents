// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  MAIN_QUERY_SOURCE,
  OpenAiCompatibleClient,
  QUERY_SOURCES,
  ResponsesApiClient,
  isHousekeepingQuerySource,
  isQuerySource,
  type CompletionRequest,
} from '../src/llms/client.js'

function jsonResponse(body: Record<string, unknown>): Response {
  return new Response(JSON.stringify(body), { headers: { 'Content-Type': 'application/json' } })
}

function taggedRequest(): CompletionRequest {
  return {
    model: 'gpt-4o',
    messages: [{ role: 'user', content: 'hi' }],
    querySource: 'compaction',
  }
}

test('query sources classify every call and narrow untrusted values', () => {
  expect(isQuerySource('compaction')).toBe(true)
  expect(isQuerySource('MAIN')).toBe(false)
  expect(isQuerySource(undefined)).toBe(false)
  expect(isQuerySource({ toString: () => 'main' })).toBe(false)

  // Only the user-facing loop is billed as the user's own work; every other
  // source is housekeeping, including any added after this test was written.
  expect(isHousekeepingQuerySource(MAIN_QUERY_SOURCE)).toBe(false)
  const housekeeping = QUERY_SOURCES.filter(source => source !== MAIN_QUERY_SOURCE)
  expect(housekeeping.length).toBe(QUERY_SOURCES.length - 1)
  expect(housekeeping.every(isHousekeepingQuerySource)).toBe(true)
  expect(QUERY_SOURCES).toContain('session_title')
  expect(QUERY_SOURCES).toContain('memory_extraction')
  expect(QUERY_SOURCES).toContain('tool_result_summary')
  expect(QUERY_SOURCES).toContain('speculation')
})

test('querySource stays local: chat-completions payloads never carry it on the wire', async () => {
  let sentBody: Record<string, unknown> | undefined
  const client = new OpenAiCompatibleClient({
    providerName: 'openai',
    apiKey: 'test-key',
    baseUrl: 'https://api.openai.com/v1',
    fetchImplementation: async (_input, init) => {
      sentBody = JSON.parse(String(init?.body)) as Record<string, unknown>
      return jsonResponse({ choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }] })
    },
  })

  const completion = await client.complete(taggedRequest())

  expect(completion.content).toBe('ok')
  // Providers that reject unknown body fields would 400 on a leaked annotation.
  expect(sentBody).not.toHaveProperty('querySource')
  expect(sentBody).not.toHaveProperty('query_source')
  expect(sentBody).toMatchObject({ model: 'gpt-4o', stream: false })
})

test('querySource stays local: Responses payloads never carry it on the wire', async () => {
  let sentBody: Record<string, unknown> | undefined
  const client = new ResponsesApiClient({
    providerName: 'openai',
    apiKey: 'test-key',
    baseUrl: 'https://api.openai.com/v1',
    fetchImplementation: async (_input, init) => {
      sentBody = JSON.parse(String(init?.body)) as Record<string, unknown>
      return jsonResponse({
        output: [{ type: 'message', content: [{ type: 'output_text', text: 'ok' }] }],
        status: 'completed',
      })
    },
  })

  const completion = await client.complete(taggedRequest())

  expect(completion.content).toBe('ok')
  expect(sentBody).not.toHaveProperty('querySource')
  expect(sentBody).not.toHaveProperty('query_source')
})
