// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { AnthropicMessagesClient } from '../src/llms/anthropic.js'
import type { CompletionRequest } from '../src/llms/client.js'
import { joinSystemSegments } from '../src/streaming/promptCaching.js'

test('Anthropic puts the cache breakpoint after the stable sources of a segmented prompt', async () => {
  const first = await capturedSystem(joinSystemSegments(daemonSegments('Recalled: prefer bun test.')))
  const second = await capturedSystem(joinSystemSegments(daemonSegments('Recalled: prefer bun test.\nAnd more.')))

  expect(first).toEqual([
    {
      type: 'text',
      text: 'Bootstrap contract.\n\nAgent instructions.\n\nWorkspace is trusted.',
      cache_control: { type: 'ephemeral' },
    },
    { type: 'text', text: '\n\nRecalled: prefer bun test.' },
  ])
  // A memory write between turns must leave the cached block untouched,
  // otherwise every substantive turn re-pays for the whole system prefix.
  expect(systemBlocks(second)[0]).toEqual(systemBlocks(first)[0])
})

test('Anthropic keeps the whole-string breakpoint for prompts assembled without segments', async () => {
  expect(await capturedSystem('Bootstrap contract.\n\nRecalled: prefer bun test.')).toEqual([{
    type: 'text',
    text: 'Bootstrap contract.\n\nRecalled: prefer bun test.',
    cache_control: { type: 'ephemeral' },
  }])
})

test('Anthropic omits cache markers entirely when prompt caching is disabled', async () => {
  const system = joinSystemSegments(daemonSegments('Recalled: prefer bun test.'))
  expect(await capturedSystem(system, { promptCaching: false })).toBe(system)
})

function systemBlocks(value: unknown): readonly unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(`expected an Anthropic system block array, saw ${JSON.stringify(value)}`)
  }
  return value as readonly unknown[]
}

/** Daemon-shaped assembly: memory is declared before the static addendum. */
function daemonSegments(memory: string) {
  return [
    { name: 'bootstrap', text: 'Bootstrap contract.' },
    { name: 'agent', text: 'Agent instructions.' },
    { name: 'memory', text: memory, volatile: true },
    { name: 'workspace-addendum', text: 'Workspace is trusted.' },
  ] as const
}

async function capturedSystem(
  systemPrompt: string,
  options: { readonly promptCaching?: boolean } = {},
): Promise<unknown> {
  let payload: Record<string, unknown> | undefined
  const client = new AnthropicMessagesClient({
    apiKey: 'test-key',
    ...(options.promptCaching === undefined ? {} : { promptCaching: options.promptCaching }),
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return sseResponse([{ type: 'message_stop' }])
    },
  })
  const request: CompletionRequest = {
    model: 'claude-sonnet-4-6',
    messages: [{ role: 'system', content: systemPrompt }, { role: 'user', content: 'Go.' }],
  }
  for await (const _event of client.stream(request)) {
    // Drain the stream so the request has certainly been issued.
  }
  return payload?.system
}

function sseResponse(events: readonly Record<string, unknown>[]): Response {
  const encoder = new TextEncoder()
  return new Response(new ReadableStream({
    start(controller) {
      for (const event of events) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`))
      }
      controller.enqueue(encoder.encode('data: [DONE]\n\n'))
      controller.close()
    },
  }), { headers: { 'Content-Type': 'text/event-stream' } })
}
