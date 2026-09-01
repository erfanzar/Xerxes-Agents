// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { AnthropicMessagesClient } from '../src/llms/anthropic.js'
import { OpenAiCompatibleClient, ResponsesApiClient } from '../src/llms/client.js'
import {
  anthropicSupportsToolReferences,
  completionsDeferredToolsMode,
  responsesDeferredToolsMode,
  splitDeferredTools,
} from '../src/llms/deferredTools.js'
import type { ChatMessage } from '../src/types/messages.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'

const SEARCH: ToolDefinition = {
  type: 'function',
  function: { name: 'ToolSearchTool', description: 'Search tools.', parameters: { type: 'object', properties: {} } },
}
const DEFERRED: ToolDefinition = {
  type: 'function',
  function: {
    name: 'deploy_service',
    description: 'Deploy a service.',
    parameters: { type: 'object', properties: { name: { type: 'string' } }, required: ['name'] },
  },
}

const HISTORY_WITH_LOAD: ChatMessage[] = [
  { role: 'user', content: 'deploy it' },
  {
    role: 'assistant',
    content: '',
    tool_calls: [{
      id: 'call_search',
      type: 'function',
      function: { name: 'ToolSearchTool', arguments: { query: 'deploy' } },
    }],
  },
  {
    role: 'tool',
    tool_call_id: 'call_search',
    content: '{"loaded_tool":"deploy_service"}',
    added_tool_names: ['deploy_service'],
  },
]

test('splitDeferredTools keeps used tools immediate and added-but-unused ones deferred', () => {
  const split = splitDeferredTools([SEARCH, DEFERRED], HISTORY_WITH_LOAD, true)
  expect(split.immediate.map(tool => tool.function.name)).toEqual(['ToolSearchTool'])
  expect([...split.deferred.keys()]).toEqual(['deploy_service'])

  // A tool whose load point precedes its first use stays deferred: its schema
  // rides the load item (tool_reference / additional_tools / tool-search pair),
  // so marking it again is the correct replay (pi-ai splitDeferredTools).
  const loadedThenUsed = splitDeferredTools([SEARCH, DEFERRED], [
    ...HISTORY_WITH_LOAD,
    {
      role: 'assistant',
      content: '',
      tool_calls: [{
        id: 'call_deploy',
        type: 'function',
        function: { name: 'deploy_service', arguments: { name: 'api' } },
      }],
    },
  ], true)
  expect([...loadedThenUsed.deferred.keys()]).toEqual(['deploy_service'])

  // A load marker for a tool the model had already used is stale; it stays
  // immediate rather than being re-anchored.
  const staleMarker = splitDeferredTools([SEARCH, DEFERRED], [
    {
      role: 'assistant',
      content: '',
      tool_calls: [{
        id: 'call_deploy',
        type: 'function',
        function: { name: 'deploy_service', arguments: { name: 'api' } },
      }],
    },
    ...HISTORY_WITH_LOAD,
  ], true)
  expect(staleMarker.deferred.size).toBe(0)

  const disabled = splitDeferredTools([SEARCH, DEFERRED], HISTORY_WITH_LOAD, false)
  expect(disabled.immediate.length).toBe(2)
  expect(disabled.deferred.size).toBe(0)
})

test('deferred mode resolution follows the pi-ai catalog and first-party defaults', () => {
  expect(responsesDeferredToolsMode('openai', 'openai/gpt-5.5')).toBe('additional-tools')
  expect(responsesDeferredToolsMode('openai-codex', 'codex/gpt-5.5')).toBe('tool-search')
  expect(responsesDeferredToolsMode('openai', 'openai/gpt-4o')).toBeUndefined()
  expect(completionsDeferredToolsMode('kimi', 'kimi/kimi-k3')).toBe('kimi')
  expect(completionsDeferredToolsMode('openai', 'openai/gpt-4o')).toBeUndefined()
  expect(anthropicSupportsToolReferences('anthropic/claude-sonnet-4-6', 'anthropic')).toBe(true)
  expect(anthropicSupportsToolReferences('anthropic/claude-haiku-4-5-20251001', 'anthropic')).toBe(false)
  expect(anthropicSupportsToolReferences('anthropic/claude-3-5-sonnet', 'anthropic')).toBe(false)
})

test('anthropic tool-reference mode defers loaded tools and displaces result content to siblings', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new AnthropicMessagesClient({
    apiKey: 'k',
    baseUrl: 'https://example.invalid',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return new Response(JSON.stringify({
        content: [{ type: 'text', text: 'done' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 5, output_tokens: 2 },
      }))
    },
  })

  await client.complete({
    model: 'anthropic/claude-sonnet-4-6',
    messages: HISTORY_WITH_LOAD,
    tools: [SEARCH, DEFERRED],
  })

  const tools = payload?.tools as Record<string, unknown>[]
  expect(tools.map(tool => tool.name)).toEqual(['ToolSearchTool', 'deploy_service'])
  expect(tools[0]?.defer_loading).toBeUndefined()
  expect(tools[1]?.defer_loading).toBe(true)

  const messages = payload?.messages as Record<string, unknown>[]
  const resultMessage = messages.at(-1)
  expect(resultMessage?.role).toBe('user')
  const blocks = resultMessage?.content as Record<string, unknown>[]
  // Reference replaces the tool_result content; the search text is displaced
  // to a sibling text block after it.
  expect(blocks[0]).toMatchObject({
    type: 'tool_result',
    tool_use_id: 'call_search',
    content: [{ type: 'tool_reference', tool_name: 'deploy_service' }],
  })
  expect(blocks[1]).toMatchObject({ type: 'text', text: '{"loaded_tool":"deploy_service"}' })
})

test('anthropic tool-reference mode promotes everything when nothing is immediate', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new AnthropicMessagesClient({
    apiKey: 'k',
    baseUrl: 'https://example.invalid',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return new Response(JSON.stringify({
        content: [{ type: 'text', text: 'done' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 5, output_tokens: 2 },
      }))
    },
  })

  await client.complete({
    model: 'anthropic/claude-sonnet-4-6',
    messages: HISTORY_WITH_LOAD,
    tools: [DEFERRED],
  })
  expect((payload?.tools as Record<string, unknown>[])[0]?.defer_loading).toBeUndefined()
})

test('responses additional-tools mode emits developer load items at the revealing result', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new ResponsesApiClient({
    providerName: 'openai',
    apiKey: 'k',
    baseUrl: 'https://example.invalid/v1',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return new Response(JSON.stringify({ status: 'completed', output: [] }))
    },
  })

  await client.complete({
    model: 'openai/gpt-5.5',
    messages: HISTORY_WITH_LOAD,
    tools: [SEARCH, DEFERRED],
  })

  // Only the immediate tool rides the top-level array.
  expect(payload?.tools).toEqual([{
    type: 'function',
    name: 'ToolSearchTool',
    description: 'Search tools.',
    parameters: { type: 'object', properties: {} },
  }])
  expect(payload?.input).toEqual([
    { role: 'user', content: 'deploy it' },
    { type: 'function_call', call_id: 'call_search', name: 'ToolSearchTool', arguments: '{"query":"deploy"}' },
    { type: 'function_call_output', call_id: 'call_search', output: '{"loaded_tool":"deploy_service"}' },
    {
      type: 'additional_tools',
      role: 'developer',
      tools: [{
        type: 'function',
        name: 'deploy_service',
        description: 'Deploy a service.',
        parameters: DEFERRED.function.parameters,
      }],
    },
  ])
})

test('responses tool-search mode replays a synthetic client-executed search pair', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new ResponsesApiClient({
    providerName: 'openai-codex',
    apiKey: 'k',
    baseUrl: 'https://example.invalid/codex',
    fetchImplementation: async (_input, init) => {
      // Codex compresses its SSE request bodies with zstd (pi-ai parity).
      const rawBody = init?.body
      payload = JSON.parse(rawBody instanceof Uint8Array
        ? new TextDecoder().decode(Bun.zstdDecompressSync(rawBody))
        : String(rawBody)) as Record<string, unknown>
      const encoder = new TextEncoder()
      return new Response(new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode('data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":2,"output_tokens":1}}}\n\n'))
          controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          controller.close()
        },
      }))
    },
  })

  const events = []
  for await (const event of client.stream({
    model: 'codex/gpt-5.5',
    messages: HISTORY_WITH_LOAD,
    tools: [SEARCH, DEFERRED],
  })) events.push(event)

  const input = payload?.input as Record<string, unknown>[]
  const searchCall = input.find(item => item.type === 'tool_search_call')
  const searchOutput = input.find(item => item.type === 'tool_search_output')
  expect(searchCall).toMatchObject({
    execution: 'client',
    status: 'completed',
    arguments: { query: 'deploy_service', limit: 1 },
  })
  expect(String(searchCall?.call_id)).toMatch(/^xerxes_tool_load_[0-9a-f]{12}$/)
  expect(searchOutput).toMatchObject({
    call_id: searchCall?.call_id,
    execution: 'client',
    status: 'completed',
  })
  expect((searchOutput?.tools as Record<string, unknown>[])[0]).toMatchObject({
    name: 'deploy_service',
    defer_loading: true,
  })
})

test('kimi deferred mode moves loaded tools into a system message after the result', async () => {
  let payload: Record<string, unknown> | undefined
  const client = new OpenAiCompatibleClient({
    providerName: 'kimi',
    apiKey: 'k',
    baseUrl: 'https://example.invalid/v1',
    fetchImplementation: async (_input, init) => {
      payload = JSON.parse(String(init?.body)) as Record<string, unknown>
      return new Response(JSON.stringify({
        choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }],
      }))
    },
  })

  await client.complete({
    model: 'kimi/kimi-k3',
    messages: HISTORY_WITH_LOAD,
    tools: [SEARCH, DEFERRED],
  })

  expect(payload?.tools).toEqual([{
    type: 'function',
    function: {
      name: 'ToolSearchTool',
      description: 'Search tools.',
      parameters: { type: 'object', properties: {} },
    },
  }])
  const messages = payload?.messages as Record<string, unknown>[]
  const synthetic = messages.at(-1)
  expect(synthetic?.role).toBe('system')
  expect(synthetic?.content).toBeUndefined()
  expect((synthetic?.tools as Record<string, unknown>[])[0]).toMatchObject({
    type: 'function',
    function: { name: 'deploy_service' },
  })
})
