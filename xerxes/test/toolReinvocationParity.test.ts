// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import type {
  CompletionRequest,
  LlmClient,
  LlmDelta,
} from '../src/llms/client.js'
import { createAgentState, type StreamEvent } from '../src/streaming/events.js'
import { runTurn } from '../src/streaming/loop.js'
import { parseLlamaToolCalls } from '../src/streaming/toolCallParsers.js'
import type { ToolCall, ToolDefinition } from '../src/types/toolCalls.js'

const EXPOSED_READ_FILE: ToolDefinition = {
  type: 'function',
  function: {
    name: 'ReadFile',
    description: 'Read a file.',
    parameters: { type: 'object', properties: { path: { type: 'string' } } },
  },
}

function unknownProviderCall(): ToolCall {
  return {
    id: 'bad_1',
    type: 'function',
    function: {
      name: 'user_request_reinterpretation',
      arguments: { prompt: 're do it again' },
    },
  }
}

async function collect(
  events: AsyncIterable<StreamEvent>,
): Promise<StreamEvent[]> {
  const result: StreamEvent[] = []
  for await (const event of events) result.push(event)
  return result
}

test('tagged function markup parses typed parameter values for a reinvoked native tool turn', () => {
  const text = [
    '<function=web.search_query>',
    '<parameter=q>',
    'latest OpenAI news',
    '</parameter>',
    '<parameter=search_type>',
    'news',
    '</parameter>',
    '<parameter=n_results>',
    '5',
    '</parameter>',
    '<parameter=include_archived>',
    'false',
    '</parameter>',
    '</function>',
  ].join('\n')

  expect(parseLlamaToolCalls(text)).toEqual([
    {
      name: 'web.search_query',
      arguments: {
        q: 'latest OpenAI news',
        search_type: 'news',
        n_results: 5,
        include_archived: false,
      },
      rawId: '',
    },
  ])
})

test('an unknown provider tool call fails explicitly without executing and lets the provider recover', async () => {
  const requests: CompletionRequest[] = []
  const llm: LlmClient = {
    async *stream(request): AsyncGenerator<LlmDelta> {
      requests.push(request)
      if (requests.length === 1) {
        yield {
          content: 'There is no prior task to redo.',
          toolCalls: [unknownProviderCall()],
        }
        return
      }
      yield { content: 'There is no configured redo tool, so I stopped safely.' }
    },
  }
  let executions = 0
  const state = createAgentState()

  const events = await collect(
    runTurn(
      {
        model: 'gpt-4o-mini',
        permissionMode: 'accept-all',
        state,
        tools: [EXPOSED_READ_FILE],
        userMessage: 're do it again',
      },
      {
        llm,
        toolExecutor: {
          async execute(): Promise<string> {
            executions += 1
            return 'unreachable'
          },
        },
      },
    ),
  )

  expect(requests).toHaveLength(2)
  expect(executions).toBe(0)
  expect(events.some((event) => event.type === 'tool_start')).toBe(false)
  expect(events).toContainEqual(expect.objectContaining({
    type: 'tool_end',
    result: expect.objectContaining({
      name: 'user_request_reinterpretation',
      result: 'Tool execution failed: user_request_reinterpretation was not configured for this turn.',
      toolCallId: 'bad_1',
    }),
  }))
  expect(requests[1]?.messages).toContainEqual({
    role: 'tool',
    name: 'user_request_reinterpretation',
    tool_call_id: 'bad_1',
    content: 'Tool execution failed: user_request_reinterpretation was not configured for this turn.',
  })
  expect(state.messages.at(-1)).toEqual({
    role: 'assistant',
    content: 'There is no configured redo tool, so I stopped safely.',
  })
})
