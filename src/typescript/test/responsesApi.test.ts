// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ResponsesEventTranslator } from '../src/streaming/responsesApi.js'

test('Responses API translator streams text and thinking while assembling function calls and usage', () => {
  const translator = new ResponsesEventTranslator()
  const deltas = [...translator.translateAll([
    { type: 'response.output_text.delta', delta: 'I will inspect it.' },
    { type: 'response.reasoning.delta', delta: 'Need the project file.' },
    {
      type: 'response.output_item.added',
      item: { type: 'function_call', id: 'item_1', call_id: 'call_1', name: 'ReadFile' },
    },
    { type: 'response.function_call_arguments.delta', item_id: 'item_1', delta: '{"path":' },
    { type: 'response.function_call_arguments.delta', item_id: 'item_1', delta: '"README.md"}' },
    {
      type: 'response.output_item.done',
      item: { type: 'function_call', id: 'item_1', call_id: 'call_1', name: 'ReadFile' },
    },
    {
      type: 'response.completed',
      response: {
        status: 'completed',
        usage: {
          input_tokens: 12,
          output_tokens: 7,
          input_tokens_details: { cached_tokens: 3 },
          output_tokens_details: { reasoning_tokens: 2 },
        },
      },
    },
  ])]

  expect(deltas).toEqual([
    { content: 'I will inspect it.' },
    { thinking: 'Need the project file.' },
    {
      finishReason: 'completed',
      usage: { inputTokens: 12, outputTokens: 7, cacheReadTokens: 3, reasoningTokens: 2 },
      toolCalls: [{
        id: 'item_1',
        type: 'function',
        function: { name: 'ReadFile', arguments: { path: 'README.md' } },
      }],
    },
  ])
  expect(translator.usage).toEqual({
    inputTokens: 12,
    outputTokens: 7,
    cacheReadTokens: 3,
    reasoningTokens: 2,
    toolCalls: [{
      id: 'item_1',
      type: 'function',
      function: { name: 'ReadFile', arguments: { path: 'README.md' } },
    }],
    finishReason: 'completed',
  })
})

test('Responses API translator flushes unfinished calls and marks failed streams', () => {
  const translator = new ResponsesEventTranslator()
  const deltas = [...translator.translateAll([
    {
      type: 'response.output_item.added',
      item: { type: 'tool_call', id: 'call_2', name: 'ListDir' },
    },
    { type: 'response.function_call_arguments.delta', call_id: 'call_2', delta: '{"directory":"."}' },
    { type: 'response.failed' },
  ])]

  expect(deltas).toEqual([{
    finishReason: 'error',
    usage: { inputTokens: 0, outputTokens: 0 },
    toolCalls: [{
      id: 'call_2',
      type: 'function',
      function: { name: 'ListDir', arguments: { directory: '.' } },
    }],
  }])
})
