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
      finishReason: 'tool_calls',
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
    finishReason: 'tool_calls',
  })
})

test('Responses API translator flushes unfinished calls when the transport truncates', () => {
  const translator = new ResponsesEventTranslator()
  const deltas = [...translator.translateAll([
    {
      type: 'response.output_item.added',
      item: { type: 'tool_call', id: 'call_2', name: 'ListDir' },
    },
    { type: 'response.function_call_arguments.delta', call_id: 'call_2', delta: '{"directory":"."}' },
  ])]

  expect(deltas).toEqual([{
    finishReason: 'stop',
    usage: { inputTokens: 0, outputTokens: 0 },
    toolCalls: [{
      id: 'call_2',
      type: 'function',
      function: { name: 'ListDir', arguments: { directory: '.' } },
    }],
  }])
})

test('Responses API translator maps completed status to neutral finish reasons', () => {
  const plain = [...new ResponsesEventTranslator().translateAll([
    { type: 'response.output_text.delta', delta: 'done' },
    {
      type: 'response.completed',
      response: { status: 'completed', usage: { input_tokens: 4, output_tokens: 2 } },
    },
  ])]
  expect(plain).toEqual([
    { content: 'done' },
    { finishReason: 'stop', usage: { inputTokens: 4, outputTokens: 2 } },
  ])

  const withCall = [...new ResponsesEventTranslator().translateAll([
    {
      type: 'response.output_item.done',
      item: { type: 'function_call', id: 'call-1', name: 'ListDir', arguments: '{"directory":"."}' },
    },
    { type: 'response.completed', response: { status: 'completed' } },
  ])]
  expect(withCall).toEqual([{
    finishReason: 'tool_calls',
    usage: { inputTokens: 0, outputTokens: 0 },
    toolCalls: [{
      id: 'call-1',
      type: 'function',
      function: { name: 'ListDir', arguments: { directory: '.' } },
    }],
  }])
})

test('Responses API translator merges entries aliased by item_id and call_id', () => {
  const deltas = [...new ResponsesEventTranslator().translateAll([
    { type: 'response.function_call_arguments.delta', call_id: 'call_9', delta: '{"path":' },
    {
      type: 'response.output_item.added',
      item: { type: 'function_call', id: 'item_9', call_id: 'call_9', name: 'ReadFile' },
    },
    { type: 'response.function_call_arguments.delta', item_id: 'item_9', delta: '"a.txt"}' },
    {
      type: 'response.output_item.done',
      item: { type: 'function_call', id: 'item_9', call_id: 'call_9', name: 'ReadFile' },
    },
    { type: 'response.completed', response: { status: 'completed' } },
  ])]

  expect(deltas).toEqual([{
    finishReason: 'tool_calls',
    usage: { inputTokens: 0, outputTokens: 0 },
    toolCalls: [{
      id: 'item_9',
      type: 'function',
      function: { name: 'ReadFile', arguments: { path: 'a.txt' } },
    }],
  }])
})

test('Responses API translator throws on failed and error events with the provider payload', () => {
  expect(() => [...new ResponsesEventTranslator().translateAll([
    { type: 'response.output_text.delta', delta: 'partial' },
    {
      type: 'response.failed',
      response: { status: 'failed', error: { code: 'server_error', message: 'Model exploded' } },
    },
  ])]).toThrow('stream returned API error (server_error): Model exploded')

  expect(() => [...new ResponsesEventTranslator().translateAll([
    { type: 'error', code: 'rate_limit_exceeded', message: 'Slow down' },
  ])]).toThrow('stream returned API error (rate_limit_exceeded): Slow down')
})

test('Responses API translator surfaces incomplete responses with a mapped finish reason', () => {
  const truncated = [...new ResponsesEventTranslator().translateAll([
    { type: 'response.output_text.delta', delta: 'cut off' },
    {
      type: 'response.incomplete',
      response: {
        status: 'incomplete',
        incomplete_details: { reason: 'max_output_tokens' },
        usage: { input_tokens: 5, output_tokens: 9 },
      },
    },
  ])]
  expect(truncated).toEqual([
    { content: 'cut off' },
    { finishReason: 'length', usage: { inputTokens: 5, outputTokens: 9 } },
  ])

  const filtered = [...new ResponsesEventTranslator().translateAll([
    {
      type: 'response.incomplete',
      response: { status: 'incomplete', incomplete_details: { reason: 'content_filter' } },
    },
  ])]
  expect(filtered).toEqual([{ finishReason: 'content_filter', usage: { inputTokens: 0, outputTokens: 0 } }])
})
