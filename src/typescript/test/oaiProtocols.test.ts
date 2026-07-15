// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  OpenAiProtocolValidationError,
  chatMessagesFromOpenAi,
  chatMessagesToOpenAi,
  completionRequestFromOpenAi,
  createOpenAiChatCompletionResponse,
  createOpenAiCompletionStreamResponse,
  parseOpenAiChatCompletionRequest,
  parseOpenAiCompletionRequest,
  usageInfoFromTokenUsage,
} from '../src/types/oaiProtocols.js'

test('OpenAI chat requests normalize defaults, preserve extensions, and become portable completion requests', () => {
  const request = parseOpenAiChatCompletionRequest({
    model: 'gpt-test',
    messages: [
      { role: 'system', content: 'Be concise.' },
      {
        role: 'assistant',
        content: null,
        function_call: { name: 'ReadFile', arguments: '{"path":"README.md"}' },
      },
    ],
    max_tokens: 64,
    temperature: 0.2,
    top_p: 0.8,
    stop: 'END',
    tools: [{
      type: 'function',
      function: {
        name: 'ReadFile',
        parameters: { type: 'object', properties: { path: { type: 'string' } } },
      },
    }],
    tool_choice: 'required',
    x_provider_hint: 'keep-me',
  })

  expect(request).toMatchObject({
    model: 'gpt-test',
    max_tokens: 64,
    temperature: 0.2,
    top_p: 0.8,
    stop: ['END'],
    stream: false,
    n: 1,
    extensions: { x_provider_hint: 'keep-me' },
  })

  expect(completionRequestFromOpenAi(request, { legacyToolCallId: () => 'call-legacy' })).toEqual({
    model: 'gpt-test',
    messages: [
      { role: 'system', content: 'Be concise.' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [{
          id: 'call-legacy',
          type: 'function',
          function: { name: 'ReadFile', arguments: { path: 'README.md' } },
        }],
      },
    ],
    maxTokens: 64,
    temperature: 0.2,
    topP: 0.8,
    stop: ['END'],
    tools: [{
      type: 'function',
      function: {
        name: 'ReadFile',
        description: '',
        parameters: { type: 'object', properties: { path: { type: 'string' } } },
      },
    }],
    toolChoice: 'any',
  })
})

test('canonical messages round trip through OpenAI protocol messages without creating another tool-call type', () => {
  const original = [
    { role: 'user' as const, content: [{ type: 'text' as const, text: 'Inspect this.' }] },
    {
      role: 'assistant' as const,
      content: '',
      tool_calls: [{
        id: 'call-1',
        type: 'function' as const,
        function: { name: 'ReadFile', arguments: { path: 'README.md' } },
      }],
    },
    { role: 'tool' as const, tool_call_id: 'call-1', name: 'ReadFile', content: 'contents' },
  ]

  expect(chatMessagesFromOpenAi(chatMessagesToOpenAi(original))).toEqual(original)
})

test('OpenAI protocol parsing accepts pre-decoded tool arguments and normalizes them through the canonical tool-call parser', () => {
  const request = parseOpenAiChatCompletionRequest({
    model: 'gpt-test',
    messages: [{
      role: 'assistant',
      content: null,
      tool_calls: [{
        id: 'call-object',
        type: 'function',
        function: { name: 'ReadFile', arguments: { path: 'README.md' } },
      }],
    }],
  })

  expect(completionRequestFromOpenAi(request).messages).toEqual([{
    role: 'assistant',
    content: '',
    tool_calls: [{
      id: 'call-object',
      type: 'function',
      function: { name: 'ReadFile', arguments: { path: 'README.md' } },
    }],
  }])
})

test('protocol validation rejects invalid bounds and unsupported legacy-only runtime conversions', () => {
  expect(() => parseOpenAiChatCompletionRequest({
    model: 'gpt-test',
    messages: [],
    max_tokens: 0,
  })).toThrow(OpenAiProtocolValidationError)

  expect(() => chatMessagesFromOpenAi([{ role: 'function', name: 'OldFunction', content: 'result' }])).toThrow(
    OpenAiProtocolValidationError,
  )

  const forcedNamedTool = parseOpenAiChatCompletionRequest({
    model: 'gpt-test',
    messages: [],
    tool_choice: { type: 'function', function: { name: 'ReadFile' } },
  })
  expect(() => completionRequestFromOpenAi(forcedNamedTool)).toThrow(OpenAiProtocolValidationError)

  expect(() => parseOpenAiCompletionRequest({ model: 'gpt-test', prompt: ['ok', 1] })).toThrow(
    OpenAiProtocolValidationError,
  )
})

test('response builders preserve explicit ids and timestamps while usage conversion carries timing metadata', () => {
  const usage = usageInfoFromTokenUsage(
    { inputTokens: 11, outputTokens: 3 },
    { processingTime: 0.5, tokensPerSecond: 6 },
  )
  expect(usage).toEqual({
    prompt_tokens: 11,
    completion_tokens: 3,
    total_tokens: 14,
    processing_time: 0.5,
    tokens_per_second: 6,
  })

  expect(createOpenAiChatCompletionResponse({
    id: 'chat-test',
    created: 1_700_000_000,
    model: 'gpt-test',
    choices: [{
      index: 0,
      message: { role: 'assistant', content: 'hello' },
      finish_reason: 'stop',
    }],
    usage,
  })).toEqual({
    id: 'chat-test',
    object: 'chat.completion',
    created: 1_700_000_000,
    model: 'gpt-test',
    choices: [{
      index: 0,
      message: { role: 'assistant', content: 'hello' },
      finish_reason: 'stop',
    }],
    usage,
  })

  expect(createOpenAiCompletionStreamResponse({
    id: 'cmpl-test',
    created: 1_700_000_001,
    model: 'gpt-test',
    choices: [{ index: 0, text: 'hello', finish_reason: null }],
  })).toEqual({
    id: 'cmpl-test',
    object: 'text_completion.chunk',
    created: 1_700_000_001,
    model: 'gpt-test',
    choices: [{ index: 0, text: 'hello', finish_reason: null }],
  })
})
