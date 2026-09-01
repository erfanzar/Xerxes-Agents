// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { OpenAiCompatibleClient, ResponsesApiClient, type LlmDelta } from '../src/llms/client.js'
import {
  appendGrammarInput,
  createGrammarInputAccumulator,
  createGrammarToolInputProperties,
  grammarToolInput,
  resolveGrammar,
} from '../src/llms/grammarTools.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'

const GRAMMAR_TOOL: ToolDefinition = {
  type: 'function',
  function: {
    name: 'sql',
    description: 'Run a SQL query.',
    parameters: {
      type: 'object',
      properties: { query: { type: 'string' } },
      required: ['query'],
    },
  },
  constrainedSampling: {
    type: 'grammar',
    variants: {
      openai_lark: 'start: "SELECT" .*',
      openai_regex: '^SELECT .*$',
    },
  },
}

// gpt-5.2's generated catalog compat enables grammar tools; gpt-4o does not.
const GRAMMAR_MODEL = 'openai/gpt-5.2'
const PLAIN_MODEL = 'openai/gpt-4o'

test('resolveGrammar prefers lark, falls back to regex, and validates the schema contract', () => {
  expect(resolveGrammar(GRAMMAR_TOOL, true)).toEqual({
    definition: 'start: "SELECT" .*',
    inputProperty: 'query',
    syntax: 'lark',
  })

  const regexOnly: ToolDefinition = {
    ...GRAMMAR_TOOL,
    constrainedSampling: { type: 'grammar', variants: { openai_regex: '^SELECT .*$' } },
  }
  expect(resolveGrammar(regexOnly, true)?.syntax).toBe('regex')

  expect(resolveGrammar(GRAMMAR_TOOL, false)).toBeUndefined()
  expect(resolveGrammar({ ...GRAMMAR_TOOL, constrainedSampling: false }, true)).toBeUndefined()

  expect(() => resolveGrammar({
    ...GRAMMAR_TOOL,
    constrainedSampling: { type: 'grammar', variants: {} },
  }, true)).toThrow(/no supported grammar variant/)

  expect(() => resolveGrammar({
    ...GRAMMAR_TOOL,
    function: {
      ...GRAMMAR_TOOL.function,
      parameters: { type: 'object', properties: { a: { type: 'string' }, b: { type: 'string' } }, required: ['a', 'b'] },
    },
  }, true)).toThrow(/exactly one required string property/)
})

test('grammarToolInput requires the declared string property on replay', () => {
  expect(grammarToolInput('sql', 'query', { query: 'SELECT 1' })).toBe('SELECT 1')
  expect(() => grammarToolInput('sql', 'query', { query: 42 as unknown as string }))
    .toThrow(/requires argument "query" to be a string/)
})

test('grammar input accumulation re-wraps raw text as growing JSON and enforces monotonicity', () => {
  const state = createGrammarInputAccumulator()
  expect(appendGrammarInput(state, 'SELECT ', false, 'query')).toBe('{"query":"SELECT ')
  expect(appendGrammarInput(state, '* FROM t', false, 'query')).toBe('* FROM t')
  expect(appendGrammarInput(state, 'SELECT * FROM t LIMIT 1', true, 'query')).toBe(' LIMIT 1"}')
  expect(state.input).toBe('SELECT * FROM t LIMIT 1')

  expect(() => appendGrammarInput(state, 'more', false, 'query')).toThrow(/changed after it was closed/)

  const broken = createGrammarInputAccumulator()
  appendGrammarInput(broken, 'SELECT ', false, 'query')
  expect(() => appendGrammarInput(broken, 'DROP TABLE', true, 'query')).toThrow(/non-monotonically/)

  const escaped = createGrammarInputAccumulator()
  expect(appendGrammarInput(escaped, 'say "hi"\n', true, 'query')).toBe('{"query":"say \\"hi\\"\\n"}')
})

test('completions payload serializes grammar tools as custom only for capable models', async () => {
  const payloads: Record<string, unknown>[] = []
  const fetchImplementation = async (_input: unknown, init?: RequestInit) => {
    payloads.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
    return new Response(JSON.stringify({
      choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }],
    }))
  }

  const capable = new OpenAiCompatibleClient({ providerName: 'openai', apiKey: 'k', baseUrl: 'https://example.invalid/v1', fetchImplementation })
  await capable.complete({ model: GRAMMAR_MODEL, messages: [{ role: 'user', content: 'q' }], tools: [GRAMMAR_TOOL] })

  const plain = new OpenAiCompatibleClient({ providerName: 'openai', apiKey: 'k', baseUrl: 'https://example.invalid/v1', fetchImplementation })
  await plain.complete({ model: PLAIN_MODEL, messages: [{ role: 'user', content: 'q' }], tools: [GRAMMAR_TOOL] })

  expect(payloads[0]?.tools).toEqual([{
    type: 'custom',
    custom: {
      name: 'sql',
      description: 'Run a SQL query.',
      format: { type: 'grammar', grammar: { syntax: 'lark', definition: 'start: "SELECT" .*' } },
    },
  }])
  expect(payloads[1]?.tools).toEqual([{
    type: 'function',
    function: {
      name: 'sql',
      description: 'Run a SQL query.',
      parameters: GRAMMAR_TOOL.function.parameters,
    },
  }])
})

test('completions stream accumulates custom tool input as raw grammar text', async () => {
  const encoder = new TextEncoder()
  const frames = [
    { choices: [{ delta: { tool_calls: [{ index: 0, id: 'call_1', custom: { name: 'sql', input: 'SELECT ' } }] } }] },
    { choices: [{ delta: { tool_calls: [{ index: 0, custom: { input: '* FROM t' } }] } }] },
    { choices: [{ delta: {}, finish_reason: 'tool_calls' }], usage: { prompt_tokens: 3, completion_tokens: 1 } },
  ]
  const client = new OpenAiCompatibleClient({
    providerName: 'openai',
    apiKey: 'k',
    baseUrl: 'https://example.invalid/v1',
    fetchImplementation: async () => new Response(new ReadableStream({
      start(controller) {
        for (const frame of frames) controller.enqueue(encoder.encode('data: ' + JSON.stringify(frame) + '\n\n'))
        controller.enqueue(encoder.encode('data: [DONE]\n\n'))
        controller.close()
      },
    })),
  })

  const events: LlmDelta[] = []
  for await (const event of client.stream({
    model: GRAMMAR_MODEL,
    messages: [{ role: 'user', content: 'q' }],
    tools: [GRAMMAR_TOOL],
  })) events.push(event)

  const calls = events.find(event => event.toolCalls)?.toolCalls
  expect(calls).toEqual([{
    id: 'call_1',
    type: 'function',
    function: { name: 'sql', arguments: { query: 'SELECT * FROM t' } },
  }])
})

test('responses payload serializes grammar tools flat and replays custom calls natively', async () => {
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
    model: GRAMMAR_MODEL,
    messages: [
      { role: 'user', content: 'q' },
      {
        role: 'assistant',
        content: '',
        tool_calls: [{
          id: 'call_1',
          type: 'function',
          function: { name: 'sql', arguments: { query: 'SELECT 1' } },
        }],
      },
      { role: 'tool', tool_call_id: 'call_1', content: '[{"n":1}]' },
    ],
    tools: [GRAMMAR_TOOL],
  })

  expect(payload?.tools).toEqual([{
    type: 'custom',
    name: 'sql',
    description: 'Run a SQL query.',
    format: { type: 'grammar', syntax: 'lark', definition: 'start: "SELECT" .*' },
  }])
  expect(payload?.input).toEqual([
    { role: 'user', content: 'q' },
    { type: 'custom_tool_call', call_id: 'call_1', name: 'sql', input: 'SELECT 1' },
    { type: 'custom_tool_call_output', call_id: 'call_1', output: '[{"n":1}]' },
  ])
})

test('responses stream resolves custom_tool_call items onto their grammar input property', async () => {
  const client = new ResponsesApiClient({
    providerName: 'openai',
    apiKey: 'k',
    baseUrl: 'https://example.invalid/v1',
    fetchImplementation: async () => {
      const encoder = new TextEncoder()
      return new Response(new ReadableStream({
        start(controller) {
          for (const event of [
            { type: 'response.output_item.added', item: { type: 'custom_tool_call', id: 'item_1', call_id: 'call_1', name: 'sql' } },
            { type: 'response.custom_tool_call_input.delta', item_id: 'item_1', delta: 'SELECT ' },
            { type: 'response.custom_tool_call_input.done', item_id: 'item_1', input: 'SELECT * FROM t' },
            { type: 'response.completed', response: { status: 'completed', usage: { input_tokens: 2, output_tokens: 1 } } },
          ]) {
            controller.enqueue(encoder.encode('data: ' + JSON.stringify(event) + '\n\n'))
          }
          controller.enqueue(encoder.encode('data: [DONE]\n\n'))
          controller.close()
        },
      }))
    },
  })

  const events: LlmDelta[] = []
  for await (const event of client.stream({
    model: GRAMMAR_MODEL,
    messages: [{ role: 'user', content: 'q' }],
    tools: [GRAMMAR_TOOL],
  })) events.push(event)

  const calls = events.find(event => event.toolCalls)?.toolCalls
  expect(calls).toEqual([{
    id: 'call_1',
    type: 'function',
    function: { name: 'sql', arguments: { query: 'SELECT * FROM t' } },
  }])
})

test('responses stream rejects a custom tool input done that contradicts its deltas', async () => {
  const client = new ResponsesApiClient({
    providerName: 'openai',
    apiKey: 'k',
    baseUrl: 'https://example.invalid/v1',
    fetchImplementation: async () => {
      const encoder = new TextEncoder()
      return new Response(new ReadableStream({
        start(controller) {
          for (const event of [
            { type: 'response.output_item.added', item: { type: 'custom_tool_call', id: 'item_1', call_id: 'call_1', name: 'sql' } },
            { type: 'response.custom_tool_call_input.delta', item_id: 'item_1', delta: 'SELECT ' },
            { type: 'response.custom_tool_call_input.done', item_id: 'item_1', input: 'DROP TABLE t' },
          ]) {
            controller.enqueue(encoder.encode('data: ' + JSON.stringify(event) + '\n\n'))
          }
          controller.close()
        },
      }))
    },
  })

  await expect((async () => {
    for await (const _event of client.stream({
      model: GRAMMAR_MODEL,
      messages: [{ role: 'user', content: 'q' }],
      tools: [GRAMMAR_TOOL],
    })) { /* drain */ }
  })()).rejects.toThrow(/non-monotonically/)
})

test('createGrammarToolInputProperties maps only grammar-capable tools', () => {
  const plain: ToolDefinition = {
    type: 'function',
    function: { name: 'read', description: 'Read.', parameters: { type: 'object', properties: {} } },
  }
  const properties = createGrammarToolInputProperties([GRAMMAR_TOOL, plain], true)
  expect(properties.get('sql')).toBe('query')
  expect(properties.has('read')).toBe(false)
  expect(createGrammarToolInputProperties([GRAMMAR_TOOL], false).size).toBe(0)
})
