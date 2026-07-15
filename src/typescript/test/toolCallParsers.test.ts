// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import type { JsonObject } from '../src/types/toolCalls.js'
import {
  TOOL_CALL_PARSER_REGISTRY,
  detectToolCallFormat,
  getToolCallParser,
  parseLlamaToolCalls,
  parseMistralToolCalls,
  parseToolCallBlocks,
  parseToolCallsForModel,
} from '../src/streaming/toolCallParsers.js'

test('tool-call parser registry contains every raw-text provider format', () => {
  expect(Object.keys(TOOL_CALL_PARSER_REGISTRY).sort()).toEqual([
    'deepseek_v3',
    'deepseek_v3_1',
    'glm45',
    'glm47',
    'kimi_k2',
    'llama',
    'longcat',
    'mistral',
    'qwen',
    'qwen3_coder',
    'xml_tool_call',
  ])
  expect(getToolCallParser('missing')).toBeUndefined()
  expect(getToolCallParser('__proto__')).toBeUndefined()
})

test('model format detection preserves provider precedence', () => {
  expect(detectToolCallFormat('nous/hermes-3-llama-3.1-8b')).toBe('xml_tool_call')
  expect(detectToolCallFormat('meta/llama-3.1-70b')).toBe('llama')
  expect(detectToolCallFormat('Qwen/Qwen3-Coder-32B')).toBe('qwen3_coder')
  expect(detectToolCallFormat('Qwen/Qwen-2.5-72B')).toBe('qwen')
  expect(detectToolCallFormat('mistralai/Mixtral-8x22B')).toBe('mistral')
  expect(detectToolCallFormat('deepseek-v3.1-chat')).toBe('deepseek_v3_1')
  expect(detectToolCallFormat('deepseek-v3-base')).toBe('deepseek_v3')
  expect(detectToolCallFormat('Zhipu/glm-4.7-air')).toBe('glm47')
  expect(detectToolCallFormat('Zhipu/glm-4.5')).toBe('glm45')
  expect(detectToolCallFormat('moonshot/kimi-k2')).toBe('kimi_k2')
  expect(detectToolCallFormat('longcat-32b')).toBe('longcat')
  expect(detectToolCallFormat('random-model')).toBeUndefined()
})

test('tag parsers recognize provider variants and aliases', () => {
  const fixtures: ReadonlyArray<readonly [string, string, string, JsonObject]> = [
    [
      'xml_tool_call',
      '<tool_call>{"function":"ReadFile","input":{"path":"a.txt"}}</tool_call>',
      'ReadFile',
      { path: 'a.txt' },
    ],
    ['qwen', '<tool_call>{"name":"ListDir","arguments":{"path":"."}}</tool_call>', 'ListDir', { path: '.' }],
    [
      'qwen3_coder',
      '|<function_call_start|>{"name":"Run","parameters":{"command":"pwd"}}|<function_call_end|>',
      'Run',
      { command: 'pwd' },
    ],
    [
      'deepseek_v3_1',
      '<tool>{"name":"Search","arguments":{"query":"needle"}}</tool>',
      'Search',
      { query: 'needle' },
    ],
    ['glm45', '<tool_call>{"name":"Read","arguments":{}}</tool_call>', 'Read', {}],
    [
      'glm47',
      '<function_call>{"name":"Write","arguments":{"path":"note.txt"}}</function_call>',
      'Write',
      { path: 'note.txt' },
    ],
    ['kimi_k2', '<|tool_call|>{"name":"Edit","arguments":{"path":"a.ts"}}<|/tool_call|>', 'Edit', { path: 'a.ts' }],
    ['longcat', '<longcat:tool>{"name":"Echo","arguments":{"text":"hi"}}</longcat:tool>', 'Echo', { text: 'hi' }],
  ]

  for (const [format, text, name, arguments_] of fixtures) {
    expect(getToolCallParser(format)?.parse(text)).toEqual([{ name, arguments: arguments_, rawId: '' }])
  }

  const fullwidthBar = '\uff5c'
  const lowerEighthBlock = '\u2581'
  const openTag = '<' + fullwidthBar + 'tool' + lowerEighthBlock + 'call'
    + lowerEighthBlock + 'begin' + fullwidthBar + '>'
  const closeTag = '<' + fullwidthBar + 'tool' + lowerEighthBlock + 'call'
    + lowerEighthBlock + 'end' + fullwidthBar + '>'
  expect(getToolCallParser('deepseek_v3')?.parse(
    openTag + '{"name":"DeepSeek","arguments":{"value":1}}' + closeTag,
  )).toEqual([{ name: 'DeepSeek', arguments: { value: 1 }, rawId: '' }])
})

test('JSON-aware tag scanning retains close-tag strings and only accepts completed blocks', () => {
  const text = '<tool_call>{"name":"Emit","arguments":{"text":"literal </tool_call> stays in JSON"}}</tool_call>'
  expect(parseToolCallBlocks(text, '<tool_call>', '</tool_call>')).toEqual([{
    name: 'Emit',
    arguments: { text: 'literal </tool_call> stays in JSON' },
    rawId: '',
  }])
  expect(parseToolCallBlocks('<tool_call>{"name":"Open","arguments":{}}', '<tool_call>', '</tool_call>')).toEqual([])
  expect(parseToolCallBlocks('<tool_call>not-json</tool_call>', '<tool_call>', '</tool_call>')).toEqual([])
  expect(parseToolCallBlocks(
    '<tool_call>{"name":"One","arguments":{}}</tool_call><tool_call>{"name":"Two","arguments":{"n":2}}</tool_call>',
    '<tool_call>',
    '</tool_call>',
  )).toEqual([
    { name: 'One', arguments: {}, rawId: '' },
    { name: 'Two', arguments: { n: 2 }, rawId: '' },
  ])
  expect(parseToolCallBlocks('<tool_call>not-json'.repeat(100), '<tool_call>', '</tool_call>')).toEqual([])
  expect(parseToolCallBlocks('{"name":"NoTag"}', '', '</tool_call>')).toEqual([])
})

test('Llama handles both formats and Mistral preserves provider ids', () => {
  const llama = [
    '<|python_tag|>{"name":"ReadFile","parameters":{"path":"README.md"}}</eom>',
    '<|python_tag|>{"name":"ListDir","arguments":{"path":"src"}}<|eom_id|>',
    '<function=Greet>{"name":"Ada","text":"literal </function> value"}</function>',
  ].join('')
  expect(parseLlamaToolCalls(llama)).toEqual([
    { name: 'ListDir', arguments: { path: 'src' }, rawId: '' },
    { name: 'Greet', arguments: { name: 'Ada', text: 'literal </function> value' }, rawId: '' },
  ])

  expect(parseMistralToolCalls(
    '[TOOL_CALLS][{"id":"provider-1","name":"ReadFile","arguments":{"path":"a.txt"}},{"name":"Skip","arguments":[]}]',
  )).toEqual([
    { name: 'ReadFile', arguments: { path: 'a.txt' }, rawId: 'provider-1' },
    { name: 'Skip', arguments: {}, rawId: '' },
  ])
  expect(parseMistralToolCalls('nothing here')).toEqual([])
  expect(parseMistralToolCalls('[TOOL_CALLS][{"name":"unfinished"')).toEqual([])
})

test('model helper leaves unknown models inert and returns completed matching blocks', () => {
  expect(parseToolCallsForModel('unknown-model', '<tool_call>{"name":"Run","arguments":{}}</tool_call>')).toEqual([])
  expect(parseToolCallsForModel(
    'Qwen/Qwen-2.5-72B',
    '<tool_call>{"name":"Run","arguments":{"command":"pwd"}}</tool_call>',
  )).toEqual([{ name: 'Run', arguments: { command: 'pwd' }, rawId: '' }])
})
