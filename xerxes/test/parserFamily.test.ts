// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { parseLlamaToolCalls, parseToolCallBlocks, parseXmlToolCalls } from '../src/streaming/parsers/common.js'
import {
  DEEPSEEK_V3_CLOSE_TAG,
  DEEPSEEK_V3_OPEN_TAG,
  parseDeepSeekV3ToolCalls,
  parseDeepSeekV31ToolCalls,
} from '../src/streaming/parsers/deepseek.js'
import { parseGlm45ToolCalls, parseGlm47ToolCalls } from '../src/streaming/parsers/glm.js'
import { normalizeParsedToolCalls } from '../src/streaming/parsers/index.js'
import { parseKimiK2ToolCalls } from '../src/streaming/parsers/kimi.js'
import { parseLongCatToolCalls } from '../src/streaming/parsers/longcat.js'
import { parseMistralToolCalls } from '../src/streaming/parsers/mistral.js'
import { parseQwen3CoderToolCalls, parseQwenToolCalls } from '../src/streaming/parsers/qwen.js'
import { deterministicToolCallId } from '../src/streaming/toolCallIds.js'

test('provider parser modules extract their concrete completed formats', () => {
  expect(parseXmlToolCalls('<tool_call>{"name":"Xml","arguments":{"a":1}}</tool_call>')).toEqual([
    { name: 'Xml', arguments: { a: 1 }, rawId: '' },
  ])
  expect(parseDeepSeekV3ToolCalls(
    DEEPSEEK_V3_OPEN_TAG + '{"name":"DeepSeek","arguments":{}}' + DEEPSEEK_V3_CLOSE_TAG,
  )).toEqual([{ name: 'DeepSeek', arguments: {}, rawId: '' }])
  expect(parseDeepSeekV31ToolCalls('<tool>{"name":"DeepSeek31","input":{"q":1}}</tool>')).toEqual([
    { name: 'DeepSeek31', arguments: { q: 1 }, rawId: '' },
  ])
  expect(parseGlm45ToolCalls('<tool_call>{"name":"Glm45","arguments":{}}</tool_call>')).toEqual([
    { name: 'Glm45', arguments: {}, rawId: '' },
  ])
  expect(parseGlm47ToolCalls('<function_call>{"name":"Glm47","arguments":{}}</function_call>')).toEqual([
    { name: 'Glm47', arguments: {}, rawId: '' },
  ])
  expect(parseKimiK2ToolCalls('<|tool_call|>{"name":"Kimi","parameters":{"x":1}}<|/tool_call|>')).toEqual([
    { name: 'Kimi', arguments: { x: 1 }, rawId: '' },
  ])
  expect(parseLongCatToolCalls('<longcat:tool>{"name":"LongCat","arguments":{}}</longcat:tool>')).toEqual([
    { name: 'LongCat', arguments: {}, rawId: '' },
  ])
  expect(parseQwenToolCalls('<tool_call>{"name":"Qwen","arguments":{}}</tool_call>')).toEqual([
    { name: 'Qwen', arguments: {}, rawId: '' },
  ])
  expect(parseQwen3CoderToolCalls(
    '|<function_call_start|>{"name":"QwenCoder","parameters":{"path":"a.ts"}}|<function_call_end|>',
  )).toEqual([{ name: 'QwenCoder', arguments: { path: 'a.ts' }, rawId: '' }])
})

test('common parser handles nested JSON, literal closing tags, and Llama forms', () => {
  expect(parseToolCallBlocks(
    '<tool_call>{"name":"Emit","arguments":{"text":"</tool_call>","nested":{"ok":true}}}</tool_call>',
    '<tool_call>',
    '</tool_call>',
  )).toEqual([{ name: 'Emit', arguments: { text: '</tool_call>', nested: { ok: true } }, rawId: '' }])
  expect(parseToolCallBlocks('<tool_call>{"name":"Partial","arguments":{}}', '<tool_call>', '</tool_call>')).toEqual([])
  expect(parseLlamaToolCalls(
    '<|python_tag|>{"name":"Read","parameters":{"path":"README.md"}}<|eom_id|>'
      + '<function=Write>{"path":"note.txt"}</function>',
  )).toEqual([
    { name: 'Read', arguments: { path: 'README.md' }, rawId: '' },
    { name: 'Write', arguments: { path: 'note.txt' }, rawId: '' },
  ])
})

test('Mistral preserves provider IDs and raw parser calls normalize to runtime ToolCalls', () => {
  const parsed = parseMistralToolCalls(
    '[TOOL_CALLS][{"id":"provider-1","name":"Read","arguments":{"path":"a.txt"}},{"name":"List","parameters":{"path":"."}}]',
  )
  expect(parsed).toEqual([
    { name: 'Read', arguments: { path: 'a.txt' }, rawId: 'provider-1' },
    { name: 'List', arguments: { path: '.' }, rawId: '' },
  ])
  expect(normalizeParsedToolCalls(parsed)).toEqual([
    { id: 'provider-1', type: 'function', function: { name: 'Read', arguments: { path: 'a.txt' } } },
    {
      id: deterministicToolCallId('List', { path: '.' }),
      type: 'function',
      function: { name: 'List', arguments: { path: '.' } },
    },
  ])
})

test('identical raw-text calls in one batch get unique suffixed fallback IDs', () => {
  const normalized = normalizeParsedToolCalls([
    { name: 'List', arguments: { path: '.' }, rawId: '' },
    { name: 'List', arguments: { path: '.' }, rawId: '' },
    { name: 'List', arguments: { path: '.' }, rawId: '' },
    { name: 'Read', arguments: { path: 'a.txt' }, rawId: 'provider-1' },
  ])
  const baseId = deterministicToolCallId('List', { path: '.' })

  expect(normalized.map(call => call.id)).toEqual([
    baseId,
    `${baseId}#2`,
    `${baseId}#3`,
    'provider-1',
  ])
  expect(new Set(normalized.map(call => call.id)).size).toBe(4)

  // Occurrence suffixing is scoped to one batch, keeping replays stable.
  expect(normalizeParsedToolCalls([{ name: 'List', arguments: { path: '.' }, rawId: '' }])[0]?.id)
    .toBe(baseId)
})

test('partial model output stays inert while invalid normalized calls fail explicitly', () => {
  expect(parseMistralToolCalls('[TOOL_CALLS][{"name":"unfinished"')).toEqual([])
  expect(() => parseToolCallBlocks(null as unknown as string, '<tool_call>', '</tool_call>'))
    .toThrow('tool-call parser text must be a string')
  expect(() => normalizeParsedToolCalls([
    { name: '', arguments: {}, rawId: '' },
  ])).toThrow('parsedToolCalls[0].name')
})
