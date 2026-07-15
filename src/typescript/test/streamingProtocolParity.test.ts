// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { messagesToOpenAi } from '../src/streaming/messages.js'
import { checkPermission, isSafeShellCommand } from '../src/streaming/permissions.js'
import { ThinkingParser, splitThinkingTags } from '../src/streaming/thinkingParser.js'
import { extractAssistantToolCallMarkers, stripAssistantToolCallMarkers } from '../src/streaming/toolMarkers.js'

test('thinking parsing preserves Python-compatible variants, nested text, empty blocks, and batch normalization', () => {
  const parser = new ThinkingParser()
  expect(parser.process('A <think> First </think> B <thinking> Second </thinking> C')).toEqual([
    { type: 'text', text: 'A ' },
    { type: 'thinking', text: ' First ' },
    { type: 'text', text: ' B ' },
    { type: 'thinking', text: ' Second ' },
    { type: 'text', text: ' C' },
  ])

  const nested = new ThinkingParser()
  expect(nested.process('<think> Outer <think> Inner </think> Outer cont </think> End')).toEqual([
    { type: 'thinking', text: ' Outer <think> Inner ' },
    { type: 'text', text: ' Outer cont </think> End' },
  ])

  const empty = new ThinkingParser()
  expect(empty.process('<think></think>Done')).toEqual([{ type: 'text', text: 'Done' }])
  expect(splitThinkingTags('<think> First </think>Middle<think> Second </think>End')).toEqual({
    visible: 'MiddleEnd',
    thinking: 'First  Second',
  })
  expect(splitThinkingTags('<think> Mismatched </thinking> The answer is 42.')).toEqual({
    visible: ' The answer is 42.',
    thinking: 'Mismatched',
  })
})

test('OpenAI message conversion includes non-empty reasoning only for assistant messages', () => {
  const withThinking = messagesToOpenAi([
    { role: 'user', content: 'list files' },
    {
      role: 'assistant',
      content: '',
      thinking: 'Need to call ListDir.',
      tool_calls: [{
        id: 'call-1',
        type: 'function',
        function: { name: 'ListDir', arguments: { path: '.' } },
      }],
    },
  ])
  expect(withThinking[1]).toMatchObject({
    role: 'assistant',
    content: null,
    reasoning_content: 'Need to call ListDir.',
    tool_calls: [{ function: { name: 'ListDir' } }],
  })
  expect(withThinking[1]).not.toHaveProperty('thinking')

  const withoutThinking = messagesToOpenAi([
    { role: 'assistant', content: 'hello' },
    { role: 'assistant', content: 'ok', thinking: '' },
  ])
  expect(withoutThinking[0]).not.toHaveProperty('reasoning_content')
  expect(withoutThinking[1]).not.toHaveProperty('reasoning_content')
})

test('permission safety allows readonly cd flows while retaining approval gates for shell metacharacters and writes', () => {
  expect(isSafeShellCommand('cd')).toBe(true)
  expect(isSafeShellCommand('cd /tmp')).toBe(true)
  expect(isSafeShellCommand('cd ../project && pwd')).toBe(true)
  expect(isSafeShellCommand('cd /tmp && git diff')).toBe(true)
  expect(isSafeShellCommand('cd /tmp && rg TODO')).toBe(true)
  expect(isSafeShellCommand('cd /tmp; rm -rf build')).toBe(false)
  expect(isSafeShellCommand('cd $(mktemp -d)')).toBe(false)
  expect(isSafeShellCommand('cd /tmp && npm install')).toBe(false)
  expect(checkPermission({ function: { name: 'exec_command', arguments: { cmd: 'cd /tmp && git diff' } } }, 'auto')).toBe(true)
  expect(checkPermission({ function: { name: 'AskUserQuestionTool', arguments: { question: 'Continue?' } } }, 'auto')).toBe(true)
})

test('provider marker cleanup accepts glued ids, Python-repr tool context, and typed invoke parameters', () => {
  const glued = extractAssistantToolCallMarkers(
    "TOOL: {'ok': True, 'path': '/tmp/result.md'}\n"
      + 'TOOL_CALL_ID: call_cc_0ASSISTANT_TOOL_CALLS: [{"id":"call_1","name":"SetInteractionModeTool",'
      + '"input":{"mode":"researcher"}}]\n'
      + 'Round 2 complete.',
  )
  expect(glued).toEqual({
    text: 'Round 2 complete.',
    toolCalls: [{ id: 'call_1', name: 'SetInteractionModeTool', input: { mode: 'researcher' } }],
  })

  const invoke = extractAssistantToolCallMarkers([
    'I will verify.',
    '<invoke name="exec_command">',
    '<parameter name="cmd">bun test test/streaming</parameter>',
    '<parameter name="yield_time_ms">120000</parameter>',
    '</invoke>',
    '<system-reminder>Tool result pending.</system-reminder>',
  ].join('\n'), 'call_cc')
  expect(invoke).toEqual({
    text: 'I will verify.',
    toolCalls: [{
      id: 'call_cc_0',
      name: 'exec_command',
      input: { cmd: 'bun test test/streaming', yield_time_ms: 120000 },
    }],
  })
  expect(stripAssistantToolCallMarkers('Confirmed.\nTOOL: {"raw":true}\nTOOL_CALL_ID: call_1\nNext.'))
    .toBe('Confirmed.\nNext.')
})
