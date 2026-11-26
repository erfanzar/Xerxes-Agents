// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { sessionHistoryActions } from '../src/daemon/historyPage.js'
import { historyBlocks } from '../src/desktop/renderer/history.js'
import { BlockBuilder } from '../src/desktop/renderer/blocks.js'

test('paged history has unique stable keys and does not replace live streaming output', () => {
  const ids = new Map<string, number>()
  const page = (start: number) => ({ before: null, has_more: false, actions: Array.from({ length: 100 }, (_, i) => ({ id: String(start + i), messages: [{ role: 'user', content: `Action ${start + i}` }], executions: [], thinking: [] })) })
  const tail = historyBlocks(page(100), ids)
  const builder = new BlockBuilder()
  builder.reset(tail)
  builder.push('text_part', { text: 'Still streaming' })
  builder.prepend(historyBlocks(page(0), ids))
  builder.push('text_part', { text: ' after paging' })
  const blocks = builder.snapshot(true)
  expect(blocks).toHaveLength(201)
  expect(new Set(blocks.map(block => block.id)).size).toBe(201)
  expect(blocks.at(-1)).toMatchObject({ text: 'Still streaming after paging', streaming: true })
  expect(historyBlocks(page(100), ids).map(block => block.id)).toEqual(tail.map(block => block.id))
})

test('flat assistant turns place thinking and tool calls above the final message in both daemon and legacy-remote history replay', () => {
  const actions = sessionHistoryActions({
    messages: [
      { role: 'user', content: 'Deploy esurge to all 128 hosts' },
      {
        role: 'assistant',
        content: 'Deployment Summary: esurge started cleanly.',
        thinking: 'Checking TPU slice health first.',
        tool_calls: [
          { id: 'c1', type: 'function', function: { name: 'exec_command', arguments: '{"cmd":"eray tpu status"}' } },
          { id: 'c2', type: 'function', function: { name: 'read_file', arguments: '{"path":"deploy.yaml"}' } },
        ],
      },
      { role: 'tool', tool_call_id: 'c1', content: '128/128 ready' },
      { role: 'tool', tool_call_id: 'c2', content: 'port: 8000' },
    ],
    toolExecutions: [
      { toolCallId: 'c1', name: 'exec_command', inputs: { cmd: 'eray tpu status' }, result: '128/128 ready', permitted: true, durationMs: 250 },
      { toolCallId: 'c2', name: 'read_file', inputs: { path: 'deploy.yaml' }, result: 'port: 8000', permitted: true, durationMs: 40 },
    ],
    thinkingContent: [],
  })

  expect(actions.map(action => action.id)).toEqual(['0:message', '1:thinking', '1:tool:0', '1:tool:1', '1:message'])
  const fromNewDaemon = historyBlocks({ before: null, has_more: false, actions })
  expect(fromNewDaemon.map(block => block.kind)).toEqual(['user', 'thinking', 'tools', 'tools', 'agent'])
  expect(fromNewDaemon.at(-1)).toMatchObject({ kind: 'agent', text: 'Deployment Summary: esurge started cleanly.' })

  // Older remote daemon emitting 1:message (thinking + final text) BEFORE 1:tool:* still replays tools above the final answer.
  const legacyPage = {
    before: null,
    has_more: false,
    actions: [
      { id: '0:message', messages: [{ role: 'user', content: 'Deploy esurge' }], executions: [], thinking: [] },
      { id: '1:message', messages: [{ role: 'assistant', content: 'Deployment Summary: all 128 hosts live.' }], executions: [], thinking: ['Planning rollout'] },
      {
        id: '1:tool:0',
        messages: [
          { role: 'assistant', content: '', tool_calls: [{ id: 'c1', type: 'function', function: { name: 'exec_command', arguments: '{"cmd":"deploy"}' } }] },
          { role: 'tool', tool_call_id: 'c1', content: 'ok' },
        ],
        executions: [{ toolCallId: 'c1', name: 'exec_command', inputs: { cmd: 'deploy' }, result: 'ok', permitted: true, durationMs: 120 }],
        thinking: [],
      },
      {
        id: '1:tool:1',
        messages: [
          { role: 'assistant', content: '', tool_calls: [{ id: 'c2', type: 'function', function: { name: 'read_file', arguments: '{"path":"logs"}' } }] },
          { role: 'tool', tool_call_id: 'c2', content: 'healthy' },
        ],
        executions: [{ toolCallId: 'c2', name: 'read_file', inputs: { path: 'logs' }, result: 'healthy', permitted: true, durationMs: 60 }],
        thinking: [],
      },
    ],
  }
  const legacyIds = new Map<string, number>()
  const reordered = historyBlocks(legacyPage, legacyIds)
  expect(reordered.map(block => block.kind)).toEqual(['user', 'thinking', 'tools', 'tools', 'agent'])
  expect(reordered[1]).toMatchObject({ kind: 'thinking', text: 'Planning rollout' })
  expect(reordered.at(-1)).toMatchObject({ kind: 'agent', text: 'Deployment Summary: all 128 hosts live.' })
  // Deterministic negative IDs remain stable across repeated pages.
  expect(historyBlocks(legacyPage, legacyIds).map(block => block.id)).toEqual(reordered.map(block => block.id))
})

test('multi-step turns with a later assistant reply keep intermediate narration before its tools', () => {
  const actions = sessionHistoryActions({
    messages: [
      { role: 'user', content: 'Check logs and summarize' },
      {
        role: 'assistant',
        content: 'Checking logs now.',
        tool_calls: [{ id: 'c1', type: 'function', function: { name: 'exec_command', arguments: '{"cmd":"tail -n 20"}' } }],
      },
      { role: 'tool', tool_call_id: 'c1', content: 'all green' },
      { role: 'assistant', content: 'Logs are clean.' },
    ],
    toolExecutions: [{ toolCallId: 'c1', name: 'exec_command', inputs: { cmd: 'tail -n 20' }, result: 'all green', permitted: true, durationMs: 50 }],
    thinkingContent: [],
  })
  expect(actions.map(action => action.id)).toEqual(['0:message', '1:message', '1:tool:0', '3:message'])
  const blocks = historyBlocks({ before: null, has_more: false, actions })
  expect(blocks.map(block => block.kind)).toEqual(['user', 'agent', 'tools', 'agent'])
  expect(blocks[1]).toMatchObject({ kind: 'agent', text: 'Checking logs now.' })
  expect(blocks[3]).toMatchObject({ kind: 'agent', text: 'Logs are clean.' })
})
