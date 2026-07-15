// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  WireEventEmitter,
  WireEventSerializationError,
  serializeWireFrame,
  toDaemonWireEventName,
  toKimiWireEventName,
  wireEventFrame,
  wireRequestFrame,
  type BridgeWireFrame,
  type WireEventSink,
} from '../src/bridge/wireEvents.js'
import type { JsonRpcPayload } from '../src/protocol/jsonRpc.js'

class RecordingSink implements WireEventSink {
  readonly frames: BridgeWireFrame[] = []

  emit(frame: BridgeWireFrame): void {
    this.frames.push(frame)
  }
}

function idFactory(): () => string {
  let id = 0
  return () => `${(++id).toString().padStart(32, '0')}`
}

function eventPayloads(sink: RecordingSink, type: string): JsonRpcPayload[] {
  return sink.frames.flatMap(frame => frame.method === 'event' && frame.params.type === type ? [frame.params.payload] : [])
}

test('wire frames use the shared daemon protocol by default and preserve opt-in Kimi names', () => {
  expect(toKimiWireEventName('text_part')).toBe('TextPart')
  expect(toDaemonWireEventName('QuestionRequest')).toBe('question_request')
  expect(wireEventFrame('TextPart', { text: 'hello' })).toEqual({
    jsonrpc: '2.0',
    method: 'event',
    params: { type: 'text_part', payload: { text: 'hello' } },
  })
  expect(wireEventFrame('text_part', { text: 'hello' }, 'kimi').params.type).toBe('TextPart')
  expect(wireRequestFrame('request-1', 'QuestionRequest', { questions: [] })).toEqual({
    id: 'request-1',
    jsonrpc: '2.0',
    method: 'request',
    params: { type: 'question_request', payload: { questions: [] } },
  })
  expect(serializeWireFrame(wireEventFrame('turn_end', {}))).toBe(
    '{"jsonrpc":"2.0","method":"event","params":{"type":"turn_end","payload":{}}}',
  )

  const circular: JsonRpcPayload = {}
  circular.self = circular
  expect(() => serializeWireFrame(wireEventFrame('notification', circular))).toThrow(WireEventSerializationError)
})

test('wire emitter suppresses split function tags and frames tools, requests, notifications, and host-supplied status', () => {
  const sink = new RecordingSink()
  const emitter = new WireEventEmitter({ idFactory: idFactory(), sink })

  emitter.emitText('before <function=call>{"hidden":true}')
  emitter.emitText('</function> after')
  emitter.emitText('  ')
  emitter.emitText('{"name":"Run","arguments":{}}')
  expect(eventPayloads(sink, 'text_part')).toEqual([{ text: 'before ' }, { text: ' after' }])

  const toolId = emitter.emitToolStart('', 'Read', { path: '/tmp/readme' })
  expect(toolId).toBe('call_000000000000')
  emitter.emitToolArgsPart('{"path":')
  emitter.emitToolResult({ returnValue: 'contents', durationMs: 12.5, permitted: false })
  expect(eventPayloads(sink, 'tool_call')).toEqual([{
    id: toolId,
    name: 'Read',
    arguments: '{"path":"/tmp/readme"}',
  }])
  expect(eventPayloads(sink, 'tool_call_part')).toEqual([{ arguments_part: '{"path":' }])
  expect(eventPayloads(sink, 'tool_result')).toEqual([{
    tool_call_id: toolId,
    return_value: 'contents',
    duration_ms: 12.5,
    display_blocks: [],
  }])

  const permissionId = emitter.emitPermissionRequest(toolId, 'Read', 'Read a file')
  const questionId = emitter.emitQuestionRequest([{ id: 'answer', question: 'Continue?', options: [] }])
  expect(permissionId).toBe('00000000000000000000000000000002')
  expect(questionId).toBe('00000000000000000000000000000003')
  expect(sink.frames.filter(frame => frame.method === 'request')).toEqual([
    {
      id: permissionId,
      jsonrpc: '2.0',
      method: 'request',
      params: {
        type: 'approval_request',
        payload: { id: permissionId, tool_call_id: toolId, action: 'Read', description: 'Read a file' },
      },
    },
    {
      id: questionId,
      jsonrpc: '2.0',
      method: 'request',
      params: {
        type: 'question_request',
        payload: { id: questionId, questions: [{ id: 'answer', question: 'Continue?', options: [] }] },
      },
    },
  ])

  emitter.emitNotification({
    id: 'notice-1',
    category: 'slash',
    type: 'result',
    severity: 'info',
    title: '',
    body: 'Saved.',
  })
  emitter.emitInitDone({ cwd: '/workspace', model: 'gpt-4o' })
  emitter.emitStatus({ context_tokens: 123, max_context: 456 })
  expect(eventPayloads(sink, 'notification')).toEqual([{
    id: 'notice-1',
    category: 'slash',
    type: 'result',
    severity: 'info',
    title: '',
    body: 'Saved.',
    payload: {},
  }])
  expect(eventPayloads(sink, 'init_done')).toEqual([{ cwd: '/workspace', model: 'gpt-4o' }])
  expect(eventPayloads(sink, 'status_update')).toEqual([{ context_tokens: 123, max_context: 456 }])
})

test('subagent summaries retain rolling previews and nest inner tool events under the active parent', () => {
  const sink = new RecordingSink()
  const emitter = new WireEventEmitter({ idFactory: idFactory(), sink, subagentPreviewChars: 10 })
  emitter.emitToolStart('parent-tool', 'AgentTool', {})

  expect(emitter.emitSubagentSummary('agent_spawn', {
    agent_name: 'researcher',
    depth: 2,
    prompt: 'Find the important behavior.',
    task_id: 'task-abcdefghijk',
  })).toBeTrue()
  emitter.emitSubagentSummary('agent_text', { agent_name: 'researcher', task_id: 'task-abcdefghijk', text: 'alpha beta ' })
  emitter.emitSubagentSummary('agent_text', { agent_name: 'researcher', task_id: 'task-abcdefghijk', text: 'gamma delta' })
  emitter.emitSubagentSummary('agent_thinking', { agent_name: 'researcher', task_id: 'task-abcdefghijk', text: 'reasoning' })
  emitter.emitSubagentSummary('agent_tool_start', {
    agent_type: 'researcher',
    inputs: { path: '/workspace/a.ts' },
    task_id: 'task-abcdefghijk',
    tool_name: 'Read',
  })
  emitter.emitSubagentSummary('agent_tool_end', {
    agent_type: 'researcher',
    duration_ms: 12.7,
    permitted: true,
    result: 'file body',
    task_id: 'task-abcdefghijk',
    tool_name: 'Read',
  })
  emitter.emitSubagentSummary('agent_done', { agent_name: 'researcher', task_id: 'task-abcdefghijk' })

  const nested = eventPayloads(sink, 'subagent_event')
  expect(nested).toHaveLength(2)
  expect(nested[0]).toMatchObject({
    parent_tool_call_id: 'parent-tool',
    agent_id: 'task-abcdefghijk',
    subagent_type: 'researcher',
    event: { type: 'ToolCall', payload: { name: 'Read', arguments: '{"path":"/workspace/a.ts"}' } },
  })
  const innerId = ((nested[0]?.event as { payload: { id: string } }).payload.id)
  expect(nested[1]).toMatchObject({
    parent_tool_call_id: 'parent-tool',
    event: { type: 'ToolResult', payload: { tool_call_id: innerId, return_value: 'file body', display_blocks: [] } },
  })

  const previews = eventPayloads(sink, 'notification').filter(payload => payload.category === 'subagent_stream')
  expect(previews.some(payload => typeof payload.body === 'string' && payload.body.startsWith('…'))).toBeTrue()
  expect(previews).toEqual(expect.arrayContaining([
    expect.objectContaining({ body: 'starting…', payload: { task_id: 'task-abcdefghijk', label: 'researcher#task-abc…' } }),
    expect.objectContaining({
      body: 'reasoning',
      payload: { task_id: 'task-abcdefghijk', label: 'researcher#task-abc… (thinking)' },
    }),
    expect.objectContaining({ body: '◐ Read(/workspace/a.ts)' }),
    expect.objectContaining({ body: '✓ Read — 13ms' }),
    expect.objectContaining({ body: '' }),
  ]))

  const uncorrelated = new WireEventEmitter({ idFactory: idFactory(), sink: new RecordingSink() })
  expect(uncorrelated.emitSubagentToolEvent('orphan', 'researcher', { tool_name: 'Read' }, 'start')).toBeFalse()
  expect(uncorrelated.emitSubagentSummary('agent_unknown', {})).toBeFalse()
})

test('Kimi mode converts emitted event names without changing the JSON-RPC event envelope', () => {
  const sink = new RecordingSink()
  const emitter = new WireEventEmitter({ eventNameStyle: 'kimi', idFactory: idFactory(), sink })
  emitter.emitThink('planning')
  emitter.emitTurnEnd()
  expect(sink.frames).toEqual([
    { jsonrpc: '2.0', method: 'event', params: { type: 'ThinkPart', payload: { think: 'planning' } } },
    { jsonrpc: '2.0', method: 'event', params: { type: 'TurnEnd', payload: {} } },
  ])
})
