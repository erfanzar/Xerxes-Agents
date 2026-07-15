// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  WIRE_EVENT_NAME_BY_INTERNAL,
  WireEventValidationError,
  createWireEvent,
  eventFromDict,
  eventFromJsonRpcMessage,
  eventToDict,
  jsonRpcEventMessage,
  toInternalEventName,
  toKimiEventName,
} from '../src/streaming/wireEvents.js'

test('the focused codec freezes aliases for every event emitted by the native stream adapter', () => {
  expect(Object.keys(WIRE_EVENT_NAME_BY_INTERNAL).sort()).toEqual([
    'approval_request',
    'notification',
    'status_update',
    'text_part',
    'think_part',
    'tool_call',
    'tool_result',
  ])
  for (const [internal, kimi] of Object.entries(WIRE_EVENT_NAME_BY_INTERNAL)) {
    expect(toKimiEventName(internal)).toBe(kimi)
    expect(toInternalEventName(kimi)).toBe(internal)
  }
})

test('text and thinking deltas round-trip between native and Kimi spellings', () => {
  const text = eventFromDict({ type: 'TextPart', payload: { text: 'hello' } })
  const thinking = eventFromDict({ type: 'think_part', think: 'plan first' })

  expect(text).toEqual({ event_type: 'text_part', payload: { text: 'hello' } })
  expect(thinking).toEqual({ event_type: 'think_part', payload: { think: 'plan first' } })
  expect(eventToDict(text)).toEqual({ type: 'TextPart', payload: { text: 'hello' } })
  expect(eventToDict(thinking)).toEqual({ type: 'ThinkPart', payload: { think: 'plan first' } })
})

test('tool, approval, and result events retain the extension fields emitted by AgentTurnRunner', () => {
  const call = eventFromDict({
    type: 'tool_call',
    payload: { id: 'call-1', tool_call_id: 'call-1', name: 'Read', arguments: '{"path":"README.md"}' },
  })
  const approval = eventFromDict({
    type: 'ApprovalRequest',
    payload: {
      id: 'request-1', request_id: 'request-1', action: 'Read', description: 'Read the project guide.',
      name: 'Read', tool_name: 'Read', inputs: { path: 'README.md' },
    },
  })
  const result = eventFromDict({
    type: 'ToolResult',
    payload: {
      tool_call_id: 'call-1', return_value: 'contents', result: 'contents', permitted: true, duration_ms: 4.5,
      display_blocks: [{ type: 'brief', body: 'Read README.md' }],
    },
  })

  expect(call).toMatchObject({ event_type: 'tool_call', payload: { tool_call_id: 'call-1' } })
  expect(approval).toMatchObject({ event_type: 'approval_request', payload: { inputs: { path: 'README.md' } } })
  expect(result).toMatchObject({ event_type: 'tool_result', payload: { permitted: true, duration_ms: 4.5 } })
  expect(eventToDict(result)).toMatchObject({ type: 'ToolResult', payload: { display_blocks: [{ type: 'brief' }] } })
})

test('status and notification payload variants stay flexible while known fields are checked', () => {
  const status = eventFromDict({
    type: 'StatusUpdate',
    payload: { model: 'gpt-4o', context_tokens: 20, usage: { inputTokens: 12 }, cache_read_tokens: 3 },
  })
  const notification = eventFromDict({
    type: 'notification',
    payload: { level: 'warning', message: 'retrying', retry: { attempt: 2 }, custom: 'retained' },
  })

  expect(status).toMatchObject({ event_type: 'status_update', payload: { cache_read_tokens: 3 } })
  expect(notification).toMatchObject({
    event_type: 'notification',
    payload: { retry: { attempt: 2 }, custom: 'retained' },
  })
  expect(eventToDict(notification)).toEqual({
    type: 'Notification',
    payload: { level: 'warning', message: 'retrying', retry: { attempt: 2 }, custom: 'retained' },
  })
})

test('unknown event names remain inspectable and malformed known payloads fail explicitly', () => {
  expect(eventFromDict({ type: 'FutureEvent', payload: { revision: 2 } })).toEqual({
    event_type: 'generic', raw: { type: 'FutureEvent', payload: { revision: 2 } },
  })
  expect(() => eventFromDict({ type: 'TextPart', payload: { text: 12 } })).toThrow(WireEventValidationError)
  expect(() => eventFromDict({
    type: 'ToolResult',
    payload: { tool_call_id: 'call-1', duration_ms: NaN, display_blocks: [] },
  })).toThrow('finite number')
  expect(() => eventFromDict({
    type: 'ApprovalRequest',
    payload: { action: 'Read', description: 'Read.' },
  })).toThrow('id or request_id')
  expect(() => eventFromDict(['not an event'])).toThrow('must be an object')
})

test('JSON-RPC envelopes use the same codec without owning a transport', () => {
  const message = jsonRpcEventMessage(createWireEvent('text_part', { text: 'native transport boundary' }))
  expect(message).toEqual({
    jsonrpc: '2.0', method: 'event', params: { type: 'TextPart', payload: { text: 'native transport boundary' } },
  })
  expect(eventFromJsonRpcMessage(message)).toEqual({
    event_type: 'text_part', payload: { text: 'native transport boundary' },
  })
  expect(() => eventFromJsonRpcMessage({ ...message, method: 'request' })).toThrow('must equal "event"')
})
