// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { OutboundMessageRegistry, registerSendMessageTool, sendMessage } from '../src/tools/sendMessage.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

function call(arguments_: JsonObject): ToolCall {
  return { id: crypto.randomUUID(), type: 'function', function: { name: 'send_message', arguments: arguments_ } }
}

test('outbound message registry validates calls and normalizes platform routing', async () => {
  const registry = new OutboundMessageRegistry()
  const calls: unknown[] = []
  registry.register('Telegram', async (platform, recipient, payload, context) => {
    calls.push({ platform, recipient, payload, context })
    return { ok: true, message_id: '42' }
  })

  expect(await sendMessage({ platform: '', recipient: 'chat', text: 'hi' }, registry)).toEqual({
    ok: false,
    error: 'platform required',
  })
  expect(await sendMessage({ platform: 'missing', recipient: 'chat', text: 'hi' }, registry)).toEqual({
    ok: false,
    error: 'unknown platform: missing',
  })
  expect(await sendMessage({
    platform: 'TELEGRAM',
    recipient: 'chat',
    text: 'hi',
    files: ['report.txt'],
    replyTo: '41',
  }, registry, { metadata: { source: 'test' } })).toEqual({ ok: true, message_id: '42' })
  expect(calls).toEqual([expect.objectContaining({
    platform: 'telegram',
    recipient: 'chat',
    payload: { text: 'hi', files: ['report.txt'], replyTo: '41' },
  })])
})

test('send_message tool exposes the dispatcher without a channel-specific implementation', async () => {
  const dispatcher = new OutboundMessageRegistry()
  dispatcher.register('discord', (_platform, recipient, payload) => ({ ok: true, recipient, text: payload.text }))
  const tools = new ToolRegistry()
  registerSendMessageTool(tools, { registry: dispatcher })
  const output = JSON.parse(await tools.execute(call({
    platform: 'discord',
    recipient: 'room',
    text: 'native Bun message',
  }), { metadata: {} }))
  expect(output).toEqual({ ok: true, recipient: 'room', text: 'native Bun message' })
})
