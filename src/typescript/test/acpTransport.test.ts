// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { AcpServer } from '../src/acp/server.js'
import { StdioJsonRpcServer, serveACPStdio } from '../src/acp/transport.js'

test('ACP stdio transport preserves NDJSON framing, aliases, streamed updates, and final result', async () => {
  const server = new AcpServer({
    promptHandler: async ({ text, emit }) => {
      if (emit) {
        await emit({ kind: 'text_delta', text: 'hello ' })
        await emit({ kind: 'text_delta', text })
        await emit({ kind: 'turn_end', output_tokens: 2, model: 'gpt-4o' })
      }
      return { ok: true, echo: text, output_tokens: 2 }
    },
    toolListProvider: () => [{
      type: 'function',
      function: { name: 'echo', description: 'Echo text.', parameters: { type: 'object' } },
    }],
    modelListProvider: () => [{ id: 'gpt-4o', name: 'GPT-4o' }],
  })
  const sessionId = String(server.openSession('/tmp').session_id)
  const input = readableChunks([
    '{"jsonrpc":"2.0","id":1,"method":"initial',
    'ize","params":{"client_info":{"name":"editor"}}}\n',
    '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}\n',
    `${JSON.stringify({ jsonrpc: '2.0', id: 3, method: 'session/prompt', params: { session_id: sessionId, text: 'world' } })}\n`,
  ])
  const output: string[] = []

  await serveACPStdio(server, input, line => {
    output.push(line)
  })

  const frames = output.map(line => JSON.parse(line) as Record<string, unknown>)
  const byId = new Map(frames.filter(frame => typeof frame.id === 'number').map(frame => [frame.id, frame]))
  expect(byId.get(1)?.result).toMatchObject({ server_name: 'xerxes', capabilities: { protocol_version: '0.9' } })
  expect(byId.get(2)?.result).toEqual([{
    type: 'function',
    function: { name: 'echo', description: 'Echo text.', parameters: { type: 'object' } },
  }])
  expect(byId.get(3)?.result).toEqual({ ok: true, echo: 'world', output_tokens: 2 })

  const updates = frames.filter(frame => frame.method === 'session/update')
  expect(updates.map(frame => (frame.params as { event: { kind: string; text?: string } }).event.kind)).toEqual([
    'text_delta', 'text_delta', 'turn_end',
  ])
  expect(updates.map(frame => (frame.params as { event: { text?: string } }).event.text).filter(Boolean)).toEqual(['hello ', 'world'])
  expect(updates.every(frame => (frame.params as { session_id: string; request_id: number }).session_id === sessionId)).toBe(true)
  expect(updates.every(frame => (frame.params as { request_id: number }).request_id === 3)).toBe(true)
})

test('ACP stdio transport reports parse and method errors but does not answer notifications', async () => {
  const server = new AcpServer({ promptHandler: () => ({ ok: true }) })
  const output: string[] = []
  const input = readableChunks([
    '{not json}\n',
    '{"jsonrpc":"2.0","id":1,"method":"does_not_exist","params":{}}\n',
    '{"jsonrpc":"2.0","method":"does_not_exist","params":{}}\n',
    '{"jsonrpc":"2.0","id":2,"method":"initialize","params":{}}\n',
  ])

  await new StdioJsonRpcServer(server).serve(input, line => {
    output.push(line)
  })

  const frames = output.map(line => JSON.parse(line) as Record<string, unknown>)
  expect(frames.filter(frame => frame.error).map(frame => (frame.error as { code: number }).code)).toEqual([-32700, -32601])
  expect(frames.some(frame => frame.id === 2 && frame.result)).toBe(true)
  expect(frames.filter(frame => frame.id === undefined)).toHaveLength(0)
})

function readableChunks(chunks: readonly string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder()
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk))
      }
      controller.close()
    },
  })
}
