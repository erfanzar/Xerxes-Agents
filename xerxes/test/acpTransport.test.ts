// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { AcpServer } from '../src/acp/server.js'
import { AcpAgentRunner } from '../src/acp/runner.js'
import { MAX_ACP_FRAME_BYTES, StdioJsonRpcServer, serveACPStdio } from '../src/acp/transport.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'

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

test('ACP respond_permission requires a strictly boolean allow parameter', async () => {
  const server = new AcpServer({ promptHandler: () => ({ ok: true }) })
  const output: string[] = []
  const input = readableChunks([
    '{"jsonrpc":"2.0","id":1,"method":"respond_permission","params":{"permission_id":"p1","allow":"false"}}\n',
    '{"jsonrpc":"2.0","id":2,"method":"respond_permission","params":{"permission_id":"p1","allow":1}}\n',
    '{"jsonrpc":"2.0","id":3,"method":"respond_permission","params":{"permission_id":"p1","allow":false}}\n',
    '{"jsonrpc":"2.0","id":4,"method":"permission/respond","params":{"permission_id":"p1","allow":true}}\n',
  ])

  await serveACPStdio(server, input, line => {
    output.push(line)
  })

  const frames = output.map(line => JSON.parse(line) as Record<string, unknown>)
  const byId = new Map(frames.map(frame => [frame.id, frame]))
  expect((byId.get(1)?.error as { code: number } | undefined)?.code).toBe(-32602)
  expect((byId.get(2)?.error as { code: number } | undefined)?.code).toBe(-32602)
  expect(byId.get(3)?.result).toEqual({ ok: false })
  expect(byId.get(4)?.result).toEqual({ ok: false })
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

test('ACP EOF aborts active runner prompts before awaiting transport workers', async () => {
  const started = Promise.withResolvers<void>()
  const llm: LlmClient = {
    async *stream(_request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
      started.resolve()
      await new Promise<void>((resolve) => {
        if (signal?.aborted) return resolve()
        signal?.addEventListener('abort', () => resolve(), { once: true })
      })
      throw signal?.reason ?? new Error('expected ACP shutdown cancellation')
    },
  }
  const runner = new AcpAgentRunner({ llm, model: 'test-model' })
  const server = new AcpServer({ runner })
  const sessionId = String(server.openSession('/tmp').session_id)
  const output: string[] = []
  const serving = serveACPStdio(server, readableChunks([
    `${JSON.stringify({ jsonrpc: '2.0', id: 1, method: 'session/prompt', params: { session_id: sessionId, text: 'wait' } })}\n`,
  ]), line => {
    output.push(line)
  })

  await started.promise
  await expect(Promise.race([
    serving.then(() => 'done'),
    Bun.sleep(500).then(() => 'timeout'),
  ])).resolves.toBe('done')
  expect(output.some(line => line.includes('ACP prompt cancelled'))).toBe(true)
})

test('ACP stdio transport rejects an unterminated oversized frame and stops reading', async () => {
  const server = new AcpServer({ promptHandler: () => ({ ok: true }) })
  const output: string[] = []
  // A peer that never sends `\n` must not grow the read buffer without bound.
  const input = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(`{"jsonrpc":"2.0","id":1,"padding":"${'x'.repeat(MAX_ACP_FRAME_BYTES)}`))
      // The stream never closes and never terminates the frame.
    },
  })

  await expect(Promise.race([
    serveACPStdio(server, input, line => {
      output.push(line)
    }).then(() => 'done'),
    Bun.sleep(2_000).then(() => 'timeout'),
  ])).resolves.toBe('done')

  const frames = output.map(line => JSON.parse(line) as { error?: { code: number; message: string } })
  expect(frames).toHaveLength(1)
  expect(frames[0]?.error?.code).toBe(-32600)
  expect(frames[0]?.error?.message).toContain('maximum size')
}, 10_000)

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
