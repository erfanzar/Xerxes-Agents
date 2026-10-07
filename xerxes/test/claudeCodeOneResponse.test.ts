// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ClaudeCodeClient, type ClaudeCodeLauncher } from '../src/llms/claudeCode.js'
import type { LlmDelta } from '../src/llms/client.js'

/**
 * When a response stops at the output limit, Claude Code asks the model to
 * continue and streams that as a further response in the same run. One run
 * produced 135,461 output tokens and 1,176 tool calls that way. A request is
 * one model response: the adapter ends with the first.
 */

const event = (inner: Record<string, unknown>) => JSON.stringify({ type: 'stream_event', event: inner })
const text = (value: string) => event({ type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: value } })
const call = (offset: number) => `<function=read_file>{"path":"core.mm","offset":${offset}}</function>\n`
const response = (body: string[], stopReason: string) => [
  event({ type: 'message_start', message: { usage: { input_tokens: 2, cache_read_input_tokens: 400_000, output_tokens: 1 } } }),
  ...body.map(text),
  event({ type: 'message_delta', delta: { stop_reason: stopReason }, usage: { output_tokens: 32_000 } }),
  event({ type: 'message_stop' }),
]

function launcher(lines: string[]): { launch: ClaudeCodeLauncher; killed: () => number } {
  let killed = 0
  return {
    launch: () => ({
      lines: (async function* () { for (const line of lines) yield line })(),
      exited: Promise.resolve(0),
      stderr: Promise.resolve(''),
      kill: () => { killed += 1 },
    }),
    killed: () => killed,
  }
}

async function run(lines: string[]) {
  const fake = launcher(lines)
  const client = new ClaudeCodeClient({ executable: '/bin/claude', launch: fake.launch, workingDirectory: '/tmp', environment: { PATH: '/bin' } })
  const deltas: LlmDelta[] = []
  const tools = [{ type: 'function' as const, function: { name: 'read_file', description: 'Read', parameters: { type: 'object', properties: { path: { type: 'string' }, offset: { type: 'number' } } } } }]
  for await (const delta of client.stream({ model: 'claude-code/opus', messages: [{ role: 'user', content: 'read it' }], tools })) deltas.push(delta)
  return { deltas, killed: fake.killed() }
}

test('a response cut at the output limit is not extended by Claude Code\'s automatic continuation', async () => {
  const first = response(['Reading the file.\n', call(0), call(40), '<function=read_file>{"path":"core.mm","off'], 'max_tokens')
  const continuation = response([call(80), call(120), call(160)], 'max_tokens')
  const { deltas, killed } = await run([...first, ...continuation, JSON.stringify({ type: 'result', is_error: false, usage: { input_tokens: 6, output_tokens: 135_461 } })])
  const calls = deltas.flatMap(delta => delta.toolCalls ?? [])
  // Only the first response's complete calls; its cut-off last call is dropped.
  expect(calls.map(item => item.function.arguments)).toEqual([{ path: 'core.mm', offset: 0 }, { path: 'core.mm', offset: 40 }])
  expect(killed).toBeGreaterThan(0)
  // Usage is the first response's, not the stitched run's.
  expect(deltas.find(delta => delta.usage)?.usage?.outputTokens).toBe(32_000)
})

test('a single complete response is read to its end as before', async () => {
  const only = response(['Two reads.\n', call(0), call(40)], 'end_turn')
  const { deltas } = await run([...only, JSON.stringify({ type: 'result', is_error: false, usage: { input_tokens: 2, output_tokens: 90 } })])
  expect(deltas.flatMap(delta => delta.toolCalls ?? [])).toHaveLength(2)
  expect(deltas.find(delta => delta.usage)?.usage?.outputTokens).toBe(90)
})

test('the call block\'s close ends the reply, as the API\'s stop sequence would', async () => {
  const { FunctionCallExtractor } = await import('../src/llms/claudeCode.js')
  const extractor = new FunctionCallExtractor()
  const visible = extractor.push('Pinning the dylib, then reading.\n<tool_calls>\n' + call(0) + call(40) + '</tool_calls>\n' + call(80) + call(120))
  expect(extractor.done).toBe(true)
  expect(extractor.calls.map(item => item.arguments)).toEqual([{ path: 'core.mm', offset: 0 }, { path: 'core.mm', offset: 40 }])
  expect(visible.trim()).toBe('Pinning the dylib, then reading.')
})

test('the block close stops a live run at once, and calls without a block still work', async () => {
  const runaway = response(['Reading.\n<tool_calls>\n', call(0), '</tool_calls>\n', call(40), call(80)], 'end_turn')
  const { deltas, killed } = await run([...runaway, JSON.stringify({ type: 'result', is_error: false, usage: { input_tokens: 2, output_tokens: 50 } })])
  expect(deltas.flatMap(delta => delta.toolCalls ?? []).map(item => item.function.arguments)).toEqual([{ path: 'core.mm', offset: 0 }])
  expect(killed).toBeGreaterThan(0)
  const unwrapped = response(['Reading.\n', call(0), call(40)], 'end_turn')
  const plain = await run([...unwrapped, JSON.stringify({ type: 'result', is_error: false, usage: { input_tokens: 2, output_tokens: 50 } })])
  expect(plain.deltas.flatMap(delta => delta.toolCalls ?? [])).toHaveLength(2)
})

test('the protocol asks for one closed block of calls', async () => {
  const { claudeCodeToolProtocol } = await import('../src/llms/claudeCode.js')
  const protocol = claudeCodeToolProtocol([{ type: 'function', function: { name: 'read_file', description: 'Read', parameters: { type: 'object', properties: {} } } }], 'auto')
  expect(protocol).toContain('<tool_calls>\n<function=TOOL_NAME>')
  expect(protocol).toContain('</tool_calls>\nThe arguments are one JSON object. Put every call this step needs in that one block; closing it ends your reply.')
})

test('with thinking on, a small reply cap does not starve the model: its own reported limit stands', async () => {
  const { claudeCodeEnvironment } = await import('../src/llms/claudeCode.js')
  const cap = (maxTokens: number | undefined, thinkingOff: boolean, limit?: number) => claudeCodeEnvironment({}, maxTokens, thinkingOff, limit).CLAUDE_CODE_MAX_OUTPUT_TOKENS
  // A compaction summary's 2,048 with opus thinking: the model's 32,000 applies.
  expect(cap(2_048, false, 32_000)).toBe('32000')
  // Limit not reported yet: Claude Code's own default, never the small cap.
  expect(cap(2_048, false)).toBeUndefined()
  // A cap at or above the model's limit is the caller asking for more room.
  expect(cap(64_000, false, 32_000)).toBe('64000')
  // Thinking off: the reply is all there is, so the cap is exact.
  expect(cap(2_048, true, 32_000)).toBe('2048')
  expect(cap(undefined, false, 32_000)).toBeUndefined()
})
