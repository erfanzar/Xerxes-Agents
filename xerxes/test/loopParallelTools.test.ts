// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import type { LlmClient, LlmDelta } from '../src/llms/client.js'
import { createAgentState, type StreamEvent } from '../src/streaming/events.js'
import { groupToolDecisions, runTurn } from '../src/streaming/loop.js'
import type { ToolCall, ToolDefinition } from '../src/types/toolCalls.js'

const TOOLS: ToolDefinition[] = ['Read', 'Write'].map(name => ({
  type: 'function',
  function: { name, description: name, parameters: { type: 'object', properties: {} } },
}))

function call(id: string, name: string): ToolCall {
  return { id, type: 'function', function: { name, arguments: {} } }
}

/** One provider round emitting the given calls, then a plain text answer. */
function client(calls: readonly ToolCall[]): LlmClient {
  let round = 0
  return {
    async *stream(): AsyncGenerator<LlmDelta> {
      if (round++ === 0) {
        yield { toolCalls: [...calls] }
        return
      }
      yield { content: 'done' }
    },
  }
}

const SAFE = { concurrencySafe: true, interruptBehavior: 'cancel' as const }
const UNSAFE = { concurrencySafe: false, interruptBehavior: 'cancel' as const }

test('consecutive concurrency-safe calls run together but report in model-emitted order', async () => {
  const calls = [call('a', 'Read'), call('b', 'Read'), call('c', 'Read')]
  const started: string[] = []
  let peakConcurrency = 0
  let live = 0
  // Deliberately inverted durations: the last call finishes first, so a loop
  // that reported in completion order would visibly reorder the transcript.
  const delays: Record<string, number> = { a: 40, b: 20, c: 1 }

  const events: StreamEvent[] = []
  for await (const event of runTurn(
    { model: 'm', state: createAgentState([]), userMessage: 'go', tools: TOOLS },
    {
      llm: client(calls),
      capabilities: () => SAFE,
      toolExecutor: {
        async execute(toolCall) {
          started.push(toolCall.id)
          live += 1
          peakConcurrency = Math.max(peakConcurrency, live)
          await Bun.sleep(delays[toolCall.id] ?? 0)
          live -= 1
          return `result-${toolCall.id}`
        },
      },
    },
  )) events.push(event)

  expect(peakConcurrency).toBe(3)
  expect(started).toEqual(['a', 'b', 'c'])
  const ends = events.flatMap(event => event.type === 'tool_end' ? [event.result.toolCallId] : [])
  expect(ends).toEqual(['a', 'b', 'c'])
  const starts = events.flatMap(event => event.type === 'tool_start' ? [event.call.id] : [])
  expect(starts).toEqual(['a', 'b', 'c'])
})

test('capability refinement receives each effective call arguments', async () => {
  const calls: ToolCall[] = [
    { id: 'safe', type: 'function', function: { name: 'Read', arguments: { mode: 'read' } } },
    { id: 'unsafe', type: 'function', function: { name: 'Read', arguments: { mode: 'write' } } },
  ]
  const observed: Array<Readonly<Record<string, unknown>> | undefined> = []

  for await (const _ of runTurn(
    { agentId: 'worker', model: 'm', state: createAgentState([]), userMessage: 'go', tools: TOOLS },
    {
      llm: client(calls),
      capabilities: (_name, _agentId, args) => {
        observed.push(args)
        return args?.mode === 'read' ? SAFE : UNSAFE
      },
      toolExecutor: { execute: async () => 'ok' },
    },
  )) { /* drain */ }

  expect(observed).toContainEqual({ mode: 'read' })
  expect(observed).toContainEqual({ mode: 'write' })
})

test('an unsafe call is a barrier: neither side overlaps it', async () => {
  const calls = [call('a', 'Read'), call('w', 'Write'), call('b', 'Read')]
  const order: string[] = []
  let live = 0
  let peak = 0

  for await (const _ of runTurn(
    { model: 'm', state: createAgentState([]), userMessage: 'go', tools: TOOLS },
    {
      llm: client(calls),
      capabilities: name => (name === 'Write' ? UNSAFE : SAFE),
      toolExecutor: {
        async execute(toolCall) {
          live += 1
          peak = Math.max(peak, live)
          order.push(toolCall.id)
          await Bun.sleep(5)
          live -= 1
          return 'ok'
        },
      },
    },
  )) { /* drain */ }

  // Never more than one at a time: each read is separated from the other by the
  // write, so no run of length > 1 can form.
  expect(peak).toBe(1)
  expect(order).toEqual(['a', 'w', 'b'])
})

test('a failing member does not fail its siblings and every call still gets a result', async () => {
  const calls = [call('a', 'Read'), call('b', 'Read'), call('c', 'Read')]
  const events: StreamEvent[] = []
  for await (const event of runTurn(
    { model: 'm', state: createAgentState([]), userMessage: 'go', tools: TOOLS },
    {
      llm: client(calls),
      capabilities: () => SAFE,
      toolExecutor: {
        async execute(toolCall) {
          if (toolCall.id === 'b') throw new Error('b exploded')
          return `result-${toolCall.id}`
        },
      },
    },
  )) events.push(event)

  const ends = events.flatMap(event => event.type === 'tool_end' ? [event.result] : [])
  expect(ends.map(result => result.toolCallId)).toEqual(['a', 'b', 'c'])
  expect(ends[0]?.result).toBe('result-a')
  expect(ends[1]?.result).toContain('exploded')
  expect(ends[2]?.result).toBe('result-c')
})

test('a parallel round writes exactly one tool message per call, in order', async () => {
  const calls = [call('a', 'Read'), call('b', 'Read')]
  const state = createAgentState([])
  for await (const _ of runTurn(
    { model: 'm', state, userMessage: 'go', tools: TOOLS },
    {
      llm: client(calls),
      capabilities: () => SAFE,
      toolExecutor: { execute: async toolCall => `r-${toolCall.id}` },
    },
  )) { /* drain */ }

  const toolMessages = state.messages.filter(message => message.role === 'tool')
  expect(toolMessages).toHaveLength(2)
  expect(toolMessages.map(message => message.role === 'tool' ? message.tool_call_id : '')).toEqual(['a', 'b'])
})

test('grouping never reorders across a barrier and honors the concurrency cap', () => {
  const decisions = [
    { call: call('1', 'Read'), kind: 'allowed' },
    { call: call('2', 'Read'), kind: 'allowed' },
    { call: call('3', 'Write'), kind: 'allowed' },
    { call: call('4', 'Read'), kind: 'allowed' },
    { call: call('5', 'Read'), kind: 'denied' },
  ]
  const groups = groupToolDecisions(decisions, c => ({ concurrencySafe: c.function.name === 'Read' }), 2)
  expect(groups.map(group => group.map(decision => decision.call.id))).toEqual([
    ['1', '2'], ['3'], ['4'], ['5'],
  ])

  const capped = groupToolDecisions(
    Array.from({ length: 5 }, (_, index) => ({ call: call(String(index), 'Read'), kind: 'allowed' })),
    () => ({ concurrencySafe: true }),
    2,
  )
  expect(capped.map(group => group.length)).toEqual([2, 2, 1])
})
