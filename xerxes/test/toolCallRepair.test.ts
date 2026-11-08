// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { createAgentState, type StreamEvent } from '../src/streaming/events.js'
import { runTurn } from '../src/streaming/loop.js'
import { repairExecCommandInputs, repairToolCall } from '../src/streaming/toolCallRepair.js'
import type { JsonObject, ToolCall, ToolDefinition } from '../src/types/toolCalls.js'

async function collect(events: AsyncIterable<StreamEvent>): Promise<StreamEvent[]> {
  const result: StreamEvent[] = []
  for await (const event of events) result.push(event)
  return result
}

test('a cmd of plain words becomes the program plus arguments, ahead of any args', () => {
  expect(repairExecCommandInputs({ cmd: 'wc -l', args: ['bank.csv', 'ledger.csv'] })).toEqual({ cmd: 'wc', args: ['-l', 'bank.csv', 'ledger.csv'] })
  expect(repairExecCommandInputs({ cmd: 'python3 -c', args: ['print(1)'] })).toEqual({ cmd: 'python3', args: ['-c', 'print(1)'] })
  expect(repairExecCommandInputs({ cmd: '  ls  -la  ' })).toEqual({ cmd: 'ls', args: ['-la'] })
  expect(repairExecCommandInputs({ cmd: ' ls ' })).toEqual({ cmd: 'ls' })
})

test('args sent as a JSON string of a string array becomes that array', () => {
  expect(repairExecCommandInputs({ cmd: 'python3', args: '["-c", "import csv\\nprint(1)"]' })).toEqual({ cmd: 'python3', args: ['-c', 'import csv\nprint(1)'] })
})

test('anything a shell would read differently is left for validation to reject', () => {
  const unchanged: JsonObject[] = [
    { cmd: "awk -F, 'NR>1 {print $2}' orders.csv" },
    { cmd: 'ls *.csv' },
    { cmd: 'cat ~/notes.txt' },
    { cmd: 'echo $HOME' },
    { cmd: 'wc -l | sort' },
    { cmd: 'FOO=1 env' },
    { cmd: 'rm -rf build # cleanup' },
    { cmd: 'wc', args: 'not json' },
    { cmd: 'wc', args: '[1, 2]' },
    { cmd: 'wc -l', args: 'not json' },
    { cmd: 'wc' },
  ]
  for (const inputs of unchanged) expect(repairExecCommandInputs(inputs)).toBe(inputs)
})

test('only exec_command calls are touched, and an unchanged call is the same object', () => {
  const other: ToolCall = { id: 'r', type: 'function', function: { name: 'ReadFile', arguments: { file_path: 'a b' } } }
  const fine: ToolCall = { id: 'e', type: 'function', function: { name: 'exec_command', arguments: { cmd: 'ls' } } }
  expect(repairToolCall(other)).toBe(other)
  expect(repairToolCall(fine)).toBe(fine)
})

const EXEC: ToolDefinition = { type: 'function', function: { name: 'exec_command', description: 'Run.', parameters: { type: 'object', properties: { cmd: { type: 'string' }, args: { type: 'array', items: { type: 'string' } } } } } }

test('the permission check and the tool both see the repaired command', async () => {
  const state = createAgentState()
  const executed: JsonObject[] = []
  let round = 0
  const events = await collect(runTurn({ model: 'test', state, userMessage: 'count lines', tools: [EXEC], permissionMode: 'auto' }, {
    llm: { async *stream() {
      round += 1
      if (round === 1) yield { toolCalls: [{ id: 'c1', type: 'function', function: { name: 'exec_command', arguments: { cmd: 'wc -l', args: ['bank.csv'] } } }] }
      else yield { content: 'done' }
    } },
    toolExecutor: { execute: async (call: ToolCall) => { executed.push(call.function.arguments); return '{"exitCode":0}' } },
  }))
  // `wc -l bank.csv` is read-only once it is seen as argv, so it runs without a prompt.
  expect(events.some(event => event.type === 'permission_request')).toBe(false)
  expect(executed).toEqual([{ cmd: 'wc', args: ['-l', 'bank.csv'] }])
})
