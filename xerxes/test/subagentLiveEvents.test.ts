// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { SubAgentManager, type SubAgentEvent } from '../src/agents/subagentManager.js'
import { daemonEventFromSubagent } from '../src/daemon/subagentHost.js'
import { agentStepsFromEvents } from '../src/hub/server.js'

test('live subagent events carry the exact tool input, result and text; the mailbox keeps previews only', async () => {
  const live: SubAgentEvent[] = []
  const command = `bun test ${'xerxes/test/very/long/path/'.repeat(10)}`
  const result = 'line\n'.repeat(5000)
  const reply = 'a'.repeat(300) + 'b'.repeat(300)
  const manager = new SubAgentManager({
    idFactory: () => 'live-task',
    onEvent: event => live.push(event),
    runner: async request => {
      request.report.toolStart({ toolCallId: 'call-1', name: 'ExecuteShell', inputs: { command, timeout: 30 } })
      request.report.toolEnd({ toolCallId: 'call-1', name: 'ExecuteShell', permitted: true, result })
      request.report.text(reply)
      return { content: 'finished' }
    },
  })
  const task = await manager.spawn({ prompt: 'run the tests' })
  await manager.wait(task.id, 1_000)

  const start = live.find(event => event.type === 'tool_start')!
  const end = live.find(event => event.type === 'tool_end')!
  const burst = live.find(event => event.type === 'text_burst')!
  expect(JSON.parse(String(start.data.inputFull))).toEqual({ command, timeout: 30 })
  expect(String(start.data.inputPreview).length).toBeLessThanOrEqual(200)
  // Results are clipped at the source to bound the journal.
  expect(String(end.data.resultFull)).toHaveLength(16_000)
  expect(String(end.data.resultFull).startsWith(result.slice(0, 15_999))).toBe(true)
  expect(burst.data.textFull).toBe(reply)

  // The 512-event mailbox never holds the full payloads.
  for (const event of manager.peekMailbox()) {
    expect(event.data).not.toHaveProperty('inputFull')
    expect(event.data).not.toHaveProperty('resultFull')
    expect(event.data).not.toHaveProperty('textFull')
  }
  expect(manager.peekMailbox().find(event => event.type === 'tool_start')?.data.inputPreview).toBe(start.data.inputPreview)

  // The daemon event keeps the preview fields and adds the full ones.
  const call = daemonEventFromSubagent(start, undefined)!
  const done = daemonEventFromSubagent(end, undefined)!
  const text = daemonEventFromSubagent(burst, undefined)!
  const nested = (event: typeof call) => (event.payload.event as { payload: Record<string, unknown> }).payload
  expect(nested(call)).toMatchObject({ id: 'call-1', name: 'ExecuteShell', arguments: start.data.inputPreview, arguments_full: start.data.inputFull })
  expect(nested(done)).toMatchObject({ tool_call_id: 'call-1', return_value: end.data.resultPreview, result_full: end.data.resultFull })
  expect(nested(text)).toMatchObject({ text: burst.data.preview, text_full: reply })

  // agents.inspect builds its steps from the full fields.
  const steps = agentStepsFromEvents([call, done, text].map(event => ({ type: event.type, payload: event.payload })), 'live-task')
  expect(steps).toEqual([
    { kind: 'tool', name: 'ExecuteShell', args: String(start.data.inputFull), output: String(end.data.resultFull), state: 'done' },
    { kind: 'text', text: reply },
  ])
  await manager.close()
})

test('agent steps fall back to the preview fields of older daemons', () => {
  const sub = (event: Record<string, unknown>) => ({ type: 'subagent_event', payload: { agent_id: 'a', event } })
  expect(agentStepsFromEvents([
    sub({ type: 'tool_call', payload: { id: 'c', name: 'Read', arguments: 'file_path=a.ts' } }),
    sub({ type: 'tool_result', payload: { tool_call_id: 'c', name: 'Read', return_value: 'short', permitted: true } }),
    sub({ type: 'text_part', payload: { text: '…tail' } }),
  ], 'a')).toEqual([
    { kind: 'tool', name: 'Read', args: 'file_path=a.ts', output: 'short', state: 'done' },
    { kind: 'text', text: '…tail' },
  ])
})
