// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { renderTrajectory, trajectorySteps } from '../src/context/trajectory.js'

test('trajectory grammar tags chronological text, actions and results with their shared call IDs', () => {
  const messages = [
    { role: 'user', content: 'Read the file.' },
    { role: 'assistant', content: 'Reading.', tool_calls: [
      { id: 'read-1', function: { name: 'ReadFile', arguments: '{"z":2,"a":1}' } },
      { id: 'test-1', name: 'exec_command', input: { command: 'bun test' } },
    ] },
    { role: 'tool', tool_call_id: 'test-1', content: '1 fail', is_error: true },
    { role: 'tool', tool_call_id: 'read-1', content: 'file contents' },
  ]
  expect(trajectorySteps(messages)).toEqual([
    { type: 'user', text: 'Read the file.' },
    { type: 'assistant', text: 'Reading.' },
    { type: 'action', tool: 'ReadFile', callId: 'read-1', args: '{"a":1,"z":2}' },
    { type: 'action', tool: 'exec_command', callId: 'test-1', args: '{"command":"bun test"}' },
    { type: 'error', callId: 'test-1', text: '1 fail' },
    { type: 'observation', callId: 'read-1', text: 'file contents', originalChars: 13, clipped: false },
  ])
  expect(renderTrajectory(messages)).toBe([
    '[USER] Read the file.',
    '[ASSISTANT] Reading.',
    '[ACTION] ReadFile({"a":1,"z":2}) call_id="read-1"',
    '[ACTION] exec_command({"command":"bun test"}) call_id="test-1"',
    '[ERROR] call_id="test-1" 1 fail',
    '[OBSERVATION] call_id="read-1" file contents',
  ].join('\n'))
})

test('error detection uses explicit flags, failure statuses, the runtime failure prefix and whole JSON envelopes', () => {
  const failures = [
    { is_error: true, content: 'flag' },
    { isError: true, content: 'camel flag' },
    { status: 'error', content: 'error status' },
    { status: 'failed', content: 'failed status' },
    { content: 'Tool execution failed: not configured' },
    { content: '{"ok":false,"error":"refused"}' },
    { content: { ok: false, error: 'refused' } },
    { content: '{"isError":true,"content":[]}' },
    { content: [{ type: 'text', text: '{"ok":false}' }] },
  ]
  for (const failure of failures) {
    expect(trajectorySteps([{ role: 'tool', tool_call_id: 'f', ...failure }])[0]?.type).toBe('error')
  }
  expect(trajectorySteps([{ role: 'error', content: 'standalone failure' }])).toEqual([
    { type: 'error', text: 'standalone failure' },
  ])
})

test('error-like prose, nested errors and quoted examples are not invented failures', () => {
  for (const content of [
    '0 errors, all tests passed',
    'Example: Tool execution failed: missing file',
    'Documentation example: {"ok":false}',
    '{"ok":true,"example":{"ok":false}}',
    '{"error":"this field alone is not a failure flag"}',
    '{"is_error":"true","ok":"false"}',
  ]) {
    expect(trajectorySteps([{ role: 'tool', content }])[0]?.type).toBe('observation')
  }
})

test('observation clips report original size and attribute clipping to the renderer, not the tool', () => {
  const output = 'x'.repeat(4_821) + 'END'
  const messages = [{ role: 'tool', tool_call_id: 'large', content: output }]
  expect(trajectorySteps(messages)).toEqual([
    { type: 'observation', callId: 'large', text: output, originalChars: 4_824, clipped: false },
  ])
  const rendered = renderTrajectory(messages, { maxObservationChars: 2_000 })
  expect(rendered).toContain('x'.repeat(2_000))
  expect(rendered).not.toContain('x'.repeat(2_001))
  expect(rendered).toContain('[preview clipped by the renderer: showing 2000 of 4824 chars;')
  expect(rendered).toContain("the tool's real output was not clipped by the renderer]")
})

test('argument and error previews are explicitly attributed and retain their full source sizes', () => {
  const rendered = renderTrajectory([
    { role: 'assistant', tool_calls: [{ id: 'a', name: 'WriteFile', input: { body: 'x'.repeat(300) } }] },
    { role: 'tool', tool_call_id: 'a', content: 'Tool execution failed: ' + 'e'.repeat(300) },
  ], { maxArgumentChars: 20, maxObservationChars: 30 })
  expect(rendered).toContain('[argument preview clipped by the renderer: showing 20 of 311 chars; original arguments unchanged]')
  expect(rendered).toContain('[ERROR] call_id="a" Tool execution failed: ')
  expect(rendered).toContain(`[preview clipped by the renderer: showing 30 of ${('Tool execution failed: ' + 'e'.repeat(300)).length} chars;`)
})

test('unclipped text, whitespace, empty turns, repeated result IDs and malformed tool payloads are never silently dropped', () => {
  const rendered = renderTrajectory([
    { role: 'system', content: 'policy' },
    { role: 'developer', content: 'additional instruction' },
    { role: 'user', content: '  exact\n\nspacing  ' },
    { role: 'assistant', content: '' },
    { role: 'tool', tool_call_id: 'dup', content: 'first result' },
    { role: 'tool', tool_call_id: 'dup', content: 'second result' },
    { role: 'assistant', tool_calls: { unexpected: 'preserve me' } },
    { role: 'assistant', tool_calls: [17] },
  ])
  expect(rendered).toContain('[USER] [source role="system"] policy')
  expect(rendered).toContain('[USER] [source role="developer"] additional instruction')
  expect(rendered).toContain('[USER]   exact\n\nspacing  ')
  expect(rendered).toContain('[ASSISTANT] ')
  expect(rendered.match(/first result/g)).toHaveLength(1)
  expect(rendered.match(/second result/g)).toHaveLength(1)
  expect(rendered).toContain('tool_calls={"unexpected":"preserve me"}')
  expect(rendered).toContain('unknown_tool(17)')
})

test('structured outputs with text fields and malformed calls retain their remaining data', () => {
  const rendered = renderTrajectory([
    { role: 'tool', content: { text: 'keep text', extra: 'keep this too', ok: false } },
    { role: 'assistant', tool_calls: [{ unexpected: 'keep malformed call' }, { name: 'tool', input: null }] },
  ])
  expect(rendered).toContain('[ERROR] {"extra":"keep this too","ok":false,"text":"keep text"}')
  expect(rendered).toContain('unknown_tool({"unexpected":"keep malformed call"})')
  expect(rendered).toContain('tool(null)')
})

test('head/tail clipping names exactly the omitted characters and whole/partial steps', () => {
  const messages = Array.from({ length: 20 }, (_, index) => ({ role: 'user', content: `turn-${index}:` + 'x'.repeat(100) }))
  const full = renderTrajectory(messages)
  const clipped = renderTrajectory(messages, { maxChars: 600 })
  expect(clipped.length).toBeLessThanOrEqual(600)
  expect(clipped.startsWith('[USER] turn-0:')).toBe(true)
  expect(clipped.endsWith('x'.repeat(100))).toBe(true)
  expect(clipped).not.toContain('turn-10:')
  const marker = clipped.match(/\n\[transcript truncated by the renderer: omitted (\d+) of (\d+) chars; (\d+) whole steps omitted, (\d+) steps partially clipped of (\d+) total;[^\n]+\]\n/)
  expect(marker).not.toBeNull()
  const head = clipped.slice(0, marker!.index)
  const tail = clipped.slice(marker!.index! + marker![0].length)
  expect(Number(marker![1])).toBe(full.length - head.length - tail.length)
  expect(Number(marker![2])).toBe(full.length)
  const removedStart = head.length
  const removedEnd = full.length - tail.length
  const spans = [...full.matchAll(/\[USER\] [^\n]*/g)]
  const whole = spans.filter(span => span.index! >= removedStart && span.index! + span[0].length <= removedEnd).length
  const touched = spans.filter(span => span.index! < removedEnd && span.index! + span[0].length > removedStart).length
  expect(Number(marker![3])).toBe(whole)
  expect(Number(marker![4])).toBe(touched - whole)
  expect(Number(marker![5])).toBe(20)
  expect(clipped).toContain('partial steps are renderer previews, not incomplete tool output')
})

test('a tiny whole-transcript budget never silently clips away the disclosure', () => {
  const rendered = renderTrajectory([{ role: 'user', content: 'head' + 'x'.repeat(1_000) + 'tail' }], { maxChars: 1 })
  expect(rendered).toContain('[transcript truncated by the renderer: omitted ')
  expect(rendered.startsWith('[')).toBe(true)
  expect(rendered.endsWith('l')).toBe(true)
})

test('disabled preview limits preserve all source text and complete short renders are unchanged', () => {
  const messages = [{ role: 'tool', content: 'long output '.repeat(1_000) }]
  const full = renderTrajectory(messages, { maxObservationChars: 0, maxChars: 0 })
  expect(full).toEndWith('long output '.repeat(1_000))
  expect(full).not.toContain('clipped')
  expect(renderTrajectory([], { maxChars: 1 })).toBe('')
  expect(renderTrajectory([{ content: null }])).toBe('[USER] null')
  expect(renderTrajectory([{ role: 'assistant' }])).toBe('[ASSISTANT] ')
})

test('rendering is deterministic, uses stable nested JSON and does not mutate frozen input', () => {
  const messages = Object.freeze([
    Object.freeze({ role: 'assistant', tool_calls: Object.freeze([
      Object.freeze({ id: 'stable', function: Object.freeze({ name: 'tool', arguments: Object.freeze({ z: 1, a: { y: 2, b: 3 } }) }) }),
    ]) }),
    Object.freeze({ role: 'tool', tool_call_id: 'stable', content: Object.freeze({ z: 1, a: 2 }) }),
  ])
  const before = JSON.stringify(messages)
  expect(renderTrajectory(messages)).toBe(renderTrajectory(messages))
  expect(renderTrajectory(messages)).toContain('tool({"a":{"b":3,"y":2},"z":1})')
  expect(renderTrajectory(messages)).toContain('{"a":2,"z":1}')
  expect(JSON.stringify(messages)).toBe(before)
})

test('text and image content arrays retain text and disclose image omissions without dumping base64', () => {
  const rendered = renderTrajectory([{ role: 'user', content: [
    { type: 'text', text: 'Before' },
    { type: 'image_url', image_url: { url: 'data:image/png;base64,PRIVATE_IMAGE' } },
    { type: 'image', source: { type: 'base64', data: 'OTHER_IMAGE' } },
    { type: 'text', text: 'After' },
    { type: 'unknown', value: 'keep this block' },
  ] }])
  expect(rendered).toBe('[USER] Before\n[image omitted]\n[image omitted]\nAfter\n{"type":"unknown","value":"keep this block"}')
  expect(rendered).not.toContain('PRIVATE_IMAGE')
  expect(rendered).not.toContain('OTHER_IMAGE')
})

test('orphan results remain observations, and missing results are stated without fabricating a tool failure', () => {
  const rendered = renderTrajectory([
    { role: 'tool', tool_call_id: 'orphan', content: 'earlier result' },
    { role: 'tool', content: 'no identifier' },
    { role: 'assistant', tool_calls: [{ id: 'pending', name: 'ReadFile' }] },
  ])
  expect(rendered).toContain('[OBSERVATION] call_id="orphan" earlier result')
  expect(rendered).toContain('[OBSERVATION] no identifier')
  expect(rendered).toContain('[ACTION] ReadFile() call_id="pending" (no result in this slice)')
  expect(rendered).not.toContain('[ERROR]')
})

test('Anthropic content-block actions and results share the same grammar without duplicate observations', () => {
  const rendered = renderTrajectory([
    { role: 'assistant', content: [{ type: 'tool_use', id: 'a', name: 'ReadFile', input: { path: '/repo/file' } }] },
    { role: 'user', content: [
      { type: 'tool_result', tool_use_id: 'a', content: [{ type: 'text', text: 'result' }] },
      { type: 'tool_result', tool_use_id: 'orphan', content: 'failed', is_error: true },
    ] },
  ])
  expect(rendered).toBe([
    '[ACTION] ReadFile({"path":"/repo/file"}) call_id="a"',
    '[OBSERVATION] call_id="a" result',
    '[ERROR] call_id="orphan" failed',
  ].join('\n'))
})

test('all preview boundaries and whole-transcript head/tail boundaries keep Unicode surrogate pairs intact', () => {
  const rendered = renderTrajectory([
    { role: 'assistant', tool_calls: [{ id: 'u', name: 'tool', args: 'a'.repeat(19) + '😀tail' }] },
    { role: 'tool', tool_call_id: 'u', content: 'b'.repeat(29) + '🚀tail' },
  ], { maxArgumentChars: 20, maxObservationChars: 30 })
  expect(rendered.isWellFormed()).toBe(true)
  expect(rendered).toContain('showing 19 of 25 chars')
  expect(rendered).toContain('showing 29 of 35 chars')
  for (const budget of [1, 300, 301, 302, 303]) {
    expect(renderTrajectory([{ role: 'user', content: '😀'.repeat(1_000) }], { maxChars: budget }).isWellFormed()).toBe(true)
  }
})
