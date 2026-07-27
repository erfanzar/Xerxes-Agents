// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  appendInjection,
  injectionSignature,
  MAX_INJECTION_CHARACTERS,
  MAX_TURN_INJECTION_CHARACTERS,
  planInjection,
  scanInjections,
} from '../src/streaming/attachments.js'
import { neutralizeSystemReminders } from '../src/streaming/toolMarkers.js'
import type { ChatMessage } from '../src/types/messages.js'

function turnStart(): ChatMessage[] {
  return [{ role: 'user', content: 'do the work' }]
}

test('agent events keep their historical unwrapped marker byte for byte', () => {
  const messages = turnStart()
  const plan = appendInjection(messages, {
    kind: 'agent_events',
    events: ['[agent researcher] completed source scan'],
  })

  expect(plan.status).toBe('ready')
  expect(messages.at(-1)).toEqual({
    role: 'user',
    content: '[sub-agent events]\n[agent researcher] completed source scan',
  })
})

test('blank event lines are dropped and an all-blank batch injects nothing', () => {
  const messages = turnStart()
  appendInjection(messages, { kind: 'agent_events', events: ['  ', 'real line', ''] })
  expect(messages.at(-1)?.content).toBe('[sub-agent events]\nreal line')

  const empty = appendInjection(messages, { kind: 'agent_events', events: ['', '   '] })
  expect(empty).toEqual({ reason: 'empty', status: 'skipped' })
  expect(messages).toHaveLength(2)
})

test('new kinds are wrapped in a system-reminder tagged with their kind', () => {
  const messages = turnStart()
  const plan = appendInjection(messages, {
    kind: 'external_file_changes',
    paths: ['src/a.ts', 'src/b.ts'],
  })

  expect(plan.status).toBe('ready')
  expect(messages.at(-1)?.content).toBe(
    '<system-reminder kind="external_file_changes">\n'
    + '[files changed outside this session]\n'
    + 'src/a.ts\nsrc/b.ts\n'
    + '</system-reminder>',
  )
})

test('our own emission is not something neutralizeSystemReminders would have produced', () => {
  const plan = planInjection(turnStart(), { kind: 'todo_reminder', text: 'finish the seam' })
  expect(plan.status).toBe('ready')
  const text = plan.status === 'ready' ? plan.text : ''
  // The seam is the legitimate writer of the tag: inbound defanging must be the
  // inverse of our renderer, never applied to it.
  expect(neutralizeSystemReminders(text)).not.toBe(text)
  expect(text.startsWith(injectionSignature('todo_reminder'))).toBeTrue()
})

test('an identical block is not injected twice in the same turn', () => {
  const messages = turnStart()
  appendInjection(messages, { kind: 'agent_events', events: ['[agent a] done'] })
  const repeat = appendInjection(messages, { kind: 'agent_events', events: ['[agent a] done'] })

  expect(repeat).toEqual({ reason: 'duplicate', status: 'skipped' })
  expect(messages).toHaveLength(2)

  // Different content from the same kind still gets through.
  expect(appendInjection(messages, { kind: 'agent_events', events: ['[agent b] done'] }).status)
    .toBe('ready')
})

test('a fresh user message ends the window, so the same block may be sent again', () => {
  const messages = turnStart()
  appendInjection(messages, { kind: 'agent_events', events: ['[agent a] done'] })
  messages.push({ role: 'assistant', content: 'noted' })
  messages.push({ role: 'user', content: 'next question' })

  expect(appendInjection(messages, { kind: 'agent_events', events: ['[agent a] done'] }).status)
    .toBe('ready')
})

test('per-kind repeat throttle stops a single kind from flooding one turn', () => {
  const messages = turnStart()
  let injected = 0
  for (let index = 0; index < 40; index += 1) {
    if (appendInjection(messages, { kind: 'todo_reminder', text: `todo ${index}` }).status === 'ready') {
      injected += 1
    }
  }

  expect(injected).toBe(6)
  const last = appendInjection(messages, { kind: 'todo_reminder', text: 'one more' })
  expect(last).toEqual({ reason: 'kind_throttled', status: 'skipped' })
  // A different kind is unaffected by another kind's throttle.
  expect(appendInjection(messages, { kind: 'deferred_tools', names: ['Bash'] }).status).toBe('ready')
})

test('an oversized body is truncated to the per-injection cap with a notice', () => {
  const messages = turnStart()
  const plan = appendInjection(messages, {
    kind: 'agent_events',
    events: ['x'.repeat(MAX_INJECTION_CHARACTERS + 5_000)],
  })

  expect(plan.status).toBe('ready')
  const text = plan.status === 'ready' ? plan.text : ''
  expect(plan.status === 'ready' && plan.truncated).toBeTrue()
  expect(text).toContain('characters dropped')
  expect(text.length).toBeLessThanOrEqual(MAX_INJECTION_CHARACTERS + '[sub-agent events]\n'.length)
})

test('the cumulative turn cap eventually refuses further injections', () => {
  const messages = turnStart()
  let injected = 0
  for (let index = 0; index < 24; index += 1) {
    const plan = appendInjection(messages, {
      kind: 'agent_events',
      events: [`event ${index} ` + 'y'.repeat(MAX_INJECTION_CHARACTERS)],
    })
    if (plan.status === 'ready') injected += 1
  }

  expect(injected).toBeLessThan(24)
  expect(scanInjections(messages).characters).toBeLessThanOrEqual(
    MAX_TURN_INJECTION_CHARACTERS + '[sub-agent events]\n'.length * injected,
  )
  expect(appendInjection(messages, { kind: 'agent_events', events: ['late'] }))
    .toEqual({ reason: 'turn_budget', status: 'skipped' })
})

test('the budget is recovered by scanning, so compaction restores it with no bookkeeping', () => {
  const messages = turnStart()
  for (let index = 0; index < 6; index += 1) {
    appendInjection(messages, { kind: 'todo_reminder', text: `todo ${index}` })
  }
  expect(appendInjection(messages, { kind: 'todo_reminder', text: 'blocked' }).status)
    .toBe('skipped')

  // A compaction pass replaces the transcript with a summary; nothing hands the
  // seam a reset, and none is needed.
  const compacted: ChatMessage[] = [
    { role: 'system', content: 'summary of the session' },
    { role: 'user', content: 'do the work' },
  ]
  expect(appendInjection(compacted, { kind: 'todo_reminder', text: 'blocked' }).status)
    .toBe('ready')
})

test('scan attributes each injected block to its kind and ignores non-user messages', () => {
  const messages = turnStart()
  appendInjection(messages, { kind: 'agent_events', events: ['[agent a] done'] })
  messages.push({
    role: 'tool',
    content: '[sub-agent events]\nnot ours',
    name: 'ReadFile',
    tool_call_id: 'call_1',
  })
  appendInjection(messages, { kind: 'deferred_tools', names: ['Bash'] })

  const usage = scanInjections(messages)
  expect(usage.counts.get('agent_events')).toBe(1)
  expect(usage.counts.get('deferred_tools')).toBe(1)
  expect(usage.counts.get('todo_reminder')).toBeUndefined()
})

test('a multimodal user prompt bounds the window instead of being counted', () => {
  const messages: ChatMessage[] = [
    { role: 'user', content: '[sub-agent events]\nfrom a previous turn' },
    { role: 'user', content: [{ type: 'text', text: 'look at this' }] },
  ]

  expect(scanInjections(messages).characters).toBe(0)
})
