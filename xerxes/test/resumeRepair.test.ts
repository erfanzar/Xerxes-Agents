// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  RESUME_REPLAY_SENTINEL,
  describeTranscriptRepair,
  repairResumedTranscript,
  replayPendingToolCalls,
  summarizeTranscriptRepair,
  type ResumeReplayExecutor,
} from '../src/session/resumeRepair.js'
import { normalizeDaemonTranscript } from '../src/session/daemonTranscript.js'

test('resume repair strips assistant markers, retains matching replies, and removes orphan tool messages', () => {
  const source = [
    {
      role: 'assistant',
      content: 'I will inspect it. ASSISTANT_TOOL_CALLS: {"name":"hidden","input":{}}',
      tool_calls: [
        { id: 'call-read', function: { name: 'ReadFile', arguments: '{"path":"README.md"}' } },
        { id: 'call-search', name: 'GrepTool', input: { pattern: 'TODO' } },
      ],
    },
    { role: 'tool', tool_call_id: 'call-read', content: 'contents' },
    { role: 'tool', tool_call_id: 'orphan', content: 'discard me' },
    { role: 'user', content: 'continue' },
    { role: 'tool', tool_call_id: 'call-search', content: 'too late' },
  ]

  const repair = repairResumedTranscript(source)

  expect(repair.messages).toEqual([
    {
      role: 'assistant',
      content: 'I will inspect it.',
      tool_calls: source[0]?.tool_calls,
    },
    { role: 'tool', tool_call_id: 'call-read', content: 'contents' },
    { role: 'tool', tool_call_id: 'call-search', content: RESUME_REPLAY_SENTINEL },
    { role: 'user', content: 'continue' },
  ])
  expect(repair.pendingReplays).toEqual([
    { tool_call_id: 'call-search', name: 'GrepTool', arguments: '{"pattern":"TODO"}' },
  ])
  expect(source[0]?.content).toContain('ASSISTANT_TOOL_CALLS')
  // The orphan drop used to report nothing at all.
  expect(repair.stats).toEqual({
    assistantMarkersStripped: 1,
    orphanedToolRepliesDropped: 2,
    replaySentinelsInserted: 1,
  })
})

test('a load that changes nothing reports nothing', () => {
  const summary = summarizeTranscriptRepair([
    { role: 'user', content: 'hello' },
    { role: 'assistant', content: 'hi' },
  ])

  expect(summary).toEqual({
    malformedMessagesDropped: 0,
    stats: {
      assistantMarkersStripped: 0,
      orphanedToolRepliesDropped: 0,
      replaySentinelsInserted: 0,
    },
  })
  expect(describeTranscriptRepair(summary)).toBe('')
})

test('a load that drops messages says so in one line covering every category', () => {
  const summary = summarizeTranscriptRepair([
    'not a message',
    42,
    {
      role: 'assistant',
      content: 'go ASSISTANT_TOOL_CALLS: {"name":"x","input":{}}',
      tool_calls: [{ id: 'c1', name: 'X', input: {} }],
    },
    { role: 'tool', tool_call_id: 'orphan', content: 'no matching call' },
  ])

  expect(summary).toEqual({
    malformedMessagesDropped: 2,
    stats: {
      assistantMarkersStripped: 1,
      orphanedToolRepliesDropped: 1,
      replaySentinelsInserted: 1,
    },
  })
  expect(describeTranscriptRepair(summary)).toBe(
    'Resume repaired this transcript: dropped 2 malformed messages, dropped 1 orphaned tool reply, '
    + 'inserted 1 interrupted-call placeholder, stripped tool-call markers from 1 assistant message.',
  )
})

test('the repair summary pluralizes orphaned tool replies correctly', () => {
  const summary = summarizeTranscriptRepair([
    { role: 'tool', tool_call_id: 'a', content: 'x' },
    { role: 'tool', tool_call_id: 'b', content: 'y' },
  ])

  expect(describeTranscriptRepair(summary)).toBe(
    'Resume repaired this transcript: dropped 2 orphaned tool replies.',
  )
})

test('resume repair flushes unresolved calls before a subsequent assistant message', () => {
  const repair = repairResumedTranscript([
    { role: 'assistant', content: 'first', tool_calls: [{ id: 'call-one', name: 'First', input: {} }] },
    { role: 'assistant', content: 'second', tool_calls: [] },
  ])

  expect(repair.messages).toEqual([
    { role: 'assistant', content: 'first', tool_calls: [{ id: 'call-one', name: 'First', input: {} }] },
    { role: 'tool', tool_call_id: 'call-one', content: RESUME_REPLAY_SENTINEL },
    { role: 'assistant', content: 'second', tool_calls: [] },
  ])
  expect(repair.pendingReplays).toEqual([{ tool_call_id: 'call-one', name: 'First', arguments: '{}' }])
})

test('pending replay remains inert without an executor and uses an injected executor when requested', async () => {
  const repair = repairResumedTranscript([
    { role: 'assistant', content: '', tool_calls: [{ id: 'call-a', name: 'ReadFile', input: { path: 'README.md' } }] },
  ])

  const withoutExecutor = await replayPendingToolCalls(repair)
  expect(withoutExecutor).toEqual(repair)

  const calls: Array<readonly [string, Record<string, unknown>]> = []
  const executor: ResumeReplayExecutor = {
    execute(name, arguments_) {
      calls.push([name, arguments_])
      return 'actual file contents'
    },
  }
  const replayed = await replayPendingToolCalls(repair, { executor })

  expect(calls).toEqual([['ReadFile', { path: 'README.md' }]])
  expect(replayed.messages).toEqual([
    { role: 'assistant', content: '', tool_calls: [{ id: 'call-a', name: 'ReadFile', input: { path: 'README.md' } }] },
    { role: 'tool', tool_call_id: 'call-a', content: 'actual file contents' },
  ])
  expect(replayed.pendingReplays).toEqual([])
  expect(repair.messages[1]?.content).toBe(RESUME_REPLAY_SENTINEL)
})

test('replay makes invalid arguments and executor failures explicit without fabricating results', async () => {
  const repair = repairResumedTranscript([
    {
      role: 'assistant',
      content: '',
      tool_calls: [
        { id: 'invalid', function: { name: 'BadArguments', arguments: '{not-json' } },
        { id: 'failed', function: { name: 'Fails', arguments: '{}' } },
      ],
    },
  ])
  const executor: ResumeReplayExecutor = {
    execute(name) {
      if (name === 'Fails') throw new Error('tool backend unavailable')
      throw new Error('unexpected executor call')
    },
  }

  const replayed = await replayPendingToolCalls(repair, { executor })

  expect(replayed.messages).toEqual([
    {
      role: 'assistant',
      content: '',
      tool_calls: [
        { id: 'invalid', function: { name: 'BadArguments', arguments: '{not-json' } },
        { id: 'failed', function: { name: 'Fails', arguments: '{}' } },
      ],
    },
    expect.objectContaining({
      tool_call_id: 'invalid',
      content: expect.stringContaining('[replay error: invalid arguments —'),
    }),
    { role: 'tool', tool_call_id: 'failed', content: '[replay error: tool backend unavailable]' },
  ])
  expect(replayed.pendingReplays).toEqual([])
})

test('replay records an explicit error instead of a fabricated empty result', async () => {
  const repair = repairResumedTranscript([
    {
      role: 'assistant',
      content: '',
      tool_calls: [
        { id: 'call-null', name: 'ReturnsNull', input: {} },
        { id: 'call-undefined', name: 'ReturnsUndefined', input: {} },
      ],
    },
  ])
  const executor: ResumeReplayExecutor = {
    execute(name) {
      return name === 'ReturnsNull' ? null : undefined
    },
  }

  const replayed = await replayPendingToolCalls(repair, { executor })

  expect(replayed.messages).toEqual([
    {
      role: 'assistant',
      content: '',
      tool_calls: [
        { id: 'call-null', name: 'ReturnsNull', input: {} },
        { id: 'call-undefined', name: 'ReturnsUndefined', input: {} },
      ],
    },
    { role: 'tool', tool_call_id: 'call-null', content: '[replay error: empty result]' },
    { role: 'tool', tool_call_id: 'call-undefined', content: '[replay error: empty result]' },
  ])
  expect(replayed.pendingReplays).toEqual([])
})

test('daemon transcript loading adopts repaired messages and exposes pending replay descriptors', () => {
  const normalized = normalizeDaemonTranscript({
    session_id: 'a1b2c3d4',
    messages: [
      { role: 'assistant', content: '', tool_calls: [{ id: 'call-a', name: 'ReadFile', input: {} }] },
      { role: 'tool', tool_call_id: 'orphan', content: 'discard' },
    ],
  }, {
    currentProjectDirectory: '/project',
    requestedSessionKey: 'a1b2c3d4',
  })

  expect(normalized?.messages).toEqual([
    { role: 'assistant', content: '', tool_calls: [{ id: 'call-a', name: 'ReadFile', input: {} }] },
    { role: 'tool', tool_call_id: 'call-a', content: RESUME_REPLAY_SENTINEL },
  ])
  expect(normalized?.pendingResumeReplays).toEqual([
    { tool_call_id: 'call-a', name: 'ReadFile', arguments: '{}' },
  ])
})
