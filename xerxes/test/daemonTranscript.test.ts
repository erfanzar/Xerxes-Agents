// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import {
  DAEMON_SESSION_FORMAT,
  INTERRUPTED_TOOL_RESULT,
  DaemonTranscriptStore,
  normalizeDaemonTranscript,
  repairToolPairs,
} from '../src/session/daemonTranscript.js'

test('daemon transcript normalizer repairs only orphaned contiguous tool calls', () => {
  const repaired = repairToolPairs([
    {
      role: 'assistant',
      content: '',
      tool_calls: [
        { id: 'call-a', name: 'ReadFile' },
        { id: 'call-b', function: { name: 'GrepTool' } },
      ],
    },
    { role: 'tool', tool_call_id: 'call-a', name: 'ReadFile', content: 'ok' },
    { role: 'user', content: 'continue' },
  ])

  expect(repaired).toEqual([
    {
      role: 'assistant',
      content: '',
      tool_calls: [
        { id: 'call-a', name: 'ReadFile' },
        { id: 'call-b', function: { name: 'GrepTool' } },
      ],
    },
    { role: 'tool', tool_call_id: 'call-a', name: 'ReadFile', content: 'ok' },
    { role: 'tool', tool_call_id: 'call-b', content: INTERRUPTED_TOOL_RESULT },
    { role: 'user', content: 'continue' },
  ])
})

test('normalizer drops malformed messages instead of rejecting the whole transcript', async () => {
  const normalized = normalizeDaemonTranscript({
    session_id: 'a1b2c3d4',
    messages: [
      { role: 'user', content: 'kept' },
      'garbage',
      null,
      42,
      ['array-is-not-a-message'],
      { role: 'assistant', content: 'also kept' },
    ],
    turn_count: 2,
  }, { currentProjectDirectory: '/project', requestedSessionKey: 'a1b2c3d4' })

  expect(normalized).toBeDefined()
  expect(normalized?.messages).toEqual([
    { role: 'user', content: 'kept' },
    { role: 'assistant', content: 'also kept' },
  ])

  // A persisted transcript with one corrupt entry still loads through the store:
  // returning undefined here would let the next save overwrite all history.
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-transcript-corrupt-'))
  try {
    const sessionId = 'deadbeefdeadbeef'
    const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' })
    await Bun.write(store.pathFor(sessionId), JSON.stringify({
      session_id: sessionId,
      messages: [{ role: 'user', content: 'survives' }, 'corrupt-entry', 12345],
      turn_count: 1,
      updated_at: '2026-01-01T00:00:00.000Z',
    }))
    const loaded = await store.load(sessionId)
    expect(loaded?.messages).toEqual([
      { role: 'user', content: 'survives' },
    ])
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('list sorts malformed updated_at timestamps as the epoch instead of NaN', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-transcript-list-'))
  try {
    const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' })
    const make = (sessionId: string, updatedAt: string) => {
      const transcript = normalizeDaemonTranscript({
        session_id: sessionId,
        updated_at: updatedAt,
        messages: [{ role: 'user', content: 'hi' }],
        turn_count: 1,
      }, { requestedSessionKey: sessionId, currentProjectDirectory: '/project' })
      if (!transcript) throw new Error('expected transcript to normalize')
      return transcript
    }
    await store.save(make('aaaa1111', '2026-01-01T00:00:00.000Z'))
    await store.save(make('bbbb2222', 'not-a-date'))
    await store.save(make('cccc3333', '2026-06-01T00:00:00.000Z'))

    const listed = await store.list()
    expect(listed.map(transcript => transcript.sessionId)).toEqual(['cccc3333', 'aaaa1111', 'bbbb2222'])
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('store writes Python-readable v2 supersets and resumes only explicit IDs', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-transcript-'))
  const sessionId = 'a1b2c3d4e5f6a7b8'
  const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: '/projects/current', workspaceRoot: '/users/.xerxes/agents' })
  const normalized = normalizeDaemonTranscript({
    session_id: sessionId,
    key: 'old-slot-key',
    cwd: '/users/.xerxes/agents/default',
    messages: [{ role: 'user', content: 'hello' }],
    turn_count: 1,
    mode: 'code',
    extra_future_field: { preserve: true },
  }, { requestedSessionKey: sessionId, currentProjectDirectory: '/projects/current', workspaceRoot: '/users/.xerxes/agents' })
  if (!normalized) {
    throw new Error('expected transcript to normalize')
  }
  expect(normalized.totalApiCalls).toBeUndefined()
  expect(normalized.apiCallsComplete).toBeUndefined()
  await store.save(normalized)
  const loaded = await store.load(sessionId)
  expect(loaded).toMatchObject({
    format: 'bun-v2',
    key: sessionId,
    cwd: '/projects/current',
    extra: { extra_future_field: { preserve: true } },
  })
  expect(await store.load('tui:default')).toBeUndefined()
  const raw = JSON.parse(await Bun.file(store.pathFor(sessionId)).text()) as Record<string, unknown>
  expect(raw.format).toBe(DAEMON_SESSION_FORMAT)
  expect(raw.extra_future_field).toEqual({ preserve: true })
  expect(raw.total_api_calls).toBeUndefined()
  expect(raw.api_calls_complete).toBeUndefined()
  expect(await store.remove(sessionId)).toBe(true)
  expect(await Bun.file(store.pathFor(sessionId)).exists()).toBe(false)
  expect(await store.remove(sessionId)).toBe(false)
  await rm(directory, { recursive: true, force: true })
})
