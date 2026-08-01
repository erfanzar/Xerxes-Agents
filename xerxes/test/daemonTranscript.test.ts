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

test('header reads answer a listing without parsing the message history', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-transcript-header-'))
  try {
    const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' })
    const sessionId = 'aaaabbbbcccc'
    const transcript = normalizeDaemonTranscript({
      session_id: sessionId,
      key: 'tui:slot',
      updated_at: '2026-02-02T00:00:00.000Z',
      messages: [{ role: 'user', content: 'measure me' }],
      turn_count: 3,
      metadata: { title: 'header title', project_root: '/project' },
    }, { requestedSessionKey: sessionId, currentProjectDirectory: '/project' })
    if (!transcript) throw new Error('expected transcript to normalize')
    await store.save(transcript)

    const raw = await Bun.file(store.pathFor(sessionId)).text()
    // Everything the listing renders has to precede `messages`, or the head
    // read cannot answer and every row costs a full parse.
    for (const key of ['"turn_count"', '"message_count"', '"metadata"']) {
      expect(raw.indexOf(key)).toBeLessThan(raw.indexOf('\n  "messages"'))
    }

    const result = await store.readHeader(sessionId)
    expect(result.kind).toBe('header')
    if (result.kind !== 'header') throw new Error('expected a header')
    expect(result.header).toMatchObject({
      agentId: 'default',
      // The normalizer rebinds the key to the requested resume id before saving.
      key: sessionId,
      messageCount: 1,
      sessionId,
      turnCount: 3,
      updatedAt: '2026-02-02T00:00:00.000Z',
    })
    expect(result.header.metadata.title).toBe('header title')

    // Only the head is read: a record whose message array is unparseable still
    // yields a usable header, while a full load rejects it.
    const brokenId = 'ddddeeeeffff'
    await Bun.write(store.pathFor(brokenId), [
      '{',
      '  "session_id": "ddddeeeeffff",',
      '  "updated_at": "2026-03-03T00:00:00.000Z",',
      '  "turn_count": 1,',
      '  "message_count": 2,',
      '  "metadata": {"title": "head only"},',
      '  "messages": [ this is not json',
    ].join('\n'))
    const brokenHeader = await store.readHeader(brokenId)
    expect(brokenHeader.kind).toBe('header')
    expect(await store.load(brokenId)).toBeUndefined()

    // A record written before the summary fields were hoisted is well-formed,
    // just unanswerable from its head.
    const legacyId = 'a1a1b2b2c3c3'
    await Bun.write(store.pathFor(legacyId), JSON.stringify({
      session_id: legacyId,
      messages: [{ role: 'user', content: 'legacy layout' }],
      turn_count: 1,
      metadata: { title: 'legacy' },
    }, null, 2))
    expect((await store.readHeader(legacyId)).kind).toBe('truncated')

    const garbageId = 'b0b0b0b0b0b0'
    await Bun.write(store.pathFor(garbageId), 'not a transcript at all')
    expect((await store.readHeader(garbageId)).kind).toBe('unreadable')

    const entries = await store.listEntries()
    expect(entries.map(entry => entry.sessionId).sort()).toEqual(
      [sessionId, brokenId, legacyId, garbageId].sort(),
    )
    expect(entries.every(entry => entry.sizeBytes > 0)).toBeTrue()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('journalled messages survive a crash between the append and the next save', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-transcript-journal-'))
  try {
    const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' })
    const sessionId = 'c0ffee00c0ffee00'
    const transcript = normalizeDaemonTranscript({
      session_id: sessionId,
      messages: [{ role: 'user', content: 'saved turn' }],
      turn_count: 1,
      updated_at: '2026-04-04T00:00:00.000Z',
    }, { requestedSessionKey: sessionId, currentProjectDirectory: '/project' })
    if (!transcript) throw new Error('expected transcript to normalize')
    await store.save(transcript)

    // Index 0 is already covered by the saved snapshot and must not be
    // duplicated; 1 and 2 are the turn the crash interrupted.
    await store.appendMessage(sessionId, { role: 'user', content: 'saved turn' }, 0)
    await store.appendMessage(sessionId, { role: 'assistant', content: 'unsaved answer' }, 1)
    await store.appendMessage(sessionId, { role: 'user', content: 'unsaved follow-up' }, 2)
    // A crash mid-append leaves a partial final line, which is discarded
    // without taking the intact lines with it.
    await Bun.write(
      store.journalPathFor(sessionId),
      `${await Bun.file(store.journalPathFor(sessionId)).text()}{"index":3,"message":{"role":"as`,
    )

    const recovered = await store.load(sessionId)
    expect(recovered?.messages).toEqual([
      { role: 'user', content: 'saved turn' },
      { role: 'assistant', content: 'unsaved answer' },
      { role: 'user', content: 'unsaved follow-up' },
    ])

    // A successful save subsumes the journal, so the next load must not replay it.
    if (!recovered) throw new Error('expected a recovered transcript')
    await store.save(recovered)
    expect(await Bun.file(store.journalPathFor(sessionId)).exists()).toBeFalse()
    expect((await store.load(sessionId))?.messages).toHaveLength(3)

    // A gap stops the replay instead of reordering the transcript around a lost write.
    await store.appendMessage(sessionId, { role: 'assistant', content: 'after a gap' }, 7)
    expect((await store.load(sessionId))?.messages).toHaveLength(3)

    // Deleting the transcript takes the journal with it, so the id cannot
    // resurrect its own history.
    await store.remove(sessionId)
    expect(await Bun.file(store.journalPathFor(sessionId)).exists()).toBeFalse()
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

// The fixture paths are POSIX-shaped (`/projects/current`); on Windows the path
// normalizer correctly rewrites them to `C:\...`, so the exact-match below only
// holds on POSIX hosts.
test.skipIf(process.platform === 'win32')('store writes Python-readable v2 supersets and resumes only explicit IDs', async () => {
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
