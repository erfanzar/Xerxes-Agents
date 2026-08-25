// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm, utimes } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { pathToFileURL } from 'node:url'

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

test('store distinguishes a missing transcript from corrupt persisted bytes', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-transcript-load-result-'))
  try {
    const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' })
    expect(await store.loadResult('deadbeef')).toEqual({ kind: 'missing' })

    await Bun.write(store.pathFor('deadbeef'), '{corrupt bytes')
    expect(await store.loadResult('deadbeef')).toEqual({ kind: 'corrupt' })
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

test('a journal append racing a save remains recoverable', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-transcript-journal-race-'))
  try {
    const sessionId = 'facefeedfacefeed'
    const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' })
    const transcript = normalizeDaemonTranscript({
      session_id: sessionId,
      messages: [{ role: 'user', content: 'saved turn' }],
      turn_count: 1,
    }, { requestedSessionKey: sessionId, currentProjectDirectory: '/project' })
    if (!transcript) throw new Error('expected transcript to normalize')
    await store.save(transcript)

    await Promise.all([
      store.appendMessage(sessionId, { role: 'assistant', content: 'raced answer' }, 1),
      store.save(transcript),
    ])

    expect((await store.load(sessionId))?.messages).toEqual([
      { role: 'user', content: 'saved turn' },
      { role: 'assistant', content: 'raced answer' },
    ])
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('an authorized undo rewrite survives a store restart without resurrecting removed messages', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-transcript-undo-restart-'))
  try {
    const sessionId = '0ddba1100ddba110'
    const first = new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' })
    const original = normalizeDaemonTranscript({
      session_id: sessionId,
      messages: [
        { role: 'user', content: 'keep' },
        { role: 'assistant', content: 'kept answer' },
        { role: 'user', content: 'undo me' },
        { role: 'assistant', content: 'remove me' },
      ],
      turn_count: 2,
    }, { requestedSessionKey: sessionId, currentProjectDirectory: '/project' })
    if (!original) throw new Error('expected transcript to normalize')
    let initial = 0
    await first.save(original, { mode: 'rewrite', expectedGeneration: 0, onSavedGeneration: value => { initial = value } })

    const resumed = await new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' }).load(sessionId)
    if (!resumed) throw new Error('expected transcript after restart')
    await first.save({ ...resumed, messages: resumed.messages.slice(0, 2), turnCount: 1 }, {
      mode: 'rewrite',
      expectedGeneration: initial,
    })

    expect((await new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' }).load(sessionId))?.messages)
      .toEqual(original.messages.slice(0, 2))
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('an authorized compaction rewrite survives a store restart without merging old history', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-transcript-compact-restart-'))
  try {
    const sessionId = 'c04fac7c04fac7'
    const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' })
    const original = normalizeDaemonTranscript({
      session_id: sessionId,
      messages: [
        { role: 'user', content: 'old request' },
        { role: 'assistant', content: 'old answer' },
        { role: 'user', content: 'latest request' },
      ],
      turn_count: 2,
    }, { requestedSessionKey: sessionId, currentProjectDirectory: '/project' })
    if (!original) throw new Error('expected transcript to normalize')
    let initial = 0
    await store.save(original, { mode: 'rewrite', expectedGeneration: 0, onSavedGeneration: value => { initial = value } })
    const compacted = [{ role: 'system', content: 'summary of old history' }, original.messages[2]!]
    await store.save({ ...original, messages: compacted }, {
      mode: 'rewrite',
      expectedGeneration: initial,
    })

    expect((await new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' }).load(sessionId))?.messages)
      .toEqual(compacted)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('transcript locks recover stale empty and PID-reused owners but preserve active owners with bounded waits', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-transcript-locks-'))
  try {
    const sessionId = '10cc10cc10cc10cc'
    const store = new DaemonTranscriptStore({
      directory,
      currentProjectDirectory: '/project',
      lockStaleMs: 100,
      lockWaitMs: 35,
    })
    const transcript = normalizeDaemonTranscript({
      session_id: sessionId,
      messages: [{ role: 'user', content: 'locked write' }],
      turn_count: 1,
    }, { requestedSessionKey: sessionId, currentProjectDirectory: '/project' })
    if (!transcript) throw new Error('expected transcript to normalize')
    const lockPath = `${store.pathFor(sessionId)}.lock`

    // A creator can crash between exclusive creation and metadata write.
    await Bun.write(lockPath, '')
    const staleTime = new Date(Date.now() - 60_000)
    await utimes(lockPath, staleTime, staleTime)
    await store.save(transcript)
    expect(await Bun.file(lockPath).exists()).toBeFalse()

    // A stale PID can have been reused by an unrelated live process. Age must
    // allow recovery; PID liveness alone would wait forever.
    await Bun.write(lockPath, JSON.stringify({
      pid: process.pid,
      token: 'dead-owner-token',
      createdAt: Date.now() - 60_000,
    }))
    await utimes(lockPath, staleTime, staleTime)
    await store.save(transcript)
    expect(await Bun.file(lockPath).exists()).toBeFalse()

    // A fresh owner is never stolen, even when it names this live PID. Waiting
    // is bounded and failure leaves the owner's token untouched.
    const active = JSON.stringify({ pid: process.pid, token: 'active-owner-token', createdAt: Date.now() })
    await Bun.write(lockPath, active)
    const started = performance.now()
    expect(store.save(transcript)).rejects.toThrow('Timed out waiting for transcript lock')
    expect(performance.now() - started).toBeLessThan(500)
    expect(await Bun.file(lockPath).text()).toBe(active)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('two processes merge distinct transcript messages instead of overwriting either write', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-transcript-process-race-'))
  try {
    const sessionId = 'cabba9e0cabba9e0'
    const sourceUrl = pathToFileURL(join(import.meta.dir, '../src/session/daemonTranscript.ts')).href
    const workerPath = join(directory, 'writer.ts')
    await Bun.write(workerPath, `
      const [sourceUrl, directory, sessionId, label] = process.argv.slice(2)
      const { DaemonTranscriptStore, normalizeDaemonTranscript } = await import(sourceUrl!)
      const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' })
      const transcript = normalizeDaemonTranscript({
        session_id: sessionId,
        messages: [
          { role: 'user', content: 'shared' },
          { role: 'assistant', content: label },
        ],
        turn_count: 1,
      }, { requestedSessionKey: sessionId, currentProjectDirectory: '/project' })
      if (!transcript) throw new Error('expected transcript to normalize')
      await Bun.write(directory + '/ready-' + label, '')
      while (!(await Bun.file(directory + '/go').exists())) await Bun.sleep(1)
      await store.save(transcript, { mode: 'append', expectedGeneration: 0, expectedMessageCount: 1 })
    `)
    const workers = ['first', 'second'].map(label => Bun.spawn([
      process.execPath,
      workerPath,
      sourceUrl,
      directory,
      sessionId,
      label,
    ], { stdout: 'pipe', stderr: 'pipe' }))
    while (!(await Bun.file(join(directory, 'ready-first')).exists())
      || !(await Bun.file(join(directory, 'ready-second')).exists())) await Bun.sleep(1)
    await Bun.write(join(directory, 'go'), '')
    const exits = await Promise.all(workers.map(worker => worker.exited))
    expect(exits).toEqual([0, 0])

    const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' })
    const contents = (await store.load(sessionId))?.messages.map(message => message.content)
    expect(contents?.[0]).toBe('shared')
    expect(contents?.slice(1).sort()).toEqual(['first', 'second'])
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('stale append writers preserve distinct suffixes while divergent and stale rewrite conflicts fail', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-transcript-conflicts-'))
  try {
    const sessionId = 'd1a7e6e0d1a7e6e0'
    const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' })
    const base = normalizeDaemonTranscript({
      session_id: sessionId,
      messages: [{ role: 'user', content: 'shared' }],
      turn_count: 1,
    }, { requestedSessionKey: sessionId, currentProjectDirectory: '/project' })
    if (!base) throw new Error('expected transcript to normalize')
    let generation = 0
    await store.save(base, { mode: 'rewrite', expectedGeneration: 0, onSavedGeneration: value => { generation = value } })
    await store.save({ ...base, messages: [...base.messages, { role: 'assistant', content: 'first' }] }, {
      mode: 'append', expectedGeneration: generation, expectedMessageCount: 1,
    })
    await store.save({ ...base, messages: [...base.messages, { role: 'assistant', content: 'second' }] }, {
      mode: 'append', expectedGeneration: generation, expectedMessageCount: 1,
    })
    expect((await store.load(sessionId))?.messages.map(message => message.content)).toEqual(['shared', 'first', 'second'])

    expect(store.save({ ...base, messages: [{ role: 'user', content: 'different' }, { role: 'assistant', content: 'bad' }] }, {
      mode: 'append', expectedGeneration: generation, expectedMessageCount: 1,
    })).rejects.toThrow('divergent append conflicts')
    expect(store.save({ ...base, messages: [{ role: 'system', content: 'stale summary' }] }, {
      mode: 'rewrite', expectedGeneration: generation,
    })).rejects.toThrow('stale rewrite conflicts')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('journal entries covered by a repaired snapshot are not re-spliced as duplicates', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-transcript-journal-shrink-'))
  try {
    const sessionId = 'f00df00df00df00d'
    const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' })
    // Raw persisted snapshot whose orphan tool reply resume repair will drop,
    // shrinking the live history below the raw list the journal indexes use.
    await Bun.write(store.pathFor(sessionId), JSON.stringify({
      generation: 3,
      messages: [
        { role: 'user', content: 'saved turn' },
        { role: 'tool', tool_call_id: 'ghost-call', content: 'orphan reply' },
      ],
      session_id: sessionId,
      turn_count: 1,
    }))
    // Journalled at raw position 2, after both persisted entries.
    await store.appendMessage(sessionId, { role: 'assistant', content: 'live answer' }, 2)

    const recovered = await store.load(sessionId)
    expect(recovered?.messages).toEqual([
      { role: 'user', content: 'saved turn' },
      { role: 'assistant', content: 'live answer' },
    ])

    await store.save(recovered!)
    // Coverage must be judged against the raw pre-repair length (3), not the
    // repaired length (2): otherwise this entry survives and the next load
    // splices a second copy of an already-persisted message.
    expect(await Bun.file(store.journalPathFor(sessionId)).exists()).toBeFalse()
    expect((await store.load(sessionId))?.messages).toEqual([
      { role: 'user', content: 'saved turn' },
      { role: 'assistant', content: 'live answer' },
    ])
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('new-era journal entries survive a shrink save and a stale follow-up save', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-transcript-journal-era-'))
  try {
    const sessionId = 'e12a34e12a34e12a'
    const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' })
    await Bun.write(store.pathFor(sessionId), JSON.stringify({
      generation: 7,
      messages: [
        { role: 'user', content: 'turn zero' },
        { role: 'tool', tool_call_id: 'ghost-call', content: 'orphan reply' },
      ],
      session_id: sessionId,
      turn_count: 1,
    }))
    // Old-era entry journalled against the raw list (position 2 of 2+orphan).
    await store.appendMessage(sessionId, { role: 'assistant', content: 'live answer' }, 2)

    // Load shrinks the history to two messages; saving it must publish the
    // shrunken coverage base so later entries numbered from it are judged
    // against the new era, not the frozen pre-repair length.
    const recovered = await store.load(sessionId)
    expect(recovered).toMatchObject({ messages: [{ content: 'turn zero' }, { content: 'live answer' }] })
    expect(recovered?.rawMessageCount).toBe(3)
    await store.save(recovered!)

    // New-era append numbering from the shrunken base (2), then a crash
    // before its message is ever saved.
    await store.appendMessage(sessionId, { role: 'user', content: 'new-era message' }, 2)

    // A stale writer — still holding the pre-shrink transcript whose frozen
    // raw count is 3 — saves again after another writer bumped the generation.
    // The old threshold would delete index 2 although that message was never
    // persisted anywhere.
    const rawSnapshot = JSON.parse(await Bun.file(store.pathFor(sessionId)).text()) as Record<string, unknown>
    rawSnapshot.generation = typeof rawSnapshot.generation === 'number' ? rawSnapshot.generation + 1 : 1
    await Bun.write(store.pathFor(sessionId), JSON.stringify(rawSnapshot))
    await store.save(
      { ...recovered!, generation: recovered!.generation ?? 0 },
      {
        mode: 'append',
        expectedGeneration: recovered!.generation ?? 0,
        expectedMessageCount: recovered!.messages.length,
      },
    )

    expect((await store.load(sessionId))?.messages).toEqual([
      { role: 'user', content: 'turn zero' },
      { role: 'assistant', content: 'live answer' },
      { role: 'user', content: 'new-era message' },
    ])
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('marker-stripped resume appends do not fabricate divergent conflicts', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-transcript-marker-resume-'))
  try {
    const sessionId = 'c1a5c1a5c1a5c1a5'
    const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' })
    await Bun.write(store.pathFor(sessionId), JSON.stringify({
      generation: 4,
      messages: [
        { role: 'user', content: 'read it' },
        {
          role: 'assistant',
          content: 'ASSISTANT_TOOL_CALLS: [{"id":"call-1","name":"ReadFile","input":{"path":"a.ts"}}]',
        },
      ],
      session_id: sessionId,
      turn_count: 1,
    }))

    const resumed = await store.load(sessionId)
    if (!resumed) throw new Error('expected transcript to load')
    expect(JSON.stringify(resumed.messages[1])).not.toContain('ASSISTANT_TOOL_CALLS')

    // Another writer advances the generation while the persisted bytes still
    // carry the raw provider markers that resume repair strips on load.
    const rawSnapshot = JSON.parse(await Bun.file(store.pathFor(sessionId)).text()) as Record<string, unknown>
    rawSnapshot.generation = typeof rawSnapshot.generation === 'number' ? rawSnapshot.generation + 1 : 1
    await Bun.write(store.pathFor(sessionId), JSON.stringify(rawSnapshot))

    // The stripped prefix differs byte-wise from disk but means the same
    // history, so the append merges instead of failing as divergent.
    await store.save(
      { ...resumed, messages: [...resumed.messages, { role: 'user', content: 'follow-up question' }] },
      {
        mode: 'append',
        expectedGeneration: resumed.generation ?? 0,
        expectedMessageCount: resumed.messages.length,
      },
    )

    const merged = await store.load(sessionId)
    expect(merged?.messages).toHaveLength(3)
    expect(merged?.messages.at(-1)?.content).toBe('follow-up question')
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('remove waits behind a journal append and cannot leave an orphaned sidecar', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-transcript-remove-race-'))
  try {
    const sessionId = 'deadc0dedeadc0de'
    const first = new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' })
    const second = new DaemonTranscriptStore({ directory, currentProjectDirectory: '/project' })
    const transcript = normalizeDaemonTranscript({
      session_id: sessionId,
      messages: [{ role: 'user', content: 'saved turn' }],
      turn_count: 1,
    }, { requestedSessionKey: sessionId, currentProjectDirectory: '/project' })
    if (!transcript) throw new Error('expected transcript to normalize')
    await first.save(transcript)

    await Promise.all([
      first.appendMessage(sessionId, { role: 'assistant', content: 'raced answer' }, 1),
      second.remove(sessionId),
    ])

    expect(await Bun.file(first.pathFor(sessionId)).exists()).toBeFalse()
    expect(await Bun.file(first.journalPathFor(sessionId)).exists()).toBeFalse()
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
