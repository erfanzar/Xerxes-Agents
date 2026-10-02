// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readdir, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonTranscriptStore } from '../src/session/daemonTranscript.js'

test('a live session that loses a save conflict is preserved before the daemon reloads the saved copy', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-divergent-'))
  const sessionDirectory = join(directory, 'sessions')
  const store = new DaemonTranscriptStore({ directory: sessionDirectory, currentProjectDirectory: directory })
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, transcriptStore: store })
  try {
    const session = await runtime.openSession('task')
    session.messages.push({ role: 'user', content: 'first ask' }, { role: 'assistant', content: 'first answer' })
    await runtime.flushSessions()

    // Another writer replaces the transcript on disk with an older history.
    const path = store.pathFor(session.id)
    const disk = await Bun.file(path).json() as Record<string, unknown>
    await Bun.write(path, JSON.stringify({ ...disk, generation: (disk.generation as number) + 5, messages: [{ role: 'user', content: 'older ask' }, { role: 'assistant', content: 'older answer' }] }))

    // An hour of work that only this live copy holds.
    session.messages.push({ role: 'user', content: 'latest ask' }, { role: 'assistant', content: 'latest answer' })
    await runtime.flushSessions()

    const kept = await readdir(join(sessionDirectory, 'divergent'))
    expect(kept).toHaveLength(1)
    const preserved = await Bun.file(join(sessionDirectory, 'divergent', kept[0]!)).json() as { messages: Array<{ content: string }>; divergent_reason: string }
    expect(preserved.messages.map(message => message.content)).toEqual(['first ask', 'first answer', 'latest ask', 'latest answer'])
    expect(preserved.divergent_reason).toContain('conflicts with persisted history')
    // The preserved copy is not a session of its own in listings.
    expect((await store.listEntries()).map(entry => entry.sessionId)).toEqual([session.id])
  } finally {
    await runtime.shutdown()
    await rm(directory, { recursive: true, force: true })
  }
})

test('a session evicted for a save conflict is reloaded under the slot key its window still uses', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-divergent-reload-'))
  const sessionDirectory = join(directory, 'sessions')
  const store = new DaemonTranscriptStore({ directory: sessionDirectory, currentProjectDirectory: directory })
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, transcriptStore: store })
  try {
    // A New-task slot key is not an id, so it can never resume by itself.
    const session = await runtime.openSession('desktop-ab12xyz9')
    session.messages.push({ role: 'user', content: 'first ask' }, { role: 'assistant', content: 'first answer' })
    await runtime.flushSessions()

    const path = store.pathFor(session.id)
    const disk = await Bun.file(path).json() as Record<string, unknown>
    await Bun.write(path, JSON.stringify({ ...disk, generation: (disk.generation as number) + 5, messages: [{ role: 'user', content: 'saved ask' }, { role: 'assistant', content: 'saved answer' }] }))
    session.messages.push({ role: 'user', content: 'diverged ask' }, { role: 'assistant', content: 'diverged answer' })
    await runtime.flushSessions()

    // The window's next message must reach the same conversation, not a new one.
    const reopened = await runtime.openSession('desktop-ab12xyz9')
    expect(reopened.id).toBe(session.id)
    expect(reopened.messages.map(message => message.content)).toEqual(['saved ask', 'saved answer'])
    // And it saves cleanly against the reloaded generation.
    reopened.messages.push({ role: 'user', content: 'next ask' }, { role: 'assistant', content: 'next answer' })
    await runtime.flushSessions()
    expect((await store.load(session.id))?.messages.map(message => message.content)).toEqual(['saved ask', 'saved answer', 'next ask', 'next answer'])
  } finally {
    await runtime.shutdown()
    await rm(directory, { recursive: true, force: true })
  }
})
