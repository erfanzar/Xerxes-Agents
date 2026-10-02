// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  InMemoryDaemonRuntime,
  type DaemonEvent,
  type DaemonSession,
  type TurnRunControls,
  type TurnRunner,
} from '../src/daemon/runtime.js'
import { DaemonTranscriptStore } from '../src/session/daemonTranscript.js'

/**
 * A state-managing runner shaped like the native one: it journals every
 * message as it is produced and synchronizes the session only at turn end.
 * A prompt starting with "hang" journals its messages and then waits for
 * cancellation, which is where a crash would find it.
 */
class JournallingRunner implements TurnRunner {
  readonly managesSessionState = true
  hanging = Promise.withResolvers<void>()
  release = Promise.withResolvers<void>()

  async *run(session: DaemonSession, text: string, signal: AbortSignal, controls?: TurnRunControls): AsyncGenerator<DaemonEvent> {
    const base = session.messages.length
    const produced: DaemonSession['messages'] = [
      { role: 'user', content: text },
      { role: 'assistant', content: `reply to ${text}` },
    ]
    produced.forEach((message, index) => controls?.journal?.(message, base + index))
    if (text.startsWith('hang')) {
      this.hanging.resolve()
      await new Promise<void>((resolve, reject) => {
        signal.addEventListener('abort', () => reject(signal.reason), { once: true })
        void this.release.promise.then(resolve)
      })
    }
    session.messages = [...session.messages, ...produced]
    session.turnCount += 1
    yield { type: 'text_part', payload: { text: `reply to ${text}` } }
    yield { type: 'status_update', payload: { stop_reason: 'completed' } }
  }
}

async function journalRows(store: DaemonTranscriptStore, sessionId: string, count: number): Promise<void> {
  // Journal appends are fire-and-forget; wait until they reached the file.
  for (let attempt = 0; attempt < 200; attempt += 1) {
    const file = Bun.file(store.journalPathFor(sessionId))
    if (await file.exists() && (await file.text()).split('\n').filter(Boolean).length >= count) return
    await Bun.sleep(5)
  }
  throw new Error(`journal for ${sessionId} never reached ${count} rows`)
}

/** What a restarted daemon would read back: a fresh store over the same files. */
async function reloadedContents(directory: string, sessionId: string): Promise<unknown[]> {
  const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: directory })
  return ((await store.load(sessionId))?.messages ?? []).map(message => message.content)
}

test('a crash during a later turn recovers that turn even when the saved turn ended in an unjournalled message', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-crash-journal-'))
  const sessions = join(directory, 'sessions')
  const store = new DaemonTranscriptStore({ directory: sessions, currentProjectDirectory: directory })
  const runner = new JournallingRunner()
  const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: directory, transcriptStore: store })
  try {
    const session = await runtime.openSession('tui:crash')
    // Turn one ends normally, with a steer saved for the next turn as its last
    // stored message: the journal never carried that message.
    const first = runtime.submitTurn(session.sessionKey, 'hang first', () => {})
    await runner.hanging.promise
    expect(runtime.steerTurn(session.sessionKey, 'keep going')).toBe(true)
    runner.release.resolve()
    await first
    expect(session.messages.at(-1)?.content).toBe('[steer from user saved for next turn]\nkeep going')

    // Turn two journals its work, then the daemon dies before turn end.
    runner.hanging = Promise.withResolvers()
    runner.release = Promise.withResolvers()
    const second = runtime.submitTurn(session.sessionKey, 'hang second', () => {})
    await runner.hanging.promise
    await journalRows(store, session.id, 4)

    expect(await reloadedContents(sessions, session.id)).toEqual([
      'hang first', 'reply to hang first', '[steer from user saved for next turn]\nkeep going',
      'hang second', 'reply to hang second',
    ])
    runtime.cancelTurn(session.sessionKey)
    await second
  } finally {
    await runtime.shutdown()
    await rm(directory, { recursive: true, force: true })
  }
})

test('a rewrite flush for one session cannot mark another session\'s running turn as saved', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-crash-rewrite-'))
  const sessions = join(directory, 'sessions')
  const store = new DaemonTranscriptStore({ directory: sessions, currentProjectDirectory: directory })
  const runner = new JournallingRunner()
  const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: directory, transcriptStore: store })
  try {
    const compacted = await runtime.openSession('tui:compacted')
    await runtime.submitTurn(compacted.sessionKey, 'compact me', () => {})
    const running = await runtime.openSession('tui:running')
    await runtime.submitTurn(running.sessionKey, 'earlier', () => {})

    const turn = runtime.submitTurn(running.sessionKey, 'hang long goal round', () => {})
    await runner.hanging.promise
    await journalRows(store, running.id, 4)
    const expected = ['earlier', 'reply to earlier', 'hang long goal round', 'reply to hang long goal round']

    // /compact, auto-compact and /undo on the other session rewrite it.
    await runtime.flushSessions('rewrite', compacted.sessionKey)
    expect(await reloadedContents(sessions, running.id)).toEqual(expected)
    // Even an unscoped rewrite leaves a running turn's journal replayable.
    await runtime.flushSessions('rewrite')
    expect(await reloadedContents(sessions, running.id)).toEqual(expected)

    runtime.cancelTurn(running.sessionKey)
    await turn
  } finally {
    await runtime.shutdown()
    await rm(directory, { recursive: true, force: true })
  }
})

test('a steer that lands while the finished turn is saving is saved for the next turn, not stranded in the queue', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-late-steer-'))
  const sessions = join(directory, 'sessions')
  const store = new DaemonTranscriptStore({ directory: sessions, currentProjectDirectory: directory })
  const runtime = new InMemoryDaemonRuntime(new JournallingRunner(), { currentProjectDirectory: directory, transcriptStore: store })
  try {
    const session = await runtime.openSession('tui:late-steer')
    let accepted: boolean | undefined
    await runtime.submitTurn(session.sessionKey, 'finish', event => {
      if (event.type === 'turn_end') accepted = runtime.steerTurn(session.sessionKey, 'one more thing')
    })

    expect(accepted).toBe(true)
    // A stranded steer blocks every later goal round from being admitted.
    expect(runtime.hasPendingSteer(session.sessionKey)).toBe(false)
    expect(session.messages.at(-1)?.content).toBe('[steer from user saved for next turn]\none more thing')
    // It is durable, not only in memory.
    expect((await reloadedContents(sessions, session.id)).at(-1)).toBe('[steer from user saved for next turn]\none more thing')
  } finally {
    await runtime.shutdown()
    await rm(directory, { recursive: true, force: true })
  }
})
