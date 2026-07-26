// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { DaemonServer } from '../src/daemon/server.js'
import {
  InMemoryDaemonRuntime,
  type DaemonEvent,
  type DaemonSession,
  type TurnRunner,
} from '../src/daemon/runtime.js'
import {
  DaemonTranscriptStore,
  normalizeDaemonTranscript,
} from '../src/session/daemonTranscript.js'

async function withTempDirectory(run: (directory: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-history-survival-'))
  try {
    await run(directory)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
}

function runtimeFor(directory: string, model?: string): InMemoryDaemonRuntime {
  return new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    ...(model === undefined ? {} : { model }),
    sessionDirectory: join(directory, 'sessions'),
  })
}

test('transcript store never deletes persisted history as a side effect of an empty save', async () => {
  await withTempDirectory(async directory => {
    const sessionId = 'abc123def456'
    const store = new DaemonTranscriptStore({ directory, currentProjectDirectory: directory })
    const transcript = normalizeDaemonTranscript(
      {
        session_id: sessionId,
        messages: [
          { role: 'user', content: 'keep me' },
          { role: 'assistant', content: 'kept too' },
        ],
        turn_count: 1,
      },
      { currentProjectDirectory: directory, requestedSessionKey: sessionId },
    )
    if (!transcript) throw new Error('expected transcript to normalize')
    await store.save(transcript)

    // A routine save of an empty in-memory session bound to the same id must
    // leave the persisted record untouched; only remove() may delete it.
    await store.save({ ...transcript, messages: [], turnCount: 0 })

    expect(await Bun.file(store.pathFor(sessionId)).exists()).toBe(true)
    const loaded = await store.load(sessionId)
    expect(loaded?.messages).toHaveLength(2)
    expect(loaded?.turnCount).toBe(1)
    expect((await store.list()).map(entry => entry.sessionId)).toEqual([sessionId])

    expect(await store.remove(sessionId)).toBe(true)
    expect(await Bun.file(store.pathFor(sessionId)).exists()).toBe(false)
  })
})

test('session history survives a full daemon restart through resume', async () => {
  await withTempDirectory(async directory => {
    const first = runtimeFor(directory)
    const session = await first.openSession('tui:slot')
    await first.submitTurn('tui:slot', 'persist across restart', () => {})
    await first.flushSessions()

    const second = runtimeFor(directory)
    const resumed = await second.openSession(session.id, undefined, { resume: true })
    expect(resumed.id).toBe(session.id)
    expect(resumed.messages.map(message => message.role)).toEqual(['user', 'assistant'])
    expect(resumed.turnCount).toBe(1)
  })
})

test('flushing a fresh empty session bound to a persisted id does not erase its history', async () => {
  await withTempDirectory(async directory => {
    const first = runtimeFor(directory)
    const session = await first.openSession('tui:origin')
    await first.submitTurn('tui:origin', 'original history', () => {})
    await first.flushSessions()

    // A client that reopens the id without resuming (initialize with a hex
    // session_key but no resume flag, or a skipped resume) gets a fresh empty
    // session bound to the same id. Its shutdown flush must not delete the
    // persisted transcript.
    const second = runtimeFor(directory)
    const impostor = await second.openSession(session.id, undefined, { resume: false })
    expect(impostor.messages).toHaveLength(0)
    expect(impostor.turnCount).toBe(0)
    await second.flushSessions()

    const third = runtimeFor(directory)
    const survived = await third.openSession(session.id, undefined, { resume: true })
    expect(survived.messages.map(message => message.role)).toEqual(['user', 'assistant'])
    expect(survived.turnCount).toBe(1)
  })
})

test('resume folds a live duplicate registered under a stale key instead of racing saves', async () => {
  await withTempDirectory(async directory => {
    const runtime = runtimeFor(directory)
    const original = await runtime.openSession('tui:slot')
    await runtime.submitTurn('tui:slot', 'first turn', () => {})
    // Simulate state newer than the last save (idle steer, metadata edit).
    original.messages.push({ role: 'user', content: 'unsaved edit' })

    const adopted = await runtime.openSession(original.id, undefined, { resume: true })

    expect(adopted.id).toBe(original.id)
    expect(runtime.listSessions().filter(session => session.id === original.id)).toHaveLength(1)
    // The live copy's unsaved state was persisted before the stale key was
    // dropped, so the adopted session loses nothing.
    expect(adopted.messages.some(message => message.content === 'unsaved edit')).toBe(true)

    await runtime.flushSessions()
    const restarted = runtimeFor(directory)
    const resumed = await restarted.openSession(original.id, undefined, { resume: true })
    expect(resumed.messages.some(message => message.content === 'unsaved edit')).toBe(true)
    expect(resumed.turnCount).toBe(1)
  })
})

test('resume rejects adopting a session that is running a turn under another connection', async () => {
  await withTempDirectory(async directory => {
    const blockingRunner: TurnRunner = {
      managesSessionState: true,
      async *run(
        _session: DaemonSession,
        _text: string,
        signal: AbortSignal,
      ): AsyncGenerator<DaemonEvent> {
        await new Promise<void>(resolve => {
          if (signal.aborted) {
            resolve()
            return
          }
          signal.addEventListener('abort', () => resolve(), { once: true })
        })
      },
    }
    const runtime = new InMemoryDaemonRuntime(blockingRunner, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, 'sessions'),
    })
    const session = await runtime.openSession('tui:blocked')
    const pending = runtime.submitTurn('tui:blocked', 'long work', () => {})
    await Bun.sleep(20)
    expect(session.activeTurnId).not.toBe('')

    await expect(
      runtime.openSession(session.id, undefined, { resume: true }),
    ).rejects.toThrow(/still running a turn/)

    expect(runtime.cancelTurn('tui:blocked')).toBe(true)
    await pending
  })
})

interface TestFrame {
  readonly error?: { readonly code?: number; readonly message?: string }
  readonly id?: number
  readonly method?: string
  readonly params?: {
    readonly payload?: Record<string, unknown>
    readonly type?: string
  }
  readonly result?: Record<string, unknown>
}

function eventFrame(type: string): (frame: TestFrame) => boolean {
  return frame => frame.method === 'event' && frame.params?.type === type
}

class HistoryTestClient {
  private buffer = ''
  private readonly frames: TestFrame[] = []
  private readonly waiters: Array<{
    predicate: (frame: TestFrame) => boolean
    resolve: (frame: TestFrame) => void
  }> = []

  private constructor(private readonly socket: Socket) {
    socket.setEncoding('utf8')
    socket.on('data', chunk =>
      this.receive(typeof chunk === 'string' ? chunk : new TextDecoder().decode(chunk)))
  }

  static async connect(socketPath: string): Promise<HistoryTestClient> {
    const socket = connect({ path: socketPath })
    await new Promise<void>((resolve, reject) => {
      socket.once('connect', resolve)
      socket.once('error', reject)
    })
    return new HistoryTestClient(socket)
  }

  close(): void {
    this.socket.destroy()
  }

  next(predicate: (frame: TestFrame) => boolean): Promise<TestFrame> {
    const index = this.frames.findIndex(predicate)
    if (index >= 0) {
      const frame = this.frames.splice(index, 1)[0]
      if (frame) return Promise.resolve(frame)
    }
    return new Promise(resolve => this.waiters.push({ predicate, resolve }))
  }

  send(frame: Record<string, unknown>): void {
    this.socket.write(`${JSON.stringify(frame)}\n`)
  }

  private receive(chunk: string): void {
    this.buffer += chunk
    let newline = this.buffer.indexOf('\n')
    while (newline >= 0) {
      const line = this.buffer.slice(0, newline)
      this.buffer = this.buffer.slice(newline + 1)
      if (line.trim()) this.handle(JSON.parse(line) as TestFrame)
      newline = this.buffer.indexOf('\n')
    }
  }

  private handle(frame: TestFrame): void {
    const index = this.waiters.findIndex(waiter => waiter.predicate(frame))
    if (index >= 0) {
      const [waiter] = this.waiters.splice(index, 1)
      waiter?.resolve(frame)
      return
    }
    this.frames.push(frame)
  }
}

async function submitTurnAndWait(client: HistoryTestClient, id: number, text: string): Promise<void> {
  client.send({ jsonrpc: '2.0', id, method: 'turn.submit', params: { text } })
  await client.next(frame => frame.id === id)
  await client.next(eventFrame('turn_end'))
}

async function listSavedIds(client: HistoryTestClient, id: number): Promise<readonly string[]> {
  client.send({ jsonrpc: '2.0', id, method: 'session.list', params: {} })
  const frame = await client.next(candidate => candidate.id === id)
  const sessions = Array.isArray(frame.result?.sessions) ? frame.result.sessions : []
  return sessions.map(session => String((session as Record<string, unknown>).session_id ?? ''))
}

test('daemon undo of the last remaining turn removes the transcript explicitly but keeps the session usable', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-history-undo-'))
  const socketPath = join(directory, 'daemon.sock')
  const server = new DaemonServer({
    socketPath,
    runtime: runtimeFor(directory, 'history-test-model'),
  })
  await server.start()
  const client = await HistoryTestClient.connect(socketPath)
  try {
    client.send({
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: { session_key: 'tui:undo-all', project_dir: directory },
    })
    const init = await client.next(frame => frame.id === 1)
    const sessionId = String(init.result?.session_id ?? '')
    expect(sessionId).not.toBe('')

    await submitTurnAndWait(client, 2, 'only turn')
    expect(await listSavedIds(client, 3)).toContain(sessionId)

    client.send({ jsonrpc: '2.0', id: 4, method: 'session.undo', params: {} })
    expect((await client.next(frame => frame.id === 4)).result).toMatchObject({ ok: true, dropped: 2 })

    // Undoing down to zero turns clears the persisted record on purpose...
    expect(await listSavedIds(client, 5)).not.toContain(sessionId)

    // ...while the live session stays usable and persists new history.
    await submitTurnAndWait(client, 6, 'fresh start')
    expect(await listSavedIds(client, 7)).toContain(sessionId)
  } finally {
    client.close()
    await server.stop()
    await rm(directory, { recursive: true, force: true })
  }
})

test('daemon partial undo keeps the persisted transcript and its remaining history', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-history-partial-undo-'))
  const socketPath = join(directory, 'daemon.sock')
  const server = new DaemonServer({
    socketPath,
    runtime: runtimeFor(directory, 'history-test-model'),
  })
  await server.start()
  const client = await HistoryTestClient.connect(socketPath)
  try {
    client.send({
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: { session_key: 'tui:partial', project_dir: directory },
    })
    const init = await client.next(frame => frame.id === 1)
    const sessionId = String(init.result?.session_id ?? '')

    await submitTurnAndWait(client, 2, 'first turn')
    await submitTurnAndWait(client, 3, 'second turn')

    client.send({ jsonrpc: '2.0', id: 4, method: 'session.undo', params: {} })
    expect((await client.next(frame => frame.id === 4)).result).toMatchObject({ ok: true, dropped: 2 })

    client.send({ jsonrpc: '2.0', id: 5, method: 'session.list', params: {} })
    const listed = await client.next(frame => frame.id === 5)
    const sessions = Array.isArray(listed.result?.sessions) ? listed.result.sessions : []
    const row = sessions.find(
      session => String((session as Record<string, unknown>).session_id ?? '') === sessionId,
    ) as Record<string, unknown> | undefined
    expect(row).toBeDefined()
    expect(row?.turn_count).toBe(1)
    expect(row?.message_count).toBe(2)
  } finally {
    client.close()
    await server.stop()
    await rm(directory, { recursive: true, force: true })
  }
})
