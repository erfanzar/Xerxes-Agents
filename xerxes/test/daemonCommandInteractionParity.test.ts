// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, readFile, realpath, rm, writeFile } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { COMPACTION_REFERENCE_PREFIX } from '../src/context/compressor.js'
import { DaemonInteractionBoard } from '../src/daemon/interactions.js'
import { InMemoryDaemonRuntime, type DaemonEvent, type DaemonSession, type TurnRunControls, type TurnRunner } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'

test('daemon completion preserves command and path semantics while native skills remain invocable by slash', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-daemon-completion-parity-'))
  const socketPath = join(directory, 'daemon.sock')
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    sessionDirectory: join(directory, 'sessions'),
  })
  const server = new DaemonServer({ socketPath, runtime })
  await writeFile(join(directory, 'alpha.txt'), 'alpha', 'utf8')
  await writeFile(join(directory, 'my file.md'), 'space', 'utf8')
  await writeFile(join(directory, '.hidden'), 'hidden', 'utf8')
  await mkdir(join(directory, 'alphabeta'))
  await mkdir(join(directory, 'direct-output'))
  await writeFile(join(directory, 'alphabeta', 'nested-alpha.md'), 'nested', 'utf8')
  await server.start()
  const client = await DaemonParityClient.connect(socketPath)
  try {
    await initialize(client, 1, 'completion', directory)

    client.send({ jsonrpc: '2.0', id: 2, method: 'complete', params: { text: '/prov' } })
    const slash = await client.next(frame => frame.id === 2)
    expect(slash.result).toMatchObject({ ok: true, kind: 'slash' })
    expect(slash.result?.completions).toEqual(expect.arrayContaining([
      expect.objectContaining({ value: '/provider', label: 'provider', meta: expect.any(String) }),
    ]))

    client.send({ jsonrpc: '2.0', id: 3, method: 'complete', params: { text: '/zzzz-not-a-command' } })
    expect((await client.next(frame => frame.id === 3)).result?.completions).toEqual([])

    client.send({ jsonrpc: '2.0', id: 4, method: 'complete', params: { text: './alph' } })
    expect((await client.next(frame => frame.id === 4)).result?.completions).toEqual(expect.arrayContaining([
      { value: './alpha.txt', label: 'alpha.txt', meta: 'file' },
      { value: './alphabeta/', label: 'alphabeta/', meta: 'dir' },
    ]))

    client.send({ jsonrpc: '2.0', id: 5, method: 'complete', params: { text: '@./al' } })
    expect((await client.next(frame => frame.id === 5)).result?.completions).toEqual([
      { value: '@alpha.txt', label: 'alpha.txt', meta: 'file' },
      { value: '@alphabeta/nested-alpha.md', label: 'alphabeta/nested-alpha.md', meta: 'file' },
    ])

    client.send({ jsonrpc: '2.0', id: 11, method: 'complete', params: { text: 'inspect @alph' } })
    expect((await client.next(frame => frame.id === 11)).result?.completions).toEqual([
      { value: '@alpha.txt', label: 'alpha.txt', meta: 'file' },
      { value: '@alphabeta/nested-alpha.md', label: 'alphabeta/nested-alpha.md', meta: 'file' },
    ])

    client.send({ jsonrpc: '2.0', id: 12, method: 'complete', params: { text: '@' } })
    expect((await client.next(frame => frame.id === 12)).result?.completions).toEqual([])

    client.send({ jsonrpc: '2.0', id: 13, method: 'complete', params: { text: '@my' } })
    expect((await client.next(frame => frame.id === 13)).result?.completions).toEqual([
      { value: '@"my file.md"', label: 'my file.md', meta: 'file' },
    ])

    client.send({ jsonrpc: '2.0', id: 14, method: 'complete', params: { text: '@direct-output' } })
    expect((await client.next(frame => frame.id === 14)).result?.completions).toEqual([
      { value: '@direct-output/', label: 'direct-output/', meta: 'dir' },
    ])

    client.send({ jsonrpc: '2.0', id: 6, method: 'complete', params: { text: './' } })
    expect((await client.next(frame => frame.id === 6)).result?.completions).not.toEqual(expect.arrayContaining([
      expect.objectContaining({ label: '.hidden' }),
    ]))

    client.send({ jsonrpc: '2.0', id: 7, method: 'complete', params: { text: './.' } })
    expect((await client.next(frame => frame.id === 7)).result?.completions).toEqual([
      { value: './.hidden', label: '.hidden', meta: 'file' },
    ])

    client.send({ jsonrpc: '2.0', id: 8, method: 'complete', params: { text: 'bare word' } })
    expect((await client.next(frame => frame.id === 8)).result?.completions).toEqual([])

    client.send({ jsonrpc: '2.0', id: 9, method: 'commands.catalog', params: {} })
    expect((await client.next(frame => frame.id === 9)).result).toMatchObject({ skill_count: expect.any(Number), sub: {} })
    client.send({ jsonrpc: '2.0', id: 10, method: 'complete', params: { text: '/deepscan' } })
    expect((await client.next(frame => frame.id === 10)).result?.completions).toEqual([])
  } finally {
    client.close()
    await server.stop()
    await rm(directory, { recursive: true, force: true })
  }
})

test('daemon question interactions emit a request, resolve matching answers, and fail safely when unwired or cancelled', async () => {
  const board = new DaemonInteractionBoard()
  await expect(board.ask('missing-session', { question: 'What should happen?' })).rejects.toThrow(
    'outside an active daemon turn',
  )

  const events: DaemonEvent[] = []
  const release = board.bind('question-session', event => events.push(event))
  try {
    const answer = board.ask('question-session', {
      question: 'What is the goal?',
      toolCallId: 'tool-call-1',
    })
    const request = events[0]
    const requestId = String(request?.payload.id)
    expect(request).toEqual({
      type: 'question_request',
      payload: {
        id: requestId,
        tool_call_id: 'tool-call-1',
        questions: [{ id: 'answer', question: 'What is the goal?', options: [], allow_free_form: true }],
      },
    })
    expect(board.respondQuestion('unknown-request', { answer: 'ignored' })).toBe(false)
    expect(board.respondQuestion(requestId, { answer: 'ship the native port' })).toBe(true)
    await expect(answer).resolves.toBe('ship the native port')

    const controller = new AbortController()
    const cancelled = board.ask('question-session', { question: 'Wait for cancellation.' }, controller.signal)
    expect(board.pendingQuestionIds()).toHaveLength(1)
    controller.abort()
    await expect(cancelled).resolves.toBe('')
    expect(board.pendingQuestionIds()).toEqual([])
  } finally {
    release()
  }
})

test('slash steering stays on the issuing connection session and queues at the active turn boundary', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-daemon-slash-steer-parity-'))
  const socketPath = join(directory, 'daemon.sock')
  const runner = new SteerBoundaryRunner()
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: directory,
    model: 'steer-model',
    sessionDirectory: join(directory, 'sessions'),
  })
  const server = new DaemonServer({ socketPath, runtime })
  await Bun.write(join(directory, 'steer.md'), 'steer attachment')
  const canonicalDirectory = await realpath(directory)
  await server.start()
  const other = await DaemonParityClient.connect(socketPath)
  const target = await DaemonParityClient.connect(socketPath)
  try {
    await initialize(other, 1, 'other-session', directory)
    await initialize(target, 2, 'target-session', directory)

    target.send({ jsonrpc: '2.0', id: 3, method: 'turn.submit', params: { text: 'start a controlled turn' } })
    expect((await target.next(frame => frame.id === 3)).result).toEqual({ ok: true })
    await target.next(eventFrame('turn_begin'))
    expect((await target.next(eventFrame('text_part'))).params?.payload).toEqual({ text: 'waiting for steer' })

    target.send({ jsonrpc: '2.0', id: 4, method: 'slash', params: { command: '/steer focus @steer.md' } })
    expect((await target.next(frame => frame.id === 4)).result).toEqual({ ok: true })
    expect((await target.next(eventFrame('steer_input'))).params?.payload).toEqual({
      content: 'focus @steer.md',
      mentioned_files: [join(canonicalDirectory, 'steer.md')],
    })
    expect((await target.next(eventFrame('notification'))).params?.payload).toMatchObject({
      category: 'slash',
      body: 'Steer accepted.',
    })

    other.send({ jsonrpc: '2.0', id: 5, method: 'session.status', params: {} })
    expect((await other.next(frame => frame.id === 5)).result?.session).toMatchObject({ key: 'other-session', messages: 0 })

    runner.release()
    const steered = (await target.next(eventFrame('text_part'))).params?.payload?.text
    expect(steered).toContain('<attached_files>')
    expect(steered).toContain('1 | steer attachment')
    expect(steered).toContain('focus @steer.md')
    await target.next(eventFrame('turn_end'))

    target.send({ jsonrpc: '2.0', id: 6, method: 'slash', params: { command: '/steer' } })
    expect((await target.next(frame => frame.id === 6)).result).toEqual({ ok: false, error: 'steer text is required' })
    expect((await target.next(eventFrame('notification'))).params?.payload).toMatchObject({
      category: 'slash',
      severity: 'warning',
      body: 'Usage: `/steer <hint>`.',
    })
  } finally {
    other.close()
    target.close()
    await server.stop()
    await rm(directory, { recursive: true, force: true })
  }
})

test('slash compact rewrites and persists the active native session without submitting a model turn', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-daemon-compact-parity-'))
  const sessionDirectory = join(directory, 'sessions')
  const socketPath = join(directory, 'daemon.sock')
  const runtime = new InMemoryDaemonRuntime(new UnexpectedTurnRunner(), {
    currentProjectDirectory: directory,
    model: 'compact-model',
    sessionDirectory,
  })
  const server = new DaemonServer({ socketPath, runtime })
  const nativeFetch = globalThis.fetch
  const providerRequests: unknown[] = []
  globalThis.fetch = (async (_input: unknown, init?: RequestInit) => {
    providerRequests.push(typeof init?.body === 'string' ? JSON.parse(init.body) : undefined)
    return new Response(
      JSON.stringify({
        choices: [{ message: { content: 'durable parity summary' } }],
      }),
    )
  }) as typeof globalThis.fetch
  await server.start()
  const client = await DaemonParityClient.connect(socketPath)
  try {
    await initialize(client, 1, 'compact-session', directory)
    const session = runtime.sessionStatus('compact-session')
    if (!session) {
      throw new Error('expected initialized compact session')
    }
    session.messages = Array.from({ length: 12 }, (_, index) => ({
      role: index % 2 ? 'assistant' : 'user',
      content: `message-${index} ${'context '.repeat(24)}`,
    }))

    client.send({ jsonrpc: '2.0', id: 2, method: 'slash', params: { command: '/compact' } })
    const compacted = await client.next(frame => frame.id === 2)
    expect(compacted.result).toMatchObject({ ok: true, compacted: true, tokens_before: expect.any(Number), tokens_after: expect.any(Number) })
    expect((await client.next(eventFrame('notification'))).params?.payload).toMatchObject({
      category: 'slash',
      body: expect.stringContaining('Compacted'),
    })
    expect((await client.next(eventFrame('status_update'))).params?.payload).toMatchObject({
      context_tokens: expect.any(Number),
      max_context: 128_000,
    })

    // Model-backed compaction: the provider was invoked with the transcript
    // and its summary replaced the compactable middle (no model turn ran).
    expect(providerRequests.length).toBeGreaterThan(0)
    expect(session.messages.length).toBeLessThan(12)
    const summary = session.messages.find(message =>
      typeof message.content === 'string' && message.content.includes(COMPACTION_REFERENCE_PREFIX),
    )
    expect(summary?.content).toEqual(expect.stringContaining('durable parity summary'))
    expect(session.metadata.last_compaction).toMatchObject({
      tokens_before: expect.any(Number),
      tokens_after: expect.any(Number),
    })

    const persisted = JSON.parse(await readFile(join(sessionDirectory, `${session.id}.json`), 'utf8')) as {
      readonly messages: unknown[]
      readonly metadata: Record<string, unknown>
    }
    expect(persisted.messages).toEqual(session.messages)
    expect(persisted.metadata.last_compaction).toEqual(session.metadata.last_compaction)
  } finally {
    globalThis.fetch = nativeFetch
    client.close()
    await server.stop()
    await rm(directory, { recursive: true, force: true })
  }
})

class SteerBoundaryRunner implements TurnRunner {
  private releaseGate: (() => void) | undefined
  private readonly gate = new Promise<void>(resolve => { this.releaseGate = resolve })

  release(): void {
    this.releaseGate?.()
  }

  async *run(_session: DaemonSession, _text: string, _signal: AbortSignal, controls?: TurnRunControls): AsyncGenerator<DaemonEvent> {
    yield { type: 'text_part', payload: { text: 'waiting for steer' } }
    await this.gate
    yield { type: 'text_part', payload: { text: `steer:${controls?.drainSteer?.().join('|') ?? ''}` } }
  }
}

class UnexpectedTurnRunner implements TurnRunner {
  async *run(): AsyncGenerator<DaemonEvent> {
    throw new Error('/compact must not submit a model turn')
  }
}

interface Frame {
  readonly id?: number | string
  readonly method?: string
  readonly params?: {
    readonly payload?: Record<string, unknown>
    readonly type?: string
  }
  readonly result?: Record<string, unknown>
}

async function initialize(client: DaemonParityClient, id: number, sessionKey: string, directory: string): Promise<void> {
  client.send({ jsonrpc: '2.0', id, method: 'initialize', params: { session_key: sessionKey, project_dir: directory } })
  expect((await client.next(frame => frame.id === id)).result).toMatchObject({ ok: true, session: { key: sessionKey } })
  await client.next(eventFrame('init_done'))
  await client.next(eventFrame('status_update'))
}

function eventFrame(type: string): (frame: Frame) => boolean {
  return frame => frame.method === 'event' && frame.params?.type === type
}

class DaemonParityClient {
  private buffer = ''
  private readonly frames: Frame[] = []
  private readonly waiters: Array<{ predicate: (frame: Frame) => boolean; resolve: (frame: Frame) => void }> = []

  private constructor(private readonly socket: Socket) {
    socket.setEncoding('utf8')
    socket.on('data', chunk => this.receive(typeof chunk === 'string' ? chunk : new TextDecoder().decode(chunk)))
  }

  static async connect(socketPath: string): Promise<DaemonParityClient> {
    const socket = connect({ path: socketPath })
    await new Promise<void>((resolveConnection, rejectConnection) => {
      socket.once('connect', resolveConnection)
      socket.once('error', rejectConnection)
    })
    return new DaemonParityClient(socket)
  }

  close(): void {
    this.socket.destroy()
  }

  next(predicate: (frame: Frame) => boolean): Promise<Frame> {
    const index = this.frames.findIndex(predicate)
    if (index >= 0) {
      const frame = this.frames.splice(index, 1)[0]
      if (frame) {
        return Promise.resolve(frame)
      }
    }
    return new Promise(resolveFrame => this.waiters.push({ predicate, resolve: resolveFrame }))
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
      if (line.trim()) {
        this.handle(JSON.parse(line) as Frame)
      }
      newline = this.buffer.indexOf('\n')
    }
  }

  private handle(frame: Frame): void {
    const index = this.waiters.findIndex(waiter => waiter.predicate(frame))
    const waiter = index >= 0 ? this.waiters.splice(index, 1)[0] : undefined
    if (waiter) {
      waiter.resolve(frame)
      return
    }
    this.frames.push(frame)
  }
}
