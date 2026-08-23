// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, realpath, rm } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { DaemonTranscriptStore } from '../src/session/daemonTranscript.js'

test('explicit daemon resume replays visible turns but filters internal prompts and tool output', async () => {
  // Canonicalize the temp dir: on macOS /tmp is a symlink to /private/tmp, and
  // the daemon rejects resuming a transcript whose stored project dir differs
  // from the (realpath-resolved) project dir the client connects with.
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-daemon-replay-parity-')))
  const sessionDirectory = join(directory, 'sessions')
  const socketPath = join(directory, 'daemon.sock')
  const sessionId = 'b1c2d3e4'
  const store = new DaemonTranscriptStore({
    currentProjectDirectory: directory,
    directory: sessionDirectory,
    workspaceRoot: join(directory, 'agents'),
  })
  await store.save({
    agentId: 'default',
    cwd: directory,
    extra: {},
    format: 'bun-v2',
    interactionMode: 'code',
    key: sessionId,
    messages: [
      { role: 'user', content: 'visible question' },
      {
        role: 'assistant',
        content: [{ text: 'visible answer' }, { content: 'with a second line' }],
        thinking: 'persisted reasoning trace',
        tool_calls: [{ id: 'tool-1', name: 'ReadFile', input: { path: 'README.md' } }],
      },
      { role: 'tool', tool_call_id: 'tool-1', content: 'large tool payload must not reach scrollback' },
      { role: 'user', content: "[Skill 'autoresearch' activated]\n\nprivate instructions" },
      { role: 'user', content: '[sub-agent events]\n[Agent scan] -> ReadFile' },
      { role: 'user', content: 'Please compact this conversation: internal compaction prompt' },
      { role: 'user', content: 'visible follow up' },
      { role: 'assistant', content: 'visible final answer' },
    ],
    metadata: {},
    pendingResumeReplays: [],
    planMode: false,
    schemaVersion: undefined,
    sessionId,
    thinkingContent: ['persisted reasoning trace'],
    toolExecutions: [
      {
        tool_call_id: 'tool-1',
        name: 'ReadFile',
        permitted: true,
        duration_ms: 120,
        inputs: { path: 'README.md' },
      },
    ],
    totalInputTokens: 0,
    totalOutputTokens: 0,
    turnCount: 2,
    updatedAt: '2026-07-13T00:00:00.000Z',
    workspace: join(directory, 'agents', 'default'),
  })
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      model: 'gpt-4o',
      sessionDirectory,
      workspaceRoot: join(directory, 'agents'),
    }),
  })
  await server.start()
  const client = await ReplayTestClient.connect(socketPath)
  try {
    client.send({
      id: 1,
      jsonrpc: '2.0',
      method: 'initialize',
      params: { project_dir: directory, resume_session_id: sessionId },
    })

    expect((await client.next(frame => frame.id === 1)).result).toMatchObject({
      ok: true,
      session: { id: sessionId, key: sessionId, messages: 8 },
    })
    await client.next(eventFrame('init_done'))
    await client.next(eventFrame('status_update'))
    const resumed = await client.next(eventFrame('notification', 'resumed'))

    const history = client.observed.filter(frame => frame.method === 'event'
      && frame.params?.type === 'notification'
      && frame.params.payload?.category === 'history')
      .map(frame => ({
        body: frame.params?.payload?.body,
        payload: frame.params?.payload?.payload,
        type: frame.params?.payload?.type,
      }))
    // Current contract (since 51cb13e, "resumed sessions replay thinking and
    // tool-call rows"): thinking rides the replay_assistant payload, and each
    // tool call replays as one semantic replay_tool row reconciled with its
    // persisted execution (name, ok, duration, compact arguments context) —
    // never the tool's output. The filtering this test guards is unchanged and
    // asserted below.
    expect(history).toEqual([
      { type: 'replay_user', body: '✨ visible question', payload: {} },
      {
        type: 'replay_assistant',
        body: 'visible answer\nwith a second line',
        payload: { thinking: 'persisted reasoning trace' },
      },
      {
        type: 'replay_tool',
        body: '✓ ReadFile',
        payload: {
          name: 'ReadFile',
          ok: true,
          context: '{"path":"README.md"}',
          duration_ms: 120,
        },
      },
      { type: 'replay_user', body: '✨ visible follow up', payload: {} },
      { type: 'replay_assistant', body: 'visible final answer', payload: {} },
      { type: 'resumed', body: `── resumed session ${sessionId} (4 messages) ──`, payload: {} },
    ])
    expect(resumed.params?.payload?.body).toContain('(4 messages)')
    expect(JSON.stringify(history)).not.toContain('private instructions')
    expect(JSON.stringify(history)).not.toContain('sub-agent events')
    expect(JSON.stringify(history)).not.toContain('internal compaction prompt')
    expect(JSON.stringify(history)).not.toContain('large tool payload')
  } finally {
    client.close()
    await server.stop()
    await rm(directory, { recursive: true, force: true })
  }
})

interface Frame {
  readonly id?: number | string
  readonly method?: string
  readonly params?: {
    readonly payload?: Record<string, unknown>
    readonly type?: string
  }
  readonly result?: Record<string, unknown>
}

function eventFrame(eventType: string, notificationType?: string): (frame: Frame) => boolean {
  return frame => frame.method === 'event'
    && frame.params?.type === eventType
    && (notificationType === undefined || frame.params.payload?.type === notificationType)
}

class ReplayTestClient {
  private buffer = ''
  private readonly frames: Frame[] = []
  readonly observed: Frame[] = []
  private readonly waiters: Array<{ predicate: (frame: Frame) => boolean; resolve: (frame: Frame) => void }> = []

  private constructor(private readonly socket: Socket) {
    socket.setEncoding('utf8')
    socket.on('data', chunk => this.receive(typeof chunk === 'string' ? chunk : new TextDecoder().decode(chunk)))
  }

  static async connect(socketPath: string): Promise<ReplayTestClient> {
    const socket = connect({ path: socketPath })
    await new Promise<void>((resolveConnection, rejectConnection) => {
      socket.once('connect', resolveConnection)
      socket.once('error', rejectConnection)
    })
    return new ReplayTestClient(socket)
  }

  close(): void {
    this.socket.destroy()
  }

  next(predicate: (frame: Frame) => boolean): Promise<Frame> {
    const index = this.frames.findIndex(predicate)
    if (index >= 0) {
      const frame = this.frames.splice(index, 1)[0]
      if (frame) return Promise.resolve(frame)
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
      if (line.trim()) this.handle(JSON.parse(line) as Frame)
      newline = this.buffer.indexOf('\n')
    }
  }

  private handle(frame: Frame): void {
    this.observed.push(frame)
    const index = this.waiters.findIndex(waiter => waiter.predicate(frame))
    const waiter = index >= 0 ? this.waiters.splice(index, 1)[0] : undefined
    if (waiter) {
      waiter.resolve(frame)
      return
    }
    this.frames.push(frame)
  }
}
