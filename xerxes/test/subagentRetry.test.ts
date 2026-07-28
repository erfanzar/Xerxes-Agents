// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import type { AgentDefinition } from '../src/agents/definitions.js'
import { SubAgentManager } from '../src/agents/subagentManager.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { createNativeSubagentHost, SUBAGENT_RETRY_CONTINUATION_PROMPT } from '../src/daemon/subagentHost.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import type { SpawnedAgentSnapshot } from '../src/operators/subagents.js'
import { DaemonTranscriptStore } from '../src/session/daemonTranscript.js'

function agentDefinition(name: string): AgentDefinition {
  return {
    allowedTools: null,
    description: `${name} test agent`,
    excludeTools: [],
    isolation: '',
    maxDepth: 3,
    model: '',
    name,
    source: 'test',
    systemPrompt: `You are the ${name} test agent.`,
    tools: [],
  }
}

async function waitFor(predicate: () => boolean, timeoutMs = 2_000): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error(`condition was not met within ${timeoutMs}ms`)
    await Bun.sleep(2)
  }
}

/** Fails `failNext` streams with a non-retryable provider error, then answers. */
class FlakyChildClient implements LlmClient {
  requests: CompletionRequest[] = []

  constructor(private failNext = 0) {}

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    if (this.failNext > 0) {
      this.failNext -= 1
      const error = new Error('connection reset by peer') as Error & { status?: number }
      error.status = 401
      throw error
    }
    const lastUser = [...request.messages].reverse().find(message => message.role === 'user')
    yield { content: `answer:${typeof lastUser?.content === 'string' ? lastUser.content : ''}` }
  }
}

function hostWith(transcripts: DaemonTranscriptStore, client: LlmClient) {
  const registry = new ToolRegistry()
  return createNativeSubagentHost({
    agentDefinitions: new Map([['coder', agentDefinition('coder')]]),
    cwd: process.cwd(),
    eventBus: new DaemonSubagentEventBus(),
    llm: client,
    model: 'test-model',
    permissionMode: 'accept-all',
    toolExecutor: registry,
    tools: registry.definitions(),
    transcriptStore: transcripts,
  })
}

function nonSystemMessages(request: CompletionRequest): Array<[string, unknown]> {
  return request.messages
    .filter(message => message.role !== 'system')
    .map(message => [message.role, message.content])
}

test('manager retry restarts a failed task under the same identity with the supplied input', async () => {
  const runs: string[] = []
  const manager = new SubAgentManager({
    runner: async request => {
      runs.push(request.prompt)
      if (request.prompt.includes('boom')) throw new Error('provider exploded')
      return `done:${request.prompt}`
    },
  })

  try {
    const task = await manager.spawn({ name: 'worker', prompt: 'boom task' })
    await manager.wait(task.id, 1_000)
    expect(task.status).toBe('failed')

    const retried = await manager.retry('worker', 'fixed input')
    expect(retried).toBeDefined()
    expect(retried?.id).toBe(task.id)
    expect(retried?.name).toBe('worker')
    expect(retried?.status).not.toBe('failed')

    await manager.wait(task.id, 1_000)
    expect(task.status).toBe('completed')
    expect(runs).toEqual(['boom task', 'fixed input'])
  } finally {
    await manager.shutdown()
  }
})

test('manager retry is idempotent against double invocation and running tasks', async () => {
  const runs: string[] = []
  const gate = Promise.withResolvers<void>()
  const manager = new SubAgentManager({
    runner: async request => {
      runs.push(request.prompt)
      if (runs.length === 1) throw new Error('first attempt failed')
      await gate.promise
      return 'recovered'
    },
  })

  try {
    const task = await manager.spawn({ name: 'worker', prompt: 'original' })
    await manager.wait(task.id, 1_000)
    expect(task.status).toBe('failed')

    const [first, second] = await Promise.all([manager.retry(task.id), manager.retry(task.id)])
    expect(first?.id).toBe(task.id)
    expect(second?.id).toBe(task.id)
    await waitFor(() => runs.length === 2)

    // A retry of an already-running attempt returns the live task instead of
    // starting a duplicate run.
    const third = await manager.retry(task.id)
    expect(third?.id).toBe(task.id)
    await Bun.sleep(20)
    expect(runs).toHaveLength(2)

    gate.resolve()
    await manager.wait(task.id, 1_000)
    expect(task.status).toBe('completed')
    expect(runs).toEqual(['original', 'original'])
  } finally {
    gate.resolve()
    await manager.shutdown()
  }
})

test('manager retry resurrects an evicted terminal task from its archive with the same identity', async () => {
  const runs: string[] = []
  const manager = new SubAgentManager({
    maxRetainedTerminalTasks: 1,
    runner: async request => {
      runs.push(request.prompt)
      if (request.prompt.startsWith('task')) throw new Error('attempt failed')
      return `done:${request.prompt}`
    },
  })

  try {
    const alpha = await manager.spawn({ name: 'worker-a', prompt: 'task a' })
    await manager.wait(alpha.id, 1_000)
    const beta = await manager.spawn({ name: 'worker-b', prompt: 'task b' })
    await manager.wait(beta.id, 1_000)

    // Retention bound 1 evicts the oldest terminal task (alpha) from live
    // state once the failing attempt's monitor finishes its bookkeeping.
    await waitFor(() => manager.listTasks().length === 1)
    expect(manager.listTasks().map(task => task.name)).toEqual(['worker-b'])
    expect(manager.findTask('worker-a')).toMatchObject({ archived: true, id: alpha.id, status: 'failed' })

    const retried = await manager.retry('worker-a', 'resume a')
    expect(retried).toBeDefined()
    expect(retried?.id).toBe(alpha.id)
    expect(retried?.name).toBe('worker-a')
    expect(manager.findTask('worker-a')).toMatchObject({ archived: false, id: alpha.id })

    await manager.wait(alpha.id, 1_000)
    expect(retried?.status).toBe('completed')
    expect(runs).toEqual(['task a', 'task b', 'resume a'])
  } finally {
    await manager.shutdown()
  }
})

test('host retry of a connection-killed task continues its persisted conversation under the same identity', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-retry-history-'))
  const transcripts = new DaemonTranscriptStore({
    currentProjectDirectory: process.cwd(),
    directory: join(directory, 'sessions'),
  })
  const client = new FlakyChildClient(1)
  const host = hostWith(transcripts, client)

  try {
    const spawned = await host.managerPort.spawn({
      message: 'inspect area one',
      nickname: 'flaky-worker',
      promptProfile: 'coder',
      sourceAgentId: 'parent-session',
      title: 'Flaky worker',
    })
    // The first attempt dies to the provider failure; the turn surfaces the
    // error as its final output and lands in a terminal state.
    await host.managerPort.wait([spawned.id], 2_000)
    expect(client.requests).toHaveLength(1)
    const historySessionId = spawned.historySessionId
    if (!historySessionId) throw new Error('expected a persisted child history id')

    const retried = await host.retry('flaky-worker')
    expect(retried.id).toBe(spawned.id)
    expect(retried.name).toBe('flaky-worker')
    expect(retried.historySessionId).toBe(historySessionId)
    expect(retried.status === 'idle' || retried.status === 'running').toBe(true)

    const settled = await host.managerPort.wait([spawned.id], 2_000)
    expect(settled.completed[0]).toMatchObject({ id: spawned.id, status: 'completed' })
    expect(client.requests).toHaveLength(2)

    // The second attempt continued the persisted conversation: the original
    // prompt is still there and the retry nudge replaces a duplicate resubmit.
    const continued = nonSystemMessages(client.requests[1]!)
    expect(continued[0]).toEqual(['user', 'inspect area one'])
    expect(continued.at(-1)).toEqual(['user', SUBAGENT_RETRY_CONTINUATION_PROMPT])

    // Same stable identity: one persisted conversation, still completed.
    const transcript = await transcripts.load(historySessionId, { currentProjectDirectory: process.cwd() })
    expect(transcript?.metadata).toMatchObject({
      session_kind: 'subagent',
      status: 'completed',
      subagent_id: spawned.id,
    })
    expect(transcript?.messages.some(message => (
      message.role === 'assistant'
      && typeof message.content === 'string'
      && message.content.includes('answer:')
    ))).toBe(true)
  } finally {
    await host.manager.shutdown()
    await rm(directory, { force: true, recursive: true })
  }
})

test('host retry of a cancelled task resumes the retained conversation instead of starting fresh', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-retry-cancelled-'))
  const transcripts = new DaemonTranscriptStore({
    currentProjectDirectory: process.cwd(),
    directory: join(directory, 'sessions'),
  })
  const started = Promise.withResolvers<void>()
  let calls = 0
  const requests: CompletionRequest[] = []
  const client: LlmClient = {
    async *stream(request: CompletionRequest, signal?: AbortSignal): AsyncGenerator<LlmDelta> {
      requests.push(request)
      calls += 1
      if (calls > 1) {
        yield { content: 'resumed and finished' }
        return
      }
      started.resolve()
      await new Promise<void>((_resolve, reject) => {
        if (!signal) {
          reject(new Error('expected a delegated cancellation signal'))
          return
        }
        const cancel = (): void => reject(signal.reason ?? new Error('cancelled'))
        if (signal.aborted) cancel()
        else signal.addEventListener('abort', cancel, { once: true })
      })
      yield { content: 'unreachable' }
    },
  }
  const host = hostWith(transcripts, client)

  try {
    const spawned = await host.managerPort.spawn({
      message: 'inspect until cancelled',
      nickname: 'cancelled-worker',
      promptProfile: 'coder',
      sourceAgentId: 'parent-session',
    })
    await started.promise
    host.managerPort.close(spawned.id)
    const historySessionId = spawned.historySessionId
    if (!historySessionId) throw new Error('expected a persisted child history id')
    // Wait for the cancellation to settle and persist before retrying.
    await waitFor(() => requests.length === 1 && host.manager.listTasks()[0]?.status === 'cancelled')
    // Poll for the status this asserts, not merely for the file to appear. The
    // manager marks the task cancelled in memory before the transcript carrying
    // that status is written, so waiting only for existence loaded the earlier
    // `running` snapshot and asserted against it — passing or failing on how
    // quickly the host happened to flush. Still non-vacuous: a status that never
    // settles exhausts the budget and the assertion below fails on the last
    // snapshot read.
    let transcript = await transcripts.load(historySessionId, { currentProjectDirectory: process.cwd() })
    for (let attempt = 0; attempt < 400 && transcript?.metadata?.status !== 'cancelled'; attempt += 1) {
      await Bun.sleep(10)
      transcript = await transcripts.load(historySessionId, { currentProjectDirectory: process.cwd() })
    }
    expect(transcript?.metadata).toMatchObject({ status: 'cancelled' })

    const retried = await host.retry(spawned.id)
    expect(retried.id).toBe(spawned.id)
    expect(retried.historySessionId).toBe(historySessionId)

    const settled = await host.managerPort.wait([spawned.id], 2_000)
    expect(settled.completed[0]).toMatchObject({ id: spawned.id, status: 'completed' })
    expect(requests).toHaveLength(2)
    const continued = nonSystemMessages(requests[1]!)
    expect(continued[0]).toEqual(['user', 'inspect until cancelled'])
    expect(continued.at(-1)).toEqual(['user', SUBAGENT_RETRY_CONTINUATION_PROMPT])

    transcript = await transcripts.load(historySessionId, { currentProjectDirectory: process.cwd() })
    expect(transcript?.metadata).toMatchObject({ status: 'completed' })
  } finally {
    await host.manager.shutdown()
    await rm(directory, { force: true, recursive: true })
  }
})

test('a dead agent stays retryable after a daemon restart with the same identity and conversation', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-retry-restart-'))
  const transcripts = new DaemonTranscriptStore({
    currentProjectDirectory: process.cwd(),
    directory: join(directory, 'sessions'),
  })
  const firstClient = new FlakyChildClient(1)
  const firstHost = hostWith(transcripts, firstClient)

  try {
    const spawned = await firstHost.managerPort.spawn({
      message: 'review the boundary',
      nickname: 'restart-worker',
      promptProfile: 'coder',
      sourceAgentId: 'parent-session',
      title: 'Restart worker',
    })
    const settled = await firstHost.managerPort.wait([spawned.id], 2_000)
    const terminal = settled.completed[0]
    if (!terminal) throw new Error('expected the first attempt to reach a terminal state')
    await firstHost.manager.shutdown()

    // The terminal record survives the "daemon restart" on disk.
    const historySessionId = spawned.historySessionId
    if (!historySessionId) throw new Error('expected a persisted child history id')
    const persisted = await transcripts.load(historySessionId, { currentProjectDirectory: process.cwd() })
    expect(persisted).toBeDefined()

    // A fresh host (new process) recovers the tombstone and retries it.
    const secondClient = new FlakyChildClient(0)
    const secondHost = hostWith(transcripts, secondClient)
    try {
      const restorable = secondHost.managerPort as unknown as {
        restoreSnapshots(snapshots: readonly SpawnedAgentSnapshot[]): number
      }
      expect(restorable.restoreSnapshots([terminal])).toBe(1)

      const retried = await secondHost.retry('restart-worker')
      expect(retried.id).toBe(spawned.id)
      expect(retried.name).toBe('restart-worker')
      expect(retried.historySessionId).toBe(historySessionId)

      const finished = await secondHost.managerPort.wait([spawned.id], 2_000)
      expect(finished.completed[0]).toMatchObject({ id: spawned.id, status: 'completed' })
      expect(secondClient.requests).toHaveLength(1)
      const continued = nonSystemMessages(secondClient.requests[0]!)
      expect(continued[0]).toEqual(['user', 'review the boundary'])
      expect(continued.at(-1)).toEqual(['user', SUBAGENT_RETRY_CONTINUATION_PROMPT])
    } finally {
      await secondHost.manager.shutdown()
    }
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
})

interface Frame {
  readonly id?: number
  readonly method?: string
  readonly result?: Record<string, unknown>
}

class SocketTestClient {
  private buffer = ''
  private readonly frames: Frame[] = []
  private readonly waiters: Array<{
    predicate: (frame: Frame) => boolean
    resolve: (frame: Frame) => void
  }> = []

  private constructor(private readonly socket: Socket) {
    socket.setEncoding('utf8')
    socket.on('data', chunk => this.receive(typeof chunk === 'string' ? chunk : new TextDecoder().decode(chunk)))
  }

  static async connect(socketPath: string): Promise<SocketTestClient> {
    const socket = connect({ path: socketPath })
    await new Promise<void>((resolve, reject) => {
      socket.once('connect', resolve)
      socket.once('error', reject)
    })
    return new SocketTestClient(socket)
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
      if (line.trim()) this.handle(JSON.parse(line) as Frame)
      newline = this.buffer.indexOf('\n')
    }
  }

  private handle(frame: Frame): void {
    const index = this.waiters.findIndex(waiter => waiter.predicate(frame))
    if (index >= 0) {
      const [waiter] = this.waiters.splice(index, 1)
      waiter?.resolve(frame)
      return
    }
    this.frames.push(frame)
  }
}

test('daemon socket exposes subagent.retry and validates the task argument', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-retry-rpc-'))
  const socketPath = join(directory, 'daemon.sock')
  const received: Array<{
    readonly message?: string
    readonly sessionKey?: string
    readonly task: string
  }> = []
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: 'protocol-model',
    sessionDirectory: join(directory, 'sessions'),
    subagentRetry: async request => {
      received.push(request)
      return {
        ok: true,
        agent: {
          id: 'subagent_dead1',
          name: request.task,
          status: 'idle',
          history_session_id: 'beefbeefbeef',
        },
      }
    },
  })
  const server = new DaemonServer({ socketPath, runtime })
  await server.start()
  const client = await SocketTestClient.connect(socketPath)

  try {
    client.send({
      jsonrpc: '2.0',
      id: 1,
      method: 'subagent.retry',
      params: { task: 'dead-worker', message: 'try again' },
    })
    const retried = await client.next(frame => frame.id === 1)
    expect(retried.result).toMatchObject({
      ok: true,
      agent: { id: 'subagent_dead1', name: 'dead-worker', status: 'idle' },
    })
    expect(received).toHaveLength(1)
    expect(received[0]).toMatchObject({ task: 'dead-worker', message: 'try again' })
    expect(typeof received[0]?.sessionKey).toBe('string')

    client.send({ jsonrpc: '2.0', id: 2, method: 'subagent.retry', params: {} })
    const missing = await client.next(frame => frame.id === 2)
    expect(missing.result).toMatchObject({
      ok: false,
      error: 'subagent.retry requires a task id or stable name',
    })
  } finally {
    client.close()
    await server.stop()
    await rm(directory, { force: true, recursive: true })
  }
})

test('daemon socket reports honestly when the runtime has no subagent retry port', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-retry-rpc-off-'))
  const socketPath = join(directory, 'daemon.sock')
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, 'sessions'),
    }),
  })
  await server.start()
  const client = await SocketTestClient.connect(socketPath)

  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'subagent.retry', params: { task: 'dead-worker' } })
    const response = await client.next(frame => frame.id === 1)
    expect(response.result).toEqual({
      ok: false,
      error: 'This daemon runtime does not expose subagent retry.',
    })
  } finally {
    client.close()
    await server.stop()
    await rm(directory, { force: true, recursive: true })
  }
})
