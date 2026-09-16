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
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { registerFileTools } from '../src/tools/fileTools.js'
import { registerProjectSetupTool } from '../src/tools/projectSetup.js'
import { WorkspacePathResolver } from '../src/tools/pathSafety.js'

test('startup loads existing commands, init adds workflows, and shell preprocessing requires explicit trust', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-init-workflows-'))
  const skillDirectory = join(root, 'skill-home')
  const tools = new ToolRegistry()
  const paths = new WorkspacePathResolver(root)
  registerFileTools(tools, paths)
  registerProjectSetupTool(tools, paths, { skillsDirectory: skillDirectory })
  const requests: string[] = []
  const runner: TurnRunner = {
    async *run(_session, text) {
      requests.push(text)
      if (text.startsWith('Initialize this repository')) {
        const result = await tools.execute({ id: 'create', type: 'function', function: {
          name: 'create_project_setup', arguments: { artifacts: [
            { kind: 'command', name: 'repo-test', description: 'Run repo tests', instructions: 'Inspect tests for $ARGUMENTS.' },
          ] },
        } }, { metadata: {} })
        yield { type: 'text_part', payload: { text: result } }
      } else yield { type: 'text_part', payload: { text: 'Workflow invoked' } }
    },
  }
  const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: root, sessionDirectory: join(root, 'sessions') })
  const socketPath = join(root, 'daemon.sock')
  await mkdir(join(root, '.xerxes/commands'), { recursive: true })
  const existing = '---\ndescription: Existing workflow\n---\nReview $ARGUMENTS.'
  await writeFile(join(root, '.xerxes/commands/existing.md'), existing)
  const server = new DaemonServer({ socketPath, runtime, skillDirectory, machineSettingsPath: join(root, 'machines.json') })
  await server.start()
  const client = await DaemonParityClient.connect(socketPath)
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { session_key: 'setup', project_dir: root } })
    await client.next(frame => frame.id === 1)
    client.send({ jsonrpc: '2.0', id: 105, method: 'slash', params: { command: '/features' } })
    expect((await client.next(frame => frame.id === 105)).result).toMatchObject({ ok: true, output: expect.stringContaining('/machine') })
    expect(requests).toHaveLength(0)
    client.send({ jsonrpc: '2.0', id: 104, method: 'slash', params: { command: '/machine' } })
    expect((await client.next(frame => frame.id === 104)).result).toMatchObject({ ok: true, machines: [] })
    client.send({ jsonrpc: '2.0', id: 106, method: 'slash', params: { command: '/machine add test host /srv/project' } })
    expect((await client.next(frame => frame.id === 106)).result).toMatchObject({ ok: true })
    client.send({ jsonrpc: '2.0', id: 107, method: 'slash', params: { command: '/machine connect test' } })
    expect((await client.next(frame => frame.id === 107)).result).toMatchObject({ ok: true, machine: { target: 'host', workspacePath: '/srv/project' } })
    client.send({ jsonrpc: '2.0', id: 108, method: 'slash', params: { command: '/custom-agents' } })
    expect((await client.next(frame => frame.id === 108)).result).toMatchObject({ ok: true, agents: [] })
    client.send({ jsonrpc: '2.0', id: 101, method: 'complete', params: { text: '/existing' } })
    expect((await client.next(frame => frame.id === 101)).result?.completions).toContainEqual(expect.objectContaining({ value: '/existing ' }))
    expect(await readFile(join(root, '.xerxes/commands/existing.md'), 'utf8')).toBe(existing)
    expect(await Bun.file(join(skillDirectory, '.hub/trusted_hashes.json')).exists()).toBe(false)
    client.send({ jsonrpc: '2.0', id: 102, method: 'slash', params: { command: '/existing src' } })
    expect((await client.next(frame => frame.id === 102)).result).toMatchObject({ ok: true, queued: true })
    await client.next(frame => frame.method === 'event' && frame.params?.type === 'turn_end')
    expect(requests.pop()).toContain('Review src.')
    client.send({ jsonrpc: '2.0', id: 2, method: 'slash', params: { command: '/init focus on tests' } })
    expect((await client.next(frame => frame.id === 2)).result).toMatchObject({ ok: true, queued: true })
    await client.next(frame => frame.method === 'event' && frame.params?.type === 'notification' && String(frame.params.payload?.body).includes('Project initialization turn finished'))
    expect(requests[0]).toContain('create_project_setup')
    expect(requests[0]).toContain('focus on tests')
    expect(requests[0]).toContain('Do not overwrite existing')
    client.send({ jsonrpc: '2.0', id: 3, method: 'complete', params: { text: '/repo-test' } })
    expect((await client.next(frame => frame.id === 3)).result?.completions).toContainEqual(expect.objectContaining({ value: '/repo-test ' }))
    client.send({ jsonrpc: '2.0', id: 4, method: 'slash', params: { command: '/repo-test kernels' } })
    expect((await client.next(frame => frame.id === 4)).result).toMatchObject({ ok: true, queued: true })
    await client.next(frame => frame.method === 'event' && frame.params?.type === 'text_part' && frame.params.payload?.text === 'Workflow invoked')
    expect(requests.at(-1)).toContain('Inspect tests for kernels.')
    await client.next(frame => frame.method === 'event' && frame.params?.type === 'turn_end')
    await writeFile(join(root, '.xerxes/commands/existing.md'), '---\ndescription: Existing workflow\n---\nReview !`echo shell-expanded`.')
    client.send({ jsonrpc: '2.0', id: 5, method: 'slash', params: { command: '/existing' } })
    expect(JSON.stringify(await client.next(frame => frame.id === 5))).toContain('/skills trust')
    client.send({ jsonrpc: '2.0', id: 6, method: 'slash', params: { command: '/skills trust existing' } })
    expect((await client.next(frame => frame.id === 6)).result).toMatchObject({ ok: true, name: 'existing' })
    client.send({ jsonrpc: '2.0', id: 103, method: 'slash', params: { command: '/existing' } })
    expect((await client.next(frame => frame.id === 103)).result).toMatchObject({ ok: true, queued: true })
    await client.next(frame => frame.method === 'event' && frame.params?.type === 'turn_end')
    expect(requests.at(-1)).toContain('Review shell-expanded.')
    client.send({ jsonrpc: '2.0', id: 7, method: 'complete', params: { text: '/existing' } })
    expect((await client.next(frame => frame.id === 7)).result?.completions).toContainEqual(expect.objectContaining({ value: '/existing ' }))
    client.send({ jsonrpc: '2.0', id: 8, method: 'slash', params: { command: '/skills trust missing' } })
    expect((await client.next(frame => frame.id === 8)).result).toMatchObject({ ok: false })
  } finally { client.close(); await server.stop(); await rm(root, { recursive: true, force: true }) }
})

test('daemon completion preserves command and path semantics while native skills remain invocable by slash', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-daemon-completion-parity-'))
  const socketPath = join(directory, 'daemon.sock')
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    sessionDirectory: join(directory, 'sessions'),
  })
  // Seeded skill library: the shorthand assertions below must be hermetic —
  // default discovery would read the developer's real ~/.xerxes/skills AND
  // the bundled xerxes/skills. It lives OUTSIDE the project directory so
  // path completions never see it.
  const skillLibrary = await mkdtemp(join(tmpdir(), 'xerxes-daemon-skill-lib-'))
  const skillDirectory = join(skillLibrary, 'deepscan')
  await mkdir(skillDirectory, { recursive: true })
  await writeFile(
    join(skillDirectory, 'SKILL.md'),
    '---\nname: deepscan\ndescription: Deep scan the workspace for issues\n---\nScan.',
    'utf8',
  )
  const server = new DaemonServer({ socketPath, skillDirectories: [skillLibrary], runtime })
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
    await mkdir(join(directory, 'folder with spaces'))
    await writeFile(join(directory, 'folder with spaces', 'notes.txt'), 'notes')
    client.send({ jsonrpc: '2.0', id: 101, method: 'complete', params: { path_prefix: './' } })
    expect((await client.next(frame => frame.id === 101)).result?.completions).toEqual(expect.arrayContaining([
      { value: './alpha.txt', label: 'alpha.txt', meta: 'file' },
      { value: './folder with spaces/', label: 'folder with spaces/', meta: 'dir' },
    ]))
    client.send({ jsonrpc: '2.0', id: 102, method: 'complete', params: { path_prefix: './folder with spaces/' } })
    expect((await client.next(frame => frame.id === 102)).result?.completions).toEqual([
      { value: './folder with spaces/notes.txt', label: 'notes.txt', meta: 'file' },
    ])
    client.send({ jsonrpc: '2.0', id: 103, method: 'complete', params: { path_prefix: './missing-directory/' } })
    expect((await client.next(frame => frame.id === 103)).error).toBeDefined()
    client.send({ jsonrpc: '2.0', id: 104, method: 'complete', params: { path_prefix: 'my f' } })
    expect((await client.next(frame => frame.id === 104)).result?.completions).toEqual([
      { value: 'my file.md', label: 'my file.md', meta: 'file' },
    ])


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
    // A skill name completes as its own slash shorthand — `/deepscan` invokes
    // the skill directly, no `/skill` prefix needed. This is the FIRST skill
    // completion this daemon serves: the registry must self-refresh here, not
    // depend on a prior catalog or `/skill` call having warmed it.
    client.send({ jsonrpc: '2.0', id: 10, method: 'complete', params: { text: '/deepscan' } })
    expect((await client.next(frame => frame.id === 10)).result?.completions).toEqual([
      { value: '/deepscan ', label: 'deepscan', meta: 'Deep scan the workspace for issues' },
    ])

    client.send({ jsonrpc: '2.0', id: 15, method: 'complete', params: { text: '/deep' } })
    expect((await client.next(frame => frame.id === 15)).result?.completions).toEqual([
      { value: '/deepscan ', label: 'deepscan', meta: 'Deep scan the workspace for issues' },
    ])
  } finally {
    client.close()
    await server.stop()
    await rm(directory, { recursive: true, force: true })
    await rm(skillLibrary, { recursive: true, force: true })
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
    expect((await target.next(eventFrame('text_part'))).params?.payload).toMatchObject({ text: 'waiting for steer' })

    target.send({ jsonrpc: '2.0', id: 4, method: 'slash', params: { command: '/steer focus @steer.md' } })
    expect((await target.next(frame => frame.id === 4)).result).toEqual({ ok: true })
    expect((await target.next(eventFrame('steer_input'))).params?.payload).toMatchObject({
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
  // The provider is pinned rather than inherited. The stubbed fetch below returns
  // an OpenAI-shaped completion, so the compaction client has to be the OpenAI
  // transport for this test to mean anything — and with no explicit provider the
  // daemon falls back to the built-in `claude-code` profile, which has no client
  // adapter at all. That made the test pass only on a machine whose ambient
  // ~/.xerxes profile happened to be OpenAI-compatible, and fail on a clean one.
  const runtime = new InMemoryDaemonRuntime(new UnexpectedTurnRunner(), {
    currentProjectDirectory: directory,
    model: 'compact-model',
    runtimeSettings: {
      base_url: 'https://api.openai.com/v1',
      model: 'compact-model',
      provider: 'openai',
    },
    sessionDirectory,
  })
  const server = new DaemonServer({ socketPath, runtime })
  const nativeFetch = globalThis.fetch
  const providerRequests: unknown[] = []
  globalThis.fetch = (async (_input: unknown, init?: RequestInit) => {
    providerRequests.push(typeof init?.body === 'string' ? JSON.parse(init.body) : undefined)
    return new Response('data: ' + JSON.stringify({ choices: [{ delta: { content: 'durable parity summary' }, finish_reason: 'stop' }] }) + '\n\ndata: [DONE]\n\n', { headers: { 'content-type': 'text/event-stream' } })
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
    // Compaction announces itself before the provider call: it is a single
    // long await, and without this the screen showed nothing at all until the
    // summary came back.
    expect((await client.next(eventFrame('notification'))).params?.payload).toMatchObject({
      category: 'slash',
      body: expect.stringContaining('Compacting'),
    })
    expect((await client.next(eventFrame('notification'))).params?.payload).toMatchObject({
      category: 'history',
      body: expect.stringContaining('Compacted'),
      payload: { automatic: false },
    })
    expect((await client.next(eventFrame('status_update'))).params?.payload).toMatchObject({
      context_tokens: expect.any(Number),
      max_context: 0,
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
  readonly error?: { readonly code: number; readonly message: string }
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

test('idle-only runtime restart rejects active work and closes admission before restarting', async () => {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-idle-update-'))
  let activeSubagents = 1
  let restarts = 0
  const runtime = new InMemoryDaemonRuntime({ async *run() {} }, {
    currentProjectDirectory: root,
    sessionDirectory: join(root, 'sessions'),
    statusInventory: () => ({ activeSubagents }),
  })
  const server = new DaemonServer({ socketPath: join(root, 'daemon.sock'), runtime, onRestart: () => { restarts += 1 } })
  await server.start()
  const client = await DaemonParityClient.connect(join(root, 'daemon.sock'))
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'runtime.restart_if_idle', params: {} })
    expect((await client.next(frame => frame.id === 1)).result).toEqual({ ok: false, busy: true })
    expect(restarts).toBe(0)
    activeSubagents = 0
    const session = await runtime.openSession("busy", undefined, { cwd: root })
    session.status = "working"
    client.send({ jsonrpc: "2.0", id: 4, method: "runtime.restart_if_idle", params: {} })
    expect((await client.next(frame => frame.id === 4)).result).toEqual({ ok: false, busy: true })
    expect(restarts).toBe(0)
    session.status = "idle"
    client.send({ jsonrpc: '2.0', id: 2, method: 'runtime.restart_if_idle', params: {} })
    expect((await client.next(frame => frame.id === 2)).result).toEqual({ ok: true })
    client.send({ jsonrpc: '2.0', id: 3, method: 'initialize', params: { session_key: 'late' } })
    expect((await client.next(frame => frame.id === 3)).result).toMatchObject({ ok: false, error: expect.stringContaining('restarting') })
    await new Promise(resolve => setTimeout(resolve, 40))
    expect(restarts).toBe(1)
  } finally { client.close(); await server.stop(); await rm(root, { recursive: true, force: true }) }
})
