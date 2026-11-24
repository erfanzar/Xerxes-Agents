// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { generateKeyPairSync, randomUUID } from 'node:crypto'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { AcpStdioClient, type AcpSpawnedProcess } from '../src/acp/client.js'
import { JobStore } from '../src/cron/jobs.js'
import type { AcpExternalAgentInfo } from '../src/acp/externalAgents.js'
import { DaemonAcpBridgeHost } from '../src/daemon/acpBridgeHost.js'
import { AGY_TASK_ID_PREFIX, AgyBrainBridge } from '../src/daemon/agyBrainBridge.js'
import { InMemoryDaemonRuntime, inflightToolContext } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import type { LlmClient, LlmDelta } from '../src/llms/client.js'
import type { JsonRpcPayload } from '../src/protocol/jsonRpc.js'
import { paramsHash, signHubAuth, type HubAuthClaims } from '../src/security/hubAuthEnvelope.js'
import { enrollHub } from '../src/security/hubIdentityVerifier.js'

const encoder = new TextEncoder()

interface InMemoryAcpHarness {
  readonly pushAgentFrame: (frame: unknown) => void
  readonly receivedFrames: Array<Record<string, unknown>>
  readonly terminate: (code?: number | null) => void
  readonly transport: AcpSpawnedProcess
}

function createInMemoryAcpTransport(
  onClientFrame?: (frame: Record<string, unknown>, harness: InMemoryAcpHarness) => void | Promise<void>,
): InMemoryAcpHarness {
  const receivedFrames: Array<Record<string, unknown>> = []
  let stdoutController: ReadableStreamDefaultController<Uint8Array> | undefined
  let stderrController: ReadableStreamDefaultController<Uint8Array> | undefined
  let resolveExit: ((code: number | null) => void) | undefined
  let exited = false

  const stdout = new ReadableStream<Uint8Array>({
    start(controller) {
      stdoutController = controller
    },
  })
  const stderr = new ReadableStream<Uint8Array>({
    start(controller) {
      stderrController = controller
    },
  })
  const exitedPromise = new Promise<number | null>(resolve => {
    resolveExit = resolve
  })

  const terminate = (code: number | null = 0) => {
    if (exited) return
    exited = true
    try { stdoutController?.close() } catch { /* already closed */ }
    try { stderrController?.close() } catch { /* already closed */ }
    resolveExit?.(code)
  }

  const pushAgentFrame = (frame: unknown) => {
    if (exited) return
    stdoutController?.enqueue(encoder.encode(`${JSON.stringify(frame)}\n`))
  }

  const harness: InMemoryAcpHarness = {
    receivedFrames,
    pushAgentFrame,
    terminate,
    transport: {
      stdout,
      stderr,
      exited: exitedPromise,
      writeLine: (line: string) => {
        const parsed = JSON.parse(line.trim()) as Record<string, unknown>
        receivedFrames.push(parsed)
        if (onClientFrame) void onClientFrame(parsed, harness)
      },
      kill: () => terminate(0),
    },
  }

  return harness
}

interface RpcFrame {
  readonly error?: { readonly code: number; readonly message: string }
  readonly id?: number
  readonly method?: string
  readonly params?: { readonly payload?: JsonRpcPayload; readonly type?: string } & JsonRpcPayload
  readonly result?: JsonRpcPayload
}

class SocketRpcClient {
  readonly frames: RpcFrame[] = []
  private readonly socket: Socket
  private buffer = ''
  private nextId = 1
  private readonly pending = new Map<number, { resolve: (r: JsonRpcPayload) => void; reject: (e: Error) => void }>()

  private constructor(socket: Socket) {
    this.socket = socket
    socket.setEncoding('utf8')
    socket.on('data', (chunk: string) => {
      this.buffer += chunk
      for (;;) {
        const idx = this.buffer.indexOf('\n')
        if (idx < 0) break
        const line = this.buffer.slice(0, idx).trim()
        this.buffer = this.buffer.slice(idx + 1)
        if (!line) continue
        const frame = JSON.parse(line) as RpcFrame
        this.frames.push(frame)
        if (typeof frame.id === 'number') {
          const entry = this.pending.get(frame.id)
          if (entry) {
            this.pending.delete(frame.id)
            if (frame.error) entry.reject(new Error(frame.error.message))
            else entry.resolve(frame.result ?? {})
          }
        }
      }
    })
  }

  static async open(socketPath: string): Promise<SocketRpcClient> {
    const socket = await new Promise<Socket>((resolve, reject) => {
      const s = connect(socketPath)
      s.once('connect', () => resolve(s))
      s.once('error', reject)
    })
    return new SocketRpcClient(socket)
  }

  call(method: string, params: JsonRpcPayload = {}): Promise<JsonRpcPayload> {
    const id = this.nextId++
    return new Promise<JsonRpcPayload>((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      this.socket.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`)
    })
  }

  async waitForEvent(
    predicate: (frame: RpcFrame) => boolean,
    timeoutMs = 5000,
  ): Promise<RpcFrame> {
    const deadline = Date.now() + timeoutMs
    for (;;) {
      const found = this.frames.find(predicate)
      if (found) return found
      if (Date.now() >= deadline) throw new Error('Timed out waiting for daemon event')
      await Bun.sleep(15)
    }
  }

  close(): void {
    this.socket.destroy()
  }
}

function makeUnusedNativeLlm(): LlmClient {
  return {
    async *stream(): AsyncGenerator<LlmDelta> {
      throw new Error('Native LLM must not be invoked when connected to Antigravity ACP')
    },
  }
}

const tempRoots: string[] = []

afterEach(async () => {
  while (tempRoots.length > 0) {
    const dir = tempRoots.pop()!
    await rm(dir, { recursive: true, force: true })
  }
})

async function makeTempDir(prefix: string): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), prefix))
  tempRoots.push(dir)
  return dir
}

describe('Antigravity background tasks & inflight tool context', () => {
  test('inflightToolContext extracts AGY PascalCase tool parameters cleanly', () => {
    expect(
      inflightToolContext('run_command', {
        CommandLine: 'eray tpu job status esurge-qwen38',
        toolSummary: 'Check TPU job',
      }),
    ).toBe('eray tpu job status esurge-qwen38')
    expect(
      inflightToolContext('schedule', {
        DurationSeconds: 720,
        Prompt: 'Check esurge compilation progress',
      }),
    ).toBe('in 720s · Check esurge compilation progress')
    expect(
      inflightToolContext('manage_task', { Action: 'kill', TaskId: 'conv/task-4' }),
    ).toBe('kill · conv/task-4')
  })

  test('AgyBrainBridge lists background commands & schedules from transcript.jsonl and inspects live task-*.log safely', async () => {
    const fakeHome = await makeTempDir('agy-brain-bg-')
    const convId = 'conv-bg-1234'
    const sessionDir = join(fakeHome, '.gemini', 'antigravity', 'brain', convId)
    const logsDir = join(sessionDir, '.system_generated', 'logs')
    const tasksDir = join(sessionDir, '.system_generated', 'tasks')
    await mkdir(logsDir, { recursive: true })
    await mkdir(tasksDir, { recursive: true })

    const nowIso = new Date().toISOString()
    const taskLogPath = join(tasksDir, 'task-1.log')
    await writeFile(taskLogPath, 'Compiling replica 1/4...\nCompiled 2/9 buckets\n')

    const lines = [
      JSON.stringify({
        step_index: 1,
        type: 'PLANNER_RESPONSE',
        status: 'DONE',
        created_at: nowIso,
        tool_calls: [
          {
            name: 'run_command',
            args: {
              CommandLine: 'eray tpu job logs esurge-qwen38',
              toolSummary: 'Tail eSurge TPU logs',
            },
          },
          {
            name: 'schedule',
            args: {
              DurationSeconds: 720,
              Prompt: 'Check esurge-qwen38 compilation progress and report metrics',
            },
          },
          {
            name: 'run_command',
            args: {
              CommandLine: 'echo done',
              toolSummary: 'Quick echo',
            },
          },
        ],
      }),
      JSON.stringify({
        step_index: 2,
        type: 'GENERIC',
        status: 'DONE',
        created_at: nowIso,
        content: `Tool is running as a background task with task id: ${convId}/task-1\nTask logs are available at: file://${taskLogPath}\nLog output:\nStarting...`,
      }),
      JSON.stringify({
        step_index: 3,
        type: 'GENERIC',
        status: 'DONE',
        created_at: nowIso,
        content: `Tool is running as a background task with task id: ${convId}/task-2`,
      }),
      JSON.stringify({
        step_index: 4,
        type: 'GENERIC',
        status: 'DONE',
        created_at: nowIso,
        content: `Tool is running as a background task with task id: ${convId}/task-3`,
      }),
      JSON.stringify({
        step_index: 5,
        type: 'SYSTEM_MESSAGE',
        status: 'DONE',
        created_at: nowIso,
        content: `[Message] timestamp=${nowIso} sender=${convId}/task-3 priority=MESSAGE_PRIORITY_HIGH content=Task id "${convId}/task-3" finished with result:\nCommand exited with code 0.`,
      }),
    ]
    await writeFile(join(logsDir, 'transcript.jsonl'), `${lines.join('\n')}\n`)

    const bridge = new AgyBrainBridge({ homeDir: fakeHome })
    const loc = bridge.resolveLocation({ externalSessionId: convId })
    expect(loc).toBeDefined()
    const tasks = bridge.listBackgroundTasksSync(loc!)
    expect(tasks).toHaveLength(3)

    const runningCmd = tasks.find(t => t.rawTaskId === `${convId}/task-1`)
    expect(runningCmd).toMatchObject({
      id: `${AGY_TASK_ID_PREFIX}${convId}/task-1`,
      kind: 'shell',
      label: 'Tail eSurge TPU logs',
      state: 'running',
      logFileName: 'task-1.log',
    })

    const activeSched = tasks.find(t => t.rawTaskId === `${convId}/task-2`)
    expect(activeSched).toMatchObject({
      id: `${AGY_TASK_ID_PREFIX}${convId}/task-2`,
      kind: 'schedule',
      state: 'scheduled',
      durationSeconds: 720,
    })
    expect(activeSched!.nextRunAt).toBeTruthy()

    const finishedCmd = tasks.find(t => t.rawTaskId === `${convId}/task-3`)
    expect(finishedCmd).toMatchObject({
      kind: 'shell',
      state: 'completed',
      exitCode: 0,
    })

    // Live task log inspection reads .system_generated/tasks/task-1.log
    const inspected = bridge.inspectBackgroundTask(loc!, `${AGY_TASK_ID_PREFIX}${convId}/task-1`, 4096)
    expect(inspected).not.toBeNull()
    expect(inspected!.running).toBe(true)
    expect(inspected!.canInterrupt).toBe(true)
    expect(inspected!.output).toContain('Compiled 2/9 buckets')
  })

  test('DaemonServer shows AGY background tasks and logs, sends agy only what the user typed, and refuses Xerxes-composed prompts E2E', async () => {
    const root = await makeTempDir('xerxes-agy-e2e-')
    const fakeHome = await makeTempDir('xerxes-agy-home-')
    const xerxesSkillsDir = join(root, 'bundled-xerxes-skills')
    await mkdir(join(xerxesSkillsDir, 'immortals-coordinator'), { recursive: true })
    await writeFile(
      join(xerxesSkillsDir, 'immortals-coordinator', 'SKILL.md'),
      '---\nname: immortals-coordinator\ndescription: Native Xerxes skill\n---\nNative only.',
    )

    // Project XERXES.md + .xerxes/commands/deploy.md + .xerxes/skills/tpu-bench/SKILL.md
    await writeFile(join(root, 'XERXES.md'), '# Workspace Policy\nAlways benchmark at 32k input / 1k output.')
    await mkdir(join(root, '.xerxes', 'commands'), { recursive: true })
    await writeFile(
      join(root, '.xerxes', 'commands', 'deploy.md'),
      '---\ndescription: Deploy eSurge model\n---\nDeploy `$ARGUMENTS` across TPU hosts.',
    )
    await mkdir(join(root, '.xerxes', 'skills', 'tpu-bench'), { recursive: true })
    await writeFile(
      join(root, '.xerxes', 'skills', 'tpu-bench', 'SKILL.md'),
      '---\nname: tpu-bench\ndescription: Run TPU throughput benchmark\n---\nRun decode TPS test.',
    )

    // Create brain task log on disk so terminal.inspect returns real live output
    const extSessionId = 'agy-bg-sess-1'
    const tasksDir = join(fakeHome, '.gemini', 'antigravity', 'brain', extSessionId, '.system_generated', 'tasks')
    await mkdir(tasksDir, { recursive: true })
    const logPath = join(tasksDir, 'task-42.log')
    await writeFile(logPath, '[replica-0] compiled 4/9 buckets\n[replica-1] compiled 4/9 buckets\n')

    const receivedPromptFrames: Array<Record<string, unknown>> = []
    let turnNumber = 0

    const agentInfo: AcpExternalAgentInfo = {
      id: 'antigravity',
      name: 'Antigravity',
      provider: 'antigravity',
      transport: 'acp',
      available: true,
      command: 'agy',
      args: ['acp'],
      description: 'Mock Antigravity ACP',
    }

    const bridgeHost = new DaemonAcpBridgeHost({
      homeDir: fakeHome,
      discoverAgents: () => [agentInfo],
      createAcpClient: opts => {
        const harness = createInMemoryAcpTransport(frame => {
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                protocolVersion: 1,
                agentInfo: { name: 'Antigravity ACP', version: '1.5.15' },
                agentCapabilities: {
                  promptCapabilities: { image: true, embeddedContext: true },
                },
              },
            })
            return
          }
          if (frame.method === 'session/new') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: extSessionId,
                update: {
                  sessionUpdate: 'available_commands_update',
                  availableCommands: [
                    { name: 'compact', description: 'Compact conversation' },
                  ],
                },
              },
            })
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: { sessionId: extSessionId, configOptions: [] },
            })
            return
          }
          if (frame.method === 'session/prompt') {
            turnNumber += 1
            const params = frame.params as Record<string, unknown>
            receivedPromptFrames.push(params)

            if (turnNumber === 1) {
              // Emit a backgrounded run_command AND a schedule timer
              harness.pushAgentFrame({
                jsonrpc: '2.0',
                method: 'session/update',
                params: {
                  sessionId: extSessionId,
                  update: {
                    sessionUpdate: 'tool_call',
                    toolCallId: 'tc-bg-cmd',
                    title: 'Run command',
                    status: 'in_progress',
                    rawInput: {
                      CommandLine: 'eray tpu job logs esurge-qwen38 --follow',
                      toolSummary: 'Follow eSurge compilation logs',
                    },
                  },
                },
              })
              harness.pushAgentFrame({
                jsonrpc: '2.0',
                method: 'session/update',
                params: {
                  sessionId: extSessionId,
                  update: {
                    sessionUpdate: 'tool_call_update',
                    toolCallId: 'tc-bg-cmd',
                    status: 'completed',
                    rawOutput: `Tool is running as a background task with task id: ${extSessionId}/task-42\nTask logs are available at: file://${logPath}`,
                  },
                },
              })
              harness.pushAgentFrame({
                jsonrpc: '2.0',
                method: 'session/update',
                params: {
                  sessionId: extSessionId,
                  update: {
                    sessionUpdate: 'tool_call',
                    toolCallId: 'tc-bg-sched',
                    title: 'Schedule',
                    status: 'in_progress',
                    rawInput: {
                      DurationSeconds: 720,
                      Prompt: 'Check esurge-qwen38 compilation progress',
                    },
                  },
                },
              })
              harness.pushAgentFrame({
                jsonrpc: '2.0',
                method: 'session/update',
                params: {
                  sessionId: extSessionId,
                  update: {
                    sessionUpdate: 'tool_call_update',
                    toolCallId: 'tc-bg-sched',
                    status: 'completed',
                    rawOutput: `Tool is running as a background task with task id: ${extSessionId}/task-43`,
                  },
                },
              })
            }

            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: extSessionId,
                update: {
                  sessionUpdate: 'agent_message_chunk',
                  content: { type: 'text', text: `Handled turn ${turnNumber}` },
                },
              },
            })
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: { stopReason: 'end_turn' },
            })
            return
          }
          if (frame.method === 'session/close') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: {} })
          }
        })
        return new AcpStdioClient({ ...opts, transport: harness.transport })
      },
    })

    const runtime = new InMemoryDaemonRuntime(
      new AgentTurnRunner({ llm: makeUnusedNativeLlm(), model: '' }),
      {
        currentProjectDirectory: root,
        model: '',
        sessionDirectory: join(root, 'sessions'),
      },
    )
    const socketPath = join(root, 'agy-bg-e2e.sock')
    // A Hub that may drive this workspace, and a private schedule store (never the real one).
    const hubKeys = generateKeyPairSync('ed25519')
    const hubEnrollments = join(root, 'hub-enrollments.json')
    enrollHub(hubEnrollments, { hub_id: 'hub-1', public_key_pem: hubKeys.publicKey.export({ type: 'spki', format: 'pem' }).toString(), project_dirs: [root] })
    const server = new DaemonServer({
      socketPath,
      projectDirectory: root,
      skillDirectory: xerxesSkillsDir,
      runtime,
      acpBridgeHost: bridgeHost,
      agyBrainBridge: new AgyBrainBridge({ homeDir: fakeHome }),
      autoTitle: false,
      autoSnapshotTurns: false,
      eventJournal: { directory: join(root, 'events') },
      hubEnrollmentPath: hubEnrollments,
      cronStoreFactory: () => new JobStore(join(root, 'jobs.json')),
      cronLeasePath: join(root, 'cron.lease'),
      cronArchiveDirectory: join(root, 'cron-archive'),
    })
    await server.start()

    const client = await SocketRpcClient.open(socketPath)
    try {
      const init = await client.call('initialize', { session_key: 'bg-s1', project_dir: root })
      const sessionId = String(init.session_id)

      const conn = await client.call('acp.connect', { session_id: sessionId, agent_id: 'antigravity' })
      expect(conn.ok).toBe(true)

      // Everything agy is sent, flattened, so no Xerxes file content can hide in any block.
      const promptDump = (): string => JSON.stringify(receivedPromptFrames)
      const xerxesOnlyMarkers = [
        'Always benchmark at 32k input / 1k output.',
        'xerxes://workspace-context',
        'xerxes_workspace_context',
        'Deploy `',
        'Run decode TPS test.',
        '[Xerxes',
      ]

      // 1. Neither completions nor acp.catalog advertise .xerxes commands or skills to agy.
      const slashCompletions = await client.call('complete', { text: '/', session_key: 'bg-s1' })
      const labels = (slashCompletions.completions as Array<{ label: string }>).map(c => c.label)
      expect(labels).not.toContain('deploy')
      expect(labels).not.toContain('tpu-bench')
      expect(labels).not.toContain('immortals-coordinator')

      const catalog = await client.call('acp.catalog', { session_id: sessionId })
      expect((catalog.commands as Array<{ name: string }>).map(c => c.name)).not.toContain('deploy')
      expect((catalog.skills as Array<{ name: string }>).map(s => s.name)).not.toContain('tpu-bench')

      // 2. Turn 1 reaches agy exactly as typed: no expansion, no attached Xerxes context.
      await client.call('turn.submit', { session_key: 'bg-s1', text: '/deploy qwen38-v5p' })
      await client.waitForEvent(
        f => f.method === 'event' && f.params?.type === 'turn_end',
      )

      expect(receivedPromptFrames).toHaveLength(1)
      const turn1Prompt = receivedPromptFrames[0]!.prompt as Array<Record<string, unknown>>
      expect(turn1Prompt).toEqual([{ type: 'text', text: '/deploy qwen38-v5p' }])

      // 3. Background activity, status, live log inspection, and schedule listing all reflect AGY background work!
      const bgStatus = await client.call('background.status', { session_key: 'bg-s1' })
      expect(bgStatus).toMatchObject({ ok: true, shells: 1, watchers: 0, schedules: 1 })

      const bgActivity = await client.call('background.activity', { session_key: 'bg-s1' })
      const rows = bgActivity.rows as Array<Record<string, unknown>>
      expect(rows.some(r => r.id === `${AGY_TASK_ID_PREFIX}${extSessionId}/task-42` && r.kind === 'shell' && r.state === 'running')).toBe(true)
      expect(rows.some(r => r.id === `${AGY_TASK_ID_PREFIX}${extSessionId}/task-43` && r.kind === 'schedule' && r.state === 'scheduled')).toBe(true)

      const inspected = await client.call('terminal.inspect', {
        session_key: 'bg-s1',
        terminal_id: `${AGY_TASK_ID_PREFIX}${extSessionId}/task-42`,
      })
      expect(inspected.ok).toBe(true)
      expect((inspected.terminal as Record<string, unknown>).output).toContain('[replica-1] compiled 4/9 buckets')
      expect((inspected.terminal as Record<string, unknown>).canInterrupt).toBe(true)

      const schedules = await client.call('schedule.list', { session_key: 'bg-s1' })
      const jobs = schedules.jobs as Array<Record<string, unknown>>
      expect(jobs.some(j => j.id === `${AGY_TASK_ID_PREFIX}${extSessionId}/task-43` && j.source === 'antigravity')).toBe(true)

      // 4. Editing XERXES.md and adding a global one still sends agy nothing but the typed text.
      await mkdir(join(fakeHome, '.xerxes', 'commands'), { recursive: true })
      await writeFile(join(fakeHome, '.xerxes', 'XERXES.md'), '# Global Policy\nAlways prefer Bun over Node.')
      await writeFile(join(root, 'XERXES.md'), '# Workspace Policy\nAlways benchmark at 32k input / 1k output and report ITL p99.')
      client.frames.length = 0
      await client.call('turn.submit', { session_key: 'bg-s1', text: 'Report final TPS' })
      await client.waitForEvent(
        f => f.method === 'event' && f.params?.type === 'turn_end',
      )
      expect(receivedPromptFrames).toHaveLength(2)
      expect(receivedPromptFrames[1]!.prompt).toEqual([{ type: 'text', text: 'Report final TPS' }])

      // 5. Board and background controls no longer write prompts for agy.
      const cancelRes = await client.call('schedule.cancel', {
        session_key: 'bg-s1',
        schedule_id: `${AGY_TASK_ID_PREFIX}${extSessionId}/task-43`,
      })
      expect(cancelRes.ok).toBe(false)
      expect(String(cancelRes.error)).toContain('written by Xerxes')
      const postRes = await client.call('swarm.post', { session_key: 'bg-s1', text: '@worker-1 status?' })
      expect(postRes.ok).toBe(false)
      expect(String(postRes.error)).toContain('written by Xerxes')
      const killRes = await client.call('swarm.kill', { session_key: 'bg-s1', name: 'all' })
      expect(killRes.ok).toBe(false)
      expect(String(killRes.error)).toContain('written by Xerxes')

      // 6. A prompt Xerxes composes itself (/init) is refused, visibly, before reaching agy.
      client.frames.length = 0
      await client.call('slash', { command: '/init' })
      const refused = await client.waitForEvent(
        f => f.method === 'event' && f.params?.type === 'notification'
          && String((f.params?.payload as Record<string, unknown> | undefined)?.message ?? '').includes('only receives messages you type'),
      )
      expect(refused).toBeDefined()
      await Bun.sleep(50)
      expect(receivedPromptFrames).toHaveLength(2)
      for (const marker of xerxesOnlyMarkers) expect(promptDump()).not.toContain(marker)

      // 7. Xerxes goal mode is off for Antigravity sessions; agy has its own.
      const goalRes = await client.call('session.goal', { session_key: 'bg-s1', input: 'Ship the TPU benchmark report' })
      expect(goalRes.ok).toBe(false)
      expect(String(goalRes.error)).toContain('own goal mode')
      const goalShow = await client.call('session.goal', { session_key: 'bg-s1', input: '' })
      expect(goalShow.ok).not.toBe(false)
      await Bun.sleep(30)
      expect(receivedPromptFrames).toHaveLength(2)

      // 8. Hub users (even with operator rights) cannot drive agy on the owner's Google sign-in.
      const hubCall = (method: string, params: JsonRpcPayload) => {
        const claims: HubAuthClaims = {
          hub_id: 'hub-1', principal_id: 'bob', principal_kind: 'user', user_id: 'user-bob', device_id: 'bob-phone',
          display_name: 'Bob', workspace_id: 'ws-1', session_id: sessionId, method, params_hash: paramsHash(params),
          request_id: randomUUID(), nonce: randomUUID(), issued_at: Date.now(), expires_at: Date.now() + 60_000,
          scopes: ['session.read', 'session.submit', 'session.steer', 'session.cancel'],
        }
        return client.call(method, { ...params, hub_auth: signHubAuth(claims, hubKeys.privateKey) as unknown as JsonRpcPayload })
      }
      const hubSubmit = await hubCall('turn.submit', { session_key: 'bg-s1', text: 'run the benchmark for me' })
      expect(hubSubmit).toMatchObject({ ok: false, code: 'antigravity-local-only' })
      const hubSteer = await hubCall('turn.steer', { session_key: 'bg-s1', content: 'also check p99' })
      expect(hubSteer).toMatchObject({ ok: false, code: 'antigravity-local-only' })
      const hubCancel = await hubCall('turn.cancel', { session_key: 'bg-s1' })
      expect(hubCancel).toMatchObject({ ok: false, code: 'antigravity-local-only' })

      // 9. A scheduled run never reaches agy, even when its prompt is plain user text.
      const created = await client.call('schedule.create', {
        session_key: 'bg-s1', target: 'session', prompt: 'Report TPS', schedule: '0 9 * * *', paused: true,
        max_runs: 3, expires_at: new Date(Date.now() + 86_400_000).toISOString(),
      })
      expect((created.job as { target_session_id: string | null }).target_session_id).toBe(sessionId)
      const scheduleId = String((created.job as { id: string }).id)
      // The run fails, and says why.
      await expect(client.call('schedule.run', { session_key: 'bg-s1', schedule_id: scheduleId })).rejects.toThrow('only receives messages you type')
      await Bun.sleep(30)
      expect(receivedPromptFrames).toHaveLength(2)
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('refuses Hub Autopilot/remote answers on Antigravity permission & question prompts and keeps steered @mentions verbatim', async () => {
    const root = await makeTempDir('xerxes-agy-hub-guard-')
    const fakeHome = await makeTempDir('xerxes-agy-hub-home-')
    await writeFile(join(root, 'notes.txt'), 'secret-note-content\n')

    const extSessionId = 'agy-guard-sess-1'
    const receivedPromptFrames: Array<Record<string, unknown>> = []
    const agentReceivedResponses: Array<Record<string, unknown>> = []
    let turnNumber = 0
    let pendingPromptId: number | undefined
    let activeHarness: InMemoryAcpHarness | undefined

    const agentInfo: AcpExternalAgentInfo = {
      id: 'antigravity',
      name: 'Antigravity',
      provider: 'antigravity',
      transport: 'acp',
      available: true,
      command: 'agy',
      args: ['acp'],
      description: 'Mock Antigravity ACP',
    }

    const bridgeHost = new DaemonAcpBridgeHost({
      homeDir: fakeHome,
      discoverAgents: () => [agentInfo],
      createAcpClient: opts => {
        const harness = createInMemoryAcpTransport(frame => {
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                protocolVersion: 1,
                agentInfo: { name: 'Antigravity ACP', version: '1.5.15' },
                agentCapabilities: {
                  promptCapabilities: { image: false, embeddedContext: false },
                },
              },
            })
            return
          }
          if (frame.method === 'session/new') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                sessionId: extSessionId,
                modes: {
                  currentModeId: 'default',
                  availableModes: [{ id: 'default', name: 'Default' }],
                },
              },
            })
            return
          }
          if (frame.method === 'session/set_mode' || frame.method === 'session/set_config_option') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: {} })
            return
          }
          if (frame.method === 'session/prompt') {
            turnNumber += 1
            const params = frame.params as Record<string, unknown>
            receivedPromptFrames.push(params)
            if (turnNumber === 1) {
              pendingPromptId = frame.id as number
              // Emit both a permission request and an elicitation question on Turn 1.
              harness.pushAgentFrame({
                jsonrpc: '2.0',
                id: 'perm-1',
                method: 'session/request_permission',
                params: {
                  sessionId: extSessionId,
                  toolCall: { toolCallId: 'tc-perm', title: 'Run shell command', kind: 'execute' },
                  options: [
                    { optionId: 'allow', name: 'Allow once', kind: 'allow_once' },
                    { optionId: 'deny', name: 'Deny', kind: 'reject_once' },
                  ],
                },
              })
              return
            }
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: extSessionId,
                update: {
                  sessionUpdate: 'agent_message_chunk',
                  content: { type: 'text', text: `Turn ${turnNumber} ok` },
                },
              },
            })
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: { stopReason: 'end_turn' },
            })
            return
          }
          if (frame.id === 'perm-1') {
            agentReceivedResponses.push(frame)
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: 'elicit-1',
              method: 'elicitation/create',
              params: {
                sessionId: extSessionId,
                toolCallId: 'tc-ask',
                mode: 'form',
                message: 'Pick an option',
                requestedSchema: {
                  type: 'object',
                  required: ['choice'],
                  properties: {
                    choice: { type: 'string', title: 'Choice', enum: ['a', 'b'] },
                  },
                },
              },
            })
            return
          }
          if (frame.id === 'elicit-1') {
            agentReceivedResponses.push(frame)
            if (pendingPromptId !== undefined) {
              const id = pendingPromptId
              pendingPromptId = undefined
              harness.pushAgentFrame({
                jsonrpc: '2.0',
                method: 'session/update',
                params: {
                  sessionId: extSessionId,
                  update: {
                    sessionUpdate: 'agent_message_chunk',
                    content: { type: 'text', text: 'Turn 1 ok' },
                  },
                },
              })
              harness.pushAgentFrame({
                jsonrpc: '2.0',
                id,
                result: { stopReason: 'end_turn' },
              })
            }
          }
        })
        activeHarness = harness
        return new AcpStdioClient({ ...opts, transport: harness.transport })
      },
    })

    const runtime = new InMemoryDaemonRuntime(
      new AgentTurnRunner({ llm: makeUnusedNativeLlm(), model: '' }),
      {
        currentProjectDirectory: root,
        model: '',
        sessionDirectory: join(root, 'sessions'),
      },
    )
    const hubKeys = generateKeyPairSync('ed25519')
    const hubEnrollPath = join(root, 'hub-enrollments.json')
    enrollHub(hubEnrollPath, {
      hub_id: 'hub-1',
      public_key_pem: hubKeys.publicKey.export({ type: 'spki', format: 'pem' }).toString(),
      project_dirs: [root],
      workspace_ids: ['ws-1'],
    })
    const socketPath = join(root, 'agy-hub-guard.sock')
    const server = new DaemonServer({
      socketPath,
      projectDirectory: root,
      runtime,
      acpBridgeHost: bridgeHost,
      agyBrainBridge: new AgyBrainBridge({ homeDir: fakeHome }),
      hubEnrollmentPath: hubEnrollPath,
      autoTitle: false,
      autoSnapshotTurns: false,
      eventJournal: { directory: join(root, 'events') },
    })
    await server.start()

    const client = await SocketRpcClient.open(socketPath)
    try {
      const init = await client.call('initialize', { session_key: 'hg-s1', project_dir: root })
      const sessionId = String(init.session_id)
      // Connect explicitly in `default` mode so `session/request_permission` waits for a response.
      const conn = await client.call('acp.connect', { session_id: sessionId, agent_id: 'antigravity', mode_id: 'default' })
      expect(conn.ok).toBe(true)

      const hubInteractionCall = (
        method: 'permission_response' | 'question_response',
        params: JsonRpcPayload,
        principalKind: 'agent' | 'user',
        scopes: readonly string[],
      ) => {
        const claims: HubAuthClaims = {
          hub_id: 'hub-1',
          principal_id: principalKind === 'agent' ? 'xerxes-supervisor' : 'bob',
          principal_kind: principalKind,
          ...(principalKind === 'user' ? { user_id: 'user-bob', device_id: 'bob-phone' } : {}),
          display_name: principalKind === 'agent' ? 'XERXES Supervisor' : 'Bob',
          workspace_id: 'ws-1',
          session_id: sessionId,
          method,
          params_hash: paramsHash(params),
          request_id: randomUUID(),
          nonce: randomUUID(),
          issued_at: Date.now(),
          expires_at: Date.now() + 60_000,
          scopes,
        }
        return client.call(method, { ...params, hub_auth: signHubAuth(claims, hubKeys.privateKey) as unknown as JsonRpcPayload })
      }

      await client.call('turn.submit', { session_key: 'hg-s1', text: 'Start task' })

      // 1. Wait for approval_request; Hub Autopilot and Hub user must BOTH be refused.
      const approvalEvent = await client.waitForEvent(
        f => f.method === 'event' && f.params?.type === 'approval_request',
      )
      const approvalId = String((approvalEvent.params?.payload as Record<string, unknown>).request_id)

      const autopilotPerm = await hubInteractionCall(
        'permission_response',
        { request_id: approvalId, response: 'approve' },
        'agent',
        ['permission.delegated'],
      )
      expect(autopilotPerm).toMatchObject({ ok: false, code: 'antigravity-local-only' })

      const remoteUserPerm = await hubInteractionCall(
        'permission_response',
        { request_id: approvalId, response: 'approve' },
        'user',
        ['permission.respond'],
      )
      expect(remoteUserPerm).toMatchObject({ ok: false, code: 'antigravity-local-only' })
      expect(agentReceivedResponses).toHaveLength(0)

      // Queue a local steer containing `@notes.txt` while Turn 1 is active: must stay verbatim (no `<attached_files>`).
      const steerRes = await client.call('turn.steer', { session_key: 'hg-s1', content: 'Inspect @notes.txt' })
      expect(steerRes.ok).toBe(true)

      // Local owner approves the permission request -> elicitation question arrives.
      const localPerm = await client.call('permission_response', { request_id: approvalId, response: 'approve' })
      expect(localPerm.ok).toBe(true)

      const questionEvent = await client.waitForEvent(
        f => f.method === 'event' && f.params?.type === 'question_request',
      )
      const questionId = String((questionEvent.params?.payload as Record<string, unknown>).id)

      // 2. Hub Autopilot and Hub user must BOTH be refused on `question_response`.
      const autopilotQuestion = await hubInteractionCall(
        'question_response',
        { request_id: questionId, answers: { choice: 'a' } },
        'agent',
        ['question.delegated'],
      )
      expect(autopilotQuestion).toMatchObject({ ok: false, code: 'antigravity-local-only' })

      const remoteUserQuestion = await hubInteractionCall(
        'question_response',
        { request_id: questionId, answers: { choice: 'a' } },
        'user',
        ['question.respond'],
      )
      expect(remoteUserQuestion).toMatchObject({ ok: false, code: 'antigravity-local-only' })
      expect(agentReceivedResponses).toHaveLength(1)

      // Local owner answers the question -> Turn 1 finishes and queued steer triggers Turn 2.
      const localQuestion = await client.call('question_response', { request_id: questionId, answers: { choice: 'a' } })
      expect(localQuestion.ok).toBe(true)

      // Wait for Turn 2 (the steered continuation) to settle.
      const deadline = Date.now() + 5000
      while (receivedPromptFrames.length < 2 && Date.now() < deadline) {
        await Bun.sleep(15)
      }
      expect(receivedPromptFrames).toHaveLength(2)
      expect(receivedPromptFrames[1]!.prompt).toEqual([{ type: 'text', text: 'Inspect @notes.txt' }])
      expect(JSON.stringify(receivedPromptFrames[1])).not.toContain('<attached_files>')
      expect(JSON.stringify(receivedPromptFrames[1])).not.toContain('secret-note-content')
      expect(activeHarness).toBeDefined()
    } finally {
      client.close()
      await server.stop()
    }
  })
})
