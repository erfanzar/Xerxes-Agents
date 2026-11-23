// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { AcpStdioClient, type AcpSpawnedProcess } from '../src/acp/client.js'
import type { AcpExternalAgentInfo } from '../src/acp/externalAgents.js'
import {
  expandXerxesAcpSlashCommand,
  loadXerxesWorkspaceBridgeForAcp,
} from '../src/acp/xerxesWorkspaceBridge.js'
import { DaemonAcpBridgeHost } from '../src/daemon/acpBridgeHost.js'
import { AGY_TASK_ID_PREFIX, AgyBrainBridge } from '../src/daemon/agyBrainBridge.js'
import { InMemoryDaemonRuntime, inflightToolContext } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import type { LlmClient, LlmDelta } from '../src/llms/client.js'
import type { JsonRpcPayload } from '../src/protocol/jsonRpc.js'

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

describe('Xerxes workspace bridge for Antigravity ACP', () => {
  test('discovers XERXES.md, XERXES.local.md, .xerxes notes/skills/agents/commands, deduplicates .agents/skills, skips AGENTS.md, and expands slash commands', async () => {
    const root = await makeTempDir('xerxes-acp-bridge-')
    const emptyDir = await makeTempDir('xerxes-acp-empty-')

    // Empty workspace returns null (zero noise on projects without Xerxes files)
    expect(await loadXerxesWorkspaceBridgeForAcp(emptyDir)).toBeNull()

    // Populate Xerxes + Antigravity workspace files
    await writeFile(join(root, 'AGENTS.md'), '# Native AGY instructions (should NOT be duplicated)')
    await writeFile(join(root, 'XERXES.md'), '# Project Xerxes Guide\nAlways run `bun test` before shipping.')
    await writeFile(join(root, 'XERXES.local.md'), 'Local machine TPU host: alham')
    await mkdir(join(root, '.xerxes', 'skills', 'tpu-bench'), { recursive: true })
    await mkdir(join(root, '.xerxes', 'skills', 'shared-skill'), { recursive: true })
    await mkdir(join(root, '.agents', 'skills', 'shared-skill'), { recursive: true })
    await mkdir(join(root, '.xerxes', 'agents'), { recursive: true })
    await mkdir(join(root, '.xerxes', 'commands'), { recursive: true })

    await writeFile(
      join(root, '.xerxes', 'architecture.md'),
      '# Architecture Note\nUse eSurge hybrid KV cache.',
    )
    await writeFile(
      join(root, '.xerxes', 'skills', 'tpu-bench', 'SKILL.md'),
      '---\nname: tpu-bench\ndescription: Benchmark TPU serving throughput\n---\nMeasure TTFT and decode TPS.',
    )
    await writeFile(
      join(root, '.xerxes', 'skills', 'shared-skill', 'SKILL.md'),
      '---\nname: shared-skill\ndescription: Duplicated in .agents/skills\n---\nDuplicate body.',
    )
    await writeFile(
      join(root, '.agents', 'skills', 'shared-skill', 'SKILL.md'),
      '---\nname: shared-skill\ndescription: Native AGY skill\n---\nNative body.',
    )
    await writeFile(
      join(root, '.xerxes', 'agents', 'reviewer.md'),
      '---\nname: reviewer\ndescription: Strict diff reviewer\n---\nCheck edge cases and types.',
    )
    await writeFile(
      join(root, '.xerxes', 'commands', 'deploy.md'),
      '---\ndescription: Deploy serving cluster\n---\nDeploy target `$ARGUMENTS` and verify health.',
    )

    const bridge = await loadXerxesWorkspaceBridgeForAcp(root)
    expect(bridge).not.toBeNull()
    expect(bridge!.instructionFiles.map(f => f.label)).toEqual(['Project XERXES.md', 'Project XERXES.local.md'])
    expect(bridge!.notes.map(n => n.relativePath)).toEqual(['architecture.md'])
    expect(bridge!.skills.map(s => s.name)).toEqual(['tpu-bench']) // shared-skill deduped against .agents/skills
    expect(bridge!.agents.map(a => a.name)).toEqual(['reviewer'])
    expect(bridge!.commands.map(c => c.name)).toEqual(['deploy'])
    expect(bridge!.firstTurnContextBlock).toContain('Always run `bun test` before shipping.')
    expect(bridge!.firstTurnContextBlock).toContain('Local machine TPU host: alham')
    expect(bridge!.firstTurnContextBlock).toContain('tpu-bench')
    expect(bridge!.firstTurnContextBlock).toContain('reviewer')
    expect(bridge!.firstTurnContextBlock).toContain('/deploy')
    expect(bridge!.firstTurnContextBlock).not.toContain('should NOT be duplicated')

    // Command expansion with $ARGUMENTS
    const expandedCmd = await expandXerxesAcpSlashCommand('/deploy staging-v5p', bridge)
    expect(expandedCmd).not.toBeNull()
    expect(expandedCmd!.displayText).toBe('/deploy staging-v5p')
    expect(expandedCmd!.matchedName).toBe('deploy')
    expect(expandedCmd!.expandedPrompt).toContain('Deploy target `staging-v5p` and verify health.')

    // Skill expansion
    const expandedSkill = await expandXerxesAcpSlashCommand('/tpu-bench qwen38', bridge)
    expect(expandedSkill).not.toBeNull()
    expect(expandedSkill!.matchedName).toBe('tpu-bench')
    expect(expandedSkill!.expandedPrompt).toContain('Measure TTFT and decode TPS.')
    expect(expandedSkill!.expandedPrompt).toContain('qwen38')

    // Never hijacks an ACP-owned slash command
    expect(
      await expandXerxesAcpSlashCommand('/deploy prod', bridge, new Set(['deploy'])),
    ).toBeNull()
  })
})

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

  test('DaemonServer integrates AGY background tasks, live log inspection, schedule cancel, and first-turn XERXES.md + .xerxes slash bridging E2E', async () => {
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
    })
    await server.start()

    const client = await SocketRpcClient.open(socketPath)
    try {
      const init = await client.call('initialize', { session_key: 'bg-s1', project_dir: root })
      const sessionId = String(init.session_id)

      const conn = await client.call('acp.connect', { session_id: sessionId, agent_id: 'antigravity' })
      expect(conn.ok).toBe(true)

      // 1. Completions and acp.catalog include project .xerxes commands & skills, never built-in Xerxes skills
      const slashCompletions = await client.call('complete', { text: '/', session_key: 'bg-s1' })
      const labels = (slashCompletions.completions as Array<{ label: string }>).map(c => c.label)
      expect(labels).toContain('deploy')
      expect(labels).toContain('tpu-bench')
      expect(labels).not.toContain('immortals-coordinator')

      const catalog = await client.call('acp.catalog', { session_id: sessionId })
      expect((catalog.commands as Array<{ name: string }>).map(c => c.name)).toContain('deploy')
      expect((catalog.skills as Array<{ name: string }>).map(s => s.name)).toContain('tpu-bench')
      expect((catalog.skills as Array<{ name: string }>).map(s => s.name)).not.toContain('immortals-coordinator')

      // 2. Turn 1 via `/deploy qwen38-v5p`: expands .xerxes command AND attaches xerxes://workspace-context ONCE
      await client.call('turn.submit', { session_key: 'bg-s1', text: '/deploy qwen38-v5p' })
      await client.waitForEvent(
        f => f.method === 'event' && f.params?.type === 'turn_end',
      )

      expect(receivedPromptFrames).toHaveLength(1)
      const turn1Prompt = receivedPromptFrames[0]!.prompt as Array<Record<string, unknown>>
      const resourceBlock = turn1Prompt.find(b => b.type === 'resource') as
        | { resource?: { uri?: string; text?: string } }
        | undefined
      expect(resourceBlock?.resource?.uri).toBe('xerxes://workspace-context')
      expect(resourceBlock?.resource?.text).toContain('Always benchmark at 32k input / 1k output.')
      const textBlock1 = turn1Prompt.find(b => b.type === 'text') as { text?: string } | undefined
      expect(textBlock1?.text).toContain('Deploy `qwen38-v5p` across TPU hosts.')

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

      // 4. Turn 2 does NOT re-inject xerxes://workspace-context when files are unchanged
      client.frames.length = 0
      await client.call('turn.submit', { session_key: 'bg-s1', text: 'Report final TPS' })
      await client.waitForEvent(
        f => f.method === 'event' && f.params?.type === 'turn_end',
      )
      expect(receivedPromptFrames).toHaveLength(2)
      const turn2Prompt = receivedPromptFrames[1]!.prompt as Array<Record<string, unknown>>
      expect(turn2Prompt.some(b => b.type === 'resource')).toBe(false)

      // 4b. Editing XERXES.md or adding ~/.xerxes/XERXES.md mid-session dynamically refreshes xerxes://workspace-context on the next turn!
      await mkdir(join(fakeHome, '.xerxes', 'commands'), { recursive: true })
      await writeFile(join(fakeHome, '.xerxes', 'XERXES.md'), '# Global Policy\nAlways prefer Bun over Node.')
      await writeFile(
        join(fakeHome, '.xerxes', 'commands', 'global-audit.md'),
        '---\ndescription: Global audit checklist\n---\nAudit `$ARGUMENTS` thoroughly.',
      )
      await writeFile(join(root, 'XERXES.md'), '# Workspace Policy\nAlways benchmark at 32k input / 1k output and report ITL p99.')
      client.frames.length = 0
      await client.call('turn.submit', { session_key: 'bg-s1', text: '/global-audit gdn-kernel' })
      await client.waitForEvent(
        f => f.method === 'event' && f.params?.type === 'turn_end',
      )
      expect(receivedPromptFrames).toHaveLength(3)
      const turn3Prompt = receivedPromptFrames[2]!.prompt as Array<Record<string, unknown>>
      const refreshedRes = turn3Prompt.find(b => b.type === 'resource') as
        | { resource?: { uri?: string; text?: string } }
        | undefined
      expect(refreshedRes?.resource?.uri).toBe('xerxes://workspace-context')
      expect(refreshedRes?.resource?.text).toContain('report ITL p99')
      expect(refreshedRes?.resource?.text).toContain('Always prefer Bun over Node.')
      const turn3Text = turn3Prompt.find(b => b.type === 'text') as { text?: string } | undefined
      expect(turn3Text?.text).toContain('Audit `gdn-kernel` thoroughly.')

      // 5. schedule.cancel cancels the AGY timer and dispatches manage_task(kill)
      client.frames.length = 0
      const cancelRes = await client.call('schedule.cancel', {
        session_key: 'bg-s1',
        schedule_id: `${AGY_TASK_ID_PREFIX}${extSessionId}/task-43`,
      })
      expect(cancelRes).toMatchObject({ ok: true, requested: true, dispatched: true })
      const deadline = Date.now() + 2000
      while (receivedPromptFrames.length < 4 && Date.now() < deadline) {
        await Bun.sleep(10)
      }
      const killPrompt = receivedPromptFrames.at(-1)?.prompt as Array<{ text?: string }> | undefined
      expect(killPrompt?.some(b => b.text?.includes('manage_task(Action: "kill"') && b.text?.includes(`${extSessionId}/task-43`))).toBe(true)
    } finally {
      client.close()
      await server.stop()
    }
  })
})
