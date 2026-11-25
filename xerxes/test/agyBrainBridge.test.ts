// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { createHash } from 'node:crypto'
import { existsSync } from 'node:fs'
import { mkdir, mkdtemp, readFile, realpath, rm, symlink, writeFile } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { AcpStdioClient, type AcpSpawnedProcess } from '../src/acp/client.js'
import type { AcpExternalAgentInfo } from '../src/acp/externalAgents.js'
import {
  DaemonAcpBridgeHost,
  makeAntigravityUpdateFocus,
  makeAntigravityUpdateStepsPlans,
} from '../src/daemon/acpBridgeHost.js'
import {
  AgyBrainBridge,
  artifactIdForAgyPath,
  parseAgyArtifactFilename,
  parseAgyDirFlags,
  parseExternalBoardMentions,
  resolveAgyBrainLocation,
} from '../src/daemon/agyBrainBridge.js'
import {
  previewWorkspaceFile,
  previewWorkspaceFileRich,
  resolvePreviewPath,
  writeWorkspaceFile,
} from '../src/daemon/filePreview.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { readRemoteChatImage } from '../src/desktop/main/chatMedia.js'
import type { JsonRpcPayload } from '../src/protocol/jsonRpc.js'
import type { SwarmLifecycleEvent } from '../src/swarm/types.js'

const encoder = new TextEncoder()
const tempRoots: string[] = []
const originalXerxesHome = process.env.XERXES_HOME

afterEach(async () => {
  if (originalXerxesHome === undefined) {
    delete process.env.XERXES_HOME
  } else {
    process.env.XERXES_HOME = originalXerxesHome
  }
  for (const dir of tempRoots.splice(0)) {
    await rm(dir, { recursive: true, force: true })
  }
})

async function makeTempDir(prefix = 'xrx-agy-brain-'): Promise<string> {
  const baseTmp = process.platform === 'win32' ? tmpdir() : (existsSync('/tmp') ? '/tmp' : tmpdir())
  const dir = await realpath(await mkdtemp(join(baseTmp, prefix)))
  tempRoots.push(dir)
  process.env.XERXES_HOME = join(dir, 'xerxes-home')
  await mkdir(process.env.XERXES_HOME, { recursive: true })
  return dir
}

async function waitUntil(predicate: () => boolean | Promise<boolean>, timeoutMs = 4_000): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (await predicate()) return
    await Bun.sleep(20)
  }
  throw new Error(`Timed out after ${timeoutMs}ms waiting for condition`)
}

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
    try { stdoutController?.close() } catch { /* ignore */ }
    try { stderrController?.close() } catch { /* ignore */ }
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
        if (onClientFrame) {
          void onClientFrame(parsed, harness)
        }
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
  readonly params?: {
    readonly payload?: JsonRpcPayload
    readonly type?: string
  } & JsonRpcPayload
  readonly result?: JsonRpcPayload
}

class SocketRpcClient {
  readonly frames: RpcFrame[] = []
  private readonly waiters: Array<{
    readonly predicate: (frame: RpcFrame) => boolean
    readonly resolve: (frame: RpcFrame) => void
  }> = []
  private buffer = ''
  private nextId = 0

  private constructor(readonly socket: Socket) {
    socket.setEncoding('utf8')
    socket.on('data', (chunk: string) => {
      this.buffer += chunk
      let newline = this.buffer.indexOf('\n')
      while (newline >= 0) {
        const raw = this.buffer.slice(0, newline).trim()
        this.buffer = this.buffer.slice(newline + 1)
        if (raw) {
          const parsed = JSON.parse(raw) as RpcFrame
          this.frames.push(parsed)
          for (let i = 0; i < this.waiters.length; i += 1) {
            const waiter = this.waiters[i]!
            if (waiter.predicate(parsed)) {
              this.waiters.splice(i, 1)
              waiter.resolve(parsed)
              break
            }
          }
        }
        newline = this.buffer.indexOf('\n')
      }
    })
  }

  static async open(socketPath: string): Promise<SocketRpcClient> {
    const socket = await new Promise<Socket>((resolveSocket, reject) => {
      const conn = connect(socketPath)
      conn.once('connect', () => resolveSocket(conn))
      conn.once('error', reject)
    })
    return new SocketRpcClient(socket)
  }

  async next(predicate: (frame: RpcFrame) => boolean, timeoutMs = 5_000): Promise<RpcFrame> {
    const existing = this.frames.find(predicate)
    if (existing) return existing
    return new Promise<RpcFrame>((resolveFrame, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`Timed out waiting for daemon frame after ${timeoutMs}ms`))
      }, timeoutMs)
      this.waiters.push({
        predicate,
        resolve: frame => {
          clearTimeout(timer)
          resolveFrame(frame)
        },
      })
    })
  }

  async call(method: string, params: JsonRpcPayload = {}): Promise<JsonRpcPayload> {
    const id = ++this.nextId
    this.socket.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`)
    const frame = await this.next(candidate => candidate.id === id)
    if (frame.error) {
      throw new Error(`RPC ${method} failed (${frame.error.code}): ${frame.error.message}`)
    }
    return frame.result ?? {}
  }

  close(): void {
    this.socket.destroy()
  }
}

const AGY_AGENTS: readonly AcpExternalAgentInfo[] = [
  {
    id: 'xerxes',
    name: 'Xerxes',
    provider: 'xerxes',
    transport: 'native',
    available: true,
    description: 'Native runtime',
  },
  {
    id: 'antigravity',
    name: 'Antigravity',
    provider: 'antigravity',
    transport: 'acp',
    available: true,
    command: '/usr/local/bin/agy',
    args: ['acp'],
    binaryPath: '/usr/local/bin/agy',
    description: 'Antigravity ACP',
  },
]

async function seedMockAgyBrain(homeDir: string, extId: string): Promise<{
  readonly brainRoot: string
  readonly sessionDir: string
  readonly swarmDir: string
  readonly narratorConvId: string
  readonly workerConvId: string
  readonly subagentConvId: string
}> {
  const brainRoot = join(homeDir, '.gemini', 'antigravity-cli', 'brain')
  const sessionDir = join(brainRoot, extId)
  const swarmDir = join(sessionDir, '.system_generated', 'swarm')
  const subagentsDir = join(sessionDir, '.system_generated', 'subagents')
  const narratorConvId = 'narrator-conv-001'
  const workerConvId = 'worker-conv-002'
  const monitorConvId = 'monitor-conv-003'
  const stoppedConvId = 'stopped-conv-004'
  const killedConvId = 'killed-conv-005'
  const subagentConvId = 'subagent-conv-099'

  await mkdir(swarmDir, { recursive: true })
  await mkdir(subagentsDir, { recursive: true })
  await mkdir(join(brainRoot, narratorConvId, 'swarm_view'), { recursive: true })
  await mkdir(join(brainRoot, workerConvId, '.system_generated', 'logs'), { recursive: true })
  await mkdir(join(brainRoot, subagentConvId, '.system_generated', 'logs'), { recursive: true })

  await writeFile(
    join(swarmDir, 'swarm.json'),
    JSON.stringify({
      version: 1,
      backend: 'local',
      swarm_id: extId,
      created_at: '2026-11-22T18:13:32Z',
    }),
    'utf8',
  )

  await writeFile(
    join(swarmDir, 'members.json'),
    JSON.stringify({
      title: 'Prismcore Audit Swarm',
      members: {
        coordinator: {
          conversation_id: extId,
          role: 'coordinator',
          desired_state: 'RUNNING',
          applied_model: 'gemini-4-neon',
          workspace_mode: 'inherit',
          phase: 'RUNNING',
          wake_mode: 'mentions',
          wait_since: 2,
          last_read_seq: 2,
          killed: false,
          model: 'gemini-4-neon',
        },
        monitor: {
          conversation_id: monitorConvId,
          role: 'worker',
          desired_state: 'RUNNING',
          applied_model: 'gemini-4-carbon',
          workspace_mode: 'inherit',
          phase: 'RUNNING',
          wake_mode: 'all',
          wait_since: 2,
          last_read_seq: 2,
          killed: false,
        },
        narrator: {
          conversation_id: narratorConvId,
          role: 'worker',
          desired_state: 'RUNNING',
          applied_model: 'gemini-4-carbon',
          workspace_mode: 'inherit',
          phase: 'RUNNING',
          wake_mode: 'mentions',
          wait_since: 2,
          last_read_seq: 2,
          killed: false,
        },
        'alpha-worker': {
          conversation_id: workerConvId,
          role: 'worker',
          desired_state: 'RUNNING',
          applied_model: 'gemini-4-neon',
          workspace_mode: 'share',
          phase: 'RUNNING',
          wake_mode: 'working',
          wait_since: 1,
          last_read_seq: 2,
          killed: false,
        },
        'paused-worker': {
          conversation_id: stoppedConvId,
          role: 'worker',
          desired_state: 'STOPPED',
          applied_model: 'gemini-4-carbon',
          workspace_mode: 'branch',
          phase: 'STOPPED',
          wake_mode: 'mentions',
          wait_since: 1,
          last_read_seq: 1,
          killed: false,
        },
        'done-worker': {
          conversation_id: killedConvId,
          role: 'worker',
          desired_state: 'RUNNING',
          applied_model: 'gemini-4-carbon',
          workspace_mode: 'inherit',
          phase: 'KILLED',
          wake_mode: 'mentions',
          wait_since: 1,
          last_read_seq: 1,
          killed: true,
        },
      },
    }),
    'utf8',
  )

  await writeFile(
    join(swarmDir, 'board.jsonl'),
    [
      JSON.stringify({
        seq: 1,
        agent: 'coordinator',
        text: '@alpha-worker Audit core math in @#backend and report findings.',
        time: '2026-11-22 21:13:32',
        kind: 'brief',
      }),
      JSON.stringify({
        seq: 2,
        agent: 'alpha-worker',
        text: '@coordinator @all Found 2 edge cases in quantizer.',
        time: '2026-11-22 21:14:10',
        kind: 'message',
      }),
    ].join('\n') + '\n',
    'utf8',
  )

  await writeFile(
    join(swarmDir, 'plan.json'),
    JSON.stringify({
      goal: { title: 'Prismcore Audit', line: 'Audit math and quantizers' },
      tasks: { 'alpha-worker': 'Check quantizers' },
    }),
    'utf8',
  )

  await writeFile(
    join(brainRoot, narratorConvId, 'swarm_view', 'view.json'),
    JSON.stringify({
      version: 1,
      as_of_seq: 2,
      updated_at: '2026-11-22T18:14:30Z',
      headline: 'Audit in progress',
      phase: 'execution',
      goal: 'Verify quantizer stability',
      running_summary: 'alpha-worker found 2 edge cases.',
      workstreams: [{ title: 'Quantizers', status: 'working', owner: 'alpha-worker', summary: 'Checking scales' }],
      risks: ['NaN propagation on zero variance'],
    }),
    'utf8',
  )

  await writeFile(
    join(brainRoot, workerConvId, '.system_generated', 'logs', 'transcript.jsonl'),
    [
      JSON.stringify({
        step_index: 0,
        source: 'SYSTEM',
        type: 'SYSTEM_MESSAGE',
        status: 'DONE',
        created_at: '2026-11-22T18:13:33Z',
        content: '<SYSTEM_MESSAGE>\n[Message] content=You have just started. Audit core math.\n</SYSTEM_MESSAGE>',
      }),
      JSON.stringify({
        step_index: 1,
        source: 'MODEL',
        type: 'PLANNER_RESPONSE',
        status: 'DONE',
        created_at: '2026-11-22T18:13:35Z',
        model: 'gemini-4-neon',
        input_tokens: 1200,
        cache_read_tokens: 300,
        output_tokens: 240,
        thinking: '**Inspecting quantizer files**\nReading _affine.py first.',
        tool_calls: [
          { name: 'read_file', args: { Path: '"/repo/prismcore/_affine.py"' } },
          { name: 'run_command', args: { CommandLine: '"pytest -q"', toolSummary: '"Run quantizer tests"' } },
        ],
      }),
      JSON.stringify({
        step_index: 2,
        source: 'MODEL',
        type: 'GENERIC',
        status: 'DONE',
        created_at: '2026-11-22T18:13:36Z',
        content: 'Showing lines 1-40 of _affine.py',
      }),
    ].join('\n') + '\n',
    'utf8',
  )

  // Artifacts in brain session dir
  await writeFile(
    join(sessionDir, 'audit_report.md'),
    '# Prismcore Audit Report\n\nAll findings summarized here.\n',
    'utf8',
  )
  await writeFile(
    join(sessionDir, 'audit_report.md.metadata.json'),
    JSON.stringify({
      summary: 'Audit findings summary',
      updatedAt: '2026-11-22T18:15:00Z',
      userFacing: true,
    }),
    'utf8',
  )
  await writeFile(
    join(sessionDir, 'interactive_chart.html'),
    '<!doctype html><title>Quantizer Chart</title><div>Chart</div>',
    'utf8',
  )

  // Subagents: one Swarm Worker (MUST be filtered out) + one non-swarm subagent (MUST be included)
  await writeFile(
    join(subagentsDir, `${workerConvId}.json`),
    JSON.stringify({
      conversationId: workerConvId,
      subagentDescriptor: { typeName: 'worker', role: 'Swarm Worker (alpha-worker)' },
      state: 'SUBAGENT_STATE_ALIVE',
      spawnStepIndex: 5,
    }),
    'utf8',
  )
  await writeFile(
    join(subagentsDir, `${subagentConvId}.json`),
    JSON.stringify({
      conversationId: subagentConvId,
      subagentDescriptor: { typeName: 'researcher', role: 'Benchmarks Analyst' },
      state: 'SUBAGENT_STATE_COMPLETED',
      spawnStepIndex: 12,
    }),
    'utf8',
  )
  await writeFile(
    join(brainRoot, subagentConvId, '.system_generated', 'logs', 'transcript.jsonl'),
    [
      JSON.stringify({
        step_index: 0,
        source: 'USER_EXPLICIT',
        type: 'USER_INPUT',
        status: 'DONE',
        created_at: '2026-11-22T18:14:00Z',
        content: 'Analyze throughput benchmarks.',
      }),
      JSON.stringify({
        step_index: 1,
        source: 'MODEL',
        type: 'PLANNER_RESPONSE',
        status: 'DONE',
        created_at: '2026-11-22T18:14:05Z',
        model: 'gemini-4-carbon',
        input_tokens: 500,
        output_tokens: 120,
        content: 'Throughput improved by 18% on fp16 workloads.',
      }),
    ].join('\n') + '\n',
    'utf8',
  )

  return { brainRoot, sessionDir, swarmDir, narratorConvId, workerConvId, subagentConvId }
}

describe('AgyBrainBridge — resolver, swarm projection, inspector, artifacts & subagents', () => {
  test('resolves brain root honoring --gemini-dir/--app-data-dir flags and antigravity-cli -> antigravity fallback', async () => {
    const tmp = await makeTempDir()
    const home = join(tmp, 'home')
    const extId = 'ext-sess-123'

    // 1. Flag parsing
    expect(parseAgyDirFlags('agy acp --gemini-dir .custom-gem --app-data-dir=my-agy')).toEqual({
      geminiDir: '.custom-gem',
      appDataDir: 'my-agy',
    })

    // 2. Fallback to ~/.gemini/antigravity/brain/<extId> when only IDE brain exists
    const ideBrainDir = join(home, '.gemini', 'antigravity', 'brain', extId)
    await mkdir(ideBrainDir, { recursive: true })
    const fallbackLoc = resolveAgyBrainLocation({ externalSessionId: extId, homeDir: home })
    expect(fallbackLoc?.sessionDir).toBe(ideBrainDir)

    // 3. CLI brain takes precedence once it exists
    const cliBrainDir = join(home, '.gemini', 'antigravity-cli', 'brain', extId)
    await mkdir(join(cliBrainDir, '.system_generated', 'swarm'), { recursive: true })
    const cliLoc = resolveAgyBrainLocation({ externalSessionId: extId, homeDir: home })
    expect(cliLoc?.sessionDir).toBe(cliBrainDir)

    // 4. Explicit flags override defaults
    const flagBrainDir = join(home, '.custom-gem', 'my-agy', 'brain', extId)
    await mkdir(flagBrainDir, { recursive: true })
    const flagLoc = resolveAgyBrainLocation({
      bindingCommand: 'agy acp --gemini-dir .custom-gem --app-data-dir my-agy',
      externalSessionId: extId,
      homeDir: home,
    })
    expect(flagLoc?.sessionDir).toBe(flagBrainDir)

    // 5. Reject path traversal externalSessionId & artifact filenames
    expect(resolveAgyBrainLocation({ externalSessionId: '../etc', homeDir: home })).toBeUndefined()
    expect(resolveAgyBrainLocation({ externalSessionId: 'a/b', homeDir: home })).toBeUndefined()
    expect(parseAgyArtifactFilename('agy:../escape.md')).toBeUndefined()
    expect(parseAgyArtifactFilename('agy:sub/file.md')).toBeUndefined()
    expect(parseAgyArtifactFilename('agy:.secret.md')).toBeUndefined()
    expect(parseAgyArtifactFilename('agy:run.sh')).toBeUndefined()
    expect(parseAgyArtifactFilename('agy:valid_plan.md')).toBe('valid_plan.md')
    expect(artifactIdForAgyPath(cliLoc!, join(cliBrainDir, 'report.html'))).toBe('agy:report.html')
    expect(artifactIdForAgyPath(cliLoc!, join(cliBrainDir, 'scratch', 'temp.md'))).toBeUndefined()
  })

  test('reads swarm status, board mentions, narrator view overview, worker inspector, artifacts, and non-swarm subagents', async () => {
    const tmp = await makeTempDir()
    const home = join(tmp, 'home')
    const extId = 'ext-prism-001'
    const seeded = await seedMockAgyBrain(home, extId)

    const bridge = new AgyBrainBridge({ homeDir: home })
    const loc = bridge.resolveLocation({ externalSessionId: extId })!
    expect(loc.sessionDir).toBe(seeded.sessionDir)

    const displayMap = new Map([
      ['gemini-4-neon', 'Gemini 4 Neon'],
      ['gemini-4-carbon', 'Gemini 4 Carbon'],
    ])
    const status = await bridge.readSwarmStatus('xrx-sess-1', loc, displayMap)
    expect(status.active).toBe(true)
    expect(status.session_id).toBe('xrx-sess-1')
    expect(status.title).toBe('Prismcore Audit Swarm')
    expect(status.head).toBe(2)
    expect(status.overview?.summary).toBe('Audit in progress')
    expect(status.overview?.markdown).toContain('## Audit in progress (execution)')
    expect(status.overview?.markdown).toContain('Quantizers')
    expect(status.overview?.author).toBe('narrator')

    const byName = new Map(status.members.map(m => [m.name, m]))
    expect(byName.get('coordinator')).toMatchObject({
      role: 'coordinator',
      state: 'waiting',
      mentions_only: true,
      parked_since: 2,
      conversation_id: extId,
      model_display_name: 'Gemini 4 Neon',
      last_post_seq: 1,
    })
    expect(byName.get('monitor')).toMatchObject({
      role: 'monitor',
      state: 'waiting',
      mentions_only: false,
      parked_since: 2,
      conversation_id: 'monitor-conv-003',
    })
    expect(byName.get('narrator')).toMatchObject({
      role: 'narrator',
      state: 'waiting',
      mentions_only: true,
      conversation_id: seeded.narratorConvId,
    })
    expect(byName.get('alpha-worker')).toMatchObject({
      role: 'worker',
      state: 'working',
      workspace: 'share',
      added_by: 'coordinator',
      conversation_id: seeded.workerConvId,
      last_post_seq: 2,
      current_activity: 'run_command: Run quantizer tests',
    })
    expect(byName.get('paused-worker')).toMatchObject({
      state: 'stopped',
      killed: false,
      workspace: 'branch',
    })
    expect(byName.get('done-worker')).toMatchObject({
      state: 'killed',
      killed: true,
    })

    // Board parsing & mention extraction
    const board = await bridge.readSwarmBoard(loc)
    expect(board.head).toBe(2)
    expect(board.messages).toHaveLength(2)
    expect(board.messages[0]).toMatchObject({
      seq: 1,
      agent: 'coordinator',
      channel: 'main',
      mentions: ['alpha-worker'],
      channel_mentions: ['backend'],
      broadcast: false,
    })
    expect(board.messages[1]).toMatchObject({
      seq: 2,
      agent: 'alpha-worker',
      mentions: ['coordinator'],
      broadcast: true,
    })
    expect(parseExternalBoardMentions('@#main @all @foo')).toEqual({
      mentions: ['foo'],
      channel_mentions: ['main'],
      broadcast: true,
    })

    // Worker inspector & live activity
    const inspected = await bridge.inspectSwarmMember('xrx-sess-1', loc, 'alpha-worker', {
      modelDisplayMap: displayMap,
    })
    expect(inspected).toBeDefined()
    expect(inspected!.conversation_id).toBe(seeded.workerConvId)
    expect(inspected!.activity.phase).toBe('tool')
    expect(inspected!.activity.active_tools).toHaveLength(1)
    expect(inspected!.activity.active_tools[0]!.name).toBe('run_command')
    expect(inspected!.activity.recent_tools).toHaveLength(1)
    expect(inspected!.activity.recent_tools[0]!.name).toBe('read_file')
    expect(inspected!.messages.length).toBeGreaterThanOrEqual(3)

    // Artifacts: list, read, pin, update
    const artifacts = await bridge.listArtifacts('xrx-sess-1', loc)
    expect(artifacts.map(a => a.id).sort()).toEqual(['agy:audit_report.md', 'agy:interactive_chart.html'])
    const mdArtifact = await bridge.readArtifact('xrx-sess-1', loc, 'agy:audit_report.md')
    expect(mdArtifact?.title).toBe('Prismcore Audit Report')
    expect(mdArtifact?.kind).toBe('markdown')
    const htmlArtifact = await bridge.readArtifact('xrx-sess-1', loc, 'agy:interactive_chart.html')
    expect(htmlArtifact?.title).toBe('Quantizer Chart')
    expect(htmlArtifact?.kind).toBe('html')

    const pinned = await bridge.pinArtifact('xrx-sess-1', loc, 'agy:interactive_chart.html', true)
    expect(pinned?.pinned).toBe(true)
    // The bridge is read-only: it exposes no way to write into an Antigravity brain directory.
    expect((bridge as unknown as Record<string, unknown>).updateArtifact).toBeUndefined()

    // Subagents: Swarm Worker filtered out; non-swarm subagent surfaced with history!
    const subagents = bridge.listSubagentSnapshotsSync(loc)
    expect(subagents).toHaveLength(1)
    expect(subagents[0]).toMatchObject({
      id: seeded.subagentConvId,
      name: 'Benchmarks Analyst',
      agent_id: 'researcher',
      status: 'completed',
      model: 'gemini-4-carbon',
      input_tokens: 500,
      output_tokens: 120,
    })
    const subInspect = await bridge.inspectSubagent(loc, seeded.subagentConvId, { limit: 10 })
    expect(subInspect?.agent.output).toContain('Throughput improved by 18%')
    expect(subInspect?.history_session_id).toBe(seeded.subagentConvId)
    expect(Array.isArray((subInspect?.history as { actions?: unknown[] })?.actions)).toBe(true)

    bridge.dispose()
  })

  test('handles large (>1 MiB) worker transcript.jsonl safely via tail-bounded read', async () => {
    const tmp = await makeTempDir()
    const home = join(tmp, 'home')
    const extId = 'ext-big-transcript'
    const seeded = await seedMockAgyBrain(home, extId)

    const transcriptPath = join(seeded.brainRoot, seeded.workerConvId, '.system_generated', 'logs', 'transcript.jsonl')
    const pad = 'x'.repeat(2048)
    const lines: string[] = []
    for (let i = 0; i < 650; i += 1) {
      lines.push(JSON.stringify({
        step_index: i,
        source: 'MODEL',
        type: 'GENERIC',
        status: 'DONE',
        created_at: '2026-11-22T18:13:40Z',
        content: `step-${i}-${pad}`,
      }))
    }
    lines.push(JSON.stringify({
      step_index: 650,
      source: 'MODEL',
      type: 'PLANNER_RESPONSE',
      status: 'DONE',
      created_at: '2026-11-22T18:15:00Z',
      model: 'gemini-4-neon',
      content: 'Final tail synthesis answer.',
    }))
    await writeFile(transcriptPath, lines.join('\n') + '\n', 'utf8')

    const bridge = new AgyBrainBridge({ homeDir: home })
    const loc = bridge.resolveLocation({ externalSessionId: extId })!
    const inspected = await bridge.inspectSwarmMember('xrx-big', loc, 'alpha-worker', { limit: 5 })
    expect(inspected).toBeDefined()
    expect(inspected!.messages.at(-1)?.text).toContain('Final tail synthesis answer.')
    expect(inspected!.activity.phase).toBe('responding')
    bridge.dispose()
  })

  test('live watcher emits activated, member_updated, message_posted, overview_updated and cleans up timers/watchers', async () => {
    const tmp = await makeTempDir()
    const home = join(tmp, 'home')
    const extId = 'ext-watch-001'
    const brainRoot = join(home, '.gemini', 'antigravity-cli', 'brain')
    const sessionDir = join(brainRoot, extId)
    const swarmDir = join(sessionDir, '.system_generated', 'swarm')
    const narratorConvId = 'narr-watch-1'
    await mkdir(join(sessionDir, '.system_generated'), { recursive: true })
    await mkdir(join(brainRoot, narratorConvId, 'swarm_view'), { recursive: true })

    const events: SwarmLifecycleEvent[] = []
    const bridge = new AgyBrainBridge({ homeDir: home, debounceMs: 25, pollIntervalMs: 80 })
    bridge.watchSession({
      sessionId: 'sess-watch',
      workspaceDir: tmp,
      externalSessionId: extId,
      onSwarmEvent: event => events.push(event),
    })
    expect(bridge.watchedSessionCount()).toBe(1)

    // Create swarm files live
    await mkdir(swarmDir, { recursive: true })
    await writeFile(
      join(swarmDir, 'swarm.json'),
      JSON.stringify({ version: 1, backend: 'local', swarm_id: extId, created_at: '2026-11-22T18:20:00Z' }),
      'utf8',
    )
    await writeFile(
      join(swarmDir, 'members.json'),
      JSON.stringify({
        title: 'Live Watch Swarm',
        members: {
          coordinator: {
            conversation_id: extId,
            role: 'coordinator',
            phase: 'RUNNING',
            wake_mode: 'mentions',
            wait_since: 1,
            last_read_seq: 1,
            killed: false,
            model: 'gemini-4-neon',
          },
          narrator: {
            conversation_id: narratorConvId,
            role: 'worker',
            phase: 'RUNNING',
            wake_mode: 'mentions',
            wait_since: 1,
            last_read_seq: 1,
            killed: false,
            model: 'gemini-4-carbon',
          },
          worker1: {
            conversation_id: 'w1-conv',
            role: 'worker',
            phase: 'RUNNING',
            wake_mode: 'working',
            last_read_seq: 1,
            killed: false,
            model: 'gemini-4-neon',
          },
        },
      }),
      'utf8',
    )
    await writeFile(
      join(swarmDir, 'board.jsonl'),
      JSON.stringify({ seq: 1, agent: 'coordinator', text: '@worker1 Go!', time: '2026-11-22 21:20:00', kind: 'brief' }) + '\n',
      'utf8',
    )
    bridge.refreshSession('sess-watch')
    await waitUntil(() => events.some(e => e.kind === 'activated') && events.some(e => e.kind === 'message_posted'))

    // While worker1 is working, fallback poll timer is active
    expect(bridge.activeTimerCount()).toBeGreaterThanOrEqual(1)

    // Transition worker1 -> waiting, append seq #2, and write narrator view.json
    await writeFile(
      join(swarmDir, 'members.json'),
      JSON.stringify({
        title: 'Live Watch Swarm',
        members: {
          coordinator: {
            conversation_id: extId,
            role: 'coordinator',
            phase: 'RUNNING',
            wake_mode: 'mentions',
            wait_since: 2,
            last_read_seq: 2,
            killed: false,
            model: 'gemini-4-neon',
          },
          narrator: {
            conversation_id: narratorConvId,
            role: 'worker',
            phase: 'RUNNING',
            wake_mode: 'mentions',
            wait_since: 2,
            last_read_seq: 2,
            killed: false,
            model: 'gemini-4-carbon',
          },
          worker1: {
            conversation_id: 'w1-conv',
            role: 'worker',
            phase: 'RUNNING',
            wake_mode: 'mentions',
            wait_since: 2,
            last_read_seq: 2,
            killed: false,
            model: 'gemini-4-neon',
          },
        },
      }),
      'utf8',
    )
    await writeFile(
      join(swarmDir, 'board.jsonl'),
      [
        JSON.stringify({ seq: 1, agent: 'coordinator', text: '@worker1 Go!', time: '2026-11-22 21:20:00', kind: 'brief' }),
        JSON.stringify({ seq: 2, agent: 'worker1', text: '@coordinator [done] All set.', time: '2026-11-22 21:21:00' }),
      ].join('\n') + '\n',
      'utf8',
    )
    await writeFile(
      join(brainRoot, narratorConvId, 'swarm_view', 'view.json'),
      JSON.stringify({
        version: 1,
        as_of_seq: 2,
        updated_at: '2026-11-22T18:21:05Z',
        headline: 'Worker1 completed',
        phase: 'complete',
      }),
      'utf8',
    )
    bridge.refreshSession('sess-watch')

    await waitUntil(() =>
      events.some(e => e.kind === 'member_updated' && e.member?.name === 'worker1' && e.member.state === 'waiting') &&
      events.some(e => e.kind === 'message_posted' && e.message?.seq === 2) &&
      events.some(e => e.kind === 'overview_updated' && e.status.overview?.summary === 'Worker1 completed'),
    )

    // Once zero members are working, poll timer MUST be cleared (0 CPU idle!)
    await waitUntil(() => bridge.activeTimerCount() === 0)

    bridge.unwatchSession('sess-watch')
    expect(bridge.watchedSessionCount()).toBe(0)
    expect(bridge.activeTimerCount()).toBe(0)
    bridge.dispose()
  })

  test('falls back to plan.json overview when view.json is absent and tolerates malformed trailing board/transcript lines', async () => {
    const tmp = await makeTempDir()
    const home = join(tmp, 'home')
    const extId = 'ext-plan-fallback'
    const brainRoot = join(home, '.gemini', 'antigravity-cli', 'brain')
    const swarmDir = join(brainRoot, extId, '.system_generated', 'swarm')
    await mkdir(swarmDir, { recursive: true })
    await writeFile(
      join(swarmDir, 'members.json'),
      JSON.stringify({
        title: 'Plan Fallback Swarm',
        members: {
          coordinator: {
            conversation_id: extId,
            role: 'coordinator',
            phase: 'RUNNING',
            wake_mode: 'mentions',
            wait_since: 1,
            last_read_seq: 1,
            killed: false,
            model: 'gemini-4-neon',
          },
        },
      }),
      'utf8',
    )
    await writeFile(
      join(swarmDir, 'plan.json'),
      JSON.stringify({
        goal: { title: 'Refactor Engine', line: 'Cleanly split modules' },
        tasks: { parser: 'Rewrite parser', verifier: 'Run tests' },
        threads: [{ id: 't1', title: 'Core', goal: 'Zero regressions' }],
      }),
      'utf8',
    )
    // Include an intentionally truncated trailing NDJSON line (mid-append safety)
    await writeFile(
      join(swarmDir, 'board.jsonl'),
      JSON.stringify({ seq: 1, agent: 'coordinator', text: 'Starting plan', time: '2026-11-22 21:00:00' }) +
        '\n{"seq":2,"agent":"partial',
      'utf8',
    )

    const bridge = new AgyBrainBridge({ homeDir: home })
    const loc = bridge.resolveLocation({ externalSessionId: extId })!
    const status = await bridge.readSwarmStatus('sess-plan', loc)
    expect(status.overview).not.toBeNull()
    expect(status.overview?.author).toBe('coordinator')
    expect(status.overview?.summary).toBe('Refactor Engine')
    expect(status.overview?.markdown).toContain('**@parser**: Rewrite parser')
    expect(status.overview?.markdown).toContain('**Core**')
    const board = await bridge.readSwarmBoard(loc)
    expect(board.messages).toHaveLength(1)
    expect(board.head).toBe(1)
    bridge.dispose()
  })
})

describe('DaemonServer — Antigravity Swarm, Artifacts, Subagents, Offline Reads & Autonomous Wakeups', () => {
  test('serves swarm/artifacts/subagents live + offline, never mutates AGY swarm files, attaches artifact_id, and streams/persists out-of-turn wakeups', async () => {
    const tmp = await makeTempDir()
    const home = join(tmp, 'home')
    const workspace = join(tmp, 'workspace')
    const extId = 'agy-live-ext-42'
    await mkdir(workspace, { recursive: true })
    const seeded = await seedMockAgyBrain(home, extId)

    let activeHarness: InMemoryAcpHarness | undefined
    const receivedPrompts: string[] = []
    let currentModel = 'gemini-4-neon'

    const acpHost = new DaemonAcpBridgeHost({
      discoverAgents: () => AGY_AGENTS,
      resolveBinary: cmd => cmd,
      createAcpClient: options => {
        activeHarness = createInMemoryAcpTransport((frame, harness) => {
          const id = frame.id as number | undefined
          const method = frame.method as string | undefined
          if (method === 'initialize') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id,
              result: {
                protocolVersion: 1,
                agentInfo: { name: 'antigravity', title: 'Antigravity', version: '1.5.14' },
                agentCapabilities: { loadSession: true },
              },
            })
            return
          }
          if (method === 'session/new' || method === 'session/load') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id,
              result: {
                sessionId: extId,
                configOptions: [
                  {
                    id: 'model',
                    name: 'Model',
                    category: 'model',
                    type: 'select',
                    currentValue: currentModel,
                    options: [
                      { value: 'gemini-4-neon', name: 'Gemini 4 Neon' },
                      { value: 'gemini-4-carbon', name: 'Gemini 4 Carbon' },
                    ],
                  },
                  {
                    id: 'mode',
                    name: 'Mode',
                    category: 'mode',
                    type: 'select',
                    currentValue: 'auto',
                    options: [
                      { value: 'default', name: 'Default' },
                      { value: 'auto', name: 'Auto' },
                    ],
                  },
                ],
              },
            })
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: extId,
                update: {
                  sessionUpdate: 'available_commands_update',
                  availableCommands: [{ name: 'swarm', description: 'Run a swarm' }],
                },
              },
            })
            return
          }
          if (method === 'session/set_config_option') {
            const p = frame.params as { configId?: string; value?: string }
            if (p.configId === 'model' && p.value) currentModel = p.value
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id,
              result: {
                configOptions: [
                  {
                    id: 'model',
                    name: 'Model',
                    category: 'model',
                    type: 'select',
                    currentValue: currentModel,
                    options: [
                      { value: 'gemini-4-neon', name: 'Gemini 4 Neon' },
                      { value: 'gemini-4-carbon', name: 'Gemini 4 Carbon' },
                    ],
                  },
                ],
              },
            })
            return
          }
          if (method === 'session/prompt') {
            const p = frame.params as { prompt?: Array<{ text?: string }> }
            const promptText = p.prompt?.map(b => b.text ?? '').join('\n') ?? ''
            receivedPrompts.push(promptText)
            if (promptText.includes('write artifact now')) {
              const artifactPath = join(seeded.sessionDir, 'audit_report.md')
              harness.pushAgentFrame({
                jsonrpc: '2.0',
                method: 'session/update',
                params: {
                  sessionId: extId,
                  update: {
                    sessionUpdate: 'tool_call',
                    toolCallId: 'tc-art-1',
                    name: 'write_file',
                    status: 'in_progress',
                    rawInput: {
                      Path: artifactPath,
                      Content: '# Prismcore Audit Report\n',
                      ArtifactMetadata: { summary: 'Audit', userFacing: true },
                    },
                  },
                },
              })
              harness.pushAgentFrame({
                jsonrpc: '2.0',
                method: 'session/update',
                params: {
                  sessionId: extId,
                  update: {
                    sessionUpdate: 'tool_call_update',
                    toolCallId: 'tc-art-1',
                    name: 'write_file',
                    status: 'completed',
                    content: 'Wrote file.',
                  },
                },
              })
            }
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: extId,
                update: {
                  sessionUpdate: 'agent_message_chunk',
                  content: { type: 'text', text: 'Acknowledged.' },
                },
              },
            })
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id,
              result: { stopReason: 'end_turn' },
            })
          }
        })
        return new AcpStdioClient({
          ...options,
          transport: activeHarness!.transport,
        })
      },
    })

    const socketPath = join(tmp, 'daemon.sock')
    const runtime = new InMemoryDaemonRuntime()
    const server = new DaemonServer({
      socketPath,
      projectDirectory: workspace,
      runtime,
      acpBridgeHost: acpHost,
      agyBrainHomeDir: home,
      acpWakeupIdleMs: 60,
    })
    await server.start()
    const client = await SocketRpcClient.open(socketPath)

    try {
      const opened = await client.call('session.open', { session_key: 'main', project_dir: workspace })
      const sessionId = (opened.session as { id: string }).id

      const connected = await client.call('acp.connect', {
        session_id: sessionId,
        agent_id: 'antigravity',
        cwd: workspace,
      })
      expect(connected.ok).toBe(true)

      // 1. Swarm read RPCs served directly from AgyBrainBridge
      const status = await client.call('swarm.status', { session_id: sessionId })
      expect(status.active).toBe(true)
      expect(status.title).toBe('Prismcore Audit Swarm')
      expect((status.members as Array<{ name: string }>).length).toBe(6)

      const board = await client.call('swarm.board', { session_id: sessionId })
      expect(board.head).toBe(2)
      expect((board.messages as unknown[]).length).toBe(2)

      const inspected = await client.call('swarm.inspect', { session_id: sessionId, name: 'alpha-worker' })
      expect((inspected.member as { conversation_id?: string }).conversation_id).toBe(seeded.workerConvId)

      // 2. Host `/swarm status`, `/swarm board`, `/swarm channels` run locally while `/swarm <task>` forwards to AGY ACP
      const slashRes = await client.call('slash', { command: '/swarm status' })
      expect(slashRes.ok).toBe(true)
      expect(slashRes.forwarded_to_acp).toBeUndefined()
      expect(String(slashRes.text ?? '')).toContain('Prismcore Audit Swarm')

      const slashBoard = await client.call('slash', { command: '/swarm board' })
      expect(slashBoard.ok).toBe(true)
      expect(slashBoard.open_board).toBe(true)

      const slashChannels = await client.call('slash', { command: '/swarm channels' })
      expect(slashChannels.ok).toBe(true)
      expect(String(slashChannels.text ?? '')).toContain('#main')

      const slashActivate = await client.call('slash', { command: '/swarm audit all shaders' })
      expect(slashActivate.ok).toBe(true)
      expect(slashActivate.forwarded_to_acp).toBe(true)
      await waitUntil(() => receivedPrompts.some(p => p.includes('/swarm audit all shaders')))
      await waitUntil(() => !acpHost.hasActiveTurn(sessionId))

      // 3. Artifacts merged (`agy:audit_report.md` & `agy:interactive_chart.html`) + pin; edits are refused
      const artList = await client.call('artifacts.list')
      expect((artList.artifacts as Array<{ id: string }>).map(a => a.id).sort()).toEqual([
        'agy:audit_report.md',
        'agy:interactive_chart.html',
      ])
      const artRead = await client.call('artifacts.read', { id: 'agy:audit_report.md' })
      expect((artRead.artifact as { title: string }).title).toBe('Prismcore Audit Report')
      const artPin = await client.call('artifacts.pin', { id: 'agy:audit_report.md', pinned: true })
      expect((artPin.artifact as { pinned: boolean }).pinned).toBe(true)
      const reportBefore = await readFile(join(seeded.sessionDir, 'audit_report.md'), 'utf8')
      const artUpdate = await client.call('artifacts.update', {
        id: 'agy:audit_report.md',
        content: '# Prismcore Audit Report\n\nUpdated via RPC.\n',
      })
      expect(artUpdate.ok).toBe(false)
      expect(String(artUpdate.error)).toContain('read-only')
      expect(await readFile(join(seeded.sessionDir, 'audit_report.md'), 'utf8')).toBe(reportBefore)
      const artTraversal = await client.call('artifacts.update', {
        id: 'agy:../escape.md',
        content: 'nope',
      })
      expect(artTraversal.ok).toBe(false)

      // 4. Non-swarm subagents merged into `session.status` & `subagent.inspect` (swarm workers excluded!)
      const sessStatus = await client.call('session.status', {})
      const subRows = ((sessStatus.session as { subagent_snapshots?: Array<{ id: string }> }).subagent_snapshots ?? [])
      expect(subRows.map(r => r.id)).toEqual([seeded.subagentConvId])
      const subInspect = await client.call('subagent.inspect', {
        task: seeded.subagentConvId,
        history_limit: 10,
      })
      expect(subInspect.ok).toBe(true)
      expect((subInspect.agent as { name: string }).name).toBe('Benchmarks Analyst')
      expect(subInspect.history_session_id).toBe(seeded.subagentConvId)

      // 5. `write_file` targeting brain dir attaches `artifact_id: 'agy:audit_report.md'` onto `tool_result`
      await client.call('turn.submit', { text: 'write artifact now' })
      const artToolResult = await client.next(
        f => f.method === 'event' && f.params?.type === 'tool_result' && f.params.payload?.name === 'write_file',
      )
      expect(artToolResult.params?.payload?.artifact_id).toBe('agy:audit_report.md')
      await client.next(f => f.method === 'event' && f.params?.type === 'turn_end')

      // 6. Board write controls never write prompts for agy and never mutate AGY files on disk.
      //    Only the coordinator model switch, a real ACP set-model RPC, still works.
      const boardBefore = await readFile(join(seeded.swarmDir, 'board.jsonl'), 'utf8')
      const membersBefore = await readFile(join(seeded.swarmDir, 'members.json'), 'utf8')
      const promptsBeforeControls = receivedPrompts.length

      const postRes = await client.call('swarm.post', {
        session_id: sessionId,
        text: 'Focus on fp16 edge cases @alpha-worker',
        as_feedback: true,
      })
      expect(postRes.ok).toBe(false)
      expect(String(postRes.error)).toContain('written by Xerxes')

      const coordModelRes = await client.call('swarm.model', {
        session_id: sessionId,
        name: 'coordinator',
        model: 'gemini-4-carbon',
      })
      expect(coordModelRes.ok).toBe(true)
      expect(coordModelRes.model).toBe('gemini-4-carbon')
      expect(currentModel).toBe('gemini-4-carbon')

      const workerModelRes = await client.call('swarm.model', {
        session_id: sessionId,
        name: 'alpha-worker',
        model: 'gemini-4-carbon',
        reason: 'quota headroom',
      })
      expect(workerModelRes.ok).toBe(false)
      expect(String(workerModelRes.error)).toContain('written by Xerxes')

      const workerKillRes = await client.call('swarm.kill', {
        session_id: sessionId,
        name: 'alpha-worker',
      })
      expect(workerKillRes.ok).toBe(false)
      expect(String(workerKillRes.error)).toContain('written by Xerxes')

      await Bun.sleep(30)
      expect(receivedPrompts).toHaveLength(promptsBeforeControls)
      expect(receivedPrompts.some(p => p.includes('[Xerxes'))).toBe(false)

      expect(await readFile(join(seeded.swarmDir, 'board.jsonl'), 'utf8')).toBe(boardBefore)
      expect(await readFile(join(seeded.swarmDir, 'members.json'), 'utf8')).toBe(membersBefore)

      // 7. Out-of-turn autonomous coordinator wakeup streams as agent turn AND persists to session.messages!
      const frameCountBeforeWake = client.frames.length
      activeHarness!.pushAgentFrame({
        jsonrpc: '2.0',
        method: 'session/update',
        params: {
          sessionId: extId,
          update: {
            sessionUpdate: 'agent_thought_chunk',
            content: { type: 'text', text: 'Synthesizing completed workers...' },
          },
        },
      })
      activeHarness!.pushAgentFrame({
        jsonrpc: '2.0',
        method: 'session/update',
        params: {
          sessionId: extId,
          update: {
            sessionUpdate: 'tool_call',
            toolCallId: 'tc-wake-board',
            name: 'read_board',
            status: 'in_progress',
            rawInput: { since: 1 },
          },
        },
      })
      activeHarness!.pushAgentFrame({
        jsonrpc: '2.0',
        method: 'session/update',
        params: {
          sessionId: extId,
          update: {
            sessionUpdate: 'tool_call_update',
            toolCallId: 'tc-wake-board',
            name: 'read_board',
            status: 'completed',
            content: '[{"seq":2}]',
          },
        },
      })
      activeHarness!.pushAgentFrame({
        jsonrpc: '2.0',
        method: 'session/update',
        params: {
          sessionId: extId,
          update: {
            sessionUpdate: 'agent_message_chunk',
            content: { type: 'text', text: 'Autonomous coordinator synthesis: all audits complete.' },
          },
        },
      })

      await waitUntil(() => {
        const recent = client.frames.slice(frameCountBeforeWake)
        return (
          recent.some(f => f.method === 'event' && f.params?.type === 'turn_begin' && f.params.payload?.origin === 'agent') &&
          recent.some(f => f.method === 'event' && f.params?.type === 'think_part') &&
          recent.some(f => f.method === 'event' && f.params?.type === 'tool_call' && f.params.payload?.name === 'read_board') &&
          recent.some(f => f.method === 'event' && f.params?.type === 'tool_result' && f.params.payload?.name === 'read_board') &&
          recent.some(f => f.method === 'event' && f.params?.type === 'text_part' && String(f.params.payload?.text ?? '').includes('Autonomous coordinator synthesis')) &&
          recent.some(f => f.method === 'event' && f.params?.type === 'turn_end' && f.params.payload?.origin === 'agent')
        )
      })

      const liveSession = runtime.listSessions().find(s => s.id === sessionId)!
      expect(
        liveSession.messages.some(
          m => m.role === 'assistant' && typeof m.content === 'string' && m.content.includes('Autonomous coordinator synthesis: all audits complete.'),
        ),
      ).toBe(true)

      // 8. Offline / disconnected historical Antigravity session: reads work, writes return clean offline error & never mutate disk!
      await acpHost.disconnect(sessionId)
      // Simulate persisted `session.metadata.acp_binding` on a reopened session (without active live bridge)
      liveSession.metadata.acp_binding = {
        session_id: sessionId,
        agent_id: 'antigravity',
        name: 'Antigravity',
        provider: 'antigravity',
        transport: 'acp',
        cwd: workspace,
        external_session_id: extId,
        connected: false,
      }

      const offlineStatus = await client.call('swarm.status', { session_id: sessionId })
      expect(offlineStatus.active).toBe(true)
      expect(offlineStatus.title).toBe('Prismcore Audit Swarm')

      const offlineBoard = await client.call('swarm.board', { session_id: sessionId })
      expect(offlineBoard.head).toBe(2)

      const offlineInspect = await client.call('swarm.inspect', { session_id: sessionId, name: 'alpha-worker' })
      expect((offlineInspect.member as { name: string }).name).toBe('alpha-worker')

      const offlinePost = await client.call('swarm.post', { session_id: sessionId, text: 'hello offline' })
      expect(offlinePost).toEqual({
        ok: false,
        error: 'This Antigravity control is not available from Xerxes: it would send agy instructions written by Xerxes rather than you. Ask Antigravity directly in the chat instead.',
      })
      const offlineKill = await client.call('swarm.kill', { session_id: sessionId, name: 'alpha-worker' })
      expect(offlineKill).toEqual({
        ok: false,
        error: 'This Antigravity control is not available from Xerxes: it would send agy instructions written by Xerxes rather than you. Ask Antigravity directly in the chat instead.',
      })
      const offlineModel = await client.call('swarm.model', {
        session_id: sessionId,
        name: 'alpha-worker',
        model: 'gemini-4-carbon',
      })
      expect(offlineModel).toEqual({
        ok: false,
        error: 'Reconnect Antigravity to send swarm actions; board remains readable offline.',
      })

      expect(await readFile(join(seeded.swarmDir, 'board.jsonl'), 'utf8')).toBe(boardBefore)
      expect(await readFile(join(seeded.swarmDir, 'members.json'), 'utf8')).toBe(membersBefore)
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('auto-approves out-of-turn coordinator permission requests when Turbo mode is active', async () => {
    const tmp = await makeTempDir()
    const workspace = join(tmp, 'workspace')
    await mkdir(workspace, { recursive: true })
    const extId = 'agy-turbo-wake-01'
    let activeHarness: InMemoryAcpHarness | undefined

    const acpHost = new DaemonAcpBridgeHost({
      discoverAgents: () => AGY_AGENTS,
      resolveBinary: cmd => cmd,
      createAcpClient: options => {
        activeHarness = createInMemoryAcpTransport((frame, harness) => {
          const id = frame.id as number | undefined
          const method = frame.method as string | undefined
          if (method === 'initialize') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id,
              result: { protocolVersion: 1, agentInfo: { name: 'antigravity', version: '1.5.14' } },
            })
            return
          }
          if (method === 'session/new') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id,
              result: {
                sessionId: extId,
                configOptions: [
                  {
                    id: 'mode',
                    name: 'Mode',
                    category: 'mode',
                    type: 'select',
                    currentValue: 'default',
                    options: [
                      { value: 'default', name: 'Default' },
                      { value: 'auto', name: 'Auto' },
                    ],
                  },
                ],
              },
            })
            return
          }
          if (method === 'session/set_config_option') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id,
              result: {
                configOptions: [
                  {
                    id: 'mode',
                    name: 'Mode',
                    category: 'mode',
                    type: 'select',
                    currentValue: 'auto',
                    options: [
                      { value: 'default', name: 'Default' },
                      { value: 'auto', name: 'Auto' },
                    ],
                  },
                ],
              },
            })
          }
        })
        return new AcpStdioClient({ ...options, transport: activeHarness!.transport })
      },
    })

    const conn = await acpHost.connect({ sessionId: 's-turbo', agentId: 'antigravity', cwd: workspace })
    expect(conn.ok).toBe(true)
    const modeRes = await acpHost.setMode('s-turbo', 'turbo')
    expect(modeRes.ok).toBe(true)

    // Simulate an out-of-turn permission request from AGY's autonomous coordinator wakeup!
    activeHarness!.pushAgentFrame({
      jsonrpc: '2.0',
      id: 901,
      method: 'session/request_permission',
      params: {
        sessionId: extId,
        toolCall: { toolCallId: 'tc-wake-cmd', title: 'Run tests' },
        options: [
          { optionId: 'opt-allow', name: 'Allow once', kind: 'allow_once' },
          { optionId: 'opt-deny', name: 'Deny', kind: 'reject_once' },
        ],
      },
    })

    await waitUntil(() =>
      activeHarness!.receivedFrames.some(
        f => f.id === 901 && (f.result as { outcome?: { outcome?: string; optionId?: string } } | undefined)?.outcome?.optionId === 'opt-allow',
      ),
    )

    await acpHost.disposeAll()
  })

  test('filePreview allows safe read-only preview of allowedReadRoots while blocking traversal, symlink escapes, sensitive dirs, and writes', async () => {
    const tmp = await makeTempDir()
    const workspace = join(tmp, 'workspace')
    const home = join(tmp, 'home')
    const extId = 'eaa65661-2d98-4bac-9995-0588436d16d8'
    await mkdir(workspace, { recursive: true })
    const seeded = await seedMockAgyBrain(home, extId)

    const planPath = join(seeded.sessionDir, 'prismcore_audit_and_improvement_plan.md')
    const planContent = '# Prismcore Audit and Improvement Plan\n\nDetailed findings.\n'
    await writeFile(planPath, planContent, 'utf8')
    await writeFile(join(workspace, 'local.md'), '# Local\n', 'utf8')

    // Sensitive internal dirs & secret files inside the brain dir
    const credsDir = join(seeded.sessionDir, '.system_generated', 'credentials')
    await mkdir(credsDir, { recursive: true })
    const credFile = join(credsDir, 'oauth.json')
    await writeFile(credFile, '{"access_token":"secret"}', 'utf8')
    const envFile = join(seeded.sessionDir, '.env.local')
    await writeFile(envFile, 'API_KEY=secret\n', 'utf8')

    // Sibling unrelated brain dir & arbitrary home file
    const siblingDir = join(seeded.brainRoot, 'unrelated-sibling-999')
    await mkdir(siblingDir, { recursive: true })
    const siblingFile = join(siblingDir, 'private_notes.md')
    await writeFile(siblingFile, '# Private\n', 'utf8')
    const homeSecret = join(home, 'secret.txt')
    await writeFile(homeSecret, 'top-secret', 'utf8')

    // Symlink escapes (from workspace -> brain, from brain -> outside, dangling symlink, dir symlink -> outside)
    await symlink(planPath, join(workspace, 'ws-to-brain.md'))
    await symlink(homeSecret, join(seeded.sessionDir, 'escape-link.md'))
    await symlink(join(home, 'does-not-exist.txt'), join(seeded.sessionDir, 'dangling-escape.md'))
    await symlink(home, join(seeded.sessionDir, 'home-dir-link'))

    const allowedReadRoots = [seeded.sessionDir]
    const allKinds = { kinds: new Set(['table', 'image', 'pdf', 'sheet', 'document', 'html'] as const), sheet: 0 }

    // 1. Workspace file stays primary & editable
    const wsPreview = await previewWorkspaceFile(workspace, 'local.md', allowedReadRoots)
    expect(wsPreview).toMatchObject({ ok: true, path: 'local.md', content: '# Local\n', truncated: false })
    expect('version' in wsPreview).toBe(true)
    expect('read_only' in wsPreview).toBe(false)

    // 2. Brain artifact opens cleanly as read-only via text & rich preview
    const resolved = await resolvePreviewPath(workspace, planPath, allowedReadRoots)
    expect(resolved).toEqual({ file: planPath, local: planPath, readOnly: true })

    const textRes = await previewWorkspaceFile(workspace, planPath, allowedReadRoots)
    expect(textRes).toEqual({
      ok: true,
      path: planPath,
      content: planContent,
      truncated: false,
      read_only: true,
    })
    expect('version' in textRes).toBe(false)

    const richMdRes = await previewWorkspaceFileRich(workspace, planPath, allKinds, allowedReadRoots)
    expect(richMdRes).toEqual({
      ok: true,
      kind: 'text',
      path: planPath,
      content: planContent,
      truncated: false,
      size: Buffer.byteLength(planContent, 'utf8'),
      read_only: true,
    })
    expect('version' in richMdRes).toBe(false)

    const htmlPath = join(seeded.sessionDir, 'interactive_chart.html')
    const richHtmlRes = await previewWorkspaceFileRich(workspace, htmlPath, allKinds, allowedReadRoots)
    expect(richHtmlRes).toMatchObject({
      ok: true,
      kind: 'html',
      path: htmlPath,
      read_only: true,
      truncated: false,
    })
    expect('version' in richHtmlRes).toBe(false)

    // 3. Relative escape, workspace symlink to brain, traversal, sibling brain, arbitrary home, sensitive dirs, and symlink escapes MUST fail
    const blockedPaths = [
      'ws-to-brain.md',
      '../../etc/passwd',
      join(seeded.sessionDir, '..', '..', '..', '..', 'etc', 'passwd'),
      join(seeded.sessionDir, '..', 'unrelated-sibling-999', 'private_notes.md'),
      siblingFile,
      homeSecret,
      credFile,
      envFile,
      join(seeded.sessionDir, 'escape-link.md'),
      join(seeded.sessionDir, 'dangling-escape.md'),
      join(seeded.sessionDir, 'home-dir-link', 'secret.txt'),
      join(seeded.sessionDir, 'home-dir-link', 'nonexistent.txt'),
    ]
    for (const blocked of blockedPaths) {
      await expect(previewWorkspaceFile(workspace, blocked, allowedReadRoots)).rejects.toThrow(
        'File preview is limited to the selected workspace',
      )
      await expect(previewWorkspaceFileRich(workspace, blocked, allKinds, allowedReadRoots)).rejects.toThrow(
        'File preview is limited to the selected workspace',
      )
    }

    // 4. writeWorkspaceFile remains strictly workspace-only
    const planHash = createHash('sha256').update(planContent).digest('hex')
    await expect(writeWorkspaceFile(workspace, planPath, '# Overwritten\n', planHash)).rejects.toThrow(
      'File editing is limited to the selected workspace',
    )
    expect(await readFile(planPath, 'utf8')).toBe(planContent)
  })

  test('DaemonServer workspace.filePreview opens Antigravity brain & verified swarm member files (live + offline) while rejecting escapes, sibling brains, and fileWrite', async () => {
    const tmp = await makeTempDir()
    const home = join(tmp, 'home')
    const workspace = join(tmp, 'workspace')
    const extId = 'eaa65661-2d98-4bac-9995-0588436d16d8'
    await mkdir(workspace, { recursive: true })
    const seeded = await seedMockAgyBrain(home, extId)

    const planPath = join(seeded.sessionDir, 'prismcore_audit_and_improvement_plan.md')
    const planContent = '# Prismcore Audit & Improvement Plan\n\n- Fix affine scale clamp\n'
    await writeFile(planPath, planContent, 'utf8')

    // Verified swarm member file under location.brainRoot/<workerConvId>/
    const workerNotesPath = join(seeded.brainRoot, seeded.workerConvId, 'worker_findings.md')
    await writeFile(workerNotesPath, '# Alpha Worker Findings\n', 'utf8')

    // Unrelated sibling brain dir NOT in members.json
    const siblingDir = join(seeded.brainRoot, 'other-unrelated-session-777')
    await mkdir(siblingDir, { recursive: true })
    const siblingFile = join(siblingDir, 'other_plan.md')
    await writeFile(siblingFile, '# Other Session Plan\n', 'utf8')

    // Arbitrary home file & sensitive internal credential file
    const homeSecret = join(home, 'private_notes.txt')
    await writeFile(homeSecret, 'private home data', 'utf8')
    const credsDir = join(seeded.sessionDir, '.system_generated', 'credentials')
    await mkdir(credsDir, { recursive: true })
    const credFile = join(credsDir, 'oauth.json')
    await writeFile(credFile, '{"token":"nope"}', 'utf8')

    // Symlink in brainRoot attempting to hijack a member dir outside brainRoot
    const maliciousMemberLink = join(seeded.brainRoot, 'malicious-conv-escape')
    await symlink(home, maliciousMemberLink)

    // Inject malicious traversal + symlink member IDs into members.json to verify strict verification
    const rawMembers = JSON.parse(await readFile(join(seeded.swarmDir, 'members.json'), 'utf8')) as {
      title: string
      members: Record<string, unknown>
    }
    rawMembers.members['evil-traversal'] = {
      conversation_id: '../../etc',
      role: 'worker',
      phase: 'RUNNING',
      killed: false,
    }
    rawMembers.members['evil-symlink'] = {
      conversation_id: 'malicious-conv-escape',
      role: 'worker',
      phase: 'RUNNING',
      killed: false,
    }
    await writeFile(join(seeded.swarmDir, 'members.json'), JSON.stringify(rawMembers), 'utf8')

    let activeHarness: InMemoryAcpHarness | undefined
    const acpHost = new DaemonAcpBridgeHost({
      discoverAgents: () => AGY_AGENTS,
      resolveBinary: cmd => cmd,
      createAcpClient: options => {
        activeHarness = createInMemoryAcpTransport((frame, harness) => {
          const id = frame.id as number | undefined
          const method = frame.method as string | undefined
          if (method === 'initialize') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id,
              result: {
                protocolVersion: 1,
                agentInfo: { name: 'antigravity', title: 'Antigravity', version: '1.5.14' },
                agentCapabilities: { loadSession: true },
              },
            })
            return
          }
          if (method === 'session/new' || method === 'session/load') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id,
              result: { sessionId: extId, configOptions: [] },
            })
          }
        })
        return new AcpStdioClient({ ...options, transport: activeHarness!.transport })
      },
    })

    const socketPath = join(tmp, 'daemon-preview.sock')
    const runtime = new InMemoryDaemonRuntime()
    const server = new DaemonServer({
      socketPath,
      projectDirectory: workspace,
      runtime,
      acpBridgeHost: acpHost,
      agyBrainHomeDir: home,
    })
    await server.start()
    const client = await SocketRpcClient.open(socketPath)

    const allKinds = ['table', 'image', 'pdf', 'sheet', 'document', 'html']
    const planHash = createHash('sha256').update(planContent).digest('hex')

    const assertPreviewAllowed = async (sessionParams: JsonRpcPayload) => {
      // Text preview (`workspace.filePreview` without `kinds`)
      const textReply = await client.call('workspace.filePreview', {
        ...sessionParams,
        path: planPath,
      })
      expect(textReply).toEqual({
        ok: true,
        path: planPath,
        content: planContent,
        truncated: false,
        read_only: true,
      })
      expect('version' in textReply).toBe(false)

      // Rich preview (`workspace.filePreview` with `kinds`)
      const richReply = await client.call('workspace.filePreview', {
        ...sessionParams,
        path: planPath,
        kinds: allKinds,
      })
      expect(richReply).toEqual({
        ok: true,
        kind: 'text',
        path: planPath,
        content: planContent,
        truncated: false,
        size: Buffer.byteLength(planContent, 'utf8'),
        read_only: true,
      })
      expect('version' in richReply).toBe(false)

      // Rich HTML preview inside brain dir
      const htmlPath = join(seeded.sessionDir, 'interactive_chart.html')
      const htmlReply = await client.call('workspace.filePreview', {
        ...sessionParams,
        path: htmlPath,
        kinds: allKinds,
      })
      expect(htmlReply).toMatchObject({
        ok: true,
        kind: 'html',
        path: htmlPath,
        read_only: true,
      })
      expect('version' in htmlReply).toBe(false)

      // Verified swarm worker brain file
      const workerReply = await client.call('workspace.filePreview', {
        ...sessionParams,
        path: workerNotesPath,
        kinds: allKinds,
      })
      expect(workerReply).toMatchObject({
        ok: true,
        kind: 'text',
        path: workerNotesPath,
        content: '# Alpha Worker Findings\n',
        read_only: true,
      })

      // Artifacts list & read work seamlessly alongside filePreview
      const artList = await client.call('artifacts.list', sessionParams)
      expect((artList.artifacts as Array<{ id: string }>).map(a => a.id)).toContain(
        'agy:prismcore_audit_and_improvement_plan.md',
      )
      const artRead = await client.call('artifacts.read', {
        ...sessionParams,
        id: 'agy:prismcore_audit_and_improvement_plan.md',
      })
      expect(artRead).toMatchObject({
        ok: true,
        artifact: {
          id: 'agy:prismcore_audit_and_improvement_plan.md',
          content: planContent,
        },
      })
    }

    const assertBlockedAndReadOnly = async (sessionParams: JsonRpcPayload) => {
      for (const badPath of [
        '../../etc/passwd',
        join(seeded.sessionDir, '../../etc/passwd'),
        siblingFile,
        homeSecret,
        credFile,
        join(maliciousMemberLink, 'private_notes.txt'),
      ]) {
        await expect(
          client.call('workspace.filePreview', { ...sessionParams, path: badPath }),
        ).rejects.toThrow('File preview is limited to the selected workspace')
        await expect(
          client.call('workspace.filePreview', { ...sessionParams, path: badPath, kinds: allKinds }),
        ).rejects.toThrow('File preview is limited to the selected workspace')
      }

      // workspace.fileWrite to brain dir MUST remain rejected
      await expect(
        client.call('workspace.fileWrite', {
          ...sessionParams,
          path: planPath,
          content: '# Hacked\n',
          base_version: planHash,
        }),
      ).rejects.toThrow('File editing is limited to the selected workspace')
      expect(await readFile(planPath, 'utf8')).toBe(planContent)
    }

    try {
      const opened = await client.call('session.open', { session_key: 'agy-preview', project_dir: workspace })
      const sessionId = (opened.session as { id: string }).id

      // Before Antigravity binding, brain path MUST be rejected
      await expect(
        client.call('workspace.filePreview', { session_key: 'agy-preview', path: planPath }),
      ).rejects.toThrow('File preview is limited to the selected workspace')

      // Connect live Antigravity ACP binding
      const connected = await client.call('acp.connect', {
        session_id: sessionId,
        agent_id: 'antigravity',
        cwd: workspace,
      })
      expect(connected.ok).toBe(true)

      // 1. Live Antigravity session: allowed brain files + swarm member files + artifacts + blocked escapes + blocked write
      await assertPreviewAllowed({ session_key: 'agy-preview' })
      await assertBlockedAndReadOnly({ session_key: 'agy-preview' })

      // Also verify `/file <brainPath>` slash command works
      const slashFile = await client.call('slash', { command: `/file ${planPath}` })
      expect(slashFile.ok).toBe(true)
      expect(String(slashFile.output ?? '')).toContain('Fix affine scale clamp')

      // 2. Saved / offline Antigravity session (disconnected live ACP bridge, persisted metadata.acp_binding)
      await acpHost.disconnect(sessionId)
      const liveSession = runtime.listSessions().find(s => s.id === sessionId)!
      liveSession.metadata.acp_binding = {
        session_id: sessionId,
        agent_id: 'antigravity',
        name: 'Antigravity',
        provider: 'antigravity',
        transport: 'acp',
        cwd: workspace,
        external_session_id: extId,
        connected: false,
      }

      await assertPreviewAllowed({ session_key: 'agy-preview' })
      await assertPreviewAllowed({ session_id: sessionId })
      await assertBlockedAndReadOnly({ session_key: 'agy-preview' })

      // 3. Separate native session without Antigravity binding MUST NOT access another session's brain dir
      await client.call('session.open', { session_key: 'native-only', project_dir: workspace })
      await expect(
        client.call('workspace.filePreview', { session_key: 'native-only', path: planPath }),
      ).rejects.toThrow('File preview is limited to the selected workspace')
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('projects read-only brain images as validated data URLs, rejects spoofed/escaping files, and serves remote brain PNGs via artifacts.read', async () => {
    const root = await makeTempDir('xrx-agy-img-')
    const homeDir = join(root, 'home')
    const extId = 'img-conv-001'
    const sessionDir = join(homeDir, '.gemini', 'antigravity-cli', 'brain', extId)
    await mkdir(sessionDir, { recursive: true })

    // 1x1 transparent PNG
    const pngBytes = Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=',
      'base64',
    )
    const svgText = '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><rect width="8" height="8"/></svg>'
    await writeFile(join(sessionDir, 'hero_render.png'), pngBytes)
    await writeFile(join(sessionDir, 'diagram.svg'), svgText, 'utf8')
    await writeFile(join(sessionDir, 'spoofed.png'), '<html><body>not a png</body></html>', 'utf8')
    await writeFile(join(sessionDir, 'notes.txt'), 'secret text', 'utf8')
    const outsidePng = join(root, 'outside.png')
    await writeFile(outsidePng, pngBytes)
    await symlink(outsidePng, join(sessionDir, 'escape.png'))
    await symlink(join(sessionDir, 'notes.txt'), join(sessionDir, 'text_alias.png'))

    const bridge = new AgyBrainBridge({ homeDir })
    const location = resolveAgyBrainLocation({ externalSessionId: extId, homeDir })!
    expect(location).toBeDefined()

    const listed = await bridge.listArtifacts('sess-img', location)
    expect(listed.map(item => item.id).sort()).toEqual(['agy:diagram.svg', 'agy:hero_render.png'])
    expect(listed.every(item => item.kind === 'image')).toBe(true)

    const pngArtifact = await bridge.readArtifact('sess-img', location, 'agy:hero_render.png')
    expect(pngArtifact?.kind).toBe('image')
    expect(pngArtifact?.title).toBe('Hero Render')
    expect(pngArtifact?.content).toBe(`data:image/png;base64,${pngBytes.toString('base64')}`)

    const svgArtifact = await bridge.readArtifact('sess-img', location, 'agy:diagram.svg')
    expect(svgArtifact?.kind).toBe('image')
    expect(svgArtifact?.content).toStartWith('data:image/svg+xml;base64,')

    // Spoofed bytes, symlink escaping brain dir, and symlink to non-image file MUST all be rejected.
    expect(await bridge.readArtifact('sess-img', location, 'agy:spoofed.png')).toBeUndefined()
    expect(await bridge.readArtifact('sess-img', location, 'agy:escape.png')).toBeUndefined()
    expect(await bridge.readArtifact('sess-img', location, 'agy:text_alias.png')).toBeUndefined()

    // Remote SSH chat image reader resolves Antigravity brain PNGs read-only via artifacts.read.
    const calls: Array<{ method: string; params: Record<string, unknown> }> = []
    const remoteResult = await readRemoteChatImage(
      {
        path: `file://${join(sessionDir, 'hero_render.png')}`,
        sessionKey: 'sess-img',
      },
      async (method, params) => {
        calls.push({ method, params })
        if (method === 'artifacts.read' && params.id === 'agy:hero_render.png') {
          return { ok: true, artifact: pngArtifact as unknown as Record<string, unknown> }
        }
        return { ok: false, error: 'not found' }
      },
    )
    expect(remoteResult).toEqual({
      ok: true,
      url: `data:image/png;base64,${pngBytes.toString('base64')}`,
    })
    expect(calls).toEqual([{ method: 'artifacts.read', params: { id: 'agy:hero_render.png', session_key: 'sess-img' } }])

    // Non-brain remote PNGs still stay blocked by workspace.filePreview text-only policy.
    const workspacePng = await readRemoteChatImage(
      { path: '/workspace/photo.png', sessionKey: 'sess-img' },
      async () => ({ ok: false }),
    )
    expect(workspacePng.ok).toBe(false)
  })

  test('DaemonAcpBridgeHost drives native agy acp --login without copying credentials (start, code submit, error, cancel)', async () => {
    const writtenLines: string[] = []
    let killed = false
    let resolveExit!: (code: number | null) => void
    let spawnedArgs: readonly string[] = []
    let spawnedEnv: Readonly<Record<string, string | undefined>> = {}

    const host = new DaemonAcpBridgeHost({
      env: {
        HOME: '/tmp/agy-login-home',
        SSH_CONNECTION: '10.0.0.1 55555 10.0.0.2 22',
        DISPLAY: ':0',
        XERXES_DAEMON_BUILD_ID: 'secret-daemon',
      },
      discoverAgents: () => [
        {
          id: 'antigravity',
          name: 'Antigravity',
          provider: 'antigravity',
          transport: 'acp',
          available: true,
          command: '/usr/local/bin/agy',
          binaryPath: '/usr/local/bin/agy',
          args: ['acp'],
          description: 'Antigravity CLI',
        },
      ],
      spawnLoginProcess: ({ args, env }) => {
        spawnedArgs = args
        spawnedEnv = env
        let outCtrl!: ReadableStreamDefaultController<Uint8Array>
        const stdout = new ReadableStream<Uint8Array>({
          start(controller) {
            outCtrl = controller
            controller.enqueue(
              encoder.encode(
                'Open this link in a browser to log in with Google:\n\n'
                + '  https://accounts.google.com/o/oauth2/auth?client_id=test&redirect_uri=https%3A%2F%2Fantigravity.google%2Foauth-callback&state=abc\n\n'
                + 'Then paste the authorization code, or the full URL of the page you land on, and press Enter:\n',
              ),
            )
          },
        })
        const stderr = new ReadableStream<Uint8Array>({ start(controller) { controller.close() } })
        const exited = new Promise<number | null>(resolve => {
          resolveExit = resolve
        })
        return {
          stdout,
          stderr,
          exited,
          writeLine: line => {
            writtenLines.push(line)
            try { outCtrl.close() } catch { /* ignore */ }
            resolveExit(0)
          },
          kill: () => {
            killed = true
            try { outCtrl.close() } catch { /* ignore */ }
            resolveExit(143)
          },
        }
      },
    })

    const started = await host.startAgyLogin({ cwd: '/tmp' })
    expect(started.ok).toBe(true)
    expect(started.phase).toBe('awaiting_code')
    expect(started.needs_code).toBe(true)
    expect(started.authorization_url).toStartWith('https://accounts.google.com/o/oauth2/auth?')
    expect(spawnedArgs).toEqual(['acp', '--login'])
    expect(spawnedEnv.DISPLAY).toBeUndefined()
    expect(spawnedEnv.XERXES_DAEMON_BUILD_ID).toBeUndefined()

    // Rejects blank / newline-injected codes cleanly without writing to stdin
    const badCode = await host.submitAgyLoginCode(started.login_id, 'abc\ndef')
    expect(badCode.ok).toBe(false)
    expect(writtenLines).toEqual([])

    const completed = await host.submitAgyLoginCode(started.login_id, '4/0A-test-auth-code')
    expect(completed.ok).toBe(true)
    expect(completed.phase).toBe('done')
    expect(writtenLines).toEqual(['4/0A-test-auth-code'])

    // Cancel path kills an active login child process cleanly
    killed = false
    const second = await host.startAgyLogin({ cwd: '/tmp' })
    expect(second.phase).toBe('awaiting_code')
    const cancelled = host.cancelAgyLogin(second.login_id)
    expect(cancelled.phase).toBe('cancelled')
    expect(killed).toBe(true)
  })

  test('Antigravity update_steps variants, update_focus live status, and brain transcript fallback work end-to-end', async () => {
    const updateSteps = makeAntigravityUpdateStepsPlans()
    expect(
      updateSteps({
        type: 'tool_call',
        sessionId: 's1',
        toolCallId: 'tc-str',
        name: 'Running update_steps',
        title: 'Running update_steps',
        kind: 'other',
        rawInput: {
          steps: JSON.stringify([
            { title: 'Inspect codebase', status: 'done' },
            { title: 'Skip old path', status: 'skipped' },
            { title: 'Patch bridge', status: 'in_progress' },
            { title: 'Ship release', status: 'pending' },
          ]),
        },
      }),
    ).toEqual({
      type: 'plan',
      sessionId: 's1',
      entries: [
        { content: 'Inspect codebase', priority: 'medium', status: 'completed' },
        { content: 'Skip old path', priority: 'medium', status: 'completed' },
        { content: 'Patch bridge', priority: 'medium', status: 'in_progress' },
        { content: 'Ship release', priority: 'medium', status: 'pending' },
      ],
    })
    expect(
      updateSteps({
        type: 'tool_call_update',
        sessionId: 's1',
        toolCallId: 'tc-str',
        status: 'completed',
      }),
    ).toBeUndefined()

    const updateFocus = makeAntigravityUpdateFocus()
    expect(
      updateFocus({
        type: 'tool_call',
        sessionId: 's1',
        toolCallId: 'tc-focus',
        name: 'Run update_focus',
        title: 'Run update_focus',
        kind: 'other',
        rawInput: { focus: 'Inspecting remote SSH readiness' },
      }),
    ).toEqual({
      type: 'focus',
      sessionId: 's1',
      focus: 'Inspecting remote SSH readiness',
    })
    expect(
      updateFocus({
        type: 'tool_call_update',
        sessionId: 's1',
        toolCallId: 'tc-focus',
        status: 'completed',
      }),
    ).toBeUndefined()

    // Brain transcript read-only fallback extracts the latest update_steps call
    const tmp = await makeTempDir()
    const home = join(tmp, 'home')
    const extId = 'a1b2c3d4-1111-2222-3333-444455556666'
    const seeded = await seedMockAgyBrain(home, extId)
    const logsDir = join(seeded.sessionDir, '.system_generated', 'logs')
    await mkdir(logsDir, { recursive: true })
    await writeFile(
      join(logsDir, 'transcript.jsonl'),
      [
        JSON.stringify({
          step_index: 1,
          source: 'MODEL',
          type: 'PLANNER_RESPONSE',
          status: 'DONE',
          tool_calls: [
            {
              name: 'update_steps',
              args: {
                steps: JSON.stringify([
                  { title: 'Old step', status: 'in_progress' },
                ]),
              },
            },
          ],
        }),
        JSON.stringify({
          step_index: 5,
          source: 'MODEL',
          type: 'PLANNER_RESPONSE',
          status: 'DONE',
          tool_calls: [
            {
              name: 'update_steps',
              args: {
                steps: JSON.stringify([
                  { title: 'Inspect codebase', status: 'done' },
                  { title: 'Wire steps UI', status: 'in_progress' },
                  { title: 'Verify build', status: 'pending' },
                ]),
              },
            },
          ],
        }),
      ].join('\n') + '\n',
      'utf8',
    )
    const bridge = new AgyBrainBridge({ homeDir: home })
    const location = bridge.resolveLocation({ externalSessionId: extId })
    expect(location).toBeDefined()
    expect(bridge.readLatestUpdateStepsSync(location!)).toEqual([
      { id: 'plan-1', content: 'Inspect codebase', status: 'completed' },
      { id: 'plan-2', content: 'Wire steps UI', status: 'in_progress' },
      { id: 'plan-3', content: 'Verify build', status: 'pending' },
    ])
  })
})
