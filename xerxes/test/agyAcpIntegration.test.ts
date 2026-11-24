// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { existsSync } from 'node:fs'
import { mkdir, mkdtemp, realpath, rm, writeFile } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  AcpStdioClient,
  normalizeAcpSessionUpdate,
  parseElicitationRequest,
  type AcpClientEvent,
  type AcpSpawnedProcess,
} from '../src/acp/client.js'
import {
  bridgeEventFromAcpClientEvent,
  buildElicitationResponseFromAnswers,
  elicitationToQuestionItems,
  formatElicitationAnswersSummary,
  type AcpExternalAgentInfo,
  type AcpToolMetaSnapshot,
} from '../src/acp/externalAgents.js'
import { resolveCommand } from '../src/bridge/commands.js'
import { augmentModesWithHostTurbo, DaemonAcpBridgeHost } from '../src/daemon/acpBridgeHost.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import type { LlmClient, LlmDelta } from '../src/llms/client.js'
import type { JsonRpcPayload } from '../src/protocol/jsonRpc.js'
import { adaptDaemonEvent, sessionInfoFromInit } from '../src/ui/gatewayAdapter.js'
import { GatewayClient } from '../src/ui/gatewayClient.js'

const encoder = new TextEncoder()

function extractSpecPromptText(params: Record<string, unknown> | undefined): string {
  if (!params) return ''
  if (typeof params.text === 'string') return params.text
  if (Array.isArray(params.prompt)) {
    return params.prompt
      .map(block => {
        if (!block || typeof block !== 'object') return ''
        const rec = block as Record<string, unknown>
        if (rec.type === 'text' && typeof rec.text === 'string') return rec.text
        return ''
      })
      .filter(Boolean)
      .join('\n')
  }
  return ''
}

// Minimal valid 1x1 transparent PNG (68 bytes) for deterministic multimodal tests.
const SAMPLE_PNG_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='

interface InMemoryAcpHarness {
  readonly killed: () => boolean
  readonly pushAgentFrame: (frame: unknown) => void
  readonly pushStderr: (text: string) => void
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
  let killed = false

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
    try {
      stdoutController?.close()
    } catch {
      // Already closed.
    }
    try {
      stderrController?.close()
    } catch {
      // Already closed.
    }
    resolveExit?.(code)
  }

  const pushAgentFrame = (frame: unknown) => {
    if (exited) return
    stdoutController?.enqueue(encoder.encode(`${JSON.stringify(frame)}\n`))
  }

  const pushStderr = (text: string) => {
    if (exited) return
    stderrController?.enqueue(encoder.encode(text))
  }

  const harness: InMemoryAcpHarness = {
    receivedFrames,
    killed: () => killed,
    pushAgentFrame,
    pushStderr,
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
      kill: () => {
        killed = true
        terminate(0)
      },
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

const AGY_AGENT_ROSTER: readonly AcpExternalAgentInfo[] = [
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

function makeAgyConfigOptions(model: string, thoughtLevel: string, mode: string) {
  const levelsByModel: Record<string, Array<{ value: string; name: string }>> = {
    'gemini-4-carbon': [
      { value: 'low', name: 'Low' },
      { value: 'medium', name: 'Medium' },
      { value: 'high', name: 'High' },
    ],
    'gemini-4-neon': [
      { value: 'low', name: 'Low' },
      { value: 'medium', name: 'Medium' },
      { value: 'high', name: 'High' },
      { value: 'xhigh', name: 'Extra' },
    ],
    'gemini-3.1-pro': [
      { value: 'low', name: 'Low' },
      { value: 'high', name: 'High' },
    ],
  }
  const thoughtChoices = levelsByModel[model] ?? levelsByModel['gemini-4-carbon']!
  const effectiveThought = thoughtChoices.some(c => c.value === thoughtLevel)
    ? thoughtLevel
    : thoughtChoices[0]!.value

  return [
    {
      id: 'model',
      name: 'Model',
      category: 'model',
      type: 'select',
      currentValue: model,
      options: [
        { value: 'gemini-4-carbon', name: 'Gemini 4 Carbon' },
        { value: 'gemini-4-neon', name: 'Gemini 4 Neon' },
        { value: 'gemini-3.1-pro', name: 'Gemini 3.1 Pro' },
      ],
    },
    {
      id: 'thought_level',
      name: 'Thinking Level',
      category: 'thought_level',
      type: 'select',
      currentValue: effectiveThought,
      options: thoughtChoices,
    },
    {
      id: 'mode',
      name: 'Mode',
      category: 'mode',
      type: 'select',
      currentValue: mode,
      options: [
        { value: 'default', name: 'Default' },
        { value: 'accept-edits', name: 'Accept Edits' },
        { value: 'auto', name: 'Auto' },
        { value: 'plan', name: 'Plan' },
      ],
    },
  ]
}

function makeUnusedNativeLlm(): LlmClient {
  return {
    async complete() {
      throw new Error('Native LLM should not be invoked when external ACP bridge is active')
    },
    async *stream(): AsyncIterable<LlmDelta> {
      throw new Error('Native LLM should not be invoked when external ACP bridge is active')
    },
  }
}

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

async function makeTempDir(prefix = 'xrx-agy-'): Promise<string> {
  const baseTmp = process.platform === 'win32' ? tmpdir() : (existsSync('/tmp') ? '/tmp' : tmpdir())
  const dir = await realpath(await mkdtemp(join(baseTmp, prefix)))
  tempRoots.push(dir)
  process.env.XERXES_HOME = join(dir, 'home')
  await mkdir(process.env.XERXES_HOME, { recursive: true })
  return dir
}

describe('Antigravity ACP client & event normalization (Fixes 3, 6, 9, 10, 11, 13)', () => {
  test('normalizes usage_update, session_info_update, available_commands_update, structured diffs, and strict terminal-only tool settlement', () => {
    // available_commands_update must NEVER fall through to continuation
    const cmdEvents = normalizeAcpSessionUpdate('s1', {
      sessionUpdate: 'available_commands_update',
      availableCommands: [
        { name: 'goal', description: 'Set goal' },
        { name: 'compact', description: 'Compact context' },
      ],
    })
    expect(cmdEvents).toEqual([
      {
        type: 'available_commands',
        sessionId: 's1',
        commands: [
          { name: 'goal', description: 'Set goal' },
          { name: 'compact', description: 'Compact context' },
        ],
      },
    ])

    // usage_update
    const usageEvents = normalizeAcpSessionUpdate('s1', {
      sessionUpdate: 'usage_update',
      used: 9140,
      size: 1048576,
    })
    expect(usageEvents).toEqual([
      { type: 'usage_update', sessionId: 's1', contextUsed: 9140, contextSize: 1048576 },
    ])

    // session_info_update
    const infoEvents = normalizeAcpSessionUpdate('s1', {
      sessionUpdate: 'session_info_update',
      title: 'Bat and Ball Riddle Solution',
    })
    expect(infoEvents).toEqual([
      { type: 'session_info', sessionId: 's1', title: 'Bat and Ball Riddle Solution' },
    ])

    // user_message_chunk emits structured user_message_chunk (never assistant text_delta/continuation)
    expect(
      normalizeAcpSessionUpdate('s1', {
        sessionUpdate: 'user_message_chunk',
        content: { type: 'text', text: 'replayed prompt' },
      }),
    ).toEqual([{ type: 'user_message_chunk', sessionId: 's1', text: 'replayed prompt' }])

    // Intermediate pending/in_progress tool_call_update carrying diff content emits live tool_update (NOT premature tool_result)
    const toolNames = new Map<string, string>()
    const toolMeta = new Map<string, AcpToolMetaSnapshot>()
    const settled = new Set<string>()

    const startEvents = normalizeAcpSessionUpdate('s1', {
      sessionUpdate: 'tool_call',
      toolCallId: 'tc-edit',
      name: 'edit_file',
      kind: 'edit',
      status: 'in_progress',
      locations: [{ path: '/repo/a.ts', line: 4 }],
      rawInput: { Path: '/repo/a.ts', Target: 'old', Replacement: 'new' },
    })
    const bridgedStart = bridgeEventFromAcpClientEvent(startEvents[0]!, toolNames, toolMeta, settled)
    expect(bridgedStart).toMatchObject({
      type: 'tool_call',
      toolCallId: 'tc-edit',
      name: 'edit_file',
      locations: [{ path: '/repo/a.ts', line: 4 }],
    })

    const pendingUpdate = normalizeAcpSessionUpdate('s1', {
      sessionUpdate: 'tool_call_update',
      toolCallId: 'tc-edit',
      status: 'pending',
      content: [{ type: 'diff', path: '/repo/a.ts', oldText: 'const a = 1\n', newText: 'const a = 2\n' }],
    })
    expect(bridgeEventFromAcpClientEvent(pendingUpdate[0]!, toolNames, toolMeta, settled)).toMatchObject({
      type: 'tool_update',
      toolCallId: 'tc-edit',
      status: 'pending',
      diff: { path: '/repo/a.ts', old_text: 'const a = 1\n', new_text: 'const a = 2\n' },
    })

    const inProgressUpdate = normalizeAcpSessionUpdate('s1', {
      sessionUpdate: 'tool_call_update',
      toolCallId: 'tc-edit',
      status: 'in_progress',
      content: [{ type: 'diff', path: '/repo/a.ts', oldText: 'const a = 1\n', newText: 'const a = 2\n' }],
    })
    expect(bridgeEventFromAcpClientEvent(inProgressUpdate[0]!, toolNames, toolMeta, settled)).toMatchObject({
      type: 'tool_update',
      toolCallId: 'tc-edit',
      status: 'in_progress',
      diff: { path: '/repo/a.ts', old_text: 'const a = 1\n', new_text: 'const a = 2\n' },
    })

    // Terminal completed update emits exactly ONE tool_result with full structured diff preserved (not flattened!)
    const completedUpdate = normalizeAcpSessionUpdate('s1', {
      sessionUpdate: 'tool_call_update',
      toolCallId: 'tc-edit',
      status: 'completed',
      content: [{ type: 'diff', path: '/repo/a.ts', oldText: 'const a = 1\n', newText: 'const a = 2\n' }],
    })
    expect(completedUpdate[0]).toMatchObject({
      type: 'tool_call_update',
      diff: { path: '/repo/a.ts', oldText: 'const a = 1\n', newText: 'const a = 2\n' },
    })
    const bridgedDone = bridgeEventFromAcpClientEvent(completedUpdate[0]!, toolNames, toolMeta, settled)
    expect(bridgedDone).toMatchObject({
      type: 'tool_result',
      toolCallId: 'tc-edit',
      name: 'edit_file',
      diff: { path: '/repo/a.ts', old_text: 'const a = 1\n', new_text: 'const a = 2\n' },
      locations: [{ path: '/repo/a.ts', line: 4 }],
    })
    // Duplicate completed update for same toolCallId is deduplicated
    expect(bridgeEventFromAcpClientEvent(completedUpdate[0]!, toolNames, toolMeta, settled)).toBeUndefined()
  })

  test('advertises honest terminal: false, handles cancel grace cleanly, calls session/close on disconnect, and surfaces actionable auth error', async () => {
    // 1. Unauthenticated agy acp -> actionable login message
    const unauthHost = new DaemonAcpBridgeHost({
      discoverAgents: () => AGY_AGENT_ROSTER,
      createAcpClient: opts => {
        const h = createInMemoryAcpTransport((frame, harness) => {
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                protocolVersion: 1,
                agentInfo: { name: 'Antigravity ACP Server', version: '1.5.14' },
                agentCapabilities: {
                  promptCapabilities: { image: true, audio: false, embeddedContext: true },
                  sessionCapabilities: { resume: {}, close: {}, load: {}, list: {} },
                },
                authMethods: [{ id: 'oauth-personal', name: 'Log in with Google' }],
              },
            })
            return
          }
          if (frame.method === 'session/new') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              error: { code: -32000, message: 'Authentication required' },
            })
          }
        })
        return new AcpStdioClient({ ...opts, transport: h.transport })
      },
    })
    const authFail = await unauthHost.connect({
      sessionId: 'unauth-1',
      agentId: 'antigravity',
      cwd: '/tmp',
    })
    expect(authFail.ok).toBe(false)
    if (!authFail.ok) {
      expect(authFail.error).toBe(
        'Antigravity is not signed in. Run `agy acp --login` (or `agy`) in your terminal to sign in with Google, then click Retry.',
      )
      expect(authFail.install_hint).toBeUndefined()
    }

    // 2. Honest clientCapabilities, clean cancel grace settling with stopReason: "cancelled", and session/close on disconnect
    let harnessRef: InMemoryAcpHarness | undefined
    const host = new DaemonAcpBridgeHost({
      discoverAgents: () => AGY_AGENT_ROSTER,
      createAcpClient: opts => {
        let promptRpcId: unknown
        harnessRef = createInMemoryAcpTransport((frame, h) => {
          if (frame.method === 'initialize') {
            h.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                protocolVersion: 1,
                agentCapabilities: {
                  promptCapabilities: { image: true, embeddedContext: true },
                  sessionCapabilities: { resume: {}, close: {} },
                },
              },
            })
            return
          }
          if (frame.method === 'session/new') {
            h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { sessionId: 'agy-clean-1' } })
            return
          }
          if (frame.method === 'session/prompt') {
            promptRpcId = frame.id
            return
          }
          if (frame.method === 'session/cancel') {
            // Real agy acp settles in-flight session/prompt cleanly within ~20ms
            setTimeout(() => {
              h.pushAgentFrame({
                jsonrpc: '2.0',
                id: promptRpcId,
                result: { stopReason: 'cancelled' },
              })
            }, 15)
            return
          }
          if (frame.method === 'session/close') {
            h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: {} })
          }
        })
        return new AcpStdioClient({ ...opts, transport: harnessRef.transport })
      },
    })

    const conn = await host.connect({ sessionId: 'clean-1', agentId: 'antigravity', cwd: '/tmp' })
    expect(conn.ok).toBe(true)
    expect(harnessRef!.receivedFrames[0]).toMatchObject({
      method: 'initialize',
      params: {
        protocolVersion: 2,
        clientInfo: { name: 'xerxes-agents', title: 'Xerxes Agents' },
        clientCapabilities: {
          fs: { readTextFile: true, writeTextFile: false },
          terminal: false,
          session: { configOptions: { boolean: {} }, notices: {}, compaction: {} },
          elicitation: { form: {}, url: {} },
        },
      },
    })

    const ac = new AbortController()
    const run = host.runPrompt({
      sessionId: 'clean-1',
      text: 'Long task',
      signal: ac.signal,
      onEvent: () => undefined,
    })
    await Bun.sleep(15)
    ac.abort()
    const promptRes = await run
    expect(promptRes.cancelled).toBe(true)
    expect(promptRes.stopReason).toBe('cancelled')

    await host.disconnect('clean-1')
    expect(harnessRef!.receivedFrames.some(f => f.method === 'session/close')).toBe(true)
    expect(harnessRef!.killed()).toBe(true)
  })
})

describe('End-to-end DaemonServer Antigravity ACP integration (Fixes 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14)', () => {
  test('supports providerless turns, rich approvals respecting AGY mode, multimodal images & @files, mid-turn steer, dynamic thought_level, usage_update, session_info title, plan todos, persistence, session/resume, unexpected crash, cancel_all, completions, and /btw hint', async () => {
    const root = await makeTempDir('xrx-agy-e2e-')
    await writeFile(join(root, 'notes.ts'), 'export const answer = 42\n', 'utf8')

    let activeModel = 'gemini-4-carbon'
    let activeThought = 'medium'
    let activeMode = 'default'
    let spawnCount = 0
    const harnesses: InMemoryAcpHarness[] = []
    const receivedPrompts: Array<Record<string, unknown>> = []
    const receivedMethods: string[] = []
    let releaseSlowTurn: (() => void) | undefined

    const bridgeHost = new DaemonAcpBridgeHost({
      discoverAgents: () => AGY_AGENT_ROSTER,
      createAcpClient: opts => {
        spawnCount += 1
        let pendingPromptId: unknown
        let slowTurnActive = false
        let permCount = 0
        let newSessionCount = 0
        const h = createInMemoryAcpTransport((frame, harness) => {
          if (typeof frame.method === 'string') {
            receivedMethods.push(frame.method)
          }
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                protocolVersion: 1,
                agentInfo: { name: 'Antigravity ACP Server', version: '1.5.14' },
                agentCapabilities: {
                  loadSession: true,
                  promptCapabilities: { image: true, audio: false, embeddedContext: true },
                  sessionCapabilities: { resume: {}, close: {}, load: {}, list: {} },
                },
                authMethods: [{ id: 'oauth-personal', name: 'Log in with Google' }],
              },
            })
            return
          }
          if (frame.method === 'session/new' || frame.method === 'session/resume') {
            newSessionCount += 1
            const sid = frame.method === 'session/resume'
              ? String((frame.params as Record<string, unknown>).sessionId)
              : newSessionCount === 1
                ? `agy-ext-${spawnCount}`
                : `agy-ext-${spawnCount}-side-${newSessionCount}`
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: sid,
                update: {
                  sessionUpdate: 'available_commands_update',
                  availableCommands: [
                    { name: 'goal', description: 'Long-running goal' },
                    { name: 'compact', description: 'Compact AGY context' },
                  ],
                },
              },
            })
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: sid,
                update: { sessionUpdate: 'usage_update', used: 4096, size: 1048576 },
              },
            })
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                sessionId: sid,
                configOptions: makeAgyConfigOptions(activeModel, activeThought, activeMode),
              },
            })
            return
          }
          if (frame.method === 'session/set_config_option') {
            const p = frame.params as Record<string, unknown>
            if (p.configId === 'model' && typeof p.value === 'string') {
              activeModel = p.value
            } else if (p.configId === 'thought_level' && typeof p.value === 'string') {
              activeThought = p.value
            } else if (p.configId === 'mode' && typeof p.value === 'string') {
              activeMode = p.value
            }
            const nextCfg = makeAgyConfigOptions(activeModel, activeThought, activeMode)
            const thoughtEntry = nextCfg.find(item => item.id === 'thought_level')
            if (thoughtEntry) activeThought = thoughtEntry.currentValue
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: { configOptions: nextCfg },
            })
            return
          }
          if (frame.method === 'session/close') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: {} })
            return
          }
          if (frame.method === 'session/prompt') {
            const params = frame.params as Record<string, unknown>
            if (slowTurnActive) {
              // Simulate an agent rejecting concurrent mid-turn session/prompt so fallback queued_steer is exercised.
              harness.pushAgentFrame({
                jsonrpc: '2.0',
                id: frame.id,
                error: { code: -32002, message: 'session prompt already in progress' },
              })
              return
            }
            pendingPromptId = frame.id
            receivedPrompts.push(params)
            const sid = String(params.sessionId)
            const text = extractSpecPromptText(params)

            if (text.includes('Quick question?')) {
              harness.pushAgentFrame({
                jsonrpc: '2.0',
                method: 'session/update',
                params: {
                  sessionId: sid,
                  update: {
                    sessionUpdate: 'agent_message_chunk',
                    content: { type: 'text', text: 'Prelude, loop, and coda answered on Antigravity.' },
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

            if (text.includes('SLOW_STEER_TURN')) {
              slowTurnActive = true
              const slowPromptId = frame.id
              harness.pushAgentFrame({
                jsonrpc: '2.0',
                method: 'session/update',
                params: {
                  sessionId: sid,
                  update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'Working on slow turn.' } },
                },
              })
              releaseSlowTurn = () => {
                slowTurnActive = false
                harness.pushAgentFrame({
                  jsonrpc: '2.0',
                  id: slowPromptId,
                  result: { stopReason: 'end_turn' },
                })
              }
              return
            }

            if (text.includes('HANG_FOR_CANCEL_ALL')) {
              return
            }

            // The steer reaches agy exactly as typed, with no Xerxes label around it.
            if (text === 'Also check edge cases') {
              harness.pushAgentFrame({
                jsonrpc: '2.0',
                method: 'session/update',
                params: {
                  sessionId: sid,
                  update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'Applied follow-up steer.' } },
                },
              })
              harness.pushAgentFrame({
                jsonrpc: '2.0',
                id: pendingPromptId,
                result: { stopReason: 'end_turn' },
              })
              return
            }

            // Full rich AGY turn: title, thinking, plan, edit_file permission with diff + only allow_once/reject_once,
            // intermediate pending update, completed edit_file with diff, failed run_command (runtime error, NOT permission denied!),
            // usage_update + prompt result usage!
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: sid,
                update: { sessionUpdate: 'session_info_update', title: 'Notes Refactor Plan' },
              },
            })
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: sid,
                update: { sessionUpdate: 'agent_thought_chunk', content: { type: 'text', text: 'Inspecting notes.ts carefully.' } },
              },
            })
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: sid,
                update: {
                  sessionUpdate: 'plan',
                  entries: [
                    { content: 'Edit notes.ts', status: 'completed', priority: 'high' },
                    { content: 'Verify build', status: 'in_progress', priority: 'medium' },
                  ],
                },
              },
            })
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: sid,
                update: {
                  sessionUpdate: 'tool_call',
                  toolCallId: 'tc-edit-1',
                  name: 'edit_file',
                  kind: 'edit',
                  status: 'in_progress',
                  locations: [{ path: join(root, 'notes.ts'), line: 1 }],
                  rawInput: { Path: join(root, 'notes.ts'), Target: '42', Replacement: '99' },
                },
              },
            })
            // Intermediate pending update with diff must NOT emit a premature tool_result
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: sid,
                update: {
                  sessionUpdate: 'tool_call_update',
                  toolCallId: 'tc-edit-1',
                  status: 'pending',
                  content: [{
                    type: 'diff',
                    path: join(root, 'notes.ts'),
                    oldText: 'export const answer = 42\n',
                    newText: 'export const answer = 99\n',
                  }],
                },
              },
            })
            permCount += 1
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: `perm-${permCount}`,
              method: 'session/request_permission',
              params: {
                sessionId: sid,
                toolCall: {
                  toolCallId: 'tc-edit-1',
                  name: 'edit_file',
                  title: 'Edit notes.ts',
                  kind: 'edit',
                  locations: [{ path: join(root, 'notes.ts'), line: 1 }],
                  rawInput: { Path: join(root, 'notes.ts'), Target: '42', Replacement: '99' },
                  content: [{
                    type: 'diff',
                    path: join(root, 'notes.ts'),
                    oldText: 'export const answer = 42\n',
                    newText: 'export const answer = 99\n',
                  }],
                },
                options: [
                  { optionId: 'opt-allow-once', name: 'Allow', kind: 'allow_once' },
                  { optionId: 'opt-deny-once', name: 'Deny', kind: 'reject_once' },
                ],
              },
            })
            return
          }
          if (typeof frame.id === 'string' && frame.id.startsWith('perm-')) {
            const sid = `agy-ext-${spawnCount}`
            const outcome = (frame.result as Record<string, unknown> | undefined)?.outcome as Record<string, unknown> | undefined
            expect(outcome?.optionId).toBe('opt-allow-once')

            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: sid,
                update: {
                  sessionUpdate: 'tool_call_update',
                  toolCallId: 'tc-edit-1',
                  status: 'completed',
                  content: [{
                    type: 'diff',
                    path: join(root, 'notes.ts'),
                    oldText: 'export const answer = 42\n',
                    newText: 'export const answer = 99\n',
                  }],
                },
              },
            })
            // Runtime failing command (must keep permitted: true!)
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: sid,
                update: {
                  sessionUpdate: 'tool_call',
                  toolCallId: 'tc-cmd-1',
                  name: 'run_command',
                  kind: 'execute',
                  rawInput: { CommandLine: 'exit 1' },
                },
              },
            })
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: sid,
                update: {
                  sessionUpdate: 'tool_call_update',
                  toolCallId: 'tc-cmd-1',
                  status: 'failed',
                  rawOutput: { output: 'command exited 1', exit_code: 1 },
                },
              },
            })
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: sid,
                update: { sessionUpdate: 'usage_update', used: 9140, size: 1048576 },
              },
            })
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: sid,
                update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'Updated notes.ts.' } },
              },
            })
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: pendingPromptId,
              result: {
                stopReason: 'end_turn',
                usage: {
                  inputTokens: 8663,
                  outputTokens: 114,
                  cachedReadTokens: 18164,
                  cachedWriteTokens: 0,
                  thoughtTokens: 190,
                  totalTokens: 26941,
                },
              },
            })
          }
        })
        harnesses.push(h)
        return new AcpStdioClient({ ...opts, transport: h.transport })
      },
    })

    // NO native Xerxes model configured (model: '') AND permissionMode: 'accept-all' to test Fix #1 & Fix #3!
    const runtime = new InMemoryDaemonRuntime(
      new AgentTurnRunner({ llm: makeUnusedNativeLlm(), model: '' }),
      {
        currentProjectDirectory: root,
        model: '',
        permissionMode: 'accept-all',
        sessionDirectory: join(root, 'sessions'),
      },
    )
    const socketPath = join(root, 'agy-e2e.sock')
    const server = new DaemonServer({
      socketPath,
      projectDirectory: root,
      runtime,
      acpBridgeHost: bridgeHost,
      autoTitle: true,
      autoSnapshotTurns: false,
      eventJournal: { directory: join(root, 'events') },
    })
    await server.start()

    const client = await SocketRpcClient.open(socketPath)
    try {
      const init = await client.call('initialize', {
        session_key: 'agy-s1',
        project_dir: root,
      })
      const sessionId = String(init.session_id)

      // 1. Connect Antigravity -> populates config_options, thought_level, thought_levels, available_commands
      //    and defaults mode to Turbo on fresh connect.
      const conn = await client.call('acp.connect', {
        session_id: sessionId,
        agent_id: 'antigravity',
      })
      expect(conn.ok).toBe(true)
      expect(conn.binding).toMatchObject({
        agent_id: 'antigravity',
        external_session_id: 'agy-ext-1',
        current_model: 'gemini-4-carbon',
        current_mode: 'turbo',
        current_mode_name: 'Turbo',
        thought_level: 'medium',
        thought_level_config_id: 'thought_level',
        thought_levels: [
          { id: 'low', name: 'Low' },
          { id: 'medium', name: 'Medium' },
          { id: 'high', name: 'High' },
        ],
        available_commands: [
          { name: 'goal', description: 'Long-running goal' },
          { name: 'compact', description: 'Compact AGY context' },
        ],
      })

      // session.open includes live acp_binding AND initial context usage (4096 / 1048576)
      const openLive = await client.call('session.open', { session_key: 'agy-s1', project_dir: root })
      expect((openLive.session as Record<string, unknown>).acp_binding).toMatchObject({
        agent_id: 'antigravity',
        connected: true,
        thought_level: 'medium',
      })
      expect((openLive.session as Record<string, unknown>).saved_acp_binding).toBeUndefined()
      expect((openLive.session as Record<string, unknown>).context_tokens).toBe(4096)
      expect((openLive.session as Record<string, unknown>).max_context).toBe(1048576)

      // 2. Dynamic config_options & thought_levels via acp.set_config_option + acp.set_model + /acp thinking
      const setThought = await client.call('acp.set_config_option', {
        session_id: sessionId,
        config_id: 'thought_level',
        value: 'high',
      })
      expect(setThought).toMatchObject({
        ok: true,
        binding: { thought_level: 'high' },
      })

      const setNeon = await client.call('acp.set_model', {
        session_id: sessionId,
        model_id: 'gemini-4-neon',
      })
      expect(setNeon).toMatchObject({
        ok: true,
        binding: {
          current_model: 'gemini-4-neon',
          thought_levels: [
            { id: 'low', name: 'Low' },
            { id: 'medium', name: 'Medium' },
            { id: 'high', name: 'High' },
            { id: 'xhigh', name: 'Extra' },
          ],
        },
      })

      const slashThinking = await client.call('slash', { command: '/acp thinking xhigh' })
      expect(slashThinking.ok).toBe(true)
      expect(String(slashThinking.text)).toContain('Extra')

      const setPro = await client.call('acp.set_config_option', {
        session_id: sessionId,
        config_id: 'model',
        value: 'gemini-3.1-pro',
      })
      expect(setPro).toMatchObject({
        ok: true,
        binding: {
          current_model: 'gemini-3.1-pro',
          thought_levels: [
            { id: 'low', name: 'Low' },
            { id: 'high', name: 'High' },
          ],
        },
      })

      // 3. /acp tab-completions (Fix #14)
      const acpSubComplete = await client.call('complete', { text: '/acp th', session_key: 'agy-s1' })
      expect(acpSubComplete.completions).toEqual([
        expect.objectContaining({ label: 'thinking', value: '/acp thinking ' }),
      ])
      const acpThoughtComplete = await client.call('complete', { text: '/acp thinking ', session_key: 'agy-s1' })
      expect((acpThoughtComplete.completions as Array<{ label: string }>).map(c => c.label)).toEqual(['low', 'high'])
      const acpConfigValComplete = await client.call('complete', { text: '/acp config mode pl', session_key: 'agy-s1' })
      expect((acpConfigValComplete.completions as Array<{ label: string }>).map(c => c.label)).toEqual(['plan'])

      // 4. Providerless turn with image + @notes.ts + rich approval (NOT auto-approved despite Xerxes accept-all!)
      // Switch from default Turbo to Default mode first so AGY asks for edit permission.
      const switchToDefault = await client.call('acp.set_mode', {
        session_id: sessionId,
        mode_id: 'default',
      })
      expect(switchToDefault).toMatchObject({
        ok: true,
        binding: { current_mode: 'default', current_mode_name: 'Default' },
      })
      const submitRes = await client.call('turn.submit', {
        session_key: 'agy-s1',
        text: 'Inspect @notes.ts and image',
        images: [{ media_type: 'image/png', data: SAMPLE_PNG_BASE64 }],
      })
      expect(submitRes.ok).toBe(true)

      // Approval MUST be prompted even though Xerxes permissionMode is 'accept-all'
      const approvalFrame = await client.next(
        f => f.method === 'event' && f.params?.type === 'approval_request',
      )
      const approvalPayload = approvalFrame.params!.payload!
      expect(approvalPayload).toMatchObject({
        action: 'edit_file',
        cwd: root,
        allow_permanent: false,
        inputs: { Path: join(root, 'notes.ts'), Target: '42', Replacement: '99' },
        diff: {
          path: join(root, 'notes.ts'),
          old_text: 'export const answer = 42\n',
          new_text: 'export const answer = 99\n',
        },
        options: [
          { option_id: 'opt-allow-once', name: 'Allow', kind: 'allow_once' },
          { option_id: 'opt-deny-once', name: 'Deny', kind: 'reject_once' },
        ],
      })

      // Verify multimodal prompt blocks sent to AGY
      const firstPromptBlocks = receivedPrompts[0]!.prompt as Array<Record<string, unknown>>
      expect(firstPromptBlocks.some(b => b.type === 'text')).toBe(true)
      expect(firstPromptBlocks.some(b => b.type === 'resource' && String((b.resource as Record<string, unknown>).uri).endsWith('notes.ts'))).toBe(true)
      expect(firstPromptBlocks.some(b => b.type === 'image' && b.mimeType === 'image/png' && b.data === SAMPLE_PNG_BASE64)).toBe(true)

      // Reply with explicit option_id via permission_response
      const permRes = await client.call('permission_response', {
        request_id: String(approvalPayload.request_id),
        response: 'approve',
        option_id: 'opt-allow-once',
      })
      expect(permRes.ok).toBe(true)

      await client.next(f => f.method === 'event' && f.params?.type === 'turn_end')

      // Check strict single tool_result for tc-edit-1 with full structured diff + display_blocks diff
      const editResults = client.frames.filter(
        f => f.method === 'event' && f.params?.type === 'tool_result' && f.params.payload?.tool_call_id === 'tc-edit-1',
      )
      expect(editResults).toHaveLength(1)
      expect(editResults[0]!.params!.payload).toMatchObject({
        permitted: true,
        diff: {
          path: join(root, 'notes.ts'),
          old_text: 'export const answer = 42\n',
          new_text: 'export const answer = 99\n',
        },
        locations: [{ path: join(root, 'notes.ts'), line: 1 }],
      })

      // Check runtime failure on tc-cmd-1 does NOT falsely mark permitted: false
      const cmdResults = client.frames.filter(
        f => f.method === 'event' && f.params?.type === 'tool_result' && f.params.payload?.tool_call_id === 'tc-cmd-1',
      )
      expect(cmdResults).toHaveLength(1)
      expect(cmdResults[0]!.params!.payload).toMatchObject({
        permitted: true,
        error: expect.stringContaining('command exited 1'),
      })

      // Check session_info_update title broadcast
      expect(
        client.frames.some(
          f => f.method === 'event' && f.params?.type === 'session_title' && f.params.payload?.title === 'Notes Refactor Plan',
        ),
      ).toBe(true)

      // Check usage_update + prompt usage updated session totals & ContextRing status_update
      const liveSession = runtime.listSessions().find(s => s.id === sessionId)!
      expect(liveSession.totalInputTokens).toBe(8663)
      expect(liveSession.totalOutputTokens).toBe(114)
      expect(
        client.frames.some(
          f =>
            f.method === 'event' &&
            f.params?.type === 'status_update' &&
            f.params.payload?.context_tokens === 9140 &&
            f.params.payload?.max_context === 1048576,
        ),
      ).toBe(true)

      // Check persisted toolExecutions (with diff & inputs), thinkingContent, and todos
      expect(liveSession.thinkingContent).toContain('Inspecting notes.ts carefully.')
      expect(
        liveSession.toolExecutions.some(
          exec =>
            typeof exec === 'object' &&
            exec !== null &&
            (exec as Record<string, unknown>).toolCallId === 'tc-edit-1' &&
            ((exec as Record<string, unknown>).diff as Record<string, unknown> | undefined)?.new_text ===
              'export const answer = 99\n',
        ),
      ).toBe(true)
      const reopenedAfterTurn = await client.call('session.open', { session_key: 'agy-s1', project_dir: root })
      expect((reopenedAfterTurn.session as Record<string, unknown>).todos).toEqual([
        { id: 'todo-1', content: 'Edit notes.ts', status: 'completed' },
        { id: 'todo-2', content: 'Verify build', status: 'in_progress' },
      ])

      // 5. Mid-turn steer on AGY ACP queues cleanly and drains via continuation (Fix #5)
      const beforeSteerIdx = client.frames.length
      await client.call('turn.submit', { session_key: 'agy-s1', text: 'SLOW_STEER_TURN' })
      await client.next(
        f => client.frames.indexOf(f) >= beforeSteerIdx && f.method === 'event' && f.params?.type === 'turn_begin',
      )
      const steerRes = await client.call('turn.steer', {
        session_key: 'agy-s1',
        content: 'Also check edge cases',
      })
      expect(steerRes).toMatchObject({ ok: true, disposition: 'queued_steer' })
      releaseSlowTurn?.()
      await client.next(
        f =>
          client.frames.indexOf(f) >= beforeSteerIdx &&
          f.method === 'event' &&
          f.params?.type === 'text_part' &&
          f.params.payload?.text === 'Applied follow-up steer.',
      )
      await client.next(
        f =>
          client.frames.indexOf(f) >= beforeSteerIdx &&
          f.method === 'event' &&
          f.params?.type === 'turn_end' &&
          client.frames.filter(
            (cand, idx) => idx >= beforeSteerIdx && cand.method === 'event' && cand.params?.type === 'turn_end',
          ).length >= 2,
      )

      // 6. /btw routes strictly through Antigravity ACP (session/new -> session/prompt -> session/close)
      //    with zero "Xerxes" preamble and never falls back to native providers — even if a stale native model is passed.
      const promptsBeforeBtw = receivedPrompts.length
      const btwRes = await client.call('session.side_question', {
        session_id: sessionId,
        question: 'Quick question?',
        model: 'claude-haiku-4-5',
      })
      expect(btwRes).toMatchObject({
        ok: true,
        answer: 'Prelude, loop, and coda answered on Antigravity.',
        model: 'gemini-3.1-pro',
      })
      expect(receivedPrompts).toHaveLength(promptsBeforeBtw + 1)
      const btwPromptText = extractSpecPromptText(receivedPrompts[promptsBeforeBtw]!)
      expect(btwPromptText).toContain('Quick question?')
      expect(btwPromptText).not.toContain('Xerxes')
      expect(receivedMethods).toContain('session/close')

      // 7. cancel_all cancels active ACP turn (Fix #13)
      const beforeCancelAllIdx = client.frames.length
      await client.call('turn.submit', { session_key: 'agy-s1', text: 'HANG_FOR_CANCEL_ALL' })
      await client.next(
        f => client.frames.indexOf(f) >= beforeCancelAllIdx && f.method === 'event' && f.params?.type === 'turn_begin',
      )
      const cancelAllRes = await client.call('cancel_all', {})
      expect(cancelAllRes.ok).toBe(true)
      expect(Number(cancelAllRes.cancelled)).toBeGreaterThanOrEqual(1)
      await client.next(
        f =>
          client.frames.indexOf(f) >= beforeCancelAllIdx &&
          f.method === 'event' &&
          f.params?.type === 'turn_end' &&
          f.params.payload?.cancelled === true,
      )

      // 8. Unexpected agy acp crash cleans up binding, broadcasts disconnected acp_status with stderr, and exposes saved_acp_binding (Fixes #2 & #8)
      const beforeCrashIdx = client.frames.length
      harnesses[0]!.pushStderr('fatal: auth token revoked\n')
      harnesses[0]!.terminate(1)
      const crashStatus = await client.next(
        f =>
          client.frames.indexOf(f) >= beforeCrashIdx &&
          f.method === 'event' &&
          f.params?.type === 'acp_status' &&
          (f.params.payload?.binding as Record<string, unknown> | undefined)?.connected === false,
      )
      expect(String((crashStatus.params!.payload!.binding as Record<string, unknown>).error)).toContain(
        'fatal: auth token revoked',
      )
      expect(bridgeHost.hasActiveBridge(sessionId)).toBe(false)

      const openDisconnected = await client.call('session.open', { session_key: 'agy-s1', project_dir: root })
      expect((openDisconnected.session as Record<string, unknown>).acp_binding).toBeUndefined()
      expect((openDisconnected.session as Record<string, unknown>).saved_acp_binding).toMatchObject({
        agent_id: 'antigravity',
        external_session_id: 'agy-ext-1',
        connected: false,
      })

      // 9. Reconnecting automatically calls session/resume with the saved external_session_id (Fix #8)
      const reconnected = await client.call('acp.connect', {
        session_id: sessionId,
        agent_id: 'antigravity',
      })
      expect(reconnected).toMatchObject({
        ok: true,
        binding: {
          agent_id: 'antigravity',
          external_session_id: 'agy-ext-1',
          connected: true,
        },
      })
      expect(receivedMethods).toContain('session/resume')

      // 10. GatewayClient.acpSetConfigOption works end-to-end
      const gw = new GatewayClient({
        projectDir: root,
        externalSocketPath: socketPath,
        sessionKey: 'agy-s1',
      })
      await gw.start()
      try {
        const cfgRes = await gw.acpSetConfigOption({
          session_id: sessionId,
          config_id: 'thought_level',
          value: 'low',
        })
        expect(cfgRes).toMatchObject({
          ok: true,
          binding: { thought_level: 'low' },
        })
      } finally {
        gw.close()
      }
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('falls back cleanly when agent lacks embeddedContext/image capabilities and adapts TUI acp_status + /acp command metadata (Fixes 4, 14)', async () => {
    // 1. Agent without image/embeddedContext falls back to inline text and non-fatal note
    const receivedPrompts: Array<Record<string, unknown>> = []
    const host = new DaemonAcpBridgeHost({
      discoverAgents: () => AGY_AGENT_ROSTER,
      createAcpClient: opts => {
        const h = createInMemoryAcpTransport((frame, harness) => {
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                protocolVersion: 1,
                agentCapabilities: {
                  promptCapabilities: { image: false, embeddedContext: false },
                },
              },
            })
            return
          }
          if (frame.method === 'session/new') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { sessionId: 'no-mm-1' } })
            return
          }
          if (frame.method === 'session/prompt') {
            receivedPrompts.push(frame.params as Record<string, unknown>)
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { stopReason: 'end_turn' } })
          }
        })
        return new AcpStdioClient({ ...opts, transport: h.transport })
      },
    })
    await host.connect({ sessionId: 's-no-mm', agentId: 'antigravity', cwd: '/tmp' })
    await host.runPrompt({
      sessionId: 's-no-mm',
      text: 'Check @a.ts',
      fallbackInlineText: '<attached_files>\n<file path="/tmp/a.ts">\nconst x = 1\n</file>\n</attached_files>\n\nCheck @a.ts',
      resources: [{ path: '/tmp/a.ts', uri: 'file:///tmp/a.ts', mimeType: 'text/typescript', text: 'const x = 1' }],
      images: [{ data: SAMPLE_PNG_BASE64, mediaType: 'image/png' }],
      onEvent: () => undefined,
    })
    const promptBlocks = receivedPrompts[0]!.prompt as Array<Record<string, unknown>>
    expect(promptBlocks).toHaveLength(1)
    expect(promptBlocks[0]!.type).toBe('text')
    expect(String(promptBlocks[0]!.text)).toContain('<attached_files>')
    // Antigravity sessions never inject Xerxes synthetic image-omitted notes into user prompt text.
    expect(String(promptBlocks[0]!.text)).not.toContain('image attachment')
    await host.disposeAll()

    // 2. /acp command registry metadata includes model/mode/thinking/config
    const acpDef = resolveCommand('/acp')
    expect(acpDef?.argsHint).toContain('model <id>')
    expect(acpDef?.argsHint).toContain('mode <id>')
    expect(acpDef?.argsHint).toContain('thinking <level>')
    expect(acpDef?.argsHint).toContain('config <id> <value>')

    // 3. TUI gatewayAdapter maps acp_status (connected, error, disconnected) & sessionInfoFromInit
    const connectedEvents = adaptDaemonEvent('acp_status', {
      session_id: 's1',
      binding: {
        session_id: 's1',
        agent_id: 'antigravity',
        name: 'Antigravity',
        provider: 'antigravity',
        transport: 'acp',
        cwd: '/tmp',
        connected: true,
        current_model: 'gemini-4-carbon',
        current_model_name: 'Gemini 4 Carbon',
        thought_level: 'high',
        current_mode: 'default',
        current_mode_name: 'Default',
      },
    })
    expect(connectedEvents.some(e => e.type === 'acp_status')).toBe(true)
    expect(connectedEvents.find(e => e.type === 'session.info')?.payload).toMatchObject({
      model: 'Antigravity: Gemini 4 Carbon',
      profile_name: 'Antigravity',
      mode: 'Default',
      reasoning_effort: 'high',
    })
    expect(connectedEvents.find(e => e.type === 'status.update')?.payload).toMatchObject({
      text: 'Antigravity · Gemini 4 Carbon · thinking: high · mode: Default',
    })

    const errorEvents = adaptDaemonEvent('acp_status', {
      session_id: 's1',
      binding: {
        session_id: 's1',
        agent_id: 'antigravity',
        name: 'Antigravity',
        provider: 'antigravity',
        transport: 'acp',
        cwd: '/tmp',
        connected: false,
        error: 'Antigravity ACP process exited unexpectedly (code 1)',
      },
    })
    expect(errorEvents.find(e => e.type === 'notification.show')?.payload).toMatchObject({
      level: 'error',
      text: 'Antigravity ACP process exited unexpectedly (code 1)',
    })

    const initInfo = sessionInfoFromInit({
      cwd: '/tmp',
      acp_binding: {
        agent_id: 'antigravity',
        name: 'Antigravity',
        transport: 'acp',
        connected: true,
        current_model_name: 'Gemini 4 Neon',
        thought_level: 'xhigh',
        current_mode_name: 'Plan',
      },
    })
    expect(initInfo).toMatchObject({
      model: 'Antigravity: Gemini 4 Neon',
      profile_name: 'Antigravity',
      mode: 'Plan',
      reasoning_effort: 'xhigh',
    })
  })

  test('enforces strict ACP runtime discovery, acp.catalog RPC, zero Xerxes native command/skill leaks, colliding slash routing, live updates, and clean native restoration on disconnect', async () => {
    const root = await makeTempDir('xrx-agy-isolation-')
    const fakeHome = join(root, 'fake-home')

    // Workspace Antigravity skill (`.agents/skills/academic-cv-builder/SKILL.md`)
    await mkdir(join(root, '.agents', 'skills', 'academic-cv-builder'), { recursive: true })
    await writeFile(
      join(root, '.agents', 'skills', 'academic-cv-builder', 'SKILL.md'),
      '---\nname: academic-cv-builder\ndescription: Format academic CVs with publications and grants\n---\n# Academic CV Builder\n',
      'utf8',
    )
    // Antigravity builtin skill (`~/.gemini/antigravity/builtin/skills/antigravity_guide/SKILL.md`)
    await mkdir(join(fakeHome, '.gemini', 'antigravity', 'builtin', 'skills', 'antigravity_guide'), {
      recursive: true,
    })
    await writeFile(
      join(fakeHome, '.gemini', 'antigravity', 'builtin', 'skills', 'antigravity_guide', 'SKILL.md'),
      '---\nname: antigravity-guide\ndescription: Guide for Google Antigravity CLI and customizations\n---\n',
      'utf8',
    )
    // Xerxes native skills directory with `immortals-coordinator` & `immortals-monitor` (MUST NOT leak when on ACP, MUST appear when on Xerxes!)
    const xerxesSkillsDir = join(root, 'xerxes-native-skills')
    await mkdir(join(xerxesSkillsDir, 'immortals-coordinator'), { recursive: true })
    await writeFile(
      join(xerxesSkillsDir, 'immortals-coordinator', 'SKILL.md'),
      '---\nname: immortals-coordinator\ndescription: Xerxes native Immortals coordinator\n---\n# Immortals Coordinator\n',
      'utf8',
    )
    await mkdir(join(xerxesSkillsDir, 'immortals-monitor'), { recursive: true })
    await writeFile(
      join(xerxesSkillsDir, 'immortals-monitor', 'SKILL.md'),
      '---\nname: immortals-monitor\ndescription: Xerxes native Immortals monitor\n---\n# Immortals Monitor\n',
      'utf8',
    )

    const receivedAcpPrompts: string[] = []
    let activeHarness: InMemoryAcpHarness | undefined

    const bridgeHost = new DaemonAcpBridgeHost({
      homeDir: fakeHome,
      discoverAgents: () => AGY_AGENT_ROSTER,
      createAcpClient: opts => {
        const h = createInMemoryAcpTransport((frame, harness) => {
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                protocolVersion: 1,
                agentInfo: { name: 'Antigravity ACP Server', version: '1.5.14' },
                agentCapabilities: {
                  loadSession: true,
                  promptCapabilities: { image: true, audio: false, embeddedContext: true },
                  sessionCapabilities: {
                    resume: {},
                    close: {},
                    load: {},
                    list: {},
                    additionalDirectories: {},
                  },
                  mcpCapabilities: { http: false, sse: false },
                  _meta: { antigravity: { tier: true } },
                },
                authMethods: [
                  { id: 'oauth-personal', name: 'Log in with Google', description: 'Google sign-in' },
                ],
              },
            })
            return
          }
          if (frame.method === 'session/new') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: 'agy-iso-1',
                update: {
                  sessionUpdate: 'available_commands_update',
                  availableCommands: [
                    { name: 'goal', description: 'Run until goal finishes', input: { hint: '<goal>' } },
                    { name: 'compact', description: 'Compact AGY conversation' },
                    { name: 'plan', description: 'Plan carefully before executing' },
                    { name: 'swarm', description: 'Run an Antigravity swarm' },
                    { name: 'boost', description: 'Invoke Boost orchestrator' },
                    { name: 'academic-cv-builder', description: 'Format academic CVs with publications and grants' },
                    { name: 'antigravity-guide', description: 'Guide for Google Antigravity CLI and customizations' },
                  ],
                },
              },
            })
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                sessionId: 'agy-iso-1',
                configOptions: [
                  ...makeAgyConfigOptions('gemini-4-carbon', 'medium', 'default'),
                  {
                    id: 'grounding',
                    name: 'Web Grounding',
                    type: 'boolean',
                    currentValue: true,
                    options: [
                      { value: 'true', name: 'On' },
                      { value: 'false', name: 'Off' },
                    ],
                  },
                ],
              },
            })
            return
          }
          if (frame.method === 'session/set_config_option') {
            const p = frame.params as Record<string, unknown>
            const nextMode = p.configId === 'mode' && typeof p.value === 'string' ? p.value : 'default'
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                configOptions: [
                  ...makeAgyConfigOptions('gemini-4-carbon', 'medium', nextMode),
                  {
                    id: 'grounding',
                    name: 'Web Grounding',
                    type: 'boolean',
                    currentValue: true,
                    options: [
                      { value: 'true', name: 'On' },
                      { value: 'false', name: 'Off' },
                    ],
                  },
                ],
              },
            })
            return
          }
          if (frame.method === 'session/prompt') {
            const params = frame.params as Record<string, unknown>
            const promptText = extractSpecPromptText(params)
            receivedAcpPrompts.push(promptText)
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: 'agy-iso-1',
                update: {
                  sessionUpdate: 'agent_message_chunk',
                  content: { type: 'text', text: `AGY handled: ${promptText}` },
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
        activeHarness = h
        return new AcpStdioClient({ ...opts, transport: h.transport })
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
    const socketPath = join(root, 'agy-iso.sock')
    const server = new DaemonServer({
      socketPath,
      projectDirectory: root,
      skillDirectory: xerxesSkillsDir,
      runtime,
      acpBridgeHost: bridgeHost,
      autoTitle: false,
      autoSnapshotTurns: false,
      eventJournal: { directory: join(root, 'events') },
    })
    await server.start()

    const client = await SocketRpcClient.open(socketPath)
    try {
      const init = await client.call('initialize', {
        session_key: 'iso-s1',
        project_dir: root,
      })
      const sessionId = String(init.session_id)

      // 0. Before connecting ACP: acp.catalog reports disconnected, and native complete("/") exposes Xerxes commands & immortals-* skills
      const nativeCatalog = await client.call('acp.catalog', { session_id: sessionId })
      expect(nativeCatalog).toEqual({
        ok: true,
        connected: false,
        commands: [],
        skills: [],
        config_options: [],
      })
      const nativeCompleteBefore = await client.call('complete', { text: '/', session_key: 'iso-s1' })
      const nativeLabelsBefore = (nativeCompleteBefore.completions as Array<{ label: string }>).map(c => c.label)
      expect(nativeLabelsBefore).toContain('immortals')
      expect(nativeLabelsBefore).toContain('provider')
      expect(nativeLabelsBefore).toContain('immortals-coordinator')

      // 1. Connect Antigravity ACP -> rich capabilities, classified commands, discovered skills, extra config_options
      const conn = await client.call('acp.connect', {
        session_id: sessionId,
        agent_id: 'antigravity',
      })
      expect(conn.ok).toBe(true)

      const acpCatalog = await client.call('acp.catalog', { session_id: sessionId })
      expect(acpCatalog).toMatchObject({
        ok: true,
        connected: true,
        binding: {
          agent_id: 'antigravity',
          agent_title: 'Antigravity ACP Server',
          agent_version: '1.5.14',
        },
        capabilities: {
          prompt: { image: true, audio: false, embedded_context: true },
          session: { load: true, resume: true, list: true, close: true, additional_directories: true },
          mcp: { http: false, sse: false },
          auth_methods: [{ id: 'oauth-personal', name: 'Log in with Google', description: 'Google sign-in' }],
          meta: { antigravity: { tier: true } },
        },
      })
      const catalogCommands = acpCatalog.commands as Array<Record<string, unknown>>
      const catalogSkills = acpCatalog.skills as Array<Record<string, unknown>>
      const catalogConfig = acpCatalog.config_options as Array<Record<string, unknown>>
      expect(catalogCommands.map(c => c.name)).toEqual(['goal', 'compact', 'plan', 'swarm', 'boost'])
      expect(catalogCommands.find(c => c.name === 'goal')?.input_hint).toBe('<goal>')
      expect(catalogSkills).toEqual([
        {
          name: 'academic-cv-builder',
          description: 'Format academic CVs with publications and grants',
          source: 'Workspace',
          path: join(root, '.agents', 'skills', 'academic-cv-builder', 'SKILL.md'),
          slash_command: '/academic-cv-builder',
        },
        {
          name: 'antigravity-guide',
          description: 'Guide for Google Antigravity CLI and customizations',
          source: 'Antigravity',
          path: join(fakeHome, '.gemini', 'antigravity', 'builtin', 'skills', 'antigravity_guide', 'SKILL.md'),
          slash_command: '/antigravity-guide',
        },
      ])
      expect(catalogConfig.some(o => o.id === 'grounding' && o.type === 'boolean' && o.current_value === true)).toBe(
        true,
      )
      // ZERO Xerxes native skills in acp.catalog
      expect(catalogSkills.some(s => String(s.name).startsWith('immortals'))).toBe(false)

      // 2. STRICT runtime isolation in complete("/") and complete("/skill ")
      const acpCompleteAll = await client.call('complete', { text: '/', session_key: 'iso-s1' })
      const acpItems = acpCompleteAll.completions as Array<Record<string, unknown>>
      const acpLabels = acpItems.map(item => String(item.label))

      // Contains ACP commands + ACP skills + minimal host bridge (/acp, /new, /clear, /export)
      expect(acpLabels).toContain('goal')
      expect(acpLabels).toContain('compact')
      expect(acpLabels).toContain('plan')
      expect(acpLabels).toContain('swarm')
      expect(acpLabels).toContain('boost')
      expect(acpLabels).toContain('academic-cv-builder')
      expect(acpLabels).toContain('antigravity-guide')
      expect(acpLabels).toContain('acp')
      expect(acpLabels).toContain('new')
      expect(acpLabels).toContain('clear')
      expect(acpLabels).toContain('export')

      // Check clean kind & source tags on completed items
      expect(acpItems.find(item => item.label === 'academic-cv-builder')).toMatchObject({
        value: '/academic-cv-builder',
        kind: 'skill',
        source: 'Workspace',
      })
      expect(acpItems.find(item => item.label === 'swarm')).toMatchObject({
        value: '/swarm',
        kind: 'command',
        source: 'Antigravity',
      })
      expect(acpItems.find(item => item.label === 'acp')).toMatchObject({
        value: '/acp',
        kind: 'command',
        source: 'Xerxes host',
      })

      // ZERO Xerxes native daemon commands or Xerxes native skills leak!
      const forbiddenXerxesLabels = [
        'immortals',
        'immortals-coordinator',
        'immortals-monitor',
        'provider',
        'model',
        'agents',
        'memory',
        'worktree',
        'skills',
        'skill',
        'mcp',
        'hooks',
        'plugins',
        'budget',
        'channels',
        'voice',
        'btw',
        'xerxes',
        'forge',
      ]
      for (const forbidden of forbiddenXerxesLabels) {
        expect(acpLabels).not.toContain(forbidden)
      }

      // `/skill ` subcommand completion MUST NOT leak Xerxes native skills while on ACP!
      const acpSkillSubComplete = await client.call('complete', { text: '/skill ', session_key: 'iso-s1' })
      expect(acpSkillSubComplete.completions).toEqual([])
      const acpImmortalsComplete = await client.call('complete', { text: '/immortals', session_key: 'iso-s1' })
      expect(acpImmortalsComplete.completions).toEqual([])

      // 3. Live available_commands_update refreshes catalog & broadcasts acp_status
      const beforeUpdateIdx = client.frames.length
      activeHarness!.pushAgentFrame({
        jsonrpc: '2.0',
        method: 'session/update',
        params: {
          sessionId: 'agy-iso-1',
          update: {
            sessionUpdate: 'available_commands_update',
            availableCommands: [
              { name: 'goal', description: 'Run until goal finishes' },
              { name: 'compact', description: 'Compact AGY conversation' },
              { name: 'plan', description: 'Plan carefully before executing' },
              { name: 'swarm', description: 'Run an Antigravity swarm' },
              { name: 'boost', description: 'Invoke Boost orchestrator' },
              { name: 'new', description: 'AGY native new command' },
              { name: 'academic-cv-builder', description: 'Format academic CVs with publications and grants' },
              { name: 'antigravity-guide', description: 'Guide for Google Antigravity CLI and customizations' },
            ],
          },
        },
      })
      await client.next(
        f =>
          client.frames.indexOf(f) >= beforeUpdateIdx &&
          f.method === 'event' &&
          f.params?.type === 'acp_status' &&
          Array.isArray((f.params.payload?.binding as Record<string, unknown> | undefined)?.available_commands) &&
          ((f.params.payload?.binding as Record<string, unknown>).available_commands as Array<{ name: string }>).some(
            c => c.name === 'new',
          ),
      )
      // Since ACP now advertises `/new`, host `/new` must NOT collide/duplicate it
      const afterNewUpdate = await client.call('complete', { text: '/new', session_key: 'iso-s1' })
      const newMatches = (afterNewUpdate.completions as Array<Record<string, unknown>>).filter(
        c => c.label === 'new',
      )
      expect(newMatches).toHaveLength(1)
      expect(newMatches[0]?.source).toBe('Antigravity')

      // 4. Colliding slash commands & discovered skills route straight to external ACP (NEVER Xerxes native handlers!)
      const slashCommandsToForward = [
        '/compact',
        '/goal finish tests',
        '/swarm audit modules',
        '/plan design auth',
        '/boost speed up build',
        '/academic-cv-builder draft my CV',
      ]
      for (const cmd of slashCommandsToForward) {
        const beforeIdx = client.frames.length
        const res = await client.call('slash', { command: cmd })
        expect(res).toMatchObject({ ok: true, forwarded_to_acp: true, agent_id: 'antigravity', command: cmd })
        await client.next(
          f => client.frames.indexOf(f) >= beforeIdx && f.method === 'event' && f.params?.type === 'turn_end',
        )
      }
      expect(receivedAcpPrompts).toEqual(slashCommandsToForward)

      // Xerxes native skill shorthand MUST NOT invoke Xerxes skill registry while on ACP
      const blockedNativeSkill = await client.call('slash', { command: '/immortals-coordinator run' })
      expect(blockedNativeSkill.ok).toBe(false)
      expect(String(blockedNativeSkill.error)).toContain('not available in Antigravity')

      // `/acp status` and `/acp catalog` stay on the Xerxes host and report discovered capabilities/skills/commands/config!
      const acpStatusSlash = await client.call('slash', { command: '/acp status' })
      expect(acpStatusSlash.ok).toBe(true)
      expect(String(acpStatusSlash.text)).toContain('Antigravity v1.5.14')
      expect(String(acpStatusSlash.text)).toContain('6 commands · 2 skills · 4 config options')
      expect(String(acpStatusSlash.text)).toContain('Capabilities: prompt[images, embedded-context]')

      const acpCatalogSlash = await client.call('slash', { command: '/acp catalog' })
      expect(acpCatalogSlash.ok).toBe(true)
      expect(String(acpCatalogSlash.text)).toContain('/academic-cv-builder [Workspace]')
      expect(String(acpCatalogSlash.text)).toContain('/antigravity-guide [Antigravity]')

      // GatewayClient.acpCatalog works end-to-end & TUI gatewayAdapter maps ACP discovered skills cleanly
      const gw = new GatewayClient({
        projectDir: root,
        externalSocketPath: socketPath,
        sessionKey: 'iso-s1',
      })
      await gw.start()
      try {
        const gwCatalog = await gw.acpCatalog({ session_id: sessionId })
        expect(gwCatalog.ok).toBe(true)
        expect(gwCatalog.connected).toBe(true)
        expect(gwCatalog.skills.map(s => s.name)).toEqual(['academic-cv-builder', 'antigravity-guide'])
        expect(gwCatalog.commands.map(c => c.name)).toContain('swarm')
        expect(gwCatalog.capabilities?.session.additional_directories).toBe(true)
      } finally {
        gw.close()
      }

      const adaptedStatus = adaptDaemonEvent('acp_status', {
        session_id: sessionId,
        binding: acpCatalog.binding as Record<string, unknown>,
      })
      expect(adaptedStatus.find(e => e.type === 'session.info')?.payload).toMatchObject({
        skills: { Antigravity: ['academic-cv-builder', 'antigravity-guide'] },
      })
      expect(String((adaptedStatus.find(e => e.type === 'status.update')?.payload as Record<string, unknown>).text)).toContain(
        '2 skills · 5 cmds',
      )

      // 5. Disconnecting back to Xerxes IMMEDIATELY restores full native commands & skills!
      const disc = await client.call('acp.disconnect', { session_id: sessionId })
      expect(disc).toMatchObject({ ok: true, disconnected: true, binding: null })

      const catalogAfterDisconnect = await client.call('acp.catalog', { session_id: sessionId })
      expect(catalogAfterDisconnect).toEqual({
        ok: true,
        connected: false,
        commands: [],
        skills: [],
        config_options: [],
      })

      const restoredComplete = await client.call('complete', { text: '/', session_key: 'iso-s1' })
      const restoredLabels = (restoredComplete.completions as Array<{ label: string }>).map(c => c.label)
      expect(restoredLabels).toContain('immortals')
      expect(restoredLabels).toContain('provider')
      expect(restoredLabels).toContain('skills')
      expect(restoredLabels).toContain('immortals-coordinator')
      expect(restoredLabels).toContain('immortals-monitor')
      expect(restoredLabels).not.toContain('antigravity-guide')

      const restoredSkillSub = await client.call('complete', { text: '/skill ', session_key: 'iso-s1' })
      const restoredSkillLabels = (restoredSkillSub.completions as Array<{ label: string }>).map(c => c.label)
      expect(restoredSkillLabels).toContain('immortals-coordinator')
    } finally {
      client.close()
      await server.stop()
    }
  })
})

describe('Antigravity ACP elicitation/create (ask_question) end-to-end', () => {
  test('parses single-select, multi-select, companion _other fields, comma-containing titles, Skip decline, and generic JSON Schema types', () => {
    const req = parseElicitationRequest({
      sessionId: 'agy-elicit-1',
      toolCallId: 'call_ask_1',
      mode: 'form',
      message: 'The agent has 3 questions.',
      requestedSchema: {
        type: 'object',
        properties: {
          q1: {
            type: 'string',
            title: 'Which Xerxes agents should I design?',
            oneOf: [
              { const: '1', title: '(Recommended) Core roster: Xerxes, Coder, Reviewer' },
              { const: '2', title: 'Preview roster only' },
            ],
          },
          q1_other: {
            type: 'string',
            title: 'Other',
            description: 'Optional free-form answer to: Which Xerxes agents should I design?',
          },
          q2: {
            type: 'array',
            title: 'Languages?',
            items: {
              anyOf: [
                { const: '1', title: 'TypeScript' },
                { const: '2', title: 'Rust' },
                { const: '3', title: 'Go' },
              ],
            },
          },
          q2_other: {
            type: 'string',
            title: 'Other',
            description: 'Optional free-form answer to: Languages?',
          },
          q3: {
            type: 'string',
            title: 'Theme?',
            oneOf: [
              { const: '1', title: 'Dark' },
              { const: '2', title: 'Light' },
            ],
          },
          q3_other: {
            type: 'string',
            title: 'Other',
            description: 'Optional free-form answer to: Theme?',
          },
          confirm: {
            type: 'boolean',
            title: 'Proceed with edits?',
          },
          retries: {
            type: 'integer',
            title: 'Retry count',
          },
          strict_env: {
            type: 'string',
            title: 'Environment',
            enum: ['dev', 'prod'],
            enumNames: ['Development', 'Production'],
          },
        },
      },
    })

    const specs = elicitationToQuestionItems(req)
    // Companion q1_other, q2_other, q3_other are folded into allowFreeform: true, NEVER duplicated as separate items.
    expect(specs.map(s => s.id)).toEqual(['q1', 'q2', 'q3', 'confirm', 'retries', 'strict_env'])
    expect(specs[0]).toMatchObject({
      id: 'q1',
      question: 'Which Xerxes agents should I design?',
      options: ['(Recommended) Core roster: Xerxes, Coder, Reviewer', 'Preview roster only'],
      allowFreeform: true,
      multiSelect: false,
      otherFieldKey: 'q1_other',
    })
    expect(specs[1]).toMatchObject({
      id: 'q2',
      question: 'Languages?',
      options: ['TypeScript', 'Rust', 'Go'],
      allowFreeform: true,
      multiSelect: true,
      otherFieldKey: 'q2_other',
    })
    expect(specs[3]).toMatchObject({ id: 'confirm', options: ['Yes', 'No'], allowFreeform: false })
    expect(specs[4]).toMatchObject({ id: 'retries', options: [], allowFreeform: true })
    expect(specs[5]).toMatchObject({ id: 'strict_env', options: ['Development', 'Production'], allowFreeform: false })

    // Comma-containing option title stays intact, newline-joined multi-select + custom Other + typed fields map cleanly.
    const accepted = buildElicitationResponseFromAnswers(specs, {
      q1: '(Recommended) Core roster: Xerxes, Coder, Reviewer',
      q2: 'TypeScript\nGo\nZig',
      q3: 'Custom magenta',
      confirm: 'Yes',
      retries: '3',
      strict_env: 'Production',
    })
    expect(accepted).toEqual({
      action: 'accept',
      content: {
        q1: '1',
        q2: ['1', '3'],
        q2_other: 'Zig',
        q3_other: 'Custom magenta',
        confirm: true,
        retries: 3,
        strict_env: 'prod',
      },
    })
    expect(formatElicitationAnswersSummary(specs, accepted)).toContain('Languages? -> TypeScript, Go, Zig')
    expect(formatElicitationAnswersSummary(specs, accepted)).toContain('Theme? -> Custom magenta')

    // Array answers (string[]) work identically for multi-select.
    const arrayAccepted = buildElicitationResponseFromAnswers(specs, {
      q2: ['Rust', 'Go'],
    })
    expect(arrayAccepted).toEqual({
      action: 'accept',
      content: { q2: ['2', '3'] },
    })

    // Clicking Skip (all items answered "Skipped") translates to `{ action: "decline" }`.
    const skipped = buildElicitationResponseFromAnswers(specs.slice(0, 3), {
      q1: 'Skipped',
      q2: 'Skipped',
      q3: 'Skipped',
    })
    expect(skipped).toEqual({ action: 'decline' })
    expect(formatElicitationAnswersSummary(specs.slice(0, 3), skipped)).toBe('User skipped.')

    // URL mode elicitation is safely ignored (not advertised in clientCapabilities).
    const urlReq = parseElicitationRequest({
      sessionId: 'agy-elicit-1',
      mode: 'url',
      url: 'https://example.com/auth',
      message: 'Sign in',
    })
    expect(elicitationToQuestionItems(urlReq)).toEqual([])
  })

  test('AcpStdioClient advertises elicitation.form only when configured and answers elicitation/create JSON-RPC requests', async () => {
    // Without createElicitation handler -> does NOT advertise elicitation capability and returns -32601.
    const noHandlerHarness = createInMemoryAcpTransport((frame, harness) => {
      if (frame.method === 'initialize') {
        harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { protocolVersion: 1 } })
      }
    })
    const noHandlerClient = new AcpStdioClient({ command: 'fake-agy', transport: noHandlerHarness.transport })
    await noHandlerClient.initialize()
    const noHandlerInit = noHandlerHarness.receivedFrames[0] as Record<string, Record<string, unknown>>
    expect((noHandlerInit.params?.clientCapabilities as Record<string, unknown>)?.elicitation).toBeUndefined()
    noHandlerHarness.pushAgentFrame({
      jsonrpc: '2.0',
      id: 'elicit-err-1',
      method: 'elicitation/create',
      params: { sessionId: 's1', mode: 'form', message: 'Q?', requestedSchema: { type: 'object', properties: {} } },
    })
    await Bun.sleep(20)
    expect(noHandlerHarness.receivedFrames.find(f => f.id === 'elicit-err-1')).toMatchObject({
      error: { code: -32601 },
    })
    await noHandlerClient.dispose()

    // With createElicitation handler -> advertises elicitation: { form: {} } and replies cleanly.
    const withHandlerHarness = createInMemoryAcpTransport((frame, harness) => {
      if (frame.method === 'initialize') {
        harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { protocolVersion: 1 } })
      }
    })
    const withHandlerClient = new AcpStdioClient({
      command: 'fake-agy',
      transport: withHandlerHarness.transport,
      handlers: {
        createElicitation: req => ({
          action: 'accept',
          content: { q1: req.fields[0]?.oneOf?.[1]?.const ?? '1' },
        }),
      },
    })
    await withHandlerClient.initialize()
    expect(withHandlerHarness.receivedFrames[0]).toMatchObject({
      method: 'initialize',
      params: { clientCapabilities: { elicitation: { form: {} } } },
    })
    withHandlerHarness.pushAgentFrame({
      jsonrpc: '2.0',
      id: 'elicit-ok-1',
      method: 'elicitation/create',
      params: {
        sessionId: 's1',
        toolCallId: 'tc-ask-1',
        mode: 'form',
        message: 'The agent has a question.',
        requestedSchema: {
          type: 'object',
          properties: {
            q1: {
              type: 'string',
              title: 'Favorite color?',
              oneOf: [
                { const: '1', title: '(Recommended) Blue' },
                { const: '2', title: 'Green' },
              ],
            },
          },
        },
      },
    })
    await Bun.sleep(20)
    expect(withHandlerHarness.receivedFrames.find(f => f.id === 'elicit-ok-1')).toEqual({
      jsonrpc: '2.0',
      id: 'elicit-ok-1',
      result: { action: 'accept', content: { q1: '2' } },
    })
    await withHandlerClient.dispose()
  })

  test('DaemonServer bridges ACP ask_question elicitation/create to question_request/question_response (multi-question, multi-select, Other, Skip, and cancel)', async () => {
    const root = await makeTempDir('xrx-agy-elicit-e2e-')
    const elicitationResults: Array<Record<string, unknown>> = []
    let activePromptId: unknown
    let activeSid = 'agy-elicit-sess'
    let activeToolCallId = ''

    const bridgeHost = new DaemonAcpBridgeHost({
      discoverAgents: () => AGY_AGENT_ROSTER,
      createAcpClient: opts => {
        const harness = createInMemoryAcpTransport((frame, h) => {
          if (frame.method === 'initialize') {
            h.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                protocolVersion: 1,
                agentInfo: { name: 'Antigravity ACP Server', version: '1.5.14' },
                agentCapabilities: {
                  promptCapabilities: { image: true, embeddedContext: true },
                  sessionCapabilities: { close: {} },
                },
              },
            })
            return
          }
          if (frame.method === 'session/new') {
            h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { sessionId: activeSid } })
            return
          }
          if (frame.method === 'session/close') {
            h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: {} })
            return
          }
          if (frame.method === 'session/cancel') {
            setTimeout(() => {
              h.pushAgentFrame({ jsonrpc: '2.0', id: activePromptId, result: { stopReason: 'cancelled' } })
            }, 15)
            return
          }
          if (frame.method === 'session/prompt') {
            activePromptId = frame.id
            const text = extractSpecPromptText(frame.params as Record<string, unknown>)
            if (text.includes('MULTI_Q')) {
              activeToolCallId = 'call_ask_multi'
              h.pushAgentFrame({
                jsonrpc: '2.0',
                method: 'session/update',
                params: {
                  sessionId: activeSid,
                  update: {
                    sessionUpdate: 'tool_call',
                    toolCallId: activeToolCallId,
                    name: 'ask_question',
                    status: 'in_progress',
                    kind: 'other',
                    title: 'Run ask_question',
                    rawInput: { questions: [{ question: 'Favorite color?' }] },
                  },
                },
              })
              h.pushAgentFrame({
                jsonrpc: '2.0',
                method: 'session/update',
                params: {
                  sessionId: activeSid,
                  update: {
                    sessionUpdate: 'tool_call_update',
                    toolCallId: activeToolCallId,
                    name: 'ask_question',
                    status: 'pending',
                  },
                },
              })
              h.pushAgentFrame({
                jsonrpc: '2.0',
                id: 'elicit-multi',
                method: 'elicitation/create',
                params: {
                  sessionId: activeSid,
                  toolCallId: activeToolCallId,
                  mode: 'form',
                  message: 'The agent has 3 questions.',
                  requestedSchema: {
                    type: 'object',
                    properties: {
                      q1: {
                        type: 'string',
                        title: 'Which roster?',
                        oneOf: [
                          { const: '1', title: '(Recommended) Core roster: Xerxes, Coder, Reviewer' },
                          { const: '2', title: 'Preview only' },
                        ],
                      },
                      q1_other: { type: 'string', title: 'Other' },
                      q2: {
                        type: 'array',
                        title: 'Languages?',
                        items: {
                          anyOf: [
                            { const: '1', title: 'TypeScript' },
                            { const: '2', title: 'Rust' },
                            { const: '3', title: 'Go' },
                          ],
                        },
                      },
                      q2_other: { type: 'string', title: 'Other' },
                      q3: {
                        type: 'string',
                        title: 'Theme?',
                        oneOf: [
                          { const: '1', title: 'Dark' },
                          { const: '2', title: 'Light' },
                        ],
                      },
                      q3_other: { type: 'string', title: 'Other' },
                    },
                  },
                },
              })
              return
            }
            if (text.includes('SKIP_Q') || text.includes('CANCEL_Q')) {
              activeToolCallId = text.includes('SKIP_Q') ? 'call_ask_skip' : 'call_ask_cancel'
              const reqId = text.includes('SKIP_Q') ? 'elicit-skip' : 'elicit-cancel'
              h.pushAgentFrame({
                jsonrpc: '2.0',
                method: 'session/update',
                params: {
                  sessionId: activeSid,
                  update: {
                    sessionUpdate: 'tool_call',
                    toolCallId: activeToolCallId,
                    name: 'ask_question',
                    status: 'in_progress',
                  },
                },
              })
              h.pushAgentFrame({
                jsonrpc: '2.0',
                id: reqId,
                method: 'elicitation/create',
                params: {
                  sessionId: activeSid,
                  toolCallId: activeToolCallId,
                  mode: 'form',
                  message: 'The agent has a question.',
                  requestedSchema: {
                    type: 'object',
                    properties: {
                      q1: {
                        type: 'string',
                        title: 'Pick A or B?',
                        oneOf: [
                          { const: '1', title: 'A' },
                          { const: '2', title: 'B' },
                        ],
                      },
                      q1_other: { type: 'string', title: 'Other' },
                    },
                  },
                },
              })
              return
            }
          }
          if (typeof frame.id === 'string' && frame.id.startsWith('elicit-')) {
            const res = (frame.result ?? {}) as Record<string, unknown>
            elicitationResults.push({ id: frame.id, ...res })
            if (frame.id === 'elicit-cancel') {
              return
            }
            h.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: activeSid,
                update: {
                  sessionUpdate: 'tool_call_update',
                  toolCallId: activeToolCallId,
                  name: 'ask_question',
                  status: 'completed',
                },
              },
            })
            h.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: activeSid,
                update: {
                  sessionUpdate: 'agent_message_chunk',
                  content: {
                    type: 'text',
                    text: res.action === 'decline' ? 'Skipped noted.' : `Received ${JSON.stringify(res.content)}`,
                  },
                },
              },
            })
            h.pushAgentFrame({ jsonrpc: '2.0', id: activePromptId, result: { stopReason: 'end_turn' } })
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
        permissionMode: 'accept-all',
        sessionDirectory: join(root, 'sessions'),
      },
    )
    const socketPath = join(root, 'daemon.sock')
    const server = new DaemonServer({
      socketPath,
      projectDirectory: root,
      runtime,
      acpBridgeHost: bridgeHost,
      autoTitle: false,
      autoSnapshotTurns: false,
      eventJournal: { directory: join(root, 'events') },
    })
    await server.start()
    const client = await SocketRpcClient.open(socketPath)
    try {
      const init = await client.call('initialize', {
        session_key: 'elicit-s1',
        project_dir: root,
        session_owned_turns: true,
      })
      const sessionId = String(init.session_id)
      const conn = await client.call('acp.connect', { session_id: sessionId, agent_id: 'antigravity', cwd: root })
      expect(conn.ok).toBe(true)

      // 1. Multi-question with comma-containing single-select, multi-select, and custom Other
      await client.call('turn.submit', { text: 'MULTI_Q', session_key: 'elicit-s1' })
      const qReqFrame = await client.next(f => f.method === 'event' && f.params?.type === 'question_request')
      const qPayload = qReqFrame.params!.payload!
      const qRequestId = String(qPayload.id)
      expect(qPayload.tool_call_id).toBe('call_ask_multi')
      expect(qPayload.questions).toEqual([
        {
          id: 'q1',
          question: 'Which roster?',
          options: ['(Recommended) Core roster: Xerxes, Coder, Reviewer', 'Preview only'],
          allow_free_form: true,
          placeholder: 'Other',
        },
        {
          id: 'q2',
          question: 'Languages?',
          options: ['TypeScript', 'Rust', 'Go'],
          allow_free_form: true,
          multi_select: true,
          placeholder: 'Other',
        },
        {
          id: 'q3',
          question: 'Theme?',
          options: ['Dark', 'Light'],
          allow_free_form: true,
          placeholder: 'Other',
        },
      ])

      // Reconnecting client sees pending ACP question_request in initialize.pending_interactions!
      const observer = await SocketRpcClient.open(socketPath)
      try {
        const obsInit = await observer.call('initialize', {
          session_key: 'elicit-s1',
          project_dir: root,
          session_owned_turns: true,
        })
        const pending = obsInit.pending_interactions as Array<{ type: string; payload: { id: string } }>
        expect(pending.some(item => item.type === 'question_request' && item.payload.id === qRequestId)).toBe(true)
      } finally {
        observer.close()
      }

      // Empty answer map is rejected cleanly
      const emptyRes = await client.call('question_response', { request_id: qRequestId, answers: {} })
      expect(emptyRes.ok).toBe(false)

      // Submit real answers (comma-containing title + newline-joined multi-select + freeform Other)
      const ansRes = await client.call('question_response', {
        request_id: qRequestId,
        answers: {
          q1: '(Recommended) Core roster: Xerxes, Coder, Reviewer',
          q2: 'TypeScript\nGo',
          q3: 'Custom magenta',
        },
      })
      expect(ansRes.ok).toBe(true)

      await client.next(f => f.method === 'event' && f.params?.type === 'turn_end')
      expect(elicitationResults[0]).toEqual({
        id: 'elicit-multi',
        action: 'accept',
        content: {
          q1: '1',
          q2: ['1', '3'],
          q3_other: 'Custom magenta',
        },
      })

      // ask_question tool_result settled cleanly (NOT failed!) with human-readable Q&A summary
      const askToolResult = client.frames.find(
        f => f.method === 'event' && f.params?.type === 'tool_result' && f.params.payload?.tool_call_id === 'call_ask_multi',
      )
      expect(askToolResult?.params?.payload).toMatchObject({
        name: 'ask_question',
        permitted: true,
      })
      expect(askToolResult?.params?.payload?.error).toBeUndefined()
      expect(String(askToolResult?.params?.payload?.return_value)).toContain('Languages? -> TypeScript, Go')
      expect(String(askToolResult?.params?.payload?.return_value)).toContain('Theme? -> Custom magenta')

      // 2. Skip flow -> sends `{ action: "decline" }` and completes cleanly
      const beforeSkipIdx = client.frames.length
      await client.call('turn.submit', { text: 'SKIP_Q', session_key: 'elicit-s1' })
      const skipReq = await client.next(
        f => client.frames.indexOf(f) >= beforeSkipIdx && f.method === 'event' && f.params?.type === 'question_request',
      )
      const skipId = String(skipReq.params!.payload!.id)
      expect((await client.call('question_response', { request_id: skipId, answers: { q1: 'Skipped' } })).ok).toBe(true)
      await client.next(
        f => client.frames.indexOf(f) >= beforeSkipIdx && f.method === 'event' && f.params?.type === 'turn_end',
      )
      expect(elicitationResults[1]).toEqual({ id: 'elicit-skip', action: 'decline' })

      // 3. Cancel turn while elicitation is pending -> sends `{ action: "cancel" }` and unblocks cleanly!
      const beforeCancelIdx = client.frames.length
      await client.call('turn.submit', { text: 'CANCEL_Q', session_key: 'elicit-s1' })
      await client.next(
        f => client.frames.indexOf(f) >= beforeCancelIdx && f.method === 'event' && f.params?.type === 'question_request',
      )
      await client.call('cancel', { session_key: 'elicit-s1' })
      const cancelEnd = await client.next(
        f => client.frames.indexOf(f) >= beforeCancelIdx && f.method === 'event' && f.params?.type === 'turn_end',
      )
      expect(cancelEnd.params?.payload?.cancelled).toBe(true)
      expect(elicitationResults[2]).toEqual({ id: 'elicit-cancel', action: 'cancel' })
    } finally {
      client.close()
      await server.stop()
    }
  })
})

describe('Antigravity ACP Turbo mode, Auto sync, resume restoration & mid-turn unlock', () => {
  test('augmentModesWithHostTurbo places Turbo right after Auto, skips agents with native bypass/turbo, and preserves empty lists', () => {
    expect(augmentModesWithHostTurbo([])).toEqual([])
    const agyModes = augmentModesWithHostTurbo([
      { id: 'default', name: 'Default' },
      { id: 'accept-edits', name: 'Accept edits' },
      { id: 'auto', name: 'Auto' },
      { id: 'plan', name: 'Plan' },
    ])
    expect(agyModes.map(m => m.id)).toEqual(['default', 'accept-edits', 'auto', 'turbo', 'plan'])
    expect(agyModes[3]).toEqual({
      id: 'turbo',
      name: 'Turbo',
      description: 'Run all commands without restrictions.',
    })

    // Native unrestricted mode (e.g. Claude Code bypassPermissions or native turbo) is never duplicated
    const withBypass = augmentModesWithHostTurbo([
      { id: 'default', name: 'Default' },
      { id: 'bypassPermissions', name: 'Bypass permissions' },
    ])
    expect(withBypass.map(m => m.id)).toEqual(['default', 'bypassPermissions'])
  })

  test('exposes Turbo after Auto, maps Turbo->auto on AGY, survives live auto updates & model switches, auto-approves immediately (including mid-turn), restores mode across session/resume, and restores prompts when leaving Turbo', async () => {
    const root = await makeTempDir('xrx-agy-turbo-')
    let spawnCount = 0
    let activeModel = 'gemini-4-carbon'
    let activeThought = 'medium'
    let activeMode = 'default'
    let activeHarness: InMemoryAcpHarness | undefined
    const setConfigCalls: Array<{ configId: string; value: string }> = []
    const permissionOutcomes: Array<{ id: string; optionId?: string; outcome?: string }> = []

    const bridgeHost = new DaemonAcpBridgeHost({
      discoverAgents: () => AGY_AGENT_ROSTER,
      createAcpClient: opts => {
        spawnCount += 1
        let pendingPromptId: unknown
        let activeSid = `agy-turbo-${spawnCount}`
        let step = 0
        const harness = createInMemoryAcpTransport((frame, h) => {
          if (frame.method === 'initialize') {
            h.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                protocolVersion: 1,
                agentInfo: { name: 'Antigravity ACP Server', version: '1.5.14' },
                agentCapabilities: {
                  loadSession: true,
                  promptCapabilities: { image: true, embeddedContext: true },
                  sessionCapabilities: { resume: {}, close: {}, load: {} },
                },
              },
            })
            return
          }
          if (frame.method === 'session/new' || frame.method === 'session/resume') {
            activeSid =
              frame.method === 'session/resume'
                ? String((frame.params as Record<string, unknown>).sessionId)
                : `agy-turbo-${spawnCount}`
            // Mirror real agy acp v1.5.14 behavior: session/resume ALWAYS resets mode back to 'default'!
            activeMode = 'default'
            h.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                sessionId: activeSid,
                modes: {
                  currentModeId: activeMode,
                  availableModes: [
                    { id: 'default', name: 'Default' },
                    { id: 'accept-edits', name: 'Accept edits' },
                    { id: 'auto', name: 'Auto' },
                    { id: 'plan', name: 'Plan' },
                  ],
                },
                configOptions: makeAgyConfigOptions(activeModel, activeThought, activeMode),
              },
            })
            return
          }
          if (frame.method === 'session/set_config_option') {
            const p = frame.params as Record<string, unknown>
            const configId = String(p.configId ?? '')
            const value = String(p.value ?? '')
            setConfigCalls.push({ configId, value })
            if (configId === 'mode' && value === 'turbo') {
              // Real agy acp v1.5.14 rejects unknown mode 'turbo'!
              h.pushAgentFrame({
                jsonrpc: '2.0',
                id: frame.id,
                error: { code: -32602, message: 'Unknown mode: turbo' },
              })
              return
            }
            if (configId === 'model') activeModel = value
            else if (configId === 'thought_level') activeThought = value
            else if (configId === 'mode') activeMode = value
            const nextCfg = makeAgyConfigOptions(activeModel, activeThought, activeMode)
            h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { configOptions: nextCfg } })
            return
          }
          if (frame.method === 'session/close') {
            h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: {} })
            return
          }
          if (frame.method === 'session/prompt') {
            pendingPromptId = frame.id
            const text = extractSpecPromptText(frame.params as Record<string, unknown>)
            // Real agy acp emits current_mode_update + config_option_update at prompt start reporting native 'auto'!
            h.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: activeSid,
                update: { sessionUpdate: 'current_mode_update', currentModeId: activeMode },
              },
            })
            h.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: activeSid,
                update: {
                  sessionUpdate: 'config_option_update',
                  configOptions: makeAgyConfigOptions(activeModel, activeThought, activeMode),
                },
              },
            })
            step = 1
            const permId = text.includes('MIDTURN_UNLOCK') ? 'perm-mid-1' : `perm-run-${spawnCount}-${text.slice(0, 8)}`
            h.pushAgentFrame({
              jsonrpc: '2.0',
              id: permId,
              method: 'session/request_permission',
              params: {
                sessionId: activeSid,
                toolCall: {
                  toolCallId: `tc-${permId}`,
                  name: 'run_command',
                  title: 'wc -l README.md | sort -rn',
                  kind: 'execute',
                  rawInput: {
                    CommandLine: 'wc -l README.md | sort -rn',
                    WaitMsBeforeAsync: 5000,
                    toolSummary: 'Count README lines',
                  },
                },
                options: [
                  { optionId: 'allow', name: 'Allow', kind: 'allow_once' },
                  { optionId: 'deny', name: 'Deny', kind: 'reject_once' },
                ],
              },
            })
            return
          }
          if (typeof frame.id === 'string' && frame.id.startsWith('perm-')) {
            const outcome = (frame.result as Record<string, unknown> | undefined)?.outcome as
              | Record<string, unknown>
              | undefined
            permissionOutcomes.push({
              id: frame.id,
              ...(typeof outcome?.outcome === 'string' ? { outcome: outcome.outcome } : {}),
              ...(typeof outcome?.optionId === 'string' ? { optionId: outcome.optionId } : {}),
            })
            if (frame.id === 'perm-mid-1' && step === 1) {
              // Second permission prompt in the SAME turn after mid-turn switch to Turbo
              step = 2
              h.pushAgentFrame({
                jsonrpc: '2.0',
                id: 'perm-mid-2',
                method: 'session/request_permission',
                params: {
                  sessionId: activeSid,
                  toolCall: {
                    toolCallId: 'tc-mid-2',
                    name: 'run_command',
                    title: 'echo step2',
                    kind: 'execute',
                    rawInput: { CommandLine: 'echo step2' },
                  },
                  options: [
                    { optionId: 'allow', name: 'Allow', kind: 'allow_once' },
                    { optionId: 'deny', name: 'Deny', kind: 'reject_once' },
                  ],
                },
              })
              return
            }
            h.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: activeSid,
                update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'DONE' } },
              },
            })
            h.pushAgentFrame({ jsonrpc: '2.0', id: pendingPromptId, result: { stopReason: 'end_turn' } })
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
        permissionMode: 'accept-all',
        sessionDirectory: join(root, 'sessions'),
      },
    )
    const socketPath = join(root, 'agy-turbo.sock')
    const server = new DaemonServer({
      socketPath,
      projectDirectory: root,
      runtime,
      acpBridgeHost: bridgeHost,
      autoTitle: false,
      autoSnapshotTurns: false,
      eventJournal: { directory: join(root, 'events') },
    })
    await server.start()

    const client = await SocketRpcClient.open(socketPath)
    try {
      const init = await client.call('initialize', {
        session_key: 'turbo-s1',
        project_dir: root,
        session_owned_turns: true,
      })
      const sessionId = String(init.session_id)

      // 1. Connect AGY -> Turbo appears right after Auto and before Plan in both modes & config_options
      //    AND is selected by default on a fresh Antigravity connect (sending native 'auto' to AGY)!
      setConfigCalls.length = 0
      const conn = await client.call('acp.connect', { session_id: sessionId, agent_id: 'antigravity', cwd: root })
      expect(conn.ok).toBe(true)
      const initialBinding = conn.binding as Record<string, unknown>
      expect(initialBinding).toMatchObject({
        current_mode: 'turbo',
        current_mode_name: 'Turbo',
      })
      expect(setConfigCalls).toEqual([{ configId: 'mode', value: 'auto' }])
      expect((initialBinding.modes as Array<{ id: string }>).map(m => m.id)).toEqual([
        'default',
        'accept-edits',
        'auto',
        'turbo',
        'plan',
      ])
      const modeConfig = (initialBinding.config_options as Array<{ id: string; current_value: string; options: Array<{ value: string }> }>)
        .find(opt => opt.id === 'mode')
      expect(modeConfig?.current_value).toBe('turbo')
      expect(modeConfig?.options.map(o => o.value)).toEqual(['default', 'accept-edits', 'auto', 'turbo', 'plan'])

      // Tab completion exposes turbo
      const modeComplete = await client.call('complete', { text: '/acp mode tu', session_key: 'turbo-s1' })
      expect((modeComplete.completions as Array<{ label: string }>).map(c => c.label)).toEqual(['turbo'])

      // 2. Switching to Default then back to Turbo via acp.set_mode -> sends native 'auto' to AGY, keeps binding in 'turbo'
      await client.call('acp.set_mode', { session_id: sessionId, mode_id: 'default' })
      setConfigCalls.length = 0
      const setTurbo = await client.call('acp.set_mode', { session_id: sessionId, mode_id: 'turbo' })
      expect(setTurbo).toMatchObject({
        ok: true,
        binding: {
          current_mode: 'turbo',
          current_mode_name: 'Turbo',
        },
      })
      expect(setConfigCalls).toEqual([{ configId: 'mode', value: 'auto' }])
      const turboModeCfg = (
        (setTurbo.binding as Record<string, unknown>).config_options as Array<{ id: string; current_value: string }>
      ).find(opt => opt.id === 'mode')
      expect(turboModeCfg?.current_value).toBe('turbo')

      // Switching model while in Turbo MUST NOT revert mode back to 'auto'
      const switchModel = await client.call('acp.set_model', { session_id: sessionId, model_id: 'gemini-4-neon' })
      expect(switchModel).toMatchObject({
        ok: true,
        binding: {
          current_model: 'gemini-4-neon',
          current_mode: 'turbo',
          current_mode_name: 'Turbo',
        },
      })

      // 3. Running a turn in Turbo auto-approves session/request_permission with ZERO approval_request events,
      // AND survives AGY's incoming current_mode_update('auto') & config_option_update(mode='auto')!
      const beforeTurboTurnIdx = client.frames.length
      await client.call('turn.submit', { session_key: 'turbo-s1', text: 'TURBO_RUN' })
      await client.next(
        f => client.frames.indexOf(f) >= beforeTurboTurnIdx && f.method === 'event' && f.params?.type === 'turn_end',
      )
      const turboApprovalRequests = client.frames
        .slice(beforeTurboTurnIdx)
        .filter(f => f.method === 'event' && f.params?.type === 'approval_request')
      expect(turboApprovalRequests).toHaveLength(0)
      expect(permissionOutcomes.at(-1)).toMatchObject({ outcome: 'selected', optionId: 'allow' })
      expect(bridgeHost.getBinding(sessionId)?.current_mode).toBe('turbo')

      // 4. Crash & reconnect via session/resume -> mock AGY resets to 'default', Xerxes MUST restore 'turbo' (sending 'auto' to AGY)!
      const beforeCrashIdx = client.frames.length
      activeHarness!.terminate(1)
      await client.next(
        f =>
          client.frames.indexOf(f) >= beforeCrashIdx &&
          f.method === 'event' &&
          f.params?.type === 'acp_status' &&
          (f.params.payload?.binding as Record<string, unknown> | undefined)?.connected === false,
      )
      setConfigCalls.length = 0
      const resumedTurbo = await client.call('acp.connect', {
        session_id: sessionId,
        agent_id: 'antigravity',
        resume_session_id: 'agy-turbo-1',
      })
      expect(resumedTurbo).toMatchObject({
        ok: true,
        binding: {
          external_session_id: 'agy-turbo-1',
          current_mode: 'turbo',
          current_mode_name: 'Turbo',
        },
      })
      expect(setConfigCalls).toEqual([{ configId: 'mode', value: 'auto' }])

      // 5. Switch back to Auto -> restores normal ACP permission prompts immediately!
      setConfigCalls.length = 0
      const setAuto = await client.call('acp.set_config_option', {
        session_id: sessionId,
        config_id: 'mode',
        value: 'auto',
      })
      expect(setAuto).toMatchObject({
        ok: true,
        binding: { current_mode: 'auto', current_mode_name: 'Auto' },
      })
      expect(setConfigCalls).toEqual([{ configId: 'mode', value: 'auto' }])

      const beforeAutoTurnIdx = client.frames.length
      await client.call('turn.submit', { session_key: 'turbo-s1', text: 'AUTO_PROMPT_CHECK' })
      const autoAppr = await client.next(
        f =>
          client.frames.indexOf(f) >= beforeAutoTurnIdx &&
          f.method === 'event' &&
          f.params?.type === 'approval_request',
      )
      await client.call('permission_response', {
        request_id: String(autoAppr.params!.payload!.request_id),
        response: 'approve',
      })
      await client.next(
        f => client.frames.indexOf(f) >= beforeAutoTurnIdx && f.method === 'event' && f.params?.type === 'turn_end',
      )

      // 6. Crash & reconnect while in Auto -> restores 'auto' cleanly (not silently reverting to 'default'!)
      const beforeCrash2Idx = client.frames.length
      activeHarness!.terminate(1)
      await client.next(
        f =>
          client.frames.indexOf(f) >= beforeCrash2Idx &&
          f.method === 'event' &&
          f.params?.type === 'acp_status' &&
          (f.params.payload?.binding as Record<string, unknown> | undefined)?.connected === false,
      )
      setConfigCalls.length = 0
      const resumedAuto = await client.call('acp.connect', {
        session_id: sessionId,
        agent_id: 'antigravity',
        resume_session_id: 'agy-turbo-1',
      })
      expect(resumedAuto).toMatchObject({
        ok: true,
        binding: { current_mode: 'auto', current_mode_name: 'Auto' },
      })
      expect(setConfigCalls).toEqual([{ configId: 'mode', value: 'auto' }])

      // 7. Mid-turn unlock: turn starts in Default and blocks on approval_request -> user switches to Turbo via `/acp mode turbo` ->
      // waiting card auto-approves immediately AND second command in the same turn runs with 0 prompts!
      await client.call('acp.set_mode', { session_id: sessionId, mode_id: 'default' })
      const beforeMidIdx = client.frames.length
      await client.call('turn.submit', { session_key: 'turbo-s1', text: 'MIDTURN_UNLOCK' })
      await client.next(
        f =>
          client.frames.indexOf(f) >= beforeMidIdx &&
          f.method === 'event' &&
          f.params?.type === 'approval_request',
      )
      const slashTurbo = await client.call('slash', { command: '/acp mode turbo' })
      expect(slashTurbo.ok).toBe(true)
      expect(String(slashTurbo.text)).toContain('Turbo')
      await client.next(
        f => client.frames.indexOf(f) >= beforeMidIdx && f.method === 'event' && f.params?.type === 'turn_end',
      )
      const midApprovals = client.frames
        .slice(beforeMidIdx)
        .filter(f => f.method === 'event' && f.params?.type === 'approval_request')
      // Only the initial pre-Turbo request appeared; the second command in the same turn was auto-approved!
      expect(midApprovals).toHaveLength(1)
      expect(
        client.frames
          .slice(beforeMidIdx)
          .some(f => f.method === 'event' && f.params?.type === 'approval_response'),
      ).toBe(true)
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('Antigravity subprocess env strips private vars of a parent agy sandbox but never touches SSH session vars or retries sign-in', async () => {
    // 1. A parent Antigravity sandbox's private env is stripped; SSH session vars and GEMINI_API_KEY pass through untouched.
    const seenOuterEnvs: Array<Readonly<Record<string, string | undefined>> | undefined> = []
    const outerHost = new DaemonAcpBridgeHost({
      discoverAgents: () => AGY_AGENT_ROSTER,
      env: {
        PATH: '/usr/bin:/bin',
        SSH_CONNECTION: '127.0.0.1 12345 127.0.0.1 22',
        SSH_AUTH_SOCK: '/tmp/ssh-agent.sock',
        AGY_ACP_FORCE_FILE_STORAGE: '1',
        ANTIGRAVITY_APP_DATA_DIR: '/tmp/outer-host/antigravity-acp',
        ANTIGRAVITY_CSRF_TOKEN: 'outer-csrf',
        GEMINI_HOME: '/tmp/outer-host',
        GEMINI_API_KEY: 'test-gemini-key',
      },
      createAcpClient: opts => {
        seenOuterEnvs.push(opts.env)
        const h = createInMemoryAcpTransport((frame, harness) => {
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { protocolVersion: 1 } })
          } else if (frame.method === 'session/new') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { sessionId: 'agy-outer-1' } })
          }
        })
        return new AcpStdioClient({ ...opts, transport: h.transport })
      },
    })
    const outerRes = await outerHost.connect({ sessionId: 'outer-1', agentId: 'antigravity', cwd: '/tmp' })
    expect(outerRes.ok).toBe(true)
    expect(seenOuterEnvs).toHaveLength(1)
    expect(seenOuterEnvs[0]?.AGY_ACP_FORCE_FILE_STORAGE).toBeUndefined()
    expect(seenOuterEnvs[0]?.ANTIGRAVITY_APP_DATA_DIR).toBeUndefined()
    expect(seenOuterEnvs[0]?.ANTIGRAVITY_CSRF_TOKEN).toBeUndefined()
    expect(seenOuterEnvs[0]?.GEMINI_HOME).toBeUndefined()
    expect(seenOuterEnvs[0]?.SSH_CONNECTION).toBe('127.0.0.1 12345 127.0.0.1 22')
    expect(seenOuterEnvs[0]?.SSH_AUTH_SOCK).toBe('/tmp/ssh-agent.sock')
    expect(seenOuterEnvs[0]?.GEMINI_API_KEY).toBe('test-gemini-key')
    await outerHost.disposeAll()

    // 2. A sign-in failure under SSH is reported once. Xerxes never retries with SSH vars
    //    removed to steer agy toward a different token store.
    const sshEnvs: Array<Readonly<Record<string, string | undefined>> | undefined> = []
    const sshHost = new DaemonAcpBridgeHost({
      discoverAgents: () => AGY_AGENT_ROSTER,
      env: {
        PATH: '/usr/bin:/bin',
        SSH_CONNECTION: '10.0.0.1 54321 10.0.0.2 22',
      },
      createAcpClient: opts => {
        sshEnvs.push(opts.env)
        const h = createInMemoryAcpTransport((frame, harness) => {
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { protocolVersion: 1 } })
          } else if (frame.method === 'session/new') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              error: { code: -32000, message: 'Authentication required' },
            })
          }
        })
        return new AcpStdioClient({ ...opts, transport: h.transport })
      },
    })
    const sshRes = await sshHost.connect({ sessionId: 'ssh-no-retry-1', agentId: 'antigravity', cwd: '/tmp' })
    expect(sshRes.ok).toBe(false)
    expect(sshEnvs).toHaveLength(1)
    expect(sshEnvs[0]?.SSH_CONNECTION).toBe('10.0.0.1 54321 10.0.0.2 22')
    if (!sshRes.ok) {
      expect(sshRes.error).toContain('agy acp --login')
      expect(sshRes.install_hint).toBeUndefined()
    }
  })
})
