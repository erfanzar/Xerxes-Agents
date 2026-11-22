// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { existsSync } from 'node:fs'
import { mkdir, mkdtemp, rm, symlink, writeFile } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  AcpStdioClient,
  normalizeAcpSessionUpdate,
  type AcpClientEvent,
  type AcpSpawnedProcess,
} from '../src/acp/client.js'
import {
  classifyAcpRuntimeFeatures,
  ClaudeStreamJsonRunner,
  DEFAULT_CLAUDE_CODE_MODELS,
  discoverExternalAcpAgents,
  discoverRuntimeSkillEntries,
  parseMarkdownSkillFrontmatter,
  type AcpBridgeStreamEvent,
  type ClaudeStreamJsonProcess,
} from '../src/acp/externalAgents.js'
import {
  DaemonAcpBridgeHost,
  extractAcpSessionModelAndModeState,
} from '../src/daemon/acpBridgeHost.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import type { LlmClient, LlmDelta } from '../src/llms/client.js'
import type { JsonRpcPayload } from '../src/protocol/jsonRpc.js'
import { GatewayClient } from '../src/ui/gatewayClient.js'
import type { GatewayEvent } from '../src/ui/gatewayTypes.js'
import {
  applyBtwEvent,
  askSideQuestion,
  getSideQuestionSliceForSession,
  parseSideQuestionInput,
  resetSideQuestionStore,
  setSideQuestionModel,
} from '../src/ui/app/sideQuestionStore.js'
import { $sideQuestionStore } from '../src/ui/app/sideQuestionStore.js'

const encoder = new TextEncoder()

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
  readonly id?: number
  readonly method?: string
  readonly params?: {
    readonly type?: string
    readonly payload?: JsonRpcPayload
  } & JsonRpcPayload
  readonly result?: JsonRpcPayload
  readonly error?: { readonly code: number; readonly message: string }
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

async function makeTempDir(prefix = 'xrx-acp-'): Promise<string> {
  const baseTmp = process.platform === 'win32' ? tmpdir() : (existsSync('/tmp') ? '/tmp' : tmpdir())
  const dir = await mkdtemp(join(baseTmp, prefix))
  tempRoots.push(dir)
  process.env.XERXES_HOME = join(dir, 'home')
  await mkdir(process.env.XERXES_HOME, { recursive: true })
  return dir
}

function makeFakeLlm(text = 'native-fallback'): LlmClient {
  return {
    async complete() {
      return { content: text, toolCalls: [] }
    },
    async *stream(): AsyncIterable<LlmDelta> {
      yield { content: text, finishReason: 'stop' }
    },
  }
}

describe('AcpStdioClient bidirectional JSON-RPC 2.0 transport', () => {
  test('handles initialize, session/new, session/load, all session/update shapes, agent requests, post-settle continuations, and clean dispose', async () => {
    const permissionResponses: Array<Record<string, unknown>> = []
    const fsResponses: Array<Record<string, unknown>> = []
    let readRequestedPath = ''
    let writeRequested: { path: string; content: string } | undefined

    const harness = createInMemoryAcpTransport((frame, h) => {
      if (frame.method === 'initialize') {
        h.pushAgentFrame({
          jsonrpc: '2.0',
          id: frame.id,
          result: {
            protocolVersion: 1,
            agentInfo: { name: 'antigravity-acp', version: '2.0.0' },
            agentCapabilities: { loadSession: true },
          },
        })
        return
      }
      if (frame.method === 'session/new') {
        h.pushAgentFrame({
          jsonrpc: '2.0',
          id: frame.id,
          result: {
            sessionId: 'agy-sess-1',
            modes: { currentModeId: 'code' },
            models: [{ id: 'gemini-3-pro' }],
          },
        })
        return
      }
      if (frame.method === 'session/load') {
        h.pushAgentFrame({
          jsonrpc: '2.0',
          id: frame.id,
          result: { sessionId: String((frame.params as Record<string, unknown>).sessionId) },
        })
        return
      }
      if (frame.method === 'session/prompt') {
        const promptId = frame.id
        // Stream every standard ACP session/update variant + agent-to-client requests!
        h.pushAgentFrame({
          jsonrpc: '2.0',
          method: 'session/update',
          params: {
            sessionId: 'agy-sess-1',
            update: { sessionUpdate: 'agent_thought_chunk', content: { type: 'text', text: 'Inspecting workspace...' } },
          },
        })
        h.pushAgentFrame({
          jsonrpc: '2.0',
          method: 'session/update',
          params: {
            sessionId: 'agy-sess-1',
            update: {
              sessionUpdate: 'plan',
              entries: [{ content: 'Read config', status: 'in_progress', priority: 'high' }],
            },
          },
        })
        h.pushAgentFrame({
          jsonrpc: '2.0',
          method: 'session/update',
          params: {
            sessionId: 'agy-sess-1',
            update: { sessionUpdate: 'current_mode_update', currentModeId: 'architect' },
          },
        })
        h.pushAgentFrame({
          jsonrpc: '2.0',
          method: 'session/update',
          params: {
            sessionId: 'agy-sess-1',
            update: {
              sessionUpdate: 'tool_call',
              toolCallId: 'tc-1',
              title: 'Edit app.ts',
              kind: 'edit',
              status: 'in_progress',
              rawInput: { path: 'app.ts' },
            },
          },
        })
        // Agent-to-client permission request
        h.pushAgentFrame({
          jsonrpc: '2.0',
          id: 'perm-1',
          method: 'session/request_permission',
          params: {
            sessionId: 'agy-sess-1',
            toolCall: { toolCallId: 'tc-1', title: 'Write app.ts', kind: 'edit' },
            options: [
              { optionId: 'allow-once', name: 'Allow once', kind: 'allow_once' },
              { optionId: 'reject-once', name: 'Reject', kind: 'reject_once' },
            ],
          },
        })
        // Agent-to-client fs requests
        h.pushAgentFrame({
          jsonrpc: '2.0',
          id: 'fs-read-1',
          method: 'fs/read_text_file',
          params: { sessionId: 'agy-sess-1', path: 'app.ts', line: 1, limit: 10 },
        })
        h.pushAgentFrame({
          jsonrpc: '2.0',
          id: 'fs-write-1',
          method: 'fs/write_text_file',
          params: { sessionId: 'agy-sess-1', path: 'out.ts', content: 'export const ok = true\n' },
        })
        setTimeout(() => {
          h.pushAgentFrame({
            jsonrpc: '2.0',
            method: 'session/update',
            params: {
              sessionId: 'agy-sess-1',
              update: {
                sessionUpdate: 'tool_call_update',
                toolCallId: 'tc-1',
                status: 'completed',
                content: [{ type: 'content', content: { type: 'text', text: 'Wrote out.ts' } }],
              },
            },
          })
          h.pushAgentFrame({
            jsonrpc: '2.0',
            method: 'session/update',
            params: {
              sessionId: 'agy-sess-1',
              update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'Done updating app.' } },
            },
          })
          h.pushAgentFrame({
            jsonrpc: '2.0',
            id: promptId,
            result: { stopReason: 'end_turn' },
          })
        }, 20)
        return
      }
      // Capture client's responses to agent-to-client requests!
      if (frame.id === 'perm-1') {
        permissionResponses.push(frame)
      } else if (frame.id === 'fs-read-1' || frame.id === 'fs-write-1') {
        fsResponses.push(frame)
      }
    })

    const allEvents: AcpClientEvent[] = []
    const client = new AcpStdioClient({
      command: 'antigravity-acp',
      transport: harness.transport,
      onEvent: ev => allEvents.push(ev),
      handlers: {
        requestPermission: async req => {
          expect(req.toolCallId).toBe('tc-1')
          return { outcome: 'selected', optionId: 'allow-once' }
        },
        readTextFile: async req => {
          readRequestedPath = req.path
          return { content: 'const x = 1\n' }
        },
        writeTextFile: async req => {
          writeRequested = { path: req.path, content: req.content }
        },
      },
    })

    const init = await client.initialize()
    expect(init.supportsLoadSession).toBe(true)
    expect(init.agentInfo?.name).toBe('antigravity-acp')
    const sentInit = harness.receivedFrames[0]!
    expect(sentInit).toMatchObject({
      jsonrpc: '2.0',
      method: 'initialize',
      params: {
        protocolVersion: 1,
        clientInfo: { name: 'xerxes-agents' },
        clientCapabilities: { fs: { readTextFile: true, writeTextFile: true }, terminal: false },
      },
    })

    const sess = await client.newSession({ cwd: '/workspace' })
    expect(sess.sessionId).toBe('agy-sess-1')

    const loaded = await client.loadSession({ sessionId: 'agy-sess-1', cwd: '/workspace' })
    expect(loaded.sessionId).toBe('agy-sess-1')

    const promptEvents: AcpClientEvent[] = []
    const res = await client.prompt({
      sessionId: 'agy-sess-1',
      text: 'Update app.ts',
      onEvent: ev => promptEvents.push(ev),
    })
    expect(res.stopReason).toBe('end_turn')
    expect(readRequestedPath).toBe('app.ts')
    expect(writeRequested).toEqual({ path: 'out.ts', content: 'export const ok = true\n' })
    expect(permissionResponses[0]).toMatchObject({
      jsonrpc: '2.0',
      id: 'perm-1',
      result: { outcome: { outcome: 'selected', optionId: 'allow-once' } },
    })
    expect(fsResponses.find(f => f.id === 'fs-read-1')).toMatchObject({
      result: { content: 'const x = 1\n' },
    })
    expect(fsResponses.find(f => f.id === 'fs-write-1')).toMatchObject({
      result: {},
    })

    expect(promptEvents.some(e => e.type === 'thought_delta' && e.text === 'Inspecting workspace...')).toBe(true)
    expect(promptEvents.some(e => e.type === 'plan' && e.entries[0]?.content === 'Read config')).toBe(true)
    expect(promptEvents.some(e => e.type === 'mode_update' && e.modeId === 'architect')).toBe(true)
    expect(promptEvents.some(e => e.type === 'tool_call' && e.toolCallId === 'tc-1')).toBe(true)
    expect(promptEvents.some(e => e.type === 'tool_call_update' && e.content === 'Wrote out.ts')).toBe(true)
    expect(promptEvents.some(e => e.type === 'text_delta' && e.text === 'Done updating app.')).toBe(true)

    // Antigravity-style post-settle continuation frame after session/prompt response returned!
    harness.pushAgentFrame({
      jsonrpc: '2.0',
      method: 'session/update',
      params: {
        sessionId: 'agy-sess-1',
        update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: ' Post-settle summary.' } },
      },
    })
    await Bun.sleep(15)
    expect(allEvents.some(e => e.type === 'continuation' && e.text === ' Post-settle summary.')).toBe(true)

    await client.dispose()
    expect(harness.killed()).toBe(true)
    expect(client.isDisposed).toBe(true)
  })

  test('defaults to safe deny on session/request_permission when no handler approves, cancels on AbortSignal, and rejects pending RPCs on dispose', async () => {
    const replies: Array<Record<string, unknown>> = []
    const harness = createInMemoryAcpTransport((frame, h) => {
      if (frame.method === 'initialize') {
        h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { protocolVersion: 1 } })
        return
      }
      if (frame.method === 'session/new') {
        h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { sessionId: 's-deny' } })
        return
      }
      if (frame.method === 'session/prompt') {
        h.pushAgentFrame({
          jsonrpc: '2.0',
          id: 'perm-deny',
          method: 'session/request_permission',
          params: {
            sessionId: 's-deny',
            toolCall: { toolCallId: 'tc-rm', title: 'Delete disk' },
            options: [
              { optionId: 'allow-1', name: 'Allow', kind: 'allow_once' },
              { optionId: 'deny-1', name: 'Reject', kind: 'reject_once' },
            ],
          },
        })
        // Never resolve prompt so we can test AbortSignal cancellation!
        return
      }
      if (frame.id === 'perm-deny') {
        replies.push(frame)
      }
    })

    const client = new AcpStdioClient({
      command: 'test-acp',
      transport: harness.transport,
    })

    await client.initialize()
    await client.newSession({ cwd: '/tmp' })

    const ac = new AbortController()
    const promptPromise = client.prompt({ sessionId: 's-deny', text: 'Run rm', signal: ac.signal })
    await Bun.sleep(20)
    expect(replies[0]).toMatchObject({
      id: 'perm-deny',
      result: { outcome: { outcome: 'selected', optionId: 'deny-1' } },
    })

    ac.abort()
    await expect(promptPromise).rejects.toThrow(/cancelled/i)
    await Bun.sleep(10)
    expect(harness.receivedFrames.some(f => f.method === 'session/cancel')).toBe(true)

    // Unanswered RPC rejects cleanly on dispose()
    const hanging = client.newSession({ cwd: '/tmp' })
    // Override transport handler behavior: trigger a new call whose response never arrives
    const harness2 = createInMemoryAcpTransport(() => {
      // intentionally never replies
    })
    const client2 = new AcpStdioClient({ command: 'hang-acp', transport: harness2.transport })
    const pendingInit = client2.initialize()
    await client2.dispose()
    await expect(pendingInit).rejects.toThrow(/disposed/i)
    await hanging.catch(() => undefined)
    await client.dispose()
  })
})

describe('discoverExternalAcpAgents priority order & custom config validation', () => {
  test('prefers XERXES_CLAUDE_ACP_CMD > claude-agent-acp > claude CLI, and XERXES_ANTIGRAVITY_ACP_CMD > antigravity-acp > agy > antigravity', () => {
    // 1. Env overrides win
    const withOverrides = discoverExternalAcpAgents({
      env: {
        XERXES_CLAUDE_ACP_CMD: '/custom/bin/my-claude-acp --stdio',
        XERXES_ANTIGRAVITY_ACP_CMD: '/custom/bin/my-agy-acp --mode acp',
      },
      binaryExists: p => p === '/custom/bin/my-claude-acp' || p === '/custom/bin/my-agy-acp',
      readConfigText: () => undefined,
    })
    expect(withOverrides[0]).toMatchObject({ id: 'xerxes', transport: 'native', available: true })
    expect(withOverrides.find(a => a.id === 'claude-code')).toMatchObject({
      transport: 'acp',
      available: true,
      binaryPath: '/custom/bin/my-claude-acp',
      args: ['--stdio'],
    })
    expect(withOverrides.find(a => a.id === 'antigravity')).toMatchObject({
      transport: 'acp',
      available: true,
      binaryPath: '/custom/bin/my-agy-acp',
      args: ['--mode', 'acp'],
    })

    // 2. Fallback to official `claude` CLI (`claude-stream-json`) and `agy` (`agy acp`)
    const withClis = discoverExternalAcpAgents({
      env: { HOME: '/Users/tester', PATH: '/usr/bin' },
      homeDir: '/Users/tester',
      binaryExists: p => p === '/Users/tester/.local/bin/claude' || p === '/Users/tester/.gemini/antigravity/bin/agy',
      readConfigText: () => undefined,
    })
    expect(withClis.find(a => a.id === 'claude-code')).toMatchObject({
      transport: 'claude-stream-json',
      available: true,
      binaryPath: '/Users/tester/.local/bin/claude',
    })
    expect(withClis.find(a => a.id === 'antigravity')).toMatchObject({
      transport: 'acp',
      available: true,
      binaryPath: '/Users/tester/.gemini/antigravity/bin/agy',
      args: ['acp'],
    })

    // 3. Missing binaries return available: false with actionable installHint (never fake available!)
    const missing = discoverExternalAcpAgents({
      env: { HOME: '/Users/nobody', PATH: '' },
      homeDir: '/Users/nobody',
      binaryExists: () => false,
      readConfigText: () => JSON.stringify([
        { id: 'xerxes', name: 'Hijack', command: '/bin/sh' }, // ignored: cannot shadow xerxes
        { id: 'bad-entry', name: 'Missing Command' }, // ignored: invalid
        { id: 'codex-acp', name: 'Codex ACP', command: 'codex-acp', provider: 'codex', args: ['--stdio'] },
      ]),
    })
    const claudeMissing = missing.find(a => a.id === 'claude-code')
    expect(claudeMissing?.available).toBe(false)
    expect(claudeMissing?.installHint).toContain('@zed-industries/claude-agent-acp')

    const agyMissing = missing.find(a => a.id === 'antigravity')
    expect(agyMissing?.available).toBe(false)
    expect(agyMissing?.installHint).toContain('XERXES_ANTIGRAVITY_ACP_CMD')

    const custom = missing.find(a => a.id === 'codex-acp')
    expect(custom).toMatchObject({
      id: 'codex-acp',
      name: 'Codex ACP',
      provider: 'codex',
      transport: 'acp',
      available: false,
      args: ['--stdio'],
    })
    expect(missing.find(a => a.id === 'xerxes')?.transport).toBe('native')
  })
})

describe('ClaudeStreamJsonRunner NDJSON streaming & --resume continuity', () => {
  test('translates stream-json events into unified AcpBridgeStreamEvent and passes --resume on turn 2', async () => {
    const spawnedCalls: Array<{ binary: string; args: readonly string[]; input: string }> = []
    let turn = 0

    const runner = new ClaudeStreamJsonRunner({
      binaryPath: '/Users/erfan/.local/bin/claude',
      cwd: '/workspace',
      permissionMode: 'auto',
      spawn: (binary, args, opts): ClaudeStreamJsonProcess => {
        turn += 1
        spawnedCalls.push({ binary, args, input: opts.input })
        const lines = turn === 1
          ? [
              { type: 'system', subtype: 'init', session_id: 'cc-sess-99', model: 'claude-sonnet-4-5' },
              { type: 'stream_event', event: { type: 'content_block_delta', delta: { type: 'thinking_delta', thinking: 'Planning read...' } } },
              { type: 'stream_event', event: { type: 'content_block_delta', delta: { type: 'text_delta', text: 'Checking file. ' } } },
              {
                type: 'assistant',
                session_id: 'cc-sess-99',
                message: {
                  content: [
                    { type: 'text', text: 'Checking file. ' }, // already streamed via delta; must NOT duplicate
                    { type: 'tool_use', id: 'tu-1', name: 'Read', input: { file_path: 'README.md' } },
                  ],
                },
              },
              {
                type: 'user',
                message: {
                  content: [{ type: 'tool_result', tool_use_id: 'tu-1', content: '# Xerxes\n' }],
                },
              },
              {
                type: 'assistant',
                session_id: 'cc-sess-99',
                message: { content: [{ type: 'text', text: 'README says Xerxes.' }] },
              },
              {
                type: 'result',
                subtype: 'success',
                session_id: 'cc-sess-99',
                result: 'README says Xerxes.',
                total_cost_usd: 0.012,
                usage: { input_tokens: 120, output_tokens: 45 },
              },
            ]
          : [
              { type: 'assistant', session_id: 'cc-sess-99', message: { content: [{ type: 'text', text: 'Turn 2 resumed cleanly.' }] } },
              { type: 'result', subtype: 'success', session_id: 'cc-sess-99' },
            ]

        const stdout = new ReadableStream<Uint8Array>({
          start(controller) {
            for (const row of lines) {
              controller.enqueue(encoder.encode(`${JSON.stringify(row)}\n`))
            }
            controller.close()
          },
        })
        return {
          stdout,
          stderr: null,
          exited: Promise.resolve(0),
          kill: () => undefined,
        }
      },
    })

    const turn1Events: AcpBridgeStreamEvent[] = []
    const res1 = await runner.prompt({
      text: 'Inspect README.md',
      onEvent: ev => turn1Events.push(ev),
    })
    expect(res1).toEqual({ cancelled: false, externalSessionId: 'cc-sess-99', stopReason: 'end_turn' })
    expect(spawnedCalls[0]!.args).toContain('--output-format')
    expect(spawnedCalls[0]!.args).toContain('stream-json')
    expect(spawnedCalls[0]!.args).toContain('--permission-mode')
    expect(spawnedCalls[0]!.args).toContain('acceptEdits')
    expect(spawnedCalls[0]!.args).not.toContain('--resume')

    expect(turn1Events.filter(e => e.type === 'text_delta').map(e => e.text).join('')).toBe(
      'Checking file. README says Xerxes.',
    )
    expect(turn1Events.some(e => e.type === 'thought_delta' && e.text === 'Planning read...')).toBe(true)
    expect(turn1Events.some(e => e.type === 'tool_call' && e.toolCallId === 'tu-1' && e.name === 'Read')).toBe(true)
    expect(turn1Events.some(e => e.type === 'tool_result' && e.toolCallId === 'tu-1' && e.output.includes('Xerxes'))).toBe(true)
    expect(turn1Events.some(e => e.type === 'usage' && e.inputTokens === 120 && e.outputTokens === 45)).toBe(true)

    const turn2Events: AcpBridgeStreamEvent[] = []
    const res2 = await runner.prompt({
      text: 'Follow up',
      onEvent: ev => turn2Events.push(ev),
    })
    expect(res2.externalSessionId).toBe('cc-sess-99')
    expect(spawnedCalls[1]!.args).toContain('--resume')
    expect(spawnedCalls[1]!.args).toContain('cc-sess-99')
    expect(turn2Events.filter(e => e.type === 'text_delta').map(e => e.text).join('')).toBe('Turn 2 resumed cleanly.')
  })

  test('kills active subprocess cleanly on AbortSignal cancellation', async () => {
    let killed = false
    let resolveExit: ((code: number | null) => void) | undefined
    const runner = new ClaudeStreamJsonRunner({
      binaryPath: '/usr/local/bin/claude',
      cwd: '/workspace',
      spawn: (): ClaudeStreamJsonProcess => ({
        stdout: new ReadableStream<Uint8Array>({
          start() {
            // stays open until cancelled
          },
        }),
        stderr: null,
        exited: new Promise<number | null>(resolve => {
          resolveExit = resolve
        }),
        kill: () => {
          killed = true
          resolveExit?.(143)
        },
      }),
    })

    const ac = new AbortController()
    const run = runner.prompt({ text: 'Slow task', signal: ac.signal, onEvent: () => undefined })
    await Bun.sleep(15)
    ac.abort()
    const res = await run
    expect(killed).toBe(true)
    expect(res.cancelled).toBe(true)
    expect(res.stopReason).toBe('cancelled')
  })
})

describe('DaemonServer ACP Bridge RPCs, transparent turn routing, permission bridging, /acp slash, and GatewayClient', () => {
  test('connects Antigravity ACP, bridges permission approval & turn events, persists transcript, supports /acp & acp.disconnect, and cleans up on stop', async () => {
    const root = await makeTempDir('xrx-acp-daemon-')
    await writeFile(join(root, 'sample.txt'), 'hello from workspace\n', 'utf8')

    let activeHarness: InMemoryAcpHarness | undefined
    const bridgeHost = new DaemonAcpBridgeHost({
      discoverAgents: () => [
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
        {
          id: 'claude-code',
          name: 'Claude Code',
          provider: 'claude-code',
          transport: 'acp',
          available: false,
          installHint: 'Install Claude Code (claude) or @zed-industries/claude-agent-acp',
          description: 'Missing Claude Code',
        },
      ],
      createAcpClient: opts => {
        let pendingPromptId: unknown
        activeHarness = createInMemoryAcpTransport((frame, h) => {
          if (frame.method === 'initialize') {
            h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { protocolVersion: 1 } })
            return
          }
          if (frame.method === 'session/new') {
            h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { sessionId: 'ext-agy-42' } })
            return
          }
          if (frame.method === 'session/prompt') {
            pendingPromptId = frame.id
            h.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: 'ext-agy-42',
                update: { sessionUpdate: 'agent_thought_chunk', content: { type: 'text', text: 'Checking sample.txt' } },
              },
            })
            h.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: 'ext-agy-42',
                update: {
                  sessionUpdate: 'tool_call',
                  toolCallId: 'agy-tc-1',
                  name: 'WriteFile',
                  title: 'Write greeting.txt',
                  rawInput: { path: 'greeting.txt' },
                },
              },
            })
            h.pushAgentFrame({
              jsonrpc: '2.0',
              id: 'agy-perm-1',
              method: 'session/request_permission',
              params: {
                sessionId: 'ext-agy-42',
                toolCall: { toolCallId: 'agy-tc-1', name: 'WriteFile', title: 'Write greeting.txt' },
                options: [
                  { optionId: 'opt-allow', name: 'Allow once', kind: 'allow_once' },
                  { optionId: 'opt-deny', name: 'Reject', kind: 'reject_once' },
                ],
              },
            })
            return
          }
          if (frame.id === 'agy-perm-1') {
            const approved =
              ((frame.result as Record<string, unknown> | undefined)?.outcome as Record<string, unknown> | undefined)
                ?.optionId === 'opt-allow'
            h.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: 'ext-agy-42',
                update: {
                  sessionUpdate: 'tool_call_update',
                  toolCallId: 'agy-tc-1',
                  status: approved ? 'completed' : 'failed',
                  content: approved ? 'Wrote greeting.txt' : 'Denied by user',
                },
              },
            })
            h.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: 'ext-agy-42',
                update: {
                  sessionUpdate: 'agent_message_chunk',
                  content: { type: 'text', text: approved ? 'Created greeting.txt via Antigravity ACP.' : 'Denied.' },
                },
              },
            })
            h.pushAgentFrame({
              jsonrpc: '2.0',
              id: pendingPromptId,
              result: { stopReason: 'end_turn' },
            })
          }
        })
        return new AcpStdioClient({ ...opts, transport: activeHarness.transport })
      },
    })

    const runtime = new InMemoryDaemonRuntime(
      new AgentTurnRunner({ llm: makeFakeLlm('native-reply'), model: 'gpt-4.1' }),
      {
        currentProjectDirectory: root,
        model: 'gpt-4.1',
        permissionMode: 'manual',
        sessionDirectory: join(root, 'sessions'),
      },
    )
    const socketPath = join(root, 'acp-daemon.sock')
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
        session_key: 'acp-sess-1',
        project_dir: root,
      })
      const sessionId = String(init.session_id)

      // 1. List agents via `acp.agents`
      const listed = await client.call('acp.agents', { session_id: sessionId })
      expect(listed.ok).toBe(true)
      expect(listed.active).toBeNull()
      expect((listed.agents as Array<{ id: string }>).map(a => a.id)).toEqual([
        'xerxes',
        'antigravity',
        'claude-code',
      ])

      // 2. Attempting to connect an unavailable agent fails with typed error + install_hint
      const missingRes = await client.call('acp.connect', {
        session_id: sessionId,
        agent_id: 'claude-code',
      })
      expect(missingRes).toMatchObject({
        ok: false,
        install_hint: expect.stringContaining('claude-agent-acp'),
      })

      // 3. Connect Antigravity ACP -> stores binding & broadcasts `acp_status`
      const connected = await client.call('acp.connect', {
        session_id: sessionId,
        agent_id: 'antigravity',
      })
      expect(connected).toMatchObject({
        ok: true,
        binding: {
          session_id: sessionId,
          agent_id: 'antigravity',
          provider: 'antigravity',
          transport: 'acp',
          external_session_id: 'ext-agy-42',
          connected: true,
        },
      })
      const statusEvent = await client.next(
        f => f.method === 'event' && f.params?.type === 'acp_status',
      )
      expect(statusEvent.params?.payload).toMatchObject({
        session_id: sessionId,
        binding: { agent_id: 'antigravity', external_session_id: 'ext-agy-42' },
      })

      // 4. Submit normal turn -> routes transparently through Antigravity ACP + bridges approval!
      const submitRes = await client.call('turn.submit', {
        session_key: 'acp-sess-1',
        text: 'Create greeting.txt',
      })
      expect(submitRes.ok).toBe(true)

      const approvalFrame = await client.next(
        f => f.method === 'event' && f.params?.type === 'approval_request',
      )
      const approvalRequestId = String(approvalFrame.params?.payload?.request_id ?? approvalFrame.params?.payload?.id)
      expect(approvalRequestId.length).toBeGreaterThan(0)

      const permReply = await client.call('permission_response', {
        request_id: approvalRequestId,
        response: 'approve',
      })
      expect(permReply.ok).toBe(true)

      const turnEnd = await client.next(
        f => f.method === 'event' && f.params?.type === 'turn_end',
      )
      expect(turnEnd.params?.payload).toMatchObject({ session_id: sessionId, stop_reason: 'end_turn' })

      expect(
        client.frames.some(
          f => f.method === 'event' && f.params?.type === 'think_part' && f.params.payload?.think === 'Checking sample.txt',
        ),
      ).toBe(true)
      expect(
        client.frames.some(
          f =>
            f.method === 'event' &&
            f.params?.type === 'text_part' &&
            f.params.payload?.text === 'Created greeting.txt via Antigravity ACP.',
        ),
      ).toBe(true)
      expect(
        client.frames.some(
          f => f.method === 'event' && f.params?.type === 'tool_result' && f.params.payload?.return_value === 'Wrote greeting.txt',
        ),
      ).toBe(true)

      // Verify transcript persisted onto session so history and /btw see the ACP turn!
      const liveSession = runtime.listSessions().find(s => s.id === sessionId)!
      expect(liveSession.messages.some(m => m.role === 'user' && m.content === 'Create greeting.txt')).toBe(true)
      expect(
        liveSession.messages.some(
          m => m.role === 'assistant' && m.content === 'Created greeting.txt via Antigravity ACP.',
        ),
      ).toBe(true)

      // 5. `/acp status` and `/acp disconnect` slash commands
      const slashStatus = await client.call('slash', { command: '/acp status' })
      expect(slashStatus.ok).toBe(true)
      expect(String(slashStatus.text)).toContain('Antigravity')

      const slashDisconnect = await client.call('slash', { command: '/acp disconnect' })
      expect(slashDisconnect.ok).toBe(true)
      expect(activeHarness?.killed()).toBe(true)
      expect(bridgeHost.getBinding(sessionId)).toBeNull()

      // 6. Verify GatewayClient ACP methods + live acp_status event over the same socket!
      const gw = new GatewayClient({
        projectDir: root,
        externalSocketPath: socketPath,
        sessionKey: 'gw-acp-client',
      })
      const gwEvents: GatewayEvent[] = []
      gw.on('event', (ev: GatewayEvent) => gwEvents.push(ev))
      await gw.start()
      try {
        await gw.request('session.create', {})
        const gwAgents = await gw.acpAgents()
        expect(gwAgents.ok).toBe(true)
        expect(gwAgents.agents.some(a => a.id === 'antigravity')).toBe(true)
        const gwConn = await gw.acpConnect({ agent_id: 'antigravity' })
        expect(gwConn.ok).toBe(true)
        expect(gwConn.binding?.agent_id).toBe('antigravity')
        const gwCancel = await gw.acpCancel()
        expect(gwCancel.ok).toBe(true)
        const gwDisc = await gw.acpDisconnect()
        expect(gwDisc.ok).toBe(true)
        expect(gwDisc.binding).toBeNull()
        await Bun.sleep(20)
        expect(gwEvents.some(ev => ev.type === 'acp_status')).toBe(true)
      } finally {
        gw.close()
      }
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('acp.cancel aborts an in-flight external ACP turn cleanly and sideQuestionStore handles model selection & tool badges', async () => {
    const root = await makeTempDir('xrx-acp-cancel-')
    let cancelReceived = false
    const bridgeHost = new DaemonAcpBridgeHost({
      discoverAgents: () => [
        {
          id: 'xerxes',
          name: 'Xerxes',
          provider: 'xerxes',
          transport: 'native',
          available: true,
          description: 'Native',
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
          description: 'Antigravity',
        },
      ],
      createAcpClient: opts => {
        const h = createInMemoryAcpTransport((frame, harness) => {
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { protocolVersion: 1 } })
          } else if (frame.method === 'session/new') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { sessionId: 'agy-slow-1' } })
          } else if (frame.method === 'session/cancel') {
            cancelReceived = true
          }
          // session/prompt intentionally hangs until cancelled!
        })
        return new AcpStdioClient({ ...opts, transport: h.transport })
      },
    })

    const runtime = new InMemoryDaemonRuntime(
      new AgentTurnRunner({ llm: makeFakeLlm(), model: 'gpt-4.1' }),
      {
        currentProjectDirectory: root,
        model: 'gpt-4.1',
        sessionDirectory: join(root, 'sessions'),
      },
    )
    const socketPath = join(root, 'acp-cancel.sock')
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
      const init = await client.call('initialize', { session_key: 'cancel-sess', project_dir: root })
      const sessionId = String(init.session_id)
      const conn = await client.call('acp.connect', { session_id: sessionId, agent_id: 'antigravity' })
      expect(conn.ok).toBe(true)

      await client.call('turn.submit', { session_key: 'cancel-sess', text: 'Long running ACP task' })
      await client.next(f => f.method === 'event' && f.params?.type === 'turn_begin')

      const cancelRes = await client.call('acp.cancel', { session_id: sessionId })
      expect(cancelRes).toEqual({ ok: true, cancelled: true })

      const turnEnd = await client.next(f => f.method === 'event' && f.params?.type === 'turn_end')
      expect(turnEnd.params?.payload).toMatchObject({ session_id: sessionId, cancelled: true })
      expect(cancelReceived).toBe(true)
    } finally {
      client.close()
      await server.stop()
    }

    // Verify OpenTUI sideQuestionStore model selection, --model parsing, and tool badge folding
    resetSideQuestionStore()
    expect(parseSideQuestionInput('--model gemini-2.5-flash Explain ACP')).toEqual({
      model: 'gemini-2.5-flash',
      question: 'Explain ACP',
    })
    expect(parseSideQuestionInput('-m=gpt-4.1-mini How does cancel work?')).toEqual({
      model: 'gpt-4.1-mini',
      question: 'How does cancel work?',
    })

    setSideQuestionModel('gemini-2.5-flash', 'ui-s1')
    expect(getSideQuestionSliceForSession($sideQuestionStore.get(), 'ui-s1').selectedModel).toBe('gemini-2.5-flash')

    const seenRpc: Array<{ method: string; params: Record<string, unknown> }> = []
    const res = await askSideQuestion(
      async <T,>(method: string, params: Record<string, unknown> = {}) => {
        seenRpc.push({ method, params })
        applyBtwEvent(
          {
            session_id: 'ui-s1',
            request_id: String(params.request_id),
            kind: 'tool',
            tool: { name: 'ReadFile', target: 'server.ts', status: 'running' },
          },
          'ui-s1',
        )
        applyBtwEvent(
          {
            session_id: 'ui-s1',
            request_id: String(params.request_id),
            kind: 'tool',
            tool: { name: 'ReadFile', target: 'server.ts', status: 'done' },
          },
          'ui-s1',
        )
        return {
          ok: true,
          request_id: String(params.request_id),
          session_id: 'ui-s1',
          question: String(params.question),
          answer: 'Checked server.ts.',
          model: String(params.model),
        } as T
      },
      { sessionId: 'ui-s1', question: 'Where is handleAcpRpc?' },
    )
    expect(res?.ok).toBe(true)
    expect(seenRpc[0]?.params).toMatchObject({
      session_id: 'ui-s1',
      question: 'Where is handleAcpRpc?',
      model: 'gemini-2.5-flash',
    })
    const entry = getSideQuestionSliceForSession($sideQuestionStore.get(), 'ui-s1').entries[0]!
    expect(entry.model).toBe('gemini-2.5-flash')
    expect(entry.tools).toEqual([{ name: 'ReadFile', target: 'server.ts', status: 'done' }])
    resetSideQuestionStore()
  })
})

describe('External ACP model & mode discovery, fallback switching, Claude --model, and daemon RPCs', () => {
  test('AcpStdioClient setModel & setMode support dual session/set_model / session/set_mode and session/set_config_option fallbacks and normalize update events', async () => {
    // 1. No configId -> tries session/set_model, falls back to session/set_config_option on -32601
    const fallbackMethods: Array<{ method: string; params: unknown }> = []
    const fallbackHarness = createInMemoryAcpTransport((frame, h) => {
      if (typeof frame.method === 'string') {
        fallbackMethods.push({ method: frame.method, params: frame.params })
      }
      if (frame.method === 'initialize') {
        h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { protocolVersion: 1 } })
        return
      }
      if (frame.method === 'session/new') {
        h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { sessionId: 'sess-fb' } })
        return
      }
      if (frame.method === 'session/set_model' || frame.method === 'session/set_mode') {
        h.pushAgentFrame({
          jsonrpc: '2.0',
          id: frame.id,
          error: { code: -32601, message: 'Method not found' },
        })
        return
      }
      if (frame.method === 'session/set_config_option') {
        const p = frame.params as Record<string, unknown>
        h.pushAgentFrame({
          jsonrpc: '2.0',
          id: frame.id,
          result: {
            configOptions: [
              { id: p.configId, category: p.configId, currentValue: p.value, options: [] },
            ],
          },
        })
      }
    })

    const client1 = new AcpStdioClient({ command: 'agy', transport: fallbackHarness.transport })
    await client1.initialize()
    await client1.newSession({ cwd: '/workspace' })

    const modelRes = await client1.setModel({ sessionId: 'sess-fb', modelId: 'gemini-4-carbon-high' })
    expect(Array.isArray(modelRes.configOptions)).toBe(true)
    expect(fallbackMethods.map(m => m.method)).toEqual([
      'initialize',
      'session/new',
      'session/set_model',
      'session/set_config_option',
    ])
    expect(fallbackMethods[3]?.params).toEqual({
      sessionId: 'sess-fb',
      configId: 'model',
      value: 'gemini-4-carbon-high',
    })

    const modeRes = await client1.setMode({ sessionId: 'sess-fb', modeId: 'plan' })
    expect(Array.isArray(modeRes.configOptions)).toBe(true)
    expect(fallbackMethods.slice(4).map(m => m.method)).toEqual([
      'session/set_mode',
      'session/set_config_option',
    ])
    expect(fallbackMethods[5]?.params).toEqual({
      sessionId: 'sess-fb',
      configId: 'mode',
      value: 'plan',
    })
    await client1.dispose()

    // 2. Explicit configId -> tries session/set_config_option first, falls back to session/set_model / session/set_mode on -32601,
    //    and propagates non-(-32601) errors without swallowing!
    const reverseMethods: Array<{ method: string; params: unknown }> = []
    const reverseHarness = createInMemoryAcpTransport((frame, h) => {
      if (typeof frame.method === 'string') {
        reverseMethods.push({ method: frame.method, params: frame.params })
      }
      if (frame.method === 'initialize') {
        h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { protocolVersion: 1 } })
        return
      }
      if (frame.method === 'session/new') {
        h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { sessionId: 'sess-rev' } })
        return
      }
      if (frame.method === 'session/set_config_option') {
        const p = frame.params as Record<string, unknown>
        if (p.value === 'invalid-model') {
          h.pushAgentFrame({
            jsonrpc: '2.0',
            id: frame.id,
            error: { code: -32602, message: 'Unknown model id' },
          })
          return
        }
        h.pushAgentFrame({
          jsonrpc: '2.0',
          id: frame.id,
          error: { code: -32601, message: 'Method not found' },
        })
        return
      }
      if (frame.method === 'session/set_model' || frame.method === 'session/set_mode') {
        h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { ok: true } })
      }
    })

    const client2 = new AcpStdioClient({ command: 'claude-agent-acp', transport: reverseHarness.transport })
    await client2.initialize()
    await client2.newSession({ cwd: '/workspace' })

    await client2.setModel({ sessionId: 'sess-rev', modelId: 'claude-opus-4-6', configId: 'agy.model' })
    await client2.setMode({ sessionId: 'sess-rev', modeId: 'architect', configId: 'agy.mode' })
    expect(reverseMethods.slice(2).map(m => m.method)).toEqual([
      'session/set_config_option',
      'session/set_model',
      'session/set_config_option',
      'session/set_mode',
    ])
    expect(reverseMethods[2]?.params).toEqual({
      sessionId: 'sess-rev',
      configId: 'agy.model',
      value: 'claude-opus-4-6',
    })

    await expect(
      client2.setModel({ sessionId: 'sess-rev', modelId: 'invalid-model', configId: 'agy.model' }),
    ).rejects.toThrow(/Unknown model id/)
    await client2.dispose()

    // 3. normalizeAcpSessionUpdate handles current_mode_update, current_model_update, and config_option_update
    expect(
      normalizeAcpSessionUpdate('s1', {
        sessionUpdate: 'current_mode_update',
        currentModeId: 'plan',
        currentModeName: 'Planning',
      }),
    ).toEqual([{ type: 'mode_update', sessionId: 's1', modeId: 'plan', modeName: 'Planning' }])

    expect(
      normalizeAcpSessionUpdate('s1', {
        sessionUpdate: 'current_model_update',
        currentModelId: 'gemini-4-carbon-high',
        currentModelName: 'Gemini 4 Carbon (High)',
      }),
    ).toEqual([
      {
        type: 'model_update',
        sessionId: 's1',
        modelId: 'gemini-4-carbon-high',
        modelName: 'Gemini 4 Carbon (High)',
      },
    ])

    const cfgEvents = normalizeAcpSessionUpdate('s1', {
      sessionUpdate: 'config_option_update',
      configOptions: [
        {
          id: 'model',
          category: 'model',
          currentValue: 'gemini-4-flash',
          options: [{ value: 'gemini-4-flash', name: 'Gemini 4 Flash' }],
        },
      ],
    })
    expect(cfgEvents).toHaveLength(1)
    expect(cfgEvents[0]).toMatchObject({
      type: 'config_update',
      sessionId: 's1',
      configOptions: [{ id: 'model', category: 'model', currentValue: 'gemini-4-flash' }],
    })
  })

  test('extractAcpSessionModelAndModeState parses models, modes, and flat/grouped configOptions cleanly', () => {
    // Legacy models + modes shape
    const legacy = extractAcpSessionModelAndModeState({
      models: {
        currentModelId: 'claude-sonnet-4-6',
        availableModels: [
          { modelId: 'claude-opus-4-6', name: 'Claude Opus 4.6', description: 'Flagship' },
          { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6' },
        ],
      },
      modes: {
        currentModeId: 'code',
        availableModes: [
          { id: 'code', name: 'Code', description: 'Edit files' },
          { id: 'ask', name: 'Ask' },
        ],
      },
    })
    expect(legacy).toEqual({
      current_model: 'claude-sonnet-4-6',
      current_model_name: 'Claude Sonnet 4.6',
      models: [
        { id: 'claude-opus-4-6', name: 'Claude Opus 4.6', description: 'Flagship' },
        { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6' },
      ],
      current_mode: 'code',
      current_mode_name: 'Code',
      modes: [
        { id: 'code', name: 'Code', description: 'Edit files' },
        { id: 'ask', name: 'Ask' },
      ],
    })

    // ACP configOptions with grouped model options + flat mode options + custom config IDs
    const fromConfig = extractAcpSessionModelAndModeState({
      raw: {
        configOptions: [
          {
            id: 'agy_model',
            name: 'Model',
            category: 'model',
            type: 'select',
            currentValue: 'gemini-4-carbon-high',
            options: [
              {
                group: 'gemini',
                name: 'Gemini',
                options: [
                  {
                    value: 'gemini-4-carbon-high',
                    name: 'Gemini 4 Carbon (High)',
                    description: 'Deep reasoning',
                  },
                  { value: 'gemini-4-flash', name: 'Gemini 4 Flash' },
                ],
              },
              {
                group: 'claude',
                name: 'Claude',
                options: [{ value: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6' }],
              },
            ],
          },
          {
            id: 'mode',
            name: 'Mode',
            category: 'mode',
            type: 'select',
            currentValue: 'agent',
            options: [
              { value: 'agent', name: 'Agent', description: 'Autonomous edits' },
              { value: 'plan', name: 'Plan', description: 'Design first' },
            ],
          },
        ],
      },
    })
    expect(fromConfig).toEqual({
      current_model: 'gemini-4-carbon-high',
      current_model_name: 'Gemini 4 Carbon (High)',
      model_config_id: 'agy_model',
      models: [
        { id: 'gemini-4-carbon-high', name: 'Gemini 4 Carbon (High)', description: 'Deep reasoning' },
        { id: 'gemini-4-flash', name: 'Gemini 4 Flash' },
        { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6' },
      ],
      current_mode: 'agent',
      current_mode_name: 'Agent',
      mode_config_id: 'mode',
      modes: [
        { id: 'agent', name: 'Agent', description: 'Autonomous edits' },
        { id: 'plan', name: 'Plan', description: 'Design first' },
      ],
      config_options: [
        {
          id: 'agy_model',
          name: 'Model',
          category: 'model',
          type: 'select',
          current_value: 'gemini-4-carbon-high',
          options: [
            { value: 'gemini-4-carbon-high', name: 'Gemini 4 Carbon (High)', description: 'Deep reasoning' },
            { value: 'gemini-4-flash', name: 'Gemini 4 Flash' },
            { value: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6' },
          ],
        },
        {
          id: 'mode',
          name: 'Mode',
          category: 'mode',
          type: 'select',
          current_value: 'agent',
          options: [
            { value: 'agent', name: 'Agent', description: 'Autonomous edits' },
            { value: 'plan', name: 'Plan', description: 'Design first' },
          ],
        },
      ],
    })
  })

  test('ClaudeStreamJsonRunner exposes DEFAULT_CLAUDE_CODE_MODELS, passes --model when set, and emits model on system init frames', async () => {
    expect(DEFAULT_CLAUDE_CODE_MODELS.map(m => m.id)).toEqual([
      'claude-opus-4-6',
      'claude-sonnet-4-6',
      'claude-haiku-4-5',
      'opus',
      'sonnet',
      'haiku',
    ])

    const spawnedArgs: Array<readonly string[]> = []
    let turn = 0
    const runner = new ClaudeStreamJsonRunner({
      binaryPath: '/usr/local/bin/claude',
      cwd: '/workspace',
      spawn: (_bin, args): ClaudeStreamJsonProcess => {
        turn += 1
        spawnedArgs.push([...args])
        const reportedModel = turn === 1 ? 'claude-sonnet-4-6' : 'claude-opus-4-6'
        const rows = [
          { type: 'system', subtype: 'init', session_id: 'cc-model-sess', model: reportedModel },
          { type: 'assistant', session_id: 'cc-model-sess', message: { content: [{ type: 'text', text: `Using ${reportedModel}` }] } },
          { type: 'result', subtype: 'success', session_id: 'cc-model-sess' },
        ]
        return {
          stdout: new ReadableStream<Uint8Array>({
            start(controller) {
              for (const row of rows) {
                controller.enqueue(encoder.encode(`${JSON.stringify(row)}\n`))
              }
              controller.close()
            },
          }),
          stderr: null,
          exited: Promise.resolve(0),
          kill: () => undefined,
        }
      },
    })

    expect(runner.getModel()).toBeUndefined()
    const host = new DaemonAcpBridgeHost({
      discoverAgents: () => [
        {
          id: 'claude-code',
          name: 'Claude Code',
          provider: 'claude-code',
          transport: 'claude-stream-json',
          available: true,
          command: '/usr/local/bin/claude',
          binaryPath: '/usr/local/bin/claude',
          description: 'Claude CLI',
        },
      ],
      createClaudeRunner: () => runner,
    })

    const conn = await host.connect({ sessionId: 'cc-s1', agentId: 'claude-code', cwd: '/workspace' })
    expect(conn.ok).toBe(true)
    if (!conn.ok) return
    expect(conn.binding.models?.map(m => m.id)).toEqual(DEFAULT_CLAUDE_CODE_MODELS.map(m => m.id))

    // Turn 1 without explicit override -> no --model flag, captures model from system init
    const t1Events: AcpBridgeStreamEvent[] = []
    await host.runPrompt({ sessionId: 'cc-s1', text: 'Hi', onEvent: ev => t1Events.push(ev) })
    expect(spawnedArgs[0]).not.toContain('--model')
    expect(t1Events.some(e => e.type === 'session_init' && e.model === 'claude-sonnet-4-6')).toBe(true)
    expect(host.getBinding('cc-s1')?.current_model).toBe('claude-sonnet-4-6')
    expect(host.getBinding('cc-s1')?.current_model_name).toBe('Claude Sonnet 4.6')

    // Switch model via host.setModel -> passes --model on turn 2 and updates binding
    const switched = await host.setModel('cc-s1', 'claude-opus-4-6')
    expect(switched.ok).toBe(true)
    if (switched.ok) {
      expect(switched.binding.current_model).toBe('claude-opus-4-6')
      expect(switched.binding.current_model_name).toBe('Claude Opus 4.6')
    }
    expect(runner.getModel()).toBe('claude-opus-4-6')

    const t2Events: AcpBridgeStreamEvent[] = []
    await host.runPrompt({ sessionId: 'cc-s1', text: 'Next', onEvent: ev => t2Events.push(ev) })
    expect(spawnedArgs[1]).toContain('--model')
    expect(spawnedArgs[1]).toContain('claude-opus-4-6')
    expect(t2Events.some(e => e.type === 'session_init' && e.model === 'claude-opus-4-6')).toBe(true)

    // Mode switching is explicitly unsupported on claude-stream-json
    const modeErr = await host.setMode('cc-s1', 'plan')
    expect(modeErr.ok).toBe(false)

    // Clearing model on runner removes --model
    runner.setModel(undefined)
    expect(runner.getModel()).toBeUndefined()
    await host.disposeAll()
  })

  test('DaemonServer supports acp.set_model, acp.set_mode, /acp model & /acp mode slash, live config_option_update, and GatewayClient methods', async () => {
    const root = await makeTempDir('xrx-acp-models-daemon-')
    let activeHarness: InMemoryAcpHarness | undefined
    let activeModel = 'gemini-4-carbon-high'
    let activeMode = 'agent'

    const makeConfigOptions = () => [
      {
        id: 'agy_model',
        name: 'Model',
        category: 'model',
        type: 'select',
        currentValue: activeModel,
        options: [
          {
            group: 'google',
            name: 'Google',
            options: [
              { value: 'gemini-4-carbon-high', name: 'Gemini 4 Carbon (High)', description: 'Deep reasoning' },
              { value: 'gemini-4-flash', name: 'Gemini 4 Flash', description: 'Fast' },
            ],
          },
        ],
      },
      {
        id: 'agy_mode',
        name: 'Mode',
        category: 'mode',
        type: 'select',
        currentValue: activeMode,
        options: [
          { value: 'agent', name: 'Agent', description: 'Autonomous coding' },
          { value: 'plan', name: 'Planning', description: 'Plan before edits' },
        ],
      },
    ]

    const bridgeHost = new DaemonAcpBridgeHost({
      discoverAgents: () => [
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
      ],
      createAcpClient: opts => {
        activeHarness = createInMemoryAcpTransport((frame, h) => {
          if (frame.method === 'initialize') {
            h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { protocolVersion: 1 } })
            return
          }
          if (frame.method === 'session/new') {
            h.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                sessionId: '39c61d78',
                configOptions: makeConfigOptions(),
              },
            })
            return
          }
          if (frame.method === 'session/set_config_option') {
            const p = frame.params as Record<string, unknown>
            if (p.configId === 'agy_model' && typeof p.value === 'string') {
              activeModel = p.value
            } else if (p.configId === 'agy_mode' && typeof p.value === 'string') {
              activeMode = p.value
            }
            h.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: { configOptions: makeConfigOptions() },
            })
          }
        })
        return new AcpStdioClient({ ...opts, transport: activeHarness.transport })
      },
    })

    const runtime = new InMemoryDaemonRuntime(
      new AgentTurnRunner({ llm: makeFakeLlm('native'), model: 'gpt-6.2-astra' }),
      {
        currentProjectDirectory: root,
        model: 'gpt-6.2-astra',
        sessionDirectory: join(root, 'sessions'),
      },
    )
    const socketPath = join(root, 'acp-models.sock')
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
        session_key: 'acp-models-sess',
        project_dir: root,
      })
      const sessionId = String(init.session_id)

      // Calling acp.set_model before connecting fails cleanly
      const notConnected = await client.call('acp.set_model', {
        session_id: sessionId,
        model_id: 'gemini-4-flash',
      })
      expect(notConnected.ok).toBe(false)

      // Connect Antigravity ACP -> binding immediately carries models, modes, and config IDs
      const connected = await client.call('acp.connect', {
        session_id: sessionId,
        agent_id: 'antigravity',
      })
      expect(connected).toMatchObject({
        ok: true,
        binding: {
          agent_id: 'antigravity',
          external_session_id: '39c61d78',
          current_model: 'gemini-4-carbon-high',
          current_model_name: 'Gemini 4 Carbon (High)',
          model_config_id: 'agy_model',
          current_mode: 'agent',
          current_mode_name: 'Agent',
          mode_config_id: 'agy_mode',
        },
      })
      const initialBinding = connected.binding as Record<string, unknown>
      expect(initialBinding.models).toEqual([
        { id: 'gemini-4-carbon-high', name: 'Gemini 4 Carbon (High)', description: 'Deep reasoning' },
        { id: 'gemini-4-flash', name: 'Gemini 4 Flash', description: 'Fast' },
      ])
      expect(initialBinding.modes).toEqual([
        { id: 'agent', name: 'Agent', description: 'Autonomous coding' },
        { id: 'turbo', name: 'Turbo', description: 'Run all commands without restrictions.' },
        { id: 'plan', name: 'Planning', description: 'Plan before edits' },
      ])

      // Switch model via acp.set_model RPC -> updates metadata & broadcasts acp_status
      const beforeFrames = client.frames.length
      const setModelRes = await client.call('acp.set_model', {
        session_id: sessionId,
        model_id: 'gemini-4-flash',
      })
      expect(setModelRes).toMatchObject({
        ok: true,
        binding: {
          current_model: 'gemini-4-flash',
          current_model_name: 'Gemini 4 Flash',
        },
      })
      const liveSession = runtime.listSessions().find(s => s.id === sessionId)!
      expect(liveSession.metadata.acp_binding).toMatchObject({
        current_model: 'gemini-4-flash',
        current_model_name: 'Gemini 4 Flash',
      })
      const modelStatusEv = await client.next(
        f =>
          client.frames.indexOf(f) >= beforeFrames &&
          f.method === 'event' &&
          f.params?.type === 'acp_status' &&
          (f.params.payload?.binding as Record<string, unknown> | undefined)?.current_model === 'gemini-4-flash',
      )
      expect(modelStatusEv.params?.payload?.session_id).toBe(sessionId)

      // Switch mode via acp.set_mode RPC -> updates metadata & broadcasts acp_status
      const beforeModeFrames = client.frames.length
      const setModeRes = await client.call('acp.set_mode', {
        session_id: sessionId,
        mode_id: 'plan',
      })
      expect(setModeRes).toMatchObject({
        ok: true,
        binding: {
          current_mode: 'plan',
          current_mode_name: 'Planning',
        },
      })
      expect(liveSession.metadata.acp_binding).toMatchObject({
        current_mode: 'plan',
        current_mode_name: 'Planning',
      })
      await client.next(
        f =>
          client.frames.indexOf(f) >= beforeModeFrames &&
          f.method === 'event' &&
          f.params?.type === 'acp_status' &&
          (f.params.payload?.binding as Record<string, unknown> | undefined)?.current_mode === 'plan',
      )

      // Slash commands `/acp model <id>` and `/acp mode <id>`
      const slashModel = await client.call('slash', { command: '/acp model gemini-4-carbon-high' })
      expect(slashModel.ok).toBe(true)
      expect(String(slashModel.text)).toContain('Gemini 4 Carbon (High)')

      const slashMode = await client.call('slash', { command: '/acp mode agent' })
      expect(slashMode.ok).toBe(true)
      expect(String(slashMode.text)).toContain('Agent')

      // Live out-of-turn config_option_update from the external agent updates binding & broadcasts acp_status
      const beforeLiveFrames = client.frames.length
      activeModel = 'gemini-4-flash'
      activeHarness?.pushAgentFrame({
        jsonrpc: '2.0',
        method: 'session/update',
        params: {
          sessionId: '39c61d78',
          update: {
            sessionUpdate: 'config_option_update',
            configOptions: makeConfigOptions(),
          },
        },
      })
      await client.next(
        f =>
          client.frames.indexOf(f) >= beforeLiveFrames &&
          f.method === 'event' &&
          f.params?.type === 'acp_status' &&
          (f.params.payload?.binding as Record<string, unknown> | undefined)?.current_model === 'gemini-4-flash',
      )
      expect(bridgeHost.getBinding(sessionId)?.current_model).toBe('gemini-4-flash')

      // Verify GatewayClient.acpSetModel & acpSetMode over the same socket
      const gw = new GatewayClient({
        projectDir: root,
        externalSocketPath: socketPath,
        sessionKey: 'gw-acp-models',
      })
      await gw.start()
      try {
        await gw.request('session.create', {})
        const gwConn = await gw.acpConnect({ agent_id: 'antigravity' })
        expect(gwConn.ok).toBe(true)
        expect(gwConn.binding?.models?.length).toBe(2)

        const gwModel = await gw.acpSetModel({ model_id: 'gemini-4-carbon-high' })
        expect(gwModel.ok).toBe(true)
        expect(gwModel.binding?.current_model).toBe('gemini-4-carbon-high')
        expect(gwModel.binding?.current_model_name).toBe('Gemini 4 Carbon (High)')

        const gwMode = await gw.request('acp.set_mode', { mode_id: 'plan' })
        expect(gwMode.ok).toBe(true)
        expect(gwMode.binding?.current_mode).toBe('plan')
        expect(gwMode.binding?.current_mode_name).toBe('Planning')
      } finally {
        gw.close()
      }
    } finally {
      client.close()
      await server.stop()
    }
  })
})

describe('Automatic ACP Runtime Feature & Skill Discovery (Track A)', () => {
  test('parses YAML frontmatter (single-line, folded multiline, disable-slash-command) and classifies Antigravity, Claude Code, and generic ACP runtimes with zero Xerxes skill leaks', async () => {
    const root = await makeTempDir('xrx-acp-discovery-')
    const fakeHome = join(root, 'user-home')

    // 1. Folded multiline frontmatter + disable-slash-command parsing
    const folded = parseMarkdownSkillFrontmatter([
      '---',
      'name: antigravity-guide',
      'description: >-',
      '  Comprehensive guide for',
      '  Google Antigravity CLI.',
      '---',
      '# Heading',
    ].join('\n'))
    expect(folded).toEqual({
      name: 'antigravity-guide',
      description: 'Comprehensive guide for Google Antigravity CLI.',
    })
    expect(
      parseMarkdownSkillFrontmatter('---\nname: hidden-worker\ndisable-slash-command: true\n---\nBody'),
    ).toMatchObject({ name: 'hidden-worker', disableSlashCommand: true })

    // 2. Populate workspace + home skill directories AND a Xerxes bundled skill directory that MUST NEVER leak!
    await mkdir(join(root, '.agents', 'skills', 'resume-tailor'), { recursive: true })
    await writeFile(
      join(root, '.agents', 'skills', 'resume-tailor', 'SKILL.md'),
      '---\nname: resume-tailor\ndescription: Customize resume for target roles\n---\n# Resume Tailor\n',
      'utf8',
    )
    await mkdir(join(fakeHome, '.gemini', 'antigravity', 'builtin', 'skills', 'antigravity_guide'), {
      recursive: true,
    })
    await writeFile(
      join(fakeHome, '.gemini', 'antigravity', 'builtin', 'skills', 'antigravity_guide', 'SKILL.md'),
      '---\nname: antigravity-guide\ndescription: >-\n  Official Antigravity guide.\n---\n',
      'utf8',
    )
    await mkdir(join(fakeHome, '.gemini', 'antigravity', 'builtin', 'skills', 'swarm_monitor'), {
      recursive: true,
    })
    await writeFile(
      join(fakeHome, '.gemini', 'antigravity', 'builtin', 'skills', 'swarm_monitor', 'SKILL.md'),
      '---\nname: swarm_monitor\ndescription: Internal monitor\ndisable-slash-command: true\n---\n',
      'utf8',
    )
    // Xerxes native bundled skill inside workspace MUST NEVER appear in external ACP discovery!
    await mkdir(join(root, 'xerxes', 'skills', 'immortals-coordinator'), { recursive: true })
    await writeFile(
      join(root, 'xerxes', 'skills', 'immortals-coordinator', 'SKILL.md'),
      '---\nname: immortals-coordinator\ndescription: Xerxes native swarm coordinator\n---\n',
      'utf8',
    )

    // Antigravity classification: built-in shortcuts -> command; workspace/builtin skills -> skill; unlisted/Xerxes -> excluded!
    const agyClassified = classifyAcpRuntimeFeatures({
      provider: 'antigravity',
      agentName: 'Antigravity',
      cwd: root,
      homeDir: fakeHome,
      rawCommands: [
        { name: 'goal', description: 'Run until goal completes', input_hint: '<objective>' },
        { name: 'swarm', description: 'Run AGY swarm' },
        { name: 'resume-tailor', description: 'Customize resume for target roles' },
        { name: 'antigravity-guide', description: 'Official Antigravity guide.' },
      ],
    })
    expect(agyClassified.available_commands).toEqual([
      {
        name: 'goal',
        description: 'Run until goal completes',
        input_hint: '<objective>',
        kind: 'command',
        source: 'Antigravity',
      },
      {
        name: 'swarm',
        description: 'Run AGY swarm',
        kind: 'command',
        source: 'Antigravity',
      },
      {
        name: 'resume-tailor',
        description: 'Customize resume for target roles',
        kind: 'skill',
        source: 'Workspace',
      },
      {
        name: 'antigravity-guide',
        description: 'Official Antigravity guide.',
        kind: 'skill',
        source: 'Antigravity',
      },
    ])
    expect(agyClassified.discovered_skills).toEqual([
      {
        name: 'resume-tailor',
        description: 'Customize resume for target roles',
        source: 'Workspace',
        path: join(root, '.agents', 'skills', 'resume-tailor', 'SKILL.md'),
        slash_command: '/resume-tailor',
      },
      {
        name: 'antigravity-guide',
        description: 'Official Antigravity guide.',
        source: 'Antigravity',
        path: join(fakeHome, '.gemini', 'antigravity', 'builtin', 'skills', 'antigravity_guide', 'SKILL.md'),
        slash_command: '/antigravity-guide',
      },
    ])
    expect(agyClassified.discovered_skills.some(s => s.name.includes('immortals') || s.name === 'swarm_monitor')).toBe(
      false,
    )

    // 3. Claude Code discovery: .claude/skills, .claude/commands, ~/.claude/skills (including symlinks!), ~/.claude/commands
    await mkdir(join(root, '.claude', 'skills', 'release-check'), { recursive: true })
    await writeFile(
      join(root, '.claude', 'skills', 'release-check', 'SKILL.md'),
      '---\nname: release-check\ndescription: Verify release readiness\n---\n',
      'utf8',
    )
    await mkdir(join(root, '.claude', 'commands'), { recursive: true })
    await writeFile(
      join(root, '.claude', 'commands', 'deploy.md'),
      '---\ndescription: Deploy workspace build\n---\nRun deploy steps\n',
      'utf8',
    )
    await mkdir(join(fakeHome, '.claude', 'skills'), { recursive: true })
    await symlink(
      join(root, '.agents', 'skills', 'resume-tailor'),
      join(fakeHome, '.claude', 'skills', 'resume-tailor'),
    )
    await mkdir(join(fakeHome, '.claude', 'commands'), { recursive: true })
    await writeFile(
      join(fakeHome, '.claude', 'commands', 'audit.md'),
      '# Security Audit\nCheck all auth paths.\n',
      'utf8',
    )

    const claudeEntries = discoverRuntimeSkillEntries({
      provider: 'claude-code',
      cwd: root,
      homeDir: fakeHome,
    })
    expect(claudeEntries.map(e => `${e.name}:${e.kind}:${e.source}`)).toEqual([
      'release-check:skill:Workspace',
      'deploy:command:Workspace',
      'resume-tailor:skill:Claude Code',
      'audit:command:Claude Code',
    ])
    const claudeClassified = classifyAcpRuntimeFeatures({
      provider: 'claude-code',
      agentName: 'Claude Code',
      cwd: root,
      homeDir: fakeHome,
      includeFilesystemUnlisted: true,
      rawCommands: [{ name: 'compact', description: 'Compact Claude context' }],
    })
    expect(claudeClassified.discovered_skills.map(s => s.name)).toEqual(['release-check', 'resume-tailor'])
    expect(
      claudeClassified.available_commands.filter(c => c.kind === 'command').map(c => c.name),
    ).toEqual(['compact', 'deploy', 'audit'])
    expect(claudeClassified.discovered_skills.some(s => s.name.includes('immortals'))).toBe(false)

    // 4. Generic / custom ACP agent: parses input.hint + _meta skill hints + ALL select/boolean/string configOptions + full capabilities
    const normalizedGeneric = normalizeAcpSessionUpdate('gen-1', {
      sessionUpdate: 'available_commands_update',
      availableCommands: [
        { name: '/lint', description: 'Run linter', input: { hint: '[--fix]' } },
        { name: 'db-migrate', description: 'Run DB migration skill', _meta: { kind: 'skill', source: 'Plugin' } },
      ],
    })
    expect(normalizedGeneric).toEqual([
      {
        type: 'available_commands',
        sessionId: 'gen-1',
        commands: [
          { name: 'lint', description: 'Run linter', input_hint: '[--fix]' },
          { name: 'db-migrate', description: 'Run DB migration skill', kind: 'skill', source: 'Plugin' },
        ],
      },
    ])

    const genericHost = new DaemonAcpBridgeHost({
      homeDir: fakeHome,
      discoverAgents: () => [
        {
          id: 'custom-acp',
          name: 'Custom ACP',
          provider: 'acp',
          transport: 'acp',
          available: true,
          command: '/usr/local/bin/custom-acp',
          args: [],
          binaryPath: '/usr/local/bin/custom-acp',
          description: 'Custom ACP runtime',
        },
      ],
      createAcpClient: opts => {
        const h = createInMemoryAcpTransport((frame, harness) => {
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                protocolVersion: 1,
                agentInfo: { name: 'Custom ACP Runtime', title: 'Custom ACP Studio', version: '3.2.1' },
                agentCapabilities: {
                  loadSession: true,
                  promptCapabilities: { image: true, audio: true, embeddedContext: true },
                  sessionCapabilities: {
                    resume: {},
                    close: {},
                    load: {},
                    list: {},
                    additionalDirectories: {},
                  },
                  mcpCapabilities: { http: true, sse: true },
                  _meta: { vendor: { streaming: true } },
                },
                authMethods: [{ id: 'api-key', name: 'API Key', description: 'Use vendor API key' }],
              },
            })
            return
          }
          if (frame.method === 'session/new') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: 'gen-sess-1',
                update: {
                  sessionUpdate: 'available_commands_update',
                  availableCommands: [
                    { name: 'lint', description: 'Run linter', input: { hint: '[--fix]' } },
                    { name: 'db-migrate', description: 'DB migration helper', _meta: { kind: 'skill', source: 'Vendor' } },
                  ],
                },
              },
            })
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                sessionId: 'gen-sess-1',
                configOptions: [
                  {
                    id: 'model',
                    name: 'Model',
                    category: 'model',
                    type: 'select',
                    currentValue: 'vendor-pro',
                    options: [{ value: 'vendor-pro', name: 'Vendor Pro' }],
                  },
                  {
                    id: 'web_search',
                    name: 'Web Search',
                    type: 'boolean',
                    currentValue: true,
                    options: [
                      { value: 'true', name: 'Enabled' },
                      { value: 'false', name: 'Disabled' },
                    ],
                  },
                  {
                    id: 'system_persona',
                    name: 'Persona',
                    type: 'string',
                    currentValue: 'terse',
                    options: [],
                  },
                ],
              },
            })
          }
        })
        return new AcpStdioClient({ ...opts, transport: h.transport })
      },
    })

    const genericConn = await genericHost.connect({
      sessionId: 'gen-local-1',
      agentId: 'custom-acp',
      cwd: root,
    })
    expect(genericConn.ok).toBe(true)
    if (genericConn.ok) {
      expect(genericConn.binding.agent_title).toBe('Custom ACP Studio')
      expect(genericConn.binding.agent_version).toBe('3.2.1')
      expect(genericConn.binding.capabilities).toEqual({
        prompt: { image: true, audio: true, embedded_context: true },
        session: { load: true, resume: true, list: true, close: true, additional_directories: true },
        mcp: { http: true, sse: true },
        auth_methods: [{ id: 'api-key', name: 'API Key', description: 'Use vendor API key' }],
        meta: { vendor: { streaming: true } },
      })
      expect(genericConn.binding.config_options).toEqual([
        {
          id: 'model',
          name: 'Model',
          category: 'model',
          type: 'select',
          current_value: 'vendor-pro',
          options: [{ value: 'vendor-pro', name: 'Vendor Pro' }],
        },
        {
          id: 'web_search',
          name: 'Web Search',
          type: 'boolean',
          current_value: 'true',
          options: [
            { value: 'true', name: 'Enabled' },
            { value: 'false', name: 'Disabled' },
          ],
        },
        {
          id: 'system_persona',
          name: 'Persona',
          type: 'string',
          current_value: 'terse',
          options: [],
        },
      ])
      expect(genericConn.binding.discovered_skills).toEqual([
        {
          name: 'db-migrate',
          description: 'DB migration helper',
          source: 'Vendor',
          slash_command: '/db-migrate',
        },
      ])
      expect(genericConn.binding.available_commands).toEqual([
        { name: 'lint', description: 'Run linter', input_hint: '[--fix]', kind: 'command', source: 'Custom ACP' },
        { name: 'db-migrate', description: 'DB migration helper', kind: 'skill', source: 'Vendor' },
      ])
    }
    await genericHost.disposeAll()

    // 5. ClaudeStreamJsonRunner parses slash_commands & skills on system/init frames
    const claudeInitEvents: AcpBridgeStreamEvent[] = []
    const claudeRunner = new ClaudeStreamJsonRunner({
      binaryPath: '/usr/local/bin/claude',
      cwd: root,
      spawn: (): ClaudeStreamJsonProcess => ({
        stdout: new ReadableStream<Uint8Array>({
          start(controller) {
            controller.enqueue(
              encoder.encode(
                `${JSON.stringify({
                  type: 'system',
                  subtype: 'init',
                  session_id: 'cc-disc-1',
                  model: 'claude-sonnet-4-6',
                  slash_commands: ['compact', 'review'],
                  skills: ['release-check'],
                })}\n${JSON.stringify({
                  type: 'result',
                  subtype: 'success',
                  session_id: 'cc-disc-1',
                  result: 'ok',
                })}\n`,
              ),
            )
            controller.close()
          },
        }),
        stderr: null,
        exited: Promise.resolve(0),
        kill: () => undefined,
      }),
    })
    await claudeRunner.prompt({ text: 'hi', onEvent: ev => claudeInitEvents.push(ev) })
    const cmdEv = claudeInitEvents.find(ev => ev.type === 'available_commands')
    expect(cmdEv).toEqual({
      type: 'available_commands',
      commands: [
        { name: 'release-check', kind: 'skill', source: 'Claude Code' },
        { name: 'compact', kind: 'command', source: 'Claude Code' },
        { name: 'review', kind: 'command', source: 'Claude Code' },
      ],
    })
  })
})
