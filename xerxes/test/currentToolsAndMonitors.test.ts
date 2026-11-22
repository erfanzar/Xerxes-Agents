// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { createHmac } from 'node:crypto'
import { existsSync } from 'node:fs'
import { mkdir, mkdtemp, readFile, rename, rm, symlink, writeFile } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import type { DaemonConfig } from '../src/daemon/config.js'
import { daemonMonitorWebhookHub } from '../src/daemon/monitorWebhooks.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { DaemonSwarmHost } from '../src/daemon/swarmHost.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import { ToolRegistry, type ToolExecutionContext } from '../src/executors/toolRegistry.js'
import { lintHarness } from '../src/extensions/harnessLint.js'
import {
  BUNDLED_SKILLS_DIRECTORY,
  parseSkillMarkdown,
  SkillRegistry,
} from '../src/extensions/skills.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { PtySessionManager } from '../src/operators/pty.js'
import type { JsonRpcPayload } from '../src/protocol/jsonRpc.js'
import { nativeFileMonitorSource } from '../src/runtime/fileMonitorSource.js'
import { ProcessRegistry } from '../src/runtime/processRegistry.js'
import { ReactionMailbox } from '../src/runtime/reactionMailbox.js'
import { RunHistory } from '../src/runtime/runHistory.js'
import { TerminalMonitors } from '../src/runtime/terminalMonitors.js'
import { TerminalRegistry } from '../src/runtime/terminalRegistry.js'
import { nativeWebSocketMonitorSource } from '../src/runtime/websocketMonitorSource.js'
import { BackgroundCommandManager } from '../src/tools/backgroundCommands.js'
import { registerClaudeSkillTool, registerCoreTools } from '../src/tools/index.js'
import { registerMonitorTools } from '../src/tools/monitorTools.js'
import type { JsonObject } from '../src/types/toolCalls.js'

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
          for (let index = 0; index < this.waiters.length; index += 1) {
            const waiter = this.waiters[index]!
            if (waiter.predicate(parsed)) {
              this.waiters.splice(index, 1)
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
        resolve: (frame) => {
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

async function makeTempDir(prefix = 'xrx-ctm-'): Promise<string> {
  const baseTmp = process.platform === 'win32' ? tmpdir() : (existsSync('/tmp') ? '/tmp' : tmpdir())
  const dir = await mkdtemp(join(baseTmp, prefix))
  tempRoots.push(dir)
  process.env.XERXES_HOME = join(dir, 'home')
  await mkdir(process.env.XERXES_HOME, { recursive: true })
  return dir
}

async function waitUntil(
  predicate: () => boolean | Promise<boolean>,
  timeoutMs = 5_000,
): Promise<void> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (await predicate()) return
    await Bun.sleep(15)
  }
  throw new Error(`Condition did not settle within ${timeoutMs}ms`)
}

function makeFakeLlm(
  responder: (request: CompletionRequest) => LlmDelta[] | Promise<LlmDelta[]>,
): LlmClient {
  return {
    async complete() {
      return { content: 'ok', toolCalls: [] }
    },
    async *stream(request: CompletionRequest): AsyncIterable<LlmDelta> {
      const chunks = await responder(request)
      for (const chunk of chunks) {
        yield chunk
      }
    },
  }
}

async function callTool<T = unknown>(
  registry: ToolRegistry,
  name: string,
  args: JsonObject,
  context: ToolExecutionContext,
): Promise<T> {
  const raw = await registry.execute(
    {
      id: `call_${name}_${crypto.randomUUID().slice(0, 8)}`,
      type: 'function',
      function: { name, arguments: args },
    },
    context,
  )
  try {
    return JSON.parse(raw) as T
  } catch {
    return raw as unknown as T
  }
}

describe('Session Monitors End-to-End (Terminal, File, WebSocket, Webhook)', () => {
  test('monitor_terminal watches output matches, completion exit triggers, list_monitors, and stop_monitor without killing the source process', async () => {
    const root = await makeTempDir()
    const workspace = join(root, 'workspace')
    await mkdir(workspace, { recursive: true })

    const history = new RunHistory(join(root, 'history.sqlite'))
    const mailbox = new ReactionMailbox(join(root, 'reactions.sqlite'))
    const terminals = new TerminalRegistry({ runHistory: history })
    const monitors = new TerminalMonitors(terminals, history, undefined, undefined, mailbox)
    const backgroundCommands = new BackgroundCommandManager(new ProcessRegistry(), terminals)

    const tools = new ToolRegistry()
    registerMonitorTools(tools, monitors)
    registerCoreTools(tools, {
      workspaceRoot: workspace,
      backgroundCommands,
      terminals,
    })

    const sessionId = 'sess-term-mon'
    const ctx: ToolExecutionContext = { sessionId, metadata: {} }

    try {
      const started = await callTool<{ procId: string; running: boolean }>(
        tools,
        'exec_command',
        {
          cmd: 'sh',
          args: ['-c', 'sleep 0.12; echo "SERVER_READY port=4321"; sleep 0.22; echo "DRAIN_COMPLETE"; exit 0'],
          run_in_background: true,
        },
        ctx,
      )
      expect(started.running).toBe(true)

      const outputWatch = await callTool<{ id: string; state: string; trigger: string }>(
        tools,
        'monitor_terminal',
        {
          terminal_id: started.procId,
          trigger: 'output',
          match: 'server_ready',
          duration_seconds: 60,
        },
        ctx,
      )
      expect(outputWatch.state).toBe('watching')
      expect(outputWatch.trigger).toBe('output')

      const completionWatch = await callTool<{ id: string; state: string; trigger: string }>(
        tools,
        'monitor_terminal',
        {
          terminal_id: started.procId,
          trigger: 'completion',
          duration_seconds: 60,
        },
        ctx,
      )
      expect(completionWatch.state).toBe('watching')
      expect(completionWatch.trigger).toBe('completion')

      await waitUntil(async () => {
        const listed = await callTool<Array<{ id: string; events: Array<{ text: string }> }>>(
          tools,
          'list_monitors',
          {},
          ctx,
        )
        const target = listed.find(item => item.id === outputWatch.id)
        return Boolean(target && target.events.some(ev => ev.text.includes('SERVER_READY port=4321')))
      })

      // Stopping the output monitor must leave the underlying process and completion monitor running!
      const stoppedOutput = await callTool<{ id: string; state: string }>(
        tools,
        'stop_monitor',
        { monitor_id: outputWatch.id },
        ctx,
      )
      expect(stoppedOutput.state).toBe('stopped')
      expect(terminals.inspect(sessionId, started.procId)?.running).toBe(true)

      await waitUntil(async () => {
        const listed = await callTool<Array<{ id: string; state: string; events: Array<{ text: string }> }>>(
          tools,
          'list_monitors',
          {},
          ctx,
        )
        const target = listed.find(item => item.id === completionWatch.id)
        return target?.state === 'source-ended' && target.events.length === 1
      })

      const finalMonitors = await callTool<Array<{ id: string; state: string; events: Array<{ text: string }> }>>(
        tools,
        'list_monitors',
        {},
        ctx,
      )
      const finishedCompletion = finalMonitors.find(item => item.id === completionWatch.id)!
      const completionPayload = JSON.parse(finishedCompletion.events[0]!.text) as {
        exit_code: number
        output: string
      }
      expect(completionPayload.exit_code).toBe(0)
      expect(completionPayload.output).toContain('DRAIN_COMPLETE')
    } finally {
      await backgroundCommands.disposeAll()
      monitors.close()
      mailbox.close()
      history.close()
    }
  })

  test('monitor_file detects direct writes and atomic rename replacements, rejects outside-workspace and symlink escapes, and stops cleanly', async () => {
    const root = await makeTempDir()
    const workspace = join(root, 'workspace')
    const outsideDir = join(root, 'outside')
    await mkdir(workspace, { recursive: true })
    await mkdir(outsideDir, { recursive: true })

    const targetPath = join(workspace, 'status.json')
    const outsideFile = join(outsideDir, 'secret.txt')
    const escapedSymlink = join(workspace, 'escape-link.txt')
    await writeFile(targetPath, '{"step":1}\n', 'utf8')
    await writeFile(outsideFile, 'top-secret\n', 'utf8')
    await symlink(outsideFile, escapedSymlink)

    const history = new RunHistory(join(root, 'history.sqlite'))
    const terminals = new TerminalRegistry({ runHistory: history })
    const monitors = new TerminalMonitors(
      terminals,
      history,
      undefined,
      undefined,
      undefined,
      {
        source: nativeFileMonitorSource,
        resolveWorkspace: () => workspace,
      },
    )

    const tools = new ToolRegistry()
    registerMonitorTools(tools, monitors)
    const ctx: ToolExecutionContext = { sessionId: 'sess-file-mon', metadata: {} }

    try {
      await expect(
        callTool(tools, 'monitor_file', { file_path: '../outside/secret.txt' }, ctx),
      ).rejects.toThrow(/parent traversal/i)

      await expect(
        callTool(tools, 'monitor_file', { file_path: outsideFile }, ctx),
      ).rejects.toThrow(/inside the workspace/i)

      await expect(
        callTool(tools, 'monitor_file', { file_path: 'escape-link.txt' }, ctx),
      ).rejects.toThrow(/outside the workspace/i)

      const watch = await callTool<{ id: string; state: string; trigger: string }>(
        tools,
        'monitor_file',
        { file_path: 'status.json', duration_seconds: 60 },
        ctx,
      )
      expect(watch.state).toBe('watching')
      expect(watch.trigger).toBe('change')

      // 1. Direct file modification
      await Bun.sleep(60)
      await writeFile(targetPath, '{"step":2,"mode":"direct"}\n', 'utf8')
      await waitUntil(async () => {
        const listed = await callTool<Array<{ id: string; events: Array<{ text: string }> }>>(
          tools,
          'list_monitors',
          {},
          ctx,
        )
        return (listed.find(item => item.id === watch.id)?.events.length ?? 0) >= 1
      })

      // 2. Atomic write + rename replacement
      await Bun.sleep(80)
      const stagingPath = join(workspace, 'status.json.tmp')
      await writeFile(stagingPath, '{"step":3,"mode":"atomic-rename-replacement"}\n', 'utf8')
      await rename(stagingPath, targetPath)

      await waitUntil(async () => {
        const listed = await callTool<Array<{ id: string; events: Array<{ text: string }> }>>(
          tools,
          'list_monitors',
          {},
          ctx,
        )
        return (listed.find(item => item.id === watch.id)?.events.length ?? 0) >= 2
      })

      const stopped = await callTool<{ id: string; state: string }>(
        tools,
        'stop_monitor',
        { monitor_id: watch.id },
        ctx,
      )
      expect(stopped.state).toBe('stopped')
    } finally {
      monitors.close()
      history.close()
    }
  })

  test('monitor_websocket matches loopback Bun.serve WebSocket text frames and ignores non-matching frames', async () => {
    const root = await makeTempDir()
    const workspace = join(root, 'workspace')
    await mkdir(workspace, { recursive: true })

    const clients = new Set<Bun.ServerWebSocket<unknown>>()
    const wsServer = Bun.serve({
      hostname: '127.0.0.1',
      port: 0,
      fetch(req, server) {
        if (server.upgrade(req)) return undefined
        return new Response('not upgraded', { status: 400 })
      },
      websocket: {
        open(ws) {
          clients.add(ws)
        },
        close(ws) {
          clients.delete(ws)
        },
        message() {},
      },
    })

    const history = new RunHistory(join(root, 'history.sqlite'))
    const terminals = new TerminalRegistry({ runHistory: history })
    const monitors = new TerminalMonitors(
      terminals,
      history,
      undefined,
      undefined,
      undefined,
      undefined,
      {
        source: nativeWebSocketMonitorSource,
        resolveWorkspace: () => workspace,
      },
    )

    const tools = new ToolRegistry()
    registerMonitorTools(tools, monitors)
    const ctx: ToolExecutionContext = { sessionId: 'sess-ws-mon', metadata: {} }

    try {
      const watch = await callTool<{ id: string; state: string }>(
        tools,
        'monitor_websocket',
        {
          websocket_url: `ws://127.0.0.1:${wsServer.port}/stream`,
          match: 'canary_tripwire',
          duration_seconds: 60,
        },
        ctx,
      )
      expect(watch.state).toBe('watching')
      await waitUntil(() => clients.size === 1)

      for (const ws of clients) {
        ws.send('status=healthy shard=1')
        ws.send('ALERT: CANARY_TRIPWIRE triggered on shard=2')
      }

      await waitUntil(async () => {
        const listed = await callTool<Array<{ id: string; events: Array<{ text: string }> }>>(
          tools,
          'list_monitors',
          {},
          ctx,
        )
        const found = listed.find(item => item.id === watch.id)
        return Boolean(found && found.events.length === 1)
      })

      const listed = await callTool<Array<{ id: string; events: Array<{ text: string }> }>>(
        tools,
        'list_monitors',
        {},
        ctx,
      )
      const active = listed.find(item => item.id === watch.id)!
      expect(active.events).toHaveLength(1)
      expect(active.events[0]!.text).toContain('CANARY_TRIPWIRE triggered on shard=2')

      const stopped = await callTool<{ id: string; state: string }>(
        tools,
        'stop_monitor',
        { monitor_id: watch.id },
        ctx,
      )
      expect(stopped.state).toBe('stopped')
    } finally {
      monitors.close()
      await wsServer.stop(true)
      history.close()
    }
  })

  test('monitor_webhook lists sources without secrets, records valid signed deliveries, and rejects unsigned or bad-HMAC payloads', async () => {
    const root = await makeTempDir()
    const workspace = join(root, 'workspace')
    await mkdir(workspace, { recursive: true })

    const secret = '0123456789abcdef0123456789abcdef-webhook-secret'
    const makeDaemonCfg = (monitorWebhooks: unknown): DaemonConfig => ({
      channels: {},
      control: {},
      maxConcurrentTurns: 1,
      projectDirectory: workspace,
      runtime: { monitor_webhooks: monitorWebhooks },
      workspace: {},
    })

    expect(() =>
      daemonMonitorWebhookHub(
        makeDaemonCfg({
          port: 0,
          sources: [
            { name: 'dup', secret_env: 'WH_SECRET' },
            { name: 'dup', secret_env: 'WH_SECRET' },
          ],
        }),
        { WH_SECRET: secret },
      ),
    ).toThrow(TypeError)

    const hub = daemonMonitorWebhookHub(
      makeDaemonCfg({
        host: '127.0.0.1',
        port: 0,
        sources: [{ name: 'ci-hooks', secret_env: 'WH_SECRET' }],
      }),
      { WH_SECRET: secret },
    )!
    hub.start()

    const history = new RunHistory(join(root, 'history.sqlite'))
    const terminals = new TerminalRegistry({ runHistory: history })
    const monitors = new TerminalMonitors(
      terminals,
      history,
      undefined,
      undefined,
      undefined,
      undefined,
      undefined,
      {
        source: hub,
        resolveWorkspace: () => workspace,
      },
    )

    const tools = new ToolRegistry()
    registerMonitorTools(tools, monitors)
    const ctx: ToolExecutionContext = { sessionId: 'sess-wh-mon', metadata: {} }

    const signPayload = (body: string, deliveryId: string, key = secret): RequestInit => {
      const timestamp = String(Math.floor(Date.now() / 1_000))
      const digest = createHmac('sha256', key)
        .update(`${timestamp}.${deliveryId}.${body}`)
        .digest('hex')
      return {
        method: 'POST',
        body,
        headers: {
          'x-xerxes-timestamp': timestamp,
          'x-xerxes-delivery-id': deliveryId,
          'x-xerxes-signature': `sha256=${digest}`,
        },
      }
    }

    try {
      const sources = await callTool<{ webhooks: Array<{ name: string }> }>(
        tools,
        'list_monitor_sources',
        {},
        ctx,
      )
      expect(sources).toEqual({ webhooks: [{ name: 'ci-hooks' }] })
      expect(JSON.stringify(sources)).not.toContain(secret)

      const watch = await callTool<{ id: string; state: string }>(
        tools,
        'monitor_webhook',
        {
          webhook_name: 'ci-hooks',
          match: 'deploy_panic',
          duration_seconds: 60,
        },
        ctx,
      )
      expect(watch.state).toBe('watching')

      const endpoint = new URL('/monitors/ci-hooks', hub.url!).href

      // Unsigned delivery -> 401
      const unsignedRes = await fetch(endpoint, { method: 'POST', body: 'deploy_panic unsigned' })
      expect(unsignedRes.status).toBe(401)

      // Bad signature delivery -> 401
      const badSigRes = await fetch(
        endpoint,
        signPayload('deploy_panic forged', 'deliv-bad', 'wrong-secret-0123456789abcdef0123456789'),
      )
      expect(badSigRes.status).toBe(401)

      // Valid signature, non-matching body -> 202, 0 recorded matches
      const nonMatchRes = await fetch(endpoint, signPayload('deploy_ok green', 'deliv-ok'))
      expect(nonMatchRes.status).toBe(202)

      // Valid signature, matching body -> 202, recorded in monitor events
      const matchRes = await fetch(endpoint, signPayload('CRITICAL: DEPLOY_PANIC in eu-west', 'deliv-match'))
      expect(matchRes.status).toBe(202)

      await waitUntil(async () => {
        const listed = await callTool<Array<{ id: string; events: Array<{ text: string }> }>>(
          tools,
          'list_monitors',
          {},
          ctx,
        )
        return (listed.find(item => item.id === watch.id)?.events.length ?? 0) === 1
      })

      const listed = await callTool<Array<{ id: string; events: Array<{ text: string }> }>>(
        tools,
        'list_monitors',
        {},
        ctx,
      )
      expect(listed.find(item => item.id === watch.id)!.events[0]!.text).toContain('DEPLOY_PANIC in eu-west')

      const stopped = await callTool<{ id: string; state: string }>(
        tools,
        'stop_monitor',
        { monitor_id: watch.id },
        ctx,
      )
      expect(stopped.state).toBe('stopped')
      expect((await fetch(endpoint, signPayload('DEPLOY_PANIC after stop', 'deliv-after'))).status).toBe(410)
    } finally {
      monitors.close()
      await hub.stop()
      history.close()
    }
  })
})

describe('Core File, Process, Interactive PTY, and Skill Tools End-to-End', () => {
  test('ReadFile, WriteFile, FileEditTool, GlobTool, GrepTool, ListDir, exec/check/list/kill_command, pty_open/write/close, and SkillTool underscore/hyphen lookup work end-to-end', async () => {
    const root = await makeTempDir()
    const workspace = join(root, 'project')
    await mkdir(workspace, { recursive: true })

    const history = new RunHistory(join(root, 'history.sqlite'))
    const terminals = new TerminalRegistry({ runHistory: history })
    const backgroundCommands = new BackgroundCommandManager(new ProcessRegistry(), terminals)
    const ptySessions = new PtySessionManager({
      terminals,
      workspaceRoot: workspace,
    })

    const skillRegistry = new SkillRegistry()
    await skillRegistry.discover(BUNDLED_SKILLS_DIRECTORY)

    // Verify exact-match priority alongside `-` <-> `_` normalization and legacy `swarm-*` -> `immortals-*` aliases.
    expect(skillRegistry.has('immortals-monitor')).toBe(true)
    expect(skillRegistry.has('immortals_monitor')).toBe(true)
    expect(skillRegistry.has('immortals-narrator')).toBe(true)
    expect(skillRegistry.has('immortals-coordinator')).toBe(true)
    expect(skillRegistry.has('swarm-monitor')).toBe(true)
    expect(skillRegistry.has('swarm_monitor')).toBe(true)
    expect(skillRegistry.has('swarm_narrator')).toBe(true)
    expect(skillRegistry.has('swarm_coordinator')).toBe(true)
    expect(skillRegistry.get('immortals_monitor')?.metadata.name).toBe('immortals-monitor')
    expect(skillRegistry.get('swarm_monitor')?.metadata.name).toBe('immortals-monitor')
    expect(skillRegistry.get('swarm_narrator')?.metadata.name).toBe('immortals-narrator')
    expect(skillRegistry.get('swarm_coordinator')?.metadata.name).toBe('immortals-coordinator')
    expect(skillRegistry.names).not.toContain('swarm-coordinator')
    expect(skillRegistry.names).not.toContain('swarm-monitor')
    expect(skillRegistry.names).not.toContain('swarm-narrator')

    const customExactRegistry = new SkillRegistry()
    customExactRegistry.register(
      parseSkillMarkdown('---\nname: pair-skill\ndescription: hyphen variant\n---\nHyphen body', join(root, 'a/SKILL.md')),
    )
    customExactRegistry.register(
      parseSkillMarkdown('---\nname: pair_skill\ndescription: underscore variant\n---\nUnderscore body', join(root, 'b/SKILL.md')),
    )
    expect(customExactRegistry.get('pair-skill')?.instructions).toBe('Hyphen body')
    expect(customExactRegistry.get('pair_skill')?.instructions).toBe('Underscore body')

    // Ensure updated bundled Immortals skills pass harnessLint with zero findings.
    const bundledSwarmDocs = await Promise.all(
      ['immortals-coordinator', 'immortals-monitor', 'immortals-narrator'].map(async (name) => {
        const skill = skillRegistry.get(name)!
        return {
          path: skill.sourcePath,
          content: await Bun.file(skill.sourcePath).text(),
        }
      }),
    )
    expect(lintHarness(bundledSwarmDocs)).toEqual([])

    const tools = new ToolRegistry()
    registerCoreTools(tools, {
      workspaceRoot: workspace,
      includeCodingTools: true,
      backgroundCommands,
      ptySessions,
      terminals,
    })
    registerClaudeSkillTool(tools, skillRegistry, 'default', () => workspace)

    const ctx: ToolExecutionContext = { sessionId: 'sess-core-tools', metadata: {} }

    try {
      // 1. WriteFile + ReadFile + FileEditTool + ListDir + GlobTool + GrepTool
      await callTool(
        tools,
        'WriteFile',
        {
          file_path: 'src/math/adder.ts',
          content: 'export function addValues(a: number, b: number): number {\n  return a + b\n}\n',
        },
        ctx,
      )
      const initialRead = await callTool<string>(tools, 'ReadFile', { file_path: 'src/math/adder.ts' }, ctx)
      expect(String(initialRead)).toContain('export function addValues')

      await callTool(
        tools,
        'FileEditTool',
        {
          file_path: 'src/math/adder.ts',
          old_string: 'return a + b',
          new_string: 'return a + b + 0 // verified edit',
        },
        ctx,
      )
      expect(await readFile(join(workspace, 'src/math/adder.ts'), 'utf8')).toContain('// verified edit')

      const dirListing = await callTool<string>(tools, 'ListDir', { directory_path: 'src', recursive: true }, ctx)
      expect(JSON.stringify(dirListing)).toContain('adder.ts')

      const globOut = await callTool<string>(tools, 'GlobTool', { pattern: '**/*.ts' }, ctx)
      expect(JSON.stringify(globOut)).toContain('src/math/adder.ts')

      const grepOut = await callTool<string>(
        tools,
        'GrepTool',
        { pattern: 'verified edit', output_mode: 'content' },
        ctx,
      )
      expect(JSON.stringify(grepOut)).toContain('adder.ts')
      expect(JSON.stringify(grepOut)).toContain('verified edit')

      // 2. Foreground & background exec_command + list_commands + check_command + kill_command
      const fg = await callTool<{ exitCode: number; stdout: string }>(
        tools,
        'exec_command',
        { cmd: 'sh', args: ['-c', 'printf "fg-ok"'] },
        ctx,
      )
      expect(fg.exitCode).toBe(0)
      expect(fg.stdout).toBe('fg-ok')

      const bg = await callTool<{ procId: string; running: boolean }>(
        tools,
        'exec_command',
        {
          cmd: 'sh',
          args: ['-c', 'echo "bg-line-1"; sleep 10'],
          run_in_background: true,
        },
        ctx,
      )
      expect(bg.running).toBe(true)

      const listedCmds = await callTool<{ processes: Array<{ procId: string }> }>(
        tools,
        'list_commands',
        {},
        ctx,
      )
      expect(listedCmds.processes.some(p => p.procId === bg.procId)).toBe(true)

      await waitUntil(async () => {
        const checked = await callTool<{ stdout: string; running: boolean }>(
          tools,
          'check_command',
          { proc_id: bg.procId },
          ctx,
        )
        return checked.stdout.includes('bg-line-1')
      })

      const killed = await callTool<{ killed?: boolean; signalled?: boolean }>(
        tools,
        'kill_command',
        { proc_id: bg.procId, signal: 'SIGKILL' },
        ctx,
      )
      expect(killed).toBeDefined()

      // 3. Interactive PTY lifecycle: pty_open + pty_write (stateful env persistence) + pty_list + pty_close
      const openedPty = await callTool<{ sessionId: string; running: boolean }>(
        tools,
        'pty_open',
        { command: 'export XERXES_PTY_STATE=persisted_42; exec bash', yield_time_ms: 250 },
        ctx,
      )
      expect(typeof openedPty.sessionId).toBe('string')
      expect(openedPty.sessionId.startsWith('pty_')).toBe(true)

      const firstWrite = await callTool<{ stdout: string; running: boolean }>(
        tools,
        'pty_write',
        {
          session_id: openedPty.sessionId,
          chars: 'echo "OUT:$XERXES_PTY_STATE"\n',
          yield_time_ms: 2_000,
        },
        ctx,
      )
      expect(firstWrite.stdout).toContain('OUT:persisted_42')

      const ptyList = await callTool<{ sessions: Array<{ sessionId: string; running: boolean }> }>(
        tools,
        'pty_list',
        {},
        ctx,
      )
      expect(ptyList.sessions.some(s => s.sessionId === openedPty.sessionId)).toBe(true)

      const closedPty = await callTool<{ closed: boolean }>(
        tools,
        'pty_close',
        { session_id: openedPty.sessionId },
        ctx,
      )
      expect(closedPty.closed).toBe(true)

      // 4. SkillTool resolving canonical (`immortals-monitor`) and legacy aliases (`swarm-monitor`, `swarm_monitor`)
      const byCanonical = await callTool<string>(
        tools,
        'SkillTool',
        { skill_name: 'immortals-monitor' },
        ctx,
      )
      const byHyphen = await callTool<string>(
        tools,
        'SkillTool',
        { skill_name: 'swarm-monitor' },
        ctx,
      )
      const byUnderscore = await callTool<string>(
        tools,
        'SkillTool',
        { skill_name: 'swarm_monitor' },
        ctx,
      )
      expect(String(byCanonical)).toContain('[Skill immortals-monitor activated]')
      expect(String(byCanonical)).toContain('TimerCondition: "any"')
      expect(String(byCanonical)).toContain('manage_task')
      expect(byHyphen).toBe(byCanonical)
      expect(byUnderscore).toBe(byCanonical)
    } finally {
      await Promise.all([backgroundCommands.disposeAll(), ptySessions.disposeAll()])
      history.close()
    }
  })
})

describe('DaemonSwarmHost Permission Mode Enforcement & Shared Terminals/Monitors Visibility', () => {
  test('parent session plan mode blocks worker file writes while auto mode allows worker tools and shares /terminals & /monitors with parent session', async () => {
    const root = await makeTempDir()
    const workspace = join(root, 'repo')
    await mkdir(workspace, { recursive: true })
    await writeFile(join(workspace, 'existing.txt'), 'hello-plan-check\n', 'utf8')

    const history = new RunHistory(join(root, 'history.sqlite'))
    const terminals = new TerminalRegistry({ runHistory: history })
    const monitors = new TerminalMonitors(
      terminals,
      history,
      undefined,
      undefined,
      undefined,
      {
        source: nativeFileMonitorSource,
        resolveWorkspace: () => workspace,
      },
    )
    const backgroundCommands = new BackgroundCommandManager(new ProcessRegistry(), terminals)

    let workerAttemptedPlanWrite = false
    let workerPlanWriteDenied = false

    const workerLlm = makeFakeLlm(async (req) => {
      const sys = String(req.messages.find(m => m.role === 'system')?.content ?? '')
      const lastUser = [...req.messages].reverse().find(m => m.role === 'user')
      const lastUserIdx = req.messages.lastIndexOf(lastUser!)
      const toolMsgs = req.messages.slice(lastUserIdx + 1).filter(m => m.role === 'tool')
      if (toolMsgs.some(m => m.name === 'swarm_wait')) {
        return [{ content: '', finishReason: 'stop' }]
      }

      if (sys.includes('You are "plan-worker"')) {
        if (toolMsgs.length === 0) {
          workerAttemptedPlanWrite = true
          return [{
            toolCalls: [
              {
                id: 'pw_read',
                type: 'function',
                function: { name: 'ReadFile', arguments: { file_path: 'existing.txt' } },
              },
              {
                id: 'pw_write',
                type: 'function',
                function: { name: 'WriteFile', arguments: { file_path: 'forbidden.txt', content: 'nope\n' } },
              },
            ],
            finishReason: 'tool_calls',
          }]
        }
        const writeOutcome = toolMsgs.find(m => m.tool_call_id === 'pw_write')
        if (writeOutcome && /refused|denied|reject/i.test(writeOutcome.content)) {
          workerPlanWriteDenied = true
        }
        if (!toolMsgs.some(m => m.name === 'post_message')) {
          return [{
            toolCalls: [
              {
                id: 'pw_post',
                type: 'function',
                function: {
                  name: 'post_message',
                  arguments: { text: `@coordinator [done] Read existing.txt; write blocked=${workerPlanWriteDenied}` },
                },
              },
              {
                id: 'pw_wait',
                type: 'function',
                function: { name: 'swarm_wait', arguments: { since: 1, mentions_only: true } },
              },
            ],
            finishReason: 'tool_calls',
          }]
        }
        return [{ content: '', finishReason: 'stop' }]
      }

      // `auto-worker`: runs in `auto` mode, starts a file monitor + foreground command, and parks!
      if (sys.includes('You are "auto-worker"')) {
        if (toolMsgs.length === 0) {
          return [{
            toolCalls: [
              {
                id: 'aw_write',
                type: 'function',
                function: { name: 'WriteFile', arguments: { file_path: 'auto-created.txt', content: 'auto-ok\n' } },
              },
              {
                id: 'aw_mon',
                type: 'function',
                function: { name: 'monitor_file', arguments: { file_path: 'existing.txt', duration_seconds: 120 } },
              },
              {
                id: 'aw_cmd',
                type: 'function',
                function: { name: 'exec_command', arguments: { cmd: 'sh', args: ['-c', 'echo "auto-cmd-ran"'] } },
              },
            ],
            finishReason: 'tool_calls',
          }]
        }
        return [{
          toolCalls: [{
            id: 'aw_wait',
            type: 'function',
            function: { name: 'swarm_wait', arguments: { since: 10, mentions_only: true } },
          }],
          finishReason: 'tool_calls',
        }]
      }

      return [{ content: 'ok', finishReason: 'stop' }]
    })

    const runtime = new InMemoryDaemonRuntime(
      new AgentTurnRunner({ llm: workerLlm, model: 'gpt-4.1' }),
      {
        currentProjectDirectory: workspace,
        model: 'gpt-4.1',
        sessionDirectory: join(root, 'sessions'),
      },
    )

    const swarmHost = new DaemonSwarmHost({
      storageRootDir: join(root, 'swarm-state'),
      fallbackLlm: workerLlm,
      defaultModel: () => 'gpt-4.1',
      backgroundCommands,
      terminalRegistry: terminals,
      terminalMonitors: monitors,
      sessionLookup: id => runtime.listSessions().find(s => s.id === id),
      permissionMode: (session) => {
        if (session?.interactionMode === 'plan') return 'plan'
        return (session?.permissionMode as 'auto' | 'accept-all' | undefined) ?? 'auto'
      },
    })

    const socketPath = join(root, 'daemon.sock')
    const server = new DaemonServer({
      socketPath,
      projectDirectory: workspace,
      runtime,
      swarmHost,
      terminalRegistry: terminals,
      monitors,
      runHistory: history,
      autoTitle: false,
      autoSnapshotTurns: false,
      eventJournal: { directory: join(root, 'events') },
    })
    await server.start()

    const client = await SocketRpcClient.open(socketPath)
    try {
      const init = await client.call('initialize', {
        session_key: 'perm-session',
        project_dir: workspace,
      })
      const sessionId = String(init.session_id)

      // 1. Switch parent session into `/plan` mode and spawn `plan-worker`
      const planSlash = await client.call('slash', { command: '/plan' })
      expect(planSlash.ok).toBe(true)

      const manager = await swarmHost.ensureSession({ id: sessionId, cwd: workspace, model: 'gpt-4.1' })
      await manager.add({ name: 'plan-worker', brief: 'Inspect existing.txt in plan mode.' })
      await manager.waitForIdle()

      expect(workerAttemptedPlanWrite).toBe(true)
      expect(workerPlanWriteDenied).toBe(true)
      expect(existsSync(join(workspace, 'forbidden.txt'))).toBe(false)

      // 2. Leave plan mode into `auto` permission mode and spawn `auto-worker`
      await client.call('slash', { command: '/plan off' })
      await client.call('slash', { command: '/permissions auto' })

      await manager.add({ name: 'auto-worker', brief: 'Create auto-created.txt and monitor existing.txt.' })
      await manager.waitForIdle()

      expect(await readFile(join(workspace, 'auto-created.txt'), 'utf8')).toBe('auto-ok\n')

      // Verify parent session sees the worker's file monitor AND terminal in RPCs and `/monitors` + `/loops`!
      const monitorList = await client.call('monitor.list', {})
      const sessionMonitors = monitorList.monitors as Array<{ id: string; state: string }>
      expect(sessionMonitors.some(m => m.state === 'watching')).toBe(true)

      const terminalList = await client.call('terminal.list', {})
      const sessionTerminals = terminalList.terminals as Array<{ id: string; command: string }>
      expect(sessionTerminals.some(t => t.command.includes('auto-cmd-ran'))).toBe(true)

      // Verify `/monitors stop <unknown>` handles errors gracefully without crashing slash RPC
      const badStop = await client.call('slash', { command: '/monitors stop missing-watch-id' })
      expect(badStop.ok).toBe(false)

      // Verify `/loops` alias works alongside `/loop`
      const loopsList = await client.call('slash', { command: '/loops' })
      expect(loopsList.ok).toBe(true)
    } finally {
      client.close()
      await backgroundCommands.disposeAll()
      monitors.close()
      await server.stop()
      history.close()
    }
  })
})

describe('Swarm Silent-Stall Monitor, TimerCondition Early-Cancel, Narrator Feedback, and Cleanup', () => {
  test('silent stall wakes monitor via schedule{DurationSeconds, TimerCondition:"any"}, monitor nudges @verifier [stall], narrator curates swarm_overview with user feedback, and [done] cleans up timers', async () => {
    const root = await makeTempDir()
    const workspace = join(root, 'swarm-proj')
    await mkdir(workspace, { recursive: true })

    const skillRegistry = new SkillRegistry()
    await skillRegistry.discover(BUNDLED_SKILLS_DIRECTORY)

    const monitorWakePrompts: string[] = []
    const narratorWakePrompts: string[] = []
    let verifierCompleted = false
    let monitorCleanedUpOnDone = false

    const swarmLlm = makeFakeLlm(async (req) => {
      const sys = String(req.messages.find(m => m.role === 'system')?.content ?? '')
      const lastUser = [...req.messages].reverse().find(m => m.role === 'user')
      const lastUserText = typeof lastUser?.content === 'string' ? lastUser.content : ''

      // Find tool messages belonging ONLY to the current turn (after the last user message).
      const lastUserIdx = req.messages.lastIndexOf(lastUser!)
      const turnToolMsgs = req.messages.slice(lastUserIdx + 1).filter(m => m.role === 'tool')
      if (turnToolMsgs.some(m => m.name === 'swarm_wait')) {
        return [{ content: '', finishReason: 'stop' }]
      }

      // --- AUTHOR: writes `module.ts`, posts WITHOUT tagging @verifier, and parks in silence ---
      if (sys.includes('You are "author"')) {
        if (turnToolMsgs.length === 0) {
          return [{
            toolCalls: [{
              id: 'auth_write_1',
              type: 'function',
              function: {
                name: 'WriteFile',
                arguments: { file_path: 'module.ts', content: 'export const status = "ready"\n' },
              },
            }],
            finishReason: 'tool_calls',
          }]
        }
        if (!turnToolMsgs.some(m => m.name === 'post_message')) {
          return [{
            toolCalls: [
              {
                id: 'auth_post_unaddressed',
                type: 'function',
                function: {
                  name: 'post_message',
                  // Deliberate silent-handoff bug: forgot to mention `@verifier`!
                  arguments: { text: 'Finished writing module.ts and unit checks.' },
                },
              },
              {
                id: 'auth_wait',
                type: 'function',
                function: { name: 'swarm_wait', arguments: { since: 5, mentions_only: true } },
              },
            ],
            finishReason: 'tool_calls',
          }]
        }
        return [{ content: '', finishReason: 'stop' }]
      }

      // --- VERIFIER: parks on startup waiting for `@verifier`; verifies once monitor nudges it! ---
      if (sys.includes('You are "verifier"')) {
        if (lastUserText.includes('You have just started')) {
          return [{
            toolCalls: [{
              id: 'ver_initial_wait',
              type: 'function',
              function: { name: 'swarm_wait', arguments: { since: 4, mentions_only: true } },
            }],
            finishReason: 'tool_calls',
          }]
        }

        // Woken by `@verifier [stall] ...` from `monitor`!
        if (!turnToolMsgs.some(m => m.name === 'read_board')) {
          return [{
            toolCalls: [
              {
                id: 'ver_read_board',
                type: 'function',
                function: { name: 'read_board', arguments: { since: 0, limit: 20 } },
              },
              {
                id: 'ver_read_file',
                type: 'function',
                function: { name: 'ReadFile', arguments: { file_path: 'module.ts' } },
              },
            ],
            finishReason: 'tool_calls',
          }]
        }
        if (!turnToolMsgs.some(m => m.name === 'post_message')) {
          verifierCompleted = true
          return [{
            toolCalls: [
              {
                id: 'ver_post_done',
                type: 'function',
                function: {
                  name: 'post_message',
                  arguments: { text: '@coordinator [done] Verified module.ts after monitor stall recovery.' },
                },
              },
              {
                id: 'ver_wait_done',
                type: 'function',
                function: { name: 'swarm_wait', arguments: { since: 6, mentions_only: true } },
              },
            ],
            finishReason: 'tool_calls',
          }]
        }
        return [{ content: '', finishReason: 'stop' }]
      }

      // --- NARRATOR: never calls post_message; updates swarm_overview & incorporates user feedback ---
      if (sys.includes('You are "narrator"')) {
        narratorWakePrompts.push(lastUserText)
        if (!turnToolMsgs.some(m => m.name === 'read_board')) {
          return [{
            toolCalls: [{
              id: 'nar_read',
              type: 'function',
              function: { name: 'read_board', arguments: { since: 0, limit: 50 } },
            }],
            finishReason: 'tool_calls',
          }]
        }
        const boardData = JSON.parse(turnToolMsgs.find(m => m.name === 'read_board')!.content) as {
          head: number
          messages: Array<{ text: string }>
        }
        const feedbackEntry = boardData.messages.find(m => m.text.startsWith('@narrator user feedback:'))
        const summary = feedbackEntry
          ? `Incorporated feedback: ${feedbackEntry.text}`
          : `Tracking swarm at #${boardData.head}`
        if (!turnToolMsgs.some(m => m.name === 'swarm_overview')) {
          return [{
            toolCalls: [
              {
                id: 'nar_overview',
                type: 'function',
                function: {
                  name: 'swarm_overview',
                  arguments: {
                    summary,
                    markdown: `## Objective\nVerify module.ts\n\n## Notes\n${summary}`,
                  },
                },
              },
              {
                id: 'nar_wait',
                type: 'function',
                function: { name: 'swarm_wait', arguments: { since: boardData.head, mentions_only: false } },
              },
            ],
            finishReason: 'tool_calls',
          }]
        }
        return [{ content: '', finishReason: 'stop' }]
      }

      // --- MONITOR: loads `swarm_monitor` skill, arms silent-stall timer, catches stall, cleans up on [done] ---
      if (sys.includes('You are "monitor"')) {
        if (turnToolMsgs.length === 0) {
          monitorWakePrompts.push(lastUserText)
        }

        // First step on every turn: inspect roster status + board messages + active tasks
        if (!turnToolMsgs.some(m => m.name === 'read_board')) {
          return [{
            toolCalls: [
              ...(lastUserText.includes('You have just started')
                ? [{
                    id: 'mon_skill',
                    type: 'function' as const,
                    function: { name: 'SkillTool', arguments: { skill_name: 'swarm_monitor' } },
                  }]
                : []),
              {
                id: 'mon_status',
                type: 'function',
                function: { name: 'read_board', arguments: { status: true } },
              },
              {
                id: 'mon_messages',
                type: 'function',
                function: { name: 'read_board', arguments: { since: 0, limit: 50 } },
              },
              {
                id: 'mon_tasks',
                type: 'function',
                function: { name: 'manage_task', arguments: { Action: 'list' } },
              },
            ],
            finishReason: 'tool_calls',
          }]
        }

        const boardMsgs = JSON.parse(
          turnToolMsgs.find(m => m.tool_call_id === 'mon_messages')!.content,
        ) as {
          head: number
          messages: Array<{ seq: number; agent: string; text: string }>
        }
        const statusData = JSON.parse(
          turnToolMsgs.find(m => m.tool_call_id === 'mon_status')!.content,
        ) as {
          members: Array<{ name: string; role: string; state: string; killed: boolean }>
        }
        const taskList = JSON.parse(
          turnToolMsgs.find(m => m.tool_call_id === 'mon_tasks')!.content,
        ) as {
          tasks: Array<{ task_id: string }>
        }

        // Check if coordinator concluded with `[done]` -> kill any active schedule and finish!
        const coordinatorDone = boardMsgs.messages.some(
          m => m.agent === 'coordinator' && m.text.includes('[done]'),
        )
        if (coordinatorDone) {
          if (taskList.tasks.length > 0 && !turnToolMsgs.some(m => m.tool_call_id === 'mon_kill_task')) {
            return [{
              toolCalls: taskList.tasks.map((t, idx) => ({
                id: idx === 0 ? 'mon_kill_task' : `mon_kill_task_${idx}`,
                type: 'function' as const,
                function: { name: 'manage_task', arguments: { Action: 'kill', TaskId: t.task_id } },
              })),
              finishReason: 'tool_calls',
            }]
          }
          monitorCleanedUpOnDone = true
          return [{ content: '', finishReason: 'stop' }]
        }

        // On `[schedule]` wakeup: if all non-monitor/non-narrator workers are waiting and verifier hasn't posted [done], nudge `@verifier [stall]`!
        const activeWorkers = statusData.members.filter(
          m => !m.killed && m.role === 'worker',
        )
        const allWorkersIdle = activeWorkers.length > 0 && activeWorkers.every(m => m.state === 'waiting')
        const verifierDoneMsg = boardMsgs.messages.some(m => m.agent === 'verifier' && m.text.includes('[done]'))
        const alreadyStallPosted = boardMsgs.messages.some(m => m.agent === 'monitor' && m.text.includes('[stall]'))
        const authorUnaddressed = boardMsgs.messages.find(
          m => m.agent === 'author' && m.text.includes('module.ts'),
        )

        if (
          lastUserText.startsWith('[schedule]')
          && allWorkersIdle
          && authorUnaddressed
          && !verifierDoneMsg
          && !alreadyStallPosted
          && !turnToolMsgs.some(m => m.name === 'post_message')
        ) {
          return [{
            toolCalls: [
              {
                id: 'mon_post_stall',
                type: 'function',
                function: {
                  name: 'post_message',
                  arguments: {
                    text: `@verifier [stall] All workers are waiting and #${authorUnaddressed.seq} finished module.ts without tagging @verifier — please verify #${authorUnaddressed.seq}.`,
                  },
                },
              },
              {
                id: 'mon_rearm_after_stall',
                type: 'function',
                function: {
                  name: 'schedule',
                  arguments: {
                    DurationSeconds: 0.25,
                    TimerCondition: 'any',
                    Prompt: 'Follow up after stall nudge',
                  },
                },
              },
              {
                id: 'mon_wait_after_stall',
                type: 'function',
                function: { name: 'swarm_wait', arguments: { since: boardMsgs.head + 1, mentions_only: false } },
              },
            ],
            finishReason: 'tool_calls',
          }]
        }

        // Otherwise stay completely silent, arm silent-stall timer (`TimerCondition: "any"`), and park!
        if (!turnToolMsgs.some(m => m.name === 'schedule')) {
          return [{
            toolCalls: [
              {
                id: 'mon_arm_timer',
                type: 'function',
                function: {
                  name: 'schedule',
                  arguments: {
                    DurationSeconds: 0.15,
                    TimerCondition: 'any',
                    Prompt: 'Check swarm roster and board for silent stall',
                  },
                },
              },
              {
                id: 'mon_park',
                type: 'function',
                function: { name: 'swarm_wait', arguments: { since: boardMsgs.head, mentions_only: false } },
              },
            ],
            finishReason: 'tool_calls',
          }]
        }

        return [{ content: '', finishReason: 'stop' }]
      }

      return [{ content: 'ok', finishReason: 'stop' }]
    })

    const runtime = new InMemoryDaemonRuntime(
      new AgentTurnRunner({ llm: swarmLlm, model: 'gpt-4.1' }),
      {
        currentProjectDirectory: workspace,
        model: 'gpt-4.1',
        sessionDirectory: join(root, 'sessions'),
      },
    )

    const swarmHost = new DaemonSwarmHost({
      storageRootDir: join(root, 'swarm-state'),
      fallbackLlm: swarmLlm,
      defaultModel: () => 'gpt-4.1',
      skillRegistryForWorkspace: () => skillRegistry,
      sessionLookup: id => runtime.listSessions().find(s => s.id === id),
    })

    const socketPath = join(root, 'daemon.sock')
    const server = new DaemonServer({
      socketPath,
      projectDirectory: workspace,
      runtime,
      skillRegistry,
      swarmHost,
      autoTitle: false,
      autoSnapshotTurns: false,
      eventJournal: { directory: join(root, 'events') },
    })
    await server.start()

    const client = await SocketRpcClient.open(socketPath)
    try {
      const init = await client.call('initialize', {
        session_key: 'stall-recovery-session',
        project_dir: workspace,
      })
      const sessionId = String(init.session_id)
      const manager = await swarmHost.ensureSession({ id: sessionId, cwd: workspace, model: 'gpt-4.1' })

      await manager.setTitle({ title: 'Silent Stall & Narrator Suite' })
      await manager.add({ name: 'verifier', brief: 'Wait for @verifier handoff, then verify module.ts.' })
      await manager.add({ name: 'narrator', brief: 'Curate overview via swarm_overview.' })
      await manager.add({ name: 'monitor', brief: 'Monitor swarm for silent stalls.' })
      await manager.add({ name: 'author', brief: 'Implement module.ts.' })

      // Wait for silent-stall timer to fire, wake `monitor`, post `@verifier [stall] ...`, and wake `verifier`!
      await waitUntil(() => verifierCompleted, 5_000)
      await manager.waitForIdle()

      // Confirm `monitor` actually received a `[schedule] ...` timer wakeup and emitted ZERO board messages except the single `[stall]` post!
      expect(monitorWakePrompts.some(p => p.startsWith('[schedule] Check swarm roster and board for silent stall'))).toBe(true)

      const boardSnapshot = await manager.readBoard({ limit: 50 })
      const allMessages = 'messages' in boardSnapshot ? boardSnapshot.messages : []
      const monitorPosts = allMessages.filter(m => m.agent === 'monitor')
      expect(monitorPosts).toHaveLength(1)
      expect(monitorPosts[0]!.text).toContain('@verifier [stall]')

      // Confirm `narrator` NEVER posted to the board, yet updated `swarm_overview`!
      expect(allMessages.filter(m => m.agent === 'narrator')).toHaveLength(0)
      expect(manager.status().overview?.author).toBe('narrator')

      // Send user feedback to `@narrator` via `swarm.post` (`as_feedback: true`) and verify `swarm_overview` updates!
      await client.call('swarm.post', {
        session_id: sessionId,
        text: 'Highlight @author and @verifier test status concisely.',
        as_feedback: true,
      })
      await waitUntil(() => {
        const summary = manager.status().overview?.summary ?? ''
        return summary.includes('Incorporated feedback: @narrator user feedback: Highlight author and verifier test status concisely.')
      }, 3_000)

      // Still zero board posts from `narrator`!
      const boardAfterFeedback = await manager.readBoard({ limit: 50 })
      const msgsAfterFeedback = 'messages' in boardAfterFeedback ? boardAfterFeedback.messages : []
      expect(msgsAfterFeedback.filter(m => m.agent === 'narrator')).toHaveLength(0)

      // Post final `[done]` from coordinator and verify `monitor` cancels its schedule and parks cleanly!
      await manager.postMessage({
        caller: 'coordinator',
        text: '@monitor [done] All slices verified and integrated.',
      })
      await waitUntil(() => monitorCleanedUpOnDone, 3_000)
      await manager.waitForIdle()
      expect(swarmHost.activeWorkerScheduleCount(sessionId, 'monitor')).toBe(0)
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('TimerCondition ("any" and "<sender>") cancels early on matching board traffic and timers clean up on kill, disposeSession, and close', async () => {
    const root = await makeTempDir()
    const workspace = join(root, 'timer-lifecycle')
    await mkdir(workspace, { recursive: true })

    const scheduleWakes: Array<{ member: string; prompt: string }> = []

    const timerLlm = makeFakeLlm(async (req) => {
      const sys = String(req.messages.find(m => m.role === 'system')?.content ?? '')
      const lastUser = [...req.messages].reverse().find(m => m.role === 'user')
      const lastUserText = typeof lastUser?.content === 'string' ? lastUser.content : ''
      const lastUserIdx = req.messages.lastIndexOf(lastUser!)
      const turnToolMsgs = req.messages.slice(lastUserIdx + 1).filter(m => m.role === 'tool')

      if (lastUserText.startsWith('[schedule]')) {
        const nameMatch = /You are "([^"]+)"/.exec(sys)
        scheduleWakes.push({ member: nameMatch?.[1] ?? 'unknown', prompt: lastUserText })
        return [{ content: '', finishReason: 'stop' }]
      }

      if (sys.includes('You are "watcher-any"')) {
        if (turnToolMsgs.length === 0) {
          return [{
            toolCalls: [
              {
                id: 'wa_sched',
                type: 'function',
                function: {
                  name: 'schedule',
                  arguments: {
                    DurationSeconds: 0.25,
                    TimerCondition: 'any',
                    Prompt: 'Should cancel on any peer post',
                  },
                },
              },
              // Own outgoing post MUST NOT cancel own `TimerCondition: "any"` timer!
              {
                id: 'wa_own_post',
                type: 'function',
                function: {
                  name: 'post_message',
                  arguments: { text: 'watcher-any armed its timer.' },
                },
              },
              {
                id: 'wa_wait',
                type: 'function',
                function: { name: 'swarm_wait', arguments: { since: 10, mentions_only: true } },
              },
            ],
            finishReason: 'tool_calls',
          }]
        }
        return [{ content: '', finishReason: 'stop' }]
      }

      if (sys.includes('You are "watcher-sender"')) {
        if (turnToolMsgs.length === 0) {
          return [{
            toolCalls: [
              {
                id: 'ws_sched',
                type: 'function',
                function: {
                  name: 'schedule',
                  arguments: {
                    DurationSeconds: 0.3,
                    TimerCondition: 'verifier',
                    Prompt: 'Should cancel only when verifier posts',
                  },
                },
              },
              {
                id: 'ws_wait',
                type: 'function',
                function: { name: 'swarm_wait', arguments: { since: 10, mentions_only: true } },
              },
            ],
            finishReason: 'tool_calls',
          }]
        }
        return [{ content: '', finishReason: 'stop' }]
      }

      if (sys.includes('You are "long-timer-worker"')) {
        if (turnToolMsgs.length === 0) {
          return [{
            toolCalls: [
              {
                id: 'lt_sched',
                type: 'function',
                function: {
                  name: 'schedule',
                  arguments: {
                    DurationSeconds: 60,
                    TimerCondition: 'never',
                    Prompt: 'Long timer to test kill/dispose/close cleanup',
                  },
                },
              },
              {
                id: 'lt_cron',
                type: 'function',
                function: {
                  name: 'schedule',
                  arguments: {
                    CronExpression: '*/5 * * * *',
                    MaxIterations: 3,
                    Prompt: 'Recurring cron to test cleanup',
                  },
                },
              },
              {
                id: 'lt_wait',
                type: 'function',
                function: { name: 'swarm_wait', arguments: { since: 10, mentions_only: true } },
              },
            ],
            finishReason: 'tool_calls',
          }]
        }
        return [{ content: '', finishReason: 'stop' }]
      }

      // Passive helper workers (`author`, `verifier`) park immediately.
      if (turnToolMsgs.length === 0) {
        return [{
          toolCalls: [{
            id: 'passive_wait',
            type: 'function',
            function: { name: 'swarm_wait', arguments: { since: 10, mentions_only: true } },
          }],
          finishReason: 'tool_calls',
        }]
      }
      return [{ content: '', finishReason: 'stop' }]
    })

    const swarmHost = new DaemonSwarmHost({
      storageRootDir: join(root, 'swarm-state'),
      fallbackLlm: timerLlm,
      defaultModel: () => 'gpt-4.1',
    })

    try {
      const sessionA = 'sess-timer-a'
      const managerA = await swarmHost.ensureSession({ id: sessionA, cwd: workspace, model: 'gpt-4.1' })

      await managerA.add({ name: 'author', brief: 'Passive author.' })
      await managerA.add({ name: 'verifier', brief: 'Passive verifier.' })
      await managerA.add({ name: 'watcher-sender', brief: 'Arm TimerCondition:verifier.' })
      await managerA.add({ name: 'long-timer-worker', brief: 'Arm 60s timer + cron.' })
      await managerA.waitForIdle()

      // Spawn `watcher-any` after peer startup briefs have settled so no coordinator brief cancels it early.
      await managerA.add({ name: 'watcher-any', brief: 'Arm TimerCondition:any.' })
      await managerA.waitForIdle()

      // `watcher-any`'s own post did NOT cancel its own timer; all 4 timers are active!
      expect(swarmHost.activeWorkerScheduleCount(sessionA, 'watcher-any')).toBe(1)
      expect(swarmHost.activeWorkerScheduleCount(sessionA, 'watcher-sender')).toBe(1)
      expect(swarmHost.activeWorkerScheduleCount(sessionA, 'long-timer-worker')).toBe(2)

      // Post from `author`: cancels `watcher-any` (`TimerCondition: "any"`), does NOT cancel `watcher-sender` (`TimerCondition: "verifier"`).
      await managerA.postMessage({ caller: 'author', text: 'Progress update from author.' })
      expect(swarmHost.activeWorkerScheduleCount(sessionA, 'watcher-any')).toBe(0)
      expect(swarmHost.activeWorkerScheduleCount(sessionA, 'watcher-sender')).toBe(1)

      // Post from `verifier`: cancels `watcher-sender` (`TimerCondition: "verifier"`).
      await managerA.postMessage({ caller: 'verifier', text: 'Review note from verifier.' })
      expect(swarmHost.activeWorkerScheduleCount(sessionA, 'watcher-sender')).toBe(0)

      // Sleep past both 0.25s / 0.30s windows to prove neither early-cancelled timer ever fires!
      await Bun.sleep(360)
      expect(scheduleWakes).toEqual([])

      // Killing `long-timer-worker` cancels both its 60s timer and its cron immediately!
      await managerA.killFromHost('long-timer-worker')
      expect(swarmHost.activeWorkerScheduleCount(sessionA, 'long-timer-worker')).toBe(0)

      // Spawn new timer workers across two sessions to verify `disposeSession()` and `close()` cleanup.
      const sessionB = 'sess-timer-b'
      const sessionC = 'sess-timer-c'
      const managerB = await swarmHost.ensureSession({ id: sessionB, cwd: workspace, model: 'gpt-4.1' })
      const managerC = await swarmHost.ensureSession({ id: sessionC, cwd: workspace, model: 'gpt-4.1' })
      await managerB.add({ name: 'long-timer-worker', brief: 'Arm timers in B.' })
      await managerC.add({ name: 'long-timer-worker', brief: 'Arm timers in C.' })
      await Promise.all([managerB.waitForIdle(), managerC.waitForIdle()])

      expect(swarmHost.activeWorkerScheduleCount(sessionB)).toBe(2)
      expect(swarmHost.activeWorkerScheduleCount(sessionC)).toBe(2)

      await swarmHost.disposeSession(sessionB)
      expect(swarmHost.activeWorkerScheduleCount(sessionB)).toBe(0)
      expect(swarmHost.activeWorkerScheduleCount(sessionC)).toBe(2)

      await swarmHost.close()
      expect(swarmHost.activeWorkerScheduleCount()).toBe(0)
    } finally {
      await swarmHost.close()
    }
  })
})
