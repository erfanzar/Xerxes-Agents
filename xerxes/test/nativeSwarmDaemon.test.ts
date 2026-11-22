// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { existsSync } from 'node:fs'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import type { ProfileQuota } from '../src/auth/profileUsage.js'
import { COMMAND_REGISTRY, listCommands, resolveCommand } from '../src/bridge/commands.js'
import { FEATURES_GUIDE } from '../src/bridge/features.js'
import { ProfileStore } from '../src/bridge/profiles.js'
import { BridgeSlashRouter } from '../src/bridge/slashRouter.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import {
  createDaemonSwarmModelCatalogPort,
  createDaemonSwarmWorkspacePort,
  DaemonSwarmHost,
} from '../src/daemon/swarmHost.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { lintHarness } from '../src/extensions/harnessLint.js'
import { BUNDLED_SKILLS_DIRECTORY, SkillRegistry } from '../src/extensions/skills.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import type { JsonRpcPayload } from '../src/protocol/jsonRpc.js'
import { createAgentState } from '../src/streaming/events.js'
import { registerCoreTools } from '../src/tools/index.js'

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

async function makeTempDir(prefix = 'xrx-sw-'): Promise<string> {
  const baseTmp = process.platform === 'win32' ? tmpdir() : (existsSync('/tmp') ? '/tmp' : tmpdir())
  const dir = await mkdtemp(join(baseTmp, prefix))
  tempRoots.push(dir)
  process.env.XERXES_HOME = join(dir, 'home')
  await mkdir(process.env.XERXES_HOME, { recursive: true })
  return dir
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

describe('Immortals bundled skills and bridge slash command registry', () => {
  test('bundled immortals-coordinator, immortals-monitor, and immortals-narrator skills load cleanly with zero lint findings and legacy swarm-* alias lookup', async () => {
    const registry = new SkillRegistry()
    await registry.discover(BUNDLED_SKILLS_DIRECTORY)

    const coordinator = registry.get('immortals-coordinator')
    const monitor = registry.get('immortals-monitor')
    const narrator = registry.get('immortals-narrator')

    expect(coordinator).toBeDefined()
    expect(monitor).toBeDefined()
    expect(narrator).toBeDefined()
    expect(coordinator!.metadata.name).toBe('immortals-coordinator')
    expect(monitor!.metadata.name).toBe('immortals-monitor')
    expect(narrator!.metadata.name).toBe('immortals-narrator')

    // Legacy `swarm-*` and underscore lookups resolve transparently without duplicating `all()`/`names`.
    expect(registry.get('swarm-coordinator')).toBe(coordinator)
    expect(registry.get('swarm_coordinator')).toBe(coordinator)
    expect(registry.get('swarm-monitor')).toBe(monitor)
    expect(registry.get('swarm_monitor')).toBe(monitor)
    expect(registry.get('swarm-narrator')).toBe(narrator)
    expect(registry.get('swarm_narrator')).toBe(narrator)
    expect(registry.has('swarm-coordinator')).toBe(true)
    expect(registry.has('swarm_monitor')).toBe(true)
    expect(registry.has('swarm_narrator')).toBe(true)
    expect(registry.all().map(s => s.metadata.name)).not.toContain('swarm-coordinator')
    expect(registry.all().map(s => s.metadata.name)).not.toContain('swarm-monitor')
    expect(registry.all().map(s => s.metadata.name)).not.toContain('swarm-narrator')

    expect(coordinator!.instructions).toContain('sidecar://immortals/board/?auto_open=false')
    expect(coordinator!.instructions).toContain('immortals-monitor')
    expect(coordinator!.instructions).toContain('immortals-narrator')
    expect(monitor!.instructions).toContain('[stall]')
    expect(monitor!.instructions).toContain('[done]')
    expect(narrator!.instructions).toContain('swarm_overview')
    expect(narrator!.instructions).toContain('Never call `post_message`')

    const docs = await Promise.all(
      [coordinator!, monitor!, narrator!].map(async skill => ({
        path: skill.sourcePath,
        content: await Bun.file(skill.sourcePath).text(),
      })),
    )
    expect(lintHarness(docs)).toEqual([])
  })

  test('COMMAND_REGISTRY, FEATURES_GUIDE, and BridgeSlashRouter expose /immortals cleanly with /swarm legacy alias', async () => {
    const cmd = resolveCommand('/immortals build auth')
    expect(cmd).toMatchObject({
      name: 'immortals',
      aliases: ['swarm'],
      category: 'activity',
      argsHint: '[<task>|board|status|channels|kill <name|all>|stop]',
    })
    expect(resolveCommand('/swarm build auth')).toBe(cmd)
    expect(listCommands('activity').map(entry => entry.name)).toContain('immortals')
    expect(COMMAND_REGISTRY.filter(entry => entry.name === 'immortals')).toHaveLength(1)
    expect(FEATURES_GUIDE).toContain('/immortals')

    const calls: string[] = []
    const router = new BridgeSlashRouter({
      cwd: '/workspace',
      state: createAgentState(),
      config: { model: 'gpt-4o' },
      host: {
        swarm: (args) => {
          calls.push(args)
          return `immortals:${args || 'board'}`
        },
      },
    })
    expect(await router.dispatch('/immortals')).toMatchObject({ status: 'handled', output: 'immortals:board' })
    expect(await router.dispatch('/swarm')).toMatchObject({ status: 'handled', output: 'immortals:board' })
    expect(await router.dispatch('/immortals stop')).toMatchObject({ status: 'handled', output: 'immortals:stop' })
    expect(await router.dispatch('/swarm stop')).toMatchObject({ status: 'handled', output: 'immortals:stop' })
    expect((await router.dispatch('/help')).output).toContain('/immortals')
    expect(calls).toEqual(['', '', 'stop', 'stop'])
  })
})

describe('Daemon v35 Swarm RPCs, /swarm slash commands, and live swarm_event streaming', () => {
  test('swarm.status, swarm.activate, swarm.post (including as_feedback), swarm.board, swarm.model, swarm.kill, and swarm_event work end-to-end', async () => {
    const root = await makeTempDir()
    const profilesPath = join(root, 'profiles.json')
    const profileStore = new ProfileStore(profilesPath)
    profileStore.save({
      name: 'primary',
      provider: 'openai',
      model: 'gpt-4.1',
      apiKey: 'sk-test-1',
      baseUrl: 'https://api.openai.com/v1',
      sampling: {},
      setActive: true,
    })
    profileStore.save({
      name: 'backup',
      provider: 'openai',
      model: 'gpt-4.1-mini',
      apiKey: 'sk-test-2',
      baseUrl: 'https://api.openai.com/v1',
      sampling: {},
      setActive: false,
    })
    profileStore.save({
      name: 'exhausted',
      provider: 'openai',
      model: 'gpt-exhausted',
      apiKey: 'sk-test-3',
      baseUrl: 'https://api.openai.com/v1',
      sampling: {},
      setActive: false,
    })

    const quotaFetcher = async (profile: { readonly name: string }): Promise<ProfileQuota> => {
      if (profile.name === 'exhausted') {
        return {
          status: 'available',
          remaining_tokens: null,
          source: 'provider_usage_endpoint',
          scope: 'profile_credentials',
          provider_profile: 'exhausted',
          observed_at: '2026-11-21T07:00:00.000Z',
          windows: [{
            label: '5-hour',
            used_percent: 100,
            resets_at: '2026-11-21T12:00:00.000Z',
            reset_after_seconds: 3600,
            detail: null,
          }],
        }
      }
      return {
        status: 'available',
        remaining_tokens: null,
        source: 'provider_usage_endpoint',
        scope: 'profile_credentials',
        provider_profile: profile.name,
        observed_at: '2026-11-21T07:00:00.000Z',
        windows: [{
          label: '5-hour',
          used_percent: 25,
          resets_at: '2026-11-21T12:00:00.000Z',
          reset_after_seconds: 3600,
          detail: null,
        }],
      }
    }

    const workerLlm = makeFakeLlm(async (req) => {
      const hasPark = req.messages.some(msg => msg.role === 'tool' && msg.name === 'swarm_wait')
      if (hasPark) {
        return [{ content: '', finishReason: 'stop' }]
      }
      return [{
        toolCalls: [{
          id: 'call_park_1',
          type: 'function',
          function: {
            name: 'swarm_wait',
            arguments: { since: 1, mentions_only: true },
          },
        }],
        finishReason: 'tool_calls',
      }]
    })

    const skillRegistry = new SkillRegistry()
    await skillRegistry.discover(BUNDLED_SKILLS_DIRECTORY)

    const swarmHost = new DaemonSwarmHost({
      storageRootDir: join(root, 'swarm-state'),
      profileStore,
      quotaFetcher,
      fallbackLlm: workerLlm,
      resolveWorkerLlm: () => workerLlm,
      defaultModel: () => 'gpt-4.1',
      skillRegistryForWorkspace: () => skillRegistry,
    })

    const runtime = new InMemoryDaemonRuntime(
      new AgentTurnRunner({ llm: workerLlm, model: 'gpt-4.1' }),
      {
        currentProjectDirectory: root,
        model: 'gpt-4.1',
        sessionDirectory: join(root, 'sessions'),
      },
    )

    const socketPath = join(root, 'daemon.sock')
    const server = new DaemonServer({
      socketPath,
      projectDirectory: root,
      runtime,
      profileStore,
      skillRegistry,
      swarmHost,
      autoTitle: false,
      autoSnapshotTurns: false,
      eventJournal: { directory: join(root, 'events') },
    })
    await server.start()

    const client = await SocketRpcClient.open(socketPath)
    const subscriber = await SocketRpcClient.open(socketPath)
    try {
      const init = await client.call('initialize', {
        session_key: 'ui-main',
        project_dir: root,
        session_owned_turns: true,
      })
      const sessionId = String(init.session_id)
      expect(sessionId.length).toBeGreaterThan(0)

      const sub = await subscriber.call('events.subscribe', {
        project_dirs: [root],
        session_ids: [sessionId],
        detail: 'summary',
      })
      expect(sub.ok).toBe(true)

      const coldStatus = await client.call('swarm.status', { session_id: sessionId })
      expect(coldStatus).toMatchObject({
        active: false,
        session_id: sessionId,
        title: '',
        board: 'local',
        open_board: '[Open the Immortals UI](sidecar://immortals/board/?auto_open=false)',
        head: 0,
        overview: null,
      })

      const activated = await client.call('swarm.activate', { session_id: sessionId, enabled: true })
      expect(activated).toMatchObject({
        ok: true,
        active: true,
      })
      expect(String(activated.prompt_overlay)).toContain('<swarm_coordinator>')
      expect(String(activated.prompt_overlay)).toContain('Xerxes Immortals Coordinator Guide')

      // Spawn a worker and a narrator through the session's SwarmManager and verify live swarm_event frames.
      const manager = await swarmHost.ensureSession({ id: sessionId, cwd: root, model: 'gpt-4.1' })
      const modelsList = await manager.listModels({ all: true })
      expect(modelsList.models.find(m => m.id === 'gpt-4.1')?.quota_remaining).toBe(75)
      expect(modelsList.models.find(m => m.id === 'gpt-exhausted')?.quota_exhausted).toBe(true)

      await expect(
        manager.add({ name: 'bad-quota', brief: 'Should fail', model: 'gpt-exhausted' }),
      ).rejects.toThrow(/exhausted its quota/i)

      await manager.setTitle({ title: 'Daemon RPC Contract Suite' })
      await manager.add({ name: 'author', brief: 'Inspect module A and wait.' })
      await manager.add({ name: 'narrator', brief: 'Curate overview.' })
      await manager.waitForIdle()

      // Verify socket observer received live `swarm_event` frames.
      const liveSwarmFrame = await client.next(
        frame => frame.method === 'event' && frame.params?.type === 'swarm_event',
      )
      expect(liveSwarmFrame.params?.payload).toMatchObject({
        session_id: sessionId,
      })

      // Verify EventSubscriptions journal also recorded `swarm_event`.
      const journalPage = await subscriber.call('events.read', {
        subscription_id: sub.subscription_id,
        after_cursor: sub.watermark,
      })
      const journalEvents = journalPage.events as Array<{ readonly type: string; readonly payload: JsonRpcPayload }>
      expect(journalEvents.some(item => item.type === 'swarm_event')).toBe(true)

      // Post normal message and narrator UI feedback (`as_feedback: true` strips @ before member names).
      const normalPost = await client.call('swarm.post', {
        session_id: sessionId,
        text: 'Focus on deterministic error paths first.',
      })
      expect(normalPost).toMatchObject({
        status: 'posted',
        channel: 'main',
      })

      const feedbackPost = await client.call('swarm.post', {
        session_id: sessionId,
        text: 'Please track @author progress and @all milestones in a table.',
        as_feedback: true,
      })
      expect(feedbackPost).toMatchObject({
        status: 'posted',
        channel: 'main',
      })

      const boardAll = await client.call('swarm.board', { session_id: sessionId, limit: 20 })
      const messages = boardAll.messages as Array<{
        readonly seq: number
        readonly agent: string
        readonly text: string
        readonly mentions: string[]
      }>
      const feedbackMsg = messages.find(m => m.seq === feedbackPost.seq)
      expect(feedbackMsg).toBeDefined()
      expect(feedbackMsg!.text).toBe(
        '@narrator user feedback: Please track author progress and all milestones in a table.',
      )
      expect(feedbackMsg!.mentions).toEqual(['narrator'])

      const boardSeqs = await client.call('swarm.board', {
        session_id: sessionId,
        seqs: [Number(normalPost.seq)],
      })
      expect((boardSeqs.messages as Array<{ readonly seq: number }>).map(m => m.seq)).toEqual([
        Number(normalPost.seq),
      ])

      await manager.waitForIdle()

      // Switch waiting worker's model via `swarm.model` RPC.
      const switched = await client.call('swarm.model', {
        session_id: sessionId,
        name: 'author',
        model: 'gpt-4.1-mini',
      })
      expect(switched).toEqual({
        status: 'switched',
        name: 'author',
        model: 'gpt-4.1-mini',
      })
      await manager.waitForIdle()

      const statusAfterSwitch = await client.call('swarm.status', { session_id: sessionId })
      const members = statusAfterSwitch.members as Array<{ readonly name: string; readonly model: string }>
      expect(members.find(m => m.name === 'author')?.model).toBe('gpt-4.1-mini')

      // Inspect worker conversation & live activity via `swarm.inspect` RPC.
      const inspected = await client.call('swarm.inspect', {
        session_id: sessionId,
        name: 'author',
        limit: 20,
      })
      expect(inspected).toMatchObject({
        session_id: sessionId,
        member: { name: 'author', model: 'gpt-4.1-mini' },
      })
      expect(Array.isArray(inspected.messages)).toBe(true)

      // Test `/immortals status`, `/immortals channels`, and legacy `/swarm` aliases via daemon `slash` RPC.
      const slashStatus = await client.call('slash', { command: '/immortals status' })
      expect(slashStatus.ok).toBe(true)
      expect(String(slashStatus.text)).toContain('Immortals: Daemon RPC Contract Suite')
      expect(String(slashStatus.text)).toContain('author [worker]')
      expect(String(slashStatus.text)).toContain('Usage: /immortals')

      const legacyStatus = await client.call('slash', { command: '/swarm status' })
      expect(legacyStatus.ok).toBe(true)
      expect(String(legacyStatus.text)).toContain('Immortals: Daemon RPC Contract Suite')

      const slashChannels = await client.call('slash', { command: '/immortals channels' })
      expect(slashChannels.ok).toBe(true)
      expect(String(slashChannels.text)).toContain('Immortals channels')
      expect(String(slashChannels.text)).toContain('#main')

      // Verify `xerxes immortals` CLI (and `xerxes swarm` alias) against live daemon socket.
      const cliEntry = join(import.meta.dir, '..', 'src', 'cli.ts')
      const cliStatusProc = Bun.spawn(
        ['bun', cliEntry, 'immortals', 'status', '--socket', socketPath, '--session', sessionId],
        { stdout: 'pipe', stderr: 'pipe' },
      )
      const [cliStatusCode, cliStatusOut] = await Promise.all([
        cliStatusProc.exited,
        new Response(cliStatusProc.stdout).text(),
      ])
      expect(cliStatusCode).toBe(0)
      expect(cliStatusOut).toContain('Immortals: Daemon RPC Contract Suite')

      const cliAliasProc = Bun.spawn(
        ['bun', cliEntry, 'swarm', 'channels', '--socket', socketPath, '--session', sessionId],
        { stdout: 'pipe', stderr: 'pipe' },
      )
      const [cliAliasCode, cliAliasOut] = await Promise.all([
        cliAliasProc.exited,
        new Response(cliAliasProc.stdout).text(),
      ])
      expect(cliAliasCode).toBe(0)
      expect(cliAliasOut).toContain('Immortals channels')

      const killedRes = await client.call('swarm.kill', {
        session_id: sessionId,
        name: 'all',
      })
      expect(killedRes.status).toBe('killed')
      expect((killedRes.killed as string[]).sort()).toEqual(['author', 'narrator'])
    } finally {
      client.close()
      subscriber.close()
      await server.stop()
    }
  })

  test('coordinator dynamically unlocks board tools after first add, runs workers via runTurn, and auto-wakes idle coordinator on @coordinator', async () => {
    const root = await makeTempDir()
    const coordinatorToolSnapshots: string[][] = []
    const coordinatorUserPrompts: string[] = []
    let coordinatorTurnNumber = 0
    let finalSynthesisDelivered = false

    const unifiedLlm = makeFakeLlm(async (req) => {
      const sysMsg = req.messages.find(m => m.role === 'system')
      const systemText = typeof sysMsg?.content === 'string' ? sysMsg.content : ''
      const isWorker = systemText.includes('<swarm_worker>')

      if (isWorker) {
        const toolResults = req.messages.filter(m => m.role === 'tool')
        const wroteFile = toolResults.some(m => m.name === 'WriteFile')
        const postedDone = toolResults.some(m => m.name === 'post_message')
        if (!wroteFile) {
          return [{
            toolCalls: [{
              id: 'worker_write_1',
              type: 'function',
              function: {
                name: 'WriteFile',
                arguments: {
                  file_path: 'swarm-output.txt',
                  content: 'verified-by-worker\n',
                },
              },
            }],
            finishReason: 'tool_calls',
          }]
        }
        if (!postedDone) {
          return [{
            toolCalls: [{
              id: 'worker_post_1',
              type: 'function',
              function: {
                name: 'post_message',
                arguments: {
                  text: '@coordinator [done] Created swarm-output.txt and verified contents.',
                },
              },
            }],
            finishReason: 'tool_calls',
          }]
        }
        return [{
          toolCalls: [{
            id: 'worker_wait_1',
            type: 'function',
            function: {
              name: 'swarm_wait',
              arguments: { since: 2, mentions_only: true },
            },
          }],
          finishReason: 'tool_calls',
        }]
      }

      // Coordinator turn handling:
      const toolNames = (req.tools ?? []).map(t => t.function.name)
      const toolMessages = req.messages.filter(m => m.role === 'tool')
      const lastUser = [...req.messages].reverse().find(m => m.role === 'user')
      const lastUserText = typeof lastUser?.content === 'string' ? lastUser.content : ''

      if (toolMessages.length === 0 || (coordinatorTurnNumber === 1 && !lastUserText.includes('[board]'))) {
        if (coordinatorTurnNumber === 0) {
          coordinatorTurnNumber = 1
          coordinatorToolSnapshots.push(toolNames)
          coordinatorUserPrompts.push(lastUserText)
          return [{
            toolCalls: [{
              id: 'coord_add_1',
              type: 'function',
              function: {
                name: 'swarm',
                arguments: {
                  action: 'add',
                  name: 'builder',
                  brief: 'Create swarm-output.txt and post @coordinator [done] when complete.',
                  fork: 'coordinator',
                },
              },
            }],
            finishReason: 'tool_calls',
          }]
        }
      }

      // End of Coordinator Turn 1 (right after `swarm add` returned): let coordinator go idle
      // so the worker's `@coordinator [done]` post triggers the daemon's idle coordinator auto-wakeup!
      if (coordinatorTurnNumber === 1 && !lastUserText.includes('addressed @coordinator')) {
        return [{
          content: 'Spawned builder; waiting for completion.',
          finishReason: 'stop',
        }]
      }

      // Coordinator Turn 2 (auto-woken by `@coordinator` post!):
      if (coordinatorTurnNumber === 1 && lastUserText.includes('addressed @coordinator')) {
        coordinatorTurnNumber = 2
        coordinatorToolSnapshots.push(toolNames)
        coordinatorUserPrompts.push(lastUserText)
        return [{
          toolCalls: [{
            id: 'coord_read_1',
            type: 'function',
            function: {
              name: 'read_board',
              arguments: { since: 0, limit: 20 },
            },
          }],
          finishReason: 'tool_calls',
        }]
      }

      finalSynthesisDelivered = true
      return [{
        content: 'All swarm work verified! [Open the Immortals UI](sidecar://immortals/board/?auto_open=false)',
        finishReason: 'stop',
      }]
    })

    const skillRegistry = new SkillRegistry()
    await skillRegistry.discover(BUNDLED_SKILLS_DIRECTORY)

    const tools = new ToolRegistry({ onDuplicateRegistration: () => undefined })
    registerCoreTools(tools, { workspaceRoot: root, includeCodingTools: true })

    // Delay worker startup slightly so Coordinator Turn 1 finishes and enters `idle` first,
    // exercising the idle-coordinator reactive auto-wakeup path deterministically!
    const delayedWorkerLlm = makeFakeLlm(async (req) => {
      const sysMsg = req.messages.find(m => m.role === 'system')
      const systemText = typeof sysMsg?.content === 'string' ? sysMsg.content : ''
      if (systemText.includes('<swarm_worker>')) {
        await Bun.sleep(60)
      }
      const chunks: LlmDelta[] = []
      for await (const item of unifiedLlm.stream(req)) {
        chunks.push(item)
      }
      return chunks
    })

    const runner = new AgentTurnRunner({
      llm: delayedWorkerLlm,
      model: 'gpt-4.1',
      tools: tools.definitions(),
      toolRegistry: tools,
      toolExecutor: tools,
    })

    const runtime = new InMemoryDaemonRuntime(runner, {
      currentProjectDirectory: root,
      model: 'gpt-4.1',
      sessionDirectory: join(root, 'sessions'),
    })

    const swarmHost = new DaemonSwarmHost({
      storageRootDir: join(root, 'swarm-state'),
      fallbackLlm: delayedWorkerLlm,
      defaultModel: () => 'gpt-4.1',
      skillRegistryForWorkspace: () => skillRegistry,
      sessionLookup: id => runtime.listSessions().find(s => s.id === id),
    })

    const socketPath = join(root, 'daemon.sock')
    const server = new DaemonServer({
      socketPath,
      projectDirectory: root,
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
        session_key: 'coord-session',
        project_dir: root,
        session_owned_turns: true,
      })
      const sessionId = String(init.session_id)

      // Launch via `/immortals <task>` slash command!
      const slashRes = await client.call('slash', {
        command: '/immortals Create swarm-output.txt with a worker and verify it',
      })
      expect(slashRes).toMatchObject({
        ok: true,
        active: true,
        queued: true,
      })

      try {
        await waitUntil(() => finalSynthesisDelivered, 3_500)
      } catch (err) {
        throw new Error(
          `${err instanceof Error ? err.message : String(err)}\nFrames: ${JSON.stringify(client.frames.slice(-12))}\nSnapshots: ${JSON.stringify(coordinatorToolSnapshots)}\nPrompts: ${JSON.stringify(coordinatorUserPrompts)}`,
        )
      }

      // Verify worker actually wrote `swarm-output.txt` into the workspace via `runTurn` + `WriteFile`.
      const written = await readFile(join(root, 'swarm-output.txt'), 'utf8')
      expect(written).toBe('verified-by-worker\n')

      // Verify 1:1 one-turn-late tool unlocking:
      // Turn 1 had ONLY `swarm` (no `post_message`, `read_board`, `swarm_wait`, `swarm_overview`).
      expect(coordinatorToolSnapshots.length).toBeGreaterThanOrEqual(2)
      expect(coordinatorToolSnapshots[0]).toContain('swarm')
      expect(coordinatorToolSnapshots[0]).not.toContain('post_message')
      expect(coordinatorToolSnapshots[0]).not.toContain('read_board')
      expect(coordinatorToolSnapshots[0]).not.toContain('swarm_wait')
      expect(coordinatorToolSnapshots[0]).not.toContain('swarm_overview')

      // Turn 2 (auto-woken after first `add`) unlocked `post_message`, `read_board`, and `swarm_wait`!
      expect(coordinatorToolSnapshots[1]).toContain('swarm')
      expect(coordinatorToolSnapshots[1]).toContain('post_message')
      expect(coordinatorToolSnapshots[1]).toContain('read_board')
      expect(coordinatorToolSnapshots[1]).toContain('swarm_wait')
      expect(coordinatorToolSnapshots[1]).not.toContain('swarm_overview')

      // Verify `[board] the latest message is #...` notice was delivered on the wake turn.
      expect(coordinatorUserPrompts[1]).toContain('[board] the latest message is #')

      // Wait for session transcript to settle and verify final assistant synthesis persisted.
      await waitUntil(() => {
        const live = runtime.listSessions().find(s => s.id === sessionId)
        return Boolean(
          live
          && live.status === 'idle'
          && live.messages.some(m => m.role === 'assistant' && String(m.content ?? '').includes('All swarm work verified!')),
        )
      }, 5_000)
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('DaemonSwarmWorkspacePort allocates and cleans up share git worktrees and branch local clones', async () => {
    const repoDir = await makeTempDir('xerxes-swarm-git-')
    const gitEnv = {
      ...Bun.env,
      GIT_AUTHOR_NAME: 'Test',
      GIT_AUTHOR_EMAIL: 'test@example.com',
      GIT_COMMITTER_NAME: 'Test',
      GIT_COMMITTER_EMAIL: 'test@example.com',
    }
    const runGit = async (args: string[]) => {
      const proc = Bun.spawn(['git', ...args], { cwd: repoDir, env: gitEnv, stdout: 'pipe', stderr: 'pipe' })
      const code = await proc.exited
      expect(code).toBe(0)
    }
    await runGit(['init'])
    await writeFile(join(repoDir, 'README.md'), '# Repo\n', 'utf8')
    await mkdir(join(repoDir, 'src'), { recursive: true })
    await writeFile(join(repoDir, 'src', 'index.ts'), 'export const x = 1\n', 'utf8')
    await runGit(['add', '.'])
    await runGit(['commit', '-m', 'init'])

    const port = createDaemonSwarmWorkspacePort()

    const inherited = await port.createWorkspace({
      mode: 'inherit',
      baseDir: repoDir,
      memberName: 'reader',
      sessionId: 'sess-ws',
    })
    expect(inherited.path).toBe(repoDir)

    const shared = await port.createWorkspace({
      mode: 'share',
      baseDir: repoDir,
      memberName: 'author-share',
      sessionId: 'sess-ws',
    })
    expect(shared.path).not.toBe(repoDir)
    expect(existsSync(join(shared.path, 'src', 'index.ts'))).toBe(true)

    const branched = await port.createWorkspace({
      mode: 'branch',
      baseDir: repoDir,
      memberName: 'author-branch',
      sessionId: 'sess-ws',
    })
    expect(branched.path).toContain(join('.xerxes', 'swarm-workspaces'))
    expect(existsSync(join(branched.path, 'src', 'index.ts'))).toBe(true)

    // Modify files in both isolated workspaces, then verify cleanup deletes them cleanly!
    await writeFile(join(shared.path, 'dirty.txt'), 'dirty\n', 'utf8')
    await writeFile(join(branched.path, 'dirty.txt'), 'dirty\n', 'utf8')

    await port.cleanupWorkspace({
      mode: 'share',
      baseDir: repoDir,
      workspacePath: shared.path,
      memberName: 'author-share',
      sessionId: 'sess-ws',
      ...(shared.handle_id === undefined ? {} : { handleId: shared.handle_id }),
    })
    expect(existsSync(shared.path)).toBe(false)

    await port.cleanupWorkspace({
      mode: 'branch',
      baseDir: repoDir,
      workspacePath: branched.path,
      memberName: 'author-branch',
      sessionId: 'sess-ws',
    })
    expect(existsSync(branched.path)).toBe(false)
    expect(existsSync(join(repoDir, 'src', 'index.ts'))).toBe(true)

    // Also verify model catalog standalone helper.
    const catalog = createDaemonSwarmModelCatalogPort({
      defaultModel: () => 'claude-sonnet-4-5',
    })
    const models = await catalog.listModels({ all: true })
    expect(models).toEqual([
      { id: 'claude-sonnet-4-5', display_name: 'claude-sonnet-4-5', suggested: true },
    ])
  })

  test('fork clones coordinator transcript history and stopped workers revive across daemon restarts via swarm.post', async () => {
    const root = await makeTempDir('xrx-sw-revive-')
    const storageRootDir = join(root, 'swarm-storage')
    const forkedUserContents: string[] = []
    let revivedRuns = 0

    const makeWorkerLlm = (phase: 'first' | 'restarted') =>
      makeFakeLlm(async (req) => {
        for (const msg of req.messages) {
          if (msg.role === 'user' && typeof msg.content === 'string') {
            forkedUserContents.push(msg.content)
          }
        }
        if (phase === 'restarted') {
          revivedRuns += 1
        }
        const alreadyWaited = req.messages.some(m => m.role === 'tool' && m.name === 'swarm_wait')
        if (alreadyWaited) {
          return [{ content: '', finishReason: 'stop' }]
        }
        return [{
          toolCalls: [{
            id: `park_${phase}_${revivedRuns}`,
            type: 'function',
            function: {
              name: 'swarm_wait',
              arguments: { since: 1, mentions_only: true },
            },
          }],
          finishReason: 'tool_calls',
        }]
      })

    const runtime1 = new InMemoryDaemonRuntime(
      new AgentTurnRunner({ llm: makeWorkerLlm('first'), model: 'gpt-4.1' }),
      {
        currentProjectDirectory: root,
        model: 'gpt-4.1',
        sessionDirectory: join(root, 'sessions'),
      },
    )

    const host1 = new DaemonSwarmHost({
      storageRootDir,
      fallbackLlm: makeWorkerLlm('first'),
      defaultModel: () => 'gpt-4.1',
      sessionLookup: id => runtime1.listSessions().find(s => s.id === id),
    })

    const socket1 = join(root, 'd1.sock')
    const server1 = new DaemonServer({
      socketPath: socket1,
      projectDirectory: root,
      runtime: runtime1,
      swarmHost: host1,
      autoTitle: false,
      autoSnapshotTurns: false,
      eventJournal: { directory: join(root, 'events1') },
    })
    await server1.start()

    let sessionId = ''
    const client1 = await SocketRpcClient.open(socket1)
    try {
      const init = await client1.call('initialize', {
        session_key: 'persist-key',
        project_dir: root,
      })
      sessionId = String(init.session_id)

      // Seed coordinator session transcript so `fork: 'coordinator'` clones it!
      const liveSession = runtime1.listSessions().find(s => s.id === sessionId)!
      liveSession.messages.push(
        { role: 'user', content: 'Context from coordinator exploration: auth lives in src/auth.ts' },
        { role: 'assistant', content: 'Noted auth module path.' },
      )

      const manager1 = await host1.ensureSession({ id: sessionId, cwd: root, model: 'gpt-4.1' })
      await manager1.add({
        name: 'auditor',
        brief: 'Audit auth module.',
        fork: 'coordinator',
      })
      await manager1.waitForIdle()

      expect(forkedUserContents.some(text => text.includes('auth lives in src/auth.ts'))).toBe(true)

      // Verify `/swarm`, `/swarm board`, and `/swarm kill` validation error.
      const boardSlash = await client1.call('slash', { command: '/swarm board' })
      expect(boardSlash.ok).toBe(true)
      expect(boardSlash.open_board).toBe(true)

      const invalidKill = await client1.call('slash', { command: '/swarm kill' })
      expect(invalidKill.ok).toBe(false)
    } finally {
      client1.close()
      await server1.stop()
    }

    // Restart a fresh DaemonServer + DaemonSwarmHost against the same storageRootDir!
    const runtime2 = new InMemoryDaemonRuntime(
      new AgentTurnRunner({ llm: makeWorkerLlm('restarted'), model: 'gpt-4.1' }),
      {
        currentProjectDirectory: root,
        model: 'gpt-4.1',
        sessionDirectory: join(root, 'sessions'),
      },
    )
    const host2 = new DaemonSwarmHost({
      storageRootDir,
      fallbackLlm: makeWorkerLlm('restarted'),
      defaultModel: () => 'gpt-4.1',
      sessionLookup: id => runtime2.listSessions().find(s => s.id === id),
    })
    const socket2 = join(root, 'd2.sock')
    const server2 = new DaemonServer({
      socketPath: socket2,
      projectDirectory: root,
      runtime: runtime2,
      swarmHost: host2,
      autoTitle: false,
      autoSnapshotTurns: false,
      eventJournal: { directory: join(root, 'events2') },
    })
    await server2.start()

    const client2 = await SocketRpcClient.open(socket2)
    try {
      const reloadedStatus = await client2.call('swarm.status', { session_id: sessionId })
      const members = reloadedStatus.members as Array<{ readonly name: string; readonly state: string }>
      expect(members.find(m => m.name === 'auditor')?.state).toBe('stopped')

      // Posting `@auditor` via `swarm.post` revives the stopped worker!
      await client2.call('swarm.post', {
        session_id: sessionId,
        text: '@auditor resume your check after restart',
      })
      await waitUntil(() => revivedRuns >= 1, 3_000)
    } finally {
      client2.close()
      await server2.stop()
    }
  })

  test('complete({ text: "/immortals" }) returns /immortals Command first with zero duplicate skills (and /swarm alias resolves), and session.side_question (/btw) runs read-only without mutating session transcript', async () => {
    const root = await makeTempDir('xrx-sw-complete-btw-')
    await writeFile(join(root, 'notes.md'), 'Port 9091 is used by the local metrics server.\n', 'utf8')

    const seenToolSets: string[][] = []
    const sideLlm = makeFakeLlm(async (req) => {
      const toolNames = (req.tools ?? []).map(t => t.function.name)
      seenToolSets.push(toolNames)
      const hasRead = req.messages.some(m => m.role === 'tool' && m.name === 'ReadFile')
      if (!hasRead) {
        return [{
          toolCalls: [{
            id: 'btw_read_1',
            type: 'function',
            function: {
              name: 'ReadFile',
              arguments: { file_path: 'notes.md' },
            },
          }],
          finishReason: 'tool_calls',
        }]
      }
      return [
        { content: 'According to `notes.md`, ' },
        { content: 'metrics runs on port **9091**.', finishReason: 'stop' },
      ]
    })

    const skillRegistry = new SkillRegistry()
    await skillRegistry.discover(BUNDLED_SKILLS_DIRECTORY)

    const runtime = new InMemoryDaemonRuntime(
      new AgentTurnRunner({ llm: sideLlm, model: 'gpt-4.1' }),
      {
        currentProjectDirectory: root,
        model: 'gpt-4.1',
        sessionDirectory: join(root, 'sessions'),
      },
    )

    const socketPath = join(root, 'daemon-btw.sock')
    const server = new DaemonServer({
      socketPath,
      projectDirectory: root,
      runtime,
      skillRegistry,
      autoTitle: false,
      autoSnapshotTurns: false,
      eventJournal: { directory: join(root, 'events') },
    })
    await server.start()

    const client = await SocketRpcClient.open(socketPath)
    try {
      const init = await client.call('initialize', {
        session_key: 'btw-session',
        project_dir: root,
      })
      const sessionId = String(init.session_id)

      // 1. `/immortals` autocomplete returns `/immortals` Command at index 0 with zero duplicate skills!
      const completed = await client.call('complete', { text: '/immortals' })
      const completions = completed.completions as Array<{
        readonly value: string
        readonly label: string
        readonly meta: string
        readonly category?: string
      }>
      expect(completions[0]).toMatchObject({
        value: '/immortals',
        label: 'immortals',
        category: 'activity',
      })
      const skillLabels = completions
        .filter(entry => entry.category === undefined)
        .map(entry => entry.label)
      expect(skillLabels).toContain('immortals-coordinator')
      expect(skillLabels).toContain('immortals-monitor')
      expect(skillLabels).toContain('immortals-narrator')
      expect(skillLabels).not.toContain('swarm-coordinator')
      expect(skillLabels).not.toContain('swarm-monitor')
      expect(skillLabels).not.toContain('swarm-narrator')
      expect(new Set(skillLabels).size).toBe(skillLabels.length)

      // Legacy `/swarm` alias still surfaces `/immortals` Command in completion!
      const completedAlias = await client.call('complete', { text: '/swarm' })
      const aliasCompletions = completedAlias.completions as Array<{
        readonly value: string
        readonly label: string
        readonly category?: string
      }>
      expect(aliasCompletions[0]).toMatchObject({
        value: '/immortals',
        label: 'immortals',
        category: 'activity',
      })

      const catalog = await client.call('commands.catalog', {})
      const pairs = catalog.pairs as Array<readonly [string, string]>
      const canon = catalog.canon as Record<string, string>
      expect(pairs.some(pair => pair[0] === '/immortals')).toBe(true)
      expect(canon['/swarm']).toBe('/immortals')

      // 2. Seed session transcript and ask an ephemeral `session.side_question`!
      const liveSession = runtime.listSessions().find(s => s.id === sessionId)!
      liveSession.messages.push(
        { role: 'user', content: 'Refactor metrics exporter' },
        { role: 'assistant', content: 'Inspecting exporter configuration.' },
      )
      const messageCountBefore = liveSession.messages.length

      const sideRes = await client.call('session.side_question', {
        session_id: sessionId,
        request_id: 'req-btw-1',
        question: 'Which port is in notes.md?',
      })
      expect(sideRes).toEqual({
        ok: true,
        request_id: 'req-btw-1',
        session_id: sessionId,
        question: 'Which port is in notes.md?',
        answer: 'According to `notes.md`, metrics runs on port **9091**.',
        model: 'gpt-4.1',
      })

      // Verify only read-only tools were exposed to the side question turn!
      expect(seenToolSets.length).toBeGreaterThan(0)
      for (const name of seenToolSets[0]!) {
        expect(['ReadFile', 'GlobTool', 'GrepTool', 'FileOutlineTool', 'ListDir']).toContain(name)
      }

      // Verify live `btw_event` frames (`begin`, `delta`, `end`) arrived and session transcript stayed untouched!
      const btwEvents = client.frames
        .filter(f => f.method === 'event' && f.params?.type === 'btw_event')
        .map(f => f.params?.payload)
      expect(btwEvents.some(p => p?.kind === 'begin' && p?.request_id === 'req-btw-1')).toBe(true)
      expect(btwEvents.some(p => p?.kind === 'delta' && p?.request_id === 'req-btw-1')).toBe(true)
      expect(btwEvents.some(p => p?.kind === 'end' && p?.request_id === 'req-btw-1')).toBe(true)
      expect(liveSession.messages).toHaveLength(messageCountBefore)

      // Verify `/btw <question>` slash handler routes through side_question instead of steering!
      const slashBtw = await client.call('slash', {
        command: '/btw Confirm the port again',
      })
      expect(slashBtw.ok).toBe(true)
      expect(String(slashBtw.answer)).toContain('9091')
      expect(liveSession.messages).toHaveLength(messageCountBefore)
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('idle coordinator automatically wakes on worker quota/provider failure with model-switch remediation notice', async () => {
    const root = await makeTempDir('xrx-sw-err-wake-')
    const coordinatorWakePrompts: string[] = []

    const coordinatorLlm = makeFakeLlm(async (req) => {
      const lastUser = [...req.messages].reverse().find(m => m.role === 'user')
      const text = typeof lastUser?.content === 'string' ? lastUser.content : ''
      coordinatorWakePrompts.push(text)
      return [{ content: 'Noted worker quota failure; switching worker model.', finishReason: 'stop' }]
    })

    const failingWorkerLlm: LlmClient = {
      async complete() {
        throw new Error('insufficient_quota: model quota exhausted')
      },
      async *stream(): AsyncIterable<LlmDelta> {
        yield* []
        throw new Error('insufficient_quota: model quota exhausted')
      },
    }

    const runtime = new InMemoryDaemonRuntime(
      new AgentTurnRunner({ llm: coordinatorLlm, model: 'gpt-4.1' }),
      {
        currentProjectDirectory: root,
        model: 'gpt-4.1',
        sessionDirectory: join(root, 'sessions'),
      },
    )

    const swarmHost = new DaemonSwarmHost({
      storageRootDir: join(root, 'swarm-state'),
      fallbackLlm: failingWorkerLlm,
      resolveWorkerLlm: () => failingWorkerLlm,
      defaultModel: () => 'gpt-4.1',
      sessionLookup: id => runtime.listSessions().find(s => s.id === id),
    })

    const socketPath = join(root, 'daemon-err.sock')
    const server = new DaemonServer({
      socketPath,
      projectDirectory: root,
      runtime,
      swarmHost,
      autoTitle: false,
      autoSnapshotTurns: false,
      eventJournal: { directory: join(root, 'events') },
    })
    await server.start()

    const client = await SocketRpcClient.open(socketPath)
    try {
      const init = await client.call('initialize', {
        session_key: 'err-wake-session',
        project_dir: root,
      })
      const sessionId = String(init.session_id)
      const manager = await swarmHost.ensureSession({ id: sessionId, cwd: root, model: 'gpt-4.1' })
      await manager.add({ name: 'flaky-worker', brief: 'Run task' })

      await waitUntil(() => coordinatorWakePrompts.length >= 1, 3_500)
      expect(coordinatorWakePrompts[0]).toContain('flaky-worker')
      expect(coordinatorWakePrompts[0]).toMatch(/quota|model/i)

      // Also verify `swarm.model` RPC forwards optional `reason` to `switchModel` and worker resume history!
      const switched = await client.call('swarm.model', {
        name: 'flaky-worker',
        model: 'gpt-4.1',
        reason: 'quota exhausted on primary model',
      })
      expect(switched).toMatchObject({
        status: 'switched',
        name: 'flaky-worker',
        model: 'gpt-4.1',
        reason: 'quota exhausted on primary model',
      })
      await waitUntil(async () => {
        const inspected = await client.call('swarm.inspect', { name: 'flaky-worker' })
        const workerMsgs = (inspected.messages ?? []) as Array<{ text: string }>
        return workerMsgs.some(m => m.text.includes('quota exhausted on primary model'))
      }, 2_000)
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('session.side_question supports changeable model across profiles, emits btw_event tool pills, and parses /btw --model / -m flags without altering session model', async () => {
    const root = await makeTempDir('xrx-btw-model-')
    await writeFile(join(root, 'config.txt'), 'cache_ttl=300\n', 'utf8')

    const profileStore = new ProfileStore(join(root, 'profiles.json'))
    profileStore.save({
      name: 'openai-main',
      provider: 'openai',
      model: 'gpt-4.1',
      apiKey: 'sk-main',
      baseUrl: 'https://api.openai.com/v1',
      sampling: {},
      setActive: true,
    })
    profileStore.save({
      name: 'gemini-fast',
      provider: 'gemini',
      model: 'gemini-2.5-flash',
      apiKey: 'gm-fast',
      baseUrl: 'https://generativelanguage.googleapis.com/v1beta',
      sampling: {},
      setActive: false,
    })
    profileStore.replaceModelCapabilities('gemini-fast', {
      'gemini-2.5-flash': { context_limit: 1_000_000 },
      'gemini-2.5-pro': { context_limit: 1_000_000 },
    })

    const factoryCalls: Array<{ model: string; profileName: string | undefined; provider: string | undefined }> = []
    const defaultLlmModels: string[] = []

    const mainLlm = makeFakeLlm(async req => {
      defaultLlmModels.push(req.model)
      return [{ content: `Main runtime answered on ${req.model}.`, finishReason: 'stop' }]
    })

    const altProfileLlm = makeFakeLlm(async req => {
      const hasRead = req.messages.some(m => m.role === 'tool' && m.name === 'ReadFile')
      if (!hasRead) {
        return [{
          toolCalls: [{
            id: 'btw_alt_read_1',
            type: 'function',
            function: {
              name: 'ReadFile',
              arguments: { file_path: 'config.txt' },
            },
          }],
          finishReason: 'tool_calls',
        }]
      }
      return [{ content: `Alt model (${req.model}) confirmed cache_ttl=300.`, finishReason: 'stop' }]
    })

    const runtime = new InMemoryDaemonRuntime(
      new AgentTurnRunner({ llm: mainLlm, model: 'gpt-4.1' }),
      {
        currentProjectDirectory: root,
        model: 'gpt-4.1',
        sessionDirectory: join(root, 'sessions'),
      },
    )

    const socketPath = join(root, 'daemon-btw-model.sock')
    const server = new DaemonServer({
      socketPath,
      projectDirectory: root,
      runtime,
      profileStore,
      sideQuestionClientFactory: (model, profile) => {
        factoryCalls.push({ model, profileName: profile?.name, provider: profile?.provider })
        return altProfileLlm
      },
      autoTitle: false,
      autoSnapshotTurns: false,
      eventJournal: { directory: join(root, 'events') },
    })
    await server.start()

    const client = await SocketRpcClient.open(socketPath)
    try {
      const init = await client.call('initialize', {
        session_key: 'btw-model-sess',
        project_dir: root,
      })
      const sessionId = String(init.session_id)

      // 1. Explicit model matching a distinct profile (`gemini-2.5-pro` in `gemini-fast` catalog)
      const sideRes = await client.call('session.side_question', {
        session_id: sessionId,
        request_id: 'req-alt-1',
        question: 'What is cache_ttl in config.txt?',
        model: 'gemini-2.5-pro',
      })
      expect(sideRes).toEqual({
        ok: true,
        request_id: 'req-alt-1',
        session_id: sessionId,
        question: 'What is cache_ttl in config.txt?',
        answer: 'Alt model (gemini-2.5-pro) confirmed cache_ttl=300.',
        model: 'gemini-2.5-pro',
      })
      expect(factoryCalls[0]).toEqual({
        model: 'gemini-2.5-pro',
        profileName: 'gemini-fast',
        provider: 'gemini',
      })

      // Verify btw_event `kind: "tool"` frames (`running` -> `done`) were broadcast!
      const toolEvents = client.frames
        .filter(
          f =>
            f.method === 'event' &&
            f.params?.type === 'btw_event' &&
            f.params.payload?.kind === 'tool' &&
            f.params.payload?.request_id === 'req-alt-1',
        )
        .map(f => f.params?.payload?.tool)
      expect(toolEvents).toEqual([
        { name: 'ReadFile', target: 'config.txt', status: 'running' },
        { name: 'ReadFile', target: 'config.txt', status: 'done' },
      ])

      // 2. `/btw --model gemini-2.5-flash ...` parses flag and routes cleanly
      const slashFlag = await client.call('slash', {
        command: '/btw --model gemini-2.5-flash Check cache_ttl',
      })
      expect(slashFlag).toMatchObject({
        ok: true,
        model: 'gemini-2.5-flash',
        question: 'Check cache_ttl',
      })
      expect(factoryCalls[1]).toEqual({
        model: 'gemini-2.5-flash',
        profileName: 'gemini-fast',
        provider: 'gemini',
      })

      // 3. `/btw -m custom-stub-model ...` falls back to runtimeClient when no profile matches
      const slashShort = await client.call('slash', {
        command: '/btw -m custom-stub-model Quick question',
      })
      expect(slashShort).toMatchObject({
        ok: true,
        model: 'custom-stub-model',
        answer: 'Main runtime answered on custom-stub-model.',
      })
      expect(defaultLlmModels).toContain('custom-stub-model')

      // Main session model and transcript must remain completely unchanged!
      const liveSession = runtime.listSessions().find(s => s.id === sessionId)!
      expect(liveSession.model).toBe('gpt-4.1')
      expect(liveSession.messages).toHaveLength(0)
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('leased session-owned clients receive broadcastEvent (swarm_event, btw_event, coordinator wake text_part) exactly once with origin: "agent"', async () => {
    const root = await makeTempDir('xrx-lease-dedup-')

    const coordinatorLlm = makeFakeLlm(async (req) => {
      const lastUser = [...req.messages].reverse().find(m => m.role === 'user')
      const prompt = typeof lastUser?.content === 'string' ? lastUser.content : ''
      if (prompt.includes('Quick check?')) {
        return [
          { content: 'Side ' },
          { content: 'answer.', finishReason: 'stop' },
        ]
      }
      return [
        { content: 'All ' },
        { content: 'eight ' },
        { content: 'workers ' },
        { content: 'done.', finishReason: 'stop' },
      ]
    })

    const runtime = new InMemoryDaemonRuntime(
      new AgentTurnRunner({ llm: coordinatorLlm, model: 'gpt-4.1' }),
      {
        currentProjectDirectory: root,
        model: 'gpt-4.1',
        sessionDirectory: join(root, 'sessions'),
      },
    )

    const swarmHost = new DaemonSwarmHost({
      storageRootDir: join(root, 'swarm-state'),
      fallbackLlm: coordinatorLlm,
      resolveWorkerLlm: () => coordinatorLlm,
      defaultModel: () => 'gpt-4.1',
      sessionLookup: id => runtime.listSessions().find(s => s.id === id),
    })

    const socketPath = join(root, 'daemon-lease-dedup.sock')
    const server = new DaemonServer({
      socketPath,
      projectDirectory: root,
      runtime,
      swarmHost,
      autoTitle: false,
      autoSnapshotTurns: false,
      eventJournal: { directory: join(root, 'events') },
    })
    await server.start()

    const client = await SocketRpcClient.open(socketPath)
    try {
      // Exact Desktop handshake: enable connection.lease + session_owned_turns so
      // `this.connections` holds the raw socket while `this.sessionObservers` holds
      // `connectionLeases.owner(raw)` wrapping the exact same socket!
      const lease = await client.call('connection.lease', {})
      expect(lease.ok).toBe(true)

      const init = await client.call('initialize', {
        session_key: 'leased-dedup-sess',
        project_dir: root,
        session_owned_turns: true,
      })
      const sessionId = String(init.session_id)

      // 1. `btw_event` must arrive exactly once per chunk (no doubled words)
      const sideRes = await client.call('session.side_question', {
        session_id: sessionId,
        request_id: 'btw-dedup-1',
        question: 'Quick check?',
      })
      expect(sideRes.ok).toBe(true)

      const btwFrames = client.frames.filter(
        f =>
          f.method === 'event' &&
          f.params?.type === 'btw_event' &&
          f.params.payload?.request_id === 'btw-dedup-1',
      )
      expect(btwFrames.filter(f => f.params?.payload?.kind === 'begin')).toHaveLength(1)
      expect(btwFrames.filter(f => f.params?.payload?.kind === 'end')).toHaveLength(1)
      const btwDeltaText = btwFrames
        .filter(f => f.params?.payload?.kind === 'delta')
        .map(f => String(f.params?.payload?.delta ?? ''))
        .join('')
      expect(btwDeltaText).toBe('Side answer.')

      // 2. `swarm_event` and coordinator wake turn (`turn_begin`, `text_part`, `turn_end`)
      //    must each arrive EXACTLY ONCE, and wake `turn_begin` must carry `origin: "agent"`!
      const manager = await swarmHost.ensureSession({ id: sessionId, cwd: root, model: 'gpt-4.1' })
      const framesBeforePost = client.frames.length

      // Post addressing @coordinator so idle coordinator wakes automatically!
      await manager.postMessage({
        caller: 'user',
        text: '@coordinator [done] All findings ready.',
      })

      // Wait for coordinator wake turn to finish
      await waitUntil(
        () =>
          client.frames
            .slice(framesBeforePost)
            .some(f => f.method === 'event' && f.params?.type === 'turn_end'),
        4_000,
      )
      // Give any accidental duplicate frames a tick to land before asserting exact counts
      await Bun.sleep(40)

      const wakeFrames = client.frames.slice(framesBeforePost).filter(f => f.method === 'event')

      const swarmMsgEvents = wakeFrames.filter(
        f =>
          f.params?.type === 'swarm_event' &&
          f.params.payload?.kind === 'message_posted' &&
          (f.params.payload?.message as Record<string, unknown> | undefined)?.agent === 'user',
      )
      expect(swarmMsgEvents).toHaveLength(1)

      const turnBegins = wakeFrames.filter(f => f.params?.type === 'turn_begin')
      expect(turnBegins).toHaveLength(1)
      expect(turnBegins[0]?.params?.payload?.origin).toBe('agent')

      const textParts = wakeFrames
        .filter(f => f.params?.type === 'text_part')
        .map(f => String(f.params?.payload?.text ?? ''))
      expect(textParts.join('')).toBe('All eight workers done.')

      const turnEnds = wakeFrames.filter(f => f.params?.type === 'turn_end')
      expect(turnEnds).toHaveLength(1)
    } finally {
      client.close()
      await server.stop()
    }
  })
})
