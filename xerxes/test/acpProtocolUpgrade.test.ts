// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { generateKeyPairSync, randomUUID } from 'node:crypto'
import { mkdir, mkdtemp, realpath, rm, symlink, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  AcpRpcError,
  AcpStdioClient,
  isAcpAuthRequiredError,
  normalizeAcpSessionUpdate,
  parseElicitationRequest,
  type AcpClientHostHandlers,
  type AcpSpawnedProcess,
} from '../src/acp/client.js'
import type { AcpExternalAgentInfo } from '../src/acp/externalAgents.js'
import { AGY_STRIPPED_XERXES_ENV_KEYS, buildAgySubprocessEnv } from '../src/auth/agyRuntimeEnv.js'
import {
  antigravitySteersWithPrompt,
  buildAcpPromptContentBlocks,
  buildAcpWireMcpServers,
  buildAgyLaunchArgs,
  DaemonAcpBridgeHost,
  isAntigravityUserInputRequest,
  makeAntigravityReadRepair,
  makeAntigravityUpdateStepsPlans,
  parseConfigOptions,
} from '../src/daemon/acpBridgeHost.js'
import { processAtMentions } from '../src/daemon/atMentions.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import type { DaemonTransportConnection } from '../src/daemon/transport.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import type { LlmClient, LlmDelta } from '../src/llms/client.js'
import type { JsonRpcPayload } from '../src/protocol/jsonRpc.js'
import { paramsHash, signHubAuth, type HubAuthClaims } from '../src/security/hubAuthEnvelope.js'
import { enrollHub } from '../src/security/hubIdentityVerifier.js'

const encoder = new TextEncoder()

const SAMPLE_PNG_BASE64 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='

interface InMemoryAcpHarness {
  readonly closeStdinCalled: () => boolean
  readonly killSignals: Array<string | number | undefined>
  readonly pushAgentFrame: (frame: unknown) => void
  readonly receivedFrames: Array<Record<string, unknown>>
  readonly terminate: (code?: number | null) => void
  readonly transport: AcpSpawnedProcess
}

function createInMemoryAcpTransport(
  onClientFrame?: (frame: Record<string, unknown>, harness: InMemoryAcpHarness) => void | Promise<void>,
  options?: {
    readonly exitOnStdinClose?: boolean
    readonly exitOnKill?: boolean
  },
): InMemoryAcpHarness {
  const receivedFrames: Array<Record<string, unknown>> = []
  const killSignals: Array<string | number | undefined> = []
  let stdoutController: ReadableStreamDefaultController<Uint8Array> | undefined
  let stderrController: ReadableStreamDefaultController<Uint8Array> | undefined
  let resolveExit: ((code: number | null) => void) | undefined
  let exited = false
  let stdinClosed = false

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
      // already closed
    }
    try {
      stderrController?.close()
    } catch {
      // already closed
    }
    resolveExit?.(code)
  }

  let buffer = ''
  const harness: InMemoryAcpHarness = {
    closeStdinCalled: () => stdinClosed,
    killSignals,
    receivedFrames,
    pushAgentFrame(frame: unknown) {
      if (exited) return
      stdoutController?.enqueue(encoder.encode(`${JSON.stringify(frame)}\n`))
    },
    terminate,
    transport: {
      writeLine(line: string) {
        const trimmed = line.trim()
        if (!trimmed) return
        const parsed = JSON.parse(trimmed) as Record<string, unknown>
        receivedFrames.push(parsed)
        queueMicrotask(() => {
          void onClientFrame?.(parsed, harness)
        })
      },
      closeStdin() {
        stdinClosed = true
        if (options?.exitOnStdinClose) {
          queueMicrotask(() => terminate(0))
        }
      },
      stdout,
      stderr,
      exited: exitedPromise,
      kill(signal?: number | NodeJS.Signals) {
        killSignals.push(signal)
        if (options?.exitOnKill !== false) {
          terminate(0)
        }
      },
    },
  }
  return harness
}

interface DaemonFrame {
  readonly id?: unknown
  readonly method?: string
  readonly params?: {
    readonly type?: string
    readonly payload?: Record<string, unknown>
  }
  readonly result?: Record<string, unknown>
  readonly error?: { readonly code: number; readonly message: string }
}

/**
 * In-memory DaemonTransportConnection driving DaemonServer.handleLine directly
 * without binding a Unix domain socket, keeping tests fast and sandbox-clean.
 */
class InMemoryDaemonClient {
  readonly frames: DaemonFrame[] = []
  private readonly waiters: Array<{
    readonly match: (frame: DaemonFrame) => boolean
    readonly resolve: (frame: DaemonFrame) => void
    readonly reject: (error: Error) => void
  }> = []
  private nextId = 1
  readonly connection: DaemonTransportConnection

  constructor(private readonly server: DaemonServer, sessionKey = 'inmem:1') {
    this.connection = {
      activeSessionKey: sessionKey,
      send: (frame: object) => {
        const parsed = frame as DaemonFrame
        this.frames.push(parsed)
        for (let i = 0; i < this.waiters.length; i += 1) {
          const waiter = this.waiters[i]!
          if (waiter.match(parsed)) {
            this.waiters.splice(i, 1)
            waiter.resolve(parsed)
            return
          }
        }
      },
    }
  }

  async call(method: string, params: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    const id = this.nextId++
    const responsePromise = this.next(f => f.id === id)
    const raw = JSON.stringify({ jsonrpc: '2.0', id, method, params })
    await (
      this.server as unknown as {
        handleLine(conn: DaemonTransportConnection, line: string): Promise<void>
      }
    ).handleLine(this.connection, raw)
    const reply = await responsePromise
    if (reply.error) {
      throw new Error(`RPC ${method} failed (${reply.error.code}): ${reply.error.message}`)
    }
    return reply.result ?? {}
  }

  next(match: (frame: DaemonFrame) => boolean, timeoutMs = 4_000): Promise<DaemonFrame> {
    const existing = this.frames.find(match)
    if (existing) return Promise.resolve(existing)
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        const idx = this.waiters.findIndex(w => w.resolve === resolve)
        if (idx >= 0) this.waiters.splice(idx, 1)
        const seen = this.frames.map(f => (f.method === 'event' ? `event:${f.params?.type}` : `rpc:${String(f.id)}:${JSON.stringify(f.result ?? f.error)}`))
        reject(new Error(`Timed out waiting for daemon frame (${timeoutMs}ms); seen=${JSON.stringify(seen)}`))
      }, timeoutMs)
      this.waiters.push({
        match,
        resolve: frame => {
          clearTimeout(timer)
          resolve(frame)
        },
        reject: err => {
          clearTimeout(timer)
          reject(err)
        },
      })
    })
  }

  close(): void {
    ;(
      this.server as unknown as {
        disconnect(conn: DaemonTransportConnection): void
      }
    ).disconnect(this.connection)
  }
}

function makeUnusedNativeLlm(): LlmClient {
  return {
    async *stream(): AsyncIterable<LlmDelta> {
      throw new Error('Native LLM should not be invoked during ACP protocol tests')
    },
  }
}

function extractPromptText(params: Record<string, unknown> | undefined): string {
  if (!params || !Array.isArray(params.prompt)) return ''
  return params.prompt
    .map(block => {
      if (!block || typeof block !== 'object') return ''
      const rec = block as Record<string, unknown>
      return rec.type === 'text' && typeof rec.text === 'string' ? rec.text : ''
    })
    .filter(Boolean)
    .join('\n')
}

const AGY_AGENT: AcpExternalAgentInfo = {
  id: 'antigravity',
  name: 'Antigravity',
  provider: 'antigravity',
  transport: 'acp',
  available: true,
  command: 'agy',
  args: ['acp'],
  description: 'Mock Antigravity ACP',
}

const GENERIC_AGENT: AcpExternalAgentInfo = {
  id: 'custom-acp',
  name: 'Custom ACP',
  provider: 'acp',
  transport: 'acp',
  available: true,
  command: 'custom-acp',
  args: [],
  description: 'Mock Generic ACP',
}

const tempDirs: string[] = []
async function makeTempDir(prefix = 'xrx-acp-upg-'): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), prefix))
  tempDirs.push(dir)
  return dir
}

afterEach(async () => {
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop()!
    await rm(dir, { recursive: true, force: true }).catch(() => undefined)
  }
})

describe('ACP Protocol, Daemon & Wire Hygiene Upgrade', () => {
  test('Task 1: strict wire hygiene, gated fs/terminal capabilities, AcpRpcError.data, graceful EOF dispose, safe unknown update, and plan clear', async () => {
    // 1a. Honest capability gating (no handlers -> fs false, terminal false, elicitation omitted)
    const noCapHarness = createInMemoryAcpTransport(
      (frame, h) => {
        if (frame.method === 'initialize') {
          h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { protocolVersion: 1 } })
        } else if (frame.method === 'session/new') {
          h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { sessionId: 'spec-s1' } })
        } else if (frame.method === 'session/prompt') {
          h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { stopReason: 'end_turn' } })
        }
      },
      { exitOnStdinClose: true },
    )
    const noCapClient = new AcpStdioClient({ command: 'mock', transport: noCapHarness.transport })
    await noCapClient.initialize()
    const initParams = noCapHarness.receivedFrames[0]!.params as Record<string, unknown>
    expect(initParams.clientInfo).toMatchObject({ name: 'xerxes-agents' })
    expect(initParams.clientCapabilities).toEqual({
      auth: { terminal: true },
      fs: { readTextFile: false, writeTextFile: false },
      session: { configOptions: { boolean: {} }, notices: {}, compaction: {} },
      terminal: false,
    })

    // 1b. Strict session/prompt & session/cancel wire payloads (NO session_id, NO top-level text)
    const sess = await noCapClient.newSession({ cwd: '/tmp' })
    await noCapClient.prompt({ sessionId: sess.sessionId, text: 'hello strict spec' })
    const promptFrame = noCapHarness.receivedFrames.find(f => f.method === 'session/prompt')!
    const promptParams = promptFrame.params as Record<string, unknown>
    expect(Object.keys(promptParams).sort()).toEqual(['prompt', 'sessionId'])
    expect('session_id' in promptParams).toBe(false)
    expect('text' in promptParams).toBe(false)
    expect(promptParams.prompt).toEqual([{ type: 'text', text: 'hello strict spec' }])

    await noCapClient.cancel(sess.sessionId)
    await Bun.sleep(10)
    const cancelFrame = noCapHarness.receivedFrames.find(f => f.method === 'session/cancel')!
    const cancelParams = cancelFrame.params as Record<string, unknown>
    expect(Object.keys(cancelParams)).toEqual(['sessionId'])
    expect('session_id' in cancelParams).toBe(false)

    // 1c. Graceful dispose closes stdin and resolves cleanly on EOF WITHOUT sending SIGTERM
    await noCapClient.dispose()
    expect(noCapHarness.closeStdinCalled()).toBe(true)
    expect(noCapHarness.killSignals).toEqual([])

    // 1d. Partial fs handler gating (read only) + AcpRpcError.data preservation + SIGTERM fallback when stdin EOF is ignored
    const errHarness = createInMemoryAcpTransport(
      (frame, h) => {
        if (frame.method === 'initialize') {
          h.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { protocolVersion: 1 } })
        } else if (frame.method === 'session/new') {
          h.pushAgentFrame({
            jsonrpc: '2.0',
            id: frame.id,
            error: {
              code: -32000,
              message: 'Authentication required',
              data: { authRequired: true, detail: 'Please sign in with Google' },
            },
          })
        }
      },
      { exitOnStdinClose: false, exitOnKill: true },
    )
    const readOnlyClient = new AcpStdioClient({
      command: 'mock',
      transport: errHarness.transport,
      handlers: {
        readTextFile: async () => ({ content: 'ok' }),
      },
    })
    await readOnlyClient.initialize()
    expect((errHarness.receivedFrames[0]!.params as Record<string, unknown>).clientCapabilities).toMatchObject({
      fs: { readTextFile: true, writeTextFile: false },
      terminal: false,
    })

    let caughtError: unknown
    try {
      await readOnlyClient.newSession({ cwd: '/tmp' })
    } catch (error) {
      caughtError = error
    }
    expect(caughtError).toBeInstanceOf(AcpRpcError)
    const rpcErr = caughtError as AcpRpcError
    expect(rpcErr.code).toBe(-32000)
    expect(rpcErr.data).toEqual({ authRequired: true, detail: 'Please sign in with Google' })
    expect(rpcErr.message).toContain('Please sign in with Google')

    await readOnlyClient.dispose()
    expect(errHarness.closeStdinCalled()).toBe(true)
    expect(errHarness.killSignals).toContain('SIGTERM')

    // 1e. Unknown session/update kind must NOT emit text_delta; empty plan & plan_removed clear plans
    expect(
      normalizeAcpSessionUpdate('s1', {
        sessionUpdate: 'future_experimental_telemetry_update',
        text: 'internal telemetry must never leak into assistant prose',
        content: { type: 'text', text: 'also do not leak' },
      }),
    ).toEqual([])
    expect(normalizeAcpSessionUpdate('s1', { sessionUpdate: 'plan', entries: [] })).toEqual([
      { type: 'plan', sessionId: 's1', entries: [] },
    ])
    expect(normalizeAcpSessionUpdate('s1', { sessionUpdate: 'plan_removed' })).toEqual([
      { type: 'plan', sessionId: 's1', entries: [] },
    ])
  })

  test('Tasks 2 & 3: raw @mention ACP resource text (including literal </file>), rich prompt blocks, and verbatim Antigravity image omission', async () => {
    const root = await makeTempDir('xrx-acp-raw-')
    const trickyContent = 'const xml = "<file>inner</file>";\nexport const ok = true;\n'
    const trickyPath = join(root, 'tricky.ts')
    await writeFile(trickyPath, trickyContent, 'utf8')

    // 2a. processAtMentions exposes raw un-framed resource text with literal </file> intact
    const processed = await processAtMentions('Inspect @tricky.ts', root)
    expect(processed.resources).toHaveLength(1)
    expect(processed.resources![0]!.path).toBe(await realpath(trickyPath))
    expect(processed.resources![0]!.text).toBe(trickyContent)
    expect(processed.resources![0]!.text).not.toContain('1 | ')
    expect(processed.resources![0]!.text).not.toContain(': lines 1-')

    // 2b. buildAcpPromptContentBlocks supports resource.blob, audio, resource_link & Antigravity verbatim prompt
    const richBlocks = buildAcpPromptContentBlocks({
      agentName: 'Generic',
      provider: 'acp',
      capabilities: { image: true, audio: true, embeddedContext: true },
      text: 'Analyze assets',
      audio: [{ data: 'UklGRg==', mediaType: 'audio/wav' }],
      resources: [
        { path: trickyPath, uri: `file://${trickyPath}`, mimeType: 'text/typescript', text: trickyContent },
        { path: '/tmp/bin.wasm', uri: 'file:///tmp/bin.wasm', mimeType: 'application/wasm', blob: 'AGFzbQE=' },
        { type: 'resource_link', path: '/tmp/doc.pdf', uri: 'file:///tmp/doc.pdf', mimeType: 'application/pdf', name: 'Spec PDF', title: 'ACP Spec' },
      ],
    })
    expect(richBlocks).toEqual([
      { type: 'text', text: 'Analyze assets' },
      { type: 'resource', resource: { uri: `file://${trickyPath}`, mimeType: 'text/typescript', text: trickyContent } },
      { type: 'resource', resource: { uri: 'file:///tmp/bin.wasm', mimeType: 'application/wasm', blob: 'AGFzbQE=' } },
      { type: 'resource_link', uri: 'file:///tmp/doc.pdf', name: 'Spec PDF', mimeType: 'application/pdf', title: 'ACP Spec' },
      { type: 'audio', data: 'UklGRg==', mimeType: 'audio/wav' },
    ])

    // 3. Antigravity omits synthetic image-omitted note when promptCapabilities.image === false; non-Antigravity keeps it
    const agyNoImg = buildAcpPromptContentBlocks({
      agentName: 'Antigravity',
      provider: 'antigravity',
      capabilities: { image: false, audio: false, embeddedContext: true },
      text: 'Look at this screenshot',
      images: [{ data: SAMPLE_PNG_BASE64, mediaType: 'image/png' }],
    })
    expect(agyNoImg).toEqual([{ type: 'text', text: 'Look at this screenshot' }])

    const genericNoImg = buildAcpPromptContentBlocks({
      agentName: 'Custom ACP',
      provider: 'acp',
      capabilities: { image: false, audio: false, embeddedContext: true },
      text: 'Look at this screenshot',
      images: [{ data: SAMPLE_PNG_BASE64, mediaType: 'image/png' }],
    })
    expect((genericNoImg[0] as { text: string }).text).toContain('image attachment omitted')

    // 2c. End-to-end DaemonServer @mention sends raw file text (with literal </file>) in resource block
    const receivedPrompts: Array<Record<string, unknown>> = []
    const bridgeHost = new DaemonAcpBridgeHost({
      discoverAgents: () => [GENERIC_AGENT],
      createAcpClient: opts => {
        const h = createInMemoryAcpTransport((frame, harness) => {
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                protocolVersion: 1,
                agentCapabilities: { promptCapabilities: { image: true, audio: false, embeddedContext: true } },
              },
            })
          } else if (frame.method === 'session/new') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { sessionId: 'raw-ext-1' } })
          } else if (frame.method === 'session/prompt') {
            receivedPrompts.push(frame.params as Record<string, unknown>)
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { stopReason: 'end_turn' } })
          }
        })
        return new AcpStdioClient({ ...opts, transport: h.transport })
      },
    })
    const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm: makeUnusedNativeLlm(), model: '' }), {
      currentProjectDirectory: root,
      model: '',
      sessionDirectory: join(root, 'sessions'),
    })
    const server = new DaemonServer({
      socketPath: join(root, 'raw.sock'),
      projectDirectory: root,
      runtime,
      acpBridgeHost: bridgeHost,
      autoTitle: false,
      autoSnapshotTurns: false,
    })
    const client = new InMemoryDaemonClient(server, 'raw-s1')
    try {
      const init = await client.call('initialize', { session_key: 'raw-s1', project_dir: root })
      const sessionId = String(init.session_id)
      expect((await client.call('acp.connect', { session_id: sessionId, agent_id: 'custom-acp' })).ok).toBe(true)
      await client.call('turn.submit', { session_key: 'raw-s1', text: 'Check @tricky.ts' })
      await client.next(f => f.method === 'event' && f.params?.type === 'turn_end')
      const promptBlocks = receivedPrompts[0]!.prompt as Array<Record<string, unknown>>
      const resourceBlock = promptBlocks.find(b => b.type === 'resource') as { resource: { text: string } }
      expect(resourceBlock).toBeDefined()
      expect(resourceBlock.resource.text).toBe(trickyContent)
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('Task 4 & Task 5: Hub turn.cancel blocked on live Antigravity sessions & buildAgySubprocessEnv strips daemon XERXES_* vars', async () => {
    // Task 5: buildAgySubprocessEnv hygiene
    const env = buildAgySubprocessEnv({
      PATH: '/usr/bin',
      SSH_CONNECTION: '10.0.0.1 1234 10.0.0.2 22',
      GEMINI_API_KEY: 'test-key',
      XERXES_CWD: '/repo',
      XERXES_PROJECT_DIR: '/repo',
      XERXES_BUN_DAEMON: '1',
      XERXES_DAEMON_BUILD_ID: 'abc123',
      XERXES_CUSTOM_INTERNAL: 'leak',
      XERXES_TEST_KEEP: 'keep-me',
      ANTIGRAVITY_AGENT: '1',
    })
    for (const key of AGY_STRIPPED_XERXES_ENV_KEYS) {
      expect(env[key]).toBeUndefined()
    }
    expect(env.XERXES_CUSTOM_INTERNAL).toBeUndefined()
    expect(env.ANTIGRAVITY_AGENT).toBeUndefined()
    expect(env.XERXES_TEST_KEEP).toBe('keep-me')
    expect(env.SSH_CONNECTION).toBe('10.0.0.1 1234 10.0.0.2 22')
    expect(env.GEMINI_API_KEY).toBe('test-key')

    // Task 4: Hub turn.cancel refusal parity
    const root = await makeTempDir('xrx-acp-hub-')
    const hubKeys = generateKeyPairSync('ed25519')
    const hubEnrollments = join(root, 'hub-enrollments.json')
    enrollHub(hubEnrollments, {
      hub_id: 'hub-1',
      public_key_pem: hubKeys.publicKey.export({ type: 'spki', format: 'pem' }).toString(),
      project_dirs: [root],
    })
    const bridgeHost = new DaemonAcpBridgeHost({
      discoverAgents: () => [AGY_AGENT],
      createAcpClient: opts => {
        const h = createInMemoryAcpTransport((frame, harness) => {
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { protocolVersion: 1 } })
          } else if (frame.method === 'session/new') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { sessionId: 'agy-hub-1' } })
          }
        })
        return new AcpStdioClient({ ...opts, transport: h.transport })
      },
    })
    const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm: makeUnusedNativeLlm(), model: '' }), {
      currentProjectDirectory: root,
      model: '',
      sessionDirectory: join(root, 'sessions'),
    })
    const server = new DaemonServer({
      socketPath: join(root, 'hub.sock'),
      projectDirectory: root,
      runtime,
      acpBridgeHost: bridgeHost,
      hubEnrollmentPath: hubEnrollments,
      autoTitle: false,
      autoSnapshotTurns: false,
    })
    const client = new InMemoryDaemonClient(server, 'hub-s1')
    try {
      const init = await client.call('initialize', { session_key: 'hub-s1', project_dir: root })
      const sessionId = String(init.session_id)
      expect((await client.call('acp.connect', { session_id: sessionId, agent_id: 'antigravity' })).ok).toBe(true)

      const cancelParams: JsonRpcPayload = { session_key: 'hub-s1' }
      const claims: HubAuthClaims = {
        hub_id: 'hub-1',
        principal_id: 'bob',
        principal_kind: 'user',
        user_id: 'user-bob',
        device_id: 'bob-phone',
        display_name: 'Bob',
        workspace_id: 'ws-1',
        session_id: sessionId,
        method: 'turn.cancel',
        params_hash: paramsHash(cancelParams),
        request_id: randomUUID(),
        nonce: randomUUID(),
        issued_at: Date.now(),
        expires_at: Date.now() + 60_000,
        scopes: ['session.read', 'session.submit', 'session.steer', 'session.cancel'],
      }
      const res = await client.call('turn.cancel', {
        ...cancelParams,
        hub_auth: signHubAuth(claims, hubKeys.privateKey) as unknown as JsonRpcPayload,
      })
      expect(res).toMatchObject({ ok: false, code: 'antigravity-local-only' })
      expect(String(res.error)).toContain('Antigravity')
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('Task 6: out-of-turn permission & elicitation fallback handlers resolve cleanly and abort on disconnect', async () => {
    const root = await makeTempDir('xrx-acp-oot-')
    let activeHarness: InMemoryAcpHarness | undefined
    const bridgeHost = new DaemonAcpBridgeHost({
      discoverAgents: () => [AGY_AGENT],
      createAcpClient: opts => {
        activeHarness = createInMemoryAcpTransport((frame, harness) => {
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { protocolVersion: 1 } })
          } else if (frame.method === 'session/new') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                sessionId: 'agy-oot-1',
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
          } else if (frame.method === 'session/set_config_option' || frame.method === 'session/close') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: {} })
          }
        })
        return new AcpStdioClient({ ...opts, transport: activeHarness.transport })
      },
    })
    const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm: makeUnusedNativeLlm(), model: '' }), {
      currentProjectDirectory: root,
      model: '',
      sessionDirectory: join(root, 'sessions'),
    })
    const server = new DaemonServer({
      socketPath: join(root, 'oot.sock'),
      projectDirectory: root,
      runtime,
      acpBridgeHost: bridgeHost,
      autoTitle: false,
      autoSnapshotTurns: false,
    })
    const client = new InMemoryDaemonClient(server, 'oot-s1')
    try {
      const init = await client.call('initialize', { session_key: 'oot-s1', project_dir: root })
      const sessionId = String(init.session_id)
      // Connect in explicit 'default' mode so permissions prompt the user (not auto-approved via Turbo)
      expect(
        (await client.call('acp.connect', { session_id: sessionId, agent_id: 'antigravity', mode_id: 'default' })).ok,
      ).toBe(true)

      // 6a. Out-of-turn permission request (no active turn!)
      activeHarness!.pushAgentFrame({
        jsonrpc: '2.0',
        id: 'oot-perm-1',
        method: 'session/request_permission',
        params: {
          sessionId: 'agy-oot-1',
          toolCall: { toolCallId: 'tc-bg-1', title: 'Run background script', kind: 'execute' },
          options: [
            { optionId: 'opt-allow', name: 'Allow once', kind: 'allow_once' },
            { optionId: 'opt-deny', name: 'Deny', kind: 'reject_once' },
          ],
        },
      })
      const approvalEv = await client.next(f => f.method === 'event' && f.params?.type === 'approval_request')
      const approvalId = String(approvalEv.params!.payload!.request_id ?? approvalEv.params!.payload!.id)
      expect((await client.call('permission_response', { request_id: approvalId, response: 'approve' })).ok).toBe(true)
      await Bun.sleep(20)
      const permReply = activeHarness!.receivedFrames.find(f => f.id === 'oot-perm-1')
      expect(permReply).toMatchObject({
        id: 'oot-perm-1',
        result: { outcome: { outcome: 'selected', optionId: 'opt-allow' } },
      })

      // 6b. Out-of-turn elicitation request (no active turn!)
      activeHarness!.pushAgentFrame({
        jsonrpc: '2.0',
        id: 'oot-elicit-1',
        method: 'elicitation/create',
        params: {
          sessionId: 'agy-oot-1',
          toolCallId: 'tc-bg-q',
          mode: 'form',
          message: 'Select region',
          requestedSchema: {
            type: 'object',
            required: ['region'],
            properties: {
              region: { type: 'string', title: 'Region', enum: ['us-east', 'eu-west'] },
            },
          },
        },
      })
      const qEv = await client.next(f => f.method === 'event' && f.params?.type === 'question_request')
      const qId = String(qEv.params!.payload!.id)
      const firstQ = (qEv.params!.payload!.questions as Array<{ id: string }>)[0]!
      expect((await client.call('question_response', { request_id: qId, answers: { [firstQ.id]: 'eu-west' } })).ok).toBe(true)
      await Bun.sleep(20)
      const elicitReply = activeHarness!.receivedFrames.find(f => f.id === 'oot-elicit-1')
      expect(elicitReply).toMatchObject({
        id: 'oot-elicit-1',
        result: { action: 'accept', content: { region: 'eu-west' } },
      })

      // 6c. Pending out-of-turn permission aborts cleanly when disconnecting
      activeHarness!.pushAgentFrame({
        jsonrpc: '2.0',
        id: 'oot-perm-abort',
        method: 'session/request_permission',
        params: {
          sessionId: 'agy-oot-1',
          toolCall: { toolCallId: 'tc-bg-2', title: 'Abort me on disconnect' },
          options: [{ optionId: 'opt-allow', name: 'Allow', kind: 'allow_once' }],
        },
      })
      await client.next(
        f =>
          f.method === 'event' &&
          f.params?.type === 'approval_request' &&
          f.params.payload?.tool_call_id === 'tc-bg-2',
      )
      await client.call('acp.disconnect', { session_id: sessionId })
      await Bun.sleep(20)
      const abortReply = activeHarness!.receivedFrames.find(f => f.id === 'oot-perm-abort')
      expect(abortReply).toMatchObject({
        id: 'oot-perm-abort',
        result: { outcome: { outcome: 'cancelled' } },
      })
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('Task 7: end-to-end boolean & grouped configOptions (init, update, and acp.set_config_option)', async () => {
    // 7a. parseConfigOptions preserves boolean values (including false) & SessionConfigSelectGroup group labels
    const parsed = parseConfigOptions([
      {
        id: 'web_search',
        name: 'Web Search',
        type: 'boolean',
        currentValue: false,
      },
      {
        id: 'model',
        name: 'Model',
        category: 'model',
        type: 'select',
        currentValue: 'gemini-4',
        options: [
          {
            group: 'google',
            name: 'Google Models',
            options: [{ value: 'gemini-4', name: 'Gemini 4' }],
          },
          {
            group: 'anthropic',
            name: 'Anthropic Models',
            options: [{ value: 'claude-4', name: 'Claude 4' }],
          },
        ],
      },
    ])
    expect(parsed[0]).toMatchObject({ id: 'web_search', type: 'boolean', current_value: false })
    expect(parsed[1]!.options).toEqual([
      { value: 'gemini-4', name: 'Gemini 4', group: 'Google Models' },
      { value: 'claude-4', name: 'Claude 4', group: 'Anthropic Models' },
    ])

    // 7b. End-to-end DaemonServer acp.set_config_option boolean RPC & boolean config_option_update
    const root = await makeTempDir('xrx-acp-bool-')
    let activeHarness: InMemoryAcpHarness | undefined
    let groundingEnabled = true
    const setConfigCalls: Array<Record<string, unknown>> = []
    const bridgeHost = new DaemonAcpBridgeHost({
      discoverAgents: () => [GENERIC_AGENT],
      createAcpClient: opts => {
        activeHarness = createInMemoryAcpTransport((frame, harness) => {
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { protocolVersion: 1 } })
          } else if (frame.method === 'session/new') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                sessionId: 'bool-ext-1',
                configOptions: [
                  { id: 'grounding', name: 'Web Grounding', type: 'boolean', currentValue: groundingEnabled },
                ],
              },
            })
          } else if (frame.method === 'session/set_config_option') {
            const p = frame.params as Record<string, unknown>
            setConfigCalls.push(p)
            if (p.configId === 'grounding' && typeof p.value === 'boolean') {
              groundingEnabled = p.value
            }
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                configOptions: [
                  { id: 'grounding', name: 'Web Grounding', type: 'boolean', currentValue: groundingEnabled },
                ],
              },
            })
          }
        })
        return new AcpStdioClient({ ...opts, transport: activeHarness.transport })
      },
    })
    const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm: makeUnusedNativeLlm(), model: '' }), {
      currentProjectDirectory: root,
      model: '',
      sessionDirectory: join(root, 'sessions'),
    })
    const server = new DaemonServer({
      socketPath: join(root, 'bool.sock'),
      projectDirectory: root,
      runtime,
      acpBridgeHost: bridgeHost,
      autoTitle: false,
      autoSnapshotTurns: false,
    })
    const client = new InMemoryDaemonClient(server, 'bool-s1')
    try {
      const init = await client.call('initialize', { session_key: 'bool-s1', project_dir: root })
      const sessionId = String(init.session_id)
      const conn = await client.call('acp.connect', { session_id: sessionId, agent_id: 'custom-acp' })
      expect(conn.ok).toBe(true)
      expect((activeHarness!.receivedFrames[0]!.params as Record<string, unknown>).clientCapabilities).toMatchObject({
        session: { configOptions: { boolean: {} }, notices: {}, compaction: {} },
      })

      const setRes = await client.call('acp.set_config_option', {
        session_id: sessionId,
        config_id: 'grounding',
        value: false,
        type: 'boolean',
      })
      expect(setRes.ok).toBe(true)
      expect(setConfigCalls[0]).toEqual({
        sessionId: 'bool-ext-1',
        configId: 'grounding',
        type: 'boolean',
        value: false,
      })
      const bindingAfter = setRes.binding as { config_options: Array<{ id: string; current_value: unknown }> }
      expect(bindingAfter.config_options[0]!.current_value).toBe(false)

      // Push live boolean config_option_update notification (toggling back to true)
      activeHarness!.pushAgentFrame({
        jsonrpc: '2.0',
        method: 'session/update',
        params: {
          sessionId: 'bool-ext-1',
          update: {
            sessionUpdate: 'config_option_update',
            configId: 'grounding',
            currentValue: true,
          },
        },
      })
      await Bun.sleep(25)
      expect(bridgeHost.getBinding(sessionId)?.config_options?.[0]?.current_value).toBe(true)
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('Task 8: elicitation url mode (accept omits content, decline, elicitation/complete) & JSON Schema field constraints', async () => {
    // 8a. Schema constraints preserved on AcpElicitationField
    const parsedReq = parseElicitationRequest({
      sessionId: 's1',
      elicitationId: 'el-schema-1',
      mode: 'form',
      message: 'Configure webhook',
      requestedSchema: {
        type: 'object',
        required: ['endpoint', 'retries'],
        properties: {
          endpoint: {
            type: 'string',
            title: 'Endpoint',
            format: 'uri',
            minLength: 8,
            maxLength: 200,
            pattern: '^https://',
          },
          retries: { type: 'integer', title: 'Retries', minimum: 0, maximum: 10 },
          tags: { type: 'array', title: 'Tags', minItems: 1, maxItems: 5, items: { type: 'string', enum: ['a', 'b'] } },
        },
      },
    })
    expect(parsedReq?.elicitationId).toBe('el-schema-1')
    const endpointField = parsedReq?.fields.find(f => f.key === 'endpoint')
    const retriesField = parsedReq?.fields.find(f => f.key === 'retries')
    const tagsField = parsedReq?.fields.find(f => f.key === 'tags')
    expect(endpointField).toMatchObject({ format: 'uri', minLength: 8, maxLength: 200, pattern: '^https://' })
    expect(retriesField).toMatchObject({ minimum: 0, maximum: 10 })
    expect(tagsField).toMatchObject({ minItems: 1, maxItems: 5 })

    // 8b. End-to-end URL mode elicitation: accept omits content, decline sends decline, elicitation/complete auto-resolves
    const root = await makeTempDir('xrx-acp-url-')
    let activeHarness: InMemoryAcpHarness | undefined
    const bridgeHost = new DaemonAcpBridgeHost({
      discoverAgents: () => [GENERIC_AGENT],
      createAcpClient: opts => {
        activeHarness = createInMemoryAcpTransport((frame, harness) => {
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { protocolVersion: 1 } })
          } else if (frame.method === 'session/new') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { sessionId: 'url-ext-1' } })
          }
        })
        return new AcpStdioClient({ ...opts, transport: activeHarness.transport })
      },
    })
    const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm: makeUnusedNativeLlm(), model: '' }), {
      currentProjectDirectory: root,
      model: '',
      sessionDirectory: join(root, 'sessions'),
    })
    const server = new DaemonServer({
      socketPath: join(root, 'url.sock'),
      projectDirectory: root,
      runtime,
      acpBridgeHost: bridgeHost,
      autoTitle: false,
      autoSnapshotTurns: false,
    })
    const client = new InMemoryDaemonClient(server, 'url-s1')
    try {
      const init = await client.call('initialize', { session_key: 'url-s1', project_dir: root })
      const sessionId = String(init.session_id)
      expect((await client.call('acp.connect', { session_id: sessionId, agent_id: 'custom-acp' })).ok).toBe(true)

      // Advertises both form and url elicitation capabilities
      expect((activeHarness!.receivedFrames[0]!.params as Record<string, unknown>).clientCapabilities).toMatchObject({
        elicitation: { form: {}, url: {} },
      })

      // Case 1: URL accept -> response strictly omits `content`!
      activeHarness!.pushAgentFrame({
        jsonrpc: '2.0',
        id: 'url-req-1',
        method: 'elicitation/create',
        params: {
          sessionId: 'url-ext-1',
          elicitationId: 'el-url-accept',
          mode: 'url',
          url: 'https://accounts.example.com/authorize',
          message: 'Authorize MCP server',
        },
      })
      const q1 = await client.next(
        f => f.method === 'event' && f.params?.type === 'question_request' && f.params.payload?.elicitation_id === 'el-url-accept',
      )
      expect(q1.params!.payload).toMatchObject({
        mode: 'url',
        url: 'https://accounts.example.com/authorize',
        elicitation_id: 'el-url-accept',
        message: 'Authorize MCP server',
      })
      await client.call('question_response', {
        request_id: String(q1.params!.payload!.id),
        answers: { url_consent: 'Open link & continue' },
      })
      await Bun.sleep(20)
      const acceptReply = activeHarness!.receivedFrames.find(f => f.id === 'url-req-1')!
      expect(acceptReply.result).toEqual({ action: 'accept' })
      expect('content' in (acceptReply.result as Record<string, unknown>)).toBe(false)

      // Case 2: URL decline
      activeHarness!.pushAgentFrame({
        jsonrpc: '2.0',
        id: 'url-req-2',
        method: 'elicitation/create',
        params: {
          sessionId: 'url-ext-1',
          elicitationId: 'el-url-decline',
          mode: 'url',
          url: 'https://accounts.example.com/decline',
          message: 'Decline test',
        },
      })
      const q2 = await client.next(
        f => f.method === 'event' && f.params?.type === 'question_request' && f.params.payload?.elicitation_id === 'el-url-decline',
      )
      await client.call('question_response', {
        request_id: String(q2.params!.payload!.id),
        answers: { url_consent: 'Decline' },
      })
      await Bun.sleep(20)
      expect(activeHarness!.receivedFrames.find(f => f.id === 'url-req-2')?.result).toEqual({ action: 'decline' })

      // Case 3: Agent pushes `elicitation/complete` notification -> auto-resolves & broadcasts question_resolved
      activeHarness!.pushAgentFrame({
        jsonrpc: '2.0',
        id: 'url-req-3',
        method: 'elicitation/create',
        params: {
          sessionId: 'url-ext-1',
          elicitationId: 'el-url-complete',
          mode: 'url',
          url: 'https://accounts.example.com/oauth',
          message: 'Complete in browser',
        },
      })
      const q3 = await client.next(
        f => f.method === 'event' && f.params?.type === 'question_request' && f.params.payload?.elicitation_id === 'el-url-complete',
      )
      const q3Id = String(q3.params!.payload!.id)
      activeHarness!.pushAgentFrame({
        jsonrpc: '2.0',
        method: 'elicitation/complete',
        params: { elicitationId: 'el-url-complete' },
      })
      const resolvedEv = await client.next(
        f => f.method === 'event' && f.params?.type === 'question_resolved' && f.params.payload?.id === q3Id,
      )
      expect(resolvedEv.params!.payload).toMatchObject({
        id: q3Id,
        session_id: sessionId,
        elicitation_id: 'el-url-complete',
        reason: 'complete',
      })
      await Bun.sleep(20)
      expect(activeHarness!.receivedFrames.find(f => f.id === 'url-req-3')?.result).toEqual({ action: 'accept' })
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('Task 9: multi-diff, rich content blocks, live incremental tool_update, kind updates, _meta, and usage_update.cost', async () => {
    // 9a. Direct normalization of multi-diff, rich contentBlocks, agent_message_chunk blocks, and cost
    const agentRich = normalizeAcpSessionUpdate('s1', {
      sessionUpdate: 'agent_message_chunk',
      content: [
        { type: 'text', text: 'Chart generated.' },
        { type: 'image', data: SAMPLE_PNG_BASE64, mimeType: 'image/png' },
        { type: 'resource_link', uri: 'file:///repo/report.pdf', name: 'report.pdf', mimeType: 'application/pdf' },
      ],
      _meta: { traceId: 'tr-1' },
    })
    expect(agentRich).toHaveLength(1)
    expect(agentRich[0]).toMatchObject({
      type: 'text_delta',
      text: 'Chart generated.',
      meta: { traceId: 'tr-1' },
    })
    expect((agentRich[0] as { contentBlocks?: unknown[] }).contentBlocks).toHaveLength(3)

    const usageCost = normalizeAcpSessionUpdate('s1', {
      sessionUpdate: 'usage_update',
      used: 12000,
      size: 200000,
      cost: { amount: 0.0425, currency: 'USD' },
    })
    expect(usageCost).toEqual([
      {
        type: 'usage_update',
        sessionId: 's1',
        contextUsed: 12000,
        contextSize: 200000,
        cost: { amount: 0.0425, currency: 'USD' },
        costUsd: 0.0425,
      },
    ])

    // 9b. End-to-end DaemonServer turn emitting live tool_update, multi-diff diffs[], content_blocks[], approval diffs[], and cost_usd
    const root = await makeTempDir('xrx-acp-rich-')
    const bridgeHost = new DaemonAcpBridgeHost({
      discoverAgents: () => [GENERIC_AGENT],
      createAcpClient: opts => {
        const h = createInMemoryAcpTransport((frame, harness) => {
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { protocolVersion: 1 } })
          } else if (frame.method === 'session/new') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { sessionId: 'rich-ext-1' } })
          } else if (frame.method === 'session/prompt') {
            const promptId = frame.id
            // 1. Initial tool_call
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: 'rich-ext-1',
                update: {
                  sessionUpdate: 'tool_call',
                  toolCallId: 'tc-multi',
                  name: 'multi_edit',
                  title: 'Patch a.ts and b.ts',
                  kind: 'other',
                  status: 'pending',
                },
              },
            })
            // 2. Non-terminal in_progress tool_call_update with updated kind + partial diff
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: 'rich-ext-1',
                update: {
                  sessionUpdate: 'tool_call_update',
                  toolCallId: 'tc-multi',
                  kind: 'edit',
                  status: 'in_progress',
                  content: [
                    { type: 'diff', path: join(root, 'a.ts'), oldText: 'a = 1\n', newText: 'a = 2\n' },
                  ],
                },
              },
            })
            // 3. Permission request carrying 2 diffs
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: 'perm-multi-diff',
              method: 'session/request_permission',
              params: {
                sessionId: 'rich-ext-1',
                toolCall: {
                  toolCallId: 'tc-multi',
                  name: 'multi_edit',
                  title: 'Apply two file diffs',
                  kind: 'edit',
                  content: [
                    { type: 'diff', path: join(root, 'a.ts'), oldText: 'a = 1\n', newText: 'a = 2\n' },
                    { type: 'diff', path: join(root, 'b.ts'), oldText: 'b = 1\n', newText: 'b = 2\n' },
                  ],
                },
                options: [{ optionId: 'allow', name: 'Allow', kind: 'allow_once' }],
              },
            })
            // Wait for permission response handler below by storing promptId on harness
            ;(harness as unknown as { __promptId?: unknown }).__promptId = promptId
          } else if (frame.id === 'perm-multi-diff') {
            const promptId = (harness as unknown as { __promptId?: unknown }).__promptId
            // 4. Terminal completed tool_call_update with 2 diffs + image + terminal content block
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: 'rich-ext-1',
                update: {
                  sessionUpdate: 'tool_call_update',
                  toolCallId: 'tc-multi',
                  kind: 'edit',
                  status: 'completed',
                  content: [
                    { type: 'diff', path: join(root, 'a.ts'), oldText: 'a = 1\n', newText: 'a = 2\n' },
                    { type: 'diff', path: join(root, 'b.ts'), oldText: 'b = 1\n', newText: 'b = 2\n' },
                    { type: 'image', data: SAMPLE_PNG_BASE64, mimeType: 'image/png' },
                    { type: 'terminal', terminalId: 'term-99' },
                  ],
                },
              },
            })
            // 5. Usage update with real USD cost
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: 'rich-ext-1',
                update: {
                  sessionUpdate: 'usage_update',
                  used: 5000,
                  size: 100000,
                  cost: { amount: 0.0315, currency: 'USD' },
                },
              },
            })
            harness.pushAgentFrame({ jsonrpc: '2.0', id: promptId, result: { stopReason: 'end_turn' } })
          }
        })
        return new AcpStdioClient({ ...opts, transport: h.transport })
      },
    })
    const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm: makeUnusedNativeLlm(), model: '' }), {
      currentProjectDirectory: root,
      model: '',
      permissionMode: 'auto',
      sessionDirectory: join(root, 'sessions'),
    })
    const server = new DaemonServer({
      socketPath: join(root, 'rich.sock'),
      projectDirectory: root,
      runtime,
      acpBridgeHost: bridgeHost,
      autoTitle: false,
      autoSnapshotTurns: false,
    })
    const client = new InMemoryDaemonClient(server, 'rich-s1')
    try {
      const init = await client.call('initialize', { session_key: 'rich-s1', project_dir: root })
      const sessionId = String(init.session_id)
      expect((await client.call('acp.connect', { session_id: sessionId, agent_id: 'custom-acp' })).ok).toBe(true)

      await client.call('turn.submit', { session_key: 'rich-s1', text: 'Patch both files' })

      // Verify live non-terminal tool_update event emitted before approval/completion
      const liveUpdate = await client.next(
        f => f.method === 'event' && f.params?.type === 'tool_update' && f.params.payload?.tool_call_id === 'tc-multi',
      )
      expect(liveUpdate.params!.payload).toMatchObject({
        status: 'in_progress',
        kind: 'edit',
        diff: { path: join(root, 'a.ts'), old_text: 'a = 1\n', new_text: 'a = 2\n' },
      })

      // Verify approval_request carries multi-diff diffs[]
      const approvalEv = await client.next(f => f.method === 'event' && f.params?.type === 'approval_request')
      expect((approvalEv.params!.payload!.diffs as unknown[]).length).toBe(2)
      await client.call('permission_response', {
        request_id: String(approvalEv.params!.payload!.request_id ?? approvalEv.params!.payload!.id),
        response: 'approve',
      })

      // Verify tool_result carries both diffs[] and structured content_blocks[]
      const resultEv = await client.next(
        f => f.method === 'event' && f.params?.type === 'tool_result' && f.params.payload?.tool_call_id === 'tc-multi',
      )
      expect((resultEv.params!.payload!.diffs as unknown[]).length).toBe(2)
      expect(resultEv.params!.payload!.diff).toMatchObject({ path: join(root, 'a.ts') })
      const contentBlocks = resultEv.params!.payload!.content_blocks as Array<Record<string, unknown>>
      expect(contentBlocks.some(b => b.type === 'image')).toBe(true)
      expect(contentBlocks.some(b => b.type === 'terminal' && b.terminalId === 'term-99')).toBe(true)

      // Verify cost_usd propagated on status_update AND persisted session wire payload
      await client.next(f => f.method === 'event' && f.params?.type === 'turn_end')
      const statusWithCost = client.frames.find(
        f => f.method === 'event' && f.params?.type === 'status_update' && f.params.payload?.cost_usd === 0.0315,
      )
      expect(statusWithCost).toBeDefined()
      const reopened = await client.call('session.open', { session_key: 'rich-s1', project_dir: root })
      expect((reopened.session as Record<string, unknown>).cost_usd).toBe(0.0315)
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('Task 10: authenticate RPC, full authMethods (agent & terminal), auth_required connect failure, and /acp auth', async () => {
    const root = await makeTempDir('xrx-acp-auth-')
    let authenticated = false
    const authCalls: string[] = []
    let activeClientRef: AcpStdioClient | undefined

    const bridgeHost = new DaemonAcpBridgeHost({
      discoverAgents: () => [GENERIC_AGENT],
      createAcpClient: opts => {
        const h = createInMemoryAcpTransport((frame, harness) => {
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                protocolVersion: 1,
                authMethods: [
                  { id: 'agent-sso', name: 'Agent SSO', type: 'agent', description: 'Built-in agent auth' },
                  {
                    id: 'cli-login',
                    name: 'Terminal Login',
                    type: 'terminal',
                    args: ['--login', '--no-browser'],
                    env: { AUTH_FLOW: 'interactive' },
                    _meta: { 'terminal-auth': { command: 'custom-acp', args: ['login'], label: 'Sign in via CLI' } },
                  },
                ],
              },
            })
          } else if (frame.method === 'authenticate') {
            const methodId = String((frame.params as Record<string, unknown>).methodId)
            authCalls.push(methodId)
            authenticated = true
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: {} })
          } else if (frame.method === 'session/new') {
            if (!authenticated) {
              harness.pushAgentFrame({
                jsonrpc: '2.0',
                id: frame.id,
                error: { code: -32000, message: 'Authentication required', data: { authRequired: true } },
              })
            } else {
              harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { sessionId: 'auth-ext-1' } })
            }
          }
        })
        activeClientRef = new AcpStdioClient({ ...opts, transport: h.transport })
        return activeClientRef
      },
    })
    const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm: makeUnusedNativeLlm(), model: '' }), {
      currentProjectDirectory: root,
      model: '',
      sessionDirectory: join(root, 'sessions'),
    })
    const server = new DaemonServer({
      socketPath: join(root, 'auth.sock'),
      projectDirectory: root,
      runtime,
      acpBridgeHost: bridgeHost,
      autoTitle: false,
      autoSnapshotTurns: false,
    })
    const client = new InMemoryDaemonClient(server, 'auth-s1')
    try {
      const init = await client.call('initialize', { session_key: 'auth-s1', project_dir: root })
      const sessionId = String(init.session_id)

      // 10a. Initial connect fails with auth_required: true and structured auth_methods
      const failConn = await client.call('acp.connect', { session_id: sessionId, agent_id: 'custom-acp' })
      expect(failConn).toMatchObject({ ok: false, auth_required: true })
      const methods = failConn.auth_methods as Array<Record<string, unknown>>
      expect(methods).toHaveLength(2)
      expect(methods[0]).toMatchObject({ id: 'agent-sso', type: 'agent' })
      expect(methods[1]).toMatchObject({ id: 'cli-login', type: 'terminal' })

      // 10b. acp.authenticate with terminal method returns launch recipe (never scrapes tokens)
      const termAuth = await client.call('acp.authenticate', {
        session_id: sessionId,
        agent_id: 'custom-acp',
        method_id: 'cli-login',
      })
      expect(termAuth).toMatchObject({
        ok: true,
        mode: 'terminal',
        command: 'custom-acp',
        args: ['--login', '--no-browser'],
        env: { AUTH_FLOW: 'interactive' },
        label: 'Sign in via CLI',
      })

      // 10c. acp.authenticate with agent method calls ACP `authenticate` RPC
      const agentAuth = await client.call('acp.authenticate', {
        session_id: sessionId,
        agent_id: 'custom-acp',
        method_id: 'agent-sso',
      })
      expect(agentAuth).toMatchObject({ ok: true, mode: 'agent' })
      expect(authCalls).toEqual(['agent-sso'])

      // 10d. Now connect succeeds and client.authenticate rejects terminal method IDs directly
      const okConn = await client.call('acp.connect', { session_id: sessionId, agent_id: 'custom-acp' })
      expect(okConn.ok).toBe(true)
      await expect(activeClientRef!.authenticate('cli-login')).rejects.toThrow(/terminal/i)

      // 10e. /acp auth slash command lists methods or runs auth
      const slashAuth = await client.call('slash', { command: '/acp auth' })
      expect(String(slashAuth.text)).toContain('agent-sso')
      expect(String(slashAuth.text)).toContain('cli-login')
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('Task 11: session/list, session/fork, /acp sessions & /acp fork, /btw fork preference, and session/load replay hydration', async () => {
    const root = await makeTempDir('xrx-acp-sf-')
    const receivedMethods: string[] = []
    let sideQuestionPromptSid = ''

    const bridgeHost = new DaemonAcpBridgeHost({
      discoverAgents: () => [GENERIC_AGENT],
      createAcpClient: opts => {
        const h = createInMemoryAcpTransport((frame, harness) => {
          if (typeof frame.method === 'string') receivedMethods.push(frame.method)
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                protocolVersion: 1,
                agentCapabilities: {
                  loadSession: true,
                  sessionCapabilities: { list: {}, fork: {}, load: {}, close: {} },
                },
              },
            })
          } else if (frame.method === 'session/new') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { sessionId: 'main-ext-1' } })
          } else if (frame.method === 'session/list') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                sessions: [
                  { sessionId: 'hist-1', cwd: root, title: 'Refactor auth', updatedAt: '2026-11-24T12:00:00Z' },
                ],
                nextCursor: 'cursor-2',
              },
            })
          } else if (frame.method === 'session/fork') {
            const parent = String((frame.params as Record<string, unknown>).sessionId)
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: { sessionId: `${parent}-forked` },
            })
          } else if (frame.method === 'session/load') {
            const sid = String((frame.params as Record<string, unknown>).sessionId)
            // Replay history during session/load!
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: sid,
                update: { sessionUpdate: 'user_message_chunk', content: { type: 'text', text: 'Replayed user ask' } },
              },
            })
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: sid,
                update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'Replayed agent reply' } },
              },
            })
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { sessionId: sid } })
          } else if (frame.method === 'session/prompt') {
            sideQuestionPromptSid = String((frame.params as Record<string, unknown>).sessionId)
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: sideQuestionPromptSid,
                update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'Forked answer!' } },
              },
            })
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { stopReason: 'end_turn' } })
          } else if (frame.method === 'session/close') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: {} })
          }
        })
        return new AcpStdioClient({ ...opts, transport: h.transport })
      },
    })
    const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm: makeUnusedNativeLlm(), model: '' }), {
      currentProjectDirectory: root,
      model: '',
      sessionDirectory: join(root, 'sessions'),
    })
    const server = new DaemonServer({
      socketPath: join(root, 'sf.sock'),
      projectDirectory: root,
      runtime,
      acpBridgeHost: bridgeHost,
      autoTitle: false,
      autoSnapshotTurns: false,
    })
    const client = new InMemoryDaemonClient(server, 'sf-s1')
    try {
      const init = await client.call('initialize', { session_key: 'sf-s1', project_dir: root })
      const sessionId = String(init.session_id)

      // Connect by loading historical session `hist-1` to test session/load replay hydration
      const conn = await client.call('acp.connect', {
        session_id: sessionId,
        agent_id: 'custom-acp',
        external_session_id: 'hist-1',
      })
      expect(conn.ok).toBe(true)
      expect((conn.binding as { capabilities: { session: { fork: boolean } } }).capabilities.session.fork).toBe(true)
      await Bun.sleep(20)
      const liveSession = runtime.listSessions().find(s => s.id === sessionId)!
      expect(liveSession.messages.map(m => ({ role: m.role, content: m.content }))).toEqual([
        { role: 'user', content: 'Replayed user ask' },
        { role: 'assistant', content: 'Replayed agent reply' },
      ])

      // acp.sessions -> session/list
      const listed = await client.call('acp.sessions', { session_id: sessionId })
      expect(listed).toMatchObject({
        ok: true,
        next_cursor: 'cursor-2',
        sessions: [{ session_id: 'hist-1', title: 'Refactor auth' }],
      })

      // /acp sessions slash command
      const slashList = await client.call('slash', { command: '/acp sessions' })
      expect(String(slashList.text)).toContain('hist-1')

      // runSideQuestion prefers session/fork when supported and sends forkedPrompt verbatim without replay
      const sideRes = await bridgeHost.runSideQuestion({
        sessionId,
        prompt: 'Fallback with replay: Quick fork check?',
        forkedPrompt: 'Quick fork check?',
      })
      expect(sideRes.ok).toBe(true)
      expect(sideRes.answer).toBe('Forked answer!')
      expect(sideQuestionPromptSid).toBe('hist-1-forked')
      expect(receivedMethods).toContain('session/fork')

      // Daemon session.side_question also uses Antigravity ACP session/fork cleanly (zero native fallback)
      const daemonSideRes = await client.call('session.side_question', {
        session_id: sessionId,
        question: 'Quick fork check?',
      })
      expect(daemonSideRes).toMatchObject({
        ok: true,
        answer: 'Forked answer!',
      })

      // acp.fork RPC forks active external session
      const forked = await client.call('acp.fork', { session_id: sessionId })
      expect(forked.ok).toBe(true)
      expect((forked.binding as { external_session_id: string }).external_session_id).toBe('hist-1-forked')
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('Task 12: native mid-turn live steering (concurrent session/prompt) & automatic fallback to queued_steer', async () => {
    const root = await makeTempDir('xrx-acp-steer-')
    let rejectConcurrent = false
    let activePromptId: unknown
    let activeHarness: InMemoryAcpHarness | undefined
    const receivedPrompts: string[] = []

    const bridgeHost = new DaemonAcpBridgeHost({
      discoverAgents: () => [AGY_AGENT],
      createAcpClient: opts => {
        activeHarness = createInMemoryAcpTransport((frame, harness) => {
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                protocolVersion: 1,
                agentInfo: { name: 'Antigravity ACP Server', version: '1.5.15' },
              },
            })
          } else if (frame.method === 'session/new') {
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { sessionId: 'agy-steer-1' } })
          } else if (frame.method === 'session/prompt') {
            const text = extractPromptText(frame.params as Record<string, unknown>)
            if (activePromptId !== undefined) {
              if (rejectConcurrent) {
                harness.pushAgentFrame({
                  jsonrpc: '2.0',
                  id: frame.id,
                  error: { code: -32002, message: 'session prompt already in progress' },
                })
                return
              }
              // Live concurrent steer accepted!
              receivedPrompts.push(`LIVE:${text}`)
              harness.pushAgentFrame({
                jsonrpc: '2.0',
                method: 'session/update',
                params: {
                  sessionId: 'agy-steer-1',
                  update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: ` [steered: ${text}]` } },
                },
              })
              harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { stopReason: 'end_turn' } })
              return
            }
            receivedPrompts.push(text)
            if (text.startsWith('HOLD')) {
              activePromptId = frame.id
              harness.pushAgentFrame({
                jsonrpc: '2.0',
                method: 'session/update',
                params: {
                  sessionId: 'agy-steer-1',
                  update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'Working...' } },
                },
              })
              return
            }
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: 'agy-steer-1',
                update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: `Queued handled: ${text}` } },
              },
            })
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { stopReason: 'end_turn' } })
          }
        })
        return new AcpStdioClient({ ...opts, transport: activeHarness.transport })
      },
    })
    const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm: makeUnusedNativeLlm(), model: '' }), {
      currentProjectDirectory: root,
      model: '',
      sessionDirectory: join(root, 'sessions'),
    })
    const server = new DaemonServer({
      socketPath: join(root, 'steer.sock'),
      projectDirectory: root,
      runtime,
      acpBridgeHost: bridgeHost,
      autoTitle: false,
      autoSnapshotTurns: false,
    })
    const client = new InMemoryDaemonClient(server, 'steer-s1')
    try {
      const init = await client.call('initialize', { session_key: 'steer-s1', project_dir: root })
      const sessionId = String(init.session_id)
      expect((await client.call('acp.connect', { session_id: sessionId, agent_id: 'antigravity' })).ok).toBe(true)

      // 12a. Live steer succeeds concurrently on agy 1.5.15
      await client.call('turn.submit', { session_key: 'steer-s1', text: 'HOLD_LIVE' })
      await client.next(f => f.method === 'event' && f.params?.type === 'text_part' && f.params.payload?.text === 'Working...')
      const liveSteer = await client.call('turn.steer', { session_key: 'steer-s1', content: 'Focus on perf' })
      expect(liveSteer).toMatchObject({ ok: true, disposition: 'live_steer' })
      await client.next(
        f => f.method === 'event' && f.params?.type === 'text_part' && f.params.payload?.text === ' [steered: Focus on perf]',
      )
      const holdId = activePromptId
      activePromptId = undefined
      activeHarness!.pushAgentFrame({ jsonrpc: '2.0', id: holdId, result: { stopReason: 'end_turn' } })
      await client.next(f => f.method === 'event' && f.params?.type === 'turn_end')
      await Bun.sleep(25)
      expect(receivedPrompts).toContain('LIVE:Focus on perf')

      // 12b. When agent rejects concurrent prompt, automatically falls back to queued_steer and drains cleanly
      rejectConcurrent = true
      const startIdx = client.frames.length
      await client.call('turn.submit', { session_key: 'steer-s1', text: 'HOLD_FALLBACK' })
      await client.next(
        f => client.frames.indexOf(f) >= startIdx && f.method === 'event' && f.params?.type === 'text_part',
      )
      const queuedSteer = await client.call('turn.steer', { session_key: 'steer-s1', content: 'Run tests next' })
      expect(queuedSteer).toMatchObject({ ok: true, disposition: 'queued_steer' })
      const hold2Id = activePromptId
      activePromptId = undefined
      activeHarness!.pushAgentFrame({ jsonrpc: '2.0', id: hold2Id, result: { stopReason: 'end_turn' } })
      await client.next(
        f =>
          client.frames.indexOf(f) >= startIdx &&
          f.method === 'event' &&
          f.params?.type === 'text_part' &&
          f.params.payload?.text === 'Queued handled: Run tests next',
      )
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('Task 13: MCP pass-through filtering, agy flags (--process-per-session, --add-dir, --ide-mcp-port, --agent, --sandbox), additionalDirectories fs scope, and opt-in terminal/* handlers', async () => {
    // 13a. buildAcpWireMcpServers filters http/sse by mcpCapabilities while always keeping stdio
    const filtered = buildAcpWireMcpServers(
      [
        { name: 'local-stdio', transport: 'stdio', command: 'mcp-stdio', args: ['--flag'] },
        { name: 'remote-http', transport: 'streamable_http', url: 'http://127.0.0.1:8080/mcp' },
        { name: 'remote-sse', transport: 'sse', url: 'http://127.0.0.1:8081/sse' },
      ],
      { http: true, sse: false },
    )
    expect(filtered).toEqual([
      { name: 'local-stdio', command: 'mcp-stdio', args: ['--flag'], env: [] },
      { name: 'remote-http', type: 'http', url: 'http://127.0.0.1:8080/mcp', headers: [] },
    ])

    // 13b. buildAgyLaunchArgs adds --process-per-session for HTTP MCP, plus --add-dir, --ide-mcp-port, --agent, --sandbox
    const agyArgs = buildAgyLaunchArgs('agy', ['acp'], {
      additionalDirectories: ['/extra/dir1', '/extra/dir2'],
      agent: 'architect',
      ideMcpPort: 41234,
      sandbox: true,
      mcpServers: [{ name: 'remote-http', transport: 'streamable_http', url: 'http://127.0.0.1:8080/mcp' }],
    })
    expect(agyArgs).toEqual([
      'acp',
      '--process-per-session',
      '--add-dir',
      '/extra/dir1',
      '--add-dir',
      '/extra/dir2',
      '--ide-mcp-port',
      '41234',
      '--agent',
      'architect',
      '--sandbox',
    ])

    // 13c. additionalDirectories passed in session/new AND whitelisted for fs/read_text_file (while outside dir denied)
    const root = await makeTempDir('xrx-acp-dirs-')
    const extraDir = await makeTempDir('xrx-acp-extra-')
    const outsideDir = await makeTempDir('xrx-acp-out-')
    const extraFile = join(extraDir, 'shared.txt')
    const outsideFile = join(outsideDir, 'secret.txt')
    await writeFile(extraFile, 'allowed extra file', 'utf8')
    await writeFile(outsideFile, 'denied outside file', 'utf8')

    let spawnedArgs: readonly string[] = []
    let newSessionParams: Record<string, unknown> | undefined
    let activeHarness: InMemoryAcpHarness | undefined

    const host = new DaemonAcpBridgeHost({
      discoverAgents: () => [AGY_AGENT],
      createAcpClient: opts => {
        spawnedArgs = opts.args ?? []
        activeHarness = createInMemoryAcpTransport((frame, harness) => {
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                protocolVersion: 1,
                agentCapabilities: {
                  mcpCapabilities: { http: true, sse: false },
                  sessionCapabilities: { additionalDirectories: {} },
                },
              },
            })
          } else if (frame.method === 'session/new') {
            newSessionParams = frame.params as Record<string, unknown>
            harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { sessionId: 'dirs-ext-1' } })
          }
        })
        return new AcpStdioClient({ ...opts, transport: activeHarness.transport })
      },
    })

    const conn = await host.connect({
      sessionId: 'dirs-s1',
      agentId: 'antigravity',
      cwd: root,
      additionalDirectories: [extraDir],
      agent: 'planner',
      ideMcpPort: 9222,
      sandbox: true,
      mcpServers: [
        { name: 'http-tools', transport: 'streamable_http', url: 'http://127.0.0.1:9999/mcp' },
        { name: 'sse-tools', transport: 'sse', url: 'http://127.0.0.1:9998/sse' },
      ],
    })
    expect(conn.ok).toBe(true)
    expect(spawnedArgs).toContain('--process-per-session')
    expect(spawnedArgs).toContain('--add-dir')
    expect(spawnedArgs).toContain(extraDir)
    expect(spawnedArgs).toContain('--ide-mcp-port')
    expect(spawnedArgs).toContain('9222')
    expect(spawnedArgs).toContain('--agent')
    expect(spawnedArgs).toContain('planner')
    expect(spawnedArgs).toContain('--sandbox')
    expect(newSessionParams?.additionalDirectories).toEqual([extraDir])
    expect(newSessionParams?.mcpServers).toEqual([
      { name: 'http-tools', type: 'http', url: 'http://127.0.0.1:9999/mcp', headers: [] },
    ])

    // Agent reads file inside additionalDirectories -> allowed!
    activeHarness!.pushAgentFrame({
      jsonrpc: '2.0',
      id: 'fs-extra',
      method: 'fs/read_text_file',
      params: { sessionId: 'dirs-ext-1', path: extraFile },
    })
    // Agent reads file outside workspace & additionalDirectories -> denied!
    activeHarness!.pushAgentFrame({
      jsonrpc: '2.0',
      id: 'fs-outside',
      method: 'fs/read_text_file',
      params: { sessionId: 'dirs-ext-1', path: outsideFile },
    })
    await Bun.sleep(30)
    expect(activeHarness!.receivedFrames.find(f => f.id === 'fs-extra')?.result).toEqual({
      content: 'allowed extra file',
    })
    expect(activeHarness!.receivedFrames.find(f => f.id === 'fs-outside')?.error).toBeDefined()
    await host.disposeAll()

    // 13d. Optional terminal/* handlers strictly gate clientCapabilities.terminal: true and dispatch all 5 RPCs
    const terminalCalls: string[] = []
    const termHarness = createInMemoryAcpTransport((frame, harness) => {
      if (frame.method === 'initialize') {
        harness.pushAgentFrame({ jsonrpc: '2.0', id: frame.id, result: { protocolVersion: 1 } })
      }
    })
    const handlers: AcpClientHostHandlers = {
      createTerminal: async req => {
        terminalCalls.push(`create:${req.command}`)
        return { terminalId: 'term-1' }
      },
      getTerminalOutput: async req => {
        terminalCalls.push(`output:${req.terminalId}`)
        return { output: 'hello\n', truncated: false }
      },
      waitForTerminalExit: async req => {
        terminalCalls.push(`wait:${req.terminalId}`)
        return { exitCode: 0 }
      },
      killTerminal: async req => {
        terminalCalls.push(`kill:${req.terminalId}`)
      },
      releaseTerminal: async req => {
        terminalCalls.push(`release:${req.terminalId}`)
      },
    }
    const termClient = new AcpStdioClient({ command: 'mock', transport: termHarness.transport, handlers })
    await termClient.initialize()
    expect((termHarness.receivedFrames[0]!.params as Record<string, unknown>).clientCapabilities).toMatchObject({
      terminal: true,
    })

    termHarness.pushAgentFrame({ jsonrpc: '2.0', id: 't1', method: 'terminal/create', params: { sessionId: 's1', command: 'echo', args: ['hi'] } })
    termHarness.pushAgentFrame({ jsonrpc: '2.0', id: 't2', method: 'terminal/output', params: { sessionId: 's1', terminalId: 'term-1' } })
    termHarness.pushAgentFrame({ jsonrpc: '2.0', id: 't3', method: 'terminal/wait_for_exit', params: { sessionId: 's1', terminalId: 'term-1' } })
    termHarness.pushAgentFrame({ jsonrpc: '2.0', id: 't4', method: 'terminal/kill', params: { sessionId: 's1', terminalId: 'term-1' } })
    termHarness.pushAgentFrame({ jsonrpc: '2.0', id: 't5', method: 'terminal/release', params: { sessionId: 's1', terminalId: 'term-1' } })
    await Bun.sleep(30)

    expect(terminalCalls).toEqual(['create:echo', 'output:term-1', 'wait:term-1', 'kill:term-1', 'release:term-1'])
    expect(termHarness.receivedFrames.find(f => f.id === 't1')?.result).toEqual({ terminalId: 'term-1' })
    expect(termHarness.receivedFrames.find(f => f.id === 't2')?.result).toEqual({ output: 'hello\n', truncated: false })
    expect(termHarness.receivedFrames.find(f => f.id === 't3')?.result).toEqual({ exitCode: 0 })
    expect(termHarness.receivedFrames.find(f => f.id === 't4')?.result).toEqual({})
    expect(termHarness.receivedFrames.find(f => f.id === 't5')?.result).toEqual({})
    await termClient.dispose()
  })

  test('Task 14: T3 Code agy parity (fs.writeTextFile=false, protocolVersion=2, interaction_* questions in Turbo, agy.security.warning, update_steps->plan, client_view_file repair, symlink/binary fs guard, PDF resource_link, version-gated steer, notice/compaction, Crashpad env)', async () => {
    // 14a. Env hygiene matches T3 Code (strips CHROME_CRASHPAD_PIPE_NAME & ELECTRON_RUN_AS_NODE, sets PYTHONUNBUFFERED=1)
    const cleanEnv = buildAgySubprocessEnv({
      PATH: '/usr/bin',
      CHROME_CRASHPAD_PIPE_NAME: '/tmp/crashpad',
      ELECTRON_RUN_AS_NODE: '1',
    })
    expect(cleanEnv.CHROME_CRASHPAD_PIPE_NAME).toBeUndefined()
    expect(cleanEnv.ELECTRON_RUN_AS_NODE).toBeUndefined()
    expect(cleanEnv.PYTHONUNBUFFERED).toBe('1')

    // 14b. Steering version gate (1.4.x queues; 1.5.0+ steers live) & -32603 "No browser is available here" auth detection
    expect(antigravitySteersWithPrompt('1.4.9')).toBe(false)
    expect(antigravitySteersWithPrompt('1.5.0')).toBe(true)
    expect(antigravitySteersWithPrompt('1.5.15')).toBe(true)
    expect(antigravitySteersWithPrompt(undefined)).toBe(true)
    expect(isAcpAuthRequiredError(new AcpRpcError('authenticate', -32603, 'No browser is available here to complete sign-in'))).toBe(true)

    // 14c. PDFs sent as resource_link for Antigravity instead of base64 blob
    const pdfBlocks = buildAcpPromptContentBlocks({
      agentName: 'Antigravity',
      provider: 'antigravity',
      capabilities: { image: true, audio: true, embeddedContext: true },
      text: 'Summarize report',
      resources: [
        {
          uri: 'file:///tmp/report.pdf',
          path: '/tmp/report.pdf',
          mimeType: 'application/pdf',
          blob: 'JVBERi0xLjQ=',
          size: 2048,
        },
      ],
    })
    expect(pdfBlocks).toEqual([
      { type: 'text', text: 'Summarize report' },
      {
        type: 'resource_link',
        uri: 'file:///tmp/report.pdf',
        name: 'report.pdf',
        mimeType: 'application/pdf',
        size: 2048,
      },
    ])

    // 14d. ACP v2 session updates: notice, compaction_update, compaction_summary_chunk, plan_update
    const noticeEvents = normalizeAcpSessionUpdate('s1', {
      sessionUpdate: 'notice',
      severity: 'warning',
      title: 'Quota low',
      description: 'Approaching 5-hour window limit.',
    })
    expect(noticeEvents).toEqual([
      {
        type: 'notice',
        sessionId: 's1',
        severity: 'warning',
        title: 'Quota low',
        description: 'Approaching 5-hour window limit.',
      },
    ])
    const compactionEvents = normalizeAcpSessionUpdate('s1', {
      sessionUpdate: 'compaction_update',
      compactionId: 'cmp-1',
      status: 'completed',
      summary: [{ type: 'text', text: 'Compacted 42 turns' }],
    })
    expect(compactionEvents[0]).toMatchObject({
      type: 'tool_call_update',
      toolCallId: 'acp-compaction:cmp-1',
      title: 'Compact context',
      kind: 'think',
      status: 'completed',
      content: 'Compacted 42 turns',
    })
    const planUpdateEvents = normalizeAcpSessionUpdate('s1', {
      sessionUpdate: 'plan_update',
      plan: {
        planId: 'p1',
        type: 'items',
        entries: [{ content: 'Build bundle', status: 'in_progress' }],
      },
    })
    expect(planUpdateEvents).toEqual([
      { type: 'plan', sessionId: 's1', entries: [{ content: 'Build bundle', status: 'in_progress' }] },
    ])

    // 14e. Run update_steps -> live plan event & drop noise updates
    const updateSteps = makeAntigravityUpdateStepsPlans()
    expect(
      updateSteps({
        type: 'tool_call',
        sessionId: 's1',
        toolCallId: 'tc-steps',
        name: 'Run update_steps',
        title: 'Run update_steps',
        kind: 'other',
        rawInput: {
          steps: [
            { title: 'Audit', status: 'done' },
            { title: 'Build', status: 'in_progress' },
          ],
        },
      }),
    ).toEqual({
      type: 'plan',
      sessionId: 's1',
      entries: [
        { content: 'Audit', priority: 'medium', status: 'completed' },
        { content: 'Build', priority: 'medium', status: 'in_progress' },
      ],
    })
    expect(
      updateSteps({
        type: 'tool_call_update',
        sessionId: 's1',
        toolCallId: 'tc-steps',
        status: 'completed',
      }),
    ).toBeUndefined()

    // 14f. client_view_file read repair suppresses spurious failed overwrite
    const readRepair = makeAntigravityReadRepair()
    const announced = readRepair.repair({
      type: 'tool_call',
      sessionId: 's1',
      toolCallId: 'tc-read-1',
      name: 'Running client_view_file',
      title: 'Running client_view_file',
      status: 'in_progress',
      locations: [{ path: '/ws/a.ts' }],
    })
    expect(announced).toMatchObject({ status: 'in_progress' })
    expect(readRepair.attempted('/ws/a.ts', true)).toEqual({ sessionId: 's1', toolCallId: 'tc-read-1' })
    const failedLater = readRepair.repair({
      type: 'tool_call_update',
      sessionId: 's1',
      toolCallId: 'tc-read-1',
      status: 'failed',
      isError: true,
      rawOutput: 'unrelated error',
    })
    expect(failedLater).toMatchObject({ status: 'completed' })
    expect((failedLater as Record<string, unknown>).isError).toBeUndefined()

    // 14g. E2E DaemonServer test: fs.writeTextFile=false, interaction_* question bypasses Turbo,
    //      agy.security.warning forwarded on approval options, symlink & binary fs/read_text_file rejected
    const root = await makeTempDir('xerxes-t3-parity-')
    const outsideDir = await makeTempDir('xerxes-t3-outside-')
    const secretFile = join(outsideDir, 'secret.txt')
    await writeFile(secretFile, 'do not read')
    await symlink(secretFile, join(root, 'escape-link.txt'))
    await writeFile(join(root, 'binary.bin'), Buffer.from([0x00, 0x01, 0x02, 0x03]))
    await writeFile(join(root, 'ok.txt'), 'line1\nline2\nline3\n')

    let activeHarness: InMemoryAcpHarness | undefined
    const receivedResponses: Array<Record<string, unknown>> = []
    let pendingPromptId: number | undefined

    const bridgeHost = new DaemonAcpBridgeHost({
      discoverAgents: () => [
        {
          id: 'antigravity',
          name: 'Antigravity',
          provider: 'antigravity',
          transport: 'acp',
          available: true,
          command: 'agy',
          args: ['acp'],
          description: 'Mock Antigravity ACP',
        },
      ],
      createAcpClient: opts => {
        const harness = createInMemoryAcpTransport(frame => {
          if (frame.method === 'initialize') {
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                protocolVersion: 2,
                agentInfo: { name: 'antigravity-acp', title: 'Google Antigravity', version: '1.5.15' },
                agentCapabilities: {
                  promptCapabilities: { image: true, audio: true, embeddedContext: true },
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
                sessionId: 'agy-t3-ext-1',
                modes: {
                  currentModeId: 'default',
                  availableModes: [
                    { id: 'default', name: 'Default' },
                    { id: 'auto', name: 'Auto' },
                  ],
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
            pendingPromptId = frame.id as number
            // Even while Turbo mode is active, an `interaction_` question must ask the user!
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: 'perm-interaction-1',
              method: 'session/request_permission',
              params: {
                sessionId: 'agy-t3-ext-1',
                toolCall: {
                  toolCallId: 'interaction_pick_1',
                  title: 'Which target should we deploy?',
                },
                options: [
                  { optionId: 'opt_staging', name: 'Staging', kind: 'allow_once' },
                  { optionId: 'opt_prod', name: 'Production', kind: 'allow_once' },
                ],
              },
            })
            return
          }
          if (frame.id === 'perm-interaction-1') {
            receivedResponses.push(frame)
            // Also emit a notice and update_steps during the turn
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: 'agy-t3-ext-1',
                update: {
                  sessionUpdate: 'notice',
                  severity: 'info',
                  title: 'Connected',
                  description: 'Antigravity ready',
                },
              },
            })
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              method: 'session/update',
              params: {
                sessionId: 'agy-t3-ext-1',
                update: {
                  sessionUpdate: 'tool_call',
                  toolCallId: 'tc-steps-e2e',
                  title: 'Run update_steps',
                  kind: 'other',
                  rawInput: { steps: [{ title: 'Deploy staging', status: 'done' }] },
                },
              },
            })
            harness.pushAgentFrame({
              jsonrpc: '2.0',
              id: pendingPromptId,
              result: { stopReason: 'end_turn' },
            })
            return
          }
          if (frame.id === 'perm-warn-1') {
            receivedResponses.push(frame)
          }
        })
        activeHarness = harness
        return new AcpStdioClient({ ...opts, transport: harness.transport })
      },
    })

    const runtime = new InMemoryDaemonRuntime(
      new AgentTurnRunner({ llm: makeUnusedNativeLlm(), model: '' }),
      { currentProjectDirectory: root, model: '', sessionDirectory: join(root, 'sessions') },
    )
    const server = new DaemonServer({
      socketPath: join(root, 'unused.sock'),
      projectDirectory: root,
      runtime,
      acpBridgeHost: bridgeHost,
      autoTitle: false,
    })
    const client = new InMemoryDaemonClient(server, 't3-parity-s1')
    try {
      await client.call('initialize', { session_key: 't3-parity-s1', project_dir: root })
      const connRes = await client.call('acp.connect', { session_key: 't3-parity-s1', agent_id: 'antigravity' })
      expect(connRes.ok).toBe(true)
      // Verify T3 Code handshake: protocolVersion 2, honest title, fs.writeTextFile=false, notices+compaction
      expect(activeHarness!.receivedFrames[0]).toMatchObject({
        method: 'initialize',
        params: {
          protocolVersion: 2,
          clientInfo: { name: 'xerxes-agents', title: 'Xerxes Agents' },
          clientCapabilities: {
            fs: { readTextFile: true, writeTextFile: false },
            session: { configOptions: { boolean: {} }, notices: {}, compaction: {} },
          },
        },
      })
      // Verify symlink escape and binary files are blocked by fs/read_text_file, while valid 1-indexed line slice works
      activeHarness!.pushAgentFrame({
        jsonrpc: '2.0',
        id: 'fs-symlink',
        method: 'fs/read_text_file',
        params: { sessionId: 'agy-t3-ext-1', path: join(root, 'escape-link.txt') },
      })
      activeHarness!.pushAgentFrame({
        jsonrpc: '2.0',
        id: 'fs-bin',
        method: 'fs/read_text_file',
        params: { sessionId: 'agy-t3-ext-1', path: join(root, 'binary.bin') },
      })
      activeHarness!.pushAgentFrame({
        jsonrpc: '2.0',
        id: 'fs-slice',
        method: 'fs/read_text_file',
        params: { sessionId: 'agy-t3-ext-1', path: join(root, 'ok.txt'), line: 2, limit: 1 },
      })
      await Bun.sleep(40)
      expect(activeHarness!.receivedFrames.find(f => f.id === 'fs-symlink')?.error).toBeDefined()
      expect(activeHarness!.receivedFrames.find(f => f.id === 'fs-bin')?.error).toBeDefined()
      expect(activeHarness!.receivedFrames.find(f => f.id === 'fs-slice')?.result).toEqual({ content: 'line2' })

      // Verify Turbo mode is active, yet `interaction_pick_1` emits question_request instead of auto-approving!
      expect((connRes.binding as { current_mode?: string }).current_mode).toBe('turbo')
      expect(
        isAntigravityUserInputRequest({
          sessionId: 'agy-t3-ext-1',
          toolCallId: 'interaction_pick_1',
          title: 'Pick',
          toolName: 'Pick',
          options: [{ optionId: 'opt_prod', name: 'Production' }],
        }),
      ).toBe(true)

      const turnPromise = client.call('turn.submit', { session_key: 't3-parity-s1', text: 'deploy' })
      const qEvent = await client.next(f => f.method === 'event' && f.params?.type === 'question_request')
      const qPayload = qEvent.params!.payload as { id: string; questions: Array<{ id: string; options: string[] }> }
      expect(qPayload.questions[0]?.options).toEqual(['Staging', 'Production'])
      await client.call('question_response', {
        request_id: qPayload.id,
        answers: { interaction_pick_1: 'Production' },
      })
      await turnPromise
      await client.next(f => f.method === 'event' && f.params?.type === 'turn_end')

      expect(receivedResponses.find(f => f.id === 'perm-interaction-1')?.result).toEqual({
        outcome: { outcome: 'selected', optionId: 'opt_prod' },
      })
      expect(
        client.frames.some(
          f =>
            f.method === 'event' &&
            f.params?.type === 'notification' &&
            String((f.params.payload as { message?: string }).message ?? '').includes('Connected: Antigravity ready'),
        ),
      ).toBe(true)
      expect(
        client.frames.some(
          f =>
            f.method === 'event' &&
            f.params?.type === 'tool_result' &&
            (f.params.payload as { name?: string; return_value?: string }).name === 'TodoWriteTool' &&
            String((f.params.payload as { return_value?: string }).return_value ?? '').includes('Deploy staging'),
        ),
      ).toBe(true)

      // Switch to default mode and verify `agy.security.warning` is forwarded on `approval_request.options`
      await client.call('acp.set_mode', { session_key: 't3-parity-s1', mode_id: 'default' })
      activeHarness!.pushAgentFrame({
        jsonrpc: '2.0',
        id: 'perm-warn-1',
        method: 'session/request_permission',
        params: {
          sessionId: 'agy-t3-ext-1',
          toolCall: { toolCallId: 'tc-sh-1', title: 'Run shell command', kind: 'execute' },
          options: [
            { optionId: 'allow_once', name: 'Allow once', kind: 'allow_once' },
            {
              optionId: 'allow_always',
              name: 'Allow always',
              kind: 'allow_always',
              _meta: {
                'agy.security.warning': {
                  message: 'Allows arbitrary shell execution in this workspace.',
                },
              },
            },
          ],
        },
      })
      const approvalEvent = await client.next(f => f.method === 'event' && f.params?.type === 'approval_request')
      const approvalPayload = approvalEvent.params!.payload as {
        id: string
        options?: Array<{ option_id: string; warning?: string }>
      }
      expect(approvalPayload.options?.find(o => o.option_id === 'allow_always')?.warning).toBe(
        'Allows arbitrary shell execution in this workspace.',
      )
      await client.call('permission_response', {
        request_id: approvalPayload.id,
        response: 'approve',
        option_id: 'allow_once',
      })
      await Bun.sleep(20)
      expect(receivedResponses.find(f => f.id === 'perm-warn-1')?.result).toEqual({
        outcome: { outcome: 'selected', optionId: 'allow_once' },
      })
    } finally {
      client.close()
      await server.stop()
    }
  })

  test('Strict ACP isolation: agy acp never discovers .xerxes/skills or bundled Xerxes skills, and never leaks a phantom native opus coordinator in swarm.status', async () => {
    const root = await makeTempDir('xrx-acp-isolation-')
    // Create BOTH a project .xerxes skill (must be ignored!) AND an Antigravity .agents skill (must be discovered!)
    await mkdir(join(root, '.xerxes', 'skills', 'xerxes-only-skill'), { recursive: true })
    await writeFile(
      join(root, '.xerxes', 'skills', 'xerxes-only-skill', 'SKILL.md'),
      '---\nname: xerxes-only-skill\ndescription: Must never appear on Antigravity ACP\n---\nSecret Xerxes instructions.\n',
    )
    await mkdir(join(root, '.agents', 'skills', 'agy-workspace-skill'), { recursive: true })
    await writeFile(
      join(root, '.agents', 'skills', 'agy-workspace-skill', 'SKILL.md'),
      '---\nname: agy-workspace-skill\ndescription: Antigravity workspace skill\n---\nAGY workspace instructions.\n',
    )

    const bridgeHost = new DaemonAcpBridgeHost({
      discoverAgents: () => [AGY_AGENT],
      createAcpClient: opts => {
        const harness = createInMemoryAcpTransport((frame, h) => {
          if (frame.method === 'initialize') {
            h.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                protocolVersion: 1,
                agentInfo: { name: 'Antigravity ACP Server', version: '1.5.15' },
              },
            })
          } else if (frame.method === 'session/new') {
            h.pushAgentFrame({
              jsonrpc: '2.0',
              id: frame.id,
              result: {
                sessionId: 'agy-iso-ext-1',
                configOptions: [
                  {
                    id: 'model',
                    name: 'Model',
                    category: 'model',
                    type: 'select',
                    currentValue: 'gemini-4-neon',
                    options: [{ value: 'gemini-4-neon', name: '4 Neon' }],
                  },
                ],
              },
            })
            queueMicrotask(() => {
              h.pushAgentFrame({
                jsonrpc: '2.0',
                method: 'session/update',
                params: {
                  sessionId: 'agy-iso-ext-1',
                  update: {
                    sessionUpdate: 'available_commands_update',
                    availableCommands: [
                      { name: 'agy-workspace-skill', description: 'Antigravity workspace skill' },
                      { name: 'swarm', description: 'Launch an Antigravity swarm' },
                    ],
                  },
                },
              })
            })
          }
        })
        return new AcpStdioClient({ ...opts, transport: harness.transport })
      },
    })

    // Seed native runtime with claude-code/opus to verify ACP never leaks it into swarm.status!
    const runtime = new InMemoryDaemonRuntime(new AgentTurnRunner({ llm: makeUnusedNativeLlm(), model: 'claude-code/opus' }), {
      currentProjectDirectory: root,
      model: 'claude-code/opus',
      sessionDirectory: join(root, 'sessions'),
    })
    const server = new DaemonServer({
      socketPath: join(root, 'iso.sock'),
      projectDirectory: root,
      runtime,
      acpBridgeHost: bridgeHost,
      agyBrainHomeDir: join(root, 'empty-agy-home'),
      autoTitle: false,
      autoSnapshotTurns: false,
    })
    const client = new InMemoryDaemonClient(server, 'iso-s1')
    try {
      const init = await client.call('initialize', { session_key: 'iso-s1', project_dir: root })
      const sessionId = String(init.session_id)
      const conn = await client.call('acp.connect', {
        session_key: 'iso-s1',
        session_id: sessionId,
        agent_id: 'antigravity',
        cwd: root,
      })
      expect(conn.ok).toBe(true)
      await Bun.sleep(20)

      // 1. capabilities.list returns ONLY Antigravity skills — zero .xerxes/skills or bundled Xerxes skills
      const caps = await client.call('capabilities.list', { session_key: 'iso-s1' })
      expect(caps.ok).toBe(true)
      const capSkillNames = (caps.skills as Array<{ name: string }>).map(s => s.name)
      expect(capSkillNames).toContain('agy-workspace-skill')
      expect(capSkillNames).not.toContain('xerxes-only-skill')
      expect(capSkillNames).not.toContain('immortals-coordinator')
      expect(capSkillNames).not.toContain('swarm-coordinator')

      // 2. capabilities.inspect refuses .xerxes skill and inspects Antigravity skill cleanly
      const blockedInspect = await client.call('capabilities.inspect', {
        session_key: 'iso-s1',
        name: 'xerxes-only-skill',
      })
      expect(blockedInspect.ok).toBe(false)
      const allowedInspect = await client.call('capabilities.inspect', {
        session_key: 'iso-s1',
        name: 'agy-workspace-skill',
      })
      expect(allowedInspect.ok).toBe(true)
      expect(String(allowedInspect.instructions)).toContain('AGY workspace instructions.')

      // 3. /skills slash command lists ONLY Antigravity skills
      const skillsSlash = await client.call('slash', { session_key: 'iso-s1', command: '/skills' })
      expect(skillsSlash.ok).toBe(true)
      const slashNames = (skillsSlash.skills as Array<{ name: string }>).map(s => s.name)
      expect(slashNames).toContain('agy-workspace-skill')
      expect(slashNames).not.toContain('xerxes-only-skill')
      expect(slashNames).not.toContain('immortals-coordinator')

      // 4. skill_suggestions returns empty array under external ACP
      const suggestions = await client.call('skill_suggestions', { session_key: 'iso-s1' })
      expect(suggestions).toEqual({ ok: true, suggestions: [] })

      // 5. swarm.status on a brand-new Antigravity thread returns empty members (NEVER leaks claude-code/opus coordinator!)
      const swarmStatus = await client.call('swarm.status', { session_key: 'iso-s1', session_id: sessionId })
      expect(swarmStatus.active).toBe(false)
      expect(swarmStatus.members).toEqual([])
      expect(JSON.stringify(swarmStatus)).not.toContain('claude-code/opus')
    } finally {
      client.close()
      await server.stop()
    }
  })
})
