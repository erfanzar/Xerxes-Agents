// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { AcpStdioClient, type AcpClientEvent, type AcpSpawnedProcess } from '../src/acp/client.js'
import { CodexExecJsonRunner, discoverExternalAcpAgents, type AcpBridgeStreamEvent } from '../src/acp/externalAgents.js'
import { DaemonAcpBridgeHost } from '../src/daemon/acpBridgeHost.js'
import {
  cleanAgyUserPromptText,
  discoverImportableExternalSessions,
  loadExternalSessionTranscript,
} from '../src/daemon/externalSessionImporter.js'
import { ConnectionLeases } from '../src/daemon/connectionLease.js'
import { LOCAL_PROVIDER_BINDING, RemoteProviderBindings } from '../src/daemon/remoteProviderBindings.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer, gitBranch } from '../src/daemon/server.js'
import type { DaemonTransportConnection } from '../src/daemon/transport.js'
import type { JsonRpcPayload } from '../src/protocol/jsonRpc.js'
import { sameRemote } from '../src/desktop/main/contextNavigation.js'
import { DaemonEventBatcher } from '../src/desktop/main/ipcEventBatcher.js'
import { DesktopProviderForwarding } from '../src/desktop/main/providerForwarding.js'
import { BlockBuilder } from '../src/desktop/renderer/blocks.js'
import { Store, applyInflightTurn, isPlanReview } from '../src/desktop/renderer/store.js'
import type { DaemonEvent } from '../src/desktop/renderer/types.js'

function mockTransport(key = 'sess-1'): { conn: DaemonTransportConnection; frames: object[] } {
  const frames: object[] = []
  return {
    frames,
    conn: {
      activeSessionKey: key,
      send(frame: object) {
        frames.push(frame)
      },
    },
  }
}

describe('Production audit regressions — leases, relay, IPC batcher & remote paths', () => {
  test('ConnectionLeases enforces PROTOCOL.md detached-only reclaim and preserves replay frames', () => {
    let expired = 0
    const leases = new ConnectionLeases(() => {
      expired += 1
    }, 5_000)
    const oldT = mockTransport('s1')
    const newT = mockTransport('s1')
    const token = leases.enable(oldT.conn)
    const originalOwner = leases.owner(oldT.conn)

    // Connected owner cannot be taken over while attached (PROTOCOL.md invariant).
    expect(() => leases.resume(newT.conn, token)).toThrow('already attached')

    // Once detached, replacement transport reclaims the lease and receives buffered replay frames.
    expect(leases.disconnect(oldT.conn)).toBe(true)
    originalOwner.send({ type: 'buffered' })
    const owner = leases.resume(newT.conn, token)
    expect(owner).toBe(originalOwner)
    expect(leases.takeReplay(owner)).toEqual([{ type: 'buffered' }])
    owner.send({ type: 'ping' })
    expect(newT.frames).toEqual([{ type: 'ping' }])
    expect(oldT.frames).toEqual([])
    expect(expired).toBe(0)
    leases.close()
  })

  test('RemoteProviderBindings preserves restorable:true on send failure and timeout', async () => {
    const bindings = new RemoteProviderBindings(20)
    const owner = {}
    const session = { id: 's-restorable', cwd: '/repo', metadata: {} as Record<string, unknown> }
    let sendShouldSucceed = false
    bindings.bind(
      owner,
      session,
      { source: 'mac', profile: 'codex', model: 'gpt-5' },
      () => sendShouldSucceed,
      { restorable: true },
    )
    const client = bindings.client(session, 'gpt-5')!
    await expect(
      (async () => {
        for await (const _chunk of client.stream({ model: 'gpt-5', messages: [{ role: 'user', content: 'hi' }] })) {
          // noop
        }
      })(),
    ).rejects.toMatchObject({ code: 'grant_unavailable', restorable: true })

    // Timeout case also preserves restorable: true
    sendShouldSucceed = true
    await expect(
      (async () => {
        for await (const _chunk of client.stream({ model: 'gpt-5', messages: [{ role: 'user', content: 'hi' }] })) {
          // noop
        }
      })(),
    ).rejects.toMatchObject({ code: 'grant_unavailable', restorable: true })
    bindings.close()
  })

  test('DaemonEventBatcher coalesces think_part on payload.think and preserves session_id', () => {
    const singles: Array<{ type: string; payload: Record<string, unknown> }> = []
    const batches: Array<readonly { type: string; payload: Record<string, unknown> }[]> = []
    const batcher = new DaemonEventBatcher({
      flushMs: 100,
      sendSingle: (type, payload) => singles.push({ type, payload }),
      sendBatch: frames => batches.push(frames),
    })
    batcher.push('think_part', { session_id: 's1', think: 'step ' })
    batcher.push('think_part', { session_id: 's1', think: 'one' })
    batcher.push('text_part', { session_id: 's1', text: 'hel' })
    batcher.push('text_part', { session_id: 's1', text: 'lo' })
    batcher.flush()
    expect(singles).toEqual([])
    expect(batches).toEqual([
      [
        { type: 'think_part', payload: { session_id: 's1', think: 'step one' } },
        { type: 'text_part', payload: { session_id: 's1', text: 'hello' } },
      ],
    ])
    batcher.dispose()
  })

  test('sameRemote normalizes trailing slashes on workspacePath', () => {
    expect(
      sameRemote(
        { alias: 'gpu', target: 'u@gpu', workspacePath: '/home/u/repo/' },
        { alias: 'gpu', target: 'u@gpu', workspacePath: '/home/u/repo' },
      ),
    ).toBe(true)
  })

  test('DesktopProviderForwarding blocks useLocal/current on external ACP and stops restore timer on disconnect', async () => {
    let remoteStatusCalls = 0
    const forwarding = new DesktopProviderForwarding(
      async () => ({ ok: true, profiles: [] }),
      async method => {
        if (method === 'session.status') {
          remoteStatusCalls += 1
          return {
            ok: true,
            provider_binding_session_guard_supported: true,
            session: {
              key: 'k-agy',
              cwd: '/srv/repo',
              status: 'idle',
              turn_count: 1,
              acp_binding: { agent_id: 'antigravity', transport: 'acp', connected: true },
            },
          }
        }
        return { ok: true }
      },
      'u@srv',
      '/srv/repo',
      async () => 'claude-code',
    )
    await expect(forwarding.useLocal('claude-code')).rejects.toThrow(/External ACP/i)
    expect(await forwarding.current()).toEqual({
      ok: true,
      bound: false,
      sessionKey: 'k-agy',
      destination: 'u@srv',
    })
    expect(remoteStatusCalls).toBe(2)

    // Disconnect during an in-flight failing follow must NOT arm a restoreTimer.
    let rejectInspect!: (err: Error) => void
    let followAttempts = 0
    const failing = new DesktopProviderForwarding(
      async () => ({ ok: true, profiles: [] }),
      async () => {
        followAttempts += 1
        return new Promise((_resolve, reject) => {
          rejectInspect = reject
        })
      },
      'u@srv',
      '/srv/repo',
      async () => 'claude-code',
    )
    const pendingFollow = failing.follow()
    await failing.disconnect()
    rejectInspect(new Error('socket closed'))
    expect(await pendingFollow).toMatchObject({ status: 'failed' })
    expect(followAttempts).toBe(1)
  })
})

describe('Production audit regressions — ACP bridge, wakeups, cancellations & custom env', () => {
  test('discoverExternalAcpAgents preserves custom env entries from acp-agents.json', () => {
    const agents = discoverExternalAcpAgents({
      configPath: '/tmp/fake-acp-agents.json',
      readConfigText: () =>
        JSON.stringify([
          {
            id: 'my-custom',
            name: 'My Custom',
            command: 'custom-acp-bin',
            args: ['--stdio'],
            env: { MY_CUSTOM_FLAG: '1' },
          },
        ]),
      resolveBinary: (name: string) => (name === 'custom-acp-bin' ? '/usr/local/bin/custom-acp-bin' : undefined),
    })
    const custom = agents.find(item => item.id === 'my-custom')
    expect(custom).toBeDefined()
    expect(custom?.env).toEqual({ MY_CUSTOM_FLAG: '1' })
  })

  test('DaemonAcpBridgeHost rejects forkSession mid-turn and clears Host Turbo on agent /plan mode_update', async () => {
    let emitClientEvent!: (event: AcpClientEvent) => void
    let resolvePrompt!: (value: { stopReason: string }) => void
    const fakeClient = {
      pid: 4242,
      isDisposed: false,
      hasExited: false,
      recentStderr: '',
      onEvent: (listener: (event: AcpClientEvent) => void) => {
        emitClientEvent = listener
        return () => {}
      },
      subscribeSession: () => () => {},
      initialize: async () => ({
        protocolVersion: 2,
        agentInfo: { name: 'antigravity', version: '1.6.0' },
        promptCapabilities: { image: true, audio: false, embeddedContext: true },
        sessionCapabilities: { load: true, resume: true, fork: true, list: true, close: true, additionalDirectories: true },
        mcpCapabilities: { http: false, sse: false },
        authMethods: [],
      }),
      newSession: async () => ({
        sessionId: 'ext-1',
        modes: [
          { id: 'default', name: 'Default' },
          { id: 'auto', name: 'Auto' },
          { id: 'plan', name: 'Plan' },
        ],
        currentModeId: 'default',
        raw: {},
      }),
      setMode: async () => ({}),
      prompt: () =>
        new Promise<{ stopReason: string }>(resolve => {
          resolvePrompt = resolve
        }),
      forkSession: async () => ({ sessionId: 'ext-fork', raw: {} }),
      cancel: async () => {},
      closeSession: async () => {},
      dispose: async () => {
        fakeClient.isDisposed = true
      },
    }

    const host = new DaemonAcpBridgeHost({
      createAcpClient: () => fakeClient as unknown as AcpStdioClient,
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
          description: 'Antigravity',
        },
      ],
    })

    const connected = await host.connect({
      sessionId: 's-turbo',
      agentId: 'antigravity',
      cwd: '/tmp',
      initialModeId: 'turbo',
    })
    expect(connected.ok).toBe(true)
    expect(host.isTurboMode('s-turbo')).toBe(true)

    // Agent-initiated /plan switch must clear Host Turbo instead of re-projecting Turbo over Plan.
    emitClientEvent({ type: 'mode_update', sessionId: 'ext-1', modeId: 'plan', modeName: 'Plan' })
    expect(host.isTurboMode('s-turbo')).toBe(false)
    expect(host.getBinding('s-turbo')?.current_mode).toBe('plan')

    // Forking while a turn is active must fail cleanly instead of mutating externalSessionId mid-turn.
    const promptPromise = host.runPrompt({
      sessionId: 's-turbo',
      text: 'hello',
      onEvent: () => {},
    })
    const forkDuringTurn = await host.forkSession('s-turbo')
    expect(forkDuringTurn.ok).toBe(false)
    if (!forkDuringTurn.ok) {
      expect(forkDuringTurn.error).toContain('turn is active')
    }
    resolvePrompt({ stopReason: 'end_turn' })
    await promptPromise
    await host.disposeAll()
  })

  test('AcpStdioClient preserves { outcome: "cancelled" } permission responses instead of rewriting to reject_once', async () => {
    const encoder = new TextEncoder()
    const receivedFrames: Array<Record<string, unknown>> = []
    let stdoutController!: ReadableStreamDefaultController<Uint8Array>
    let resolveExit!: (code: number | null) => void
    const stdout = new ReadableStream<Uint8Array>({ start(c) { stdoutController = c } })
    const stderr = new ReadableStream<Uint8Array>({ start() {} })
    const exited = new Promise<number | null>(resolve => { resolveExit = resolve })
    const pushAgentFrame = (frame: unknown) => {
      stdoutController.enqueue(encoder.encode(`${JSON.stringify(frame)}\n`))
    }
    const transport: AcpSpawnedProcess = {
      stdout,
      stderr,
      exited,
      writeLine: line => {
        const parsed = JSON.parse(line.trim()) as Record<string, unknown>
        receivedFrames.push(parsed)
        if (parsed.method === 'initialize') {
          pushAgentFrame({ jsonrpc: '2.0', id: parsed.id, result: { protocolVersion: 1, agentInfo: { name: 'antigravity', version: '1.5.16' } } })
        }
      },
      kill: () => {
        try { stdoutController.close() } catch { /* ignore */ }
        resolveExit(0)
      },
    }
    const client = new AcpStdioClient({
      command: 'agy',
      args: ['acp'],
      transport,
      handlers: {
        requestPermission: async () => ({ outcome: 'cancelled' }),
      },
    })
    await client.initialize()
    pushAgentFrame({
      jsonrpc: '2.0',
      id: 77,
      method: 'session/request_permission',
      params: {
        sessionId: 'ext-cancel',
        toolCall: { toolCallId: 'interaction_q1', title: 'Pick option' },
        options: [
          { optionId: 'allow', name: 'Allow', kind: 'allow_once' },
          { optionId: 'deny', name: 'Deny', kind: 'reject_once' },
        ],
      },
    })
    await Bun.sleep(40)
    const reply = receivedFrames.find(frame => frame.id === 77)
    expect(reply).toBeDefined()
    expect((reply?.result as { outcome?: { outcome?: string } } | undefined)?.outcome).toEqual({ outcome: 'cancelled' })
    await client.dispose()
  })

  test('out-of-turn wakeup never duplicates continuation text, drains queued steers, and aborts interaction_* on cancel', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'xrx-audit-wakeup-'))
    const socketPath = join(dir, 'daemon.sock')
    let emitClientEvent!: (event: AcpClientEvent) => void
    let triggerPermission!: (req: {
      sessionId: string
      toolCallId: string
      toolName: string
      title: string
      options: Array<{ optionId: string; name: string; kind?: string }>
    }) => Promise<{ outcome: string; optionId?: string }>
    const promptsReceived: string[] = []
    let promptCount = 0

    const host = new DaemonAcpBridgeHost({
      createAcpClient: options => {
        triggerPermission = async req => await options.handlers!.requestPermission!(req as never)
        return {
          pid: 9999,
          isDisposed: false,
          hasExited: false,
          recentStderr: '',
          onEvent: (listener: (event: AcpClientEvent) => void) => {
            emitClientEvent = listener
            return () => {}
          },
          subscribeSession: () => () => {},
          initialize: async () => ({
            protocolVersion: 2,
            agentInfo: { name: 'antigravity', version: '1.6.0' },
            promptCapabilities: { image: true, audio: false, embeddedContext: true },
            sessionCapabilities: { load: false, resume: false, fork: false, list: false, close: true, additionalDirectories: true },
            mcpCapabilities: { http: false, sse: false },
            authMethods: [],
          }),
          newSession: async () => ({ sessionId: 'ext-wake', raw: {} }),
          prompt: async (req: { text: string; signal?: AbortSignal }) => {
            promptCount += 1
            promptsReceived.push(req.text)
            if (promptCount === 2) {
              // Trigger an Antigravity interaction_* question, then wait for cancel abort.
              const permOutcome = await triggerPermission({
                sessionId: 'ext-wake',
                toolCallId: 'interaction_choice_1',
                toolName: 'ask_user',
                title: 'Pick one',
                options: [
                  { optionId: 'a', name: 'Alpha', kind: 'allow_once' },
                  { optionId: 'b', name: 'Beta', kind: 'allow_once' },
                ],
              })
              return { stopReason: permOutcome.outcome === 'cancelled' ? 'cancelled' : 'end_turn' }
            }
            return { stopReason: 'end_turn' }
          },
          cancel: async () => {},
          closeSession: async () => {},
          dispose: async () => {},
        } as unknown as AcpStdioClient
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
          description: 'Antigravity',
        },
      ],
    })

    const runtime = new InMemoryDaemonRuntime()
    const server = new DaemonServer({
      runtime,
      socketPath,
      projectDirectory: dir,
      acpBridgeHost: host,
      acpWakeupIdleMs: 40,
      remoteProviderBindings: new RemoteProviderBindings(),
    })

    const events: Array<{ type: string; payload: JsonRpcPayload }> = []
    let socket: Socket | undefined
    const pending = new Map<number, (frame: Record<string, unknown>) => void>()
    let nextId = 0
    const callRpc = async (method: string, params: JsonRpcPayload = {}): Promise<JsonRpcPayload> => {
      const id = ++nextId
      return await new Promise<JsonRpcPayload>((resolve, reject) => {
        const timer = setTimeout(() => {
          pending.delete(id)
          reject(new Error(`RPC ${method} timed out`))
        }, 4_000)
        pending.set(id, frame => {
          clearTimeout(timer)
          if (frame.error) reject(new Error(JSON.stringify(frame.error)))
          else resolve((frame.result as JsonRpcPayload) ?? {})
        })
        socket!.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`)
      })
    }

    try {
      await server.start()
      socket = await new Promise<Socket>((resolve, reject) => {
        const conn = connect(socketPath)
        conn.once('connect', () => resolve(conn))
        conn.once('error', reject)
      })
      socket.setEncoding('utf8')
      let buf = ''
      socket.on('data', (chunk: string) => {
        buf += chunk
        for (let nl = buf.indexOf('\n'); nl >= 0; nl = buf.indexOf('\n')) {
          const line = buf.slice(0, nl).trim()
          buf = buf.slice(nl + 1)
          if (!line) continue
          const frame = JSON.parse(line) as Record<string, unknown>
          if (typeof frame.id === 'number') {
            const waiter = pending.get(frame.id)
            if (waiter) {
              pending.delete(frame.id)
              waiter(frame)
            }
          } else if (frame.method === 'event' && frame.params && typeof frame.params === 'object') {
            const ev = frame.params as { type?: string; payload?: JsonRpcPayload }
            if (ev.type) events.push({ type: ev.type, payload: ev.payload ?? {} })
          }
        }
      })

      const init = await callRpc('initialize', {
        session_key: 'wake-key',
        project_dir: dir,
        client_capabilities: { session_owned_turns: true },
      })
      const sessionId = String(init.session_id)
      const liveSession = runtime.listSessions().find(s => s.id === sessionId)!
      liveSession.metadata[LOCAL_PROVIDER_BINDING] = { version: 1, binding: 'a'.repeat(32), workspace: dir, source: 'mac', profile: 'codex', model: 'gpt-5' }
      liveSession.metadata.local_provider_profile = 'codex'

      const connected = await callRpc('acp.connect', { session_id: sessionId, agent_id: 'antigravity', cwd: dir })
      expect(connected.ok).toBe(true)
      // Connecting external ACP must clear any stale LOCAL_PROVIDER_BINDING marker.
      expect(Object.hasOwn(liveSession.metadata, LOCAL_PROVIDER_BINDING)).toBe(false)

      // 1. Out-of-turn wakeup emits BOTH text_delta and continuation — must NOT duplicate text!
      emitClientEvent({ type: 'text_delta', sessionId: 'ext-wake', text: 'Worker done.' })
      emitClientEvent({ type: 'continuation', sessionId: 'ext-wake', text: 'Worker done.', rawUpdate: {} })

      // 2. Steer submitted during an out-of-turn wakeup must execute via continuation when wakeup settles!
      const steerRes = await callRpc('turn.steer', { session_key: 'wake-key', content: 'now run tests' })
      expect(steerRes.ok).toBe(true)

      await Bun.sleep(220)
      const wakeTexts = events
        .filter(ev => ev.type === 'text_part')
        .map(ev => String(ev.payload.text ?? ''))
      expect(wakeTexts).toEqual(['Worker done.'])
      expect(promptsReceived).toEqual(['now run tests'])

      // 3. Antigravity interaction_* question must abort cleanly on turn.cancel with zero leaked pendingAcpQuestions.
      await callRpc('turn.submit', { session_key: 'wake-key', text: 'ask me something' })
      await Bun.sleep(60)
      expect(events.some(ev => ev.type === 'question_request' && ev.payload.tool_call_id === 'interaction_choice_1')).toBe(true)

      await callRpc('turn.cancel', { session_key: 'wake-key' })
      await Bun.sleep(80)
      const statusAfterCancel = await callRpc('session.status', { session_key: 'wake-key', history_limit: 0 })
      const sessionStatus = statusAfterCancel.session as { status: string; active_turn_id?: string }
      expect(sessionStatus.status).toBe('idle')
      expect(sessionStatus.active_turn_id || '').toBe('')
    } finally {
      socket?.destroy()
      await server.stop()
      await rm(dir, { recursive: true, force: true })
    }
  })
})

describe('Production audit regressions — Desktop Store', () => {
  function makeStoreBridge() {
    const handlers = new Set<(event: DaemonEvent) => void>()
    const calls: Array<{ method: string; params: Record<string, unknown> }> = []
    let responder: (method: string, params: Record<string, unknown>) => Promise<Record<string, unknown>> | Record<string, unknown> =
      method => {
        if (method === 'initialize') {
          return {
            ok: true,
            session_id: 's1',
            cwd: '/repo',
            model: 'opus',
            connection_lease_supported: true,
            session: { id: 's1', key: 'k1', cwd: '/repo', status: 'idle', messages: [], history: { actions: [], before: null, has_more: false } },
          }
        }
        return { ok: true }
      }
    const bridge = {
      calls,
      call: async (method: string, params: Record<string, unknown> = {}) => {
        calls.push({ method, params })
        return await responder(method, params)
      },
      subscribe: (handler: (event: DaemonEvent) => void) => {
        handlers.add(handler)
        return () => handlers.delete(handler)
      },
      push: (type: string, payload: Record<string, unknown> = {}) => {
        for (const handler of handlers) handler({ type, payload })
      },
      respondWith: (fn: typeof responder) => {
        responder = fn
      },
    }
    return bridge
  }

  function withStoreWindow<T>(bridge: ReturnType<typeof makeStoreBridge>, run: () => Promise<T>): Promise<T> {
    const prev = (globalThis as { window?: unknown }).window
    ;(globalThis as { window?: unknown }).window = {
      xerxes: { onEvent: (handler: (event: DaemonEvent) => void) => bridge.subscribe(handler) },
    }
    return run().finally(() => {
      ;(globalThis as { window?: unknown }).window = prev
    })
  }

  test('isPlanReview rejects multi-question payloads and non-freeform Antigravity interaction_* questions', () => {
    expect(
      isPlanReview({
        requestId: 'q1',
        toolCallId: 'tc1',
        items: [{ id: 'i1', question: 'Approve the plan?', options: ['Approve', 'Revise'], allowFreeform: true }],
      }),
    ).toBe(true)
    expect(
      isPlanReview({
        requestId: 'q2',
        toolCallId: 'interaction_1',
        items: [{ id: 'interaction_1', question: 'Approve plan?', options: ['Approve', 'Reject'], allowFreeform: false }],
      }),
    ).toBe(false)
    expect(
      isPlanReview({
        requestId: 'q3',
        toolCallId: 'tc3',
        items: [
          { id: 'i1', question: 'Approve plan?', options: ['Yes', 'No'], allowFreeform: true },
          { id: 'i2', question: 'Extra question', options: ['A', 'B'], allowFreeform: true },
        ],
      }),
    ).toBe(false)
  })

  test('applyInflightTurn deduplicates repeated hydration and assigns deterministic fallback tool IDs', () => {
    const builder = new BlockBuilder()
    const inflight = {
      user: 'run build',
      thinking: 'checking files',
      tools: [{ name: 'ReadFileTool', arguments: '{"path":"a.ts"}' }],
      assistant: 'Working on it',
      steers: ['also check tests'],
    }
    applyInflightTurn(builder, inflight)
    applyInflightTurn(builder, inflight)
    const blocks = builder.snapshot(true)
    expect(blocks.filter(b => b.kind === 'user').map(b => b.text)).toEqual(['run build', 'also check tests'])
    expect(blocks.filter(b => b.kind === 'tools').flatMap(b => (b.kind === 'tools' ? b.items : []))).toHaveLength(1)
  })

  test('turn_begin dedupes pre-rAF optimistic submit, multi-steer continuation, and rolls back denied edits', async () => {
    const bridge = makeStoreBridge()
    await withStoreWindow(bridge, async () => {
      const store = new Store()
      store.start(bridge as never)
      await Bun.sleep(20)

      // 1. Optimistic submit followed immediately by turn_begin (before rAF) must produce 1 user bubble.
      await store.submit('hello world')
      bridge.push('turn_begin', { session_id: 's1', text: 'hello world' })
      const userBlocks = store.getSnapshot().blocks.filter(b => b.kind === 'user')
      expect(userBlocks.map(b => b.text)).toEqual(['hello world'])

      // 2. Denied tool_result rolls back optimistic tool_call edit fold.
      bridge.push('tool_call', {
        session_id: 's1',
        id: 'edit-denied',
        name: 'FileEditTool',
        arguments: JSON.stringify({ file_path: 'src/x.ts', old_string: 'a', new_string: 'b' }),
      })
      expect(store.getSnapshot().changes.map(c => c.path)).toEqual(['src/x.ts'])
      bridge.push('tool_result', {
        session_id: 's1',
        id: 'edit-denied',
        name: 'FileEditTool',
        permitted: false,
        error: 'Denied by user',
      })
      expect(store.getSnapshot().changes).toEqual([])

      // 3. Multi-turn edits to same file record file.turns so earlier turns stay visible.
      bridge.push('tool_call', {
        session_id: 's1',
        id: 'edit-ok-1',
        name: 'FileEditTool',
        arguments: JSON.stringify({ file_path: 'src/multi.ts', old_string: 'a', new_string: 'b' }),
      })
      bridge.push('tool_result', { session_id: 's1', id: 'edit-ok-1', name: 'FileEditTool', permitted: true })
      bridge.push('turn_end', { session_id: 's1', stop_reason: 'end_turn' })

      bridge.push('turn_begin', { session_id: 's1', text: 'second edit' })
      bridge.push('tool_call', {
        session_id: 's1',
        id: 'edit-ok-2',
        name: 'FileEditTool',
        arguments: JSON.stringify({ file_path: 'src/multi.ts', old_string: 'b', new_string: 'c' }),
      })
      bridge.push('tool_result', { session_id: 's1', id: 'edit-ok-2', name: 'FileEditTool', permitted: true })
      bridge.push('turn_end', { session_id: 's1', stop_reason: 'end_turn' })

      const multiFile = store.getSnapshot().changes.find(c => c.path === 'src/multi.ts')
      expect(multiFile?.turns).toEqual([1, 2])

      // 4. Multi-line joined steer continuation ("S1\nS2") dedupes against already-echoed steers.
      bridge.push('turn_begin', { session_id: 's1', text: 'turn 3' })
      await store.steer('S1')
      await store.steer('S2')
      bridge.push('turn_end', { session_id: 's1', stop_reason: 'end_turn' })
      bridge.push('steer_applied', { session_id: 's1', contents: ['S1', 'S2'], count: 2 })
      bridge.push('turn_begin', { session_id: 's1', text: 'S1\nS2' })
      const tailUsers = store.getSnapshot().blocks.filter(b => b.kind === 'user').map(b => b.text)
      expect(tailUsers.filter(t => t === 'S1\nS2')).toEqual([])
      expect(tailUsers.filter(t => t === 'S1')).toHaveLength(1)
      expect(tailUsers.filter(t => t === 'S2')).toHaveLength(1)

      // 5. retryFailed() works even when failed.lastUser is empty (e.g. cold reload or wakeup error).
      bridge.push('turn_end', { session_id: 's1', stop_reason: 'turn_failed', failed: true, error: 'rate limit' })
      expect(store.getSnapshot().failed).not.toBeNull()
      ;(store as unknown as { failure: { error: string; turn: number; lastUser: string } }).failure = {
        error: 'rate limit',
        turn: 3,
        lastUser: '',
      }
      ;(store as unknown as { patch: (p: Record<string, unknown>) => void }).patch({
        failed: { error: 'rate limit', turn: 3, lastUser: '' },
      })
      const submitsBefore = bridge.calls.filter(c => c.method === 'turn.submit').length
      store.retryFailed()
      await Bun.sleep(10)
      expect(bridge.calls.filter(c => c.method === 'turn.submit').length).toBe(submitsBefore + 1)
    })
  })

  test('steerNow skips live_steer when promotedFromQueue or pending approval/question, and cleans up failed live_steer', async () => {
    const bridge = makeStoreBridge()
    bridge.respondWith(method => {
      if (method === 'initialize') {
        return {
          ok: true,
          session_id: 's1',
          cwd: '/repo',
          model: 'opus',
          connection_lease_supported: true,
          session: {
            id: 's1',
            key: 'k1',
            cwd: '/repo',
            status: 'idle',
            messages: [],
            history: { actions: [], before: null, has_more: false },
            acp_binding: { agent_id: 'antigravity', provider: 'antigravity', transport: 'acp', connected: true },
          },
        }
      }
      if (method === 'turn.cancel') {
        queueMicrotask(() => bridge.push('turn_end', { session_id: 's1', stop_reason: 'cancelled' }))
        return { ok: true }
      }
      return { ok: true }
    })
    await withStoreWindow(bridge, async () => {
      const store = new Store()
      store.start(bridge as never)
      await Bun.sleep(20)
      bridge.push('turn_begin', { session_id: 's1', text: 'initial' })
      bridge.push('approval_request', {
        session_id: 's1',
        id: 'apr-1',
        tool_name: 'RunCommand',
        summary: 'Run ls',
      })
      expect(store.getSnapshot().approval).not.toBeNull()

      // Pending interaction active -> steerNow must cancel + submit, NOT call turn.steer({ live: true }).
      await store.steerNow('redirect now')
      const liveSteerCalls = bridge.calls.filter(c => c.method === 'turn.steer' && c.params.live === true)
      expect(liveSteerCalls).toHaveLength(0)
      expect(bridge.calls.some(c => c.method === 'turn.cancel')).toBe(true)
      expect(bridge.calls.some(c => c.method === 'turn.submit' && c.params.text === 'redirect now')).toBe(true)

      // Promoted from queue -> also skips live steer so drop_pending_steers atomically drops the queued item.
      bridge.push('turn_begin', { session_id: 's1', text: 'redirect now' })
      await store.steerNow('promoted msg', [], { promotedFromQueue: true })
      expect(bridge.calls.filter(c => c.method === 'turn.steer' && c.params.live === true)).toHaveLength(0)
      expect(bridge.calls.some(c => c.method === 'turn.cancel' && c.params.drop_pending_steers === true)).toBe(true)
    })
  })

  test('reconnect_events replay before finalize and suppress already-resolved pending_interactions', async () => {
    const bridge = makeStoreBridge()
    await withStoreWindow(bridge, async () => {
      const store = new Store()
      store.start(bridge as never)
      await Bun.sleep(20)
      bridge.push('turn_begin', { session_id: 's1', text: 'work' })
      bridge.push('text_part', { session_id: 's1', text: 'Part 1 ' })
      bridge.push('desktop_connection', { online: false })

      bridge.respondWith(method => {
        if (method === 'initialize') {
          return {
            ok: true,
            session_id: 's1',
            cwd: '/repo',
            model: 'opus',
            connection_lease_supported: true,
            reconnect_events: [
              { method: 'event', params: { type: 'text_part', payload: { session_id: 's1', text: 'Part 2.' } } },
              { method: 'event', params: { type: 'question_response', payload: { session_id: 's1', request_id: 'q-resolved' } } },
              { method: 'event', params: { type: 'turn_end', payload: { session_id: 's1', stop_reason: 'end_turn' } } },
            ],
            pending_interactions: [
              {
                type: 'question_request',
                payload: { id: 'q-resolved', session_id: 's1', questions: [{ id: 'a', question: 'Old?', options: ['Yes'] }] },
              },
            ],
            session: { id: 's1', key: 'k1', cwd: '/repo', status: 'idle', messages: [], history: { actions: [], before: null, has_more: false } },
          }
        }
        return { ok: true }
      })

      await store.retryConnection()
      const snap = store.getSnapshot()
      const agentBlocks = snap.blocks.filter(b => b.kind === 'agent')
      expect(agentBlocks).toHaveLength(1)
      expect(agentBlocks[0]?.kind === 'agent' ? agentBlocks[0].text : '').toBe('Part 1 Part 2.')
      expect(snap.question).toBeNull()
      expect(snap.turnActive).toBe(false)
    })
  })

  test('ageOf handles both epoch seconds and epoch ms; applyInflightTurn preserves chronological steer order and live tools; steerNow interrupts blocking Antigravity tools', async () => {
    const { ageOf, applyInflightTurn } = await import('../src/desktop/renderer/store.js')
    const { BlockBuilder } = await import('../src/desktop/renderer/blocks.js')

    const tenMinutesAgoMs = Date.now() - 10 * 60_000
    expect(ageOf(tenMinutesAgoMs)).toBe('10m')
    expect(ageOf(Math.floor(tenMinutesAgoMs / 1000))).toBe('10m')

    // Chronological inflight messages keep an answered steer BEFORE the assistant's response and keep running tools `working`.
    const builder = new BlockBuilder()
    applyInflightTurn(builder, {
      user: 'start distillation',
      started_at: Math.floor(tenMinutesAgoMs / 1000),
      messages: [
        { role: 'user', content: 'start distillation' },
        { role: 'assistant', content: 'Probing B=32.' },
        { role: 'user', content: 'also turn on wandb' },
        {
          role: 'assistant',
          content: 'W&B authenticated cleanly.',
          tool_calls: [{ id: 'tc-live', type: 'function', function: { name: 'exec_command', arguments: '{"cmd":"gcloud ssh"}' } }],
        },
      ],
      executions: [],
    })
    const hydrated = builder.snapshot(true)
    expect(hydrated.map(b => b.kind)).toEqual(['user', 'agent', 'user', 'tools', 'agent'])
    expect(hydrated[2]).toMatchObject({ kind: 'user', text: 'also turn on wandb' })
    expect(hydrated[3]).toMatchObject({ kind: 'tools', running: true, items: [{ id: 'tc-live', state: 'working' }] })
    expect(hydrated[4]).toMatchObject({ kind: 'agent', text: 'W&B authenticated cleanly.' })

    // On Antigravity, steerNow uses live_steer when thinking/streaming, but interrupts via turn.cancel when a tool is actively blocking.
    const bridge = makeStoreBridge()
    bridge.respondWith(method => {
      if (method === 'initialize') {
        return {
          ok: true,
          session_id: 's-agy',
          cwd: '/repo',
          model: 'gemini-4-neon',
          connection_lease_supported: true,
          session: {
            id: 's-agy',
            key: 'k-agy',
            cwd: '/repo',
            status: 'idle',
            messages: [],
            history: { actions: [], before: null, has_more: false },
            acp_binding: {
              session_id: 's-agy',
              agent_id: 'antigravity',
              name: 'Antigravity',
              provider: 'antigravity',
              transport: 'acp',
              cwd: '/repo',
              connected: true,
            },
          },
        }
      }
      if (method === 'turn.steer') return { ok: true, disposition: 'live_steer' }
      if (method === 'turn.cancel') {
        setTimeout(() => bridge.push('turn_end', { session_id: 's-agy', cancelled: true }), 5)
        return { ok: true }
      }
      return { ok: true }
    })
    await withStoreWindow(bridge, async () => {
      const store = new Store()
      store.start(bridge as never)
      await Bun.sleep(20)
      bridge.push('turn_begin', { session_id: 's-agy', text: 'run training' })
      bridge.push('tool_call', { session_id: 's-agy', id: 'tc-block', name: 'run_command', arguments: '{"cmd":"gcloud ssh"}' })
      bridge.calls.length = 0

      // Blocking tool is actively running -> steerNow must cancel + submit instead of passive live_steer.
      const interrupted = await store.steerNow('is it reporting to wandb ?')
      expect(interrupted).toBe(true)
      expect(bridge.calls.some(c => c.method === 'turn.cancel' && c.params.drop_pending_steers === true)).toBe(true)
      expect(bridge.calls.some(c => c.method === 'turn.submit' && c.params.text === 'is it reporting to wandb ?')).toBe(true)
      expect(bridge.calls.some(c => c.method === 'turn.steer' && c.params.live === true)).toBe(false)
    })
  })
})

describe('SSH & remote performance regressions — wire history budget & direct git HEAD', () => {
  test('DaemonServer caps oversized tool outputs and >4 MiB history pages with a valid before cursor, and reads .git/HEAD directly', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'xrx-ssh-perf-'))
    const repoDir = join(dir, 'repo')
    const worktreeDir = join(dir, 'wt')
    const gitCommonDir = join(repoDir, '.git')
    const wtGitDir = join(gitCommonDir, 'worktrees', 'wt')
    await mkdir(wtGitDir, { recursive: true })
    await mkdir(worktreeDir, { recursive: true })
    await writeFile(join(gitCommonDir, 'HEAD'), 'ref: refs/heads/feature/fast-ssh\n')
    await writeFile(join(wtGitDir, 'HEAD'), 'ref: refs/heads/worktree-branch\n')
    await writeFile(join(worktreeDir, '.git'), `gitdir: ${wtGitDir}\n`)

    const socketPath = join(dir, 'daemon.sock')
    const runtime = new InMemoryDaemonRuntime()
    const server = new DaemonServer({
      runtime,
      socketPath,
      projectDirectory: repoDir,
    })

    let socket: Socket | undefined
    const pending = new Map<number, (frame: Record<string, unknown>) => void>()
    let nextId = 0
    const callRpc = async (method: string, params: JsonRpcPayload = {}): Promise<JsonRpcPayload> => {
      const id = ++nextId
      return await new Promise<JsonRpcPayload>((resolve, reject) => {
        const timer = setTimeout(() => {
          pending.delete(id)
          reject(new Error(`RPC ${method} timed out`))
        }, 5_000)
        pending.set(id, frame => {
          clearTimeout(timer)
          if (frame.error) reject(new Error(JSON.stringify(frame.error)))
          else resolve((frame.result as JsonRpcPayload) ?? {})
        })
        socket!.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`)
      })
    }

    try {
      await server.start()
      socket = await new Promise<Socket>((resolve, reject) => {
        const conn = connect(socketPath)
        conn.once('connect', () => resolve(conn))
        conn.once('error', reject)
      })
      socket.setEncoding('utf8')
      let buf = ''
      socket.on('data', (chunk: string) => {
        buf += chunk
        for (let nl = buf.indexOf('\n'); nl >= 0; nl = buf.indexOf('\n')) {
          const line = buf.slice(0, nl).trim()
          buf = buf.slice(nl + 1)
          if (!line) continue
          const frame = JSON.parse(line) as Record<string, unknown>
          if (typeof frame.id === 'number') {
            const waiter = pending.get(frame.id)
            if (waiter) {
              pending.delete(frame.id)
              waiter(frame)
            }
          }
        }
      })

      // 1. Direct .git/HEAD fast path works for both standard repos and linked worktrees, and reflects branch changes immediately.
      expect(await gitBranch(repoDir)).toBe('feature/fast-ssh')
      expect(await gitBranch(worktreeDir)).toBe('worktree-branch')
      const repoBranch = await callRpc('git.branch', {})
      expect(repoBranch.branch).toBe('feature/fast-ssh')
      await writeFile(join(gitCommonDir, 'HEAD'), 'ref: refs/heads/main\n')
      const updatedBranch = await callRpc('git.branch', {})
      expect(updatedBranch.branch).toBe('main')

      // 2. Wire history page truncates huge tool results and enforces <= 4 MiB wire page budget with valid re-anchored `before` cursor.
      const init = await callRpc('initialize', { session_key: 'heavy-session', project_dir: repoDir })
      const sessionId = String(init.session_id)
      const session = runtime.listSessions().find(s => s.id === sessionId)!
      const bigTool = 'HEAD-MARKER ' + 'x'.repeat(180_000) + ' TAIL-MARKER'
      const bigAssistant = 'A'.repeat(220_000)
      for (let idx = 0; idx < 28; idx += 1) {
        session.messages.push({ role: 'user', content: `turn ${idx}` })
        session.messages.push({ role: 'tool', content: bigTool, name: 'RunCommand', tool_call_id: `tc-${idx}` } as never)
        session.messages.push({ role: 'assistant', content: `reply ${idx}: ${bigAssistant}` })
      }

      const latestResp = await callRpc('session.history', { session_key: 'heavy-session', history_limit: 80 })
      const latestPage = latestResp.history as {
        actions: Array<{ messages: Array<{ role?: string; content?: string }> }>
        has_more: boolean
        before: string | null
      }
      const actions = latestPage.actions
      expect(latestPage.has_more).toBe(true)
      expect(typeof latestPage.before).toBe('string')
      expect(Buffer.byteLength(JSON.stringify(actions), 'utf8')).toBeLessThanOrEqual(4 * 1024 * 1024)
      const latestMessages = actions.flatMap(a => a.messages)
      const toolAction = latestMessages.find(m => m.role === 'tool')
      expect(toolAction).toBeDefined()
      expect(toolAction?.content?.length ?? 0).toBeLessThan(70_000)
      expect(toolAction?.content).toContain('HEAD-MARKER')
      expect(toolAction?.content).toContain('TAIL-MARKER')
      expect(toolAction?.content).toContain('in history preview')
      // Stored session message on disk/memory stays completely untouched.
      expect(String(session.messages[1]?.content ?? '').length).toBe(bigTool.length)

      // Re-anchored `before` cursor pages cleanly to earlier actions without error or duplication.
      const olderResp = await callRpc('session.history', { session_key: 'heavy-session', before: latestPage.before, history_limit: 40 })
      const olderPage = olderResp.history as { actions: Array<{ messages: Array<{ role?: string; content?: string }> }> }
      expect(olderPage.actions.length).toBeGreaterThan(0)
      expect(olderPage.actions.flatMap(a => a.messages).some(m => m.content === 'turn 0')).toBe(true)

      // 3. Saved transcripts preserve lean ACP runtime provider metadata for fast session.list without bulky mode/model lists.
      session.metadata.acp_binding = {
        session_id: session.id,
        agent_id: 'antigravity',
        name: 'Antigravity',
        provider: 'antigravity',
        transport: 'acp',
        cwd: repoDir,
        connected: false,
        available_models: Array.from({ length: 40 }, (_, i) => ({ id: `m-${i}`, name: `Model ${i}` })),
        available_modes: [{ id: 'code', name: 'Code' }],
      }
      await runtime.flushSessions()
      const listedResp = await callRpc('session.list', { kind: 'main', scope: 'global', limit: 20 })
      const listedRows = (listedResp.sessions as Array<Record<string, unknown>>) ?? []
      const listedHeavy = listedRows.find(row => row.id === sessionId)
      expect(listedHeavy).toBeDefined()
      expect(listedHeavy?.acp_provider).toBe('antigravity')
      expect(listedHeavy?.acp_agent_id).toBe('antigravity')
      expect(listedHeavy?.available_models).toBeUndefined()
    } finally {
      socket?.destroy()
      await server.stop()
      await rm(dir, { recursive: true, force: true })
    }
  })

  test('run_workflow settles synchronous results, labels background tasks in Activity, and completes on background_changed', async () => {
    const handlers = new Set<(event: DaemonEvent) => void>()
    let bgState: 'running' | 'completed' = 'running'
    const bridge = {
      call: async (method: string) => {
        if (method === 'initialize') {
          return {
            ok: true,
            session_id: 's-wf',
            cwd: '/repo',
            model: 'opus',
            connection_lease_supported: true,
            session: { id: 's-wf', key: 'k-wf', cwd: '/repo', status: 'idle', messages: [], history: { actions: [], before: null, has_more: false } },
          }
        }
        if (method === 'session.background') {
          return {
            ok: true,
            items: [{ id: 'agy-task:conv-1/task-3269', rawTaskId: 'conv-1/task-3269', state: bgState, outputPreview: 'Wired 61 families cleanly.' }],
          }
        }
        return { ok: true }
      },
      subscribe: (handler: (event: DaemonEvent) => void) => {
        handlers.add(handler)
        return () => handlers.delete(handler)
      },
      push: (type: string, payload: Record<string, unknown> = {}) => {
        for (const handler of handlers) handler({ type, payload })
      },
    }
    const prev = (globalThis as { window?: unknown }).window
    ;(globalThis as { window?: unknown }).window = {
      xerxes: { onEvent: (handler: (event: DaemonEvent) => void) => bridge.subscribe(handler) },
    }
    try {
      const store = new Store()
      store.start(bridge as never)
      await Bun.sleep(20)
      bridge.push('turn_begin', { session_id: 's-wf', text: 'run workflows' })

      // 1. Backgrounded run_workflow sets Activity label instead of stuck on Starting.
      bridge.push('tool_call', { session_id: 's-wf', id: 'wf-bg', name: 'run_workflow', arguments: JSON.stringify({ Description: 'Wire 61 families' }) })
      bridge.push('tool_result', {
        session_id: 's-wf',
        id: 'wf-bg',
        name: 'run_workflow',
        return_value: 'Tool is running as a background task with task id: conv-1/task-3269',
      })
      const getMembers = () => store.getSnapshot().blocks.find(b => b.kind === 'agents')?.members ?? []
      expect(getMembers().find(m => m.key === 'wf-bg:0')).toMatchObject({
        status: 'working',
        backgroundTaskId: 'conv-1/task-3269',
        activity: 'Running in Activity (task-3269)',
      })

      // 2. Synchronous run_workflow settles immediately to completed.
      bridge.push('tool_call', { session_id: 's-wf', id: 'wf-sync', name: 'run_workflow', arguments: JSON.stringify({ Description: 'Quick sync check' }) })
      bridge.push('tool_result', {
        session_id: 's-wf',
        id: 'wf-sync',
        name: 'run_workflow',
        return_value: 'All 5 checks passed',
      })
      expect(getMembers().find(m => m.key === 'wf-sync:0')).toMatchObject({
        status: 'completed',
        summary: 'All 5 checks passed',
      })

      // 3. background_changed settles the backgrounded workflow when Activity finishes.
      bgState = 'completed'
      bridge.push('background_changed', { session_id: 's-wf' })
      await Bun.sleep(25)
      expect(getMembers().find(m => m.key === 'wf-bg:0')).toMatchObject({
        status: 'completed',
        summary: 'Wired 61 families cleanly.',
      })
    } finally {
      ;(globalThis as { window?: unknown }).window = prev
    }
  })

  test('strict session runtime/model pinning blocks disconnected external ACP fallback and restart auto-resume', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'xerxes-pin-guard-'))
    const sessionsDir = join(dir, 'sessions')
    const socketPath = join(dir, 'daemon.sock')
    let nativeTurnRuns = 0
    const runtime = new InMemoryDaemonRuntime(
      {
        async *run() {
          nativeTurnRuns += 1
          yield { type: 'text_part', payload: { text: 'native reply' } }
        },
      },
      {
        model: 'pinned-native-model',
        currentProjectDirectory: dir,
        sessionDirectory: sessionsDir,
      },
    )
    runtime.reload({ provider_profile: 'native-prof', model: 'pinned-native-model' })
    const server = new DaemonServer({
      runtime,
      socketPath,
      projectDirectory: dir,
      onRestart: async () => {},
    })
    let socket: Socket | undefined
    const pending = new Map<number, (frame: Record<string, unknown>) => void>()
    const pushedEvents: Array<{ type: string; payload: Record<string, unknown> }> = []
    let nextId = 0
    const callRpc = async (method: string, params: JsonRpcPayload = {}): Promise<JsonRpcPayload> => {
      const id = ++nextId
      return await new Promise<JsonRpcPayload>((resolve, reject) => {
        const timer = setTimeout(() => {
          pending.delete(id)
          reject(new Error(`RPC ${method} timed out`))
        }, 5_000)
        pending.set(id, frame => {
          clearTimeout(timer)
          if (frame.error) reject(new Error(JSON.stringify(frame.error)))
          else resolve((frame.result as JsonRpcPayload) ?? {})
        })
        socket!.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`)
      })
    }
    try {
      await server.start()
      socket = await new Promise<Socket>((resolve, reject) => {
        const conn = connect(socketPath)
        conn.once('connect', () => resolve(conn))
        conn.once('error', reject)
      })
      socket.setEncoding('utf8')
      let buf = ''
      socket.on('data', (chunk: string) => {
        buf += chunk
        for (let nl = buf.indexOf('\n'); nl >= 0; nl = buf.indexOf('\n')) {
          const line = buf.slice(0, nl).trim()
          buf = buf.slice(nl + 1)
          if (!line) continue
          const frame = JSON.parse(line) as Record<string, unknown>
          if (typeof frame.id === 'number') {
            const waiter = pending.get(frame.id)
            if (waiter) {
              pending.delete(frame.id)
              waiter(frame)
            }
          } else if (frame.method === 'event' && frame.params && typeof frame.params === 'object') {
            const p = frame.params as { type?: string; payload?: Record<string, unknown> }
            if (typeof p.type === 'string') {
              pushedEvents.push({ type: p.type, payload: p.payload ?? {} })
            }
          }
        }
      })

      // 1. Disconnected saved Antigravity session must NEVER fall through to native Xerxes,
      //    never run native compaction, and never auto-resume after a runtime restart.
      await callRpc('session.open', { session_key: 'agy-slot', project_dir: dir })
      const agySession = runtime.sessionStatus('agy-slot')!
      agySession.messages.push(
        { role: 'user', content: 'Train looped Qwen' },
        { role: 'assistant', content: 'Running on prism-v5p-128_dev_1.' },
      )
      agySession.turnCount = 1
      agySession.metadata.acp_binding = {
        session_id: agySession.id,
        agent_id: 'antigravity',
        name: 'Antigravity',
        provider: 'antigravity',
        transport: 'acp',
        cwd: dir,
        current_model: 'MODEL_4_NEON',
        thought_level: 'xhigh',
        connected: false,
      }
      await runtime.flushSessions()

      // 1a. turn.submit on disconnected Antigravity session fails closed (native runner never called).
      pushedEvents.length = 0
      await callRpc('turn.submit', { session_key: 'agy-slot', text: 'hello after disconnect' })
      await Bun.sleep(40)
      expect(nativeTurnRuns).toBe(0)
      expect(agySession.messages).toHaveLength(2)
      expect(
        pushedEvents.some(
          ev =>
            ev.type === 'notification'
            && ev.payload.level === 'error'
            && /Antigravity is disconnected/i.test(String(ev.payload.message ?? '')),
        ),
      ).toBe(true)

      // 1b. session.compress refuses native compaction on saved disconnected Antigravity session.
      const compactRes = await callRpc('session.compress', { session_key: 'agy-slot' })
      expect(compactRes.ok).toBe(false)
      expect(String(compactRes.error ?? '')).toMatch(/managed by Antigravity/i)

      // 1c. Stale turn_resume_after_restart marker on an Antigravity session is dropped without native fallback.
      agySession.metadata.turn_resume_after_restart = { at: Date.now(), origin: 'human' }
      pushedEvents.length = 0
      await callRpc('session.open', { session_key: 'agy-slot', project_dir: dir })
      await Bun.sleep(40)
      expect(nativeTurnRuns).toBe(0)
      expect(agySession.metadata.turn_resume_after_restart).toBeUndefined()
      expect(agySession.messages).toHaveLength(2)
      expect(
        pushedEvents.some(
          ev => ev.type === 'notification' && /Reconnect Antigravity/i.test(String(ev.payload.message ?? '')),
        ),
      ).toBe(true)

      // 1d. Native session pins its model + provider_profile once a turn runs,
      //     so later global default changes cannot drift it to another model.
      const nativeSession = await runtime.openSession('native-slot', undefined, { cwd: dir })
      nativeSession.modelPinned = false
      delete nativeSession.metadata.provider_profile
      await runtime.submitTurn('native-slot', 'first native prompt', () => {})
      expect(nativeTurnRuns).toBe(1)
      expect(nativeSession.model).toBe('pinned-native-model')
      expect(nativeSession.modelPinned).toBe(true)
      expect(nativeSession.metadata.provider_profile).toBe('native-prof')
      runtime.reload({ model: 'brand-new-global-default' })
      expect(nativeSession.model).toBe('pinned-native-model')

      // 2. Forced runtime.restart_if_idle stamps pinned { model, provider_profile } onto
      //    interrupted native sessions, and NEVER stamps turn_resume_after_restart onto external ACP sessions.
      agySession.activeTurnId = 'busy-agy-turn'
      agySession.status = 'working'
      nativeSession.activeTurnId = 'busy-native-turn'
      nativeSession.status = 'working'
      const restartRes = await callRpc('runtime.restart_if_idle', { force: true, requested_by: 'test' })
      expect(restartRes.ok).toBe(true)
      expect(agySession.metadata.turn_resume_after_restart).toBeUndefined()
      expect(nativeSession.metadata.turn_resume_after_restart).toMatchObject({
        model: 'pinned-native-model',
        provider_profile: 'native-prof',
      })
      agySession.activeTurnId = ''
      agySession.status = 'idle'
      nativeSession.activeTurnId = ''
      nativeSession.status = 'idle'
    } finally {
      socket?.destroy()
      await server.stop()
      await rm(dir, { recursive: true, force: true })
    }
  })

  test('session.rewind truncates transcript, prunes tool executions, persists rewrite, and syncs Desktop Store', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'xerxes-rewind-'))
    const sessionsDir = join(dir, 'sessions')
    const socketPath = join(dir, 'daemon.sock')
    const runtime = new InMemoryDaemonRuntime(undefined, {
      model: 'test-model',
      currentProjectDirectory: dir,
      sessionDirectory: sessionsDir,
    })
    const server = new DaemonServer({
      runtime,
      socketPath,
      projectDirectory: dir,
    })
    let socket: Socket | undefined
    const pending = new Map<number, (frame: Record<string, unknown>) => void>()
    let nextId = 0
    const callRpc = async (method: string, params: JsonRpcPayload = {}): Promise<JsonRpcPayload> => {
      const id = ++nextId
      return await new Promise<JsonRpcPayload>((resolve, reject) => {
        const timer = setTimeout(() => {
          pending.delete(id)
          reject(new Error(`RPC ${method} timed out`))
        }, 5_000)
        pending.set(id, frame => {
          clearTimeout(timer)
          if (frame.error) reject(new Error(JSON.stringify(frame.error)))
          else resolve((frame.result as JsonRpcPayload) ?? {})
        })
        socket!.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`)
      })
    }
    try {
      await server.start()
      socket = await new Promise<Socket>((resolve, reject) => {
        const conn = connect(socketPath)
        conn.once('connect', () => resolve(conn))
        conn.once('error', reject)
      })
      socket.setEncoding('utf8')
      let buf = ''
      socket.on('data', (chunk: string) => {
        buf += chunk
        for (let nl = buf.indexOf('\n'); nl >= 0; nl = buf.indexOf('\n')) {
          const line = buf.slice(0, nl).trim()
          buf = buf.slice(nl + 1)
          if (!line) continue
          const frame = JSON.parse(line) as Record<string, unknown>
          if (typeof frame.id === 'number') {
            const waiter = pending.get(frame.id)
            if (waiter) {
              pending.delete(frame.id)
              waiter(frame)
            }
          }
        }
      })

      await callRpc('session.open', { session_key: 'rw-slot', project_dir: dir })
      const session = runtime.sessionStatus('rw-slot')!
      session.messages.push(
        { role: 'user', content: 'first prompt' },
        {
          role: 'assistant',
          content: 'first reply',
          tool_calls: [{ id: 'tc-1', type: 'function', function: { name: 'ReadFile', arguments: '{}' } }],
        },
        { role: 'tool', tool_call_id: 'tc-1', content: 'file content' },
        { role: 'user', content: 'second prompt' },
        {
          role: 'assistant',
          content: 'second reply',
          tool_calls: [{ id: 'tc-2', type: 'function', function: { name: 'RunCmd', arguments: '{}' } }],
        },
        { role: 'tool', tool_call_id: 'tc-2', content: 'cmd output' },
        { role: 'user', content: 'Continue after the runtime restart', origin: 'agent' },
        { role: 'assistant', content: '[Stopped: Client claude-code: Prompt is too long]' },
      )
      session.toolExecutions.push(
        { toolCallId: 'tc-1', name: 'ReadFile', inputs: {}, result: 'file content' },
        { toolCallId: 'tc-2', name: 'RunCmd', inputs: {}, result: 'cmd output' },
      )
      session.thinkingContent.push('thought 1', 'thought 2', 'thought 3')
      session.turnCount = 3
      session.metadata.turn_resume_after_restart = { at: Date.now(), origin: 'human' }
      await runtime.flushSessions('rewrite', session.sessionKey)

      // Rewind blocked while a turn is active.
      session.activeTurnId = 'active-1'
      const blocked = await callRpc('session.rewind', { session_key: 'rw-slot', message_index: 6 })
      expect(blocked.ok).toBe(false)
      expect(String(blocked.error ?? '')).toMatch(/Cannot rewind while a turn is running/i)
      session.activeTurnId = ''

      // 1. Rewind synthetic restart turn (index 6): restores empty prompt, drops last 2 messages, clears restart marker.
      const rwRestart = await callRpc('session.rewind', {
        session_key: 'rw-slot',
        message_index: 6,
        inclusive: true,
      })
      expect(rwRestart.ok).toBe(true)
      expect(rwRestart.dropped).toBe(2)
      expect(rwRestart.restored_prompt).toBe('')
      expect(rwRestart.turn_count).toBe(2)
      expect(session.metadata.turn_resume_after_restart).toBeUndefined()
      expect(session.toolExecutions).toHaveLength(2)

      // 2. Rewind to 'second prompt' (index 3): returns restored_prompt and prunes tc-2 execution + extra thinking.
      const rwSecond = await callRpc('session.rewind', {
        session_key: 'rw-slot',
        message_index: 3,
        match_text: 'second prompt',
        inclusive: true,
      })
      expect(rwSecond.ok).toBe(true)
      expect(rwSecond.dropped).toBe(3)
      expect(rwSecond.remaining_messages).toBe(3)
      expect(rwSecond.turn_count).toBe(1)
      expect(rwSecond.restored_prompt).toBe('second prompt')
      expect(session.toolExecutions.map(exec => (exec as { toolCallId?: string }).toolCallId)).toEqual(['tc-1'])
      expect(session.thinkingContent).toEqual(['thought 1'])

      // Verify persisted transcript on disk matches the rewound state.
      const reloadedRuntime = new InMemoryDaemonRuntime(undefined, {
        currentProjectDirectory: dir,
        sessionDirectory: sessionsDir,
      })
      try {
        const reopened = await reloadedRuntime.openSession(session.id, undefined, { cwd: dir, resume: true })
        expect(reopened.messages).toHaveLength(3)
        expect(reopened.turnCount).toBe(1)
        expect(reopened.modelPinned).toBe(true)
      } finally {
        await reloadedRuntime.shutdown()
      }

      // 3. Desktop Store.rewindToMessage clears failed state and adopts the returned rewound session.
      const rewoundSessionWire = rwSecond.session as Record<string, unknown>
      const handlers = new Set<(event: DaemonEvent) => void>()
      const storeBridge = {
        call: async (method: string) => {
          if (method === 'initialize') {
            return {
              ok: true,
              session_id: 's-rw',
              cwd: dir,
              model: 'test-model',
              connection_lease_supported: true,
              session: { id: 's-rw', key: 'k-rw', cwd: dir, status: 'idle', messages: [], history: { actions: [], before: null, has_more: false } },
            }
          }
          if (method === 'session.rewind') {
            return {
              ok: true,
              dropped: 3,
              restored_prompt: 'second prompt',
              session: rewoundSessionWire,
            }
          }
          return { ok: true }
        },
        subscribe: (handler: (event: DaemonEvent) => void) => {
          handlers.add(handler)
          return () => handlers.delete(handler)
        },
        push: (type: string, payload: Record<string, unknown> = {}) => {
          for (const handler of handlers) handler({ type, payload })
        },
      }
      const prevWindow = (globalThis as { window?: unknown }).window
      ;(globalThis as { window?: unknown }).window = {
        xerxes: { onEvent: (handler: (event: DaemonEvent) => void) => storeBridge.subscribe(handler) },
      }
      try {
        const desktopStore = new Store()
        desktopStore.start(storeBridge as never)
        await Bun.sleep(20)
        storeBridge.push('turn_begin', { session_id: 's-rw', text: 'second prompt' })
        storeBridge.push('turn_end', { session_id: 's-rw', stop_reason: 'turn_failed', failed: true, error: 'Prompt is too long' })
        expect(desktopStore.getSnapshot().failed).not.toBeNull()
        const res = await desktopStore.rewindToMessage({ messageIndex: 3, text: 'second prompt' })
        expect(res.ok).toBe(true)
        expect(res.restoredPrompt).toBe('second prompt')
        expect(desktopStore.getSnapshot().failed).toBeNull()
        expect(desktopStore.getSnapshot().blocks.filter(b => b.kind === 'user').map(b => b.text)).toEqual(['first prompt'])
      } finally {
        ;(globalThis as { window?: unknown }).window = prevWindow
      }
    } finally {
      socket?.destroy()
      await server.stop()
      await rm(dir, { recursive: true, force: true })
    }
  })

  test('first-class Codex ACP discovery, CodexExecJsonRunner streaming, and DaemonAcpBridgeHost controls', async () => {
    // 1. Discovery priority: XERXES_CODEX_ACP_CMD > codex-acp > codex (codex-exec-json) > installHint.
    const withOverride = discoverExternalAcpAgents({
      env: { XERXES_CODEX_ACP_CMD: '/opt/bin/codex-acp --stdio' },
      binaryExists: p => p === '/opt/bin/codex-acp',
      readConfigText: () => undefined,
    })
    expect(withOverride.find(a => a.id === 'codex')).toMatchObject({
      provider: 'codex',
      transport: 'acp',
      available: true,
      binaryPath: '/opt/bin/codex-acp',
      args: ['--stdio'],
    })

    const withCodexAcpBin = discoverExternalAcpAgents({
      env: { HOME: '/Users/dev', PATH: '/usr/local/bin' },
      homeDir: '/Users/dev',
      binaryExists: p => p === '/usr/local/bin/codex-acp',
      readConfigText: () => undefined,
    })
    expect(withCodexAcpBin.find(a => a.id === 'codex')).toMatchObject({
      provider: 'codex',
      transport: 'acp',
      available: true,
      binaryPath: '/usr/local/bin/codex-acp',
    })

    const withCodexCli = discoverExternalAcpAgents({
      env: { HOME: '/Users/dev', PATH: '/usr/local/bin' },
      homeDir: '/Users/dev',
      binaryExists: p => p === '/usr/local/bin/codex',
      readConfigText: () => undefined,
    })
    expect(withCodexCli.find(a => a.id === 'codex')).toMatchObject({
      provider: 'codex',
      transport: 'codex-exec-json',
      available: true,
      binaryPath: '/usr/local/bin/codex',
    })

    // 2. CodexExecJsonRunner translates NDJSON items + passes resume/model/effort/mode flags.
    const spawnedCalls: Array<{ bin: string; args: readonly string[] }> = []
    const makeStream = (lines: readonly string[]): ReadableStream<Uint8Array> =>
      new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(`${lines.join('\n')}\n`))
          controller.close()
        },
      })
    const runner = new CodexExecJsonRunner({
      binaryPath: '/usr/local/bin/codex',
      cwd: '/repo',
      model: 'gpt-5.4',
      thoughtLevel: 'xhigh',
      mode: 'turbo',
      spawn: (bin, args) => {
        spawnedCalls.push({ bin, args: [...args] })
        return {
          pid: 999,
          stdout: makeStream([
            JSON.stringify({ type: 'thread.started', thread_id: 'codex-thread-42' }),
            JSON.stringify({ type: 'item.updated', item: { id: 'r1', type: 'reasoning', text: 'Checking files' } }),
            JSON.stringify({ type: 'item.started', item: { id: 'cmd1', type: 'command_execution', command: 'ls -la', status: 'in_progress' } }),
            JSON.stringify({ type: 'item.completed', item: { id: 'cmd1', type: 'command_execution', command: 'ls -la', aggregated_output: 'README.md', exit_code: 0, status: 'completed' } }),
            JSON.stringify({ type: 'item.updated', item: { id: 'm1', type: 'agent_message', text: 'Hello ' } }),
            JSON.stringify({ type: 'item.completed', item: { id: 'm1', type: 'agent_message', text: 'Hello world!' } }),
            JSON.stringify({ type: 'turn.completed', usage: { input_tokens: 120, cached_input_tokens: 40, output_tokens: 18 } }),
          ]),
          stderr: makeStream([]),
          exited: Promise.resolve(0),
          kill: () => {},
        }
      },
    })
    const events: AcpBridgeStreamEvent[] = []
    const turn1 = await runner.prompt({ text: 'inspect repo', onEvent: ev => events.push(ev) })
    expect(turn1.externalSessionId).toBe('codex-thread-42')
    expect(events.filter(e => e.type === 'text_delta').map(e => (e.type === 'text_delta' ? e.text : '')).join('')).toBe('Hello world!')
    expect(events.some(e => e.type === 'thought_delta' && e.text === 'Checking files')).toBe(true)
    expect(events.some(e => e.type === 'tool_call' && e.name === 'Bash')).toBe(true)
    expect(events.some(e => e.type === 'tool_result' && e.output === 'README.md')).toBe(true)
    expect(events.some(e => e.type === 'usage' && e.inputTokens === 120 && e.outputTokens === 18)).toBe(true)
    expect(spawnedCalls[0]?.args).toContain('--model')
    expect(spawnedCalls[0]?.args).toContain('gpt-5.4')
    expect(spawnedCalls[0]?.args).toContain('model_reasoning_effort="xhigh"')
    expect(spawnedCalls[0]?.args).toContain('--dangerously-bypass-approvals-and-sandbox')

    await runner.prompt({ text: 'follow up', onEvent: () => {} })
    expect(spawnedCalls[1]?.args.slice(0, 2)).toEqual(['exec', 'resume'])
    expect(spawnedCalls[1]?.args).toContain('codex-thread-42')

    // 3. DaemonAcpBridgeHost connects codex-exec-json and mutates model, mode, and thought_level.
    const host = new DaemonAcpBridgeHost({
      discoverAgents: () => [
        { id: 'xerxes', name: 'Xerxes', provider: 'xerxes', transport: 'native', available: true, description: 'Native' },
        { id: 'codex', name: 'Codex', provider: 'codex', transport: 'codex-exec-json', available: true, binaryPath: '/usr/local/bin/codex', command: '/usr/local/bin/codex', description: 'Codex' },
      ],
      createCodexRunner: () => runner,
    })
    const connRes = await host.connect({ sessionId: 's-codex', agentId: 'codex', cwd: '/repo' })
    expect(connRes.ok).toBe(true)
    if (connRes.ok) {
      expect(connRes.binding.provider).toBe('codex')
      expect(connRes.binding.transport).toBe('codex-exec-json')
      expect(connRes.binding.thought_levels?.length).toBeGreaterThan(0)
    }
    const modelRes = await host.setModel('s-codex', 'gpt-5.3-codex')
    expect(modelRes.ok && modelRes.binding.current_model).toBe('gpt-5.3-codex')
    const effortRes = await host.setConfigOption('s-codex', 'thought_level', 'medium')
    expect(effortRes.ok && effortRes.binding.thought_level).toBe('medium')
    const modeRes = await host.setMode('s-codex', 'manual')
    expect(modeRes.ok && modeRes.binding.current_mode).toBe('manual')
    await host.disposeAll()
  })

  test('cross-runtime chat importer reads Antigravity, Claude Code, and Codex transcripts read-only and serves daemon RPCs', async () => {
    const fakeHome = await mkdtemp(join(tmpdir(), 'xerxes-import-home-'))
    const repoDir = join(fakeHome, 'work', 'easydel-src')
    const sessionsDir = join(fakeHome, 'xerxes-sessions')
    const socketPath = join(fakeHome, 'daemon.sock')
    await mkdir(repoDir, { recursive: true })

    // 1. Seed Antigravity brain directory (strictly read-only source).
    const agyId = '11111111-2222-3333-4444-555555555555'
    const agyBrain = join(fakeHome, '.gemini', 'antigravity-cli', 'brain', agyId)
    await mkdir(join(agyBrain, '.system_generated', 'logs'), { recursive: true })
    await writeFile(join(agyBrain, 'task.md'), '# Transforming Qwen Into Loop Transformer\n\n- [x] Run 1k single-pass\n')
    await writeFile(
      join(agyBrain, '.system_generated', 'logs', 'transcript_full.jsonl'),
      [
        JSON.stringify({
          step_index: 0,
          source: 'USER_EXPLICIT',
          type: 'USER_INPUT',
          status: 'DONE',
          content: `<USER_REQUEST>\nTrain looped Qwen3.8 on 1k single pass\n</USER_REQUEST>\n<ADDITIONAL_METADATA>\nCommand Working Directory: ${repoDir}\n</ADDITIONAL_METADATA>`,
        }),
        JSON.stringify({
          step_index: 1,
          source: 'MODEL',
          type: 'PLANNER_RESPONSE',
          status: 'DONE',
          model: 'gemini-4-neon-high',
          thinking: 'Checking dataset shards',
          tool_calls: [{ name: 'run_command', args: { CommandLine: 'ls data/' } }],
        }),
        JSON.stringify({
          step_index: 2,
          source: 'MODEL',
          type: 'GENERIC',
          status: 'DONE',
          content: 'Command exited with code 0.\nshard_001.parquet',
        }),
        JSON.stringify({
          step_index: 3,
          source: 'MODEL',
          type: 'PLANNER_RESPONSE',
          status: 'DONE',
          model: 'gemini-4-neon-high',
          content: 'Verified all shards and launched single-pass 1k step run.',
        }),
      ].join('\n') + '\n',
    )

    // 2. Seed Claude Code project JSONL.
    const claudeId = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
    const claudeProj = join(fakeHome, '.claude', 'projects', '-tmp-easydel-src')
    await mkdir(claudeProj, { recursive: true })
    await writeFile(
      join(claudeProj, `${claudeId}.jsonl`),
      [
        JSON.stringify({ type: 'custom-title', customTitle: 'Claude TPU Fix', sessionId: claudeId }),
        JSON.stringify({ type: 'user', cwd: repoDir, sessionId: claudeId, message: { role: 'user', content: 'Audit TPU scripts' } }),
        JSON.stringify({ type: 'assistant', cwd: repoDir, sessionId: claudeId, message: { id: 'msg_1', role: 'assistant', model: 'claude-opus-4-6', content: [{ type: 'thinking', thinking: 'Scanning scripts' }] } }),
        JSON.stringify({ type: 'assistant', cwd: repoDir, sessionId: claudeId, message: { id: 'msg_1', role: 'assistant', model: 'claude-opus-4-6', content: [{ type: 'tool_use', id: 'tu_1', name: 'Read', input: { file_path: 'train.py' } }] } }),
        JSON.stringify({ type: 'user', cwd: repoDir, sessionId: claudeId, message: { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'tu_1', content: 'def main(): pass' }] } }),
        JSON.stringify({ type: 'assistant', cwd: repoDir, sessionId: claudeId, message: { id: 'msg_2', role: 'assistant', model: 'claude-opus-4-6', content: [{ type: 'text', text: 'Audit complete.' }] } }),
      ].join('\n') + '\n',
    )

    // 3. Seed Codex rollout JSONL.
    const codexId = '99999999-8888-7777-6666-555555555555'
    const codexDayDir = join(fakeHome, '.codex', 'sessions', '2026', '11', '26')
    await mkdir(codexDayDir, { recursive: true })
    await writeFile(
      join(codexDayDir, `rollout-2026-11-26T10-00-00-${codexId}.jsonl`),
      [
        JSON.stringify({ type: 'session_meta', payload: { session_id: codexId, cwd: repoDir } }),
        JSON.stringify({ type: 'turn_context', payload: { model: 'gpt-5.4', effort: 'xhigh', cwd: repoDir } }),
        JSON.stringify({ type: 'response_item', payload: { type: 'message', role: 'user', content: [{ type: 'input_text', text: '# AGENTS.md instructions for /repo\nignored boilerplate' }] } }),
        JSON.stringify({ type: 'response_item', payload: { type: 'message', role: 'user', content: [{ type: 'input_text', text: 'Optimize JAX kernel' }] } }),
        JSON.stringify({ type: 'response_item', payload: { type: 'reasoning', summary: [{ text: 'Analyzing Pallas tiling' }] } }),
        JSON.stringify({ type: 'response_item', payload: { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'Tuned block sizes to 128x128.' }] } }),
      ].join('\n') + '\n',
    )

    // Verify direct read-only discovery + transcript parsers across all 3 runtimes.
    expect(cleanAgyUserPromptText('<USER_REQUEST>\nHello\n</USER_REQUEST>\n<ADDITIONAL_METADATA>x</ADDITIONAL_METADATA>')).toBe('Hello')
    const listed = discoverImportableExternalSessions({ homeDir: fakeHome, cwd: repoDir, allProjects: true })
    expect(listed.map(item => `${item.runtime_id}:${item.id}`).sort()).toEqual([
      `antigravity:${agyId}`,
      `claude-code:${claudeId}`,
      `codex:${codexId}`,
    ])

    const agyLoaded = loadExternalSessionTranscript({ homeDir: fakeHome, runtimeId: 'antigravity', externalSessionId: agyId })
    expect(agyLoaded?.title).toBe('Transforming Qwen Into Loop Transformer')
    expect(agyLoaded?.model).toBe('gemini-4-neon-high')
    expect(agyLoaded?.thoughtLevel).toBe('high')
    expect(agyLoaded?.turnCount).toBe(1)
    expect(agyLoaded?.messages.map(m => m.role)).toEqual(['user', 'assistant', 'tool', 'assistant'])
    expect(agyLoaded?.toolExecutions).toHaveLength(1)

    const claudeLoaded = loadExternalSessionTranscript({ homeDir: fakeHome, runtimeId: 'claude-code', externalSessionId: claudeId })
    expect(claudeLoaded?.title).toBe('Claude TPU Fix')
    expect(claudeLoaded?.model).toBe('claude-opus-4-6')
    expect(claudeLoaded?.messages.map(m => m.role)).toEqual(['user', 'assistant', 'tool', 'assistant'])
    expect(claudeLoaded?.toolExecutions).toHaveLength(1)

    const codexLoaded = loadExternalSessionTranscript({ homeDir: fakeHome, runtimeId: 'codex', externalSessionId: codexId })
    expect(codexLoaded?.title).toBe('Optimize JAX kernel')
    expect(codexLoaded?.model).toBe('gpt-5.4')
    expect(codexLoaded?.thoughtLevel).toBe('xhigh')
    expect(codexLoaded?.messages.map(m => m.role)).toEqual(['user', 'assistant'])
    expect(codexLoaded?.messages[0]?.content).toBe('Optimize JAX kernel')

    // Verify end-to-end daemon RPC (`acp.importable_sessions` + `acp.import_session`) with fake HOME.
    const prevHome = process.env.HOME
    process.env.HOME = fakeHome
    const runtime = new InMemoryDaemonRuntime(undefined, {
      model: 'native-default',
      currentProjectDirectory: repoDir,
      sessionDirectory: sessionsDir,
    })
    const server = new DaemonServer({ runtime, socketPath, projectDirectory: repoDir })
    let socket: Socket | undefined
    const pending = new Map<number, (frame: Record<string, unknown>) => void>()
    let nextId = 0
    const callRpc = async (method: string, params: JsonRpcPayload = {}): Promise<JsonRpcPayload> => {
      const id = ++nextId
      return await new Promise<JsonRpcPayload>((resolve, reject) => {
        const timer = setTimeout(() => { pending.delete(id); reject(new Error(`RPC ${method} timed out`)) }, 5_000)
        pending.set(id, frame => {
          clearTimeout(timer)
          if (frame.error) reject(new Error(JSON.stringify(frame.error)))
          else resolve((frame.result as JsonRpcPayload) ?? {})
        })
        socket!.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`)
      })
    }
    try {
      await server.start()
      socket = await new Promise<Socket>((resolve, reject) => {
        const conn = connect(socketPath)
        conn.once('connect', () => resolve(conn))
        conn.once('error', reject)
      })
      socket.setEncoding('utf8')
      let buf = ''
      socket.on('data', (chunk: string) => {
        buf += chunk
        for (let nl = buf.indexOf('\n'); nl >= 0; nl = buf.indexOf('\n')) {
          const line = buf.slice(0, nl).trim()
          buf = buf.slice(nl + 1)
          if (!line) continue
          const frame = JSON.parse(line) as Record<string, unknown>
          if (typeof frame.id === 'number') {
            const waiter = pending.get(frame.id)
            if (waiter) { pending.delete(frame.id); waiter(frame) }
          }
        }
      })

      const importableBefore = await callRpc('acp.importable_sessions', { cwd: repoDir, all_projects: true })
      expect(importableBefore.ok).toBe(true)
      expect((importableBefore.sessions as Array<{ id: string }>).length).toBe(3)

      const imported = await callRpc('acp.import_session', {
        runtime_id: 'antigravity',
        external_session_id: agyId,
        cwd: repoDir,
        connect: false,
      })
      expect(imported.ok).toBe(true)
      expect(imported.title).toBe('Transforming Qwen Into Loop Transformer')
      expect(imported.imported_messages).toBe(4)
      expect(imported.imported_turns).toBe(1)

      // Subsequent listing marks the imported chat as `already_imported: true`.
      const importableAfter = await callRpc('acp.importable_sessions', { cwd: repoDir, all_projects: true })
      const agyRow = (importableAfter.sessions as Array<{ id: string; already_imported?: boolean }>).find(s => s.id === agyId)
      expect(agyRow?.already_imported).toBe(true)
    } finally {
      process.env.HOME = prevHome
      socket?.destroy()
      await server.stop()
      await rm(fakeHome, { recursive: true, force: true })
    }
  })
})
