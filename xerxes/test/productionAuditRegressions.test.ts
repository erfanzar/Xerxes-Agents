// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { AcpStdioClient, type AcpClientEvent, type AcpSpawnedProcess } from '../src/acp/client.js'
import { discoverExternalAcpAgents } from '../src/acp/externalAgents.js'
import { DaemonAcpBridgeHost } from '../src/daemon/acpBridgeHost.js'
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
})
