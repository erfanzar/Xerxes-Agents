// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHmac } from 'node:crypto'
import { chmod, mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Database } from 'bun:sqlite'
import { afterEach, describe, expect, test } from 'bun:test'

import { AcpPermissionBoard, routePermission } from '../src/acp/permissions.js'
import { AcpAgentRunner } from '../src/acp/runner.js'
import { AcpServer } from '../src/acp/server.js'
import { AcpFrameByteCounter, MAX_ACP_FRAME_BYTES, serveACPStdio } from '../src/acp/transport.js'
import {
  ChannelManager,
  ChannelRoutingError,
  ChannelTurnRouter,
  ChannelWebhookServer,
  chunkText as discordChunkText,
  createChannelMessage,
  JsonChannelSessionIndex,
  MessageDirection,
  verifySlackSignature,
  WhatsAppChannel,
  whatsAppWebhookChallenge,
  type Channel,
  type ChannelMessage,
  type InboundHandler,
} from '../src/channels/index.js'
import { chunkText, sliceWithoutLoneSurrogate } from '../src/channels/textChunking.js'
import { DaemonInteractionBoard } from '../src/daemon/interactions.js'
import type { DaemonEvent, DaemonRuntime, DaemonSession, OpenSessionOptions } from '../src/daemon/runtime.js'
import { cursorScopeKey, decodeCursor, encodeCursor, resolveClientCursor } from '../src/hub/cursor.js'
import { emptyProjection, replaceSessionSummaries } from '../src/hub/projections.js'
import type { HubEventSink, UpstreamEvent } from '../src/hub/ports.js'
import { HUB_LIMITS, HubProtocolError, utf8ByteLength } from '../src/hub/protocol.js'
import { HubRuntimePool, type PoolWorkspace } from '../src/hub/runtimePool.js'
import { agentRowFromSnapshot, agentStepsFromEvents } from '../src/hub/server.js'
import { createSupervisorHandlers } from '../src/hub/supervisor/handlers.js'
import { SupervisorThreads } from '../src/hub/supervisor/threads.js'
import { HubVoiceModelStore, VoiceModelsUnavailable } from '../src/hub/voice/modelStore.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import type { JsonRpcPayload } from '../src/protocol/jsonRpc.js'
import type { RunSafety } from '../src/runtime/runSafety.js'
import type { PermissionRequest } from '../src/streaming/events.js'
import { disposeAll, FakeIdentity, standardUsers, startFixture, tempDir } from './hubHarness.js'

const localCleanups: Array<() => Promise<void> | void> = []

afterEach(async () => {
  for (const fn of localCleanups.splice(0).reverse()) {
    await fn()
  }
  await disposeAll()
})

function hasLoneSurrogate(text: string): boolean {
  for (let i = 0; i < text.length; i += 1) {
    const code = text.charCodeAt(i)
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = i + 1 < text.length ? text.charCodeAt(i + 1) : 0
      if (next < 0xdc00 || next > 0xdfff) return true
      i += 1
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      return true
    }
  }
  return false
}

describe('Hub surfaces: cursor scope & strict decoding', () => {
  test('distinguishes empty sessionIds from workspace-wide subscriptions and rejects malformed base64url cursors', () => {
    const allSessions = cursorScopeKey({ workspaceIds: ['w1'] })
    const noSessions = cursorScopeKey({ workspaceIds: ['w1'], sessionIds: [] })
    const namedSession = cursorScopeKey({ workspaceIds: ['w1'], sessionIds: ['s1'] })

    expect(allSessions).not.toBe(noSessions)
    expect(namedSession).not.toBe(noSessions)

    const emptyCursor = encodeCursor({ streamId: 'stream-1', sequence: 50, scopeKey: noSessions })
    expect(() => resolveClientCursor(emptyCursor, { streamId: 'stream-1', scopeKey: allSessions }))
      .toThrow(HubProtocolError)

    // Non-base64url suffix and empty stream/scope keys must fail closed.
    const valid = encodeCursor({ streamId: 'stream-1', sequence: 1, scopeKey: allSessions })
    expect(() => decodeCursor(`${valid}===`)).toThrow(HubProtocolError)
    const blankStream = 'hc1.' + Buffer.from(JSON.stringify({ s: '', n: 0, k: allSessions }), 'utf8').toString('base64url')
    expect(() => decodeCursor(blankStream)).toThrow(HubProtocolError)
  })
})

describe('Hub surfaces: runtime pool backfill cursor, delta coalescing, and cleanup', () => {
  async function startMockDaemon() {
    const dir = await mkdtemp(join('/tmp', 'xh-swarm-'))
    const socketPath = join(dir, 'd.sock')
    const requests: Array<{ id: number; method: string; params: Record<string, unknown> }> = []
    let readHandler: ((params: Record<string, unknown>) => Record<string, unknown>) | undefined
    const sockets = new Set<Bun.Socket<{ buffer: string }>>()
    const server = Bun.listen<{ buffer: string }>({
      unix: socketPath,
      socket: {
        open(socket) {
          sockets.add(socket)
          socket.data = { buffer: '' }
        },
        close(socket) {
          sockets.delete(socket)
        },
        data(socket, bytes) {
          socket.data.buffer += bytes.toString()
          for (;;) {
            const nl = socket.data.buffer.indexOf('\n')
            if (nl < 0) return
            const line = socket.data.buffer.slice(0, nl)
            socket.data.buffer = socket.data.buffer.slice(nl + 1)
            const frame = JSON.parse(line) as { id: number; method: string; params: Record<string, unknown> }
            requests.push(frame)
            let result: Record<string, unknown> = { ok: true }
            if (frame.method === 'initialize') {
              result = { ok: true, daemon_protocol: 35, session: { id: 'meta', key: frame.params.session_key } }
            } else if (frame.method === 'runtime.status') {
              result = {
                ok: true,
                daemon_protocol: 35,
                runtime_id: 'rt-1',
                instance_id: 'inst-1',
                event_stream_supported: true,
                principal_auth_v1_supported: true,
              }
            } else if (frame.method === 'session.active_list') {
              result = { ok: true, sessions: [{ id: 's0', key: 'k0', cwd: '/a', title: 'Task', status: 'idle', model: 'm' }] }
            } else if (frame.method === 'events.subscribe') {
              result = { ok: true, subscription_id: `sub-${requests.length}`, stream_id: 'stream-1', watermark: 'c0', oldest_cursor: 'c0', reset_required: false }
            } else if (frame.method === 'events.read') {
              result = readHandler ? readHandler(frame.params) : { ok: true, events: [], through_cursor: frame.params.after_cursor ?? 'c0' }
            }
            socket.write(JSON.stringify({ jsonrpc: '2.0', id: frame.id, result }) + '\n')
          }
        },
      },
    })
    localCleanups.push(async () => {
      for (const s of sockets) s.terminate()
      server.stop(true)
      await rm(dir, { recursive: true, force: true })
    })
    return {
      socketPath,
      requests,
      onRead(fn: (params: Record<string, unknown>) => Record<string, unknown>) { readHandler = fn },
      push(type: string, payload: Record<string, unknown>) {
        for (const s of sockets) {
          s.write(JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { type, payload } }) + '\n')
        }
      },
    }
  }

  test('advances live subscription to final backfill page cursor and flushes trailing backfill deltas before going online', async () => {
    const daemon = await startMockDaemon()
    const mkDelta = (sequence: number, text: string): UpstreamEvent => ({
      event_id: `e${sequence}`,
      sequence,
      cursor: `c${sequence}`,
      runtime_id: 'rt-1',
      instance_id: 'inst-1',
      project_dir: '/a',
      session_id: 's0',
      type: 'text_delta',
      payload: { delta: text },
      occurred_at: new Date().toISOString(),
    })

    daemon.onRead(params => params.after_cursor === 'c1'
      ? { ok: true, events: [mkDelta(2, 'Hel'), mkDelta(3, 'lo')], through_cursor: 'c3', has_more: false }
      : { ok: true, events: [mkDelta(1, 'Hi ')], next_cursor: 'c1', has_more: true })

    const rows: UpstreamEvent[] = []
    let committed: string | undefined
    const sink: HubEventSink = {
      ingest(_rt, _ws, events) {
        rows.push(...events)
        committed = events.at(-1)?.cursor ?? committed
      },
      upstreamCursor: () => committed,
    }
    const pool = new HubRuntimePool({ sink, coalesceMs: 60_000, pollIntervalMs: 60_000 })
    localCleanups.push(() => pool.stop())
    const ws: PoolWorkspace = { id: 'w1', label: 'W1', endpoint: { kind: 'local', socketPath: daemon.socketPath }, projectDir: '/a', autoConnect: true }
    await pool.start([ws])

    expect(pool.health('w1')?.state).toBe('online')
    // Trailing backfill deltas were flushed immediately (despite 60s coalesceMs) and live subscribe resumed at c3, not c1.
    expect(rows).toHaveLength(1)
    expect(rows[0]).toMatchObject({ type: 'text_part', sequence: 3, cursor: 'c3', payload: { text: 'Hi Hello' } })
    expect(committed).toBe('c3')
    const subscribes = daemon.requests.filter(r => r.method === 'events.subscribe')
    expect(subscribes.at(-1)?.params.after_cursor).toBe('c3')
  })

  test('coalesces deltas regardless of payload key order and drops pending delta timers on workspace removal', async () => {
    const daemon = await startMockDaemon()
    const rows: UpstreamEvent[] = []
    const sink: HubEventSink = {
      ingest(_rt, _ws, events) { rows.push(...events) },
      upstreamCursor: () => undefined,
    }
    const pool = new HubRuntimePool({ sink, coalesceMs: 80, pollIntervalMs: 60_000 })
    localCleanups.push(() => pool.stop())
    await pool.start([{ id: 'w1', label: 'W1', endpoint: { kind: 'local', socketPath: daemon.socketPath }, projectDir: '/a', autoConnect: true }])
    const subId = String(daemon.requests.filter(r => r.method === 'events.subscribe').at(-1)?.params.project_dirs ? 'sub-4' : 'sub-4')
    // Find actual subscription id returned to the pool (`sub-${requests.length}` when events.subscribe was handled).
    const subReqIndex = daemon.requests.findIndex(r => r.method === 'events.subscribe') + 1
    const activeSubId = `sub-${subReqIndex}`

    daemon.push('event_batch', {
      subscription_id: activeSubId,
      events: [
        {
          event_id: 'e1', sequence: 1, cursor: 'c1', runtime_id: 'rt-1', instance_id: 'inst-1',
          project_dir: '/a', session_id: 's0', type: 'text_part',
          payload: { text: 'One ', meta_a: 1, meta_b: 2 }, occurred_at: new Date().toISOString(),
        },
        {
          event_id: 'e2', sequence: 2, cursor: 'c2', runtime_id: 'rt-1', instance_id: 'inst-1',
          project_dir: '/a', session_id: 's0', type: 'text_part',
          payload: { meta_b: 2, meta_a: 1, text: 'Two' }, occurred_at: new Date().toISOString(),
        },
        {
          event_id: 'e3', sequence: 3, cursor: 'c3', runtime_id: 'rt-1', instance_id: 'inst-1',
          project_dir: '/a', session_id: 's0', type: 'turn_end',
          payload: {}, occurred_at: new Date().toISOString(),
        },
      ],
    })
    const deadline = Date.now() + 2000
    while (rows.length < 2 && Date.now() < deadline) await Bun.sleep(5)
    expect(rows).toHaveLength(2)
    expect(rows[0]).toMatchObject({ type: 'text_part', sequence: 2, payload: { text: 'One Two', meta_a: 1, meta_b: 2 } })

    // Buffer a new delta and immediately remove the workspace before coalesceMs elapses.
    daemon.push('event_batch', {
      subscription_id: activeSubId,
      events: [{
        event_id: 'e4', sequence: 4, cursor: 'c4', runtime_id: 'rt-1', instance_id: 'inst-1',
        project_dir: '/a', session_id: 's0', type: 'text_part',
        payload: { text: 'dropped on remove' }, occurred_at: new Date().toISOString(),
      }],
    })
    await Bun.sleep(10)
    await pool.removeWorkspace('w1')
    await Bun.sleep(120)
    expect(rows).toHaveLength(2)
    expect(subId).toBeDefined()
  })
})

describe('Hub surfaces: frame bounds, surrogate-safe clipping, and voice mirror validation', () => {
  test('clipText helpers on agent rows/steps never produce lone UTF-16 surrogates', () => {
    const emojiTitle = `${'a'.repeat(398)}🚀tail`
    const row = agentRowFromSnapshot({ id: 'ag-1', status: 'running', name: 'worker', title: emojiTitle })
    expect(row).toBeDefined()
    expect(hasLoneSurrogate(String(row!.title))).toBe(false)
    expect(JSON.stringify(row)).not.toMatch(/\\ud[89ab][0-9a-f]{2}/i)

    const steps = agentStepsFromEvents([{
      type: 'subagent_event',
      payload: {
        agent_id: 'ag-1',
        event: { type: 'text_part', payload: { text_full: `${'x'.repeat(7998)}🎉more` } },
      },
    }], 'ag-1')
    expect(steps).toHaveLength(1)
    expect(hasLoneSurrogate(steps[0]!.text!)).toBe(false)
  })

  test('supervisor.thread.open bounds large transcripts inside HUB_LIMITS.normalFrameBytes', async () => {
    const db = new Database(':memory:', { strict: true })
    localCleanups.push(() => db.close())
    const identity = standardUsers(new FakeIdentity())
    const threads = new SupervisorThreads({ db, identity })
    localCleanups.push(() => threads.close())
    const shared = threads.ensureShared()

    // Insert 60 large XERXES replies (~20 KiB each => ~1.2 MiB total > 1 MiB maxFrameBytes).
    for (let i = 0; i < 60; i += 1) {
      threads.append({
        threadId: shared.id,
        role: 'xerxes',
        text: `reply-${i}:${'日本語😀'.repeat(2_200)}`,
      })
    }

    const handlers = createSupervisorHandlers({
      threads,
      actions: { cardsFor: () => [], respond: async () => { throw new Error('unused') }, status: async () => { throw new Error('unused') } },
      modelId: 'test-model',
      usage: () => ({ tokens: 0, costUsd: 0, calls: 0, windowStart: new Date(0).toISOString() }),
      submit: () => { throw new Error('unused') },
    })
    const openHandler = handlers['supervisor.thread.open']!
    const result = await openHandler({
      context: identity.context('owner'),
      grants: [],
    }, {})

    const bytes = utf8ByteLength(JSON.stringify(result))
    expect(bytes).toBeLessThanOrEqual(HUB_LIMITS.normalFrameBytes)
    const messages = result.messages as Array<{ text: string }>
    expect(messages.length).toBeGreaterThan(0)
    expect(messages.at(-1)?.text.startsWith('reply-59:')).toBe(true)
  })

  test('hub.snapshot degrades gracefully below normalFrameBytes when many workspaces hold pending cards', async () => {
    const workspaceList = Array.from({ length: 30 }, (_, idx) => ({ id: `ws-${idx}`, label: `Workspace ${idx}` }))
    const projections = new Map(workspaceList.map(ws => [
      ws.id,
      replaceSessionSummaries(emptyProjection(ws.id), [{
        id: 'sess-1',
        title: 'Task',
        status: 'needs_you',
        pending_interaction_count: 1,
      }], new Date().toISOString()),
    ]))

    const fixture = await startFixture({
      server: {
        workspaces: {
          list: () => workspaceList,
          health: () => ({ state: 'online', degraded: false }),
          projection: wsId => projections.get(wsId),
        },
      },
    })
    for (const ws of workspaceList) {
      fixture.store.append({
        workspace_id: ws.id,
        session_id: 'sess-1',
        type: 'question_request',
        payload: {
          id: `q-${ws.id}`,
          request_id: `q-${ws.id}`,
          questions: [{ id: 'q1', question: 'x'.repeat(12_000), options: ['yes', 'no'] }],
        },
      })
    }

    const client = await fixture.client('owner')
    const result = await client.ok('hub.snapshot', { limit: 1 })
    expect(utf8ByteLength(JSON.stringify(result))).toBeLessThanOrEqual(HUB_LIMITS.normalFrameBytes)
    expect(result.truncated).toBe(true)
  })

  test('HubVoiceModelStore rejects mirror baseUrls carrying query strings or hash fragments', async () => {
    const dir = await tempDir()
    const store = new HubVoiceModelStore({
      stateDir: dir,
      environment: { XERXES_HUB_VOICE_MODELS_URL: 'https://mirror.example.test/models?token=bad' },
      fetch: async () => {
        throw new Error('fetch must not be called for invalid mirror URL')
      },
    })
    localCleanups.push(() => store.close())
    const firstFile = store.files('light')[0]!
    await expect(store.get(firstFile)).rejects.toBeInstanceOf(VoiceModelsUnavailable)
  })
})

describe('ACP surfaces: frame byte counter, stream cancellation, and permission cleanup', () => {
  test('AcpFrameByteCounter stays exceeded across trailing newline and serveACPStdio cancels reader on oversized line', async () => {
    const counter = new AcpFrameByteCounter(10)
    counter.append(new TextEncoder().encode('01234567890\n'))
    expect(counter.exceedsLimit).toBe(true)

    let cancelled = false
    const server = new AcpServer({ promptHandler: () => ({ ok: true }) })
    const oversizedLine = `{"jsonrpc":"2.0","id":1,"padding":"${'a'.repeat(MAX_ACP_FRAME_BYTES)}"}\n`
    const input = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(oversizedLine))
        // Stream intentionally left open to verify reader.cancel() is invoked.
      },
      cancel() {
        cancelled = true
      },
    })
    const output: string[] = []
    await serveACPStdio(server, input, line => { output.push(line) })
    expect(cancelled).toBe(true)
    expect(output).toHaveLength(1)
    expect(JSON.parse(output[0]!)).toMatchObject({ error: { code: -32600 } })
  })

  test('AcpServer cancel, closeSession, and shutdown drop pending permissions without hanging waiters', async () => {
    const board = new AcpPermissionBoard()
    const p1 = routePermission({ sessionId: 's1', toolName: 'Bash', description: 'run 1', inputs: {} })
    const p2 = routePermission({ sessionId: 's2', toolName: 'Bash', description: 'run 2', inputs: {} })
    const wait1 = board.submit(p1)
    const wait2 = board.submit(p2)

    expect(board.dropSession('s1')).toBe(1)
    await expect(wait1).resolves.toBe(false)
    expect(board.snapshotPending().map(r => r.id)).toEqual([p2.id])
    expect(board.clear()).toBe(1)
    await expect(wait2).resolves.toBe(false)
    expect(board.snapshotPending()).toEqual([])

    const server = new AcpServer({ promptHandler: () => ({ ok: true }) })
    const sA = String(server.openSession('/tmp').session_id)
    const sB = String(server.openSession('/tmp').session_id)
    server.requestPermission({ sessionId: sA, toolName: 'WriteFile', description: 'write a', inputs: {} })
    server.requestPermission({ sessionId: sB, toolName: 'WriteFile', description: 'write b', inputs: {} })
    expect(server.pendingPermissions()).toHaveLength(2)

    server.closeSession(sA)
    expect(server.pendingPermissions().map(p => p.session_id)).toEqual([sB])
    server.shutdown()
    expect(server.pendingPermissions()).toEqual([])
  })

  test('AcpAgentRunner aborts before model execution if cancelled while runSafetyFor is initializing', async () => {
    let llmCalled = false
    const llm: LlmClient = {
      async *stream(_req: CompletionRequest): AsyncGenerator<LlmDelta> {
        llmCalled = true
        yield { content: 'should not run' }
      },
    }
    const safetyStarted = Promise.withResolvers<void>()
    const releaseSafety = Promise.withResolvers<RunSafety | undefined>()
    const runner = new AcpAgentRunner({
      llm,
      model: 'test-model',
      runSafetyFor: async () => {
        safetyStarted.resolve()
        return releaseSafety.promise
      },
    })
    const server = new AcpServer({ runner })
    const sessionId = String(server.openSession('/tmp').session_id)
    const session = server.sessions.get(sessionId)!

    const running = runner.runPrompt({ session, text: 'hello' })
    await safetyStarted.promise
    expect(server.cancel(sessionId)).toEqual({ ok: true })
    releaseSafety.resolve(undefined)

    await expect(running).resolves.toMatchObject({ ok: true, cancelled: true })
    expect(llmCalled).toBe(false)
  })
})

describe('Channels surfaces: surrogate/fence chunking, turnRouter waits/identity, sessionIndex, and webhooks', () => {
  test('chunkText preserves UTF-16 surrogate pairs and balances Markdown fenced code blocks across chunks', () => {
    // Surrogate boundary right at limit (emoji straddling index 9..10).
    const emojiInput = `${'a'.repeat(9)}🚀${'b'.repeat(9)}`
    const emojiChunks = chunkText(emojiInput, 10)
    for (const chunk of emojiChunks) {
      expect(chunk.length).toBeLessThanOrEqual(10)
      expect(hasLoneSurrogate(chunk)).toBe(false)
    }
    expect(emojiChunks.join('')).toBe(emojiInput)

    // Discord wrapper delegates to the same implementation.
    const discordChunks = discordChunkText(emojiInput, 10)
    expect(discordChunks).toEqual(emojiChunks)

    // Markdown fenced code block split across multiple chunks keeps balanced fences and indentation.
    const codeLines = Array.from({ length: 12 }, (_, idx) => `  const value${idx} = ${idx}`)
    const markdown = ['Intro paragraph:', '```ts', ...codeLines, '```', 'Summary line.'].join('\n')
    const chunks = chunkText(markdown, 120)
    expect(chunks.length).toBeGreaterThan(1)
    for (const chunk of chunks) {
      expect(chunk.length).toBeLessThanOrEqual(120)
      const fenceCount = (chunk.match(/^`{3}/gm) ?? []).length
      expect(fenceCount % 2).toBe(0)
    }
    expect(chunks[0]).toContain('```ts\n  const value0 = 0')
    expect(chunks[0]?.endsWith('\n```')).toBe(true)
    expect(chunks[1]?.startsWith('```ts\n  const value')).toBe(true)

    expect(sliceWithoutLoneSurrogate('abcd🚀ef', 5)).toBe('abcd')
    expect(sliceWithoutLoneSurrogate('abcd🚀ef', 6)).toBe('abcd🚀')
  })

  test('chunkText handles back-to-back mixed fences, long info strings, and whitespace-only fence prefixes without looping or mismatching markers', () => {
    // Exact verifier repro: earlier tilde fence with long info string followed by a backtick fence.
    const mixedRepro = [
      `~~~${'a'.repeat(44)}`,
      '   ~~~~',
      '`'.repeat(12),
      'b'.repeat(30),
    ].join('\n')
    for (let limit = 1; limit <= mixedRepro.length + 5; limit += 1) {
      const pieces = chunkText(mixedRepro, limit)
      expect(pieces.length).toBeGreaterThan(0)
      for (const piece of pieces) {
        expect(piece.length).toBeLessThanOrEqual(limit)
        expect(hasLoneSurrogate(piece)).toBe(false)
      }
    }

    // Earlier tilde fence WITH body lines + second backtick fence: when stepping back lands inside the tilde fence,
    // it must close with tildes (`~~~`), never with the later fence's backticks.
    const mixedWithBody = [
      `~~~${'x'.repeat(20)}`,
      'tilde_body_line_1',
      '~~~~',
      '``````',
      'backtick_body_line_1',
      '``````',
    ].join('\n')
    for (let limit = 30; limit <= 80; limit += 1) {
      const pieces = chunkText(mixedWithBody, limit)
      for (const piece of pieces) {
        expect(piece.length).toBeLessThanOrEqual(limit)
        // Every chunk that opens ~~~xxx... without reaching ~~~~ must never close with backticks.
        if (piece.startsWith('~~~x') && !piece.includes('\n~~~~')) {
          expect(piece.endsWith('\n``````')).toBe(false)
          if (limit >= 45) {
            expect(piece.endsWith('\n~~~')).toBe(true)
          }
        }
        // Once every fence line fits comfortably, every chunk has zero unclosed fences.
        if (limit >= 45) {
          const sixBackticks = (piece.match(/^`{6}$/gm) ?? []).length
          expect(sixBackticks % 2).toBe(0)
        }
      }
    }

    // Whitespace-only prefix after an opening fence line must never emit an empty fenced block (` ```lang\n``` `).
    const whitespaceAfterFence = ['   ```typescript   ', '   ', '  ', 'const answer = 42', '```'].join('\n')
    for (let limit = 15; limit <= 55; limit += 1) {
      const pieces = chunkText(whitespaceAfterFence, limit)
      for (const piece of pieces) {
        expect(piece.length).toBeLessThanOrEqual(limit)
        expect(piece).not.toMatch(/^ *`{3}[a-z]* *\n`{3}$/)
      }
    }
  })

  test('ChannelTurnRouter rejects blank/group-roomless identities and handles slash commands cleanly during waits', async () => {
    class TestChannel implements Channel {
      readonly name = 'slack'
      readonly sent: ChannelMessage[] = []
      private inbound: InboundHandler | undefined
      async start(onInbound: InboundHandler): Promise<void> { this.inbound = onInbound }
      async stop(): Promise<void> { this.inbound = undefined }
      async send(message: ChannelMessage): Promise<void> { this.sent.push(message) }
      async receive(message: ChannelMessage): Promise<void> { await this.inbound?.(message) }
    }
    class ParkableRuntime implements DaemonRuntime {
      private sessions = new Map<string, DaemonSession>()
      private controller = new AbortController()
      constructor(private readonly board: DaemonInteractionBoard) {}
      cancelAllTurns(): number { return 0 }
      cancelTurn(): boolean { this.controller.abort(); return true }
      evictSession(key: string): void { this.sessions.delete(key) }
      async flushSessions(): Promise<void> {}
      async listSavedSessions(): Promise<readonly []> { return [] }
      listSessions(): readonly DaemonSession[] { return [...this.sessions.values()] }
      async openSession(sessionKey: string, agentId = 'default', _options: OpenSessionOptions = {}): Promise<DaemonSession> {
        const existing = this.sessions.get(sessionKey)
        if (existing) return existing
        const created: DaemonSession = {
          activeTurnId: '', agentId, cancelRequested: false, cwd: '/workspace', extra: {},
          id: `${sessionKey}-id`, interactionMode: 'code', lastActive: 0, messages: [], metadata: {},
          model: 'gpt-test', planMode: false, sessionKey, status: 'idle', thinkingContent: [],
          toolExecutions: [], totalInputTokens: 10, totalOutputTokens: 5, turnCount: 1, workspace: '/workspace',
        }
        this.sessions.set(sessionKey, created)
        return created
      }
      reload(): JsonRpcPayload { return {} }
      async setSessionMode(key: string): Promise<DaemonSession | undefined> { return this.sessions.get(key) }
      sessionStatus(key: string): DaemonSession | undefined { return this.sessions.get(key) }
      steerTurn(): boolean { return false }
      status(): JsonRpcPayload { return { runtime: 'bun-typescript', model: 'gpt-test' } }
      async submitTurn(sessionKey: string, _text: string, emit: (event: DaemonEvent) => void): Promise<void> {
        this.controller = new AbortController()
        const session = this.sessions.get(sessionKey)!
        const release = this.board.bind(session.id, emit)
        try {
          const req: PermissionRequest = {
            requestId: 'perm-1',
            description: 'Run deploy script',
            inputs: {},
            toolCall: { id: 'tc-1', type: 'function', function: { name: 'Bash', arguments: {} } },
          }
          emit({ type: 'approval_request', payload: { id: req.requestId, request_id: req.requestId, description: req.description } })
          const decision = await this.board.permissionBroker(session.id).request(req, this.controller.signal)
          emit({ type: 'text_part', payload: { text: `done:${decision}` } })
        } finally {
          release()
        }
      }
    }

    const channel = new TestChannel()
    const board = new DaemonInteractionBoard()
    const runtime = new ParkableRuntime(board)
    const errors: unknown[] = []
    const manager = new ChannelManager({ channels: [['slack', channel]] })
    const router = new ChannelTurnRouter({
      channels: manager,
      runtime,
      interactions: board,
      streamPreviews: false,
      onError: err => { errors.push(err) },
    })
    manager.setInboundHandler(msg => router.handle(msg))
    await manager.enable('slack')

    // Empty string channelUserId and roomId (as produced by stringOrEmpty on malformed events) must fail closed.
    await channel.receive(createChannelMessage({
      channel: 'slack',
      channelUserId: '   ',
      roomId: '',
      direction: MessageDirection.INBOUND,
      text: 'anonymous',
    }))
    // Group message with missing roomId must also fail closed instead of pooling into slack:chat::thread:main.
    await channel.receive(createChannelMessage({
      channel: 'slack',
      channelUserId: 'U1',
      roomId: '',
      metadata: { chat_type: 'group' },
      direction: MessageDirection.INBOUND,
      text: 'group without room',
    }))
    expect(errors).toHaveLength(2)
    expect(errors.every(e => e instanceof ChannelRoutingError)).toBe(true)

    // Start a turn that parks on approval.
    const msg = (text: string) => createChannelMessage({
      channel: 'slack',
      channelUserId: 'U1',
      roomId: 'C1',
      direction: MessageDirection.INBOUND,
      text,
    })
    const activeTurn = channel.receive(msg('deploy now'))
    const deadline = Date.now() + 1000
    while (!channel.sent.some(s => s.text.includes('Approval needed')) && Date.now() < deadline) {
      await Bun.sleep(5)
    }

    // Read-only /status and /help respond immediately without deadlocking behind the parked turn.
    await channel.receive(msg('/status'))
    expect(channel.sent.at(-1)?.text).toContain('Xerxes status:')
    // Non-approval mutating command (/new) hits answerWait guidance instead of hanging in pendingBySession.
    await channel.receive(msg('/new'))
    expect(channel.sent.at(-1)?.text).toContain('A tool is waiting for approval.')

    // Approve to finish cleanly.
    await channel.receive(msg('/approve'))
    await activeTurn
    expect(channel.sent.at(-1)?.text).toBe('done:approve')
  })

  test('JsonChannelSessionIndex does not poison in-memory state on persist failure and preserves __proto__ keys', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'xerxes-sess-idx-'))
    localCleanups.push(() => rm(dir, { recursive: true, force: true }))
    const subDir = join(dir, 'locked')
    await Bun.write(join(subDir, '.keep'), '')
    const indexPath = join(subDir, 'sessions.json')
    const index = new JsonChannelSessionIndex(indexPath)

    // Make parent directory read-only so first persist fails.
    await chmod(subDir, 0o500)
    try {
      await expect(index.set('telegram:private:1', 'sess-1')).rejects.toThrow()
      expect(await index.get('telegram:private:1')).toBeUndefined()
    } finally {
      await chmod(subDir, 0o700)
    }

    // Retry with identical (key, id) must still write to disk now that permissions are restored.
    await index.set('telegram:private:1', 'sess-1')
    await index.set('__proto__', 'sess-proto')

    const reloaded = new JsonChannelSessionIndex(indexPath)
    expect(await reloaded.get('telegram:private:1')).toBe('sess-1')
    expect(await reloaded.get('__proto__')).toBe('sess-proto')
  })

  test('webhook security helpers reject empty WhatsApp verifyToken, overflowed Slack timestamps, and invalid 127.x hosts', async () => {
    expect(whatsAppWebhookChallenge({ 'hub.mode': 'subscribe', 'hub.verify_token': '', 'hub.challenge': 'c' }, '')).toBeUndefined()
    expect(whatsAppWebhookChallenge({ 'hub.mode': 'subscribe', 'hub.verify_token': 'secret', 'hub.challenge': 'ok' }, 'secret')).toBe('ok')

    // Uppercase hex X-Hub-Signature-256 is accepted when valid.
    const body = new TextEncoder().encode(JSON.stringify({
      entry: [{ changes: [{ value: { messages: [{ from: '1555', id: 'm1', type: 'text', text: { body: 'hi' } }] } }] }],
    }))
    const upperSig = 'SHA256=' + createHmac('sha256', 'app-secret').update(body).digest('hex').toUpperCase()
    const wa = new WhatsAppChannel({ accessToken: 'tok', phoneNumberId: 'pid', appSecret: 'app-secret' })
    const received: string[] = []
    await wa.start(async m => { received.push(m.text) })
    expect(await wa.handleWebhook({ 'X-Hub-Signature-256': upperSig }, body)).toEqual({ status: 200, body: 'ok' })
    expect(received).toEqual(['hi'])

    // Overflowed >16-digit Slack timestamp fails closed.
    const hugeTs = '9'.repeat(400)
    expect(verifySlackSignature('sec', {
      'x-slack-request-timestamp': hugeTs,
      'x-slack-signature': 'v0=' + '0'.repeat(64),
    }, body)).toBe(false)

    // Invalid IPv4 octet 127.999.999.999 must not bypass non-loopback authToken enforcement.
    const manager = new ChannelManager({ channels: [] })
    expect(() => new ChannelWebhookServer({ host: '127.999.999.999', manager }))
      .toThrow('non-loopback webhook listeners require authToken')
  })
})
