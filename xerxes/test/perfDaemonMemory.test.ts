// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { SubAgentManager, type SubAgentEvent } from '../src/agents/subagentManager.js'
import { DaemonAcpBridgeHost } from '../src/daemon/acpBridgeHost.js'
import { EventSubscriptions } from '../src/daemon/eventSubscriptions.js'
import { InMemoryDaemonRuntime, type DaemonEvent, type TurnRunner } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { daemonEventFromSubagent } from '../src/daemon/subagentHost.js'
import { DaemonSwarmHost } from '../src/daemon/swarmHost.js'
import { TransportEventJournal } from '../src/daemon/transportEventJournal.js'
import type { JsonRpcPayload } from '../src/protocol/jsonRpc.js'
import { MemoryGuard, type MemoryGuardPorts } from '../src/runtime/memoryGuard.js'
import { RunHistory } from '../src/runtime/runHistory.js'
import { TerminalMonitors } from '../src/runtime/terminalMonitors.js'
import { TerminalRegistry } from '../src/runtime/terminalRegistry.js'
import { DaemonTranscriptStore, normalizeDaemonTranscript } from '../src/session/daemonTranscript.js'
import { SwarmBoardStore, type PersistedSwarmState } from '../src/swarm/board.js'
import { SwarmManager } from '../src/swarm/manager.js'
import { MAX_RETAINED_WORKER_TOOL_CONTENT_CHARS } from '../src/swarm/runner.js'
import type { SwarmLifecycleEvent } from '../src/swarm/types.js'

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
    const pending = this.next(frame => frame.id === id)
    this.socket.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`)
    const frame = await pending
    if (frame.error) {
      throw new Error(`RPC ${method} failed (${frame.error.code}): ${frame.error.message}`)
    }
    return frame.result ?? {}
  }

  close(): void {
    this.socket.destroy()
  }
}

async function withTempDir<T>(fn: (dir: string) => Promise<T>): Promise<T> {
  const dir = await mkdtemp(join(tmpdir(), 'xerxes-perf-daemon-'))
  try {
    return await fn(dir)
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
}

describe('Daemon & Swarm CPU/RAM Performance Contracts', () => {
  test('SwarmManager evicts killed worker histories from RAM, lazy-loads on inspectMember & reload, and caps oversized tool output', async () => {
    await withTempDir(async (root) => {
      const workspace = join(root, 'ws')
      const storage = join(root, 'swarm')
      const giantToolPayload = 'Z'.repeat(MAX_RETAINED_WORKER_TOOL_CONTENT_CHARS + 20_000)

      const swarm = await SwarmManager.open({
        sessionId: 'sess-evict-workers',
        workspaceDir: workspace,
        storageDir: storage,
        workerRunner: async (input) => ({
          messages: [
            ...input.messages,
            { role: 'user', content: input.prompt },
            {
              role: 'assistant',
              content: `Finished work for ${input.member.name}.`,
              tool_calls: [{ id: `tc_${input.member.name}`, type: 'function', function: { name: 'ReadFile', arguments: {} } }],
            },
            {
              role: 'tool',
              tool_call_id: `tc_${input.member.name}`,
              name: 'ReadFile',
              content: giantToolPayload,
            },
            { role: 'assistant', content: `Summary from ${input.member.name}` },
          ],
        }),
      })

      await swarm.add({ name: 'worker-a', brief: 'Task A' })
      await swarm.add({ name: 'worker-b', brief: 'Task B' })
      await swarm.waitForIdle()

      expect(swarm.inMemoryHistoryCount()).toBe(2)

      // Oversized tool output was capped cleanly in retained history rather than keeping the giant blob.
      const liveHistory = swarm.getMemberHistory('worker-a')
      const toolMsg = liveHistory.find(m => m.role === 'tool')
      expect(typeof toolMsg?.content).toBe('string')
      expect((toolMsg?.content as string).length).toBeLessThan(MAX_RETAINED_WORKER_TOOL_CONTENT_CHARS + 128)
      expect(toolMsg?.content).toContain('[truncated')

      // Killing worker-a persists its history to disk and evicts it from RAM immediately.
      await swarm.kill({ caller: 'coordinator', name: 'worker-a' })
      expect(swarm.inMemoryHistoryCount()).toBe(1)

      // inspectMember still serves worker-a's full bounded history lazily from disk!
      const inspected = swarm.inspectMember('worker-a')
      expect(inspected.member.killed).toBe(true)
      expect(inspected.total_messages).toBe(4)
      expect(inspected.messages.at(-1)?.text).toBe('Summary from worker-a')
      // Lazy inspect uses a separate bounded cache and never re-pins into memberHistories.
      expect(swarm.inMemoryHistoryCount()).toBe(1)

      await swarm.dispose()

      // Reloading a persisted swarm loads ZERO inactive (stopped/killed) histories into RAM eagerly!
      const reloaded = await SwarmManager.open({
        sessionId: 'sess-evict-workers',
        workspaceDir: workspace,
        storageDir: storage,
      })
      expect(reloaded.inMemoryHistoryCount()).toBe(0)
      expect(reloaded.inspectMember('worker-a').messages.at(-1)?.text).toBe('Summary from worker-a')
      expect(reloaded.inspectMember('worker-b').messages.at(-1)?.text).toBe('Summary from worker-b')
      expect(reloaded.inMemoryHistoryCount()).toBe(0)
      await reloaded.dispose()
    })
  })

  test('SwarmManager deduplicates identical activity snapshots and enforces >=500ms same-phase throttle', async () => {
    await withTempDir(async (root) => {
      let nowMs = 1_700_000_000_000
      const activityEvents: SwarmLifecycleEvent[] = []
      const releaseTurn = Promise.withResolvers<void>()
      const started = Promise.withResolvers<void>()

      const swarm = await SwarmManager.open({
        sessionId: 'sess-activity-dedup',
        workspaceDir: join(root, 'ws'),
        storageDir: join(root, 'swarm'),
        now: () => new Date(nowMs),
        onEvent: (event) => {
          if (event.kind === 'member_activity') activityEvents.push(event)
        },
        workerRunner: async () => {
          started.resolve()
          await releaseTurn.promise
          return { messages: [] }
        },
      })

      await swarm.add({ name: 'thinker', brief: 'Think deeply' })
      await started.promise

      // First thinking token emits once.
      swarm.recordWorkerStreamEvent('thinker', { type: 'thinking', text: 'step 1' })
      expect(activityEvents).toHaveLength(1)

      // Subsequent thinking deltas keep identical phase/summary/liveText -> deduplicated even past 1000ms!
      nowMs += 600
      swarm.recordWorkerStreamEvent('thinker', { type: 'thinking', text: 'step 2' })
      nowMs += 600
      swarm.recordWorkerStreamEvent('thinker', { type: 'thinking', text: 'step 3' })
      expect(activityEvents).toHaveLength(1)

      // Phase transition thinking -> responding emits past 250ms.
      nowMs += 300
      swarm.recordWorkerStreamEvent('thinker', { type: 'text', text: 'Hello ' })
      expect(activityEvents).toHaveLength(2)

      // Same-phase responding deltas within 500ms are throttled.
      nowMs += 300
      swarm.recordWorkerStreamEvent('thinker', { type: 'text', text: 'world ' })
      expect(activityEvents).toHaveLength(2)

      // Advancing past 500ms lets the updated responding text emit.
      nowMs += 250
      swarm.recordWorkerStreamEvent('thinker', { type: 'text', text: 'again!' })
      expect(activityEvents).toHaveLength(3)
      expect(activityEvents[2]?.activity?.live_text_preview).toBe('Hello world again!')

      releaseTurn.resolve()
      await swarm.waitForIdle()
      await swarm.dispose()
    })
  })

  test('DaemonServer emits compact member_activity without full status and skips TransportEventJournal persistence', async () => {
    await withTempDir(async (root) => {
      const socketPath = join(root, 'daemon.sock')
      const journalDir = join(root, 'journal')
      const swarmHost = new DaemonSwarmHost({ storageRootDir: join(root, 'swarms') })
      const runtime = new InMemoryDaemonRuntime(undefined, {
        currentProjectDirectory: root,
        sessionDirectory: join(root, 'sessions'),
      })
      const server = new DaemonServer({
        socketPath,
        projectDirectory: root,
        runtime,
        swarmHost,
        eventJournal: { directory: journalDir, maxBytes: 256 * 1024, segmentBytes: 64 * 1024 },
      })
      await server.start()
      const client = await SocketRpcClient.open(socketPath)
      const subscriber = await SocketRpcClient.open(socketPath)
      try {
        const init = await client.call('initialize', {
          session_id: 'aabbccddeeff',
          project_dir: root,
        })
        const sessionId = String(init.session_id)

        const sub = await subscriber.call('events.subscribe', {
          project_dirs: [root],
          detail: 'conversation',
        })

        const manager = await swarmHost.ensureSession({ id: sessionId, cwd: root, model: 'gpt-4.1' })
        await manager.add({ name: 'alpha', brief: 'Work on alpha' })
        await manager.add({ name: 'beta', brief: 'Work on beta' })

        // Force live member_activity events on alpha.
        manager.recordWorkerStreamEvent('alpha', { type: 'text', text: 'streaming compact delta' })

        const activityFrame = await client.next(
          frame =>
            frame.method === 'event'
            && frame.params?.type === 'swarm_event'
            && (frame.params?.payload as JsonRpcPayload | undefined)?.kind === 'member_activity',
        )
        const activityPayload = activityFrame.params?.payload as JsonRpcPayload
        expect(activityPayload.member_name).toBe('alpha')
        expect(activityPayload.status).toBeUndefined()
        expect(activityPayload.member).toBeDefined()
        expect(activityPayload.activity).toBeDefined()

        // Structural swarm events ARE journaled, while ephemeral member_activity is NEVER journaled!
        const page = await subscriber.call('events.read', {
          subscription_id: sub.subscription_id,
          after_cursor: sub.watermark,
          limit: 200,
        })
        const journalEvents = page.events as Array<{ readonly type: string; readonly payload: JsonRpcPayload }>
        expect(journalEvents.some(e => e.type === 'swarm_event' && e.payload.kind === 'member_added')).toBe(true)
        expect(journalEvents.some(e => e.type === 'swarm_event' && e.payload.kind === 'member_activity')).toBe(false)

        // init_done journal copy strips heavy skill_descriptions while live client still receives it.
        const initJournalEvents = journalEvents.filter(e => e.type === 'init_done')
        for (const ev of initJournalEvents) {
          expect(ev.payload).not.toHaveProperty('skill_descriptions')
        }
      } finally {
        client.close()
        subscriber.close()
        await server.stop()
      }
    })
  })

  test('EventSubscriptions filters member_activity for summary detail and enforces queue byte cap', async () => {
    await withTempDir(async (root) => {
      const journal = new TransportEventJournal({ directory: join(root, 'journal'), maxBytes: 128 * 1024 })
      await journal.start()
      const delivered: Array<{ readonly type: string; readonly payload: JsonRpcPayload }> = []
      let backlogged = true
      const subscriptions = new EventSubscriptions(journal, 2_048, () => backlogged)
      try {
        const owner = {}
        const sub = subscriptions.subscribe(owner, {
          project_dirs: [root],
          session_ids: ['sess-pinned'],
          detail: 'summary',
        }, (type, payload) => {
          delivered.push({ type, payload })
        })
        expect(subscriptions.hasSessionSubscription('sess-pinned')).toBe(true)
        expect(subscriptions.hasSessionSubscription('sess-other')).toBe(false)

        // Ephemeral member_activity is ignored by summary subscribers.
        journal.append({
          project_dir: root,
          session_id: 'sess-pinned',
          type: 'swarm_event',
          payload: { kind: 'member_activity', session_id: 'sess-pinned' },
        })
        // Flood beyond the 2 KiB queue cap while transport is backlogged -> triggers retention gap.
        for (let i = 0; i < 12; i += 1) {
          journal.append({
            project_dir: root,
            session_id: 'sess-pinned',
            type: 'tool_result',
            payload: { output: 'x'.repeat(400), seq: i },
          })
        }
        await journal.flush()
        await Bun.sleep(20)
        expect(delivered.some(d => d.type === 'event_gap' && d.payload.reason === 'retention')).toBe(true)
        expect(
          delivered
            .filter(d => d.type === 'event_batch')
            .flatMap(d => d.payload.events as Array<{ readonly payload: JsonRpcPayload }>)
            .some(e => e.payload.kind === 'member_activity'),
        ).toBe(false)

        backlogged = false
        subscriptions.unsubscribe(owner, { subscription_id: sub.subscription_id })
        expect(subscriptions.hasSessionSubscription('sess-pinned')).toBe(false)
      } finally {
        subscriptions.close()
        await journal.close()
      }
    })
  })

  test('subagent streaming deltas omit repeated static goal/rules/toolsets while boundaries retain them', async () => {
    const events: SubAgentEvent[] = []
    const manager = new SubAgentManager({
      idFactory: () => 'sub-compact-1',
      onEvent: ev => events.push(ev),
      runner: async (req) => {
        req.report.thinking('analyzing...')
        req.report.text('progress chunk')
        req.report.toolStart({ toolCallId: 'c1', name: 'ReadFile', inputs: { path: 'a.ts' } })
        req.report.toolEnd({ toolCallId: 'c1', name: 'ReadFile', permitted: true, result: 'ok' })
        return { content: 'done' }
      },
    })
    try {
      const task = await manager.spawn({
        prompt: 'G'.repeat(2_000),
        rules: ['permission:accept-all'],
        toolsets: ['ReadFile', 'EditFile'],
      })
      await manager.wait(task.id, 2_000)

      const spawnEv = daemonEventFromSubagent(events.find(e => e.type === 'spawn')!, undefined)!
      const thinkEv = daemonEventFromSubagent(events.find(e => e.type === 'thinking')!, undefined)!
      const textEv = daemonEventFromSubagent(events.find(e => e.type === 'text_burst')!, undefined)!
      const doneEv = daemonEventFromSubagent(events.find(e => e.type === 'done')!, undefined)!

      expect(spawnEv.payload.goal).toBe('G'.repeat(2_000))
      expect(spawnEv.payload.toolsets).toEqual(['ReadFile', 'EditFile'])
      expect(doneEv.payload.goal).toBe('G'.repeat(2_000))

      // High-frequency streaming deltas omit static prompt & toolset arrays.
      expect(thinkEv.payload).not.toHaveProperty('goal')
      expect(thinkEv.payload).not.toHaveProperty('toolsets')
      expect(thinkEv.payload).not.toHaveProperty('rules')
      expect(textEv.payload).not.toHaveProperty('goal')
      expect(textEv.payload).not.toHaveProperty('toolsets')
    } finally {
      await manager.close()
    }
  })

  test('InMemoryDaemonRuntime LRU-evicts idle sessions to disk while preserving pinned & active sessions', async () => {
    await withTempDir(async (root) => {
      const turnStarted = Promise.withResolvers<void>()
      const releaseTurn = Promise.withResolvers<void>()
      const runner: TurnRunner = {
        async *run(session, text): AsyncGenerator<DaemonEvent> {
          if (session.id === '111111111111') {
            turnStarted.resolve()
            await releaseTurn.promise
          }
          yield { type: 'text_part', payload: { text: `reply:${text}` } }
        },
      }
      const pinnedIds = new Set<string>(['222222222222'])
      const runtime = new InMemoryDaemonRuntime(runner, {
        currentProjectDirectory: root,
        sessionDirectory: join(root, 'sessions'),
        maxLoadedSessions: 2,
        isSessionPinned: session => pinnedIds.has(session.id),
      })

      // Session 1: active turn (must NEVER be evicted).
      await runtime.openSession('111111111111', undefined, { cwd: root })
      const inFlight = runtime.submitTurn('111111111111', 'busy', () => undefined)
      await turnStarted.promise

      // Session 2: host-pinned (e.g., active swarm / goal / ACP / terminal / subscription).
      await runtime.openSession('222222222222', undefined, { cwd: root })
      await runtime.submitTurn('222222222222', 'pinned prompt', () => undefined)

      // Session 3: idle with history -> oldest evictable candidate.
      await runtime.openSession('333333333333', undefined, { cwd: root })
      await runtime.submitTurn('333333333333', 'history preserved', () => undefined)

      // Session 4: newest idle session.
      await runtime.openSession('444444444444', undefined, { cwd: root })
      await runtime.submitTurn('444444444444', 'recent prompt', () => undefined)

      const evicted = await runtime.evictIdleSessions({ maxLoaded: 2 })
      // Session 3 and 4 are evictable while 1 (active turn) and 2 (pinned) stay in RAM.
      expect(evicted).toContain('333333333333')
      expect(runtime.sessionById('111111111111')).toBeDefined()
      expect(runtime.sessionById('222222222222')).toBeDefined()
      expect(runtime.sessionById('333333333333')).toBeUndefined()

      // Re-opening evicted session 3 transparently reloads its persisted transcript from disk!
      const rehydrated = await runtime.openSession('333333333333', undefined, { cwd: root, resume: true })
      expect(rehydrated.messages.some(m => m.content === 'reply:history preserved')).toBe(true)

      releaseTurn.resolve()
      await inFlight
      await runtime.shutdown()
    })
  })

  test('DaemonTranscriptStore progressive readHeader handles >32 KiB metadata without full load and skips covered journal reads', async () => {
    await withTempDir(async (root) => {
      const store = new DaemonTranscriptStore({
        directory: root,
        currentProjectDirectory: '/project',
      })
      const sessionId = 'abcdef123456'
      const bigMetadata = {
        title: 'Large Metadata Session',
        project_root: '/project',
        file_reads: { notes: 'R'.repeat(60_000) },
      }
      const transcript = normalizeDaemonTranscript({
        session_id: sessionId,
        cwd: '/project',
        turn_count: 2,
        updated_at: '2026-11-22T01:00:00.000Z',
        metadata: bigMetadata,
        messages: [
          { role: 'user', content: 'hello' },
          { role: 'assistant', content: 'world' },
        ],
      }, { requestedSessionKey: sessionId, currentProjectDirectory: '/project' })!

      await store.appendMessage(sessionId, { role: 'user', content: 'hello' }, 0, '/project')
      await store.appendMessage(sessionId, { role: 'assistant', content: 'world' }, 1, '/project')
      await store.save(transcript, {
        mode: 'rewrite',
        expectedGeneration: 0,
        expectedMessageCount: 0,
        coversEventLog: true,
      })

      // Progressive header read succeeds even though metadata exceeds 32 KiB!
      const headerRes = await store.readHeader(sessionId)
      expect(headerRes.kind).toBe('header')
      if (headerRes.kind === 'header') {
        expect(headerRes.header.metadata.title).toBe('Large Metadata Session')
      }

      // Listing header strips bulky non-listing metadata keys before caching.
      const entries = await store.listEntries()
      const listing = await store.readListingHeader(sessionId, entries[0])
      expect(listing.kind).toBe('header')
      if (listing.kind === 'header') {
        expect(listing.header.metadata.title).toBe('Large Metadata Session')
        expect(listing.header.metadata).not.toHaveProperty('file_reads')
      }

      // Saved snapshot covers the entire crash journal (`event_log_offset === jsonl.size`).
      const savedRaw = JSON.parse(await readFile(store.pathFor(sessionId), 'utf8')) as { event_log_offset?: number }
      const journalBytes = (await readFile(store.journalPathFor(sessionId))).byteLength
      expect(savedRaw.event_log_offset).toBe(journalBytes)

      // Loading succeeds cleanly and still replays any newly appended uncovered tail!
      await store.appendMessage(sessionId, { role: 'user', content: 'after crash' }, 2, '/project')
      const reloaded = await store.load(sessionId)
      expect(reloaded?.messages).toHaveLength(3)
      expect(reloaded?.messages[2]?.content).toBe('after crash')
    })
  })

  test('MemoryGuard skips ps subprocess when hasChildren() is false, and TerminalRegistry + TerminalMonitors stay strictly bounded', async () => {
    let listCalls = 0
    let hasChild = false
    const ports: MemoryGuardPorts = {
      hasChildren: () => hasChild,
      listProcesses: async () => {
        listCalls += 1
        return []
      },
      footprintBytes: () => 0,
      kill: () => undefined,
    }
    const guard = new MemoryGuard({
      rootPid: 1234,
      limitBytes: () => 1024 ** 3,
      ports,
    })

    // Idle tick: zero `listProcesses()` (`ps`) calls.
    expect(await guard.tick()).toEqual([])
    expect(listCalls).toBe(0)

    // Once a child appears, `listProcesses()` runs.
    hasChild = true
    expect(await guard.tick()).toEqual([])
    expect(listCalls).toBe(1)

    // TerminalRegistry TailBuffer stays strictly bounded under heavy small-chunk writes.
    const registry = new TerminalRegistry({ mirrorCapacity: 1_024, historyLimit: 5 })
    const handle = registry.open({
      id: 'pty-chatty',
      kind: 'pty',
      command: 'yes',
      cwd: '/tmp',
      ownerSessionId: 'sess-term',
    })
    expect(registry.hasRunning('sess-term')).toBe(true)
    for (let i = 0; i < 2_000; i += 1) {
      handle.append(`line-${String(i).padStart(4, '0')}\n`)
    }
    const inspected = registry.inspect('sess-term', 'pty-chatty', 2_048)!
    expect(inspected.output.length).toBeLessThanOrEqual(1_024)
    expect(inspected.outputTruncated).toBe(true)
    expect(inspected.output.endsWith('line-1999\n')).toBe(true)
    handle.close(0)
    expect(registry.hasRunning('sess-term')).toBe(false)

    // TerminalMonitors reports active watches accurately and cleans up on stop.
    await withTempDir(async (root) => {
      const history = new RunHistory(join(root, 'runs.sqlite'))
      const monitors = new TerminalMonitors(registry, history)
      try {
        const liveHandle = registry.open({
          id: 'pty-watch',
          kind: 'pty',
          command: 'tail -f',
          cwd: root,
          ownerSessionId: 'sess-mon',
        })
        const watch = monitors.start('sess-mon', { terminalId: 'pty-watch', match: 'ERR' })
        expect(monitors.hasActiveWatches('sess-mon')).toBe(true)
        monitors.stop('sess-mon', watch.id)
        expect(monitors.hasActiveWatches('sess-mon')).toBe(false)
        liveHandle.close(0)
      } finally {
        monitors.close()
        history.close()
      }
    })

    // DaemonAcpBridgeHost exposes hasActiveBridge cleanly.
    const acp = new DaemonAcpBridgeHost()
    expect(acp.hasActiveBridge('sess-none')).toBe(false)
  })

  test('SwarmManager bounds warm idle worker histories in RAM and reloads evicted workers seamlessly on wake', async () => {
    await withTempDir(async (root) => {
      const wokenHistoryLengths: number[] = []
      const swarm = await SwarmManager.open({
        sessionId: 'sess-warm-cap',
        workspaceDir: join(root, 'ws'),
        storageDir: join(root, 'swarm'),
        workerRunner: async (input) => {
          wokenHistoryLengths.push(input.messages.length)
          return {
            messages: [
              ...input.messages,
              { role: 'user', content: input.prompt },
              { role: 'assistant', content: `Reply from ${input.member.name}` },
            ],
          }
        },
      })

      // Spawn 12 workers that complete and park in 'waiting'.
      await swarm.add({ name: 'batch', n: 12, brief: 'Complete slice' })
      await swarm.waitForIdle()

      // Warm idle history cap (8) keeps RAM bounded even with 12 living idle workers.
      expect(swarm.inMemoryHistoryCount()).toBeLessThanOrEqual(8)

      // Inspecting an evicted idle worker still returns its persisted transcript.
      const firstInspect = swarm.inspectMember('batch-1')
      expect(firstInspect.total_messages).toBe(2)
      expect(firstInspect.messages.at(-1)?.text).toBe('Reply from batch-1')

      // Waking an evicted worker reloads its persisted history transparently before running turn 2!
      const beforeWakes = wokenHistoryLengths.length
      await swarm.postMessage({ caller: 'coordinator', text: '@batch-1 follow up' })
      await swarm.waitForIdle()
      expect(wokenHistoryLengths[beforeWakes]).toBe(2)

      await swarm.dispose()
    })
  })

  test('DaemonServer protects sessions with active swarms, terminals, or event subscriptions during idle LRU eviction', async () => {
    await withTempDir(async (root) => {
      const socketPath = join(root, 'daemon-pin.sock')
      const swarmHost = new DaemonSwarmHost({ storageRootDir: join(root, 'swarms') })
      const terminalRegistry = new TerminalRegistry({ mirrorCapacity: 4_096, historyLimit: 10 })
      const runtime = new InMemoryDaemonRuntime(undefined, {
        currentProjectDirectory: root,
        sessionDirectory: join(root, 'sessions'),
        maxLoadedSessions: 0,
      })
      const server = new DaemonServer({
        socketPath,
        projectDirectory: root,
        runtime,
        swarmHost,
        terminalRegistry,
        eventJournal: { directory: join(root, 'journal'), maxBytes: 128 * 1024 },
      })
      await server.start()
      const subscriber = await SocketRpcClient.open(socketPath)
      try {
        // 1. Session with an active living swarm.
        await runtime.openSession('aaaa1111aaaa', undefined, { cwd: root })
        await runtime.submitTurn('aaaa1111aaaa', 'swarm session', () => undefined)
        const swarm = await swarmHost.ensureSession({ id: 'aaaa1111aaaa', cwd: root, model: 'gpt-4.1' })
        await swarm.add({ name: 'worker-live', brief: 'Stay waiting' })

        // 2. Session with a live running terminal.
        await runtime.openSession('bbbb2222bbbb', undefined, { cwd: root })
        await runtime.submitTurn('bbbb2222bbbb', 'terminal session', () => undefined)
        const liveTerm = terminalRegistry.open({
          id: 'term-1',
          kind: 'background',
          command: 'sleep 60',
          cwd: root,
          ownerSessionId: 'bbbb2222bbbb',
        })

        // 3. Session with an explicit event subscription.
        await runtime.openSession('cccc3333cccc', undefined, { cwd: root })
        await runtime.submitTurn('cccc3333cccc', 'subscribed session', () => undefined)
        await subscriber.call('events.subscribe', {
          project_dirs: [root],
          session_ids: ['cccc3333cccc'],
          detail: 'summary',
        })

        // 4. Truly idle, unpinned session -> MUST be evicted when cap=1.
        await runtime.openSession('dddd4444dddd', undefined, { cwd: root })
        await runtime.submitTurn('dddd4444dddd', 'idle session', () => undefined)

        const evicted = await runtime.evictIdleSessions({ maxLoaded: 1 })
        expect(evicted).toEqual(['dddd4444dddd'])
        expect(runtime.sessionById('aaaa1111aaaa')).toBeDefined()
        expect(runtime.sessionById('bbbb2222bbbb')).toBeDefined()
        expect(runtime.sessionById('cccc3333cccc')).toBeDefined()
        expect(runtime.sessionById('dddd4444dddd')).toBeUndefined()

        liveTerm.close(0)
      } finally {
        subscriber.close()
        await server.stop()
      }
    })
  })

  test('SwarmBoardStore preserves strict write ordering across saveState -> appendMessagesAndSaveState -> saveState without stale overwrites', async () => {
    await withTempDir(async (root) => {
      const store = new SwarmBoardStore(join(root, 'swarm-ordering'))
      const nowIso = '2026-11-22T05:15:00.000Z'
      const makeState = (title: string, head: number): PersistedSwarmState => ({
        version: 1,
        session_id: 'sess-ordering',
        workspace_dir: root,
        title,
        coordinator_activated: true,
        coordinator_model: 'gpt-4.1',
        coordinator_model_display_name: 'gpt-4.1',
        coordinator_last_read_seq: head,
        head,
        overview: null,
        channels: [],
        members: [],
      })

      // Fire S1 -> append(msg, S2) -> S3 -> S4 back-to-back without awaiting in between:
      // S1's queued slot must be cancelled, S2 writes with msg, and S3+S4 coalesce AFTER S2
      // so S4 is guaranteed to be the final persisted state on disk (never overwritten by S2).
      const p1 = store.saveState(makeState('S1-stale', 1))
      const p2 = store.appendMessagesAndSaveState([
        {
          seq: 2,
          agent: 'coordinator',
          channel: 'main',
          text: 'hello ordered board',
          mentions: [],
          channel_mentions: [],
          broadcast: false,
          time: nowIso,
        },
      ], makeState('S2-mid', 2))
      const p3 = store.saveState(makeState('S3-coalesced', 3))
      const p4 = store.saveState(makeState('S4-final', 4))

      await Promise.all([p1, p2, p3, p4])
      await store.flush()

      const reloaded = await store.load({
        sessionId: 'sess-ordering',
        workspaceDir: root,
        coordinatorModel: 'gpt-4.1',
        coordinatorModelDisplayName: 'gpt-4.1',
        nowIso,
      })
      expect(reloaded?.state.title).toBe('S4-final')
      expect(reloaded?.state.head).toBe(4)
      expect(reloaded?.messages).toHaveLength(1)
      expect(reloaded?.messages[0]?.text).toBe('hello ordered board')
    })
  })
})
