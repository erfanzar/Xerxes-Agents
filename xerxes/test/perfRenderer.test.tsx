// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { Stream, TRANSCRIPT_TAIL_GROUPS, attachTranscriptExpandListeners } from '../src/desktop/renderer/App.js'
import { isBackdropTrackingActive, trackBackdropImage } from '../src/desktop/renderer/backdropLayers.js'
import { BlockBuilder, MAX_TOOL_DETAIL_CHARS, capToolDetail } from '../src/desktop/renderer/blocks.js'
import { createListPoll } from '../src/desktop/renderer/listPoll.js'
import {
  MARKDOWN_CACHE_LIMIT,
  clearMarkdownCache,
  markdownBlocks,
  markdownCacheSize,
} from '../src/desktop/renderer/markdown.js'
import {
  installPerfPauseListeners,
  isPerfPaused,
  pageIsInactive,
  syncPerfPaused,
} from '../src/desktop/renderer/pageVisibility.js'
import { LOG_CAP, Store, type Snapshot, type XerxesLike } from '../src/desktop/renderer/store.js'
import { nextTranscriptFollowState } from '../src/desktop/renderer/transcriptScroll.js'
import { SWARM_BOARD_TAIL_LIMIT, SwarmBoardView } from '../src/desktop/renderer/SwarmBoard.js'
import { SwarmGraph } from '../src/desktop/renderer/SwarmGraph.js'
import {
  SWARM_MESSAGE_MEMORY_CAP,
  applySwarmStatusDelta,
  mergeSwarmMessages,
  parseSwarmEvent,
  type SwarmBoardMessage,
  type SwarmMemberSnapshot,
  type SwarmStatusSnapshot,
} from '../src/desktop/renderer/swarmState.js'
import type { Block, DaemonEvent } from '../src/desktop/renderer/types.js'
import { XERXES_3D_RUNTIME_SCRIPT } from '../src/desktop/renderer/xerxes3dRuntime.js'

const RENDERER_DIR = join(import.meta.dir, '../src/desktop/renderer')

class FakeDesktopBridge implements XerxesLike {
  readonly calls: Array<{ method: string; params: Record<string, unknown> }> = []
  private handlers = new Set<(event: DaemonEvent) => void>()

  constructor(
    private readonly respond: (method: string, params: Record<string, unknown>) => Record<string, unknown> = () => ({ ok: true }),
  ) {}

  onEvent(handler: (event: DaemonEvent) => void): () => void {
    this.handlers.add(handler)
    return () => { this.handlers.delete(handler) }
  }

  push(type: string, payload: Record<string, unknown>): void {
    for (const handler of this.handlers) handler({ type, payload })
  }

  async call(method: string, params: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
    this.calls.push({ method, params })
    return this.respond(method, params)
  }

  async getWorkspace(): Promise<string> {
    return '/repo'
  }
}

function makeSnapshot(overrides: Partial<Snapshot> = {}): Snapshot {
  return {
    connection: 'online',
    cwd: '/repo',
    model: 'claude-sonnet-4-6',
    currentAgentPreset: 'default',
    agentPresets: [],
    models: [],
    reasoningLevels: [],
    contextTokens: null,
    contextMax: null,
    ttftMs: null,
    tokensPerSecond: null,
    llmDurationMs: 0,
    llmSteps: 0,
    toolDurationMs: 0,
    toolSteps: 0,
    inputTokens: 0,
    outputTokens: 0,
    metricPhase: null,
    metricPhaseStartedAt: null,
    cacheHitRate: null,
    sessions: [],
    live: [],
    fleet: [],
    skillSuggestions: [],
    creatorTrace: [],
    backgroundJobs: [],
    currentId: 'sess-perf',
    sessionOpenRevision: 1,
    currentTitle: 'Perf Audit',
    sessionKey: 'perf-key',
    branch: 'main',
    daemonWarning: null,
    costUsd: null,
    mcpStatus: {},
    goal: '',
    approval: null,
    question: null,
    interactionHidden: false,
    planMode: false,
    turnActive: false,
    turnFailed: false,
    turnSeconds: 0,
    turnStartedAt: null,
    blocks: [],
    error: null,
    tab: 'activity',
    turnCount: 0,
    queue: [],
    changes: [],
    changesKept: false,
    plan: null,
    todos: null,
    log: [],
    failed: null,
    stopArmed: false,
    settingsOpen: false,
    settingsTab: 'general',
    paletteOpen: false,
    commands: [],
    pickerOpen: false,
    wsMenuOpen: false,
    sessionMenu: null,
    streamThinking: true,
    taskModalOpen: false,
    providers: [],
    providerModels: {},
    providerModelLoading: [],
    providerModelWarnings: {},
    providerTypes: [],
    permissionMode: 'auto',
    snippets: {},
    ...overrides,
  } as unknown as Snapshot
}

function makeMember(name: string, overrides: Partial<SwarmMemberSnapshot> = {}): SwarmMemberSnapshot {
  return {
    name,
    role: name === 'coordinator' ? 'coordinator' : 'worker',
    ...(name === 'coordinator' ? {} : { added_by: 'coordinator' }),
    model: 'claude-sonnet-4-6',
    model_display_name: 'Claude Sonnet 4.6',
    workspace: 'share',
    channels: ['main'],
    state: 'waiting',
    killed: false,
    last_read_seq: 1,
    ...overrides,
  }
}

function makeMessages(count: number): SwarmBoardMessage[] {
  return Array.from({ length: count }, (_, idx) => ({
    seq: idx + 1,
    agent: idx === 0 ? 'coordinator' : `worker-${(idx % 5) + 1}`,
    channel: 'main',
    text: `Update #${idx + 1} completed cleanly.`,
    mentions: [],
    channel_mentions: [],
    broadcast: false,
    time: `2026-11-22T01:${String(idx % 60).padStart(2, '0')}:00.000Z`,
  }))
}

describe('Renderer Performance — Store Coalescing, Bounds & Structural Sharing', () => {
  const origWindow = (globalThis as { window?: unknown }).window
  const origRaf = (globalThis as { requestAnimationFrame?: unknown }).requestAnimationFrame
  const origCancelRaf = (globalThis as { cancelAnimationFrame?: unknown }).cancelAnimationFrame

  afterEach(() => {
    ;(globalThis as { window?: unknown }).window = origWindow
    ;(globalThis as { requestAnimationFrame?: unknown }).requestAnimationFrame = origRaf
    ;(globalThis as { cancelAnimationFrame?: unknown }).cancelAnimationFrame = origCancelRaf
  })

  test('rAF-coalesces rapid streaming text_part, think_part, btw_event, and member_activity deltas while flushing synchronously on turn_end', async () => {
    const rafQueue: Array<{ id: number; cb: FrameRequestCallback }> = []
    let nextRafId = 1
    ;(globalThis as { requestAnimationFrame?: unknown }).requestAnimationFrame = (cb: FrameRequestCallback) => {
      const id = nextRafId++
      rafQueue.push({ id, cb })
      return id
    }
    ;(globalThis as { cancelAnimationFrame?: unknown }).cancelAnimationFrame = (id: number) => {
      const idx = rafQueue.findIndex(item => item.id === id)
      if (idx >= 0) rafQueue.splice(idx, 1)
    }

    const bridge = new FakeDesktopBridge(method => {
      if (method === 'initialize') {
        return {
          ok: true,
          session_id: 's-perf',
          session_key: 'k-perf',
          cwd: '/repo',
          model: 'claude-sonnet-4-6',
        }
      }
      return { ok: true, sessions: [] }
    })
    ;(globalThis as { window?: unknown }).window = { xerxes: bridge }

    const testStore = new Store()
    testStore.start(bridge)
    await Bun.sleep(15)

    // Hydrate initial 57-member swarm status once.
    const initialMembers = [
      makeMember('coordinator', { role: 'coordinator', state: 'waiting' }),
      ...Array.from({ length: 56 }, (_, i) =>
        makeMember(`worker-${i + 1}`, { state: i === 0 ? 'working' : 'waiting' }),
      ),
    ]
    bridge.push('swarm_event', {
      session_id: 's-perf',
      kind: 'activated',
      head: 5,
      status: {
        active: true,
        session_id: 's-perf',
        title: '57-Member Run',
        head: 5,
        channels: [],
        members: initialMembers,
      },
    })
    bridge.push('turn_begin', { text: 'benchmark streaming' })
    // Ensure turnStartedAt is stamped (no global 1s setInterval needed).
    expect(typeof testStore.getSnapshot().turnStartedAt).toBe('number')

    // Wait past the 50ms pacing interval so the next streaming burst enters rAF immediately.
    await Bun.sleep(60)
    const beforeCalls = bridge.calls.filter(c => c.method === 'swarm.status').length
    const originalWorker2 = testStore.getSnapshot().swarm!.members.find(m => m.name === 'worker-2')

    let notifyCount = 0
    const unsub = testStore.subscribe(() => { notifyCount += 1 })

    // Burst 60 text_part deltas + 10 lightweight member_activity ticks (omitting status!).
    for (let i = 0; i < 60; i += 1) {
      bridge.push('text_part', { text: 'a' })
    }
    for (let i = 0; i < 10; i += 1) {
      bridge.push('swarm_event', {
        session_id: 's-perf',
        kind: 'member_activity',
        head: 5,
        member_name: 'worker-1',
        activity: { phase: 'tool', summary: `Reading file ${i}`, active_tools: [], recent_tools: [] },
      })
    }

    // Zero synchronous fan-out during the 70-event burst; 1 queued rAF frame.
    expect(notifyCount).toBe(0)
    expect(rafQueue.length).toBe(1)

    // Synchronous getSnapshot() still materializes the latest accumulated state accurately.
    const midSnap = testStore.getSnapshot()
    expect(midSnap.blocks.some(b => b.kind === 'agent' && b.text === 'a'.repeat(60))).toBe(true)
    expect(midSnap.swarm!.members.find(m => m.name === 'worker-1')?.current_activity).toBe('Reading file 9')
    // Structural sharing: untouched worker-2 preserved exact object reference!
    expect(midSnap.swarm!.members.find(m => m.name === 'worker-2')).toBe(originalWorker2)
    // Lightweight member_activity must NOT trigger fallback swarm.status RPCs!
    expect(bridge.calls.filter(c => c.method === 'swarm.status').length).toBe(beforeCalls)

    // Flush the single rAF -> exactly 1 listener notification for all 70 events.
    const frame = rafQueue.shift()!
    frame.cb(16)
    expect(notifyCount).toBe(1)

    // Push more tokens, then verify turn_end flushes synchronously and cancels pending frame.
    bridge.push('text_part', { text: '!' })
    bridge.push('turn_end', {})
    expect(notifyCount).toBe(2)
    expect(testStore.getSnapshot().turnActive).toBe(false)
    expect(testStore.getSnapshot().turnStartedAt).toBeNull()
    expect(testStore.getSnapshot().blocks.some(b => b.kind === 'agent' && b.text === `${'a'.repeat(60)}!` && !b.streaming)).toBe(true)
    unsub()
  })

  test('caps Store log ring at LOG_CAP (300) and caps oversized tool input/output payloads cleanly', async () => {
    const bridge = new FakeDesktopBridge(method => {
      if (method === 'initialize') return { ok: true, session_id: 's-cap', session_key: 'k-cap', cwd: '/repo', model: 'm' }
      return { ok: true, sessions: [] }
    })
    ;(globalThis as { window?: unknown }).window = { xerxes: bridge }
    ;(globalThis as { requestAnimationFrame?: unknown }).requestAnimationFrame = undefined

    const testStore = new Store()
    testStore.start(bridge)
    await Bun.sleep(10)

    for (let i = 0; i < LOG_CAP + 120; i += 1) {
      bridge.push('text_part', { text: 'x' })
    }
    const snap = testStore.getSnapshot()
    expect(snap.log.length).toBe(LOG_CAP)
    expect(snap.log[snap.log.length - 1]!.type).toBe('text_part')

    // BlockBuilder caps multi-hundred-KB tool payloads with a clear truncation notice.
    const giantOutput = 'Z'.repeat(200_000)
    const giantInput = JSON.stringify({ content: 'Y'.repeat(150_000) })
    const builder = new BlockBuilder()
    builder.push('tool_call', { id: 'big-1', name: 'ReadFile', arguments: giantInput })
    builder.push('tool_result', { tool_call_id: 'big-1', name: 'ReadFile', result: giantOutput })
    builder.finalize()
    const toolsBlock = builder.all()[0]!
    expect(toolsBlock.kind).toBe('tools')
    if (toolsBlock.kind === 'tools') {
      const item = toolsBlock.items[0]!
      expect(item.input!.length).toBeLessThan(MAX_TOOL_DETAIL_CHARS + 64)
      expect(item.input!).toContain('truncated,')
      expect(item.output!.length).toBeLessThan(MAX_TOOL_DETAIL_CHARS + 64)
      expect(item.output!).toContain('truncated, 195 KB total')
    }
    expect(capToolDetail('short')).toBe('short')
  })

  test('BlockBuilder preserves structural sharing (=== identity) for committed blocks and settled runs during streaming', () => {
    const builder = new BlockBuilder()
    builder.pushUser('first prompt')
    builder.push('think_part', { think: 'initial reasoning' })
    builder.push('tool_call', { id: 't1', name: 'ReadFile', arguments: '{"path":"a.ts"}' })
    builder.push('tool_result', { tool_call_id: 't1', name: 'ReadFile', result: 'ok' })
    builder.push('text_part', { text: 'Hello' })

    const snap1 = builder.snapshot(true)
    expect(snap1.map(b => b.kind)).toEqual(['user', 'thinking', 'tools', 'agent'])
    // Repeated read without mutation returns identical snapshot array reference.
    expect(builder.snapshot(true)).toBe(snap1)

    // Streaming another token into trailing agent run keeps user, thinking, and tools blocks `===`.
    builder.push('text_part', { text: ' world' })
    const snap2 = builder.snapshot(true)
    expect(snap2).not.toBe(snap1)
    expect(snap2[0]).toBe(snap1[0])
    expect(snap2[1]).toBe(snap1[1])
    expect(snap2[2]).toBe(snap1[2])
    expect(snap2[3]).not.toBe(snap1[3])
  })

  test('swarmState lightweight delta parser and bounded message cap', () => {
    const parsed = parseSwarmEvent({
      session_id: 's1',
      kind: 'member_activity',
      head: 12,
      member_name: 'worker-4',
      member_state: 'working',
      activity: { phase: 'tool', summary: 'Running tests', active_tools: [], recent_tools: [] },
    })
    expect(parsed?.status).toBeUndefined()
    expect(parsed?.member_name).toBe('worker-4')
    expect(parsed?.current_activity).toBe('Running tests')

    const status: SwarmStatusSnapshot = {
      active: true,
      session_id: 's1',
      title: 'Test',
      board: 'local',
      open_board: '',
      head: 10,
      overview: null,
      channels: [],
      members: [makeMember('coordinator'), makeMember('worker-4', { state: 'waiting' }), makeMember('worker-5')],
    }
    const next = applySwarmStatusDelta(status, parsed!)
    expect(next.head).toBe(12)
    expect(next.members[0]).toBe(status.members[0])
    expect(next.members[2]).toBe(status.members[2])
    expect(next.members[1]?.state).toBe('working')
    expect(next.members[1]?.current_activity).toBe('Running tests')

    // Message memory cap keeps only latest SWARM_MESSAGE_MEMORY_CAP entries.
    const merged = mergeSwarmMessages([], makeMessages(SWARM_MESSAGE_MEMORY_CAP + 40))
    expect(merged.length).toBe(SWARM_MESSAGE_MEMORY_CAP)
    expect(merged[0]!.seq).toBe(41)
    expect(merged[merged.length - 1]!.seq).toBe(SWARM_MESSAGE_MEMORY_CAP + 40)
  })
})

describe('Renderer Performance — Windowed DOM, LRU Markdown Cache & Idle SMIL Suppression', () => {
  test('Stream tail-windows long transcripts with Show earlier messages (N) and zero #RRGGBB hex literals', () => {
    const blocks: Block[] = []
    for (let i = 1; i <= TRANSCRIPT_TAIL_GROUPS + 20; i += 1) {
      blocks.push({ kind: 'user', id: i, text: `Prompt number ${i}` })
    }
    const html = renderToStaticMarkup(createElement(Stream, { snap: makeSnapshot({ blocks }) }))
    expect(html).toContain('Show earlier messages (20)')
    expect(html).toContain(`Prompt number ${TRANSCRIPT_TAIL_GROUPS + 20}`)
    expect(html).not.toContain('Prompt number 1<')
    expect(html.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])
  })

  test('attachTranscriptExpandListeners expands tail-windowed Stream on native macOS xerxes:find, Cmd/Ctrl+F, and top scroll', () => {
    const elListeners = new Map<string, Set<EventListener>>()
    const winListeners = new Map<string, Set<EventListener>>()
    const fakeEl = {
      scrollTop: 500,
      addEventListener: (type: string, fn: EventListener) => {
        const set = elListeners.get(type) ?? new Set()
        set.add(fn)
        elListeners.set(type, set)
      },
      removeEventListener: (type: string, fn: EventListener) => {
        elListeners.get(type)?.delete(fn)
      },
    }
    const fakeWin = {
      addEventListener: (type: string, fn: EventListener) => {
        const set = winListeners.get(type) ?? new Set()
        set.add(fn)
        winListeners.set(type, set)
      },
      removeEventListener: (type: string, fn: EventListener) => {
        winListeners.get(type)?.delete(fn)
      },
    }

    let expandMoreCount = 0
    let expandAllCount = 0
    const detach = attachTranscriptExpandListeners(
      fakeEl as unknown as HTMLElement,
      fakeWin as unknown as Window,
      () => { expandMoreCount += 1 },
      () => { expandAllCount += 1 },
    )

    // Scrolling below threshold does not expand.
    for (const fn of elListeners.get('scroll') ?? []) fn(new Event('scroll'))
    expect(expandMoreCount).toBe(0)

    // Scrolling near top expands another chunk.
    fakeEl.scrollTop = 80
    for (const fn of elListeners.get('scroll') ?? []) fn(new Event('scroll'))
    expect(expandMoreCount).toBe(1)

    // Native macOS Edit > Find (Cmd+F menu accelerator) dispatches `xerxes:find` without DOM keydown.
    for (const fn of winListeners.get('xerxes:find') ?? []) fn(new CustomEvent('xerxes:find'))
    expect(expandAllCount).toBe(1)

    // DOM Cmd+F / Ctrl+F keydown also expands all groups.
    for (const fn of winListeners.get('keydown') ?? []) {
      fn({ metaKey: true, ctrlKey: false, key: 'f' } as unknown as Event)
    }
    expect(expandAllCount).toBe(2)

    // Unrelated keydown does not expand.
    for (const fn of winListeners.get('keydown') ?? []) {
      fn({ metaKey: false, ctrlKey: false, key: 'f' } as unknown as Event)
    }
    expect(expandAllCount).toBe(2)

    // Detach removes all listeners cleanly.
    detach()
    expect(elListeners.get('scroll')?.size ?? 0).toBe(0)
    expect(winListeners.get('keydown')?.size ?? 0).toBe(0)
    expect(winListeners.get('xerxes:find')?.size ?? 0).toBe(0)

    // Verify both Stream (App.tsx) and SwarmBoard (SwarmBoard.tsx) wire `xerxes:find` expansion.
    const appSrc = readFileSync(join(RENDERER_DIR, 'App.tsx'), 'utf8')
    const boardSrc = readFileSync(join(RENDERER_DIR, 'SwarmBoard.tsx'), 'utf8')
    expect(appSrc).toContain('attachTranscriptExpandListeners(')
    expect(appSrc).toContain("addEventListener('xerxes:find'")
    expect(boardSrc).toContain("addEventListener('xerxes:find'")
  })

  test('SwarmBoardView tail-windows 203-post feeds with Show earlier posts (N) and renders latest 50 posts', () => {
    const messages = makeMessages(203)
    const status: SwarmStatusSnapshot = {
      active: false,
      session_id: 's-eyvan',
      title: 'Eyvan Study',
      board: 'local',
      open_board: '',
      head: 203,
      overview: null,
      channels: [{ name: 'main', topic: 'Board', created_by: 'coordinator', archived: false, subscriber_count: 57 }],
      members: [makeMember('coordinator')],
    }
    const html = renderToStaticMarkup(
      createElement(SwarmBoardView, {
        status,
        channels: status.channels,
        members: status.members,
        counts: { total: 1, working: 0, waiting: 1, stopped: 0, killed: 0, livingWorkers: 0 },
        messages,
        visibleMessages: messages,
        memoizeRows: true,
        models: [],
        online: true,
        loading: false,
        busyAction: '',
        error: '',
        activeTab: 'board',
        selectedChannel: '*',
        highlightedSeq: null,
        feedbackOpen: false,
        feedbackDraft: '',
        postDraft: '',
        switchingMember: '',
        modelDrafts: {},
        onRefresh: () => {},
        onStopAll: () => {},
        onToggleFeedback: () => {},
        onChangeFeedbackDraft: () => {},
        onSubmitFeedback: () => {},
        onCloseFeedback: () => {},
        onSelectChannel: () => {},
        onInsertMention: () => {},
        onInsertSeqRef: () => {},
        onJumpToSeq: () => {},
        onKillMember: () => {},
        onToggleModelSwitch: () => {},
        onChangeModelDraft: () => {},
        onSubmitModelSwitch: () => {},
        onCancelModelSwitch: () => {},
        onChangePostDraft: () => {},
        onSubmitPost: () => {},
      }),
    )
    expect(html).toContain(`Show earlier posts (${203 - SWARM_BOARD_TAIL_LIMIT})`)
    expect(html).toContain('id="swarm-msg-203"')
    expect(html).toContain(`id="swarm-msg-${203 - SWARM_BOARD_TAIL_LIMIT + 1}"`)
    expect(html).not.toContain('id="swarm-msg-1"')
    expect(html.match(/#[0-9a-fA-F]{6}\b/g) ?? []).toEqual([])
  })

  test('markdownBlocks caches finalized parse trees and enforces MARKDOWN_CACHE_LIMIT (256) LRU bound', () => {
    clearMarkdownCache()
    const first = markdownBlocks('Hello **world** with `code.ts`')
    const second = markdownBlocks('Hello **world** with `code.ts`')
    expect(second).toBe(first)
    expect(markdownCacheSize()).toBe(1)

    for (let i = 0; i < MARKDOWN_CACHE_LIMIT + 60; i += 1) {
      markdownBlocks(`Paragraph entry #${i} with **bold** text.`)
    }
    expect(markdownCacheSize()).toBe(MARKDOWN_CACHE_LIMIT)
    clearMarkdownCache()
  })

  test('SwarmGraph skips idle <animateMotion> SMIL loops when no workers are active and not replaying', () => {
    const messages: SwarmBoardMessage[] = [
      {
        seq: 1,
        agent: 'worker-1',
        channel: 'main',
        text: '@coordinator Completed analysis.',
        mentions: ['coordinator'],
        channel_mentions: [],
        broadcast: false,
        time: '2026-11-22 01:00:00',
      },
    ]
    const settledStatus: SwarmStatusSnapshot = {
      active: false,
      session_id: 's-idle',
      title: 'Settled Run',
      board: 'local',
      open_board: '',
      head: 1,
      overview: null,
      channels: [],
      members: [
        makeMember('coordinator', { state: 'waiting' }),
        makeMember('worker-1', { state: 'waiting' }),
      ],
    }
    const idleHtml = renderToStaticMarkup(
      createElement(SwarmGraph, { status: settledStatus, messages, initialMode: 'battlefield' }),
    )
    expect(idleHtml).not.toContain('<animateMotion')

    const activeStatus: SwarmStatusSnapshot = {
      ...settledStatus,
      active: true,
      members: [
        makeMember('coordinator', { state: 'waiting' }),
        makeMember('worker-1', { state: 'working' }),
      ],
    }
    const activeHtml = renderToStaticMarkup(
      createElement(SwarmGraph, { status: activeStatus, messages, initialMode: 'battlefield' }),
    )
    // The map draws dispatch arrows only; it runs no SMIL loop even while workers are active.
    expect(activeHtml).not.toContain('<animateMotion')
  })
})

describe('Renderer Performance — Visibility Pausing, Pollers, Backdrop & Widget Cleanup', () => {
  test('installPerfPauseListeners toggles data-perf-paused="true" and pauses/unpauses SVG SMIL timelines on blur/focus/visibility/occlusion', () => {
    const attrs = new Map<string, string>()
    let hidden = false
    let focused = true
    let pausedCount = 0
    let unpausedCount = 0
    const docListeners = new Map<string, Set<() => void>>()
    const winListeners = new Map<string, Set<() => void>>()

    const fakeSvg = {
      pauseAnimations() { pausedCount += 1 },
      unpauseAnimations() { unpausedCount += 1 },
    }
    const fakeDoc = {
      get visibilityState() { return hidden ? 'hidden' : 'visible' },
      get hidden() { return hidden },
      hasFocus: () => focused,
      documentElement: {
        getAttribute: (name: string) => attrs.get(name) ?? null,
        setAttribute: (name: string, val: string) => { attrs.set(name, val) },
        removeAttribute: (name: string) => { attrs.delete(name) },
        hasAttribute: (name: string) => attrs.has(name),
      },
      querySelectorAll: (selector: string) => (selector === 'svg' ? [fakeSvg] : []),
      addEventListener: (type: string, fn: () => void) => {
        const set = docListeners.get(type) ?? new Set()
        set.add(fn)
        docListeners.set(type, set)
      },
      removeEventListener: (type: string, fn: () => void) => { docListeners.get(type)?.delete(fn) },
    } as unknown as Document

    const fakeWin = {
      addEventListener: (type: string, fn: () => void) => {
        const set = winListeners.get(type) ?? new Set()
        set.add(fn)
        winListeners.set(type, set)
      },
      removeEventListener: (type: string, fn: () => void) => { winListeners.get(type)?.delete(fn) },
    } as unknown as Window

    const teardown = installPerfPauseListeners(fakeWin, fakeDoc)
    try {
      expect(isPerfPaused(fakeDoc)).toBe(false)
      expect(attrs.has('data-perf-paused')).toBe(false)
      expect(unpausedCount).toBe(1)

      // Blur -> paused!
      focused = false
      for (const fn of winListeners.get('blur') ?? []) fn()
      expect(attrs.get('data-perf-paused')).toBe('true')
      expect(pageIsInactive(fakeDoc)).toBe(true)
      expect(pausedCount).toBe(1)

      // Focus -> resumed!
      focused = true
      for (const fn of winListeners.get('focus') ?? []) fn()
      expect(attrs.has('data-perf-paused')).toBe(false)
      expect(unpausedCount).toBe(2)

      // Visibility hidden -> paused!
      hidden = true
      for (const fn of docListeners.get('visibilitychange') ?? []) fn()
      expect(attrs.get('data-perf-paused')).toBe('true')
      expect(pausedCount).toBe(2)

      // Visible again, then occluded -> stays paused when occluded!
      hidden = false
      attrs.set('data-occluded', '')
      expect(syncPerfPaused(fakeDoc)).toBe(true)
      expect(attrs.get('data-perf-paused')).toBe('true')
    } finally {
      teardown()
    }
  })

  test('listPoll skips polls while hidden/unfocused and catches up cleanly on resume()', async () => {
    let paused = true
    let calls = 0
    const applied: string[] = []
    const poll = createListPoll<string>(
      async key => {
        calls += 1
        return `loaded:${key}`
      },
      value => applied.push(value),
      () => {},
      { pauseWhenInactive: true, isPaused: () => paused },
    )

    // Periodic poll while paused is skipped and deferred.
    expect(poll.request('sess-1')).toBe(false)
    expect(calls).toBe(0)

    // Unpause -> resume() executes the deferred fetch once.
    paused = false
    expect(poll.resume()).toBe(true)
    await Bun.sleep(0)
    expect(calls).toBe(1)
    expect(applied).toEqual(['loaded:sess-1'])

    // Explicit forced call still works even if paused (e.g. initial open/navigation).
    paused = true
    expect(poll.request('sess-2', true)).toBe(true)
    await Bun.sleep(0)
    expect(calls).toBe(2)
    expect(applied).toEqual(['loaded:sess-1', 'loaded:sess-2'])
  })

  test('backdropLayers tears down observers when no backdrop is active and CSS pauses animations & contains offscreen rows', () => {
    trackBackdropImage(undefined)
    expect(isBackdropTrackingActive()).toBe(false)

    const appCss = readFileSync(join(RENDERER_DIR, 'app.css'), 'utf8')
    const atelierCss = readFileSync(join(RENDERER_DIR, 'atelier.css'), 'utf8')
    const layoutCss = readFileSync(join(RENDERER_DIR, 'layout.css'), 'utf8')
    expect(appCss).toContain('[data-perf-paused="true"]')
    expect(appCss).toContain('animation-play-state: paused !important')
    expect(atelierCss).toContain(":root[data-perf-paused='true']")
    expect(atelierCss).toContain('content-visibility:auto')
    expect(atelierCss).toContain(':root:not([data-backdrop]) .atelier :is(.top,.side,.desktop-rail.studio-sheet){-webkit-backdrop-filter:none;backdrop-filter:none}')
    expect(layoutCss).toContain('content-visibility:auto')
  })

  test('WidgetFrame revokes blob URLs on unmount/source toggle and Xerxes3D pauses rAF on xerxes-widget-pause', () => {
    const widgetSrc = readFileSync(join(RENDERER_DIR, 'WidgetFrame.tsx'), 'utf8')
    expect(widgetSrc).toContain('URL.createObjectURL')
    expect(widgetSrc).toContain('URL.revokeObjectURL')
    expect(widgetSrc).toContain("if (showSource)")
    expect(widgetSrc).toContain("type: 'xerxes-widget-pause'")
    expect(widgetSrc).toContain('subscribeDocumentTheme')

    // Execute XERXES_3D_RUNTIME_SCRIPT in a lightweight sandbox mock to verify rAF pausing.
    const listeners = new Map<string, Array<(ev: unknown) => void>>()
    const rafCallbacks: Array<() => void> = []
    let canceled = 0
    const mockWin: Record<string, unknown> = {
      devicePixelRatio: 1,
      addEventListener: (type: string, fn: (ev: unknown) => void) => {
        const list = listeners.get(type) ?? []
        list.push(fn)
        listeners.set(type, list)
      },
      removeEventListener: () => {},
      requestAnimationFrame: (cb: () => void) => {
        rafCallbacks.push(cb)
        return rafCallbacks.length
      },
      cancelAnimationFrame: () => { canceled += 1 },
    }
    const mockCtx = {
      save() {}, restore() {}, setTransform() {}, clearRect() {}, fillRect() {},
      beginPath() {}, moveTo() {}, lineTo() {}, closePath() {}, fill() {}, stroke() {},
      arc() {}, fillText() {}, measureText: () => ({ width: 10 }), createLinearGradient: () => ({ addColorStop() {} }),
      rect() {},
    }
    const mockCanvas = {
      width: 200,
      height: 200,
      clientWidth: 200,
      clientHeight: 200,
      style: {},
      getContext: () => mockCtx,
      addEventListener: () => {},
      removeEventListener: () => {},
    }
    const boot = new Function('window', 'document', XERXES_3D_RUNTIME_SCRIPT)
    const mockDoc = {
      hidden: false,
      documentElement: { getAttribute: () => 'dark' },
      addEventListener: () => {},
      removeEventListener: () => {},
    }
    boot(mockWin, mockDoc)
    const xerxes3d = mockWin.Xerxes3D as { createScene: (canvas: unknown, opts?: unknown) => { animate: (cb: () => void) => void; dispose: () => void } }
    const scene = xerxes3d.createScene(mockCanvas, { interactive: false })
    scene.animate(() => {})
    expect(rafCallbacks.length).toBe(1)

    // Host sends pause message -> cancels rAF and does not schedule new frames while paused.
    for (const fn of listeners.get('message') ?? []) fn({ data: { type: 'xerxes-widget-pause', paused: true } })
    expect(canceled).toBe(1)

    // Host resumes -> schedules next frame cleanly.
    for (const fn of listeners.get('message') ?? []) fn({ data: { type: 'xerxes-widget-pause', paused: false } })
    expect(rafCallbacks.length).toBe(2)

    scene.dispose()
    expect(canceled).toBe(2)
  })

  test('nextTranscriptFollowState unpins immediately on subtle upward scroll and variable-height stream/swarm rows omit content-visibility:auto', () => {
    // User is pinned at bottom (scrollTop=1000, clientHeight=600, scrollHeight=1600) and scrolls UP by just 12px.
    // Must immediately unpin so trackpad scrolling never rubber-bands back to bottom.
    const nudgedUp = nextTranscriptFollowState({
      wasFollowing: true,
      prevScrollTop: 1000,
      scrollTop: 988,
      clientHeight: 600,
      scrollHeight: 1600,
      lastFollowTop: 1000,
    })
    expect(nudgedUp.following).toBe(false)
    expect(nudgedUp.nextLastFollowTop).toBeNull()

    // Staying slightly above bottom (18px away) while NOT moving downward stays unpinned.
    const stillUp = nextTranscriptFollowState({
      wasFollowing: false,
      prevScrollTop: 982,
      scrollTop: 982,
      clientHeight: 600,
      scrollHeight: 1600,
      lastFollowTop: null,
    })
    expect(stillUp.following).toBe(false)

    // Scrolling DOWN back into the bottom 24px band re-pins cleanly.
    const scrolledDown = nextTranscriptFollowState({
      wasFollowing: false,
      prevScrollTop: 960,
      scrollTop: 990,
      clientHeight: 600,
      scrollHeight: 1600,
      lastFollowTop: null,
    })
    expect(scrolledDown.following).toBe(true)

    // Programmatic follow jump downward keeps following even if new streaming content arrived mid-frame.
    const programmaticFollow = nextTranscriptFollowState({
      wasFollowing: true,
      prevScrollTop: 950,
      scrollTop: 1000,
      clientHeight: 600,
      scrollHeight: 1700,
      lastFollowTop: 1000,
    })
    expect(programmaticFollow.following).toBe(true)

    // Non-overflowing content always stays pinned.
    const fitsViewport = nextTranscriptFollowState({
      wasFollowing: false,
      prevScrollTop: 0,
      scrollTop: 0,
      clientHeight: 600,
      scrollHeight: 500,
      lastFollowTop: null,
    })
    expect(fitsViewport.following).toBe(true)

    // macOS elastic overscroll past the bottom edge rebounding (1035 -> 1020 -> 1000) MUST stay pinned
    // and never trigger an unpin/re-pin oscillation loop.
    const overscrollRebound = nextTranscriptFollowState({
      wasFollowing: true,
      prevScrollTop: 1035,
      scrollTop: 1020,
      clientHeight: 600,
      scrollHeight: 1600,
      lastFollowTop: 1035,
    })
    expect(overscrollRebound.following).toBe(true)
    const overscrollSettled = nextTranscriptFollowState({
      wasFollowing: true,
      prevScrollTop: 1020,
      scrollTop: 1000,
      clientHeight: 600,
      scrollHeight: 1600,
      lastFollowTop: 1000,
    })
    expect(overscrollSettled.following).toBe(true)

    // .jump-latest-dock takes 0px of flow height so mounting/unmounting the jump button never alters scrollHeight.
    const appCss = readFileSync(join(RENDERER_DIR, 'app.css'), 'utf8')
    expect(appCss).toContain('.jump-latest-dock')
    expect(appCss).toContain('height: 0')

    // Variable-height transcript groups and swarm messages must NOT use content-visibility:auto
    // (causes scroll viewport teleporting when paired with overflow-anchor:none).
    const atelierCss = readFileSync(join(RENDERER_DIR, 'atelier.css'), 'utf8')
    expect(atelierCss).not.toMatch(/\.stream__group[^{]*\{[^}]*content-visibility:\s*auto/)
    expect(atelierCss).not.toMatch(/\.swarm-msg[^{]*\{[^}]*content-visibility:\s*auto/)
  })
})
