// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createServer, type Socket as NetSocket } from 'node:net'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { describe, expect, test } from 'bun:test'

import { createBridge } from '../src/desktop/bridgeApi.js'
import { DaemonRpc } from '../src/desktop/main/daemon.js'
import { createHubHealth, HUB_FOCUS_RECHECK_MS, type HubHealthPort } from '../src/desktop/main/hubHealth.js'
import { FrameDecoder } from '../src/desktop/main/hubSocket.js'
import {
  DAEMON_EVENT_BATCH_CHANNEL,
  DAEMON_EVENT_CHANNEL,
  DaemonEventBatcher,
  type DaemonEventFrame,
} from '../src/desktop/main/ipcEventBatcher.js'
import {
  applyChromiumPerformanceSwitches,
  CHROMIUM_PERFORMANCE_SWITCHES,
  createIdleCacheTrimmer,
  IDLE_VIEW_HIBERNATE_MS,
  MAX_WARM_BACKGROUND_VIEWS,
  pickEvictableBackgroundViews,
  planStartupSurfaces,
  resolveSuspendedBrowserUrl,
  setBoundedMapEntry,
  surfaceIdentityKey,
} from '../src/desktop/main/perfHygiene.js'
import { createChannelDispatcher } from '../src/desktop/main/preloadDispatcher.js'
import type { SavedWindow } from '../src/desktop/main/windowState.js'

const BOUNDS = { x: 0, y: 0, width: 1440, height: 900 }

const savedLocal = (workspace: string, sessionId: string | null, over: Partial<SavedWindow> = {}): SavedWindow => ({
  windowGroup: '1',
  active: false,
  workspace,
  sessionId,
  remote: null,
  bounds: BOUNDS,
  maximized: false,
  fullscreen: false,
  ...over,
})

const savedRemote = (alias: string, target: string, workspacePath: string, sessionId: string | null, over: Partial<SavedWindow> = {}): SavedWindow => ({
  windowGroup: '1',
  active: false,
  workspace: null,
  sessionId,
  remote: { alias, target, workspacePath },
  bounds: BOUNDS,
  maximized: false,
  fullscreen: false,
  ...over,
})

describe('Chromium performance switches', () => {
  test('registers safe background-service disable switches without overriding GPU/media features', () => {
    const applied: Array<[string, string | undefined]> = []
    applyChromiumPerformanceSwitches({
      appendSwitch(flag, value) { applied.push([flag, value]) },
    })
    expect(applied.length).toBe(CHROMIUM_PERFORMANCE_SWITCHES.length)
    const names = applied.map(([flag]) => flag)
    expect(names).toContain('disable-background-networking')
    expect(names).toContain('disable-component-update')
    expect(names).toContain('disable-domain-reliability')
    expect(names).toContain('disable-sync')
    expect(names).not.toContain('disable-gpu')
    expect(names).not.toContain('enable-features')
    expect(names).not.toContain('disable-features')
  })
})

describe('Startup surface planner (lazy multi-view restoration)', () => {
  test('deduplicates accumulated views and boots only the active surface per OS window group', () => {
    const saved: SavedWindow[] = [
      savedLocal('/repo/xerxes', 's1'),
      savedLocal('/repo/eyvan', 's2'),
      savedRemote('tpu', '34.34.5.117', '/home/erfan/src/', 'r1'),
      savedLocal('/repo/xerxes', 's3'),
      savedLocal('/repo/xerxes', 's3'), // duplicate local session
      savedRemote('tpu', '34.34.5.117', '/home/erfan/src', 'r1'), // duplicate remote session (trailing slash normalized)
      savedLocal('/repo/eyvan', 's-active', { active: true }),
    ]
    const plans = planStartupSurfaces(saved)
    expect(plans).toHaveLength(1)
    expect(plans[0]!.group).toBe('1')
    expect(plans[0]!.primary.sessionId).toBe('s-active')
    expect(plans[0]!.primary.active).toBe(true)
    // 7 raw rows -> 1 primary + 4 unique deferred (2 duplicates collapsed)
    expect(plans[0]!.deferred).toHaveLength(4)
    expect(new Set(plans[0]!.deferred.map(surfaceIdentityKey)).size).toBe(4)
  })

  test('preserves multiple OS window groups and falls back to the latest row when active is absent', () => {
    const plans = planStartupSurfaces([
      savedLocal('/repo/a', 'a1', { windowGroup: '1', active: false }),
      savedLocal('/repo/a', 'a2', { windowGroup: '1', active: false }),
      savedLocal('/repo/b', 'b1', { windowGroup: '2', active: true }),
      savedLocal('/repo/b', 'b2', { windowGroup: '2', active: false }),
    ])
    expect(plans).toHaveLength(2)
    expect(plans[0]!.primary.sessionId).toBe('a2')
    expect(plans[1]!.primary.sessionId).toBe('b1')
  })

  test('preserves active flag when an earlier duplicate was active and a later duplicate was inactive', () => {
    const plans = planStartupSurfaces([
      savedLocal('/repo/a', 'dup', { active: true }),
      savedLocal('/repo/a', 'dup', { active: false }),
      savedLocal('/repo/b', 'other', { active: false }),
    ])
    expect(plans[0]!.primary.sessionId).toBe('dup')
    expect(plans[0]!.deferred.map(d => d.sessionId)).toEqual(['other'])
  })
})

describe('Background WebContentsView eviction & idle hibernation', () => {
  test('enforces warm view cap LRU-first and hibernates long-idle views while protecting active/base/running/busy views', () => {
    const now = 1_000_000
    const victims = pickEvictableBackgroundViews([
      { id: 1, isBasePage: true, isActive: false, hasActiveTurn: false, isBusy: false, lastActiveAt: 0 },
      { id: 2, isBasePage: false, isActive: true, hasActiveTurn: false, isBusy: false, lastActiveAt: now },
      { id: 3, isBasePage: false, isActive: false, hasActiveTurn: true, isBusy: false, lastActiveAt: 0 },
      { id: 4, isBasePage: false, isActive: false, hasActiveTurn: false, isBusy: true, lastActiveAt: 0 },
      { id: 5, isBasePage: false, isActive: false, hasActiveTurn: false, isBusy: false, lastActiveAt: now - IDLE_VIEW_HIBERNATE_MS - 1 },
      { id: 6, isBasePage: false, isActive: false, hasActiveTurn: false, isBusy: false, lastActiveAt: now - 30_000 },
      { id: 7, isBasePage: false, isActive: false, hasActiveTurn: false, isBusy: false, lastActiveAt: now - 10_000 },
    ], now, { maxWarmBackgroundViews: MAX_WARM_BACKGROUND_VIEWS, idleHibernateMs: IDLE_VIEW_HIBERNATE_MS })

    // id 1 (base), id 2 (active), id 3 (running turn), id 4 (busy) are protected.
    // id 5 is past idleHibernateMs; after retiring 5, remaining background views = 4 (3,4,6,7) > cap 2, so oldest evictable (6, 7) are also retired.
    expect(victims).toEqual([5, 6, 7])
  })

  test('keeps recent background views up to maxWarmBackgroundViews', () => {
    const now = 500_000
    const victims = pickEvictableBackgroundViews([
      { id: 1, isBasePage: true, isActive: false, hasActiveTurn: false, isBusy: false, lastActiveAt: 0 },
      { id: 2, isBasePage: false, isActive: true, hasActiveTurn: false, isBusy: false, lastActiveAt: now },
      { id: 3, isBasePage: false, isActive: false, hasActiveTurn: false, isBusy: false, lastActiveAt: now - 5_000 },
      { id: 4, isBasePage: false, isActive: false, hasActiveTurn: false, isBusy: false, lastActiveAt: now - 2_000 },
    ], now, { maxWarmBackgroundViews: 2, idleHibernateMs: 600_000 })
    expect(victims).toEqual([])
  })
})

describe('Bounded Map & idle cache trimmer', () => {
  test('setBoundedMapEntry caps size and evicts oldest keys', () => {
    const map = new Map<string, string>()
    setBoundedMapEntry(map, 'a', '1', 2)
    setBoundedMapEntry(map, 'b', '2', 2)
    setBoundedMapEntry(map, 'a', '1b', 2) // refreshes 'a'
    setBoundedMapEntry(map, 'c', '3', 2) // evicts 'b'
    expect([...map.entries()]).toEqual([['a', '1b'], ['c', '3']])
  })

  test('idle cache trimmer cancels on foreground, respects cooldown, and stops on dispose', async () => {
    let clock = 100_000
    let trims = 0
    const timers: Array<{ id: number; fn: () => void; ms: number; cleared: boolean }> = []
    let nextId = 1

    const trimmer = createIdleCacheTrimmer({
      trim: () => { trims += 1 },
      idleDelayMs: 5_000,
      minIntervalMs: 30_000,
      now: () => clock,
      setTimer: (fn, ms) => {
        const entry = { id: nextId++, fn, ms, cleared: false }
        timers.push(entry)
        return entry.id as unknown as ReturnType<typeof setTimeout>
      },
      clearTimer: handle => {
        const found = timers.find(t => t.id === (handle as unknown as number))
        if (found) found.cleared = true
      },
    })

    // Background -> foreground cancels before firing
    trimmer.markBackground()
    expect(timers).toHaveLength(1)
    expect(timers[0]!.ms).toBe(5_000)
    trimmer.markForeground()
    expect(timers[0]!.cleared).toBe(true)
    expect(trims).toBe(0)

    // Background fires after delay
    trimmer.markBackground()
    clock += 5_000
    timers[1]!.fn()
    await Promise.resolve()
    expect(trims).toBe(1)

    // Immediate triggerNow within cooldown is suppressed
    clock += 10_000
    trimmer.triggerNow()
    await Promise.resolve()
    expect(trims).toBe(1)

    // After cooldown expires, triggerNow runs
    clock += 25_000
    trimmer.triggerNow()
    await Promise.resolve()
    expect(trims).toBe(2)

    // Dispose cancels pending timer
    clock += 40_000
    trimmer.markBackground()
    trimmer.dispose()
    expect(timers.at(-1)!.cleared).toBe(true)
  })

  test('idle cache trimmer guards overlapping async trims and swallows best-effort trim errors', async () => {
    let clock = 100_000
    let starts = 0
    const pending = Promise.withResolvers<void>()
    const trimmer = createIdleCacheTrimmer({
      trim: async () => {
        starts += 1
        if (starts === 1) await pending.promise
        throw new Error('transient cache lock')
      },
      minIntervalMs: 1_000,
      now: () => clock,
    })

    trimmer.triggerNow()
    clock += 5_000
    trimmer.triggerNow() // suppressed while first async trim is still in flight
    expect(starts).toBe(1)

    pending.resolve()
    await Bun.sleep(0)
    clock += 5_000
    trimmer.triggerNow()
    await Bun.sleep(0)
    expect(starts).toBe(2)
    trimmer.dispose()
  })
})

describe('DaemonEventBatcher IPC coalescing & ordering', () => {
  function harness(flushMs = 16, maxBatchSize = 128) {
    const singles: DaemonEventFrame[] = []
    const batches: Array<readonly DaemonEventFrame[]> = []
    let pendingFn: (() => void) | undefined
    const batcher = new DaemonEventBatcher({
      flushMs,
      maxBatchSize,
      sendSingle: (type, payload) => singles.push({ type, payload }),
      sendBatch: frames => batches.push(frames),
      setTimer: fn => {
        pendingFn = fn
        return 1 as unknown as ReturnType<typeof setTimeout>
      },
      clearTimer: () => {
        pendingFn = undefined
      },
    })
    return {
      batcher,
      singles,
      batches,
      tick: () => {
        const fn = pendingFn
        pendingFn = undefined
        fn?.()
      },
    }
  }

  test('coalesces consecutive same-session text_part and think_part chunks into single IPC frames', () => {
    const h = harness()
    h.batcher.push('text_part', { session_id: 's1', text: 'Hello ' })
    h.batcher.push('text_part', { session_id: 's1', text: 'world' })
    h.batcher.push('text_part', { session_id: 's1', text: '!' })
    expect(h.singles).toHaveLength(0)
    h.tick()
    expect(h.batches).toHaveLength(0)
    expect(h.singles).toEqual([{ type: 'text_part', payload: { session_id: 's1', text: 'Hello world!' } }])
  })

  test('never merges text chunks across different sessions or event types, shipping them as one batch', () => {
    const h = harness()
    h.batcher.push('text_part', { session_id: 's1', text: 'A' })
    h.batcher.push('text_part', { session_id: 's2', text: 'B' })
    h.batcher.push('think_part', { session_id: 's2', text: 'C' })
    h.batcher.push('think_part', { session_id: 's2', text: 'D' })
    h.tick()
    expect(h.singles).toHaveLength(0)
    expect(h.batches).toEqual([[
      { type: 'text_part', payload: { session_id: 's1', text: 'A' } },
      { type: 'text_part', payload: { session_id: 's2', text: 'B' } },
      { type: 'think_part', payload: { session_id: 's2', text: 'CD' } },
    ]])
  })

  test('supersedes stale member_activity frames per (session_id, member_name) while preserving any status snapshot', () => {
    const h = harness()
    h.batcher.push('swarm_event', { session_id: 's1', kind: 'member_activity', member_name: 'worker-1', head: 10, status: { members: 57 } })
    h.batcher.push('swarm_event', { session_id: 's1', kind: 'member_activity', member_name: 'worker-2', head: 11 })
    h.batcher.push('swarm_event', { session_id: 's1', kind: 'member_activity', member_name: 'worker-1', head: 12, activity: { step: 2 } })
    h.tick()
    expect(h.batches).toEqual([[
      { type: 'swarm_event', payload: { session_id: 's1', kind: 'member_activity', member_name: 'worker-1', head: 12, activity: { step: 2 }, status: { members: 57 } } },
      { type: 'swarm_event', payload: { session_id: 's1', kind: 'member_activity', member_name: 'worker-2', head: 11 } },
    ]])
  })

  test('flushes queued deltas BEFORE delivering immediate lifecycle/control events in strict FIFO order', () => {
    const h = harness()
    h.batcher.push('text_part', { session_id: 's1', text: 'Part 1 ' })
    h.batcher.push('text_part', { session_id: 's1', text: 'Part 2' })
    h.batcher.push('turn_end', { session_id: 's1', stop_reason: 'completed' })
    expect(h.singles).toEqual([
      { type: 'text_part', payload: { session_id: 's1', text: 'Part 1 Part 2' } },
      { type: 'turn_end', payload: { session_id: 's1', stop_reason: 'completed' } },
    ])
  })

  test('flushes immediately when maxBatchSize is reached and drops queued frames on dispose', () => {
    const h = harness(16, 2)
    h.batcher.push('text_part', { session_id: 's1', text: 'one' })
    h.batcher.push('text_part', { session_id: 's2', text: 'two' })
    expect(h.batches).toHaveLength(1)
    h.batcher.push('text_part', { session_id: 's1', text: 'after' })
    h.batcher.dispose()
    h.tick()
    expect(h.singles).toHaveLength(0)
    expect(h.batches).toHaveLength(1)
  })
})

describe('Preload channel dispatcher & bridge integration', () => {
  test('multiplexes subscribers onto one ipcRenderer listener per channel, unpacks daemon:event-batch, and cleans up with zero leaks', () => {
    const listeners = new Map<string, Set<(_event: unknown, value: unknown) => void>>()
    const fakeIpc = {
      on(channel: string, fn: (_event: unknown, value: unknown) => void) {
        let set = listeners.get(channel)
        if (!set) { set = new Set(); listeners.set(channel, set) }
        set.add(fn)
      },
      removeListener(channel: string, fn: (_event: unknown, value: unknown) => void) {
        listeners.get(channel)?.delete(fn)
        if (listeners.get(channel)?.size === 0) listeners.delete(channel)
      },
    }
    const emit = (channel: string, value: unknown) => {
      for (const fn of [...(listeners.get(channel) ?? [])]) fn({}, value)
    }

    const dispatcher = createChannelDispatcher(fakeIpc)
    const bridge = createBridge({
      invoke: async () => ({}),
      on: (channel, listener) => dispatcher.on(channel, listener),
      setZoomFactor: () => {},
    })

    const receivedA: unknown[] = []
    const receivedB: unknown[] = []
    const offA = bridge.onEvent((e: unknown) => receivedA.push(e))
    const offB = bridge.onEvent((e: unknown) => receivedB.push(e))
    const offThrow = bridge.onEvent(() => { throw new Error('broken listener') })

    // 3 subscribers -> only 1 daemon:event + 1 daemon:event-batch underlying binding!
    expect(listeners.get(DAEMON_EVENT_CHANNEL)?.size).toBe(1)
    expect(listeners.get(DAEMON_EVENT_BATCH_CHANNEL)?.size).toBe(1)
    expect(dispatcher.bindingCount()).toBe(2)

    // Single frame delivery
    emit(DAEMON_EVENT_CHANNEL, { type: 'turn_begin', payload: { session_id: 's1' } })
    // Batched frame delivery
    emit(DAEMON_EVENT_BATCH_CHANNEL, [
      { type: 'text_part', payload: { text: 'Hello world' } },
      { type: 'turn_end', payload: { stop_reason: 'completed' } },
    ])

    expect(receivedA).toEqual([
      { type: 'turn_begin', payload: { session_id: 's1' } },
      { type: 'text_part', payload: { text: 'Hello world' } },
      { type: 'turn_end', payload: { stop_reason: 'completed' } },
    ])
    expect(receivedB).toEqual(receivedA)

    // Unsubscribing all releases every underlying ipcRenderer listener cleanly (idempotent)
    offA()
    offA()
    expect(dispatcher.bindingCount()).toBe(2)
    offB()
    offThrow()
    expect(dispatcher.bindingCount()).toBe(0)
    expect(listeners.size).toBe(0)
  })
})

describe('DaemonRpc framing efficiency, HubSocket buffer release, & HubHealth dedup', () => {
  test('DaemonRpc handles a multi-hundred-frame burst in a single chunk cleanly and extends deadline lazily', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'xpe-'))
    const socketPath = join(dir, 'd.sock')
    const sockets: NetSocket[] = []
    const server = createServer(socket => {
      sockets.push(socket)
      socket.setEncoding('utf8')
      socket.on('error', () => {})
      socket.on('data', chunk => {
        for (const line of String(chunk).split('\n')) {
          if (!line.trim()) continue
          const req = JSON.parse(line) as { id?: number; method?: string }
          if (req.method === 'warm') {
            socket.write(`${JSON.stringify({ jsonrpc: '2.0', id: req.id, result: { ok: true } })}\n`)
          }
        }
      })
    })
    await new Promise<void>(resolve => server.listen(socketPath, resolve))
    const rpc = new DaemonRpc({ projectDir: dir, socketPath, deadlineMs: 120 })
    const events: string[] = []
    rpc.onEvent((type, payload) => events.push(`${type}:${String(payload.i ?? '')}`))
    try {
      await rpc.call('warm')
      const feed = (rpc as unknown as { onData: (chunk: string) => void }).onData.bind(rpc)
      // Burst of 250 NDJSON lines in one chunk + partial tail split across next chunk
      const lines = Array.from({ length: 250 }, (_, i) => JSON.stringify({ method: 'event', params: { type: 'text_part', payload: { i } } }))
      const splitLine = JSON.stringify({ method: 'event', params: { type: 'text_part', payload: { i: 250 } } })
      feed(`${lines.join('\n')}\n${splitLine.slice(0, 18)}`)
      feed(`${splitLine.slice(18)}\n`)
      expect(events).toHaveLength(251)
      expect(events[0]).toBe('text_part:0')
      expect(events[250]).toBe('text_part:250')
    } finally {
      rpc.dispose()
      for (const s of sockets) s.destroy()
      server.close()
      rmSync(dir, { recursive: true, force: true })
    }
  })

  test('FrameDecoder releases parent buffer reference once all bytes are consumed', () => {
    const decoder = new FrameDecoder(1024 * 1024)
    const payload = Buffer.from('hello')
    const wire = Buffer.concat([Buffer.from([0x81, payload.length]), payload])
    const frames = decoder.push(wire)
    expect(frames).toHaveLength(1)
    expect(frames[0]!.payload.toString('utf8')).toBe('hello')
    const internalBuffer = (decoder as unknown as { buffer: Buffer }).buffer
    expect(internalBuffer.byteLength).toBe(0)
    expect(internalBuffer.buffer.byteLength).toBe(0)
  })

  test('HubHealth suppresses redundant onChange broadcasts when status is unchanged on focus recheck', async () => {
    let clock = 1_000_000
    const port: HubHealthPort = {
      readHub: () => undefined,
      get: async () => ({ status: 200, body: '{}' }),
      readDismissed: () => ({ not_configured: false, unreachable: false }),
      writeDismissed: () => {},
      now: () => clock,
    }
    const health = createHubHealth(port)
    let broadcasts = 0
    health.onChange(() => { broadcasts += 1 })
    await health.check()
    expect(broadcasts).toBe(1)
    clock += HUB_FOCUS_RECHECK_MS + 1
    health.recheckOnFocus()
    await Promise.resolve()
    await Promise.resolve()
    expect(broadcasts).toBe(1)
  })
})

describe('Electron main & BrowserTab performance posture', () => {
  const desktopDir = join(import.meta.dir, '..', 'src', 'desktop')

  test('main.ts throttles covered workspace views while keeping the active window unthrottled (no macOS blackout)', () => {
    const main = readFileSync(join(desktopDir, 'main.ts'), 'utf8')
    expect(main).toContain('applyChromiumPerformanceSwitches(app.commandLine)')
    expect(main).toContain('backgroundThrottling: false')
    expect(main).toContain('setBackgroundThrottling(occluded)')
    expect(main).toContain('spellcheck: false')
    expect(main).toContain('enableWebSQL: false')
    expect(main).toContain('planStartupSurfaces(saved)')
    expect(main).toContain("powerMonitor.on('suspend'")
    expect(main).toContain("powerMonitor.on('lock-screen'")
    expect(main).toContain('createIdleCacheTrimmer')
    expect(main).not.toContain('clearStorageData')
  })

  test('browserTab.ts enables backgroundThrottling and suspends hidden WebContentsViews cleanly', () => {
    const browserTab = readFileSync(join(desktopDir, 'main', 'browserTab.ts'), 'utf8')
    expect(browserTab).toContain('backgroundThrottling: true')
    expect(browserTab).toContain('DEFAULT_BROWSER_HIDDEN_SUSPEND_MS')
    expect(browserTab).toContain('suspend(): void')
    expect(browserTab).toContain('resolveSuspendedBrowserUrl(currentUrl, this.retryUrl)')
    expect(browserTab).toContain('session.clearCache()')
  })

  test('resolveSuspendedBrowserUrl falls back to retryUrl when suspended mid-load before did-navigate commits', () => {
    expect(resolveSuspendedBrowserUrl('https://example.com/loaded', 'https://example.com/old')).toBe('https://example.com/loaded')
    expect(resolveSuspendedBrowserUrl('', 'http://localhost:8080/dashboard')).toBe('http://localhost:8080/dashboard')
    expect(resolveSuspendedBrowserUrl('about:blank', 'https://app.internal/page')).toBe('https://app.internal/page')
    expect(resolveSuspendedBrowserUrl('file:///etc/passwd', 'javascript:alert(1)')).toBe('')
  })
})
