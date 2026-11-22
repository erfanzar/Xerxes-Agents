// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, it, vi } from 'vitest'

import { VoiceLink } from '../../client/voiceLink.js'
import { createVoiceStateStore } from '../../client/voiceState.js'
import { createHubHttp } from '../api.js'
import { pairingAwait, startPairingSession, type PairAction, type PairStorage } from '../pairing.js'
import { CALL_WAITING_FOR_HUB, CallEngine, type CallOutput } from '../voice/call/engine.js'
import type { CallModels } from '../voice/call/protocol.js'
import { VoiceClientCore } from '../voice/core.js'
import type { LiveCapture } from '../voice/liveCapture.js'
import { FakeVoiceHub, flush } from './voiceFakes.js'

const vadFrame = (value = 0): Float32Array => new Float32Array(512).fill(value)
const drainMicrotasks = async (): Promise<void> => {
  for (let i = 0; i < 30; i++) await Promise.resolve()
}

function memoryPairStorage(initial: Record<string, string> = {}): PairStorage {
  const map = new Map<string, string>(Object.entries(initial))
  return {
    getItem: key => map.get(key) ?? null,
    setItem: (key, value) => {
      map.set(key, value)
    },
    removeItem: key => {
      map.delete(key)
    },
  }
}

describe('Mobile PWA client/gateway surface fixes', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('CallEngine flushes deferred offline pendingSend after transient noise settles on a reconnected link', async () => {
    let onMicFrame: ((samples: Float32Array) => void) | undefined
    const capture: LiveCapture = {
      sampleRate: 16_000,
      onFrame: cb => {
        onMicFrame = cb
      },
      close: () => undefined,
    }
    const transcripts: Array<string | null> = ['Deploy staging', null]
    let vadProb = 0
    const models: CallModels = {
      load: async tier => ({ tier }),
      vad: async () => vadProb,
      resetVad: () => undefined,
      transcribe: async () => transcripts.shift() ?? null,
      synthesize: async () => ({ samples: new Float32Array(160), sampleRate: 16_000 }),
      dispose: () => undefined,
    }
    const output: CallOutput = {
      unlock: async () => undefined,
      play: async () => undefined,
      speak: async () => undefined,
      flush: () => undefined,
      close: () => undefined,
    }
    const sent: Array<{ text: string; messageId: string }> = []
    const engine = new CallEngine({
      capture: { open: async () => capture },
      models: () => models,
      output: () => output,
      reply: async (text, _signal, _onTimeout, messageId) => {
        sent.push({ text, messageId })
        return { id: 'reply-1', text: 'Deploying now.', role: 'assistant' }
      },
      messageId: () => 'msg-fixed-1',
    })

    await engine.start('standard')
    engine.setConnected(false)

    // 1. User speaks while Hub is offline -> stored in pendingSend.
    vadProb = 0.9
    for (let i = 0; i < 6; i++) {
      onMicFrame?.(vadFrame(0.3))
      await drainMicrotasks()
    }
    vadProb = 0
    for (let i = 0; i < 23; i++) {
      onMicFrame?.(vadFrame(0))
      await drainMicrotasks()
    }
    expect(sent).toEqual([])
    expect(engine.getSnapshot().notice).toBe(CALL_WAITING_FOR_HUB)

    // 2. Background noise triggers VAD `hearing` right before Hub reconnects.
    vadProb = 0.9
    for (let i = 0; i < 6; i++) {
      onMicFrame?.(vadFrame(0.2))
      await drainMicrotasks()
    }
    expect(engine.getSnapshot().phase).toBe('hearing')
    engine.setConnected(true)
    expect(sent).toEqual([])

    // 3. Noise ends with unreadable/null transcript -> settle() must flush the deferred pendingSend!
    vadProb = 0
    for (let i = 0; i < 23; i++) {
      onMicFrame?.(vadFrame(0))
      await drainMicrotasks()
    }
    await drainMicrotasks()

    expect(sent).toEqual([{ text: 'Deploy staging', messageId: 'msg-fixed-1' }])
    engine.hangup()
  })

  it('VoiceClientCore deduplicates concurrent ensureLink() calls and rejects pending waiters on dropLink()', async () => {
    const hub = new FakeVoiceHub()
    let created = 0
    const createLink = (): VoiceLink => {
      created += 1
      return new VoiceLink({ url: 'wss://hub.test/api/hub/v1/voice', socketFactory: hub.factory })
    }

    const core = new VoiceClientCore({
      createLink,
      output: undefined,
      state: createVoiceStateStore(),
    })

    const first = core.ensureLink()
    const second = core.ensureLink()
    expect(created).toBe(1)
    expect(hub.sockets).toHaveLength(1)
    expect(hub.sockets[0]!.closedWith).toBeUndefined()

    await flush()
    const [link1, link2] = await Promise.all([first, second])
    expect(link1).toBe(link2)
    expect(hub.sockets[0]!.closedWith).toBeUndefined()

    const readyWaiter = core.waitReady('utt-1', 10_000)
    core.dropLink()
    await expect(readyWaiter).rejects.toThrow('voice connection closed')
    expect(hub.sockets[0]!.closedWith).toBe(1000)
  })

  it('startPairingSession consumes a one-time pairing code only once across remounts and clears poll timers on cancel', async () => {
    vi.useFakeTimers()
    const storage = memoryPairStorage()
    const consumed = { current: false }
    const actions: PairAction[] = []
    let pairedCount = 0
    let consumeCalls = 0
    let refreshCalls = 0
    let confirmed = false

    const http = {
      consumePairing: async () => {
        consumeCalls += 1
        if (consumeCalls > 1) return { status: 'rejected' as const, message: 'Token already spent' }
        return { status: 'accepted' as const, active: false, comparisonCode: '4829' }
      },
      refresh: async () => {
        refreshCalls += 1
        return confirmed ? { status: 'ok' as const, csrfToken: 'csrf-1' } : { status: 'pending' as const }
      },
    }

    const read = {
      kind: 'found' as const,
      fragment: { pairingId: 'pair_1', code: 'one-time-secret' },
    }

    // First mount consumes the one-time ticket and enters polling, then unmounts (StrictMode / remount).
    const stopFirst = startPairingSession({
      read,
      consumed,
      http,
      deviceLabel: 'iPhone',
      storage,
      dispatch: action => actions.push(action),
      onPaired: () => {
        pairedCount += 1
      },
      pollMs: 500,
      maxWaitMs: 30_000,
      now: () => 1_000,
    })
    await drainMicrotasks()
    expect(consumeCalls).toBe(1)
    expect(consumed.current).toBe(true)
    expect(refreshCalls).toBe(1)

    stopFirst()
    expect(vi.getTimerCount()).toBe(0)

    // Second mount reuses `consumed.current === true`, skipping consumePairing and polling until confirmed.
    const stopSecond = startPairingSession({
      read,
      consumed,
      http,
      deviceLabel: 'iPhone',
      storage,
      dispatch: action => actions.push(action),
      onPaired: () => {
        pairedCount += 1
      },
      pollMs: 500,
      maxWaitMs: 30_000,
      now: () => 1_500,
    })
    await drainMicrotasks()
    expect(consumeCalls).toBe(1)

    confirmed = true
    await vi.advanceTimersByTimeAsync(500)
    await drainMicrotasks()

    expect(pairedCount).toBe(1)
    expect(actions).toContainEqual({ type: 'confirmed' })
    stopSecond()
  })

  it('pairingAwait rejects non-finite expiry values and HubHttp keepAlive re-arms on 202 pending responses', async () => {
    const storage = memoryPairStorage({ 'xerxes.pair.awaiting': '{"until":1e400,"comparison":"1234"}' })
    expect(pairingAwait(storage, 1_000)).toBeUndefined()
    expect(storage.getItem('xerxes.pair.awaiting')).toBeNull()

    let nowMs = 10_000
    const scheduled: Array<{ cb: () => void; ms: number; cleared: boolean }> = []
    const timers = {
      setTimeout(cb: () => void, ms: number) {
        const entry = { cb, ms, cleared: false }
        scheduled.push(entry)
        return entry
      },
      clearTimeout(handle: unknown) {
        if (handle && typeof handle === 'object') (handle as { cleared: boolean }).cleared = true
      },
    }

    let statusSequence = [200, 202, 200]
    const http = createHubHttp(
      async () => {
        const status = statusSequence.shift() ?? 200
        return new Response(
          JSON.stringify(status === 200 ? { csrf_token: 'tok', session_expires_at: new Date(nowMs + 100_000).toISOString() } : {}),
          { status },
        )
      },
      { now: () => nowMs, timers },
    )

    const stop = http.keepAlive(() => true)
    await http.refresh()
    const firstKeepAlive = scheduled.findLast(item => !item.cleared)
    expect(firstKeepAlive).toBeDefined()

    // Trigger keep-alive tick -> receives 202 pending -> must re-arm instead of stopping forever.
    firstKeepAlive!.cb()
    await drainMicrotasks()
    const rearmedAfterPending = scheduled.findLast(item => !item.cleared && item !== firstKeepAlive)
    expect(rearmedAfterPending?.ms).toBe(30_000)
    stop()
  })
})
