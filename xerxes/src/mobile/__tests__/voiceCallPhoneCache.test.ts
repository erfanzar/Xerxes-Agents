// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it, vi } from 'vitest'

import { tierDownloadBytes, VOICE_MODEL_TIERS, VOICE_MODELS, voiceModelFileUrlPath, type VoiceModelTier } from '../../client/voiceModelManifest.js'
import { cachedTierBytes, KOKORO_VOICES_CACHE_NAME, TRANSFORMERS_CACHE_NAME, type PhoneCacheStorage } from '../voice/call/phoneCache.js'
import { voiceStateLine } from '../voice/call/presentation.js'
import { VoiceCallService, type VoiceCallSnapshot } from '../voice/call/service.js'
import { voiceServiceFixture } from './voiceCallServiceFakes.js'

const ORIGIN = 'https://hub.example'
const KOKORO_VOICE = 'https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/'

function tierKeys(tier: VoiceModelTier): { key: string; size: number }[] {
  return VOICE_MODEL_TIERS[tier].flatMap(id => VOICE_MODELS[id].files.map(file => ({
    key: file.path.startsWith('voices/') ? `${KOKORO_VOICES_CACHE_NAME}|${KOKORO_VOICE}${file.path}` : `${TRANSFORMERS_CACHE_NAME}|${ORIGIN}${voiceModelFileUrlPath(VOICE_MODELS[id], file.path)}`,
    size: file.size,
  })))
}
function fakeCaches(keys: readonly string[]): PhoneCacheStorage {
  const stored = new Set(keys)
  return { match: vi.fn(async (request: RequestInfo | URL, options?: MultiCacheQueryOptions) => stored.has(`${options?.cacheName}|${String(request)}`) ? new Response('x') : undefined) }
}

describe('cachedTierBytes', () => {
  it('reports every byte cached when all tier files are in the browser caches', async () => {
    const keys = tierKeys('standard').map(entry => entry.key)
    expect(await cachedTierBytes('standard', ORIGIN, fakeCaches(keys))).toEqual({ cachedBytes: tierDownloadBytes('standard'), totalBytes: tierDownloadBytes('standard') })
  })
  it('sums only the cached files when some are missing', async () => {
    const entries = tierKeys('light')
    const some = entries.slice(0, 2)
    const expected = some.reduce((sum, entry) => sum + entry.size, 0)
    expect(await cachedTierBytes('light', `${ORIGIN}/`, fakeCaches(some.map(entry => entry.key)))).toEqual({ cachedBytes: expected, totalBytes: tierDownloadBytes('light') })
  })
  it('reports nothing cached for an empty cache, a missing CacheStorage or a failing lookup', async () => {
    expect(await cachedTierBytes('standard', ORIGIN, fakeCaches([]))).toEqual({ cachedBytes: 0, totalBytes: tierDownloadBytes('standard') })
    expect(await cachedTierBytes('standard', ORIGIN, undefined)).toEqual({ cachedBytes: 0, totalBytes: tierDownloadBytes('standard') })
    const failing: PhoneCacheStorage = { match: async () => { throw new Error('SecurityError') } }
    expect(await cachedTierBytes('light', ORIGIN, failing)).toEqual({ cachedBytes: 0, totalBytes: tierDownloadBytes('light') })
  })
})

function loadingState(phoneCache: VoiceCallSnapshot['phoneCache'], loaded: number): VoiceCallSnapshot {
  const total = 200_000_000
  return { view: 'voice', tier: 'standard', connected: true, checking: false, downloadNoticeSeen: true, debugCaptureEnabled: false,
    call: { phase: 'loading', tier: 'standard', muted: false, heard: '', reply: '', progress: { loaded, total } }, ...(phoneCache ? { phoneCache } : {}) }
}

describe('loading state line', () => {
  it('says Starting voice when the whole tier is cached', () => {
    expect(voiceStateLine(loadingState({ cachedBytes: 200_000_000, totalBytes: 200_000_000 }, 150_000_000))).toBe('Starting voice')
  })
  it('counts only the uncached remainder for a partial cache', () => {
    expect(voiceStateLine(loadingState({ cachedBytes: 150_000_000, totalBytes: 200_000_000 }, 100_000_000))).toBe('Downloading voice (0 MB of 50 MB)')
    expect(voiceStateLine(loadingState({ cachedBytes: 150_000_000, totalBytes: 200_000_000 }, 170_000_000))).toBe('Downloading voice (20 MB of 50 MB)')
    expect(voiceStateLine(loadingState({ cachedBytes: 150_000_000, totalBytes: 200_000_000 }, 400_000_000))).toBe('Downloading voice (50 MB of 50 MB)')
  })
  it('shows the full download for an empty or unknown cache', () => {
    expect(voiceStateLine(loadingState({ cachedBytes: 0, totalBytes: 200_000_000 }, 80_000_000))).toBe('Downloading voice (80 MB of 200 MB)')
    expect(voiceStateLine(loadingState(undefined, 80_000_000))).toBe('Downloading voice (80 MB of 200 MB)')
  })
})

describe('service phone cache and persistent storage', () => {
  function service(extra: Partial<ConstructorParameters<typeof VoiceCallService>[0]>) {
    const f = voiceServiceFixture()
    const s = new VoiceCallService({ engine: f.createEngine, modelsStatus: f.modelsStatus, canStart: () => true, ...extra })
    return { f, s }
  }
  it('fills phoneCache on loadStatus, refreshes it on setTier and after a call ends', async () => {
    const phoneCache = vi.fn(async (tier: VoiceModelTier) => ({ cachedBytes: tier === 'light' ? 5 : 7, totalBytes: tierDownloadBytes(tier) }))
    const { f, s } = service({ phoneCache })
    await s.loadStatus()
    expect(s.getSnapshot().phoneCache).toEqual({ cachedBytes: 7, totalBytes: tierDownloadBytes('standard') })
    s.setTier('light')
    await vi.waitFor(() => expect(s.getSnapshot().phoneCache).toEqual({ cachedBytes: 5, totalBytes: tierDownloadBytes('light') }))
    await s.start()
    const calls = phoneCache.mock.calls.length
    f.engine.hangup()
    await vi.waitFor(() => expect(phoneCache.mock.calls.length).toBe(calls + 1))
    s.dispose()
  })
  it('requests persistent storage once per lifetime without delaying the call', async () => {
    let grant: (value: boolean) => void = () => undefined
    const persistStorage = vi.fn(() => new Promise<boolean>(resolve => { grant = resolve }))
    const { f, s } = service({ persistStorage })
    await s.loadStatus()
    await s.start()
    expect(f.engine.start).toHaveBeenCalledOnce()
    expect(s.getSnapshot().storagePersisted).toBeUndefined()
    grant(true)
    await vi.waitFor(() => expect(s.getSnapshot().storagePersisted).toBe('granted'))
    f.engine.hangup()
    await s.start()
    expect(persistStorage).toHaveBeenCalledOnce()
    s.dispose()
  })
  it('reports unsupported when persist is missing or rejects, and still starts the call', async () => {
    const missing = service({})
    await missing.s.loadStatus()
    await missing.s.start()
    expect(missing.s.getSnapshot()).toMatchObject({ storagePersisted: 'unsupported', call: { phase: 'listening' } })
    missing.s.dispose()
    const rejecting = service({ persistStorage: vi.fn(async () => { throw new Error('denied by policy') }) })
    await rejecting.s.loadStatus()
    await rejecting.s.start()
    expect(rejecting.s.getSnapshot().call.phase).toBe('listening')
    expect(rejecting.s.getSnapshot().error).toBeUndefined()
    await vi.waitFor(() => expect(rejecting.s.getSnapshot().storagePersisted).toBe('unsupported'))
    rejecting.s.dispose()
  })
})
