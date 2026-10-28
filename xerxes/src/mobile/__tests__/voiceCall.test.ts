// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, it, vi } from 'vitest'

import { prepareCallSpeech } from '../../client/callSpeech.js'
import { parseSupervisorMessage } from '../../client/hubParsers.js'
import { VOICE_MODELS, tierDownloadBytes, voiceModelFileUrlPath, type VoiceModelTier } from '../../client/voiceModelManifest.js'
import { fetchDecision } from '../pwa/swCore.js'
import { CallEngine, type CallOutput } from '../voice/call/engine.js'
import { CALL_SAMPLE_RATE, CallResampler, VadFrames, VadSegmenter, callFetchUrl, chooseCallTier, createCallFetch, fetchCallModelsStatus, matchCallReply, waitForCallReply, type CallReply } from '../voice/call/logic.js'
import type { CallModels } from '../voice/call/protocol.js'
import type { LiveCapture } from '../voice/liveCapture.js'

const frame = (value = 0): Float32Array => new Float32Array(512).fill(value)
const settle = async (): Promise<void> => { for (let i = 0; i < 25; i++) await Promise.resolve() }
afterEach(() => vi.useRealTimers())

describe('on-phone audio logic', () => {
  it('starts only after 250 ms above 0.5, and ends after 700 ms strictly below 0.35', () => {
    const vad = new VadSegmenter()
    for (let i = 0; i < 7; i++) expect(vad.push(frame(), 0.5, false).hearing).toBe(false)
    expect(vad.push(frame(), 0.5, false).hearing).toBe(true)
    for (let i = 0; i < 40; i++) expect(vad.push(frame(), 0.35, false).utterance).toBeUndefined()
    for (let i = 0; i < 21; i++) expect(vad.push(frame(), 0.34, false).utterance).toBeUndefined()
    expect(vad.push(frame(), 0.34, false).utterance?.length).toBeGreaterThan(0)
  })
  it('retains exactly 200 ms of pre-roll plus the entire start-confirmation window', () => {
    const vad = new VadSegmenter()
    for (let i = 0; i < 10; i++) vad.push(frame(0.1), 0.1, false)
    for (let i = 0; i < 8; i++) vad.push(frame(0.8), 0.8, false)
    let utterance: Float32Array | undefined
    for (let i = 0; i < 22; i++) utterance = vad.push(frame(), 0, false).utterance ?? utterance
    expect(utterance?.length).toBe(3200 + 8 * 512 + 22 * 512)
    expect(utterance?.[3199]).toBeCloseTo(0.1)
    expect(utterance?.[3200]).toBeCloseTo(0.8)
  })
  it('does not treat short noise or threshold flapping as a speech start', () => {
    const vad = new VadSegmenter()
    for (let i = 0; i < 100; i++) expect(vad.push(frame(), i % 8 === 7 ? 0.49 : 0.8, false).hearing).toBe(false)
  })
  it('caps an utterance at exactly 30 seconds and resets', () => {
    const vad = new VadSegmenter()
    let utterance: Float32Array | undefined
    for (let i = 0; i < 938; i++) utterance = vad.push(frame(), 0.8, false).utterance ?? utterance
    expect(utterance?.length).toBe(30 * CALL_SAMPLE_RATE)
    expect(vad.push(frame(), 0, false).hearing).toBe(false)
  })
  it('barges in only after at least 300 ms of consecutive speech while speaking', () => {
    const vad = new VadSegmenter()
    for (let i = 0; i < 9; i++) expect(vad.push(frame(), 0.8, true).bargeIn).toBe(false)
    expect(vad.push(frame(), 0.8, true).bargeIn).toBe(true)
    vad.reset()
    for (let i = 0; i < 20; i++) expect(vad.push(frame(), 0.9, false).bargeIn).toBe(false)
  })
  it('rejects invalid VAD frames and rates', () => {
    expect(() => new VadSegmenter().push(new Float32Array(10), 1, false)).toThrow('Invalid VAD frame')
    expect(() => new VadSegmenter().push(frame(), NaN, false)).toThrow('Invalid VAD frame')
    expect(() => new CallResampler(0)).toThrow('sample rate')
  })
  it.each([48_000, 44_100, 16_000, 8_000])('resamples %i Hz continuously without callback-boundary drift', rate => {
    const source = new Float32Array(rate).map((_, index) => Math.sin(index / 60))
    const whole = new CallResampler(rate).push(source)
    const streaming = new CallResampler(rate)
    const chunks: number[] = []
    for (let start = 0; start < source.length; start += 137) chunks.push(...streaming.push(source.subarray(start, start + 137)))
    expect(whole.length).toBe(16_000)
    expect(chunks).toEqual([...whole])
  })
  it('area-averages downsampled input and makes exact VAD frames across worklet chunks', () => {
    expect([...new CallResampler(48_000).push(Float32Array.from([0, 1, 2, 3, 4, 5]))]).toEqual([1, 4])
    const frames = new VadFrames()
    expect(frames.push(new Float32Array(320))).toHaveLength(0)
    expect(frames.push(new Float32Array(320))).toHaveLength(1)
    expect(frames.push(new Float32Array(384))).toHaveLength(1)
  })
  it('cleans code, markdown and URLs and bounds sentences without dropping long first sentences', () => {
    expect(prepareCallSpeech('# Hello.\n```ts\nsecret()\n```\nSee [the docs](https://example.com). https://example.com **Great!**')).toEqual(['Hello.', 'See the docs.', 'Great!'])
    expect(prepareCallSpeech('```code```')).toEqual([])
    expect(prepareCallSpeech('a'.repeat(600)).map(value => value.length)).toEqual([240, 240, 120])
    expect(prepareCallSpeech('z'.repeat(9000)).join('').length).toBe(4000)
  })
  it('chooses the low-memory tier, preserves explicit choice and uses manifest download sizes', () => {
    expect(chooseCallTier(undefined)).toBe('standard')
    expect(chooseCallTier(4)).toBe('standard')
    expect(chooseCallTier(2)).toBe('light')
    expect(chooseCallTier(8, 'light')).toBe('light')
    expect(chooseCallTier(2, 'standard')).toBe('standard')
    expect(tierDownloadBytes('standard')).toBeGreaterThan(tierDownloadBytes('light'))
  })
})

describe('same-origin model policy', () => {
  const origin = 'https://hub.example'
  it('rewrites Kokoro’s hard-coded voice URL to the exact pinned same-origin file', () => {
    expect(callFetchUrl(`https://huggingface.co/${VOICE_MODELS.tts.repo}/resolve/main/voices/af_heart.bin`, origin))
      .toBe(`${origin}${voiceModelFileUrlPath(VOICE_MODELS.tts, 'voices/af_heart.bin')}`)
    expect(callFetchUrl('/voice-ort/ort-wasm-simd-threaded.wasm', origin)).toBe(`${origin}/voice-ort/ort-wasm-simd-threaded.wasm`)
  })
  it.each(['https://evil.example/file', 'http://huggingface.co/model', 'https://huggingface.co.evil/file', 'https://user:pass@huggingface.co/file', 'data:application/wasm,aaa', 'blob:https://hub.example/id', '//evil.example/file', 'https://huggingface.co:8443/file'])('refuses external URL %s', input => {
    expect(() => callFetchUrl(input, origin)).toThrow('cross-origin')
  })
  it('does not reinterpret a same-origin double-slash pathname as a protocol-relative external URL', () => {
    expect(callFetchUrl(`${origin}//evil.example/file`, origin)).toBe(`${origin}//evil.example/file`)
  })
  it('forces credentials and redirect refusal at the actual fetch boundary', async () => {
    const native = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => new Response('ok'))
    const wrapped = createCallFetch(origin, native)
    await wrapped('/voice-ort/file.wasm', { credentials: 'include', redirect: 'follow' })
    expect(native).toHaveBeenCalledWith(`${origin}/voice-ort/file.wasm`, expect.objectContaining({ credentials: 'same-origin', redirect: 'error' }))
    await expect(wrapped('https://evil.example/file')).rejects.toThrow('cross-origin')
    expect(native).toHaveBeenCalledTimes(1)
    const abort = new AbortController()
    await wrapped(new Request(`${origin}/voice-models/file`, { method: 'HEAD', signal: abort.signal }))
    expect(native).toHaveBeenLastCalledWith(`${origin}/voice-models/file`, expect.objectContaining({ method: 'HEAD', signal: expect.any(AbortSignal) }))
  })
  it('keeps optional/unlisted metadata on the Hub to receive a real 404, never a third-party fallback', () => {
    expect(callFetchUrl(`https://huggingface.co/${VOICE_MODELS.vad.repo}/resolve/main/config.json`, origin)).toBe(`${origin}/voice-models/${VOICE_MODELS.vad.repo}/resolve/main/config.json`)
  })
  it('never lets the service worker precache/intercept model requests, even a navigation', () => {
    expect(fetchDecision({ method: 'GET', url: `${origin}/voice-models/x`, origin, mode: 'navigate', shell: new Set(['/voice-models/x']) })).toBe('network')
  })
  it.each([404, 503, 401])('gives an actionable model-status message on HTTP %i', async status => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(new Response('{}', { status }))
    await expect(fetchCallModelsStatus(fetcher)).rejects.toThrow(status === 404 ? 'does not support' : 'unavailable')
  })
  it('validates enabled and tier data at the boundary', async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(Response.json({ enabled: false }))
    await expect(fetchCallModelsStatus(fetcher)).rejects.toThrow('disabled')
    fetcher.mockResolvedValue(Response.json({ enabled: true, tiers: { standard: { bytes: -1, ready: true } } }))
    await expect(fetchCallModelsStatus(fetcher)).rejects.toThrow('invalid')
    const tiers = { standard: { bytes: 160_000_000, ready: true }, light: { bytes: 32_000_000, ready: false } }
    fetcher.mockResolvedValue(Response.json({ enabled: true, tiers }))
    await expect(fetchCallModelsStatus(fetcher)).resolves.toEqual(tiers)
  })
})

describe('supervisor reply identity', () => {
  it('preserves causal and thread IDs from the wire', () => {
    expect(parseSupervisorMessage({ id: 'reply', text: 'Hi', role: 'xerxes', thread_id: 'shared', causal_message_id: 'submit', origin: 'wake' })).toMatchObject({ threadId: 'shared', causalMessageId: 'submit', origin: 'wake' })
  })
  it('ignores history, other threads, other causal IDs and non-XERXES messages; falls back for older Hubs', () => {
    const baseline = new Set(['old'])
    const messages: CallReply[] = [
      { id: 'old', role: 'xerxes', text: '' }, { id: 'user', role: 'user', text: '' },
      { id: 'other', role: 'xerxes', text: '', causalMessageId: 'other' },
      { id: 'private', role: 'xerxes', text: '', threadId: 'autopilot' },
      { id: 'wake', role: 'xerxes', text: 'Autopilot woke', origin: 'wake' },
      { id: 'causal-wake', role: 'xerxes', text: 'Watch woke', causalMessageId: 'submit', origin: 'wake' },
      { id: 'right', role: 'xerxes', text: 'yes', causalMessageId: 'submit', threadId: 'shared' },
    ]
    expect(matchCallReply(messages, baseline, 'submit', 'shared')?.id).toBe('right')
    expect(matchCallReply([{ id: 'new', role: 'xerxes', text: 'legacy' }], baseline, 'submit')?.text).toBe('legacy')
  })
  it('does not lose a reply that arrives before the submission ACK and cleans up its observer', async () => {
    const messages: CallReply[] = []
    let listener = (): void => undefined
    const unsubscribe = vi.fn()
    const result = waitForCallReply({ messageId: 'submit', messages: () => messages, subscribe: next => { listener = next; return unsubscribe }, signal: new AbortController().signal,
      send: async () => { messages.push({ id: 'new', role: 'xerxes', text: 'Done.', causalMessageId: 'submit' }); listener(); return true },
    })
    await expect(result).resolves.toMatchObject({ text: 'Done.' })
    expect(unsubscribe).toHaveBeenCalledOnce()
  })
  it('times out in 120 seconds and does not replay the submitted message', async () => {
    vi.useFakeTimers()
    const send = vi.fn(async () => true)
    const result = waitForCallReply({ messageId: 'submit', messages: () => [], subscribe: () => () => undefined, send, signal: new AbortController().signal })
    const assertion = expect(result).rejects.toThrow('120 seconds')
    await vi.advanceTimersByTimeAsync(120_000)
    await assertion
    expect(send).toHaveBeenCalledOnce()
  })
  it('rejects failed sends and aborts without a timeout leak', async () => {
    vi.useFakeTimers()
    const signal = new AbortController()
    const input = { messageId: 'submit', messages: () => [], subscribe: () => () => undefined, send: async () => false, signal: signal.signal }
    await expect(waitForCallReply(input)).rejects.toThrow('not accepted')
    const result = waitForCallReply({ ...input, send: async () => true })
    signal.abort()
    await expect(result).rejects.toThrow('cancelled')
    expect(vi.getTimerCount()).toBe(0)
  })
})

function callFixture() {
  let listener: (samples: Float32Array) => void = () => undefined
  const close = vi.fn()
  const capture: LiveCapture = { sampleRate: 16_000, onFrame: next => { listener = next }, close }
  const model: CallModels = {
    load: vi.fn(async (tier: VoiceModelTier) => ({ tier })),
    vad: vi.fn(async samples => Number(samples[0])), resetVad: vi.fn(),
    transcribe: vi.fn(async () => 'What needs me?'),
    synthesize: vi.fn(async () => ({ samples: frame(), sampleRate: 24_000 })),
    dispose: vi.fn(),
  }
  let playbackSignal: AbortSignal | undefined
  const output: CallOutput = {
    unlock: vi.fn(async () => undefined),
    play: vi.fn(async (_audio, signal) => { playbackSignal = signal; await new Promise<void>(resolve => { signal.addEventListener('abort', () => resolve(), { once: true }) }) }),
    speak: vi.fn(async () => undefined), flush: vi.fn(), close: vi.fn(),
  }
  const reply = vi.fn(async (_text: string, _signal: AbortSignal, _onTimeout: () => void): Promise<CallReply> => ({ id: 'reply', role: 'xerxes', text: 'First sentence. Second sentence.' }))
  const engine = new CallEngine({ capture: { open: async () => capture }, models: () => model, output: () => output, reply })
  const feed = async (count: number, probability: number): Promise<void> => { for (let i = 0; i < count; i++) { listener(frame(probability)); await settle() } }
  const utterance = async (): Promise<void> => { await feed(10, 0.8); await feed(22, 0); await settle() }
  return { engine, model, output, reply, close, feed, utterance, playbackSignal: () => playbackSignal }
}

describe('call lifecycle with injected fake models (no downloads)', () => {
  it('transcribes, sends a normal message and plays sentence-by-sentence; hang-up releases all resources', async () => {
    const f = callFixture()
    await f.engine.start('standard')
    expect(f.engine.getSnapshot().phase).toBe('listening')
    await f.utterance()
    expect(f.reply).toHaveBeenCalledWith('What needs me?', expect.any(AbortSignal), expect.any(Function))
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'speaking', heard: 'What needs me?', reply: 'First sentence. Second sentence.' })
    expect(f.model.synthesize).toHaveBeenCalledTimes(1)
    f.engine.hangup()
    expect(f.close).toHaveBeenCalledOnce()
    expect(f.model.dispose).toHaveBeenCalledOnce()
    expect(f.output.close).toHaveBeenCalledOnce()
    expect(f.playbackSignal()?.aborted).toBe(true)
    expect(f.engine.getSnapshot().phase).toBe('idle')
  })
  it('barge-in aborts playback AND pending synthesis and never plays the remaining sentence', async () => {
    const f = callFixture()
    await f.engine.start('standard')
    await f.utterance()
    const before = vi.mocked(f.output.flush).mock.calls.length
    await f.feed(9, 0.9)
    expect(f.playbackSignal()?.aborted).toBe(false)
    await f.feed(1, 0.9)
    expect(f.playbackSignal()?.aborted).toBe(true)
    expect(f.output.flush).toHaveBeenCalledTimes(before + 1)
    expect(f.model.synthesize).toHaveBeenCalledTimes(1)
    expect(f.engine.getSnapshot().phase).toBe('hearing')
    f.engine.hangup()
    const pending = callFixture()
    let synthesisSignal: AbortSignal | undefined
    vi.mocked(pending.model.synthesize).mockImplementation(async (_text, signal) => {
      synthesisSignal = signal
      return new Promise((_resolve, reject) => { signal.addEventListener('abort', () => reject(new Error('cancelled')), { once: true }) })
    })
    await pending.engine.start('standard')
    await pending.utterance()
    await pending.feed(10, 0.9)
    expect(synthesisSignal?.aborted).toBe(true)
    expect(pending.output.play).not.toHaveBeenCalled()
    expect(pending.engine.getSnapshot().phase).toBe('hearing')
    pending.engine.hangup()
  })
  it('mutes processing, resumes, and uses the phone voice in Light mode', async () => {
    const f = callFixture()
    await f.engine.start('light')
    f.engine.mute(true)
    await f.utterance()
    expect(f.model.transcribe).not.toHaveBeenCalled()
    f.engine.mute(false)
    await f.utterance()
    expect(f.output.speak).toHaveBeenCalledTimes(2)
    expect(f.model.synthesize).not.toHaveBeenCalled()
    expect(f.engine.getSnapshot().phase).toBe('listening')
    f.engine.hangup()
  })
  it('displays the once-per-load Kokoro fallback notice and switches all call logic to Light', async () => {
    const f = callFixture()
    vi.mocked(f.model.load).mockResolvedValue({ tier: 'light', notice: 'Kokoro could not load. Using Light.' })
    await f.engine.start('standard')
    expect(f.engine.getSnapshot()).toMatchObject({ tier: 'light', notice: 'Kokoro could not load. Using Light.' })
    await f.utterance()
    expect(f.output.speak).toHaveBeenCalled()
    expect(f.model.synthesize).not.toHaveBeenCalled()
    f.engine.hangup()
  })
  it('releases resources on real errors and preserves the error when the connection drops', async () => {
    const f = callFixture()
    f.reply.mockRejectedValue(new Error('The message was refused.'))
    await f.engine.start('standard')
    await f.utterance()
    f.engine.connectionLost()
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'error', error: 'The message was refused.' })
    expect(f.close).toHaveBeenCalledOnce()
    expect(f.model.dispose).toHaveBeenCalledOnce()
  })
  it('survives seconds of slow TTS while independent VAD keeps answering and barges in', async () => {
    vi.useFakeTimers()
    const f = callFixture()
    let synthesisSignal: AbortSignal | undefined
    vi.mocked(f.model.synthesize).mockImplementation(async (_text, signal) => {
      synthesisSignal = signal
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => resolve({ samples: frame(), sampleRate: 24_000 }), 5000)
        signal.addEventListener('abort', () => { clearTimeout(timer); reject(new Error('cancelled')) }, { once: true })
      })
    })
    await f.engine.start('standard')
    await f.utterance()
    const before = vi.mocked(f.model.vad).mock.calls.length
    for (let i = 0; i < 100; i++) { await f.feed(1, 0); await vi.advanceTimersByTimeAsync(32) }
    expect(f.model.vad).toHaveBeenCalledTimes(before + 100)
    expect(f.engine.getSnapshot().phase).toBe('speaking')
    expect(f.model.dispose).not.toHaveBeenCalled()
    await f.feed(10, 0.9)
    expect(synthesisSignal?.aborted).toBe(true)
    expect(f.engine.getSnapshot().phase).toBe('hearing')
    expect(f.output.play).not.toHaveBeenCalled()
    f.engine.hangup()
  })
  it('drops a full VAD backlog rather than hanging up, but fails if VAD itself never answers for 10 s', async () => {
    vi.useFakeTimers()
    const f = callFixture()
    vi.mocked(f.model.vad).mockImplementation(async () => new Promise<number>(() => undefined))
    await f.engine.start('standard')
    await f.feed(100, 0)
    await vi.advanceTimersByTimeAsync(9999)
    expect(f.engine.getSnapshot().phase).toBe('listening')
    expect(f.model.dispose).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(1)
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'error', error: expect.stringContaining('VAD stopped responding') })
    expect(f.model.dispose).toHaveBeenCalledOnce()
  })
  it.each([false, true])('keeps the call loaded at reply timeout; late reply is spoken only without new speech (new speech: %s)', async newSpeech => {
    vi.useFakeTimers()
    const f = callFixture()
    const messages: CallReply[] = []
    let listener = (): void => undefined
    const unsubscribe = vi.fn()
    f.reply.mockImplementation((_text, signal, onTimeout) => waitForCallReply({
      messageId: 'submit', messages: () => messages, subscribe: next => { listener = next; return unsubscribe },
      send: async () => true, signal, onTimeout,
    }))
    await f.engine.start('light')
    await f.utterance()
    await vi.advanceTimersByTimeAsync(120_000)
    await settle()
    expect(f.engine.getSnapshot()).toMatchObject({ phase: 'listening', notice: expect.stringContaining('still working') })
    expect(f.close).not.toHaveBeenCalled()
    expect(f.model.dispose).not.toHaveBeenCalled()
    expect(f.output.speak).toHaveBeenCalledWith(expect.stringContaining('still working'), expect.any(AbortSignal))
    if (newSpeech) await f.feed(8, 0.9)
    messages.push({ id: 'late', role: 'xerxes', text: 'Late answer.', causalMessageId: 'submit' })
    listener()
    await settle()
    expect(f.output.speak).toHaveBeenCalledTimes(newSpeech ? 1 : 2)
    if (!newSpeech) expect(f.engine.getSnapshot().reply).toBe('Late answer.')
    f.engine.hangup()
    expect(unsubscribe).toHaveBeenCalledOnce()
  })
  it('hangs up after 60 seconds hidden, cancels that timer on return', async () => {
    vi.useFakeTimers()
    const f = callFixture()
    await f.engine.start('standard')
    f.engine.visibility(true)
    await vi.advanceTimersByTimeAsync(59_999)
    expect(f.close).not.toHaveBeenCalled()
    f.engine.visibility(false)
    await vi.advanceTimersByTimeAsync(60_000)
    expect(f.close).not.toHaveBeenCalled()
    f.engine.visibility(true)
    await vi.advanceTimersByTimeAsync(60_000)
    expect(f.close).toHaveBeenCalledOnce()
    expect(f.model.dispose).toHaveBeenCalledOnce()
  })
  it('does not acquire the microphone after a loading call was hung up', async () => {
    const f = callFixture()
    let ready: (() => void) | undefined
    vi.mocked(f.model.load).mockImplementation(async tier => { await new Promise<void>(resolve => { ready = resolve }); return { tier } })
    const start = f.engine.start('standard')
    await settle()
    f.engine.hangup()
    ready?.()
    await start
    expect(f.engine.getSnapshot().phase).toBe('idle')
    expect(f.model.dispose).toHaveBeenCalledOnce()
    expect(f.close).not.toHaveBeenCalled()
  })
})
