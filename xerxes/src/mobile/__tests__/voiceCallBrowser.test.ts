// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, it, vi } from 'vitest'

import { createHubStore } from '../../client/hubStore.js'
import { browserCallModels, browserCallOutput, browserVoiceCallService, SPEECH_RESTART_TIMEOUT_MS } from '../voice/call/browser.js'
import { liveState, stubController } from './helpers.js'
import type { CallWorkerRequest, CallWorkerResponse } from '../voice/call/protocol.js'
import { browserLiveCaptureDeps } from '../voice/liveCapture.js'

class FakeWorker {
  static current: FakeWorker
  static instances: FakeWorker[] = []
  readonly messages: CallWorkerRequest[] = []
  readonly transfers: Transferable[][] = []
  onmessage: ((event: MessageEvent<CallWorkerResponse>) => void) | undefined
  onerror: ((event: ErrorEvent) => void) | undefined
  terminate = vi.fn()
  constructor(readonly url: string, readonly options: WorkerOptions) { FakeWorker.current = this; FakeWorker.instances.push(this) }
  postMessage(request: CallWorkerRequest, transfers: Transferable[] = []): void { this.messages.push(request); this.transfers.push(transfers) }
  reply(message: CallWorkerResponse): void { this.onmessage?.(new MessageEvent('message', { data: message })) }
}
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); FakeWorker.instances = [] })

describe('call worker browser adapter', () => {
  it('loads only on demand, transfers audio, aborts pending synthesis and disposes before terminating', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('Worker', FakeWorker)
    const model = browserCallModels()
    const worker = FakeWorker.current
    const vadWorker = FakeWorker.instances[0]!
    expect(vadWorker.url).toBe('/voice-vad-worker.js')
    expect(worker.url).toBe('/voice-call-worker.js')
    expect(worker.options.type).toBe('module')
    expect(worker.messages).toHaveLength(0)
    const abort = new AbortController()
    const progress = vi.fn()
    const loading = model.load('standard', progress, abort.signal)
    const vadLoad = vadWorker.messages.at(-1)!
    vadWorker.reply({ id: vadLoad.id, type: 'loaded', tier: 'standard' })
    await Promise.resolve()
    await Promise.resolve()
    const load = worker.messages.at(-1)!
    worker.reply({ id: load.id, type: 'progress', progress: { loaded: 100, total: 200 } })
    worker.reply({ id: load.id, type: 'loaded', tier: 'standard' })
    await expect(loading).resolves.toEqual({ tier: 'standard' })
    expect(progress).toHaveBeenCalledWith({ loaded: 100, total: 200 })
    const frame = new Float32Array(512)
    const probability = model.vad(frame, abort.signal)
    const vad = vadWorker.messages.at(-1)!
    expect(vadWorker.transfers.at(-1)).toHaveLength(1)
    expect('samples' in vad && vad.samples).not.toBe(frame)
    vadWorker.reply({ id: vad.id, type: 'probability', probability: 0.8 })
    await expect(probability).resolves.toBe(0.8)
    const synthesis = model.synthesize('Hi.', abort.signal)
    const synth = worker.messages.at(-1)!
    abort.abort()
    await expect(synthesis).rejects.toThrow('cancelled')
    expect(worker.messages.at(-1)).toEqual({ type: 'cancel', id: synth.id })
    worker.reply({ id: synth.id, type: 'audio', audio: { samples: frame, sampleRate: 24_000 } })
    model.resetVad()
    const reset = vadWorker.messages.at(-1)!
    model.dispose()
    const vadDispose = vadWorker.messages.at(-1)!
    const dispose = worker.messages.at(-1)!
    expect(dispose.type).toBe('dispose')
    expect(worker.terminate).not.toHaveBeenCalled()
    // A late reset ACK must not be mistaken for the disposal ACK.
    vadWorker.reply({ id: reset.id, type: 'done' })
    expect(worker.terminate).not.toHaveBeenCalled()
    expect(vadWorker.terminate).not.toHaveBeenCalled()
    worker.reply({ id: dispose.id, type: 'done' })
    vadWorker.reply({ id: vadDispose.id, type: 'done' })
    expect(worker.terminate).toHaveBeenCalledOnce()
    expect(vadWorker.terminate).toHaveBeenCalledOnce()
    expect(vi.getTimerCount()).toBe(0)
  })
  it('forcibly frees a worker whose model initialization never acknowledges disposal', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('Worker', FakeWorker)
    const model = browserCallModels()
    const loading = model.load('light', () => undefined, new AbortController().signal)
    model.dispose()
    await expect(loading).rejects.toThrow('ended')
    await vi.advanceTimersByTimeAsync(2000)
    expect(FakeWorker.current.terminate).toHaveBeenCalledOnce()
  })
  it('recovers failed GPU model loading in a fresh WASM worker without reloading VAD', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('Worker', FakeWorker)
    const model = browserCallModels()
    const vad = FakeWorker.instances[0]!
    const first = FakeWorker.current
    const loading = model.load('light', () => undefined, new AbortController().signal)
    vad.reply({ id: vad.messages.at(-1)!.id, type: 'loaded', tier: 'light' })
    await Promise.resolve()
    await Promise.resolve()
    first.reply({ id: first.messages.at(-1)!.id, type: 'retry-wasm' })
    await Promise.resolve()
    await Promise.resolve()
    const fallback = FakeWorker.current
    expect(fallback.messages.at(-1)).toMatchObject({ type: 'load', forceWasm: true })
    fallback.reply({ id: fallback.messages.at(-1)!.id, type: 'loaded', tier: 'light' })
    await expect(loading).resolves.toMatchObject({ tier: 'light' })
    expect(vad.messages.filter(message => message.type === 'load')).toHaveLength(1)
    model.dispose()
    await vi.advanceTimersByTimeAsync(2000)
  })
  it.each([false, true])('retries GPU transcription in a fresh WASM worker exactly once (retry failure: %s)', async retryFailure => {
    vi.useFakeTimers()
    vi.stubGlobal('Worker', FakeWorker)
    const model = browserCallModels()
    const first = FakeWorker.current
    const samples = new Float32Array(512)
    const transcription = model.transcribe(samples, new AbortController().signal)
    const assertion = retryFailure ? expect(transcription).rejects.toThrow('WASM retry') : expect(transcription).resolves.toBe('Recovered.')
    const request = first.messages.at(-1)!
    expect('samples' in request && request.samples).not.toBe(samples)
    first.reply({ id: request.id, type: 'retry-wasm' })
    await Promise.resolve()
    await Promise.resolve()
    const fallback = FakeWorker.current
    expect(fallback).not.toBe(first)
    expect(fallback.messages.at(-1)).toMatchObject({ type: 'load', forceWasm: true })
    fallback.reply({ id: fallback.messages.at(-1)!.id, type: 'loaded', tier: 'standard' })
    await Promise.resolve()
    await Promise.resolve()
    const retry = fallback.messages.at(-1)!
    expect(retry.type).toBe('transcribe')
    expect('samples' in retry && retry.samples).toBe(samples)
    fallback.reply(retryFailure ? { id: retry.id, type: 'retry-wasm' } : { id: retry.id, type: 'transcript', text: 'Recovered.' })
    await assertion
    expect(FakeWorker.instances).toHaveLength(3) // VAD stays intact; only the speech worker is replaced
    model.dispose()
    await vi.advanceTimersByTimeAsync(2000)
    for (const worker of FakeWorker.instances) expect(worker.terminate).toHaveBeenCalledOnce()
  })
  it('makes worker startup failures observable to the caller', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('Worker', FakeWorker)
    const model = browserCallModels()
    const loading = model.load('light', () => undefined, new AbortController().signal)
    FakeWorker.instances[0]!.onerror?.({ preventDefault() {} } as ErrorEvent)
    await expect(loading).rejects.toThrow('worker stopped')
    model.dispose()
    await vi.advanceTimersByTimeAsync(2000)
  })
})

const flushWorkerJobs = async (): Promise<void> => { for (let i = 0; i < 12; i++) await Promise.resolve() }

async function loadedSpeechFixture() {
  vi.useFakeTimers()
  vi.setSystemTime(0)
  vi.stubGlobal('Worker', FakeWorker)
  const model = browserCallModels()
  const original = FakeWorker.current
  const vad = FakeWorker.instances[0]!
  const loading = model.load('standard', () => undefined, new AbortController().signal)
  vad.reply({ id: vad.messages.at(-1)!.id, type: 'loaded', tier: 'standard' })
  await flushWorkerJobs()
  original.reply({ id: original.messages.at(-1)!.id, type: 'loaded', tier: 'standard' })
  await loading
  return { model, original, vad }
}

async function recoverDroppedClip(model: ReturnType<typeof browserCallModels>): Promise<FakeWorker> {
  const previous = FakeWorker.current
  const pending = model.transcribe(new Float32Array(64_000), new AbortController().signal)
  previous.reply({ id: previous.messages.at(-1)!.id, type: 'restart-speech' })
  await flushWorkerJobs()
  const fresh = FakeWorker.current
  fresh.reply({ id: fresh.messages.at(-1)!.id, type: 'loaded', tier: 'standard' })
  await expect(pending).resolves.toBeNull()
  return fresh
}

describe('poisoned WASM speech-runtime recovery', () => {
  it('drops a failed mid-call utterance, terminates the poisoned worker immediately and reloads both STT/Kokoro, preserving VAD and speech output', async () => {
    const f = await loadedSpeechFixture()
    const restarting = vi.fn()
    const progress = vi.fn()
    const pending = f.model.transcribe(new Float32Array(64_000), new AbortController().signal, { restarting, progress })
    f.original.reply({ id: f.original.messages.at(-1)!.id, type: 'restart-speech' })
    await flushWorkerJobs()
    const fresh = FakeWorker.current
    expect(fresh).not.toBe(f.original)
    expect(f.original.terminate).toHaveBeenCalledOnce()
    expect(f.vad.terminate).not.toHaveBeenCalled()
    expect(restarting).toHaveBeenCalledOnce()
    expect(fresh.messages).toEqual([{ id: 1, type: 'load', tier: 'standard', forceWasm: true }])
    fresh.reply({ id: 1, type: 'progress', progress: { loaded: 12, total: 20 } })
    expect(progress).toHaveBeenCalledWith({ loaded: 12, total: 20 })
    fresh.reply({ id: 1, type: 'loaded', tier: 'standard' })
    await expect(pending).resolves.toBeNull()
    expect(fresh.messages.some(message => message.type === 'transcribe')).toBe(false)
    const audio = { samples: new Float32Array(240), sampleRate: 24000 }
    const spoken = f.model.synthesize('Speech still works.', new AbortController().signal)
    fresh.reply({ id: fresh.messages.at(-1)!.id, type: 'audio', audio })
    await expect(spoken).resolves.toEqual(audio)
    const next = f.model.transcribe(new Float32Array(16_000), new AbortController().signal)
    fresh.reply({ id: fresh.messages.at(-1)!.id, type: 'transcript', text: 'The next utterance works.' })
    await expect(next).resolves.toBe('The next utterance works.')
    expect(f.vad.messages.filter(message => message.type === 'load')).toHaveLength(1)
    f.model.dispose()
    await vi.advanceTimersByTimeAsync(2000)
    for (const worker of FakeWorker.instances) expect(worker.terminate).toHaveBeenCalledOnce()
    expect(vi.getTimerCount()).toBe(0)
  })
  it('stops on the third restart request within 60 seconds instead of allocating more workers', async () => {
    const f = await loadedSpeechFixture()
    await recoverDroppedClip(f.model)
    await recoverDroppedClip(f.model)
    const current = FakeWorker.current
    const pending = f.model.transcribe(new Float32Array(64_000), new AbortController().signal)
    const assertion = expect(pending).rejects.toThrow('3 restarts within 60 seconds')
    current.reply({ id: current.messages.at(-1)!.id, type: 'restart-speech' })
    await assertion
    expect(FakeWorker.instances).toHaveLength(4) // one VAD + original speech + two fresh speech workers
    expect(current.terminate).toHaveBeenCalledOnce()
    f.model.dispose()
    await vi.advanceTimersByTimeAsync(2000)
    for (const worker of FakeWorker.instances) expect(worker.terminate).toHaveBeenCalledOnce()
  })
  it('expires old restarts in a rolling 60-second window', async () => {
    const f = await loadedSpeechFixture()
    await recoverDroppedClip(f.model)
    await recoverDroppedClip(f.model)
    await vi.advanceTimersByTimeAsync(60_000)
    await recoverDroppedClip(f.model)
    expect(FakeWorker.instances).toHaveLength(5)
    f.model.dispose()
    await vi.advanceTimersByTimeAsync(2000)
  })
  it.each([false, true])('hang-up/cancellation during a restart kills the replacement and ignores late load/transcript responses (dispose: %s)', async dispose => {
    const f = await loadedSpeechFixture()
    const abort = new AbortController()
    const progress = vi.fn()
    const pending = f.model.transcribe(new Float32Array(64_000), abort.signal, { restarting: vi.fn(), progress })
    const assertion = expect(pending).rejects.toThrow(/Call (?:ended|cancelled)/u)
    f.original.reply({ id: f.original.messages.at(-1)!.id, type: 'restart-speech' })
    await flushWorkerJobs()
    const fresh = FakeWorker.current
    const late = fresh.onmessage
    const loadId = fresh.messages.at(-1)!.id
    if (dispose) f.model.dispose()
    else abort.abort()
    await assertion
    expect(fresh.terminate).toHaveBeenCalledOnce()
    late?.(new MessageEvent('message', { data: { id: loadId, type: 'progress', progress: { loaded: 20, total: 20 } } }))
    late?.(new MessageEvent('message', { data: { id: loadId, type: 'loaded', tier: 'standard' } }))
    late?.(new MessageEvent('message', { data: { id: loadId, type: 'transcript', text: 'Stale' } }))
    expect(progress).not.toHaveBeenCalled()
    expect(FakeWorker.instances).toHaveLength(3)
    f.model.dispose()
    await vi.advanceTimersByTimeAsync(2000)
    for (const worker of FakeWorker.instances) expect(worker.terminate).toHaveBeenCalledOnce()
    expect(vi.getTimerCount()).toBe(0)
  })
  it('ignores stale transcripts from the old worker even when request IDs collide with the replacement', async () => {
    const f = await loadedSpeechFixture()
    const oldHandler = f.original.onmessage
    const fresh = await recoverDroppedClip(f.model)
    const pending = f.model.transcribe(new Float32Array(16_000), new AbortController().signal)
    const id = fresh.messages.at(-1)!.id
    let settled = false
    void pending.then(() => { settled = true })
    oldHandler?.(new MessageEvent('message', { data: { id, type: 'transcript', text: 'Wrong worker, same ID.' } }))
    await flushWorkerJobs()
    expect(settled).toBe(false)
    fresh.reply({ id, type: 'transcript', text: 'Fresh worker transcript.' })
    await expect(pending).resolves.toBe('Fresh worker transcript.')
    f.model.dispose()
    await vi.advanceTimersByTimeAsync(2000)
  })
  it('fails clearly and kills a replacement whose reload never finishes', async () => {
    const f = await loadedSpeechFixture()
    const pending = f.model.transcribe(new Float32Array(64_000), new AbortController().signal)
    const assertion = expect(pending).rejects.toThrow('too long to restart. Choose Light mode')
    f.original.reply({ id: f.original.messages.at(-1)!.id, type: 'restart-speech' })
    await flushWorkerJobs()
    const fresh = FakeWorker.current
    await vi.advanceTimersByTimeAsync(SPEECH_RESTART_TIMEOUT_MS)
    await assertion
    expect(fresh.terminate).toHaveBeenCalledOnce()
    f.model.dispose()
    await vi.advanceTimersByTimeAsync(2000)
    expect(vi.getTimerCount()).toBe(0)
  })
  it('does not silently switch to Light if Kokoro fails to reload mid-call', async () => {
    const f = await loadedSpeechFixture()
    const pending = f.model.transcribe(new Float32Array(64_000), new AbortController().signal)
    const assertion = expect(pending).rejects.toThrow('Speech output could not reload. Choose Light mode')
    f.original.reply({ id: f.original.messages.at(-1)!.id, type: 'restart-speech' })
    await flushWorkerJobs()
    const fresh = FakeWorker.current
    fresh.reply({ id: fresh.messages.at(-1)!.id, type: 'loaded', tier: 'light' })
    await assertion
    expect(fresh.terminate).toHaveBeenCalledOnce()
    f.model.dispose()
    await vi.advanceTimersByTimeAsync(2000)
  })
})

describe('debug capture browser preference', () => {
  it('defaults off, restores the opt-in and persists only a boolean, never samples or transcript', () => {
    const values = new Map<string, string>()
    const setItem = vi.fn((key: string, value: string) => { values.set(key, value) })
    vi.stubGlobal('localStorage', { getItem: (key: string) => values.get(key) ?? null, setItem })
    const controller = stubController()
    const store = createHubStore(liveState())
    const first = browserVoiceCallService(controller, store)
    expect(first.getSnapshot().debugCaptureEnabled).toBe(false)
    first.setDebugCaptureEnabled(true)
    expect(setItem).toHaveBeenCalledExactlyOnceWith('xerxes-call-debug-utterance', 'true')
    const restored = browserVoiceCallService(controller, store)
    expect(restored.getSnapshot().debugCaptureEnabled).toBe(true)
    restored.setDebugCaptureEnabled(false)
    expect(setItem).toHaveBeenLastCalledWith('xerxes-call-debug-utterance', 'false')
    expect(restored.getSnapshot().call.debugCapture).toBeUndefined()
    first.dispose()
    restored.dispose()
  })
  it('defaults off when localStorage is unavailable and still allows a memory-only opt-in', () => {
    vi.stubGlobal('localStorage', { getItem() { throw new Error('Storage blocked') }, setItem() { throw new Error('Storage blocked') } })
    const service = browserVoiceCallService(stubController(), createHubStore(liveState()))
    expect(service.getSnapshot().debugCaptureEnabled).toBe(false)
    service.setDebugCaptureEnabled(true)
    expect(service.getSnapshot().debugCaptureEnabled).toBe(true)
    service.dispose()
  })
})

describe('shared worklet microphone cancellation', () => {
  it('releases granted microphone tracks immediately when hang-up happens during worklet loading', async () => {
    const stop = vi.fn()
    const close = vi.fn(async () => undefined)
    let loaded = (): void => undefined
    const module = new Promise<void>(resolve => { loaded = resolve })
    vi.stubGlobal('navigator', { mediaDevices: { getUserMedia: vi.fn(async () => ({ getTracks: () => [{ stop }] })) } })
    vi.stubGlobal('AudioContext', class { audioWorklet = { addModule: () => module }; close = close })
    vi.stubGlobal('AudioWorkletNode', class {})
    const abort = new AbortController()
    const opening = browserLiveCaptureDeps()!.open(abort.signal)
    for (let i = 0; i < 10; i++) await Promise.resolve()
    abort.abort()
    expect(stop).toHaveBeenCalledOnce()
    expect(close).toHaveBeenCalledOnce()
    loaded()
    await expect(opening).rejects.toMatchObject({ kind: 'cancelled' })
    expect(stop).toHaveBeenCalledOnce()
  })
  it('releases a microphone grant that arrives after hang-up during the permission prompt', async () => {
    const stop = vi.fn()
    const close = vi.fn(async () => undefined)
    let grant = (): void => undefined
    vi.stubGlobal('navigator', { mediaDevices: { getUserMedia: () => new Promise(resolve => { grant = () => resolve({ getTracks: () => [{ stop }] }) }) } })
    vi.stubGlobal('AudioContext', class { resume = async () => undefined; close = close })
    vi.stubGlobal('AudioWorkletNode', class {})
    const abort = new AbortController()
    const opening = browserLiveCaptureDeps()!.open(abort.signal)
    abort.abort()
    expect(close).toHaveBeenCalledOnce()
    grant()
    await expect(opening).rejects.toMatchObject({ kind: 'cancelled' })
    expect(stop).toHaveBeenCalledOnce()
  })
  it('creates and resumes the capture AudioContext synchronously inside the tap, before the permission prompt', () => {
    const order: string[] = []
    vi.stubGlobal('navigator', { mediaDevices: { getUserMedia: () => { order.push('getUserMedia'); return new Promise(() => undefined) } } })
    vi.stubGlobal('AudioContext', class { constructor() { order.push('context') } resume = () => { order.push('resume'); return Promise.resolve() }; close = async () => undefined })
    vi.stubGlobal('AudioWorkletNode', class {})
    void browserLiveCaptureDeps()!.open(new AbortController().signal)
    // All three happened before open() yielded once.
    expect(order).toEqual(['context', 'resume', 'getUserMedia'])
  })
  it('never asks for the microphone when the audio context cannot be created', async () => {
    const getUserMedia = vi.fn(async () => ({ getTracks: () => [] }))
    vi.stubGlobal('navigator', { mediaDevices: { getUserMedia } })
    vi.stubGlobal('AudioContext', class { constructor() { throw new Error('No audio device') } })
    vi.stubGlobal('AudioWorkletNode', class {})
    await expect(browserLiveCaptureDeps()!.open(new AbortController().signal)).rejects.toThrow('could not initialize')
    expect(getUserMedia).not.toHaveBeenCalled()
  })
})

class FakeUtterance {
  onend: (() => void) | null = null
  onerror: (() => void) | null = null
  constructor(readonly text: string) {}
}

describe('real playback-level metering', () => {
  it('reads Web Audio output for the orb and stops the meter when playback is flushed', async () => {
    let nextFrame: FrameRequestCallback | undefined
    const cancel = vi.fn()
    vi.stubGlobal('requestAnimationFrame', vi.fn((callback: FrameRequestCallback) => { nextFrame = callback; return 1 }))
    vi.stubGlobal('cancelAnimationFrame', cancel)
    const analyser = { fftSize: 256, connect: vi.fn(), disconnect: vi.fn(), getFloatTimeDomainData: (samples: Float32Array) => samples.fill(0.125) }
    const source = { buffer: undefined as unknown, connect: vi.fn(), disconnect: vi.fn(), start: vi.fn(), stop: vi.fn(), onended: null as (() => void) | null }
    vi.stubGlobal('AudioContext', class {
      state = 'running'
      destination = {}
      resume = async () => undefined
      close = async () => undefined
      createAnalyser = () => analyser
      createBuffer = () => ({ getChannelData: () => new Float32Array(512) })
      createBufferSource = () => source
    })
    const output = browserCallOutput()
    const level = vi.fn()
    output.onLevel?.(level)
    const playing = output.play({ samples: new Float32Array(512), sampleRate: 16000 }, new AbortController().signal)
    await Promise.resolve()
    await Promise.resolve()
    nextFrame?.(0)
    expect(source.connect).toHaveBeenCalledWith(analyser)
    expect(level).toHaveBeenCalledWith(0.5)
    output.flush()
    await playing
    expect(level).toHaveBeenLastCalledWith(0)
    expect(cancel).toHaveBeenCalled()
    output.close()
    expect(analyser.disconnect).toHaveBeenCalledOnce()
  })
})

describe('local phone voice output', () => {
  function outputFixture(local: boolean) {
    const close = vi.fn(async () => undefined)
    vi.stubGlobal('AudioContext', class { state = 'running'; resume = async () => undefined; close = close })
    vi.stubGlobal('SpeechSynthesisUtterance', FakeUtterance)
    const spoken: FakeUtterance[] = []
    const synth = { getVoices: () => [{ localService: local, lang: 'en-US' }], speak: vi.fn((utterance: FakeUtterance) => spoken.push(utterance)), cancel: vi.fn() }
    vi.stubGlobal('speechSynthesis', synth)
    return { output: browserCallOutput(), synth, spoken, close }
  }
  it('refuses network English voices instead of sending reply text off the phone', async () => {
    const f = outputFixture(false)
    await f.output.unlock()
    await expect(f.output.speak('Private reply', new AbortController().signal)).rejects.toThrow('local English voice')
    expect(f.synth.speak).not.toHaveBeenCalled()
    f.output.close()
    expect(f.close).toHaveBeenCalledOnce()
  })
  it('reports synthesis errors and resolves cancelled utterances without late completion callbacks', async () => {
    const f = outputFixture(true)
    const error = f.output.speak('Hello', new AbortController().signal)
    f.spoken.at(-1)?.onerror?.()
    await expect(error).rejects.toThrow('could not speak')
    const abort = new AbortController()
    const cancelled = f.output.speak('Another reply', abort.signal)
    const utterance = f.spoken.at(-1)!
    abort.abort()
    await expect(cancelled).resolves.toBeUndefined()
    expect(f.synth.cancel).toHaveBeenCalledOnce()
    expect(utterance.onend).toBeNull()
    expect(utterance.onerror).toBeNull()
    f.output.close()
  })
})
