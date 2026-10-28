// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, it, vi } from 'vitest'

import { browserCallModels, browserCallOutput } from '../voice/call/browser.js'
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
    await Promise.resolve()
    await Promise.resolve()
    abort.abort()
    expect(stop).toHaveBeenCalledOnce()
    expect(close).toHaveBeenCalledOnce()
    loaded()
    await expect(opening).rejects.toMatchObject({ kind: 'cancelled' })
    expect(stop).toHaveBeenCalledOnce()
  })
  it('releases tracks if creating the audio context fails', async () => {
    const stop = vi.fn()
    vi.stubGlobal('navigator', { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [{ stop }] }) } })
    vi.stubGlobal('AudioContext', class { constructor() { throw new Error('No audio device') } })
    vi.stubGlobal('AudioWorkletNode', class {})
    await expect(browserLiveCaptureDeps()!.open(new AbortController().signal)).rejects.toThrow('could not initialize')
    expect(stop).toHaveBeenCalledOnce()
  })
})

class FakeUtterance {
  onend: (() => void) | null = null
  onerror: (() => void) | null = null
  constructor(readonly text: string) {}
}

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
