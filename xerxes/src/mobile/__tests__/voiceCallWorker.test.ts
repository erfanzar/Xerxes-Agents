// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { VOICE_MODELS } from '../../client/voiceModelManifest.js'
import type { CallWorkerRequest, CallWorkerResponse } from '../voice/call/protocol.js'

// Mock the actual ML boundary, not the worker adapter. Importing this test never downloads a model.
const fake = vi.hoisted(() => {
  class Tensor {
    readonly data: Float32Array | number[] | bigint[]
    readonly dims: number[]
    dispose = vi.fn()
    constructor(_type: string, data: Float32Array | number[] | bigint[], dims: number[]) { this.data = data; this.dims = dims }
  }
  const vad = Object.assign(vi.fn(async (_input: Record<string, Tensor>) => ({ output: new Tensor('float32', [0.75], [1, 1]), stateN: new Tensor('float32', new Float32Array(256), [2, 1, 128]) })), { dispose: vi.fn(async () => undefined) })
  const ttsModel = { dispose: vi.fn(async () => undefined) }
  const stt = Object.assign(vi.fn(async () => ({ text: 'Hello XERXES.' })), { dispose: vi.fn(async () => undefined) })
  return {
    Tensor, vad, ttsModel, stt,
    env: { allowLocalModels: true, remoteHost: '', remotePathTemplate: '', backends: { onnx: { wasm: { wasmPaths: '', numThreads: 4, proxy: true } } } },
    autoModel: vi.fn(async (_repo: string, _options: unknown) => vad),
    ttsLoad: vi.fn(async (_repo: string, _options: unknown) => ttsModel),
    tokenizer: vi.fn(async (_repo: string, _options: unknown) => ({})),
    pipeline: vi.fn(async (_task: string, _repo: string, _options: unknown) => stt),
    generate: vi.fn(async (_text: string, _options: unknown) => ({ audio: new Float32Array(240), sampling_rate: 24_000 })),
  }
})
vi.mock('@huggingface/transformers', () => ({
  AutoModel: { from_pretrained: fake.autoModel }, AutoTokenizer: { from_pretrained: fake.tokenizer },
  PretrainedConfig: class { constructor(readonly value: unknown) { Object.assign(this, value) } },
  StyleTextToSpeech2Model: { from_pretrained: fake.ttsLoad }, Tensor: fake.Tensor, env: fake.env, pipeline: fake.pipeline,
}))
vi.mock('kokoro-js', () => ({ KokoroTTS: class {
  constructor(readonly model: unknown, readonly tokenizer: unknown) {}
  generate = fake.generate
} }))

let messages: CallWorkerResponse[]
async function ticks(): Promise<void> { for (let i = 0; i < 8; i++) await new Promise(resolve => setTimeout(resolve, 1)) }
function send(request: CallWorkerRequest): void {
  const worker = globalThis as unknown as { onmessage: (event: MessageEvent<CallWorkerRequest>) => void }
  worker.onmessage(new MessageEvent('message', { data: request }))
}
async function dispatch(request: CallWorkerRequest): Promise<CallWorkerResponse | undefined> {
  send(request)
  await ticks()
  return messages.filter(message => message.id === request.id && message.type !== 'progress').at(-1)
}

beforeEach(async () => {
  vi.resetModules()
  vi.clearAllMocks()
  fake.autoModel.mockResolvedValue(fake.vad)
  fake.ttsLoad.mockResolvedValue(fake.ttsModel)
  fake.pipeline.mockResolvedValue(fake.stt)
  fake.generate.mockResolvedValue({ audio: new Float32Array(240), sampling_rate: 24_000 })
  fake.stt.mockResolvedValue({ text: 'Hello XERXES.' })
  messages = []
  vi.stubGlobal('location', { origin: 'https://hub.example' })
  vi.stubGlobal('navigator', { gpu: {} })
  vi.stubGlobal('postMessage', (message: CallWorkerResponse) => { messages.push(message) })
  vi.stubGlobal('onmessage', undefined)
  vi.stubGlobal('fetch', vi.fn(async () => new Response('fixture')))
  await import('../voice/call/worker.js')
})
afterEach(() => { vi.unstubAllGlobals() })

describe('call inference worker with fake transformers and Kokoro', () => {
  it('configures same-origin single-thread inference and leaves VAD to its independent worker', async () => {
    expect(fake.env).toMatchObject({ allowLocalModels: false, remoteHost: 'https://hub.example/voice-models/', remotePathTemplate: '{model}/resolve/{revision}/', backends: { onnx: { wasm: { wasmPaths: 'https://hub.example/voice-ort/', numThreads: 1, proxy: false } } } })
    expect(fake.autoModel).not.toHaveBeenCalled()
    expect(await dispatch({ id: 1, type: 'load', tier: 'light' })).toMatchObject({ type: 'loaded', tier: 'light' })
    expect(fake.autoModel).not.toHaveBeenCalled()
    expect(fake.pipeline).toHaveBeenCalledWith('automatic-speech-recognition', VOICE_MODELS.stt_tiny.repo, expect.objectContaining({ revision: VOICE_MODELS.stt_tiny.revision, device: 'wasm', dtype: VOICE_MODELS.stt_tiny.dtype }))
    expect(fake.ttsLoad).not.toHaveBeenCalled()
  })
  it('runs Silero independently with its custom config, context and recurrent state, then disposes it', async () => {
    await import('../voice/call/vadWorker.js')
    await dispatch({ id: 1, type: 'load', tier: 'light' })
    expect(fake.autoModel).toHaveBeenCalledWith(VOICE_MODELS.vad.repo, expect.objectContaining({ config: expect.objectContaining({ model_type: 'custom' }), revision: VOICE_MODELS.vad.revision, device: 'wasm', dtype: VOICE_MODELS.vad.dtype }))
    expect(await dispatch({ id: 2, type: 'vad', samples: new Float32Array(512) })).toMatchObject({ type: 'probability', probability: 0.75 })
    expect(fake.vad).toHaveBeenCalledWith(expect.objectContaining({ input: expect.objectContaining({ dims: [1, 576] }), state: expect.objectContaining({ dims: [2, 1, 128] }), sr: expect.objectContaining({ data: [16000n], dims: [] }) }))
    expect(fake.pipeline).not.toHaveBeenCalled()
    await dispatch({ id: 3, type: 'dispose' })
    expect(fake.vad.dispose).toHaveBeenCalledOnce()
  })
  it('never runs quantized speech recognition on WebGPU, even when the browser offers it', async () => {
    // Measured in Chrome: q8 Moonshine on WebGPU decoded repeated multilingual junk; on WASM it was correct.
    expect(VOICE_MODELS.stt_base.dtype).toBe('q8')
    await dispatch({ id: 1, type: 'load', tier: 'standard' })
    expect(fake.pipeline).toHaveBeenCalledWith('automatic-speech-recognition', VOICE_MODELS.stt_base.repo, expect.objectContaining({ device: 'wasm' }))
    expect(fake.pipeline).not.toHaveBeenCalledWith(expect.anything(), expect.anything(), expect.objectContaining({ device: 'webgpu' }))
    fake.stt.mockRejectedValueOnce(new Error('inference failed'))
    // A WASM failure is a plain error, never a WebGPU-recovery loop.
    expect(await dispatch({ id: 2, type: 'transcribe', samples: new Float32Array(16_000) })).toMatchObject({ type: 'error' })
    expect(fake.pipeline).toHaveBeenCalledTimes(1)
    await dispatch({ id: 3, type: 'dispose' })
    expect(fake.stt.dispose).toHaveBeenCalledOnce()
  })
  it('honors forced WASM recovery and does not loop when that inference also fails', async () => {
    await dispatch({ id: 1, type: 'load', tier: 'light', forceWasm: true })
    expect(fake.pipeline).toHaveBeenCalledWith('automatic-speech-recognition', VOICE_MODELS.stt_tiny.repo, expect.objectContaining({ device: 'wasm' }))
    fake.stt.mockRejectedValueOnce(new Error('WASM failed'))
    expect(await dispatch({ id: 2, type: 'transcribe', samples: new Float32Array(16_000) })).toMatchObject({ type: 'error' })
    expect(fake.stt).toHaveBeenCalledTimes(1)
    expect(fake.pipeline).toHaveBeenCalledTimes(1)
  })
  it('pins Kokoro’s model AND tokenizer revisions, primes only af_heart, and returns 24 kHz PCM', async () => {
    await dispatch({ id: 1, type: 'load', tier: 'standard' })
    expect(fake.ttsLoad).toHaveBeenCalledWith(VOICE_MODELS.tts.repo, expect.objectContaining({ revision: VOICE_MODELS.tts.revision, dtype: 'q8', device: 'wasm' }))
    expect(fake.tokenizer).toHaveBeenCalledWith(VOICE_MODELS.tts.repo, expect.objectContaining({ revision: VOICE_MODELS.tts.revision }))
    expect(fake.generate).toHaveBeenCalledWith('Ready.', { voice: 'af_heart' })
    expect(await dispatch({ id: 2, type: 'synthesize', text: 'Hello.' })).toMatchObject({ type: 'audio', audio: { sampleRate: 24_000, samples: expect.any(Float32Array) } })
  })
  it('reports a WASM speech-recognition load failure as an error, never a WebGPU retry', async () => {
    fake.pipeline.mockRejectedValueOnce(new Error('No adapter'))
    expect(await dispatch({ id: 1, type: 'load', tier: 'light' })).toMatchObject({ type: 'error' })
    expect(messages.some(message => message.type === 'retry-wasm')).toBe(false)
    expect(fake.pipeline).toHaveBeenCalledTimes(1)
  })
  it('disposes failed Kokoro and loads Moonshine tiny when voice initialization fails', async () => {
    fake.generate.mockRejectedValueOnce(new Error('phonemizer failed'))
    expect(await dispatch({ id: 1, type: 'load', tier: 'standard' })).toMatchObject({ type: 'loaded', tier: 'light', notice: expect.stringContaining('Kokoro could not load') })
    expect(fake.ttsModel.dispose).toHaveBeenCalledOnce()
    expect(fake.pipeline).toHaveBeenCalledWith('automatic-speech-recognition', VOICE_MODELS.stt_tiny.repo, expect.anything())
  })
  it('disposes the speech models and acknowledges hang-up without owning a VAD model', async () => {
    await dispatch({ id: 1, type: 'load', tier: 'standard' })
    expect(await dispatch({ id: 2, type: 'dispose' })).toMatchObject({ type: 'done' })
    expect(fake.vad.dispose).not.toHaveBeenCalled()
    expect(fake.stt.dispose).toHaveBeenCalledOnce()
    expect(fake.ttsModel.dispose).toHaveBeenCalledOnce()
  })
  it('skips cancelled queued STT/TTS and suppresses an active synthesis result after barge-in', async () => {
    await dispatch({ id: 1, type: 'load', tier: 'standard' })
    let finish = (_value: { audio: Float32Array<ArrayBuffer>; sampling_rate: number }): void => undefined
    fake.generate.mockImplementationOnce(async () => new Promise(resolve => { finish = resolve }))
    send({ id: 2, type: 'synthesize', text: 'Slow sentence.' })
    await ticks()
    send({ id: 3, type: 'transcribe', samples: new Float32Array(16000) })
    send({ id: 4, type: 'synthesize', text: 'Stale sentence.' })
    send({ id: 2, type: 'cancel' })
    send({ id: 3, type: 'cancel' })
    send({ id: 4, type: 'cancel' })
    finish({ audio: new Float32Array(240), sampling_rate: 24000 })
    await ticks()
    expect(fake.stt).not.toHaveBeenCalled()
    expect(fake.generate).toHaveBeenCalledTimes(2) // warm-up + active sentence; queued sentence never starts
    expect(messages.some(message => [2, 3, 4].includes(message.id))).toBe(false)
  })
  it('sanitizes model failures instead of echoing URLs or paths', async () => {
    fake.pipeline.mockRejectedValue(new Error('secret URL https://private.example'))
    const result = await dispatch({ id: 1, type: 'load', tier: 'light', forceWasm: true })
    expect(result).toMatchObject({ type: 'error', message: expect.stringContaining('On-phone load failed') })
    expect(JSON.stringify(result)).not.toContain('private.example')
  })
})
