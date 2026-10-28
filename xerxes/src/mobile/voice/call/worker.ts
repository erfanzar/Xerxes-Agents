// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { AutoTokenizer, StyleTextToSpeech2Model, pipeline, type AutomaticSpeechRecognitionPipeline, type ProgressCallback } from '@huggingface/transformers'
import { KokoroTTS } from 'kokoro-js'

import { VOICE_CALL_TTS_VOICE, VOICE_MODELS, tierDownloadBytes, type VoiceModelTier } from '../../../client/voiceModelManifest.js'
import type { CallWorkerRequest, CallWorkerResponse } from './protocol.js'
import { configureCallWorker, workerErrorDetail } from './workerEnvironment.js'

const scope = configureCallWorker()
let stt: AutomaticSpeechRecognitionPipeline | undefined
let sttDevice: 'webgpu' | 'wasm' = 'wasm'
let tier: VoiceModelTier = 'standard'
let tts: KokoroTTS | undefined
let stopping = false
const queue: CallWorkerRequest[] = []
const liveIds = new Set<number>()
const cancelled = new Set<number>()
let running = false
// Yield before starting queued work and after inference, so pending cancel messages are processed even after blocking WASM.
const yieldToMessages = (): Promise<void> => new Promise(resolve => setTimeout(resolve, 0))

function emit(response: CallWorkerResponse, transfer?: Transferable[]): void {
  if (!cancelled.has(response.id) && (!stopping || response.type === 'done')) scope.postMessage(response, transfer)
}
async function loadStt(device: 'webgpu' | 'wasm', progress?: ProgressCallback): Promise<AutomaticSpeechRecognitionPipeline> {
  const spec = tier === 'standard' ? VOICE_MODELS.stt_base : VOICE_MODELS.stt_tiny
  const loaded = await pipeline<'automatic-speech-recognition'>('automatic-speech-recognition', spec.repo, {
    revision: spec.revision, dtype: spec.dtype, device, ...(progress ? { progress_callback: progress } : {}),
  })
  sttDevice = device
  return loaded
}

async function load(request: Extract<CallWorkerRequest, { type: 'load' }>): Promise<void> {
  tier = request.tier
  let notice: string | undefined
  const totals = new Map<string, number>()
  const progress: ProgressCallback = value => {
    if (value.status === 'progress') totals.set(`${value.name}/${value.file}`, value.loaded)
    if (value.status === 'done') {
      const model = Object.values(VOICE_MODELS).find(model => model.repo === value.name)
      const file = model?.files.find(file => file.path === value.file)
      if (file) totals.set(`${value.name}/${value.file}`, file.size)
    }
    const total = tierDownloadBytes(tier)
    const vadBytes = VOICE_MODELS.vad.files.reduce((sum, file) => sum + file.size, 0)
    emit({ id: request.id, type: 'progress', progress: { loaded: Math.min(total, vadBytes + [...totals.values()].reduce((a, b) => a + b, 0)), total } })
  }
  if (tier === 'standard') {
    const spec = VOICE_MODELS.tts
    let model: StyleTextToSpeech2Model | undefined
    try {
      // q8 Kokoro is reliable on WASM; WebGPU is recommended for fp32, which we deliberately do not download.
      model = await StyleTextToSpeech2Model.from_pretrained(spec.repo, { revision: spec.revision, dtype: spec.dtype, device: 'wasm', progress_callback: progress })
      const tokenizer = await AutoTokenizer.from_pretrained(spec.repo, { revision: spec.revision, progress_callback: progress })
      tts = new KokoroTTS(model, tokenizer)
      await tts.generate('Ready.', { voice: VOICE_CALL_TTS_VOICE })
    } catch {
      await model?.dispose()
      tts = undefined
      tier = 'light'
      notice = 'Kokoro could not load. Using Light with the phone’s built-in English voice for this call.'
    }
  }
  if (stopping) return
  // Quantized (q8) Moonshine decodes garbage on WebGPU (measured in Chrome: repeated multilingual
  // tokens after 8.5 s) and is correct on WASM (0.3 s). Only full-precision weights may use WebGPU.
  const sttSpec = tier === 'standard' ? VOICE_MODELS.stt_base : VOICE_MODELS.stt_tiny
  const useGpu = sttSpec.dtype === 'fp32' && scope.navigator.gpu !== undefined && request.forceWasm !== true
  try { stt = await loadStt(useGpu ? 'webgpu' : 'wasm', progress) } catch (error) {
    if (!useGpu) throw error
    emit({ id: request.id, type: 'retry-wasm' })
    return
  }
  emit({ id: request.id, type: 'loaded', tier, ...(notice ? { notice } : {}) })
}

async function handle(request: CallWorkerRequest): Promise<void> {
  if (request.type === 'dispose') {
    await Promise.all([stt?.dispose(), tts?.model.dispose()])
    stt = undefined
    tts = undefined
    emit({ id: request.id, type: 'done' })
  } else if (request.type === 'load') await load(request)
  else if (request.type === 'transcribe') {
    if (!stt || request.samples.length > 480_000) throw new Error('Transcription not ready or audio too long.')
    let result
    try { result = await stt(request.samples, { max_new_tokens: 256 }) } catch (error) {
      if (sttDevice !== 'webgpu' || cancelled.has(request.id) || stopping) throw error
      // transformers.js chains inference promises internally. An execution error can poison that chain;
      // ask the page to retry once in a fresh WASM speech worker rather than reusing this runtime.
      await yieldToMessages()
      emit({ id: request.id, type: 'retry-wasm' })
      return
    }
    await yieldToMessages()
    const text = Array.isArray(result) ? result.map(item => item.text).join(' ') : result.text
    emit({ id: request.id, type: 'transcript', text: text.trim() })
  } else if (request.type === 'synthesize') {
    if (!tts || request.text.length > 300) throw new Error('Speech output not ready or sentence too long.')
    const audio = await tts.generate(request.text, { voice: VOICE_CALL_TTS_VOICE })
    await yieldToMessages()
    if (cancelled.has(request.id) || stopping) return
    const samples = Float32Array.from(audio.audio)
    emit({ id: request.id, type: 'audio', audio: { samples, sampleRate: audio.sampling_rate } }, [samples.buffer])
  } else throw new Error('Unsupported speech worker job.')
}

async function drain(): Promise<void> {
  if (running) return
  running = true
  try {
    while (queue.length > 0) {
      await yieldToMessages()
      const request = queue.shift()!
      try {
        if (!cancelled.has(request.id) && (!stopping || request.type === 'dispose')) await handle(request)
      } catch (error) {
        emit({ id: request.id, type: 'error', message: `On-phone ${request.type} failed (${workerErrorDetail(error)}). Try Light mode or check that the Hub’s call models are available.` })
      } finally { liveIds.delete(request.id); cancelled.delete(request.id) }
    }
  } finally { running = false }
}

scope.onmessage = event => {
  const request = event.data
  if (request.type === 'cancel') {
    if (liveIds.has(request.id)) cancelled.add(request.id)
    return
  }
  if (request.type === 'dispose') stopping = true
  if (stopping && request.type !== 'dispose') return
  liveIds.add(request.id)
  queue.push(request)
  void drain()
}
