// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { AutoModel, PretrainedConfig, Tensor, type PreTrainedModel } from '@huggingface/transformers'

import { VOICE_MODELS } from '../../../client/voiceModelManifest.js'
import { VAD_FRAME_SAMPLES } from './logic.js'
import type { CallWorkerRequest } from './protocol.js'
import { configureCallWorker, workerErrorDetail } from './workerEnvironment.js'

// Silero is intentionally isolated: synchronous STT/TTS WASM must never block turn detection or barge-in.
const scope = configureCallWorker()
let model: PreTrainedModel | undefined
let state = new Tensor('float32', new Float32Array(256), [2, 1, 128])
let context = new Float32Array(64)
let disposed = false
let queue: Promise<void> = Promise.resolve()

async function handle(request: CallWorkerRequest): Promise<void> {
  if (request.type === 'dispose') {
    disposed = true
    await model?.dispose()
    model = undefined
    state.dispose()
    scope.postMessage({ id: request.id, type: 'done' })
    return
  }
  if (disposed || request.type === 'cancel') return
  if (request.type === 'load') {
    const spec = VOICE_MODELS.vad
    model = await AutoModel.from_pretrained(spec.repo, {
      config: new PretrainedConfig({ model_type: 'custom' }), revision: spec.revision,
      dtype: spec.dtype, device: 'wasm',
      progress_callback: progress => {
        if (progress.status === 'progress') scope.postMessage({ id: request.id, type: 'progress', progress: { loaded: progress.loaded, total: spec.files[0]!.size } })
      },
    })
    scope.postMessage({ id: request.id, type: 'loaded', tier: request.tier })
  } else if (request.type === 'reset-vad') {
    state.dispose()
    state = new Tensor('float32', new Float32Array(256), [2, 1, 128])
    context = new Float32Array(64)
    scope.postMessage({ id: request.id, type: 'done' })
  } else if (request.type === 'vad') {
    if (!model || request.samples.length !== VAD_FRAME_SAMPLES) throw new Error('VAD not ready.')
    const input = new Float32Array(64 + VAD_FRAME_SAMPLES)
    input.set(context)
    input.set(request.samples, 64)
    const inputTensor = new Tensor('float32', input, [1, input.length])
    const sr = new Tensor('int64', [16000n], [])
    let result: Record<string, unknown>
    try { result = await model({ input: inputTensor, state, sr }) } finally { inputTensor.dispose(); sr.dispose() }
    const probability = result.output
    const nextState = result.stateN
    if (!(probability instanceof Tensor) || !(nextState instanceof Tensor)) throw new Error('Invalid Silero output.')
    const value = Number(probability.data[0])
    probability.dispose()
    state.dispose()
    state = nextState
    context = request.samples.slice(-64)
    scope.postMessage({ id: request.id, type: 'probability', probability: value })
  } else throw new Error('Unsupported VAD job.')
}

scope.onmessage = event => {
  const request = event.data
  // One recurrent-state owner at a time, on a worker that never executes a speech-recognition/synthesis job.
  queue = queue.then(() => handle(request)).catch((error: unknown) => {
    scope.postMessage({ id: request.id, type: 'error', message: `On-phone VAD failed (${workerErrorDetail(error)}). Restart the call or try another browser.` })
  })
}
