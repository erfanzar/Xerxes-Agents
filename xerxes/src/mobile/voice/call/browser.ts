// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { HubController } from '../../../client/hubController.js'
import type { HubStore } from '../../../client/hubStore.js'
import type { VoiceControl } from '../../../client/voiceControl.js'
import { tierDownloadBytes, type VoiceModelTier } from '../../../client/voiceModelManifest.js'
import { browserLiveCaptureDeps } from '../liveCapture.js'
import { CallEngine, type CallOutput } from './engine.js'
import { fetchCallModelsStatus, waitForCallReply } from './logic.js'
import { VoiceCallService } from './service.js'
import type { CallModels, CallProgress, CallTranscriptionRecovery, CallWorkerRequest, CallWorkerResponse } from './protocol.js'

export const CALL_WORKER_URL = '/voice-call-worker.js'
export const CALL_VAD_WORKER_URL = '/voice-vad-worker.js'
export const SPEECH_RESTART_WINDOW_MS = 60_000
export const SPEECH_RESTART_LIMIT = 3
export const SPEECH_RESTART_TIMEOUT_MS = 30_000
type WithoutId<T> = T extends { readonly id: number } ? Omit<T, 'id'> : never
type WorkerJob = WithoutId<Exclude<CallWorkerRequest, { readonly type: 'cancel' }>>

function callWorkerClient(url: string) {
  const worker = new Worker(url, { type: 'module', name: url === CALL_VAD_WORKER_URL ? 'xerxes-call-vad' : 'xerxes-call-speech' })
  let id = 0
  let disposed = false
  let disposeId: number | undefined
  let failure: Error | undefined
  const pending = new Map<number, { resolve: (response: CallWorkerResponse) => void; reject: (error: Error) => void; progress?: (value: CallProgress) => void }>()
  let terminateTimer: ReturnType<typeof setTimeout> | undefined
  let terminated = false
  const terminate = (): void => {
    if (terminated) return
    terminated = disposed = true
    if (terminateTimer !== undefined) clearTimeout(terminateTimer)
    for (const call of pending.values()) call.reject(new Error('Call ended.'))
    pending.clear()
    worker.onmessage = null
    worker.onerror = null
    worker.terminate()
  }
  worker.onmessage = (event: MessageEvent<CallWorkerResponse>) => {
    const response = event.data
    if (disposed) { if (response.id === disposeId && response.type === 'done') terminate(); return }
    const call = pending.get(response.id)
    if (!call) return
    if (response.type === 'progress') { call.progress?.(response.progress); return }
    pending.delete(response.id)
    if (response.type === 'error') call.reject(new Error(response.message))
    else call.resolve(response)
  }
  worker.onerror = event => {
    event.preventDefault()
    failure = new Error('The on-phone voice worker stopped. Check browser support and try Light mode.')
    for (const call of pending.values()) call.reject(failure)
    pending.clear()
  }
  function request(job: WorkerJob, signal?: AbortSignal, progress?: (value: CallProgress) => void): Promise<CallWorkerResponse> {
    const message: CallWorkerRequest = { ...job, id: ++id }
    return new Promise((resolve, reject) => {
      if (disposed || signal?.aborted || failure) { reject(failure ?? new Error('Call cancelled.')); return }
      const abort = (): void => {
        pending.delete(message.id)
        worker.postMessage({ type: 'cancel', id: message.id } satisfies CallWorkerRequest)
        reject(new Error('Call cancelled.'))
      }
      const clean = (): void => signal?.removeEventListener('abort', abort)
      pending.set(message.id, { resolve: response => { clean(); resolve(response) }, reject: error => { clean(); reject(error) }, ...(progress ? { progress } : {}) })
      signal?.addEventListener('abort', abort, { once: true })
      try {
        worker.postMessage(message, 'samples' in message ? [message.samples.buffer] : [])
      } catch {
        pending.delete(message.id)
        clean()
        reject(new Error('Could not send audio to the on-phone worker.'))
      }
    })
  }
  return {
    request,
    /** Poisoned/cancelled speech runtimes must release WASM immediately, without waiting for dispose. */
    terminate,
    dispose() {
      if (disposed) return
      disposed = true
      for (const call of pending.values()) call.reject(new Error('Call ended.'))
      pending.clear()
      disposeId = ++id
      worker.postMessage({ id: disposeId, type: 'dispose' } satisfies CallWorkerRequest)
      // Also frees resources if ONNX initialization is stuck and cannot acknowledge dispose.
      terminateTimer = setTimeout(terminate, 2000)
    },
  }
}

export function browserCallModels(): CallModels {
  const vad = callWorkerClient(CALL_VAD_WORKER_URL)
  let speech: ReturnType<typeof callWorkerClient>
  let activeTier: VoiceModelTier = 'standard'
  let disposed = false
  let recovering = false
  const restarts: number[] = []
  const checkActive = (signal: AbortSignal): void => { if (disposed || signal.aborted) throw new Error('Call cancelled.') }
  try { speech = callWorkerClient(CALL_WORKER_URL) } catch (error) { vad.dispose(); throw error }
  const restartSpeech = async (signal: AbortSignal, recovery?: CallTranscriptionRecovery): Promise<void> => {
    checkActive(signal)
    const now = Date.now()
    while (restarts.length && restarts[0]! <= now - SPEECH_RESTART_WINDOW_MS) restarts.shift()
    restarts.push(now)
    // Fail on the third recovery request, before allocating another runtime or retrying bad audio.
    if (restarts.length >= SPEECH_RESTART_LIMIT) {
      speech.terminate()
      throw new Error('Speech recognition needed 3 restarts within 60 seconds. End this call and choose Light mode.')
    }
    recovering = true
    speech.terminate()
    const restart = new AbortController()
    let timedOut = false
    let ready = false
    const abort = (): void => restart.abort()
    signal.addEventListener('abort', abort, { once: true })
    const timer = setTimeout(() => { timedOut = true; restart.abort() }, SPEECH_RESTART_TIMEOUT_MS)
    try {
      speech = callWorkerClient(CALL_WORKER_URL)
      recovery?.restarting()
      checkActive(signal)
      const loaded = await speech.request({ type: 'load', tier: activeTier, forceWasm: true }, restart.signal, progress => recovery?.progress(progress))
      checkActive(signal)
      if (loaded.type !== 'loaded') throw new Error('Speech recognition could not restart. Choose Light mode and start a new call.')
      // Restart both Kokoro and STT. Never silently change the engine's tier during mid-call recovery.
      if (loaded.tier !== activeTier) throw new Error('Speech output could not reload. Choose Light mode and start a new call.')
      ready = true
    } catch (error) {
      if (timedOut && !disposed && !signal.aborted) throw new Error('Speech recognition took too long to restart. Choose Light mode and start a new call.')
      throw error
    } finally {
      clearTimeout(timer)
      signal.removeEventListener('abort', abort)
      if (!ready) speech.terminate()
      recovering = false
    }
  }
  return {
    async load(tier, progress, signal) {
      const initialized = await vad.request({ type: 'load', tier }, signal, value => progress({ loaded: value.loaded, total: tierDownloadBytes(tier) }))
      if (initialized.type !== 'loaded') throw new Error('Invalid VAD load response.')
      let response = await speech.request({ type: 'load', tier }, signal, progress)
      if (response.type === 'retry-wasm') {
        if (disposed || signal.aborted) throw new Error('Call cancelled.')
        speech.dispose()
        speech = callWorkerClient(CALL_WORKER_URL)
        response = await speech.request({ type: 'load', tier, forceWasm: true }, signal, progress)
      }
      if (response.type !== 'loaded') throw new Error('Invalid model load response.')
      activeTier = response.tier
      return { tier: response.tier, ...(response.notice ? { notice: response.notice } : {}) }
    },
    async vad(samples, signal) {
      const response = await vad.request({ type: 'vad', samples: samples.slice() }, signal)
      if (response.type !== 'probability') throw new Error('Invalid VAD response.')
      return response.probability
    },
    resetVad() { void vad.request({ type: 'reset-vad' }).catch(() => undefined) },
    async transcribe(samples, signal, recovery) {
      checkActive(signal)
      if (recovering) throw new Error('Speech recognition is still restarting.')
      // Retain one bounded copy only for the existing GPU-to-WASM retry. WASM failures drop the clip.
      let response = await speech.request({ type: 'transcribe', samples: samples.slice() }, signal)
      checkActive(signal)
      if (response.type === 'retry-wasm') {
        await restartSpeech(signal)
        checkActive(signal)
        response = await speech.request({ type: 'transcribe', samples }, signal)
        checkActive(signal)
      }
      if (response.type === 'restart-speech') {
        await restartSpeech(signal, recovery)
        checkActive(signal)
        return null
      }
      if (response.type !== 'transcript') throw new Error('Transcription failed after the WASM retry.')
      return response.text
    },
    async synthesize(text, signal) {
      const response = await speech.request({ type: 'synthesize', text }, signal)
      if (response.type !== 'audio') throw new Error('Invalid speech response.')
      return response.audio
    },
    dispose() { if (disposed) return; disposed = true; vad.dispose(); if (recovering) speech.terminate(); else speech.dispose() },
  }
}

/** Local PCM playback and strictly local English speechSynthesis (never a network voice). */
export function browserCallOutput(): CallOutput {
  const context = new AudioContext()
  const sources = new Map<AudioBufferSourceNode, () => void>()
  const analyser = context.createAnalyser?.()
  if (analyser) { analyser.fftSize = 256; analyser.connect(context.destination) }
  const levels = new Float32Array(256)
  let reportLevel: ((level: number) => void) | undefined
  let meterFrame: number | undefined
  const stopMeter = (): void => {
    if (meterFrame !== undefined) cancelAnimationFrame(meterFrame)
    meterFrame = undefined
    reportLevel?.(0)
  }
  const startMeter = (): void => {
    if (!analyser || !reportLevel || meterFrame !== undefined || typeof requestAnimationFrame === 'undefined') return
    const tick = (): void => {
      if (sources.size === 0) { stopMeter(); return }
      analyser.getFloatTimeDomainData(levels)
      let energy = 0
      for (const sample of levels) energy += sample * sample
      reportLevel?.(Math.min(1, Math.sqrt(energy / levels.length) * 4))
      meterFrame = requestAnimationFrame(tick)
    }
    meterFrame = requestAnimationFrame(tick)
  }
  let endSpeech: (() => void) | undefined
  const synth = typeof speechSynthesis === 'undefined' ? undefined : speechSynthesis
  const flush = (): void => {
    for (const [source, finish] of sources) {
      try { source.stop() } catch { /* already ended */ }
      finish()
    }
    sources.clear()
    stopMeter()
    synth?.cancel()
    endSpeech?.()
  }
  return {
    async unlock() {
      const resumed = context.resume()
      // iOS requires speechSynthesis to be unlocked in a gesture as well.
      if (synth) {
        const voice = synth.getVoices().find(voice => voice.localService && /^en(?:-|$)/iu.test(voice.lang))
        if (voice) {
          const silent = new SpeechSynthesisUtterance(' ')
          silent.voice = voice
          silent.volume = 0
          synth.speak(silent)
        }
      }
      await resumed
      if (context.state !== 'running') throw new Error('Tap Start again to allow call audio.')
    },
    async play(audio, signal) {
      if (signal.aborted) return
      await context.resume()
      if (context.state !== 'running') throw new Error('The phone blocked call audio. Hang up and tap Start again.')
      if (signal.aborted) return
      const buffer = context.createBuffer(1, audio.samples.length, audio.sampleRate)
      buffer.getChannelData(0).set(audio.samples)
      const source = context.createBufferSource()
      source.buffer = buffer
      source.connect(analyser ?? context.destination)
      await new Promise<void>(resolve => {
        let ended = false
        const finish = (): void => {
          if (ended) return
          ended = true
          signal.removeEventListener('abort', abort)
          source.onended = null
          source.disconnect()
          sources.delete(source)
          if (sources.size === 0) stopMeter()
          resolve()
        }
        const abort = (): void => { try { source.stop() } catch { /* already ended */ }; finish() }
        source.onended = finish
        sources.set(source, finish)
        signal.addEventListener('abort', abort, { once: true })
        source.start()
        startMeter()
      })
    },
    async speak(text, signal) {
      if (!synth) throw new Error('Light mode needs an installed English voice. This browser has no speech synthesis.')
      if (signal.aborted) return
      if (synth.getVoices().length === 0) {
        await new Promise<void>(resolve => {
          const done = (): void => { clearTimeout(timer); synth.removeEventListener('voiceschanged', done); signal.removeEventListener('abort', done); resolve() }
          const timer = setTimeout(done, 1500)
          synth.addEventListener('voiceschanged', done, { once: true })
          signal.addEventListener('abort', done, { once: true })
        })
      }
      if (signal.aborted) return
      const voice = synth.getVoices().find(voice => voice.localService && /^en(?:-|$)/iu.test(voice.lang))
      if (!voice) throw new Error('Light mode needs a local English voice installed in your phone’s speech settings. Network voices are not used.')
      await new Promise<void>((resolve, reject) => {
        const utterance = new SpeechSynthesisUtterance(text)
        utterance.voice = voice
        utterance.lang = voice.lang
        let ended = false
        const clean = (): boolean => {
          if (ended) return false
          ended = true
          signal.removeEventListener('abort', abort)
          utterance.onend = null
          utterance.onerror = null
          if (endSpeech === finish) endSpeech = undefined
          return true
        }
        const finish = (): void => { if (clean()) resolve() }
        const abort = (): void => { synth.cancel(); finish() }
        endSpeech = finish
        utterance.onend = finish
        utterance.onerror = () => {
          if (!clean()) return
          if (signal.aborted) resolve()
          else reject(new Error('The phone’s built-in voice could not speak.'))
        }
        signal.addEventListener('abort', abort, { once: true })
        synth.speak(utterance)
      })
    },
    onLevel(listener) { reportLevel = listener },
    flush,
    close() { flush(); analyser?.disconnect(); reportLevel = undefined; void context.close().catch(() => undefined) },
  }
}

export function browserCallEngine(controller: HubController, store: HubStore): CallEngine {
  const capture = browserLiveCaptureDeps()
  if (!capture || typeof Worker === 'undefined') throw new Error('Call mode needs a secure HTTPS page, microphone access, AudioWorklet and Web Workers. This browser does not offer them.')
  return new CallEngine({
    capture,
    models: browserCallModels,
    output: browserCallOutput,
    reply(text, signal, onTimeout) {
      const messageId = crypto.randomUUID()
      const threadId = store.getSnapshot().supervisor.threadId
      return waitForCallReply({
        messageId, ...(threadId ? { threadId } : {}), signal, onTimeout,
        messages: () => store.getSnapshot().supervisor.messages,
        subscribe: store.subscribe,
        send: () => controller.supervisorSend(text, 'voice', messageId),
      })
    },
  })
}

/** Created once by MobileApp; no worker/microphone is opened until Start voice is pressed. */
export function browserVoiceCallService(controller: HubController, store: HubStore, voice?: VoiceControl): VoiceCallService {
  const memory = typeof navigator === 'undefined' ? undefined : (navigator as Navigator & { deviceMemory?: number }).deviceMemory
  return new VoiceCallService({
    engine: () => browserCallEngine(controller, store),
    modelsStatus: signal => fetchCallModelsStatus(fetch, signal),
    canStart: () => {
      const state = store.getSnapshot()
      return state.connection.state === 'live' && state.supervisor.phase === 'ready' && state.supervisorConfig.status !== 'unconfigured'
    },
    beforeStart: () => { voice?.cancel(); voice?.live?.stop(); voice?.interrupt?.(); void voice?.armReply?.(false) },
    ...(memory === undefined ? {} : { deviceMemory: memory }),
    noticeSeen: () => { try { return localStorage.getItem('xerxes-call-download-notice') === 'seen' } catch { return false } },
    rememberNotice: () => { try { localStorage.setItem('xerxes-call-download-notice', 'seen') } catch { /* storage is optional */ } },
    debugEnabled: () => { try { return localStorage.getItem('xerxes-call-debug-utterance') === 'true' } catch { return false } },
    rememberDebugEnabled: enabled => { try { localStorage.setItem('xerxes-call-debug-utterance', String(enabled)) } catch { /* only the preference, never audio, is stored */ } },
  })
}
