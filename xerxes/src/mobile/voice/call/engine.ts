// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { prepareCallSpeech } from '../../../client/callSpeech.js'
import type { VoiceModelTier } from '../../../client/voiceModelManifest.js'
import type { LiveCapture, LiveCaptureDeps } from '../liveCapture.js'
import { CallResampler, VadFrames, VadSegmenter, type CallReply } from './logic.js'
import type { CallAudio, CallModels, CallProgress } from './protocol.js'

export const VAD_RESPONSE_TIMEOUT_MS = 10_000
export const CALL_STILL_WORKING = "XERXES is still working; I'll read the reply when it arrives."
export type CallPhase = 'idle' | 'loading' | 'listening' | 'hearing' | 'transcribing' | 'thinking' | 'speaking' | 'error'
export interface CallSnapshot {
  readonly phase: CallPhase
  readonly tier: VoiceModelTier
  readonly muted: boolean
  readonly heard: string
  readonly reply: string
  readonly notice?: string
  readonly error?: string
  readonly progress?: CallProgress
}
export interface CallOutput {
  unlock(): Promise<void>
  play(audio: CallAudio, signal: AbortSignal): Promise<void>
  speak(text: string, signal: AbortSignal): Promise<void>
  flush(): void
  close(): void
}
export interface CallEngineDeps {
  readonly capture: LiveCaptureDeps
  readonly models: () => CallModels
  readonly output: () => CallOutput
  readonly reply: (text: string, signal: AbortSignal, onTimeout: () => void) => Promise<CallReply>
}

/** Call lifecycle with injected devices/models; VAD owns a worker independent of blocking STT/TTS. */
export class CallEngine {
  private snapshot: CallSnapshot = { phase: 'idle', tier: 'standard', muted: false, heard: '', reply: '' }
  private readonly observers = new Set<() => void>()
  private lifetime: AbortController | undefined
  private turn: AbortController | undefined
  private lateTurn: AbortController | undefined
  private speech: AbortController | undefined
  private models: CallModels | undefined
  private output: CallOutput | undefined
  private capture: LiveCapture | undefined
  private vad = new VadSegmenter()
  private frames: Float32Array[] = []
  private processing = false
  private inputEpoch = 0
  private hiddenTimer: ReturnType<typeof setTimeout> | undefined
  constructor(private readonly deps: CallEngineDeps) {}
  readonly getSnapshot = (): CallSnapshot => this.snapshot
  readonly subscribe = (listener: () => void): (() => void) => { this.observers.add(listener); return () => { this.observers.delete(listener) } }
  private update(patch: Partial<CallSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...patch }
    for (const observer of this.observers) observer()
  }

  async start(tier: VoiceModelTier): Promise<void> {
    this.hangup()
    const lifetime = new AbortController()
    this.lifetime = lifetime
    this.snapshot = { phase: 'loading', tier, muted: false, heard: '', reply: '' }
    this.update({})
    try {
      const output = this.deps.output()
      this.output = output
      await output.unlock()
      if (lifetime.signal.aborted) return
      const models = this.deps.models()
      this.models = models
      const loaded = await models.load(tier, progress => { if (!lifetime.signal.aborted) this.update({ progress }) }, lifetime.signal)
      if (lifetime.signal.aborted) return
      this.update({ tier: loaded.tier, ...(loaded.notice ? { notice: loaded.notice } : {}) })
      const capture = await this.deps.capture.open(lifetime.signal)
      if (lifetime.signal.aborted) { capture.close(); return }
      this.capture = capture
      const resampler = new CallResampler(capture.sampleRate)
      const splitter = new VadFrames()
      this.update({ phase: 'listening' })
      capture.onFrame(samples => {
        // Discard input during transcription/thinking rather than replaying stale microphone frames later.
        if (lifetime.signal.aborted || this.snapshot.muted || !this.acceptsAudio()) return
        this.frames.push(...splitter.push(resampler.push(samples)))
        // Keep recent audio, not an unbounded latency backlog. A busy speech worker must never end the call.
        if (this.frames.length > 64) this.frames.splice(0, this.frames.length - 64)
        void this.drain(lifetime)
      })
    } catch (error) { if (!lifetime.signal.aborted) this.fail(error) }
  }

  private acceptsAudio(): boolean { return this.snapshot.phase === 'listening' || this.snapshot.phase === 'hearing' || this.snapshot.phase === 'speaking' }
  private resetInput(): void {
    this.inputEpoch += 1
    this.frames = []
    this.vad.reset()
    this.models?.resetVad()
  }
  private async probability(frame: Float32Array, signal: AbortSignal): Promise<number> {
    const models = this.models
    if (!models) throw new Error('VAD is unavailable.')
    let timer: ReturnType<typeof setTimeout> | undefined
    let abort = (): void => undefined
    const deadline = new Promise<never>((_resolve, reject) => {
      timer = setTimeout(() => reject(new Error('On-phone VAD stopped responding for 10 seconds. Restart the call.')), VAD_RESPONSE_TIMEOUT_MS)
      abort = () => reject(new Error('Call cancelled.'))
      signal.addEventListener('abort', abort, { once: true })
      if (signal.aborted) abort()
    })
    try { return await Promise.race([models.vad(frame, signal), deadline]) } finally {
      if (timer !== undefined) clearTimeout(timer)
      signal.removeEventListener('abort', abort)
    }
  }
  private async drain(lifetime: AbortController): Promise<void> {
    if (this.processing) return
    this.processing = true
    try {
      while (this.frames.length > 0 && !lifetime.signal.aborted) {
        const frame = this.frames.shift()!
        if (!this.models || !this.acceptsAudio() || this.snapshot.muted) continue
        const epoch = this.inputEpoch
        const probability = await this.probability(frame, lifetime.signal)
        if (epoch !== this.inputEpoch || lifetime.signal.aborted || !this.acceptsAudio() || this.snapshot.muted) continue
        const step = this.vad.push(frame, probability, this.snapshot.phase === 'speaking')
        if (step.bargeIn && this.snapshot.phase === 'speaking') {
          this.lateTurn?.abort()
          this.lateTurn = undefined
          this.speech?.abort()
          this.output?.flush()
          this.update({ phase: 'hearing' })
        } else if (step.hearing && this.snapshot.phase === 'listening') {
          // A new utterance retires the old late-reply observer before any new transcript is submitted.
          this.lateTurn?.abort()
          this.lateTurn = undefined
          this.update({ phase: 'hearing' })
        }
        if (step.utterance) {
          this.resetInput()
          if (this.snapshot.phase !== 'speaking') void this.respond(step.utterance, lifetime)
        }
      }
    } catch (error) { if (!lifetime.signal.aborted) this.fail(error) } finally {
      this.processing = false
      if (this.frames.length > 0 && this.lifetime && this.lifetime !== lifetime) void this.drain(this.lifetime)
    }
  }

  private async say(text: string, turn: AbortController): Promise<void> {
    const models = this.models
    const output = this.output
    if (!models || !output || turn.signal.aborted) return
    const sentences = prepareCallSpeech(text)
    if (sentences.length === 0) { this.resetInput(); this.update({ phase: 'listening' }); return }
    this.speech?.abort()
    const speech = new AbortController()
    this.speech = speech
    const abort = (): void => speech.abort()
    turn.signal.addEventListener('abort', abort, { once: true })
    this.resetInput()
    this.update({ phase: 'speaking' })
    try {
      for (const sentence of sentences) {
        if (speech.signal.aborted) break
        if (this.snapshot.tier === 'light') await output.speak(sentence, speech.signal)
        else {
          const audio = await models.synthesize(sentence, speech.signal)
          if (!speech.signal.aborted) await output.play(audio, speech.signal)
        }
      }
    } catch (error) { if (!speech.signal.aborted) throw error } finally {
      turn.signal.removeEventListener('abort', abort)
      if (this.speech === speech) {
        this.speech = undefined
        if (!turn.signal.aborted && !speech.signal.aborted) { this.resetInput(); this.update({ phase: 'listening' }) }
      }
    }
  }

  private async respond(samples: Float32Array, lifetime: AbortController): Promise<void> {
    const models = this.models
    if (!models) return
    this.turn?.abort()
    const turn = new AbortController()
    this.turn = turn
    const abort = (): void => turn.abort()
    lifetime.signal.addEventListener('abort', abort, { once: true })
    try {
      this.update({ phase: 'transcribing' })
      const text = (await models.transcribe(samples, turn.signal)).trim()
      if (turn.signal.aborted) return
      if (!text) { this.resetInput(); this.update({ phase: 'listening' }); return }
      this.update({ heard: text, reply: '', phase: 'thinking' })
      const reply = await this.deps.reply(text, turn.signal, () => {
        if (turn.signal.aborted || this.turn !== turn) return
        this.lateTurn = turn
        this.resetInput()
        this.update({ phase: 'listening', notice: CALL_STILL_WORKING })
        // Keep the observer and models alive. A new utterance aborts this observer and notice speech.
        void this.say(CALL_STILL_WORKING, turn).catch(error => { if (!turn.signal.aborted) this.fail(error) })
      })
      if (turn.signal.aborted) return
      this.lateTurn = undefined
      this.speech?.abort()
      this.output?.flush()
      this.update({ reply: reply.text })
      await this.say(reply.text, turn)
    } catch (error) { if (!turn.signal.aborted) this.fail(error) } finally {
      lifetime.signal.removeEventListener('abort', abort)
      if (this.turn === turn) this.turn = undefined
      if (this.lateTurn === turn) this.lateTurn = undefined
    }
  }

  mute(muted: boolean): void {
    this.resetInput()
    this.update({ muted, ...(this.snapshot.phase === 'hearing' ? { phase: 'listening' as const } : {}) })
  }
  connectionLost(): void {
    if (this.snapshot.phase !== 'error' && this.snapshot.phase !== 'idle') this.hangup('Connection lost. The call ended; your messages remain in the thread.')
  }
  visibility(hidden: boolean): void {
    if (this.hiddenTimer !== undefined) clearTimeout(this.hiddenTimer)
    this.hiddenTimer = hidden && this.lifetime ? setTimeout(() => { this.hangup('Call ended after the tab was hidden for 60 seconds.') }, 60_000) : undefined
  }
  hangup(notice?: string): void {
    this.lifetime?.abort()
    this.lifetime = undefined
    this.turn?.abort()
    this.turn = undefined
    this.lateTurn = undefined
    this.speech?.abort()
    this.speech = undefined
    if (this.hiddenTimer !== undefined) clearTimeout(this.hiddenTimer)
    this.hiddenTimer = undefined
    this.capture?.close()
    this.capture = undefined
    this.output?.flush()
    this.output?.close()
    this.output = undefined
    this.models?.dispose()
    this.models = undefined
    this.inputEpoch += 1
    this.frames = []
    this.vad.reset()
    this.update({ phase: 'idle', muted: false, ...(notice ? { notice } : {}) })
  }
  private fail(error: unknown): void {
    this.hangup()
    this.update({ phase: 'error', error: error instanceof Error ? error.message : 'The call could not continue.' })
  }
}
