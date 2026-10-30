// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { prepareCallSpeech } from '../../../client/callSpeech.js'
import { tierDownloadBytes, type VoiceModelTier } from '../../../client/voiceModelManifest.js'
import type { CaptureInterruption, LiveCapture, LiveCaptureDeps } from '../liveCapture.js'
import { createDebugCapture, type CallDebugCapture } from './debugCapture.js'
import { CallResampler, CallSendError, VadFrames, VadSegmenter, type CallReply, type VadUtteranceOffsets } from './logic.js'
import { CallStepError, type CallAudio, type CallModels, type CallProgress } from './protocol.js'

export const VAD_RESPONSE_TIMEOUT_MS = 10_000
export const CALL_STILL_WORKING = "XERXES is still working; I'll read the reply when it arrives."
export const CALL_DIDNT_CATCH = "Didn't catch that — say it again."
export const CALL_RECONNECTING = 'Reconnecting to the Hub… the call stays open.'
export const CALL_WAITING_FOR_HUB = "Not connected to the Hub. I'll send that as soon as it is back."
export const CALL_RESENDING = 'Could not reach the Hub. Trying that again…'
export const CALL_AUDIO_PAUSED = 'Audio paused by the phone — waiting to resume…'
export const CALL_AUDIO_RESTORING = 'Restoring the microphone…'
export const CALL_AUDIO_LOST = 'The microphone stopped (a phone call, Siri or another app took it). Tap Restart voice to call again.'
export const CALL_AUDIO_STALLED = 'Call audio stalled, so the rest of that reply was not spoken. It is in the thread.'
export const CALL_TRANSCRIBE_STALLED = 'Speech recognition took too long. Say that again.'
export const CALL_SYNTH_STALLED = 'XERXES’s voice took too long, so that reply was not spoken. It is in the thread.'
export const CALL_ENDED_HIDDEN = 'Call ended after the app was hidden for 60 seconds.'
export const SPEECH_RESTART_LOADING_DELAY_MS = 750
/** Longest first spoken chunk; it breaks at a comma when one is near. */
export const FIRST_SPEECH_CHARS = 120
/** The phone paused call audio (phone call, Siri): after this long without resuming, the microphone is reopened. */
export const AUDIO_PAUSE_LIMIT_MS = 8000
/** Microphone reopen attempts allowed per minute before the call ends with a visible reason. */
export const AUDIO_REOPEN_LIMIT = 2
/** A clip that has not finished this long after its own duration is treated as stalled playback. */
export const PLAYBACK_GRACE_MS = 4000
export const SPEAK_MIN_MS = 8000
export const SPEAK_MS_PER_CHAR = 120
/** Backstops above the worker's own deadlines (which also replace a hung speech runtime). */
export const TRANSCRIBE_DEADLINE_MS = 75_000
export const SYNTH_DEADLINE_MS = 60_000
/** A send whose outcome is unknown is retried once with the same message id (the Hub deduplicates it). */
export const RESEND_DELAY_MS = 3000
export type CallPhase = 'idle' | 'loading' | 'listening' | 'hearing' | 'transcribing' | 'thinking' | 'speaking' | 'error'
export interface CallSnapshot {
  readonly phase: CallPhase
  readonly tier: VoiceModelTier
  readonly muted: boolean
  readonly heard: string
  readonly reply: string
  readonly notice?: string
  readonly error?: string
  /** Why the call ended without the person pressing End (shown with a Restart button). */
  readonly endReason?: string | undefined
  /** No local voice can speak: replies are shown on screen, not spoken. */
  readonly textOnly?: boolean
  readonly progress?: CallProgress
  /** The speech engine is being replaced after a failure; models come from the phone's cache, not a download. */
  readonly restoring?: boolean
  /** Display-only meters/caption; never used for VAD or turn decisions. */
  readonly inputLevel?: number
  readonly outputLevel?: number
  readonly spokenSentence?: string
  /** Opt-in, local-only last model input. Never part of a supervisor request. */
  readonly debugCapture?: CallDebugCapture | undefined
}
export interface CallOutput {
  unlock(): Promise<void>
  play(audio: CallAudio, signal: AbortSignal): Promise<void>
  speak(text: string, signal: AbortSignal): Promise<void>
  flush(): void
  close(): void
  /** Optional display-only output meter (PCM has real Web Audio levels; browser voices expose no PCM). */
  onLevel?(listener: (level: number) => void): void
  /** Checks that the loaded tier can speak; returns the reason when it cannot (Light without a local voice). */
  prepare?(tier: VoiceModelTier): Promise<string | undefined>
  /** Resume output audio after the phone paused it; true when it runs again. */
  resume?(): Promise<boolean>
}
export interface CallEngineDeps {
  readonly capture: LiveCaptureDeps
  readonly models: () => CallModels
  readonly output: () => CallOutput
  /** Submit `text` as message `messageId` and resolve with XERXES's reply (a resend reuses the id). */
  readonly reply: (text: string, signal: AbortSignal, onTimeout: () => void, messageId: string) => Promise<CallReply>
  readonly messageId?: () => string
}

/** Runs `work` with a deadline. On expiry its signal aborts, `onExpire` runs and the call step fails recoverably. */
async function withDeadline<T>(parent: AbortSignal, ms: number, message: string, work: (signal: AbortSignal) => Promise<T>, onExpire?: () => void): Promise<T> {
  const child = new AbortController()
  const abort = (): void => child.abort()
  parent.addEventListener('abort', abort, { once: true })
  if (parent.aborted) child.abort()
  let timer: ReturnType<typeof setTimeout> | undefined
  let cancelled = (): void => undefined
  const deadline = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => { onExpire?.(); child.abort(); reject(new CallStepError(message)) }, ms)
    // A cancelled call never waits on work that ignores its signal.
    cancelled = () => {
      // Cleared synchronously: a hung-up call leaves no deadline timer behind.
      if (timer !== undefined) clearTimeout(timer)
      timer = undefined
      reject(new Error('Call cancelled.'))
    }
    parent.addEventListener('abort', cancelled, { once: true })
    if (parent.aborted) cancelled()
  })
  try { return await Promise.race([work(child.signal), deadline]) } finally {
    if (timer !== undefined) clearTimeout(timer)
    parent.removeEventListener('abort', abort)
    parent.removeEventListener('abort', cancelled)
  }
}

/** Call lifecycle with injected devices/models; VAD owns a worker independent of blocking STT/TTS. */
export class CallEngine {
  private snapshot: CallSnapshot = { phase: 'idle', tier: 'standard', muted: false, heard: '', reply: '' }
  private readonly observers = new Set<() => void>()
  private lifetime: AbortController | undefined
  /** The current STT job. */
  private transcription: AbortController | undefined
  /** The latest submitted turn (thinking, late or speaking its reply). */
  private turn: AbortController | undefined
  /** The turn whose reply is awaited right now (the call shows Thinking for it). */
  private awaiting: AbortController | undefined
  /** A turn past the 120 s notice whose late reply is still observed. */
  private lateTurn: AbortController | undefined
  /** A reply that arrived while the person was speaking; spoken unless their new words replace it. */
  private held: { readonly turn: AbortController; readonly reply: CallReply } | undefined
  /** Text kept while the Hub is unreachable; sent with the same id when the connection returns. */
  private pendingSend: { readonly text: string; readonly messageId: string } | undefined
  private resend: { readonly text: string; readonly messageId: string; readonly timer: ReturnType<typeof setTimeout> } | undefined
  private speech: AbortController | undefined
  private models: CallModels | undefined
  private output: CallOutput | undefined
  private capture: LiveCapture | undefined
  private vad = new VadSegmenter()
  private frames: Float32Array[] = []
  private processing = false
  private inputEpoch = 0
  private keepDebugUtterance = false
  private debugEpoch = 0
  private connected = true
  private hidden = false
  private reopening = false
  private reopens: number[] = []
  private hiddenTimer: ReturnType<typeof setTimeout> | undefined
  private pauseTimer: ReturnType<typeof setTimeout> | undefined
  constructor(private readonly deps: CallEngineDeps) {}
  readonly getSnapshot = (): CallSnapshot => this.snapshot
  readonly subscribe = (listener: () => void): (() => void) => { this.observers.add(listener); return () => { this.observers.delete(listener) } }
  private update(patch: Partial<CallSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...patch }
    for (const observer of this.observers) observer()
  }
  private newMessageId(): string { return this.deps.messageId?.() ?? crypto.randomUUID() }

  async start(tier: VoiceModelTier): Promise<void> {
    this.hangup()
    const lifetime = new AbortController()
    this.lifetime = lifetime
    this.reopens = []
    this.snapshot = { phase: 'loading', tier, muted: false, heard: '', reply: '' }
    this.update({})
    try {
      // Everything that needs the tap runs before the first await: output audio, the microphone
      // permission prompt and the capture AudioContext. The model download comes after.
      const output = this.deps.output()
      this.output = output
      output.onLevel?.(outputLevel => { if (!lifetime.signal.aborted) this.update({ outputLevel }) })
      const opening = this.deps.capture.open(lifetime.signal).then(capture => {
        if (lifetime.signal.aborted) { capture.close(); return undefined }
        this.capture = capture
        return capture
      })
      // A denied microphone ends the call at once, without waiting for the download.
      opening.catch((error: unknown) => { if (!lifetime.signal.aborted) this.fail(error) })
      await output.unlock()
      if (lifetime.signal.aborted) return
      const models = this.deps.models()
      this.models = models
      const loaded = await models.load(tier, progress => { if (!lifetime.signal.aborted) this.update({ progress }) }, lifetime.signal)
      if (lifetime.signal.aborted) return
      this.update({ tier: loaded.tier, ...(loaded.notice ? { notice: loaded.notice } : {}) })
      const cannotSpeak = await output.prepare?.(loaded.tier)
      if (lifetime.signal.aborted) return
      if (cannotSpeak !== undefined) {
        // The person chose Light: say so before the call starts. A fallback to Light keeps the call as text.
        if (tier === 'light') throw new Error(cannotSpeak)
        this.update({ textOnly: true, notice: `${cannotSpeak} Replies will show on screen instead.` })
      }
      const capture = await opening
      if (!capture || lifetime.signal.aborted) return
      this.attach(capture, lifetime)
      this.update({ phase: 'listening' })
    } catch (error) { if (!lifetime.signal.aborted) this.fail(error) }
  }

  private attach(capture: LiveCapture, lifetime: AbortController): void {
    this.capture = capture
    const resampler = new CallResampler(capture.sampleRate)
    const splitter = new VadFrames()
    capture.onFrame(samples => {
      if (lifetime.signal.aborted || this.capture !== capture) return
      // Display-only RMS: inference still receives the original, unmodified frames and uses Silero probabilities.
      let energy = 0
      for (const sample of samples) energy += sample * sample
      const inputLevel = this.snapshot.muted ? 0 : Math.min(1, Math.sqrt(energy / Math.max(1, samples.length)) * 4)
      if (Math.abs(inputLevel - (this.snapshot.inputLevel ?? 0)) > 0.01) this.update({ inputLevel })
      // Discard input during transcription rather than replaying stale microphone frames later.
      if (this.snapshot.muted || !this.acceptsAudio()) return
      this.frames.push(...splitter.push(resampler.push(samples)))
      // Keep recent audio, not an unbounded latency backlog. A busy speech worker must never end the call.
      if (this.frames.length > 64) this.frames.splice(0, this.frames.length - 64)
      void this.drain(lifetime)
    })
    capture.onInterruption?.(event => this.captureInterrupted(event, capture, lifetime))
  }

  /** The VAD keeps listening while XERXES thinks, so the person can add to or replace a request. */
  private acceptsAudio(): boolean {
    const phase = this.snapshot.phase
    return phase === 'listening' || phase === 'hearing' || phase === 'speaking' || phase === 'thinking'
  }
  private busyHearing(): boolean { return this.snapshot.phase === 'hearing' || this.transcription !== undefined }
  private restingPhase(): CallPhase { return this.awaiting && !this.awaiting.signal.aborted ? 'thinking' : 'listening' }
  private resetInput(): void {
    this.inputEpoch += 1
    this.frames = []
    this.vad.reset()
    this.models?.resetVad()
  }
  /** Back to listening (or thinking), speaking a held reply when nothing newer replaced it. */
  private settle(notice?: string): void {
    // The person is mid-sentence: keep their audio and phase; only the notice changes.
    if (this.busyHearing()) { if (notice !== undefined) this.update({ notice }); return }
    this.resetInput()
    const held = this.held
    if (held && !held.turn.signal.aborted && !this.busyHearing()) {
      this.held = undefined
      if (notice !== undefined) this.update({ notice })
      void this.speakReply(held.reply, held.turn)
      return
    }
    this.update({ phase: this.restingPhase(), ...(notice !== undefined ? { notice } : {}) })
  }
  /** A recoverable step failure keeps the call; anything else ends it. */
  private recover(error: unknown): void {
    if (error instanceof CallStepError) {
      this.update({ spokenSentence: '', outputLevel: 0 })
      this.settle(error.message)
    } else this.fail(error)
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
          // Stops the speech only. A late-reply observer stays until new words actually replace it.
          this.speech?.abort()
          this.output?.flush()
          this.update({ phase: 'hearing', outputLevel: 0 })
        } else if (step.hearing && (this.snapshot.phase === 'listening' || this.snapshot.phase === 'thinking')) {
          this.update({ phase: 'hearing' })
        }
        if (step.utterance) {
          this.resetInput()
          if (this.snapshot.phase !== 'speaking') void this.respond(step.utterance, step.offsets, lifetime)
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
    if (this.snapshot.textOnly === true) {
      this.resetInput()
      this.update({ spokenSentence: text, phase: this.restingPhase() })
      return
    }
    // A short first chunk gets the first audio out sooner; later chunks keep full sentences.
    const sentences = prepareCallSpeech(text, 4000, 240, FIRST_SPEECH_CHARS)
    if (sentences.length === 0) { this.resetInput(); this.update({ phase: this.restingPhase() }); return }
    this.speech?.abort()
    const speech = new AbortController()
    this.speech = speech
    const abort = (): void => speech.abort()
    turn.signal.addEventListener('abort', abort, { once: true })
    this.resetInput()
    this.update({ phase: 'speaking' })
    try {
      if (this.snapshot.tier === 'light') {
        for (const sentence of sentences) {
          if (speech.signal.aborted) break
          this.update({ spokenSentence: sentence })
          // The phone's voice can stop calling back after an interruption; never wait on it forever.
          await withDeadline(speech.signal, Math.max(SPEAK_MIN_MS, sentence.length * SPEAK_MS_PER_CHAR), CALL_AUDIO_STALLED,
            signal => output.speak(sentence, signal), () => output.flush())
        }
      } else {
        // Streamed playback: the next sentence is synthesized while the current one plays, so speech
        // flows without a pause between sentences. A barge-in aborts both through the same signal.
        const synth = (sentence: string): Promise<CallAudio> => {
          const pending = withDeadline(speech.signal, SYNTH_DEADLINE_MS, CALL_SYNTH_STALLED, signal => models.synthesize(sentence, signal))
          pending.catch(() => undefined)
          return pending
        }
        let next: Promise<CallAudio> | undefined = synth(sentences[0]!)
        for (let index = 0; index < sentences.length && next; index++) {
          const audio = await next
          if (speech.signal.aborted) break
          next = index + 1 < sentences.length ? synth(sentences[index + 1]!) : undefined
          this.update({ spokenSentence: sentences[index]! })
          // A suspended AudioContext (phone call, Siri) never fires `ended`: a wall-clock watchdog ends the clip.
          const seconds = audio.sampleRate > 0 ? audio.samples.length / audio.sampleRate : 0
          await withDeadline(speech.signal, seconds * 1000 + PLAYBACK_GRACE_MS, CALL_AUDIO_STALLED,
            signal => output.play(audio, signal), () => output.flush())
        }
      }
    } catch (error) {
      if (!speech.signal.aborted) throw error
    } finally {
      turn.signal.removeEventListener('abort', abort)
      if (this.speech === speech) {
        this.speech = undefined
        if (!turn.signal.aborted && !speech.signal.aborted) { this.resetInput(); this.update({ phase: this.restingPhase() }) }
      }
    }
  }

  private async speakReply(reply: CallReply, turn: AbortController): Promise<void> {
    this.speech?.abort()
    this.output?.flush()
    // A failed XERXES turn ("XERXES stopped: …") is spoken and stays visible.
    this.update({ reply: reply.text, notice: reply.role === 'system' ? reply.text : '' })
    try { await this.say(reply.text, turn) } catch (error) { if (!turn.signal.aborted) this.recover(error) }
  }

  /** New words replace whatever XERXES was still answering locally (the Hub keeps its own turn). */
  private supersede(): void {
    for (const turn of [this.turn, this.lateTurn, this.awaiting, this.held?.turn]) turn?.abort()
    this.turn = this.lateTurn = this.awaiting = this.held = undefined
    this.pendingSend = undefined
    if (this.resend) clearTimeout(this.resend.timer)
    this.resend = undefined
    this.speech?.abort()
    this.output?.flush()
  }

  private async respond(samples: Float32Array, offsets: VadUtteranceOffsets, lifetime: AbortController): Promise<void> {
    const models = this.models
    if (!models) return
    this.transcription?.abort()
    const job = new AbortController()
    this.transcription = job
    const abort = (): void => job.abort()
    lifetime.signal.addEventListener('abort', abort, { once: true })
    let reloadTimer: ReturnType<typeof setTimeout> | undefined
    let reloadProgress: CallProgress = { loaded: 0, total: tierDownloadBytes(this.snapshot.tier) }
    const stopReloadTimer = (): void => { if (reloadTimer !== undefined) clearTimeout(reloadTimer); reloadTimer = undefined }
    job.signal.addEventListener('abort', stopReloadTimer, { once: true })
    try {
      const debugEpoch = ++this.debugEpoch
      this.update({ phase: 'transcribing', debugCapture: this.keepDebugUtterance ? createDebugCapture(samples, offsets) : undefined })
      const transcript = await withDeadline(job.signal, TRANSCRIBE_DEADLINE_MS, CALL_TRANSCRIBE_STALLED, signal => models.transcribe(samples, signal, {
        restarting: () => {
          if (job.signal.aborted || this.transcription !== job) return
          this.resetInput()
          this.update({ notice: "Didn't catch that — restoring speech…", spokenSentence: '' })
          stopReloadTimer()
          reloadTimer = setTimeout(() => {
            if (!job.signal.aborted && this.transcription === job) this.update({ phase: 'loading', progress: reloadProgress, restoring: true })
          }, SPEECH_RESTART_LOADING_DELAY_MS)
        },
        progress: progress => {
          if (job.signal.aborted || this.transcription !== job) return
          reloadProgress = progress
          if (this.snapshot.phase === 'loading') this.update({ progress })
        },
      }))
      stopReloadTimer()
      if (this.snapshot.restoring) this.update({ restoring: false })
      if (job.signal.aborted) return
      if (this.transcription === job) this.transcription = undefined
      if (debugEpoch === this.debugEpoch && this.snapshot.debugCapture) this.update({ debugCapture: { ...this.snapshot.debugCapture, transcript: transcript ?? '' } })
      // A cough or an unreadable clip never cancels what XERXES is still answering.
      if (transcript === null) { this.settle(CALL_DIDNT_CATCH); return }
      const text = transcript.trim()
      if (!text) { this.settle(); return }
      this.supersede()
      await this.ask(text, this.newMessageId(), 0)
    } catch (error) {
      if (!job.signal.aborted) {
        if (this.transcription === job) this.transcription = undefined
        if (this.snapshot.restoring) this.update({ restoring: false })
        this.recover(error)
      }
    } finally {
      stopReloadTimer()
      job.signal.removeEventListener('abort', stopReloadTimer)
      lifetime.signal.removeEventListener('abort', abort)
      if (this.transcription === job) this.transcription = undefined
    }
  }

  private async ask(text: string, messageId: string, attempt: number): Promise<void> {
    const lifetime = this.lifetime
    if (!lifetime || lifetime.signal.aborted) return
    if (!this.connected) {
      this.pendingSend = { text, messageId }
      this.resetInput()
      this.update({ heard: text, reply: '', spokenSentence: '', phase: this.restingPhase(), notice: CALL_WAITING_FOR_HUB })
      return
    }
    const turn = new AbortController()
    this.turn = turn
    this.awaiting = turn
    const abort = (): void => turn.abort()
    lifetime.signal.addEventListener('abort', abort, { once: true })
    this.update({ heard: text, reply: '', spokenSentence: '', phase: 'thinking', notice: '' })
    try {
      const reply = await this.deps.reply(text, turn.signal, () => {
        if (turn.signal.aborted || this.awaiting !== turn) return
        // Keep the observer and models alive. New words (not a noise) replace it.
        this.awaiting = undefined
        this.lateTurn = turn
        if (this.busyHearing()) { this.update({ notice: CALL_STILL_WORKING }); return }
        this.resetInput()
        this.update({ phase: 'listening', notice: CALL_STILL_WORKING })
        void this.say(CALL_STILL_WORKING, turn).catch((error: unknown) => { if (!turn.signal.aborted) this.recover(error) })
      }, messageId)
      if (turn.signal.aborted) return
      if (this.awaiting === turn) this.awaiting = undefined
      if (this.lateTurn === turn) this.lateTurn = undefined
      if (this.busyHearing()) {
        // The person is talking: hold the reply until their words are transcribed.
        this.held = { turn, reply }
        this.update({ reply: reply.text })
        return
      }
      await this.speakReply(reply, turn)
    } catch (error) {
      if (turn.signal.aborted) return
      if (this.awaiting === turn) this.awaiting = undefined
      if (this.lateTurn === turn) this.lateTurn = undefined
      if (error instanceof CallSendError && error.kind !== 'refused' && (!this.connected || attempt === 0)) this.queueSend(text, messageId, attempt)
      else this.recover(error)
    } finally {
      lifetime.signal.removeEventListener('abort', abort)
      if (this.turn === turn && !this.speech) this.turn = undefined
    }
  }

  /** The send's outcome is unknown: keep the text and resend it with the same id (never a new message). */
  private queueSend(text: string, messageId: string, attempt: number): void {
    const lifetime = this.lifetime
    if (!lifetime) return
    this.resetInput()
    if (!this.connected) {
      this.pendingSend = { text, messageId }
      this.update({ phase: this.restingPhase(), notice: CALL_WAITING_FOR_HUB })
      return
    }
    const timer = setTimeout(() => {
      this.resend = undefined
      if (this.lifetime === lifetime) void this.ask(text, messageId, attempt + 1)
    }, RESEND_DELAY_MS)
    this.resend = { text, messageId, timer }
    this.update({ phase: this.restingPhase(), notice: CALL_RESENDING })
  }

  /** Hub link state. A drop pauses sending with a notice; it never ends the call or unloads the models. */
  setConnected(connected: boolean): void {
    if (connected === this.connected) return
    this.connected = connected
    if (!this.lifetime) return
    if (!connected) {
      if (this.resend) {
        clearTimeout(this.resend.timer)
        this.pendingSend = { text: this.resend.text, messageId: this.resend.messageId }
        this.resend = undefined
      }
      this.update({ notice: this.pendingSend ? CALL_WAITING_FOR_HUB : CALL_RECONNECTING })
      return
    }
    if (this.snapshot.notice === CALL_RECONNECTING || this.snapshot.notice === CALL_WAITING_FOR_HUB) this.update({ notice: '' })
    const pending = this.pendingSend
    this.pendingSend = undefined
    if (pending && !this.busyHearing() && this.snapshot.phase !== 'loading') void this.ask(pending.text, pending.messageId, 0)
    else if (pending) this.pendingSend = pending
  }

  private captureInterrupted(event: CaptureInterruption, capture: LiveCapture, lifetime: AbortController): void {
    if (lifetime.signal.aborted || this.capture !== capture) return
    if (event === 'lost') { void this.audioLost(lifetime); return }
    if (event === 'paused') {
      this.update({ notice: CALL_AUDIO_PAUSED })
      // While hidden the phone may pause audio legitimately; it is checked again on return.
      if (this.pauseTimer === undefined && !this.hidden) {
        this.pauseTimer = setTimeout(() => { this.pauseTimer = undefined; void this.audioLost(lifetime) }, AUDIO_PAUSE_LIMIT_MS)
      }
      return
    }
    if (this.pauseTimer !== undefined) clearTimeout(this.pauseTimer)
    this.pauseTimer = undefined
    this.resetInput()
    if (this.snapshot.notice === CALL_AUDIO_PAUSED) this.update({ notice: '' })
  }

  /** The microphone ended or stayed paused: reopen it, or end the call with the reason on screen. */
  private async audioLost(lifetime: AbortController): Promise<void> {
    if (this.reopening || lifetime.signal.aborted) return
    if (this.pauseTimer !== undefined) clearTimeout(this.pauseTimer)
    this.pauseTimer = undefined
    const now = Date.now()
    this.reopens = this.reopens.filter(at => at > now - 60_000)
    if (this.reopens.length >= AUDIO_REOPEN_LIMIT) { this.hangup(CALL_AUDIO_LOST); return }
    this.reopens.push(now)
    this.reopening = true
    const previous = this.capture
    this.capture = undefined
    previous?.close()
    this.resetInput()
    this.update({ notice: CALL_AUDIO_RESTORING })
    try {
      const capture = await this.deps.capture.open(lifetime.signal)
      if (lifetime.signal.aborted) { capture.close(); return }
      this.attach(capture, lifetime)
      this.update({ notice: '' })
    } catch {
      if (!lifetime.signal.aborted) this.hangup(CALL_AUDIO_LOST)
    } finally { this.reopening = false }
  }

  private async recheckAudio(lifetime: AbortController): Promise<void> {
    const capture = this.capture
    const [input] = await Promise.all([capture?.resume?.() ?? Promise.resolve(true), this.output?.resume?.() ?? Promise.resolve(true)])
    if (lifetime.signal.aborted || this.capture !== capture || this.reopening) return
    if (!input) { void this.audioLost(lifetime); return }
    if (this.pauseTimer !== undefined) clearTimeout(this.pauseTimer)
    this.pauseTimer = undefined
    if (this.snapshot.notice === CALL_AUDIO_PAUSED) this.update({ notice: '' })
  }

  setDebugCaptureEnabled(enabled: boolean): void {
    if (enabled === this.keepDebugUtterance) return
    this.keepDebugUtterance = enabled
    this.debugEpoch += 1
    if (!enabled) this.update({ debugCapture: undefined })
  }

  /** Local voice-view interrupt; does not cancel the supervisor turn or dispose the session. */
  interrupt(): void {
    if (this.snapshot.phase !== 'speaking') return
    this.speech?.abort()
    this.output?.flush()
    this.resetInput()
    this.update({ phase: this.restingPhase(), outputLevel: 0, spokenSentence: '' })
  }
  mute(muted: boolean): void {
    const hearing = this.snapshot.phase === 'hearing'
    this.resetInput()
    this.update({ muted, inputLevel: 0, ...(hearing ? { phase: this.restingPhase() } : {}) })
    if (hearing) this.settle()
  }
  visibility(hidden: boolean): void {
    this.hidden = hidden
    if (this.hiddenTimer !== undefined) clearTimeout(this.hiddenTimer)
    this.hiddenTimer = undefined
    const lifetime = this.lifetime
    if (!lifetime) return
    if (hidden) this.hiddenTimer = setTimeout(() => { this.hangup(CALL_ENDED_HIDDEN) }, 60_000)
    else if (this.capture) void this.recheckAudio(lifetime)
  }
  /** Ends the call. `reason` (an unrequested end) stays on screen with a Restart button. */
  hangup(reason?: string): void {
    this.lifetime?.abort()
    this.lifetime = undefined
    this.supersede()
    this.transcription?.abort()
    this.transcription = undefined
    this.speech = undefined
    if (this.hiddenTimer !== undefined) clearTimeout(this.hiddenTimer)
    this.hiddenTimer = undefined
    if (this.pauseTimer !== undefined) clearTimeout(this.pauseTimer)
    this.pauseTimer = undefined
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
    this.debugEpoch += 1
    this.update({ phase: 'idle', muted: false, inputLevel: 0, outputLevel: 0, spokenSentence: '', debugCapture: undefined, endReason: reason, ...(reason ? { notice: '' } : {}) })
  }
  private fail(error: unknown): void {
    this.hangup()
    this.update({ phase: 'error', error: error instanceof Error ? error.message : 'The call could not continue.' })
  }
}
