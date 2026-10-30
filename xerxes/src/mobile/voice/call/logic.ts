// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { resolveVoiceModelFile, voiceModelFileUrlPath, type VoiceModelTier } from '../../../client/voiceModelManifest.js'
import { CallStepError } from './protocol.js'

export const CALL_SAMPLE_RATE = 16_000
export const VAD_FRAME_SAMPLES = 512
export const REPLY_TIMEOUT_MS = 120_000
/** Longest wait for the Hub to acknowledge a spoken message before the call says so and keeps listening. */
export const REPLY_SEND_TIMEOUT_MS = 30_000

/** Voice barge-in while XERXES speaks: Silero probability and how long it must last. */
export const BARGE_IN_PROBABILITY = 0.85
export const BARGE_IN_SECONDS = 0.5
/** Shortest speech that starts an utterance while listening: short words ("yes", "no", "stop") must count. */
export const VAD_MIN_SPEECH_SECONDS = 0.15
/** Shortest speech that starts an utterance while XERXES speaks (its own echo must not). */
export const VAD_MIN_SPEECH_WHILE_SPEAKING_SECONDS = 0.25
/** Consecutive low frames that end a speech-start candidate; one or two dips inside a word do not. */
export const VAD_START_HANGOVER_FRAMES = 3
/** After XERXES's turn goes idle with no reply, wait this long for the reply event before concluding. */
export const REPLY_IDLE_GRACE_MS = 3000
export const CALL_SEND_REFUSED = 'The message was not accepted. Check the XERXES thread before trying again.'
export const CALL_SEND_FAILED = 'The message could not be sent. Check the thread before trying again.'
export const CALL_SEND_UNCONFIRMED = 'The Hub did not confirm your message in time. Check the XERXES thread before saying it again.'
export const CALL_FINISHED_WITHOUT_REPLY = 'XERXES finished without a reply to that. Check the thread for details.'

/** One Moonshine Streaming encoder frame: 5 ms at 16 kHz. */
export const STT_FRAME_SAMPLES = 80

/**
 * Zero-pad audio to a whole number of encoder frames. Moonshine Streaming small (ONNX, WASM) throws a bare
 * numeric ONNX error for any other length, whatever the audio is (measured: every failing clip passed once
 * padded; one extra sample broke a passing clip). The VAD cuts in 512-sample frames plus pre-roll, so most
 * real utterances would otherwise crash.
 */
export function padToSttFrames(samples: Float32Array, frame = STT_FRAME_SAMPLES): Float32Array {
  const length = Math.ceil(samples.length / frame) * frame
  if (length === samples.length) return samples
  const padded = new Float32Array(length)
  padded.set(samples)
  return padded
}

export function chooseCallTier(memory: number | undefined, requested?: VoiceModelTier): VoiceModelTier {
  return requested ?? (memory !== undefined && memory < 4 ? 'light' : 'standard')
}

/** All fetches, including Kokoro's hard-coded voice URL, stay on this origin. Redirects must also be refused by the adapter. */
export function callFetchUrl(input: string, origin: string): string {
  const url = new URL(input, origin)
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Call mode refused a cross-origin request.')
  if (url.origin !== origin) {
    if (url.protocol !== 'https:' || url.hostname !== 'huggingface.co' || url.port !== '' || url.username !== '' || url.password !== '') {
      throw new Error('Call mode refused a cross-origin request.')
    }
    url.pathname = `/voice-models${url.pathname}`
  }
  if (url.pathname.startsWith('/voice-models/')) {
    const resolved = resolveVoiceModelFile(decodeURIComponent(url.pathname.slice('/voice-models/'.length)))
    if (resolved) url.pathname = voiceModelFileUrlPath(resolved.model, resolved.file.path)
  }
  // Optional transformer metadata may be absent from the allowlist: the Hub returns 404, never a remote fallback.
  const local = new URL(origin)
  local.pathname = url.pathname
  local.search = url.search
  return local.href
}

/** Browser-independent fetch adapter: credentials stay local and redirects cannot escape the policy. */
export type CallFetch = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>

export function createCallFetch(origin: string, nativeFetch: CallFetch): CallFetch {
  return async (input, init) => {
    const url = callFetchUrl(input instanceof Request ? input.url : String(input), origin)
    return nativeFetch(url, { ...(input instanceof Request ? { method: input.method, headers: input.headers, signal: input.signal } : {}), ...init, credentials: 'same-origin', redirect: 'error' })
  }
}

/** Streaming area resampler. Fractional samples carry across worklet callbacks (including at 44.1 kHz). */
export class CallResampler {
  private sum = 0
  private weight = 0
  private readonly ratio: number
  constructor(inputRate: number, outputRate = CALL_SAMPLE_RATE) {
    if (!Number.isFinite(inputRate) || inputRate <= 0 || !Number.isFinite(outputRate) || outputRate <= 0) throw new Error('Invalid audio sample rate.')
    this.ratio = inputRate / outputRate
  }
  push(input: Float32Array): Float32Array {
    const out: number[] = []
    for (const sample of input) {
      let remaining = 1
      while (remaining > 1e-9) {
        const take = Math.min(remaining, this.ratio - this.weight)
        this.sum += sample * take
        this.weight += take
        remaining -= take
        if (this.weight >= this.ratio - 1e-9) {
          out.push(this.sum / this.ratio)
          this.sum = 0
          this.weight = 0
        }
      }
    }
    return Float32Array.from(out)
  }
}

export class VadFrames {
  private pending = new Float32Array(VAD_FRAME_SAMPLES)
  private count = 0
  push(input: Float32Array): Float32Array[] {
    const out: Float32Array[] = []
    let offset = 0
    while (offset < input.length) {
      const take = Math.min(input.length - offset, VAD_FRAME_SAMPLES - this.count)
      this.pending.set(input.subarray(offset, offset + take), this.count)
      offset += take
      this.count += take
      if (this.count === VAD_FRAME_SAMPLES) {
        out.push(this.pending)
        this.pending = new Float32Array(VAD_FRAME_SAMPLES)
        this.count = 0
      }
    }
    return out
  }
}

/** Offsets in the exact returned 16 kHz clip, including its pre-roll. End is exclusive. */
export interface VadUtteranceOffsets {
  readonly startSample: number
  readonly endSample: number
  readonly preRollSamples: number
}
export type VadStep = { readonly hearing: boolean; readonly bargeIn: boolean } & (
  | { readonly utterance: Float32Array; readonly offsets: VadUtteranceOffsets }
  | { readonly utterance?: undefined; readonly offsets?: undefined }
)

/** Silero probabilities enter here; no browser or inference dependency. All timing is counted in 16 kHz samples. */
export class VadSegmenter {
  private pre = new Float32Array(0)
  private chunks: Float32Array[] = []
  private length = 0
  private preRollSamples = 0
  private candidate = 0
  private starting = false
  private lowRun = 0
  private silence = 0
  private speakingRun = 0
  private active = false
  reset(): void {
    this.pre = new Float32Array(0)
    this.chunks = []
    this.length = this.preRollSamples = this.candidate = this.lowRun = this.silence = this.speakingRun = 0
    this.active = this.starting = false
  }
  push(frame: Float32Array, probability: number, speaking: boolean): VadStep {
    if (frame.length !== VAD_FRAME_SAMPLES || !Number.isFinite(probability)) throw new Error('Invalid VAD frame.')
    // While XERXES speaks, its own voice can leak from the speaker into the mic (phone echo cancellation
    // is weak for Web Audio playback). Only clear, sustained speech interrupts; tapping the orb always does.
    this.speakingRun = speaking && probability >= BARGE_IN_PROBABILITY ? this.speakingRun + frame.length : 0
    const bargeIn = this.speakingRun >= BARGE_IN_SECONDS * CALL_SAMPLE_RATE
    if (bargeIn) this.speakingRun = 0
    if (!this.active) {
      if (probability >= 0.5 || this.starting) {
        if (probability >= 0.5) {
          if (!this.starting) {
            this.starting = true
            this.chunks = [this.pre]
            this.length = this.pre.length
            this.preRollSamples = this.pre.length
          }
          this.lowRun = 0
          this.candidate += frame.length
        } else {
          // A dip inside a word ("y-e-s") does not reset the start; a few low frames in a row do.
          this.lowRun += 1
        }
        if (this.lowRun >= VAD_START_HANGOVER_FRAMES) {
          this.starting = false
          this.candidate = this.lowRun = this.length = 0
          this.chunks = []
        } else {
          this.chunks.push(frame)
          this.length += frame.length
          const minimum = speaking ? VAD_MIN_SPEECH_WHILE_SPEAKING_SECONDS : VAD_MIN_SPEECH_SECONDS
          if (this.candidate >= minimum * CALL_SAMPLE_RATE) this.active = true
        }
      }
      const pre = new Float32Array(this.pre.length + frame.length)
      pre.set(this.pre)
      pre.set(frame, this.pre.length)
      this.pre = pre.slice(-0.2 * CALL_SAMPLE_RATE)
      return { hearing: this.active, bargeIn }
    }
    this.chunks.push(frame)
    this.length += frame.length
    this.silence = probability < 0.35 ? this.silence + frame.length : 0
    if (this.silence < 0.7 * CALL_SAMPLE_RATE && this.length < 30 * CALL_SAMPLE_RATE) return { hearing: true, bargeIn }
    const utterance = new Float32Array(Math.min(this.length, 30 * CALL_SAMPLE_RATE))
    let offset = 0
    for (const chunk of this.chunks) {
      const take = Math.min(chunk.length, utterance.length - offset)
      utterance.set(chunk.subarray(0, take), offset)
      offset += take
      if (offset === utterance.length) break
    }
    const offsets: VadUtteranceOffsets = {
      startSample: this.preRollSamples,
      endSample: Math.min(utterance.length, this.length - this.silence),
      preRollSamples: this.preRollSamples,
    }
    this.reset()
    return { hearing: false, bargeIn, utterance, offsets }
  }
}

export interface CallReply {
  readonly id: string
  readonly role: string
  readonly text: string
  readonly causalMessageId?: string
  readonly threadId?: string
  readonly origin?: string
}

/** The Hub's notice for a XERXES turn that failed ("XERXES stopped: <why>"). */
export const CALL_TURN_STOPPED_PREFIX = 'XERXES stopped'

/**
 * The reply to `messageId`: XERXES's answer, or the Hub's "XERXES stopped: …" notice when the turn failed
 * (returned with role 'system', so the call speaks the failure instead of waiting 120 s in silence).
 */
export function matchCallReply(messages: readonly CallReply[], baseline: ReadonlySet<string>, messageId: string, threadId?: string): CallReply | undefined {
  return messages.find(message => (
    (message.role === 'xerxes' && message.origin !== 'wake') || (message.role === 'system' && message.text.startsWith(CALL_TURN_STOPPED_PREFIX))
  ) && !baseline.has(message.id)
    && (threadId === undefined || message.threadId === undefined || message.threadId === threadId)
    && (message.causalMessageId === undefined || message.causalMessageId === messageId))
}

/** The message never reached XERXES (refused, or the request failed). The call keeps running. */
export class CallSendError extends CallStepError {
  constructor(message: string, readonly kind: 'refused' | 'failed' | 'unconfirmed') {
    super(message)
    this.name = 'CallSendError'
  }
}

/** Observe before submit, so a reply delivered before the submit acknowledgement is not lost. */
export function waitForCallReply(input: {
  readonly messageId: string
  readonly threadId?: string
  readonly messages: () => readonly CallReply[]
  readonly subscribe: (listener: () => void) => () => void
  readonly send: () => Promise<boolean>
  readonly signal: AbortSignal
  readonly timeoutMs?: number
  /** Keep observing late replies after this notice; the call aborts this wait when the person speaks again. */
  readonly onTimeout?: () => void
  /**
   * Whether XERXES is still running the turn (undefined when unknown). After the send is accepted, `false`
   * for longer than `idleGraceMs` with no reply ends the wait with a visible error instead of silence.
   */
  readonly working?: () => boolean | undefined
  readonly idleGraceMs?: number
  /** Longest wait for the submit acknowledgement; a lost acknowledgement must not hang the call. */
  readonly sendTimeoutMs?: number
}): Promise<CallReply> {
  const baseline = new Set(input.messages().map(message => message.id))
  return new Promise((resolve, reject) => {
    let settled = false
    let accepted = false
    let idleTimer: ReturnType<typeof setTimeout> | undefined
    let unsubscribe = (): void => undefined
    const stopIdle = (): void => { if (idleTimer !== undefined) clearTimeout(idleTimer); idleTimer = undefined }
    const finish = (error?: Error, reply?: CallReply): void => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      clearTimeout(sendTimer)
      stopIdle()
      unsubscribe()
      input.signal.removeEventListener('abort', abort)
      if (error) reject(error)
      else if (reply) resolve(reply)
    }
    const abort = (): void => finish(new Error('Call cancelled.'))
    const check = (): void => {
      if (!accepted || settled) return
      const reply = matchCallReply(input.messages(), baseline, input.messageId, input.threadId)
      if (reply) { finish(undefined, reply); return }
      if (input.working?.() !== false) { stopIdle(); return }
      // The turn ended; the reply event may still be on its way, so wait a moment before concluding.
      idleTimer ??= setTimeout(() => {
        idleTimer = undefined
        const late = matchCallReply(input.messages(), baseline, input.messageId, input.threadId)
        if (late) finish(undefined, late)
        else if (input.working?.() === false) finish(new CallStepError(CALL_FINISHED_WITHOUT_REPLY))
      }, input.idleGraceMs ?? REPLY_IDLE_GRACE_MS)
    }
    const timer = setTimeout(() => {
      if (input.onTimeout) input.onTimeout()
      else finish(new Error('XERXES did not reply within 120 seconds. The message remains in your thread.'))
    }, input.timeoutMs ?? REPLY_TIMEOUT_MS)
    const sendTimer = setTimeout(() => {
      if (!accepted) finish(new CallSendError(CALL_SEND_UNCONFIRMED, 'unconfirmed'))
    }, input.sendTimeoutMs ?? REPLY_SEND_TIMEOUT_MS)
    input.signal.addEventListener('abort', abort, { once: true })
    if (input.signal.aborted) { abort(); return }
    unsubscribe = input.subscribe(check)
    void input.send().then(ok => {
      if (!ok) finish(new CallSendError(CALL_SEND_REFUSED, 'refused'))
      else { accepted = true; clearTimeout(sendTimer); check() }
    }, () => finish(new CallSendError(CALL_SEND_FAILED, 'failed')))
  })
}

export interface CallModelsStatus {
  readonly standard: { readonly bytes: number; readonly ready: boolean }
  readonly light: { readonly bytes: number; readonly ready: boolean }
}

export async function fetchCallModelsStatus(fetcher: typeof fetch = fetch, signal?: AbortSignal): Promise<CallModelsStatus> {
  const response = await fetcher('/api/hub/v1/voice/models', { credentials: 'same-origin', ...(signal ? { signal } : {}) })
  if (response.status === 404) throw new Error('This Hub does not support on-phone call models yet. Update the Hub to use Call.')
  if (!response.ok) throw new Error('Call models are unavailable on this Hub. Try again later.')
  const raw: unknown = await response.json()
  if (typeof raw !== 'object' || raw === null || !('enabled' in raw) || raw.enabled !== true) throw new Error('On-phone call models are disabled on this Hub.')
  const tiers = 'tiers' in raw && typeof raw.tiers === 'object' && raw.tiers !== null ? raw.tiers : {}
  const tier = (key: 'standard' | 'light'): { bytes: number; ready: boolean } => {
    const value: unknown = key === 'standard' ? ('standard' in tiers ? tiers.standard : undefined) : ('light' in tiers ? tiers.light : undefined)
    if (typeof value !== 'object' || value === null || !('bytes' in value) || typeof value.bytes !== 'number' || !Number.isFinite(value.bytes) || value.bytes <= 0 || !('ready' in value) || typeof value.ready !== 'boolean') throw new Error('This Hub returned an invalid call model status.')
    return { bytes: value.bytes, ready: value.ready }
  }
  return { standard: tier('standard'), light: tier('light') }
}
