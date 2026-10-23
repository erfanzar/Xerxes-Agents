// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { AuthenticatedContext } from '../identityPort.js'
import { estimateUsd, type Reservation, type VoiceBudget } from './budgets.js'
import { allowedModes, buildCapabilities, ttsCodec, type VoiceRuntimeConfig } from './config.js'
import { FrameGate } from './frameGate.js'
import { bytesToPcm16, pcm16DurationMs, pcm16ToBytes, resamplePcm16 } from './pcm.js'
import { speechChunks, type RealtimeEvent, type RealtimeToolSpec, type RealtimeVoiceSession } from './ports.js'
import {
  RECORDER_CODECS,
  VOICE_LIMITS,
  VoiceError,
  decodeVoiceFrame,
  encodeVoiceFrame,
  encodeVoiceServerMessage,
  mediaTypeForCodec,
  parseVoiceClientMessage,
  type VoiceClientMessage,
  type VoiceCodec,
  type VoiceFrame,
  type VoiceServerMessage,
  type VoiceStateName,
} from './protocol.js'
import type { LeaseHolder, SpeakerLeases } from './speakerLease.js'
import { prepareSpeech } from './speech.js'
import type { ExactlyOnceSubmitter, SupervisorReply, SupervisorVoiceBridge } from './supervisorBridge.js'
import type { VoiceTranscriptRecord } from './transcripts.js'

/**
 * One voice connection. It owns the state machine for push-to-talk
 * (record, transcribe once, optional edit, submit, speak the reply) and for the
 * live relay (provider VAD, barge-in, bridge functions). It never persists
 * audio, never replays audio after a reconnect (a new socket is a new
 * session), and never cancels agents: interrupt only flushes playback.
 */

/** How the session talks to its socket. The route adapts a Bun WebSocket to this. */
export interface VoiceTransport {
  /** False when the frame was dropped. */
  sendText(text: string): boolean
  sendBinary(bytes: Uint8Array): boolean
  bufferedAmount(): number
  close(code: number, reason: string): void
}

export interface VoiceSessionOptions {
  readonly id: string
  readonly context: AuthenticatedContext
  readonly config: VoiceRuntimeConfig
  readonly supervisor: SupervisorVoiceBridge
  /** Shared by every session of a route, so a resubmission from a new socket is still deduplicated. */
  readonly submitter: ExactlyOnceSubmitter
  readonly leases: SpeakerLeases
  readonly budget: VoiceBudget
  readonly clock: () => number
  readonly transport: VoiceTransport
  readonly onError?: (error: unknown, context: string) => void
}

const ARM_MS = 120_000
const MAX_TRANSCRIBED = 16
const MAX_PROTOCOL_STRIKES = 5
const SPEECH_CHUNK_BYTES = 32 * 1024
const MAX_TOOL_CALLS = 40
const MAX_REPLY_TEXT = 8000

export const REALTIME_INSTRUCTIONS = [
  'You are the voice of XERXES, the supervisor of the user\'s coding agents. You are a voice front end, not a second supervisor.',
  'When the user finishes asking for something, call supervisor_submit exactly once for that request. Do not paraphrase it and do not answer it yourself.',
  'Only report what supervisor_status or a supervisor message tells you. Never claim work was done, started or stopped unless a function result says so.',
  'Call supervisor_cancel only when the user clearly asks to stop XERXES\'s current reply. It does not stop background agents, and you must say so if asked.',
  'You cannot approve, confirm, review or grant anything. If asked, tell the user to do it on the screen.',
  'Keep spoken answers short. Read supervisor messages aloud word for word when asked to.',
].join('\n')

const EMPTY_OBJECT = Object.freeze({ type: 'object', properties: {}, additionalProperties: false })
export const REALTIME_TOOLS: readonly RealtimeToolSpec[] = Object.freeze([
  { name: 'supervisor_submit', description: 'Send the user\'s most recent finished utterance, verbatim, to XERXES. Call once per request.', parameters: EMPTY_OBJECT },
  { name: 'supervisor_status', description: 'Read whether XERXES is busy and how many agents are working.', parameters: EMPTY_OBJECT },
  { name: 'supervisor_cancel', description: 'Stop XERXES\'s current reply on this conversation. Does not stop background agents.', parameters: EMPTY_OBJECT },
])

interface ActiveUtterance {
  readonly id: string
  readonly generation: number
  readonly codec: VoiceCodec
  readonly language: string | undefined
  readonly gate: FrameGate
  readonly abort: AbortController
  readonly startedAt: number
  chunks: Uint8Array[]
  phase: 'recording' | 'transcribing'
}

interface ActiveResponse {
  readonly id: string
  readonly generation: number
  readonly abort: AbortController
  readonly providerId: string | undefined
  readonly startedAt: number
  itemId: string | undefined
  sequence: number
  sentMs: number
}

interface LiveState {
  readonly rt: RealtimeVoiceSession
  readonly gate: FrameGate
  readonly clientRate: number
  readonly providerRate: number
  readonly outputRate: number
  readonly reservations: Reservation[]
  readonly startedAt: number
  readonly off: () => void
  readonly revisions: Map<string, number>
  readonly cancelled: Set<string>
  deadline: number
  lastFinal: { readonly id: string; readonly text: string } | undefined
  toolCalls: number
}

type StartMessage = Extract<VoiceClientMessage, { type: 'start' }>

export class VoiceSession {
  private counter = 0
  private responseSeq = 0
  private conversationId: string | undefined
  private offReply: (() => void) | undefined
  private utterance: ActiveUtterance | undefined
  private response: ActiveResponse | undefined
  private live: LiveState | undefined
  private liveAbort: AbortController | undefined
  private starting = false
  private closed = false
  private strikes = 0
  private lastActivity: number
  private lastState = ''
  private armedUntil = 0
  private armedMessageId: string | undefined
  private readonly transcribed = new Map<string, { readonly text: string; readonly provider: string; readonly model: string }>()

  constructor(private readonly options: VoiceSessionOptions) {
    this.lastActivity = options.clock()
  }

  get id(): string { return this.options.id }
  get isClosed(): boolean { return this.closed }
  get principalId(): string { return this.options.context.principal.id }
  get browserSessionId(): string { return this.options.context.browserSessionId }
  get deviceId(): string { return this.holder().deviceId }
  get activeConversation(): string | undefined { return this.conversationId }

  // ------------------------------------------------------------------ inbound

  async handleText(text: string): Promise<void> {
    if (this.closed) return
    let message: VoiceClientMessage
    try {
      message = parseVoiceClientMessage(text)
    } catch (error) {
      this.protocolFailure(error)
      return
    }
    this.lastActivity = this.options.clock()
    try {
      switch (message.type) {
        case 'capabilities': this.send({ type: 'capabilities', capabilities: buildCapabilities(this.options.config) }); return
        case 'start': await this.onStart(message); return
        case 'commit': await this.onCommit(message); return
        case 'submit': await this.onSubmit(message); return
        case 'interrupt': this.onInterrupt(message); return
        case 'renew': this.onRenew(); return
        case 'expect_reply': await this.onExpectReply(message); return
        case 'stop': this.end('stopped'); return
      }
    } catch (error) {
      this.fail(error)
    }
  }

  handleBinary(bytes: Uint8Array): void {
    if (this.closed) return
    let frame: VoiceFrame
    try {
      frame = decodeVoiceFrame(bytes)
    } catch (error) {
      this.send(errorMessage(error instanceof VoiceError ? error : new VoiceError('VOICE_PROTOCOL', 'malformed audio frame')))
      this.abortStream()
      return
    }
    const live = this.live
    const utterance = this.utterance?.phase === 'recording' ? this.utterance : undefined
    const gate = live?.gate ?? utterance?.gate
    if (gate === undefined) {
      this.send(errorMessage(new VoiceError('VOICE_STALE', 'no audio stream is open for this frame')))
      return
    }
    const conversation = this.conversationId
    if (conversation === undefined || !this.options.leases.isHolder(conversation, this.holder())) {
      this.send(errorMessage(new VoiceError('VOICE_PERMISSION', 'you no longer hold the speaker lease for this conversation')))
      this.abortStream()
      return
    }
    const verdict = gate.check(frame)
    if (!verdict.ok) {
      this.send(errorMessage(verdict.error))
      if (verdict.fatal) this.abortStream()
      return
    }
    this.options.leases.renew(conversation, this.holder())
    if (live !== undefined) {
      this.feedLive(live, frame)
      return
    }
    this.lastActivity = this.options.clock()
    utterance?.chunks.push(frame.payload.slice())
  }

  /** Socket closed by the peer or the server. Nothing is sent. */
  dispose(): void {
    this.teardown()
  }

  /** Periodic housekeeping: idle and session limits, lease keep-alive for live sessions. */
  tick(): void {
    if (this.closed) return
    const now = this.options.clock()
    const { settings } = this.options.config
    const live = this.live
    if (live !== undefined) {
      if (now >= live.deadline) { this.sendError(new VoiceError('VOICE_LIMIT', 'the live session reached its time limit; start a new one')); this.end('session limit'); return }
      if (now - this.lastActivity > settings.idleSeconds * 1000) { this.end('idle'); return }
      if (this.conversationId !== undefined) this.options.leases.renew(this.conversationId, this.holder())
      return
    }
    const utterance = this.utterance
    if (utterance?.phase === 'recording' && now - utterance.startedAt > settings.maxUtteranceSeconds * 1000 + 5000) {
      this.sendError(new VoiceError('VOICE_LIMIT', 'the recording exceeded the time limit'))
      this.abortStream()
      return
    }
    if (now - this.lastActivity > settings.pttIdleSeconds * 1000) this.end('idle')
  }

  // ------------------------------------------------------------------ start

  private async onStart(message: StartMessage): Promise<void> {
    const { config } = this.options
    const { settings } = config
    if (!settings.enabled) throw new VoiceError('VOICE_UNAVAILABLE', 'voice is disabled on this hub')
    if (!allowedModes(config).includes(message.mode)) {
      throw new VoiceError('VOICE_UNAVAILABLE', message.mode === 'live' ? 'live voice is not configured on this hub' : 'speech-to-text is not configured on this hub')
    }
    if (this.options.context.principal.kind !== 'user') throw new VoiceError('VOICE_PERMISSION', 'voice is only available to signed-in people')
    if (this.starting || this.utterance !== undefined || this.live !== undefined) throw new VoiceError('VOICE_BUSY', 'a recording or live session is already in progress on this connection')
    if (message.mode === 'push_to_talk' && !RECORDER_CODECS.includes(message.codec)) throw new VoiceError('VOICE_FORMAT', 'unsupported recording format')
    if (message.mode === 'live') {
      if (message.codec !== 'pcm16') throw new VoiceError('VOICE_FORMAT', 'live voice needs mono PCM16 audio')
      if (message.sample_rate === undefined) throw new VoiceError('VOICE_PROTOCOL', 'live voice needs a sample_rate')
    }
    this.starting = true
    try {
      const allowed = await this.options.supervisor.authorize(message.conversation_id, this.options.context.actor)
      if (this.closed) return
      if (!allowed) throw new VoiceError('VOICE_PERMISSION', 'you cannot use voice on this conversation')
      this.bindConversation(message.conversation_id)
      this.takeLease(message.conversation_id)
      if (message.mode === 'push_to_talk') this.beginUtterance(message)
      else await this.beginLive(message)
    } finally {
      this.starting = false
    }
  }

  private bindConversation(conversationId: string): void {
    if (this.conversationId === conversationId) return
    this.offReply?.()
    if (this.conversationId !== undefined) this.options.leases.release(this.conversationId, this.holder())
    this.conversationId = conversationId
    this.offReply = this.options.supervisor.onReply(conversationId, reply => { void this.onReply(reply).catch(error => this.fail(error)) })
  }

  private takeLease(conversationId: string): void {
    const outcome = this.options.leases.acquire(conversationId, this.holder())
    this.send({ type: 'lease', conversation_id: conversationId, held_by_you: outcome.ok, expires_at: new Date(outcome.expiresAt).toISOString() })
    if (!outcome.ok) throw new VoiceError('VOICE_BUSY', 'someone else is speaking on this conversation')
  }

  private beginUtterance(message: StartMessage): void {
    const { settings } = this.options.config
    const generation = ++this.counter
    this.utterance = {
      id: message.utterance_id,
      generation,
      codec: message.codec,
      language: message.language ?? settings.language,
      gate: new FrameGate({ generation, codec: message.codec, maxBytes: settings.maxUtteranceBytes, maxSeconds: settings.maxUtteranceSeconds, now: this.options.clock }),
      abort: new AbortController(),
      startedAt: this.options.clock(),
      chunks: [],
      phase: 'recording',
    }
    this.setState('listening', generation, { utterance_id: message.utterance_id })
  }

  // ------------------------------------------------------------------ push-to-talk

  private async onCommit(message: Extract<VoiceClientMessage, { type: 'commit' }>): Promise<void> {
    const utterance = this.utterance
    if (utterance === undefined || utterance.id !== message.utterance_id || utterance.phase !== 'recording') {
      throw new VoiceError('VOICE_STALE', 'there is no matching recording to finish')
    }
    const { config } = this.options
    const stt = config.stt
    if (stt === undefined) { this.clearUtterance(utterance); throw new VoiceError('VOICE_UNAVAILABLE', 'speech-to-text is not configured on this hub') }
    utterance.phase = 'transcribing'
    this.setState('transcribing', utterance.generation, { utterance_id: utterance.id })
    const audio = concat(utterance.chunks)
    utterance.chunks = []
    if (audio.byteLength === 0) {
      this.send({ type: 'transcript', utterance_id: utterance.id, revision: 1, text: '', final: true })
      this.clearUtterance(utterance)
      return
    }
    // The sender's timestamps are untrusted; bill for the longer of its claim and the Hub's own clock.
    const minutes = Math.max(utterance.gate.durationMs, this.options.clock() - utterance.startedAt) / 60_000
    let reservation: Reservation
    try {
      reservation = this.options.budget.reserve(this.principalId, this.id, 'stt', estimateUsd(config.settings.pricing, 'stt', minutes))
    } catch (error) {
      this.clearUtterance(utterance)
      throw error
    }
    let result
    try {
      result = await stt.transcribe({ audio, mediaType: mediaTypeForCodec(utterance.codec), ...(utterance.language === undefined ? {} : { language: utterance.language }) }, utterance.abort.signal)
    } catch (error) {
      if (utterance.abort.signal.aborted) this.options.budget.release(reservation)
      else this.options.budget.settle(reservation)
      if (this.utterance === utterance) this.clearUtterance(utterance)
      this.sendUsage()
      if (error instanceof VoiceError) throw error
      throw new VoiceError('VOICE_PROVIDER', 'the transcription provider failed', error)
    }
    this.options.budget.settle(reservation)
    this.sendUsage()
    // An interrupt or stop that arrived while the provider was working discards the result unseen.
    if (this.utterance !== utterance || this.closed) return
    // Trim here as well as in the adapter: a port that returns padding must not produce a "heard" transcript.
    const text = result.text.trim()
    this.send({ type: 'transcript', utterance_id: utterance.id, revision: 1, text, final: true })
    this.clearUtterance(utterance)
    if (text === '') return // empty recognition is never submitted
    this.transcribed.set(utterance.id, { text, provider: result.provider, model: result.model })
    while (this.transcribed.size > MAX_TRANSCRIBED) {
      const oldest = this.transcribed.keys().next().value
      if (oldest === undefined) break
      this.transcribed.delete(oldest)
    }
    this.persist({
      kind: 'transcript', conversationId: this.conversationId ?? '', utteranceId: utterance.id, userId: this.principalId, deviceId: this.holder().deviceId,
      displayName: this.options.context.actor.display_name, mode: 'push_to_talk', text, provider: result.provider, model: result.model, at: this.iso(),
    })
    if (message.submit === true && !config.settings.confirmTranscript) await this.submitUtterance(utterance.id, text, text)
  }

  private async onSubmit(message: Extract<VoiceClientMessage, { type: 'submit' }>): Promise<void> {
    const recognized = this.transcribed.get(message.utterance_id)
    if (recognized === undefined) throw new VoiceError('VOICE_STALE', 'that transcript is no longer available; send the text as a normal message')
    await this.submitUtterance(message.utterance_id, message.text, recognized.text)
  }

  private async submitUtterance(utteranceId: string, text: string, recognized: string): Promise<void> {
    const conversation = this.conversationId
    if (conversation === undefined) throw new VoiceError('VOICE_STALE', 'no conversation is open')
    const actor = this.options.context.actor
    if (!(await this.options.supervisor.authorize(conversation, actor))) throw new VoiceError('VOICE_PERMISSION', 'you cannot send to this conversation', undefined)
    this.takeLease(conversation)
    // Arm before submitting: the supervisor's reply can arrive before the submit call returns.
    this.armedUntil = this.options.clock() + ARM_MS
    this.armedMessageId = undefined
    let outcome
    try {
      outcome = await this.options.submitter.submit(conversation, text, actor, `voice:${utteranceId}`)
    } catch (error) {
      this.armedUntil = 0
      throw error
    }
    if (!outcome.accepted) {
      this.armedUntil = 0
      throw new VoiceError('VOICE_PERMISSION', 'the supervisor did not accept the message')
    }
    this.send({ type: 'submitted', utterance_id: utteranceId, duplicate: outcome.duplicate, ...(outcome.messageId === undefined ? {} : { message_id: outcome.messageId }) })
    if (outcome.duplicate) { this.armedUntil = 0; return }
    this.armedMessageId = outcome.messageId
    this.persist({
      kind: 'submitted', conversationId: conversation, utteranceId, userId: this.principalId, deviceId: this.holder().deviceId,
      displayName: actor.display_name, mode: this.live === undefined ? 'push_to_talk' : 'live', text, edited: text !== recognized, at: this.iso(),
    })
    this.setState('thinking', this.counter)
  }

  private async onExpectReply(message: Extract<VoiceClientMessage, { type: 'expect_reply' }>): Promise<void> {
    if (message.enabled === false) { this.armedUntil = 0; return }
    if (message.conversation_id !== undefined) {
      // A fresh socket (the old one idled out) names its conversation here; it is authorized like a start.
      if (!this.options.config.settings.enabled) throw new VoiceError('VOICE_UNAVAILABLE', 'voice is disabled on this hub')
      if (this.options.context.principal.kind !== 'user') throw new VoiceError('VOICE_PERMISSION', 'voice is only available to signed-in people')
      if (!(await this.options.supervisor.authorize(message.conversation_id, this.options.context.actor))) {
        throw new VoiceError('VOICE_PERMISSION', 'you cannot use voice on this conversation')
      }
      if (this.closed) return
      this.bindConversation(message.conversation_id)
    }
    if (this.conversationId === undefined) throw new VoiceError('VOICE_STALE', 'no conversation is open')
    this.armedUntil = this.options.clock() + ARM_MS
    this.armedMessageId = undefined
  }

  // ------------------------------------------------------------------ interrupt / renew

  private onInterrupt(message: Extract<VoiceClientMessage, { type: 'interrupt' }>): void {
    const response = this.response
    if (response !== undefined && (message.response_id === undefined || message.response_id === response.id)) {
      this.endResponse(response, true)
      const live = this.live
      if (live !== undefined) {
        live.rt.cancelResponse()
        if (response.providerId !== undefined) live.cancelled.add(response.providerId)
        if (live.rt.supportsTruncate && response.itemId !== undefined) live.rt.truncate(response.itemId, message.played_ms ?? response.sentMs)
      }
      this.persist({
        kind: 'interruption', conversationId: this.conversationId ?? '', responseId: response.id, userId: this.principalId, deviceId: this.holder().deviceId,
        ...(message.played_ms === undefined ? {} : { playedMs: message.played_ms }), at: this.iso(),
      })
      this.setState('interrupted', this.counter)
    }
    // Interrupting also abandons a recording or transcription in progress. It never cancels agents.
    if (this.utterance !== undefined) this.clearUtterance(this.utterance)
    this.armedUntil = 0
  }

  private onRenew(): void {
    const conversation = this.conversationId
    if (conversation === undefined) throw new VoiceError('VOICE_STALE', 'no conversation is open')
    this.takeLease(conversation)
    const live = this.live
    if (live === undefined) return
    const { settings } = this.options.config
    const reservation = this.options.budget.reserve(this.principalId, this.id, 'realtime', estimateUsd(settings.pricing, 'realtime', settings.maxSessionSeconds / 60))
    live.reservations.push(reservation)
    live.deadline = this.options.clock() + settings.maxSessionSeconds * 1000
    live.gate.extend()
    this.sendUsage()
  }

  // ------------------------------------------------------------------ replies

  private async onReply(reply: SupervisorReply): Promise<void> {
    if (this.closed || this.armedUntil <= this.options.clock()) return
    if (reply.inReplyTo !== undefined && this.armedMessageId !== undefined && reply.inReplyTo !== this.armedMessageId) return
    this.armedUntil = 0
    this.armedMessageId = undefined
    if (this.live !== undefined) { this.liveReply(this.live, reply); return }
    await this.speakReply(reply)
  }

  private async speakReply(reply: SupervisorReply): Promise<void> {
    const { config } = this.options
    const tts = config.tts
    const sentences = prepareSpeech(reply.text, VOICE_LIMITS.maxSpokenChars)
    const previous = this.response
    if (previous !== undefined) this.endResponse(previous, true)
    const response: ActiveResponse = {
      id: `${this.id}-r${++this.responseSeq}`, generation: ++this.counter, abort: new AbortController(),
      providerId: undefined, startedAt: this.options.clock(), itemId: undefined, sequence: 0, sentMs: 0,
    }
    let reservation: Reservation | undefined
    if (tts !== undefined && sentences.length > 0) {
      const chars = sentences.reduce((sum, sentence) => sum + sentence.length, 0)
      try {
        reservation = this.options.budget.reserve(this.principalId, this.id, 'tts', estimateUsd(config.settings.pricing, 'tts', chars))
      } catch (error) {
        this.fail(error)
      }
    }
    const text = reply.text.slice(0, MAX_REPLY_TEXT)
    this.send({ type: 'reply', response_id: response.id, generation: response.generation, text, spoken: reservation !== undefined, ...(reply.inReplyTo === undefined ? {} : { message_id: reply.inReplyTo }) })
    if (tts === undefined || reservation === undefined) { this.setState('idle', this.counter); return }
    this.response = response
    this.setState('buffering', response.generation, { response_id: response.id })
    this.send({ type: 'audio_start', response_id: response.id, generation: response.generation, delivery: tts.delivery, codec: ttsCodec(config.settings) })
    let first = true
    try {
      for (let index = 0; index < sentences.length; index += 1) {
        const sentence = sentences[index] ?? ''
        const last = index === sentences.length - 1
        let codec: VoiceCodec = ttsCodec(config.settings)
        for await (const chunk of speechChunks(tts, { text: sentence, format: config.settings.ttsFormat }, response.abort.signal, SPEECH_CHUNK_BYTES)) {
          if (this.response !== response) return
          codec = chunk.codec
          if (!this.sendAudio(response, codec, chunk.bytes, false, false)) { this.overloaded(response); return }
          if (first) { first = false; this.setState('speaking', response.generation, { response_id: response.id }) }
        }
        if (this.response !== response) return
        if (!this.sendAudio(response, codec, new Uint8Array(0), last, true)) { this.overloaded(response); return }
      }
      if (this.response === response) {
        this.response = undefined
        this.send({ type: 'audio_end', response_id: response.id, generation: response.generation, interrupted: false })
        if (this.utterance === undefined) this.setState('idle', this.counter)
      }
    } catch (error) {
      if (this.response === response) {
        this.endResponse(response, true)
        this.fail(error)
      }
    } finally {
      this.options.budget.settle(reservation)
      this.sendUsage()
    }
  }

  private sendAudio(response: ActiveResponse, codec: VoiceCodec, payload: Uint8Array, final: boolean, clipEnd: boolean): boolean {
    const bytes = encodeVoiceFrame(
      { stream: 'tts', codec, generation: response.generation, sequence: response.sequence, timestampMs: Math.max(0, this.options.clock() - response.startedAt), final, clipEnd },
      payload,
    )
    // Backpressure: a phone that cannot keep up gets its stream ended, never an unbounded queue.
    if (this.options.transport.bufferedAmount() + bytes.byteLength > VOICE_LIMITS.maxBufferedBytes) return false
    if (!this.options.transport.sendBinary(bytes)) return false
    response.sequence += 1
    return true
  }

  private overloaded(response: ActiveResponse): void {
    this.send(errorMessage(new VoiceError('VOICE_OVERLOADED', 'this connection is too slow to stream audio; playback was ended'), { response_id: response.id }))
    this.endResponse(response, true)
  }

  private endResponse(response: ActiveResponse, interrupted: boolean): void {
    if (this.response === response) this.response = undefined
    response.abort.abort()
    this.send({ type: 'audio_end', response_id: response.id, generation: response.generation, interrupted })
  }

  // ------------------------------------------------------------------ live relay

  private async beginLive(message: StartMessage): Promise<void> {
    const { config } = this.options
    const { settings } = config
    const port = config.realtime
    const clientRate = message.sample_rate
    if (port === undefined || clientRate === undefined) throw new VoiceError('VOICE_UNAVAILABLE', 'live voice is not configured on this hub')
    const reservation = this.options.budget.reserve(this.principalId, this.id, 'realtime', estimateUsd(settings.pricing, 'realtime', settings.maxSessionSeconds / 60))
    const abort = new AbortController()
    this.liveAbort = abort
    const language = message.language ?? settings.language
    let rt: RealtimeVoiceSession
    try {
      rt = await port.connect({ instructions: REALTIME_INSTRUCTIONS, tools: REALTIME_TOOLS, sampleRate: port.sampleRate, ...(language === undefined ? {} : { language }) }, abort.signal)
    } catch (error) {
      this.options.budget.release(reservation)
      this.liveAbort = undefined
      if (error instanceof VoiceError) throw error
      throw new VoiceError('VOICE_PROVIDER', 'the realtime provider is unavailable', error)
    }
    if (this.closed || abort.signal.aborted) {
      rt.close()
      this.options.budget.release(reservation)
      return
    }
    const generation = ++this.counter
    const now = this.options.clock()
    const live: LiveState = {
      rt, clientRate, providerRate: port.sampleRate, outputRate: port.outputSampleRate ?? port.sampleRate,
      gate: new FrameGate({
        generation, codec: 'pcm16', maxBytes: Math.ceil(clientRate * 2 * settings.maxSessionSeconds * 1.25),
        maxSeconds: settings.maxSessionSeconds, now: this.options.clock,
      }),
      reservations: [reservation], startedAt: now, deadline: now + settings.maxSessionSeconds * 1000,
      off: rt.onEvent(event => this.onRealtimeEvent(event)),
      revisions: new Map(), cancelled: new Set(), lastFinal: undefined, toolCalls: 0,
    }
    this.live = live
    this.lastActivity = now
    this.setState('listening', generation, { utterance_id: message.utterance_id })
  }

  private feedLive(live: LiveState, frame: VoiceFrame): void {
    if (frame.payload.byteLength % 2 !== 0) {
      this.sendError(new VoiceError('VOICE_FORMAT', 'PCM16 audio must have an even byte length'))
      this.end('bad audio')
      return
    }
    const pcm = resamplePcm16(bytesToPcm16(frame.payload), live.clientRate, live.providerRate)
    if (!live.rt.sendAudio(pcm16ToBytes(pcm))) {
      this.sendError(new VoiceError('VOICE_OVERLOADED', 'the realtime provider connection is backed up; the session was ended'))
      this.end('overloaded')
    }
  }

  private onRealtimeEvent(event: RealtimeEvent): void {
    const live = this.live
    if (live === undefined || this.closed) return
    switch (event.type) {
      case 'ready': this.setState('listening', this.counter); return
      case 'speech_started': {
        this.lastActivity = this.options.clock()
        const response = this.response
        if (response !== undefined) this.bargeIn(live, response)
        this.setState('listening', this.counter)
        return
      }
      case 'speech_stopped': this.lastActivity = this.options.clock(); this.setState('thinking', this.counter); return
      case 'transcript': {
        this.lastActivity = this.options.clock()
        const revision = (live.revisions.get(event.itemId) ?? 0) + 1
        live.revisions.set(event.itemId, revision)
        if (live.revisions.size > 64) { const oldest = live.revisions.keys().next().value; if (oldest !== undefined) live.revisions.delete(oldest) }
        this.send({ type: 'transcript', utterance_id: event.itemId, revision, text: event.text, final: event.final })
        if (event.final && event.text !== '') {
          live.lastFinal = { id: event.itemId, text: event.text }
          this.persist({
            kind: 'transcript', conversationId: this.conversationId ?? '', utteranceId: event.itemId, userId: this.principalId, deviceId: this.holder().deviceId,
            displayName: this.options.context.actor.display_name, mode: 'live', text: event.text, provider: this.options.config.realtime?.providerName ?? 'realtime', at: this.iso(),
          })
        }
        return
      }
      case 'audio': this.onProviderAudio(live, event); return
      case 'audio_done': this.onProviderAudioDone(live, event.responseId); return
      case 'tool_call': this.lastActivity = this.options.clock(); void this.onToolCall(live, event.callId, event.name).catch(error => this.fail(error)); return
      case 'usage': return
      case 'error': this.sendError(event.error); this.end('provider error'); return
      case 'closed': this.end('provider closed'); return
    }
  }

  private bargeIn(live: LiveState, response: ActiveResponse): void {
    if (response.providerId !== undefined) live.cancelled.add(response.providerId)
    live.rt.cancelResponse()
    if (live.rt.supportsTruncate && response.itemId !== undefined) live.rt.truncate(response.itemId, response.sentMs)
    this.endResponse(response, true)
    this.persist({
      kind: 'interruption', conversationId: this.conversationId ?? '', responseId: response.id, userId: this.principalId, deviceId: this.holder().deviceId,
      playedMs: Math.floor(response.sentMs), at: this.iso(),
    })
  }

  private onProviderAudio(live: LiveState, event: Extract<RealtimeEvent, { type: 'audio' }>): void {
    if (live.cancelled.has(event.responseId)) return
    this.lastActivity = this.options.clock()
    let response = this.response
    if (response === undefined || response.providerId !== event.responseId) {
      if (response !== undefined) this.endResponse(response, true)
      response = {
        id: `${this.id}-r${++this.responseSeq}`, generation: ++this.counter, abort: new AbortController(), providerId: event.responseId,
        startedAt: this.options.clock(), itemId: undefined, sequence: 0, sentMs: 0,
      }
      this.response = response
      this.send({ type: 'audio_start', response_id: response.id, generation: response.generation, delivery: 'streaming', codec: 'pcm16', sample_rate: live.outputRate })
      this.setState('speaking', response.generation, { response_id: response.id })
    }
    if (event.itemId !== undefined) response.itemId = event.itemId
    for (let offset = 0; offset < event.audio.byteLength; offset += SPEECH_CHUNK_BYTES) {
      const part = event.audio.subarray(offset, offset + SPEECH_CHUNK_BYTES - ((SPEECH_CHUNK_BYTES - offset) % 2 === 0 ? 0 : 0))
      if (!this.sendAudio(response, 'pcm16', part, false, true)) {
        this.overloaded(response)
        live.rt.cancelResponse()
        live.cancelled.add(event.responseId)
        return
      }
      response.sentMs += pcm16DurationMs(part.byteLength, live.outputRate)
    }
  }

  private onProviderAudioDone(live: LiveState, providerId: string): void {
    if (live.cancelled.delete(providerId)) return
    const response = this.response
    if (response === undefined || response.providerId !== providerId) return
    this.sendAudio(response, 'pcm16', new Uint8Array(0), true, true)
    this.response = undefined
    this.send({ type: 'audio_end', response_id: response.id, generation: response.generation, interrupted: false })
    this.setState('listening', this.counter)
  }

  private liveReply(live: LiveState, reply: SupervisorReply): void {
    const spoken = prepareSpeech(reply.text, VOICE_LIMITS.maxSpokenChars).join(' ')
    this.send({ type: 'reply', response_id: `${this.id}-r${++this.responseSeq}`, generation: this.counter, text: reply.text.slice(0, MAX_REPLY_TEXT), spoken: spoken !== '', ...(reply.inReplyTo === undefined ? {} : { message_id: reply.inReplyTo }) })
    if (spoken !== '') live.rt.say(spoken)
  }

  private async onToolCall(live: LiveState, callId: string, name: string): Promise<void> {
    live.toolCalls += 1
    let output: string
    try {
      if (live.toolCalls > MAX_TOOL_CALLS) throw new VoiceError('VOICE_LIMIT', 'too many function calls in this session')
      const conversation = this.conversationId
      if (conversation === undefined) throw new VoiceError('VOICE_STALE', 'no conversation is open')
      const actor = this.options.context.actor
      switch (name) {
        case 'supervisor_submit': output = await this.liveSubmit(live, conversation); break
        case 'supervisor_status': {
          const status = await this.options.supervisor.status(conversation)
          output = JSON.stringify({ ok: true, busy: status.busy, ...(status.agentsWorking === undefined ? {} : { agents_working: status.agentsWorking }), ...(status.summary === undefined ? {} : { summary: status.summary.slice(0, 800) }) })
          break
        }
        case 'supervisor_cancel': {
          const result = await this.options.supervisor.cancel(conversation, actor)
          output = JSON.stringify({ ok: true, cancelled: result.cancelled, note: 'Background agents keep running.' })
          break
        }
        default: output = JSON.stringify({ ok: false, error: 'unknown function' })
      }
    } catch (error) {
      output = JSON.stringify({ ok: false, error: error instanceof VoiceError ? error.message : 'the request failed' })
    }
    if (this.live === live && !this.closed) live.rt.sendToolResult(callId, output)
  }

  private async liveSubmit(live: LiveState, conversation: string): Promise<string> {
    const final = live.lastFinal
    if (final === undefined) return JSON.stringify({ ok: false, error: 'the user\'s words are not finished yet; wait for them to finish speaking' })
    // The key comes from the finalized utterance, never from the model: a repeated call or a second
    // path for the same words reaches the submitter with the same key and starts nothing new.
    const key = `voice:${final.id}`
    const actor = this.options.context.actor
    this.armedUntil = this.options.clock() + ARM_MS
    this.armedMessageId = undefined
    let outcome
    try {
      outcome = await this.options.submitter.submit(conversation, final.text, actor, key)
    } catch (error) {
      this.armedUntil = 0
      throw error
    }
    if (!outcome.accepted) { this.armedUntil = 0; return JSON.stringify({ ok: false, error: 'the supervisor did not accept the message' }) }
    this.send({ type: 'submitted', utterance_id: final.id, duplicate: outcome.duplicate, ...(outcome.messageId === undefined ? {} : { message_id: outcome.messageId }) })
    if (outcome.duplicate) {
      this.armedUntil = 0
      return JSON.stringify({ ok: true, duplicate: true, note: 'This request was already sent.' })
    }
    this.armedMessageId = outcome.messageId
    this.persist({
      kind: 'submitted', conversationId: conversation, utteranceId: final.id, userId: this.principalId, deviceId: this.holder().deviceId,
      displayName: actor.display_name, mode: 'live', text: final.text, edited: false, at: this.iso(),
    })
    return JSON.stringify({ ok: true, duplicate: false, note: 'Sent to XERXES. Its answer will be read to the user.' })
  }

  // ------------------------------------------------------------------ teardown

  /** Finish the session: report usage, close the socket. Idempotent. */
  end(reason: string, code = 1000): void {
    if (this.closed) return
    this.sendUsage()
    this.teardown()
    this.options.transport.close(code, reason.slice(0, 100))
  }

  private teardown(): void {
    if (this.closed && this.live === undefined) return
    this.closed = true
    this.utterance?.abort.abort()
    this.utterance = undefined
    this.response?.abort.abort()
    this.response = undefined
    this.liveAbort?.abort()
    this.endLive()
    this.offReply?.()
    this.offReply = undefined
    this.options.leases.releaseConnection(this.id)
    this.options.budget.endSession(this.id)
  }

  private endLive(): void {
    const live = this.live
    if (live === undefined) return
    this.live = undefined
    live.off()
    try { live.rt.close() } catch (error) { this.options.onError?.(error, 'close realtime') }
    const elapsedMinutes = (this.options.clock() - live.startedAt) / 60_000
    const actual = estimateUsd(this.options.config.settings.pricing, 'realtime', elapsedMinutes)
    const [first, ...rest] = live.reservations
    if (first !== undefined) this.options.budget.settle(first, actual)
    for (const extra of rest) this.options.budget.settle(extra, 0)
  }

  private abortStream(): void {
    if (this.live !== undefined) { this.end('stream error'); return }
    if (this.utterance !== undefined) this.clearUtterance(this.utterance)
  }

  private clearUtterance(utterance: ActiveUtterance): void {
    utterance.abort.abort()
    utterance.chunks = []
    if (this.utterance === utterance) this.utterance = undefined
    if (!this.closed && this.response === undefined) this.setState('idle', this.counter)
  }

  // ------------------------------------------------------------------ helpers

  private holder(): LeaseHolder {
    const principal = this.options.context.principal
    return { userId: principal.id, deviceId: principal.kind === 'user' ? principal.deviceId : '', connectionId: this.id }
  }

  private iso(): string { return new Date(this.options.clock()).toISOString() }

  private setState(state: VoiceStateName, generation: number, ids: { readonly utterance_id?: string; readonly response_id?: string } = {}): void {
    const key = `${state}:${generation}:${ids.utterance_id ?? ''}:${ids.response_id ?? ''}`
    if (key === this.lastState) return
    this.lastState = key
    this.send({ type: 'state', state, generation, ...(this.conversationId === undefined ? {} : { conversation_id: this.conversationId }), ...ids })
  }

  private send(message: VoiceServerMessage): boolean {
    try {
      return this.options.transport.sendText(encodeVoiceServerMessage(message))
    } catch (error) {
      this.options.onError?.(error, 'send')
      return false
    }
  }

  private sendError(error: VoiceError, ids: { readonly utterance_id?: string; readonly response_id?: string } = {}): void {
    this.send(errorMessage(error, ids))
  }

  private sendUsage(): void {
    const usage = this.options.budget.usage(this.principalId, this.id)
    this.send({
      type: 'usage', session_usd: usage.sessionUsd, daily_usd: usage.dailyUsd, unknown_cost: usage.unknownCost,
      ...(usage.maxSessionUsd === undefined ? {} : { max_session_usd: usage.maxSessionUsd }),
      ...(usage.maxDailyUsd === undefined ? {} : { max_daily_usd: usage.maxDailyUsd }),
    })
  }

  private fail(error: unknown): void {
    if (error instanceof VoiceError) { this.sendError(error); return }
    this.options.onError?.(error, 'voice')
    this.sendError(new VoiceError('VOICE_PROVIDER', 'voice failed unexpectedly'))
  }

  private protocolFailure(error: unknown): void {
    this.fail(error)
    this.strikes += 1
    if (this.strikes >= MAX_PROTOCOL_STRIKES) this.end('too many protocol errors', 1008)
  }

  private persist(record: VoiceTranscriptRecord): void {
    const store = this.options.config.transcripts
    if (store === undefined) return
    void store.append(record).catch(error => this.options.onError?.(error, 'persist transcript'))
  }
}

function errorMessage(error: VoiceError, ids: { readonly utterance_id?: string; readonly response_id?: string } = {}): VoiceServerMessage {
  return { type: 'error', code: error.code, message: error.message, ...ids }
}

function concat(chunks: readonly Uint8Array[]): Uint8Array {
  let total = 0
  for (const chunk of chunks) total += chunk.byteLength
  const out = new Uint8Array(total)
  let offset = 0
  for (const chunk of chunks) { out.set(chunk, offset); offset += chunk.byteLength }
  return out
}
