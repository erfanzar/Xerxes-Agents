// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { AuthenticatedContext } from '../src/hub/identityPort.js'
import type { HubActor } from '../src/hub/protocol.js'
import { VoiceBudget } from '../src/hub/voice/budgets.js'
import { DEFAULT_VOICE_SETTINGS, type VoiceRuntimeConfig, type VoiceSettings } from '../src/hub/voice/config.js'
import type { RealtimeEvent, RealtimeSessionOptions, RealtimeVoicePort, RealtimeVoiceSession, SpeechToTextPort, TextToSpeechPort } from '../src/hub/voice/ports.js'
import { VOICE_LIMITS, decodeVoiceFrame, encodeVoiceFrame, type VoiceCodec, type VoiceFrame } from '../src/hub/voice/protocol.js'
import { VoiceSession, type VoiceTransport } from '../src/hub/voice/session.js'
import { SpeakerLeases } from '../src/hub/voice/speakerLease.js'
import { ExactlyOnceSubmitter, type SupervisorReply, type SupervisorSubmitOutcome, type SupervisorThreadStatus, type SupervisorVoiceBridge } from '../src/hub/voice/supervisorBridge.js'
import { MemoryVoiceTranscriptStore } from '../src/hub/voice/transcripts.js'

export const START = Date.parse('2026-10-22T10:00:00Z')

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type Msg = { type: string; [key: string]: any }

export function settle(ms = 5): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

export function userContext(id: string, deviceId = `dev-${id}`): AuthenticatedContext {
  const actor: HubActor = { principal_kind: 'user', principal_id: id, display_name: id.toUpperCase(), user_id: id, device_id: deviceId }
  return {
    principal: { kind: 'user', id, displayName: id.toUpperCase(), deviceId, hubOwner: false },
    grants: [],
    browserSessionId: `bs-${id}-${deviceId}`,
    actor,
  }
}

export function agentContext(id: string): AuthenticatedContext {
  return {
    principal: { kind: 'agent', id, displayName: id },
    grants: [],
    browserSessionId: `bs-${id}`,
    actor: { principal_kind: 'agent', principal_id: id, display_name: id },
  }
}

export class FakeTransport implements VoiceTransport {
  readonly texts: string[] = []
  readonly binaries: Uint8Array[] = []
  closed: { code: number; reason: string } | undefined
  buffered = 0
  sendText(text: string): boolean { this.texts.push(text); return true }
  sendBinary(bytes: Uint8Array): boolean { this.binaries.push(bytes); return true }
  bufferedAmount(): number { return this.buffered }
  close(code: number, reason: string): void { this.closed = { code, reason } }
  messages(): Msg[] { return this.texts.map(text => JSON.parse(text) as Msg) }
  ofType(type: string): Msg[] { return this.messages().filter(message => message.type === type) }
  last(type: string): Msg {
    const found = this.ofType(type).at(-1)
    if (found === undefined) throw new Error(`no ${type} message was sent; got ${this.messages().map(m => m.type).join(',')}`)
    return found
  }
  errors(): string[] { return this.ofType('error').map(message => String(message.code)) }
  frames(): VoiceFrame[] { return this.binaries.map(bytes => decodeVoiceFrame(bytes)) }
}

export class FakeStt implements SpeechToTextPort {
  readonly providerName = 'fake-stt'
  readonly calls: { audio: Uint8Array; mediaType: string; language?: string }[] = []
  text = 'hello world'
  gate: Promise<void> | undefined
  fail: Error | undefined
  async transcribe(request: { audio: Uint8Array; mediaType: string; language?: string }, signal: AbortSignal): Promise<{ text: string; provider: string; model: string }> {
    this.calls.push({ audio: request.audio, mediaType: request.mediaType, ...(request.language === undefined ? {} : { language: request.language }) })
    if (this.gate !== undefined) {
      await Promise.race([this.gate, new Promise<never>((_, reject) => signal.addEventListener('abort', () => reject(new Error('aborted')), { once: true }))])
    }
    if (this.fail !== undefined) throw this.fail
    return { text: this.text, provider: 'fake-stt', model: 'fake-model' }
  }
}

export class FakeTts implements TextToSpeechPort {
  readonly providerName = 'fake-tts'
  delivery: 'buffered' | 'streaming' = 'buffered'
  readonly calls: string[] = []
  gate: Promise<void> | undefined
  size = 100
  async synthesize(request: { text: string }, signal?: AbortSignal): Promise<{ audio: Uint8Array; format: 'mp3'; model: string; voice: string }> {
    this.calls.push(request.text)
    if (this.gate !== undefined) {
      await Promise.race([this.gate, new Promise<never>((_, reject) => signal?.addEventListener('abort', () => reject(new Error('aborted')), { once: true }))])
    }
    return { audio: new Uint8Array(this.size).fill(7), format: 'mp3', model: 'fake', voice: 'fake' }
  }
}

export class FakeBridge implements SupervisorVoiceBridge {
  readonly submits: { threadId: string; text: string; actor: HubActor; key: string }[] = []
  readonly cancels: { threadId: string; actor: HubActor }[] = []
  readonly denied = new Set<string>()
  private readonly keys = new Map<string, SupervisorSubmitOutcome>()
  private readonly listeners = new Map<string, Set<(reply: SupervisorReply) => void>>()
  statusResult: SupervisorThreadStatus = { busy: true, agentsWorking: 2, summary: 'two agents are working' }
  failNextSubmit = false
  private seq = 0

  async authorize(threadId: string, actor: HubActor): Promise<boolean> {
    return !this.denied.has(`${threadId}|${actor.principal_id}`)
  }
  async submit(threadId: string, text: string, actor: HubActor, key: string): Promise<SupervisorSubmitOutcome> {
    if (this.failNextSubmit) { this.failNextSubmit = false; throw new Error('bridge down') }
    const identity = `${threadId}|${actor.principal_id}|${key}`
    const known = this.keys.get(identity)
    if (known !== undefined) return { ...known, duplicate: true }
    this.submits.push({ threadId, text, actor, key })
    const outcome: SupervisorSubmitOutcome = { accepted: true, duplicate: false, messageId: `m${++this.seq}` }
    this.keys.set(identity, outcome)
    return outcome
  }
  async status(): Promise<SupervisorThreadStatus> { return this.statusResult }
  async cancel(threadId: string, actor: HubActor): Promise<{ cancelled: boolean }> {
    this.cancels.push({ threadId, actor })
    return { cancelled: true }
  }
  onReply(threadId: string, listener: (reply: SupervisorReply) => void): () => void {
    let set = this.listeners.get(threadId)
    if (set === undefined) { set = new Set(); this.listeners.set(threadId, set) }
    set.add(listener)
    return () => { set.delete(listener) }
  }
  listenerCount(threadId: string): number { return this.listeners.get(threadId)?.size ?? 0 }
  reply(threadId: string, text: string, inReplyTo?: string): void {
    for (const listener of [...(this.listeners.get(threadId) ?? [])]) listener({ id: `reply-${++this.seq}`, text, ...(inReplyTo === undefined ? {} : { inReplyTo }) })
  }
}

export class FakeRealtimeSession implements RealtimeVoiceSession {
  supportsTruncate = true
  readonly audio: Uint8Array[] = []
  readonly truncates: { itemId: string; playedMs: number }[] = []
  readonly toolResults: { callId: string; output: string }[] = []
  readonly said: string[] = []
  cancels = 0
  closed = false
  backedUp = false
  private listener: ((event: RealtimeEvent) => void) | undefined
  onEvent(listener: (event: RealtimeEvent) => void): () => void { this.listener = listener; return () => { this.listener = undefined } }
  sendAudio(pcm: Uint8Array): boolean { this.audio.push(pcm); return !this.backedUp }
  cancelResponse(): void { this.cancels += 1 }
  truncate(itemId: string, playedMs: number): void { this.truncates.push({ itemId, playedMs }) }
  sendToolResult(callId: string, output: string): void { this.toolResults.push({ callId, output }) }
  say(text: string): void { this.said.push(text) }
  close(): void { this.closed = true }
  emit(event: RealtimeEvent): void { this.listener?.(event) }
}

export class FakeRealtimePort implements RealtimeVoicePort {
  readonly providerName = 'fake-realtime'
  readonly sampleRate = 24_000
  readonly connects: RealtimeSessionOptions[] = []
  session = new FakeRealtimeSession()
  failConnect: Error | undefined
  async connect(options: RealtimeSessionOptions): Promise<RealtimeVoiceSession> {
    this.connects.push(options)
    if (this.failConnect !== undefined) throw this.failConnect
    return this.session
  }
}

export interface RigOptions {
  readonly settings?: Partial<VoiceSettings>
  readonly stt?: SpeechToTextPort | null
  readonly tts?: TextToSpeechPort | null
  readonly realtime?: RealtimeVoicePort
}

export interface Connection {
  readonly session: VoiceSession
  readonly transport: FakeTransport
}

export class Rig {
  clockNow = START
  readonly clock = (): number => this.clockNow
  readonly bridge = new FakeBridge()
  readonly stt = new FakeStt()
  readonly tts = new FakeTts()
  readonly store = new MemoryVoiceTranscriptStore()
  readonly config: VoiceRuntimeConfig
  readonly budget: VoiceBudget
  readonly leases: SpeakerLeases
  readonly submitter: ExactlyOnceSubmitter
  readonly errors: unknown[] = []
  private n = 0

  constructor(options: RigOptions = {}) {
    const settings: VoiceSettings = {
      ...DEFAULT_VOICE_SETTINGS,
      enabled: true,
      mode: options.realtime === undefined ? DEFAULT_VOICE_SETTINGS.mode : 'both',
      ...options.settings,
    }
    const stt = options.stt === null ? undefined : options.stt ?? this.stt
    const tts = options.tts === null ? undefined : options.tts ?? this.tts
    this.config = {
      settings,
      ...(stt === undefined ? {} : { stt }),
      ...(tts === undefined ? {} : { tts }),
      ...(options.realtime === undefined ? {} : { realtime: options.realtime }),
      transcripts: this.store,
    }
    this.budget = new VoiceBudget({
      clock: this.clock,
      limits: {
        ...(settings.maxSessionUsd === undefined ? {} : { maxSessionUsd: settings.maxSessionUsd }),
        ...(settings.maxDailyUserUsd === undefined ? {} : { maxDailyUserUsd: settings.maxDailyUserUsd }),
      },
    })
    this.leases = new SpeakerLeases({ ttlMs: VOICE_LIMITS.leaseTtlMs, clock: this.clock })
    this.submitter = new ExactlyOnceSubmitter(this.bridge)
  }

  advance(ms: number): void { this.clockNow += ms }

  connect(user = 'alice', device?: string): Connection {
    const transport = new FakeTransport()
    const session = new VoiceSession({
      id: `s${++this.n}`,
      context: userContext(user, device),
      config: this.config,
      supervisor: this.bridge,
      submitter: this.submitter,
      leases: this.leases,
      budget: this.budget,
      clock: this.clock,
      transport,
      onError: error => { this.errors.push(error) },
    })
    return { session, transport }
  }
}

export function control(message: Record<string, unknown>): string {
  return JSON.stringify({ v: 1, ...message })
}

export function micFrame(generation: number, sequence: number, payload: Uint8Array | number, options: { final?: boolean; codec?: VoiceCodec; timestampMs?: number } = {}): Uint8Array {
  const bytes = typeof payload === 'number' ? new Uint8Array(payload).fill(3) : payload
  return encodeVoiceFrame(
    { stream: 'mic', codec: options.codec ?? 'webm-opus', generation, sequence, timestampMs: options.timestampMs ?? sequence * 500, final: options.final ?? false, clipEnd: false },
    bytes,
  )
}

/** Open a push-to-talk utterance and return its generation. */
export async function startPtt(conn: Connection, options: { conversation?: string; utterance?: string; codec?: VoiceCodec } = {}): Promise<number> {
  await conn.session.handleText(control({
    type: 'start', mode: 'push_to_talk', conversation_id: options.conversation ?? 'thread-1',
    utterance_id: options.utterance ?? 'u1', codec: options.codec ?? 'webm-opus',
  }))
  return Number(conn.transport.last('state').generation)
}
