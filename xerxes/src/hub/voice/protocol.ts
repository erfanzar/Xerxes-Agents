// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Wire contract of the Hub voice endpoint (`/api/hub/v1/voice`). Browser-safe:
 * no Node or Bun imports, because the phone shares this module.
 *
 * Text frames carry JSON control messages. Binary frames carry audio behind a
 * fixed 16-byte header:
 *
 *   byte 0      protocol version (1)
 *   byte 1      stream (1 = microphone upstream, 2 = speech downstream)
 *   byte 2      codec id (see VOICE_CODECS, 1-based)
 *   byte 3      flags (bit 0 = final frame of the stream, bit 1 = last frame of a
 *               self-contained clip: one synthesized sentence for container codecs)
 *   bytes 4-7   generation (u32 big-endian)
 *   bytes 8-11  sequence (u32 big-endian, starts at 0, contiguous)
 *   bytes 12-15 timestamp in ms since the stream started (u32 big-endian)
 *
 * Audio is never replayed on reconnect: a new socket is a new session.
 */

export const VOICE_PROTOCOL_VERSION = 1 as const
export const VOICE_PATH = '/api/hub/v1/voice'

export const VOICE_LIMITS = Object.freeze({
  headerBytes: 16,
  /** Largest audio payload carried by one binary frame. */
  maxFramePayloadBytes: 256 * 1024,
  maxControlBytes: 16 * 1024,
  maxUtteranceSeconds: 60,
  maxUtteranceBytes: 8 * 1024 * 1024,
  minSliceMs: 250,
  maxSliceMs: 1000,
  maxSessionSeconds: 300,
  idleSeconds: 30,
  pttIdleSeconds: 120,
  /** Outbound bytes a socket may have queued before a stream is ended. */
  maxBufferedBytes: 1024 * 1024,
  maxIdLength: 200,
  maxTextChars: 4000,
  maxSpokenChars: 1200,
  leaseTtlMs: 30_000,
})

export const VOICE_ERROR_CODES = [
  'VOICE_UNAVAILABLE',
  'VOICE_FORMAT',
  'VOICE_BUDGET',
  'VOICE_PERMISSION',
  'VOICE_PROVIDER',
  'VOICE_LIMIT',
  'VOICE_OVERLOADED',
  'VOICE_CANCELLED',
  'VOICE_STALE',
  'VOICE_PROTOCOL',
  'VOICE_BUSY',
] as const
export type VoiceErrorCode = (typeof VOICE_ERROR_CODES)[number]

/** Typed voice failure. `VOICE_UNAVAILABLE` means nothing is configured: never a simulated success. */
export class VoiceError extends Error {
  readonly code: VoiceErrorCode
  constructor(code: VoiceErrorCode, message: string, cause?: unknown) {
    super(message, cause === undefined ? undefined : { cause })
    this.name = 'VoiceError'
    this.code = code
  }
}

export const VOICE_CODECS = ['pcm16', 'webm-opus', 'ogg-opus', 'mp4-aac', 'mp3', 'aac', 'opus', 'flac', 'wav'] as const
export type VoiceCodec = (typeof VOICE_CODECS)[number]
export type VoiceMode = 'push_to_talk' | 'live'
export type VoiceStreamName = 'mic' | 'tts'
export type VoiceStateName = 'idle' | 'connecting' | 'listening' | 'transcribing' | 'thinking' | 'buffering' | 'speaking' | 'interrupted' | 'failed'

/** Codecs a browser recorder can produce for push-to-talk. */
export const RECORDER_CODECS: readonly VoiceCodec[] = ['webm-opus', 'ogg-opus', 'mp4-aac']

const MEDIA_TYPES: Readonly<Record<VoiceCodec, string>> = {
  pcm16: 'audio/pcm',
  'webm-opus': 'audio/webm',
  'ogg-opus': 'audio/ogg',
  'mp4-aac': 'audio/mp4',
  mp3: 'audio/mpeg',
  aac: 'audio/aac',
  opus: 'audio/opus',
  flac: 'audio/flac',
  wav: 'audio/wav',
}

export function mediaTypeForCodec(codec: VoiceCodec): string {
  return MEDIA_TYPES[codec]
}

/** Map a (possibly parameterized) recorder MIME type to a codec; undefined when unsupported. */
export function codecForMediaType(mediaType: string): VoiceCodec | undefined {
  const lower = mediaType.toLowerCase()
  const bare = lower.split(';')[0]?.trim() ?? ''
  if (bare === 'audio/webm') return 'webm-opus'
  if (bare === 'audio/ogg') return 'ogg-opus'
  if (bare === 'audio/mp4' || bare === 'audio/aac' || bare === 'audio/x-m4a') return bare === 'audio/aac' ? 'aac' : 'mp4-aac'
  if (bare === 'audio/mpeg' || bare === 'audio/mp3') return 'mp3'
  if (bare === 'audio/wav' || bare === 'audio/x-wav') return 'wav'
  if (bare === 'audio/flac') return 'flac'
  if (bare === 'audio/opus') return 'opus'
  if (bare === 'audio/pcm') return 'pcm16'
  return undefined
}

export function isVoiceCodec(value: unknown): value is VoiceCodec {
  return typeof value === 'string' && (VOICE_CODECS as readonly string[]).includes(value)
}

// ------------------------------------------------------------------ binary frames

export interface VoiceFrameHeader {
  readonly stream: VoiceStreamName
  readonly codec: VoiceCodec
  readonly generation: number
  readonly sequence: number
  readonly timestampMs: number
  readonly final: boolean
  /** Last frame of a self-contained clip. Always true on every PCM16 frame. */
  readonly clipEnd: boolean
}

export interface VoiceFrame {
  readonly header: VoiceFrameHeader
  readonly payload: Uint8Array
}

const MAX_U32 = 0xffff_ffff

function u32(value: number, field: string): number {
  if (!Number.isInteger(value) || value < 0 || value > MAX_U32) {
    throw new VoiceError('VOICE_PROTOCOL', `${field} must be an unsigned 32-bit integer`)
  }
  return value
}

export function encodeVoiceFrame(header: VoiceFrameHeader, payload: Uint8Array): Uint8Array {
  if (payload.byteLength > VOICE_LIMITS.maxFramePayloadBytes) {
    throw new VoiceError('VOICE_LIMIT', `audio payload exceeds ${VOICE_LIMITS.maxFramePayloadBytes} bytes`)
  }
  const out = new Uint8Array(VOICE_LIMITS.headerBytes + payload.byteLength)
  const view = new DataView(out.buffer)
  view.setUint8(0, VOICE_PROTOCOL_VERSION)
  view.setUint8(1, header.stream === 'mic' ? 1 : 2)
  view.setUint8(2, VOICE_CODECS.indexOf(header.codec) + 1)
  view.setUint8(3, (header.final ? 1 : 0) | (header.clipEnd ? 2 : 0))
  view.setUint32(4, u32(header.generation, 'generation'))
  view.setUint32(8, u32(header.sequence, 'sequence'))
  view.setUint32(12, u32(Math.round(header.timestampMs), 'timestampMs'))
  out.set(payload, VOICE_LIMITS.headerBytes)
  return out
}

/** Parse one binary frame. Throws VoiceError(VOICE_PROTOCOL | VOICE_LIMIT); never returns a partial frame. */
export function decodeVoiceFrame(bytes: Uint8Array, maxPayloadBytes: number = VOICE_LIMITS.maxFramePayloadBytes): VoiceFrame {
  if (bytes.byteLength < VOICE_LIMITS.headerBytes) throw new VoiceError('VOICE_PROTOCOL', 'audio frame is shorter than its header')
  const payloadBytes = bytes.byteLength - VOICE_LIMITS.headerBytes
  if (payloadBytes > maxPayloadBytes) throw new VoiceError('VOICE_LIMIT', `audio payload exceeds ${maxPayloadBytes} bytes`)
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  if (view.getUint8(0) !== VOICE_PROTOCOL_VERSION) throw new VoiceError('VOICE_PROTOCOL', 'unsupported audio frame version')
  const streamId = view.getUint8(1)
  if (streamId !== 1 && streamId !== 2) throw new VoiceError('VOICE_PROTOCOL', 'unknown audio stream')
  const codec = VOICE_CODECS[view.getUint8(2) - 1]
  if (codec === undefined) throw new VoiceError('VOICE_PROTOCOL', 'unknown audio codec')
  return {
    header: {
      stream: streamId === 1 ? 'mic' : 'tts',
      codec,
      generation: view.getUint32(4),
      sequence: view.getUint32(8),
      timestampMs: view.getUint32(12),
      final: (view.getUint8(3) & 1) === 1,
      clipEnd: (view.getUint8(3) & 2) === 2,
    },
    payload: bytes.subarray(VOICE_LIMITS.headerBytes),
  }
}

// ------------------------------------------------------------------ control messages

export interface VoiceCapabilities {
  readonly enabled: boolean
  readonly modes: readonly VoiceMode[]
  readonly stt: { readonly available: boolean; readonly provider?: string; readonly reason?: string }
  readonly tts: { readonly available: boolean; readonly provider?: string; readonly delivery?: 'buffered' | 'streaming'; readonly format?: VoiceCodec; readonly reason?: string }
  readonly realtime: { readonly available: boolean; readonly provider?: string; readonly sample_rate?: number; readonly reason?: string }
  readonly limits: {
    readonly max_utterance_seconds: number
    readonly max_utterance_bytes: number
    readonly slice_ms_min: number
    readonly slice_ms_max: number
    readonly max_session_seconds: number
    readonly idle_seconds: number
  }
  readonly language?: string
  readonly confirm_transcript: boolean
  readonly browser_fallback: boolean
  readonly audio_retention: 'none'
}

export type VoiceClientMessage =
  | { readonly type: 'capabilities' }
  | {
      readonly type: 'start'
      readonly mode: VoiceMode
      readonly conversation_id: string
      readonly utterance_id: string
      readonly codec: VoiceCodec
      readonly sample_rate?: number
      readonly language?: string
    }
  | { readonly type: 'commit'; readonly utterance_id: string; readonly submit?: boolean }
  | { readonly type: 'submit'; readonly utterance_id: string; readonly text: string }
  | { readonly type: 'interrupt'; readonly response_id?: string; readonly played_ms?: number }
  | { readonly type: 'renew' }
  /**
   * The phone submitted a message through the normal Hub path and wants the
   * supervisor's next reply spoken on this device. Arms one reply for 2 minutes.
   */
  | { readonly type: 'expect_reply'; readonly enabled?: boolean; readonly conversation_id?: string }
  | { readonly type: 'stop' }

export type VoiceServerMessage =
  | { readonly type: 'capabilities'; readonly capabilities: VoiceCapabilities }
  | { readonly type: 'state'; readonly state: VoiceStateName; readonly generation: number; readonly conversation_id?: string; readonly utterance_id?: string; readonly response_id?: string }
  | { readonly type: 'transcript'; readonly utterance_id: string; readonly revision: number; readonly text: string; readonly final: boolean }
  | { readonly type: 'submitted'; readonly utterance_id: string; readonly duplicate: boolean; readonly message_id?: string }
  /** Supervisor-authored text for the screen. `spoken` is true only when audio for `response_id` follows. */
  | { readonly type: 'reply'; readonly response_id: string; readonly generation: number; readonly text: string; readonly spoken: boolean; readonly message_id?: string }
  | { readonly type: 'audio_start'; readonly response_id: string; readonly generation: number; readonly delivery: 'buffered' | 'streaming'; readonly codec: VoiceCodec; readonly sample_rate?: number }
  | { readonly type: 'audio_end'; readonly response_id: string; readonly generation: number; readonly interrupted: boolean }
  | { readonly type: 'usage'; readonly session_usd: number; readonly daily_usd: number; readonly unknown_cost: boolean; readonly max_session_usd?: number; readonly max_daily_usd?: number }
  | { readonly type: 'lease'; readonly conversation_id: string; readonly held_by_you: boolean; readonly expires_at: string }
  | { readonly type: 'error'; readonly code: VoiceErrorCode; readonly message: string; readonly utterance_id?: string; readonly response_id?: string }

const ID_PATTERN = /^[A-Za-z0-9._:-]{1,200}$/

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function readId(record: Record<string, unknown>, field: string): string {
  const value = record[field]
  if (typeof value !== 'string' || !ID_PATTERN.test(value)) {
    throw new VoiceError('VOICE_PROTOCOL', `${field} must be 1-${VOICE_LIMITS.maxIdLength} characters of letters, digits and . _ : -`)
  }
  return value
}

function utf8Length(text: string): number {
  return new TextEncoder().encode(text).byteLength
}

/** Parse one client control message. Throws VoiceError(VOICE_PROTOCOL | VOICE_LIMIT). */
export function parseVoiceClientMessage(text: string): VoiceClientMessage {
  if (utf8Length(text) > VOICE_LIMITS.maxControlBytes) throw new VoiceError('VOICE_LIMIT', 'control message is too large')
  let value: unknown
  try { value = JSON.parse(text) } catch { throw new VoiceError('VOICE_PROTOCOL', 'control message is not valid JSON') }
  if (!isRecord(value)) throw new VoiceError('VOICE_PROTOCOL', 'control message must be a JSON object')
  if (value.v !== undefined && value.v !== VOICE_PROTOCOL_VERSION) throw new VoiceError('VOICE_PROTOCOL', 'unsupported voice protocol version')
  switch (value.type) {
    case 'capabilities': return { type: 'capabilities' }
    case 'renew': return { type: 'renew' }
    case 'expect_reply': {
      if (value.enabled !== undefined && typeof value.enabled !== 'boolean') throw new VoiceError('VOICE_PROTOCOL', 'enabled must be a boolean')
      return {
        type: 'expect_reply',
        ...(value.enabled === undefined ? {} : { enabled: value.enabled }),
        ...(value.conversation_id === undefined ? {} : { conversation_id: readId(value, 'conversation_id') }),
      }
    }
    case 'stop': return { type: 'stop' }
    case 'start': {
      const mode = value.mode
      if (mode !== 'push_to_talk' && mode !== 'live') throw new VoiceError('VOICE_PROTOCOL', 'mode must be push_to_talk or live')
      if (!isVoiceCodec(value.codec)) throw new VoiceError('VOICE_FORMAT', 'unsupported codec')
      const rate = value.sample_rate
      if (rate !== undefined && (typeof rate !== 'number' || !Number.isInteger(rate) || rate < 8000 || rate > 96000)) {
        throw new VoiceError('VOICE_PROTOCOL', 'sample_rate must be an integer from 8000 to 96000')
      }
      const language = value.language
      if (language !== undefined && (typeof language !== 'string' || !/^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$/.test(language))) {
        throw new VoiceError('VOICE_PROTOCOL', 'language must be a BCP-47 tag')
      }
      return {
        type: 'start', mode, codec: value.codec,
        conversation_id: readId(value, 'conversation_id'), utterance_id: readId(value, 'utterance_id'),
        ...(rate === undefined ? {} : { sample_rate: rate }),
        ...(language === undefined ? {} : { language: language as string }),
      }
    }
    case 'commit': {
      if (value.submit !== undefined && typeof value.submit !== 'boolean') throw new VoiceError('VOICE_PROTOCOL', 'submit must be a boolean')
      return { type: 'commit', utterance_id: readId(value, 'utterance_id'), ...(value.submit === undefined ? {} : { submit: value.submit }) }
    }
    case 'submit': {
      const body = value.text
      if (typeof body !== 'string' || body.trim() === '' || body.length > VOICE_LIMITS.maxTextChars) {
        throw new VoiceError('VOICE_PROTOCOL', `text must be 1-${VOICE_LIMITS.maxTextChars} characters`)
      }
      return { type: 'submit', utterance_id: readId(value, 'utterance_id'), text: body }
    }
    case 'interrupt': {
      const played = value.played_ms
      if (played !== undefined && (typeof played !== 'number' || !Number.isFinite(played) || played < 0 || played > 3_600_000)) {
        throw new VoiceError('VOICE_PROTOCOL', 'played_ms must be a non-negative number')
      }
      return {
        type: 'interrupt',
        ...(value.response_id === undefined ? {} : { response_id: readId(value, 'response_id') }),
        ...(played === undefined ? {} : { played_ms: Math.floor(played) }),
      }
    }
    default:
      throw new VoiceError('VOICE_PROTOCOL', 'unknown control message type')
  }
}

export function encodeVoiceServerMessage(message: VoiceServerMessage): string {
  const text = JSON.stringify({ v: VOICE_PROTOCOL_VERSION, ...message })
  if (utf8Length(text) > VOICE_LIMITS.maxControlBytes * 4) throw new VoiceError('VOICE_LIMIT', 'control message is too large')
  return text
}

/** Parse a server control message on the phone. Returns undefined for anything malformed or unknown. */
export function parseVoiceServerMessage(text: string): VoiceServerMessage | undefined {
  let value: unknown
  try { value = JSON.parse(text) } catch { return undefined }
  if (!isRecord(value) || value.v !== VOICE_PROTOCOL_VERSION || typeof value.type !== 'string') return undefined
  const { v: _v, ...rest } = value
  void _v
  const known = ['capabilities', 'state', 'transcript', 'submitted', 'reply', 'audio_start', 'audio_end', 'usage', 'lease', 'error']
  return known.includes(value.type) ? (rest as unknown as VoiceServerMessage) : undefined
}
