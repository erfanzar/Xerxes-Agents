// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { VOICE_LIMITS, VoiceError, type VoiceCodec, type VoiceErrorCode, type VoiceFrame } from './protocol.js'

/**
 * Admission control for the binary frames of ONE upstream stream (one
 * push-to-talk utterance or one live session). A frame is accepted only when
 * it has the stream's generation and codec, the next sequence number, and fits
 * the byte and time budget. Everything else is rejected with a typed error and
 * a verdict on whether the stream must end.
 */

export type GateVerdict =
  | { readonly ok: true }
  | { readonly ok: false; readonly error: VoiceError; readonly fatal: boolean }

export interface FrameGateOptions {
  readonly generation: number
  readonly codec: VoiceCodec
  readonly maxBytes: number
  readonly maxSeconds: number
  readonly now: () => number
}

export class FrameGate {
  private nextSequence = 0
  private bytes = 0
  private byteBase = 0
  private timeBaseMs = 0
  private lastTimestampMs = 0
  private startedAt: number
  private ended = false

  constructor(private readonly options: FrameGateOptions) {
    this.startedAt = options.now()
  }

  get generation(): number { return this.options.generation }
  get byteCount(): number { return this.bytes }
  /** Position of the last accepted frame on the sender's timeline. */
  get durationMs(): number { return this.lastTimestampMs }

  /** Start a new byte and time window without restarting the sequence (a deliberate live renewal). */
  extend(): void {
    this.byteBase = this.bytes
    this.timeBaseMs = this.lastTimestampMs
    this.startedAt = this.options.now()
  }

  check(frame: VoiceFrame): GateVerdict {
    const { header, payload } = frame
    if (this.ended) return reject('VOICE_STALE', 'this audio stream already ended', false)
    if (header.stream !== 'mic') return reject('VOICE_PROTOCOL', 'only microphone frames are accepted from the phone', false)
    if (header.generation !== this.options.generation) return reject('VOICE_STALE', 'audio frame belongs to an older generation', false)
    if (header.codec !== this.options.codec) return reject('VOICE_FORMAT', 'audio frame codec differs from the one announced at start', true)
    if (header.sequence < this.nextSequence) return reject('VOICE_STALE', 'late or duplicate audio frame', false)
    if (header.sequence > this.nextSequence) return reject('VOICE_PROTOCOL', 'audio frames arrived out of order', true)
    if (payload.byteLength > VOICE_LIMITS.maxFramePayloadBytes) return reject('VOICE_LIMIT', 'audio frame is too large', true)
    if (this.bytes - this.byteBase + payload.byteLength > this.options.maxBytes) return reject('VOICE_LIMIT', 'the recording exceeds the size limit', true)
    const limitMs = this.options.maxSeconds * 1000
    if (header.timestampMs - this.timeBaseMs > limitMs) return reject('VOICE_LIMIT', 'the recording exceeds the time limit', true)
    // The sender's timestamps are untrusted: the Hub's own clock bounds the stream too (5 s of slack for network jitter).
    if (this.options.now() - this.startedAt > limitMs + 5000) return reject('VOICE_LIMIT', 'the recording exceeds the time limit', true)
    this.nextSequence += 1
    this.bytes += payload.byteLength
    this.lastTimestampMs = Math.max(this.lastTimestampMs, header.timestampMs)
    if (header.final) this.ended = true
    return { ok: true }
  }
}

function reject(code: VoiceErrorCode, message: string, fatal: boolean): GateVerdict {
  return { ok: false, error: new VoiceError(code, message), fatal }
}
