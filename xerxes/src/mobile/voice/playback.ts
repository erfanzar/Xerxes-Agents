// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { bytesToPcm16 } from '../../hub/voice/pcm.js'
import type { VoiceCodec, VoiceFrame } from '../../hub/voice/protocol.js'

/**
 * Playback of Hub speech. `ResponsePlayback` decides what may play (current
 * response and generation only, contiguous sequence) and when playback truly
 * ended; `AudioOutput` is the sound device. Interrupting flushes the device and
 * aborts what is scheduled. It is purely local: it never touches agents.
 */

export interface PlaybackClip {
  readonly codec: VoiceCodec
  readonly bytes: Uint8Array
  readonly sampleRate?: number
}

export type PlaybackErrorKind = 'blocked' | 'decode' | 'unsupported'

export class PlaybackError extends Error {
  readonly kind: PlaybackErrorKind
  constructor(kind: PlaybackErrorKind, message: string, cause?: unknown) {
    super(message, cause === undefined ? undefined : { cause })
    this.name = 'PlaybackError'
    this.kind = kind
  }
}

export interface AudioOutput {
  /**
   * Queue a clip after what is already queued. Calls `onStart` when it starts
   * sounding and resolves when it finished (or was flushed). Rejects with
   * `PlaybackError('blocked')` when the browser refuses to play sound.
   */
  schedule(clip: PlaybackClip, signal: AbortSignal, onStart: () => void): Promise<void>
  /** Stop everything that is sounding or queued, immediately. */
  flush(): void
  /** Call inside a user gesture so a later reply is allowed to play (autoplay policy). */
  unlock?(): Promise<void>
}

export interface PlaybackEvents {
  started(responseId: string, generation: number): void
  ended(responseId: string, generation: number): void
  failed(responseId: string, generation: number, error: PlaybackError): void
}

export interface PlaybackStart {
  readonly responseId: string
  readonly generation: number
  readonly codec: VoiceCodec
  readonly sampleRate?: number
}

interface Current {
  readonly responseId: string
  readonly generation: number
  readonly codec: VoiceCodec
  readonly sampleRate: number | undefined
  readonly abort: AbortController
  nextSequence: number
  parts: Uint8Array[]
  pending: number
  scheduled: number
  streamEnded: boolean
  started: boolean
  failed: boolean
}

export class ResponsePlayback {
  private current: Current | undefined

  constructor(private readonly output: AudioOutput, private readonly events: PlaybackEvents) {}

  get activeResponse(): { readonly responseId: string; readonly generation: number } | undefined {
    const current = this.current
    return current === undefined ? undefined : { responseId: current.responseId, generation: current.generation }
  }

  get playing(): boolean {
    return this.current?.started === true
  }

  /** A new response replaces the previous one: whatever it still had is flushed. */
  start(info: PlaybackStart): void {
    this.interrupt()
    this.current = {
      responseId: info.responseId, generation: info.generation, codec: info.codec, sampleRate: info.sampleRate,
      abort: new AbortController(), nextSequence: 0, parts: [], pending: 0, scheduled: 0, streamEnded: false, started: false, failed: false,
    }
  }

  /** False when the frame was rejected (no response, wrong generation, late or duplicate). */
  frame(frame: VoiceFrame): boolean {
    const current = this.current
    const { header, payload } = frame
    if (current === undefined || current.failed || header.stream !== 'tts' || header.generation !== current.generation) return false
    if (header.sequence < current.nextSequence) return false
    if (header.sequence > current.nextSequence) {
      this.fail(current, new PlaybackError('decode', 'Audio arrived out of order.'))
      return false
    }
    current.nextSequence += 1
    if (current.codec === 'pcm16') {
      if (payload.byteLength > 0) this.schedule(current, { codec: 'pcm16', bytes: payload.slice(), ...(current.sampleRate === undefined ? {} : { sampleRate: current.sampleRate }) })
      return true
    }
    if (payload.byteLength > 0) current.parts.push(payload.slice())
    if (header.clipEnd && current.parts.length > 0) {
      let total = 0
      for (const part of current.parts) total += part.byteLength
      const bytes = new Uint8Array(total)
      let offset = 0
      for (const part of current.parts) { bytes.set(part, offset); offset += part.byteLength }
      current.parts = []
      this.schedule(current, { codec: current.codec, bytes })
    }
    return true
  }

  /** The Hub finished sending (or cut) this response. */
  end(info: { readonly responseId: string; readonly generation: number; readonly interrupted: boolean }): void {
    const current = this.current
    if (current === undefined || current.responseId !== info.responseId || current.generation !== info.generation) return
    if (info.interrupted) { this.interrupt(); return }
    current.streamEnded = true
    this.maybeFinish(current)
  }

  /** Flush the device and forget the response. Safe to call at any time. */
  interrupt(): void {
    const current = this.current
    this.current = undefined
    if (current === undefined) return
    current.abort.abort()
    this.output.flush()
  }

  private schedule(current: Current, clip: PlaybackClip): void {
    current.pending += 1
    current.scheduled += 1
    const done = (): void => {
      current.pending -= 1
      this.maybeFinish(current)
    }
    this.output
      .schedule(clip, current.abort.signal, () => {
        if (this.current !== current || current.started) return
        current.started = true
        this.events.started(current.responseId, current.generation)
      })
      .then(done, (error: unknown) => {
        current.pending -= 1
        this.fail(current, error instanceof PlaybackError ? error : new PlaybackError('decode', 'The audio could not be played.', error))
      })
  }

  private maybeFinish(current: Current): void {
    if (this.current !== current || current.failed || !current.streamEnded || current.pending > 0) return
    this.current = undefined
    // A response that never produced sound has nothing to report as ended.
    if (current.started) this.events.ended(current.responseId, current.generation)
  }

  private fail(current: Current, error: PlaybackError): void {
    if (this.current !== current || current.failed) return
    current.failed = true
    this.current = undefined
    current.abort.abort()
    this.output.flush()
    this.events.failed(current.responseId, current.generation, error)
  }
}

// ------------------------------------------------------------------ Web Audio

/** The slice of AudioContext used here. */
export interface AudioContextLike {
  readonly state: string
  readonly currentTime: number
  readonly destination: AudioNode
  resume(): Promise<void>
  decodeAudioData(data: ArrayBuffer): Promise<AudioBuffer>
  createBuffer(channels: number, length: number, sampleRate: number): AudioBuffer
  createBufferSource(): AudioBufferSourceNode
}

export function createWebAudioOutput(context: AudioContextLike): AudioOutput {
  const live = new Map<AudioBufferSourceNode, () => void>()
  let nextStart = 0
  let tail: Promise<void> = Promise.resolve()

  async function ensureRunning(): Promise<void> {
    if (context.state === 'running') return
    try { await context.resume() } catch (error) {
      throw new PlaybackError('blocked', 'The browser blocked sound. Tap the button once to allow playback.', error)
    }
    if (context.state !== 'running') throw new PlaybackError('blocked', 'The browser blocked sound. Tap the button once to allow playback.')
  }

  async function toBuffer(clip: PlaybackClip): Promise<AudioBuffer> {
    if (clip.codec === 'pcm16') {
      const samples = bytesToPcm16(clip.bytes)
      const buffer = context.createBuffer(1, Math.max(1, samples.length), clip.sampleRate ?? 24_000)
      const channel = buffer.getChannelData(0)
      for (let index = 0; index < samples.length; index += 1) channel[index] = (samples[index] ?? 0) / 0x8000
      return buffer
    }
    try {
      return await context.decodeAudioData(clip.bytes.slice().buffer)
    } catch (error) {
      throw new PlaybackError('decode', 'This audio format could not be decoded.', error)
    }
  }

  return {
    unlock: ensureRunning,
    schedule(clip, signal, onStart) {
      const previous = tail
      let release: () => void = () => undefined
      tail = new Promise<void>(resolve => { release = resolve })
      return (async () => {
        try {
          await previous
          if (signal.aborted) return
          await ensureRunning()
          const buffer = await toBuffer(clip)
          if (signal.aborted) return
          const source = context.createBufferSource()
          source.buffer = buffer
          source.connect(context.destination)
          const startAt = Math.max(context.currentTime + 0.02, nextStart)
          nextStart = startAt + buffer.duration
          const ended = new Promise<void>(resolve => {
            live.set(source, resolve)
            source.onended = () => { live.delete(source); resolve() }
          })
          source.start(startAt)
          setTimeout(() => { if (!signal.aborted) onStart() }, Math.max(0, (startAt - context.currentTime) * 1000))
          release()
          await ended
        } finally {
          release()
        }
      })()
    },
    flush() {
      for (const [source, resolve] of live) {
        source.onended = null
        try { source.stop() } catch { /* not started */ }
        resolve()
      }
      live.clear()
      nextStart = 0
    },
  }
}

export function browserAudioOutput(): AudioOutput | undefined {
  if (typeof AudioContext === 'undefined') return undefined
  return createWebAudioOutput(new AudioContext())
}
