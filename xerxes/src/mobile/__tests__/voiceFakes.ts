// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

// JSX-free on purpose (see helpers.ts): the runtime tsconfig type-checks every .ts file under src/.

import { VoiceLink, type VoiceSocketFactory, type VoiceSocketHandlers, type VoiceSocketLike } from '../../client/voiceLink.js'
import { createVoiceStateStore } from '../../client/voiceState.js'
import { decodeVoiceFrame, encodeVoiceFrame, type VoiceCapabilities, type VoiceFrame, type VoiceFrameHeader } from '../../hub/voice/protocol.js'
import { MicrophoneError, type BlobLike, type MicrophoneDeps, type RecorderLike, type StreamLike, type TrackLike } from '../voice/capture.js'
import { VoiceClientCore } from '../voice/core.js'
import type { LiveCapture, LiveCaptureDeps } from '../voice/liveCapture.js'
import type { AudioOutput, PlaybackClip } from '../voice/playback.js'

export type Json = Record<string, unknown>

export const tick = (): Promise<void> => new Promise(resolve => { setTimeout(resolve, 0) })

/** Let queued microtask chains and zero-delay timers finish. */
export async function flush(times = 6): Promise<void> {
  for (let index = 0; index < times; index += 1) await tick()
}

export function deferred<T = void>(): { promise: Promise<T>; resolve: (value: T) => void; reject: (error: unknown) => void } {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej })
  return { promise, resolve, reject }
}

export function caps(over: Partial<VoiceCapabilities> = {}): VoiceCapabilities {
  return {
    enabled: true,
    modes: ['push_to_talk', 'live'],
    stt: { available: true, provider: 'fake-stt' },
    tts: { available: true, provider: 'fake-tts', delivery: 'buffered', format: 'mp3' },
    realtime: { available: true, provider: 'fake-realtime', sample_rate: 24_000 },
    limits: { max_utterance_seconds: 60, max_utterance_bytes: 8 * 1024 * 1024, slice_ms_min: 250, slice_ms_max: 1000, max_session_seconds: 300, idle_seconds: 30 },
    confirm_transcript: true,
    browser_fallback: false,
    audio_retention: 'none',
    ...over,
  }
}

export function ttsFrameBytes(generation: number, sequence: number, payload: readonly number[] | Uint8Array, over: Partial<VoiceFrameHeader> = {}): Uint8Array {
  const bytes = payload instanceof Uint8Array ? payload : new Uint8Array(payload)
  return encodeVoiceFrame({ stream: 'tts', codec: 'mp3', generation, sequence, timestampMs: 0, final: false, clipEnd: false, ...over }, bytes)
}

export function ttsFrame(generation: number, sequence: number, payload: readonly number[] | Uint8Array, over: Partial<VoiceFrameHeader> = {}): VoiceFrame {
  return decodeVoiceFrame(ttsFrameBytes(generation, sequence, payload, over))
}

// ------------------------------------------------------------------ Hub socket

export class FakeVoiceSocket implements VoiceSocketLike {
  readonly controls: Json[] = []
  readonly frames: VoiceFrame[] = []
  closedWith: number | undefined

  constructor(readonly handlers: VoiceSocketHandlers, private readonly hub: FakeVoiceHub) {}

  send(data: string | Uint8Array): void {
    if (typeof data === 'string') {
      const message = JSON.parse(data) as Json
      this.controls.push(message)
      this.hub.onControl(this, message)
    } else {
      this.frames.push(decodeVoiceFrame(data))
    }
  }
  close(code?: number): void { this.closedWith = code ?? 1000 }
  bufferedAmount(): number { return this.hub.buffered }
  server(message: Json): void { this.handlers.text(JSON.stringify({ v: 1, ...message })) }
  serverFrame(bytes: Uint8Array): void { this.handlers.binary(bytes) }
  drop(code = 1006): void { this.handlers.close(code, '') }
  ofType(type: string): Json[] { return this.controls.filter(message => message.type === type) }
}

export class FakeVoiceHub {
  readonly sockets: FakeVoiceSocket[] = []
  buffered = 0
  capabilities: VoiceCapabilities = caps()
  generation = 7
  transcript = 'hello there'
  /** Hold `start` acknowledgements until `releaseStart()`. */
  holdStart = false
  /** Close every socket before it opens. */
  refuse = false
  /** Never answer a capabilities request. */
  silent = false
  startError: { code: string; message: string } | undefined
  commitError: { code: string; message: string } | undefined
  private held: { socket: FakeVoiceSocket; message: Json }[] = []

  readonly factory: VoiceSocketFactory = (_url, handlers) => {
    const socket = new FakeVoiceSocket(handlers, this)
    this.sockets.push(socket)
    queueMicrotask(() => { if (this.refuse) handlers.close(1006, ''); else handlers.open() })
    return socket
  }

  get socket(): FakeVoiceSocket {
    const found = this.sockets.at(-1)
    if (found === undefined) throw new Error('no voice socket was opened')
    return found
  }

  onControl(socket: FakeVoiceSocket, message: Json): void {
    switch (message.type) {
      case 'capabilities':
        if (!this.silent) socket.server({ type: 'capabilities', capabilities: this.capabilities })
        return
      case 'start':
        if (this.startError !== undefined) { socket.server({ type: 'error', ...this.startError, utterance_id: message.utterance_id }); return }
        if (this.holdStart) { this.held.push({ socket, message }); return }
        this.acceptStart(socket, message)
        return
      case 'commit':
        if (this.commitError !== undefined) { socket.server({ type: 'error', ...this.commitError, utterance_id: message.utterance_id }); return }
        socket.server({ type: 'transcript', utterance_id: message.utterance_id, revision: 1, text: this.transcript, final: true })
        return
      default:
    }
  }

  acceptStart(socket: FakeVoiceSocket, message: Json): void {
    socket.server({ type: 'state', state: 'listening', generation: this.generation, utterance_id: message.utterance_id })
  }

  releaseStart(): void {
    for (const entry of this.held.splice(0)) this.acceptStart(entry.socket, entry.message)
  }
}

// ------------------------------------------------------------------ microphone

export class FakeTrack implements TrackLike {
  stopped = 0
  stop(): void { this.stopped += 1 }
}

export class FakeStream implements StreamLike {
  readonly track = new FakeTrack()
  getTracks(): readonly TrackLike[] { return [this.track] }
}

export class FakeBlob implements BlobLike {
  constructor(readonly bytes: Uint8Array) {}
  get size(): number { return this.bytes.byteLength }
  async arrayBuffer(): Promise<ArrayBuffer> { return this.bytes.slice().buffer as ArrayBuffer }
}

export class FakeRecorder implements RecorderLike {
  state: 'inactive' | 'recording' | 'paused' = 'inactive'
  ondataavailable: RecorderLike['ondataavailable'] = null
  onstop: RecorderLike['onstop'] = null
  onerror: RecorderLike['onerror'] = null
  startedWith: number | undefined
  constructor(readonly mimeType: string) {}
  start(timesliceMs: number): void { this.state = 'recording'; this.startedWith = timesliceMs }
  stop(): void { this.state = 'inactive'; this.onstop?.() }
  emit(bytes: readonly number[] | Uint8Array): void {
    this.ondataavailable?.({ data: new FakeBlob(bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes)) })
  }
}

export class FakeMic implements MicrophoneDeps {
  supported = new Set<string>(['audio/webm;codecs=opus'])
  /** Holds the permission prompt open until resolved. */
  gate: Promise<void> | undefined
  denied: { name: string } | undefined
  opened = 0
  readonly streams: FakeStream[] = []
  readonly recorders: FakeRecorder[] = []
  onCreate: ((recorder: FakeRecorder) => void) | undefined

  async getUserMedia(): Promise<StreamLike> {
    this.opened += 1
    if (this.gate !== undefined) await this.gate
    if (this.denied !== undefined) throw this.denied
    const stream = new FakeStream()
    this.streams.push(stream)
    return stream
  }
  createRecorder(_stream: StreamLike, mimeType: string): RecorderLike {
    const recorder = new FakeRecorder(mimeType)
    this.recorders.push(recorder)
    this.onCreate?.(recorder)
    return recorder
  }
  isTypeSupported(mimeType: string): boolean { return this.supported.has(mimeType) }
}

// ------------------------------------------------------------------ live capture

export class FakeLiveCapture implements LiveCapture {
  closed = 0
  private listener: ((samples: Float32Array) => void) | undefined
  constructor(readonly sampleRate = 48_000) {}
  onFrame(listener: (samples: Float32Array) => void): void { this.listener = listener }
  close(): void { this.closed += 1; this.listener = undefined }
  emit(samples: Float32Array): void { this.listener?.(samples) }
  /** `count` 20 ms frames at the capture rate, all at `level`. */
  frames(count: number, level: number): void {
    const size = Math.round(this.sampleRate * 0.02)
    for (let index = 0; index < count; index += 1) this.emit(new Float32Array(size).fill(level))
  }
}

export class FakeLiveDeps implements LiveCaptureDeps {
  gate: Promise<void> | undefined
  denied = false
  opens = 0
  readonly capture = new FakeLiveCapture()

  async open(signal: AbortSignal): Promise<LiveCapture> {
    this.opens += 1
    if (this.gate !== undefined) await this.gate
    if (this.denied) throw new MicrophoneError('denied', 'Microphone access was denied.')
    if (signal.aborted) {
      // Same contract as the browser implementation: a late grant is released at once.
      this.capture.close()
      throw new MicrophoneError('cancelled', 'The live session was cancelled.')
    }
    return this.capture
  }
}

// ------------------------------------------------------------------ playback

export interface Scheduled {
  readonly clip: PlaybackClip
  readonly signal: AbortSignal
  start(): void
  finish(): void
  fail(error: Error): void
}

export class FakeOutput implements AudioOutput {
  readonly scheduled: Scheduled[] = []
  flushes = 0
  unlocked = 0
  schedule(clip: PlaybackClip, signal: AbortSignal, onStart: () => void): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      this.scheduled.push({ clip, signal, start: onStart, finish: resolve, fail: reject })
    })
  }
  flush(): void { this.flushes += 1 }
  async unlock(): Promise<void> { this.unlocked += 1 }
}

// ------------------------------------------------------------------ rig

export function voiceRig(options: { readonly caps?: VoiceCapabilities; readonly output?: FakeOutput } = {}) {
  const hub = new FakeVoiceHub()
  if (options.caps !== undefined) hub.capabilities = options.caps
  const mic = new FakeMic()
  const live = new FakeLiveDeps()
  const state = createVoiceStateStore()
  const createLink = (): VoiceLink => new VoiceLink({ url: 'wss://hub.test/api/hub/v1/voice', socketFactory: hub.factory })
  const core = new VoiceClientCore({ createLink, output: options.output, state })
  return { hub, mic, live, state, core, createLink }
}

export type VoiceRig = ReturnType<typeof voiceRig>
