// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { Timers } from '../../client/hubConnection.js'
import { VOICE_LIMITS } from '../../hub/voice/protocol.js'
import { chooseRecorderFormat, type RecorderFormat } from './mime.js'

/**
 * Microphone capture over MediaRecorder, behind small interfaces so tests use
 * fakes (no DOM is installed). Guarantees:
 * - the microphone is opened only after the caller has checked capabilities;
 * - a permission that resolves after `cancel` (late permission) is released
 *   immediately and never records;
 * - every path out (stop, cancel, error, limit) stops all tracks;
 * - chunks are delivered in order, and nothing is delivered after cancel.
 */

export interface TrackLike { stop(): void }
export interface StreamLike { getTracks(): readonly TrackLike[] }
export interface BlobLike { readonly size: number; arrayBuffer(): Promise<ArrayBuffer> }

export interface RecorderLike {
  readonly state: 'inactive' | 'recording' | 'paused'
  ondataavailable: ((event: { readonly data: BlobLike }) => void) | null
  onstop: (() => void) | null
  onerror: (() => void) | null
  start(timesliceMs: number): void
  stop(): void
}

export interface LevelMeter {
  /** Current microphone level in [0, 1]. */
  read(): number
  close(): void
}

export interface MicrophoneDeps {
  getUserMedia(): Promise<StreamLike>
  createRecorder(stream: StreamLike, mimeType: string): RecorderLike
  isTypeSupported(mimeType: string): boolean
  level?(stream: StreamLike): LevelMeter | undefined
}

export type MicrophoneErrorKind = 'unsupported' | 'denied' | 'unavailable' | 'format' | 'cancelled'

export class MicrophoneError extends Error {
  readonly kind: MicrophoneErrorKind
  constructor(kind: MicrophoneErrorKind, message: string, cause?: unknown) {
    super(message, cause === undefined ? undefined : { cause })
    this.name = 'MicrophoneError'
    this.kind = kind
  }
}

export function classifyMediaError(error: unknown): MicrophoneError {
  const name = typeof error === 'object' && error !== null && 'name' in error ? String((error as { name: unknown }).name) : ''
  if (name === 'NotAllowedError' || name === 'SecurityError' || name === 'PermissionDeniedError') {
    return new MicrophoneError('denied', 'Microphone access was denied. Allow it for this site in your browser settings, or type instead.', error)
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
    return new MicrophoneError('unavailable', 'No microphone was found on this device.', error)
  }
  if (name === 'NotReadableError' || name === 'AbortError') {
    return new MicrophoneError('unavailable', 'The microphone is busy or could not be started.', error)
  }
  return new MicrophoneError('unavailable', 'The microphone is not available.', error)
}

export interface CaptureOptions {
  /** Slice length; clamped to the 250-1000 ms the Hub accepts. */
  readonly sliceMs?: number
  readonly maxSeconds: number
  readonly maxBytes: number
  /** Aborted when the user cancels; also what releases a late permission grant. */
  readonly signal: AbortSignal
  readonly onChunk: (bytes: Uint8Array) => void
  readonly onLimit: (reason: 'time' | 'size') => void
  readonly onError: (error: Error) => void
  readonly timers?: Timers
}

export interface CaptureSession {
  readonly format: RecorderFormat
  /** Stop recording, deliver every pending chunk, release the microphone. */
  stop(): Promise<void>
  /** Drop everything and release the microphone. */
  cancel(): void
  level(): number
}

const REAL_TIMERS: Timers = {
  setTimeout: (callback, ms) => setTimeout(callback, ms),
  clearTimeout: handle => clearTimeout(handle as ReturnType<typeof setTimeout>),
}

export async function startCapture(mic: MicrophoneDeps, options: CaptureOptions): Promise<CaptureSession> {
  const timers = options.timers ?? REAL_TIMERS
  const format = chooseRecorderFormat(mimeType => mic.isTypeSupported(mimeType))
  if (format === undefined) throw new MicrophoneError('format', 'This browser cannot record in a format the Hub accepts. Type instead.')
  const sliceMs = Math.min(VOICE_LIMITS.maxSliceMs, Math.max(VOICE_LIMITS.minSliceMs, options.sliceMs ?? 500))

  let stream: StreamLike
  try {
    stream = await mic.getUserMedia()
  } catch (error) {
    throw classifyMediaError(error)
  }
  let meter: LevelMeter | undefined
  let released = false
  const release = (): void => {
    if (released) return
    released = true
    try { meter?.close() } catch { /* already closed */ }
    for (const track of stream.getTracks()) {
      try { track.stop() } catch { /* already stopped */ }
    }
  }
  if (options.signal.aborted) {
    // The permission prompt was answered after the user let go: never record.
    release()
    throw new MicrophoneError('cancelled', 'The recording was cancelled.')
  }
  try { meter = mic.level?.(stream) } catch { meter = undefined }

  let recorder: RecorderLike
  try {
    recorder = mic.createRecorder(stream, format.mimeType)
  } catch (error) {
    release()
    throw new MicrophoneError('unsupported', 'The recorder could not be created.', error)
  }

  let total = 0
  let dropped = false
  let limited = false
  let chain: Promise<void> = Promise.resolve()
  let timer: unknown

  const stopRecorder = (): void => {
    if (recorder.state === 'inactive') return
    try { recorder.stop() } catch { /* already stopped */ }
  }
  const limit = (reason: 'time' | 'size'): void => {
    if (limited) return
    limited = true
    options.onLimit(reason)
    stopRecorder()
  }

  const finished = new Promise<void>(resolve => { recorder.onstop = () => resolve() })
  recorder.ondataavailable = event => {
    const blob = event.data
    if (blob.size === 0) return
    chain = chain
      .then(async () => {
        if (dropped) return
        const bytes = new Uint8Array(await blob.arrayBuffer())
        if (dropped) return
        if (total + bytes.byteLength > options.maxBytes) { limit('size'); return }
        total += bytes.byteLength
        options.onChunk(bytes)
      })
      .catch(() => options.onError(new MicrophoneError('unavailable', 'A recording chunk could not be read.')))
  }
  recorder.onerror = () => options.onError(new MicrophoneError('unavailable', 'The recording failed.'))

  try {
    recorder.start(sliceMs)
  } catch (error) {
    release()
    throw new MicrophoneError('unsupported', 'Recording could not start.', error)
  }
  timer = timers.setTimeout(() => limit('time'), options.maxSeconds * 1000)

  return {
    format,
    level: () => {
      try { return meter?.read() ?? 0 } catch { return 0 }
    },
    async stop() {
      timers.clearTimeout(timer)
      let settleTimer: unknown
      const settle = new Promise<void>(resolve => { settleTimer = timers.setTimeout(resolve, 2000) })
      stopRecorder()
      await Promise.race([finished, settle])
      timers.clearTimeout(settleTimer)
      await chain
      dropped = true
      release()
    },
    cancel() {
      dropped = true
      timers.clearTimeout(timer)
      stopRecorder()
      release()
    },
  }
}

/** The browser microphone, or undefined when this page cannot record. */
export function browserMicrophone(): MicrophoneDeps | undefined {
  if (typeof navigator === 'undefined' || typeof MediaRecorder === 'undefined') return undefined
  const devices = (navigator as Navigator & { mediaDevices?: MediaDevices }).mediaDevices
  if (devices === undefined || typeof devices.getUserMedia !== 'function') return undefined
  return {
    getUserMedia: () => devices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } }),
    createRecorder: (stream, mimeType) => new MediaRecorder(stream as MediaStream, { mimeType }) as unknown as RecorderLike,
    isTypeSupported: mimeType => MediaRecorder.isTypeSupported(mimeType),
    level: stream => {
      if (typeof AudioContext === 'undefined') return undefined
      const context = new AudioContext()
      const analyser = context.createAnalyser()
      analyser.fftSize = 512
      context.createMediaStreamSource(stream as MediaStream).connect(analyser)
      const data = new Uint8Array(analyser.fftSize)
      return {
        read() {
          analyser.getByteTimeDomainData(data)
          let peak = 0
          for (const value of data) peak = Math.max(peak, Math.abs(value - 128))
          return Math.min(1, peak / 64)
        },
        close() { void context.close().catch(() => undefined) },
      }
    },
  }
}
