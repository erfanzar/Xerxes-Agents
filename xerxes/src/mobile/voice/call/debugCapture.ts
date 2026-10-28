// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { CALL_SAMPLE_RATE, type VadUtteranceOffsets } from './logic.js'

export const MAX_DEBUG_UTTERANCE_SAMPLES = 30 * CALL_SAMPLE_RATE
export interface AudioLevelStats {
  readonly durationSeconds: number
  /** Linear, unnormalised full-scale amplitude (1.0 is digital full scale). */
  readonly peak: number
  readonly rms: number
}
export interface CallDebugCapture extends AudioLevelStats {
  readonly samples: Float32Array
  readonly sampleRate: typeof CALL_SAMPLE_RATE
  /** Null while transcription is pending; an empty string means no speech was recognised. */
  readonly transcript: string | null
  readonly vad: VadUtteranceOffsets
}

/** Pure measurements of the original samples; never normalises or modifies model input. */
export function audioLevelStats(samples: Float32Array, sampleRate = CALL_SAMPLE_RATE): AudioLevelStats {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) throw new Error('Invalid audio sample rate.')
  let peak = 0
  let energy = 0
  for (const raw of samples) {
    const sample = Number.isFinite(raw) ? raw : 0
    peak = Math.max(peak, Math.abs(sample))
    energy += sample * sample
  }
  return { durationSeconds: samples.length / sampleRate, peak, rms: samples.length ? Math.sqrt(energy / samples.length) : 0 }
}

/** Copy BEFORE transcription: a worker/GPU fallback may transfer and detach the input buffer. */
export function createDebugCapture(samples: Float32Array, vad: VadUtteranceOffsets): CallDebugCapture {
  if (samples.length > MAX_DEBUG_UTTERANCE_SAMPLES) throw new Error('Debug audio exceeds 30 seconds.')
  return { samples: samples.slice(), sampleRate: CALL_SAMPLE_RATE, transcript: null, vad: { ...vad }, ...audioLevelStats(samples) }
}

/** Pure RIFF/WAVE encoder: 16 kHz, mono, signed 16-bit little-endian PCM. */
export function encodeDebugWav(samples: Float32Array): ArrayBuffer {
  if (samples.length > MAX_DEBUG_UTTERANCE_SAMPLES) throw new Error('Debug audio exceeds 30 seconds.')
  const dataBytes = samples.length * 2
  const buffer = new ArrayBuffer(44 + dataBytes)
  const view = new DataView(buffer)
  const text = (offset: number, value: string): void => { for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i)) }
  text(0, 'RIFF')
  view.setUint32(4, 36 + dataBytes, true)
  text(8, 'WAVE')
  text(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true) // PCM
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, CALL_SAMPLE_RATE, true)
  view.setUint32(28, CALL_SAMPLE_RATE * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  text(36, 'data')
  view.setUint32(40, dataBytes, true)
  for (let i = 0; i < samples.length; i++) {
    const raw = samples[i]!
    const sample = Number.isFinite(raw) ? Math.max(-1, Math.min(1, raw)) : 0
    view.setInt16(44 + i * 2, Math.round(sample * (sample < 0 ? 32768 : 32767)), true)
  }
  return buffer
}
