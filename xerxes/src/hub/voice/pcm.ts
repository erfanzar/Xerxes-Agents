// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Browser-safe PCM16 helpers shared by the Hub (resampling to a provider's
 * rate) and the phone (AudioWorklet capture). Samples are little-endian,
 * mono, signed 16-bit.
 */

/** Convert float samples in [-1, 1] to PCM16. Out-of-range input is clipped, NaN becomes silence. */
export function floatToPcm16(input: Float32Array): Int16Array {
  const out = new Int16Array(input.length)
  for (let index = 0; index < input.length; index += 1) {
    const sample = input[index] ?? 0
    const clipped = Number.isFinite(sample) ? Math.max(-1, Math.min(1, sample)) : 0
    out[index] = clipped < 0 ? Math.round(clipped * 0x8000) : Math.round(clipped * 0x7fff)
  }
  return out
}

export function pcm16ToBytes(samples: Int16Array): Uint8Array {
  const out = new Uint8Array(samples.length * 2)
  const view = new DataView(out.buffer)
  for (let index = 0; index < samples.length; index += 1) view.setInt16(index * 2, samples[index] ?? 0, true)
  return out
}

/** Decode little-endian PCM16 bytes. A trailing odd byte is dropped (a frame boundary can split a sample). */
export function bytesToPcm16(bytes: Uint8Array): Int16Array {
  const count = Math.floor(bytes.byteLength / 2)
  const out = new Int16Array(count)
  const view = new DataView(bytes.buffer, bytes.byteOffset, count * 2)
  for (let index = 0; index < count; index += 1) out[index] = view.getInt16(index * 2, true)
  return out
}

/** Linear-interpolation resampler. Rates must be positive integers. */
export function resamplePcm16(input: Int16Array, fromRate: number, toRate: number): Int16Array {
  if (!Number.isInteger(fromRate) || !Number.isInteger(toRate) || fromRate < 1 || toRate < 1) {
    throw new RangeError('sample rates must be positive integers')
  }
  if (fromRate === toRate || input.length === 0) return input.slice()
  const length = Math.max(1, Math.round((input.length * toRate) / fromRate))
  const out = new Int16Array(length)
  const ratio = fromRate / toRate
  for (let index = 0; index < length; index += 1) {
    const position = index * ratio
    const left = Math.min(input.length - 1, Math.floor(position))
    const right = Math.min(input.length - 1, left + 1)
    const fraction = position - left
    out[index] = Math.round((input[left] ?? 0) * (1 - fraction) + (input[right] ?? 0) * fraction)
  }
  return out
}

export function pcm16DurationMs(byteLength: number, sampleRate: number): number {
  return (byteLength / 2 / sampleRate) * 1000
}

/**
 * Collect small capture frames (20-40 ms) into larger messages (40-100 ms).
 * `push` returns zero or one ready message; `flush` returns the remainder.
 */
export class PcmAggregator {
  private parts: Int16Array[] = []
  private samples = 0
  constructor(private readonly sampleRate: number, private readonly targetMs = 60) {
    if (!Number.isInteger(sampleRate) || sampleRate < 8000) throw new RangeError('sampleRate must be an integer of at least 8000')
    if (targetMs < 20 || targetMs > 200) throw new RangeError('targetMs must be from 20 to 200')
  }

  push(frame: Int16Array): Int16Array | undefined {
    if (frame.length === 0) return undefined
    this.parts.push(frame)
    this.samples += frame.length
    return (this.samples / this.sampleRate) * 1000 >= this.targetMs ? this.flush() : undefined
  }

  flush(): Int16Array | undefined {
    if (this.samples === 0) return undefined
    const out = new Int16Array(this.samples)
    let offset = 0
    for (const part of this.parts) {
      out.set(part, offset)
      offset += part.length
    }
    this.parts = []
    this.samples = 0
    return out
  }
}
