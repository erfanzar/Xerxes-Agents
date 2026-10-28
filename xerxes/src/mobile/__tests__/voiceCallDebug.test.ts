// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { audioLevelStats, createDebugCapture, encodeDebugWav, MAX_DEBUG_UTTERANCE_SAMPLES } from '../voice/call/debugCapture.js'

const offsets = { startSample: 0, endSample: 4, preRollSamples: 0 }

describe('pure utterance audio measurements', () => {
  it('measures duration and unnormalised peak/RMS without changing samples', () => {
    const samples = Float32Array.from([0.5, -1, 0, 0.5])
    const original = samples.slice()
    expect(audioLevelStats(samples)).toEqual({ durationSeconds: 4 / 16000, peak: 1, rms: Math.sqrt(1.5 / 4) })
    expect(samples).toEqual(original)
    expect(audioLevelStats(Float32Array.of(2, -2))).toMatchObject({ peak: 2, rms: 2 })
  })
  it('handles empty, silent and non-finite input deterministically', () => {
    expect(audioLevelStats(new Float32Array())).toEqual({ durationSeconds: 0, peak: 0, rms: 0 })
    expect(audioLevelStats(new Float32Array(16000))).toEqual({ durationSeconds: 1, peak: 0, rms: 0 })
    expect(audioLevelStats(Float32Array.of(NaN, Infinity, -Infinity, 1))).toMatchObject({ peak: 1, rms: 0.5 })
    expect(() => audioLevelStats(new Float32Array(), 0)).toThrow('sample rate')
  })
  it('copies exact Float32 samples and offsets before a model can detach its input', () => {
    const samples = Float32Array.from([0.1, -0.3, 0.2, 1])
    const expected = samples.slice()
    const vad = { ...offsets }
    const capture = createDebugCapture(samples, vad)
    structuredClone(samples.buffer, { transfer: [samples.buffer] })
    vad.startSample = 3
    expect(samples.byteLength).toBe(0)
    expect(capture.samples).toEqual(expected)
    expect(capture.transcript).toBeNull()
    expect(capture.vad).toEqual(offsets)
    expect(capture.sampleRate).toBe(16000)
  })
  it('refuses oversized debug captures instead of silently truncating model input', () => {
    expect(() => createDebugCapture(new Float32Array(MAX_DEBUG_UTTERANCE_SAMPLES + 1), offsets)).toThrow('30 seconds')
    expect(createDebugCapture(new Float32Array(MAX_DEBUG_UTTERANCE_SAMPLES), offsets).durationSeconds).toBe(30)
  })
})

describe('pure 16-bit mono WAV encoding', () => {
  it('writes every RIFF/WAVE header field in little endian with the exact payload length', () => {
    const wav = encodeDebugWav(Float32Array.of(0, 1, -1))
    const view = new DataView(wav)
    const tag = (offset: number) => new TextDecoder().decode(new Uint8Array(wav, offset, 4))
    expect(wav.byteLength).toBe(50)
    expect(tag(0)).toBe('RIFF')
    expect(view.getUint32(4, true)).toBe(42)
    expect(tag(8)).toBe('WAVE')
    expect(tag(12)).toBe('fmt ')
    expect(view.getUint32(16, true)).toBe(16)
    expect(view.getUint16(20, true)).toBe(1)
    expect(view.getUint16(22, true)).toBe(1)
    expect(view.getUint32(24, true)).toBe(16000)
    expect(view.getUint32(28, true)).toBe(32000)
    expect(view.getUint16(32, true)).toBe(2)
    expect(view.getUint16(34, true)).toBe(16)
    expect(tag(36)).toBe('data')
    expect(view.getUint32(40, true)).toBe(6)
    expect([...new Uint8Array(wav, 44)]).toEqual([0, 0, 255, 127, 0, 128])
  })
  it('rounds signed samples, clips to PCM range, silences non-finite values and leaves input untouched', () => {
    const samples = Float32Array.from([-2, -1, -0.5, 0, 0.5, 1, 2, NaN, Infinity, -Infinity])
    const expected = [-32768, -32768, -16384, 0, 16384, 32767, 32767, 0, 0, 0]
    const original = samples.slice()
    const view = new DataView(encodeDebugWav(samples))
    expect(expected.map((_, i) => view.getInt16(44 + i * 2, true))).toEqual(expected)
    expect(samples).toEqual(original)
  })
  it('supports empty audio and exactly 30 seconds, with a hard 30-second bound', () => {
    const empty = encodeDebugWav(new Float32Array())
    expect(empty.byteLength).toBe(44)
    expect(new DataView(empty).getUint32(40, true)).toBe(0)
    expect(encodeDebugWav(new Float32Array(MAX_DEBUG_UTTERANCE_SAMPLES)).byteLength).toBe(44 + 30 * 16000 * 2)
    expect(() => encodeDebugWav(new Float32Array(MAX_DEBUG_UTTERANCE_SAMPLES + 1))).toThrow('30 seconds')
  })
})
