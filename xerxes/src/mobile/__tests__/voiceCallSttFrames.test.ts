// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'vitest'

import { STT_FRAME_SAMPLES, padToSttFrames } from '../voice/call/logic.js'

describe('speech-to-text input length', () => {
  it('pads to a whole number of 80-sample encoder frames, which Moonshine Streaming needs to not crash', () => {
    // A VAD cut: 512-sample frames plus pre-roll, so the length is rarely a multiple of 80.
    const samples = Float32Array.from({ length: 81_466 }, (_, i) => Math.sin(i / 10))
    const padded = padToSttFrames(samples)
    expect(padded.length % STT_FRAME_SAMPLES).toBe(0)
    expect(padded.length).toBe(81_520)
    expect(Array.from(padded.subarray(0, samples.length))).toEqual(Array.from(samples))
    expect(Array.from(padded.subarray(samples.length)).every(value => value === 0)).toBe(true)
  })
  it('leaves an aligned clip untouched and handles one extra sample', () => {
    const aligned = new Float32Array(44_400)
    expect(padToSttFrames(aligned)).toBe(aligned)
    expect(padToSttFrames(new Float32Array(44_401)).length).toBe(44_480)
    expect(padToSttFrames(new Float32Array(0)).length).toBe(0)
  })
})
