// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * AudioWorklet processor for live voice. It runs on the audio thread and only
 * collects mono float samples into ~20 ms frames and posts them to the page,
 * which converts to PCM16 and aggregates. Built to `/voice-worklet.js` (same
 * origin: the Hub's CSP forbids blob: scripts).
 *
 * The declarations are module-local; the worklet scope provides the real ones.
 */

declare const sampleRate: number
declare class AudioWorkletProcessorBase {
  readonly port: { postMessage(message: unknown, transfer?: Transferable[]): void }
}
declare function registerProcessor(name: string, processor: new () => AudioWorkletProcessorBase & { process(inputs: Float32Array[][]): boolean }): void
declare const AudioWorkletProcessor: typeof AudioWorkletProcessorBase

export const PCM_CAPTURE_PROCESSOR = 'xerxes-pcm-capture'

class PcmCaptureProcessor extends AudioWorkletProcessor {
  private readonly frame = new Float32Array(Math.max(128, Math.round(sampleRate * 0.02)))
  private filled = 0

  process(inputs: Float32Array[][]): boolean {
    const channel = inputs[0]?.[0]
    if (channel === undefined) return true
    let offset = 0
    while (offset < channel.length) {
      const take = Math.min(channel.length - offset, this.frame.length - this.filled)
      this.frame.set(channel.subarray(offset, offset + take), this.filled)
      this.filled += take
      offset += take
      if (this.filled === this.frame.length) {
        const out = this.frame.slice()
        this.port.postMessage(out, [out.buffer])
        this.filled = 0
      }
    }
    return true
  }
}

registerProcessor(PCM_CAPTURE_PROCESSOR, PcmCaptureProcessor)
