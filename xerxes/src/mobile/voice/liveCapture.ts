// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { classifyMediaError, MicrophoneError } from './capture.js'

/**
 * Mono float capture for live voice, behind an interface so tests use fakes.
 * `open` honours its signal: a microphone grant that arrives after the signal
 * aborted is released at once and `open` rejects as cancelled.
 */
export interface LiveCapture {
  /** Rate of the samples delivered to `onFrame` (negotiated with the Hub, never hard-coded). */
  readonly sampleRate: number
  onFrame(listener: (samples: Float32Array) => void): void
  close(): void
}

export interface LiveCaptureDeps {
  open(signal: AbortSignal): Promise<LiveCapture>
}

export const LIVE_WORKLET_URL = '/voice-worklet.js'
const PROCESSOR = 'xerxes-pcm-capture'

export function browserLiveCaptureDeps(workletUrl: string = LIVE_WORKLET_URL): LiveCaptureDeps | undefined {
  if (typeof navigator === 'undefined' || typeof AudioContext === 'undefined' || typeof AudioWorkletNode === 'undefined') return undefined
  const devices = (navigator as Navigator & { mediaDevices?: MediaDevices }).mediaDevices
  if (devices === undefined || typeof devices.getUserMedia !== 'function') return undefined
  return {
    async open(signal) {
      let stream: MediaStream
      try {
        stream = await devices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } })
      } catch (error) {
        throw classifyMediaError(error)
      }
      const stopTracks = (): void => { for (const track of stream.getTracks()) track.stop() }
      if (signal.aborted) {
        stopTracks()
        throw new MicrophoneError('cancelled', 'The live session was cancelled.')
      }
      const context = new AudioContext()
      const closeAll = (): void => {
        stopTracks()
        void context.close().catch(() => undefined)
      }
      try {
        await context.audioWorklet.addModule(workletUrl)
        if (signal.aborted) throw new MicrophoneError('cancelled', 'The live session was cancelled.')
        const source = context.createMediaStreamSource(stream)
        const node = new AudioWorkletNode(context, PROCESSOR, { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1 })
        // Some browsers only run a node that reaches the destination; a muted gain keeps the mic silent in the speaker.
        const mute = context.createGain()
        mute.gain.value = 0
        source.connect(node)
        node.connect(mute)
        mute.connect(context.destination)
        let listener: ((samples: Float32Array) => void) | undefined
        node.port.onmessage = event => {
          if (event.data instanceof Float32Array) listener?.(event.data)
        }
        return {
          sampleRate: context.sampleRate,
          onFrame: next => { listener = next },
          close: () => {
            listener = undefined
            node.port.onmessage = null
            try { source.disconnect() } catch { /* already disconnected */ }
            closeAll()
          },
        }
      } catch (error) {
        closeAll()
        if (error instanceof MicrophoneError) throw error
        throw new MicrophoneError('unavailable', 'Live voice could not start on this browser.', error)
      }
    },
  }
}
