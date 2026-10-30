// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { classifyMediaError, MicrophoneError } from './capture.js'

/**
 * What the phone did to a running capture: `paused` (a phone call, Siri or a background trip suspended
 * the audio or muted the track), `resumed` (it runs again), or `lost` (the track ended; only reopening
 * the microphone brings it back).
 */
export type CaptureInterruption = 'paused' | 'resumed' | 'lost'

/**
 * Mono float capture for live voice, behind an interface so tests use fakes.
 * `open` honours its signal: a microphone grant that arrives after the signal
 * aborted is released at once and `open` rejects as cancelled.
 */
export interface LiveCapture {
  /** Rate of the samples delivered to `onFrame` (negotiated with the Hub, never hard-coded). */
  readonly sampleRate: number
  onFrame(listener: (samples: Float32Array) => void): void
  /** Optional: audio-session interruptions, so a call can recover instead of going silently deaf. */
  onInterruption?(listener: (event: CaptureInterruption) => void): void
  /** Optional: try to resume after an interruption; true when the capture runs again. */
  resume?(): Promise<boolean>
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
      // Created and resumed synchronously, inside the tap that started the call (WebKit only lets a
      // gesture start audio). The permission prompt is also requested from the same tap, before any await.
      let context: AudioContext
      try { context = new AudioContext() } catch (error) {
        throw new MicrophoneError('unavailable', 'Microphone audio could not initialize.', error)
      }
      let resumed: Promise<void>
      try { resumed = context.resume().catch(() => undefined) } catch { resumed = Promise.resolve() }
      let stream: MediaStream | undefined
      let closed = false
      const stopTracks = (): void => { for (const track of stream?.getTracks() ?? []) track.stop() }
      const closeAll = (): void => {
        if (closed) return
        closed = true
        signal.removeEventListener('abort', closeAll)
        stopTracks()
        void context.close().catch(() => undefined)
      }
      signal.addEventListener('abort', closeAll, { once: true })
      try {
        stream = await devices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } })
      } catch (error) {
        closeAll()
        throw classifyMediaError(error)
      }
      if (signal.aborted || closed) {
        // Hang-up came while the permission prompt was open: release the grant that just arrived.
        for (const track of stream.getTracks()) track.stop()
        closeAll()
        throw new MicrophoneError('cancelled', 'The live session was cancelled.')
      }
      try {
        await context.audioWorklet.addModule(workletUrl)
        if (signal.aborted) throw new MicrophoneError('cancelled', 'The live session was cancelled.')
        await resumed
        if (context.state !== 'running') await context.resume()
        if (context.state !== 'running') throw new MicrophoneError('unavailable', 'The browser blocked microphone audio. Tap Start again to allow it.')
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
        let interrupted: ((event: CaptureInterruption) => void) | undefined
        node.port.onmessage = event => {
          if (event.data instanceof Float32Array) listener?.(event.data)
        }
        const tracks = stream.getAudioTracks?.() ?? stream.getTracks()
        const report = (event: CaptureInterruption): void => { if (!closed) interrupted?.(event) }
        const tryResume = async (): Promise<boolean> => {
          if (closed) return false
          if (tracks.some(track => track.readyState === 'ended')) return false
          if (context.state !== 'running') { try { await context.resume() } catch { /* reported below */ } }
          return !closed && context.state === 'running' && !tracks.some(track => track.muted)
        }
        const onState = (): void => {
          if (closed) return
          const state: string = context.state
          if (state === 'running') { report('resumed'); return }
          if (state === 'closed') { report('lost'); return }
          // 'suspended' or WebKit's 'interrupted': ask once, then report a pause the call can show.
          void tryResume().then(running => report(running ? 'resumed' : 'paused'))
        }
        const onEnded = (): void => report('lost')
        const onMute = (): void => report('paused')
        const onUnmute = (): void => report('resumed')
        context.addEventListener?.('statechange', onState)
        for (const track of tracks) {
          track.addEventListener?.('ended', onEnded)
          track.addEventListener?.('mute', onMute)
          track.addEventListener?.('unmute', onUnmute)
        }
        return {
          sampleRate: context.sampleRate,
          onFrame: next => { listener = next },
          onInterruption: next => { interrupted = next },
          resume: tryResume,
          close: () => {
            listener = undefined
            interrupted = undefined
            node.port.onmessage = null
            context.removeEventListener?.('statechange', onState)
            for (const track of tracks) {
              track.removeEventListener?.('ended', onEnded)
              track.removeEventListener?.('mute', onMute)
              track.removeEventListener?.('unmute', onUnmute)
            }
            try { source.disconnect() } catch { /* already disconnected */ }
            closeAll()
          },
        }
      } catch (error) {
        closeAll()
        if (signal.aborted) throw new MicrophoneError('cancelled', 'The live session was cancelled.')
        if (error instanceof MicrophoneError) throw error
        throw new MicrophoneError('unavailable', 'Live voice could not start on this browser.', error)
      }
    },
  }
}
