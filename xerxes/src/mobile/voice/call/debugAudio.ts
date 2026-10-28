// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { encodeDebugWav, type CallDebugCapture } from './debugCapture.js'

export interface DebugPlayback {
  readonly finished: Promise<void>
  stop(): void
}

/** Plays the original Float32 clip locally, without normalisation, encoding or a model worker. */
export function playDebugCapture(capture: CallDebugCapture): DebugPlayback {
  const context = new AudioContext()
  let source: AudioBufferSourceNode | undefined
  let finish = (_error?: unknown): void => undefined
  const finished = new Promise<void>((resolve, reject) => {
    let ended = false
    finish = (error?: unknown): void => {
      if (ended) return
      ended = true
      if (source) { source.onended = null; source.disconnect() }
      void context.close().catch(() => undefined)
      if (error) reject(error)
      else resolve()
    }
    try {
      const buffer = context.createBuffer(1, capture.samples.length, capture.sampleRate)
      buffer.getChannelData(0).set(capture.samples)
      const node = context.createBufferSource()
      source = node
      node.buffer = buffer
      node.connect(context.destination)
      node.onended = () => finish()
      void context.resume().then(() => {
        if (ended) return
        if (context.state !== 'running') { finish(new Error('The phone blocked debug audio playback. Tap Play again.')); return }
        try { node.start() } catch (error) { finish(error) }
      }, error => finish(error))
    } catch (error) { finish(error) }
  })
  return { finished, stop() { try { source?.stop() } catch { /* not started or already ended */ }; finish() } }
}

/** Only invoked by Save WAV. No upload: the person chooses a share target or a local download. */
export async function saveDebugCapture(capture: CallDebugCapture): Promise<void> {
  const blob = new Blob([encodeDebugWav(capture.samples)], { type: 'audio/wav' })
  const name = 'xerxes-utterance.wav'
  if (typeof File !== 'undefined' && typeof navigator !== 'undefined' && navigator.share && navigator.canShare) {
    const file = new File([blob], name, { type: blob.type })
    if (navigator.canShare({ files: [file] })) {
      try { await navigator.share({ files: [file] }) } catch (error) {
        if (!(error instanceof DOMException && error.name === 'AbortError')) throw error
      }
      return
    }
  }
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  try {
    link.href = url
    link.download = name
    document.body.append(link)
    link.click()
  } finally {
    link.remove()
    // Let the browser consume the navigation before revoking its bytes (including on mobile Safari).
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
}
