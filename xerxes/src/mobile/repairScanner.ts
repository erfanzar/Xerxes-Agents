// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import jsQR from 'jsqr'

export interface RunningScanner {
  stop(): void
}

/** A short, actionable reason the camera could not start. */
export function cameraErrorMessage(error: unknown): string {
  const name = error instanceof Error || (typeof error === 'object' && error !== null && 'name' in error) ? String((error as { name: unknown }).name) : ''
  if (name === 'NotAllowedError' || name === 'SecurityError') return 'Camera access was denied. Allow the camera for this app in Settings, or paste the pairing link below.'
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return 'No camera was found. Paste the pairing link below instead.'
  if (name === 'NotReadableError' || name === 'AbortError') return 'The camera is busy in another app. Close it and try again, or paste the pairing link below.'
  return 'The camera is not available here. Paste the pairing link below instead.'
}

/** Longest side of the frame handed to the decoder; enough for a QR code on a laptop screen. */
const FRAME_MAX = 720
const FRAME_INTERVAL_MS = 200

/**
 * Decode QR codes from the rear camera into `onText` until stopped. Frames are decoded
 * on the phone and never sent anywhere. Throws (with a DOMException name) when the
 * camera cannot start; `cameraErrorMessage` turns that into text.
 */
export async function startQrScanner(video: HTMLVideoElement, onText: (text: string) => void): Promise<RunningScanner> {
  const media = globalThis.navigator?.mediaDevices
  if (media?.getUserMedia === undefined) throw new DOMException('Camera API unavailable', 'NotSupportedError')
  const stream = await media.getUserMedia({ video: { facingMode: { ideal: 'environment' } }, audio: false })
  let stopped = false
  let timer: ReturnType<typeof setTimeout> | undefined
  const stop = (): void => {
    stopped = true
    if (timer !== undefined) clearTimeout(timer)
    for (const track of stream.getTracks()) track.stop()
    video.srcObject = null
  }
  video.muted = true
  video.setAttribute('playsinline', '')
  video.srcObject = stream
  try { await video.play() } catch (error) { stop(); throw error }
  const canvas = document.createElement('canvas')
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (context === null) { stop(); throw new DOMException('Canvas unavailable', 'NotSupportedError') }
  const tick = (): void => {
    if (stopped) return
    if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
      const scale = Math.min(1, FRAME_MAX / Math.max(video.videoWidth, video.videoHeight))
      const width = Math.round(video.videoWidth * scale)
      const height = Math.round(video.videoHeight * scale)
      if (canvas.width !== width) canvas.width = width
      if (canvas.height !== height) canvas.height = height
      context.drawImage(video, 0, 0, width, height)
      const code = jsQR(context.getImageData(0, 0, width, height).data, width, height, { inversionAttempts: 'attemptBoth' })
      if (code !== null && code.data) onText(code.data)
    }
    if (!stopped) timer = setTimeout(tick, FRAME_INTERVAL_MS)
  }
  tick()
  return { stop }
}
