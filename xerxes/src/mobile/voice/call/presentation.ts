// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { tierDownloadBytes } from '../../../client/voiceModelManifest.js'
import type { VoiceCallSnapshot } from './service.js'

export const voiceMegabytes = (bytes: number): string => `${Math.round(bytes / 1_000_000)} MB`

export function voiceStateLine(state: VoiceCallSnapshot): string {
  if (state.call.phase === 'error') return 'Voice ended'
  if (!state.connected) return 'Connecting'
  switch (state.call.phase) {
    case 'idle': return state.checking ? 'Connecting' : state.call.endReason ? 'Voice ended' : 'Ready for voice'
    case 'loading': return loadingLine(state)
    case 'listening': case 'hearing': return state.call.muted ? 'Microphone muted' : 'Listening'
    case 'thinking': case 'transcribing': return 'Thinking'
    case 'speaking': return 'Speaking'
  }
}

/**
 * Loader progress counts cached files too, so it would climb to the full tier size on every call.
 * Only bytes not already in this browser's caches are shown as a download.
 */
function loadingLine(state: VoiceCallSnapshot): string {
  // A speech-engine restart reloads models the phone already has; it is never a download.
  if (state.call.restoring === true) return 'Restoring speech'
  const cache = state.phoneCache
  const cachedBytes = cache ? Math.max(0, cache.cachedBytes) : 0
  const totalBytes = cache?.totalBytes ?? state.call.progress?.total ?? tierDownloadBytes(state.tier)
  if (cache && cachedBytes >= totalBytes) return 'Starting voice'
  const remaining = Math.max(0, totalBytes - cachedBytes)
  const loaded = Math.min(remaining, Math.max(0, (state.call.progress?.loaded ?? 0) - cachedBytes))
  return `Downloading voice (${voiceMegabytes(loaded)} of ${voiceMegabytes(remaining)})`
}

/** Meter data affects only the orb's paint, never recognition or playback. */
export function voiceOrbLevel(state: VoiceCallSnapshot): number {
  const raw = state.call.phase === 'speaking' ? state.call.outputLevel ?? 0 : state.call.muted ? 0 : state.call.inputLevel ?? 0
  return Number.isFinite(raw) ? Math.max(0, Math.min(1, raw)) : 0
}

export function shortVoiceCaption(text: string): string {
  const clean = text.replace(/\s+/gu, ' ').trim()
  return clean.length > 220 ? `${clean.slice(0, 217)}…` : clean
}
