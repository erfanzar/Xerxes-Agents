// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { tierDownloadBytes } from '../../../client/voiceModelManifest.js'
import type { VoiceCallSnapshot } from './service.js'

export const voiceMegabytes = (bytes: number): string => `${Math.round(bytes / 1_000_000)} MB`

export function voiceStateLine(state: VoiceCallSnapshot): string {
  if (state.call.phase === 'error') return 'Voice ended'
  if (!state.connected) return 'Connecting'
  switch (state.call.phase) {
    case 'idle': return state.checking ? 'Connecting' : 'Ready for voice'
    case 'loading': return `Loading voice (${voiceMegabytes(state.call.progress?.loaded ?? 0)} of ${voiceMegabytes(state.call.progress?.total ?? tierDownloadBytes(state.tier))})`
    case 'listening': case 'hearing': return state.call.muted ? 'Microphone muted' : 'Listening'
    case 'thinking': case 'transcribing': return 'Thinking'
    case 'speaking': return 'Speaking'
  }
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
