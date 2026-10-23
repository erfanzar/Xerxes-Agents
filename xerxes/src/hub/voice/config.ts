// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { VoicePricing } from './budgets.js'
import type { RealtimeVoicePort, SpeechToTextPort, TextToSpeechPort } from './ports.js'
import { VOICE_LIMITS, type VoiceCapabilities, type VoiceCodec, type VoiceMode } from './protocol.js'
import type { VoiceTranscriptStore } from './transcripts.js'

/**
 * Validated voice settings (`hub.voice.*`). Contains no credential: keys are
 * resolved by the provider factory and live only inside the provider ports.
 */
export interface VoiceSettings {
  readonly enabled: boolean
  readonly mode: 'push_to_talk' | 'live' | 'both'
  readonly language?: string
  readonly browserFallback: boolean
  readonly confirmTranscript: boolean
  readonly maxUtteranceSeconds: number
  readonly maxUtteranceBytes: number
  readonly maxSessionSeconds: number
  readonly idleSeconds: number
  /** Push-to-talk sockets idle longer than this are closed. */
  readonly pttIdleSeconds: number
  readonly maxSessionUsd?: number
  readonly maxDailyUserUsd?: number
  readonly pricing: VoicePricing
  readonly ttsFormat: 'mp3' | 'opus' | 'aac' | 'wav' | 'flac' | 'pcm'
  readonly audioRetention: 'none'
}

export const DEFAULT_VOICE_SETTINGS: VoiceSettings = Object.freeze({
  enabled: false,
  mode: 'push_to_talk',
  browserFallback: false,
  confirmTranscript: true,
  maxUtteranceSeconds: VOICE_LIMITS.maxUtteranceSeconds,
  maxUtteranceBytes: VOICE_LIMITS.maxUtteranceBytes,
  maxSessionSeconds: VOICE_LIMITS.maxSessionSeconds,
  idleSeconds: VOICE_LIMITS.idleSeconds,
  pttIdleSeconds: VOICE_LIMITS.pttIdleSeconds,
  pricing: Object.freeze({}),
  ttsFormat: 'mp3',
  audioRetention: 'none',
})

/** Why a capability is off, for the capability reply. Never contains a secret or a URL. */
export interface VoiceUnavailableReasons {
  readonly stt?: string
  readonly tts?: string
  readonly realtime?: string
}

/** Everything the voice route needs: validated settings plus already-constructed ports. */
export interface VoiceRuntimeConfig {
  readonly settings: VoiceSettings
  readonly stt?: SpeechToTextPort
  readonly tts?: TextToSpeechPort
  readonly realtime?: RealtimeVoicePort
  readonly reasons?: VoiceUnavailableReasons
  /** Final transcripts and attribution only. Audio has no path into this store. */
  readonly transcripts?: VoiceTranscriptStore
}

const TTS_CODEC: Readonly<Record<VoiceSettings['ttsFormat'], VoiceCodec>> = {
  mp3: 'mp3', opus: 'opus', aac: 'aac', wav: 'wav', flac: 'flac', pcm: 'pcm16',
}

export function ttsCodec(settings: VoiceSettings): VoiceCodec {
  return TTS_CODEC[settings.ttsFormat]
}

export function allowedModes(config: VoiceRuntimeConfig): readonly VoiceMode[] {
  if (!config.settings.enabled) return []
  const modes: VoiceMode[] = []
  if (config.stt !== undefined && config.settings.mode !== 'live') modes.push('push_to_talk')
  if (config.realtime !== undefined && config.settings.mode !== 'push_to_talk') modes.push('live')
  return modes
}

/** Capability reply. Reports provider names and honest delivery labels, never credentials or endpoints. */
export function buildCapabilities(config: VoiceRuntimeConfig): VoiceCapabilities {
  const { settings, stt, tts, realtime, reasons } = config
  const off = (specific: string | undefined): string => (settings.enabled ? specific ?? 'not configured' : 'voice is disabled on this hub')
  const modes = allowedModes(config)
  const sttOn = settings.enabled && stt !== undefined && modes.includes('push_to_talk')
  const ttsOn = settings.enabled && tts !== undefined
  const liveOn = settings.enabled && realtime !== undefined && modes.includes('live')
  return {
    enabled: settings.enabled && modes.length > 0,
    modes,
    stt: sttOn ? { available: true, provider: stt.providerName } : { available: false, reason: off(reasons?.stt) },
    tts: ttsOn
      ? { available: true, provider: tts.providerName, delivery: tts.delivery, format: ttsCodec(settings) }
      : { available: false, reason: off(reasons?.tts) },
    realtime: liveOn
      ? { available: true, provider: realtime.providerName, sample_rate: realtime.sampleRate }
      : { available: false, reason: off(reasons?.realtime) },
    limits: {
      max_utterance_seconds: settings.maxUtteranceSeconds,
      max_utterance_bytes: settings.maxUtteranceBytes,
      slice_ms_min: VOICE_LIMITS.minSliceMs,
      slice_ms_max: VOICE_LIMITS.maxSliceMs,
      max_session_seconds: settings.maxSessionSeconds,
      idle_seconds: settings.idleSeconds,
    },
    ...(settings.language === undefined ? {} : { language: settings.language }),
    confirm_transcript: settings.confirmTranscript,
    browser_fallback: settings.browserFallback,
    audio_retention: 'none',
  }
}
