// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ConfigurationError } from '../../core/errors.js'
import { OpenAiCompatibleTextToSpeechPort } from '../../tools/tts.js'
import { OpenAiCompatibleTranscriptionPort } from '../../tools/transcription.js'
import type { MediaFetch } from '../../tools/mediaHttp.js'
import type { VoicePricing } from './budgets.js'
import { DEFAULT_VOICE_SETTINGS, type VoiceRuntimeConfig, type VoiceSettings, type VoiceUnavailableReasons } from './config.js'
import { adaptBufferedTextToSpeech, adaptTranscriptionPort, type RealtimeVoicePort, type SpeechToTextPort, type TextToSpeechPort } from './ports.js'
import { GeminiLivePort } from './providers/geminiLive.js'
import { OpenAiRealtimePort } from './providers/openaiRealtime.js'
import type { RealtimeSocketFactory } from './providers/realtimeSocket.js'
import type { VoiceTranscriptStore } from './transcripts.js'

/**
 * Builds the voice runtime from explicit configuration and nothing else.
 *
 * Settings live under `hub.voice` (passed in as a plain object). Every key can
 * also come from the environment as `XERXES_HUB_VOICE_<PATH>` with dots turned
 * into underscores (`stt.base_url` -> `XERXES_HUB_VOICE_STT_BASE_URL`); a value
 * in settings wins over the environment.
 *
 *   enabled, mode (push_to_talk | live | both), language, browser_fallback,
 *   confirm_transcript, max_utterance_seconds, max_utterance_bytes,
 *   max_session_seconds, idle_seconds, max_session_usd, max_daily_user_usd,
 *   audio_retention (only "none" exists),
 *   stt.{provider=openai-compatible, base_url, model, api_key_env, allow_unauthenticated},
 *   tts.{provider=openai-compatible, base_url, model, voice, format, api_key_env, allow_unauthenticated},
 *   realtime.{provider=openai|gemini, model, base_url, voice, vad_silence_ms, api_key_env},
 *   pricing.{stt_usd_per_minute, tts_usd_per_million_chars, realtime_usd_per_minute}
 *
 * Credentials are never read from settings (a literal `api_key` is rejected).
 * They come from the environment variable named by `<section>.api_key_env`,
 * default `XERXES_VOICE_<STT|TTS|REALTIME>_API_KEY`, and stay inside the
 * provider ports. A section that is not configured, uses an unknown provider
 * or lacks a credential is simply unavailable, with a fixed reason; nothing
 * substitutes another provider.
 */

export interface VoiceFactoryInput {
  /** The `hub.voice` settings object. */
  readonly settings?: unknown
  readonly env?: Readonly<Record<string, string | undefined>>
  readonly fetchImplementation?: MediaFetch
  readonly socketFactory?: RealtimeSocketFactory
  readonly transcripts?: VoiceTranscriptStore
}

const TTS_FORMATS = ['mp3', 'opus', 'aac', 'wav', 'flac'] as const
const MODES = ['push_to_talk', 'live', 'both'] as const
const ENV_PREFIX = 'XERXES_HUB_VOICE_'

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

class Reader {
  constructor(private readonly settings: Record<string, unknown>, private readonly env: Readonly<Record<string, string | undefined>>) {}

  raw(path: string): unknown {
    let cursor: unknown = this.settings
    for (const part of path.split('.')) {
      if (!isRecord(cursor)) { cursor = undefined; break }
      cursor = cursor[part]
    }
    if (cursor !== undefined) return cursor
    const fromEnv = this.env[ENV_PREFIX + path.replace(/\./g, '_').toUpperCase()]
    return fromEnv === undefined || fromEnv.trim() === '' ? undefined : fromEnv.trim()
  }

  string(path: string): string | undefined {
    const value = this.raw(path)
    if (value === undefined) return undefined
    if (typeof value !== 'string' || value.trim() === '') throw new ConfigurationError(`hub.voice.${path}`, 'must be a non-empty string')
    return value.trim()
  }

  bool(path: string): boolean | undefined {
    const value = this.raw(path)
    if (value === undefined) return undefined
    if (typeof value === 'boolean') return value
    if (typeof value === 'string') {
      const lower = value.toLowerCase()
      if (lower === 'true' || lower === '1' || lower === 'yes' || lower === 'on') return true
      if (lower === 'false' || lower === '0' || lower === 'no' || lower === 'off') return false
    }
    throw new ConfigurationError(`hub.voice.${path}`, 'must be true or false')
  }

  number(path: string, bounds: { readonly min: number; readonly max: number; readonly integer?: boolean }): number | undefined {
    const value = this.raw(path)
    if (value === undefined) return undefined
    const parsed = typeof value === 'number' ? value : typeof value === 'string' && value.trim() !== '' ? Number(value) : Number.NaN
    if (!Number.isFinite(parsed) || parsed < bounds.min || parsed > bounds.max || (bounds.integer === true && !Number.isInteger(parsed))) {
      throw new ConfigurationError(`hub.voice.${path}`, `must be ${bounds.integer === true ? 'an integer' : 'a number'} from ${bounds.min} to ${bounds.max}`)
    }
    return parsed
  }

  oneOf<T extends string>(path: string, allowed: readonly T[]): T | undefined {
    const value = this.string(path)
    if (value === undefined) return undefined
    if (!(allowed as readonly string[]).includes(value)) throw new ConfigurationError(`hub.voice.${path}`, `must be one of ${allowed.join(', ')}`)
    return value as T
  }

  /** A section is "configured" when any of its keys is set. */
  touched(section: string, keys: readonly string[]): boolean {
    return keys.some(key => this.raw(`${section}.${key}`) !== undefined)
  }
}

function rejectLiteralKeys(settings: Record<string, unknown>): void {
  for (const section of ['stt', 'tts', 'realtime']) {
    const value = settings[section]
    if (isRecord(value) && ('api_key' in value || 'apiKey' in value || 'key' in value || 'token' in value)) {
      throw new ConfigurationError(`hub.voice.${section}`, 'credentials must not be stored in settings; name an environment variable with api_key_env')
    }
  }
}

function credential(reader: Reader, env: Readonly<Record<string, string | undefined>>, section: 'stt' | 'tts' | 'realtime'): string | undefined {
  const variable = reader.string(`${section}.api_key_env`) ?? `XERXES_VOICE_${section.toUpperCase()}_API_KEY`
  const value = env[variable]?.trim()
  return value === undefined || value === '' ? undefined : value
}

interface Built<T> { readonly port?: T; readonly reason?: string }

const SECTION_KEYS = ['provider', 'base_url', 'model', 'api_key_env', 'voice', 'format', 'vad_silence_ms', 'allow_unauthenticated'] as const

function buildStt(reader: Reader, env: Readonly<Record<string, string | undefined>>, fetchImplementation: MediaFetch | undefined, language: string | undefined): Built<SpeechToTextPort> {
  if (!reader.touched('stt', SECTION_KEYS)) return { reason: 'not configured' }
  const provider = reader.string('stt.provider') ?? 'openai-compatible'
  if (provider !== 'openai-compatible') return { reason: 'unsupported speech-to-text provider' }
  const baseUrl = reader.string('stt.base_url')
  if (baseUrl === undefined) return { reason: 'missing base_url' }
  const apiKey = credential(reader, env, 'stt')
  const open = reader.bool('stt.allow_unauthenticated') === true
  if (apiKey === undefined && !open) return { reason: 'missing credentials' }
  const model = reader.string('stt.model')
  const port = new OpenAiCompatibleTranscriptionPort({
    baseUrl,
    ...(apiKey === undefined ? { allowUnauthenticated: true } : { apiKey }),
    ...(model === undefined ? {} : { defaultModel: model }),
    ...(fetchImplementation === undefined ? {} : { fetchImplementation }),
    providerName: 'openai-compatible-stt',
  })
  return { port: adaptTranscriptionPort(port, language === undefined ? {} : { defaultLanguage: language }) }
}

function buildTts(reader: Reader, env: Readonly<Record<string, string | undefined>>, fetchImplementation: MediaFetch | undefined): Built<TextToSpeechPort> {
  if (!reader.touched('tts', SECTION_KEYS)) return { reason: 'not configured' }
  const provider = reader.string('tts.provider') ?? 'openai-compatible'
  if (provider !== 'openai-compatible') return { reason: 'unsupported speech provider' }
  const baseUrl = reader.string('tts.base_url')
  if (baseUrl === undefined) return { reason: 'missing base_url' }
  const apiKey = credential(reader, env, 'tts')
  const open = reader.bool('tts.allow_unauthenticated') === true
  if (apiKey === undefined && !open) return { reason: 'missing credentials' }
  const model = reader.string('tts.model')
  const voice = reader.string('tts.voice')
  const port = new OpenAiCompatibleTextToSpeechPort({
    baseUrl,
    ...(apiKey === undefined ? { allowUnauthenticated: true } : { apiKey }),
    ...(model === undefined ? {} : { defaultModel: model }),
    ...(voice === undefined ? {} : { defaultVoice: voice }),
    ...(fetchImplementation === undefined ? {} : { fetchImplementation }),
    providerName: 'openai-compatible-tts',
  })
  // This adapter returns the whole clip per request, so it is labelled buffered.
  return { port: adaptBufferedTextToSpeech(port) }
}

function buildRealtime(reader: Reader, env: Readonly<Record<string, string | undefined>>, socketFactory: RealtimeSocketFactory | undefined): Built<RealtimeVoicePort> {
  if (!reader.touched('realtime', SECTION_KEYS)) return { reason: 'not configured' }
  const provider = reader.string('realtime.provider')
  if (provider === undefined) return { reason: 'missing provider' }
  if (provider !== 'openai' && provider !== 'gemini') return { reason: 'unsupported realtime provider' }
  const apiKey = credential(reader, env, 'realtime')
  if (apiKey === undefined) return { reason: 'missing credentials' }
  const model = reader.string('realtime.model')
  const baseUrl = reader.string('realtime.base_url')
  const voice = reader.string('realtime.voice')
  const vad = reader.number('realtime.vad_silence_ms', { min: 100, max: 3000, integer: true })
  const common = {
    apiKey,
    ...(model === undefined ? {} : { model }),
    ...(baseUrl === undefined ? {} : { baseUrl }),
    ...(voice === undefined ? {} : { voice }),
    ...(vad === undefined ? {} : { vadSilenceMs: vad }),
    ...(socketFactory === undefined ? {} : { socketFactory }),
  }
  return { port: provider === 'openai' ? new OpenAiRealtimePort(common) : new GeminiLivePort(common) }
}

export function createVoiceRuntimeConfig(input: VoiceFactoryInput = {}): VoiceRuntimeConfig {
  const env = input.env ?? {}
  if (input.settings !== undefined && !isRecord(input.settings)) throw new ConfigurationError('hub.voice', 'must be an object')
  const raw = input.settings ?? {}
  rejectLiteralKeys(raw)
  const reader = new Reader(raw, env)

  const retention = reader.string('audio_retention')
  if (retention !== undefined && retention !== 'none') {
    throw new ConfigurationError('hub.voice.audio_retention', 'audio is never stored; the only supported value is "none"')
  }
  const language = reader.string('language')
  if (language !== undefined && !/^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$/.test(language)) {
    throw new ConfigurationError('hub.voice.language', 'must be a BCP-47 language tag')
  }
  const d = DEFAULT_VOICE_SETTINGS
  const maxSessionUsd = reader.number('max_session_usd', { min: 0, max: 100_000 })
  const maxDailyUserUsd = reader.number('max_daily_user_usd', { min: 0, max: 100_000 })
  const sttPrice = reader.number('pricing.stt_usd_per_minute', { min: 0, max: 1000 })
  const ttsPrice = reader.number('pricing.tts_usd_per_million_chars', { min: 0, max: 100_000 })
  const rtPrice = reader.number('pricing.realtime_usd_per_minute', { min: 0, max: 1000 })
  const pricing: VoicePricing = {
    ...(sttPrice === undefined ? {} : { sttUsdPerMinute: sttPrice }),
    ...(ttsPrice === undefined ? {} : { ttsUsdPerMillionChars: ttsPrice }),
    ...(rtPrice === undefined ? {} : { realtimeUsdPerMinute: rtPrice }),
  }
  const format = reader.oneOf('tts.format', TTS_FORMATS)
  const settings: VoiceSettings = {
    enabled: reader.bool('enabled') ?? d.enabled,
    mode: reader.oneOf('mode', MODES) ?? d.mode,
    browserFallback: reader.bool('browser_fallback') ?? d.browserFallback,
    confirmTranscript: reader.bool('confirm_transcript') ?? d.confirmTranscript,
    maxUtteranceSeconds: reader.number('max_utterance_seconds', { min: 1, max: 600, integer: true }) ?? d.maxUtteranceSeconds,
    maxUtteranceBytes: reader.number('max_utterance_bytes', { min: 1024, max: 25 * 1024 * 1024, integer: true }) ?? d.maxUtteranceBytes,
    maxSessionSeconds: reader.number('max_session_seconds', { min: 30, max: 3600, integer: true }) ?? d.maxSessionSeconds,
    idleSeconds: reader.number('idle_seconds', { min: 5, max: 600, integer: true }) ?? d.idleSeconds,
    pttIdleSeconds: d.pttIdleSeconds,
    pricing,
    ttsFormat: format ?? d.ttsFormat,
    audioRetention: 'none',
    ...(language === undefined ? {} : { language }),
    ...(maxSessionUsd === undefined ? {} : { maxSessionUsd }),
    ...(maxDailyUserUsd === undefined ? {} : { maxDailyUserUsd }),
  }

  if (!settings.enabled) {
    return { settings, reasons: { stt: 'voice is disabled on this hub', tts: 'voice is disabled on this hub', realtime: 'voice is disabled on this hub' }, ...(input.transcripts === undefined ? {} : { transcripts: input.transcripts }) }
  }

  const stt = buildStt(reader, env, input.fetchImplementation, language)
  const tts = buildTts(reader, env, input.fetchImplementation)
  const realtime = buildRealtime(reader, env, input.socketFactory)
  const reasons: VoiceUnavailableReasons = {
    ...(stt.reason === undefined ? {} : { stt: stt.reason }),
    ...(tts.reason === undefined ? {} : { tts: tts.reason }),
    ...(realtime.reason === undefined ? {} : { realtime: realtime.reason }),
  }
  return {
    settings,
    reasons,
    ...(stt.port === undefined ? {} : { stt: stt.port }),
    ...(tts.port === undefined ? {} : { tts: tts.port }),
    ...(realtime.port === undefined ? {} : { realtime: realtime.port }),
    ...(input.transcripts === undefined ? {} : { transcripts: input.transcripts }),
  }
}
