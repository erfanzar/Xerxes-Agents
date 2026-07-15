// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import type { JsonObject } from '../types/toolCalls.js'
import { HttpMediaClient, encodeMediaBase64, type HttpMediaClientOptions } from './mediaHttp.js'

const DEFAULT_TTS_MODEL = 'tts-1'
const DEFAULT_TTS_VOICE = 'alloy'
const AUDIO_FORMATS = ['aac', 'flac', 'mp3', 'opus', 'pcm', 'wav'] as const

export type AudioFormat = (typeof AUDIO_FORMATS)[number]

export interface TextToSpeechRequest {
  readonly format?: AudioFormat
  readonly model?: string
  readonly text: string
  readonly voice?: string
}

export interface SynthesizedAudio {
  readonly audio: Uint8Array
  readonly format: AudioFormat
  readonly model: string
  readonly voice: string
}

/** Provider port for cloud, local, or browser-hosted TTS implementations. */
export interface TextToSpeechPort {
  readonly providerName: string
  synthesize(request: TextToSpeechRequest, signal?: AbortSignal): Promise<SynthesizedAudio>
}

export interface OpenAiCompatibleTextToSpeechOptions extends HttpMediaClientOptions {
  readonly defaultModel?: string
  readonly defaultVoice?: string
}

/** OpenAI-compatible `/audio/speech` implementation backed by injected native fetch. */
export class OpenAiCompatibleTextToSpeechPort implements TextToSpeechPort {
  private readonly client: HttpMediaClient
  private readonly defaultModel: string
  private readonly defaultVoice: string
  readonly providerName: string

  constructor(options: OpenAiCompatibleTextToSpeechOptions) {
    this.client = new HttpMediaClient({ ...options, providerName: options.providerName ?? 'openai-compatible-tts' })
    this.defaultModel = requiredText(options.defaultModel ?? DEFAULT_TTS_MODEL, 'defaultModel')
    this.defaultVoice = requiredText(options.defaultVoice ?? DEFAULT_TTS_VOICE, 'defaultVoice')
    this.providerName = this.client.providerName
  }

  async synthesize(request: TextToSpeechRequest, signal?: AbortSignal): Promise<SynthesizedAudio> {
    const text = requiredText(request.text, 'text')
    const model = requiredText(request.model ?? this.defaultModel, 'model')
    const voice = requiredText(request.voice ?? this.defaultVoice, 'voice')
    const format = normalizeAudioFormat(request.format ?? 'mp3')
    const audio = await this.client.postBinary('audio/speech', {
      model,
      input: text,
      voice,
      format,
    }, signal)
    return Object.freeze({ audio, format, model, voice })
  }
}

/** Explicit provider registry for injected Edge, ElevenLabs, local, or HTTP ports. */
export class TextToSpeechRegistry {
  private readonly providers = new Map<string, TextToSpeechPort>()

  register(name: string, port: TextToSpeechPort): void {
    this.providers.set(providerKey(name), port)
  }

  names(): string[] {
    return [...this.providers.keys()].sort((left, right) => left.localeCompare(right))
  }

  async synthesize(provider: string, request: TextToSpeechRequest, signal?: AbortSignal): Promise<SynthesizedAudio> {
    const key = providerKey(provider)
    const port = this.providers.get(key)
    if (!port) {
      throw new ValidationError('provider', `is not registered; available: ${this.names().join(', ') || '(none)'}`, provider)
    }
    return port.synthesize(request, signal)
  }
}

/** Safe JSON result for a tool call. Audio stays base64 rather than a non-serializable byte array. */
export function serializableSynthesizedAudio(result: SynthesizedAudio): JsonObject {
  return {
    audio_b64: encodeMediaBase64(result.audio),
    bytes: result.audio.byteLength,
    format: result.format,
    model: result.model,
    voice: result.voice,
  }
}

export function normalizeAudioFormat(value: string): AudioFormat {
  if (!(AUDIO_FORMATS as readonly string[]).includes(value)) {
    throw new ValidationError('audio_format', `must be one of ${AUDIO_FORMATS.join(', ')}`, value)
  }
  return value as AudioFormat
}

function requiredText(value: string, field: string): string {
  const normalized = value.trim()
  if (!normalized) {
    throw new ValidationError(field, 'must be a non-empty string', value)
  }
  return normalized
}

function providerKey(value: string): string {
  return requiredText(value, 'provider').toLowerCase()
}
