// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import type { JsonObject } from '../types/toolCalls.js'
import {
  HttpMediaClient,
  jsonObject,
  jsonString,
  requiredMediaType,
  type HttpMediaClientOptions,
} from './mediaHttp.js'

const DEFAULT_TRANSCRIPTION_MODEL = 'whisper-1'

export interface AudioInput {
  readonly bytes: Uint8Array
  readonly filename: string
  readonly mediaType: string
}

export interface TranscriptionRequest {
  readonly audio: AudioInput
  readonly language?: string
  readonly model?: string
  readonly prompt?: string
  readonly temperature?: number
}

export interface TranscriptionResult {
  readonly backend: string
  readonly model: string
  readonly text: string
}

/** Provider port for a transcription service or an explicitly injected local model. */
export interface TranscriptionPort {
  readonly providerName: string
  transcribe(request: TranscriptionRequest, signal?: AbortSignal): Promise<TranscriptionResult>
}

export interface OpenAiCompatibleTranscriptionOptions extends HttpMediaClientOptions {
  readonly defaultModel?: string
}

/** OpenAI-compatible multipart `/audio/transcriptions` adapter. */
export class OpenAiCompatibleTranscriptionPort implements TranscriptionPort {
  private readonly client: HttpMediaClient
  private readonly defaultModel: string
  readonly providerName: string

  constructor(options: OpenAiCompatibleTranscriptionOptions) {
    this.client = new HttpMediaClient({ ...options, providerName: options.providerName ?? 'openai-compatible-transcription' })
    this.defaultModel = requiredText(options.defaultModel ?? DEFAULT_TRANSCRIPTION_MODEL, 'defaultModel')
    this.providerName = this.client.providerName
  }

  async transcribe(request: TranscriptionRequest, signal?: AbortSignal): Promise<TranscriptionResult> {
    const audio = normalizeAudioInput(request.audio)
    const model = requiredText(request.model ?? this.defaultModel, 'model')
    const form = new FormData()
    const buffer = Uint8Array.from(audio.bytes).buffer as ArrayBuffer
    form.set('file', new File([buffer], audio.filename, { type: audio.mediaType }))
    form.set('model', model)
    if (request.language?.trim()) form.set('language', request.language.trim())
    if (request.prompt?.trim()) form.set('prompt', request.prompt.trim())
    if (request.temperature !== undefined) {
      if (!Number.isFinite(request.temperature) || request.temperature < 0 || request.temperature > 1) {
        throw new ValidationError('temperature', 'must be a number from 0 to 1', request.temperature)
      }
      form.set('temperature', String(request.temperature))
    }
    const response = jsonObject(await this.client.postForm('audio/transcriptions', form, signal), 'response')
    return Object.freeze({
      backend: this.providerName,
      model,
      text: jsonString(response.text) ?? '',
    })
  }
}

/** Explicit registry for caller-provided local or vendor-specific transcription ports. */
export class TranscriptionRegistry {
  private readonly providers = new Map<string, TranscriptionPort>()

  register(name: string, port: TranscriptionPort): void {
    this.providers.set(providerKey(name), port)
  }

  names(): string[] {
    return [...this.providers.keys()].sort((left, right) => left.localeCompare(right))
  }

  async transcribe(provider: string, request: TranscriptionRequest, signal?: AbortSignal): Promise<TranscriptionResult> {
    const port = this.providers.get(providerKey(provider))
    if (!port) {
      throw new ValidationError('backend', `is not registered; available: ${this.names().join(', ') || '(none)'}`, provider)
    }
    return port.transcribe(request, signal)
  }
}

export function serializableTranscription(result: TranscriptionResult): JsonObject {
  return { backend: result.backend, model: result.model, text: result.text }
}

export function normalizeAudioInput(input: AudioInput): AudioInput {
  if (!(input.bytes instanceof Uint8Array) || input.bytes.byteLength === 0) {
    throw new ValidationError('audio', 'must contain non-empty audio bytes')
  }
  const filename = requiredText(input.filename, 'filename').replace(/^.*[\\/]/, '')
  if (!filename) {
    throw new ValidationError('filename', 'must include a file name', input.filename)
  }
  return Object.freeze({
    bytes: new Uint8Array(input.bytes),
    filename,
    mediaType: requiredMediaType(input.mediaType, 'audio/', 'media_type'),
  })
}

function requiredText(value: string, field: string): string {
  const normalized = value.trim()
  if (!normalized) {
    throw new ValidationError(field, 'must be a non-empty string', value)
  }
  return normalized
}

function providerKey(value: string): string {
  return requiredText(value, 'backend').toLowerCase()
}
