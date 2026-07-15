// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import type { JsonObject, JsonValue } from '../types/toolCalls.js'
import {
  HttpMediaClient,
  decodeMediaBase64,
  jsonArray,
  jsonObject,
  jsonString,
  requiredMediaType,
  type HttpMediaClientOptions,
} from './mediaHttp.js'

const DEFAULT_ANTHROPIC_VISION_MODEL = 'claude-sonnet-4-6'
const DEFAULT_OPENAI_VISION_MODEL = 'gpt-4o-mini'
const DEFAULT_VISION_PROMPT = 'Describe this image in detail.'
const DEFAULT_VISION_MAX_TOKENS = 1024

export type VisionImageSource =
  | { readonly kind: 'base64'; readonly mediaType: string; readonly data: string }
  | { readonly kind: 'url'; readonly url: string }

export interface VisionAnalysisRequest {
  readonly image: VisionImageSource
  readonly maxTokens?: number
  readonly model?: string
  readonly prompt?: string
}

export interface VisionAnalysisResult {
  readonly model: string
  readonly prompt: string
  readonly provider: string
  readonly raw?: JsonValue
  readonly response: string
}

/** Vision provider port. Implement a port for a local model rather than giving the core runtime model dependencies. */
export interface VisionPort {
  readonly providerName: string
  analyze(request: VisionAnalysisRequest, signal?: AbortSignal): Promise<VisionAnalysisResult>
}

export interface OpenAiCompatibleVisionOptions extends HttpMediaClientOptions {
  readonly defaultMaxTokens?: number
  readonly defaultModel?: string
}

/** OpenAI-compatible chat-completions vision adapter. */
export class OpenAiCompatibleVisionPort implements VisionPort {
  private readonly client: HttpMediaClient
  private readonly defaultMaxTokens: number
  private readonly defaultModel: string
  readonly providerName: string

  constructor(options: OpenAiCompatibleVisionOptions) {
    this.client = new HttpMediaClient({ ...options, providerName: options.providerName ?? 'openai-compatible-vision' })
    this.defaultModel = requiredText(options.defaultModel ?? DEFAULT_OPENAI_VISION_MODEL, 'defaultModel')
    this.defaultMaxTokens = validMaxTokens(options.defaultMaxTokens ?? DEFAULT_VISION_MAX_TOKENS)
    this.providerName = this.client.providerName
  }

  async analyze(request: VisionAnalysisRequest, signal?: AbortSignal): Promise<VisionAnalysisResult> {
    const prompt = requiredText(request.prompt ?? DEFAULT_VISION_PROMPT, 'prompt')
    const model = requiredText(request.model ?? this.defaultModel, 'model')
    const maxTokens = validMaxTokens(request.maxTokens ?? this.defaultMaxTokens)
    const imageUrl = openAiImageUrl(request.image)
    const raw = await this.client.postJson('chat/completions', {
      model,
      max_tokens: maxTokens,
      messages: [{
        role: 'user',
        content: [
          { type: 'text', text: prompt },
          { type: 'image_url', image_url: { url: imageUrl } },
        ],
      }],
    }, signal)
    return Object.freeze({
      provider: this.providerName,
      model,
      prompt,
      response: openAiResponseText(raw),
      raw,
    })
  }
}

export interface AnthropicVisionOptions extends Omit<HttpMediaClientOptions, 'authHeader' | 'authScheme'> {
  readonly defaultMaxTokens?: number
  readonly defaultModel?: string
  readonly version?: string
}

/** Direct Anthropic Messages API vision adapter. Remote URLs are intentionally not downloaded implicitly. */
export class AnthropicVisionPort implements VisionPort {
  private readonly client: HttpMediaClient
  private readonly defaultMaxTokens: number
  private readonly defaultModel: string
  readonly providerName: string

  constructor(options: AnthropicVisionOptions) {
    this.client = new HttpMediaClient({
      ...options,
      authHeader: 'x-api-key',
      authScheme: '',
      headers: {
        'anthropic-version': options.version ?? '2023-06-01',
        ...(options.headers ?? {}),
      },
      providerName: options.providerName ?? 'anthropic-vision',
    })
    this.defaultModel = requiredText(options.defaultModel ?? DEFAULT_ANTHROPIC_VISION_MODEL, 'defaultModel')
    this.defaultMaxTokens = validMaxTokens(options.defaultMaxTokens ?? DEFAULT_VISION_MAX_TOKENS)
    this.providerName = this.client.providerName
  }

  async analyze(request: VisionAnalysisRequest, signal?: AbortSignal): Promise<VisionAnalysisResult> {
    const prompt = requiredText(request.prompt ?? DEFAULT_VISION_PROMPT, 'prompt')
    const model = requiredText(request.model ?? this.defaultModel, 'model')
    const maxTokens = validMaxTokens(request.maxTokens ?? this.defaultMaxTokens)
    if (request.image.kind !== 'base64') {
      throw new ValidationError(
        'image_url',
        'Anthropic vision requires image_b64 and image_media_type; inject a downloader explicitly before calling this port',
        request.image.url,
      )
    }
    const mediaType = requiredMediaType(request.image.mediaType, 'image/', 'image_media_type')
    const data = requiredText(request.image.data, 'image_b64')
    const raw = await this.client.postJson('v1/messages', {
      model,
      max_tokens: maxTokens,
      messages: [{
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: mediaType, data } },
          { type: 'text', text: prompt },
        ],
      }],
    }, signal)
    return Object.freeze({
      provider: this.providerName,
      model,
      prompt,
      response: anthropicResponseText(raw),
      raw,
    })
  }
}

/** Explicit named registry for extension-provided vision ports. */
export class VisionRegistry {
  private readonly providers = new Map<string, VisionPort>()

  register(name: string, port: VisionPort): void {
    this.providers.set(providerKey(name), port)
  }

  names(): string[] {
    return [...this.providers.keys()].sort((left, right) => left.localeCompare(right))
  }

  async analyze(provider: string, request: VisionAnalysisRequest, signal?: AbortSignal): Promise<VisionAnalysisResult> {
    const port = this.providers.get(providerKey(provider))
    if (!port) {
      throw new ValidationError('provider', `is not registered; available: ${this.names().join(', ') || '(none)'}`, provider)
    }
    return port.analyze(request, signal)
  }
}

/** Serialize an analysis response for a tool result or RPC event. */
export function serializableVisionAnalysis(result: VisionAnalysisResult): JsonObject {
  return {
    answer: result.response,
    model: result.model,
    prompt: result.prompt,
    provider: result.provider,
    ...(result.raw === undefined ? {} : { raw: result.raw }),
  }
}

export function visionImageFromBase64(
  data: string,
  mediaType = 'image/png',
): Extract<VisionImageSource, { readonly kind: 'base64' }> {
  const normalized = data.replace(/\s/g, '')
  decodeMediaBase64(normalized, 'image_b64')
  return Object.freeze({
    kind: 'base64',
    data: normalized,
    mediaType: requiredMediaType(mediaType, 'image/', 'image_media_type'),
  })
}

export function visionImageFromUrl(url: string): Extract<VisionImageSource, { readonly kind: 'url' }> {
  const normalized = requiredText(url, 'image_url')
  let parsed: URL
  try {
    parsed = new URL(normalized)
  } catch {
    throw new ValidationError('image_url', 'must be an absolute HTTP(S) URL', url)
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new ValidationError('image_url', 'must use HTTP or HTTPS', url)
  }
  return Object.freeze({ kind: 'url', url: parsed.toString() })
}

function openAiImageUrl(image: VisionImageSource): string {
  if (image.kind === 'url') return visionImageFromUrl(image.url).url
  const mediaType = requiredMediaType(image.mediaType, 'image/', 'image_media_type')
  return `data:${mediaType};base64,${requiredText(image.data, 'image_b64')}`
}

function openAiResponseText(value: JsonValue): string {
  const root = jsonObject(value, 'response')
  const choices = jsonArray(root.choices ?? [], 'choices')
  const first = choices[0]
  if (first === undefined) return ''
  const message = jsonObject(first, 'choices[0]').message
  if (message === undefined) return ''
  const content = jsonObject(message, 'choices[0].message').content
  return contentToText(content)
}

function anthropicResponseText(value: JsonValue): string {
  const root = jsonObject(value, 'response')
  const content = jsonArray(root.content ?? [], 'content')
  const text: string[] = []
  for (let index = 0; index < content.length; index += 1) {
    const part = jsonObject(content[index]!, `content[${index}]`)
    if (jsonString(part.type) === 'text') {
      const fragment = jsonString(part.text)
      if (fragment) text.push(fragment)
    }
  }
  return text.join('')
}

function contentToText(value: JsonValue | undefined): string {
  if (typeof value === 'string') return value
  if (!Array.isArray(value)) return ''
  const text: string[] = []
  for (let index = 0; index < value.length; index += 1) {
    const part = value[index]
    if (typeof part === 'object' && part !== null && !Array.isArray(part)) {
      const fragment = jsonString(part.text)
      if (fragment) text.push(fragment)
    }
  }
  return text.join('')
}

function validMaxTokens(value: number): number {
  if (!Number.isInteger(value) || value < 1 || value > 16_384) {
    throw new ValidationError('max_tokens', 'must be an integer from 1 to 16384', value)
  }
  return value
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
