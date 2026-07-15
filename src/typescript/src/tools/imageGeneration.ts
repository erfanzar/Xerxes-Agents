// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'

import { ProviderError, ValidationError } from '../core/errors.js'
import type { JsonObject, JsonValue } from '../types/toolCalls.js'
import {
  HttpMediaClient,
  decodeMediaBase64,
  jsonArray,
  jsonObject,
  jsonString,
  type HttpMediaClientOptions,
} from './mediaHttp.js'

const DEFAULT_IMAGE_MODEL = 'gpt-image-1'
const DEFAULT_IMAGE_SIZE = '1024x1024'
const MAX_IMAGE_COUNT = 10

export interface ImageGenerationRequest {
  readonly count?: number
  readonly model?: string
  readonly prompt: string
  readonly size?: string
}

/** Provider-neutral serializable representation of an image-generation response. */
export interface GeneratedImage {
  readonly b64?: string
  readonly format: string
  readonly revisedPrompt?: string
  readonly url?: string
}

export interface ImageGenerationResult {
  readonly count: number
  readonly images: readonly GeneratedImage[]
  readonly model: string
  readonly size: string
}

/** Image-generation provider port; callers can inject FAL, local, or remote implementations. */
export interface ImageGenerationPort {
  readonly providerName: string
  generate(request: ImageGenerationRequest, signal?: AbortSignal): Promise<ImageGenerationResult>
}

export interface OpenAiCompatibleImageGenerationOptions extends HttpMediaClientOptions {
  readonly defaultModel?: string
}

/** OpenAI-compatible `/images/generations` adapter using only native fetch. */
export class OpenAiCompatibleImageGenerationPort implements ImageGenerationPort {
  private readonly client: HttpMediaClient
  private readonly defaultModel: string
  readonly providerName: string

  constructor(options: OpenAiCompatibleImageGenerationOptions) {
    this.client = new HttpMediaClient({ ...options, providerName: options.providerName ?? 'openai-compatible-image' })
    this.defaultModel = nonEmpty(options.defaultModel ?? DEFAULT_IMAGE_MODEL, 'defaultModel')
    this.providerName = this.client.providerName
  }

  async generate(request: ImageGenerationRequest, signal?: AbortSignal): Promise<ImageGenerationResult> {
    const prompt = nonEmpty(request.prompt, 'prompt')
    const model = nonEmpty(request.model ?? this.defaultModel, 'model')
    const size = nonEmpty(request.size ?? DEFAULT_IMAGE_SIZE, 'size')
    const count = request.count ?? 1
    if (!Number.isInteger(count) || count < 1 || count > MAX_IMAGE_COUNT) {
      throw new ValidationError('count', `must be an integer from 1 to ${MAX_IMAGE_COUNT}`, count)
    }

    const data = await this.client.postJson('images/generations', {
      model,
      prompt,
      size,
      n: count,
      response_format: 'b64_json',
    }, signal)
    const response = jsonObject(data, 'response')
    const entries = jsonArray(response.data ?? [], 'data')
    const images = entries.map((entry, index) => parseImage(entry, index))
    return Object.freeze({
      model,
      size,
      count: images.length,
      images: Object.freeze(images),
    })
  }
}

/** Explicit provider registry replacing Python's mutable module-global callback map. */
export class ImageGenerationRegistry {
  private readonly providers = new Map<string, ImageGenerationPort>()

  register(name: string, port: ImageGenerationPort): void {
    const normalized = providerKey(name)
    this.providers.set(normalized, port)
  }

  names(): string[] {
    return [...this.providers.keys()].sort((left, right) => left.localeCompare(right))
  }

  async generate(provider: string, request: ImageGenerationRequest, signal?: AbortSignal): Promise<ImageGenerationResult> {
    const port = this.providers.get(providerKey(provider))
    if (!port) {
      throw new ValidationError('provider', `is not registered; available: ${this.names().join(', ') || '(none)'}`, provider)
    }
    return port.generate(request, signal)
  }
}

/**
 * Persist a provider-returned base64 image when the caller has already made a
 * workspace/sandbox decision about the destination path.
 */
export async function writeGeneratedImage(image: GeneratedImage, outputPath: string): Promise<{ readonly bytes: number; readonly path: string }> {
  const b64 = image.b64
  if (!b64) {
    throw new ValidationError('image.b64', 'is required to write an image; URL-only provider responses need an explicit downloader', b64)
  }
  const bytes = decodeMediaBase64(b64, 'image.b64')
  await mkdir(dirname(outputPath), { recursive: true })
  await Bun.write(outputPath, bytes)
  return Object.freeze({ path: outputPath, bytes: bytes.byteLength })
}

/** Convert an image result to a JSON-safe tool result without exposing byte arrays. */
export function serializableImageGenerationResult(result: ImageGenerationResult): JsonObject {
  const images: JsonObject[] = result.images.map(image => ({
    format: image.format,
    ...(image.b64 === undefined ? {} : { b64: image.b64 }),
    ...(image.revisedPrompt === undefined ? {} : { revised_prompt: image.revisedPrompt }),
    ...(image.url === undefined ? {} : { url: image.url }),
  }))
  return {
    count: result.count,
    images,
    model: result.model,
    size: result.size,
  }
}

function parseImage(value: JsonValue, index: number): GeneratedImage {
  const entry = jsonObject(value, `data[${index}]`)
  const b64 = jsonString(entry.b64_json)
  const url = jsonString(entry.url)
  const revisedPrompt = jsonString(entry.revised_prompt)
  if (!b64 && !url) {
    throw new ProviderError('media', `image response data[${index}] did not include b64_json or url`)
  }
  return Object.freeze({
    format: imageFormat(entry),
    ...(b64 === undefined ? {} : { b64 }),
    ...(revisedPrompt === undefined ? {} : { revisedPrompt }),
    ...(url === undefined ? {} : { url }),
  })
}

function imageFormat(value: JsonObject): string {
  const format = jsonString(value.format)
  return format?.trim() || 'png'
}

function nonEmpty(value: string, field: string): string {
  const normalized = value.trim()
  if (!normalized) {
    throw new ValidationError(field, 'must be a non-empty string', value)
  }
  return normalized
}

function providerKey(value: string): string {
  return nonEmpty(value, 'provider').toLowerCase()
}
