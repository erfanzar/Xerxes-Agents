// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * `generate_image` — the agent-facing image-generation tool.
 *
 * Routes through the pi-ai-style image API registry (`src/images/`), so the
 * model picks any catalogued image route (default: OpenRouter's auto router)
 * and every generated file is written inside the workspace with
 * magic-byte-verified media types. Credentials are resolved by the host at
 * call time; this module never reads process state.
 */

import { mkdir } from 'node:fs/promises'
import { dirname, join } from 'node:path'

import { sniffImageMediaType } from '../core/multimodal.js'
import { ProviderError, ValidationError } from '../core/errors.js'
import {
  DEFAULT_IMAGE_MODEL_REFERENCE,
  resolveImageModel,
} from '../images/index.js'
import { generateImages } from '../images/index.js'
import type { AssistantImages, ImagesContent, ImagesModel, ImagesOptions } from '../images/index.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import { decodeMediaBase64 } from './mediaHttp.js'
import { WorkspacePathResolver } from './pathSafety.js'
import { optionalString, requiredString } from './inputs.js'

export const GENERATE_IMAGE_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'generate_image',
    description:
      'Generate images from a text prompt through a catalogued image model '
      + '(default: OpenRouter auto router). Files are saved into the workspace.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        prompt: { type: 'string', description: 'Non-empty image description.' },
        model: {
          type: 'string',
          description:
            'Catalogued image-model reference such as "openrouter/google/gemini-3-pro-image" '
            + 'or a bare route id. Defaults to the OpenRouter auto router.',
        },
        output_path: {
          type: 'string',
          description:
            'Optional workspace-relative destination file. Defaults to images/<slug>-<timestamp>.<ext>.',
        },
      },
      required: ['prompt'],
    },
  },
}

/** Host-supplied configuration; `resolveApiKey` is consulted per call. */
export interface GenerateImageToolOptions {
  /** Resolve the bearer credential for the wire host at call time. */
  readonly resolveApiKey: () => string
  /** Filesystem root the tool may write into. */
  readonly workspaceRoot: string
  /** Default catalogued model reference; `openrouter/auto` when omitted. */
  readonly defaultModel?: string
  /** Injectable transport for offline tests and policy-pinned hosts. */
  readonly fetch?: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>
  /** Overall request deadline in milliseconds. */
  readonly timeoutMs?: number
}

export interface SavedGeneratedImage {
  readonly bytes: number
  readonly mediaType: string
  readonly path: string
}

export async function generateImageTool(inputs: JsonObject, options: GenerateImageToolOptions): Promise<JsonObject> {
  const prompt = requiredString(inputs, 'prompt')
  const reference = optionalString(inputs, 'model') ?? options.defaultModel
  const outputPath = optionalString(inputs, 'output_path')
  const model = resolveImageModel(reference)
  const apiKey = options.resolveApiKey()?.trim()
  if (!apiKey) {
    throw new ValidationError(
      'api_key',
      `no image API key configured; set the credential for provider '${model.provider}' and retry`,
    )
  }

  const result = await generateImages(model, { input: [{ type: 'text', text: prompt }] }, {
    apiKey,
    ...(options.fetch ? { fetch: options.fetch } : {}),
    ...(options.timeoutMs === undefined ? {} : { timeoutMs: options.timeoutMs }),
  } satisfies ImagesOptions)
  if (result.stopReason !== 'stop') {
    throw new ProviderError('images', result.errorMessage ?? `image generation ${result.stopReason}`)
  }

  const paths = new WorkspacePathResolver(options.workspaceRoot)
  const saved: SavedGeneratedImage[] = []
  const images = result.output.filter((part): part is Extract<ImagesContent, { type: 'image' }> => part.type === 'image')
  if (images.length === 0) {
    throw new ProviderError('images', 'image generation produced no images (text-only response)')
  }
  for (const [index, image] of images.entries()) {
    saved.push(await saveGeneratedImage(paths, model, image, outputPath, images.length > 1 ? index : undefined))
  }

  return serializableGenerateImageResult(result, model, saved)
}

/** Register the tool when a host supplies real configuration (opt-in pattern). */
export function registerGenerateImageTool(registry: ToolRegistry, options: GenerateImageToolOptions): void {
  registry.register(GENERATE_IMAGE_DEFINITION, inputs => generateImageTool(inputs, options))
}

export function serializableGenerateImageResult(
  result: AssistantImages,
  model: ImagesModel,
  saved: readonly SavedGeneratedImage[],
): JsonObject {
  const usage = result.usage
  return {
    model: `${model.provider}/${model.id}`,
    provider: model.provider,
    response_id: result.responseId ?? '',
    saved: saved.map(image => ({
      bytes: image.bytes,
      media_type: image.mediaType,
      path: image.path,
    })),
    ...(result.output.some(part => part.type === 'text')
      ? {
        text: result.output
          .filter((part): part is Extract<ImagesContent, { type: 'text' }> => part.type === 'text')
          .map(part => part.text)
          .join('\n'),
      }
      : {}),
    stop_reason: result.stopReason,
    ...(usage
      ? {
        usage: {
          cost_usd: usage.cost.total,
          input_tokens: usage.input,
          output_tokens: usage.output,
          total_tokens: usage.totalTokens,
        },
      }
      : {}),
  }
}

async function saveGeneratedImage(
  paths: WorkspacePathResolver,
  model: ImagesModel,
  image: Extract<ImagesContent, { type: 'image' }>,
  outputPath: string | undefined,
  index: number | undefined,
): Promise<SavedGeneratedImage> {
  const bytes = decodeMediaBase64(image.data, 'image.data')
  // The magic-byte sniff is authoritative for the reported media type; the
  // provider's mime only picks the extension when bytes are unrecognizable.
  const mediaType = sniffImageMediaType(bytes) ?? image.mimeType
  const destination = outputPath === undefined
    ? defaultImagePath(model, mediaType, index)
    // An explicit path with several images gets -2/-3... suffixes so later
    // images cannot overwrite earlier ones.
    : index === undefined
      ? outputPath
      : withIndexSuffix(outputPath, index + 1)
  const target = await paths.resolve(destination)
  await mkdir(dirname(target), { recursive: true })
  await Bun.write(target, bytes)
  return {
    bytes: bytes.byteLength,
    mediaType,
    path: await paths.relative(target),
  }
}

function withIndexSuffix(path: string, n: number): string {
  const dot = path.lastIndexOf('.')
  const stem = dot > 0 ? path.slice(0, dot) : path
  const extension = dot > 0 ? path.slice(dot) : ''
  return `${stem}-${n}${extension}`
}

function defaultImagePath(model: ImagesModel, mediaType: string, index: number | undefined): string {
  const slug = model.id
    .toLowerCase()
    .replaceAll(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'image'
  const suffix = index === undefined ? '' : `-${index + 1}`
  return join('images', `${slug}-${Date.now()}${suffix}.${extensionFor(mediaType)}`)
}

function extensionFor(mediaType: string): string {
  const normalized = mediaType.toLowerCase()
  if (normalized === 'image/jpeg') return 'jpg'
  if (normalized === 'image/gif' || normalized === 'image/webp') return normalized.slice('image/'.length)
  return 'png'
}
