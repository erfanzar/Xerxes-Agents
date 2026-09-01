// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { imagesError } from './types.js'
import type {
  AssistantImages,
  ImagesApiProvider,
  ImagesContent,
  ImagesContext,
  ImagesModel,
  ImagesOptions,
  ImagesUsage,
} from './types.js'

/**
 * The `openrouter-images` wire API (pi-ai `api/openrouter-images.ts`).
 *
 * OpenRouter serves image generation through chat completions: a normal
 * `messages` payload plus a `modalities` array, answered with assistant text
 * and/or base64 `image_url` parts. Everything here is native fetch — no SDK.
 */
export const OPENROUTER_IMAGES_API = 'openrouter-images'

const DATA_URL_PATTERN = /^data:([^;]+);base64,(.+)$/

/** Never rejects: failures land in the returned result (pi-ai parity). */
export async function generateImagesViaOpenRouter(
  model: ImagesModel,
  context: ImagesContext,
  options: ImagesOptions = {},
): Promise<AssistantImages> {
  const startedAt = Date.now()
  try {
    const apiKey = options.apiKey
    if (!apiKey) {
      throw new Error(`no API key for provider: ${model.provider}`)
    }
    const body = JSON.stringify(buildParams(model, context))
    const response = await postWithRetry(model, body, apiKey, options)
    if (!response.ok) {
      const detail = (await response.text()).slice(0, 4_096)
      throw new Error(`openrouter-images request failed (${response.status}): ${detail}`)
    }
    const payload = asRecord(await response.json())
    if (!payload) {
      throw new Error('openrouter-images returned a non-object response body')
    }
    const parts: ImagesContent[] = []
    const responseId = typeof payload.id === 'string' && payload.id ? payload.id : undefined
    const usage = parseUsage(asRecord(payload.usage), model)
    const choices = Array.isArray(payload.choices) ? payload.choices : []
    const message = asRecord(asRecord(choices[0])?.message)
    if (message) {
      if (typeof message.content === 'string' && message.content.length > 0) {
        parts.push({ type: 'text', text: message.content })
      }
      for (const image of Array.isArray(message.images) ? message.images : []) {
        const parsed = parseImagePart(image)
        if (parsed) parts.push(parsed)
      }
    }
    const output: AssistantImages = {
      api: model.api,
      model: model.id,
      output: parts,
      provider: model.provider,
      ...(responseId === undefined ? {} : { responseId }),
      stopReason: 'stop',
      timestamp: startedAt,
      ...(usage === undefined ? {} : { usage }),
    }
    return output
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    return imagesError(model, message, options, startedAt)
  }
}

/** Registration shape expected by the image API registry. */
export function openrouterImagesApiProvider(): ImagesApiProvider {
  return { api: OPENROUTER_IMAGES_API, generateImages: generateImagesViaOpenRouter }
}

/** Chat-completions payload with `modalities` selected from the model surface. */
export function buildParams(
  model: ImagesModel,
  context: ImagesContext,
): Record<string, unknown> {
  const content = context.input.map((part): Record<string, unknown> => part.type === 'text'
    ? { type: 'text', text: sanitizeSurrogates(part.text) }
    : { type: 'image_url', image_url: { url: `data:${part.mimeType};base64,${part.data}` } })
  return {
    model: model.id,
    messages: [{ role: 'user', content }],
    stream: false,
    modalities: model.output.includes('text') ? ['image', 'text'] : ['image'],
  }
}

/** Per-million catalog pricing applied to reported token counts (pi-ai parity). */
export function parseUsage(
  rawUsage: Record<string, unknown> | undefined,
  model: ImagesModel,
): ImagesUsage | undefined {
  if (!rawUsage) return undefined
  const promptTokens = numberAt(rawUsage, 'prompt_tokens') ?? 0
  const details = asRecord(rawUsage.prompt_tokens_details)
  const reportedCachedTokens = numberAt(details, 'cached_tokens') ?? 0
  const cacheWriteTokens = numberAt(details, 'cache_write_tokens') ?? 0
  const cacheReadTokens = cacheWriteTokens > 0
    ? Math.max(0, reportedCachedTokens - cacheWriteTokens)
    : reportedCachedTokens
  const input = Math.max(0, promptTokens - cacheReadTokens - cacheWriteTokens)
  const outputTokens = numberAt(rawUsage, 'completion_tokens') ?? 0
  const inputCost = (model.cost.input / 1_000_000) * input
  const outputCost = (model.cost.output / 1_000_000) * outputTokens
  const cacheReadCost = (model.cost.cacheRead / 1_000_000) * cacheReadTokens
  const cacheWriteCost = (model.cost.cacheWrite / 1_000_000) * cacheWriteTokens
  return {
    cacheRead: cacheReadTokens,
    cacheWrite: cacheWriteTokens,
    cost: {
      cacheRead: cacheReadCost,
      cacheWrite: cacheWriteCost,
      input: inputCost,
      output: outputCost,
      total: inputCost + outputCost + cacheReadCost + cacheWriteCost,
    },
    input,
    output: outputTokens,
    totalTokens: input + outputTokens + cacheReadTokens + cacheWriteTokens,
  }
}

function parseImagePart(value: unknown): ImagesContent | undefined {
  const record = asRecord(value)
  const imageUrl = record?.image_url
  const url = typeof imageUrl === 'string'
    ? imageUrl
    : typeof imageUrl === 'object' && imageUrl !== null
      ? stringAt(asRecord(imageUrl), 'url')
      : undefined
  if (!url) return undefined
  const matches = DATA_URL_PATTERN.exec(url)
  if (!matches?.[1] || !matches[2]) return undefined
  return { type: 'image', data: matches[2], mimeType: matches[1] }
}

const RETRYABLE_STATUS = new Set([408, 409, 429])

async function postWithRetry(
  model: ImagesModel,
  body: string,
  apiKey: string,
  options: ImagesOptions,
): Promise<Response> {
  const request = options.fetch ?? fetch
  const maxRetries = options.maxRetries ?? 0
  const maxRetryDelayMs = options.maxRetryDelayMs ?? 60_000
  const deadline = createDeadline(options.timeoutMs, options.signal)
  try {
    for (let attempt = 0; ; attempt++) {
      let response: Response
      try {
        response = await request(new URL('chat/completions', withTrailingSlash(model.baseUrl)), {
          method: 'POST',
          headers: {
            Accept: 'application/json',
            Authorization: `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
            ...model.headers,
            ...options.headers,
          },
          body,
          signal: deadline.signal,
        })
      } catch (error) {
        // Transport-level failures (network, abort, timeout) are not retried,
        // matching pi-ai's SDK configuration (maxRetries: 0 on the client).
        throw error
      }
      if (response.ok) return response
      if (attempt >= maxRetries || !retryable(response)) {
        return response
      }
      await sleep(retryDelayMs(response, attempt, maxRetryDelayMs), options)
    }
  } finally {
    deadline.dispose()
  }
}

function retryable(response: Response): boolean {
  const shouldRetry = response.headers.get('x-should-retry')
  if (shouldRetry === 'true') return true
  if (shouldRetry === 'false') return false
  return response.status === 408
    || response.status === 409
    || response.status === 429
    || response.status >= 500
}

function retryDelayMs(response: Response, attempt: number, maxRetryDelayMs: number): number {
  const retryAfterMs = response.headers.get('retry-after-ms')
  if (retryAfterMs) {
    const value = Number.parseFloat(retryAfterMs)
    if (!Number.isNaN(value)) return cappedServerDelay(value, maxRetryDelayMs)
  }
  const retryAfter = response.headers.get('retry-after')
  if (retryAfter) {
    const seconds = Number.parseFloat(retryAfter)
    const delayMs = Number.isNaN(seconds) ? Date.parse(retryAfter) - Date.now() : seconds * 1_000
    if (Number.isFinite(delayMs)) return cappedServerDelay(delayMs, maxRetryDelayMs)
  }
  // Mirrors the pinned OpenAI SDK backoff: 0.5s * 2^attempt, capped at 8s,
  // with up to 25% jitter.
  return Math.min(0.5 * 2 ** attempt, 8) * 1_000 * (1 - Math.random() * 0.25)
}

function cappedServerDelay(delayMs: number, maxRetryDelayMs: number): number {
  if (maxRetryDelayMs > 0 && delayMs > maxRetryDelayMs) {
    throw new Error(`server requested ${Math.ceil(delayMs / 1_000)}s retry delay (max: ${Math.ceil(maxRetryDelayMs / 1_000)}s)`)
  }
  return delayMs
}

function sleep(ms: number, options: ImagesOptions): Promise<void> {
  const delay = Math.max(0, ms)
  if (options.sleep) return options.sleep(delay)
  return new Promise(resolve => setTimeout(resolve, delay))
}

interface RequestDeadline {
  readonly signal: AbortSignal
  dispose(): void
}

/** Combine the caller signal with an optional per-request timeout. */
function createDeadline(timeoutMs: number | undefined, signal: AbortSignal | undefined): RequestDeadline {
  if (timeoutMs === undefined) {
    if (!signal) return { signal: new AbortController().signal, dispose: () => {} }
    const controller = new AbortController()
    const abort = () => controller.abort(signal.reason)
    if (signal.aborted) abort()
    else signal.addEventListener('abort', abort, { once: true })
    return {
      signal: controller.signal,
      dispose: () => signal.removeEventListener('abort', abort),
    }
  }
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(new Error(`request timed out after ${timeoutMs}ms`)), timeoutMs)
  const abort = () => controller.abort(signal?.reason)
  if (signal?.aborted) abort()
  else signal?.addEventListener('abort', abort, { once: true })
  return {
    signal: controller.signal,
    dispose: () => {
      clearTimeout(timer)
      signal?.removeEventListener('abort', abort)
    },
  }
}

/** Removes unpaired surrogates that break JSON serialization (pi-ai parity). */
export function sanitizeSurrogates(text: string): string {
  return text.replace(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g, '')
}

function withTrailingSlash(url: string): string {
  return url.endsWith('/') ? url : `${url}/`
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined
}

function stringAt(record: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = record?.[key]
  return typeof value === 'string' ? value : undefined
}

function numberAt(record: Record<string, unknown> | undefined, key: string): number | undefined {
  const value = record?.[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}
