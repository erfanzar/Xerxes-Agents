// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Image-generation surface mirroring pi-ai 0.84.4 (`images.ts`, `types.ts`).
 *
 * The contract is deliberately close to Pi's: an {@link ImagesModel} names the
 * wire API (`api`) plus endpoint and pricing, an {@link ImagesContext} carries
 * text/image input parts, and the API implementation resolves to an
 * {@link AssistantImages} result that reports outputs, usage, and failures
 * through `stopReason` instead of rejecting.
 */

/** Wire APIs with a built-in implementation. */
export type KnownImagesApi = 'openrouter-images'

/** Any registered image wire API; unknown strings are allowed for plugins. */
export type ImagesApi = KnownImagesApi | (string & {})

/** Content kinds an image model may accept or produce. */
export type ImagesContentKind = 'text' | 'image'

/** One input or output part: text, or base64 image bytes with a mime type. */
export type ImagesContent =
  | { readonly type: 'text'; readonly text: string }
  | { readonly type: 'image'; readonly data: string; readonly mimeType: string }

export interface ImagesContext {
  readonly input: readonly ImagesContent[]
}

/** USD price per million tokens for one direction. */
export interface ImagesModelCost {
  readonly cacheRead: number
  readonly cacheWrite: number
  readonly input: number
  readonly output: number
}

/**
 * One catalogued image-generation model (pi-ai `ImagesModel`).
 *
 * `cost` entries are per-million USD; the catalog's `-1000000` sentinel means
 * "dynamic/unknown" (OpenRouter's auto router) and flows through arithmetic
 * untouched, matching Pi.
 */
export interface ImagesModel {
  /** Wire API that serves this model, e.g. `openrouter-images`. */
  readonly api: ImagesApi
  /** Route to send as the request `model` field. */
  readonly id: string
  readonly input: readonly ImagesContentKind[]
  /** Optional per-model request headers. */
  readonly headers?: Readonly<Record<string, string>>
  readonly name: string
  /** Cataloguing provider (e.g. `openrouter`), not necessarily the wire host. */
  readonly provider: string
  readonly baseUrl: string
  readonly output: readonly ImagesContentKind[]
  readonly cost: ImagesModelCost
}

export interface ImagesUsageCost {
  readonly cacheRead: number
  readonly cacheWrite: number
  readonly input: number
  readonly output: number
  readonly total: number
}

export interface ImagesUsage {
  readonly cacheRead: number
  readonly cacheWrite: number
  readonly cost: ImagesUsageCost
  readonly input: number
  readonly output: number
  readonly totalTokens: number
}

export type ImagesStopReason = 'aborted' | 'error' | 'stop'

/**
 * Terminal image-generation result. Failures never reject: they land here
 * with `stopReason: "error"` (or `"aborted"`) and an `errorMessage`.
 */
export interface AssistantImages {
  readonly api: ImagesApi
  readonly errorMessage?: string
  readonly model: string
  readonly output: readonly ImagesContent[]
  readonly provider: string
  readonly responseId?: string
  readonly stopReason: ImagesStopReason
  readonly timestamp: number
  readonly usage?: ImagesUsage
}

/** Options for one image-generation request (pi-ai `ImagesOptions`). */
export interface ImagesOptions {
  /** Bearer credential for the wire host; resolved by the caller, not from env. */
  readonly apiKey?: string
  /** Extra request headers, merged over the model's own. */
  readonly headers?: Readonly<Record<string, string>>
  /** Injectable transport for offline tests and policy-pinned hosts. */
  readonly fetch?: (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>
  /** Overall request deadline in milliseconds. */
  readonly timeoutMs?: number
  /**
   * Retry attempts for retryable HTTP failures (408/409/429/5xx, or an
   * `x-should-retry` header). Default 0, matching pi-ai.
   */
  readonly maxRetries?: number
  /** Cap on server-requested retry delays; default 60_000ms like pi-ai. */
  readonly maxRetryDelayMs?: number
  /** Injectable delay for deterministic retry tests. */
  readonly sleep?: (ms: number) => Promise<void>
  /** Abort signal for the whole request. */
  readonly signal?: AbortSignal
}

/** The uniform contract every image wire-API module exports. */
export interface ImagesApiProvider {
  readonly api: ImagesApi
  readonly generateImages: (
    model: ImagesModel,
    context: ImagesContext,
    options?: ImagesOptions,
  ) => Promise<AssistantImages>
}

/** Build a terminal error result for a failed image request. */
export function imagesError(
  model: ImagesModel,
  message: string,
  options?: ImagesOptions,
  timestamp = Date.now(),
): AssistantImages {
  return {
    api: model.api,
    ...(message ? { errorMessage: message } : {}),
    model: model.id,
    output: [],
    provider: model.provider,
    stopReason: options?.signal?.aborted ? 'aborted' : 'error',
    timestamp,
  }
}
