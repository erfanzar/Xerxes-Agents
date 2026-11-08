// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Ask a local OpenAI-compatible engine for the context window it actually
 * loaded. `/v1/models` does not report it on either engine Xerxes manages, so
 * without this the runtime only learns the window when the server refuses a
 * prompt as too long.
 *
 * - LM Studio: `GET /api/v0/models`, `loaded_context_length` on the entry whose
 *   `state` is `loaded` (one entry per loaded identifier).
 * - llama.cpp `llama-server`: `GET /props`, `default_generation_settings.n_ctx`
 *   (the per-slot window a single request can use), else top-level `n_ctx`.
 *   llama-server serves one model under any requested name, so its window
 *   applies to every model id on that origin.
 *
 * Only http(s) loopback origins are probed. The network goes through a narrow
 * fetch port, and failures come back as a typed reason instead of a throw so a
 * caller can keep its current behaviour and say why.
 */
import { isLoopbackBaseUrl } from '../llms/providerRegistry.js'

export const LOADED_WINDOW_TIMEOUT_MS = 3000
const MAX_RESPONSE_BYTES = 1_048_576

export type LocalEngineName = 'lmstudio' | 'llama-server'

/** The narrow network port: one GET, an abort signal, optional headers. */
export type WindowFetch = (
  url: string,
  init: { readonly signal: AbortSignal; readonly headers: Readonly<Record<string, string>> },
) => Promise<Response>

export type LoadedWindows =
  | {
      readonly ok: true
      readonly engine: 'lmstudio'
      /** Loaded window by API identifier; models that are not loaded are absent. */
      readonly windows: ReadonlyMap<string, number>
    }
  | { readonly ok: true; readonly engine: 'llama-server'; readonly window: number }
  | { readonly ok: false; readonly reason: string }

export interface ProbeLoadedWindowsInput {
  readonly baseUrl: string
  /** Sent as a bearer token when non-empty (an LM Studio server can require one). */
  readonly apiKey?: string
  readonly signal?: AbortSignal
  readonly timeoutMs?: number
}

export interface LmStudioModelEntry {
  readonly id: string
  readonly state: string
  readonly type?: string
  readonly loadedContextLength?: number
  readonly maxContextLength?: number
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function positiveInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0 ? value : undefined
}

/**
 * The origin to probe, or undefined when the base URL is not an http(s)
 * loopback endpoint. Unix-socket URLs count as local elsewhere but cannot be
 * fetched here.
 */
export function loopbackOrigin(baseUrl: string): string | undefined {
  if (!isLoopbackBaseUrl(baseUrl)) return undefined
  let url: URL
  try {
    url = new URL(baseUrl.trim())
  } catch {
    return undefined
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return undefined
  return url.origin
}

/** Validate LM Studio's `/api/v0/models` body. Returns undefined when it is not that shape. */
export function parseLmStudioModels(value: unknown): LmStudioModelEntry[] | undefined {
  if (!isRecord(value) || !Array.isArray(value.data)) return undefined
  const entries: LmStudioModelEntry[] = []
  for (const item of value.data) {
    if (!isRecord(item) || typeof item.id !== 'string' || item.id.trim() === '' || typeof item.state !== 'string') {
      return undefined
    }
    const loaded = positiveInteger(item.loaded_context_length)
    const max = positiveInteger(item.max_context_length)
    entries.push({
      id: item.id,
      state: item.state,
      ...(typeof item.type === 'string' ? { type: item.type } : {}),
      ...(loaded === undefined ? {} : { loadedContextLength: loaded }),
      ...(max === undefined ? {} : { maxContextLength: max }),
    })
  }
  return entries
}

/** Loaded windows by identifier from a parsed LM Studio model list. */
export function lmStudioLoadedWindows(entries: readonly LmStudioModelEntry[]): Map<string, number> {
  const windows = new Map<string, number>()
  for (const entry of entries) {
    if (entry.state === 'loaded' && entry.loadedContextLength !== undefined) windows.set(entry.id, entry.loadedContextLength)
  }
  return windows
}

/** The per-request window from llama-server's `/props`, or undefined when absent. */
export function llamaServerWindow(value: unknown): number | undefined {
  if (!isRecord(value)) return undefined
  const settings = value.default_generation_settings
  return (isRecord(settings) ? positiveInteger(settings.n_ctx) : undefined) ?? positiveInteger(value.n_ctx)
}

/** The window a probe reports for one model id, if any. */
export function windowFor(result: LoadedWindows, model: string): number | undefined {
  if (!result.ok) return undefined
  if (result.engine === 'llama-server') return result.window
  return result.windows.get(model.trim())
}

type Fetched = { readonly kind: 'json'; readonly value: unknown } | { readonly kind: 'status'; readonly status: number }

async function getJson(
  fetch: WindowFetch,
  url: string,
  headers: Readonly<Record<string, string>>,
  signal: AbortSignal,
): Promise<Fetched> {
  const response = await fetch(url, { signal, headers })
  if (!response.ok) {
    await response.body?.cancel().catch(() => undefined)
    return { kind: 'status', status: response.status }
  }
  const declared = Number(response.headers.get('content-length'))
  if (Number.isFinite(declared) && declared > MAX_RESPONSE_BYTES) {
    await response.body?.cancel().catch(() => undefined)
    throw new Error(`response from ${new URL(url).pathname} is larger than ${MAX_RESPONSE_BYTES} bytes`)
  }
  const text = await response.text()
  if (text.length > MAX_RESPONSE_BYTES) throw new Error(`response from ${new URL(url).pathname} is larger than ${MAX_RESPONSE_BYTES} bytes`)
  try {
    return { kind: 'json', value: JSON.parse(text) as unknown }
  } catch {
    throw new Error(`${new URL(url).pathname} returned invalid JSON`)
  }
}

function reasonOf(error: unknown, signal: AbortSignal): string {
  if (signal.aborted) {
    const cause = signal.reason
    return cause instanceof Error ? cause.message : 'probe cancelled'
  }
  return error instanceof Error ? error.message : String(error)
}

/**
 * Ask the engine behind `baseUrl` which windows it has loaded. LM Studio is
 * tried first; a 404 there falls through to llama-server's `/props`.
 */
export async function probeLoadedWindows(input: ProbeLoadedWindowsInput, fetch: WindowFetch): Promise<LoadedWindows> {
  const origin = loopbackOrigin(input.baseUrl)
  if (origin === undefined) return { ok: false, reason: 'not an http(s) loopback endpoint' }
  const timeoutMs = input.timeoutMs ?? LOADED_WINDOW_TIMEOUT_MS
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(new Error(`no answer from ${origin} within ${timeoutMs}ms`)), timeoutMs)
  const onAbort = () => controller.abort(input.signal?.reason)
  input.signal?.addEventListener('abort', onAbort, { once: true })
  if (input.signal?.aborted) controller.abort(input.signal.reason)
  const apiKey = input.apiKey?.trim()
  const headers: Record<string, string> = { Accept: 'application/json', ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}) }
  try {
    const lmStudio = await getJson(fetch, `${origin}/api/v0/models`, headers, controller.signal)
    if (lmStudio.kind === 'json') {
      const entries = parseLmStudioModels(lmStudio.value)
      if (entries !== undefined) return { ok: true, engine: 'lmstudio', windows: lmStudioLoadedWindows(entries) }
    } else if (lmStudio.status === 401 || lmStudio.status === 403) {
      return { ok: false, reason: `${origin}/api/v0/models refused the request (HTTP ${lmStudio.status})` }
    }
    const props = await getJson(fetch, `${origin}/props`, headers, controller.signal)
    if (props.kind === 'json') {
      const window = llamaServerWindow(props.value)
      if (window !== undefined) return { ok: true, engine: 'llama-server', window }
      return { ok: false, reason: `${origin}/props has no n_ctx` }
    }
    return {
      ok: false,
      reason: `${origin} exposes neither LM Studio's /api/v0/models nor llama-server's /props (HTTP ${props.status})`,
    }
  } catch (error) {
    return { ok: false, reason: reasonOf(error, controller.signal) }
  } finally {
    clearTimeout(timer)
    input.signal?.removeEventListener('abort', onAbort)
  }
}

/** Production port: the platform fetch, never following redirects off the loopback origin. */
export const systemWindowFetch: WindowFetch = (url, init) =>
  fetch(url, { signal: init.signal, headers: { ...init.headers }, redirect: 'error' })
