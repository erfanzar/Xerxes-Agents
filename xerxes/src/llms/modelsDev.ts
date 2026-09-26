// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Model capabilities from models.dev, fetched live — the same source Kimi
 * Code uses for every provider that does not describe its own models.
 *
 * models.dev (https://models.dev/api.json) is an open catalog of providers
 * and their models: context and output limits, whether a model reasons and
 * how (effort values, an on/off toggle, a token budget), temperature support
 * and prices. Xerxes asks it at runtime and keeps the answer for ten minutes;
 * nothing is bundled, so a model or provider added upstream shows up without
 * a Xerxes release. When models.dev cannot be reached the answer is simply
 * "unknown" — never a guess.
 *
 * A profile is matched to a models.dev provider by its base URL first (data,
 * not a name table), then by an identical provider id, then — only when it
 * is unambiguous — by the model id alone.
 */

export const MODELS_DEV_URL = 'https://models.dev/api.json'

/** How a model's reasoning can be controlled, as its provider or models.dev states it. */
export interface LiveReasoning {
  /** The model reasons at all. */
  readonly supported: boolean
  /** Reasoning can be switched off (a toggle, a budget, or a `none` effort). */
  readonly canDisable: boolean
  /** Effort values the model accepts, in the order given. */
  readonly efforts: readonly string[]
  /** The effort the provider applies when none is chosen, when it says so. */
  readonly defaultEffort?: string
  /** The value that means "off" on the wire, when it is an effort (`none`). */
  readonly offEffort?: string
  /** A token budget can be set (models.dev `budget_tokens`). */
  readonly budget?: boolean
}

/** USD per million tokens. */
export interface LiveCost {
  readonly input?: number
  readonly output?: number
  readonly cacheRead?: number
  readonly cacheWrite?: number
}

/**
 * Which wire protocol a model speaks, when its catalog entry says so —
 * gateways (OpenCode, Cloudflare AI Gateway) serve some models over the
 * Anthropic API and others over OpenAI's.
 */
export type WireApi = 'anthropic-messages' | 'openai-responses' | 'openai-completions' | 'google-generative-ai'

export interface LiveModelCapability {
  readonly displayName?: string
  readonly description?: string
  readonly contextLimit?: number
  readonly maxOutputTokens?: number
  readonly reasoning?: LiveReasoning
  /** Whether the model accepts `temperature`. */
  readonly temperature?: boolean
  readonly cost?: LiveCost
  /** The protocol this model is served over, when stated per model or per provider. */
  readonly api?: WireApi
  /** A model-specific endpoint, when its catalog entry names one. */
  readonly apiBaseUrl?: string
  /** OpenAI `custom` grammar tools (Codex: `apply_patch_tool_type: freeform`). */
  readonly grammarTools?: boolean
  /** Responses-API tool search for deferred tools (Codex: `supports_search_tool`). */
  readonly toolSearch?: boolean
  /** Responses-API `additional_tools` developer items for deferred tools, where reported. */
  readonly additionalTools?: boolean
  /** Tools added mid-conversation, Kimi style (`supports_dynamic_tools`). */
  readonly dynamicTools?: boolean
  /** Anthropic tool references for deferred tools, where a provider reports them. */
  readonly toolReferences?: boolean
}

/**
 * The protocol an AI SDK package speaks — models.dev names the package that
 * serves each model. This maps protocol identities, not models.
 */
export function wireApiFromNpm(npm: unknown): WireApi | undefined {
  if (typeof npm !== 'string') return undefined
  if (npm === '@ai-sdk/anthropic' || npm === '@ai-sdk/google-vertex/anthropic') return 'anthropic-messages'
  if (npm === '@ai-sdk/openai' || npm === '@ai-sdk/azure') return 'openai-responses'
  if (npm === '@ai-sdk/google' || npm === '@ai-sdk/google-vertex') return 'google-generative-ai'
  if (npm === '@ai-sdk/openai-compatible') return 'openai-completions'
  return undefined
}

interface ModelsDevProvider {
  readonly id: string
  readonly api?: string
  readonly models: ReadonlyMap<string, LiveModelCapability>
}

type Json = Record<string, unknown>
const record = (value: unknown): Json => (value && typeof value === 'object' && !Array.isArray(value) ? value as Json : {})
const text = (value: unknown): string | undefined => (typeof value === 'string' && value.trim() ? value.trim() : undefined)
const positive = (value: unknown): number | undefined => (typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.floor(value) : undefined)
const price = (value: unknown): number | undefined => (typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined)

/**
 * Kimi Code's reading of `reasoning_options`: effort values are the levels
 * (`none` among them is the off switch); a toggle or a token budget means
 * thinking can be switched off; efforts with neither mean it is always on.
 */
export function reasoningFromModelsDev(reasoning: unknown, options: unknown): LiveReasoning | undefined {
  if (reasoning === false) return { supported: false, canDisable: false, efforts: [] }
  if (reasoning !== true) return undefined
  const list = Array.isArray(options) ? options.map(record) : []
  const values = list.filter(option => option.type === 'effort').flatMap(option => Array.isArray(option.values) ? option.values : [])
    .filter((value): value is string => typeof value === 'string' && value.trim() !== '')
  const offEffort = values.find(value => value.toLowerCase() === 'none')
  const efforts = [...new Set(values.filter(value => value.toLowerCase() !== 'none'))]
  const budget = list.some(option => option.type === 'budget_tokens')
  const switchable = budget || list.some(option => option.type === 'toggle')
  return {
    supported: true,
    canDisable: switchable || offEffort !== undefined || !efforts.length,
    efforts,
    ...(offEffort ? { offEffort } : {}),
    ...(budget ? { budget } : {}),
  }
}

function capabilityFromModelsDev(value: unknown, providerNpm?: unknown): LiveModelCapability | undefined {
  const model = record(value)
  const route = record(model.provider)
  const api = wireApiFromNpm(route.npm ?? providerNpm)
  const apiBaseUrl = text(route.api)
  if (!text(model.id)) return undefined
  const limit = record(model.limit)
  const cost = record(model.cost)
  const reasoning = reasoningFromModelsDev(model.reasoning, model.reasoning_options)
  const input = price(cost.input), output = price(cost.output), cacheRead = price(cost.cache_read), cacheWrite = price(cost.cache_write)
  const liveCost: LiveCost = {
    ...(input === undefined ? {} : { input }),
    ...(output === undefined ? {} : { output }),
    ...(cacheRead === undefined ? {} : { cacheRead }),
    ...(cacheWrite === undefined ? {} : { cacheWrite }),
  }
  const name = text(model.name)
  const description = text(model.description)
  const contextLimit = positive(limit.context)
  const maxOutputTokens = positive(limit.output)
  return {
    ...(name ? { displayName: name } : {}),
    ...(description ? { description } : {}),
    ...(contextLimit ? { contextLimit } : {}),
    ...(maxOutputTokens ? { maxOutputTokens } : {}),
    ...(reasoning ? { reasoning } : {}),
    ...(typeof model.temperature === 'boolean' ? { temperature: model.temperature } : {}),
    ...(Object.keys(liveCost).length ? { cost: liveCost } : {}),
    ...(api ? { api } : {}),
    ...(apiBaseUrl ? { apiBaseUrl } : {}),
  }
}

export function parseModelsDev(value: unknown): ModelsDevProvider[] {
  const providers: ModelsDevProvider[] = []
  for (const [key, raw] of Object.entries(record(value))) {
    const provider = record(raw)
    const models = new Map<string, LiveModelCapability>()
    for (const [modelKey, modelValue] of Object.entries(record(provider.models))) {
      const capability = capabilityFromModelsDev(modelValue, provider.npm)
      if (capability) models.set(text(record(modelValue).id) ?? modelKey, capability)
    }
    const api = text(provider.api)
    providers.push({ id: text(provider.id) ?? key, ...(api ? { api } : {}), models })
  }
  return providers
}

/** `https://API.z.ai/api/coding/paas/v4/` → `api.z.ai/api/coding/paas/v4`, for comparing endpoints. */
export function endpointKey(url: string | undefined): string | undefined {
  if (!url?.trim()) return undefined
  try {
    const parsed = new URL(url.trim())
    return `${parsed.host.toLowerCase()}${parsed.pathname.replace(/\/+$/, '')}`
  } catch { return undefined }
}

function isLocalEndpoint(endpoint: string): boolean {
  const host = endpoint.split('/')[0]!.replace(/:\d+$/, '').replace(/^\[|\]$/g, '')
  return host === 'localhost' || host.endsWith('.local') || host === '::1'
    || /^127\./.test(host) || /^10\./.test(host) || /^192\.168\./.test(host) || /^172\.(1[6-9]|2\d|3[01])\./.test(host)
}

const sameCapability = (a: LiveModelCapability, b: LiveModelCapability) =>
  a.contextLimit === b.contextLimit && a.maxOutputTokens === b.maxOutputTokens && JSON.stringify(a.reasoning) === JSON.stringify(b.reasoning)

export interface ModelsDevQuery {
  readonly baseUrl?: string
  readonly provider?: string
  readonly model: string
}

/**
 * The models.dev answer for one model of one profile, or undefined when
 * models.dev does not describe it unambiguously.
 */
export function findInModelsDev(providers: readonly ModelsDevProvider[], query: ModelsDevQuery): LiveModelCapability | undefined {
  const model = query.model.trim()
  if (!model) return undefined
  const bare = model.includes('/') ? model.slice(model.indexOf('/') + 1) : model
  const lookup = (provider: ModelsDevProvider) => provider.models.get(model) ?? provider.models.get(bare)
  const endpoint = endpointKey(query.baseUrl)
  if (endpoint) {
    const byEndpoint = providers.filter(provider => endpointKey(provider.api) === endpoint)
    for (const provider of byEndpoint) {
      const found = lookup(provider)
      if (found) return found
    }
    // Same host, another path (a coding plan vs the general API): the same
    // vendor describing the same model.
    const host = endpoint.split('/')[0]
    const sameHost = providers.filter(provider => endpointKey(provider.api)?.split('/')[0] === host).map(lookup)
      .filter((found): found is LiveModelCapability => found !== undefined)
    if (sameHost.length && sameHost.every(found => sameCapability(found, sameHost[0]!))) return sameHost[0]
  }
  // A self-hosted server (local or private address) runs its own build of a
  // model — its window can differ from the vendor's, so an id match alone
  // does not describe it.
  if (endpoint && isLocalEndpoint(endpoint)) return undefined
  const named = providers.find(provider => provider.id === query.provider?.trim().toLowerCase())
  const byName = named ? lookup(named) : undefined
  if (byName) return byName
  const everywhere = providers.map(lookup).filter((found): found is LiveModelCapability => found !== undefined)
  if (everywhere.length && everywhere.every(found => sameCapability(found, everywhere[0]!))) return everywhere[0]
  return undefined
}

export interface ModelsDevOptions {
  readonly url?: string
  readonly ttlMs?: number
  readonly fetchImplementation?: (url: string, init?: RequestInit) => Promise<Response>
  readonly now?: () => number
  readonly environment?: Readonly<Record<string, string | undefined>>
}

/**
 * The shared, briefly cached models.dev catalog. `peek` is synchronous for
 * callers that cannot wait (context windows, cost); `load` refreshes it.
 */
export class ModelsDevCatalog {
  private providers: readonly ModelsDevProvider[] = []
  private fetchedAt = 0
  private inflight: Promise<readonly ModelsDevProvider[]> | undefined
  private readonly ttlMs: number
  private readonly now: () => number

  constructor(private readonly options: ModelsDevOptions = {}) {
    this.ttlMs = options.ttlMs ?? 10 * 60_000
    this.now = options.now ?? Date.now
  }

  private get disabled(): boolean {
    const environment = this.options.environment ?? process.env
    const setting = environment.XERXES_MODELS_DEV?.trim() ?? ''
    if (/^(0|off|false|no)$/i.test(setting)) return true
    // Test runs never reach the network unless a test injects its own fetch.
    return !this.options.fetchImplementation && environment.NODE_ENV === 'test' && !/^(1|on|true|yes)$/i.test(setting)
  }

  peek(): readonly ModelsDevProvider[] { return this.providers }

  /** Load a models.dev document directly (tests, or an answer fetched elsewhere). */
  seed(document: unknown): void {
    this.providers = parseModelsDev(document)
    this.fetchedAt = this.now()
  }

  find(query: ModelsDevQuery): LiveModelCapability | undefined {
    return findInModelsDev(this.providers, query)
  }

  /** Refresh when stale (or on demand); failures keep the last answer. */
  async load(refresh = false): Promise<readonly ModelsDevProvider[]> {
    if (this.disabled) return this.providers
    if (!refresh && this.providers.length && this.now() - this.fetchedAt < this.ttlMs) return this.providers
    this.inflight ??= this.fetch().then(providers => {
      this.providers = providers
      this.fetchedAt = this.now()
      return providers
    }).finally(() => { this.inflight = undefined })
    try {
      return await this.inflight
    } catch {
      return this.providers
    }
  }

  private async fetch(): Promise<ModelsDevProvider[]> {
    const url = (this.options.environment ?? process.env).XERXES_MODELS_DEV_URL?.trim() || this.options.url || MODELS_DEV_URL
    const fetcher = this.options.fetchImplementation ?? ((input: string, init?: RequestInit) => fetch(input, init))
    const response = await fetcher(url, { headers: { Accept: 'application/json' }, signal: AbortSignal.timeout(20_000) })
    if (!response.ok) throw new Error(`models.dev answered ${response.status}`)
    const providers = parseModelsDev(await response.json())
    if (!providers.length) throw new Error('models.dev returned no providers')
    return providers
  }
}

/** One catalog per process. */
export const modelsDev = new ModelsDevCatalog()

// ── What providers reported ─────────────────────────────────────────────

const reported = new Map<string, LiveModelCapability>()
const reportKey = (provider: string, model: string) => `${provider.trim().toLowerCase()}\u0000${model.includes('/') ? model.slice(model.indexOf('/') + 1) : model}`

/**
 * Record what a provider said about one of its models (its /models entry,
 * Codex's catalog). Request builders consult this before models.dev.
 */
export function reportModelCapability(provider: string, model: string, capability: LiveModelCapability): void {
  const key = reportKey(provider, model)
  reported.set(key, { ...reported.get(key), ...capability })
}

/** Forget reported capabilities (tests). */
export function clearReportedCapabilities(): void {
  reported.clear()
}

/**
 * Everything known about how to talk to a model: what the provider reported,
 * filled in from models.dev. Empty when nothing is known — callers then use
 * the protocol's plain default, never a guess about the model.
 */
export function wireCapability(query: ModelsDevQuery & { readonly provider: string }): LiveModelCapability {
  const own = reported.get(reportKey(query.provider, query.model))
  const catalog = modelsDev.find(query)
  if (!own) return catalog ?? {}
  if (!catalog) return own
  return { ...catalog, ...own, ...(own.reasoning ?? catalog.reasoning ? { reasoning: own.reasoning ?? catalog.reasoning! } : {}) }
}
