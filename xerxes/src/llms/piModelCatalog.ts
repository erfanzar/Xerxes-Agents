// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import catalog from './piModelCatalog.generated.json' with { type: 'json' }

export interface PiModelCapabilities {
  readonly api: string
  readonly baseUrl?: string
  readonly compat?: Readonly<Record<string, unknown>>
  readonly contextLimit: number
  readonly maxOutputTokens: number
  readonly reasoning: boolean
  readonly thinkingLevelMap?: Readonly<Record<string, string | null>>
}

interface GeneratedModelCapabilities {
  readonly api: string
  readonly base_url?: string
  readonly compat?: Readonly<Record<string, unknown>>
  readonly context_limit: number
  readonly max_output_tokens: number
  readonly reasoning: boolean
  readonly thinking_level_map?: Readonly<Record<string, string | null>>
}

const CATALOGS: Readonly<
  Record<string, Readonly<Record<string, GeneratedModelCapabilities>>>
> = catalog.providers

function normalizedProvider(provider: string): string {
  return provider.trim().toLowerCase().replaceAll('_', '-')
}

function modelId(model: string, provider: string): string {
  const configured = model.trim()
  if (provider === 'openrouter' && configured.toLowerCase().startsWith('openrouter/')) {
    return configured.slice('openrouter/'.length)
  }
  const slash = configured.indexOf('/')
  return slash >= 0 ? configured.slice(slash + 1) : configured
}

/** Resolve one model's input/output capacities from Pi's generated catalogs. */
export function piCatalogModelCapabilities(
  model: string,
  provider: string,
): PiModelCapabilities | undefined {
  const key = normalizedProvider(provider)
  const id = modelId(model, key)
  if (!id) return undefined
  const capabilities = CATALOGS[key]?.[id]
  return capabilities === undefined
    ? undefined
    : {
        api: capabilities.api,
        contextLimit: capabilities.context_limit,
        maxOutputTokens: capabilities.max_output_tokens,
        reasoning: capabilities.reasoning,
        ...(capabilities.base_url ? { baseUrl: capabilities.base_url } : {}),
        ...(capabilities.compat ? { compat: capabilities.compat } : {}),
        ...(capabilities.thinking_level_map ? { thinkingLevelMap: capabilities.thinking_level_map } : {}),
      }
}

/** Resolve a model's context capacity from Pi's generated provider catalogs. */
export function piCatalogContextWindow(model: string, provider: string): number | undefined {
  return piCatalogModelCapabilities(model, provider)?.contextLimit
}

export interface PiContextWindowOptions {
  readonly configuredContextWindow?: number
  readonly model: string
  readonly provider: string
}

/** Resolve context capacity with Pi's precedence: override, catalog, unknown. */
export function resolvePiContextWindow(options: PiContextWindowOptions): number | undefined {
  const configured = options.configuredContextWindow
  if (typeof configured === 'number' && Number.isSafeInteger(configured) && configured > 0) {
    return configured
  }
  return piCatalogContextWindow(options.model, options.provider)
}

/** Provenance for diagnostics and reproducible catalog updates. */
export const PI_MODEL_CATALOG_SOURCE = Object.freeze({ ...catalog.source })
