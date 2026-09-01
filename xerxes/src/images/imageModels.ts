// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import generated from './imageModels.generated.json' with { type: 'json' }
import type { ImagesContentKind, ImagesModel, ImagesModelCost } from './types.js'

interface GeneratedImageModelRecord {
  readonly api: string
  readonly baseUrl: string
  readonly cost: ImagesModelCost
  readonly id: string
  readonly headers?: Readonly<Record<string, string>>
  readonly input: readonly ImagesContentKind[]
  readonly name: string
  readonly output: readonly ImagesContentKind[]
  readonly provider: string
}

interface GeneratedImageCatalog {
  readonly source: {
    readonly package: string
    readonly version: string
    readonly generated_at?: string
  }
  readonly providers: Readonly<Record<string, readonly GeneratedImageModelRecord[]>>
}

const CATALOG = generated as unknown as GeneratedImageCatalog

/** Provenance of the generated image-model data. */
export const IMAGE_MODEL_SOURCE = CATALOG.source

const REGISTRY: ReadonlyMap<string, readonly ImagesModel[]> = new Map(
  Object.entries(CATALOG.providers).map(([provider, models]) => [
    provider,
    models.map(model => ({
      api: model.api,
      baseUrl: model.baseUrl,
      cost: model.cost,
      id: model.id,
      ...(model.headers ? { headers: model.headers } : {}),
      input: [...model.input],
      name: model.name,
      output: [...model.output],
      provider: model.provider,
    })),
  ]),
)

/** Every catalogued image provider (pi-ai `getImageProviders`). */
export function getImageProviders(): string[] {
  return [...REGISTRY.keys()].sort((left, right) => left.localeCompare(right))
}

/** Every catalogued model for one provider, or all providers (pi-ai `getImageModels`). */
export function getImageModels(provider?: string): readonly ImagesModel[] {
  if (provider !== undefined) return REGISTRY.get(normalizeProvider(provider)) ?? []
  return [...REGISTRY.values()].flat()
}

/** One catalogued model by provider and route id (pi-ai `getImageModel`). */
export function getImageModel(provider: string, modelId: string): ImagesModel | undefined {
  // Route ids themselves contain slashes (vendor/model), so this is an exact
  // match against the catalogued id — never a prefix strip.
  return getImageModels(provider).find(model => model.id === modelId)
}

export const DEFAULT_IMAGE_MODEL_REFERENCE = 'openrouter/auto'

/**
 * Resolve a user/model-facing image-model reference to a catalog entry.
 *
 * Accepted forms: `provider/model-id` (catalog provider prefix), a bare model
 * id searched across providers, or `undefined` for the OpenRouter auto
 * router. Throws ValidationError with the available providers when nothing
 * matches — a guessable fallback would silently send pixels to the wrong
 * host.
 */
export function resolveImageModel(reference?: string): ImagesModel {
  const requested = reference?.trim()
  if (!requested) {
    const fallback = getImageModel('openrouter', DEFAULT_IMAGE_MODEL_REFERENCE)
    if (!fallback) {
      throw new Error(`default image model '${DEFAULT_IMAGE_MODEL_REFERENCE}' is missing from the generated catalog`)
    }
    return fallback
  }
  // `provider/route-id` where the prefix names a catalogued provider; route
  // ids themselves contain slashes (vendor/model), so only a known provider
  // prefix is split.
  const slash = requested.indexOf('/')
  if (slash > 0) {
    const prefix = requested.slice(0, slash).toLowerCase()
    if (REGISTRY.has(prefix)) {
      const model = getImageModel(prefix, requested.slice(slash + 1))
      if (model) return model
    }
  }
  // Otherwise the whole reference is a route id searched across providers.
  for (const provider of getImageProviders()) {
    const model = getImageModel(provider, requested)
    if (model) return model
  }
  throw new Error(
    `unknown image model '${requested}'; available providers: ${getImageProviders().join(', ')}`,
  )
}

function normalizeProvider(provider: string): string {
  return provider.trim().toLowerCase().replaceAll('_', '-')
}
