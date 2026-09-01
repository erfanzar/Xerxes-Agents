// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ProviderError } from '../core/errors.js'
import type {
  AssistantImages,
  ImagesApi,
  ImagesApiProvider,
  ImagesContext,
  ImagesModel,
  ImagesOptions,
} from './types.js'

/**
 * Registry of image wire-API implementations (pi-ai `images-api-registry.ts`).
 *
 * One provider per `api` name; registration validates that a provider only
 * ever serves its own api, exactly like Pi's `wrapGenerateImages` guard.
 */
const providers = new Map<ImagesApi, ImagesApiProvider>()

function guardApi(provider: ImagesApiProvider): ImagesApiProvider {
  const { api, generateImages } = provider
  return {
    api,
    // Async so the api-mismatch guard rejects rather than throwing across
    // the caller's frame — every invocation yields a promise.
    generateImages: async (model, context, options) => {
      if (model.api !== api) {
        throw new ProviderError('images', `mismatched api: ${model.api} expected ${api}`)
      }
      return generateImages(model, context, options)
    },
  }
}

/** Register (or replace) the implementation for one image wire API. */
export function registerImagesApiProvider(provider: ImagesApiProvider): void {
  providers.set(provider.api, guardApi(provider))
}

export function getImagesApiProvider(api: ImagesApi): ImagesApiProvider | undefined {
  return providers.get(api)
}

/**
 * Dispatch one image-generation request to the implementation registered for
 * `model.api`. Throws when nothing is registered — callers see a typed
 * ProviderError, not a silent failure (pi-ai raises the same way).
 */
export async function generateImages(
  model: ImagesModel,
  context: ImagesContext,
  options?: ImagesOptions,
): Promise<AssistantImages> {
  const provider = providers.get(model.api)
  if (!provider) {
    throw new ProviderError('images', `no image API provider registered for api: ${model.api}`)
  }
  return provider.generateImages(model, context, options)
}
