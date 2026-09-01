// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Image-generation entry point (pi-ai `images.ts` parity): importing this
 * module registers the built-in wire APIs, mirroring Pi's side-effecting
 * `register-builtins` import, and re-exports the dispatch surface.
 */

import { generateImagesViaOpenRouter, OPENROUTER_IMAGES_API } from './openrouterImages.js'
import { registerImagesApiProvider } from './registry.js'

export * from './imageModels.js'
export * from './openrouterImages.js'
export * from './registry.js'
export * from './types.js'

registerImagesApiProvider({
  api: OPENROUTER_IMAGES_API,
  generateImages: generateImagesViaOpenRouter,
})
