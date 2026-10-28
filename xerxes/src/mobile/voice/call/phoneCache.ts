// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { tierDownloadBytes, VOICE_MODEL_TIERS, VOICE_MODELS, voiceModelFileUrlPath, type VoiceModelTier } from '../../../client/voiceModelManifest.js'

/** How much of a tier's pinned model bytes this browser already holds. Display-only. */
export interface PhoneCacheStatus {
  readonly cachedBytes: number
  readonly totalBytes: number
}
export type PhoneCacheStorage = Pick<CacheStorage, 'match'>

/** transformers.js caches every model file here, keyed by its full remote URL. */
export const TRANSFORMERS_CACHE_NAME = 'transformers-cache'
/** kokoro-js caches its voice tensors here, keyed by the upstream `main` URL it hard-codes. */
export const KOKORO_VOICES_CACHE_NAME = 'kokoro-voices'
const KOKORO_VOICE_URL_PREFIX = 'https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/'

async function cached(storage: PhoneCacheStorage, url: string, cacheName: string): Promise<boolean> {
  try { return (await storage.match(url, { cacheName })) !== undefined } catch { return false }
}

/**
 * Sums the manifest sizes of the tier's files already in the browser caches. A missing
 * CacheStorage (insecure context, old browser) or a failing lookup counts as not cached.
 */
export async function cachedTierBytes(tier: VoiceModelTier, origin: string, cacheStorage: PhoneCacheStorage | undefined): Promise<PhoneCacheStatus> {
  const totalBytes = tierDownloadBytes(tier)
  if (!cacheStorage) return { cachedBytes: 0, totalBytes }
  const base = origin.replace(/\/+$/u, '')
  const checks = VOICE_MODEL_TIERS[tier].flatMap(id => {
    const model = VOICE_MODELS[id]
    return model.files.map(async file => {
      const hit = await cached(cacheStorage, `${base}${voiceModelFileUrlPath(model, file.path)}`, TRANSFORMERS_CACHE_NAME)
        || (model.repo === VOICE_MODELS.tts.repo && file.path.startsWith('voices/')
          && await cached(cacheStorage, `${KOKORO_VOICE_URL_PREFIX}${file.path}`, KOKORO_VOICES_CACHE_NAME))
      return hit ? file.size : 0
    })
  })
  const cachedBytes = (await Promise.all(checks)).reduce((sum, size) => sum + size, 0)
  return { cachedBytes, totalBytes }
}
