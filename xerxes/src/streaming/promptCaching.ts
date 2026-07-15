// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Providers whose request protocol supports Anthropic-style prompt caching. */
export const SUPPORTS_CACHING = Object.freeze(['anthropic'] as const)

/** Provider-facing marker for an ephemeral Anthropic cache breakpoint. */
export const EPHEMERAL_CACHE_CONTROL = Object.freeze({ type: 'ephemeral' as const })

export interface CachedSystemTextBlock {
  readonly cache_control: typeof EPHEMERAL_CACHE_CONTROL
  readonly text: string
  readonly type: 'text'
}

export type CachedSystemPrompt = string | readonly CachedSystemTextBlock[]

export type CacheableToolSchema = Readonly<Record<string, unknown>>

/**
 * Convert a non-empty system prompt to an Anthropic cacheable content block.
 *
 * An empty prompt stays a string so callers retain the provider's ordinary
 * request shape when there is no stable prefix to cache.
 */
export function wrapSystemWithCache(systemText: string): CachedSystemPrompt {
  if (!systemText) {
    return ''
  }
  return [{
    type: 'text',
    text: systemText,
    cache_control: EPHEMERAL_CACHE_CONTROL,
  }]
}

/**
 * Mark the tail of a tool-definition block as an Anthropic cache breakpoint.
 *
 * The returned schemas are copies. Existing markers on earlier schemas are
 * removed, while the final schema receives the sole ephemeral marker.
 */
export function wrapToolsWithCache(
  toolSchemas: readonly CacheableToolSchema[],
): readonly CacheableToolSchema[] {
  if (!toolSchemas.length) {
    return toolSchemas
  }

  return toolSchemas.map((tool, index) => {
    const { cache_control: _cacheControl, ...withoutCacheControl } = tool
    if (index !== toolSchemas.length - 1) {
      return withoutCacheControl
    }
    return { ...withoutCacheControl, cache_control: EPHEMERAL_CACHE_CONTROL }
  })
}

/**
 * Read Anthropic cache usage counters from an SDK usage object or JSON record.
 *
 * Missing or non-finite counters are treated as zero so optional provider
 * usage does not destabilize cost accounting.
 */
export function extractCacheTokens(usage: unknown): readonly [number, number] {
  return [
    cacheTokenAt(usage, 'cache_read_input_tokens'),
    cacheTokenAt(usage, 'cache_creation_input_tokens'),
  ]
}

function cacheTokenAt(usage: unknown, name: string): number {
  if ((typeof usage !== 'object' && typeof usage !== 'function') || usage === null) {
    return 0
  }
  const value = (usage as Record<string, unknown>)[name]
  return finiteInteger(value)
}

function finiteInteger(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.trunc(value)
  }
  if (typeof value !== 'string' || !/^[+-]?\d+$/.test(value)) {
    return 0
  }
  const parsed = Number(value)
  return Number.isSafeInteger(parsed) ? parsed : 0
}
