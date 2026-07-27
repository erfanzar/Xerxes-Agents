// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Providers whose request protocol supports Anthropic-style prompt caching. */
export const SUPPORTS_CACHING = Object.freeze(['anthropic'] as const)

/** Provider-facing marker for an ephemeral Anthropic cache breakpoint. */
export const EPHEMERAL_CACHE_CONTROL = Object.freeze({ type: 'ephemeral' as const })

export interface CachedSystemTextBlock {
  /** Absent on a volatile block: marking one would rewrite the cache every turn. */
  readonly cache_control?: typeof EPHEMERAL_CACHE_CONTROL
  readonly text: string
  readonly type: 'text'
}

export type CachedSystemPrompt = string | readonly CachedSystemTextBlock[]

export type CacheableToolSchema = Readonly<Record<string, unknown>>

/** Separator the daemon has always used between system-prompt contributions. */
const SEGMENT_SEPARATOR = '\n\n'

/** How many recently joined prompts keep their segment structure available. */
const SEGMENT_REGISTRY_LIMIT = 8

/** One named contribution to the system prompt, in caller-assembled order. */
export interface SystemPromptSegment {
  /** Diagnostic label for the contributing source, e.g. `bootstrap` or `memory`. */
  readonly name: string
  readonly text: string
  /**
   * True when the text can differ between two turns of the same session:
   * agent memory, self-memory, and per-turn notices. Volatile segments are
   * moved behind the cache breakpoint, because a single byte of drift inside
   * a cached block invalidates that block and everything after it.
   */
  readonly volatile?: boolean
}

/**
 * Convert a non-empty system prompt to an Anthropic cacheable content block.
 *
 * An empty prompt stays a string so callers retain the provider's ordinary
 * request shape when there is no stable prefix to cache.
 *
 * This whole-string form caches only as well as its most volatile byte;
 * prefer {@link wrapSystemSegmentsWithCache} when the caller knows which
 * sources are stable.
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
 * Split ordered system segments into a cached stable prefix and an uncached
 * volatile tail.
 *
 * The shipped prompt tells the agent to write to its memory before ending a
 * substantive turn, and that memory section used to sit inside the single
 * cached block — so obeying our own instructions invalidated the entire
 * system prefix on the very next request. Emitting the mutable sources as a
 * separate trailing block keeps the breakpoint on bytes that survive the turn.
 *
 * The tail block carries the separator so the concatenated block texts are
 * byte-identical to {@link joinSystemSegments} of the same input: moving a
 * caller onto segments changes cache boundaries only, never what the model
 * reads.
 */
export function wrapSystemSegmentsWithCache(
  segments: readonly SystemPromptSegment[],
): CachedSystemPrompt {
  const ordered = orderSystemSegments(segments)
  if (!ordered.length) {
    return ''
  }
  const prefix = joinTexts(ordered.filter(segment => !segment.volatile))
  const tail = joinTexts(ordered.filter(segment => segment.volatile))
  if (!prefix) {
    // Nothing outlives the turn, so a breakpoint here would only pay the
    // cache-write surcharge on every request and never produce a hit.
    return [{ type: 'text', text: tail }]
  }
  const prefixBlock: CachedSystemTextBlock = {
    type: 'text',
    text: prefix,
    cache_control: EPHEMERAL_CACHE_CONTROL,
  }
  if (!tail) {
    return [prefixBlock]
  }
  return [prefixBlock, { type: 'text', text: `${SEGMENT_SEPARATOR}${tail}` }]
}

/**
 * Join ordered system segments into the single string the rest of the request
 * path still transports, and remember the structure for
 * {@link cacheableSystemPrompt}.
 *
 * `CompletionRequest` carries the system prompt as a plain string, so segment
 * structure cannot reach the provider adapter by value yet. Registering the
 * exact joined text lets the Anthropic adapter recover it without any wire
 * marker that a non-caching provider would have to strip — and a registry miss
 * degrades to the previous single-block behavior rather than to a broken
 * prompt.
 */
export function joinSystemSegments(segments: readonly SystemPromptSegment[]): string {
  const ordered = orderSystemSegments(segments)
  const joined = joinTexts(ordered)
  if (!joined) {
    return ''
  }
  // Re-inserting refreshes recency so the prompts of an actively used session
  // are not evicted by a burst of one-off sessions.
  segmentRegistry.delete(joined)
  segmentRegistry.set(joined, ordered)
  for (const key of segmentRegistry.keys()) {
    if (segmentRegistry.size <= SEGMENT_REGISTRY_LIMIT) {
      break
    }
    segmentRegistry.delete(key)
  }
  return joined
}

/**
 * Resolve the cache shape for a system prompt, using registered segments when
 * the text came from {@link joinSystemSegments} and falling back to the
 * whole-string breakpoint otherwise.
 */
export function cacheableSystemPrompt(systemText: string): CachedSystemPrompt {
  const segments = segmentRegistry.get(systemText)
  return segments ? wrapSystemSegmentsWithCache(segments) : wrapSystemWithCache(systemText)
}

/**
 * Order segments for the wire: stable sources first in caller order, then
 * volatile ones. Empty contributions drop out, mirroring the `filter(Boolean)`
 * the daemon has always applied.
 */
export function orderSystemSegments(
  segments: readonly SystemPromptSegment[],
): readonly SystemPromptSegment[] {
  const present = segments.filter(segment => segment.text !== '')
  return [...present.filter(segment => !segment.volatile), ...present.filter(segment => segment.volatile)]
}

const segmentRegistry = new Map<string, readonly SystemPromptSegment[]>()

function joinTexts(segments: readonly SystemPromptSegment[]): string {
  return segments.map(segment => segment.text).join(SEGMENT_SEPARATOR)
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

/**
 * Attach the conversation's cache breakpoint to its final message.
 *
 * The system prompt and tool schemas were already cached, but they are the part
 * that does not grow: on a long session the ~5KB prelude was cached while the
 * transcript that dwarfs it was re-sent at full input price on every request.
 * Anthropic caches everything *before* a breakpoint, so one marker on the last
 * message covers the entire history.
 *
 * Markers on earlier messages are stripped for the same reason
 * {@link wrapToolsWithCache} strips stale tool markers: a resumed transcript
 * that already carries one would otherwise spend a second of the four
 * breakpoints an account is allowed.
 */
export function markLastMessageForCache(
  messages: readonly AnthropicCacheableMessage[],
): readonly AnthropicCacheableMessage[] {
  if (!messages.length) return messages
  return messages.map((message, index) => {
    const content = index === messages.length - 1
      // A breakpoint attaches to a content block, so a plain string body has to
      // be promoted to one first. Anthropic treats the two forms identically.
      ? markLastBlock(typeof message.content === 'string'
        ? [{ type: 'text', text: message.content }]
        : message.content)
      : stripCacheControl(message.content)
    return { ...message, content } as AnthropicCacheableMessage
  })
}

/** A message whose content may carry per-block cache markers. */
export interface AnthropicCacheableMessage {
  readonly content: string | readonly Readonly<Record<string, unknown>>[]
  readonly role: string
}

function markLastBlock(
  blocks: readonly Readonly<Record<string, unknown>>[],
): readonly Readonly<Record<string, unknown>>[] {
  if (!blocks.length) return blocks
  return blocks.map((block, index) => {
    const { cache_control: _cacheControl, ...rest } = block
    return index === blocks.length - 1
      ? { ...rest, cache_control: EPHEMERAL_CACHE_CONTROL }
      : rest
  })
}

function stripCacheControl(
  content: string | readonly Readonly<Record<string, unknown>>[],
): string | readonly Readonly<Record<string, unknown>>[] {
  if (typeof content === 'string') return content
  return content.map(block => {
    const { cache_control: _cacheControl, ...rest } = block
    return rest
  })
}
