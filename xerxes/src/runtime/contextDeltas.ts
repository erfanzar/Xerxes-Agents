// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Change-driven constraint deltas.
 *
 * When a session-level constraint (mode, model, permission, reasoning effort)
 * changes between turns, the change itself is the event: it is recorded once in
 * session metadata and rendered into exactly one subsequent request as a
 * volatile system layer — instead of restating static boilerplate on every
 * request. Unchanged turns contribute nothing, which keeps the stable prefix
 * byte-identical for the provider's cache.
 *
 * State lives under one bounded metadata key rather than in the message array:
 * deltas are context bookkeeping, never chat history, so they must not reach a
 * provider as messages or survive into exports as transcript rows.
 */

export type ContextDeltaLayer = 'interaction-mode' | 'model' | 'permission' | 'reasoning'

/** One recorded constraint change, ordered by recording time. */
export interface ContextDelta {
  readonly at: number
  readonly layer: ContextDeltaLayer
  readonly value: string
}

const CONTEXT_DELTAS_KEY = 'context_deltas'

/**
 * Ring ceiling. Deltas drain at the next turn assembly, so the realistic
 * backlog is a handful of slash commands fired while idle; the cap only exists
 * so an idle session hammered by RPC cannot grow metadata without bound. The
 * oldest entries drop first — the newest constraint state is what matters.
 */
export const MAX_CONTEXT_DELTAS = 16

/** Read the pending deltas without consuming them (metadata is shared state). */
export function readContextDeltas(metadata: Readonly<Record<string, unknown>>): readonly ContextDelta[] {
  const raw = metadata[CONTEXT_DELTAS_KEY]
  if (!Array.isArray(raw)) return []
  return raw.filter(isContextDelta)
}

/** Append one delta, dropping the oldest beyond the ring cap. */
export function appendContextDelta(metadata: Record<string, unknown>, delta: ContextDelta): void {
  const pending = [...readContextDeltas(metadata), delta]
  metadata[CONTEXT_DELTAS_KEY] = pending.slice(-MAX_CONTEXT_DELTAS)
}

/**
 * Merge independently mutated delta queues without replaying the same change.
 *
 * A daemon mode RPC may append to live session metadata while the active turn
 * owns a snapshot. End-of-turn synchronization must preserve both queues. The
 * tuple is the durable identity for the current schema; duplicate identical
 * changes recorded in the same millisecond are observationally equivalent.
 */
export function mergeContextDeltas(
  first: Readonly<Record<string, unknown>>,
  second: Readonly<Record<string, unknown>>,
): readonly ContextDelta[] {
  const merged = [...readContextDeltas(first), ...readContextDeltas(second)]
  const unique = new Map<string, ContextDelta>()
  for (const delta of merged) {
    unique.set(`${delta.at}\u0000${delta.layer}\u0000${delta.value}`, delta)
  }
  return [...unique.values()]
    .sort((left, right) => left.at - right.at)
    .slice(-MAX_CONTEXT_DELTAS)
}

/**
 * Consume the backlog: render-ready deltas are removed from metadata so the
 * same change is injected exactly once. Call this at turn assembly; a turn that
 * aborts afterwards simply misses its notice — the constraint itself still
 * governs the request through its real parameter.
 */
export function takeContextDeltas(metadata: Record<string, unknown>): readonly ContextDelta[] {
  const drained = readContextDeltas(metadata)
  if (drained.length) delete metadata[CONTEXT_DELTAS_KEY]
  return drained
}

/** Human-readable label per layer, used verbatim in rendered lines. */
function layerLabel(layer: ContextDeltaLayer): string {
  switch (layer) {
    case 'interaction-mode':
      return 'interaction mode'
    case 'model':
      return 'model'
    case 'permission':
      return 'permission mode'
    case 'reasoning':
      return 'reasoning effort'
  }
}

/**
 * Render drained deltas as the volatile `[Context updated]` block.
 *
 * One line per change, newest last. Empty in, empty out: no deltas means no
 * layer, and the assembled prompt stays byte-identical to a session with none.
 */
export function renderContextDeltas(deltas: readonly ContextDelta[]): string {
  if (!deltas.length) return ''
  const lines = deltas.map(delta => `- ${layerLabel(delta.layer)}: ${delta.value}`)
  return ['[Context updated]', ...lines].join('\n')
}

/** Build the delta for a real change, or undefined when the value did not move. */
export function contextDeltaFor(
  previous: string | undefined,
  next: string,
  now: number,
  layer: ContextDeltaLayer,
): ContextDelta | undefined {
  if (previous === next) return undefined
  return { at: now, layer, value: next }
}

function isContextDelta(value: unknown): value is ContextDelta {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as Record<string, unknown>
  return typeof candidate.at === 'number'
    && Number.isFinite(candidate.at)
    && typeof candidate.value === 'string'
    && (candidate.layer === 'interaction-mode'
      || candidate.layer === 'model'
      || candidate.layer === 'permission'
      || candidate.layer === 'reasoning')
}
