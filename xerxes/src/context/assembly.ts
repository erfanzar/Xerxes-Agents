// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SystemPromptSegment } from '../streaming/promptCaching.js'

/**
 * The layered system-prompt assembler.
 *
 * Every text the daemon contributes to a request's system prompt enters through
 * one declared, ordered pipeline here, so three properties hold by construction
 * instead of by convention:
 *
 * 1. **Purity** — identical inputs assemble byte-identical segments, which is
 *    what keeps the provider's prefix cache alive across turns.
 * 2. **Stability ordering** — layers that survive a turn are emitted before
 *    layers rewritten every turn (`volatile`), so one drifting byte cannot
 *    invalidate cached bytes ahead of it.
 * 3. **Provenance** — each layer is named and individually digestible
 *    ({@link layerDigests}), which is what makes "why did this turn behave
 *    differently?" answerable after the fact.
 */

/** Named inputs to {@link assembleContextLayers}, in assembly order within each band. */
export interface ContextAssemblyInput {
  /** Stable: workspace bootstrap preamble (identity, cwd, environment). */
  readonly bootstrap: string
  /** Stable: the selected agent persona's own system prompt. */
  readonly agentPrompt: string
  /** Stable: per-tool usage-policy sections for exactly the visible surface. */
  readonly toolGuidance: string
  /** Stable: names of tools that exist but are not in this request's schemas. */
  readonly deferredCatalog?: string
  /** Stable: interaction-mode switch hint for the current mode. */
  readonly modeHint: string
  /** Stable: subagent join contract when a coordinator is attached. */
  readonly subagentJoin: string
  /** Volatile: recovery notice for subagents found in a resumed transcript. */
  readonly recoveredSubagents?: string
  /** Volatile: ranked memory recall for this turn's query. */
  readonly memoryRecall?: string
  /** Volatile: agent-self memory addendum. */
  readonly selfMemory?: string
  /** Volatile: change-driven constraint deltas emitted since the previous turn. */
  readonly contextDeltas?: string
  /** Volatile: operator/session addendum. */
  readonly addendum?: string
}

/**
 * Assemble the request's system-prompt segments from named layers.
 *
 * Empty contributions drop out; callers never hand-assemble an array, so the
 * layer set is closed under this module's tests rather than open under edit
 * accidents. Output order is stable-first then volatile-first-within-band,
 * matching what providers cache.
 */
export function assembleContextLayers(input: ContextAssemblyInput): SystemPromptSegment[] {
  return [
    { name: 'bootstrap', text: input.bootstrap },
    { name: 'agent', text: input.agentPrompt },
    { name: 'tool_guidance', text: input.toolGuidance },
    { name: 'deferred_catalog', text: input.deferredCatalog ?? '' },
    { name: 'mode_hint', text: input.modeHint },
    { name: 'subagent_join', text: input.subagentJoin },
    { name: 'recovered_subagents', text: input.recoveredSubagents ?? '', volatile: true },
    { name: 'memory', text: input.memoryRecall ?? '', volatile: true },
    { name: 'self_memory', text: input.selfMemory ?? '', volatile: true },
    { name: 'context_deltas', text: input.contextDeltas ?? '', volatile: true },
    { name: 'addendum', text: input.addendum ?? '', volatile: true },
  ].filter(segment => segment.text !== '')
}

/** One layer's provenance digest: short enough to log, long enough to diff. */
export interface LayerDigest {
  readonly hash: string
  readonly name: string
}

/**
 * Digest every assembled layer independently.
 *
 * Per-layer rather than whole-prompt hashing is the point: a whole-prompt hash
 * changes on any drift, while per-layer digests say *which* contribution moved
 * — the difference between "the prompt changed" and "the memory layer changed".
 * Truncated to 16 hex characters; collision space stays far beyond any session's
 * layer count.
 */
export function layerDigests(segments: readonly SystemPromptSegment[]): readonly LayerDigest[] {
  return segments.map(segment => ({ hash: shortSha256(segment.text), name: segment.name }))
}

function shortSha256(text: string): string {
  const hasher = new Bun.CryptoHasher('sha256')
  hasher.update(text)
  return hasher.digest('hex').slice(0, 16)
}

/**
 * How many recent assembly records a session keeps. Provenance answers
 * "which context generation produced this turn?", which is a recent-history
 * question; an unbounded log would belong in the transcript instead.
 */
export const MAX_ASSEMBLY_PROVENANCE_ENTRIES = 50

const ASSEMBLY_PROVENANCE_KEY = 'context_assembly'

/** One turn's assembled-layer fingerprint, recorded before the request fires. */
export interface AssemblyProvenance {
  readonly layers: readonly LayerDigest[]
  readonly recordedAt: number
  readonly turnId?: string
}

/** Read recorded provenance, oldest first, without consuming it. */
export function readAssemblyProvenance(
  metadata: Readonly<Record<string, unknown>>,
): readonly AssemblyProvenance[] {
  const raw = metadata[ASSEMBLY_PROVENANCE_KEY]
  if (!Array.isArray(raw)) return []
  return raw.filter(isAssemblyProvenance)
}

/**
 * Record this turn's layer digests in bounded session metadata.
 *
 * Stored per layer rather than as one whole-prompt hash on purpose: a
 * whole-prompt hash only says "something moved", while per-layer digests say
 * which contribution moved — memory versus mode hint versus addendum — which is
 * the difference between observing drift and diagnosing it.
 */
export function recordAssemblyProvenance(
  metadata: Record<string, unknown>,
  entry: AssemblyProvenance,
): void {
  const next = [...readAssemblyProvenance(metadata), entry].slice(-MAX_ASSEMBLY_PROVENANCE_ENTRIES)
  metadata[ASSEMBLY_PROVENANCE_KEY] = next
}

function isAssemblyProvenance(value: unknown): value is AssemblyProvenance {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as Record<string, unknown>
  return typeof candidate.recordedAt === 'number'
    && Array.isArray(candidate.layers)
    && candidate.layers.every(layer =>
      typeof layer === 'object' && layer !== null
      && typeof (layer as Record<string, unknown>).name === 'string'
      && typeof (layer as Record<string, unknown>).hash === 'string')
}
