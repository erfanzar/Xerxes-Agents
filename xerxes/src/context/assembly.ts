// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SystemPromptSegment } from '../streaming/promptCaching.js'

/**
 * The layered prompt assembler.
 *
 * Every text the daemon contributes to a request enters through one declared,
 * ordered pipeline here, so these properties hold by construction instead of
 * by convention:
 *
 * 1. **A stable system prompt** — every system layer is fixed for the life of
 *    a session (memory is a snapshot taken when the session starts). Providers
 *    cache by prefix and the system prompt comes before the conversation, so
 *    a single changed byte in it made every provider — Anthropic, OpenAI,
 *    Kimi, DeepSeek, Claude Code — re-read and re-bill the whole conversation
 *    on the next turn. Ordering "volatile" layers last in the system prompt
 *    did not help: the conversation still sat behind them.
 * 2. **Per-turn context rides with the message** — what changes between turns
 *    (memory written since the snapshot, the goal's round and status,
 *    instruction-file edits, constraint deltas, recovered subagents) is rendered by {@link renderTurnContext} and
 *    prepended to that turn's user message, where it stays: later requests
 *    replay it byte for byte, so it is cached like any other message.
 * 3. **Provenance** — each layer is named and individually digestible
 *    ({@link layerDigests}), which is what makes "why did this turn behave
 *    differently?" answerable after the fact.
 */

/** Named system-prompt inputs, in assembly order. Each is fixed for the session. */
export interface ContextAssemblyInput {
  /** Workspace bootstrap preamble (identity, cwd, environment). */
  readonly bootstrap: string
  /** The selected agent persona's own system prompt. */
  readonly agentPrompt: string
  /** Per-tool usage-policy sections for exactly the visible surface. */
  readonly toolGuidance: string
  /** Goal policy, when goal tools are visible (the goal's status is per-turn). */
  readonly goalPolicy?: string
  /** Names of tools that exist but are not in this request's schemas. */
  readonly deferredCatalog?: string
  /** Interaction-mode switch hint for the current mode. */
  readonly modeHint: string
  /** Subagent join contract when a coordinator is attached. */
  readonly subagentJoin: string
  /** That long conversations are summarized, when the host compacts automatically. */
  readonly compaction?: string
  /** Persistent memory as it stood when the session started. */
  readonly memory?: string
  /** Agent self-memory as it stood when the session started. */
  readonly selfMemory?: string
  /** Operator/session addendum (set when the session opens). */
  readonly addendum?: string
}

/**
 * Assemble the request's system-prompt segments from named layers.
 *
 * Empty contributions drop out; callers never hand-assemble an array, so the
 * layer set is closed under this module's tests rather than open under edit
 * accidents.
 */
export function assembleContextLayers(input: ContextAssemblyInput): SystemPromptSegment[] {
  return [
    { name: 'bootstrap', text: input.bootstrap },
    { name: 'agent', text: input.agentPrompt },
    { name: 'tool_guidance', text: input.toolGuidance },
    { name: 'goal_policy', text: input.goalPolicy ?? '' },
    { name: 'deferred_catalog', text: input.deferredCatalog ?? '' },
    { name: 'mode_hint', text: input.modeHint },
    { name: 'subagent_join', text: input.subagentJoin },
    { name: 'compaction', text: input.compaction ?? '' },
    { name: 'memory', text: input.memory ?? '' },
    { name: 'self_memory', text: input.selfMemory ?? '' },
    { name: 'addendum', text: input.addendum ?? '' },
  ].filter(segment => segment.text !== '')
}

/** What changed since the previous turn, delivered with this turn's message. */
export interface TurnContextInput {
  /** The local date, when the day changed since the prompt was dated. */
  readonly dateChange?: string
  /** The goal's status, when it changed since the model was last told. */
  readonly goalStatus?: string
  /** Recovery notice for subagents found in a resumed transcript. */
  readonly recoveredSubagents?: string
  /** Memory files written or changed since the session's memory snapshot. */
  readonly memoryChanges?: string
  /** Self-memory text, when it changed since it was last delivered. */
  readonly selfMemoryChanges?: string
  /** Change-driven constraint deltas emitted since the previous turn. */
  readonly contextDeltas?: string
  /** Fresh content of instruction files that changed since the last turn. */
  readonly instructionUpdates?: string
}

export function assembleTurnContext(input: TurnContextInput): SystemPromptSegment[] {
  return [
    { name: 'date_change', text: input.dateChange ?? '' },
    { name: 'recovered_subagents', text: input.recoveredSubagents ?? '' },
    { name: 'goal_status', text: input.goalStatus ?? '' },
    { name: 'instruction_updates', text: input.instructionUpdates ?? '' },
    { name: 'context_deltas', text: input.contextDeltas ?? '' },
    { name: 'memory_changes', text: input.memoryChanges ?? '' },
    { name: 'self_memory_changes', text: input.selfMemoryChanges ?? '' },
  ].filter(segment => segment.text.trim() !== '')
}

/**
 * The block prepended to a turn's user message, or '' when nothing changed.
 * It says who wrote it: the runtime, carrying the same authority as the
 * system prompt it updates, while fenced memory inside stays data.
 */
export function renderTurnContext(segments: readonly SystemPromptSegment[]): string {
  if (!segments.length) return ''
  return [
    '<turn-context>',
    'Xerxes attached this to the message below; the user did not type it. It updates your instructions and memory since they were loaded, with the same standing as the system prompt.',
    '',
    segments.map(segment => segment.text.trim()).join('\n\n'),
    '</turn-context>',
  ].join('\n')
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
