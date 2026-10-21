// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { SystemPromptSegment } from '../streaming/promptCaching.js'

export interface PromptSize {
  /** JavaScript string length (UTF-16 code units), not encoded byte length. */
  readonly chars: number
  /** A tokenizer-independent chars / 4 heuristic, not provider usage or a hard context limit. */
  readonly estimatedTokens: number
}

export interface LayerMeasurement extends PromptSize {
  readonly name: string
}

export interface PromptMeasurement {
  readonly layers: readonly LayerMeasurement[]
  /** Includes the two newlines between assembled system layers. */
  readonly total: PromptSize
}

/** Advisory caps in estimated tokens; omitted options inherit the defaults. */
export interface PromptBudgetOptions {
  readonly total?: number
  readonly layers?: Readonly<Record<string, number>>
}

export interface PromptBudgetOverage extends PromptSize {
  /** The named system layer, or `total` for the entire always-on prompt. */
  readonly layer: string
  /** The advisory cap in estimated tokens (the same units as estimatedTokens). */
  readonly budget: number
  readonly reason: string
}

export interface PromptBudgetReport extends PromptMeasurement {
  readonly ok: boolean
  readonly overages: readonly PromptBudgetOverage[]
}

export const DEFAULT_PROMPT_BUDGETS: {
  readonly total: number
  readonly layers: Readonly<Record<string, number>>
} = Object.freeze({
  // 24K leaves most of a typical 128K window for conversation, tool results
  // and output instead of charging an ever-growing fixed cost on every turn.
  total: 24_000,
  layers: Object.freeze({
    // Bootstrap includes workspace instructions and skill discovery, so it
    // gets half the total but should not crowd out the task itself.
    bootstrap: 12_000,
    // Usage policy should remain concise; schemas already describe the tools.
    tool_guidance: 6_000,
    // Snapshot memory should be selective rather than a second conversation.
    memory: 4_000,
    // Deferred tools are discoverable on demand; this is only their index.
    deferred_catalog: 2_000,
  }),
})

const BUDGET_REASONS: Readonly<Record<string, string>> = Object.freeze({
  total: 'The always-on prompt needs to leave room for conversation, tool results and the model reply on every turn.',
  bootstrap: 'Workspace instructions and skill discovery should orient the model without crowding out the current task.',
  tool_guidance: 'Always-on tool guidance should stay concise because tool schemas already carry their detailed contracts.',
  memory: 'Snapshot memory should supply selective durable facts rather than consume the space needed for the current conversation.',
  deferred_catalog: 'The deferred tool catalog is a discovery index; full tool details can be loaded on demand instead of repeated every turn.',
})

/** Read-only accounting: preserve all whitespace and never alter a segment. */
export function measureLayers(segments: readonly SystemPromptSegment[]): PromptMeasurement {
  const layers = segments.map(segment => ({ name: segment.name, ...sizeOf(segment.text.length) }))
  const chars = layers.reduce((sum, layer) => sum + layer.chars, 0) + Math.max(0, segments.length - 1) * 2
  return { layers, total: sizeOf(chars) }
}

/** Report every exceeded advisory cap without refusing, trimming or truncating any content. */
export function evaluateBudget(
  measure: PromptMeasurement,
  options: PromptBudgetOptions = {},
): PromptBudgetReport {
  const totalBudget = options.total ?? DEFAULT_PROMPT_BUDGETS.total
  const layerBudgets = { ...DEFAULT_PROMPT_BUDGETS.layers, ...options.layers }
  validateBudget(totalBudget, 'total')
  for (const [name, budget] of Object.entries(layerBudgets)) validateBudget(budget, name)
  const overages: PromptBudgetOverage[] = []
  for (const layer of measure.layers) {
    const budget = Object.hasOwn(layerBudgets, layer.name) ? layerBudgets[layer.name] : undefined
    if (budget !== undefined && layer.estimatedTokens > budget) {
      overages.push({
        layer: layer.name,
        chars: layer.chars,
        estimatedTokens: layer.estimatedTokens,
        budget,
        reason: (Object.hasOwn(BUDGET_REASONS, layer.name) ? BUDGET_REASONS[layer.name] : undefined)
          ?? `The always-on ${layer.name} layer should stay within its configured share so other context has room on every turn.`,
      })
    }
  }
  if (measure.total.estimatedTokens > totalBudget) {
    overages.push({ layer: 'total', ...measure.total, budget: totalBudget, reason: BUDGET_REASONS.total! })
  }
  return { ok: overages.length === 0, total: measure.total, layers: measure.layers, overages }
}

function sizeOf(chars: number): PromptSize {
  return { chars, estimatedTokens: Math.ceil(chars / 4) }
}

function validateBudget(value: number, layer: string): void {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`Prompt budget for '${layer}' must be a finite, non-negative estimated token count`)
  }
}
