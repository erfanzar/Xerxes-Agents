// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export type TokenCountInput = string | readonly Record<string, unknown>[]

/** Provider-aware token estimator with a deterministic dependency-free fallback. */
export class ProviderTokenCounter {
  static countTokensForProvider(input: TokenCountInput, provider?: string, model?: string): number {
    const text = typeof input === 'string' ? input : this.messagesToText(input)
    const resolved = provider ?? (model ? this.detectProvider(model) : undefined)
    const fallback = estimateTokens(text)
    return resolved === 'google' ? Math.ceil(fallback * 1.1) : fallback
  }

  static detectProvider(model: string): string | undefined {
    const normalized = model.toLowerCase()
    if (normalized.includes('gpt') || normalized.includes('o1')) return 'openai'
    if (normalized.includes('claude')) return 'anthropic'
    if (normalized.includes('gemini') || normalized.includes('palm')) return 'google'
    if (normalized.includes('llama')) return 'meta'
    if (normalized.includes('mistral') || normalized.includes('mixtral')) return 'mistral'
    return undefined
  }

  static messagesToText(messages: readonly Record<string, unknown>[]): string {
    return messages.map(message => {
      const role = typeof message.role === 'string' ? message.role : ''
      const parts = [`${role}: ${contentToText(message.content)}`]
      // tool_calls arguments are often the largest payload in a window; count every
      // additional token-bearing field instead of silently treating it as free.
      for (const [key, value] of Object.entries(message)) {
        if (key === 'role' || key === 'content' || value === undefined || value === null) {
          continue
        }
        const serialized = contentToText(value)
        if (serialized) {
          parts.push(`${key}=${serialized}`)
        }
      }
      return parts.join(' ')
    }).join('\n')
  }
}

export interface SmartTokenCounterOptions {
  readonly model?: string
  readonly provider?: string
}

export class SmartTokenCounter {
  readonly model: string | undefined
  readonly provider: string | undefined

  constructor(options: SmartTokenCounterOptions = {}) {
    this.model = options.model
    this.provider = options.provider ?? (options.model ? ProviderTokenCounter.detectProvider(options.model) : undefined)
  }

  countRemainingCapacity(input: TokenCountInput, maxTokens: number): number {
    return Math.max(0, maxTokens - this.countTokens(input))
  }

  countTokens(input: TokenCountInput): number {
    return ProviderTokenCounter.countTokensForProvider(input, this.provider, this.model)
  }

  estimateCompressionRatio(original: string, compressed: string): number {
    const originalTokens = this.countTokens(original)
    return originalTokens === 0 ? 0 : 1 - this.countTokens(compressed) / originalTokens
  }
}

function contentToText(value: unknown): string {
  if (typeof value === 'string') return value
  if (Array.isArray(value)) return value.map(contentToText).join(' ')
  if (value === undefined || value === null) return ''
  return JSON.stringify(value)
}

/**
 * CJK code-point ranges counted individually by the estimator: CJK punctuation
 * and symbols (U+3000-U+303F), Hiragana (U+3040-U+309F), Katakana
 * (U+30A0-U+30FF), Han unified ideographs Extension A (U+3400-U+4DBF) and the
 * URO (U+4E00-U+9FFF), Hangul syllables (U+AC00-U+D7AF), compatibility
 * ideographs (U+F900-U+FAFF), and fullwidth forms (U+FF00-U+FFEF).
 */
const CJK_CODE_POINT_PATTERN =
  /[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]/gu

/**
 * Estimated tokens per CJK code point.
 *
 * Word-splitting heuristics collapse an entire Han run into a single lexical
 * token while measured BPE tokenizers emit roughly 0.6-1.0 tokens per CJK
 * character; 3/4 sits in the middle of that band and keeps the math exact
 * enough to stay deterministic (`Math.ceil(cjkCodePoints * 0.75)`).
 */
const TOKENS_PER_CJK_CODE_POINT = 0.75

/**
 * Deterministic provider-independent token estimate.
 *
 * Formula:
 * 1. Baseline (legacy behavior, unchanged): the number of lexical tokens
 *    (word/number runs count one each, a run of punctuation/symbols counts
 *    once per run), floored at 1 and at `ceil(text.length / 4)`.
 * 2. Script-aware correction: each CJK code point (ranges documented on
 *    {@link CJK_CODE_POINT_PATTERN}) contributes ~0.75 tokens via
 *    `ceil(cjkCodePoints * 0.75)` because lexical splitting would otherwise
 *    charge a whole Han/Hangul/Kana paragraph as one token (~4x undercount).
 * 3. Result: `max(baseline, cjkContribution + nonCjkBaseline)` where
 *    nonCjkBaseline re-applies the baseline heuristic to the text with CJK
 *    code points removed. Inputs without CJK characters therefore keep their
 *    exact previous estimates; empty input still estimates 0.
 */
function estimateTokens(text: string): number {
  if (!text) return 0
  const baseline = heuristicEstimate(text)
  const cjkMatches = text.match(CJK_CODE_POINT_PATTERN)
  if (!cjkMatches?.length) {
    return baseline
  }
  const cjkContribution = Math.ceil(cjkMatches.length * TOKENS_PER_CJK_CODE_POINT)
  const nonCjkText = text.replace(CJK_CODE_POINT_PATTERN, '')
  return Math.max(baseline, cjkContribution + heuristicEstimate(nonCjkText))
}

/** Lexical-token plus chars-per-4 estimate for a single script run. */
function heuristicEstimate(text: string): number {
  if (!text) return 0
  // Words and numbers count individually while a run of punctuation/symbols counts once
  // per run. One token per non-space character over-counted punctuation-dense content
  // (code, JSON, diffs) by roughly 2x versus BPE merge behavior.
  const lexical = text.match(/\p{L}+[\p{L}\p{N}_-]*|\p{N}+|[^\s\p{L}\p{N}]+/gu)?.length ?? 0
  return Math.max(1, lexical, Math.ceil(text.length / 4))
}
