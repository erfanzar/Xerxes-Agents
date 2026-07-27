// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { repairToolMessageSequence } from './toolPairRepair.js'
import { pruneToolMessages } from './toolResultPruner.js'
import { SmartTokenCounter } from './tokenCounter.js'

export const COMPACTION_REFERENCE_PREFIX = '[CONTEXT COMPACTION — REFERENCE ONLY]'

/**
 * Typed flag identifying a summary this compressor wrote.
 *
 * Prefix sniffing alone misreads any message that merely quotes the marker —
 * a transcript discussing compaction, or a tool result echoing an earlier
 * summary — as a prior summary, and the iterative path then folds unrelated
 * text into the next summary and drops the head message carrying it.
 * Providers ignore message keys they do not model, so the flag rides along.
 */
export const COMPACTION_SUMMARY_MARKER = 'xerxes_compaction_summary'

export type ContextMessage = Record<string, unknown>
export type Summarizer = (messages: readonly ContextMessage[], budgetTokens: number) => string

export interface CompressionResult {
  readonly compressed: boolean
  readonly compressedCount: number
  readonly messages: ContextMessage[]
  readonly metadata: Readonly<Record<string, unknown>>
  readonly protectedFirst: number
  readonly protectedLast: number
  readonly prunedToolResults: number
  readonly summaryTokens: number
  readonly tokensAfter: number
  readonly tokensBefore: number
}

export interface ContextCompressorOptions {
  readonly contextWindow?: number
  /**
   * Skip the under-threshold early return so an explicit compaction still
   * summarizes the middle after tool-result pruning. Scheduled compaction
   * keeps the default prune-only behavior for contexts that already fit.
   */
  readonly forceSummarize?: boolean
  readonly model?: string
  readonly protectFirst?: number
  readonly protectLast?: number
  readonly summarizer?: Summarizer
  readonly summaryBudgetRatio?: number
  readonly summaryMaxTokens?: number
  readonly summaryMinTokens?: number
  readonly threshold?: number
  readonly tokenCounter?: SmartTokenCounter
}

/** Pre-prunes tool data then safely folds the unprotected middle into a reference-only summary. */
export class ContextCompressor {
  readonly contextWindow: number
  readonly forceSummarize: boolean
  readonly protectFirst: number
  readonly protectLast: number
  readonly summaryBudgetRatio: number
  readonly summaryMaxTokens: number
  readonly summaryMinTokens: number
  readonly threshold: number
  private readonly summarizer: Summarizer | undefined
  private readonly tokenCounter: SmartTokenCounter

  constructor(options: ContextCompressorOptions = {}) {
    this.threshold = options.threshold ?? 0.75
    if (this.threshold <= 0 || this.threshold > 1) throw new Error('threshold must be in (0.0, 1.0]')
    this.contextWindow = options.contextWindow ?? 200_000
    this.protectFirst = options.protectFirst ?? 3
    this.protectLast = options.protectLast ?? 6
    if (this.protectFirst < 0 || this.protectLast < 0) throw new Error('protectFirst and protectLast must be >= 0')
    this.summaryMinTokens = options.summaryMinTokens ?? 2_000
    this.summaryMaxTokens = options.summaryMaxTokens ?? 12_000
    this.summaryBudgetRatio = options.summaryBudgetRatio ?? 0.2
    this.forceSummarize = options.forceSummarize ?? false
    this.summarizer = options.summarizer
    this.tokenCounter = options.tokenCounter ?? new SmartTokenCounter({ model: options.model ?? 'gpt-4' })
  }

  compress(messages: readonly ContextMessage[]): CompressionResult {
    const tokensBefore = this.count(messages)
    if (messages.length === 0) return unchanged([], tokensBefore)
    const pruned = pruneToolMessages(messages, { protectLast: this.protectLast })
    const afterPrune = this.count(pruned.messages)
    if (afterPrune < this.thresholdTokens() && !this.forceSummarize) {
      if (pruned.prunedCount === 0) {
        // Already under threshold with nothing pruned: a scheduled compaction must not
        // lossily summarize a context that still fits the window.
        return unchanged([...pruned.messages], afterPrune)
      }
      return {
        messages: pruned.messages,
        compressed: true,
        tokensBefore,
        tokensAfter: afterPrune,
        protectedFirst: Math.min(this.protectFirst, pruned.messages.length),
        protectedLast: Math.min(this.protectLast, pruned.messages.length),
        compressedCount: 0,
        prunedToolResults: pruned.prunedCount,
        summaryTokens: 0,
        metadata: { strategy: 'prune-only' },
      }
    }

    const count = pruned.messages.length
    const headCount = Math.min(this.protectFirst, count)
    const tailCount = Math.min(this.protectLast, Math.max(0, count - headCount))
    let head = pruned.messages.slice(0, headCount)
    const tail = tailCount ? pruned.messages.slice(count - tailCount) : []
    let middle = tailCount ? pruned.messages.slice(headCount, count - tailCount) : pruned.messages.slice(headCount)
    if (!middle.length) {
      return {
        messages: pruned.messages,
        compressed: pruned.prunedCount > 0,
        tokensBefore,
        tokensAfter: afterPrune,
        protectedFirst: headCount,
        protectedLast: tailCount,
        compressedCount: 0,
        prunedToolResults: pruned.prunedCount,
        summaryTokens: 0,
        metadata: { strategy: 'no-middle' },
      }
    }
    if (!this.summarizer) {
      return {
        messages: pruned.messages,
        compressed: pruned.prunedCount > 0,
        tokensBefore,
        tokensAfter: afterPrune,
        protectedFirst: headCount,
        protectedLast: tailCount,
        compressedCount: 0,
        prunedToolResults: pruned.prunedCount,
        summaryTokens: 0,
        metadata: { strategy: 'no-summary-agent' },
      }
    }
    let prior: string | undefined
    let priorFromHead = false
    const headPrior = priorSummaryText(head.at(-1))
    const middlePrior = priorSummaryText(middle[0])
    if (headPrior !== undefined) {
      prior = headPrior
      head = head.slice(0, -1)
      priorFromHead = true
    } else if (middlePrior !== undefined) {
      prior = middlePrior
      middle = middle.slice(1)
    }
    const budget = this.summaryBudget(this.count(middle))
    const wrapped = wrapSummary(prior, this.summarizer(middle, budget))
    const summaryMessage: ContextMessage = { role: 'user', content: wrapped, [COMPACTION_SUMMARY_MARKER]: true }
    // Cutting the middle can leave the head ending in assistant tool_calls whose results
    // were summarized away, or the tail beginning with orphan tool results. Repair the
    // window here so every caller receives a provider-safe sequence.
    const output = repairToolMessageSequence([...head, summaryMessage, ...tail])
    return {
      messages: output,
      compressed: true,
      tokensBefore,
      tokensAfter: this.count(output),
      protectedFirst: head.length + (priorFromHead ? 1 : 0),
      protectedLast: tailCount,
      compressedCount: middle.length,
      prunedToolResults: pruned.prunedCount,
      summaryTokens: this.tokenCounter.countTokens(wrapped),
      metadata: { strategy: prior ? 'iterative' : 'first-pass' },
    }
  }

  shouldCompact(messages: readonly ContextMessage[]): boolean {
    return this.count(messages) >= this.thresholdTokens()
  }

  thresholdTokens(): number {
    return Math.floor(this.contextWindow * this.threshold)
  }

  private count(messages: readonly ContextMessage[]): number {
    return this.tokenCounter.countTokens(messages)
  }

  private summaryBudget(compressedTokens: number): number {
    return Math.min(this.summaryMaxTokens, Math.max(this.summaryMinTokens, Math.floor(compressedTokens * this.summaryBudgetRatio)))
  }
}

/** Deterministic test/dev summarizer that retains one readable line per message. */
export function naiveSummarizer(messages: readonly ContextMessage[], _budgetTokens: number): string {
  return messages.flatMap(message => {
    const content = contentToText(message.content).trim()
    const firstLine = content.split(/\r?\n/, 1)[0] ?? ''
    return firstLine ? [`- ${typeof message.role === 'string' ? message.role : '?'}: ${firstLine.slice(0, 200)}${firstLine.length > 200 ? '…' : ''}`] : []
  }).join('\n')
}

function contentToText(value: unknown): string {
  if (typeof value === 'string') return value
  if (Array.isArray(value)) return value.map(contentToText).join(' ')
  return value === undefined || value === null ? '' : JSON.stringify(value)
}

/**
 * True for a summary this compressor wrote.
 *
 * The prefix is the fallback, not the primary test: transcripts persisted
 * before the typed flag existed — and reloads that rebuild messages from a
 * fixed set of provider fields — carry the prefix and nothing else.
 */
export function isCompactionSummaryMessage(message: ContextMessage | undefined): boolean {
  if (message === undefined) return false
  if (message[COMPACTION_SUMMARY_MARKER] === true) return true
  // The fallback is deliberately narrow: only the user message this compressor
  // writes summaries as. A tool result or assistant turn that merely opens with
  // the marker — quoting an earlier summary — is content, not a summary, and
  // folding it into the iterative path would drop the message carrying it.
  return message.role === 'user'
    && typeof message.content === 'string'
    && message.content.startsWith(COMPACTION_REFERENCE_PREFIX)
}

/** Summary text of a prior summary message, or undefined when the message is not one. */
function priorSummaryText(message: ContextMessage | undefined): string | undefined {
  if (!isCompactionSummaryMessage(message)) return undefined
  return typeof message?.content === 'string' ? message.content : undefined
}

function unchanged(messages: ContextMessage[], tokens: number): CompressionResult {
  return {
    messages,
    compressed: false,
    tokensBefore: tokens,
    tokensAfter: tokens,
    protectedFirst: 0,
    protectedLast: 0,
    compressedCount: 0,
    prunedToolResults: 0,
    summaryTokens: 0,
    metadata: {},
  }
}

function wrapSummary(prior: string | undefined, summary: string): string {
  const body = prior ? `${prior.trim()}\n\n---\n\n${summary.trim()}` : summary.trim()
  return `${COMPACTION_REFERENCE_PREFIX}\n\n${body}`
}
