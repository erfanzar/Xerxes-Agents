// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { pruneToolMessages } from './toolResultPruner.js'
import { SmartTokenCounter } from './tokenCounter.js'

export const COMPACTION_REFERENCE_PREFIX = '[CONTEXT COMPACTION — REFERENCE ONLY]'

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
    this.summarizer = options.summarizer
    this.tokenCounter = options.tokenCounter ?? new SmartTokenCounter({ model: options.model ?? 'gpt-4' })
  }

  compress(messages: readonly ContextMessage[]): CompressionResult {
    const tokensBefore = this.count(messages)
    if (messages.length === 0) return unchanged([], tokensBefore)
    const pruned = pruneToolMessages(messages, { protectLast: this.protectLast })
    const afterPrune = this.count(pruned.messages)
    if (afterPrune < this.thresholdTokens() && pruned.prunedCount > 0) {
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
    const headLast = head.at(-1)
    if (isPriorSummary(headLast?.content)) {
      prior = headLast?.content as string
      head = head.slice(0, -1)
    } else if (isPriorSummary(middle[0]?.content)) {
      prior = middle[0]?.content as string
      middle = middle.slice(1)
    }
    const budget = this.summaryBudget(this.count(middle))
    const wrapped = wrapSummary(prior, this.summarizer(middle, budget))
    const output = [...head, { role: 'user', content: wrapped }, ...tail]
    return {
      messages: output,
      compressed: true,
      tokensBefore,
      tokensAfter: this.count(output),
      protectedFirst: head.length + (prior ? 1 : 0),
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

function isPriorSummary(content: unknown): content is string {
  return typeof content === 'string' && content.startsWith(COMPACTION_REFERENCE_PREFIX)
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
