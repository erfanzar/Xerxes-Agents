// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { SmartTokenCounter, type ContextMessage } from '../context/index.js'

export interface MessageCompactor {
  summarizeMessages(messages: readonly ContextMessage[]): Promise<ContextMessage[]> | ContextMessage[]
}

export interface AutoCompactAgentOptions {
  readonly autoCompact?: boolean
  readonly compactTarget?: number
  readonly compactThreshold?: number
  readonly compactionStrategy?: string
  readonly compactor?: MessageCompactor
  readonly liveTailHint?: number
  readonly maxContextTokens?: number
  readonly model?: string
  readonly preserveSystemPrompt?: boolean
  readonly tokenCounter?: SmartTokenCounter
}

export interface AutoCompactionMetadata {
  readonly reason?: 'below_threshold' | 'disabled' | 'no_summary_agent'
  readonly summaryCreated: boolean
  readonly tokensAfter: number
  readonly tokensBefore: number
}

export interface AutoCompactionResult {
  readonly messages: ContextMessage[]
  readonly metadata: AutoCompactionMetadata
}

/** Token-budget watchdog for an injected context compactor. */
export class AutoCompactAgent {
  readonly autoCompact: boolean
  readonly compactTarget: number
  readonly compactThreshold: number
  readonly compactionStrategy: string
  readonly liveTailHint: number
  readonly maxContextTokens: number
  readonly model: string
  readonly preserveSystemPrompt: boolean
  readonly targetTokens: number
  readonly thresholdTokens: number
  private compactionCount = 0
  private tokensSaved = 0
  private readonly compactor: MessageCompactor | undefined
  private readonly tokenCounter: SmartTokenCounter

  constructor(options: AutoCompactAgentOptions = {}) {
    this.model = options.model ?? ''
    this.autoCompact = options.autoCompact ?? true
    this.compactThreshold = fraction(options.compactThreshold ?? 0.8, 'compactThreshold')
    this.compactTarget = fraction(options.compactTarget ?? 0.5, 'compactTarget')
    this.maxContextTokens = positiveInteger(options.maxContextTokens ?? 8_000, 'maxContextTokens')
    this.compactionStrategy = options.compactionStrategy?.trim() || 'summarize'
    this.preserveSystemPrompt = options.preserveSystemPrompt ?? true
    this.liveTailHint = nonNegativeInteger(options.liveTailHint ?? 5, 'liveTailHint')
    this.compactor = options.compactor
    this.tokenCounter = options.tokenCounter ?? new SmartTokenCounter({ model: this.model })
    this.thresholdTokens = Math.floor(this.maxContextTokens * this.compactThreshold)
    this.targetTokens = Math.floor(this.maxContextTokens * this.compactTarget)
  }

  /** Return cumulative compaction savings and resolved thresholds. */
  getStatistics(): Readonly<Record<string, number | string>> {
    return Object.freeze({
      compactionCount: this.compactionCount,
      tokensSaved: this.tokensSaved,
      maxContextTokens: this.maxContextTokens,
      thresholdTokens: this.thresholdTokens,
      targetTokens: this.targetTokens,
      strategy: this.compactionStrategy,
    })
  }

  /** Return the configured automatic-compaction threshold and target. */
  checkUsage(): Readonly<Record<string, number>> {
    return Object.freeze({
      maxContextTokens: this.maxContextTokens,
      thresholdTokens: this.thresholdTokens,
      compactThreshold: this.compactThreshold,
      compactTarget: this.compactTarget,
    })
  }

  /** Record one externally completed compaction pass. */
  recordCompaction(tokensBefore: number, tokensAfter: number): void {
    this.compactionCount += 1
    this.tokensSaved += tokensBefore - tokensAfter
  }

  /** Return whether this message sequence meets the configured threshold. */
  shouldCompact(messages: readonly ContextMessage[]): boolean {
    return this.autoCompact && this.tokenCounter.countTokens(messages) >= this.thresholdTokens
  }

  /**
   * Compact only when the threshold is reached and a concrete compactor was
   * supplied. No deterministic excerpt is substituted for an unavailable LLM.
   */
  async compact(messages: readonly ContextMessage[]): Promise<AutoCompactionResult> {
    const original = [...messages]
    const tokensBefore = this.tokenCounter.countTokens(original)
    if (!this.autoCompact) return unchanged(original, tokensBefore, 'disabled')
    if (tokensBefore < this.thresholdTokens) return unchanged(original, tokensBefore, 'below_threshold')
    if (this.compactor === undefined) return unchanged(original, tokensBefore, 'no_summary_agent')

    const compacted = await this.compactor.summarizeMessages(original)
    const tokensAfter = this.tokenCounter.countTokens(compacted)
    const summaryCreated = !sameMessages(original, compacted)
    if (summaryCreated) this.recordCompaction(tokensBefore, tokensAfter)
    return Object.freeze({
      messages: [...compacted],
      metadata: Object.freeze({ summaryCreated, tokensBefore, tokensAfter }),
    })
  }
}

function unchanged(
  messages: ContextMessage[],
  tokens: number,
  reason: NonNullable<AutoCompactionMetadata['reason']>,
): AutoCompactionResult {
  return Object.freeze({
    messages,
    metadata: Object.freeze({ summaryCreated: false, reason, tokensBefore: tokens, tokensAfter: tokens }),
  })
}

function sameMessages(left: readonly ContextMessage[], right: readonly ContextMessage[]): boolean {
  if (left.length !== right.length) return false
  return left.every((message, index) => JSON.stringify(message) === JSON.stringify(right[index]))
}

function fraction(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0 || value > 1) throw new TypeError(`${name} must be between 0 and 1`)
  return value
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) throw new TypeError(`${name} must be a positive safe integer`)
  return value
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0) throw new TypeError(`${name} must be a non-negative safe integer`)
  return value
}
