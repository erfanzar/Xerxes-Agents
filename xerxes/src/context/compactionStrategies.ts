// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  CompactionProvisioner,
  type CompactionModelPort,
  type CompactionSummaryAgent,
} from './compactionProvisioner.js'
import type { ContextMessage } from './compressor.js'
import { SmartTokenCounter } from './tokenCounter.js'

/** Named compaction choices retained from the Python configuration surface. */
export enum CompactionStrategy {
  SUMMARIZE = 'summarize',
  SLIDING_WINDOW = 'sliding_window',
  PRIORITY_BASED = 'priority_based',
  SMART = 'smart',
  TRUNCATE = 'truncate',
  ADVANCED = 'advanced',
}

export const COMPACTION_STRATEGIES = Object.values(CompactionStrategy)

export interface CompactionStrategyStats {
  readonly compactedCount: number
  readonly messagesKept?: number
  readonly messagesSummarized?: number
  readonly originalCount: number
  readonly reason?: string
  readonly strategy: string
  readonly substrategy?: string
  readonly summaryCreated: boolean
  readonly tokensAfter?: number
  readonly tokensBefore?: number
}

export interface CompactionStrategyResult {
  readonly messages: ContextMessage[]
  readonly stats: CompactionStrategyStats
}

export interface BaseCompactionStrategyOptions {
  readonly model?: string
  readonly modelPort?: CompactionModelPort
  readonly preserveSystem?: boolean
  readonly summaryAgent?: CompactionSummaryAgent
  readonly summaryMaxTokens?: number
  readonly summaryTemperature?: number
  readonly targetTokens: number
  readonly tokenCounter?: SmartTokenCounter
}

export type PriorityScorer = (
  message: ContextMessage,
  index: number,
  metadata?: Readonly<Record<string, unknown>>,
) => number

export interface PriorityBasedStrategyOptions extends BaseCompactionStrategyOptions {
  readonly priorityScorer?: PriorityScorer
}

export interface CompactionStrategyOptions extends PriorityBasedStrategyOptions {}

/** Shared strategy scaffolding; all strategy labels delegate to one provisioner-backed path. */
export abstract class BaseCompactionStrategy {
  readonly model: string
  readonly preserveSystem: boolean
  readonly strategyName: string
  readonly targetTokens: number
  protected readonly modelPort: CompactionModelPort | undefined
  protected readonly summaryAgent: CompactionSummaryAgent | undefined
  protected readonly summaryMaxTokens: number | undefined
  protected readonly summaryTemperature: number | undefined
  protected readonly tokenCounter: SmartTokenCounter | undefined

  protected constructor(options: BaseCompactionStrategyOptions, strategyName: string) {
    this.strategyName = strategyName
    this.targetTokens = positiveInteger(options.targetTokens, 'targetTokens')
    this.model = nonEmptyText(options.model ?? 'gpt-4', 'model')
    this.preserveSystem = options.preserveSystem ?? true
    if (options.summaryAgent !== undefined && typeof options.summaryAgent !== 'function') {
      throw new TypeError('summaryAgent must be a function')
    }
    if (options.modelPort !== undefined && typeof options.modelPort !== 'function') {
      throw new TypeError('modelPort must be a function')
    }
    this.summaryAgent = options.summaryAgent
    this.modelPort = options.modelPort
    this.summaryMaxTokens = options.summaryMaxTokens
    this.summaryTemperature = options.summaryTemperature
    this.tokenCounter = options.tokenCounter
  }

  abstract compact(
    messages: readonly ContextMessage[],
    metadata?: Readonly<Record<string, unknown>>,
  ): CompactionStrategyResult

  protected runProvisioner(messages: readonly ContextMessage[], strategy = this.strategyName): CompactionStrategyResult {
    if (this.summaryAgent === undefined && this.modelPort === undefined) {
      return this.noAgentResult(messages, strategy)
    }
    const provisioner = new CompactionProvisioner({
      model: this.model,
      maxContextTokens: Math.max(this.targetTokens * 2, this.targetTokens + 1),
      targetTokens: this.targetTokens,
      thresholdTokens: 1,
      ...(this.summaryAgent === undefined ? {} : { summaryAgent: this.summaryAgent }),
      ...(this.modelPort === undefined ? {} : { modelPort: this.modelPort }),
      ...(this.summaryMaxTokens === undefined ? {} : { summaryMaxTokens: this.summaryMaxTokens }),
      ...(this.summaryTemperature === undefined ? {} : { summaryTemperature: this.summaryTemperature }),
      ...(this.tokenCounter === undefined ? {} : { tokenCounter: this.tokenCounter }),
    })
    const provision = provisioner.compact(messages, { force: true })
    if (!provision.compacted) {
      return {
        messages: [...messages],
        stats: {
          originalCount: messages.length,
          compactedCount: messages.length,
          strategy,
          summaryCreated: false,
          reason: provision.reason,
          tokensBefore: provision.tokensBefore,
          tokensAfter: provision.tokensAfter,
        },
      }
    }
    return {
      messages: provision.messages,
      stats: {
        originalCount: messages.length,
        compactedCount: provision.messages.length,
        strategy,
        summaryCreated: provision.summarizedCount > 0,
        messagesSummarized: provision.summarizedCount,
        messagesKept: provision.keptCount,
        reason: provision.reason,
        tokensBefore: provision.tokensBefore,
        tokensAfter: provision.tokensAfter,
      },
    }
  }

  private noAgentResult(messages: readonly ContextMessage[], strategy: string): CompactionStrategyResult {
    return {
      messages: [...messages],
      stats: {
        originalCount: messages.length,
        compactedCount: messages.length,
        strategy,
        summaryCreated: false,
        reason: 'no_summary_agent',
      },
    }
  }
}

/** Replace compactable history with a model-generated summary. */
export class SummarizationStrategy extends BaseCompactionStrategy {
  constructor(options: BaseCompactionStrategyOptions) {
    super(options, 'summarization')
  }

  compact(messages: readonly ContextMessage[]): CompactionStrategyResult {
    return this.runProvisioner(messages)
  }
}

/** Legacy strategy name; the safe implementation uses model-backed summary rather than dropping history. */
export class SlidingWindowStrategy extends BaseCompactionStrategy {
  constructor(options: BaseCompactionStrategyOptions) {
    super(options, 'sliding_window')
  }

  compact(messages: readonly ContextMessage[]): CompactionStrategyResult {
    return this.runProvisioner(messages)
  }
}

/** Legacy strategy name; relevance scoring remains observable but never drops messages by itself. */
export class PriorityBasedStrategy extends BaseCompactionStrategy {
  readonly priorityScorer: PriorityScorer

  constructor(options: PriorityBasedStrategyOptions) {
    super(options, 'priority_based')
    this.priorityScorer = options.priorityScorer ?? defaultPriorityScorer
  }

  compact(
    messages: readonly ContextMessage[],
    _metadata?: Readonly<Record<string, unknown>>,
  ): CompactionStrategyResult {
    return this.runProvisioner(messages)
  }
}

/** Adaptive entry point that deterministically chooses the shared summarization path. */
export class SmartCompactionStrategy extends BaseCompactionStrategy {
  constructor(options: BaseCompactionStrategyOptions) {
    super(options, 'smart')
  }

  compact(messages: readonly ContextMessage[]): CompactionStrategyResult {
    const result = this.runProvisioner(messages, 'smart')
    return {
      ...result,
      stats: {
        ...result.stats,
        substrategy: result.stats.summaryCreated
          ? 'summarization'
          : result.stats.reason === 'pruned' ? 'prune_only' : 'no_summary_agent',
      },
    }
  }
}

/** Legacy strategy name; truncation is replaced by safe summary compaction. */
export class TruncateStrategy extends BaseCompactionStrategy {
  constructor(options: BaseCompactionStrategyOptions) {
    super(options, 'truncate')
  }

  compact(messages: readonly ContextMessage[]): CompactionStrategyResult {
    return this.runProvisioner(messages)
  }
}

/** The existing compressor's prune-then-summary flow is the native advanced implementation. */
export class AdvancedCompactionStrategy extends BaseCompactionStrategy {
  constructor(options: BaseCompactionStrategyOptions) {
    super(options, 'advanced')
  }

  compact(messages: readonly ContextMessage[]): CompactionStrategyResult {
    return this.runProvisioner(messages)
  }
}

/** Return a deterministic strategy instance; unknown values safely select standard summarization. */
export function getCompactionStrategy(
  strategy: CompactionStrategy | string,
  options: CompactionStrategyOptions,
): BaseCompactionStrategy {
  switch (strategy) {
    case 'sliding_window':
      return new SlidingWindowStrategy(options)
    case 'priority_based':
      return new PriorityBasedStrategy(options)
    case 'smart':
      return new SmartCompactionStrategy(options)
    case 'truncate':
      return new TruncateStrategy(options)
    case 'advanced':
      return new AdvancedCompactionStrategy(options)
    case 'summarize':
    default:
      return new SummarizationStrategy(options)
  }
}

/** Bounded relevance hint retained for callers that inspect priority strategy behavior. */
export function defaultPriorityScorer(
  message: ContextMessage,
  index: number,
  _metadata?: Readonly<Record<string, unknown>>,
): number {
  let score = 0.5
  if (message.role === 'system') score += 0.3
  if ('function_call' in message || 'tool_calls' in message) score += 0.2
  if (String(message.content ?? '').length > 500) score += 0.1
  score += Math.min(0.1 * (index / 100), 0.1)
  return Math.min(score, 1)
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`)
  return value
}

function nonEmptyText(value: string, label: string): string {
  if (!value.trim()) throw new TypeError(`${label} must be a non-empty string`)
  return value
}
