// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  DEFAULT_SWE_WORKING_SET_BUDGET_CHARS,
  extractSweWorkingSet,
  hasSweWorkingSetEntries,
  renderSweWorkingSet,
  stripSweWorkingSet,
} from './sweWorkingSet.js'
import { SmartTokenCounter } from './tokenCounter.js'
import { repairToolMessageSequence } from './toolPairRepair.js'
import { pruneToolMessages } from './toolResultPruner.js'

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
  /** Optional external prior summary text whose `<swe-working-set>` block is carried forward. */
  readonly priorSummary?: string
  readonly protectFirst?: number
  readonly protectLast?: number
  readonly summarizer?: Summarizer
  readonly summaryBudgetRatio?: number
  readonly summaryMaxTokens?: number
  readonly summaryMinTokens?: number
  readonly threshold?: number
  readonly tokenCounter?: SmartTokenCounter
  readonly workingSetBudgetChars?: number
}

/** Pre-prunes tool data then safely folds the unprotected middle into a reference-only summary. */
export class ContextCompressor {
  readonly contextWindow: number | undefined
  readonly forceSummarize: boolean
  readonly protectFirst: number
  readonly protectLast: number
  readonly summaryBudgetRatio: number
  readonly summaryMaxTokens: number
  readonly summaryMinTokens: number
  readonly threshold: number
  private readonly priorSummary: string | undefined
  private readonly summarizer: Summarizer | undefined
  private readonly tokenCounter: SmartTokenCounter
  private readonly workingSetBudgetChars: number

  constructor(options: ContextCompressorOptions = {}) {
    this.threshold = options.threshold ?? 0.75
    if (this.threshold <= 0 || this.threshold > 1) throw new Error('threshold must be in (0.0, 1.0]')
    this.contextWindow = options.contextWindow
    if (this.contextWindow !== undefined && (!Number.isSafeInteger(this.contextWindow) || this.contextWindow <= 0)) {
      throw new Error('contextWindow must be a positive safe integer when supplied')
    }
    this.protectFirst = options.protectFirst ?? 3
    this.protectLast = options.protectLast ?? 6
    if (this.protectFirst < 0 || this.protectLast < 0) throw new Error('protectFirst and protectLast must be >= 0')
    this.summaryMinTokens = options.summaryMinTokens ?? 2_000
    this.summaryMaxTokens = options.summaryMaxTokens ?? 12_000
    this.summaryBudgetRatio = options.summaryBudgetRatio ?? 0.2
    this.forceSummarize = options.forceSummarize ?? false
    this.priorSummary = options.priorSummary
    this.summarizer = options.summarizer
    this.tokenCounter = options.tokenCounter ?? new SmartTokenCounter({ model: options.model ?? 'gpt-4' })
    this.workingSetBudgetChars = options.workingSetBudgetChars ?? DEFAULT_SWE_WORKING_SET_BUDGET_CHARS
  }

  compress(messages: readonly ContextMessage[]): CompressionResult {
    const tokensBefore = this.count(messages)
    if (messages.length === 0) return unchanged([], tokensBefore)
    const pruned = pruneToolMessages(messages, { protectLast: this.protectLast })
    const afterPrune = this.count(pruned.messages)
    const thresholdTokens = this.thresholdTokens()
    if ((thresholdTokens === undefined || afterPrune < thresholdTokens) && !this.forceSummarize) {
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
    const middleEnd = tailCount ? count - tailCount : count
    let head = pruned.messages.slice(0, headCount)
    const tail = tailCount ? pruned.messages.slice(count - tailCount) : []
    let middle = pruned.messages.slice(headCount, middleEnd)
    let rawMiddle = messages.slice(headCount, middleEnd)
    let callContextEnd = headCount
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
      callContextEnd = Math.max(0, headCount - 1)
    } else if (middlePrior !== undefined) {
      prior = middlePrior
      middle = middle.slice(1)
      rawMiddle = rawMiddle.slice(1)
    }
    if (!middle.length) {
      return {
        messages: pruned.messages,
        compressed: pruned.prunedCount > 0,
        tokensBefore,
        tokensAfter: afterPrune,
        protectedFirst: head.length + (priorFromHead ? 1 : 0),
        protectedLast: tailCount,
        compressedCount: 0,
        prunedToolResults: pruned.prunedCount,
        summaryTokens: 0,
        metadata: { strategy: 'no-middle' },
      }
    }

    const priorWorkingSetSource = prior ?? this.priorSummary
    const ledger = extractSweWorkingSet(rawMiddle, {
      ...(priorWorkingSetSource === undefined ? {} : { priorSummary: priorWorkingSetSource }),
      callContextMessages: messages.slice(0, callContextEnd),
    })

    const budget = this.summaryBudget(this.count(middle))
    const effectiveSummaryTokenCap = Math.min(this.summaryMaxTokens, Math.max(budget, 320))
    const hasWorkingSet = hasSweWorkingSetEntries(ledger) && this.workingSetBudgetChars > 0

    let wsCharBudget = hasWorkingSet
      ? Math.min(this.workingSetBudgetChars, Math.max(180, Math.floor(effectiveSummaryTokenCap * 0.48) * 4))
      : 0
    let workingSetBlock = wsCharBudget > 0 ? renderSweWorkingSet(ledger, wsCharBudget) : ''
    const reservedWsTokens = workingSetBlock ? this.tokenCounter.countTokens(workingSetBlock) : 0
    const summarizerBudget = Math.max(1, budget - reservedWsTokens)
    const rawSummary = this.summarizer(middle, summarizerBudget)

    const proseWrapped = wrapSummary(prior, rawSummary, '')
    let wrapped = workingSetBlock ? wrapSummary(prior, rawSummary, workingSetBlock) : proseWrapped

    // Keep total summary within effectiveSummaryTokenCap when adding `<swe-working-set>`.
    if (workingSetBlock) {
      const proseTokens = this.tokenCounter.countTokens(proseWrapped)
      if (proseTokens < effectiveSummaryTokenCap) {
        while (workingSetBlock && this.tokenCounter.countTokens(wrapped) > effectiveSummaryTokenCap) {
          wsCharBudget = Math.floor(wsCharBudget * 0.7)
          if (wsCharBudget < 96) {
            workingSetBlock = ''
            wrapped = proseWrapped
            break
          }
          workingSetBlock = renderSweWorkingSet(ledger, wsCharBudget)
          wrapped = workingSetBlock ? wrapSummary(prior, rawSummary, workingSetBlock) : proseWrapped
        }
      } else {
        workingSetBlock = ''
        wrapped = proseWrapped
      }
    }

    // Cutting the middle can leave the head ending in assistant tool_calls whose results
    // were summarized away, or the tail beginning with orphan tool results. Repair the
    // window here so every caller receives a provider-safe sequence.
    const buildOutput = (summaryContent: string): ContextMessage[] => {
      const summaryMessage: ContextMessage = {
        role: 'user',
        content: summaryContent,
        [COMPACTION_SUMMARY_MARKER]: true,
      }
      return repairToolMessageSequence([...head, summaryMessage, ...tail])
    }

    let output = buildOutput(wrapped)
    let tokensAfter = this.count(output)

    // Never allow `<swe-working-set>` alone to prevent a shrinking pass from shrinking below tokensBefore.
    if (workingSetBlock && tokensAfter >= tokensBefore) {
      const proseOutput = buildOutput(proseWrapped)
      const proseAfter = this.count(proseOutput)
      if (proseAfter < tokensBefore) {
        let shrinkBudget = Math.min(wsCharBudget, Math.max(96, (tokensBefore - proseAfter - 1) * 4))
        let fitted = false
        for (let step = 0; step < 4 && shrinkBudget >= 96; step += 1) {
          const candidateBlock = renderSweWorkingSet(ledger, shrinkBudget)
          if (!candidateBlock) break
          const candidateWrapped = wrapSummary(prior, rawSummary, candidateBlock)
          const candidateOutput = buildOutput(candidateWrapped)
          const candidateTokens = this.count(candidateOutput)
          if (candidateTokens < tokensBefore) {
            wrapped = candidateWrapped
            output = candidateOutput
            tokensAfter = candidateTokens
            fitted = true
            break
          }
          shrinkBudget = Math.floor(shrinkBudget * 0.65)
        }
        if (!fitted) {
          wrapped = proseWrapped
          output = proseOutput
          tokensAfter = proseAfter
        }
      }
    }

    return {
      messages: output,
      compressed: true,
      tokensBefore,
      tokensAfter,
      protectedFirst: head.length + (priorFromHead ? 1 : 0),
      protectedLast: tailCount,
      compressedCount: middle.length,
      prunedToolResults: pruned.prunedCount,
      summaryTokens: this.tokenCounter.countTokens(wrapped),
      metadata: { strategy: prior ? 'iterative' : 'first-pass' },
    }
  }

  shouldCompact(messages: readonly ContextMessage[]): boolean {
    const threshold = this.thresholdTokens()
    return threshold !== undefined && this.count(messages) >= threshold
  }

  thresholdTokens(): number | undefined {
    return this.contextWindow === undefined
      ? undefined
      : Math.floor(this.contextWindow * this.threshold)
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

/**
 * Added once after the reference prefix by `compactMessagesIfNeeded`, only
 * when the replaced history was archived; iterative passes strip it so it is
 * never repeated. The summary is lossy, the archive is not, and the model
 * only searches what it knows exists.
 */
export const COMPACTION_ARCHIVE_NOTE = 'Earlier turns are archived verbatim. If an exact value, path, error or decision is missing here, recover it with session_search.'

function stripSummaryHeaders(text: string): string {
  let rest = text.trimStart()
  for (;;) {
    if (rest.startsWith(COMPACTION_REFERENCE_PREFIX)) rest = rest.slice(COMPACTION_REFERENCE_PREFIX.length).trimStart()
    else if (rest.startsWith(COMPACTION_ARCHIVE_NOTE)) rest = rest.slice(COMPACTION_ARCHIVE_NOTE.length).trimStart()
    else return stripSweWorkingSet(rest)
  }
}

function wrapSummary(prior: string | undefined, summary: string, workingSetBlock: string): string {
  const priorBody = prior === undefined ? '' : stripSummaryHeaders(prior)
  const summaryBody = stripSummaryHeaders(summary)
  const proseBody = priorBody && summaryBody
    ? `${priorBody}\n\n---\n\n${summaryBody}`
    : priorBody || summaryBody
  const fullBody = proseBody && workingSetBlock
    ? `${proseBody}\n\n${workingSetBlock}`
    : proseBody || workingSetBlock
  return `${COMPACTION_REFERENCE_PREFIX}\n\n${fullBody}`
}
