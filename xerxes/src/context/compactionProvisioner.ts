// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  COMPACTION_REFERENCE_PREFIX,
  ContextCompressor,
  type ContextMessage,
  type Summarizer,
} from './compressor.js'
import { SmartTokenCounter } from './tokenCounter.js'
import { repairToolMessageSequence } from './toolPairRepair.js'

export { repairToolMessageSequence } from './toolPairRepair.js'

/** Prefix emitted by the shared compressor for provisioned summary messages. */
export const COMPACTION_SUMMARY_PREFIX = COMPACTION_REFERENCE_PREFIX
export const DEFAULT_COMPACTION_TAIL_RATIO = 0.35
export const DEFAULT_COMPACTION_TARGET_RATIO = 0.5
export const DEFAULT_COMPACTION_THRESHOLD_RATIO = 0.75
export const DEFAULT_COMPACTION_SUMMARY_MAX_TOKENS = 8_192
export const DEFAULT_COMPACTION_SUMMARY_TEMPERATURE = 0.2

/** Caller-supplied summary function; no provider client is constructed by context code. */
export type CompactionSummaryAgent = (
  messages: readonly ContextMessage[],
  previousSummary?: string,
) => string

/** Fully resolved request delivered to an injected model port. */
export interface CompactionModelRequest {
  readonly maxTokens: number
  readonly messages: readonly ContextMessage[]
  readonly model: string
  readonly previousSummary?: string
  readonly prompt: string
  readonly temperature: number
}

/** Model boundary for summary generation. It is intentionally synchronous to match ContextCompressor. */
export type CompactionModelPort = (request: CompactionModelRequest) => string

export interface ProviderCompactionAgentOptions {
  readonly maxTokens?: number
  readonly model: string
  readonly modelPort: CompactionModelPort
  readonly temperature?: number
}

/**
 * Adapter for a caller-owned model port.
 *
 * Despite the historical Python name, this class does not discover providers,
 * read credentials, or create network clients. The host owns those concerns.
 */
export class ProviderCompactionAgent {
  readonly maxTokens: number
  readonly model: string
  readonly temperature: number
  private readonly modelPort: CompactionModelPort

  constructor(options: ProviderCompactionAgentOptions) {
    this.model = requireText(options.model, 'model')
    if (typeof options.modelPort !== 'function') {
      throw new TypeError('modelPort must be a function')
    }
    this.modelPort = options.modelPort
    const requestedMaxTokens = positiveInteger(
      options.maxTokens ?? DEFAULT_COMPACTION_SUMMARY_MAX_TOKENS,
      'maxTokens',
    )
    this.maxTokens = Math.max(512, requestedMaxTokens)
    this.temperature = nonNegativeFiniteNumber(
      options.temperature ?? DEFAULT_COMPACTION_SUMMARY_TEMPERATURE,
      'temperature',
    )
  }

  summarize(messages: readonly ContextMessage[], previousSummary?: string): string {
    const request: CompactionModelRequest = {
      model: this.model,
      messages,
      prompt: buildCompactionPrompt(messages, previousSummary),
      maxTokens: this.maxTokens,
      temperature: this.temperature,
      ...(previousSummary === undefined ? {} : { previousSummary }),
    }
    const summary = this.modelPort(request)
    if (typeof summary !== 'string') {
      throw new TypeError('compaction model port must return a string')
    }
    return summary
  }

  toSummaryAgent(): CompactionSummaryAgent {
    return (messages, previousSummary) => this.summarize(messages, previousSummary)
  }
}

/** Create a summary agent from an explicit host-owned model invocation port. */
export function compactionSummaryAgentFromModelPort(
  model: string,
  modelPort: CompactionModelPort,
  options: Omit<ProviderCompactionAgentOptions, 'model' | 'modelPort'> = {},
): CompactionSummaryAgent {
  return new ProviderCompactionAgent({ model, modelPort, ...options }).toSummaryAgent()
}

export interface CompactionProvision {
  readonly compacted: boolean
  readonly error: string
  readonly keptCount: number
  readonly messages: ContextMessage[]
  readonly reason: string
  readonly summarizedCount: number
  readonly tokensAfter: number
  readonly tokensBefore: number
}

export interface CompactionProvisionerOptions {
  readonly maxContextTokens: number
  readonly model: string
  readonly modelPort?: CompactionModelPort
  readonly summaryAgent?: CompactionSummaryAgent
  readonly summaryMaxTokens?: number
  readonly summaryTemperature?: number
  readonly tailRatio?: number
  readonly targetRatio?: number
  readonly targetTokens?: number
  readonly thresholdRatio?: number
  readonly thresholdTokens?: number
  readonly tokenCounter?: SmartTokenCounter
}

export interface CompactionDecisionOptions {
  readonly appendedMessages?: readonly ContextMessage[]
  readonly force?: boolean
}

export interface CompactOptions {
  readonly force?: boolean
  readonly previousSummary?: string
}

/**
 * Plan and execute a model-backed compaction pass around the shared compressor.
 *
 * Thresholding and the protected live tail follow the Python provisioner. The
 * actual pruning, summary placement, iterative-summary handling, and token
 * accounting stay in ContextCompressor so the runtime has one compressor.
 */
export class CompactionProvisioner {
  readonly maxContextTokens: number
  readonly model: string
  readonly tailRatio: number
  readonly targetTokens: number
  readonly thresholdTokens: number
  private readonly summaryAgent: CompactionSummaryAgent | undefined
  private readonly tokenCounter: SmartTokenCounter

  constructor(options: CompactionProvisionerOptions) {
    this.model = requireText(options.model, 'model')
    this.maxContextTokens = positiveInteger(options.maxContextTokens, 'maxContextTokens')
    const thresholdRatio = nonNegativeFiniteNumber(
      options.thresholdRatio ?? DEFAULT_COMPACTION_THRESHOLD_RATIO,
      'thresholdRatio',
    )
    const targetRatio = nonNegativeFiniteNumber(
      options.targetRatio ?? DEFAULT_COMPACTION_TARGET_RATIO,
      'targetRatio',
    )
    const configuredThreshold = options.thresholdTokens === undefined
      ? Math.floor(this.maxContextTokens * thresholdRatio)
      : positiveInteger(options.thresholdTokens, 'thresholdTokens')
    const configuredTarget = options.targetTokens === undefined
      ? Math.floor(this.maxContextTokens * targetRatio)
      : positiveInteger(options.targetTokens, 'targetTokens')
    this.thresholdTokens = Math.max(1, configuredThreshold)
    this.targetTokens = Math.max(1, Math.min(configuredTarget, this.thresholdTokens))
    this.tailRatio = clamp(
      nonNegativeFiniteNumber(options.tailRatio ?? DEFAULT_COMPACTION_TAIL_RATIO, 'tailRatio'),
      0.05,
      0.9,
    )
    if (options.summaryAgent !== undefined && typeof options.summaryAgent !== 'function') {
      throw new TypeError('summaryAgent must be a function')
    }
    if (options.modelPort !== undefined && typeof options.modelPort !== 'function') {
      throw new TypeError('modelPort must be a function')
    }
    this.summaryAgent = options.summaryAgent ?? (options.modelPort === undefined
      ? undefined
      : compactionSummaryAgentFromModelPort(this.model, options.modelPort, {
        ...(options.summaryMaxTokens === undefined ? {} : { maxTokens: options.summaryMaxTokens }),
        ...(options.summaryTemperature === undefined ? {} : { temperature: options.summaryTemperature }),
      }))
    this.tokenCounter = options.tokenCounter ?? new SmartTokenCounter({ model: this.model })
  }

  /** Return the shared model-aware estimate for a message sequence. */
  countTokens(messages: readonly ContextMessage[]): number {
    return this.tokenCounter.countTokens(messages)
  }

  /** Decide against the full post-append context, without mutating either input. */
  shouldCompact(messages: readonly ContextMessage[], options: CompactionDecisionOptions = {}): boolean {
    if (options.force) return true
    const candidate = [...messages, ...(options.appendedMessages ?? [])]
    return this.countTokens(candidate) >= this.thresholdTokens
  }

  /** Compact only existing history before an incoming turn is appended. */
  compactBeforeAppend(
    messages: readonly ContextMessage[],
    appendedMessages: readonly ContextMessage[],
  ): CompactionProvision {
    const candidate = [...messages, ...appendedMessages]
    const tokensBefore = this.countTokens(candidate)
    if (tokensBefore < this.thresholdTokens) {
      return unchanged(messages, tokensBefore, 'below_threshold')
    }

    const provision = this.compact(messages, { force: true })
    if (!provision.compacted) {
      return {
        ...unchanged(messages, tokensBefore, provision.reason),
        error: provision.error,
      }
    }
    return {
      ...provision,
      tokensBefore,
      tokensAfter: this.countTokens([...provision.messages, ...appendedMessages]),
    }
  }

  /** Replace compactable history with a model-written reference summary. */
  compact(messages: readonly ContextMessage[], options: CompactOptions = {}): CompactionProvision {
    const original = [...messages]
    const tokensBefore = this.countTokens(original)
    if (!options.force && tokensBefore < this.thresholdTokens) {
      return unchanged(original, tokensBefore, 'below_threshold')
    }
    if (this.summaryAgent === undefined) {
      return unchanged(original, tokensBefore, 'no_summary_agent')
    }

    const systemMessages = original.filter(message => message.role === 'system')
    const conversationMessages = original.filter(message => message.role !== 'system')
    const protectedTail = this.protectedTailCount(conversationMessages)
    if (conversationMessages.length < 2 || protectedTail >= conversationMessages.length) {
      return unchanged(original, tokensBefore, 'nothing_to_compact')
    }

    let summaryCalled = false
    let summaryFailure: unknown
    let summaryWasEmpty = false
    const compressor = this.compressor(protectedTail, (compactableMessages, _budgetTokens) => {
      summaryCalled = true
      try {
        const summary = options.previousSummary === undefined
          ? this.summaryAgent!(compactableMessages)
          : this.summaryAgent!(compactableMessages, options.previousSummary)
        if (typeof summary !== 'string') {
          throw new TypeError('summaryAgent must return a string')
        }
        const trimmed = summary.trim()
        if (!trimmed) {
          summaryWasEmpty = true
          return ''
        }
        return trimmed
      } catch (error) {
        summaryFailure = error
        return ''
      }
    })
    const compressed = compressor.compress(conversationMessages)
    if (summaryFailure !== undefined) {
      return failure(original, tokensBefore, 'summary_agent_failed', summaryFailure)
    }
    if (summaryWasEmpty) {
      return unchanged(original, tokensBefore, 'empty_summary')
    }
    if (!compressed.compressed) {
      return unchanged(original, tokensBefore, 'nothing_to_compact')
    }

    const output = repairToolMessageSequence([...systemMessages, ...compressed.messages])
    const tokensAfter = this.countTokens(output)
    if (tokensAfter >= tokensBefore) {
      return {
        ...unchanged(original, tokensBefore, 'summary_did_not_shrink'),
        tokensAfter,
      }
    }
    return {
      compacted: true,
      messages: output,
      tokensBefore,
      tokensAfter,
      summarizedCount: compressed.compressedCount,
      keptCount: protectedTail,
      reason: summaryCalled ? 'compacted' : 'pruned',
      error: '',
    }
  }

  private compressor(protectLast: number, summarizer: Summarizer): ContextCompressor {
    return new ContextCompressor({
      contextWindow: this.maxContextTokens,
      model: this.model,
      protectFirst: 0,
      protectLast,
      summarizer,
      summaryMaxTokens: this.targetTokens,
      summaryMinTokens: 1,
      threshold: clamp(this.thresholdTokens / this.maxContextTokens, Number.EPSILON, 1),
      tokenCounter: this.tokenCounter,
    })
  }

  private protectedTailCount(messages: readonly ContextMessage[]): number {
    if (!messages.length) return 0
    const tailBudget = Math.max(1, Math.floor(this.targetTokens * this.tailRatio))
    let start = messages.length
    let tailTokens = 0
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index]
      if (message === undefined) continue
      const messageTokens = Math.max(1, this.countTokens([message]))
      if (start !== messages.length && tailTokens + messageTokens > tailBudget) break
      start = index
      tailTokens += messageTokens
    }
    if (start === messages.length) start = messages.length - 1
    while (start > 0 && messages[start]?.role === 'tool') {
      start -= 1
      if (messages[start]?.role === 'assistant') break
    }
    return messages.length - start
  }
}

/** Render provider-shaped message content as readable text for a summary prompt. */
export function messageContentToText(content: unknown): string {
  if (typeof content === 'string') return content
  if (Array.isArray(content)) {
    return content.map(item => isRecord(item)
      ? typeof item.text === 'string' ? item.text : stableJson(item)
      : String(item)).join('\n')
  }
  if (isRecord(content)) return stableJson(content)
  return content === undefined || content === null ? '' : String(content)
}

/** Render full messages into the explicit, deterministic prompt input supplied to a model port. */
export function renderMessagesForSummary(messages: readonly ContextMessage[]): string {
  return messages.map((message, index) => {
    const role = typeof message.role === 'string' ? message.role.toUpperCase() : 'UNKNOWN'
    const lines = [`Message ${index + 1} [${role}]`]
    const content = messageContentToText(message.content)
    if (content) lines.push(content)
    if (message.tool_calls) lines.push(`tool_calls=${stableJson(message.tool_calls)}`)
    if (message.tool_call_id) lines.push(`tool_call_id=${String(message.tool_call_id)}`)
    return lines.join('\n')
  }).join('\n\n')
}

/** Build the durable-memory instruction used by ProviderCompactionAgent's injected model port. */
export function buildCompactionPrompt(messages: readonly ContextMessage[], previousSummary?: string): string {
  const prior = previousSummary?.trim()
  return [
    'Rewrite the following conversation history into a compact, durable memory for the next model turn.',
    'Preserve concrete facts, user instructions, decisions, file paths, tool results, errors, fixes,',
    'open questions, and current task state. Drop chatter and duplicate text. Output only the summary.',
    '',
    ...(prior ? ['Existing summary to refresh:', prior, ''] : []),
    'Conversation history to compact:',
    renderMessagesForSummary(messages),
  ].join('\n')
}

function unchanged(messages: readonly ContextMessage[], tokens: number, reason: string): CompactionProvision {
  return {
    compacted: false,
    messages: [...messages],
    tokensBefore: tokens,
    tokensAfter: tokens,
    summarizedCount: 0,
    keptCount: 0,
    reason,
    error: '',
  }
}

function failure(
  messages: readonly ContextMessage[],
  tokens: number,
  reason: string,
  error: unknown,
): CompactionProvision {
  return { ...unchanged(messages, tokens, reason), error: errorMessage(error) }
}

function stableJson(value: unknown): string {
  try {
    return JSON.stringify(value, (_key, item: unknown) => {
      if (typeof item === 'bigint') return item.toString()
      if (!isRecord(item)) return item
      return Object.fromEntries(Object.keys(item).sort().map(key => [key, item[key]]))
    }) ?? String(value)
  } catch {
    return String(value)
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function requireText(value: string, label: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new TypeError(`${label} must be a non-empty string`)
  return value
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`)
  return value
}

function nonNegativeFiniteNumber(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 0) throw new RangeError(`${label} must be a non-negative finite number`)
  return value
}

function clamp(value: number, lower: number, upper: number): number {
  return Math.max(lower, Math.min(upper, value))
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
