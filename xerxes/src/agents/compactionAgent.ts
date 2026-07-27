// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  COMPACTION_SUMMARY_PREFIX,
  CompactionProvisioner,
  DEFAULT_COMPACTION_SUMMARY_MAX_TOKENS,
  buildCompactionPromptFromText,
  renderMessagesForSummary,
  stripCompactionAnalysis,
  type ContextMessage,
} from '../context/index.js'
import { SmartTokenCounter } from '../context/tokenCounter.js'

export {
  COMPACTION_LENGTH_INSTRUCTIONS,
  type CompactionTargetLength,
} from '../context/index.js'

export interface CompactionCompletionRequest {
  readonly maxTokens: number
  readonly prompt: string
  readonly stream: false
  readonly temperature: number
}

export interface CompactionChoice {
  readonly message?: {
    readonly content?: unknown
  }
}

export interface CompactionCompletion {
  readonly choices?: readonly CompactionChoice[]
  readonly content?: unknown
  readonly text?: unknown
}

/** Explicit boundary through which a host invokes its chosen summarization model. */
export type CompactionCompletionPort = (
  request: CompactionCompletionRequest,
) => Promise<CompactionCompletion | string> | CompactionCompletion | string

export interface CompactionAgentOptions {
  readonly completion: CompactionCompletionPort
  readonly model?: string
  readonly summaryMaxTokens?: number
  readonly targetLength?: string
  readonly tokenCounter?: SmartTokenCounter
}

/** Summary text, or the reason a host response could not be read as text. */
export type CompactionTextResult =
  | { readonly ok: true; readonly text: string }
  | { readonly detail: string; readonly ok: false }

/**
 * Raised when the completion port returned successfully but in a shape holding no text.
 *
 * It is deliberately distinct from whatever the port throws for transport failures: a caller
 * retrying a provider outage should not retry a response shape that will never parse.
 */
export class CompactionResponseShapeError extends TypeError {
  readonly detail: string

  constructor(detail: string) {
    super(`compaction completion returned an unusable response shape: ${detail}`)
    this.name = 'CompactionResponseShapeError'
    this.detail = detail
  }
}

const COMPACTION_PROMPT_PLACEHOLDER = '__XERXES_COMPACTION_SUMMARY_PLACEHOLDER__'

/**
 * Model-backed context compactor with an injected completion boundary.
 *
 * It never creates a provider client or derives credentials. Hosts choose the
 * model and expose the exact completion call through `completion`.
 */
export class CompactionAgent {
  readonly model: string
  readonly summaryMaxTokens: number
  readonly targetLength: string
  private readonly completion: CompactionCompletionPort
  private readonly tokenCounter: SmartTokenCounter

  constructor(options: CompactionAgentOptions) {
    if (typeof options.completion !== 'function') throw new TypeError('completion must be a function')
    this.completion = options.completion
    this.model = options.model?.trim() || 'compaction'
    this.targetLength = options.targetLength?.trim() || 'concise'
    const requestedMaxTokens = options.summaryMaxTokens ?? DEFAULT_COMPACTION_SUMMARY_MAX_TOKENS
    if (!Number.isSafeInteger(requestedMaxTokens) || requestedMaxTokens < 1) {
      throw new RangeError('summaryMaxTokens must be a positive integer')
    }
    // The template asks for eight enumerated sections including every user turn verbatim; the
    // old 2_048-token ceiling truncated that mid-summary and stored the fragment.
    this.summaryMaxTokens = requestedMaxTokens
    this.tokenCounter = options.tokenCounter ?? new SmartTokenCounter({ model: this.model })
  }

  /** Summarize a text context while preserving caller-requested topics. */
  async summarizeContext(context: string, preserveTopics: readonly string[] = []): Promise<string> {
    const result = await this.summarizeContextResult(context, preserveTopics)
    if (result.ok) return result.text
    throw new CompactionResponseShapeError(result.detail)
  }

  /** Same call as `summarizeContext`, with an unreadable response reported as data. */
  async summarizeContextResult(
    context: string,
    preserveTopics: readonly string[] = [],
  ): Promise<CompactionTextResult> {
    if (!context || context.length < 200) return { ok: true, text: context }
    const response = await this.completion({
      prompt: buildCompactionPromptFromText({ context, targetLength: this.targetLength, preserveTopics }),
      temperature: 0.3,
      maxTokens: this.summaryMaxTokens,
      stream: false,
    })
    const extracted = completionText(response)
    if (!extracted.ok) return extracted
    return { ok: true, text: stripCompactionAnalysis(extracted.text) }
  }

  /**
   * Replace compactable history with a model-written summary.
   *
   * `CompactionProvisioner` continues to own the safety rules for preserving
   * system messages, live context, tool pairs, and summary placement. The
   * provisioner is first used to determine the exact compactable window, then
   * this asynchronous agent invokes the caller-owned completion port.
   */
  async summarizeMessages(messages: readonly ContextMessage[]): Promise<ContextMessage[]> {
    const original = [...messages]
    if (original.length < 2) return original

    const currentTokens = Math.max(1, this.tokenCounter.countTokens(original))
    let compactable: readonly ContextMessage[] | undefined
    const provision = new CompactionProvisioner({
      model: this.model,
      maxContextTokens: currentTokens,
      targetTokens: Math.max(1, Math.floor(currentTokens / 2)),
      tokenCounter: this.tokenCounter,
      summaryAgent: candidate => {
        compactable = candidate
        return COMPACTION_PROMPT_PLACEHOLDER
      },
    }).compact(original, { force: true })

    if (!provision.compacted || compactable === undefined) return original
    const summary = await this.summarizeContext(renderMessagesForSummary(compactable))
    if (!summary.trim()) return original
    const replaced = provision.messages.map(message => replaceSummaryPlaceholder(message, summary))
    if (this.tokenCounter.countTokens(replaced) >= currentTokens) return original
    return replaced
  }
}

/** Construct a compaction agent from a caller-owned completion port. */
export function createCompactionAgent(options: CompactionAgentOptions): CompactionAgent {
  return new CompactionAgent(options)
}

/** Build the instruction delivered to a completion port for one compaction request. */
export function buildCompactionPrompt(
  context: string,
  targetLength: string = 'concise',
  preserveTopics: readonly string[] = [],
): string {
  return buildCompactionPromptFromText({ context, targetLength, preserveTopics })
}

/** Read summary text out of a host response, reporting an unusable shape instead of throwing. */
export function completionText(response: CompactionCompletion | string): CompactionTextResult {
  if (typeof response === 'string') return { ok: true, text: response }
  if (response === null || typeof response !== 'object') return { ok: false, detail: describeShape(response) }
  const choice = response.choices?.[0]?.message?.content
  if (typeof choice === 'string') return { ok: true, text: choice }
  if (typeof response.content === 'string') return { ok: true, text: response.content }
  if (typeof response.text === 'string') return { ok: true, text: response.text }
  return { ok: false, detail: describeShape(response) }
}

/** Name the shape only: response values can be large or carry session content into logs. */
function describeShape(response: unknown): string {
  if (response === null) return 'null'
  if (typeof response !== 'object') return typeof response
  const keys = Object.keys(response).sort().slice(0, 8)
  return keys.length ? `object with keys ${keys.join(', ')}` : 'object with no keys'
}

function replaceSummaryPlaceholder(message: ContextMessage, summary: string): ContextMessage {
  if (typeof message.content !== 'string' || !message.content.includes(COMPACTION_PROMPT_PLACEHOLDER)) {
    return message
  }
  const content = message.content.replace(COMPACTION_PROMPT_PLACEHOLDER, summary)
  if (!content.startsWith(COMPACTION_SUMMARY_PREFIX)) {
    throw new Error('compaction provisioner returned an unexpected summary message')
  }
  return { ...message, content }
}
