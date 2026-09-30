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
import { classifyError, ErrorKind } from '../runtime/errorClassifier.js'

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
  /** Cap each summary request independently of the model's context window. */
  readonly maxRequestTokens?: number
  /** Effective input budget, after reserving the model's output allowance. */
  readonly maxContextTokens?: number
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
export const DEFAULT_COMPACTION_REQUEST_TOKENS = 32_000

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
  private readonly maxContextTokens: number
  private readonly maxRequestTokens: number

  constructor(options: CompactionAgentOptions) {
    if (typeof options.completion !== 'function') throw new TypeError('completion must be a function')
    this.completion = options.completion
    this.model = options.model?.trim() || 'compaction'
    this.targetLength = options.targetLength?.trim() || 'concise'
    const requestedMaxTokens = options.summaryMaxTokens ?? DEFAULT_COMPACTION_SUMMARY_MAX_TOKENS
    if (!Number.isSafeInteger(requestedMaxTokens) || requestedMaxTokens < 1) {
      throw new RangeError('summaryMaxTokens must be a positive integer')
    }
    // Reserve enough space for the final working record; intermediate segments
    // use a smaller allowance so merging them remains tractable.
    this.summaryMaxTokens = requestedMaxTokens
    this.tokenCounter = options.tokenCounter ?? new SmartTokenCounter({ model: this.model })
    this.maxContextTokens = options.maxContextTokens ?? 64_000
    this.maxRequestTokens = options.maxRequestTokens ?? DEFAULT_COMPACTION_REQUEST_TOKENS
    if (!Number.isSafeInteger(this.maxRequestTokens) || this.maxRequestTokens < 4096) {
      throw new RangeError('maxRequestTokens must be an integer of at least 4096')
    }
    if (!Number.isSafeInteger(this.maxContextTokens) || this.maxContextTokens < 4096) {
      throw new RangeError('maxContextTokens must be an integer of at least 4096')
    }
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
    return this.summarizeWithBudget(context, preserveTopics, this.maxRequestTokens)
  }

  private async summarizeWithBudget(
    context: string,
    preserveTopics: readonly string[],
    requestBudget: number,
    outputBudget: number = this.summaryMaxTokens,
  ): Promise<CompactionTextResult> {
    if (!context || context.length < 200) return { ok: true, text: context }
    const overhead = this.tokenCounter.countTokens(buildCompactionPromptFromText({ context: '', targetLength: this.targetLength, preserveTopics }))
    const inputBudget = Math.max(512, Math.floor(Math.min(this.maxContextTokens, requestBudget) * 0.8) - overhead)
    if (this.tokenCounter.countTokens(context) > inputBudget) {
      const chunks: string[] = []
      let remaining = context
      while (remaining.length) {
        let low = 1
        let high = remaining.length
        while (low < high) {
          const middle = Math.ceil((low + high) / 2)
          if (this.tokenCounter.countTokens(remaining.slice(0, middle)) <= inputBudget) low = middle
          else high = middle - 1
        }
        // Do not split a UTF-16 surrogate pair between chunks.
        if (low < remaining.length && /[\uD800-\uDBFF]/u.test(remaining[low - 1]!)) low -= 1
        chunks.push(remaining.slice(0, low))
        remaining = remaining.slice(low)
      }
      const summaries: string[] = []
      // Only top-level, independent segments run in parallel. Timeout splits
      // remain serial so retries cannot multiply this three-request bound.
      const concurrency = requestBudget === this.maxRequestTokens ? 3 : 1
      for (let offset = 0; offset < chunks.length; offset += concurrency) {
        const parts = await Promise.allSettled(chunks.slice(offset, offset + concurrency).map(chunk =>
          this.summarizeWithBudget(chunk, preserveTopics, requestBudget, Math.min(outputBudget, 2048))))
        // Drain every in-flight request before exposing failure to the caller.
        // Otherwise its retry can overlap the previous batch and exceed the bound.
        for (const settled of parts) {
          if (settled.status === 'rejected') throw settled.reason
          const part = settled.value
          if (!part.ok) return part
          if (!part.text.trim()) throw new Error('Compaction returned an empty chunk summary; original history retained')
          summaries.push(part.text)
        }
      }
      const combined = summaries.join('\n\n--- Next chronological segment ---\n\n')
      if (this.tokenCounter.countTokens(combined) >= this.tokenCounter.countTokens(context)) {
        throw new Error('Compaction did not reduce chunk summaries; original history retained')
      }
      return this.summarizeWithBudget(combined, preserveTopics, requestBudget, outputBudget)
    }
    let response: CompactionCompletion | string
    try {
      response = await this.completion({
        prompt: buildCompactionPromptFromText({ context, targetLength: this.targetLength, preserveTopics }) + `\nKeep the complete summary within ${outputBudget} output tokens. Prioritize unfinished work, constraints and exact recovery references.`,
        temperature: 0.3,
        maxTokens: outputBudget,
        stream: false,
      })
    } catch (error) {
      // Keep completed chronological segments. Retrying the entire transcript
      // after a late timeout repeatedly charges for work that already succeeded.
      if (classifyError(error).kind === ErrorKind.TIMEOUT && requestBudget > 4096) {
        return this.summarizeWithBudget(context, preserveTopics, Math.max(4096, Math.floor(requestBudget / 4)), outputBudget)
      }
      throw error
    }
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
      maxContextTokens: Math.min(currentTokens, this.maxContextTokens),
      targetTokens: Math.max(1, Math.floor(Math.min(currentTokens, this.maxContextTokens) / 2)),
      tokenCounter: this.tokenCounter,
      summaryAgent: candidate => {
        compactable = candidate
        return COMPACTION_PROMPT_PLACEHOLDER
      },
    }).compact(original, { force: true })

    if (!provision.compacted || compactable === undefined) return original
    const summary = await this.summarizeContext(renderMessagesForSummary(compactable))
    if (!summary.trim()) return original
    let replaced = provision.messages.map(message => replaceSummaryPlaceholder(message, summary))
    // An iterative pass keeps the previous summary verbatim and appends the new
    // one, so a long-running session's summary only ever grew: one goal session
    // carried a 25K-token summary against a 2K budget, and compaction could no
    // longer make room. Past twice the budget, fold both into one summary.
    const carried = provision.messages.map(carriedSummaryOf).find((text): text is string => text !== undefined)
    if (carried !== undefined && this.tokenCounter.countTokens(carried) + this.tokenCounter.countTokens(summary) > this.summaryMaxTokens * 2) {
      const consolidated = await this.summarizeContext(
        `Earlier summary of this session:\n\n${carried}\n\n--- What happened after that ---\n\n${summary}`,
      )
      if (consolidated.trim()) {
        const merged = provision.messages.map(message => consolidateSummary(message, consolidated))
        if (this.tokenCounter.countTokens(merged) < this.tokenCounter.countTokens(replaced)) replaced = merged
      }
    }
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

/**
 * The previous summary an iterative pass carried into the placeholder message,
 * without its reference prefixes (each pass used to nest another one).
 */
function carriedSummaryOf(message: ContextMessage): string | undefined {
  if (typeof message.content !== 'string' || !message.content.includes(COMPACTION_PROMPT_PLACEHOLDER)) return undefined
  const before = message.content.slice(0, message.content.indexOf(COMPACTION_PROMPT_PLACEHOLDER))
  const carried = stripSummaryPrefixes(before).replace(/\n*---\s*$/u, '').trim()
  return carried || undefined
}

function stripSummaryPrefixes(text: string): string {
  let rest = text.trimStart()
  while (rest.startsWith(COMPACTION_SUMMARY_PREFIX)) rest = rest.slice(COMPACTION_SUMMARY_PREFIX.length).trimStart()
  return rest
}

/** The placeholder message rewritten as one consolidated summary. */
function consolidateSummary(message: ContextMessage, summary: string): ContextMessage {
  if (typeof message.content !== 'string' || !message.content.includes(COMPACTION_PROMPT_PLACEHOLDER)) return message
  return { ...message, content: `${COMPACTION_SUMMARY_PREFIX}\n\n${summary.trim()}` }
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
