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

/** Delimiters of the model's private scratchpad; `stripCompactionAnalysis` removes it before storage. */
export const COMPACTION_ANALYSIS_OPEN_TAG = '<analysis>'
export const COMPACTION_ANALYSIS_CLOSE_TAG = '</analysis>'

/** How dense each section of the compaction template should be. */
export const COMPACTION_LENGTH_INSTRUCTIONS = Object.freeze({
  brief: 'Keep every section to its shortest useful form: one clause per bullet, no narration.',
  concise: 'Keep bullets to one or two lines and drop anything the next turn will not act on.',
  detailed: 'Keep full detail in every section: exact values, identifiers, and the reasoning behind each decision.',
})

export type CompactionTargetLength = keyof typeof COMPACTION_LENGTH_INSTRUCTIONS

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
    const compressor = this.compressor(protectedTail, options.force === true, (compactableMessages, _budgetTokens) => {
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

  private compressor(protectLast: number, forceSummarize: boolean, summarizer: Summarizer): ContextCompressor {
    return new ContextCompressor({
      contextWindow: this.maxContextTokens,
      forceSummarize,
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

/**
 * Render full messages into the explicit, deterministic prompt input supplied to a model port.
 *
 * Tool traffic collapses into one `called name(args) -> outcome` line per call. Dumping raw
 * `tool_calls` JSON next to a separate result message spent most of the summarizer's input on
 * argument scaffolding and call ids — the budget the summary needs for file state and decisions.
 */
export function renderMessagesForSummary(messages: readonly ContextMessage[]): string {
  const resultsById = toolResultsById(messages)
  const folded = new Set<string>()
  const rendered: string[] = []
  for (const message of messages) {
    const role = typeof message.role === 'string' ? message.role.toUpperCase() : 'UNKNOWN'
    const callLines = renderToolCalls(message.tool_calls, resultsById, folded)
    const identifier = typeof message.tool_call_id === 'string' ? message.tool_call_id : ''
    // The result already appears inside its caller's line; repeating it here would restore
    // exactly the duplication this rendering exists to remove.
    if (role === 'TOOL' && folded.has(identifier)) continue
    const lines = [`Message ${rendered.length + 1} [${role}]`]
    const content = messageContentToText(message.content)
    if (content) lines.push(content)
    lines.push(...callLines)
    if (message.tool_call_id) lines.push(`tool_call_id=${String(message.tool_call_id)}`)
    rendered.push(lines.join('\n'))
  }
  return rendered.join('\n\n')
}

export interface CompactionPromptOptions {
  readonly context: string
  readonly preserveTopics?: readonly string[]
  readonly previousSummary?: string
  readonly targetLength?: string
}

/**
 * The single compaction template.
 *
 * Two templates used to exist — this one and a shorter set of generic bullets in the agent — and
 * only the weaker one was reachable from the daemon. Both entry points now render through here so
 * an edit to the wording cannot improve a prompt nobody runs.
 */
export function buildCompactionPromptFromText(options: CompactionPromptOptions): string {
  const lengthInstruction = COMPACTION_LENGTH_INSTRUCTIONS[options.targetLength as CompactionTargetLength]
    ?? COMPACTION_LENGTH_INSTRUCTIONS.concise
  const topics = (options.preserveTopics ?? []).filter(topic => typeof topic === 'string' && topic.trim())
  const prior = options.previousSummary?.trim()
  return [
    'You are the context-compaction engine for a long-running coding session. Rewrite the slice of',
    'transcript below into the only record of it that survives: the raw messages are dropped once you',
    'answer, so whatever you leave out is gone for the rest of the session.',
    '',
    'WHERE YOUR OUTPUT LANDS',
    'The summary is inserted between the preserved opening of the session and a preserved live tail of',
    'the most recent messages, which the agent still reads verbatim. Cover the slice below and nothing',
    'else: do not restate the system prompt, and do not re-describe the recent turns that follow you.',
    '',
    `FIRST, THINK IN ${COMPACTION_ANALYSIS_OPEN_TAG}`,
    `Open with an ${COMPACTION_ANALYSIS_OPEN_TAG} block and use it to list every user turn in the slice, the task in`,
    'flight, the files touched, and anything you are unsure about. That block is stripped before the',
    `summary is stored, so it costs the agent nothing. Close it with ${COMPACTION_ANALYSIS_CLOSE_TAG}, then write the`,
    'summary.',
    '',
    'THEN WRITE THE SUMMARY, USING EXACTLY THESE SECTIONS',
    '',
    '## User requests',
    'Every non-tool user message in the slice, in order, one bullet each — including asks that were',
    'refused, deferred, corrected, or already satisfied. Never merge two requests into one bullet and',
    'never drop one because it looks handled. Quote the operative wording of each ask.',
    '',
    '## Current task',
    'The instruction being worked on where the slice ends, quoted VERBATIM, then how far it got.',
    'Do not paraphrase the instruction itself.',
    '',
    '## Files touched',
    'One bullet per file: absolute path, what was done to it (read, created, edited, deleted), and its',
    'state now — saved, half-edited, reverted, or only inspected. Reproduce paths exactly.',
    '',
    '## Decisions',
    'Each decision with the reason behind it, so it is not reopened and re-argued.',
    '',
    '## Errors and fixes',
    'Each failure — error text, failing test, wrong output — its cause, and whether the fix landed.',
    '',
    '## Open questions',
    'Unresolved questions, unverified assumptions, and anything waiting on the user.',
    '',
    '## Next step',
    'The single concrete action to take next. If the slice ends mid-edit, name the file and the change.',
    '',
    'RULES',
    `- ${lengthInstruction}`,
    '- Reproduce identifiers exactly: paths, function names, commands, flags, versions, error strings.',
    '- Record outcomes, not narration. Drop chatter, restated plans, and duplicated text.',
    '- Never invent progress the slice does not show; write "unknown" rather than guessing.',
    '- Keep every section header, and write "none" under a section the slice leaves empty.',
    ...(topics.length ? [`- Ensure these topics are covered: ${topics.join(', ')}`] : []),
    '',
    ...(prior
      ? ['EXISTING SUMMARY TO REFRESH (fold it into the sections above; do not append to it):', prior, '']
      : []),
    'CONTEXT TO SUMMARIZE:',
    options.context,
    '',
    `Begin with ${COMPACTION_ANALYSIS_OPEN_TAG}.`,
  ].join('\n')
}

/** Build the durable-memory instruction used by ProviderCompactionAgent's injected model port. */
export function buildCompactionPrompt(messages: readonly ContextMessage[], previousSummary?: string): string {
  return buildCompactionPromptFromText({
    context: renderMessagesForSummary(messages),
    ...(previousSummary === undefined ? {} : { previousSummary }),
  })
}

/**
 * Remove the model's scratchpad from a summary before it is stored.
 *
 * Storing the reasoning would spend the compacted window on notes the next turn cannot use, and a
 * response truncated inside an unterminated block would otherwise persist as a half-written thought.
 */
export function stripCompactionAnalysis(summary: string): string {
  if (typeof summary !== 'string') return ''
  let text = summary
  for (;;) {
    const open = text.indexOf(COMPACTION_ANALYSIS_OPEN_TAG)
    if (open < 0) break
    const close = text.indexOf(COMPACTION_ANALYSIS_CLOSE_TAG, open + COMPACTION_ANALYSIS_OPEN_TAG.length)
    if (close < 0) {
      text = text.slice(0, open)
      break
    }
    text = `${text.slice(0, open)}${text.slice(close + COMPACTION_ANALYSIS_CLOSE_TAG.length)}`
  }
  return text.trim()
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

const TOOL_ARGUMENT_PREVIEW_LIMIT = 200
const TOOL_OUTCOME_PREVIEW_LIMIT = 400

function toolResultsById(messages: readonly ContextMessage[]): Map<string, ContextMessage> {
  const results = new Map<string, ContextMessage>()
  for (const message of messages) {
    if (message.role !== 'tool') continue
    const identifier = typeof message.tool_call_id === 'string' ? message.tool_call_id : ''
    if (!identifier || results.has(identifier)) continue
    results.set(identifier, message)
  }
  return results
}

function renderToolCalls(
  toolCalls: unknown,
  resultsById: ReadonlyMap<string, ContextMessage>,
  folded: Set<string>,
): string[] {
  if (toolCalls === undefined || toolCalls === null) return []
  // Unknown shapes still have to reach the summarizer; falling back to the old dump is lossy
  // in prompt budget but never lossy in content.
  if (!Array.isArray(toolCalls)) return [`tool_calls=${stableJson(toolCalls)}`]
  return toolCalls.map(call => {
    if (!isRecord(call)) return `called ${truncatePreview(String(call), TOOL_ARGUMENT_PREVIEW_LIMIT)}`
    const fn = isRecord(call.function) ? call.function : undefined
    const name = firstText(fn?.name, call.name, call.tool_name) || 'unknown_tool'
    const identifier = firstText(call.id, call.tool_call_id)
    const result = identifier ? resultsById.get(identifier) : undefined
    if (identifier && result !== undefined) folded.add(identifier)
    return `called ${name}(${shortArguments(fn?.arguments ?? call.arguments ?? call.input ?? call.args)}) `
      + `-> ${toolOutcome(result)}`
  })
}

function toolOutcome(result: ContextMessage | undefined): string {
  if (result === undefined) return '(no result in this slice)'
  const failed = result.is_error === true || result.isError === true || result.status === 'error'
  const text = collapseWhitespace(messageContentToText(result.content))
  if (!text) return failed ? 'error (no output)' : 'ok (no output)'
  const preview = truncatePreview(text, TOOL_OUTCOME_PREVIEW_LIMIT)
  return failed ? `error: ${preview}` : preview
}

function shortArguments(value: unknown): string {
  if (value === undefined || value === null) return ''
  const text = collapseWhitespace(typeof value === 'string' ? value : stableJson(value))
  return truncatePreview(text, TOOL_ARGUMENT_PREVIEW_LIMIT)
}

function collapseWhitespace(text: string): string {
  return text.replace(/\s+/g, ' ').trim()
}

function truncatePreview(text: string, limit: number): string {
  if (text.length <= limit) return text
  return `${text.slice(0, limit)}… (+${text.length - limit} chars)`
}

function firstText(...candidates: readonly unknown[]): string {
  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.trim()) return candidate.trim()
  }
  return ''
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
