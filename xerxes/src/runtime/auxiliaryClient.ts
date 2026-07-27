// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { QuerySource } from '../llms/client.js'

export const DEFAULT_AUXILIARY_MODEL = 'claude-haiku-4-5'
export const DEFAULT_AUXILIARY_MAX_TOKENS = 1_000

/**
 * Query source for a free-form auxiliary call that names no purpose of its own.
 *
 * Auxiliary work is never the user's turn, so the fallback must still be a
 * housekeeping source: defaulting to the main loop would hide exactly the
 * spend this dimension exists to expose.
 */
export const DEFAULT_AUXILIARY_QUERY_SOURCE: QuerySource = 'classification'

/** Query source each built-in helper tags its own call with. */
const SUMMARIZE_QUERY_SOURCE: QuerySource = 'compaction'
const TITLE_QUERY_SOURCE: QuerySource = 'session_title'
const EXTRACT_QUERY_SOURCE: QuerySource = 'memory_extraction'

const SUMMARY_INSTRUCTION = [
  'You are a context-compaction assistant. Summarize the following conversation concisely,',
  'preserving facts the model will need later. Capture decisions, user preferences, error states,',
  'and partial work. Do NOT continue the conversation. Output plain prose.',
].join(' ')

const TITLE_INSTRUCTION = 'Generate a short, descriptive title (max 8 words) for this conversation. Output only the title.'

/**
 * Character ceiling for a rendered transcript sent to the auxiliary model.
 * Compaction exists to shrink long sessions; sending the whole transcript
 * would defeat it with a cost blowout or an overflow of the small auxiliary
 * model's context. Oversized renders keep the head and tail with an explicit
 * marker so the model knows the middle was omitted.
 */
export const MAX_RENDERED_TRANSCRIPT_CHARS = 120_000

const TRANSCRIPT_TRUNCATION_MARKER = [
  '',
  '[... transcript truncated: middle portion omitted to fit the auxiliary-model context ...]',
  '',
].join('\n')

export interface AuxiliaryMessage {
  readonly content?: unknown
  readonly role?: string
  readonly [field: string]: unknown
}

/** Caller-owned description of one focused auxiliary-model request. */
export interface AuxiliaryRequest {
  readonly maxTokens?: number
  readonly messages: readonly AuxiliaryMessage[]
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly purpose: string
  /** Overrides the client's default source; built-in helpers set their own. */
  readonly querySource?: QuerySource
  readonly temperature?: number
}

/**
 * Fully resolved request delivered to an injected auxiliary backend.
 *
 * `querySource` is required rather than optional: tagging happens during
 * normalization, so a backend can forward it to the completion request and the
 * cost ledger without every call site remembering to attach it.
 */
export interface AuxiliaryBackendRequest {
  readonly maxTokens: number
  readonly messages: readonly AuxiliaryMessage[]
  readonly metadata: Readonly<Record<string, unknown>>
  readonly model: string
  readonly purpose: string
  readonly querySource: QuerySource
  readonly temperature: number
}

/** Optional token accounting supplied by an injected auxiliary backend. */
export interface AuxiliaryBackendResponse {
  readonly requestTokens?: number
  readonly responseTokens?: number
  readonly text: string
}

export type AuxiliaryBackendOutput = AuxiliaryBackendResponse | string
export type AuxiliaryBackend = (request: AuxiliaryBackendRequest) => AuxiliaryBackendOutput | Promise<AuxiliaryBackendOutput>

/** Completed auxiliary request with timing and resolved model metadata. */
export interface AuxiliaryResponse {
  readonly durationMs: number
  readonly model: string
  readonly purpose: string
  /** Source the call was billed under, so callers can attribute cost without re-deriving it. */
  readonly querySource: QuerySource
  readonly requestTokens: number
  readonly responseTokens: number
  readonly text: string
}

export interface AuxiliaryClientOptions {
  /** Required host-owned model invocation. This module never creates provider clients. */
  readonly backend: AuxiliaryBackend
  readonly defaultMaxTokens?: number
  /** Source applied to calls that name none; defaults to a housekeeping source, never `main`. */
  readonly defaultQuerySource?: QuerySource
  readonly model?: string
  /** Injectable monotonic clock for deterministic accounting and tests. */
  readonly monotonicNow?: () => number
}

export interface AuxiliarySummarizeOptions {
  readonly budgetTokens?: number
  readonly metadata?: Readonly<Record<string, unknown>>
  /** Overrides the compaction default, e.g. `tool_result_summary` for shrinking one tool reply. */
  readonly querySource?: QuerySource
  readonly temperature?: number
}

export interface AuxiliaryExtractOptions {
  readonly instruction: string
  readonly maxTokens?: number
  readonly metadata?: Readonly<Record<string, unknown>>
  /** Overrides the memory-extraction default for other extraction purposes. */
  readonly querySource?: QuerySource
  readonly temperature?: number
}

/**
 * Injected-only client for inexpensive titles, summaries, and extraction work.
 *
 * Backends may be synchronous or asynchronous. Their errors deliberately
 * propagate to the caller so auxiliary failures never masquerade as output.
 */
export class AuxiliaryClient {
  private readonly backend: AuxiliaryBackend
  private readonly defaultMaxTokens: number
  private readonly monotonicNow: () => number
  readonly model: string
  /** Source applied to calls that do not name one; always a housekeeping source. */
  readonly querySource: QuerySource

  constructor(options: AuxiliaryClientOptions) {
    if (typeof options.backend !== 'function') {
      throw new TypeError('auxiliary backend must be a function')
    }
    this.backend = options.backend
    this.model = nonEmptyString(options.model ?? DEFAULT_AUXILIARY_MODEL, 'auxiliary model')
    this.defaultMaxTokens = tokenBudget(options.defaultMaxTokens ?? DEFAULT_AUXILIARY_MAX_TOKENS, 'defaultMaxTokens')
    this.querySource = auxiliaryQuerySource(options.defaultQuerySource ?? DEFAULT_AUXILIARY_QUERY_SOURCE)
    this.monotonicNow = options.monotonicNow ?? (() => performance.now())
  }

  /** Dispatch a typed request through the injected backend. */
  async call(request: AuxiliaryRequest): Promise<AuxiliaryResponse> {
    const backendRequest = normalizeRequest(request, this.model, this.defaultMaxTokens, this.querySource)
    const startedAt = this.monotonicNow()
    const output = await this.backend(backendRequest)
    const durationMs = Math.max(0, this.monotonicNow() - startedAt)
    const response = normalizeBackendOutput(output)
    return Object.freeze({
      text: response.text,
      purpose: backendRequest.purpose,
      querySource: backendRequest.querySource,
      model: this.model,
      durationMs,
      requestTokens: response.requestTokens,
      responseTokens: response.responseTokens,
    })
  }

  /** Produce compact prose suitable for context compaction. */
  async summarize(
    messages: readonly AuxiliaryMessage[],
    options: AuxiliarySummarizeOptions = {},
  ): Promise<string> {
    const response = await this.call({
      purpose: 'summarize',
      querySource: options.querySource ?? SUMMARIZE_QUERY_SOURCE,
      messages: [
        { role: 'system', content: SUMMARY_INSTRUCTION },
        { role: 'user', content: renderMessages(messages) },
      ],
      ...(options.budgetTokens === undefined ? {} : { maxTokens: options.budgetTokens }),
      ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
      ...(options.metadata === undefined ? {} : { metadata: options.metadata }),
    })
    return response.text
  }

  /** Generate a short, display-safe title from the opening conversation turns. */
  async title(firstTurns: readonly AuxiliaryMessage[]): Promise<string> {
    const response = await this.call({
      purpose: 'title',
      querySource: TITLE_QUERY_SOURCE,
      messages: [
        { role: 'system', content: TITLE_INSTRUCTION },
        { role: 'user', content: renderMessages(firstTurns) },
      ],
      maxTokens: 64,
    })
    return stripTitleDelimiters(response.text)
  }

  /** Run a single extraction using an explicit caller-provided instruction. */
  async extract(text: string, options: AuxiliaryExtractOptions): Promise<string> {
    const instruction = nonEmptyString(options.instruction, 'extraction instruction')
    if (typeof text !== 'string') {
      throw new TypeError('extraction text must be a string')
    }
    const response = await this.call({
      purpose: 'extract',
      querySource: options.querySource ?? EXTRACT_QUERY_SOURCE,
      messages: [
        { role: 'system', content: instruction },
        { role: 'user', content: text },
      ],
      ...(options.maxTokens === undefined ? {} : { maxTokens: options.maxTokens }),
      ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
      ...(options.metadata === undefined ? {} : { metadata: options.metadata }),
    })
    return response.text
  }
}

function normalizeRequest(
  request: AuxiliaryRequest,
  model: string,
  defaultMaxTokens: number,
  defaultQuerySource: QuerySource,
): AuxiliaryBackendRequest {
  if (!isRecord(request)) {
    throw new TypeError('auxiliary request must be an object')
  }
  const purpose = nonEmptyString(request.purpose, 'auxiliary request purpose')
  const querySource = auxiliaryQuerySource(request.querySource ?? defaultQuerySource)
  if (!Array.isArray(request.messages)) {
    throw new TypeError('auxiliary request messages must be an array')
  }
  const maxTokens = request.maxTokens === undefined
    ? defaultMaxTokens
    : tokenBudget(request.maxTokens, 'auxiliary request maxTokens')
  const temperature = request.temperature === undefined
    ? 0
    : nonNegativeFiniteNumber(request.temperature, 'auxiliary request temperature')
  return Object.freeze({
    purpose,
    querySource,
    messages: Object.freeze(request.messages.map((message, index) => copyMessage(message, index))),
    maxTokens,
    temperature,
    metadata: copyMetadata(request.metadata),
    model,
  })
}

/**
 * Reject the main-loop source on an auxiliary call.
 *
 * Auxiliary work is housekeeping by definition; billing it as `main` would
 * hide it inside the user's turn, which is exactly the blindness the source
 * dimension exists to remove. The literal is compared locally rather than via
 * the llms helper so this module keeps its type-only dependency on the
 * provider layer and never pulls provider adapters into its module graph.
 */
function auxiliaryQuerySource(value: QuerySource): QuerySource {
  if (typeof value !== 'string' || !value.trim()) {
    throw new TypeError('auxiliary query source must be a non-empty string')
  }
  if (value === 'main') {
    throw new RangeError("auxiliary query source must not be 'main'; auxiliary calls are housekeeping")
  }
  return value
}

function normalizeBackendOutput(output: AuxiliaryBackendOutput): {
  readonly requestTokens: number
  readonly responseTokens: number
  readonly text: string
} {
  if (typeof output === 'string') {
    return { text: output, requestTokens: 0, responseTokens: 0 }
  }
  if (!isRecord(output) || typeof output.text !== 'string') {
    throw new TypeError('auxiliary backend must return a string or an object with string text')
  }
  return {
    text: output.text,
    requestTokens: optionalTokenCount(output.requestTokens, 'auxiliary backend requestTokens'),
    responseTokens: optionalTokenCount(output.responseTokens, 'auxiliary backend responseTokens'),
  }
}

function copyMessage(message: AuxiliaryMessage, index: number): AuxiliaryMessage {
  if (!isRecord(message)) {
    throw new TypeError(`auxiliary request message ${index} must be an object`)
  }
  if (message.role !== undefined && typeof message.role !== 'string') {
    throw new TypeError(`auxiliary request message ${index} role must be a string`)
  }
  return Object.freeze({ ...message })
}

function copyMetadata(metadata: Readonly<Record<string, unknown>> | undefined): Readonly<Record<string, unknown>> {
  if (metadata === undefined) return EMPTY_METADATA
  if (!isRecord(metadata)) {
    throw new TypeError('auxiliary request metadata must be an object')
  }
  return Object.freeze({ ...metadata })
}

function renderMessages(
  messages: readonly AuxiliaryMessage[],
  maximumChars: number = MAX_RENDERED_TRANSCRIPT_CHARS,
): string {
  if (!Array.isArray(messages)) {
    throw new TypeError('auxiliary messages must be an array')
  }
  const rendered = messages.map((message, index) => {
    if (!isRecord(message)) {
      throw new TypeError(`auxiliary message ${index} must be an object`)
    }
    if (message.role !== undefined && typeof message.role !== 'string') {
      throw new TypeError(`auxiliary message ${index} role must be a string`)
    }
    const role = message.role ?? 'user'
    const content = message.content === undefined ? '' : message.content
    return `[${role}] ${typeof content === 'string' ? content : String(content)}`
  }).join('\n')
  return clipRenderedTranscript(rendered, maximumChars)
}

/**
 * Bound a rendered transcript to a character budget, keeping the head and
 * tail (newest turns are usually the most relevant) joined by an explicit
 * truncation marker. A non-positive or non-finite budget disables clipping.
 */
function clipRenderedTranscript(rendered: string, maximumChars: number): string {
  if (!Number.isFinite(maximumChars) || maximumChars <= 0 || rendered.length <= maximumChars) {
    return rendered
  }
  if (maximumChars <= TRANSCRIPT_TRUNCATION_MARKER.length + 2) {
    return rendered.slice(0, maximumChars)
  }
  const available = maximumChars - TRANSCRIPT_TRUNCATION_MARKER.length
  const headLength = Math.ceil(available / 2)
  const tailLength = available - headLength
  return rendered.slice(0, headLength) + TRANSCRIPT_TRUNCATION_MARKER + rendered.slice(rendered.length - tailLength)
}

function stripTitleDelimiters(text: string): string {
  return text.trim().replace(/^["'`]+/, '').replace(/["'`]+$/, '').trim()
}

function nonEmptyString(value: string, label: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new TypeError(`${label} must be a non-empty string`)
  }
  return value.trim()
}

function tokenBudget(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${label} must be a positive safe integer`)
  }
  return value
}

function optionalTokenCount(value: unknown, label: string): number {
  if (value === undefined) return 0
  return tokenBudgetOrZero(value, label)
}

function tokenBudgetOrZero(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative safe integer`)
  }
  return value
}

function nonNegativeFiniteNumber(value: number, label: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`${label} must be a finite non-negative number`)
  }
  return value
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

const EMPTY_METADATA: Readonly<Record<string, unknown>> = Object.freeze({})
