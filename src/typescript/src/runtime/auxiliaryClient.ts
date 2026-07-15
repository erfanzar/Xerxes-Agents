// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export const DEFAULT_AUXILIARY_MODEL = 'claude-haiku-4-5'
export const DEFAULT_AUXILIARY_MAX_TOKENS = 1_000

const SUMMARY_INSTRUCTION = [
  'You are a context-compaction assistant. Summarize the following conversation concisely,',
  'preserving facts the model will need later. Capture decisions, user preferences, error states,',
  'and partial work. Do NOT continue the conversation. Output plain prose.',
].join(' ')

const TITLE_INSTRUCTION = 'Generate a short, descriptive title (max 8 words) for this conversation. Output only the title.'

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
  readonly temperature?: number
}

/** Fully resolved request delivered to an injected auxiliary backend. */
export interface AuxiliaryBackendRequest {
  readonly maxTokens: number
  readonly messages: readonly AuxiliaryMessage[]
  readonly metadata: Readonly<Record<string, unknown>>
  readonly model: string
  readonly purpose: string
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
  readonly requestTokens: number
  readonly responseTokens: number
  readonly text: string
}

export interface AuxiliaryClientOptions {
  /** Required host-owned model invocation. This module never creates provider clients. */
  readonly backend: AuxiliaryBackend
  readonly defaultMaxTokens?: number
  readonly model?: string
  /** Injectable monotonic clock for deterministic accounting and tests. */
  readonly monotonicNow?: () => number
}

export interface AuxiliarySummarizeOptions {
  readonly budgetTokens?: number
  readonly metadata?: Readonly<Record<string, unknown>>
  readonly temperature?: number
}

export interface AuxiliaryExtractOptions {
  readonly instruction: string
  readonly maxTokens?: number
  readonly metadata?: Readonly<Record<string, unknown>>
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

  constructor(options: AuxiliaryClientOptions) {
    if (typeof options.backend !== 'function') {
      throw new TypeError('auxiliary backend must be a function')
    }
    this.backend = options.backend
    this.model = nonEmptyString(options.model ?? DEFAULT_AUXILIARY_MODEL, 'auxiliary model')
    this.defaultMaxTokens = tokenBudget(options.defaultMaxTokens ?? DEFAULT_AUXILIARY_MAX_TOKENS, 'defaultMaxTokens')
    this.monotonicNow = options.monotonicNow ?? (() => performance.now())
  }

  /** Dispatch a typed request through the injected backend. */
  async call(request: AuxiliaryRequest): Promise<AuxiliaryResponse> {
    const backendRequest = normalizeRequest(request, this.model, this.defaultMaxTokens)
    const startedAt = this.monotonicNow()
    const output = await this.backend(backendRequest)
    const durationMs = Math.max(0, this.monotonicNow() - startedAt)
    const response = normalizeBackendOutput(output)
    return Object.freeze({
      text: response.text,
      purpose: backendRequest.purpose,
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
): AuxiliaryBackendRequest {
  if (!isRecord(request)) {
    throw new TypeError('auxiliary request must be an object')
  }
  const purpose = nonEmptyString(request.purpose, 'auxiliary request purpose')
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
    messages: Object.freeze(request.messages.map((message, index) => copyMessage(message, index))),
    maxTokens,
    temperature,
    metadata: copyMetadata(request.metadata),
    model,
  })
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

function renderMessages(messages: readonly AuxiliaryMessage[]): string {
  if (!Array.isArray(messages)) {
    throw new TypeError('auxiliary messages must be an array')
  }
  return messages.map((message, index) => {
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
