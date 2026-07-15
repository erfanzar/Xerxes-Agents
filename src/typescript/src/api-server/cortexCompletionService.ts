// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { messageText, type ChatMessage } from '../types/messages.js'

/** Cortex execution strategies supported by the completion adapter. */
export const CortexProcessType = {
  HIERARCHICAL: 'hierarchical',
  PARALLEL: 'parallel',
  SEQUENTIAL: 'sequential',
} as const

export type CortexProcessType = (typeof CortexProcessType)[keyof typeof CortexProcessType]

/** Optional OpenAI request extension used to choose Cortex execution behavior. */
export interface CortexCompletionMetadata {
  readonly background?: string
  readonly process_type?: string
  readonly task_mode?: boolean
}

/** The chat-completion fields consumed by the Cortex adapter. */
export interface CortexCompletionRequest {
  readonly messages: readonly ChatMessage[]
  readonly metadata?: CortexCompletionMetadata
  readonly model: string
}

/** Model- and metadata-derived execution settings. */
export interface CortexCompletionConfig {
  readonly background?: string
  readonly processType: CortexProcessType
  readonly taskMode: boolean
}

/** Shared data supplied to a Cortex execution boundary. */
export interface CortexExecutionRequest {
  readonly model: string
  readonly processType: CortexProcessType
  readonly prompt: string
}

/** Task-mode execution additionally receives optional task-planning background. */
export interface CortexTaskExecutionRequest extends CortexExecutionRequest {
  readonly background?: string
}

/** Output from a completed Cortex task that can contribute to a final answer. */
export interface CortexExecutionTaskOutput {
  readonly output: string
  readonly status?: 'failed' | 'skipped' | 'succeeded'
}

/**
 * A completed native Cortex execution.
 *
 * `rawOutput` is the terminal orchestration result. When it is not available,
 * the service aggregates successful `taskOutputs`; `output` supports execution
 * ports that expose one direct instruction result instead.
 */
export interface CortexExecutionResult {
  readonly output?: string
  readonly rawOutput?: string
  readonly taskOutputs?: readonly CortexExecutionTaskOutput[]
}

/** A tool call observed in a Cortex stream chunk. */
export interface CortexStreamingToolCall {
  readonly arguments: unknown
  readonly name: string
}

export interface CortexStreamChunkEvent {
  readonly content?: string
  readonly toolCalls?: readonly CortexStreamingToolCall[]
  readonly type: 'stream_chunk'
}

export interface CortexFunctionDetectionEvent {
  readonly message: string
  readonly type: 'function_detection'
}

export interface CortexFunctionsExtractedEvent {
  readonly functions: readonly string[]
  readonly type: 'functions_extracted'
}

export interface CortexFunctionStartEvent {
  readonly functionName: string
  readonly progress?: unknown
  readonly type: 'function_start'
}

export interface CortexFunctionCompleteEvent {
  readonly error?: string
  readonly functionName: string
  readonly result?: unknown
  readonly status: string
  readonly type: 'function_complete'
}

export interface CortexAgentSwitchEvent {
  readonly fromAgent: string
  readonly reason?: string
  readonly toAgent: string
  readonly type: 'agent_switch'
}

export interface CortexReinvokeEvent {
  readonly message: string
  readonly type: 'reinvoke'
}

export interface CortexCompletionEvent {
  readonly functionCallsExecuted?: number
  readonly type: 'completion'
}

/** The native Cortex streaming vocabulary exposed to the API layer. */
export type CortexStreamEvent =
  | CortexAgentSwitchEvent
  | CortexCompletionEvent
  | CortexFunctionCompleteEvent
  | CortexFunctionDetectionEvent
  | CortexFunctionStartEvent
  | CortexFunctionsExtractedEvent
  | CortexReinvokeEvent
  | CortexStreamChunkEvent

/**
 * Explicit execution boundary for Cortex task and instruction modes.
 *
 * The completion service owns protocol translation only. An application wires
 * planning, agent selection, persistence, and actual execution behind this
 * port; no Python bridge, shell process, or implicit global runtime is used.
 */
export interface CortexExecutionPort {
  readonly executeInstruction: (
    request: CortexExecutionRequest,
    signal?: AbortSignal,
  ) => Promise<CortexExecutionResult>
  readonly executeTask: (
    request: CortexTaskExecutionRequest,
    signal?: AbortSignal,
  ) => Promise<CortexExecutionResult>
  readonly streamInstruction: (
    request: CortexExecutionRequest,
    signal?: AbortSignal,
  ) => AsyncIterable<CortexStreamEvent>
  readonly streamTask: (
    request: CortexTaskExecutionRequest,
    signal?: AbortSignal,
  ) => AsyncIterable<CortexStreamEvent>
}

export interface CortexOpenAiUsage {
  readonly completion_tokens: number
  readonly prompt_tokens: number
  readonly total_tokens: number
}

export interface CortexOpenAiCompletion {
  readonly choices: readonly [{
    readonly finish_reason: 'stop'
    readonly index: 0
    readonly message: {
      readonly content: string
      readonly role: 'assistant'
    }
  }]
  readonly created: number
  readonly id: string
  readonly model: string
  readonly object: 'chat.completion'
  readonly usage: CortexOpenAiUsage
}

export interface CortexOpenAiStreamMetadata {
  readonly [key: string]: unknown
  readonly event?: string
}

export interface CortexOpenAiStreamFrame {
  readonly choices: readonly [{
    readonly delta: {
      readonly content?: string
      readonly role?: 'assistant'
    }
    readonly finish_reason: 'stop' | null
    readonly index: 0
  }]
  readonly created: number
  readonly id: string
  readonly metadata?: CortexOpenAiStreamMetadata
  readonly model: string
  readonly object: 'chat.completion.chunk'
  readonly usage: CortexOpenAiUsage
}

export interface CortexCompletionServiceOptions {
  readonly execution: CortexExecutionPort
  /** Millisecond clock used to generate OpenAI epoch-second timestamps. */
  readonly now?: () => number
  /** Injectable id source so one stream shares a stable OpenAI completion id. */
  readonly responseId?: () => string
}

interface CortexStreamPresentation {
  readonly content?: string
  readonly metadata?: CortexOpenAiStreamMetadata
}

/**
 * OpenAI-compatible completion adapter over an injected native Cortex runtime.
 *
 * It intentionally never guesses an execution result: normal and streaming
 * failures are allowed to propagate from the execution port to the HTTP layer.
 */
export class CortexCompletionService {
  private readonly execution: CortexExecutionPort
  private readonly now: () => number
  private readonly responseId: () => string

  constructor(options: CortexCompletionServiceOptions) {
    this.execution = options.execution
    this.now = options.now ?? Date.now
    this.responseId = options.responseId ?? (() => `chatcmpl-cortex-${crypto.randomUUID()}`)
  }

  /** Execute one Cortex request and return an OpenAI chat-completion envelope. */
  async createCompletion(request: CortexCompletionRequest, signal?: AbortSignal): Promise<CortexOpenAiCompletion> {
    throwIfAborted(signal)
    const config = deriveCortexCompletionConfig(request)
    const prompt = latestUserPrompt(request.messages)
    const result = config.taskMode
      ? await this.execution.executeTask(taskExecutionRequest(request.model, prompt, config), signal)
      : await this.execution.executeInstruction(instructionExecutionRequest(request.model, prompt, config), signal)
    throwIfAborted(signal)

    const content = aggregateCortexExecution(result)
    return {
      id: this.responseId(),
      object: 'chat.completion',
      created: this.epochSeconds(),
      model: responseModel(request.model),
      choices: [{
        index: 0,
        message: { role: 'assistant', content },
        finish_reason: 'stop',
      }],
      usage: approximateCortexUsage(prompt, content),
    }
  }

  /**
   * Convert an async Cortex event stream into OpenAI `data:` frames.
   *
   * The final stop frame and `[DONE]` sentinel are emitted only after the
   * injected stream completes normally. An execution error remains an error
   * instead of being converted into a successful-looking completion.
   */
  async *createStreamingCompletion(
    request: CortexCompletionRequest,
    signal?: AbortSignal,
  ): AsyncGenerator<string> {
    throwIfAborted(signal)
    const config = deriveCortexCompletionConfig(request)
    const prompt = latestUserPrompt(request.messages)
    const stream = config.taskMode
      ? this.execution.streamTask(taskExecutionRequest(request.model, prompt, config), signal)
      : this.execution.streamInstruction(instructionExecutionRequest(request.model, prompt, config), signal)
    const id = this.responseId()
    const created = this.epochSeconds()
    const model = responseModel(request.model)
    let emitted = 0

    for await (const event of stream) {
      throwIfAborted(signal)
      const presentation = presentCortexStreamEvent(event)
      if (!presentation) continue
      const delta: { content?: string; role?: 'assistant' } = {
        ...(emitted === 0 ? { role: 'assistant' as const } : {}),
        ...(presentation.content === undefined ? {} : { content: presentation.content }),
      }
      const frame: CortexOpenAiStreamFrame = {
        id,
        object: 'chat.completion.chunk',
        created,
        model,
        choices: [{ index: 0, delta, finish_reason: null }],
        usage: zeroUsage(),
        ...(presentation.metadata === undefined ? {} : { metadata: presentation.metadata }),
      }
      yield sseFrame(frame)
      emitted += 1
    }

    throwIfAborted(signal)
    yield sseFrame({
      id,
      object: 'chat.completion.chunk',
      created,
      model,
      choices: [{ index: 0, delta: { content: '' }, finish_reason: 'stop' }],
      usage: zeroUsage(),
    })
    yield 'data: [DONE]\n\n'
  }

  private epochSeconds(): number {
    return Math.floor(this.now() / 1_000)
  }
}

/** Derive task mode, process strategy, and task background from a request. */
export function deriveCortexCompletionConfig(request: Pick<CortexCompletionRequest, 'metadata' | 'model'>): CortexCompletionConfig {
  const model = request.model.toLowerCase()
  let taskMode = model.includes('task')
  let processType: CortexProcessType = model.includes('parallel')
    ? CortexProcessType.PARALLEL
    : model.includes('hierarchical')
      ? CortexProcessType.HIERARCHICAL
      : CortexProcessType.SEQUENTIAL

  const metadata = request.metadata
  if (typeof metadata?.task_mode === 'boolean') {
    taskMode = metadata.task_mode
  }
  if (typeof metadata?.process_type === 'string') {
    processType = parseCortexProcessType(metadata.process_type) ?? processType
  }
  return {
    taskMode,
    processType,
    ...(typeof metadata?.background === 'string' ? { background: metadata.background } : {}),
  }
}

/** Select the latest user message; retain the full transcript as a fallback. */
export function latestUserPrompt(messages: readonly ChatMessage[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message?.role === 'user') return messageText(message)
  }
  return messages.map(messageText).join('\n')
}

/** Prefer a terminal Cortex result, then aggregate successful task outputs. */
export function aggregateCortexExecution(result: CortexExecutionResult): string {
  if (result.rawOutput?.trim()) return result.rawOutput
  if (result.output?.trim()) return result.output
  return (result.taskOutputs ?? [])
    .filter(task => task.status === undefined || task.status === 'succeeded')
    .map(task => task.output)
    .filter(Boolean)
    .join('\n\n')
}

/** Approximate token usage consistently for Cortex executions without provider counters. */
export function approximateCortexUsage(prompt: string, content: string): CortexOpenAiUsage {
  const promptTokens = whitespaceTokenCount(prompt)
  const completionTokens = whitespaceTokenCount(content)
  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: promptTokens + completionTokens,
  }
}

/** Format one OpenAI SSE event frame. */
export function sseFrame(frame: CortexOpenAiStreamFrame): string {
  return `data: ${JSON.stringify(frame)}\n\n`
}

function taskExecutionRequest(model: string, prompt: string, config: CortexCompletionConfig): CortexTaskExecutionRequest {
  return {
    model: responseModel(model),
    prompt,
    processType: config.processType,
    ...(config.background === undefined ? {} : { background: config.background }),
  }
}

function instructionExecutionRequest(model: string, prompt: string, config: CortexCompletionConfig): CortexExecutionRequest {
  return { model: responseModel(model), prompt, processType: config.processType }
}

function presentCortexStreamEvent(event: CortexStreamEvent): CortexStreamPresentation | undefined {
  switch (event.type) {
    case 'stream_chunk': {
      const metadata = event.toolCalls?.length
        ? { tool_calls: event.toolCalls.map(toolCall => ({ name: toolCall.name, arguments: toolCall.arguments })) }
        : undefined
      if (!event.content && !metadata) return undefined
      return {
        ...(event.content ? { content: event.content } : {}),
        ...(metadata === undefined ? {} : { metadata }),
      }
    }
    case 'function_detection':
      return {
        content: `\n**Detecting functions: ${event.message}**\n`,
        metadata: { event: 'function_detection' },
      }
    case 'functions_extracted':
      return {
        content: `\n*Functions to execute: ${event.functions.join(', ')}*\n`,
        metadata: { event: 'functions_extracted', functions: [...event.functions] },
      }
    case 'function_start':
      return {
        content: `\n⚡ Executing ${event.functionName}...\n`,
        metadata: {
          event: 'function_start',
          function: event.functionName,
          ...(event.progress === undefined ? {} : { progress: event.progress }),
        },
      }
    case 'function_complete': {
      let content = `\n*${event.functionName} completed*\n`
      const metadata: Record<string, unknown> = {
        event: 'function_complete',
        function: event.functionName,
        status: event.status,
      }
      if (event.result !== undefined && event.result !== null) {
        content += `   Result: ${truncateStreamValue(event.result)}\n`
        metadata.has_result = true
      } else if (event.error) {
        content += `   Error: ${event.error}\n`
        metadata.error = event.error
      }
      return { content, metadata }
    }
    case 'agent_switch':
      return {
        content: `\n*Switching from ${event.fromAgent} to ${event.toAgent}*${event.reason === undefined ? '' : `\n   Reason: ${event.reason}`}\n`,
        metadata: {
          event: 'agent_switch',
          from_agent: event.fromAgent,
          to_agent: event.toAgent,
          ...(event.reason === undefined ? {} : { reason: event.reason }),
        },
      }
    case 'reinvoke':
      return { content: `\n*Reinvoke* ${event.message}\n`, metadata: { event: 'reinvoke' } }
    case 'completion':
      return {
        content: '\n*Task completed*\n',
        metadata: { event: 'completion', functions_executed: event.functionCallsExecuted ?? 0 },
      }
  }
}

function parseCortexProcessType(value: string): CortexProcessType | undefined {
  switch (value.trim().toLowerCase()) {
    case CortexProcessType.SEQUENTIAL:
      return CortexProcessType.SEQUENTIAL
    case CortexProcessType.PARALLEL:
      return CortexProcessType.PARALLEL
    case CortexProcessType.HIERARCHICAL:
      return CortexProcessType.HIERARCHICAL
    default:
      return undefined
  }
}

function responseModel(model: string): string {
  return model || 'cortex'
}

function truncateStreamValue(value: unknown): string {
  const rendered = renderStreamValue(value)
  return rendered.length > 100 ? `${rendered.slice(0, 100)}...` : rendered
}

function renderStreamValue(value: unknown): string {
  if (typeof value === 'string') return value
  try {
    const encoded = JSON.stringify(value)
    if (encoded !== undefined) return encoded
  } catch {
    // Fall through to JavaScript's stable string conversion for circular values.
  }
  return String(value)
}

function whitespaceTokenCount(value: string): number {
  const normalized = value.trim()
  return normalized ? normalized.split(/\s+/).length : 0
}

function zeroUsage(): CortexOpenAiUsage {
  return { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return
  throw signal.reason instanceof Error ? signal.reason : new Error('Cortex completion was aborted.')
}
