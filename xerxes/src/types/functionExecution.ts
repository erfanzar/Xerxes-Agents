// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  AgentSwitchTrigger,
  FUNCTION_CALLING_CAPABILITY,
} from '../agents/orchestrator.js'
import { CompactionStrategy } from '../context/compactionStrategies.js'
import {
  ExecutionStatus,
  type ExecutionStatus as RuntimeExecutionStatus,
} from '../runtime/executionRegistry.js'
import {
  isJsonObject,
  type JsonObject,
  type ToolCall,
} from './toolCalls.js'

/** Re-export the existing runtime vocabularies instead of creating competing copies. */
export {
  AgentSwitchTrigger,
  CompactionStrategy,
  ExecutionStatus,
  FUNCTION_CALLING_CAPABILITY,
}
export type { AgentCapability } from '../agents/orchestrator.js'

/** The `ExecutionStatus` union owned by `runtime/executionRegistry.ts`. */
export type FunctionExecutionStatus = RuntimeExecutionStatus

export const FunctionCallStrategy = Object.freeze({
  SEQUENTIAL: 'sequential',
  PARALLEL: 'parallel',
  CONDITIONAL: 'conditional',
  PIPELINE: 'pipeline',
} as const)

export type FunctionCallStrategy = (typeof FunctionCallStrategy)[keyof typeof FunctionCallStrategy]

/** Source for stable tool-call identifiers when an upstream provider omitted one. */
export type FunctionCallIdFactory = () => string

/** Native input shape for an in-flight tool call. */
export interface RequestFunctionCallInput {
  readonly agentId?: string
  readonly arguments: JsonObject
  /** `callId` takes precedence over `id`, matching the old dataclass invariant. */
  readonly callId?: string
  readonly dependencies?: readonly string[]
  readonly error?: string
  readonly id?: string
  readonly maxRetries?: number
  readonly name: string
  readonly result?: unknown
  readonly retryCount?: number
  readonly status?: FunctionExecutionStatus
  readonly timeout?: number
}

export interface RequestFunctionCallFactoryOptions {
  readonly idFactory?: FunctionCallIdFactory
}

/**
 * Provider-neutral function call ready for scheduling.
 *
 * This remains an immutable value. Use `withRequestFunctionCall` to make a
 * status/result transition instead of mutating a shared in-flight record.
 */
export interface RequestFunctionCall {
  readonly agentId?: string
  readonly arguments: JsonObject
  readonly callId: string
  readonly dependencies: readonly string[]
  readonly error?: string
  readonly id: string
  readonly maxRetries: number
  readonly name: string
  readonly result?: unknown
  readonly retryCount: number
  readonly status: FunctionExecutionStatus
  readonly timeout?: number
}

/** Create a validated immutable call and synchronize its `id` and `callId`. */
export function createRequestFunctionCall(
  input: RequestFunctionCallInput,
  options: RequestFunctionCallFactoryOptions = {},
): RequestFunctionCall {
  const name = requiredText(input.name, 'name')
  if (!isJsonObject(input.arguments)) {
    throw new TypeError('arguments must be a JSON object')
  }
  const callId = preferredId(input, options.idFactory ?? defaultFunctionCallId)
  const agentId = optionalText(input.agentId, 'agentId')
  const dependencies = (input.dependencies ?? []).map((dependency, index) => requiredText(dependency, `dependencies[${index}]`))
  const timeout = optionalNonNegativeNumber(input.timeout, 'timeout')
  const retryCount = nonNegativeInteger(input.retryCount ?? 0, 'retryCount')
  const maxRetries = nonNegativeInteger(input.maxRetries ?? 3, 'maxRetries')
  const error = optionalText(input.error, 'error')

  return {
    name,
    arguments: { ...input.arguments },
    id: callId,
    callId,
    dependencies,
    retryCount,
    maxRetries,
    status: input.status ?? ExecutionStatus.PENDING,
    ...(agentId === undefined ? {} : { agentId }),
    ...(timeout === undefined ? {} : { timeout }),
    ...(input.result === undefined ? {} : { result: input.result }),
    ...(error === undefined ? {} : { error }),
  }
}

/** Return a new call value after a scheduler changes status, output, or retry metadata. */
export function withRequestFunctionCall(
  call: RequestFunctionCall,
  changes: Partial<RequestFunctionCallInput>,
): RequestFunctionCall {
  return createRequestFunctionCall({
    ...call,
    ...changes,
    id: changes.id ?? call.id,
    callId: changes.callId ?? call.callId,
  })
}

/** Convert one canonical provider tool call into a schedulable function call. */
export function requestFunctionCallFromToolCall(
  call: ToolCall,
  options: Omit<RequestFunctionCallInput, 'arguments' | 'callId' | 'id' | 'name'> = {},
): RequestFunctionCall {
  return createRequestFunctionCall({
    ...options,
    id: call.id,
    callId: call.id,
    name: call.function.name,
    arguments: call.function.arguments,
  })
}

/** Convert a function-execution request back to the single canonical provider tool-call type. */
export function toolCallFromRequestFunctionCall(call: RequestFunctionCall): ToolCall {
  return {
    id: call.callId,
    type: 'function',
    function: {
      name: call.name,
      arguments: { ...call.arguments },
    },
  }
}

/** Lightweight outcome for a single function execution, distinct from the detached runtime ledger result. */
export interface FunctionExecutionResult {
  readonly error?: string
  readonly result?: unknown
  readonly status: FunctionExecutionStatus
}

/** Context retained when orchestration switches from one agent to another. */
export interface FunctionSwitchContext {
  readonly bufferedContent?: string
  readonly executionError: boolean
  readonly functionResults: readonly FunctionExecutionResult[]
}

/** A partial provider tool call received while the model is still streaming. */
export interface ToolCallStreamChunk {
  readonly arguments?: string
  readonly functionName?: string
  readonly id: string
  readonly index?: number
  readonly isComplete: boolean
  readonly type: 'function'
}

export interface StreamChunkInput {
  readonly agentId?: string
  readonly bufferedContent?: string
  readonly bufferedReasoningContent?: string
  readonly chunk?: unknown
  readonly content?: string
  readonly functionCallsDetected?: boolean
  readonly reasoningContent?: string
  readonly reinvoked?: boolean
  readonly streamingToolCalls?: readonly ToolCallStreamChunk[]
  readonly toolCalls?: readonly ToolCallStreamChunk[]
}

/** Legacy-pipeline stream fragment without a provider SDK dependency. */
export interface StreamChunk {
  readonly agentId: string
  readonly bufferedContent?: string
  readonly bufferedReasoningContent?: string
  readonly chunk?: unknown
  readonly content?: string
  readonly functionCallsDetected?: boolean
  readonly reasoningContent?: string
  readonly reinvoked: boolean
  readonly streamingToolCalls?: readonly ToolCallStreamChunk[]
  readonly toolCalls?: readonly ToolCallStreamChunk[]
  readonly type: 'stream_chunk'
}

/** Construct a normalized stream fragment without mutating raw provider chunks. */
export function createStreamChunk(input: StreamChunkInput = {}): StreamChunk {
  return {
    type: 'stream_chunk',
    agentId: input.agentId ?? '',
    reinvoked: input.reinvoked ?? false,
    ...(input.chunk === undefined ? {} : { chunk: input.chunk }),
    ...(input.content === undefined ? {} : { content: input.content }),
    ...(input.bufferedContent === undefined ? {} : { bufferedContent: input.bufferedContent }),
    ...(input.reasoningContent === undefined ? {} : { reasoningContent: input.reasoningContent }),
    ...(input.bufferedReasoningContent === undefined ? {} : { bufferedReasoningContent: input.bufferedReasoningContent }),
    ...(input.functionCallsDetected === undefined ? {} : { functionCallsDetected: input.functionCallsDetected }),
    ...(input.toolCalls === undefined ? {} : { toolCalls: [...input.toolCalls] }),
    ...(input.streamingToolCalls === undefined ? {} : { streamingToolCalls: [...input.streamingToolCalls] }),
  }
}

/**
 * Return text from an SDK-neutral Gemini-shaped chunk, with the same fallback
 * ordering as the previous provider-specific stream wrapper.
 */
export function geminiContent(stream: StreamChunk): string | undefined {
  const rawChunk = optionalRecord(stream.chunk)
  const result = optionalRecord(rawChunk?._result)
  if (result !== undefined) {
    return typeof result.text === 'string' ? result.text : stream.content ?? ''
  }
  return stream.content || undefined
}

/** Determine whether a buffered chunk contains more opening than closing reasoning tags. */
export function streamChunkIsThinking(stream: Pick<StreamChunk, 'bufferedContent'>): boolean {
  const content = stream.bufferedContent
  if (!content) return false
  const opens = content.match(/<(think|thinking|reason|reasoning)>/gi)?.length ?? 0
  const closes = content.match(/<\/(think|thinking|reason|reasoning)>/gi)?.length ?? 0
  return opens > closes
}

export interface FunctionDetection {
  readonly agentId: string
  readonly message: string
  readonly type: 'function_detection'
}

export interface FunctionCallInfo {
  readonly id: string
  readonly name: string
}

export interface FunctionCallsExtracted {
  readonly agentId: string
  readonly functionCalls: readonly FunctionCallInfo[]
  readonly type: 'function_calls_extracted'
}

export interface FunctionExecutionStart {
  readonly agentId: string
  readonly functionId: string
  readonly functionName: string
  readonly progress: string
  readonly type: 'function_execution_start'
}

export interface FunctionExecutionComplete {
  readonly agentId: string
  readonly error?: string
  readonly functionId: string
  readonly functionName: string
  readonly result?: unknown
  readonly status: string
  readonly type: 'function_execution_complete'
}

export interface AgentSwitch {
  readonly fromAgent: string
  readonly reason: string
  readonly toAgent: string
  readonly type: 'agent_switch'
}

export interface Completion {
  readonly agentId: string
  readonly executionHistory: readonly unknown[]
  readonly finalContent: string
  readonly functionCallsExecuted: number
  readonly reasoningContent: string
  readonly type: 'completion'
}

/** Structured response output from the legacy function-execution pipeline. */
export interface ResponseResult {
  readonly agentId: string
  readonly completion?: Completion
  readonly content: string
  readonly executionHistory: readonly unknown[]
  readonly functionCalls: readonly unknown[]
  readonly reasoningContent: string
  readonly reinvoked: boolean
  readonly response?: unknown
}

export interface ReinvokeSignal {
  readonly agentId: string
  readonly message: string
  readonly type: 'reinvoke_signal'
}

/** Discriminated union retained for consumers of the legacy orchestration event stream. */
export type StreamingResponseType =
  | AgentSwitch
  | Completion
  | FunctionCallsExtracted
  | FunctionDetection
  | FunctionExecutionComplete
  | FunctionExecutionStart
  | ReinvokeSignal
  | StreamChunk

function preferredId(input: RequestFunctionCallInput, idFactory: FunctionCallIdFactory): string {
  const callId = input.callId?.trim()
  if (callId) return callId
  const id = input.id?.trim()
  if (id) return id
  return requiredText(idFactory(), 'idFactory result')
}

function requiredText(value: unknown, field: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new TypeError(`${field} must be a non-empty string`)
  }
  return value
}

function optionalText(value: unknown, field: string): string | undefined {
  if (value === undefined) return undefined
  return requiredText(value, field)
}

function optionalNonNegativeNumber(value: unknown, field: string): number | undefined {
  if (value === undefined) return undefined
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new TypeError(`${field} must be a finite number greater than or equal to zero`)
  }
  return value
}

function nonNegativeInteger(value: unknown, field: string): number {
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
    throw new TypeError(`${field} must be an integer greater than or equal to zero`)
  }
  return value
}

function optionalRecord(value: unknown): Readonly<Record<string, unknown>> | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Readonly<Record<string, unknown>>
    : undefined
}

function defaultFunctionCallId(): string {
  return `call_${crypto.randomUUID().replaceAll('-', '')}`
}
