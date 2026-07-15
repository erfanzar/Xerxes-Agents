// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { FunctionExecutionError } from '../core/errors.js'
import type { ToolExecutor } from '../executors/toolRegistry.js'
import {
  createLlmClient,
  type LlmClient,
  type OpenAiCompatibleClientOptions,
} from '../llms/client.js'
import type { ProviderOverrides } from '../llms/providerRegistry.js'
import type { StreamEvent } from './events.js'
import { runTurn, type TurnDependencies, type TurnRequest } from './loop.js'
import type { ThinkingPart, ThinkingStreamParser } from './thinkingParser.js'

/** The small default keeps diagnostics from accidentally becoming full production runs. */
export const DEFAULT_DEBUG_MAX_TOOL_TURNS = 8
/** Hard ceiling for the deliberately bounded debug loop. */
export const MAX_DEBUG_TOOL_TURNS = 10

/**
 * A diagnostic turn deliberately excludes production-only objective continuation
 * and retry options. Its only thinking syntax is `<think>...</think>`.
 */
export interface DebugTurnRequest extends Omit<TurnRequest, 'interactionMode' | 'maxToolTurns'> {
  readonly maxToolTurns?: number
}

/**
 * Dependencies accepted by the diagnostic loop. Retry policy and parser choice
 * are fixed so a reproduction cannot silently drift into production behavior.
 */
export interface DebugTurnDependencies extends Omit<TurnDependencies, 'retryDelays' | 'thinkingParserFactory' | 'toolExecutor'> {
  readonly toolExecutor?: ToolExecutor
}

/** Provider options accepted by the cache-free native debug-client helper. */
export type DebugLlmClientOptions = Omit<OpenAiCompatibleClientOptions, 'promptCaching' | 'providerName'>

/**
 * Construct a native provider client for a debug reproduction.
 *
 * Anthropic's stable-prefix cache breakpoints are explicitly disabled. Other
 * providers do not use the Xerxes prompt-cache adapter, so they use the normal
 * native client implementation unchanged.
 */
export function createDebugLlmClient(
  model: string,
  overrides: ProviderOverrides = {},
  options: DebugLlmClientOptions = {},
): LlmClient {
  return createLlmClient(model, overrides, { ...options, promptCaching: false })
}

/**
 * Native diagnostic sibling of {@link runTurn}.
 *
 * This is intentionally a narrow composition layer: no prompt caching through
 * {@link createDebugLlmClient}, no provider retries, a bounded number of tool
 * turns, and the single-tag {@link DebugThinkingParser}. It preserves the
 * production event and permission vocabulary so callers can compare both loops
 * without a protocol adapter.
 */
export async function* runDebugTurn(
  request: DebugTurnRequest,
  dependencies: DebugTurnDependencies,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  if (signal?.aborted) {
    return
  }

  const debugRequest: TurnRequest = {
    ...request,
    interactionMode: 'code',
    maxToolTurns: boundedDebugToolTurns(request.maxToolTurns),
  }
  const debugDependencies: TurnDependencies = {
    ...dependencies,
    retryDelays: [],
    thinkingParserFactory: () => new DebugThinkingParser(),
    toolExecutor: dependencies.toolExecutor ?? unavailableDebugToolExecutor,
  }

  for await (const event of runTurn(debugRequest, debugDependencies, signal)) {
    yield event
  }
}

/**
 * Incrementally split only `<think>...</think>` tags.
 *
 * Keeping `<thinking>` literal is intentional: it lets a diagnostic run isolate
 * differences caused by the production loop's wider tag compatibility.
 */
export class DebugThinkingParser implements ThinkingStreamParser {
  private static readonly closeTag = '</think>'
  private static readonly openTag = '<think>'

  private buffer = ''
  private inThinking = false
  private thinkingBuffer = ''

  process(chunk: string): readonly ThinkingPart[] {
    const parts: ThinkingPart[] = []
    this.buffer += chunk
    const finalFlush = chunk.length === 0

    if (finalFlush && this.inThinking) {
      this.thinkingBuffer += this.buffer
      this.buffer = ''
      if (this.thinkingBuffer) {
        parts.push({ type: 'thinking', text: this.thinkingBuffer })
      }
      this.thinkingBuffer = ''
      this.inThinking = false
      return parts
    }

    while (this.buffer) {
      if (!this.inThinking) {
        const index = this.buffer.indexOf(DebugThinkingParser.openTag)
        if (index < 0) {
          const heldBack = finalFlush ? 0 : partialTagTail(this.buffer, DebugThinkingParser.openTag)
          const visible = heldBack ? this.buffer.slice(0, -heldBack) : this.buffer
          if (visible) {
            parts.push({ type: 'text', text: visible })
          }
          this.buffer = heldBack ? this.buffer.slice(-heldBack) : ''
          break
        }
        if (index > 0) {
          parts.push({ type: 'text', text: this.buffer.slice(0, index) })
        }
        this.buffer = this.buffer.slice(index + DebugThinkingParser.openTag.length)
        this.inThinking = true
        this.thinkingBuffer = ''
        continue
      }

      const index = this.buffer.indexOf(DebugThinkingParser.closeTag)
      if (index < 0) {
        const heldBack = finalFlush ? 0 : partialTagTail(this.buffer, DebugThinkingParser.closeTag)
        this.thinkingBuffer += heldBack ? this.buffer.slice(0, -heldBack) : this.buffer
        this.buffer = heldBack ? this.buffer.slice(-heldBack) : ''
        break
      }
      if (index > 0) {
        this.thinkingBuffer += this.buffer.slice(0, index)
      }
      this.buffer = this.buffer.slice(index + DebugThinkingParser.closeTag.length)
      this.inThinking = false
      if (this.thinkingBuffer) {
        parts.push({ type: 'thinking', text: this.thinkingBuffer })
        this.thinkingBuffer = ''
      }
    }

    return parts
  }
}

function boundedDebugToolTurns(value: number | undefined): number {
  const turns = value ?? DEFAULT_DEBUG_MAX_TOOL_TURNS
  if (!Number.isSafeInteger(turns) || turns < 1 || turns > MAX_DEBUG_TOOL_TURNS) {
    throw new RangeError(`debug maxToolTurns must be an integer from 1 to ${MAX_DEBUG_TOOL_TURNS}`)
  }
  return turns
}

const unavailableDebugToolExecutor: ToolExecutor = {
  async execute(call): Promise<string> {
    throw new FunctionExecutionError(call.function.name, 'no tool executor is configured for the diagnostic loop')
  },
}

function partialTagTail(value: string, tag: string): number {
  const limit = Math.min(value.length, tag.length - 1)
  for (let size = limit; size > 0; size -= 1) {
    if (value.endsWith(tag.slice(0, size))) {
      return size
    }
  }
  return 0
}
