// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type {
  EvaluationAgent,
  EvaluationDiagnosisRequest,
  EvaluationEvent,
  EvaluationJudgePort,
  EvaluationSessionPort,
  EvaluationStartRequest,
  EvaluationTurnOptions,
  EvaluationTurnResult,
} from './types.js'

const DEFAULT_TIMEOUT_MS = 450_000
const DEFAULT_RETRIES = 2
const TRANSIENT_MARKERS = [
  'connection',
  'reset by peer',
  'econnreset',
  'broken pipe',
  'remote disconnected',
  'rate limit',
  'ratelimit',
  '429',
  '502',
  '503',
  '504',
  'service unavailable',
  'unavailable',
  'overloaded',
  'temporarily',
  'timed out',
  'read timeout',
] as const

/** Constructor-only dependencies for a native evaluation agent. */
export interface NativeEvaluationAgentOptions {
  readonly clock?: () => number
  readonly retryDelayMs?: (attempt: number) => number
  readonly sleep?: (milliseconds: number) => Promise<void>
  readonly start: EvaluationStartRequest
  readonly transport: EvaluationSessionPort
}

/**
 * Turns an explicitly supplied runtime transport into scorer-friendly telemetry.
 *
 * This class intentionally owns no provider profile, environment credential, or
 * daemon process. The caller supplies all real execution through `transport`.
 */
export class NativeEvaluationAgent implements EvaluationAgent {
  private readonly clock: () => number
  private readonly retryDelayMs: (attempt: number) => number
  private readonly sleep: (milliseconds: number) => Promise<void>
  private readonly startRequest: EvaluationStartRequest
  private readonly transport: EvaluationSessionPort
  private activeModel = '?'

  constructor(options: NativeEvaluationAgentOptions) {
    this.clock = options.clock ?? Date.now
    this.retryDelayMs = options.retryDelayMs ?? (attempt => 2_000 * attempt)
    this.sleep = options.sleep ?? (milliseconds => Bun.sleep(milliseconds))
    this.startRequest = options.start
    this.transport = options.transport
  }

  get model(): string {
    return this.activeModel
  }

  async start(): Promise<void> {
    const result = await this.transport.start(this.startRequest)
    this.activeModel = normalizedText(result.model) || '?'
  }

  async freshSession(): Promise<void> {
    await this.transport.reset()
  }

  async turn(prompt: string, options: EvaluationTurnOptions = {}): Promise<EvaluationTurnResult> {
    const timeoutMs = positiveInteger(options.timeoutMs ?? DEFAULT_TIMEOUT_MS, 'timeoutMs')
    const retries = nonNegativeInteger(options.retries ?? DEFAULT_RETRIES, 'retries')
    let attempt = 0
    let result = await this.oneTurn(requiredText(prompt, 'prompt'), timeoutMs)

    while (attempt < retries && isTransientEvaluationFailure(result)) {
      attempt += 1
      await this.sleep(this.retryDelayMs(attempt))
      result = await this.oneTurn(prompt, timeoutMs)
    }

    return { ...result, retries: attempt }
  }

  async close(): Promise<void> {
    await this.transport.close()
  }

  private async oneTurn(prompt: string, timeoutMs: number): Promise<EvaluationTurnResult> {
    const startedAt = this.clock()
    const deadline = startedAt + timeoutMs
    const controller = new AbortController()
    const text: string[] = []
    const tools: string[] = []
    let contextTokens = 0
    let error: string | undefined
    let iterator: AsyncIterator<EvaluationEvent> | undefined

    try {
      const events = await this.transport.submit({ prompt, signal: controller.signal, timeoutMs })
      iterator = events[Symbol.asyncIterator]()
      while (true) {
        const remaining = deadline - this.clock()
        if (remaining <= 0) {
          error = 'timeout'
          break
        }
        const next = await nextBeforeDeadline(iterator, remaining)
        if (next === TIMED_OUT) {
          error = 'timeout'
          break
        }
        if (next.done) break

        const event = next.value
        if (event.type === 'text') {
          text.push(event.text)
        } else if (event.type === 'tool_call') {
          tools.push(event.name)
        } else if (event.type === 'approval_request') {
          await this.transport.approve(event.id)
        } else if (event.type === 'status') {
          if (event.contextTokens !== undefined && event.contextTokens > 0) contextTokens = event.contextTokens
        } else if (event.type === 'notification' && event.severity === 'error') {
          error = truncate(event.body ?? event.title ?? 'error', 160)
        } else if (event.type === 'turn_end') {
          break
        }
      }
    } catch (caught) {
      error = truncate(errorMessage(caught), 160)
    } finally {
      controller.abort()
      if (error !== undefined && iterator?.return !== undefined) await iterator.return()
    }

    return {
      contextTokens,
      error,
      latencyMs: Math.max(0, this.clock() - startedAt),
      retries: 0,
      text: text.join('').trim(),
      tools,
    }
  }
}

/** True only for error envelopes that warrant a retry. Turn timeouts never retry. */
export function isTransientEvaluationFailure(result: Pick<EvaluationTurnResult, 'error' | 'text'>): boolean {
  const error = result.error?.toLowerCase() ?? ''
  if (error === 'timeout') return false
  const text = result.text.trim().toLowerCase()
  const envelope = error || (text.startsWith('[error:') ? text : '')
  return envelope.length > 0 && TRANSIENT_MARKERS.some(marker => envelope.includes(marker))
}

/**
 * Describe a failure without creating a provider client.
 *
 * A `judge` is an opt-in host port. Without one, the scorer returns the grader
 * detail verbatim enough to keep reports useful while making the unavailable
 * external diagnosis explicit.
 */
export async function diagnoseEvaluationFailure(
  request: EvaluationDiagnosisRequest,
  error: string | undefined,
  judge: EvaluationJudgePort | undefined,
): Promise<string> {
  if (error === 'timeout') {
    return 'timed out — the model did not finish within the turn limit (too slow, or stuck looping).'
  }
  if (error !== undefined && error.length > 0) return `the turn errored before completing: ${truncate(error, 160)}`
  if (request.reply.trim().startsWith('[Error:')) {
    return `API/transport error that persisted through retries (not a model mistake): ${truncate(request.reply.trim(), 160)}`
  }
  if (judge === undefined) return `(auto-diagnosis unavailable: no injected judge); grader: ${truncate(request.graderDetail, 140)}`

  try {
    const diagnosis = stripThinking(await judge.diagnose(request)).trim()
    return diagnosis ? truncate(diagnosis, 320) : `(model gave no diagnosis); grader: ${truncate(request.graderDetail, 120)}`
  } catch (caught) {
    return `(auto-diagnosis unavailable: ${errorName(caught)}); grader: ${truncate(request.graderDetail, 140)}`
  }
}

const TIMED_OUT = Symbol('timed out')

async function nextBeforeDeadline<T>(
  iterator: AsyncIterator<T>,
  timeoutMs: number,
): Promise<IteratorResult<T> | typeof TIMED_OUT> {
  let timeout: ReturnType<typeof setTimeout> | undefined
  try {
    return await Promise.race([
      iterator.next(),
      new Promise<typeof TIMED_OUT>(resolve => {
        timeout = setTimeout(() => resolve(TIMED_OUT), timeoutMs)
      }),
    ])
  } finally {
    if (timeout !== undefined) clearTimeout(timeout)
  }
}

function stripThinking(value: string): string {
  return value.replace(/<think>[\s\S]*?<\/think>/gi, '')
}

function requiredText(value: string, label: string): string {
  const text = value.trim()
  if (!text) throw new Error(`${label} must not be empty`)
  return text
}

function normalizedText(value: string | undefined): string {
  return value?.trim() ?? ''
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`${label} must be a positive integer`)
  return value
}

function nonNegativeInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${label} must be a non-negative integer`)
  return value
}

function truncate(value: string, limit: number): string {
  return value.slice(0, limit)
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function errorName(error: unknown): string {
  return error instanceof Error && error.name ? error.name : 'Error'
}
