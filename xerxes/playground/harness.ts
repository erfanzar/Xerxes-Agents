// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type {
  EvaluationAgent,
  EvaluationDiagnosisRequest,
  EvaluationEvent,
  EvaluationFailureCategory,
  EvaluationJudgePort,
  EvaluationScoreRow,
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
    const normalizedPrompt = requiredText(prompt, 'prompt')
    let attempt = 0
    let result = await this.oneTurn(normalizedPrompt, timeoutMs, options.signal)

    while (attempt < retries && !options.signal?.aborted && isTransientEvaluationFailure(result)) {
      attempt += 1
      const aborted = await abortableSleep(this.sleep, this.retryDelayMs(attempt), options.signal)
      if (aborted) {
        return { ...result, error: 'cancelled', retries: attempt }
      }
      result = await this.oneTurn(normalizedPrompt, timeoutMs, options.signal)
    }

    return { ...result, retries: attempt }
  }

  async close(): Promise<void> {
    await this.transport.close()
  }

  private async oneTurn(prompt: string, timeoutMs: number, callerSignal?: AbortSignal): Promise<EvaluationTurnResult> {
    const startedAt = this.clock()
    if (callerSignal?.aborted) {
      return {
        contextTokens: 0,
        error: 'cancelled',
        latencyMs: 0,
        retries: 0,
        text: '',
        tools: [],
      }
    }
    const deadline = startedAt + timeoutMs
    const controller = new AbortController()
    const onCallerAbort = (): void => controller.abort(callerSignal?.reason)
    callerSignal?.addEventListener('abort', onCallerAbort, { once: true })
    if (callerSignal?.aborted) onCallerAbort()
    const text: string[] = []
    const tools: string[] = []
    let contextTokens = 0
    let error: string | undefined
    let iteratorDone = false
    let iterator: AsyncIterator<EvaluationEvent> | undefined

    try {
      const submitted = await raceWithDeadlineAndAbort(
        Promise.resolve(this.transport.submit({ prompt, signal: controller.signal, timeoutMs })),
        deadline - this.clock(),
        callerSignal,
      )
      if (submitted === TIMED_OUT) {
        error = 'timeout'
      } else if (submitted === CANCELLED) {
        error = 'cancelled'
      } else {
        iterator = submitted[Symbol.asyncIterator]()
        while (true) {
          if (callerSignal?.aborted) {
            error = 'cancelled'
            break
          }
          const remaining = deadline - this.clock()
          if (remaining <= 0) {
            error = 'timeout'
            break
          }
          const next = await raceWithDeadlineAndAbort(iterator.next(), remaining, callerSignal)
          if (next === TIMED_OUT) {
            error = 'timeout'
            break
          }
          if (next === CANCELLED) {
            error = 'cancelled'
            break
          }
          if (next.done) {
            iteratorDone = true
            break
          }

          const event = next.value
          if (event.type === 'text') {
            text.push(event.text)
          } else if (event.type === 'tool_call') {
            tools.push(event.name)
          } else if (event.type === 'approval_request') {
            const approveRemaining = deadline - this.clock()
            if (approveRemaining <= 0) {
              error = 'timeout'
              break
            }
            const approved = await raceWithDeadlineAndAbort(
              this.transport.approve(event.id),
              approveRemaining,
              callerSignal,
            )
            if (approved === TIMED_OUT) {
              error = 'timeout'
              break
            }
            if (approved === CANCELLED) {
              error = 'cancelled'
              break
            }
          } else if (event.type === 'status') {
            if (event.contextTokens !== undefined && event.contextTokens > 0) contextTokens = event.contextTokens
          } else if (event.type === 'notification' && event.severity === 'error') {
            error = truncate(event.body ?? event.title ?? 'error', 160)
          } else if (event.type === 'turn_end') {
            break
          }
        }
      }
    } catch (caught) {
      error = callerSignal?.aborted ? 'cancelled' : truncate(errorMessage(caught), 160)
    } finally {
      callerSignal?.removeEventListener('abort', onCallerAbort)
      controller.abort()
      if (!iteratorDone && iterator?.return !== undefined) {
        await closeIteratorBounded(iterator, 250)
      }
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

/** True only for error envelopes that warrant a retry. Turn timeouts and cancellations never retry. */
export function isTransientEvaluationFailure(result: Pick<EvaluationTurnResult, 'error' | 'text'>): boolean {
  const error = result.error?.toLowerCase() ?? ''
  if (error === 'timeout' || error === 'cancelled') return false
  const text = result.text.trim().toLowerCase()
  const envelope = error || (text.startsWith('[error:') ? text : '')
  return envelope.length > 0 && TRANSIENT_MARKERS.some(marker => envelope.includes(marker))
}

/**
 * Classify a failed evaluation score row or turn into a deterministic failure category
 * without regex-scraping human prose, separating harness/provider faults from model misses.
 */
export function classifyEvaluationFailure(
  input: Pick<EvaluationScoreRow, 'detail' | 'ok'> & { readonly turnError?: string; readonly reply?: string },
): EvaluationFailureCategory | undefined {
  if (input.ok) return undefined
  const error = input.turnError?.trim().toLowerCase()
  const detail = input.detail.trim().toLowerCase()
  if (error === 'cancelled' || detail === 'error: cancelled') return 'cancelled'
  if (
    error === 'timeout'
    || detail === 'error: timeout'
    || detail.startsWith('grader timed out')
    || detail.startsWith('behavioral probe timed out')
  ) {
    return 'timeout'
  }
  if (detail.startsWith('grader raised:')) return 'grader_exception'
  if (
    (error !== undefined && error.length > 0)
    || detail.startsWith('error:')
    || (input.reply?.trim().toLowerCase().startsWith('[error:') ?? false)
  ) {
    return 'transport_error'
  }
  return 'assertion_failed'
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
const CANCELLED = Symbol('cancelled')

async function raceWithDeadlineAndAbort<T>(
  operation: Promise<T>,
  timeoutMs: number,
  signal?: AbortSignal,
): Promise<T | typeof TIMED_OUT | typeof CANCELLED> {
  if (signal?.aborted) return CANCELLED
  if (timeoutMs <= 0) return TIMED_OUT
  let timeout: ReturnType<typeof setTimeout> | undefined
  let onAbort: (() => void) | undefined
  try {
    return await Promise.race([
      operation,
      new Promise<typeof TIMED_OUT>(resolve => {
        timeout = setTimeout(() => resolve(TIMED_OUT), timeoutMs)
      }),
      ...(signal === undefined ? [] : [
        new Promise<typeof CANCELLED>(resolve => {
          onAbort = () => resolve(CANCELLED)
          signal.addEventListener('abort', onAbort, { once: true })
          if (signal.aborted) onAbort()
        }),
      ]),
    ])
  } finally {
    if (timeout !== undefined) clearTimeout(timeout)
    if (onAbort !== undefined) signal?.removeEventListener('abort', onAbort)
  }
}

async function closeIteratorBounded(iterator: AsyncIterator<unknown>, graceMs: number): Promise<void> {
  if (iterator.return === undefined) return
  let timeout: ReturnType<typeof setTimeout> | undefined
  try {
    await Promise.race([
      Promise.resolve(iterator.return()).catch(() => undefined),
      new Promise<void>(resolve => {
        timeout = setTimeout(resolve, graceMs)
      }),
    ])
  } finally {
    if (timeout !== undefined) clearTimeout(timeout)
  }
}

async function abortableSleep(
  sleepFn: (milliseconds: number) => Promise<void>,
  delayMs: number,
  signal?: AbortSignal,
): Promise<boolean> {
  if (signal?.aborted) return true
  if (signal === undefined) {
    await sleepFn(delayMs)
    return false
  }
  let onAbort: (() => void) | undefined
  try {
    return await Promise.race([
      sleepFn(delayMs).then(() => false),
      new Promise<boolean>(resolve => {
        onAbort = () => resolve(true)
        signal.addEventListener('abort', onAbort, { once: true })
        if (signal.aborted) onAbort()
      }),
    ])
  } finally {
    if (onAbort !== undefined) signal.removeEventListener('abort', onAbort)
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
