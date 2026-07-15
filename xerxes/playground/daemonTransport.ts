// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { DaemonEvent, DaemonRuntime } from '../src/daemon/runtime.js'
import type {
  EvaluationEvent,
  EvaluationSessionPort,
  EvaluationStartRequest,
  EvaluationStartResult,
  EvaluationSubmitRequest,
} from './types.js'

/** Options for adapting an already-configured Bun daemon runtime to the evaluation harness. */
export interface DaemonEvaluationSessionPortOptions {
  /** Required only when a custom runner can still emit manual approval requests. */
  readonly approve?: (requestId: string) => Promise<void>
  /** Distinguishes evaluation sessions from a host's ordinary daemon sessions. */
  readonly sessionKeyPrefix?: string
}

/**
 * Bridges the native daemon event stream into the playground's provider-neutral transport.
 *
 * The caller owns runtime construction, including its private session/home paths and provider
 * configuration. The adapter forces accept-all permissions for the evaluation session and never
 * starts a daemon process or reads a profile itself.
 */
export class DaemonEvaluationSessionPort implements EvaluationSessionPort {
  private readonly approveHandler: ((requestId: string) => Promise<void>) | undefined
  private readonly sessionKeyPrefix: string
  private activeSessionKey = ''
  private closed = false
  private readonly sessionKeys = new Set<string>()
  private startRequest: EvaluationStartRequest | undefined

  constructor(
    private readonly runtime: DaemonRuntime,
    options: DaemonEvaluationSessionPortOptions = {},
  ) {
    this.approveHandler = options.approve
    this.sessionKeyPrefix = requiredPrefix(options.sessionKeyPrefix ?? 'evaluation')
  }

  async approve(requestId: string): Promise<void> {
    if (!this.approveHandler) {
      throw new Error('The evaluation runtime requested approval; configure DaemonEvaluationSessionPortOptions.approve.')
    }
    await this.approveHandler(requiredText(requestId, 'requestId'))
  }

  async close(): Promise<void> {
    if (this.closed) return
    this.closed = true
    for (const sessionKey of this.sessionKeys) {
      this.runtime.cancelTurn(sessionKey)
      this.runtime.evictSession(sessionKey)
    }
    this.sessionKeys.clear()
  }

  async reset(): Promise<void> {
    this.assertStarted()
    await this.openFreshSession()
  }

  async start(request: EvaluationStartRequest): Promise<EvaluationStartResult> {
    if (this.closed) throw new Error('The evaluation session port is closed')
    if (this.startRequest !== undefined) throw new Error('The evaluation session port has already started')
    this.startRequest = request
    this.runtime.reload({ permission_mode: 'accept-all' })
    const session = await this.openFreshSession()
    return session.model ? { model: session.model } : {}
  }

  submit(request: EvaluationSubmitRequest): AsyncIterable<EvaluationEvent> {
    this.assertStarted()
    requiredText(request.prompt, 'prompt')
    if (!Number.isSafeInteger(request.timeoutMs) || request.timeoutMs <= 0) {
      throw new Error('timeoutMs must be a positive integer')
    }
    return this.events(request)
  }

  private async *events(request: EvaluationSubmitRequest): AsyncGenerator<EvaluationEvent> {
    const sessionKey = this.activeSessionKey
    const pending: EvaluationEvent[] = []
    let completed = false
    let ended = false
    let wake: (() => void) | undefined
    const notify = (): void => {
      const current = wake
      wake = undefined
      current?.()
    }
    const enqueue = (event: EvaluationEvent): void => {
      if (event.type === 'turn_end') ended = true
      pending.push(event)
      notify()
    }
    const emit = (event: DaemonEvent): void => {
      const mapped = evaluationEventFromDaemon(event)
      if (mapped !== undefined) enqueue(mapped)
    }
    const cancel = (): void => {
      if (!ended) this.runtime.cancelTurn(sessionKey)
    }
    request.signal.addEventListener('abort', cancel, { once: true })
    if (request.signal.aborted) cancel()

    void this.runtime.submitTurn(sessionKey, request.prompt, emit).then(
      () => {
        completed = true
        if (!ended) enqueue({ type: 'turn_end' })
        notify()
      },
      error => {
        completed = true
        enqueue({
          type: 'notification',
          severity: 'error',
          title: undefined,
          body: `Native daemon evaluation failed: ${errorMessage(error)}`,
        })
        if (!ended) enqueue({ type: 'turn_end' })
        notify()
      },
    )

    try {
      while (!completed || pending.length) {
        const event = pending.shift()
        if (event !== undefined) {
          yield event
          continue
        }
        await new Promise<void>(resolve => {
          wake = resolve
        })
      }
    } finally {
      request.signal.removeEventListener('abort', cancel)
      if (request.signal.aborted && !ended) cancel()
    }
  }

  private async openFreshSession(): Promise<Awaited<ReturnType<DaemonRuntime['openSession']>>> {
    const request = this.startRequest
    if (request === undefined) throw new Error('The evaluation session port has not started')
    const sessionKey = `${this.sessionKeyPrefix}:${process.pid}:${crypto.randomUUID()}`
    const session = await this.runtime.openSession(sessionKey, undefined, {
      cwd: request.workspaceDirectory,
    })
    this.activeSessionKey = sessionKey
    this.sessionKeys.add(sessionKey)
    return session
  }

  private assertStarted(): void {
    if (this.closed) throw new Error('The evaluation session port is closed')
    if (this.startRequest === undefined) throw new Error('The evaluation session port has not started')
  }
}

function evaluationEventFromDaemon(event: DaemonEvent): EvaluationEvent | undefined {
  if (event.type === 'text_part') {
    const text = stringValue(event.payload.text)
    return text ? { type: 'text', text } : undefined
  }
  if (event.type === 'tool_call') {
    const name = stringValue(event.payload.name)
    return name ? { type: 'tool_call', name } : undefined
  }
  if (event.type === 'approval_request') {
    const id = stringValue(event.payload.request_id) || stringValue(event.payload.id)
    return id ? { type: 'approval_request', id } : undefined
  }
  if (event.type === 'status_update') {
    const contextTokens = numberValue(event.payload.context_tokens)
    return { type: 'status', contextTokens }
  }
  if (event.type === 'notification') {
    return {
      type: 'notification',
      severity: notificationSeverity(event.payload.severity) ?? notificationSeverity(event.payload.level) ?? 'info',
      title: optionalText(event.payload.title),
      body: optionalText(event.payload.body) ?? optionalText(event.payload.message),
    }
  }
  if (event.type === 'turn_end') return { type: 'turn_end' }
  return undefined
}

function requiredPrefix(value: string): string {
  const prefix = value.trim()
  if (!prefix) throw new Error('sessionKeyPrefix must not be empty')
  return prefix
}

function requiredText(value: string, label: string): string {
  const text = value.trim()
  if (!text) throw new Error(`${label} must not be empty`)
  return text
}

function optionalText(value: unknown): string | undefined {
  const text = stringValue(value)
  return text || undefined
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function numberValue(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined
}

function notificationSeverity(value: unknown): 'debug' | 'error' | 'info' | 'warning' | undefined {
  return value === 'debug' || value === 'error' || value === 'info' || value === 'warning' ? value : undefined
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
