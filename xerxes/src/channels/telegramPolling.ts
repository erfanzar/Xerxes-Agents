// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { TelegramUpdatesOptions } from './telegram.js'
import type { WebhookHeaders, WebhookResponse } from './webhooks.js'

const DEFAULT_POLLING_RETRY_DELAY = 2_000
const DEFAULT_POLLING_TIMEOUT = 30
/** Delivery attempts granted to one update before it is dead-lettered. */
const DEFAULT_MAX_UPDATE_ATTEMPTS = 5
/** Upper bound for the exponential delay between failed polling cycles. */
const MAX_POLLING_BACKOFF_MS = 30_000

export interface TelegramPollingChannel {
  getUpdates(options?: TelegramUpdatesOptions): Promise<Readonly<Record<string, unknown>>>
  handleWebhook(headers: WebhookHeaders, body: Uint8Array): Promise<WebhookResponse>
  /**
   * Internal ingest for polled updates. Polling is authenticated by the bot
   * token, so adapters use this to bypass webhook-only credentials such as
   * Telegram's secret-token header. Falls back to handleWebhook when absent.
   */
  ingestPolledUpdate?(body: Uint8Array): Promise<WebhookResponse>
}

export interface TelegramPollingLoopOptions {
  readonly channel: TelegramPollingChannel
  /** Delivery attempts allowed per update before it is skipped as a dead letter. */
  readonly maxUpdateAttempts?: number
  readonly onError?: (error: unknown) => void
  readonly retryDelay?: number
  readonly timeout?: number
}

/**
 * Abortable Telegram Bot API long-poll lifecycle.
 *
 * Telegram parsing remains centralized in the adapter's webhook handler, so
 * polling and webhook delivery normalize updates through exactly one path.
 *
 * Failure containment is bounded: a deterministically failing update is
 * retried up to `maxUpdateAttempts` times with growing backoff, after which
 * the loop advances the offset past the poisoned update, reports a durable
 * dead-letter signal through `onError` (the message carries the update_id),
 * and keeps delivering later updates instead of wedging the bot forever.
 */
export class TelegramPollingLoop {
  private readonly abort = new AbortController()
  private readonly channel: TelegramPollingChannel
  private readonly done: Promise<void>
  private readonly maxUpdateAttempts: number
  private readonly onError: ((error: unknown) => void) | undefined
  private readonly retryDelay: number
  private readonly timeout: number
  private backoffStep = 0
  private readonly updateFailures = new Map<number, number>()

  constructor(options: TelegramPollingLoopOptions) {
    this.channel = options.channel
    this.onError = options.onError
    this.retryDelay = nonNegativeInteger(options.retryDelay ?? DEFAULT_POLLING_RETRY_DELAY, 'retryDelay')
    this.timeout = nonNegativeInteger(options.timeout ?? DEFAULT_POLLING_TIMEOUT, 'timeout')
    this.maxUpdateAttempts = positiveInteger(
      options.maxUpdateAttempts ?? DEFAULT_MAX_UPDATE_ATTEMPTS,
      'maxUpdateAttempts',
    )
    this.done = this.poll()
  }

  /** Whether this loop has been stopped or its current request was aborted. */
  get stopped(): boolean {
    return this.abort.signal.aborted
  }

  /** Stop receiving updates and wait until the active request observes cancellation. */
  async stop(): Promise<void> {
    this.abort.abort()
    await this.done
  }

  private async poll(): Promise<void> {
    let offset: number | undefined
    while (!this.abort.signal.aborted) {
      try {
        const response = await this.channel.getUpdates({
          timeout: this.timeout,
          signal: this.abort.signal,
          ...(offset === undefined ? {} : { offset }),
        })
        for (const update of updates(response)) {
          if (this.abort.signal.aborted) return
          const updateId = integer(update.update_id)
          const delivered = await ingestPolledUpdate(this.channel, update)
          if (delivered.status >= 400) {
            const attempts = this.registerDeliveryFailure(updateId, delivered.status)
            if (attempts !== 0) {
              // The offset is advanced only after successful delivery, so an
              // update that is still under its retry budget is re-fetched
              // from Telegram after a bounded backoff.
              throw new Error(attempts < 0
                ? `Telegram update without an update_id failed delivery (${delivered.status}) and cannot be acknowledged; retrying`
                : `Telegram update ${updateId} delivery failed (${delivered.status}); `
                  + `retrying (attempt ${attempts} of ${this.maxUpdateAttempts})`)
            }
            // Dead-lettered: fall through so the offset advances past the
            // poisoned update instead of wedging the polling loop on it.
          }
          if (updateId !== undefined) {
            offset = updateId + 1
            this.updateFailures.delete(updateId)
          }
        }
        this.backoffStep = 0
      } catch (error) {
        if (this.abort.signal.aborted) return
        this.report(error)
        await sleep(this.nextBackoffDelay(), this.abort.signal)
        this.backoffStep += 1
      }
    }
  }

  /**
   * Account one failed delivery against the update's retry budget.
   *
   * Returns 0 once the budget is exhausted (the caller then skips the
   * poisoned update permanently), otherwise the failed-attempt count, or -1
   * for updates without an update_id: those can never be acknowledged by an
   * offset, so they always stay on the bounded-backoff retry path rather
   * than spinning hot on an unacknowledged dead letter.
   */
  private registerDeliveryFailure(updateId: number | undefined, status: number): number {
    if (updateId === undefined) return -1
    const attempts = (this.updateFailures.get(updateId) ?? 0) + 1
    if (attempts < this.maxUpdateAttempts) {
      this.updateFailures.set(updateId, attempts)
      return attempts
    }
    this.updateFailures.delete(updateId)
    this.report(new Error(
      `Telegram polling dead-lettered update_id ${updateId} after ${attempts} failed delivery`
        + ` attempts (last status ${status}); advancing past it`,
    ))
    return 0
  }

  /** Capped exponential delay before the next polling cycle after a failure. */
  private nextBackoffDelay(): number {
    return Math.min(this.retryDelay * 2 ** Math.min(this.backoffStep, 30), MAX_POLLING_BACKOFF_MS)
  }

  private report(error: unknown): void {
    if (!this.onError) return
    try {
      this.onError(error)
    } catch {
      // Diagnostics must never terminate a healthy polling loop.
    }
  }
}

function updates(response: Readonly<Record<string, unknown>>): readonly Record<string, unknown>[] {
  const value = response.result
  return Array.isArray(value)
    ? value.filter((item): item is Record<string, unknown> => isRecord(item))
    : []
}

function ingestPolledUpdate(
  channel: TelegramPollingChannel,
  update: Record<string, unknown>,
): Promise<WebhookResponse> {
  const body = new TextEncoder().encode(JSON.stringify(update))
  return channel.ingestPolledUpdate ? channel.ingestPolledUpdate(body) : channel.handleWebhook({}, body)
}

function integer(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isInteger(value) ? value : undefined
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(name + ' must be a non-negative safe integer')
  }
  return value
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(name + ' must be a positive safe integer')
  }
  return value
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

async function sleep(milliseconds: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted || milliseconds === 0) return
  let resolveAbort: () => void = () => undefined
  const aborted = new Promise<void>(resolve => { resolveAbort = resolve })
  const onAbort = (): void => { resolveAbort() }
  signal.addEventListener('abort', onAbort, { once: true })
  try {
    if (signal.aborted) return
    await Promise.race([Bun.sleep(milliseconds), aborted])
  } finally {
    signal.removeEventListener('abort', onAbort)
  }
}
