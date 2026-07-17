// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { TelegramUpdatesOptions } from './telegram.js'
import type { WebhookHeaders, WebhookResponse } from './webhooks.js'

const DEFAULT_POLLING_RETRY_DELAY = 2_000
const DEFAULT_POLLING_TIMEOUT = 30

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
  readonly onError?: (error: unknown) => void
  readonly retryDelay?: number
  readonly timeout?: number
}

/**
 * Abortable Telegram Bot API long-poll lifecycle.
 *
 * Telegram parsing remains centralized in the adapter's webhook handler, so
 * polling and webhook delivery normalize updates through exactly one path.
 */
export class TelegramPollingLoop {
  private readonly abort = new AbortController()
  private readonly channel: TelegramPollingChannel
  private readonly done: Promise<void>
  private readonly onError: ((error: unknown) => void) | undefined
  private readonly retryDelay: number
  private readonly timeout: number

  constructor(options: TelegramPollingLoopOptions) {
    this.channel = options.channel
    this.onError = options.onError
    this.retryDelay = nonNegativeInteger(options.retryDelay ?? DEFAULT_POLLING_RETRY_DELAY, 'retryDelay')
    this.timeout = nonNegativeInteger(options.timeout ?? DEFAULT_POLLING_TIMEOUT, 'timeout')
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
            // The offset is advanced only after a successful delivery, so a
            // failed update is acknowledged nowhere and Telegram redelivers it.
            throw new Error(`Telegram update ${updateId ?? 'unknown'} delivery failed (${delivered.status})`)
          }
          if (updateId !== undefined) offset = updateId + 1
        }
      } catch (error) {
        if (this.abort.signal.aborted) return
        this.report(error)
        await sleep(this.retryDelay, this.abort.signal)
      }
    }
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

async function sleep(milliseconds: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted || milliseconds === 0) return
  await Promise.race([
    Bun.sleep(milliseconds),
    new Promise<void>(resolve => signal.addEventListener('abort', () => resolve(), { once: true })),
  ])
}
