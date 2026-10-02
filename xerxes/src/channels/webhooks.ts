// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { Channel, InboundHandler } from './base.js'
import { ChannelTurnDeliveryError } from './turnRouter.js'
import type { ChannelMessage } from './types.js'

/** Headers and raw payload delivered by an HTTP webhook endpoint. */
export type WebhookHeaders = Readonly<Record<string, string>>

/** HTTP response returned by a webhook handler or channel. */
export interface WebhookResponse {
  readonly body: string
  readonly headers?: Readonly<Record<string, string>>
  readonly status: number
}

/** A response returned once a payload is accepted, while its inbound turns may still be running. */
export interface WebhookAcceptance extends WebhookResponse {
  /** Settles with the final delivery outcome once every accepted message has been handled. */
  readonly settled?: Promise<WebhookResponse>
}

/** A channel that exposes a raw webhook endpoint in addition to the base transport contract. */
export interface WebhookCapableChannel extends Channel {
  handleWebhook(
    headers: WebhookHeaders,
    body: Uint8Array,
  ): Promise<WebhookResponse>
}

/** Handler registered with a generic channel webhook dispatcher. */
export type WebhookHandler = (
  headers: WebhookHeaders,
  body: Uint8Array,
) => Promise<WebhookResponse>

export type WebhookFailureSource = 'dispatcher' | 'inbound_handler' | 'parse'

/** Bound on retained dispatcher failures so long-lived processes keep only recent diagnostics. */
export const WEBHOOK_FAILURE_LIMIT = 100

/**
 * Bound on recently delivered platform message ids retained to deduplicate
 * provider retries (Slack 500-retry, Telegram update re-send, and similar).
 */
export const WEBHOOK_DELIVERY_DEDUP_LIMIT = 1_000

/** A webhook error that was converted into a safe HTTP response. */
export interface WebhookFailure {
  readonly channel: string
  readonly error: unknown
  readonly source: WebhookFailureSource
}

export interface WebhookDispatcherOptions {
  readonly onFailure?: (failure: WebhookFailure) => void
}

/**
 * Route raw HTTP callbacks to named channel handlers.
 *
 * The dispatcher deliberately passes headers and bytes through unchanged:
 * provider signature verification must operate on the original request.
 */
export class WebhookDispatcher {
  private readonly failures: WebhookFailure[] = []
  private readonly handlers = new Map<string, WebhookHandler>()
  private readonly onFailure: ((failure: WebhookFailure) => void) | undefined

  constructor(options: WebhookDispatcherOptions = {}) {
    this.onFailure = options.onFailure
  }

  register(name: string, handler: WebhookHandler): void {
    this.handlers.set(name, handler)
  }

  /** Register a webhook-capable channel under its own adapter name. */
  registerChannel(channel: WebhookCapableChannel): void {
    this.register(channel.name, (headers, body) => channel.handleWebhook(headers, body))
  }

  unregister(name: string): void {
    this.handlers.delete(name)
  }

  names(): string[] {
    return [...this.handlers.keys()]
  }

  failuresSnapshot(): readonly WebhookFailure[] {
    return [...this.failures]
  }

  clearFailures(): void {
    this.failures.length = 0
  }

  async dispatch(
    name: string,
    headers: WebhookHeaders,
    body: Uint8Array,
  ): Promise<WebhookResponse> {
    const handler = this.handlers.get(name)
    if (!handler) {
      return { status: 404, body: `unknown channel '${name}'` }
    }
    try {
      return await handler(headers, body)
    } catch (error) {
      this.report({ channel: name, error, source: 'dispatcher' })
      return { status: 500, body: '' }
    }
  }

  private report(failure: WebhookFailure): void {
    this.failures.push(failure)
    if (this.failures.length > WEBHOOK_FAILURE_LIMIT) {
      this.failures.splice(0, this.failures.length - WEBHOOK_FAILURE_LIMIT)
    }
    if (!this.onFailure) {
      return
    }
    try {
      this.onFailure(failure)
    } catch {
      // A diagnostic callback must not make webhook error containment fail.
    }
  }
}

export interface WebhookChannelOptions {
  readonly onFailure?: (failure: WebhookFailure) => void
}

type WebhookDeliveryOutcome = 'completed_delivery_failure' | 'delivered' | 'retryable_failure'

/**
 * Base class for channels whose inbound transport is an HTTP webhook.
 *
 * Subclasses only parse provider bytes and send provider-specific outbound
 * traffic. This class owns handler registration and response containment.
 */
export abstract class WebhookChannel implements WebhookCapableChannel {
  abstract readonly name: string

  private readonly deliveredPlatformIds = new Map<string, true>()
  private handler: InboundHandler | undefined
  private readonly onFailure: ((failure: WebhookFailure) => void) | undefined

  constructor(options: WebhookChannelOptions = {}) {
    this.onFailure = options.onFailure
  }

  async start(onInbound: InboundHandler): Promise<void> {
    this.handler = onInbound
  }

  async stop(): Promise<void> {
    this.handler = undefined
  }

  async send(message: ChannelMessage): Promise<void> {
    await this.sendOutbound(message)
  }

  async handleWebhook(
    headers: WebhookHeaders,
    body: Uint8Array,
  ): Promise<WebhookResponse> {
    const handler = this.handler
    if (!handler) {
      return { status: 503, body: 'channel not started' }
    }
    let messages: readonly ChannelMessage[]
    try {
      messages = await this.parseInbound(headers, body)
    } catch (error) {
      this.report({ channel: this.name, error, source: 'parse' })
      return { status: 400, body: 'invalid payload' }
    }

    let failed = false
    for (const message of messages) {
      // Reserve the id before awaiting dispatch so simultaneous provider
      // retries cannot both enter the inbound handler.
      if (!this.reserveDelivery(message)) continue
      if (await this.deliverReserved(message)) failed = true
    }
    return { status: failed ? 500 : 200, body: 'ok' }
  }

  /**
   * Accept a payload and start its inbound turns without waiting for them.
   *
   * A transport that pulls updates (Telegram long polling) must keep pulling
   * while a turn runs: awaiting the whole turn kept the next getUpdates from
   * being issued, so `/stop` never reached the running turn and every other
   * chat on the bot waited behind it. Handlers are invoked in payload order
   * before this returns, so per-conversation ordering is kept by the router.
   * The status reflects acceptance only; `settled` carries the turn outcome.
   */
  protected async acceptWebhook(
    headers: WebhookHeaders,
    body: Uint8Array,
  ): Promise<WebhookAcceptance> {
    if (!this.handler) {
      return { status: 503, body: 'channel not started' }
    }
    let messages: readonly ChannelMessage[]
    try {
      messages = await this.parseInbound(headers, body)
    } catch (error) {
      this.report({ channel: this.name, error, source: 'parse' })
      return { status: 400, body: 'invalid payload' }
    }
    const deliveries = messages
      .filter(message => this.reserveDelivery(message))
      .map(message => this.deliverReserved(message))
    const settled = Promise.all(deliveries).then(failures => ({
      status: failures.some(Boolean) ? 500 : 200,
      body: 'ok',
    }))
    return { status: 200, body: 'accepted', settled }
  }

  /**
   * Dispatch one reserved message and report whether it failed. Failures that
   * happen before turn completion release the reservation for a provider
   * retry; reply delivery failures keep it because retrying would repeat the turn.
   */
  private async deliverReserved(message: ChannelMessage): Promise<boolean> {
    const outcome = await this.dispatchInbound(message)
    if (outcome === 'retryable_failure') {
      this.forgetDelivery(message)
      return true
    }
    return outcome === 'completed_delivery_failure'
  }

  /** Identity used to drop provider re-sends of an already delivered message. */
  protected deliveryKey(message: ChannelMessage): string | undefined {
    const platformMessageId = message.platformMessageId
    if (!platformMessageId) {
      return undefined
    }
    return `${message.roomId ?? ''} ${platformMessageId}`
  }

  private reserveDelivery(message: ChannelMessage): boolean {
    const key = this.deliveryKey(message)
    if (key === undefined) return true
    if (this.deliveredPlatformIds.has(key)) return false
    this.deliveredPlatformIds.set(key, true)
    while (this.deliveredPlatformIds.size > WEBHOOK_DELIVERY_DEDUP_LIMIT) {
      const oldest = this.deliveredPlatformIds.keys().next()
      if (oldest.done === true) break
      this.deliveredPlatformIds.delete(oldest.value)
    }
    return true
  }

  private forgetDelivery(message: ChannelMessage): void {
    const key = this.deliveryKey(message)
    if (key !== undefined) this.deliveredPlatformIds.delete(key)
  }

  /** Deliver one already-normalized inbound message while preserving error containment. */
  protected async dispatchInbound(message: ChannelMessage): Promise<WebhookDeliveryOutcome> {
    const handler = this.handler
    if (!handler) return 'retryable_failure'
    try {
      await handler(message)
      return 'delivered'
    } catch (error) {
      this.report({ channel: this.name, error, source: 'inbound_handler' })
      return error instanceof ChannelTurnDeliveryError
        ? 'completed_delivery_failure'
        : 'retryable_failure'
    }
  }

  protected abstract parseInbound(
    headers: WebhookHeaders,
    body: Uint8Array,
  ): Promise<readonly ChannelMessage[]> | readonly ChannelMessage[]

  protected abstract sendOutbound(message: ChannelMessage): Promise<void>

  private report(failure: WebhookFailure): void {
    if (!this.onFailure) {
      return
    }
    try {
      this.onFailure(failure)
    } catch {
      // A diagnostic callback must not make webhook error containment fail.
    }
  }
}

/** Decode a webhook body as an object, rejecting malformed JSON. */
export function parseJsonBody(body: Uint8Array): Record<string, unknown> {
  if (!body.byteLength) return {}
  try {
    const value: unknown = JSON.parse(new TextDecoder().decode(body))
    return isRecord(value) ? value : {}
  } catch (error) {
    throw new TypeError('invalid JSON webhook payload', { cause: error })
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
