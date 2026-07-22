// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { Channel, InboundHandler } from './base.js'
import type { ChannelMessage } from './types.js'

/** Headers and raw payload delivered by an HTTP webhook endpoint. */
export type WebhookHeaders = Readonly<Record<string, string>>

/** HTTP response returned by a webhook handler or channel. */
export interface WebhookResponse {
  readonly body: string
  readonly headers?: Readonly<Record<string, string>>
  readonly status: number
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
      // Provider retries repeat the same platform message id; acknowledge the
      // duplicate without routing a second agent turn. A failed dispatch is
      // deliberately not remembered so a later retry can still be delivered.
      if (this.isDuplicateDelivery(message)) continue
      if (!await this.dispatchInbound(message)) {
        failed = true
        continue
      }
      this.rememberDelivery(message)
    }
    return { status: failed ? 500 : 200, body: 'ok' }
  }

  private deliveryKey(message: ChannelMessage): string | undefined {
    const platformMessageId = message.platformMessageId
    if (!platformMessageId) {
      return undefined
    }
    return `${message.roomId ?? ''} ${platformMessageId}`
  }

  private isDuplicateDelivery(message: ChannelMessage): boolean {
    const key = this.deliveryKey(message)
    return key !== undefined && this.deliveredPlatformIds.has(key)
  }

  private rememberDelivery(message: ChannelMessage): void {
    const key = this.deliveryKey(message)
    if (key === undefined) {
      return
    }
    this.deliveredPlatformIds.set(key, true)
    while (this.deliveredPlatformIds.size > WEBHOOK_DELIVERY_DEDUP_LIMIT) {
      const oldest = this.deliveredPlatformIds.keys().next()
      if (oldest.done === true) break
      this.deliveredPlatformIds.delete(oldest.value)
    }
  }

  /** Deliver one already-normalized inbound message while preserving error containment. */
  protected async dispatchInbound(message: ChannelMessage): Promise<boolean> {
    const handler = this.handler
    if (!handler) return false
    try {
      await handler(message)
      return true
    } catch (error) {
      this.report({ channel: this.name, error, source: 'inbound_handler' })
      return false
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

/** Decode a webhook body as an object, returning an empty object for non-object JSON. */
export function parseJsonBody(body: Uint8Array): Record<string, unknown> {
  if (!body.byteLength) {
    return {}
  }
  try {
    const value: unknown = JSON.parse(new TextDecoder().decode(body))
    return isRecord(value) ? value : {}
  } catch {
    return {}
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
