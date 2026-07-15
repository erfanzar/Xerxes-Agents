// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { providerUrl, putJson, type ChannelFetch } from './http.js'
import {
  arrayValue,
  outboundDestination,
  recordValue,
  requiredOption,
  stringValue,
  type RelayChannelTransport,
} from './relay.js'
import { createChannelMessage, MessageDirection, type ChannelMessage } from './types.js'
import { parseJsonBody, WebhookChannel, type WebhookHeaders } from './webhooks.js'

/** Matrix is webhook-relay-in/client-server-API-out; `/sync` is not started here. */
export const MATRIX_TRANSPORT: RelayChannelTransport = {
  inbound: 'webhook-relay',
  outbound: 'http-api',
  unsupported: ['Matrix /sync polling', 'Matrix sliding-sync and persistent client connections'],
}

export interface MatrixChannelOptions {
  readonly accessToken: string
  readonly fetchImplementation?: ChannelFetch
  readonly homeserverUrl: string
  /** Supplies deterministic ids in tests or a caller-managed retry protocol. */
  readonly transactionId?: () => string
}

/**
 * Matrix room-event webhook relay with an idempotent client-server API sender.
 *
 * Inbound Matrix `/sync` traffic must be relayed by another component. The
 * adapter only accepts already-delivered room events through `handleWebhook`.
 */
export class MatrixChannel extends WebhookChannel {
  readonly name = 'matrix'
  readonly transport = MATRIX_TRANSPORT

  private readonly accessToken: string
  private readonly fetchImplementation: ChannelFetch | undefined
  private readonly homeserverUrl: string
  private readonly transactionId: () => string

  constructor(options: MatrixChannelOptions) {
    super()
    this.accessToken = requiredOption(options.accessToken, 'Matrix accessToken')
    this.fetchImplementation = options.fetchImplementation
    this.homeserverUrl = requiredOption(options.homeserverUrl, 'Matrix homeserverUrl')
    this.transactionId = options.transactionId ?? (() => `xerxes-${crypto.randomUUID()}`)
  }

  protected parseInbound(
    _headers: WebhookHeaders,
    body: Uint8Array,
  ): readonly ChannelMessage[] {
    const payload = parseJsonBody(body)
    const events = arrayValue(payload.events)
    const candidates = events.length ? events : [payload]
    const messages: ChannelMessage[] = []
    for (const candidate of candidates) {
      const event = recordValue(candidate)
      if (event.type !== 'm.room.message') {
        continue
      }
      const content = recordValue(event.content)
      messages.push(createChannelMessage({
        channel: this.name,
        direction: MessageDirection.INBOUND,
        text: stringValue(content.body),
        channelUserId: stringValue(event.sender),
        roomId: stringValue(event.room_id),
        platformMessageId: stringValue(event.event_id),
        metadata: { msgtype: stringValue(content.msgtype) },
      }))
    }
    return messages
  }

  protected async sendOutbound(message: ChannelMessage): Promise<void> {
    const roomId = outboundDestination(message, 'Matrix')
    const transactionId = this.transactionId()
    if (!transactionId) {
      throw new Error('Matrix transactionId provider returned an empty id')
    }
    const path = [
      '_matrix/client/v3/rooms',
      encodeURIComponent(roomId),
      'send/m.room.message',
      encodeURIComponent(transactionId),
    ].join('/')
    await putJson(providerUrl(this.homeserverUrl, path), {
      body: { msgtype: 'm.text', body: message.text },
      headers: { Authorization: `Bearer ${this.accessToken}` },
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
    })
  }
}
