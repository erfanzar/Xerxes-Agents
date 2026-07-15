// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { postJson, providerUrl, type ChannelFetch } from './http.js'
import { outboundDestination, recordValue, requiredOption, stringValue, type RelayChannelTransport } from './relay.js'
import { createChannelMessage, MessageDirection, type ChannelMessage } from './types.js'
import { parseJsonBody, WebhookChannel, type WebhookHeaders } from './webhooks.js'

/** Signal depends on a self-hosted REST bridge; it does not run signal-cli itself. */
export const SIGNAL_TRANSPORT: RelayChannelTransport = {
  inbound: 'webhook-relay',
  outbound: 'http-api',
  unsupported: ['signal-cli receive loop', 'Signal WebSocket and attachment transports'],
}

export interface SignalChannelOptions {
  readonly fetchImplementation?: ChannelFetch
  readonly restBaseUrl: string
  readonly senderNumber: string
}

/**
 * Webhook/API adapter for a self-hosted Signal REST bridge.
 *
 * A bridge such as signal-cli-rest-api must relay inbound envelopes. This
 * adapter deliberately does not own a Signal account, socket, or receive loop.
 */
export class SignalChannel extends WebhookChannel {
  readonly name = 'signal'
  readonly transport = SIGNAL_TRANSPORT

  private readonly fetchImplementation: ChannelFetch | undefined
  private readonly restBaseUrl: string
  private readonly senderNumber: string

  constructor(options: SignalChannelOptions) {
    super()
    this.fetchImplementation = options.fetchImplementation
    this.restBaseUrl = requiredOption(options.restBaseUrl, 'Signal restBaseUrl')
    this.senderNumber = requiredOption(options.senderNumber, 'Signal senderNumber')
  }

  protected parseInbound(
    _headers: WebhookHeaders,
    body: Uint8Array,
  ): readonly ChannelMessage[] {
    const payload = parseJsonBody(body)
    const envelope = Object.keys(recordValue(payload.envelope)).length
      ? recordValue(payload.envelope)
      : payload
    const dataMessage = recordValue(envelope.dataMessage)
    const bridgeMessage = recordValue(envelope.message)
    const text = stringValue(dataMessage.message)
      || stringValue(bridgeMessage.message)
      || stringValue(envelope.message)
    if (!text) {
      return []
    }
    const sender = stringValue(envelope.sourceNumber) || stringValue(envelope.source)
    return [createChannelMessage({
      channel: this.name,
      direction: MessageDirection.INBOUND,
      text,
      channelUserId: sender,
      roomId: sender,
      platformMessageId: stringValue(envelope.timestamp),
      metadata: { source_name: stringValue(envelope.sourceName) },
    })]
  }

  protected async sendOutbound(message: ChannelMessage): Promise<void> {
    const recipient = outboundDestination(message, 'Signal')
    await postJson(providerUrl(this.restBaseUrl, 'v2/send'), {
      body: { number: this.senderNumber, recipients: [recipient], message: message.text },
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
    })
  }
}
