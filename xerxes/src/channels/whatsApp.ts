// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { postJson, providerUrl, type ChannelFetch } from './http.js'
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

const WHATSAPP_GRAPH_API = 'https://graph.facebook.com/'

/** WhatsApp Cloud API webhook relay with Graph HTTP sends. */
export const WHATSAPP_TRANSPORT: RelayChannelTransport = {
  inbound: 'webhook-relay',
  outbound: 'http-api',
  unsupported: ['webhook signature verification', 'persistent WhatsApp socket transport'],
}

export interface WhatsAppChannelOptions {
  readonly accessToken: string
  readonly apiBaseUrl?: string
  readonly apiVersion?: string
  readonly fetchImplementation?: ChannelFetch
  readonly phoneNumberId: string
}

/**
 * WhatsApp Business Cloud API webhook relay and text sender.
 *
 * The HTTP edge must answer Meta's GET verification challenge and validate
 * signatures. `whatsAppWebhookChallenge` is provided for the former, while
 * this adapter accepts only the delivered JSON event body.
 */
export class WhatsAppChannel extends WebhookChannel {
  readonly name = 'whatsapp'
  readonly transport = WHATSAPP_TRANSPORT

  private readonly accessToken: string
  private readonly apiBaseUrl: string
  private readonly apiVersion: string
  private readonly fetchImplementation: ChannelFetch | undefined
  private readonly phoneNumberId: string

  constructor(options: WhatsAppChannelOptions) {
    super()
    this.accessToken = requiredOption(options.accessToken, 'WhatsApp accessToken')
    this.apiBaseUrl = options.apiBaseUrl ?? WHATSAPP_GRAPH_API
    this.apiVersion = options.apiVersion ?? 'v23.0'
    this.fetchImplementation = options.fetchImplementation
    this.phoneNumberId = requiredOption(options.phoneNumberId, 'WhatsApp phoneNumberId')
  }

  protected parseInbound(
    _headers: WebhookHeaders,
    body: Uint8Array,
  ): readonly ChannelMessage[] {
    const payload = parseJsonBody(body)
    const messages: ChannelMessage[] = []
    for (const entryValue of arrayValue(payload.entry)) {
      const entry = recordValue(entryValue)
      for (const changeValue of arrayValue(entry.changes)) {
        const change = recordValue(changeValue)
        const value = recordValue(change.value)
        for (const incomingValue of arrayValue(value.messages)) {
          const incoming = recordValue(incomingValue)
          const text = whatsAppText(incoming)
          if (!text) {
            continue
          }
          const sender = stringValue(incoming.from)
          messages.push(createChannelMessage({
            channel: this.name,
            direction: MessageDirection.INBOUND,
            text,
            channelUserId: sender,
            roomId: sender,
            platformMessageId: stringValue(incoming.id),
            metadata: { type: stringValue(incoming.type) },
          }))
        }
      }
    }
    return messages
  }

  protected async sendOutbound(message: ChannelMessage): Promise<void> {
    const recipient = outboundDestination(message, 'WhatsApp')
    await postJson(providerUrl(this.apiBaseUrl, `${this.apiVersion}/${this.phoneNumberId}/messages`), {
      body: {
        messaging_product: 'whatsapp',
        to: recipient,
        type: 'text',
        text: { body: message.text },
      },
      headers: { Authorization: `Bearer ${this.accessToken}` },
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
    })
  }
}

/** Return Meta's verification challenge only when its configured token matches. */
export function whatsAppWebhookChallenge(
  query: Readonly<Record<string, string | undefined>>,
  verifyToken: string,
): string | undefined {
  if (query['hub.mode'] !== 'subscribe' || query['hub.verify_token'] !== verifyToken) {
    return undefined
  }
  return query['hub.challenge']
}

function whatsAppText(message: Readonly<Record<string, unknown>>): string {
  return stringValue(recordValue(message.text).body)
    || stringValue(recordValue(message.button).text)
    || stringValue(recordValue(recordValue(message.interactive).button_reply).title)
    || stringValue(recordValue(recordValue(message.interactive).list_reply).title)
}
