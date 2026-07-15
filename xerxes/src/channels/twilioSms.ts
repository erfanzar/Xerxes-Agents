// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Buffer } from 'node:buffer'

import { postForm, providerUrl, type ChannelFetch } from './http.js'
import { outboundDestination, requiredOption, type RelayChannelTransport } from './relay.js'
import { createChannelMessage, MessageDirection, type ChannelMessage } from './types.js'
import { WebhookChannel, type WebhookHeaders } from './webhooks.js'

const TWILIO_API_BASE = 'https://api.twilio.com/'

/** Twilio form-webhook relay and REST form sender. */
export const TWILIO_SMS_TRANSPORT: RelayChannelTransport = {
  inbound: 'webhook-relay',
  outbound: 'http-api',
  unsupported: ['Twilio X-Twilio-Signature verification', 'MMS media download and delivery callbacks'],
}

export interface TwilioSmsChannelOptions {
  readonly accountSid: string
  readonly apiBaseUrl?: string
  readonly authToken: string
  readonly fetchImplementation?: ChannelFetch
  readonly fromNumber: string
}

/**
 * Twilio SMS form-webhook relay and `Messages.json` API adapter.
 *
 * An HTTP edge should verify Twilio's signature before calling this channel;
 * this dependency-free adapter parses the delivered form and sends SMS only.
 */
export class TwilioSmsChannel extends WebhookChannel {
  readonly name = 'sms'
  readonly transport = TWILIO_SMS_TRANSPORT

  private readonly accountSid: string
  private readonly apiBaseUrl: string
  private readonly authToken: string
  private readonly fetchImplementation: ChannelFetch | undefined
  private readonly fromNumber: string

  constructor(options: TwilioSmsChannelOptions) {
    super()
    this.accountSid = requiredOption(options.accountSid, 'Twilio accountSid')
    this.apiBaseUrl = options.apiBaseUrl ?? TWILIO_API_BASE
    this.authToken = requiredOption(options.authToken, 'Twilio authToken')
    this.fetchImplementation = options.fetchImplementation
    this.fromNumber = requiredOption(options.fromNumber, 'Twilio fromNumber')
  }

  protected parseInbound(
    _headers: WebhookHeaders,
    body: Uint8Array,
  ): readonly ChannelMessage[] {
    const fields = new URLSearchParams(new TextDecoder().decode(body))
    if (![...fields.keys()].length) {
      return []
    }
    const sender = fields.get('From') ?? ''
    return [createChannelMessage({
      channel: this.name,
      direction: MessageDirection.INBOUND,
      text: fields.get('Body') ?? '',
      channelUserId: sender,
      roomId: sender,
      platformMessageId: fields.get('MessageSid') ?? '',
      metadata: { to: fields.get('To') ?? '' },
    })]
  }

  protected async sendOutbound(message: ChannelMessage): Promise<void> {
    const recipient = outboundDestination(message, 'Twilio SMS')
    const credentials = Buffer.from(`${this.accountSid}:${this.authToken}`, 'utf8').toString('base64')
    await postForm(providerUrl(this.apiBaseUrl, `2010-04-01/Accounts/${this.accountSid}/Messages.json`), {
      body: { From: this.fromNumber, To: recipient, Body: message.text },
      headers: { Authorization: `Basic ${credentials}` },
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
    })
  }
}
