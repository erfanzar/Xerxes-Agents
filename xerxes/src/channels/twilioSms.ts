// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHmac } from 'node:crypto'
import { Buffer } from 'node:buffer'

import { postForm, providerUrl, type ChannelFetch } from './http.js'
import { outboundDestination, requiredOption, type RelayChannelTransport } from './relay.js'
import { createChannelMessage, MessageDirection, type ChannelMessage } from './types.js'
import { constantTimeEqualStrings, webhookHeaderValue } from './webhookSignatures.js'
import { WebhookChannel, type WebhookHeaders, type WebhookResponse } from './webhooks.js'

const TWILIO_API_BASE = 'https://api.twilio.com/'
/**
 * Webhook path assumed when reconstructing the signed request URL from
 * forwarded headers. It mirrors the default exposure of
 * `ChannelWebhookServer`: `<pathPrefix>/<channelName>/webhook`.
 */
const DEFAULT_WEBHOOK_PATH = '/channels/sms/webhook'

/** Twilio form-webhook relay and REST form sender with signature verification. */
export const TWILIO_SMS_TRANSPORT: RelayChannelTransport = {
  inbound: 'webhook-relay',
  outbound: 'http-api',
  unsupported: ['MMS media download and delivery callbacks'],
}

export interface TwilioSmsChannelOptions {
  readonly accountSid: string
  readonly apiBaseUrl?: string
  readonly authToken: string
  readonly fetchImplementation?: ChannelFetch
  readonly fromNumber: string
  /**
   * Exact public URL Twilio requests for inbound SMS webhooks — scheme,
   * authority, and path precisely as configured on the Twilio side.
   *
   * Signature verification needs the byte-for-byte request URL Twilio signed.
   * Supply this whenever the edge sits behind a proxy, a custom path prefix,
   * or a non-default webhook route; without it the adapter reconstructs
   * `${proto}://${host}${DEFAULT_WEBHOOK_PATH}` from forwarded/host headers,
   * which only matches the default webhook server exposure.
   */
  readonly webhookUrl?: string
}

/**
 * Twilio SMS form-webhook relay and `Messages.json` API adapter.
 *
 * Inbound webhooks are verified against Twilio's `X-Twilio-Signature`
 * (base64 HMAC-SHA1 over the request URL plus alphabetically sorted
 * concatenated parameter values) using the required auth token; requests
 * without a verifiable signature are rejected before parsing.
 */
export class TwilioSmsChannel extends WebhookChannel {
  readonly name = 'sms'
  readonly transport = TWILIO_SMS_TRANSPORT

  private readonly accountSid: string
  private readonly apiBaseUrl: string
  private readonly authToken: string
  private readonly fetchImplementation: ChannelFetch | undefined
  private readonly fromNumber: string
  private readonly webhookUrl: string

  constructor(options: TwilioSmsChannelOptions) {
    super()
    this.accountSid = requiredOption(options.accountSid, 'Twilio accountSid')
    this.apiBaseUrl = options.apiBaseUrl ?? TWILIO_API_BASE
    this.authToken = requiredOption(options.authToken, 'Twilio authToken')
    this.fetchImplementation = options.fetchImplementation
    this.fromNumber = requiredOption(options.fromNumber, 'Twilio fromNumber')
    this.webhookUrl = options.webhookUrl?.trim() ?? ''
  }

  override async handleWebhook(headers: WebhookHeaders, body: Uint8Array): Promise<WebhookResponse> {
    if (!this.signatureMatches(headers, body)) {
      // Fail closed: auth_token is mandatory, so every deployment can and
      // must present a verifiable signature before form parsing runs.
      return { status: 401, body: 'unauthorized' }
    }
    return super.handleWebhook(headers, body)
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

  /**
   * Validate Twilio's signature over the raw body parsed as
   * application/x-www-form-urlencoded: base64(HMAC-SHA1(authToken,
   * requestUrl + values sorted by parameter name and concatenated)).
   */
  private signatureMatches(headers: WebhookHeaders, body: Uint8Array): boolean {
    const provided = webhookHeaderValue(headers, 'x-twilio-signature')
    const requestUrl = this.signedRequestUrl(headers)
    if (!provided || !requestUrl) return false
    const parameters = [...new URLSearchParams(new TextDecoder().decode(body)).entries()]
      .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
      .map(([, value]) => value)
      .join('')
    const expected = createHmac('sha1', this.authToken)
      .update(requestUrl + parameters)
      .digest('base64')
    return constantTimeEqualStrings(provided, expected)
  }

  /**
   * Resolve the URL Twilio signed: the explicit webhookUrl option when set,
   * otherwise reconstruction from forwarded/host headers against the default
   * webhook server path. Without either input there is nothing verifiable to
   * compare against, so verification fails closed.
   */
  private signedRequestUrl(headers: WebhookHeaders): string | undefined {
    if (this.webhookUrl) return this.webhookUrl
    const host = webhookHeaderValue(headers, 'x-forwarded-host')?.split(',')[0]?.trim()
      || webhookHeaderValue(headers, 'host')
    if (!host) return undefined
    const proto = (webhookHeaderValue(headers, 'x-forwarded-proto')?.split(',')[0] ?? '').trim() || 'http'
    return `${proto}://${host}${DEFAULT_WEBHOOK_PATH}`
  }
}
