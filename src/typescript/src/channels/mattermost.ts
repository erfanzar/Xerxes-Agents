// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { postJson, providerUrl, type ChannelFetch } from './http.js'
import { requiredOption, stringValue, type RelayChannelTransport } from './relay.js'
import { createChannelMessage, MessageDirection, type ChannelMessage } from './types.js'
import { parseJsonBody, WebhookChannel, type WebhookHeaders } from './webhooks.js'

/** Mattermost is webhook-in/REST-out; it does not own a WebSocket connection. */
export const MATTERMOST_TRANSPORT: RelayChannelTransport = {
  inbound: 'webhook-relay',
  outbound: 'http-api',
  unsupported: ['Mattermost WebSocket and post polling transports'],
}

export interface MattermostChannelOptions {
  readonly baseUrl: string
  readonly botToken: string
  readonly fetchImplementation?: ChannelFetch
}

/**
 * Mattermost outgoing-webhook relay and v4 REST sender.
 *
 * The host must route Mattermost callbacks to `handleWebhook`; this adapter
 * intentionally does not maintain a Mattermost WebSocket session.
 */
export class MattermostChannel extends WebhookChannel {
  readonly name = 'mattermost'
  readonly transport = MATTERMOST_TRANSPORT

  private readonly baseUrl: string
  private readonly botToken: string
  private readonly fetchImplementation: ChannelFetch | undefined

  constructor(options: MattermostChannelOptions) {
    super()
    this.baseUrl = requiredOption(options.baseUrl, 'Mattermost baseUrl')
    this.botToken = requiredOption(options.botToken, 'Mattermost botToken')
    this.fetchImplementation = options.fetchImplementation
  }

  protected parseInbound(
    _headers: WebhookHeaders,
    body: Uint8Array,
  ): readonly ChannelMessage[] {
    const payload = parseJsonBody(body)
    if (!Object.keys(payload).length) {
      return []
    }
    return [createChannelMessage({
      channel: this.name,
      direction: MessageDirection.INBOUND,
      text: stringValue(payload.text),
      channelUserId: stringValue(payload.user_id),
      roomId: stringValue(payload.channel_id),
      platformMessageId: stringValue(payload.post_id),
      metadata: { team_id: stringValue(payload.team_id) },
    })]
  }

  protected async sendOutbound(message: ChannelMessage): Promise<void> {
    if (!message.roomId) {
      throw new TypeError('Mattermost outbound messages require roomId')
    }
    const payload: Record<string, unknown> = {
      channel_id: message.roomId,
      message: message.text,
    }
    if (message.replyTo) {
      payload.root_id = message.replyTo
    }
    await postJson(providerUrl(this.baseUrl, 'api/v4/posts'), {
      body: payload,
      headers: { Authorization: `Bearer ${this.botToken}` },
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
    })
  }
}
