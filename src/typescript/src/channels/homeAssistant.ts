// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { postJson, providerUrl, type ChannelFetch } from './http.js'
import { recordValue, requiredOption, stringValue, type RelayChannelTransport } from './relay.js'
import { createChannelMessage, MessageDirection, type ChannelMessage } from './types.js'
import { parseJsonBody, WebhookChannel, type WebhookHeaders } from './webhooks.js'

/** Home Assistant conversation webhook relay with persistent-notification delivery. */
export const HOME_ASSISTANT_TRANSPORT: RelayChannelTransport = {
  inbound: 'webhook-relay',
  outbound: 'http-api',
  unsupported: ['Home Assistant WebSocket event subscriptions', 'configured notify-platform delivery'],
}

export interface HomeAssistantChannelOptions {
  readonly accessToken: string
  readonly baseUrl: string
  readonly fetchImplementation?: ChannelFetch
  readonly notificationTitle?: string
}

/**
 * Home Assistant conversation-webhook relay and persistent notification sender.
 *
 * The adapter does not establish Home Assistant's WebSocket connection and
 * deliberately avoids guessing an operator's preferred notify integration.
 */
export class HomeAssistantChannel extends WebhookChannel {
  readonly name = 'home_assistant'
  readonly transport = HOME_ASSISTANT_TRANSPORT

  private readonly accessToken: string
  private readonly baseUrl: string
  private readonly fetchImplementation: ChannelFetch | undefined
  private readonly notificationTitle: string

  constructor(options: HomeAssistantChannelOptions) {
    super()
    this.accessToken = requiredOption(options.accessToken, 'Home Assistant accessToken')
    this.baseUrl = requiredOption(options.baseUrl, 'Home Assistant baseUrl')
    this.fetchImplementation = options.fetchImplementation
    this.notificationTitle = options.notificationTitle ?? 'Xerxes'
  }

  protected parseInbound(
    _headers: WebhookHeaders,
    body: Uint8Array,
  ): readonly ChannelMessage[] {
    const payload = parseJsonBody(body)
    const text = stringValue(payload.text)
      || stringValue(recordValue(payload.input).text)
      || stringValue(payload.message)
    if (!text) {
      return []
    }
    return [createChannelMessage({
      channel: this.name,
      direction: MessageDirection.INBOUND,
      text,
      channelUserId: stringValue(payload.user_id),
      roomId: stringValue(payload.conversation_id),
      platformMessageId: stringValue(payload.event_id),
      metadata: { language: stringValue(payload.language) || 'en' },
    })]
  }

  protected async sendOutbound(message: ChannelMessage): Promise<void> {
    await postJson(providerUrl(this.baseUrl, 'api/services/persistent_notification/create'), {
      body: {
        title: this.notificationTitle,
        message: message.text,
        notification_id: message.messageId,
      },
      headers: { Authorization: `Bearer ${this.accessToken}` },
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
    })
  }
}
