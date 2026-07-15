// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { postJson, providerUrl, type ChannelFetch } from './http.js'
import {
  arrayValue,
  outboundDestination,
  recordValue,
  requiredOption,
  stringValue,
  urlWithQuery,
  type RelayChannelTransport,
} from './relay.js'
import { createChannelMessage, MessageDirection, type ChannelMessage } from './types.js'
import { parseJsonBody, WebhookChannel, type WebhookHeaders } from './webhooks.js'

/** BlueBubbles webhook relay and HTTP sender; no server event socket is owned. */
export const BLUEBUBBLES_TRANSPORT: RelayChannelTransport = {
  inbound: 'webhook-relay',
  outbound: 'http-api',
  unsupported: ['BlueBubbles persistent event socket', 'attachment-only message delivery'],
}

export interface BlueBubblesChannelOptions {
  readonly fetchImplementation?: ChannelFetch
  readonly password: string
  readonly serverUrl: string
}

/**
 * Self-hosted BlueBubbles iMessage webhook relay and REST sender.
 *
 * Inbound events must be pushed by BlueBubbles to Xerxes. The adapter does
 * not establish the bridge's persistent event socket, and never logs URLs
 * that contain the server password.
 */
export class BlueBubblesChannel extends WebhookChannel {
  readonly name = 'bluebubbles'
  readonly transport = BLUEBUBBLES_TRANSPORT

  private readonly fetchImplementation: ChannelFetch | undefined
  private readonly password: string
  private readonly serverUrl: string

  constructor(options: BlueBubblesChannelOptions) {
    super()
    this.fetchImplementation = options.fetchImplementation
    this.password = requiredOption(options.password, 'BlueBubbles password')
    this.serverUrl = requiredOption(options.serverUrl, 'BlueBubbles serverUrl')
  }

  protected parseInbound(
    _headers: WebhookHeaders,
    body: Uint8Array,
  ): readonly ChannelMessage[] {
    const payload = parseJsonBody(body)
    const data = Object.keys(recordValue(payload.data)).length ? recordValue(payload.data) : payload
    const text = stringValue(data.text) || stringValue(data.body)
    if (!text) {
      return []
    }
    const chats = arrayValue(data.chats)
    const firstChat = chats.length ? recordValue(chats[0]) : recordValue(data.chat)
    const handle = recordValue(data.handle)
    return [createChannelMessage({
      channel: this.name,
      direction: MessageDirection.INBOUND,
      text,
      channelUserId: stringValue(handle.address),
      roomId: stringValue(firstChat.guid),
      platformMessageId: stringValue(data.guid),
      metadata: { is_from_me: data.isFromMe === true },
    })]
  }

  protected async sendOutbound(message: ChannelMessage): Promise<void> {
    const chatGuid = outboundDestination(message, 'BlueBubbles')
    const endpoint = urlWithQuery(
      providerUrl(this.serverUrl, 'api/v1/message/text'),
      'password',
      this.password,
    )
    await postJson(endpoint, {
      body: { chatGuid, message: message.text, method: 'private-api' },
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
    })
  }
}
