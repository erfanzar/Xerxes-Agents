// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { postJson, type ChannelFetch } from './http.js'
import {
  recordValue,
  requiredOption,
  stringValue,
  throwOnProviderErrorCode,
  type RelayChannelTransport,
} from './relay.js'
import { createChannelMessage, MessageDirection, type ChannelMessage } from './types.js'
import { parseJsonBody, WebhookChannel, type WebhookHeaders } from './webhooks.js'

/** DingTalk's signed/stream transports stay upstream; this uses webhook relay only. */
export const DINGTALK_TRANSPORT: RelayChannelTransport = {
  inbound: 'webhook-relay',
  outbound: 'incoming-webhook',
  unsupported: ['DingTalk stream-mode connections', 'incoming webhook signature verification'],
}

export interface DingTalkChannelOptions {
  readonly fetchImplementation?: ChannelFetch
  /** Conversation-scoped incoming webhook URL, including any access token. */
  readonly webhookUrl: string
}

/**
 * DingTalk outgoing-webhook relay and incoming-webhook sender.
 *
 * The configured URL determines the outbound conversation. Signature checks
 * for public inbound endpoints remain the responsibility of the HTTP edge.
 */
export class DingTalkChannel extends WebhookChannel {
  readonly name = 'dingtalk'
  readonly transport = DINGTALK_TRANSPORT

  private readonly fetchImplementation: ChannelFetch | undefined
  private readonly webhookUrl: string

  constructor(options: DingTalkChannelOptions) {
    super()
    this.fetchImplementation = options.fetchImplementation
    this.webhookUrl = requiredOption(options.webhookUrl, 'DingTalk webhookUrl')
  }

  protected parseInbound(
    _headers: WebhookHeaders,
    body: Uint8Array,
  ): readonly ChannelMessage[] {
    const payload = parseJsonBody(body)
    if (!Object.keys(payload).length) {
      return []
    }
    const text = stringValue(recordValue(payload.text).content) || stringValue(payload.content)
    if (!text) {
      return []
    }
    return [createChannelMessage({
      channel: this.name,
      direction: MessageDirection.INBOUND,
      text,
      channelUserId: stringValue(payload.senderId) || stringValue(payload.senderStaffId),
      roomId: stringValue(payload.conversationId),
      platformMessageId: stringValue(payload.msgId),
      metadata: { sender_nick: stringValue(payload.senderNick) },
    })]
  }

  protected async sendOutbound(message: ChannelMessage): Promise<void> {
    const response = await postJson(this.webhookUrl, {
      body: { msgtype: 'text', text: { content: message.text } },
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
    })
    throwOnProviderErrorCode(response, 'DingTalk', 'errcode')
  }
}
