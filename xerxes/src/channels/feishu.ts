// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { postJson, providerUrl, type ChannelFetch } from './http.js'
import {
  recordValue,
  resolveAccessToken,
  stringValue,
  throwOnProviderErrorCode,
  urlWithQuery,
  type ChannelAccessTokenProvider,
  type RelayChannelTransport,
} from './relay.js'
import { createChannelMessage, MessageDirection, type ChannelMessage } from './types.js'
import {
  parseJsonBody,
  WebhookChannel,
  type WebhookHeaders,
  type WebhookResponse,
} from './webhooks.js'

const FEISHU_API_BASE = 'https://open.feishu.cn/'

/** Feishu/Lark event callbacks are relayed here; encrypted callbacks are not decrypted. */
export const FEISHU_TRANSPORT: RelayChannelTransport = {
  inbound: 'webhook-relay',
  outbound: 'http-api',
  unsupported: ['encrypted event payload decryption', 'persistent WebSocket event delivery'],
}

export interface FeishuChannelOptions {
  /** Override for Lark-compatible deployments or tests. */
  readonly apiBaseUrl?: string
  readonly fetchImplementation?: ChannelFetch
  readonly tenantAccessToken?: string
  /** Called on every outbound request to support externally refreshed tokens. */
  readonly tokenProvider?: ChannelAccessTokenProvider
}

/**
 * Feishu/Lark webhook relay and `im/v1/messages` text sender.
 *
 * URL-verification challenges are answered, while encrypted webhook events
 * and long-lived provider connections intentionally remain outside this
 * dependency-free adapter.
 */
export class FeishuChannel extends WebhookChannel {
  readonly name = 'feishu'
  readonly transport = FEISHU_TRANSPORT

  private readonly apiBaseUrl: string
  private readonly fetchImplementation: ChannelFetch | undefined
  private readonly staticToken: string
  private readonly tokenProvider: ChannelAccessTokenProvider | undefined

  constructor(options: FeishuChannelOptions = {}) {
    super()
    this.apiBaseUrl = options.apiBaseUrl ?? FEISHU_API_BASE
    this.fetchImplementation = options.fetchImplementation
    this.staticToken = options.tenantAccessToken ?? ''
    this.tokenProvider = options.tokenProvider
  }

  override async handleWebhook(
    headers: WebhookHeaders,
    body: Uint8Array,
  ): Promise<WebhookResponse> {
    const payload = parseJsonBody(body)
    if (payload.type === 'url_verification') {
      return {
        status: 200,
        body: stringValue(payload.challenge),
        headers: { 'content-type': 'text/plain; charset=utf-8' },
      }
    }
    return super.handleWebhook(headers, body)
  }

  protected parseInbound(
    _headers: WebhookHeaders,
    body: Uint8Array,
  ): readonly ChannelMessage[] {
    const payload = parseJsonBody(body)
    if (payload.type === 'url_verification') {
      return []
    }
    const event = recordValue(payload.event)
    const message = recordValue(event.message)
    const sender = recordValue(event.sender)
    const senderId = recordValue(sender.sender_id)
    const text = feishuText(message.content)
    if (!text) {
      return []
    }
    return [createChannelMessage({
      channel: this.name,
      direction: MessageDirection.INBOUND,
      text,
      channelUserId: stringValue(senderId.open_id),
      roomId: stringValue(message.chat_id),
      platformMessageId: stringValue(message.message_id),
      metadata: { message_type: stringValue(message.message_type) },
    })]
  }

  protected async sendOutbound(message: ChannelMessage): Promise<void> {
    if (!message.roomId) {
      throw new TypeError('Feishu outbound messages require roomId')
    }
    const token = await resolveAccessToken(this.staticToken, this.tokenProvider, 'Feishu')
    const url = urlWithQuery(providerUrl(this.apiBaseUrl, 'open-apis/im/v1/messages'), 'receive_id_type', 'chat_id')
    const response = await postJson(url, {
      body: {
        receive_id: message.roomId,
        msg_type: 'text',
        content: JSON.stringify({ text: message.text }),
      },
      headers: { Authorization: `Bearer ${token}` },
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
    })
    throwOnProviderErrorCode(response, 'Feishu', 'code')
  }
}

function feishuText(value: unknown): string {
  const raw = stringValue(value)
  if (!raw) {
    return ''
  }
  try {
    return stringValue(recordValue(JSON.parse(raw)).text)
  } catch {
    return raw
  }
}
