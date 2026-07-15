// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { postJson, providerUrl, type ChannelFetch } from './http.js'
import {
  resolveAccessToken,
  stringValue,
  throwOnProviderErrorCode,
  urlWithQuery,
  type ChannelAccessTokenProvider,
  type RelayChannelTransport,
} from './relay.js'
import { createChannelMessage, MessageDirection, type ChannelMessage } from './types.js'
import { parseJsonBody, WebhookChannel, type WebhookHeaders } from './webhooks.js'

const WECOM_API_BASE = 'https://qyapi.weixin.qq.com/'

/** WeCom JSON bridge relay plus enterprise HTTP API sender. */
export const WECOM_TRANSPORT: RelayChannelTransport = {
  inbound: 'webhook-relay',
  outbound: 'http-api',
  unsupported: ['encrypted XML callback decryption', 'persistent callback connections'],
}

export interface WeComChannelOptions {
  readonly accessToken?: string
  readonly agentId: string | number
  readonly apiBaseUrl?: string
  readonly fetchImplementation?: ChannelFetch
  readonly tokenProvider?: ChannelAccessTokenProvider
}

/**
 * WeCom JSON-webhook relay and enterprise message API adapter.
 *
 * Official encrypted XML callbacks must be decrypted by an upstream relay;
 * this class intentionally accepts only their normalized JSON equivalents.
 */
export class WeComChannel extends WebhookChannel {
  readonly name = 'wecom'
  readonly transport = WECOM_TRANSPORT

  private readonly agentId: string | number
  private readonly apiBaseUrl: string
  private readonly fetchImplementation: ChannelFetch | undefined
  private readonly staticToken: string
  private readonly tokenProvider: ChannelAccessTokenProvider | undefined

  constructor(options: WeComChannelOptions) {
    super()
    if (`${options.agentId}`.trim() === '') {
      throw new TypeError('WeCom agentId must not be empty')
    }
    this.agentId = options.agentId
    this.apiBaseUrl = options.apiBaseUrl ?? WECOM_API_BASE
    this.fetchImplementation = options.fetchImplementation
    this.staticToken = options.accessToken ?? ''
    this.tokenProvider = options.tokenProvider
  }

  protected parseInbound(
    _headers: WebhookHeaders,
    body: Uint8Array,
  ): readonly ChannelMessage[] {
    const payload = parseJsonBody(body)
    const text = stringValue(payload.Content) || stringValue(payload.content)
    if (!text) {
      return []
    }
    const sender = stringValue(payload.FromUserName) || stringValue(payload.from_user)
    return [createChannelMessage({
      channel: this.name,
      direction: MessageDirection.INBOUND,
      text,
      channelUserId: sender,
      roomId: sender,
      platformMessageId: stringValue(payload.MsgId) || stringValue(payload.msg_id),
      metadata: { event: stringValue(payload.Event) || stringValue(payload.event) },
    })]
  }

  protected async sendOutbound(message: ChannelMessage): Promise<void> {
    const recipient = message.channelUserId ?? message.roomId
    if (!recipient) {
      throw new TypeError('WeCom outbound messages require channelUserId or roomId')
    }
    const token = await resolveAccessToken(this.staticToken, this.tokenProvider, 'WeCom')
    const url = urlWithQuery(providerUrl(this.apiBaseUrl, 'cgi-bin/message/send'), 'access_token', token)
    const response = await postJson(url, {
      body: {
        touser: recipient,
        msgtype: 'text',
        agentid: this.agentId,
        text: { content: message.text },
      },
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
    })
    throwOnProviderErrorCode(response, 'WeCom', 'errcode')
  }
}
