// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { postJson, type ChannelFetch } from './http.js'
import { createChannelMessage, MessageDirection, type ChannelAttachment, type ChannelMessage } from './types.js'
import {
  parseJsonBody,
  WebhookChannel,
  type WebhookHeaders,
} from './webhooks.js'

export interface GenericWebhookRequest {
  readonly body: Uint8Array
  readonly headers: WebhookHeaders
  readonly json: Readonly<Record<string, unknown>>
}

export type GenericInboundParser = (
  request: GenericWebhookRequest,
) => readonly ChannelMessage[] | Promise<readonly ChannelMessage[]>

export type GenericOutboundSender = (
  message: ChannelMessage,
) => Promise<void> | void

export interface GenericWebhookChannelOptions {
  readonly fetchImplementation?: ChannelFetch
  readonly name: string
  /** Custom parser. The default accepts a normalized JSON message payload. */
  readonly parseInbound?: GenericInboundParser
  /** Extra headers for the optional JSON outbound endpoint. */
  readonly outboundHeaders?: Readonly<Record<string, string>>
  /** Optional endpoint for a simple JSON outbound webhook. */
  readonly outboundUrl?: string
  /** Customizes the payload posted to `outboundUrl`. */
  readonly serializeOutbound?: (message: ChannelMessage) => unknown
  /** Custom outbound sender, useful for non-JSON or signed endpoints. */
  readonly sendOutbound?: GenericOutboundSender
}

/**
 * Configurable HTTP-webhook channel for integrations without a dedicated adapter.
 *
 * The default inbound shape is a JSON object (or `messages` array) containing
 * `text`, `room_id`, `channel_user_id`, optional `attachments`, and `metadata`.
 * Its adapter name always wins over any payload-supplied channel field.
 */
export class GenericWebhookChannel extends WebhookChannel {
  readonly name: string

  private readonly fetchImplementation: ChannelFetch | undefined
  private readonly inboundParser: GenericInboundParser | undefined
  private readonly outboundHeaders: Readonly<Record<string, string>>
  private readonly outboundUrl: string | undefined
  private readonly outboundSerializer: (message: ChannelMessage) => unknown
  private readonly outboundSender: GenericOutboundSender | undefined

  constructor(options: GenericWebhookChannelOptions) {
    super()
    if (!options.name.trim()) {
      throw new TypeError('generic webhook channel name must not be empty')
    }
    this.name = options.name
    this.fetchImplementation = options.fetchImplementation
    this.inboundParser = options.parseInbound
    this.outboundHeaders = { ...options.outboundHeaders }
    this.outboundUrl = options.outboundUrl
    this.outboundSerializer = options.serializeOutbound ?? defaultOutboundPayload
    this.outboundSender = options.sendOutbound
  }

  protected async parseInbound(
    headers: WebhookHeaders,
    body: Uint8Array,
  ): Promise<readonly ChannelMessage[]> {
    const json = parseJsonBody(body)
    if (this.inboundParser) {
      const parsed = await this.inboundParser({ body, headers, json })
      return parsed.map(message => ({
        ...message,
        channel: this.name,
        attachments: message.attachments.map(attachment => ({ ...attachment })),
        metadata: { ...message.metadata },
      }))
    }
    return genericMessages(this.name, json)
  }

  protected async sendOutbound(message: ChannelMessage): Promise<void> {
    if (this.outboundSender) {
      await this.outboundSender(message)
      return
    }
    if (!this.outboundUrl) {
      throw new Error(`generic webhook channel '${this.name}' has no outbound sender`)
    }
    await postJson(this.outboundUrl, {
      body: this.outboundSerializer(message),
      headers: this.outboundHeaders,
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
    })
  }
}

function genericMessages(
  channel: string,
  payload: Readonly<Record<string, unknown>>,
): readonly ChannelMessage[] {
  const candidates = Array.isArray(payload.messages) ? payload.messages : [payload]
  const messages: ChannelMessage[] = []
  for (const candidate of candidates) {
    if (!isRecord(candidate) || typeof candidate.text !== 'string') {
      continue
    }
    const userId = stringField(candidate, 'user_id')
    const channelUserId = stringField(candidate, 'channel_user_id')
    const roomId = stringField(candidate, 'room_id')
    const replyTo = stringField(candidate, 'reply_to')
    const platformMessageId = stringField(candidate, 'platform_message_id')
    messages.push(createChannelMessage({
      channel,
      text: candidate.text,
      direction: MessageDirection.INBOUND,
      ...(userId ? { userId } : {}),
      ...(channelUserId ? { channelUserId } : {}),
      ...(roomId ? { roomId } : {}),
      ...(replyTo ? { replyTo } : {}),
      ...(platformMessageId ? { platformMessageId } : {}),
      attachments: attachmentFields(candidate.attachments),
      metadata: recordField(candidate, 'metadata'),
    }))
  }
  return messages
}

function defaultOutboundPayload(message: ChannelMessage): Record<string, unknown> {
  return {
    text: message.text,
    channel: message.channel,
    room_id: message.roomId ?? null,
    reply_to: message.replyTo ?? null,
    metadata: { ...message.metadata },
  }
}

function attachmentFields(value: unknown): readonly ChannelAttachment[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.filter(isRecord).map(item => ({ ...item }))
}

function recordField(value: Readonly<Record<string, unknown>>, name: string): Record<string, unknown> {
  const candidate = value[name]
  return isRecord(candidate) ? { ...candidate } : {}
}

function stringField(value: Readonly<Record<string, unknown>>, name: string): string | undefined {
  const candidate = value[name]
  return typeof candidate === 'string' && candidate ? candidate : undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
