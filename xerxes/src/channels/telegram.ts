// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { timingSafeEqual } from 'node:crypto'

import { postJson, providerUrl, type ChannelFetch } from './http.js'
import { scanContextContent } from '../security/promptScanner.js'
import { createChannelMessage, MessageDirection, type ChannelMessage } from './types.js'
import {
  parseJsonBody,
  WebhookChannel,
  type WebhookHeaders,
} from './webhooks.js'

const TELEGRAM_API_BASE = 'https://api.telegram.org/'
const DEFAULT_MAX_PAYLOAD_BYTES = 256 * 1024
const PATH_REDACTION = /(?:\/Users\/[^\s'"]+|\/home\/[^\s'"]+|\/private\/[^\s'"]+|\/var\/[^\s'"]+|\/tmp\/[^\s'"]+|~\/\.xerxes[^\s'"]*)/g
const TRACEBACK_REDACTION = /Traceback \(most recent call last\):.*?(?=\n\n|$)/gs

type StringList = string | readonly string[] | ReadonlySet<string> | undefined

export interface TelegramChannelOptions {
  readonly acceptEditedMessages?: boolean
  /** Allowlist of Telegram user IDs. Used only when requireAllowedSender is enabled. */
  readonly allowedUserIds?: StringList
  /** Allowlist of Telegram usernames (with or without @). */
  readonly allowedUsernames?: StringList
  /** Override only for a Telegram-compatible API or tests. */
  readonly apiBaseUrl?: string
  /** Bot @username used for exact group-addressing checks. */
  readonly botUsername?: string
  readonly fetchImplementation?: ChannelFetch
  /** Per-channel hard cap enforced after the webhook body is read. */
  readonly maxPayloadBytes?: number
  /** Enforce a fail-closed Telegram sender allowlist. */
  readonly requireAllowedSender?: boolean | string
  readonly token: string
  /** Register this public callback URL when the channel starts in webhook mode. Requires webhookSecretToken. */
  readonly webhookUrl?: string
  /** Telegram's secret-token header; mandatory whenever webhookUrl is configured. */
  readonly webhookSecretToken?: string
}

export interface TelegramUpdatesOptions {
  readonly offset?: number
  readonly signal?: AbortSignal
  readonly timeout?: number
}

/** Telegram Bot API webhook adapter with optional long-poll helpers. */
export class TelegramChannel extends WebhookChannel {
  readonly name = 'telegram'
  readonly maxWebhookBodyBytes: number

  private readonly acceptEditedMessages: boolean
  private readonly allowedUserIds: ReadonlySet<string>
  private readonly allowedUsernames: ReadonlySet<string>
  private readonly apiBaseUrl: string
  private readonly botUsername: string
  private readonly fetchImplementation: ChannelFetch | undefined
  private readonly requireAllowedSender: boolean
  private readonly token: string
  private readonly webhookSecretToken: string
  private readonly webhookUrl: string

  constructor(options: TelegramChannelOptions) {
    super()
    if (!options.token) {
      throw new TypeError('Telegram bot token must not be empty')
    }
    this.token = options.token
    this.apiBaseUrl = options.apiBaseUrl ?? TELEGRAM_API_BASE
    this.fetchImplementation = options.fetchImplementation
    this.acceptEditedMessages = options.acceptEditedMessages ?? false
    this.allowedUserIds = identifierSet(options.allowedUserIds)
    this.allowedUsernames = usernameSet(options.allowedUsernames)
    this.botUsername = normalizedUsername(options.botUsername ?? '')
    this.maxWebhookBodyBytes = payloadLimit(options.maxPayloadBytes ?? DEFAULT_MAX_PAYLOAD_BYTES)
    this.requireAllowedSender = asBoolean(options.requireAllowedSender, false)
    this.webhookSecretToken = options.webhookSecretToken?.trim() ?? ''
    this.webhookUrl = options.webhookUrl?.trim() ?? ''
    if (this.webhookUrl && !this.webhookSecretToken) {
      throw new TypeError(
        'Telegram webhookUrl requires webhookSecretToken so forged updates cannot reach agent sessions',
      )
    }
  }

  override async start(onInbound: import('./base.js').InboundHandler): Promise<void> {
    await super.start(onInbound)
    if (!this.webhookUrl) return
    try {
      await this.setWebhook(this.webhookUrl)
    } catch (error) {
      await super.stop()
      throw error
    }
  }

  override async handleWebhook(
    headers: WebhookHeaders,
    body: Uint8Array,
  ): Promise<import('./webhooks.js').WebhookResponse> {
    if (body.byteLength > this.maxWebhookBodyBytes) {
      return { status: 413, body: 'payload too large' }
    }
    if (this.webhookSecretToken && !telegramSecretMatches(headers, this.webhookSecretToken)) {
      return { status: 401, body: 'unauthorized' }
    }
    return super.handleWebhook(headers, body)
  }

  /**
   * Ingest one update received through authenticated Bot API long polling.
   *
   * Polling is already authenticated by the bot token, so this path bypasses
   * the webhook-only secret-token check while retaining the shared parser and
   * inbound error containment.
   */
  async ingestPolledUpdate(body: Uint8Array): Promise<import('./webhooks.js').WebhookResponse> {
    if (body.byteLength > this.maxWebhookBodyBytes) {
      return { status: 413, body: 'payload too large' }
    }
    return super.handleWebhook({}, body)
  }

  /** Long-poll Telegram updates when a deployment does not expose a webhook. */
  async getUpdates(options: TelegramUpdatesOptions = {}): Promise<Readonly<Record<string, unknown>>> {
    const timeout = options.timeout ?? 30
    if (!Number.isInteger(timeout) || timeout < 0) {
      throw new RangeError('Telegram getUpdates timeout must be a non-negative integer')
    }
    const allowedUpdates = this.acceptEditedMessages
      ? ['message', 'edited_message']
      : ['message']
    const payload: Record<string, unknown> = {
      timeout,
      allowed_updates: allowedUpdates,
    }
    if (options.offset !== undefined) {
      payload.offset = options.offset
    }
    return asRecord(await this.request('getUpdates', payload, options.signal))
  }

  /** Send a Telegram text message and return Telegram's API envelope. */
  async sendText(
    chatId: string,
    text: string,
    replyTo?: string,
  ): Promise<Readonly<Record<string, unknown>>> {
    if (!chatId) {
      throw new TypeError('Telegram chat id must not be empty')
    }
    const payload: Record<string, unknown> = { chat_id: chatId, text: sanitizeTelegramOutbound(text) }
    if (replyTo) {
      payload.reply_to_message_id = replyTo
    }
    return asRecord(await this.request('sendMessage', payload))
  }

  /** Replace a previously sent Telegram message's text. */
  async editText(
    chatId: string,
    messageId: string,
    text: string,
  ): Promise<Readonly<Record<string, unknown>>> {
    if (!chatId || !messageId) {
      throw new TypeError('Telegram editText requires chatId and messageId')
    }
    return asRecord(await this.request('editMessageText', {
      chat_id: chatId,
      message_id: messageId,
      text: sanitizeTelegramOutbound(text),
    }))
  }

  /** Show Telegram's transient typing indicator for the target chat. */
  async sendTyping(chatId: string | undefined): Promise<void> {
    const target = chatId?.trim()
    if (!target) return
    await this.request('sendChatAction', { chat_id: target, action: 'typing' })
  }

  /** Register a public Telegram Bot API webhook with the configured secret token. */
  async setWebhook(webhookUrl: string): Promise<Readonly<Record<string, unknown>>> {
    const url = webhookUrl.trim()
    if (!url) throw new TypeError('Telegram webhook URL must not be empty')
    const payload: Record<string, unknown> = { url }
    if (this.webhookSecretToken) payload.secret_token = this.webhookSecretToken
    return asRecord(await this.request('setWebhook', payload))
  }

  /** Clear Telegram's registered webhook before switching to long polling. */
  async deleteWebhook(): Promise<Readonly<Record<string, unknown>>> {
    return asRecord(await this.request('deleteWebhook', {}))
  }

  protected parseInbound(
    _headers: WebhookHeaders,
    body: Uint8Array,
  ): readonly ChannelMessage[] {
    const update = parseJsonBody(body)
    const message = asRecord(update.message)
    const editedMessage = this.acceptEditedMessages ? asRecord(update.edited_message) : {}
    const envelope = Object.keys(message).length ? message : editedMessage
    if (!Object.keys(envelope).length) {
      return []
    }
    const sender = asRecord(envelope.from)
    const chat = asRecord(envelope.chat)
    const rawText = stringOrEmpty(envelope.text) || stringOrEmpty(envelope.caption)
    const userId = stringOrEmpty(sender.id)
    const username = stringOrEmpty(sender.username)
    const chatType = stringOrEmpty(chat.type)
    if (!this.senderAllowed(userId, username) || this.groupMessageIsNotAddressed(rawText, chatType)) {
      return []
    }
    return [createChannelMessage({
      channel: this.name,
      text: scanContextContent(rawText, 'telegram:inbound'),
      direction: MessageDirection.INBOUND,
      channelUserId: userId,
      roomId: stringOrEmpty(chat.id),
      platformMessageId: stringOrEmpty(envelope.message_id),
      metadata: {
        username,
        first_name: stringOrEmpty(sender.first_name),
        last_name: stringOrEmpty(sender.last_name),
        chat_type: chatType,
        chat_title: stringOrEmpty(chat.title),
        thread_id: stringOrEmpty(envelope.message_thread_id),
      },
    })]
  }

  protected async sendOutbound(message: ChannelMessage): Promise<void> {
    const chatId = message.roomId ?? message.channelUserId
    if (!chatId) {
      throw new TypeError('Telegram outbound messages require roomId or channelUserId')
    }
    await this.sendText(chatId, sanitizeTelegramOutbound(message.text), message.replyTo)
  }

  private senderAllowed(userId: string, username: string): boolean {
    if (!this.requireAllowedSender) return true
    if (!this.allowedUserIds.size && !this.allowedUsernames.size) return false
    return this.allowedUserIds.has(userId) || this.allowedUsernames.has(normalizedUsername(username))
  }

  private groupMessageIsNotAddressed(text: string, chatType: string): boolean {
    if (chatType.toLowerCase() !== 'group' && chatType.toLowerCase() !== 'supergroup') return false
    // Generic adapter configurations historically delivered every group message.
    // Exact-address filtering becomes active only once an operator supplies the
    // bot username needed to make the decision without guessing.
    if (!this.botUsername) return false
    const normalized = text.trim().toLowerCase()
    if (!normalized) return true
    const bot = this.botUsername
    if (bot && normalized.includes(`@${bot}`)) return false
    if (bot && (normalized.startsWith(`/${bot}`) || normalized.startsWith(`/xerxes@${bot}`))) return false
    return !(normalized === '/xerxes' || normalized.startsWith('/xerxes '))
  }

  private async request(method: string, body: unknown, signal: AbortSignal | undefined = undefined): Promise<unknown> {
    const result = await postJson(this.methodUrl(method), {
      body,
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
      ...(signal ? { signal } : {}),
    })
    if (isRecord(result) && result.ok === false) {
      const description = typeof result.description === 'string'
        ? `: ${result.description}`
        : ''
      throw new Error(`Telegram API ${method} failed${description}`)
    }
    return result
  }

  private methodUrl(method: string): string {
    return providerUrl(this.apiBaseUrl, `bot${this.token}/${method}`)
  }
}

function asRecord(value: unknown): Readonly<Record<string, unknown>> {
  return isRecord(value) ? value : {}
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringOrEmpty(value: unknown): string {
  return value === undefined || value === null ? '' : String(value)
}

function identifierSet(value: StringList): ReadonlySet<string> {
  const items = typeof value === 'string' ? value.split(',') : value ? [...value] : []
  return new Set(items.map(item => String(item).trim()).filter(Boolean))
}

function usernameSet(value: StringList): ReadonlySet<string> {
  return new Set([...identifierSet(value)].map(normalizedUsername).filter(Boolean))
}

function normalizedUsername(value: string): string {
  return value.trim().replace(/^@/, '').toLowerCase()
}

function payloadLimit(value: number): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError('Telegram maxPayloadBytes must be a positive safe integer')
  }
  return value
}

function asBoolean(value: boolean | string | undefined, fallback: boolean): boolean {
  if (value === undefined) return fallback
  if (typeof value === 'boolean') return value
  return !new Set(['0', 'false', 'no', 'off', '']).has(value.trim().toLowerCase())
}

function telegramSecretMatches(headers: WebhookHeaders, expected: string): boolean {
  const actual = Object.entries(headers)
    .find(([name]) => name.toLowerCase() === 'x-telegram-bot-api-secret-token')?.[1] ?? ''
  const actualBytes = Buffer.from(actual)
  const expectedBytes = Buffer.from(expected)
  return actualBytes.byteLength === expectedBytes.byteLength && timingSafeEqual(actualBytes, expectedBytes)
}

function sanitizeTelegramOutbound(text: string): string {
  if (!text) return text
  return text
    .replace(TRACEBACK_REDACTION, '[traceback redacted]')
    .replace(PATH_REDACTION, '[path redacted]')
}
