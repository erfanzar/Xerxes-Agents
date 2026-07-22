// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHmac, timingSafeEqual } from 'node:crypto'

import { postJson, type ChannelFetch } from './http.js'
import { scanContextContent } from '../security/promptScanner.js'
import { createChannelMessage, MessageDirection, type ChannelMessage } from './types.js'
import {
  parseJsonBody,
  WebhookChannel,
  type WebhookHeaders,
  type WebhookResponse,
} from './webhooks.js'

const SLACK_POST_MESSAGE_URL = 'https://slack.com/api/chat.postMessage'
const DEFAULT_SIGNATURE_SKEW_SECONDS = 5 * 60

export interface SlackAccessToken {
  readonly accessToken: string
}

/** Minimal per-workspace token lookup contract for Slack OAuth installations. */
export interface SlackTokenProvider {
  getValidToken(
    installId: string,
  ): SlackAccessToken | undefined | Promise<SlackAccessToken | undefined>
}

export interface SlackChannelOptions {
  readonly botToken?: string
  readonly fetchImplementation?: ChannelFetch
  readonly installId?: string
  readonly now?: () => number
  /**
   * Reject inbound events when a signing secret has not been configured.
   * Defaults to true (fail closed); set false only as an explicit opt-out
   * for tests or a separately authenticated relay.
   */
  readonly requireSignature?: boolean
  readonly signingSecret?: string
  readonly tokenProvider?: SlackTokenProvider
}

export interface SlackSignatureOptions {
  readonly maxSkewSeconds?: number
  readonly now?: () => number
}

/** Verify Slack Events API HMAC signatures against the untouched request body. */
export function verifySlackSignature(
  signingSecret: string,
  headers: WebhookHeaders,
  body: Uint8Array,
  options: SlackSignatureOptions = {},
): boolean {
  if (!signingSecret) {
    return false
  }
  const timestamp = header(headers, 'x-slack-request-timestamp')
  const supplied = header(headers, 'x-slack-signature')
  if (!timestamp || !supplied || !/^\d+$/.test(timestamp)) {
    return false
  }
  const now = options.now ?? (() => Date.now() / 1000)
  const maxSkewSeconds = options.maxSkewSeconds ?? DEFAULT_SIGNATURE_SKEW_SECONDS
  if (!Number.isFinite(maxSkewSeconds) || maxSkewSeconds < 0) {
    throw new RangeError('Slack signature maxSkewSeconds must be non-negative')
  }
  if (Math.abs(now() - Number(timestamp)) > maxSkewSeconds) {
    return false
  }
  const signed = Buffer.concat([
    Buffer.from(`v0:${timestamp}:`, 'utf8'),
    Buffer.from(body),
  ])
  const expected = `v0=${createHmac('sha256', signingSecret).update(signed).digest('hex')}`
  const expectedBytes = Buffer.from(expected, 'utf8')
  const suppliedBytes = Buffer.from(supplied, 'utf8')
  return expectedBytes.length === suppliedBytes.length
    && timingSafeEqual(expectedBytes, suppliedBytes)
}

/** Slack Events API adapter with optional per-workspace OAuth token resolution. */
export class SlackChannel extends WebhookChannel {
  readonly name = 'slack'

  private readonly botToken: string
  private readonly fetchImplementation: ChannelFetch | undefined
  private readonly installId: string
  private readonly now: () => number
  private readonly requireSignature: boolean
  private readonly signingSecret: string
  private readonly tokenProvider: SlackTokenProvider | undefined

  constructor(options: SlackChannelOptions = {}) {
    super()
    this.botToken = options.botToken ?? ''
    this.fetchImplementation = options.fetchImplementation
    this.installId = options.installId ?? 'default'
    this.now = options.now ?? (() => Date.now() / 1000)
    this.requireSignature = options.requireSignature ?? true
    this.signingSecret = options.signingSecret ?? ''
    this.tokenProvider = options.tokenProvider
  }

  /** Echo Slack's verification challenge while retaining signature validation. */
  override async handleWebhook(
    headers: WebhookHeaders,
    body: Uint8Array,
  ): Promise<WebhookResponse> {
    if (!this.signatureValid(headers, body)) {
      return { status: 200, body: 'ok' }
    }
    const payload = parseJsonBody(body)
    if (payload.type === 'url_verification') {
      return {
        status: 200,
        body: typeof payload.challenge === 'string' ? payload.challenge : '',
        headers: { 'content-type': 'text/plain; charset=utf-8' },
      }
    }
    return super.handleWebhook(headers, body)
  }

  protected parseInbound(
    headers: WebhookHeaders,
    body: Uint8Array,
  ): readonly ChannelMessage[] {
    if (!this.signatureValid(headers, body)) {
      return []
    }
    const payload = parseJsonBody(body)
    if (payload.type === 'url_verification') {
      return []
    }
    const event = asRecord(payload.event)
    const eventType = stringField(event, 'type')
    if (eventType !== 'message' && eventType !== 'app_mention') {
      return []
    }
    if (event.bot_id !== undefined || event.subtype !== undefined) {
      return []
    }
    const teamId = (this.signingSecret || this.botToken)
      ? stringOrEmpty(payload.team_id)
      : ''
    return [createChannelMessage({
      channel: this.name,
      text: scanContextContent(stringOrEmpty(event.text), 'slack:inbound'),
      direction: MessageDirection.INBOUND,
      channelUserId: stringOrEmpty(event.user),
      roomId: stringOrEmpty(event.channel),
      platformMessageId: stringOrEmpty(event.ts),
      metadata: {
        team_id: teamId,
        thread_ts: stringOrEmpty(event.thread_ts),
        verified_install_id: teamId,
      },
    })]
  }

  protected async sendOutbound(message: ChannelMessage): Promise<void> {
    if (!message.roomId) {
      throw new TypeError('Slack outbound messages require roomId')
    }
    const payload: Record<string, unknown> = {
      channel: message.roomId,
      text: message.text,
    }
    if (message.replyTo) {
      payload.thread_ts = message.replyTo
    }
    const requestedInstall = stringField(message.metadata, 'verified_install_id')
    const result = await postJson<unknown>(SLACK_POST_MESSAGE_URL, {
      body: payload,
      headers: { Authorization: `Bearer ${await this.resolveToken(requestedInstall)}` },
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
    })
    if (isRecord(result) && result.ok === false) {
      const error = stringField(result, 'error')
      throw new Error(`Slack chat.postMessage failed${error ? `: ${error}` : ''}`)
    }
  }

  private signatureValid(headers: WebhookHeaders, body: Uint8Array): boolean {
    if (!this.signingSecret) {
      return !this.requireSignature
    }
    return verifySlackSignature(this.signingSecret, headers, body, {
      now: this.now,
    })
  }

  private async resolveToken(installId: string | undefined): Promise<string> {
    if (this.botToken) {
      return this.botToken
    }
    if (this.tokenProvider) {
      const token = await this.tokenProvider.getValidToken(installId ?? this.installId)
      if (token?.accessToken) {
        return token.accessToken
      }
    }
    throw new Error('Slack bot token unavailable')
  }
}

function header(headers: WebhookHeaders, wanted: string): string {
  for (const [name, value] of Object.entries(headers)) {
    if (name.toLowerCase() === wanted) {
      return value
    }
  }
  return ''
}

function asRecord(value: unknown): Readonly<Record<string, unknown>> {
  return isRecord(value) ? value : {}
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function stringField(record: Readonly<Record<string, unknown>>, field: string): string | undefined {
  const value = record[field]
  return typeof value === 'string' && value ? value : undefined
}

function stringOrEmpty(value: unknown): string {
  return value === undefined || value === null ? '' : String(value)
}
