// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Buffer } from 'node:buffer'

import type { InboundHandler } from './base.js'
import { BunSmtpTransport } from './smtpTransport.js'
import {
  createChannelMessage,
  MessageDirection,
  type ChannelAttachment,
  type ChannelMessage,
} from './types.js'
import { parseJsonBody, WebhookChannel, type WebhookHeaders } from './webhooks.js'

const DEFAULT_SMTP_HOST = 'localhost'
const DEFAULT_SMTP_PORT = 25
const DEFAULT_SUBJECT = 'Re:'
const HEADER_BREAK = /[\r\n]/
const MIME_WORD = /=\?[^?]+\?([bBqQ])\?([^?]*)\?=/g

/** A normalized inbound message emitted by a host-owned IMAP client or poller. */
export interface EmailInboundMessage {
  readonly attachments?: readonly ChannelAttachment[]
  readonly date?: Date | string
  readonly from?: string
  readonly html?: string
  readonly htmlEncoding?: string
  readonly inReplyTo?: string
  readonly messageId?: string
  readonly subject?: string
  readonly text?: string
  readonly textEncoding?: string
  readonly to?: string | readonly string[]
}

/** A host-owned IMAP boundary. The transport owns sockets, polling, and provider authentication. */
export interface EmailImapTransport {
  start(onInbound: (message: EmailInboundMessage) => Promise<void>): Promise<void> | void
  stop(): Promise<void> | void
}

/** SMTP envelope and RFC 5322/MIME payload prepared by the channel for a host-owned sender. */
export interface EmailSmtpSendRequest {
  readonly authentication?: Readonly<{ readonly password: string; readonly username: string }>
  readonly from: string
  readonly host: string
  readonly mime: string
  readonly port: number
  readonly startTls: boolean
  readonly subject: string
  readonly text: string
  readonly to: string
}

/** SMTP sender boundary. Bun supplies a direct SMTP sender by default; hosts can override it. */
export interface EmailSmtpTransport {
  send(request: EmailSmtpSendRequest): Promise<void> | void
}

export interface EmailChannelOptions {
  /** Use Bun's direct SMTP/STARTTLS sender when no host transport is supplied. */
  readonly directSmtp?: boolean
  /** Envelope/header sender. Falls back to smtpUser when omitted. */
  readonly fromAddress?: string
  /** Optional real IMAP client/poller supplied by the embedding host. */
  readonly imapTransport?: EmailImapTransport
  /** Reject startup when no injected IMAP transport is available. */
  readonly requireImapTransport?: boolean
  readonly smtpHost?: string
  readonly smtpPassword?: string
  readonly smtpPort?: number
  /** Optional real SMTP sender supplied by the embedding host. */
  readonly smtpTransport?: EmailSmtpTransport
  readonly smtpUser?: string
}

export type EmailImapChannelOptions = EmailChannelOptions

export class EmailChannelConfigurationError extends Error {
  constructor(message: string) {
    super(`Email channel configuration: ${message}`)
    this.name = new.target.name
  }
}

export class EmailOutboundMessageError extends Error {
  constructor(message: string) {
    super(`Email outbound message: ${message}`)
    this.name = new.target.name
  }
}

export class EmailTransportUnavailableError extends Error {
  readonly transport: 'IMAP' | 'SMTP'

  constructor(transport: 'IMAP' | 'SMTP') {
    super(`Email ${transport} transport is unavailable; inject a real ${transport} transport`)
    this.name = new.target.name
    this.transport = transport
  }
}

export class EmailTransportError extends Error {
  readonly cause: unknown
  readonly transport: 'IMAP' | 'SMTP'

  constructor(transport: 'IMAP' | 'SMTP', operation: string, cause: unknown) {
    super(`Email ${transport} transport ${operation} failed`)
    this.name = new.target.name
    this.cause = cause
    this.transport = transport
  }
}

/**
 * Email adapter with webhook/IMAP inbound delivery and SMTP outbound delivery.
 *
 * The adapter deliberately has no direct IMAP socket implementation. An embedding host supplies
 * real IMAP clients; this layer validates configuration, normalizes MIME-adjacent content,
 * maps inbound mail to ChannelMessage, and constructs safe text/plain MIME send requests.
 * Native SMTP is used by default and can be replaced with a host transport when desired.
 */
export class EmailChannel extends WebhookChannel {
  readonly name = 'email'
  readonly transport = {
    inbound: 'injected-imap-or-webhook',
    outbound: 'bun-native-smtp-or-injected',
    unsupported: ['direct Bun IMAP sockets', 'complete RFC 2045 multipart parsing'],
  } as const

  private readonly fromAddress: string
  private readonly imapTransport: EmailImapTransport | undefined
  private imapStarted = false
  private readonly requireImapTransport: boolean
  private readonly smtpHost: string
  private readonly smtpPassword: string
  private readonly smtpPort: number
  private readonly smtpTransport: EmailSmtpTransport | undefined
  private readonly smtpUser: string

  constructor(options: EmailChannelOptions = {}) {
    super()
    this.smtpHost = configuredHost(options.smtpHost ?? DEFAULT_SMTP_HOST)
    this.smtpPort = configuredPort(options.smtpPort ?? DEFAULT_SMTP_PORT)
    this.smtpUser = configuredOptionalHeader(options.smtpUser ?? '', 'smtpUser')
    this.smtpPassword = options.smtpPassword ?? ''
    if ((this.smtpUser && !this.smtpPassword) || (!this.smtpUser && this.smtpPassword)) {
      throw new EmailChannelConfigurationError('smtpUser and smtpPassword must be configured together')
    }
    this.fromAddress = options.fromAddress === undefined
      ? this.smtpUser
      : configuredAddress(options.fromAddress, 'fromAddress')
    this.smtpTransport = options.smtpTransport ?? (options.directSmtp === false ? undefined : new BunSmtpTransport())
    if (options.smtpTransport && !this.fromAddress) {
      throw new EmailChannelConfigurationError('fromAddress or smtpUser is required with an SMTP transport')
    }
    this.imapTransport = options.imapTransport
    this.requireImapTransport = options.requireImapTransport ?? false
  }

  override async start(onInbound: InboundHandler): Promise<void> {
    await super.start(onInbound)
    const transport = this.imapTransport
    if (!transport) {
      if (!this.requireImapTransport) {
        return
      }
      await super.stop()
      throw new EmailTransportUnavailableError('IMAP')
    }
    if (this.imapStarted) {
      return
    }
    try {
      await transport.start(message => this.ingestImap(message))
      this.imapStarted = true
    } catch (error) {
      await super.stop()
      throw new EmailTransportError('IMAP', 'start', error)
    }
  }

  override async stop(): Promise<void> {
    const transport = this.imapTransport
    try {
      if (transport && this.imapStarted) {
        await transport.stop()
      }
    } catch (error) {
      throw new EmailTransportError('IMAP', 'stop', error)
    } finally {
      this.imapStarted = false
      await super.stop()
    }
  }

  protected parseInbound(
    _headers: WebhookHeaders,
    body: Uint8Array,
  ): readonly ChannelMessage[] {
    const payload = parseJsonBody(body)
    if (Object.keys(payload).length === 0) {
      return []
    }
    return [inboundChannelMessage(this.name, payload)]
  }

  protected async sendOutbound(message: ChannelMessage): Promise<void> {
    const transport = this.smtpTransport
    if (!transport) {
      throw new EmailTransportUnavailableError('SMTP')
    }
    if (!this.fromAddress) {
      throw new EmailChannelConfigurationError('fromAddress or smtpUser is required for outbound email')
    }
    const to = outboundAddress(message.roomId ?? message.channelUserId, 'recipient')
    const subject = outboundSubject(message.metadata.subject)
    const replyTo = message.replyTo === undefined ? undefined : outboundHeader(message.replyTo, 'replyTo')
    const text = normalizeOutboundText(message.text)
    const request: EmailSmtpSendRequest = {
      from: this.fromAddress,
      host: this.smtpHost,
      mime: buildPlainTextMime(this.fromAddress, to, subject, text, replyTo),
      port: this.smtpPort,
      startTls: Boolean(this.smtpUser),
      subject,
      text,
      to,
      ...(this.smtpUser
        ? { authentication: { username: this.smtpUser, password: this.smtpPassword } }
        : {}),
    }
    try {
      await transport.send(request)
    } catch (error) {
      throw new EmailTransportError('SMTP', 'send', error)
    }
  }

  private async ingestImap(message: EmailInboundMessage): Promise<void> {
    let body: Uint8Array
    try {
      body = new TextEncoder().encode(JSON.stringify(message))
    } catch (error) {
      throw new EmailTransportError('IMAP', 'serialize inbound message', error)
    }
    const response = await this.handleWebhook({}, body)
    if (response.status >= 400) {
      throw new EmailTransportError('IMAP', `deliver inbound message (${response.status})`, response.body)
    }
  }
}

export { EmailChannel as EmailImapChannel }
export { BunSmtpTransport } from './smtpTransport.js'

/** Decode a RFC 2047 encoded-word header where the common UTF-8 B/Q forms are used. */
export function decodeMimeHeader(value: string): string {
  return value.replace(MIME_WORD, (encodedWord, encoding: string, data: string) => {
    try {
      if (encoding.toUpperCase() === 'B') {
        return Buffer.from(data, 'base64').toString('utf8')
      }
      return decodeQuotedPrintable(data.replaceAll('_', ' '))
    } catch {
      return encodedWord
    }
  })
}

/** Normalize text/plain content and decode common transfer encodings supplied by an IMAP bridge. */
export function normalizeEmailText(value: string, contentTransferEncoding = ''): string {
  const decoded = decodeContent(value, contentTransferEncoding)
  return decoded
    .replace(/\r\n?/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

/** Conservative text extraction for HTML-only mail; this is not a complete HTML or MIME renderer. */
export function htmlToEmailText(value: string): string {
  const withoutHiddenContent = value
    .replace(/<(script|style)\b[^>]*>[\s\S]*?<\/\1>/gi, '')
    .replace(/<\s*br\s*\/?\s*>/gi, '\n')
    .replace(/<\s*\/\s*(p|div|li|tr|h[1-6])\s*>/gi, '\n')
    .replace(/<[^>]*>/g, '')
  return normalizeEmailText(decodeHtmlEntities(withoutHiddenContent))
}

function inboundChannelMessage(channel: string, payload: Readonly<Record<string, unknown>>): ChannelMessage {
  const fromHeader = decodeMimeHeader(firstString(payload, 'from', 'sender'))
  const toHeader = decodeMimeHeader(firstAddress(payload.to) || firstAddress(payload.recipient))
  const subject = decodeMimeHeader(firstString(payload, 'subject'))
  const plainText = normalizeEmailText(
    firstString(payload, 'text', 'text_plain'),
    firstString(payload, 'textEncoding', 'text_encoding', 'contentTransferEncoding'),
  )
  const text = plainText || htmlToEmailText(decodeContent(
    firstString(payload, 'html', 'text_html'),
    firstString(payload, 'htmlEncoding', 'html_encoding'),
  ))
  const platformMessageId = firstString(payload, 'messageId', 'message_id', 'id')
  const replyTo = firstString(payload, 'inReplyTo', 'in_reply_to')
  const timestamp = timestampValue(payload.date ?? payload.timestamp)
  const sender = mailboxAddress(fromHeader)
  const recipient = mailboxAddress(toHeader)
  return createChannelMessage({
    channel,
    direction: MessageDirection.INBOUND,
    text,
    attachments: attachmentsValue(payload.attachments),
    ...(sender ? { channelUserId: sender } : {}),
    ...(recipient ? { roomId: recipient } : {}),
    ...(platformMessageId ? { platformMessageId } : {}),
    ...(replyTo ? { replyTo } : {}),
    ...(timestamp ? { timestamp } : {}),
    metadata: {
      subject,
      ...(fromHeader ? { from: fromHeader } : {}),
      ...(toHeader ? { to: toHeader } : {}),
    },
  })
}

function configuredHost(value: string): string {
  const host = value.trim()
  if (!host || HEADER_BREAK.test(host)) {
    throw new EmailChannelConfigurationError('smtpHost must be a non-empty host name')
  }
  return host
}

function configuredPort(value: number): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > 65_535) {
    throw new EmailChannelConfigurationError('smtpPort must be an integer between 1 and 65535')
  }
  return value
}

function configuredOptionalHeader(value: string, name: string): string {
  if (HEADER_BREAK.test(value)) {
    throw new EmailChannelConfigurationError(`${name} must not contain line breaks`)
  }
  return value.trim()
}

function configuredAddress(value: string, name: string): string {
  const address = configuredOptionalHeader(value, name)
  if (!address) {
    throw new EmailChannelConfigurationError(`${name} must not be empty`)
  }
  return address
}

function outboundAddress(value: string | undefined, name: string): string {
  if (value === undefined) {
    throw new EmailOutboundMessageError(`${name} requires roomId or channelUserId`)
  }
  return outboundHeader(value, name)
}

function outboundSubject(value: unknown): string {
  const subject = typeof value === 'string' && value.trim() ? value : DEFAULT_SUBJECT
  return outboundHeader(subject, 'subject')
}

function outboundHeader(value: string, name: string): string {
  const normalized = value.trim()
  if (!normalized || HEADER_BREAK.test(normalized)) {
    throw new EmailOutboundMessageError(`${name} must be a non-empty single-line value`)
  }
  return normalized
}

function normalizeOutboundText(value: string): string {
  return value.replace(/\r\n?/g, '\n')
}

function buildPlainTextMime(
  from: string,
  to: string,
  subject: string,
  text: string,
  replyTo: string | undefined,
): string {
  const headers = [
    `From: ${encodeMimeHeader(from)}`,
    `To: ${encodeMimeHeader(to)}`,
    `Subject: ${encodeMimeHeader(subject)}`,
    ...(replyTo === undefined ? [] : [`In-Reply-To: ${replyTo}`]),
    'MIME-Version: 1.0',
    'Content-Type: text/plain; charset=utf-8',
    'Content-Transfer-Encoding: base64',
  ]
  return `${headers.join('\r\n')}\r\n\r\n${foldBase64(Buffer.from(text, 'utf8').toString('base64'))}\r\n`
}

function encodeMimeHeader(value: string): string {
  if (/^[\x20-\x7e]*$/.test(value)) {
    return value
  }
  return `=?UTF-8?B?${Buffer.from(value, 'utf8').toString('base64')}?=`
}

function foldBase64(value: string): string {
  const lines: string[] = []
  for (let index = 0; index < value.length; index += 76) {
    lines.push(value.slice(index, index + 76))
  }
  return lines.join('\r\n')
}

function decodeContent(value: string, encoding: string): string {
  const normalized = encoding.trim().toLowerCase()
  if (normalized === 'base64') {
    if (!/^[A-Za-z0-9+/\s]*={0,2}$/.test(value)) {
      return value
    }
    try {
      return Buffer.from(value, 'base64').toString('utf8')
    } catch {
      return value
    }
  }
  if (normalized === 'quoted-printable' || normalized === 'q') {
    return decodeQuotedPrintable(value)
  }
  return value
}

function decodeQuotedPrintable(value: string): string {
  const bytes = value
    .replace(/=\r?\n/g, '')
    .replace(/=([\da-f]{2})/gi, (_match, hex: string) => String.fromCharCode(Number.parseInt(hex, 16)))
  try {
    return Buffer.from(bytes, 'latin1').toString('utf8')
  } catch {
    return bytes
  }
}

function decodeHtmlEntities(value: string): string {
  const named: Readonly<Record<string, string>> = {
    amp: '&', apos: "'", gt: '>', lt: '<', nbsp: ' ', quot: '"',
  }
  return value.replace(/&(#x[\da-f]+|#\d+|[a-z]+);/gi, (entity, body: string) => {
    const lower = body.toLowerCase()
    if (lower.startsWith('#x')) {
      return codePointEntity(Number.parseInt(lower.slice(2), 16), entity)
    }
    if (lower.startsWith('#')) {
      return codePointEntity(Number.parseInt(lower.slice(1), 10), entity)
    }
    return named[lower] ?? entity
  })
}

function codePointEntity(value: number, fallback: string): string {
  if (!Number.isInteger(value) || value < 0 || value > 0x10ffff) {
    return fallback
  }
  return String.fromCodePoint(value)
}

function firstString(payload: Readonly<Record<string, unknown>>, ...names: readonly string[]): string {
  for (const name of names) {
    const value = payload[name]
    if (typeof value === 'string') {
      return value
    }
  }
  return ''
}

function firstAddress(value: unknown): string {
  if (typeof value === 'string') {
    return value
  }
  if (!Array.isArray(value)) {
    return ''
  }
  for (const item of value) {
    if (typeof item === 'string') {
      return item
    }
  }
  return ''
}

function mailboxAddress(value: string): string {
  const match = /<([^<>\r\n]+)>/.exec(value)
  return (match?.[1] ?? value).trim()
}

function attachmentsValue(value: unknown): readonly ChannelAttachment[] {
  if (!Array.isArray(value)) {
    return []
  }
  const attachments: ChannelAttachment[] = []
  for (const item of value) {
    if (isRecord(item)) {
      attachments.push({ ...item })
    }
  }
  return attachments
}

function timestampValue(value: unknown): Date | undefined {
  if (typeof value !== 'string' && !(value instanceof Date)) {
    return undefined
  }
  const date = new Date(value)
  return Number.isNaN(date.valueOf()) ? undefined : date
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
