// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { DaemonEvent, DaemonRuntime, DaemonSession } from '../daemon/runtime.js'
import { scanContextContent } from '../security/promptScanner.js'
import { ChannelManager } from './manager.js'
import {
  createSessionResetPolicy,
  shouldReset,
  type SessionResetPolicy,
  type SessionResetPolicyInput,
} from './sessionReset.js'
import { createChannelMessage, MessageDirection, type ChannelMessage } from './types.js'

const DEFAULT_TYPING_INTERVAL = 8_000
const DEFAULT_PREVIEW_INTERVAL = 1_000
const JOURNAL_RESPONSE_MAX_CHARS = 500
const MAX_PREVIEW_CHARS = 4_096
const NO_RESPONSE_TEXT = '(no response)'
const PREVIEW_PLACEHOLDER = '...'
const PATH_REDACTION = /(?:\/Users\/[^\s'"]+|\/home\/[^\s'"]+|\/private\/[^\s'"]+|\/var\/[^\s'"]+|\/tmp\/[^\s'"]+|~\/\.xerxes[^\s'"]*)/g
const TRACEBACK_REDACTION = /Traceback \(most recent call last\):.*?(?=\n\n|$)/gs

/** Persistent Markdown context used by channel-backed agent sessions. */
export interface ChannelWorkspace {
  appendDailyNote(text: string): Promise<string>
  loadContext(): Promise<Readonly<{ readonly prompt: string }>>
}

/** Enables previews globally or selectively for normalized inbound messages. */
export type ChannelPreviewPolicy = boolean | ((message: ChannelMessage) => boolean)

/** Sets preview edit cadence globally or per normalized inbound message, in milliseconds. */
export type ChannelPreviewInterval = number | ((message: ChannelMessage) => number)

export interface ChannelTurnRouterOptions {
  /** Agent selected for channel-originated conversations. */
  readonly agentId?: string
  /** Host-owned channels used both for typing indicators and outbound replies. */
  readonly channels: ChannelManager
  /** Workspace applied when opening channel-backed daemon sessions. */
  readonly cwd?: string
  /** Receives contained delivery/turn errors without exposing channel credentials. */
  readonly onError?: (error: unknown, message: ChannelMessage) => void
  /** Minimum interval between edits sent through adapters with editable text support, in milliseconds. */
  readonly previewInterval?: ChannelPreviewInterval
  /** Optional automatic reset policy for a channel conversation. */
  readonly sessionResetPolicy?: SessionResetPolicy | SessionResetPolicyInput
  /** Native daemon runtime used for session and turn lifecycle. */
  readonly runtime: DaemonRuntime
  /** Enable native streamed previews for adapters with sendText/editText support. */
  readonly streamPreviews?: ChannelPreviewPolicy
  /** Interval for adapters that provide a live typing indicator. */
  readonly typingInterval?: number
  /** Optional Markdown workspace journal and per-turn system-prompt context. */
  readonly workspace?: ChannelWorkspace
  /** Injectable clock for session inactivity policy and deterministic tests. */
  readonly clock?: () => Date
}

/**
 * Routes one normalized platform message into a serialized native agent turn.
 *
 * Each channel conversation receives a durable daemon session. The router
 * deliberately keeps platform metadata separate from the prompt body while
 * preserving it on outbound replies for adapters such as Slack that need a
 * verified installation identifier.
 */
export class ChannelTurnRouter {
  private readonly agentId: string
  private readonly channels: ChannelManager
  private readonly clock: () => Date
  private readonly cwd: string | undefined
  private readonly onError: ((error: unknown, message: ChannelMessage) => void) | undefined
  private readonly pendingBySession = new Map<string, Promise<void>>()
  private readonly previewInterval: ChannelPreviewInterval
  private readonly resetPolicy: SessionResetPolicy
  private readonly resetState = new Map<string, ChannelSessionActivity>()
  private readonly runtime: DaemonRuntime
  private readonly streamPreviews: ChannelPreviewPolicy
  private readonly typingInterval: number
  private readonly workspace: ChannelWorkspace | undefined

  constructor(options: ChannelTurnRouterOptions) {
    this.agentId = nonBlank(options.agentId) ?? 'default'
    this.channels = options.channels
    this.clock = options.clock ?? (() => new Date())
    this.cwd = nonBlank(options.cwd)
    this.onError = options.onError
    this.previewInterval = options.previewInterval ?? DEFAULT_PREVIEW_INTERVAL
    if (typeof this.previewInterval === 'number') {
      positiveInteger(this.previewInterval, 'previewInterval')
    }
    this.resetPolicy = createSessionResetPolicy(options.sessionResetPolicy)
    this.runtime = options.runtime
    this.streamPreviews = options.streamPreviews ?? true
    this.typingInterval = positiveInteger(options.typingInterval ?? DEFAULT_TYPING_INTERVAL, 'typingInterval')
    this.workspace = options.workspace
  }

  /** Accept one inbound message, serializing concurrent deliveries for its conversation. */
  async handle(message: ChannelMessage): Promise<void> {
    if (!message.text.trim()) return
    const key = channelSessionKey(message)
    const previous = this.pendingBySession.get(key) ?? Promise.resolve()
    const current = previous.catch(() => undefined).then(() => this.run(message, key))
    this.pendingBySession.set(key, current)
    try {
      await current
    } catch (error) {
      this.report(error, message)
      throw error
    } finally {
      if (this.pendingBySession.get(key) === current) {
        this.pendingBySession.delete(key)
      }
    }
  }

  private async run(message: ChannelMessage, sessionKey: string): Promise<void> {
    await this.journalInbound(message)
    const slash = parseChannelCommand(message.text)
    if (slash) {
      await this.handleCommand(message, sessionKey, slash)
      return
    }
    await this.runTurn(message, sessionKey, formatChannelPrompt(message, message.text))
  }

  private async handleCommand(
    message: ChannelMessage,
    sessionKey: string,
    command: ChannelCommand,
  ): Promise<void> {
    if (command.name === 'ask') {
      if (!command.arguments) {
        await this.reply(message, 'Usage: /ask <prompt>')
        return
      }
      await this.runTurn(message, sessionKey, formatChannelPrompt(message, command.arguments))
      return
    }
    if (command.name === 'help' || command.name === 'commands') {
      await this.reply(message, [
        'Channel commands:',
        '/ask <prompt> — run an agent turn',
        '/status — show channel session status',
        '/context — show channel session token usage',
        '/new — start a fresh channel session',
        '/stop — cancel the active channel turn',
      ].join('\n'))
      return
    }
    if (command.name === 'new' || command.name === 'reset') {
      await this.resetSession(sessionKey)
      await this.reply(message, 'Started a new channel session.')
      return
    }
    if (command.name === 'stop' || command.name === 'cancel') {
      await this.reply(message, this.runtime.cancelTurn(sessionKey)
        ? 'Cancellation requested.'
        : 'No active channel turn to cancel.')
      return
    }
    if (command.name === 'status') {
      await this.reply(message, channelStatus(this.runtime.status()))
      return
    }
    if (command.name === 'context' || command.name === 'usage' || command.name === 'history') {
      const session = this.runtime.sessionStatus(sessionKey)
      await this.reply(message, session ? sessionUsage(session) : 'No channel session is active.')
      return
    }
    await this.reply(message, 'Unsupported channel command: /' + command.name)
  }

  private async runTurn(message: ChannelMessage, sessionKey: string, prompt: string): Promise<void> {
    await this.openSessionForTurn(sessionKey, message)
    const output: string[] = []
    const preview = this.startPreview(message)
    const typing = this.startTyping(message)
    try {
      await this.runtime.submitTurn(sessionKey, prompt, event => {
        const chunk = streamedText(event)
        if (chunk) preview?.push(chunk)
        collectOutput(output, event)
      })
    } finally {
      await typing.stop()
    }
    const response = output.join('').trim() || NO_RESPONSE_TEXT
    const previewDelivered = await preview?.finish(response) ?? false
    if (!previewDelivered) await this.reply(message, response)
    await this.journalAssistant(message, response)
  }

  private async reply(message: ChannelMessage, text: string): Promise<void> {
    await this.channels.registry.send(createChannelMessage({
      channel: message.channel,
      direction: MessageDirection.OUTBOUND,
      metadata: message.metadata,
      text,
      ...(message.channelUserId === undefined ? {} : { channelUserId: message.channelUserId }),
      ...(message.platformMessageId === undefined ? {} : { replyTo: message.platformMessageId }),
      ...(message.roomId === undefined ? {} : { roomId: message.roomId }),
    }))
  }

  private sessionOptions(): { readonly cwd?: string } {
    return this.cwd === undefined ? {} : { cwd: this.cwd }
  }

  private async openSessionForTurn(sessionKey: string, message: ChannelMessage): Promise<void> {
    const now = validDate(this.clock())
    const prior = this.resetState.get(sessionKey)
    if (shouldReset(this.resetPolicy, {
      messageCount: (prior?.messageCount ?? 0) + 1,
      ...(prior === undefined ? {} : { lastMessageAt: prior.lastMessageAt }),
      now,
    })) {
      this.runtime.evictSession(sessionKey)
      this.resetState.delete(sessionKey)
    }
    const workspacePrompt = await this.workspacePrompt(message)
    await this.runtime.openSession(sessionKey, this.agentId, {
      ...this.sessionOptions(),
      ...(workspacePrompt ? { systemPromptAddendum: workspacePrompt } : {}),
    })
    const active = this.resetState.get(sessionKey)
    this.resetState.set(sessionKey, {
      lastMessageAt: now,
      messageCount: (active?.messageCount ?? 0) + 1,
    })
  }

  private async resetSession(sessionKey: string): Promise<void> {
    this.runtime.evictSession(sessionKey)
    this.resetState.delete(sessionKey)
    await this.runtime.openSession(sessionKey, this.agentId, this.sessionOptions())
  }

  private startTyping(message: ChannelMessage): Stoppable {
    const channel = this.channels.registry.get(message.channel)
    if (!hasTypingIndicator(channel)) return NO_TYPING_LOOP
    return new TypingLoop(channel, message.roomId, this.typingInterval, error => this.report(error, message))
  }

  private startPreview(message: ChannelMessage): ChannelPreview | undefined {
    if (!this.previewsEnabled(message)) return undefined
    const channel = this.channels.registry.get(message.channel)
    if (!hasEditableText(channel)) return undefined
    const chatId = message.roomId ?? message.channelUserId
    if (!chatId) return undefined
    return new ChannelPreview(channel, chatId, message.replyTo ?? message.platformMessageId, this.previewIntervalFor(message), error => {
      this.report(error, message)
    })
  }

  private previewsEnabled(message: ChannelMessage): boolean {
    try {
      return typeof this.streamPreviews === 'function'
        ? this.streamPreviews(message)
        : this.streamPreviews
    } catch (error) {
      this.report(error, message)
      return false
    }
  }

  private previewIntervalFor(message: ChannelMessage): number {
    try {
      const interval = typeof this.previewInterval === 'function'
        ? this.previewInterval(message)
        : this.previewInterval
      return positiveInteger(interval, 'previewInterval')
    } catch (error) {
      this.report(error, message)
      return DEFAULT_PREVIEW_INTERVAL
    }
  }

  private async journalInbound(message: ChannelMessage): Promise<void> {
    const safeText = scanContextContent(message.text, message.channel + ':inbound')
    await this.appendJournal(message, [
      '[' + message.channel + ':' + channelJournalTarget(message) + '] user ' + (message.channelUserId ?? ''),
      quoteUserBlock(safeText),
    ].join(':\n'))
  }

  private async journalAssistant(message: ChannelMessage, response: string): Promise<void> {
    await this.appendJournal(message, [
      '[' + message.channel + ':' + channelJournalTarget(message) + '] xerxes:',
      sanitizeJournalOutput(response).slice(0, JOURNAL_RESPONSE_MAX_CHARS),
    ].join(' '))
  }

  private async appendJournal(message: ChannelMessage, entry: string): Promise<void> {
    if (!this.workspace) return
    try {
      await this.workspace.appendDailyNote(entry)
    } catch (error) {
      this.report(error, message)
    }
  }

  private async workspacePrompt(message: ChannelMessage): Promise<string | undefined> {
    if (!this.workspace) return undefined
    try {
      return nonBlank((await this.workspace.loadContext()).prompt)
    } catch (error) {
      this.report(error, message)
      return undefined
    }
  }

  private report(error: unknown, message: ChannelMessage): void {
    if (!this.onError) return
    try {
      this.onError(error, message)
    } catch {
      // Diagnostic callbacks must not alter platform delivery semantics.
    }
  }
}

/** Derive a durable private-or-group conversation key from trusted adapter metadata. */
export function channelSessionKey(message: ChannelMessage): string {
  const chatType = stringMetadata(message, 'chat_type').toLowerCase()
  const threadId = stringMetadata(message, 'thread_id') || 'main'
  if (chatType === 'group' || chatType === 'supergroup' || chatType === 'channel') {
    return message.channel + ':chat:' + (message.roomId ?? '') + ':thread:' + threadId
  }
  return message.channel + ':private:' + (message.channelUserId ?? message.roomId ?? '')
}

/** Make the inbound origin explicit to the model without merging platform metadata into user text. */
export function formatChannelPrompt(message: ChannelMessage, text = message.text): string {
  return [
    '[' + message.channel + ' message]',
    'room_id: ' + (message.roomId ?? ''),
    'from_user_id: ' + (message.channelUserId ?? ''),
    'thread_id: ' + stringMetadata(message, 'thread_id'),
    '',
    text,
  ].join('\n')
}

interface ChannelCommand {
  readonly arguments: string
  readonly name: string
}

interface ChannelSessionActivity {
  readonly lastMessageAt: Date
  readonly messageCount: number
}

interface TypingCapableChannel {
  sendTyping(roomId: string | undefined): Promise<void>
}

interface EditableTextChannel {
  editText(chatId: string, messageId: string, text: string): Promise<unknown>
  sendText(chatId: string, text: string, replyTo?: string): Promise<Readonly<Record<string, unknown>>>
}

interface Stoppable {
  stop(): Promise<void>
}

const NO_TYPING_LOOP: Stoppable = {
  async stop(): Promise<void> {},
}

class TypingLoop implements Stoppable {
  private readonly abort = new AbortController()
  private readonly done: Promise<void>

  constructor(
    private readonly channel: TypingCapableChannel,
    private readonly roomId: string | undefined,
    private readonly interval: number,
    private readonly onError: (error: unknown) => void,
  ) {
    this.done = this.run()
  }

  async stop(): Promise<void> {
    this.abort.abort()
    await this.done
  }

  private async run(): Promise<void> {
    while (!this.abort.signal.aborted) {
      try {
        await this.channel.sendTyping(this.roomId)
      } catch (error) {
        this.onError(error)
        return
      }
      await sleepUntilAbort(this.interval, this.abort.signal)
    }
  }
}

/** Posts an initial placeholder, then rate-limits editable channel previews. */
class ChannelPreview {
  private readonly ready: Promise<void>
  private editQueue: Promise<void>
  private failed = false
  private lastEditedAt = 0
  private lastText = ''
  private messageId = ''
  private pendingText = ''
  private scheduled: ReturnType<typeof setTimeout> | undefined

  constructor(
    private readonly channel: EditableTextChannel,
    private readonly chatId: string,
    private readonly replyTo: string | undefined,
    private readonly interval: number,
    private readonly report: (error: unknown) => void,
  ) {
    this.ready = this.channel.sendText(chatId, PREVIEW_PLACEHOLDER, replyTo)
      .then(response => {
        this.messageId = telegramMessageId(response)
      })
      .catch(error => {
        this.failed = true
        this.report(error)
      })
    this.editQueue = this.ready
  }

  push(text: string): void {
    if (!text || this.failed) return
    this.pendingText += text
    this.scheduleEdit()
  }

  async finish(text: string): Promise<boolean> {
    if (this.scheduled !== undefined) {
      clearTimeout(this.scheduled)
      this.scheduled = undefined
    }
    this.pendingText = text
    this.enqueueEdit()
    await this.editQueue
    return !this.failed && Boolean(this.messageId) && this.lastText === previewText(text)
  }

  private scheduleEdit(): void {
    if (this.scheduled !== undefined) return
    const elapsed = this.lastEditedAt ? Date.now() - this.lastEditedAt : 0
    const delay = Math.max(0, this.interval - elapsed)
    this.scheduled = setTimeout(() => {
      this.scheduled = undefined
      this.enqueueEdit()
    }, delay)
  }

  private enqueueEdit(): void {
    const proposed = previewText(this.pendingText)
    if (!proposed || proposed === this.lastText) return
    this.editQueue = this.editQueue.then(async () => {
      if (this.failed || !this.messageId) return
      const current = previewText(this.pendingText)
      if (!current || current === this.lastText) return
      try {
        await this.channel.editText(this.chatId, this.messageId, current)
        this.lastText = current
        this.lastEditedAt = Date.now()
      } catch (error) {
        this.failed = true
        this.report(error)
      }
    })
  }
}

function collectOutput(output: string[], event: DaemonEvent): void {
  if (event.type === 'text_part') {
    const text = rawStringPayload(event.payload, 'text')
    if (text) output.push(text)
    return
  }
  if (event.type !== 'notification') return
  const level = stringPayload(event.payload, 'level') || stringPayload(event.payload, 'severity')
  if (level !== 'error') return
  const message = stringPayload(event.payload, 'message')
    || stringPayload(event.payload, 'body')
    || stringPayload(event.payload, 'title')
  if (message) output.push(message)
}

function streamedText(event: DaemonEvent): string {
  return event.type === 'text_part' ? rawStringPayload(event.payload, 'text') : ''
}

function parseChannelCommand(text: string): ChannelCommand | undefined {
  const raw = text.trim()
  if (!raw.startsWith('/')) return undefined
  const [head, ...tail] = raw.slice(1).trim().split(/\s+/)
  const name = head?.toLowerCase()
  if (!name) return undefined
  return { name, arguments: tail.join(' ').trim() }
}

function channelStatus(status: Readonly<Record<string, unknown>>): string {
  const model = stringPayload(status, 'model') || '(not configured)'
  const runtime = stringPayload(status, 'runtime') || 'bun-typescript'
  return 'Xerxes status: runtime=' + runtime + ', model=' + model
}

function sessionUsage(session: DaemonSession): string {
  const total = session.totalInputTokens + session.totalOutputTokens
  return [
    'Session: ' + session.id,
    'Turns: ' + session.turnCount,
    'Input tokens: ' + session.totalInputTokens,
    'Output tokens: ' + session.totalOutputTokens,
    'Total tokens: ' + total,
  ].join('\n')
}

function hasTypingIndicator(value: unknown): value is TypingCapableChannel {
  return typeof value === 'object'
    && value !== null
    && 'sendTyping' in value
    && typeof value.sendTyping === 'function'
}

function hasEditableText(value: unknown): value is EditableTextChannel {
  return typeof value === 'object'
    && value !== null
    && 'sendText' in value
    && typeof value.sendText === 'function'
    && 'editText' in value
    && typeof value.editText === 'function'
}

function telegramMessageId(response: Readonly<Record<string, unknown>>): string {
  const result = recordPayload(response, 'result')
  return rawStringPayload(result, 'message_id') || rawStringPayload(response, 'message_id')
}

function previewText(text: string): string {
  return text.length <= MAX_PREVIEW_CHARS ? text : text.slice(-MAX_PREVIEW_CHARS)
}

function channelJournalTarget(message: ChannelMessage): string {
  return message.roomId ?? message.channelUserId ?? ''
}

function quoteUserBlock(text: string): string {
  return '~~~user\n' + text + '\n~~~'
}

function sanitizeJournalOutput(text: string): string {
  return text
    .replace(TRACEBACK_REDACTION, '[traceback redacted]')
    .replace(PATH_REDACTION, '[path redacted]')
    .trim()
}

function stringMetadata(message: ChannelMessage, key: string): string {
  return stringPayload(message.metadata, key)
}

function stringPayload(payload: Readonly<Record<string, unknown>>, key: string): string {
  return rawStringPayload(payload, key).trim()
}

function recordPayload(payload: Readonly<Record<string, unknown>>, key: string): Readonly<Record<string, unknown>> {
  const value = payload[key]
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Readonly<Record<string, unknown>>
    : {}
}

function rawStringPayload(payload: Readonly<Record<string, unknown>>, key: string): string {
  const value = payload[key]
  return typeof value === 'string' ? value : value === undefined || value === null ? '' : String(value)
}

function nonBlank(value: string | undefined): string | undefined {
  const normalized = value?.trim()
  return normalized || undefined
}

function positiveInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(name + ' must be a positive safe integer')
  }
  return value
}

function validDate(value: Date): Date {
  if (!Number.isFinite(value.getTime())) throw new TypeError('clock must return a valid Date')
  return value
}

async function sleepUntilAbort(milliseconds: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return
  await Promise.race([
    Bun.sleep(milliseconds),
    new Promise<void>(resolve => signal.addEventListener('abort', () => resolve(), { once: true })),
  ])
}
