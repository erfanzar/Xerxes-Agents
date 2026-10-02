// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import type { DaemonEvent, DaemonRuntime, DaemonSession, OpenSessionOptions } from '../daemon/runtime.js'
import { scanContextContent } from '../security/promptScanner.js'
import { ChannelManager } from './manager.js'
import type { ChannelSessionIndex } from './sessionIndex.js'
import {
  createSessionResetPolicy,
  shouldReset,
  type SessionResetPolicy,
  type SessionResetPolicyInput,
} from './sessionReset.js'
import { chunkText } from './textChunking.js'
import { createChannelMessage, MessageDirection, type ChannelMessage } from './types.js'

const DEFAULT_TYPING_INTERVAL = 8_000
const DEFAULT_PREVIEW_INTERVAL = 1_000
const DEFAULT_MAX_PENDING_PER_SESSION = 32
const DEFAULT_IDLE_SESSION_TTL_MS = 24 * 60 * 60 * 1_000
const JOURNAL_RESPONSE_MAX_CHARS = 500
const MAX_PREVIEW_CHARS = 4_096
const NO_RESPONSE_TEXT = '(no response)'
const PREVIEW_PLACEHOLDER = '...'
const PREVIEW_TRUNCATION_MARKER = '…[truncated]'
const TURN_FAILED_TEXT = '(turn failed)'
const PATH_REDACTION = /(?:\/Users\/[^\s'"]+|\/home\/[^\s'"]+|\/private\/[^\s'"]+|\/var\/[^\s'"]+|\/tmp\/[^\s'"]+|~[\/\\]\.xerxes[^\s'"]*|[A-Za-z]:\\Users\\[^\s'"]+|[A-Za-z]:\\[^\s'"]*\\\.xerxes[^\s'"]*)/g
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

/** Answers the approvals and questions a channel turn waits on (the daemon interaction board). */
export interface ChannelInteractionPort {
  respondPermission(requestId: string, response: string): boolean
  respondQuestion(requestId: string, answers: Readonly<Record<string, string>>): boolean
}

/** An approval or question a channel turn is parked on, and who may answer it. */
type ChannelWait =
  | { readonly kind: 'approval'; readonly requestId: string; readonly requester: string | undefined }
  | {
    readonly kind: 'question'
    readonly requestId: string
    readonly questionId: string
    readonly options: readonly string[]
    readonly requester: string | undefined
  }

export interface ChannelTurnRouterOptions {
  /** Agent selected for channel-originated conversations. */
  readonly agentId?: string
  /** Host-owned channels used both for typing indicators and outbound replies. */
  readonly channels: ChannelManager
  /** Workspace applied when opening channel-backed daemon sessions. */
  readonly cwd?: string
  /** Evict inactive router/session bookkeeping after this many milliseconds. */
  readonly idleSessionTtlMs?: number
  /**
   * Answers approvals and questions from the chat. Without it a turn that asks
   * is stopped with a reply saying why, rather than waiting forever.
   */
  readonly interactions?: ChannelInteractionPort
  /** Maximum active plus queued messages retained for one conversation. */
  readonly maxPendingPerSession?: number
  /** Receives contained delivery/turn errors without exposing channel credentials. */
  readonly onError?: (error: unknown, message: ChannelMessage) => void
  /** Minimum interval between edits sent through adapters with editable text support, in milliseconds. */
  readonly previewInterval?: ChannelPreviewInterval
  /** Optional automatic reset policy for a channel conversation. */
  readonly sessionResetPolicy?: SessionResetPolicy | SessionResetPolicyInput
  /** Remembers which saved session each conversation continues across restarts and eviction. */
  readonly sessionIndex?: ChannelSessionIndex
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
  private readonly idleSessionTtlMs: number
  private readonly interactions: ChannelInteractionPort | undefined
  private readonly maxPendingPerSession: number
  private readonly onError: ((error: unknown, message: ChannelMessage) => void) | undefined
  private readonly pendingBySession = new Map<string, Promise<void>>()
  private readonly pendingCounts = new Map<string, number>()
  private readonly previewInterval: ChannelPreviewInterval
  private readonly resetPolicy: SessionResetPolicy
  private readonly resetState = new Map<string, ChannelSessionActivity>()
  private readonly runtime: DaemonRuntime
  private readonly sessionIndex: ChannelSessionIndex | undefined
  private readonly streamPreviews: ChannelPreviewPolicy
  private readonly typingInterval: number
  private readonly waits = new Map<string, ChannelWait>()
  private readonly workspace: ChannelWorkspace | undefined

  constructor(options: ChannelTurnRouterOptions) {
    this.agentId = nonBlank(options.agentId) ?? 'default'
    this.channels = options.channels
    this.clock = options.clock ?? (() => new Date())
    this.cwd = nonBlank(options.cwd)
    this.idleSessionTtlMs = nonNegativeFinite(options.idleSessionTtlMs ?? DEFAULT_IDLE_SESSION_TTL_MS, 'idleSessionTtlMs')
    this.interactions = options.interactions
    this.maxPendingPerSession = positiveInteger(
      options.maxPendingPerSession ?? DEFAULT_MAX_PENDING_PER_SESSION,
      'maxPendingPerSession',
    )
    this.onError = options.onError
    this.previewInterval = options.previewInterval ?? DEFAULT_PREVIEW_INTERVAL
    if (typeof this.previewInterval === 'number') {
      positiveInteger(this.previewInterval, 'previewInterval')
    }
    this.resetPolicy = createSessionResetPolicy(options.sessionResetPolicy)
    this.runtime = options.runtime
    this.sessionIndex = options.sessionIndex
    this.streamPreviews = options.streamPreviews ?? true
    this.typingInterval = positiveInteger(options.typingInterval ?? DEFAULT_TYPING_INTERVAL, 'typingInterval')
    this.workspace = options.workspace
  }

  /** Accept one inbound message, serializing concurrent deliveries for its conversation. */
  async handle(message: ChannelMessage): Promise<void> {
    if (!message.text.trim()) return
    if (message.channelUserId === undefined && message.roomId === undefined) {
      // Without a user or room identity every anonymous sender would collapse
      // into one shared daemon session; reject the message instead of pooling.
      this.report(new ChannelRoutingError(message.channel), message)
      return
    }
    const key = channelSessionKey(message)
    const slash = parseChannelCommand(message.text)
    if (slash?.name === 'stop' || slash?.name === 'cancel') {
      await this.journalInbound(message)
      await this.handleCommand(message, key, slash)
      return
    }
    // The waiting turn holds this conversation's queue, so its answer must not
    // queue behind it.
    const wait = this.waits.get(key)
    if (wait && (!slash || slash.name === 'approve' || slash.name === 'deny' || slash.name === 'answer')) {
      await this.journalInbound(message)
      await this.answerWait(message, key, wait, slash)
      return
    }
    this.evictIdleBookkeeping(validDate(this.clock()))
    const pendingCount = this.pendingCounts.get(key) ?? 0
    if (pendingCount >= this.maxPendingPerSession) {
      const error = new ChannelQueueFullError(message.channel, key, this.maxPendingPerSession)
      this.report(error, message)
      throw error
    }
    this.pendingCounts.set(key, pendingCount + 1)
    const previous = this.pendingBySession.get(key) ?? Promise.resolve()
    const current = previous.catch(() => undefined).then(() => this.run(message, key))
    this.pendingBySession.set(key, current)
    try {
      await current
    } catch (error) {
      this.report(error, message)
      throw error
    } finally {
      const remaining = (this.pendingCounts.get(key) ?? 1) - 1
      if (remaining > 0) this.pendingCounts.set(key, remaining)
      else this.pendingCounts.delete(key)
      if (this.pendingBySession.get(key) === current) {
        this.pendingBySession.delete(key)
      }
    }
  }

  private evictIdleBookkeeping(now: Date): void {
    const cutoff = now.getTime() - this.idleSessionTtlMs
    for (const [sessionKey, activity] of this.resetState) {
      if (activity.lastMessageAt.getTime() >= cutoff || this.pendingCounts.has(sessionKey)) continue
      this.resetState.delete(sessionKey)
      this.runtime.evictSession(sessionKey)
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
        '/approve, /deny — answer a pending tool approval',
        '/answer <text> — answer a pending question',
      ].join('\n'))
      return
    }
    if (command.name === 'approve' || command.name === 'deny' || command.name === 'answer') {
      await this.reply(message, 'Nothing is waiting for an answer.')
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
        const wait = channelWait(event, message.channelUserId)
        if (wait) this.holdForAnswer(message, sessionKey, wait, event.payload)
      })
    } catch (error) {
      // Finish the placeholder (which also cancels its pending edit timer) so a
      // failed turn never leaves a live '...' message or a stray timer behind.
      await preview?.finish(TURN_FAILED_TEXT)
      throw error
    } finally {
      this.waits.delete(sessionKey)
      await typing.stop()
    }
    const response = output.join('').trim() || NO_RESPONSE_TEXT
    try {
      const previewDelivered = await preview?.finish(response) ?? false
      if (!previewDelivered) await this.reply(message, response)
    } catch (error) {
      // The agent turn is already durably complete. Expose the delivery failure
      // without asking webhook transports to retry the inbound message and run
      // its side effects again.
      throw new ChannelTurnDeliveryError(message.channel, error)
    }
    await this.journalAssistant(message, response)
  }

  /**
   * Put the turn's approval or question in front of the person in the chat.
   * The turn is parked until it is answered, so a prompt nobody can see or
   * answer stops the turn instead of hanging the conversation.
   */
  private holdForAnswer(message: ChannelMessage, sessionKey: string, wait: ChannelWait, payload: Readonly<Record<string, unknown>>): void {
    const stop = (text: string): void => {
      this.runtime.cancelTurn(sessionKey)
      void this.reply(message, text).catch(error => this.report(error, message))
    }
    const needs = wait.kind === 'approval' ? 'an approval' : 'an answer'
    if (!this.interactions) {
      stop('This turn needs ' + needs + ' that this channel cannot give, so it was stopped.')
      return
    }
    this.waits.set(sessionKey, wait)
    void this.reply(message, waitPrompt(wait, payload)).catch(error => {
      this.report(error, message)
      if (this.waits.get(sessionKey) !== wait) return
      this.waits.delete(sessionKey)
      stop('This turn needs ' + needs + ' but the request could not be delivered, so it was stopped.')
    })
  }

  private async answerWait(
    message: ChannelMessage,
    sessionKey: string,
    wait: ChannelWait,
    command: ChannelCommand | undefined,
  ): Promise<void> {
    // In a group chat only the person whose message started the turn decides.
    if (wait.requester !== undefined && message.channelUserId !== wait.requester) {
      await this.reply(message, 'Only the person who started this turn can answer it.')
      return
    }
    const interactions = this.interactions
    if (!interactions) return
    if (wait.kind === 'approval') {
      if (command?.name !== 'approve' && command?.name !== 'deny') {
        await this.reply(message, 'A tool is waiting for approval. Reply /approve or /deny; /stop cancels the turn.')
        return
      }
      if (!interactions.respondPermission(wait.requestId, command.name === 'approve' ? 'approve' : 'reject')) {
        if (this.waits.get(sessionKey) === wait) this.waits.delete(sessionKey)
        await this.reply(message, 'That approval is no longer waiting.')
        return
      }
      if (this.waits.get(sessionKey) === wait) this.waits.delete(sessionKey)
      await this.reply(message, command.name === 'approve' ? 'Approved.' : 'Denied.')
      return
    }
    if (command && command.name !== 'answer') {
      await this.reply(message, 'A question is waiting. Reply with your answer; /stop cancels the turn.')
      return
    }
    const text = (command ? command.arguments : message.text).trim()
    const index = /^\d+$/.test(text) ? Number.parseInt(text, 10) - 1 : -1
    const answer = wait.options[index] ?? text
    if (!answer || !interactions.respondQuestion(wait.requestId, { [wait.questionId]: answer })) {
      await this.reply(message, wait.options.length
        ? 'That answer was not accepted. Choose one of:\n' + numberedOptions(wait.options)
        : 'That answer was not accepted.')
      return
    }
    if (this.waits.get(sessionKey) === wait) this.waits.delete(sessionKey)
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
    const workspacePrompt = await this.workspacePrompt(message)
    const options: OpenSessionOptions = {
      ...this.sessionOptions(),
      ...(workspacePrompt ? { systemPromptAddendum: workspacePrompt } : {}),
    }
    let prior = this.resetState.get(sessionKey)
      ?? await this.resumeSavedSession(sessionKey, options, message)
    if (shouldReset(this.resetPolicy, {
      messageCount: (prior?.messageCount ?? 0) + 1,
      ...(prior === undefined ? {} : { lastMessageAt: prior.lastMessageAt }),
      now,
    })) {
      this.runtime.evictSession(sessionKey)
      this.resetState.delete(sessionKey)
      prior = undefined
    }
    const session = await this.runtime.openSession(sessionKey, this.agentId, options)
    await this.sessionIndex?.set(sessionKey, session.id)
    this.resetState.set(sessionKey, {
      lastMessageAt: now,
      messageCount: (prior?.messageCount ?? 0) + 1,
    })
  }

  /**
   * Reopen the saved session this conversation last used when the runtime no
   * longer holds it (daemon restart, runtime update, idle eviction), and
   * return its activity so the reset policy still applies to it.
   */
  private async resumeSavedSession(
    sessionKey: string,
    options: OpenSessionOptions,
    message: ChannelMessage,
  ): Promise<ChannelSessionActivity | undefined> {
    if (!this.sessionIndex || this.runtime.sessionStatus(sessionKey) !== undefined) return undefined
    const savedId = await this.sessionIndex.get(sessionKey)
    if (!savedId) return undefined
    let session: DaemonSession
    try {
      session = await this.runtime.openSession(sessionKey, this.agentId, { ...options, resumeSessionId: savedId })
    } catch (error) {
      // The saved conversation belongs to another project or is busy
      // elsewhere. Report it and start fresh rather than wedge the chat.
      if (!(error instanceof ValidationError)) throw error
      this.report(error, message)
      return undefined
    }
    return session.turnCount > 0
      ? { lastMessageAt: new Date(session.lastActive), messageCount: session.turnCount }
      : undefined
  }

  private async resetSession(sessionKey: string): Promise<void> {
    this.runtime.evictSession(sessionKey)
    this.resetState.delete(sessionKey)
    const session = await this.runtime.openSession(sessionKey, this.agentId, this.sessionOptions())
    await this.sessionIndex?.set(sessionKey, session.id)
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

/** Raised when a completed agent turn could not be delivered to the channel. */
export class ChannelTurnDeliveryError extends Error {
  readonly cause: unknown
  readonly retryInbound = false
  readonly turnCompleted = true

  constructor(channel: string, cause: unknown) {
    super(`completed channel '${channel}' turn could not be delivered`, { cause })
    this.name = new.target.name
    this.cause = cause
  }
}

/** Raised when a conversation exceeds its bounded active-plus-queued delivery capacity. */
export class ChannelQueueFullError extends Error {
  constructor(channel: string, sessionKey: string, limit: number) {
    super(`channel '${channel}' session '${sessionKey}' has reached its pending message limit (${limit})`)
    this.name = new.target.name
  }
}

/** Raised when an inbound message carries no user or room identity to route by. */
export class ChannelRoutingError extends Error {
  constructor(channel: string) {
    super(`channel '${channel}' message has no channelUserId or roomId; refusing to pool it into a shared session`)
    this.name = new.target.name
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

  /**
   * Put the final answer in the placeholder and report whether all of it
   * reached the chat; false asks the caller to send the answer itself.
   *
   * An answer longer than one message used to be edited in truncated and
   * still reported as delivered, so its tail (often the conclusion) never
   * arrived. The placeholder now takes the first chunk and the rest follows
   * as new messages.
   */
  async finish(text: string): Promise<boolean> {
    if (this.scheduled !== undefined) {
      clearTimeout(this.scheduled)
      this.scheduled = undefined
    }
    const [head = '', ...rest] = chunkText(text, MAX_PREVIEW_CHARS)
    this.pendingText = head
    this.enqueueEdit()
    await this.editQueue
    if (this.failed || !this.messageId || this.lastText !== head) return false
    for (const chunk of rest) {
      await this.channel.sendText(this.chatId, chunk)
    }
    return true
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

function channelWait(event: DaemonEvent, requester: string | undefined): ChannelWait | undefined {
  if (event.type === 'approval_request') {
    const requestId = stringPayload(event.payload, 'id') || stringPayload(event.payload, 'request_id')
    return requestId ? { kind: 'approval', requestId, requester } : undefined
  }
  if (event.type !== 'question_request') return undefined
  const requestId = stringPayload(event.payload, 'id')
  const item = firstQuestion(event.payload)
  if (!requestId || !item) return undefined
  return {
    kind: 'question',
    requestId,
    questionId: stringPayload(item, 'id') || 'answer',
    options: Array.isArray(item.options) ? item.options.filter((option): option is string => typeof option === 'string') : [],
    requester,
  }
}

function firstQuestion(payload: Readonly<Record<string, unknown>>): Readonly<Record<string, unknown>> | undefined {
  const questions = payload.questions
  const first: unknown = Array.isArray(questions) ? questions[0] : undefined
  return typeof first === 'object' && first !== null && !Array.isArray(first) ? first as Record<string, unknown> : undefined
}

function waitPrompt(wait: ChannelWait, payload: Readonly<Record<string, unknown>>): string {
  if (wait.kind === 'approval') {
    const action = stringPayload(payload, 'description') || stringPayload(payload, 'tool_name') || stringPayload(payload, 'name') || 'a tool call'
    return [
      'Approval needed: ' + action.slice(0, JOURNAL_RESPONSE_MAX_CHARS),
      'Reply /approve to allow it or /deny to refuse; /stop cancels the turn.',
    ].join('\n')
  }
  const question = stringPayload(firstQuestion(payload) ?? {}, 'question')
  return [
    question,
    ...(wait.options.length ? [numberedOptions(wait.options)] : []),
    wait.options.length
      ? 'Reply with your answer or its number; /stop cancels the turn.'
      : 'Reply with your answer; /stop cancels the turn.',
  ].join('\n')
}

function numberedOptions(options: readonly string[]): string {
  return options.map((option, index) => String(index + 1) + '. ' + option).join('\n')
}

function parseChannelCommand(text: string): ChannelCommand | undefined {
  const raw = text.trim()
  if (!raw.startsWith('/')) return undefined
  const [head, ...tail] = raw.slice(1).trim().split(/\s+/)
  // Telegram clients send menu commands in groups as '/stop@BotName'; without
  // dropping the address '/stop' missed its fast path and was rejected.
  const name = head?.toLowerCase().replace(/@.*$/, '')
  if (!name) return undefined
  // '/xerxes <prompt>' is the addressing form group adapters admit.
  return { name: name === 'xerxes' ? 'ask' : name, arguments: tail.join(' ').trim() }
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

/**
 * Keep the head of an oversized preview and mark the cut explicitly.
 *
 * Dropping the tail (the previous behavior) silently hid the end of the
 * agent's answer; keeping the head with a visible marker preserves the
 * beginning of the response and tells the reader text was elided.
 */
function previewText(text: string): string {
  if (text.length <= MAX_PREVIEW_CHARS) return text
  return text.slice(0, MAX_PREVIEW_CHARS - PREVIEW_TRUNCATION_MARKER.length) + PREVIEW_TRUNCATION_MARKER
}

function channelJournalTarget(message: ChannelMessage): string {
  return message.roomId ?? message.channelUserId ?? ''
}

function quoteUserBlock(text: string): string {
  return '~~~user\n' + text + '\n~~~'
}

export function sanitizeJournalOutput(text: string): string {
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

function nonNegativeFinite(value: number, name: string): number {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(name + ' must be a non-negative finite number')
  }
  return value
}

function validDate(value: Date): Date {
  if (!Number.isFinite(value.getTime())) throw new TypeError('clock must return a valid Date')
  return value
}

async function sleepUntilAbort(milliseconds: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return
  let resolveAbort: () => void = () => undefined
  const aborted = new Promise<void>(resolve => { resolveAbort = resolve })
  const onAbort = (): void => { resolveAbort() }
  signal.addEventListener('abort', onAbort, { once: true })
  try {
    if (signal.aborted) return
    await Promise.race([Bun.sleep(milliseconds), aborted])
  } finally {
    signal.removeEventListener('abort', onAbort)
  }
}
