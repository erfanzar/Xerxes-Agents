// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { appendFile, mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { join, resolve } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { xerxesHome } from '../daemon/paths.js'
import type { ChatMessage } from '../types/messages.js'
import { createMainChannel } from './channels.js'
import {
  COORDINATOR_NAME,
  MAIN_CHANNEL,
  type SwarmBoardMessage,
  type SwarmChannelRecord,
  type SwarmMemberRecord,
  type SwarmOverviewSnapshot,
} from './types.js'

const FORBIDDEN_MAIN_CHANNEL_MENTION = /(?:^|[\s([{"',;:])@#main(?=$|[^a-z0-9_-])/i
const CHANNEL_MENTION_REGEX = /(?:^|(?<=[\s([{"',;:]))@#([a-z0-9][a-z0-9_-]{0,31})(?=$|[^a-z0-9_-])/gi
const MEMBER_MENTION_REGEX = /(?:^|(?<=[\s([{"',;:]))@([a-z0-9][a-z0-9_-]{0,23})(?=$|[^a-z0-9_-])/gi

export interface ParsedBoardPost {
  readonly mentions: string[]
  readonly channel_mentions: string[]
  readonly broadcast: boolean
}

export function parseBoardPostText(text: string): ParsedBoardPost {
  if (FORBIDDEN_MAIN_CHANNEL_MENTION.test(text)) {
    throw new ValidationError(
      'text',
      '@#main is not allowed because #main reaches the entire swarm; use @all for broadcasts or @<name> for specific members',
    )
  }

  const channelMentions: string[] = []
  const seenChannels = new Set<string>()
  for (const match of text.matchAll(CHANNEL_MENTION_REGEX)) {
    const channelName = match[1]?.toLowerCase()
    if (channelName && !seenChannels.has(channelName)) {
      seenChannels.add(channelName)
      channelMentions.push(channelName)
    }
  }

  let broadcast = false
  const mentions: string[] = []
  const seenMentions = new Set<string>()
  for (const match of text.matchAll(MEMBER_MENTION_REGEX)) {
    const token = match[1]?.toLowerCase()
    if (!token) continue
    if (token === 'all') {
      broadcast = true
      continue
    }
    if (!seenMentions.has(token)) {
      seenMentions.add(token)
      mentions.push(token)
    }
  }

  return {
    mentions,
    channel_mentions: channelMentions,
    broadcast,
  }
}

/**
 * Format user UI feedback for `@narrator` while stripping `@` prefixes before
 * known member names, `@all`, and `@#channel` tokens so feedback never wakes workers.
 */
export function formatNarratorFeedbackText(
  rawText: string,
  knownNames: Iterable<string>,
): string {
  const trimmed = rawText.trim()
  if (!trimmed) {
    throw new ValidationError('text', 'feedback text must not be empty')
  }
  const protectedTokens = new Set<string>(['all', COORDINATOR_NAME, 'narrator', 'monitor'])
  for (const name of knownNames) {
    protectedTokens.add(name.toLowerCase())
  }
  const sanitized = trimmed
    .replace(/@#([a-z0-9][a-z0-9_-]{0,31})\b/gi, '#$1')
    .replace(/@([a-z0-9][a-z0-9_-]{0,23})\b/gi, (full, token: string) => {
      return protectedTokens.has(token.toLowerCase()) ? token : full
    })
  return `@narrator user feedback: ${sanitized}`
}

export interface PersistedSwarmState {
  readonly version: 1
  readonly session_id: string
  readonly workspace_dir: string
  readonly title: string
  readonly coordinator_activated: boolean
  readonly coordinator_model: string
  readonly coordinator_model_display_name: string
  readonly coordinator_last_read_seq: number
  readonly coordinator_last_post_seq?: number
  readonly coordinator_last_post_time?: string
  readonly head: number
  readonly overview: SwarmOverviewSnapshot | null
  readonly channels: readonly SwarmChannelRecord[]
  readonly members: readonly SwarmMemberRecord[]
}

export interface LoadedSwarmSnapshot {
  readonly state: PersistedSwarmState
  readonly messages: SwarmBoardMessage[]
  readonly histories: Map<string, ChatMessage[]>
}

function sanitizeSegment(value: string): string {
  const cleaned = value.replace(/[^a-zA-Z0-9._-]+/g, '_').slice(0, 64)
  const hash = createHash('sha256').update(value, 'utf8').digest('hex').slice(0, 12)
  return cleaned ? `${cleaned}-${hash}` : hash
}

export function defaultSwarmStorageDirectory(
  workspaceDir: string,
  sessionId: string,
  homeDir = xerxesHome(),
): string {
  const canonicalWorkspace = resolve(workspaceDir)
  const workspaceKey = createHash('sha256').update(canonicalWorkspace, 'utf8').digest('hex').slice(0, 16)
  return join(homeDir, 'swarms', workspaceKey, sanitizeSegment(sessionId))
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isMissingError(error: unknown): boolean {
  return isRecord(error) && error.code === 'ENOENT'
}

export class SwarmBoardStore {
  readonly directory: string
  private readonly statePath: string
  private readonly messagesPath: string
  private readonly historiesDir: string
  private writeChain: Promise<void> = Promise.resolve()
  private pendingStateSlot: {
    state: PersistedSwarmState | undefined
    promise: Promise<void>
  } | undefined

  constructor(directory: string) {
    this.directory = resolve(directory)
    this.statePath = join(this.directory, 'state.json')
    this.messagesPath = join(this.directory, 'messages.jsonl')
    this.historiesDir = join(this.directory, 'histories')
  }

  private enqueue<T>(operation: () => Promise<T>): Promise<T> {
    const next = this.writeChain.then(operation, operation)
    this.writeChain = next.then(() => undefined, () => undefined)
    return next
  }

  async flush(): Promise<void> {
    await this.writeChain
  }

  async load(defaults: {
    readonly sessionId: string
    readonly workspaceDir: string
    readonly coordinatorModel: string
    readonly coordinatorModelDisplayName: string
    readonly nowIso: string
    /**
     * Skip eagerly loading inactive (`killed` / `stopped`) worker histories
     * into RAM by default; `SwarmManager` lazy-loads on worker revival, fork,
     * or explicit `inspectMember`.
     */
    readonly includeInactiveHistories?: boolean
  }): Promise<LoadedSwarmSnapshot | undefined> {
    let rawStateText: string
    try {
      rawStateText = await readFile(this.statePath, 'utf8')
    } catch (error) {
      if (isMissingError(error)) return undefined
      throw error
    }

    const parsedState: unknown = JSON.parse(rawStateText)
    if (!isRecord(parsedState) || parsedState.version !== 1) {
      throw new ValidationError('swarm.state', 'invalid or unsupported persisted swarm state version')
    }

    const messages: SwarmBoardMessage[] = []
    try {
      const rawLines = await readFile(this.messagesPath, 'utf8')
      for (const line of rawLines.split('\n')) {
        const trimmed = line.trim()
        if (!trimmed) continue
        const msg = JSON.parse(trimmed) as SwarmBoardMessage
        if (typeof msg.seq === 'number' && typeof msg.agent === 'string' && typeof msg.text === 'string') {
          messages.push({
            seq: msg.seq,
            agent: msg.agent,
            channel: typeof msg.channel === 'string' && msg.channel ? msg.channel : MAIN_CHANNEL,
            text: msg.text,
            mentions: Array.isArray(msg.mentions) ? msg.mentions.filter((item): item is string => typeof item === 'string') : [],
            channel_mentions: Array.isArray(msg.channel_mentions)
              ? msg.channel_mentions.filter((item): item is string => typeof item === 'string')
              : [],
            broadcast: Boolean(msg.broadcast),
            time: typeof msg.time === 'string' ? msg.time : defaults.nowIso,
            ...(msg.kind === undefined ? {} : { kind: msg.kind }),
          })
        }
      }
    } catch (error) {
      if (!isMissingError(error)) throw error
    }

    messages.sort((left, right) => left.seq - right.seq)
    const maxMessageSeq = messages.at(-1)?.seq ?? 0
    const recordedHead = typeof parsedState.head === 'number' ? parsedState.head : 0
    const head = Math.max(recordedHead, maxMessageSeq)

    const rawChannels = Array.isArray(parsedState.channels) ? parsedState.channels : []
    const channels: SwarmChannelRecord[] = []
    let hasMain = false
    for (const item of rawChannels) {
      if (!isRecord(item) || typeof item.name !== 'string') continue
      if (item.name === MAIN_CHANNEL) hasMain = true
      channels.push({
        name: item.name,
        topic: typeof item.topic === 'string' ? item.topic : '',
        created_by: typeof item.created_by === 'string' ? item.created_by : COORDINATOR_NAME,
        created_at: typeof item.created_at === 'string' ? item.created_at : defaults.nowIso,
        archived: Boolean(item.archived),
      })
    }
    if (!hasMain) {
      channels.unshift(createMainChannel(defaults.nowIso))
    }

    const rawMembers = Array.isArray(parsedState.members) ? parsedState.members : []
    const members: SwarmMemberRecord[] = []
    const histories = new Map<string, ChatMessage[]>()
    for (const item of rawMembers) {
      if (!isRecord(item) || typeof item.name !== 'string') continue
      const killed = Boolean(item.killed) || item.state === 'killed'
      // Restored non-killed workers come back in `'stopped'` state until a board post names them.
      const restoredState = killed ? 'killed' : 'stopped'
      const record: SwarmMemberRecord = {
        name: item.name,
        ...(typeof item.group === 'string' ? { group: item.group } : {}),
        role: item.role === 'narrator' || item.role === 'monitor' || item.role === 'coordinator' ? item.role : 'worker',
        ...(typeof item.added_by === 'string' ? { added_by: item.added_by } : {}),
        model: typeof item.model === 'string' && item.model ? item.model : defaults.coordinatorModel,
        model_display_name: typeof item.model_display_name === 'string' && item.model_display_name
          ? item.model_display_name
          : defaults.coordinatorModelDisplayName,
        workspace: item.workspace === 'share' || item.workspace === 'branch' ? item.workspace : 'inherit',
        workspace_path: typeof item.workspace_path === 'string' && item.workspace_path
          ? item.workspace_path
          : defaults.workspaceDir,
        ...(typeof item.workspace_handle_id === 'string' ? { workspace_handle_id: item.workspace_handle_id } : {}),
        ...(typeof item.workspace_base_head === 'string' ? { workspace_base_head: item.workspace_base_head } : {}),
        channels: Array.isArray(item.channels)
          ? item.channels.filter((ch): ch is string => typeof ch === 'string')
          : [MAIN_CHANNEL],
        state: restoredState,
        killed,
        ...(typeof item.parked_since === 'number' ? { parked_since: item.parked_since } : {}),
        ...(typeof item.mentions_only === 'boolean' ? { mentions_only: item.mentions_only } : {}),
        last_read_seq: typeof item.last_read_seq === 'number' ? item.last_read_seq : 0,
        ...(typeof item.last_post_seq === 'number' ? { last_post_seq: item.last_post_seq } : {}),
        ...(typeof item.last_post_time === 'string' ? { last_post_time: item.last_post_time } : {}),
        ...(typeof item.error === 'string' ? { error: item.error } : {}),
      }
      members.push(record)

      // Every restored member comes back either `'killed'` or `'stopped'`.
      // Lazy-load histories on demand when a worker wakes, forks, or is inspected
      // so loading a 56-member swarm costs ~0 MB instead of 100+ MB of heap.
      if (defaults.includeInactiveHistories === true) {
        const decoded = await this.loadMemberHistory(record.name)
        if (decoded !== undefined) {
          histories.set(record.name, decoded)
        }
      }
    }

    let overview: SwarmOverviewSnapshot | null = null
    if (isRecord(parsedState.overview) && typeof parsedState.overview.markdown === 'string') {
      overview = {
        markdown: parsedState.overview.markdown,
        ...(typeof parsedState.overview.summary === 'string' ? { summary: parsedState.overview.summary } : {}),
        ...(typeof parsedState.overview.updated_at === 'string' ? { updated_at: parsedState.overview.updated_at } : {}),
        ...(typeof parsedState.overview.author === 'string' ? { author: parsedState.overview.author } : {}),
      }
    }

    let lastCoordinatorPostSeq = typeof parsedState.coordinator_last_post_seq === 'number'
      ? parsedState.coordinator_last_post_seq
      : undefined
    let lastCoordinatorPostTime = typeof parsedState.coordinator_last_post_time === 'string'
      ? parsedState.coordinator_last_post_time
      : undefined
    if (lastCoordinatorPostSeq === undefined) {
      for (let idx = messages.length - 1; idx >= 0; idx -= 1) {
        const candidate = messages[idx]
        if (candidate?.agent === COORDINATOR_NAME) {
          lastCoordinatorPostSeq = candidate.seq
          lastCoordinatorPostTime = candidate.time
          break
        }
      }
    }

    return {
      state: {
        version: 1,
        session_id: typeof parsedState.session_id === 'string' ? parsedState.session_id : defaults.sessionId,
        workspace_dir: typeof parsedState.workspace_dir === 'string' ? parsedState.workspace_dir : defaults.workspaceDir,
        title: typeof parsedState.title === 'string' ? parsedState.title : '',
        coordinator_activated: Boolean(parsedState.coordinator_activated),
        coordinator_model: typeof parsedState.coordinator_model === 'string' && parsedState.coordinator_model
          ? parsedState.coordinator_model
          : defaults.coordinatorModel,
        coordinator_model_display_name: typeof parsedState.coordinator_model_display_name === 'string'
          && parsedState.coordinator_model_display_name
          ? parsedState.coordinator_model_display_name
          : defaults.coordinatorModelDisplayName,
        coordinator_last_read_seq: typeof parsedState.coordinator_last_read_seq === 'number'
          ? parsedState.coordinator_last_read_seq
          : 0,
        ...(lastCoordinatorPostSeq === undefined ? {} : { coordinator_last_post_seq: lastCoordinatorPostSeq }),
        ...(lastCoordinatorPostTime === undefined ? {} : { coordinator_last_post_time: lastCoordinatorPostTime }),
        head,
        overview,
        channels,
        members,
      },
      messages,
      histories,
    }
  }

  async loadMemberHistory(memberName: string): Promise<ChatMessage[] | undefined> {
    const historyFile = join(this.historiesDir, `${memberName}.json`)
    try {
      const rawHistory = await readFile(historyFile, 'utf8')
      const decoded: unknown = JSON.parse(rawHistory)
      return Array.isArray(decoded) ? (decoded as ChatMessage[]) : undefined
    } catch (error) {
      if (isMissingError(error)) return undefined
      throw error
    }
  }

  loadMemberHistorySync(memberName: string): ChatMessage[] | undefined {
    const historyFile = join(this.historiesDir, `${memberName}.json`)
    try {
      const rawHistory = readFileSync(historyFile, 'utf8')
      const decoded: unknown = JSON.parse(rawHistory)
      return Array.isArray(decoded) ? (decoded as ChatMessage[]) : undefined
    } catch (error) {
      if (isMissingError(error)) return undefined
      throw error
    }
  }

  async appendMessageAndSaveState(
    message: SwarmBoardMessage,
    state: PersistedSwarmState,
  ): Promise<void> {
    await this.appendMessagesAndSaveState([message], state)
  }

  async appendMessagesAndSaveState(
    messages: readonly SwarmBoardMessage[],
    state: PersistedSwarmState,
  ): Promise<void> {
    // Detach and cancel any queued standalone state save slot so:
    // 1. the earlier queued save becomes a no-op instead of writing stale state, and
    // 2. any later `saveState()` creates a fresh slot ordered strictly AFTER this write.
    if (this.pendingStateSlot) {
      this.pendingStateSlot.state = undefined
      this.pendingStateSlot = undefined
    }
    await this.enqueue(async () => {
      await mkdir(this.directory, { recursive: true, mode: 0o700 })
      if (messages.length > 0) {
        const lines = messages.map(msg => `${JSON.stringify(msg)}\n`).join('')
        await appendFile(this.messagesPath, lines, { encoding: 'utf8', mode: 0o600 })
      }
      await this.writeStateAtomically(state)
    })
  }

  async saveState(state: PersistedSwarmState): Promise<void> {
    if (this.pendingStateSlot) {
      this.pendingStateSlot.state = state
      return this.pendingStateSlot.promise
    }
    const slot: { state: PersistedSwarmState | undefined; promise: Promise<void> } = {
      state,
      promise: Promise.resolve(),
    }
    const scheduled = this.enqueue(async () => {
      const latest = slot.state
      slot.state = undefined
      if (this.pendingStateSlot === slot) {
        this.pendingStateSlot = undefined
      }
      if (!latest) return
      await mkdir(this.directory, { recursive: true, mode: 0o700 })
      await this.writeStateAtomically(latest)
    })
    slot.promise = scheduled
    this.pendingStateSlot = slot
    return scheduled
  }

  async saveMemberHistory(memberName: string, messages: readonly ChatMessage[]): Promise<void> {
    await this.enqueue(async () => {
      await mkdir(this.historiesDir, { recursive: true, mode: 0o700 })
      const target = join(this.historiesDir, `${memberName}.json`)
      const temp = `${target}.${crypto.randomUUID()}.tmp`
      await writeFile(temp, JSON.stringify(messages), { encoding: 'utf8', mode: 0o600 })
      await rename(temp, target)
    })
  }

  private async writeStateAtomically(state: PersistedSwarmState): Promise<void> {
    const temp = `${this.statePath}.${crypto.randomUUID()}.tmp`
    await writeFile(temp, JSON.stringify(state), { encoding: 'utf8', mode: 0o600 })
    await rename(temp, this.statePath)
  }
}
