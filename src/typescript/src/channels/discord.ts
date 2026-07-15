// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { postJson, providerUrl, type ChannelFetch } from './http.js'
import {
  type DiscordApplicationCommand,
  type DiscordApplicationRestPort,
} from './discordApplications.js'
import {
  DiscordGatewayTransport,
  type DiscordGatewayDispatch,
  type DiscordGatewayPorts,
} from './discordGateway.js'
import {
  createChannelMessage,
  MessageDirection,
  type ChannelAttachment,
  type ChannelMessage,
} from './types.js'
import {
  parseJsonBody,
  WebhookChannel,
  type WebhookHeaders,
} from './webhooks.js'

const DISCORD_API_BASE = 'https://discord.com/api/v10/'
export const DISCORD_MESSAGE_LIMIT = 2_000

type StringList = string | readonly string[] | ReadonlySet<string> | undefined

interface CachedDiscordChannel {
  readonly guildId: string
  readonly name: string
  readonly parentChannelId: string
}

export interface DiscordChannelOptions {
  readonly addressNames?: StringList
  /** Discord application id used for command registration. It can also be learned from READY. */
  readonly applicationId?: string
  /** Explicit REST boundary for application commands and interaction responses. */
  readonly applicationRest?: DiscordApplicationRestPort
  readonly allowedChannelIds?: StringList
  readonly allowedChannelNames?: StringList
  readonly allowedGuildIds?: StringList
  readonly alwaysReplyInChannels?: boolean | string
  /** Override only for a Discord-compatible API or tests. */
  readonly apiBaseUrl?: string
  readonly botToken?: string
  readonly botUserId?: string
  readonly deviceName?: string
  readonly fetchImplementation?: ChannelFetch
  /** Explicit host-owned Gateway network ports. Required for `transport: "gateway"`. */
  readonly gatewayPorts?: DiscordGatewayPorts
  readonly ignoreBots?: boolean | string
  readonly instanceName?: string
  readonly maxMessageChars?: number | string
  /** Request Discord's privileged message-content intent during Gateway identify. */
  readonly messageContentIntent?: boolean | string
  /** Register `/ask`, `/skills`, `/skill`, and `/status` after the Gateway READY event. */
  readonly registerCommands?: boolean | string
  readonly requireMention?: boolean | string
  readonly suppressMentions?: boolean | string
  readonly token?: string
  /** Use a host-injected native Discord Gateway transport instead of webhook delivery. */
  readonly transport?: 'webhook' | 'gateway'
}

/** Discord REST/webhook adapter with an optional, explicitly injected Gateway lifecycle. */
export class DiscordChannel extends WebhookChannel {
  readonly name = 'discord'

  private readonly addressNames: ReadonlySet<string>
  private applicationCommandsSynced = false
  private applicationId: string
  private readonly applicationRest: DiscordApplicationRestPort | undefined
  private readonly allowedChannelIds: ReadonlySet<string>
  private readonly allowedChannelNames: ReadonlySet<string>
  private readonly allowedGuildIds: ReadonlySet<string>
  private readonly alwaysReplyInChannels: boolean
  private readonly apiBaseUrl: string
  private readonly botToken: string
  private botUserId: string
  private readonly fetchImplementation: ChannelFetch | undefined
  private readonly ignoreBots: boolean
  private readonly instanceName: string
  private readonly maxMessageChars: number
  private readonly messageContentIntent: boolean
  private readonly registerCommands: boolean
  private readonly requireMention: boolean
  private readonly suppressMentions: boolean
  private readonly gatewayChannels = new Map<string, CachedDiscordChannel>()
  private readonly gatewayGuildNames = new Map<string, string>()
  private gateway: DiscordGatewayTransport | undefined

  constructor(options: DiscordChannelOptions = {}) {
    super()
    this.botToken = String(options.botToken || options.token || '')
    if (!this.botToken) {
      throw new TypeError('Discord bot token is required')
    }
    const transport = options.transport ?? 'webhook'
    if (transport !== 'webhook' && transport !== 'gateway') {
      throw new TypeError("Discord transport must be 'webhook' or 'gateway'")
    }
    this.addressNames = normalizedNames(options.addressNames)
    this.applicationId = stringOrEmpty(options.applicationId).trim()
    this.applicationRest = options.applicationRest
    this.allowedChannelIds = idSet(options.allowedChannelIds)
    this.allowedChannelNames = normalizedNames(options.allowedChannelNames)
    this.allowedGuildIds = idSet(options.allowedGuildIds)
    this.alwaysReplyInChannels = asBoolean(options.alwaysReplyInChannels, false)
    this.apiBaseUrl = options.apiBaseUrl ?? DISCORD_API_BASE
    this.botUserId = String(options.botUserId ?? '')
    this.fetchImplementation = options.fetchImplementation
    this.ignoreBots = asBoolean(options.ignoreBots, true)
    this.instanceName = String(options.instanceName || options.deviceName || '').trim()
    this.maxMessageChars = messageLimit(options.maxMessageChars)
    this.messageContentIntent = asBoolean(options.messageContentIntent, true)
    this.registerCommands = asBoolean(options.registerCommands, true)
    this.requireMention = asBoolean(options.requireMention, false)
    this.suppressMentions = asBoolean(options.suppressMentions, true)
    if (transport === 'gateway') {
      if (!options.gatewayPorts) {
        throw new TypeError('Discord gateway transport requires explicit gatewayPorts')
      }
      if (this.registerCommands && !this.applicationRest) {
        throw new TypeError('Discord command registration requires explicit applicationRest')
      }
      this.gateway = new DiscordGatewayTransport({
        botToken: this.botToken,
        messageContentIntent: this.messageContentIntent,
        onDispatch: dispatch => this.routeGatewayDispatch(dispatch),
        ports: options.gatewayPorts,
      })
    }
  }

  /** Last contained Gateway protocol/connection failure, if gateway mode is active. */
  get gatewayError(): string {
    return this.gateway?.state.lastError ?? ''
  }

  override async start(onInbound: import('./base.js').InboundHandler): Promise<void> {
    await super.start(onInbound)
    if (!this.gateway) return
    try {
      await this.gateway.start()
    } catch (error) {
      await super.stop()
      throw error
    }
  }

  override async stop(): Promise<void> {
    try {
      await this.gateway?.stop()
    } finally {
      await super.stop()
    }
  }

  protected parseInbound(
    _headers: WebhookHeaders,
    body: Uint8Array,
  ): readonly ChannelMessage[] {
    const payload = parseJsonBody(body)
    if (!Object.keys(payload).length) {
      return []
    }
    const message = messagePayload(payload)
    const normalized = this.messageFromPayload(message)
    return normalized ? [normalized] : []
  }

  /** Post Discord's typing indicator for an active room. */
  async sendTyping(roomId: string | undefined): Promise<void> {
    if (!roomId) {
      return
    }
    await postJson(this.endpoint(`channels/${roomId}/typing`), {
      body: {},
      headers: { Authorization: `Bot ${this.botToken}` },
      ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
    })
  }

  protected async sendOutbound(message: ChannelMessage): Promise<void> {
    if (!message.roomId) {
      throw new TypeError('Discord outbound messages require roomId')
    }
    const text = labelOutbound(message.text || '(no response)', this.instanceName)
    const chunks = chunkText(text, this.maxMessageChars)
    for (const [index, chunk] of chunks.entries()) {
      const payload: Record<string, unknown> = { content: chunk }
      if (this.suppressMentions) {
        payload.allowed_mentions = { parse: [], replied_user: false }
      }
      if (index === 0 && message.replyTo) {
        payload.message_reference = {
          message_id: message.replyTo,
          fail_if_not_exists: false,
        }
      }
      await postJson(this.endpoint(`channels/${message.roomId}/messages`), {
        body: payload,
        headers: { Authorization: `Bot ${this.botToken}` },
        ...(this.fetchImplementation ? { fetchImplementation: this.fetchImplementation } : {}),
      })
    }
  }

  private endpoint(path: string): string {
    return providerUrl(this.apiBaseUrl, path)
  }

  private async routeGatewayDispatch(dispatch: DiscordGatewayDispatch): Promise<void> {
    this.cacheGatewayDispatch(dispatch)
    if (dispatch.type === 'READY') {
      const user = asRecord(dispatch.data.user)
      const userId = stringOrEmpty(user.id)
      if (userId) this.botUserId = userId
      const application = asRecord(dispatch.data.application)
      const applicationId = stringOrEmpty(application.id) || stringOrEmpty(dispatch.data.application_id)
      if (!this.applicationId && applicationId) this.applicationId = applicationId
      await this.syncApplicationCommands(guildIdsFromReady(dispatch.data))
      return
    }
    if (dispatch.type === 'INTERACTION_CREATE') {
      await this.handleGatewayInteraction(dispatch.data)
      return
    }
    if (dispatch.type !== 'MESSAGE_CREATE') return
    await this.handleWebhook({}, new TextEncoder().encode(JSON.stringify({ t: dispatch.type, d: dispatch.data })))
  }

  private async syncApplicationCommands(guildIds: ReadonlySet<string>): Promise<void> {
    if (!this.registerCommands || this.applicationCommandsSynced) return
    const applicationRest = this.applicationRest
    if (!applicationRest) {
      throw new TypeError('Discord command registration requires explicit applicationRest')
    }
    if (!this.applicationId) {
      throw new TypeError('Discord command registration requires applicationId or READY.application.id')
    }
    const commands = discordApplicationCommands()
    const targetGuildIds = this.allowedGuildIds.size ? this.allowedGuildIds : guildIds
    if (targetGuildIds.size) {
      for (const guildId of targetGuildIds) {
        await applicationRest.replaceGuildCommands({
          applicationId: this.applicationId,
          botToken: this.botToken,
          commands,
          guildId,
        })
      }
    } else {
      await applicationRest.replaceGlobalCommands({
        applicationId: this.applicationId,
        botToken: this.botToken,
        commands,
      })
    }
    this.applicationCommandsSynced = true
  }

  private async handleGatewayInteraction(payload: Readonly<Record<string, unknown>>): Promise<void> {
    const command = interactionCommand(payload)
    if (command === undefined) return
    const applicationRest = this.applicationRest
    if (!applicationRest) {
      throw new TypeError('Discord interaction acknowledgement requires explicit applicationRest')
    }
    const interactionId = stringOrEmpty(payload.id)
    const interactionToken = stringOrEmpty(payload.token)
    if (!interactionId || !interactionToken) {
      throw new TypeError('Discord interaction omitted id or token')
    }
    const channel = asRecord(payload.channel)
    const channelId = stringOrEmpty(payload.channel_id) || stringOrEmpty(channel.id)
    const cached = this.gatewayChannels.get(channelId)
    const guildId = stringOrEmpty(payload.guild_id) || cached?.guildId || ''
    const channelNames = this.channelNamesFromPayload({
      channel,
      channel_name: payload.channel_name,
      parent_channel: asRecord(payload.parent_channel),
      thread: asRecord(payload.thread),
      thread_name: payload.thread_name,
    }, cached)
    if (!this.routingAllows(channelId, channelNames, guildId, true)) {
      await applicationRest.acknowledgeInteraction({
        content: 'This Xerxes instance is not configured for this channel.',
        ephemeral: true,
        interactionId,
        interactionToken,
      })
      return
    }
    await applicationRest.acknowledgeInteraction({
      content: 'Queued.',
      ephemeral: true,
      interactionId,
      interactionToken,
    })
    const user = interactionUser(payload)
    const parentChannelId = stringOrEmpty(channel.parent_id)
      || stringOrEmpty(payload.parent_channel_id)
      || cached?.parentChannelId
      || ''
    const channelName = stringOrEmpty(channel.name)
      || stringOrEmpty(payload.channel_name)
      || cached?.name
      || ''
    const guild = asRecord(payload.guild)
    await this.dispatchInbound(createChannelMessage({
      channel: this.name,
      channelUserId: stringOrEmpty(user.id),
      direction: MessageDirection.INBOUND,
      platformMessageId: '',
      roomId: channelId,
      text: command,
      metadata: {
        guild_id: guildId,
        guild_name: stringOrEmpty(guild.name)
          || stringOrEmpty(payload.guild_name)
          || this.gatewayGuildNames.get(guildId)
          || '',
        thread_id: parentChannelId ? channelId : '',
        parent_channel_id: parentChannelId,
        channel_name: channelName,
        chat_type: guildId ? 'group' : 'private',
        discord_interaction: true,
        instance_name: this.instanceName,
      },
    }))
  }

  private messageFromPayload(payload: Readonly<Record<string, unknown>>): ChannelMessage | undefined {
    const author = asRecord(payload.author)
    if (this.ignoreBots && author.bot === true) {
      return undefined
    }
    const channelId = stringOrEmpty(payload.channel_id)
    const cached = this.gatewayChannels.get(channelId)
    const guildId = stringOrEmpty(payload.guild_id) || cached?.guildId || ''
    const channelNames = this.channelNamesFromPayload(payload, cached)
    const content = stringOrEmpty(payload.content)
    if (!this.routingAllows(channelId, channelNames, guildId, this.botMentioned(payload.mentions, content))) {
      return undefined
    }
    const attachments = attachmentsFromPayload(payload.attachments)
    const addressed = stripAddress(stripBotMention(content, this.botUserId), this.addressNames)
    if (addressed === undefined) {
      return undefined
    }
    const text = addressed.trim() || attachmentText(attachments)
    if (!text) {
      return undefined
    }
    return createChannelMessage({
      channel: this.name,
      text,
      direction: MessageDirection.INBOUND,
      channelUserId: stringOrEmpty(author.id),
      roomId: channelId,
      platformMessageId: stringOrEmpty(payload.id),
      attachments,
      metadata: {
        guild_id: guildId,
        guild_name: stringOrEmpty(payload.guild_name) || this.gatewayGuildNames.get(guildId) || '',
        thread_id: stringOrEmpty(payload.thread_id) || (cached?.parentChannelId ? channelId : ''),
        parent_channel_id: stringOrEmpty(payload.parent_channel_id) || cached?.parentChannelId || '',
        channel_name: [...channelNames][0] ?? '',
        instance_name: this.instanceName,
        chat_type: guildId ? 'group' : 'private',
        author_username: stringOrEmpty(author.username),
        author_display_name: stringOrEmpty(asRecord(payload.member).nick)
          || stringOrEmpty(author.global_name)
          || stringOrEmpty(author.username),
      },
    })
  }

  private cacheGatewayDispatch(dispatch: DiscordGatewayDispatch): void {
    const payload = dispatch.data
    if (dispatch.type === 'GUILD_CREATE' || dispatch.type === 'GUILD_UPDATE') {
      const guildId = stringOrEmpty(payload.id)
      const guildName = stringOrEmpty(payload.name)
      if (guildId && guildName) this.gatewayGuildNames.set(guildId, guildName)
      this.cacheChannelCollection(payload.channels, guildId)
      this.cacheChannelCollection(payload.threads, guildId)
      return
    }
    if (dispatch.type === 'GUILD_DELETE') {
      const guildId = stringOrEmpty(payload.id)
      if (!guildId) return
      this.gatewayGuildNames.delete(guildId)
      for (const [channelId, channel] of this.gatewayChannels) {
        if (channel.guildId === guildId) this.gatewayChannels.delete(channelId)
      }
      return
    }
    if (dispatch.type === 'THREAD_LIST_SYNC') {
      const guildId = stringOrEmpty(payload.guild_id)
      this.cacheChannelCollection(payload.channels, guildId)
      this.cacheChannelCollection(payload.threads, guildId)
      return
    }
    if (dispatch.type === 'CHANNEL_DELETE' || dispatch.type === 'THREAD_DELETE') {
      const channelId = stringOrEmpty(payload.id)
      if (channelId) this.gatewayChannels.delete(channelId)
      return
    }
    if (dispatch.type === 'CHANNEL_CREATE' || dispatch.type === 'CHANNEL_UPDATE'
      || dispatch.type === 'THREAD_CREATE' || dispatch.type === 'THREAD_UPDATE'
      || dispatch.type === 'MESSAGE_CREATE' || dispatch.type === 'INTERACTION_CREATE') {
      this.cacheChannel(payload)
      const channel = asRecord(payload.channel)
      if (Object.keys(channel).length) this.cacheChannel(channel, stringOrEmpty(payload.guild_id))
      const guild = asRecord(payload.guild)
      const guildId = stringOrEmpty(payload.guild_id) || stringOrEmpty(guild.id)
      const guildName = stringOrEmpty(payload.guild_name) || stringOrEmpty(guild.name)
      if (guildId && guildName) this.gatewayGuildNames.set(guildId, guildName)
    }
  }

  private cacheChannelCollection(value: unknown, fallbackGuildId = ''): void {
    if (!Array.isArray(value)) return
    for (const candidate of value) {
      this.cacheChannel(asRecord(candidate), fallbackGuildId)
    }
  }

  private cacheChannel(payload: Readonly<Record<string, unknown>>, fallbackGuildId = ''): void {
    const channelId = stringOrEmpty(payload.channel_id) || stringOrEmpty(payload.id)
    if (!channelId) return
    const current = this.gatewayChannels.get(channelId)
    const guildId = stringOrEmpty(payload.guild_id) || fallbackGuildId || current?.guildId || ''
    const name = stringOrEmpty(payload.channel_name) || stringOrEmpty(payload.name) || current?.name || ''
    const parentChannelId = stringOrEmpty(payload.parent_channel_id)
      || stringOrEmpty(payload.parent_id)
      || current?.parentChannelId
      || ''
    this.gatewayChannels.set(channelId, { guildId, name, parentChannelId })
  }

  private channelNamesFromPayload(
    payload: Readonly<Record<string, unknown>>,
    cached: CachedDiscordChannel | undefined,
  ): ReadonlySet<string> {
    const names = channelNamesFromPayload(payload)
    if (cached?.name) names.add(normalizeName(cached.name))
    const parentChannelId = stringOrEmpty(payload.parent_channel_id)
      || stringOrEmpty(asRecord(payload.channel).parent_id)
      || cached?.parentChannelId
      || ''
    const parent = this.gatewayChannels.get(parentChannelId)
    if (parent?.name) names.add(normalizeName(parent.name))
    return names
  }

  private routingAllows(
    channelId: string,
    channelNames: ReadonlySet<string>,
    guildId: string,
    mentioned: boolean,
  ): boolean {
    if (this.allowedGuildIds.size && guildId && !this.allowedGuildIds.has(guildId)) {
      return false
    }
    if (this.allowedChannelIds.size && !this.allowedChannelIds.has(channelId)) {
      return false
    }
    const nameMatches = [...channelNames].some(name => this.allowedChannelNames.has(name))
    if (this.allowedChannelNames.size && !nameMatches) {
      return false
    }
    if (!guildId || this.alwaysReplyInChannels) {
      return true
    }
    if (this.allowedChannelIds.has(channelId) || nameMatches) {
      return true
    }
    return !this.requireMention || mentioned
  }

  private botMentioned(value: unknown, content: string): boolean {
    if (!this.botUserId) {
      return false
    }
    if (Array.isArray(value)) {
      for (const item of value) {
        if (stringOrEmpty(asRecord(item).id) === this.botUserId) {
          return true
        }
      }
    }
    return content.includes(`<@${this.botUserId}>`) || content.includes(`<@!${this.botUserId}>`)
  }
}

/** Split text for Discord without producing empty chunks. */
export function chunkText(text: string, limit = DISCORD_MESSAGE_LIMIT): string[] {
  if (!Number.isInteger(limit) || limit < 1 || limit > DISCORD_MESSAGE_LIMIT) {
    throw new RangeError(`Discord message limit must be between 1 and ${DISCORD_MESSAGE_LIMIT}`)
  }
  if (text.length <= limit) {
    return [text]
  }
  const chunks: string[] = []
  let remaining = text
  while (remaining.length > limit) {
    let splitAt = remaining.lastIndexOf('\n', limit)
    if (splitAt < Math.max(1, Math.floor(limit / 2))) {
      splitAt = limit
    }
    const chunk = remaining.slice(0, splitAt).trimEnd()
    if (chunk) {
      chunks.push(chunk)
    }
    remaining = remaining.slice(splitAt).trimStart()
  }
  if (remaining) {
    chunks.push(remaining)
  }
  return chunks.length ? chunks : ['']
}

/** Built-in application commands registered by the native Discord Gateway adapter. */
export function discordApplicationCommands(): readonly DiscordApplicationCommand[] {
  return [
    {
      name: 'ask',
      description: 'Ask Xerxes to do something in this channel.',
      type: 1,
      options: [{
        name: 'prompt',
        description: 'Prompt to send to Xerxes.',
        required: true,
        type: 3,
      }],
    },
    {
      name: 'skills',
      description: 'List available Xerxes skills.',
      type: 1,
    },
    {
      name: 'skill',
      description: 'Run a Xerxes skill.',
      type: 1,
      options: [
        {
          name: 'name',
          description: 'Skill name, optionally with :subcommand.',
          required: true,
          type: 3,
        },
        {
          name: 'prompt',
          description: 'Optional task for the skill.',
          type: 3,
        },
      ],
    },
    {
      name: 'status',
      description: 'Show Xerxes runtime status.',
      type: 1,
    },
  ]
}

function messagePayload(payload: Readonly<Record<string, unknown>>): Readonly<Record<string, unknown>> {
  if (payload.t === 'MESSAGE_CREATE') {
    return asRecord(payload.d)
  }
  const nested = asRecord(payload.message)
  return Object.keys(nested).length ? nested : payload
}

function guildIdsFromReady(payload: Readonly<Record<string, unknown>>): ReadonlySet<string> {
  if (!Array.isArray(payload.guilds)) return new Set()
  return new Set(payload.guilds
    .map(guild => stringOrEmpty(asRecord(guild).id))
    .filter(Boolean))
}

function interactionCommand(payload: Readonly<Record<string, unknown>>): string | undefined {
  if (payload.type !== 2) return undefined
  const data = asRecord(payload.data)
  const name = stringOrEmpty(data.name).trim()
  if (name === 'ask') {
    return interactionOption(data.options, 'prompt')
  }
  if (name === 'skills') return '/skills'
  if (name === 'status') return '/status'
  if (name !== 'skill') return undefined
  const skill = interactionOption(data.options, 'name')?.trim()
  if (!skill) return undefined
  const prompt = interactionOption(data.options, 'prompt')?.trim()
  return prompt ? `/skill ${skill} ${prompt}` : `/skill ${skill}`
}

function interactionOption(options: unknown, name: string): string | undefined {
  if (!Array.isArray(options)) return undefined
  for (const candidate of options) {
    const option = asRecord(candidate)
    if (stringOrEmpty(option.name) !== name || typeof option.value !== 'string') continue
    return option.value
  }
  return undefined
}

function interactionUser(payload: Readonly<Record<string, unknown>>): Readonly<Record<string, unknown>> {
  const memberUser = asRecord(asRecord(payload.member).user)
  return Object.keys(memberUser).length ? memberUser : asRecord(payload.user)
}

function attachmentsFromPayload(value: unknown): readonly ChannelAttachment[] {
  if (!Array.isArray(value)) {
    return []
  }
  const attachments: ChannelAttachment[] = []
  for (const item of value) {
    const attachment = asRecord(item)
    if (!Object.keys(attachment).length) {
      continue
    }
    attachments.push({
      id: stringOrEmpty(attachment.id),
      filename: stringOrEmpty(attachment.filename),
      url: stringOrEmpty(attachment.url),
      content_type: stringOrEmpty(attachment.content_type),
      size: attachment.size ?? 0,
      width: attachment.width ?? null,
      height: attachment.height ?? null,
    })
  }
  return attachments
}

function attachmentText(attachments: readonly ChannelAttachment[]): string {
  const urls = attachments
    .map(attachment => attachment.url)
    .filter((url): url is string => typeof url === 'string' && Boolean(url))
  if (urls.length) {
    return `Attachments:\n${urls.join('\n')}`
  }
  return attachments.length ? 'Attachments received.' : ''
}

function channelNamesFromPayload(payload: Readonly<Record<string, unknown>>): Set<string> {
  const candidates = [
    payload.channel_name,
    payload.thread_name,
    asRecord(payload.channel).name,
    asRecord(payload.thread).name,
    asRecord(payload.parent_channel).name,
  ]
  return new Set(candidates.map(normalizeName).filter(Boolean))
}

function asBoolean(value: boolean | string | undefined, fallback: boolean): boolean {
  if (value === undefined) {
    return fallback
  }
  if (typeof value === 'boolean') {
    return value
  }
  return !new Set(['0', 'false', 'no', 'off', '']).has(value.trim().toLowerCase())
}

function idSet(value: StringList): ReadonlySet<string> {
  if (!value) {
    return new Set()
  }
  const items = typeof value === 'string' ? value.split(',') : [...value]
  return new Set(items.map(item => String(item).trim()).filter(Boolean))
}

function normalizedNames(value: StringList): ReadonlySet<string> {
  return new Set([...idSet(value)].map(normalizeName).filter(Boolean))
}

function normalizeName(value: unknown): string {
  return String(value ?? '').trim().replace(/^#/, '').toLowerCase()
}

function messageLimit(value: number | string | undefined): number {
  if (value === undefined || value === '') {
    return DISCORD_MESSAGE_LIMIT
  }
  const parsed = Number(value)
  if (!Number.isInteger(parsed)) {
    throw new TypeError('Discord maxMessageChars must be an integer')
  }
  return Math.max(1, Math.min(parsed, DISCORD_MESSAGE_LIMIT))
}

function stripBotMention(content: string, botUserId: string): string {
  if (!botUserId) {
    return content
  }
  return content.replace(new RegExp(`^<@!?${escapeRegex(botUserId)}>\\s*`), '').trim()
}

function stripAddress(text: string, addresses: ReadonlySet<string>): string | undefined {
  if (!addresses.size) {
    return text
  }
  const trimmed = text.trim()
  const lowercase = trimmed.toLowerCase()
  for (const address of addresses) {
    for (const prefix of [address, `@${address}`, `/${address}`]) {
      if (lowercase === prefix) {
        return ''
      }
      if (!lowercase.startsWith(prefix)) {
        continue
      }
      const next = lowercase.slice(prefix.length, prefix.length + 1)
      if ([' ', ':', ',', '-'].includes(next)) {
        return trimmed.slice(prefix.length).replace(/^[ :,\-]+/, '')
      }
    }
  }
  return undefined
}

function labelOutbound(text: string, instanceName: string): string {
  return instanceName ? `[${instanceName}]\n${text}` : text
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function asRecord(value: unknown): Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function stringOrEmpty(value: unknown): string {
  return value === undefined || value === null ? '' : String(value)
}
