// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { Channel, InboundHandler } from './base.js'
import { BlueBubblesChannel } from './blueBubbles.js'
import { DingTalkChannel } from './dingtalk.js'
import { DiscordChannel } from './discord.js'
import type { DiscordApplicationRestPort } from './discordApplications.js'
import type { DiscordGatewayPorts } from './discordGateway.js'
import { EmailChannel } from './emailImap.js'
import { FeishuChannel } from './feishu.js'
import { GenericWebhookChannel } from './genericWebhook.js'
import { HomeAssistantChannel } from './homeAssistant.js'
import { ChannelManager, ChannelNotConfiguredError, type ManagedChannelStatus } from './manager.js'
import { MatrixChannel } from './matrix.js'
import { MattermostChannel } from './mattermost.js'
import { SignalChannel } from './signal.js'
import { SlackChannel } from './slack.js'
import { TelegramChannel } from './telegram.js'
import { TelegramPollingLoop, type TelegramPollingChannel } from './telegramPolling.js'
import { TwilioSmsChannel } from './twilioSms.js'
import { WeComChannel } from './wecom.js'
import { WhatsAppChannel } from './whatsApp.js'
import type { ChannelMessage } from './types.js'

export type ChannelSettings = Readonly<Record<string, unknown>>

export interface ConfiguredChannelSpec {
  readonly enabled?: boolean
  readonly settings?: ChannelSettings
  readonly type?: string
}

export type ConfiguredChannelSpecs = Readonly<Record<string, ConfiguredChannelSpec | ChannelSettings>>

export interface ConfiguredChannelManagerOptions {
  readonly channels: ConfiguredChannelSpecs
  /** Explicit REST port for Discord command registration and interaction acknowledgement. */
  readonly discordApplicationRest?: DiscordApplicationRestPort
  /** Explicit network ports for Discord's live Gateway transport. */
  readonly discordGatewayPorts?: DiscordGatewayPorts
  /** Host override for adapters that need non-serializable transport ports. */
  readonly factory?: ConfiguredChannelFactory
  readonly onInbound?: InboundHandler
}

export type ConfiguredChannelFactory = (
  type: string,
  name: string,
  settings: ChannelSettings,
) => Channel

export interface ConfiguredChannelTransportPorts {
  /** Host-owned Discord application-command REST port. */
  readonly discordApplicationRest?: DiscordApplicationRestPort
  /** Host-owned Discord Gateway endpoint discovery and WebSocket ports. */
  readonly discordGatewayPorts?: DiscordGatewayPorts
}

interface LoadedChannel {
  channel: Channel | undefined
  enabled: boolean
  error: string | undefined
  readonly name: string
  readonly settings: ChannelSettings
  readonly type: string
}

/** Raised when a saved channel declaration cannot be constructed safely. */
export class ChannelConfigurationError extends Error {
  readonly channel: string

  constructor(channel: string, message: string) {
    super('channel ' + JSON.stringify(channel) + ' is misconfigured: ' + message)
    this.name = new.target.name
    this.channel = channel
  }
}

/**
 * Channel manager constructed from resolved daemon configuration.
 *
 * It records disabled and invalid declarations instead of dropping them, so
 * channel.list is an operational diagnostic and a later enable command can
 * retry construction after an in-memory configuration update.
 */
export class ConfiguredChannelManager extends ChannelManager {
  private readonly factory: ConfiguredChannelFactory
  private readonly loaded = new Map<string, LoadedChannel>()
  private readonly polling = new Map<string, TelegramPollingLoop>()

  constructor(options: ConfiguredChannelManagerOptions) {
    super({ ...(options.onInbound === undefined ? {} : { onInbound: options.onInbound }) })
    this.factory = options.factory ?? ((type, name, settings) => createConfiguredChannel(type, name, settings, {
      ...(options.discordApplicationRest === undefined
        ? {}
        : { discordApplicationRest: options.discordApplicationRest }),
      ...(options.discordGatewayPorts === undefined ? {} : { discordGatewayPorts: options.discordGatewayPorts }),
    }))
    for (const [configuredName, raw] of Object.entries(options.channels)) {
      const name = configuredName.trim()
      if (!name) continue
      const spec = normalizedSpec(raw)
      const loaded: LoadedChannel = {
        name,
        type: normalizedType(spec.type, name),
        enabled: spec.enabled === true,
        settings: spec.settings ?? {},
        channel: undefined,
        error: undefined,
      }
      this.loaded.set(name, loaded)
      this.materialize(loaded)
    }
  }

  override get hasConfiguredChannels(): boolean {
    return this.loaded.size > 0
  }

  override list(): readonly ManagedChannelStatus[] {
    return [...this.loaded.values()]
      .sort((left, right) => left.name.localeCompare(right.name))
      .map(loaded => this.loadedStatus(loaded))
  }

  override status(name: string): ManagedChannelStatus | undefined {
    const loaded = this.loaded.get(name.trim())
    return loaded ? this.loadedStatus(loaded) : undefined
  }

  override async enable(name: string): Promise<ManagedChannelStatus> {
    const loaded = this.requireLoaded(name)
    loaded.enabled = true
    this.materialize(loaded)
    if (!loaded.channel) {
      throw new ChannelConfigurationError(loaded.name, loaded.error ?? 'no channel adapter was constructed')
    }
    await super.enable(loaded.name)
    await this.startPolling(loaded)
    return this.loadedStatus(loaded)
  }

  override async disable(name: string): Promise<ManagedChannelStatus> {
    const loaded = this.requireLoaded(name)
    loaded.enabled = false
    await this.stopPolling(loaded.name)
    if (loaded.channel) {
      await super.disable(loaded.name)
    }
    return this.loadedStatus(loaded)
  }

  /** Start only declarations explicitly enabled in persisted daemon configuration. */
  async startConfigured(): Promise<readonly ManagedChannelStatus[]> {
    for (const loaded of this.loaded.values()) {
      if (!loaded.enabled) continue
      this.materialize(loaded)
      if (!loaded.channel) continue
      try {
        await super.enable(loaded.name)
        await this.startPolling(loaded)
      } catch {
        // Lifecycle errors are retained by ChannelRegistry and surfaced in list().
      }
    }
    return this.list()
  }

  /** True when a configured adapter can receive the daemon webhook endpoint. */
  hasWebhookChannels(): boolean {
    return [...this.loaded.values()].some(loaded => loaded.channel !== undefined && hasWebhookHandler(loaded.channel))
  }

  override async stopAll(): Promise<void> {
    await Promise.all([...this.polling.keys()].map(name => this.stopPolling(name)))
    await super.stopAll()
  }

  private loadedStatus(loaded: LoadedChannel): ManagedChannelStatus {
    const current = loaded.channel ? super.status(loaded.name) : undefined
    return {
      name: loaded.name,
      adapterName: loaded.type,
      enabled: current?.enabled ?? false,
      ...(current?.lastOperation === undefined ? {} : { lastOperation: current.lastOperation }),
      ...(loaded.error === undefined
        ? current?.lastError === undefined ? {} : { lastError: current.lastError }
        : { lastError: loaded.error }),
    }
  }

  private materialize(loaded: LoadedChannel): void {
    if (loaded.channel) return
    try {
      const channel = configuredChannelName(loaded.name, this.factory(loaded.type, loaded.name, loaded.settings))
      this.register(loaded.name, channel)
      loaded.channel = channel
      loaded.error = undefined
    } catch (error) {
      loaded.error = errorMessage(error)
    }
  }

  private requireLoaded(name: string): LoadedChannel {
    const normalized = name.trim()
    const loaded = this.loaded.get(normalized)
    if (!loaded) throw new ChannelNotConfiguredError(normalized || name)
    return loaded
  }

  private async startPolling(loaded: LoadedChannel): Promise<void> {
    if (loaded.type !== 'telegram' || !telegramUsesPolling(loaded) || this.polling.has(loaded.name)) return
    if (!isTelegramPollingChannel(loaded.channel)) {
      loaded.error = 'Telegram adapter does not support Bot API polling'
      return
    }
    try {
      if (isTelegramWebhookManager(loaded.channel)) {
        await loaded.channel.deleteWebhook()
      }
      const loop = new TelegramPollingLoop({
        channel: loaded.channel,
        timeout: pollingInteger(loaded.settings, 'polling_timeout', 30),
        retryDelay: pollingInteger(loaded.settings, 'polling_retry_delay', 2_000),
        onError: error => { loaded.error = 'Telegram polling: ' + errorMessage(error) },
      })
      this.polling.set(loaded.name, loop)
    } catch (error) {
      loaded.error = 'Telegram polling: ' + errorMessage(error)
    }
  }

  private async stopPolling(name: string): Promise<void> {
    const loop = this.polling.get(name)
    if (!loop) return
    this.polling.delete(name)
    await loop.stop()
  }
}

/** Build a concrete adapter from a resolved channel settings object. */
export function createConfiguredChannel(
  type: string,
  name: string,
  settings: ChannelSettings,
  ports: ConfiguredChannelTransportPorts = {},
): Channel {
  switch (type) {
    case 'telegram':
      return new TelegramChannel({
        token: requiredString(type, settings, 'token'),
        ...telegramOptions(settings),
      })
    case 'discord':
      return new DiscordChannel({
        botToken: requiredString(type, settings, 'bot_token', 'botToken', 'token'),
        ...discordOptions(settings),
        ...(ports.discordApplicationRest === undefined
          ? {}
          : { applicationRest: ports.discordApplicationRest }),
        ...(ports.discordGatewayPorts === undefined ? {} : { gatewayPorts: ports.discordGatewayPorts }),
      })
    case 'slack':
      return new SlackChannel({
        ...optionalField('botToken', optionalString(settings, 'bot_token', 'botToken')),
        ...optionalField('signingSecret', optionalString(settings, 'signing_secret', 'signingSecret')),
        ...optionalField('installId', optionalString(settings, 'install_id', 'installId')),
        ...optionalField('requireSignature', booleanSetting(settings, 'require_signature', 'requireSignature')),
      })
    case 'whatsapp':
      return new WhatsAppChannel({
        accessToken: requiredString(type, settings, 'access_token', 'accessToken'),
        phoneNumberId: requiredString(type, settings, 'phone_number_id', 'phoneNumberId'),
        ...optionalField('apiBaseUrl', optionalString(settings, 'api_base_url', 'apiBaseUrl')),
        ...optionalField('apiVersion', optionalString(settings, 'api_version', 'apiVersion')),
      })
    case 'matrix':
      return new MatrixChannel({
        accessToken: requiredString(type, settings, 'access_token', 'accessToken'),
        homeserverUrl: requiredString(type, settings, 'homeserver_url', 'homeserverUrl'),
      })
    case 'signal':
      return new SignalChannel({
        restBaseUrl: requiredString(type, settings, 'rest_base_url', 'restBaseUrl'),
        senderNumber: requiredString(type, settings, 'sender_number', 'senderNumber'),
      })
    case 'dingtalk':
      return new DingTalkChannel({
        webhookUrl: requiredString(type, settings, 'webhook_url', 'webhookUrl'),
      })
    case 'feishu':
      return new FeishuChannel({
        ...optionalField('tenantAccessToken', optionalString(settings, 'tenant_access_token', 'tenantAccessToken')),
        ...optionalField('apiBaseUrl', optionalString(settings, 'api_base_url', 'apiBaseUrl')),
      })
    case 'wecom':
      return new WeComChannel({
        agentId: requiredNumberOrString(type, settings, 'agent_id', 'agentId'),
        ...optionalField('accessToken', optionalString(settings, 'access_token', 'accessToken')),
        ...optionalField('apiBaseUrl', optionalString(settings, 'api_base_url', 'apiBaseUrl')),
      })
    case 'mattermost':
      return new MattermostChannel({
        baseUrl: requiredString(type, settings, 'base_url', 'baseUrl'),
        botToken: requiredString(type, settings, 'bot_token', 'botToken'),
      })
    case 'sms':
    case 'twilio_sms':
      return new TwilioSmsChannel({
        accountSid: requiredString(type, settings, 'account_sid', 'accountSid'),
        authToken: requiredString(type, settings, 'auth_token', 'authToken'),
        fromNumber: requiredString(type, settings, 'from_number', 'fromNumber'),
        ...optionalField('apiBaseUrl', optionalString(settings, 'api_base_url', 'apiBaseUrl')),
      })
    case 'bluebubbles':
      return new BlueBubblesChannel({
        password: requiredString(type, settings, 'password'),
        serverUrl: requiredString(type, settings, 'server_url', 'serverUrl'),
      })
    case 'home_assistant':
      return new HomeAssistantChannel({
        accessToken: requiredString(type, settings, 'access_token', 'accessToken'),
        baseUrl: requiredString(type, settings, 'base_url', 'baseUrl'),
        ...optionalField('notificationTitle', optionalString(settings, 'notification_title', 'notificationTitle')),
      })
    case 'email':
    case 'email_imap':
      return new EmailChannel({
        ...optionalField('fromAddress', optionalString(settings, 'from_address', 'fromAddress')),
        ...optionalField('smtpHost', optionalString(settings, 'smtp_host', 'smtpHost')),
        ...optionalField('smtpPassword', optionalString(settings, 'smtp_password', 'smtpPassword')),
        ...optionalField('smtpPort', optionalPort(settings, 'smtp_port', 'smtpPort')),
        ...optionalField('smtpUser', optionalString(settings, 'smtp_user', 'smtpUser')),
        ...optionalField(
          'requireImapTransport',
          booleanSetting(settings, 'require_imap_transport', 'requireImapTransport'),
        ),
      })
    case 'generic_webhook':
    case 'webhook':
      return new GenericWebhookChannel({
        name,
        ...optionalField('outboundUrl', optionalString(settings, 'outbound_url', 'outboundUrl')),
        ...optionalField('outboundHeaders', stringRecord(settings, 'outbound_headers', 'outboundHeaders')),
      })
    default:
      throw new Error('unsupported Bun channel type ' + JSON.stringify(type))
  }
}

function discordOptions(settings: ChannelSettings): Omit<ConstructorParameters<typeof DiscordChannel>[0], 'botToken'> {
  const options: Record<string, unknown> = {}
  copySetting(options, settings, 'applicationId', 'application_id', 'applicationId')
  copySetting(options, settings, 'botUserId', 'bot_user_id', 'botUserId')
  copySetting(options, settings, 'deviceName', 'device_name', 'deviceName')
  copySetting(options, settings, 'instanceName', 'instance_name', 'instanceName')
  copySetting(options, settings, 'maxMessageChars', 'max_message_chars', 'maxMessageChars')
  copySetting(options, settings, 'transport', 'transport')
  copyStringList(options, settings, 'addressNames', 'address_names', 'addressNames')
  copyStringList(options, settings, 'allowedChannelIds', 'allowed_channel_ids', 'allowedChannelIds')
  copyStringList(options, settings, 'allowedChannelNames', 'allowed_channel_names', 'allowedChannelNames')
  copyStringList(options, settings, 'allowedGuildIds', 'allowed_guild_ids', 'allowedGuildIds')
  copyBoolean(options, settings, 'alwaysReplyInChannels', 'always_reply_in_channels', 'alwaysReplyInChannels')
  copyBoolean(options, settings, 'ignoreBots', 'ignore_bots', 'ignoreBots')
  copyBoolean(options, settings, 'messageContentIntent', 'message_content_intent', 'messageContentIntent')
  copyBoolean(options, settings, 'registerCommands', 'register_commands', 'registerCommands')
  copyBoolean(options, settings, 'requireMention', 'require_mention', 'requireMention')
  copyBoolean(options, settings, 'suppressMentions', 'suppress_mentions', 'suppressMentions')
  return options as Omit<ConstructorParameters<typeof DiscordChannel>[0], 'botToken'>
}

function telegramOptions(settings: ChannelSettings): Omit<ConstructorParameters<typeof TelegramChannel>[0], 'token'> {
  const options: Record<string, unknown> = {}
  copyStringList(options, settings, 'allowedUserIds', 'allowed_user_ids', 'allowedUserIds')
  copyStringList(options, settings, 'allowedUsernames', 'allowed_usernames', 'allowedUsernames')
  copySetting(options, settings, 'botUsername', 'bot_username', 'botUsername')
  copySetting(options, settings, 'maxPayloadBytes', 'max_payload_bytes', 'maxPayloadBytes')
  copyBoolean(options, settings, 'acceptEditedMessages', 'accept_edited_messages', 'acceptEditedMessages')
  copyBoolean(options, settings, 'requireAllowedSender', 'require_allowed_sender', 'requireAllowedSender')
  copySetting(options, settings, 'webhookSecretToken', 'webhook_secret_token', 'webhookSecretToken')
  copySetting(options, settings, 'webhookUrl', 'webhook_url', 'webhookUrl')
  return options as Omit<ConstructorParameters<typeof TelegramChannel>[0], 'token'>
}

function normalizedSpec(raw: ConfiguredChannelSpec | ChannelSettings): ConfiguredChannelSpec {
  const record = raw as Record<string, unknown>
  const settings = isRecord(record.settings) ? record.settings : {}
  return {
    ...(typeof record.type === 'string' ? { type: record.type } : {}),
    ...(typeof record.enabled === 'boolean' ? { enabled: record.enabled } : {}),
    settings,
  }
}

function normalizedType(value: string | undefined, fallback: string): string {
  return (value ?? fallback).trim().toLowerCase()
}

function telegramUsesPolling(loaded: LoadedChannel): boolean {
  const transport = optionalString(loaded.settings, 'transport')?.toLowerCase() ?? 'auto'
  const webhookUrl = optionalString(loaded.settings, 'webhook_url', 'webhookUrl')
  return transport === 'polling' || (transport === 'auto' && !webhookUrl)
}

function pollingInteger(settings: ChannelSettings, key: string, fallback: number): number {
  const value = firstSetting(settings, [key])
  const parsed = typeof value === 'number'
    ? value
    : typeof value === 'string' && /^\\d+$/.test(value) ? Number.parseInt(value, 10) : Number.NaN
  return Number.isSafeInteger(parsed) && parsed >= 0 ? parsed : fallback
}

function requiredString(type: string, settings: ChannelSettings, ...keys: readonly string[]): string {
  const value = optionalString(settings, ...keys)
  if (value) return value
  throw new Error('missing required setting ' + keys[0] + ' for ' + type)
}

function requiredNumberOrString(type: string, settings: ChannelSettings, ...keys: readonly string[]): string | number {
  const value = firstSetting(settings, keys)
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) return value.trim()
  throw new Error('missing required setting ' + keys[0] + ' for ' + type)
}

function optionalString(settings: ChannelSettings, ...keys: readonly string[]): string | undefined {
  const value = firstSetting(settings, keys)
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function optionalPort(settings: ChannelSettings, ...keys: readonly string[]): number | undefined {
  const value = firstSetting(settings, keys)
  const parsed = typeof value === 'number'
    ? value
    : typeof value === 'string' && /^\\d+$/.test(value) ? Number.parseInt(value, 10) : Number.NaN
  return Number.isSafeInteger(parsed) ? parsed : undefined
}

function optionalField<Key extends string, Value>(
  key: Key,
  value: Value | undefined,
): { readonly [Property in Key]?: Value } {
  return value === undefined ? {} : { [key]: value } as { readonly [Property in Key]?: Value }
}

function booleanSetting(settings: ChannelSettings, ...keys: readonly string[]): boolean | undefined {
  const value = firstSetting(settings, keys)
  if (typeof value === 'boolean') return value
  if (typeof value === 'string') {
    if (value.toLowerCase() === 'true') return true
    if (value.toLowerCase() === 'false') return false
  }
  return undefined
}

function stringRecord(
  settings: ChannelSettings,
  ...keys: readonly string[]
): Readonly<Record<string, string>> | undefined {
  const value = firstSetting(settings, keys)
  if (!isRecord(value)) return undefined
  const entries = Object.entries(value).flatMap(([key, item]) => typeof item === 'string' ? [[key, item] as const] : [])
  return Object.fromEntries(entries)
}

function firstSetting(settings: ChannelSettings, keys: readonly string[]): unknown {
  for (const key of keys) {
    if (key in settings) return settings[key]
  }
  return undefined
}

function copySetting(
  target: Record<string, unknown>,
  settings: ChannelSettings,
  output: string,
  ...keys: readonly string[]
): void {
  const value = firstSetting(settings, keys)
  if (value !== undefined) target[output] = value
}

function copyStringList(
  target: Record<string, unknown>,
  settings: ChannelSettings,
  output: string,
  ...keys: readonly string[]
): void {
  const value = firstSetting(settings, keys)
  if (typeof value === 'string') {
    target[output] = value
    return
  }
  if (Array.isArray(value) && value.every(item => typeof item === 'string')) {
    target[output] = value
  }
}

function copyBoolean(
  target: Record<string, unknown>,
  settings: ChannelSettings,
  output: string,
  ...keys: readonly string[]
): void {
  const value = booleanSetting(settings, ...keys)
  if (value !== undefined) target[output] = value
}

function hasWebhookHandler(channel: Channel): boolean {
  return 'handleWebhook' in channel && typeof channel.handleWebhook === 'function'
}

function isTelegramPollingChannel(channel: Channel | undefined): channel is Channel & TelegramPollingChannel {
  return channel !== undefined
    && 'getUpdates' in channel
    && typeof channel.getUpdates === 'function'
    && 'handleWebhook' in channel
    && typeof channel.handleWebhook === 'function'
}

function isTelegramWebhookManager(channel: Channel): channel is Channel & { deleteWebhook(): Promise<unknown> } {
  return 'deleteWebhook' in channel && typeof channel.deleteWebhook === 'function'
}

function configuredChannelName(name: string, target: Channel): Channel {
  return new Proxy(target, {
    get(channel, property, receiver) {
      if (property === 'name') return name
      if (property === 'start') {
        return async (onInbound: InboundHandler) => channel.start(message => onInbound({
          ...message,
          channel: name,
        }))
      }
      if (property === 'send') {
        return async (message: ChannelMessage) => channel.send({
          ...message,
          channel: channel.name,
        })
      }
      const value = Reflect.get(channel, property, receiver)
      return typeof value === 'function' ? value.bind(channel) : value
    },
  })
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
