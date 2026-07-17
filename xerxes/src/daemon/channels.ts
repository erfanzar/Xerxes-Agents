// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  ChannelTurnRouter,
  ConfiguredChannelManager,
  MarkdownAgentWorkspace,
  type ConfiguredChannelFactory,
  type DiscordApplicationRestPort,
  type DiscordGatewayPorts,
} from '../channels/index.js'
import type { ChannelWebhookServerOptions } from '../channels/webhookServer.js'
import { daemonChannels, type DaemonConfig, type DaemonEnvironment } from './config.js'
import type { DaemonRuntime } from './runtime.js'

export interface DaemonChannelManagerOptions {
  /** Explicit REST port for Discord command registration and interaction acknowledgement. */
  readonly discordApplicationRest?: DiscordApplicationRestPort
  /** Explicit host-owned network ports for Discord's live Gateway transport. */
  readonly discordGatewayPorts?: DiscordGatewayPorts
  /** Explicit environment used to resolve channel `*_env` settings. */
  readonly environment: DaemonEnvironment
  /** Optional application-specific adapter constructor for injected transport ports. */
  readonly factory?: ConfiguredChannelFactory
  /** Workspace applied to daemon sessions created by channel conversations. */
  readonly projectDirectory?: string
  /** Optional explicit Markdown workspace used for channel journaling and system context. */
  readonly workspace?: MarkdownAgentWorkspace
}

/** Construct the configured adapter registry and connect it to native turn delivery. */
export function createDaemonChannelManager(
  config: DaemonConfig,
  runtime: DaemonRuntime,
  options: DaemonChannelManagerOptions,
): ConfiguredChannelManager {
  const manager = new ConfiguredChannelManager({
    channels: daemonChannels(config, options.environment),
    ...(options.discordApplicationRest === undefined
      ? {}
      : { discordApplicationRest: options.discordApplicationRest }),
    ...(options.discordGatewayPorts === undefined ? {} : { discordGatewayPorts: options.discordGatewayPorts }),
    ...(options.factory === undefined ? {} : { factory: options.factory }),
  })
  const workspace = options.workspace ?? new MarkdownAgentWorkspace(
    stringSetting(config.workspace.root) || undefined,
  )
  const router = new ChannelTurnRouter({
    channels: manager,
    cwd: options.projectDirectory ?? config.projectDirectory,
    previewInterval: message => channelPreviewInterval(config, message.channel),
    runtime,
    streamPreviews: message => channelPreviewsEnabled(config, message.channel),
    workspace,
  })
  manager.setInboundHandler(message => router.handle(message))
  return manager
}

/** Resolve the independent Bun listener used for configured channel webhooks. */
export function daemonChannelWebhookOptions(
  config: DaemonConfig,
): Omit<ChannelWebhookServerOptions, 'manager'> {
  const host = stringSetting(config.control.webhook_host)
    || stringSetting(config.control.websocket_host)
    || '127.0.0.1'
  warnOnUnsignedWebhookExposure(config, host)
  return {
    host,
    port: numericSetting(config.control.webhook_port, 11997),
  }
}

const LOOPBACK_HOSTS = new Set(['127.0.0.1', '::1', 'localhost'])

/** Fail-open webhook auth stays silent; make a public bind without verification loud. */
function warnOnUnsignedWebhookExposure(config: DaemonConfig, host: string): void {
  if (LOOPBACK_HOSTS.has(host.trim().toLowerCase())) return
  const unsigned = Object.entries(config.channels)
    .filter(([, channel]) => enabledWithoutSignatureVerification(channel))
    .map(([name]) => name)
  if (unsigned.length === 0) return
  console.warn([
    `[xerxes] SECURITY WARNING: the channel webhook listener binds to non-loopback host '${host}'`,
    `but these enabled channels perform no webhook signature verification: ${unsigned.join(', ')}.`,
    'Their webhook endpoints are unauthenticated and publicly reachable.',
    'Configure a signing secret (for example webhook_secret_token or signing_secret)',
    "or bind webhook_host to '127.0.0.1'.",
  ].join(' '))
}

function enabledWithoutSignatureVerification(channel: unknown): boolean {
  if (!isRecord(channel) || channel.enabled !== true || !isRecord(channel.settings)) return false
  const settings = channel.settings
  const type = typeof channel.type === 'string' ? channel.type.trim().toLowerCase() : ''
  if (type === 'telegram' && telegramPollsInsteadOfWebhook(settings)) return false
  return !nonEmptySetting(settings, 'webhook_secret_token', 'webhookSecretToken')
    && !nonEmptySetting(settings, 'signing_secret', 'signingSecret')
}

function telegramPollsInsteadOfWebhook(settings: Readonly<Record<string, unknown>>): boolean {
  const transport = typeof settings.transport === 'string' ? settings.transport.trim().toLowerCase() : 'auto'
  const webhookUrl = nonEmptySetting(settings, 'webhook_url', 'webhookUrl')
  return transport === 'polling' || (transport === 'auto' && !webhookUrl)
}

function nonEmptySetting(settings: Readonly<Record<string, unknown>>, ...keys: readonly string[]): string {
  for (const key of keys) {
    const value = settings[key]
    if (typeof value === 'string' && value.trim()) return value.trim()
  }
  return ''
}

function numericSetting(value: unknown, fallback: number): number {
  if (typeof value === 'number' && Number.isInteger(value) && value >= 0 && value <= 65_535) return value
  if (typeof value === 'string' && /^\d+$/.test(value)) {
    const parsed = Number.parseInt(value, 10)
    return parsed <= 65_535 ? parsed : fallback
  }
  return fallback
}

function stringSetting(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function channelPreviewsEnabled(config: DaemonConfig, channelName: string): boolean {
  const configured = config.channels[channelName]
  if (!configured) return true
  const settings = configured.settings
  if (!isRecord(settings)) return true
  const value = settings.stream_previews ?? settings.streamPreviews
  if (typeof value === 'boolean') return value
  if (typeof value !== 'string') return true
  return !new Set(['0', 'false', 'no', 'off', '']).has(value.trim().toLowerCase())
}

function channelPreviewInterval(config: DaemonConfig, channelName: string): number {
  const configured = config.channels[channelName]
  if (!configured || !isRecord(configured.settings)) return 1_000
  const settings = configured.settings
  const milliseconds = positiveDecimal(settings.preview_interval_ms ?? settings.previewIntervalMs)
  if (milliseconds !== undefined) return Math.max(1, Math.round(milliseconds))
  const seconds = positiveDecimal(settings.preview_interval ?? settings.previewInterval)
  return seconds === undefined ? 1_000 : Math.max(1, Math.round(seconds * 1_000))
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function positiveDecimal(value: unknown): number | undefined {
  const parsed = typeof value === 'number'
    ? value
    : typeof value === 'string' && /^\d+(?:\.\d+)?$/.test(value) ? Number.parseFloat(value) : Number.NaN
  return Number.isFinite(parsed) && parsed > 0 ? parsed : undefined
}
