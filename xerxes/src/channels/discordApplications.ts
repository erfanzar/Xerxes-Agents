// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { postJson, providerUrl, putJson, type ChannelFetch } from './http.js'

const DISCORD_API_BASE = 'https://discord.com/api/v10/'
const APPLICATION_COMMAND_TYPE = 1
const CHANNEL_MESSAGE_RESPONSE_TYPE = 4
/** Discord message flag that makes an interaction response visible only to its invoker. */
export const DISCORD_EPHEMERAL_MESSAGE_FLAG = 1 << 6

export interface DiscordApplicationCommandOption {
  readonly description: string
  readonly name: string
  readonly required?: boolean
  /** Discord application-command option type. Xerxes' built-ins use string options (`3`). */
  readonly type: number
}

/** One command in Discord's bulk application-command replacement payload. */
export interface DiscordApplicationCommand {
  readonly description: string
  readonly name: string
  readonly options?: readonly DiscordApplicationCommandOption[]
  /** Discord's CHAT_INPUT command type. */
  readonly type?: typeof APPLICATION_COMMAND_TYPE
}

export interface DiscordApplicationCommandRequest {
  readonly applicationId: string
  readonly botToken: string
  readonly commands: readonly DiscordApplicationCommand[]
}

export interface DiscordGuildApplicationCommandRequest extends DiscordApplicationCommandRequest {
  readonly guildId: string
}

export interface DiscordInteractionAcknowledgement {
  readonly content: string
  readonly ephemeral?: boolean
  readonly interactionId: string
  readonly interactionToken: string
}

/** Explicit REST boundary for command registration and interaction acknowledgement. */
export interface DiscordApplicationRestPort {
  acknowledgeInteraction(request: DiscordInteractionAcknowledgement): Promise<void>
  replaceGlobalCommands(request: DiscordApplicationCommandRequest): Promise<void>
  replaceGuildCommands(request: DiscordGuildApplicationCommandRequest): Promise<void>
}

/**
 * Native Discord application-command REST client.
 *
 * Hosts must construct and inject this port; it does not discover a token or
 * application id from process state. Interaction callbacks intentionally use
 * Discord's interaction token rather than a Bot authorization header.
 */
export class FetchDiscordApplicationRestPort implements DiscordApplicationRestPort {
  private readonly apiBaseUrl: string
  private readonly fetchImplementation: ChannelFetch

  constructor(
    options: { readonly apiBaseUrl?: string; readonly fetchImplementation?: ChannelFetch } = {},
  ) {
    this.apiBaseUrl = options.apiBaseUrl ?? DISCORD_API_BASE
    this.fetchImplementation = options.fetchImplementation ?? fetch
  }

  async replaceGlobalCommands(request: DiscordApplicationCommandRequest): Promise<void> {
    await putJson(this.endpoint('applications', request.applicationId, 'commands'), {
      body: request.commands,
      fetchImplementation: this.fetchImplementation,
      headers: botAuthorization(request.botToken),
    })
  }

  async replaceGuildCommands(request: DiscordGuildApplicationCommandRequest): Promise<void> {
    await putJson(this.endpoint('applications', request.applicationId, 'guilds', request.guildId, 'commands'), {
      body: request.commands,
      fetchImplementation: this.fetchImplementation,
      headers: botAuthorization(request.botToken),
    })
  }

  async acknowledgeInteraction(request: DiscordInteractionAcknowledgement): Promise<void> {
    await postJson(this.endpoint('interactions', request.interactionId, request.interactionToken, 'callback'), {
      body: {
        type: CHANNEL_MESSAGE_RESPONSE_TYPE,
        data: {
          content: requiredText(request.content, 'Discord interaction response content'),
          ...(request.ephemeral ? { flags: DISCORD_EPHEMERAL_MESSAGE_FLAG } : {}),
        },
      },
      fetchImplementation: this.fetchImplementation,
    })
  }

  private endpoint(...segments: readonly string[]): string {
    const path = segments
      .map(segment => encodeURIComponent(requiredText(segment, 'Discord API path segment')))
      .join('/')
    return providerUrl(this.apiBaseUrl, path)
  }
}

function botAuthorization(token: string): Readonly<Record<string, string>> {
  return { Authorization: `Bot ${requiredText(token, 'Discord bot token')}` }
}

function requiredText(value: string, label: string): string {
  const text = value.trim()
  if (!text) throw new TypeError(`${label} is required`)
  return text
}
