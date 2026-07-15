// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Direction of a platform-neutral channel message. */
export const MessageDirection = {
  INBOUND: 'inbound',
  OUTBOUND: 'outbound',
} as const

export type MessageDirection =
  (typeof MessageDirection)[keyof typeof MessageDirection]
export type ChannelAttachment = Readonly<Record<string, unknown>>
export type ChannelMetadata = Readonly<Record<string, unknown>>

/** One message moving between Xerxes and a messaging platform. */
export interface ChannelMessage {
  readonly attachments: readonly ChannelAttachment[]
  readonly channel: string
  readonly channelUserId?: string
  readonly direction: MessageDirection
  readonly messageId: string
  readonly metadata: ChannelMetadata
  readonly platformMessageId?: string
  readonly replyTo?: string
  readonly roomId?: string
  readonly text: string
  readonly timestamp: Date
  readonly userId?: string
}

/** Input accepted by {@link createChannelMessage}. */
export interface ChannelMessageInput {
  readonly attachments?: readonly ChannelAttachment[]
  readonly channel: string
  readonly channelUserId?: string
  readonly direction?: MessageDirection
  readonly messageId?: string
  readonly metadata?: ChannelMetadata
  readonly platformMessageId?: string
  readonly replyTo?: string
  readonly roomId?: string
  readonly text: string
  readonly timestamp?: Date
  readonly userId?: string
}

/**
 * Build a normalized message with Xerxes-owned defaults.
 *
 * The factory copies collection fields so an adapter cannot accidentally
 * share metadata or attachment arrays across messages.
 */
export function createChannelMessage(
  input: ChannelMessageInput,
): ChannelMessage {
  return {
    text: input.text,
    channel: input.channel,
    ...(input.userId === undefined ? {} : { userId: input.userId }),
    ...(input.channelUserId === undefined
      ? {}
      : { channelUserId: input.channelUserId }),
    ...(input.roomId === undefined ? {} : { roomId: input.roomId }),
    ...(input.replyTo === undefined ? {} : { replyTo: input.replyTo }),
    messageId: input.messageId ?? crypto.randomUUID(),
    ...(input.platformMessageId === undefined
      ? {}
      : { platformMessageId: input.platformMessageId }),
    attachments:
      input.attachments?.map((attachment) => ({ ...attachment })) ?? [],
    timestamp: input.timestamp ? new Date(input.timestamp) : new Date(),
    direction: input.direction ?? MessageDirection.INBOUND,
    metadata: { ...input.metadata },
  }
}
