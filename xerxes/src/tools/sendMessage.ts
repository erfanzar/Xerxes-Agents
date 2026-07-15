// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ToolRegistry, type ToolExecutionContext } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import { optionalString, optionalStringArray, requiredString } from './inputs.js'

export interface OutboundMessagePayload {
  readonly files: readonly string[]
  readonly replyTo?: string
  readonly text: string
}

export type OutboundMessageHandler = (
  platform: string,
  recipient: string,
  payload: OutboundMessagePayload,
  context?: ToolExecutionContext,
) => JsonObject | Promise<JsonObject>

/** Injectable registry shared by channel adapters and the send_message tool. */
export class OutboundMessageRegistry {
  private readonly handlers = new Map<string, OutboundMessageHandler>()

  register(platform: string, handler: OutboundMessageHandler): void {
    const normalized = normalizePlatform(platform)
    this.handlers.set(normalized, handler)
  }

  registeredPlatforms(): string[] {
    return [...this.handlers.keys()].sort((left, right) => left.localeCompare(right))
  }

  async send(
    platform: string,
    recipient: string,
    payload: OutboundMessagePayload,
    context?: ToolExecutionContext,
  ): Promise<JsonObject> {
    const normalized = normalizePlatform(platform)
    const handler = this.handlers.get(normalized)
    if (!handler) return { ok: false, error: 'unknown platform: ' + platform }
    return handler(normalized, recipient, payload, context)
  }
}

export const defaultOutboundMessageRegistry = new OutboundMessageRegistry()

/** Register an outbound platform callback on the process-local default dispatcher. */
export function registerOutboundPlatform(platform: string, handler: OutboundMessageHandler): void {
  defaultOutboundMessageRegistry.register(platform, handler)
}

/** Route a text and/or attachment payload through a configured channel callback. */
export async function sendMessage(
  options: {
    readonly files?: readonly string[]
    readonly platform: string
    readonly recipient: string
    readonly replyTo?: string
    readonly text?: string
  },
  registry: OutboundMessageRegistry = defaultOutboundMessageRegistry,
  context?: ToolExecutionContext,
): Promise<JsonObject> {
  const platform = options.platform.trim()
  const recipient = options.recipient.trim()
  const text = options.text ?? ''
  const files = [...(options.files ?? [])]
  if (!platform) return { ok: false, error: 'platform required' }
  if (!recipient) return { ok: false, error: 'recipient required' }
  if (!text && files.length === 0) return { ok: false, error: 'text or files required' }
  const payload: { files: readonly string[]; replyTo?: string; text: string } = { files, text }
  const replyTo = options.replyTo?.trim()
  if (replyTo) payload.replyTo = replyTo
  return registry.send(platform, recipient, payload, context)
}

export const SEND_MESSAGE_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'send_message',
    description: 'Send a text message and/or attachment path through a configured channel platform.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        platform: { type: 'string' },
        recipient: { type: 'string' },
        text: { type: 'string', default: '' },
        files: { type: 'array', items: { type: 'string' } },
        reply_to: { type: 'string' },
      },
      required: ['platform', 'recipient'],
    },
  },
}

export interface SendMessageToolOptions {
  readonly registry?: OutboundMessageRegistry
}

/** Register send_message only when a host has deliberately configured channel handlers. */
export function registerSendMessageTool(registry: ToolRegistry, options: SendMessageToolOptions = {}): void {
  const dispatcher = options.registry ?? defaultOutboundMessageRegistry
  registry.register(SEND_MESSAGE_DEFINITION, (inputs, context) => {
    const replyTo = optionalString(inputs, 'reply_to')
    return sendMessage({
      platform: requiredString(inputs, 'platform'),
      recipient: requiredString(inputs, 'recipient'),
      text: optionalString(inputs, 'text') ?? '',
      files: optionalStringArray(inputs, 'files'),
      ...(replyTo === undefined ? {} : { replyTo }),
    }, dispatcher, context)
  })
}

function normalizePlatform(platform: string): string {
  const normalized = platform.trim().toLowerCase()
  if (!normalized) throw new Error('platform required')
  return normalized
}
