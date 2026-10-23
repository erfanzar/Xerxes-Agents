// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ImageUrlContentPart, MessageContent, UserMessage, XerxesActor } from '../types/messages.js'

/**
 * Verified authorship of a steer queued through the Hub. Present only when the
 * daemon verified a `hub_auth` envelope; a steer without it renders as
 * `[steer from user]` exactly as before.
 */
export interface SteerAuthor {
  readonly display_name: string
  /** Verified identity; absent for a local (non-hub) caller, whose steer is still attributed by name. */
  readonly actor?: XerxesActor
  /** Daemon-assigned id of the durable queue entry this live steer mirrors. */
  readonly steer_id?: string
  readonly submission_id?: string
  readonly queue_sequence?: number
}

/**
 * One queued steer: plain text, or text with validated image parts (a
 * screenshot pasted while the agent works). Plain strings stay valid so
 * subagent steering and older callers keep their shape.
 */
export type SteerInput =
  | string
  | { readonly text: string; readonly images: readonly ImageUrlContentPart[]; readonly author?: SteerAuthor }

export function steerText(steer: SteerInput): string {
  return typeof steer === 'string' ? steer : steer.text
}

export function steerImages(steer: SteerInput): readonly ImageUrlContentPart[] {
  return typeof steer === 'string' ? [] : steer.images
}

export function steerAuthor(steer: SteerInput): SteerAuthor | undefined {
  return typeof steer === 'string' ? undefined : steer.author
}

/** A display name that can never close the `[steer from …]` marker or add lines. */
export function steerLabel(displayName: string): string {
  const cleaned = displayName.replace(/[[\]\r\n]+/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 60)
  return cleaned || 'participant'
}

/** `[steer from user…]` becomes `[steer from <name>…]` for an attributed steer. */
export function steerPrefix(prefix: string, steer: SteerInput): string {
  const author = steerAuthor(steer)
  return author ? prefix.replace('[steer from user', `[steer from ${steerLabel(author.display_name)}`) : prefix
}

/**
 * The user message a steer becomes: `<prefix>\n<text>` as a string when it has
 * no images, or a text part followed by its image parts. Undefined when the
 * steer carries nothing at all.
 */
export function steerContent(prefix: string, steer: SteerInput): MessageContent | undefined {
  const text = steerText(steer).trim()
  const images = steerImages(steer)
  if (!text && !images.length) return undefined
  const head = `${steerPrefix(prefix, steer)}\n${text}`.trimEnd()
  return images.length ? [{ type: 'text', text: head }, ...images] : head
}

/** The transcript message for a steer, carrying its verified author when it has one. */
export function steerMessage(prefix: string, steer: SteerInput): UserMessage | undefined {
  const content = steerContent(prefix, steer)
  if (content === undefined) return undefined
  const author = steerAuthor(steer)
  return { role: 'user', content, ...(author?.actor ? { xerxes_actor: author.actor } : {}) }
}
