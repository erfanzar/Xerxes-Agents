// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ImageUrlContentPart, MessageContent } from '../types/messages.js'

/**
 * One queued steer: plain text, or text with validated image parts (a
 * screenshot pasted while the agent works). Plain strings stay valid so
 * subagent steering and older callers keep their shape.
 */
export type SteerInput = string | { readonly text: string; readonly images: readonly ImageUrlContentPart[] }

export function steerText(steer: SteerInput): string {
  return typeof steer === 'string' ? steer : steer.text
}

export function steerImages(steer: SteerInput): readonly ImageUrlContentPart[] {
  return typeof steer === 'string' ? [] : steer.images
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
  const head = `${prefix}\n${text}`.trimEnd()
  return images.length ? [{ type: 'text', text: head }, ...images] : head
}
