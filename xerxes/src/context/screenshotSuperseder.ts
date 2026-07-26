// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ChatMessage } from '../types/messages.js'

/**
 * Multimodal tool results (computer_use screenshots) are JSON-stringified
 * into tool-message content with `_multimodal` as the first key. The prefix
 * check keeps the per-append sweep O(1) for ordinary tool results.
 */
export const SCREENSHOT_RESULT_PREFIX = '{"_multimodal":true'
export const SCREENSHOT_OMITTED_PREFIX = '[screenshot omitted:'

const CAPTURE_DIMENSIONS_PATTERN = /Screen capture:\s*(\d+)\s*x\s*(\d+)/
const IMAGE_BASE64_PATTERN = /;base64,([A-Za-z0-9+/=]+)/

/** True when a tool-message content string carries an inline screenshot payload. */
export function isScreenshotToolResult(content: unknown): content is string {
  return typeof content === 'string' && content.startsWith(SCREENSHOT_RESULT_PREFIX)
}

/**
 * Replace every superseded screenshot tool result with a compact marker so a
 * capture-heavy session keeps only the latest capture inline.
 *
 * Only the `content` field of earlier tool messages is rewritten; message
 * order, roles, names, and tool_call_id pairing are untouched, so
 * tool-call-sequence repair invariants are preserved. The latest screenshot
 * stays intact for the next provider round. Returns the number of results
 * superseded.
 */
export function supersedeScreenshotToolResults(messages: ChatMessage[]): number {
  const screenshotIndexes: number[] = []
  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index]
    if (message?.role === 'tool' && isScreenshotToolResult(message.content)) {
      screenshotIndexes.push(index)
    }
  }
  let superseded = 0
  for (const index of screenshotIndexes.slice(0, -1)) {
    const message = messages[index]
    if (message?.role !== 'tool') continue
    messages[index] = { ...message, content: omittedScreenshotMarker(message.content) }
    superseded += 1
  }
  return superseded
}

/** Compact stand-in naming the dropped payload's byte size and dimensions. */
export function omittedScreenshotMarker(content: string): string {
  if (content.startsWith(SCREENSHOT_OMITTED_PREFIX)) return content
  let bytes: number | undefined
  let dimensions: string | undefined
  try {
    const parsed: unknown = JSON.parse(content)
    if (typeof parsed === 'object' && parsed !== null) {
      const record = parsed as Record<string, unknown>
      const summary = typeof record['text_summary'] === 'string' ? record['text_summary'] : ''
      const match = CAPTURE_DIMENSIONS_PATTERN.exec(summary)
      if (match?.[1] && match[2]) dimensions = `${match[1]}x${match[2]}`
      const parts = record['content']
      if (Array.isArray(parts)) {
        for (const part of parts) {
          if (typeof part !== 'object' || part === null) continue
          const url = (part as Record<string, unknown>)['image_url']
          if (typeof url !== 'object' || url === null) continue
          const href = (url as Record<string, unknown>)['url']
          if (typeof href !== 'string') continue
          const base64 = IMAGE_BASE64_PATTERN.exec(href)?.[1]
          if (base64) bytes = Math.floor((base64.length * 3) / 4)
        }
      }
    }
  } catch {
    // A malformed payload still gets a marker; details are best-effort.
  }
  const size = bytes === undefined ? 'unknown bytes' : `${bytes} bytes`
  const shape = dimensions ?? 'unknown dimensions'
  return `[screenshot omitted: ${size}, ${shape}]`
}
