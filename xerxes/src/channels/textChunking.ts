// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Split long text into provider-safe chunks without producing empty pieces.
 *
 * Shared by channel adapters whose providers reject oversized messages
 * (Discord, Telegram). Newline boundaries below the limit are preferred so
 * Markdown and code blocks stay readable; every returned piece is at most
 * `limit` characters.
 */
export function chunkText(text: string, limit: number): string[] {
  if (!Number.isInteger(limit) || limit < 1) {
    throw new RangeError('message chunk limit must be a positive safe integer')
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
