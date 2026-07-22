// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export const DEFAULT_TOOL_RESULT_MAX_CHARS = 4_000
export const DEFAULT_TOOL_RESULT_HEAD_LINES = 40
export const DEFAULT_TOOL_RESULT_TAIL_LINES = 20

export interface ToolResultPruneOptions {
  readonly headLines?: number
  readonly maxChars?: number
  readonly tailLines?: number
}

export interface MessagePruneOptions extends ToolResultPruneOptions {
  readonly protectLast?: number
}

export interface PrunedToolResult<T> {
  readonly content: T | string
  readonly pruned: boolean
}

/** Cheaply shrink oversized string tool output before model-backed compaction. */
export function pruneToolResult<T>(content: T, options: ToolResultPruneOptions = {}): PrunedToolResult<T> {
  if (typeof content !== 'string') {
    return { content, pruned: false }
  }
  const maxChars = options.maxChars ?? DEFAULT_TOOL_RESULT_MAX_CHARS
  if (content.length <= maxChars) {
    return { content, pruned: false }
  }
  if (isBinaryBlob(content)) {
    return { content: `[${content.length} bytes of binary content elided by pre-pruning]`, pruned: true }
  }
  return {
    content: truncateText(content, options.headLines ?? DEFAULT_TOOL_RESULT_HEAD_LINES, options.tailLines ?? DEFAULT_TOOL_RESULT_TAIL_LINES, maxChars),
    pruned: true,
  }
}

/** Never mutates input messages; recent messages remain intact for the next turn. */
export function pruneToolMessages<T extends Record<string, unknown>>(
  messages: readonly T[],
  options: MessagePruneOptions = {},
): { readonly messages: T[]; readonly prunedCount: number } {
  const protectedStart = Math.max(0, messages.length - (options.protectLast ?? 4))
  let prunedCount = 0
  const prunedMessages = messages.map((message, index) => {
    if (message.role !== 'tool' || index >= protectedStart) {
      return message
    }
    const pruned = pruneToolResult(message.content, options)
    if (!pruned.pruned) {
      return message
    }
    prunedCount += 1
    return { ...message, content: pruned.content } as T
  })
  return { messages: prunedMessages, prunedCount }
}

function isBinaryBlob(content: string): boolean {
  if (!content) {
    return false
  }
  const sample = content.slice(0, 1_024)
  let nonPrintable = 0
  for (const character of sample) {
    if (!/^[\p{L}\p{N}\p{P}\p{S}\p{Z}\n\r\t]$/u.test(character)) {
      nonPrintable += 1
    }
  }
  return nonPrintable > sample.length * 0.3
}

function truncateText(content: string, headLines: number, tailLines: number, maxChars: number): string {
  const lines = content.split(/\r?\n/)
  if (lines.length > headLines + tailLines) {
    const omitted = lines.length - headLines - tailLines
    const head = headLines > 0 ? lines.slice(0, headLines).join('\n') : ''
    // slice(-0) === slice(0), so guard explicitly: a zero tail must not re-append every line.
    const tail = tailLines > 0 ? lines.slice(-tailLines).join('\n') : ''
    return [head, `[... ${omitted} lines omitted by pre-pruning ...]`, tail].filter(part => part.length > 0).join('\n\n')
  }
  const headCharacters = Math.max(1, Math.floor(maxChars / 2))
  const tailCharacters = Math.max(1, maxChars - headCharacters)
  const omitted = content.length - headCharacters - tailCharacters
  return `${content.slice(0, headCharacters)}\n\n[... ${omitted} chars omitted by pre-pruning ...]\n\n${content.slice(-tailCharacters)}`
}
