// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { copyTurnOutcome } from '../types/turnOutcome.js'

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
    return copyTurnOutcome(message, { ...message, content: pruned.content } as T)
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

export interface ToolResultShedOptions<T> {
  /** Stop once the conversation is at or under this many tokens. */
  readonly budgetTokens: number
  /** The newest tool results kept verbatim: the next round still reads them. */
  readonly keepRecent: number
  readonly count: (messages: readonly T[]) => number
}

/**
 * Replace the oldest tool results with a one-line note until the conversation
 * fits. A single round can be larger than any window — one assistant message
 * carried 1,176 calls and 1.25M tokens of results — and such a round can be
 * neither summarized apart from its calls nor trimmed small enough per
 * result. Each call keeps its result message, so pairing stays valid; only
 * the oldest outputs lose their text, and user and assistant messages are
 * never touched.
 */
export function shedToolResults<T extends Record<string, unknown>>(
  messages: readonly T[],
  options: ToolResultShedOptions<T>,
): { readonly messages: T[]; readonly shedCount: number } {
  const output = [...messages]
  const toolIndexes = output.flatMap((message, index) => message.role === 'tool' ? [index] : [])
  const sheddable = toolIndexes.slice(0, Math.max(0, toolIndexes.length - options.keepRecent))
  let total = options.count(output)
  let shedCount = 0
  for (const index of sheddable) {
    if (total <= options.budgetTokens) break
    const message = output[index]!
    const text = typeof message.content === 'string' ? message.content : JSON.stringify(message.content ?? '')
    if (text.startsWith(SHED_PREFIX)) continue
    const replacement = copyTurnOutcome(message, { ...message, content: `${SHED_PREFIX} (${text.length.toLocaleString('en-US')} characters)]` } as T)
    total -= options.count([message]) - options.count([replacement])
    output[index] = replacement
    shedCount += 1
  }
  return { messages: output, shedCount }
}

const SHED_PREFIX = '[result omitted to fit the context window'
