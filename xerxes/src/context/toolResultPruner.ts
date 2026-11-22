// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { copyTurnOutcome } from '../types/turnOutcome.js'
import {
  compressFailureSalientText,
  fitSurrogateSafeText,
  hasSalientFailureLines,
  isCommandOrDiagnosticTool,
  safeSlicePrefix,
} from './failureSalientPreview.js'
import { omittedScreenshotMarker, SCREENSHOT_RESULT_PREFIX } from './screenshotSuperseder.js'

export const DEFAULT_TOOL_RESULT_MAX_CHARS = 4_000
export const DEFAULT_TOOL_RESULT_HEAD_LINES = 40
export const DEFAULT_TOOL_RESULT_TAIL_LINES = 20

export interface ToolResultPruneOptions {
  readonly headLines?: number
  readonly maxChars?: number
  readonly tailLines?: number
  readonly toolName?: string
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
  const maxChars = Math.max(1, Math.trunc(options.maxChars ?? DEFAULT_TOOL_RESULT_MAX_CHARS))
  if (content.length <= maxChars) {
    return { content, pruned: false }
  }
  if (content.startsWith(SCREENSHOT_RESULT_PREFIX)) {
    return { content: omittedScreenshotMarker(content), pruned: true }
  }
  if (isBinaryBlob(content)) {
    const marker = `[${content.length} bytes of binary content elided by pre-pruning]`
    return { content: marker.length <= maxChars ? marker : safeSlicePrefix(marker, maxChars), pruned: true }
  }
  return {
    content: truncateText(
      content,
      options.headLines ?? DEFAULT_TOOL_RESULT_HEAD_LINES,
      options.tailLines ?? DEFAULT_TOOL_RESULT_TAIL_LINES,
      maxChars,
      options.toolName,
    ),
    pruned: true,
  }
}

/** Never mutates input messages; recent messages remain intact for the next turn. */
export function pruneToolMessages<T extends Record<string, unknown>>(
  messages: readonly T[],
  options: MessagePruneOptions = {},
): { readonly messages: T[]; readonly prunedCount: number } {
  const protectedStart = Math.max(0, messages.length - (options.protectLast ?? 4))
  const callToolNames = new Map<string, string>()
  for (const message of messages) {
    if (!Array.isArray(message.tool_calls)) continue
    for (const call of message.tool_calls) {
      if (typeof call !== 'object' || call === null) continue
      const record = call as Record<string, unknown>
      const id = typeof record.id === 'string' ? record.id : ''
      const fn = typeof record.function === 'object' && record.function !== null
        ? record.function as Record<string, unknown>
        : undefined
      const name = typeof fn?.name === 'string'
        ? fn.name
        : typeof record.name === 'string' ? record.name : ''
      if (id && name) callToolNames.set(id, name)
    }
  }

  let prunedCount = 0
  const prunedMessages = messages.map((message, index) => {
    if (message.role !== 'tool' || index >= protectedStart) {
      return message
    }
    const inferredName = typeof message.name === 'string'
      ? message.name
      : typeof message.tool_call_id === 'string'
        ? callToolNames.get(message.tool_call_id)
        : undefined
    const messageOptions: ToolResultPruneOptions = options.toolName !== undefined || inferredName === undefined
      ? options
      : { ...options, toolName: inferredName }
    const pruned = pruneToolResult(message.content, messageOptions)
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

function truncateText(
  content: string,
  headLines: number,
  tailLines: number,
  maxChars: number,
  toolName: string | undefined,
): string {
  const safeHeadLines = Math.max(0, Math.trunc(headLines))
  const safeTailLines = Math.max(0, Math.trunc(tailLines))
  const lines = content.split(/\r?\n/)
  const strictGuard = !isCommandOrDiagnosticTool(toolName)

  if (lines.length > safeHeadLines + safeTailLines) {
    const middleSlice = lines.slice(safeHeadLines, safeTailLines > 0 ? lines.length - safeTailLines : lines.length)
    const middleHasFailures = hasSalientFailureLines(
      middleSlice,
      strictGuard ? { strictSourceCodeGuard: true } : {},
    )
    if (middleHasFailures) {
      return compressFailureSalientText(content, {
        maxChars,
        headLines: Math.min(safeHeadLines, 16),
        tailLines: Math.min(safeTailLines, 16),
        contextLines: 2,
        omissionLabel: 'lines omitted by pre-pruning',
        annotateHeadAndTailRanges: false,
        ...(strictGuard ? { strictSourceCodeGuard: true } : {}),
      })
    }

    const omitted = lines.length - safeHeadLines - safeTailLines
    const head = safeHeadLines > 0 ? lines.slice(0, safeHeadLines).join('\n') : ''
    // slice(-0) === slice(0), so guard explicitly: a zero tail must not re-append every line.
    const tail = safeTailLines > 0 ? lines.slice(-safeTailLines).join('\n') : ''
    const plain = [head, `[... ${omitted} lines omitted by pre-pruning ...]`, tail]
      .filter(part => part.length > 0)
      .join('\n\n')
    if (plain.length <= maxChars) return plain

    return compressFailureSalientText(content, {
      maxChars,
      headLines: safeHeadLines,
      tailLines: safeTailLines,
      contextLines: 2,
      omissionLabel: 'lines omitted by pre-pruning',
      annotateHeadAndTailRanges: false,
      ...(strictGuard ? { strictSourceCodeGuard: true } : {}),
    })
  }

  if (hasSalientFailureLines(lines, strictGuard ? { strictSourceCodeGuard: true } : {})) {
    return compressFailureSalientText(content, {
      maxChars,
      headLines: Math.min(safeHeadLines, 10),
      tailLines: Math.min(safeTailLines, 12),
      contextLines: 2,
      omissionLabel: 'lines omitted by pre-pruning',
      annotateHeadAndTailRanges: false,
      ...(strictGuard ? { strictSourceCodeGuard: true } : {}),
    })
  }

  return fitSurrogateSafeText(
    content,
    maxChars,
    omitted => `\n\n[... ${omitted} chars omitted by pre-pruning ...]\n\n`,
  )
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
