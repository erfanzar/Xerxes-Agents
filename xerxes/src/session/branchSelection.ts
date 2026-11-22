// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isCompactionSummaryMessage } from '../context/compressor.js'

/** The message prefix that can be used as the history of a historical branch. */
export interface BranchTurnSelection {
  /** Number of transcript messages through the selected completed turn. */
  readonly messageCount: number
  /** Number of retained user turns in that prefix. */
  readonly turnCount: number
}

function isStandaloneUserTurn(message: Readonly<Record<string, unknown>> | undefined): boolean {
  return message?.role === 'user' && !isCompactionSummaryMessage(message) && message.origin !== 'harness'
}

/**
 * Select a completed, retained user turn as the end of a historical branch.
 *
 * Selection is based on message roles and tool-call IDs. It does not inspect
 * assistant text, mutate messages, or construct a replacement transcript.
 * Messages after the next user turn are outside the selected prefix and are
 * deliberately not validated, since this helper cannot reconstruct history
 * that was compacted away.
 */
export function selectBranchTurn(
  messages: readonly Readonly<Record<string, unknown>>[],
  turn: number,
): BranchTurnSelection {
  if (!Number.isSafeInteger(turn) || turn < 1) {
    throw new RangeError('turn must be a positive 1-based integer')
  }

  const userTurnStarts: number[] = []
  for (let index = 0; index < messages.length; index += 1) {
    if (isStandaloneUserTurn(messages[index])) userTurnStarts.push(index)
  }
  if (turn > userTurnStarts.length) {
    throw new RangeError(`retained user turn ${turn} is out of range`)
  }

  const selectedStart = userTurnStarts[turn - 1]
  if (selectedStart === undefined) {
    throw new RangeError(`retained user turn ${turn} is out of range`)
  }
  let messageCount = messages.length
  for (let index = selectedStart + 1; index < messages.length; index += 1) {
    const candidate = messages[index]
    if (isStandaloneUserTurn(candidate) || isCompactionSummaryMessage(candidate)) {
      messageCount = index
      break
    }
  }
  validatePrefix(messages, selectedStart, messageCount, turn)
  return { messageCount, turnCount: turn }
}

function validatePrefix(
  messages: readonly Readonly<Record<string, unknown>>[],
  selectedStart: number,
  messageCount: number,
  selectedTurn: number,
): void {
  let currentTurn = 0
  let turnOpen = false
  let postSummaryTail = false
  let sawAssistant = false
  let sawFinalAssistant = false
  let pendingToolCalls = new Set<string>()

  const finishSegment = (): void => {
    if (turnOpen) {
      if (pendingToolCalls.size > 0 || !sawAssistant || !sawFinalAssistant) {
        throw incompleteTurn(selectedTurn, currentTurn)
      }
    } else if (postSummaryTail) {
      if (pendingToolCalls.size > 0 || (sawAssistant && !sawFinalAssistant)) {
        throw incompleteTurn(selectedTurn, currentTurn || 1)
      }
    } else {
      return
    }
    turnOpen = false
    postSummaryTail = false
    sawAssistant = false
    sawFinalAssistant = false
    pendingToolCalls = new Set<string>()
  }

  for (let index = 0; index < messageCount; index += 1) {
    const message = messages[index]
    if (message === undefined) {
      throw new Error(`message ${index} is missing`)
    }
    const role = message.role

    // System messages are transcript preamble and do not open or close a turn,
    // though they cannot appear mid-tool-round.
    if (role === 'system') {
      if (pendingToolCalls.size > 0) {
        throw incompleteTurn(selectedTurn, currentTurn || 1)
      }
      continue
    }

    // Compaction summaries close any preceding protected head turn (verifying
    // it was complete) and open a post-summary tail segment so tail cuts that
    // CompactionProvisioner starts at `assistant` / `tool` / `origin: 'harness'`
    // remain valid ahead of the next standalone user turn.
    if (isCompactionSummaryMessage(message)) {
      finishSegment()
      postSummaryTail = true
      continue
    }

    if (role === 'user') {
      if (message.origin === 'harness') {
        if (!turnOpen && !postSummaryTail) {
          throw new Error(`harness message ${index} precedes the first user turn`)
        }
        if (pendingToolCalls.size > 0) {
          throw incompleteTurn(selectedTurn, currentTurn || 1)
        }
        continue
      }
      finishSegment()
      currentTurn += 1
      turnOpen = true
      continue
    }

    if (role === 'assistant') {
      if (!turnOpen && !postSummaryTail) {
        throw new Error(`assistant message ${index} precedes the first user turn`)
      }
      if (pendingToolCalls.size > 0) {
        throw incompleteTurn(selectedTurn, currentTurn || 1)
      }
      const toolCallIds = assistantToolCallIds(message, index, selectedTurn, currentTurn || 1)
      sawAssistant = true
      if (toolCallIds.length === 0) {
        sawFinalAssistant = true
      } else {
        sawFinalAssistant = false
        pendingToolCalls = new Set(toolCallIds)
      }
      continue
    }

    if (role === 'tool') {
      if (!turnOpen && !postSummaryTail) {
        throw new Error(`tool message ${index} precedes the first user turn`)
      }
      const toolCallId = message.tool_call_id
      if (typeof toolCallId !== 'string' || toolCallId.trim() === '' || !pendingToolCalls.delete(toolCallId)) {
        throw incompleteTurn(selectedTurn, currentTurn || 1)
      }
      continue
    }

    throw new Error(`message ${index} has unsupported role`)
  }

  // The selected prefix always contains the selected user turn. A malformed
  // preamble or an earlier incomplete turn is rejected before the branch can
  // copy it, and the selected turn is checked by the same rule.
  if (currentTurn === 0 || !turnOpen || selectedStart >= messageCount) {
    throw new RangeError(`retained user turn ${selectedTurn} is out of range`)
  }
  finishSegment()
}

function assistantToolCallIds(
  message: Readonly<Record<string, unknown>>,
  index: number,
  selectedTurn: number,
  currentTurn: number,
): string[] {
  const rawToolCalls = message.tool_calls
  if (rawToolCalls === undefined) return []
  if (!Array.isArray(rawToolCalls)) {
    throw incompleteTurn(selectedTurn, currentTurn, `assistant message ${index} has malformed tool_calls`)
  }

  const ids: string[] = []
  const seen = new Set<string>()
  for (const rawToolCall of rawToolCalls) {
    if (!isRecord(rawToolCall) || typeof rawToolCall.id !== 'string' || rawToolCall.id.trim() === '') {
      throw incompleteTurn(selectedTurn, currentTurn, `assistant message ${index} has a tool call without an ID`)
    }
    if (seen.has(rawToolCall.id)) {
      throw incompleteTurn(selectedTurn, currentTurn, `assistant message ${index} repeats a tool call ID`)
    }
    seen.add(rawToolCall.id)
    ids.push(rawToolCall.id)
  }
  return ids
}

function incompleteTurn(selectedTurn: number, currentTurn: number, detail = 'tool activity or the final response is incomplete'): Error {
  return new Error(`retained user turn ${selectedTurn} is incomplete (turn ${currentTurn}: ${detail})`)
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
