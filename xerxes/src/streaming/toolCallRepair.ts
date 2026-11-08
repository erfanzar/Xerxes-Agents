// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { JsonObject, JsonValue, ToolCall } from '../types/toolCalls.js'

/**
 * A word a shell would pass through unchanged: no quotes, globs, expansions,
 * escapes, comments, operators or whitespace.
 */
const PLAIN_WORD = /^[A-Za-z0-9_@%+,./:=-]+$/

/**
 * Rewrite tool calls whose arguments have exactly one sensible reading but fail
 * validation as written. Small local models call `exec_command` the way they
 * would call a shell, and each rejected call costs a full round with the model.
 *
 * The loop runs this before hooks, permission and data-boundary checks, so every
 * check judges the command that will actually run. Anything ambiguous is left
 * alone for validation to reject as before.
 */
export function repairToolCall(call: ToolCall): ToolCall {
  if (call.function.name !== 'exec_command') return call
  const repaired = repairExecCommandInputs(call.function.arguments)
  return repaired === call.function.arguments ? call : { ...call, function: { ...call.function, arguments: repaired } }
}

/**
 * - `args` sent as a string that parses as a JSON array of strings becomes that array.
 * - A `cmd` made only of plain words (`wc -l`) becomes its first word, with the
 *   other words placed before any `args`: what a shell would run for that text.
 *   The first word must not be a `NAME=value` assignment.
 *
 * Returns the same object when nothing changes.
 */
export function repairExecCommandInputs(inputs: JsonObject): JsonObject {
  let next = inputs
  if (typeof next.args === 'string') {
    const parsed = stringArrayFromJson(next.args)
    if (parsed !== undefined) next = { ...next, args: parsed }
  }
  const command = next.cmd
  if (typeof command !== 'string') return next
  const words = command.trim().split(/\s+/).filter(Boolean)
  if (!words.length || !words.every(word => PLAIN_WORD.test(word)) || words[0]!.includes('=')) return next
  if (words.length === 1) return words[0] === command ? next : { ...next, cmd: words[0]! }
  const existing = next.args
  if (existing !== undefined && !isStringArray(existing)) return next
  return { ...next, cmd: words[0]!, args: [...words.slice(1), ...(existing ?? [])] }
}

function stringArrayFromJson(text: string): string[] | undefined {
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    return undefined
  }
  return isStringArray(parsed) ? [...parsed] : undefined
}

function isStringArray(value: unknown): value is readonly string[] {
  return Array.isArray(value) && value.every((item: JsonValue) => typeof item === 'string')
}
