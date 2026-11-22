// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isJsonObject, type JsonObject, type JsonSchema, type JsonValue, type ToolCall } from '../types/toolCalls.js'

/**
 * A word a shell would pass through unchanged: no quotes, globs, expansions,
 * escapes, comments, operators or whitespace.
 */
const PLAIN_WORD = /^[A-Za-z0-9_@%+,./:=-]+$/

const INTEGER_LITERAL = /^-?\d+$/
const NUMBER_LITERAL = /^-?\d+(\.\d+)?$/
const CODE_FENCE_PATTERN = /^```(?:json|jsonc)?\s*\r?\n?([\s\S]*?)\r?\n?```$/i

/**
 * Unambiguous cross-dialect parameter aliases models frequently mix up across
 * built-in and custom tool surfaces. Applied only when the schema declares the
 * canonical property, omits the alias property, and the call supplied the alias
 * without the canonical key.
 */
const UNAMBIGUOUS_PARAMETER_ALIASES: Readonly<Record<string, readonly string[]>> = Object.freeze({
  cmd: ['command'],
  command: ['cmd'],
  cwd: ['workdir', 'working_directory'],
  file_path: ['path', 'filepath'],
  path: ['file_path', 'filepath'],
  workdir: ['cwd', 'working_directory'],
})

/**
 * Rewrite tool calls whose arguments have exactly one sensible reading but fail
 * validation or misreport intent to pre-execution gates as written.
 *
 * The loop runs this before hooks, permission, description, and data-boundary
 * checks, so every check judges the exact typed arguments that will actually run.
 * Anything ambiguous is left alone for validation to reject as before.
 */
export function repairToolCall(call: ToolCall, schema?: JsonSchema): ToolCall {
  let repaired = schema ? repairDeclaredArguments(call.function.arguments, schema) : call.function.arguments
  if (call.function.name === 'exec_command') {
    repaired = repairExecCommandInputs(repaired)
  }
  return repaired === call.function.arguments ? call : { ...call, function: { ...call.function, arguments: repaired } }
}

/**
 * Repair schema-declared parameter aliases, literal scalar strings (`"true"`,
 * `"0"`, `"1.5"`), and JSON-serialized container strings (including Markdown
 * code fences and trailing commas outside string literals) ahead of permission
 * and policy gates.
 *
 * Returns the same object reference when nothing changes.
 */
export function repairDeclaredArguments(inputs: JsonObject, schema: JsonSchema | undefined): JsonObject {
  if (!schema || !isRecord(schema.properties)) return inputs
  const properties = schema.properties
  let next: JsonObject | undefined

  for (const [canonical, aliases] of Object.entries(UNAMBIGUOUS_PARAMETER_ALIASES)) {
    if (!Object.hasOwn(properties, canonical) || Object.hasOwn(inputs, canonical)) continue
    const presentAliases = aliases.filter(
      alias => Object.hasOwn(inputs, alias) && inputs[alias] !== undefined && !Object.hasOwn(properties, alias),
    )
    if (presentAliases.length !== 1) continue
    const alias = presentAliases[0]!
    next ??= { ...inputs }
    next[canonical] = next[alias]!
    delete next[alias]
  }

  const current = next ?? inputs
  for (const [key, rawValue] of Object.entries(current)) {
    const property = Object.hasOwn(properties, key) ? properties[key] : undefined
    if (!isRecord(property) || typeof property.type !== 'string') continue
    const coerced = coerceDeclaredValue(rawValue, property.type)
    if (!Object.is(coerced, rawValue)) {
      next ??= { ...current }
      next[key] = coerced
    }
  }

  return next ?? inputs
}

/**
 * - `args` sent as a string that parses as a JSON array of strings (including
 *   fenced or trailing-comma JSON) becomes that array.
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

/**
 * Parse a JSON container (`array` or `object`) from a model-emitted string,
 * tolerating outer Markdown code fences and trailing commas outside strings.
 */
export function parseRepairableJsonContainer(
  text: string,
  expectedType: 'array' | 'object',
): JsonValue | undefined {
  const unfenced = stripCodeFence(text.trim())
  const expectedOpen = expectedType === 'array' ? '[' : '{'
  const expectedClose = expectedType === 'array' ? ']' : '}'
  if (!unfenced.startsWith(expectedOpen) || !unfenced.endsWith(expectedClose)) return undefined
  const direct = tryParseJsonValue(unfenced)
  const parsed = direct !== undefined ? direct : tryParseJsonValue(stripTrailingJsonCommas(unfenced))
  if (parsed === undefined) return undefined
  if (expectedType === 'array') return Array.isArray(parsed) ? parsed : undefined
  return isJsonObject(parsed) ? parsed : undefined
}

function coerceDeclaredValue(value: JsonValue, expectedType: string): JsonValue {
  if (typeof value !== 'string') return value
  if (expectedType === 'boolean') {
    if (value === 'true') return true
    if (value === 'false') return false
    return value
  }
  if (expectedType === 'integer' && INTEGER_LITERAL.test(value)) {
    const parsed = Number(value)
    return Number.isSafeInteger(parsed) ? parsed : value
  }
  if (expectedType === 'number' && NUMBER_LITERAL.test(value)) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : value
  }
  if (expectedType === 'array' || expectedType === 'object') {
    return parseRepairableJsonContainer(value, expectedType) ?? value
  }
  return value
}

function stringArrayFromJson(text: string): string[] | undefined {
  const parsed = parseRepairableJsonContainer(text, 'array')
  return isStringArray(parsed) ? [...parsed] : undefined
}

function stripCodeFence(text: string): string {
  const match = CODE_FENCE_PATTERN.exec(text)
  return match?.[1]?.trim() ?? text
}

/** Remove `,` tokens that sit outside JSON string literals immediately before `]` or `}`. */
function stripTrailingJsonCommas(text: string): string {
  let result = ''
  let inString = false
  let escaped = false
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index]!
    if (inString) {
      result += char
      if (escaped) escaped = false
      else if (char === '\\') escaped = true
      else if (char === '"') inString = false
      continue
    }
    if (char === '"') {
      inString = true
      result += char
      continue
    }
    if (char === ',') {
      let lookahead = index + 1
      while (lookahead < text.length && /\s/.test(text[lookahead]!)) lookahead += 1
      const next = text[lookahead]
      if (next === ']' || next === '}') continue
    }
    result += char
  }
  return result
}

function tryParseJsonValue(text: string): JsonValue | undefined {
  try {
    const parsed: unknown = JSON.parse(text)
    return isJsonValue(parsed) ? parsed : undefined
  } catch {
    return undefined
  }
}

function isJsonValue(value: unknown): value is JsonValue {
  if (value === null || typeof value === 'boolean' || typeof value === 'number' || typeof value === 'string') return true
  if (Array.isArray(value)) return value.every(isJsonValue)
  return isJsonObject(value) && Object.values(value).every(isJsonValue)
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isStringArray(value: unknown): value is readonly string[] {
  return Array.isArray(value) && value.every((item: JsonValue) => typeof item === 'string')
}
