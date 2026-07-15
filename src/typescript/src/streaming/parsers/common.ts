// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  isJsonObject,
  type JsonObject,
  type JsonValue,
} from '../../types/toolCalls.js'
import type { ParsedToolCall } from './types.js'

const TAGGED_PARAMETER_PATTERN =
  /<parameter=([A-Za-z0-9_.-]+)>\s*([\s\S]*?)\s*<\/parameter>/gi

/** Ensure runtime callers pass a decoded provider text fragment. */
export function assertParserText(text: string): void {
  if (typeof text !== 'string') {
    throw new TypeError('tool-call parser text must be a string')
  }
}

/**
 * Extract complete JSON object tool-call blocks delimited by literal tags.
 *
 * Parsing the JSON payload before testing the closing tag keeps strings that
 * happen to contain the delimiter from truncating a completed tool call.
 */
export function parseToolCallBlocks(
  text: string,
  openTag: string,
  closeTag: string,
): ParsedToolCall[] {
  assertParserText(text)
  if (!openTag || !closeTag) {
    return []
  }
  return parseTaggedJsonObjects(
    text,
    openTag,
    closeTag,
    normalizeTaggedToolCall,
  )
}

/** Parse the XML-like raw text format used by Hermes-style models. */
export function parseXmlToolCalls(text: string): ParsedToolCall[] {
  return parseToolCallBlocks(text, '<tool_call>', '</tool_call>')
}

/** Parse both Llama python-tag and function-name tool-call forms. */
export function parseLlamaToolCalls(text: string): ParsedToolCall[] {
  assertParserText(text)
  return [
    ...parseTaggedJsonObjects(
      text,
      '<|python_tag|>',
      '<|eom_id|>',
      normalizeLlamaPythonCall,
    ),
    ...parseLlamaFunctionCalls(text),
  ]
}

/** Decode one complete JSON object or array starting at an exact source position. */
export function decodeJsonCompositeAt(
  text: string,
  start: number,
): { readonly end: number; readonly value: JsonValue } | undefined {
  const end = findJsonCompositeEnd(text, start)
  if (end === undefined) {
    return undefined
  }
  try {
    return { end, value: JSON.parse(text.slice(start, end)) as JsonValue }
  } catch {
    return undefined
  }
}

/** Advance an index over Unicode whitespace. */
export function skipWhitespace(text: string, start: number): number {
  let position = start
  while (position < text.length && isWhitespace(text[position] ?? '')) {
    position += 1
  }
  return position
}

/** Mirror provider JSON IDs as strings, including numeric IDs. */
export function stringFromJson(value: JsonValue | undefined): string {
  return value === undefined ? '' : String(value)
}

function parseTaggedJsonObjects(
  text: string,
  openTag: string,
  closeTag: string,
  normalize: (payload: JsonObject) => ParsedToolCall | undefined,
): ParsedToolCall[] {
  const calls: ParsedToolCall[] = []
  let position = 0
  while (position < text.length) {
    const opening = text.indexOf(openTag, position)
    if (opening < 0) {
      break
    }
    const payloadStart = skipWhitespace(text, opening + openTag.length)
    if (text[payloadStart] !== '{') {
      position = opening + openTag.length
      continue
    }

    const decoded = decodeJsonCompositeAt(text, payloadStart)
    if (!decoded || !isJsonObject(decoded.value)) {
      position = opening + openTag.length
      continue
    }
    const closing = skipWhitespace(text, decoded.end)
    if (!text.startsWith(closeTag, closing)) {
      position = opening + openTag.length
      continue
    }

    const call = normalize(decoded.value)
    if (call) {
      calls.push(call)
    }
    position = closing + closeTag.length
  }
  return calls
}

function normalizeTaggedToolCall(
  payload: JsonObject,
): ParsedToolCall | undefined {
  const name = payload.name || payload.function || payload.tool
  if (!name) {
    return undefined
  }
  const argumentsValue =
    payload.arguments || payload.input || payload.parameters || {}
  return {
    name: String(name),
    arguments: isJsonObject(argumentsValue) ? argumentsValue : {},
    rawId: '',
  }
}

function normalizeLlamaPythonCall(
  payload: JsonObject,
): ParsedToolCall | undefined {
  const name = payload.name
  if (!name) {
    return undefined
  }
  const argumentsValue = payload.parameters || payload.arguments || {}
  return {
    name: String(name),
    arguments: isJsonObject(argumentsValue) ? argumentsValue : {},
    rawId: '',
  }
}

function parseLlamaFunctionCalls(text: string): ParsedToolCall[] {
  const calls: ParsedToolCall[] = []
  const openTag = '<function='
  const closeTag = '</function>'
  let position = 0
  while (position < text.length) {
    const opening = text.indexOf(openTag, position)
    if (opening < 0) {
      break
    }
    const nameStart = opening + openTag.length
    const nameEnd = text.indexOf('>', nameStart)
    if (nameEnd < 0) {
      position = nameStart
      continue
    }
    const payloadStart = skipWhitespace(text, nameEnd + 1)
    if (text[payloadStart] === '{') {
      const decoded = decodeJsonCompositeAt(text, payloadStart)
      if (!decoded || !isJsonObject(decoded.value)) {
        position = nameStart
        continue
      }
      const closing = skipWhitespace(text, decoded.end)
      if (!text.startsWith(closeTag, closing)) {
        position = nameStart
        continue
      }

      calls.push({
        name: text.slice(nameStart, nameEnd).trim(),
        arguments: decoded.value,
        rawId: '',
      })
      position = closing + closeTag.length
      continue
    }

    const closing = text.indexOf(closeTag, payloadStart)
    if (closing < 0) {
      position = nameEnd + 1
      continue
    }
    const arguments_ = parseTaggedParameters(text.slice(payloadStart, closing))
    if (!arguments_) {
      position = nameEnd + 1
      continue
    }
    calls.push({
      name: text.slice(nameStart, nameEnd).trim(),
      arguments: arguments_,
      rawId: '',
    })
    position = closing + closeTag.length
  }
  return calls
}

/** Parse Claude-style parameter tags while retaining their JSON scalar values. */
function parseTaggedParameters(body: string): JsonObject | undefined {
  const arguments_: JsonObject = {}
  let found = false
  for (const match of body.matchAll(TAGGED_PARAMETER_PATTERN)) {
    const name = match[1]
    const rawValue = match[2]?.trim() ?? ''
    if (!name || !rawValue) continue
    arguments_[name] = parseTaggedParameterValue(rawValue)
    found = true
  }
  return found ? arguments_ : undefined
}

function parseTaggedParameterValue(value: string): JsonValue {
  const shouldDecode =
    value.startsWith("'") ||
    value.startsWith('"') ||
    value.startsWith('{') ||
    value.startsWith('[') ||
    value.startsWith('-') ||
    value === 'true' ||
    value === 'false' ||
    value === 'null' ||
    /^\d+$/.test(value)
  if (!shouldDecode) return value
  try {
    return JSON.parse(value) as JsonValue
  } catch {
    return value
  }
}

function findJsonCompositeEnd(text: string, start: number): number | undefined {
  const opening = text[start]
  if (opening !== '{' && opening !== '[') {
    return undefined
  }

  const closers: string[] = [opening === '{' ? '}' : ']']
  let escaped = false
  let inString = false
  for (let position = start + 1; position < text.length; position += 1) {
    const character = text[position] ?? ''
    if (inString) {
      if (escaped) {
        escaped = false
      } else if (character === '\\') {
        escaped = true
      } else if (character === '"') {
        inString = false
      }
      continue
    }

    if (character === '"') {
      inString = true
    } else if (character === '{') {
      closers.push('}')
    } else if (character === '[') {
      closers.push(']')
    } else if (character === '}' || character === ']') {
      if (closers.at(-1) !== character) {
        return undefined
      }
      closers.pop()
      if (!closers.length) {
        return position + 1
      }
    }
  }
  return undefined
}

function isWhitespace(value: string): boolean {
  return value.trim() === ''
}
