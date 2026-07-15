// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isJsonObject, type JsonObject } from '../types/toolCalls.js'

export const ASSISTANT_TOOL_CALLS_MARKER = 'ASSISTANT_TOOL_CALLS'

export interface MarkerToolCall {
  readonly id: string
  readonly input: JsonObject
  readonly name: string
}

export interface MarkerExtraction {
  readonly text: string
  readonly toolCalls: readonly MarkerToolCall[]
}

/** Remove provider-only tool markers from visible text and normalize their calls. */
export function extractAssistantToolCallMarkers(text: string, idPrefix = 'call_marker'): MarkerExtraction {
  const calls: MarkerToolCall[] = []
  const spans: Array<readonly [number, number]> = []
  let searchFrom = 0
  while (true) {
    const markerStart = text.indexOf(ASSISTANT_TOOL_CALLS_MARKER, searchFrom)
    if (markerStart < 0) break
    let payloadStart = markerStart + ASSISTANT_TOOL_CALLS_MARKER.length
    payloadStart = skipWhitespaceAndOptionalColon(text, payloadStart)
    const decoded = decodeJsonPrefix(text, payloadStart)
    if (!decoded) {
      searchFrom = payloadStart
      continue
    }
    const markerCalls = normalizeMarkerPayload(decoded.value, idPrefix, calls.length)
    if (markerCalls.length) {
      calls.push(...markerCalls)
      spans.push([markerStart, decoded.end])
    }
    searchFrom = decoded.end
  }

  const invokeExpression = /<invoke\s+name=(["'])(.*?)\1[^>]*>([\s\S]*?)<\/invoke>/gi
  for (const match of text.matchAll(invokeExpression)) {
    const full = match[0]
    const name = match[2] ?? ''
    const body = match[3] ?? ''
    const start = match.index
    if (start === undefined) continue
    const call = normalizeInvokeBlock(name, body, idPrefix + '_' + calls.length)
    if (!call) continue
    calls.push(call)
    spans.push([start, start + full.length])
  }

  return {
    text: stripProviderToolContext(removeSpans(text, spans)).trim(),
    toolCalls: calls,
  }
}

/** Remove provider-only assistant markers without exposing their parsed calls. */
export function stripAssistantToolCallMarkers(text: string): string {
  return extractAssistantToolCallMarkers(text).text
}

function normalizeMarkerPayload(value: unknown, idPrefix: string, startIndex: number): MarkerToolCall[] {
  const items = Array.isArray(value) ? value : [value]
  return items.flatMap((item, index) => {
    const call = normalizeMarkerCall(item, idPrefix + '_' + (startIndex + index))
    return call ? [call] : []
  })
}

function normalizeMarkerCall(value: unknown, fallbackId: string): MarkerToolCall | undefined {
  if (!isJsonObject(value)) return undefined
  const functionValue = isJsonObject(value.function) ? value.function : undefined
  const name = stringValue(functionValue?.name) || stringValue(value.name) || stringValue(value.tool_name)
  if (!name) return undefined
  const rawInput = functionValue?.arguments ?? value.input ?? value.arguments
  return {
    id: stringValue(value.id) || stringValue(value.tool_call_id) || fallbackId,
    name,
    input: normalizeInput(rawInput),
  }
}

function normalizeInvokeBlock(name: string, body: string, fallbackId: string): MarkerToolCall | undefined {
  const normalizedName = name.trim()
  if (!normalizedName) return undefined
  const input: JsonObject = {}
  const parameterExpression = /<parameter\s+name=(["'])(.*?)\1[^>]*>([\s\S]*?)<\/parameter>/gi
  for (const match of body.matchAll(parameterExpression)) {
    const key = (match[2] ?? '').trim()
    if (!key) continue
    input[key] = decodeParameterValue(match[3] ?? '')
  }
  return { id: fallbackId, name: normalizedName, input }
}

function normalizeInput(value: unknown): JsonObject {
  if (isJsonObject(value)) return value
  if (typeof value !== 'string') return {}
  try {
    const parsed: unknown = JSON.parse(value)
    return isJsonObject(parsed) ? parsed : {}
  } catch {
    return {}
  }
}

function decodeParameterValue(value: string): JsonObject[string] {
  const cleaned = decodeHtml(value).trim()
  if (!cleaned) return ''
  try {
    const parsed: unknown = JSON.parse(cleaned)
    return isJsonValue(parsed) ? parsed : cleaned
  } catch {
    return cleaned
  }
}

function stripProviderToolContext(text: string): string {
  let clean = text.replace(/[ \t]*<system-reminder\b[^>]*>[\s\S]*?<\/system-reminder>[ \t]*(?:\n)?/gi, '')
  clean = clean.replace(/^TOOL_CALL_ID:\s*[^\n]*(?:\n|$)/gim, '')
  clean = stripJsonLineMarker(clean, 'TOOL:')
  return clean.replace(/^TOOL:\s*(?:\{|\[|None\b|True\b|False\b|"|')[^\n]*(?:\n|$)/gim, '').trim()
}

function stripJsonLineMarker(text: string, marker: string): string {
  const expression = new RegExp('^' + escapeRegex(marker) + '[ \\t]*', 'gim')
  const spans: Array<readonly [number, number]> = []
  for (const match of text.matchAll(expression)) {
    const start = match.index
    if (start === undefined) continue
    const decoded = decodeJsonPrefix(text, start + match[0].length)
    if (decoded) spans.push([start, decoded.end])
  }
  return removeSpans(text, spans)
}

function removeSpans(text: string, spans: readonly (readonly [number, number])[]): string {
  let clean = text
  for (const [start, end] of [...spans].sort((left, right) => right[0] - left[0])) {
    const before = clean.slice(0, start).trimEnd()
    const after = clean.slice(end).trimStart()
    clean = before && after ? before + '\n' + after : before || after
  }
  return clean
}

function decodeJsonPrefix(text: string, start: number): { readonly end: number; readonly value: unknown } | undefined {
  const first = text[start]
  if (first !== '{' && first !== '[') return undefined
  let quote = ''
  let escaped = false
  const stack: string[] = []
  for (let index = start; index < text.length; index += 1) {
    const character = text[index] ?? ''
    if (quote) {
      if (escaped) {
        escaped = false
      } else if (character === String.fromCharCode(92)) {
        escaped = true
      } else if (character === quote) {
        quote = ''
      }
      continue
    }
    if (character === '"' || character === "'") {
      quote = character
      continue
    }
    if (character === '{') stack.push('}')
    else if (character === '[') stack.push(']')
    else if (character === '}' || character === ']') {
      if (stack.pop() !== character) return undefined
      if (stack.length === 0) {
        const end = index + 1
        try {
          return { end, value: JSON.parse(text.slice(start, end)) as unknown }
        } catch {
          return undefined
        }
      }
    }
  }
  return undefined
}

function skipWhitespaceAndOptionalColon(text: string, index: number): number {
  let cursor = index
  while (/\s/.test(text[cursor] ?? '')) cursor += 1
  if (text[cursor] === ':') cursor += 1
  while (/\s/.test(text[cursor] ?? '')) cursor += 1
  return cursor
}

function decodeHtml(value: string): string {
  return value
    .replaceAll('&quot;', '"')
    .replaceAll('&apos;', "'")
    .replaceAll('&#39;', "'")
    .replaceAll('&lt;', '<')
    .replaceAll('&gt;', '>')
    .replaceAll('&amp;', '&')
}

function escapeRegex(value: string): string {
  const special = new Set(['.', '*', '+', '?', '^', String.fromCharCode(36), '{', '}', '(', ')', '|', '[', ']', String.fromCharCode(92)])
  return [...value].map(character => special.has(character) ? String.fromCharCode(92) + character : character).join('')
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function isJsonValue(value: unknown): value is JsonObject[string] {
  if (value === null || typeof value === 'boolean' || typeof value === 'number' || typeof value === 'string') return true
  if (Array.isArray(value)) return value.every(isJsonValue)
  return isJsonObject(value) && Object.values(value).every(isJsonValue)
}
