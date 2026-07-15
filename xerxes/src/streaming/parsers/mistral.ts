// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isJsonObject } from '../../types/toolCalls.js'
import { assertParserText, decodeJsonCompositeAt, skipWhitespace, stringFromJson } from './common.js'
import type { ParsedToolCall } from './types.js'

/** Parse Mistral's `[TOOL_CALLS][...]` JSON-array response format. */
export function parseMistralToolCalls(text: string): ParsedToolCall[] {
  assertParserText(text)
  const marker = text.indexOf('[TOOL_CALLS]')
  if (marker < 0) {
    return []
  }
  const start = skipWhitespace(text, marker + '[TOOL_CALLS]'.length)
  const decoded = decodeJsonCompositeAt(text, start)
  if (!decoded || !Array.isArray(decoded.value)) {
    return []
  }

  const calls: ParsedToolCall[] = []
  for (const entry of decoded.value) {
    if (!isJsonObject(entry)) {
      continue
    }
    const name = entry.name
    if (!name) {
      continue
    }
    const argumentsValue = entry.arguments || entry.parameters || {}
    calls.push({
      name: String(name),
      arguments: isJsonObject(argumentsValue) ? argumentsValue : {},
      rawId: stringFromJson(entry.id),
    })
  }
  return calls
}
