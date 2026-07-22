// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../../core/errors.js'
import { deterministicToolCallId } from '../toolCallIds.js'
import { isJsonObject, type JsonObject, type ToolCall } from '../../types/toolCalls.js'

export type ToolCallParserName =
  | 'deepseek_v3'
  | 'deepseek_v3_1'
  | 'glm45'
  | 'glm47'
  | 'kimi_k2'
  | 'llama'
  | 'longcat'
  | 'mistral'
  | 'qwen'
  | 'qwen3_coder'
  | 'xml_tool_call'

/** A tool call extracted from a model's raw text response. */
export interface ParsedToolCall {
  readonly arguments: JsonObject
  readonly name: string
  readonly rawId: string
}

/** Stateless parser for one raw-text model tool-call format. */
export interface ToolCallTextParser {
  readonly name: ToolCallParserName
  readonly parse: (text: string) => ParsedToolCall[]
}

/** Create an immutable parser record for the model-format registry. */
export function createToolCallParser(
  name: ToolCallParserName,
  parse: (text: string) => ParsedToolCall[],
): ToolCallTextParser {
  return Object.freeze({ name, parse })
}

/**
 * Normalize raw-text parser output into the runtime's canonical tool calls.
 *
 * Provider-supplied IDs are retained. Calls without IDs use the same stable
 * fallback as structured OpenAI-, Anthropic-, Gemini-, and Ollama-style
 * stream adapters.
 */
export function normalizeParsedToolCalls(calls: readonly ParsedToolCall[]): ToolCall[] {
  const generatedIdOccurrences = new Map<string, number>()
  return calls.map((candidate, index) => {
    const call = validateParsedToolCall(candidate, index)
    let id = call.rawId
    if (!id) {
      const generated = deterministicToolCallId(call.name, call.arguments)
      const occurrence = generatedIdOccurrences.get(generated) ?? 0
      generatedIdOccurrences.set(generated, occurrence + 1)
      // Identical raw-text calls hash to the same fallback ID; suffix later
      // occurrences so tool_use/tool_result correlation IDs stay unique.
      id = occurrence === 0 ? generated : `${generated}#${occurrence + 1}`
    }
    return {
      id,
      type: 'function',
      function: {
        name: call.name,
        arguments: call.arguments,
      },
    }
  })
}

function validateParsedToolCall(candidate: ParsedToolCall, index: number): ParsedToolCall {
  if (!isJsonObject(candidate)) {
    throw new ValidationError(`parsedToolCalls[${index}]`, 'must be an object', candidate)
  }
  const name = candidate.name
  if (typeof name !== 'string' || !name.trim()) {
    throw new ValidationError(`parsedToolCalls[${index}].name`, 'must be a non-empty string', name)
  }
  const arguments_ = candidate.arguments
  if (!isJsonObject(arguments_)) {
    throw new ValidationError(`parsedToolCalls[${index}].arguments`, 'must be a JSON object', arguments_)
  }
  const rawId = candidate.rawId
  if (typeof rawId !== 'string') {
    throw new ValidationError(`parsedToolCalls[${index}].rawId`, 'must be a string', rawId)
  }
  return { name: name.trim(), arguments: arguments_, rawId: rawId.trim() }
}
