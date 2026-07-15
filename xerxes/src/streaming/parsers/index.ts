// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { parseLlamaToolCalls, parseXmlToolCalls } from './common.js'
import { parseDeepSeekV3ToolCalls, parseDeepSeekV31ToolCalls } from './deepseek.js'
import { parseGlm45ToolCalls, parseGlm47ToolCalls } from './glm.js'
import { parseKimiK2ToolCalls } from './kimi.js'
import { parseLongCatToolCalls } from './longcat.js'
import { parseMistralToolCalls } from './mistral.js'
import { parseQwen3CoderToolCalls, parseQwenToolCalls } from './qwen.js'
import { createToolCallParser, type ParsedToolCall, type ToolCallParserName, type ToolCallTextParser } from './types.js'

export * from './common.js'
export * from './deepseek.js'
export * from './glm.js'
export * from './kimi.js'
export * from './longcat.js'
export * from './mistral.js'
export * from './qwen.js'
export * from './types.js'

/** Parsers keyed by raw-text model format. Every parser is pure and stateless. */
export const TOOL_CALL_PARSER_REGISTRY: Readonly<Record<ToolCallParserName, ToolCallTextParser>> = Object.freeze({
  xml_tool_call: createToolCallParser('xml_tool_call', parseXmlToolCalls),
  llama: createToolCallParser('llama', parseLlamaToolCalls),
  mistral: createToolCallParser('mistral', parseMistralToolCalls),
  qwen: createToolCallParser('qwen', parseQwenToolCalls),
  qwen3_coder: createToolCallParser('qwen3_coder', parseQwen3CoderToolCalls),
  deepseek_v3: createToolCallParser('deepseek_v3', parseDeepSeekV3ToolCalls),
  deepseek_v3_1: createToolCallParser('deepseek_v3_1', parseDeepSeekV31ToolCalls),
  glm45: createToolCallParser('glm45', parseGlm45ToolCalls),
  glm47: createToolCallParser('glm47', parseGlm47ToolCalls),
  kimi_k2: createToolCallParser('kimi_k2', parseKimiK2ToolCalls),
  longcat: createToolCallParser('longcat', parseLongCatToolCalls),
})

/** Return the parser registered under a known raw-text model format. */
export function getToolCallParser(name: string): ToolCallTextParser | undefined {
  if (!Object.hasOwn(TOOL_CALL_PARSER_REGISTRY, name)) {
    return undefined
  }
  return TOOL_CALL_PARSER_REGISTRY[name as ToolCallParserName]
}

/** Infer a raw-text tool-call format from a model identifier. */
export function detectToolCallFormat(model: string): ToolCallParserName | undefined {
  const normalized = model.toLowerCase()
  if (normalized.includes('hermes')) {
    return 'xml_tool_call'
  }
  if (normalized.includes('llama') || normalized.includes('llama-3')) {
    return 'llama'
  }
  if (normalized.includes('mistral') || normalized.includes('mixtral')) {
    return 'mistral'
  }
  if (normalized.includes('qwen3-coder') || normalized.includes('qwen-coder')) {
    return 'qwen3_coder'
  }
  if (normalized.includes('qwen')) {
    return 'qwen'
  }
  if (normalized.includes('deepseek-v3.1') || normalized.includes('deepseek-v3-1')) {
    return 'deepseek_v3_1'
  }
  if (normalized.includes('deepseek')) {
    return 'deepseek_v3'
  }
  if (normalized.includes('glm-4.7') || normalized.includes('glm47')) {
    return 'glm47'
  }
  if (normalized.includes('glm-4.5') || normalized.includes('glm45')) {
    return 'glm45'
  }
  if (normalized.includes('kimi-k2') || normalized.includes('kimi')) {
    return 'kimi_k2'
  }
  if (normalized.includes('longcat')) {
    return 'longcat'
  }
  return undefined
}

/** Detect and parse every completed raw-text tool call for a specific model. */
export function parseToolCallsForModel(model: string, text: string): ParsedToolCall[] {
  const format = detectToolCallFormat(model)
  return format ? TOOL_CALL_PARSER_REGISTRY[format].parse(text) : []
}
