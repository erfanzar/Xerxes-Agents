// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'

export type JsonPrimitive = boolean | null | number | string
export type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue }
export type JsonObject = { [key: string]: JsonValue }
export type JsonSchema = Readonly<Record<string, unknown>>

export interface FunctionDefinition {
  readonly name: string
  readonly description: string
  readonly parameters: JsonSchema
}

export interface ToolDefinition {
  readonly type: 'function'
  readonly function: FunctionDefinition
}

export type ToolChoice = 'any' | 'auto' | 'none'

export interface FunctionCall {
  readonly name: string
  readonly arguments: JsonObject
}

export interface ToolCall {
  readonly id: string
  readonly type: 'function'
  readonly function: FunctionCall
}

export interface OpenAiToolCall {
  readonly id: string
  readonly type: 'function'
  readonly function: {
    readonly name: string
    readonly arguments: string
  }
}

export function isJsonObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Parse provider arguments into the one canonical tool-call representation. */
export function parseToolArguments(value: string | JsonObject | undefined): JsonObject {
  if (value === undefined || value === '') {
    return {}
  }
  if (isJsonObject(value)) {
    return value
  }

  try {
    const parsed: unknown = JSON.parse(value)
    if (!isJsonObject(parsed)) {
      throw new ValidationError('arguments', 'must decode to a JSON object', value)
    }
    return parsed
  } catch (error) {
    if (error instanceof ValidationError) {
      throw error
    }
    throw new ValidationError('arguments', 'must be valid JSON', value)
  }
}

export function toolCallFromOpenAi(value: OpenAiToolCall): ToolCall {
  if (!value.id || !value.function?.name) {
    throw new ValidationError('tool_call', 'must include id and function name', value as unknown as JsonObject)
  }
  return {
    id: value.id,
    type: 'function',
    function: {
      name: value.function.name,
      arguments: parseToolArguments(value.function.arguments),
    },
  }
}

export function toolCallToOpenAi(value: ToolCall): OpenAiToolCall {
  return {
    id: value.id,
    type: 'function',
    function: {
      name: value.function.name,
      arguments: JSON.stringify(value.function.arguments),
    },
  }
}

export function toolDefinitionToOpenAi(value: ToolDefinition): ToolDefinition {
  return value
}
