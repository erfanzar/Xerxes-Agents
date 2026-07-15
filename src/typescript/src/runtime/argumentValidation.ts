// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { JsonObject, JsonSchema } from '../types/toolCalls.js'

/** Outcome of checking a model-emitted tool call against its declared JSON Schema subset. */
export interface ToolArgumentValidationResult {
  readonly error: string
  readonly missing: readonly string[]
  readonly ok: boolean
  readonly toolName: string
}

/**
 * Validate the lightweight JSON Schema subset Xerxes declares for native tools.
 *
 * This intentionally covers required fields, declared-property types, enum
 * values, and `additionalProperties: false` without adding a schema runtime.
 * It is a pre-execution boundary, not a general JSON Schema implementation.
 */
export function validateToolArguments(
  toolName: string,
  argumentsValue: unknown,
  schema: JsonSchema | undefined,
): ToolArgumentValidationResult {
  const name = requiredToolName(toolName)
  if (!schema || !Object.keys(schema).length) return valid(name)
  if (!isRecord(argumentsValue)) {
    return invalid(name, `${name}: expected arguments to be an object, got ${typeName(argumentsValue)}`)
  }

  const required = stringArray(schema.required)
  const properties = isRecord(schema.properties) ? schema.properties : {}
  const missing = required.filter(key => !(key in argumentsValue))
  if (missing.length) {
    return {
      ok: false,
      toolName: name,
      missing,
      error: `${name}: missing required parameter(s): ${missing.join(', ')}`,
    }
  }

  for (const [key, value] of Object.entries(argumentsValue)) {
    const property = properties[key]
    if (!isRecord(property)) {
      if (schema.additionalProperties === false) {
        return invalid(name, `${name}: unknown parameter '${key}' (schema has additionalProperties=false)`)
      }
      continue
    }
    const expectedType = typeof property.type === 'string' ? property.type : undefined
    if (expectedType && !matchesType(value, expectedType)) {
      return invalid(name, `${name}: parameter '${key}' expected ${expectedType}, got ${typeName(value)}`)
    }
    const values = Array.isArray(property.enum) ? property.enum : undefined
    if (values && !values.some(candidate => jsonEqual(candidate, value))) {
      return invalid(name, `${name}: parameter '${key}' must be ${formatEnumRequirement(values)}, got ${formatValue(value)}`)
    }
  }
  return valid(name)
}

/**
 * Return a model-facing validation error, accepting raw provider JSON strings.
 *
 * A return value of undefined means the call is safe to hand to the executor.
 */
export function validateAndFormatToolArgumentError(
  toolName: string,
  argumentsValue: JsonObject | string | unknown,
  schema: JsonSchema | undefined,
): string | undefined {
  let parsed = argumentsValue
  if (typeof argumentsValue === 'string') {
    try {
      parsed = JSON.parse(argumentsValue) as unknown
    } catch {
      return `${requiredToolName(toolName)}: arguments are not valid JSON: ${argumentsValue.slice(0, 200)}`
    }
  }
  if (!isRecord(parsed)) return `${requiredToolName(toolName)}: arguments must be a JSON object.`
  const result = validateToolArguments(toolName, parsed, schema)
  return result.ok ? undefined : result.error
}

function valid(toolName: string): ToolArgumentValidationResult {
  return { ok: true, toolName, error: '', missing: [] }
}

function invalid(toolName: string, error: string): ToolArgumentValidationResult {
  return { ok: false, toolName, error, missing: [] }
}

function requiredToolName(value: string): string {
  const name = value.trim()
  if (!name) throw new TypeError('toolName must be non-empty')
  return name
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === 'string') : []
}

function matchesType(value: unknown, expected: string): boolean {
  if (expected === 'string') return typeof value === 'string'
  if (expected === 'integer') return typeof value === 'number' && Number.isInteger(value)
  if (expected === 'number') return typeof value === 'number' && Number.isFinite(value)
  if (expected === 'boolean') return typeof value === 'boolean'
  if (expected === 'array') return Array.isArray(value)
  if (expected === 'object') return isRecord(value)
  if (expected === 'null') return value === null
  return true
}

function typeName(value: unknown): string {
  if (value === null) return 'null'
  if (Array.isArray(value)) return 'array'
  return typeof value
}

function formatEnumRequirement(values: readonly unknown[]): string {
  if (values.every((value): value is string => typeof value === 'string')) {
    return formatWords(values)
  }
  return 'one of ' + formatEnum(values)
}

function formatWords(values: readonly string[]): string {
  if (values.length === 0) return 'one of []'
  const first = values[0] ?? ''
  if (values.length === 1) return first
  const second = values[1] ?? ''
  if (values.length === 2) return first + ' or ' + second
  return values.slice(0, -1).join(', ') + ', or ' + (values.at(-1) ?? '')
}

function formatEnum(values: readonly unknown[]): string {
  return '[' + values.map(formatValue).join(', ') + ']'
}

function formatValue(value: unknown): string {
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function jsonEqual(left: unknown, right: unknown): boolean {
  return stableJson(left) === stableJson(right)
}

function stableJson(value: unknown): string {
  if (value === null) return 'null'
  if (typeof value === 'string') return JSON.stringify(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'number') return Number.isFinite(value) ? JSON.stringify(value) : JSON.stringify(String(value))
  if (Array.isArray(value)) return '[' + value.map(stableJson).join(',') + ']'
  if (isRecord(value)) {
    return '{' + Object.keys(value).sort().map(key => JSON.stringify(key) + ':' + stableJson(value[key])).join(',') + '}'
  }
  return JSON.stringify(String(value))
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
