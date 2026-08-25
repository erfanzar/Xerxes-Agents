// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { JsonObject, JsonSchema } from '../types/toolCalls.js'

/** Outcome of checking a model-emitted tool call against its declared JSON Schema subset. */
export interface ToolArgumentValidationResult {
  /**
   * The arguments as they were actually validated: provider JSON strings decoded
   * and literal string forms repaired by `coerceDeclared`. Executors must hand
   * this to the handler; passing the raw payload instead would let an accepted
   * `offset: "0"` reach the tool as a string again. Undefined only when the
   * payload was not an object at all.
   */
  readonly coerced: JsonObject | undefined
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
 *
 * Providers hand arguments over as a JSON string often enough that decoding one
 * here is cheaper than a burnt turn; the decoded object is reported back through
 * `coerced` so the caller never has to parse it a second time.
 *
 * Messages deliberately do NOT name the tool: the result carries `toolName`, and
 * the one production consumer wraps this in a FunctionExecutionError that
 * already prefixes `Function <name>: `. Repeating it here produced
 * "Function agent_memory_list: agent_memory_list: ..." — the name twice, and the
 * actual reason pushed past the point where surfaces truncate it.
 *
 * A rejection is the model's only chance to get the call right on the retry, so
 * a failure that turns on the schema states the schema.
 */
export function validateToolArguments(
  toolName: string,
  argumentsValue: unknown,
  schema: JsonSchema | undefined,
): ToolArgumentValidationResult {
  const name = requiredToolName(toolName)
  let payload = argumentsValue
  if (typeof argumentsValue === 'string') {
    try {
      payload = JSON.parse(argumentsValue) as unknown
    } catch {
      return invalid(name, `arguments are not valid JSON: ${argumentsValue.slice(0, 200)}`)
    }
  }
  if (!schema || !Object.keys(schema).length) return valid(name, isRecord(payload) ? payload : undefined)
  if (!isRecord(payload)) {
    return invalid(name, `expected arguments to be an object, got ${typeName(payload)}`)
  }

  const required = stringArray(schema.required)
  const properties = isRecord(schema.properties) ? schema.properties : {}
  const missing = required.filter(key => !(key in payload))
  if (missing.length) {
    return {
      ok: false,
      toolName: name,
      coerced: undefined,
      missing,
      error: `missing required parameter(s): ${missing.join(', ')}. ${expectedParameters(properties, required)}`,
    }
  }

  let repaired: Record<string, unknown> | undefined
  for (const [key, rawValue] of Object.entries(payload)) {
    const property = properties[key]
    if (!isRecord(property)) {
      if (schema.additionalProperties === false) {
        return invalid(name, `unknown parameter '${key}'. ${acceptedParameters(properties)}`)
      }
      continue
    }
    const expectedType = typeof property.type === 'string' ? property.type : undefined
    const value = expectedType === undefined ? rawValue : coerceDeclared(rawValue, expectedType)
    if (!Object.is(value, rawValue)) {
      repaired ??= { ...payload }
      repaired[key] = value
    }
    if (expectedType && !matchesType(value, expectedType)) {
      return invalid(name, `parameter '${key}' expected ${expectedType}, got ${typeName(value)}`)
    }
    const values = Array.isArray(property.enum) ? property.enum : undefined
    if (values && !values.some(candidate => jsonEqual(candidate, value))) {
      return invalid(name, `parameter '${key}' must be ${formatEnumRequirement(values)}, got ${formatValue(value)}`)
    }
    const elementError = expectedType === 'array' && Array.isArray(value)
      ? arrayElementError(key, value, property.items)
      : undefined
    if (elementError) return invalid(name, elementError)
  }
  return valid(name, repaired ?? payload)
}

/**
 * Check the declared element shape of an array parameter.
 *
 * Only the top level used to be checked, so a SpawnAgents batch missing a
 * title per element passed the gate cleanly and failed much deeper, where the
 * error no longer names the element that was wrong. Batch tools are exactly
 * where a model is most likely to slip, and where a vague failure costs the
 * most, so the element shape is worth one pass.
 *
 * Deliberately shallow — element type, element enum, and the required keys of
 * an object element. This stays a pre-execution boundary, not a JSON Schema
 * engine, and it never repairs: a rejection here is information for the model,
 * and silently rewriting a nested value would be a worse failure than the one
 * it prevents.
 */
function arrayElementError(key: string, values: readonly unknown[], items: unknown): string | undefined {
  if (!isRecord(items)) return undefined
  const expected = typeof items.type === 'string' ? items.type : undefined
  const allowed = Array.isArray(items.enum) ? items.enum : undefined
  const itemRequired = stringArray(items.required)
  const itemProperties = isRecord(items.properties) ? items.properties : {}
  for (const [index, element] of values.entries()) {
    const at = `parameter '${key}[${index}]'`
    if (expected && !matchesType(element, expected)) {
      return `${at} expected ${expected}, got ${typeName(element)}`
    }
    if (allowed && !allowed.some(candidate => jsonEqual(candidate, element))) {
      return `${at} must be ${formatEnumRequirement(allowed)}, got ${formatValue(element)}`
    }
    if (!itemRequired.length || !isRecord(element)) continue
    const missing = itemRequired.filter(field => !(field in element))
    if (missing.length) {
      return `${at} is missing required field(s): ${missing.join(', ')}. `
        + `${expectedParameters(itemProperties, itemRequired)}`
    }
  }
  return undefined
}

const INTEGER_LITERAL = /^-?\d+$/
const NUMBER_LITERAL = /^-?\d+(\.\d+)?$/

/**
 * Repair the exact literal string forms providers emit for typed parameters —
 * `"0"` for an integer, `"true"` for a boolean — so a well-formed call is not
 * thrown away over a quoting artifact.
 *
 * The accepted set is deliberately far narrower than `Number()` or `JSON.parse`
 * semantics: '', null, 'yes' and 'on' are left alone, and '0' never becomes a
 * boolean. Widening any of those would convert a loud rejection into a tool run
 * with a silently wrong argument, which costs far more than the rejected turn
 * this repairs. Non-string values and undeclared types pass through untouched.
 */
export function coerceDeclared(value: unknown, expectedType: string): unknown {
  if (typeof value !== 'string') return value
  if (expectedType === 'boolean') {
    if (value === 'true') return true
    if (value === 'false') return false
    return value
  }
  if (expectedType === 'integer' && INTEGER_LITERAL.test(value)) {
    const parsed = Number(value)
    // Past the safe range the parse rounds to a different integer, so keep the
    // string and let the declared-type check reject it rather than run the wrong one.
    return Number.isSafeInteger(parsed) ? parsed : value
  }
  if (expectedType === 'number' && NUMBER_LITERAL.test(value)) {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : value
  }
  return value
}

function valid(toolName: string, coerced: Record<string, unknown> | undefined): ToolArgumentValidationResult {
  return { ok: true, toolName, coerced: coerced as JsonObject | undefined, error: '', missing: [] }
}

/** Bound how much schema a rejection may quote; a wide tool must not flood the turn. */
const MAX_QUOTED_PARAMETERS = 12

/** "Expected parameters: path (string, required), limit (integer)" */
function expectedParameters(
  properties: Record<string, unknown>,
  required: readonly string[],
): string {
  const names = Object.keys(properties)
  if (!names.length) return 'This tool declares no parameters.'
  const requiredSet = new Set(required)
  const shown = names.slice(0, MAX_QUOTED_PARAMETERS).map(key => {
    const property = properties[key]
    const type = isRecord(property) && typeof property.type === 'string' ? property.type : 'any'
    return `${key} (${type}${requiredSet.has(key) ? ', required' : ''})`
  })
  const rest = names.length - shown.length
  return `Expected parameters: ${shown.join(', ')}${rest > 0 ? `, and ${rest} more` : ''}.`
}

/** "Accepted parameters: path, limit, offset" */
function acceptedParameters(properties: Record<string, unknown>): string {
  const names = Object.keys(properties)
  if (!names.length) return 'This tool accepts no parameters.'
  const shown = names.slice(0, MAX_QUOTED_PARAMETERS)
  const rest = names.length - shown.length
  return `Accepted parameters: ${shown.join(', ')}${rest > 0 ? `, and ${rest} more` : ''}.`
}

function invalid(toolName: string, error: string): ToolArgumentValidationResult {
  return { ok: false, toolName, coerced: undefined, error, missing: [] }
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
