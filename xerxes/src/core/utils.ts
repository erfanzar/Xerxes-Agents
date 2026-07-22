// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { JsonObject, JsonSchema, JsonValue, ToolDefinition } from '../types/toolCalls.js'

export type MutableRecord = Record<string, unknown>

export type FunctionParameterType = 'array' | 'boolean' | 'integer' | 'null' | 'number' | 'object' | 'string'

/** Explicit runtime metadata that replaces Python's erased annotation reflection. */
export interface FunctionParameter {
  readonly defaultValue?: unknown
  readonly description?: string
  readonly items?: FunctionParameterType
  readonly name: string
  readonly required?: boolean
  readonly type?: FunctionParameterType | readonly FunctionParameterType[]
}

/** A callable-specific schema override, analogous to Python's hand-tuned schema metadata. */
export interface CallableSchema {
  readonly description?: string
  readonly inputSchema?: JsonSchema
  readonly name?: string
  readonly parameters?: JsonSchema
}

/** Metadata used to build a tool schema when no callable-level override supplies one. */
export interface FunctionToJsonOptions {
  readonly description?: string
  readonly name?: string
  readonly parameters?: readonly FunctionParameter[]
}

/** Injectable effect boundary for debug output and local timestamp capture. */
export interface DebugOutput {
  now(): Date
  write(line: string): void
}

const CALLABLE_SCHEMAS = new WeakMap<Function, CallableSchema>()
const FUNCTION_PARAMETER_TYPES = new Set<FunctionParameterType>([
  'array',
  'boolean',
  'integer',
  'null',
  'number',
  'object',
  'string',
])

/**
 * Normalize a synchronous or asynchronous value at an async boundary.
 *
 * JavaScript cannot safely block the main event loop until a promise settles,
 * so this is the native replacement for Python's `run_sync`: callers await it
 * rather than creating a hidden worker thread.
 */
export function runAsync<T>(operation: T | PromiseLike<T>): Promise<T> {
  return Promise.resolve(operation)
}

/** Render the ANSI debug line used by the source runtime without performing I/O. */
export function formatDebugLine(args: readonly unknown[], timestamp: Date): string {
  if (!(timestamp instanceof Date) || Number.isNaN(timestamp.valueOf())) {
    throw new TypeError('timestamp must be a valid Date')
  }
  const date = [
    String(timestamp.getFullYear()).padStart(4, '0'),
    String(timestamp.getMonth() + 1).padStart(2, '0'),
    String(timestamp.getDate()).padStart(2, '0'),
  ].join('-')
  const time = [
    String(timestamp.getHours()).padStart(2, '0'),
    String(timestamp.getMinutes()).padStart(2, '0'),
    String(timestamp.getSeconds()).padStart(2, '0'),
  ].join(':')
  const message = args.map(String).join(' ')
  return `\u001b[97m[\u001b[90m${date} ${time}\u001b[97m]\u001b[90m ${message}\u001b[0m`
}

/** Write a debug line only when the caller enables debugging. */
export function debugPrint(debug: boolean, output: DebugOutput, ...args: readonly unknown[]): void {
  if (!debug) return
  if (!output || typeof output.now !== 'function' || typeof output.write !== 'function') {
    throw new TypeError('debug output must provide now() and write() functions')
  }
  output.write(formatDebugLine(args, output.now()))
}

/**
 * Concatenate string fields and recursively merge nested records into a mutable accumulator.
 *
 * Non-string scalar values, arrays, and null values are deliberately ignored,
 * matching streaming-delta merge semantics in the Python runtime.
 */
/** Keys that must never be merged, because assigning them mutates object prototypes. */
const UNSAFE_MERGE_KEYS = new Set(['__proto__', 'constructor', 'prototype'])

export function mergeFields(target: MutableRecord, source: Readonly<Record<string, unknown>>): void {
  assertPlainRecord(target, 'target')
  assertPlainRecord(source, 'source')
  for (const [key, value] of Object.entries(source)) {
    if (UNSAFE_MERGE_KEYS.has(key)) continue
    if (typeof value === 'string') {
      const existing = target[key]
      if (typeof existing !== 'string') {
        throw new TypeError(`target.${key} must be a string before it can be merged`)
      }
      target[key] = existing + value
      continue
    }
    if (isPlainRecord(value)) {
      const existing = target[key]
      if (!isPlainRecord(existing)) {
        throw new TypeError(`target.${key} must be a record before it can be merged`)
      }
      mergeFields(existing, value)
    }
  }
}

/**
 * Fold one OpenAI-compatible streaming delta into a response accumulator.
 *
 * The Python implementation removes `role` and tool-call `index` in place.
 * This native version preserves the incoming delta and removes those transport
 * fields only from its internal merge view.
 */
export function mergeChunk(finalResponse: MutableRecord, delta: Readonly<Record<string, unknown>>): void {
  assertPlainRecord(finalResponse, 'finalResponse')
  assertPlainRecord(delta, 'delta')

  const fields: MutableRecord = {}
  for (const [key, value] of Object.entries(delta)) {
    if (key !== 'role' && key !== 'tool_calls') defineOwnField(fields, key, value)
  }
  mergeFields(finalResponse, fields)

  const toolCalls = delta.tool_calls
  if (!Array.isArray(toolCalls) || toolCalls.length === 0) return
  const toolCall = toolCalls[0]
  if (!isPlainRecord(toolCall)) throw new TypeError('delta.tool_calls[0] must be a record')
  const indexValue = toolCall.index
  if (typeof indexValue !== 'number' || !Number.isInteger(indexValue) || indexValue < 0) {
    throw new TypeError('delta.tool_calls[0].index must be a non-negative integer')
  }
  const index = indexValue
  const finalToolCalls = finalResponse.tool_calls
  if (!Array.isArray(finalToolCalls)) throw new TypeError('finalResponse.tool_calls must be an array')
  const finalToolCall = finalToolCalls[index]
  if (!isPlainRecord(finalToolCall)) {
    throw new TypeError(`finalResponse.tool_calls[${index}] must be a record`)
  }
  const toolFields: MutableRecord = {}
  for (const [key, value] of Object.entries(toolCall)) {
    if (key !== 'index') defineOwnField(toolFields, key, value)
  }
  mergeFields(finalToolCall, toolFields)
}

/** Estimate token count from character length when no provider tokenizer is available. */
export function estimateTokens(text: string, charsPerToken = 4): number {
  if (typeof text !== 'string') throw new TypeError('text must be a string')
  if (!Number.isFinite(charsPerToken) || charsPerToken <= 0) {
    throw new RangeError('charsPerToken must be a positive finite number')
  }
  if (!text) return 0
  return Math.max(1, Math.trunc(text.length / charsPerToken))
}

/** Sum content token estimates plus the source runtime's four-token message overhead. */
export function estimateMessagesTokens(messages: readonly Readonly<Record<string, unknown>>[]): number {
  let total = 0
  for (const message of messages) {
    assertPlainRecord(message, 'message')
    const content = message.content
    if (content) total += estimateTokens(String(content))
    total += 4
  }
  return total
}

/** Attach immutable native schema metadata to a callable. */
export function defineCallableSchema<T extends Function>(func: T, schema: CallableSchema): T {
  requireFunction(func, 'func')
  assertStrictObject(schema, ['description', 'inputSchema', 'name', 'parameters'])
  const normalized: {
    description?: string
    inputSchema?: JsonSchema
    name?: string
    parameters?: JsonSchema
  } = {}
  const description = optionalText(schema.description, 'schema.description')
  const name = optionalText(schema.name, 'schema.name')
  if (description !== undefined) normalized.description = description
  if (name !== undefined) normalized.name = name
  if (schema.inputSchema !== undefined) {
    normalized.inputSchema = freezeJsonSchema(schema.inputSchema, 'schema.inputSchema')
  }
  if (schema.parameters !== undefined) {
    normalized.parameters = freezeJsonSchema(schema.parameters, 'schema.parameters')
  }
  CALLABLE_SCHEMAS.set(func, Object.freeze(normalized))
  return func
}

/** Return the wire-facing callable name from registered metadata or the function itself. */
export function getCallablePublicName(func: Function): string {
  requireFunction(func, 'func')
  const customName = CALLABLE_SCHEMAS.get(func)?.name
  if (customName !== undefined) return customName
  return func.name || String(func)
}

/**
 * Build an OpenAI-compatible function tool schema from explicit native metadata.
 *
 * TypeScript erases annotations and documentation at runtime. Callers therefore
 * provide parameter descriptors or register a `CallableSchema` instead of
 * parsing function source text and guessing an unsafe schema.
 */
export function functionToJson(func: Function, options: FunctionToJsonOptions = {}): ToolDefinition {
  requireFunction(func, 'func')
  assertStrictObject(options as unknown, ['description', 'name', 'parameters'])
  const callableSchema = CALLABLE_SCHEMAS.get(func)
  const customName = callableSchema?.name
  const optionName = optionalText(options.name, 'options.name')
  const name = customName ?? optionName ?? getCallablePublicName(func)
  const description = combineDescriptions(callableSchema?.description, options.description)
  const customParameters = callableSchema?.parameters ?? callableSchema?.inputSchema
  const parameters = customParameters ?? parameterSchema(options.parameters ?? [])
  return Object.freeze({
    type: 'function',
    function: Object.freeze({
      name,
      description,
      parameters,
    }),
  })
}

/** Return a JSON-only copy, rejecting unsupported values and circular structures. */
export function toJsonValue(value: unknown): JsonValue {
  return copyJsonValue(value, new Set<object>())
}

/** Serialize JSON-compatible data without invoking lossy JSON.stringify coercions. */
export function safeJsonStringify(value: unknown): string {
  const serialized = JSON.stringify(toJsonValue(value))
  if (serialized === undefined) throw new TypeError('value is not JSON-serializable')
  return serialized
}

/** Assert an object contains only the allowed fields, the native equivalent of strict model extras. */
export function assertStrictObject(
  value: unknown,
  allowedFields: readonly string[],
): asserts value is Record<string, unknown> {
  assertPlainRecord(value, 'value')
  const allowed = new Set<string>()
  for (const field of allowedFields) {
    const normalized = optionalText(field, 'allowed field')
    if (normalized === undefined) throw new TypeError('allowed field must be non-empty')
    allowed.add(normalized)
  }
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) throw new TypeError(`Unexpected field: ${key}`)
  }
}

function parameterSchema(parameters: readonly FunctionParameter[]): JsonSchema {
  if (!Array.isArray(parameters)) throw new TypeError('options.parameters must be an array')
  const properties: JsonObject = {}
  const required: string[] = []
  const names = new Set<string>()
  for (const parameter of parameters) {
    assertStrictObject(parameter as unknown, ['defaultValue', 'description', 'items', 'name', 'required', 'type'])
    const name = optionalText(parameter.name, 'parameter.name')
    if (name === undefined) throw new TypeError('parameter.name must be non-empty')
    if (name === 'context_variables') continue
    if (names.has(name)) throw new TypeError(`Duplicate parameter name: ${name}`)
    names.add(name)

    const property: JsonObject = { type: parameterTypeSchema(parameter.type) }
    if (parameter.items !== undefined) {
      if (parameter.type !== 'array') throw new TypeError(`parameter ${name} items require type array`)
      property.items = { type: requireParameterType(parameter.items, `parameter ${name} items`) }
    }
    const description = optionalText(parameter.description, `parameter ${name} description`)
    if (description !== undefined) property.description = description
    const hasDefault = Object.hasOwn(parameter, 'defaultValue')
    if (hasDefault) property.default = serializeDefaultValue(parameter.defaultValue)
    if (parameter.required !== undefined && typeof parameter.required !== 'boolean') {
      throw new TypeError(`parameter ${name} required must be a boolean`)
    }
    if (parameter.required ?? !hasDefault) required.push(name)
    properties[name] = deepFreezeJson(toJsonValue(property))
  }
  return freezeJsonSchema({ type: 'object', properties, required }, 'parameters')
}

function parameterTypeSchema(value: FunctionParameter['type']): JsonValue {
  if (value === undefined) return 'string'
  if (!Array.isArray(value)) return requireParameterType(value, 'parameter type')
  if (!value.length) throw new TypeError('parameter type union must not be empty')
  return {
    type: 'union',
    types: value.map(type => requireParameterType(type, 'parameter union type')),
  }
}

function requireParameterType(value: unknown, field: string): FunctionParameterType {
  if (typeof value !== 'string' || !FUNCTION_PARAMETER_TYPES.has(value as FunctionParameterType)) {
    throw new TypeError(`${field} must be a supported JSON schema type`)
  }
  return value as FunctionParameterType
}

function serializeDefaultValue(value: unknown): JsonValue {
  try {
    return toJsonValue(value)
  } catch (error) {
    if (error instanceof TypeError) return String(value)
    throw error
  }
}

function combineDescriptions(schemaDescription: string | undefined, optionDescription: unknown): string {
  const descriptions = [schemaDescription, optionalText(optionDescription, 'options.description')]
    .filter((value): value is string => value !== undefined)
  return descriptions.filter((value, index) => index === 0 || value !== descriptions[index - 1]).join('\n\n')
}

function optionalText(value: unknown, field: string): string | undefined {
  if (value === undefined) return undefined
  if (typeof value !== 'string') throw new TypeError(`${field} must be a string`)
  const normalized = value.trim()
  return normalized || undefined
}

function freezeJsonSchema(value: unknown, field: string): JsonSchema {
  const json = toJsonValue(value)
  if (!isJsonObject(json)) throw new TypeError(`${field} must be a JSON object`)
  return deepFreezeJson(json) as JsonSchema
}

function copyJsonValue(value: unknown, ancestors: Set<object>): JsonValue {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return value
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new TypeError('numbers must be finite JSON values')
    return value
  }
  if (Array.isArray(value)) {
    if (ancestors.has(value)) throw new TypeError('circular JSON values are not supported')
    ancestors.add(value)
    const copied = value.map(item => copyJsonValue(item, ancestors))
    ancestors.delete(value)
    return copied
  }
  if (isPlainRecord(value)) {
    if (ancestors.has(value)) throw new TypeError('circular JSON values are not supported')
    ancestors.add(value)
    const copied: JsonObject = {}
    for (const [key, item] of Object.entries(value)) {
      defineOwnField(copied, key, copyJsonValue(item, ancestors))
    }
    ancestors.delete(value)
    return copied
  }
  throw new TypeError('value must be JSON-serializable')
}

function deepFreezeJson(value: JsonValue): JsonValue {
  if (Array.isArray(value)) {
    for (const item of value) deepFreezeJson(item)
    return Object.freeze(value) as unknown as JsonValue
  }
  if (isJsonObject(value)) {
    for (const item of Object.values(value)) deepFreezeJson(item)
    return Object.freeze(value) as JsonValue
  }
  return value
}

function isJsonObject(value: JsonValue): value is JsonObject {
  return isPlainRecord(value)
}

/**
 * Assign `key` as a real own property so hostile names such as `__proto__`
 * are stored as data instead of silently mutating the target's prototype.
 */
function defineOwnField(target: MutableRecord, key: string, value: unknown): void {
  Object.defineProperty(target, key, {
    configurable: true,
    enumerable: true,
    value,
    writable: true,
  })
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

function assertPlainRecord(value: unknown, field: string): asserts value is Record<string, unknown> {
  if (!isPlainRecord(value)) throw new TypeError(`${field} must be a plain object`)
}

function requireFunction(value: unknown, field: string): asserts value is Function {
  if (typeof value !== 'function') throw new TypeError(`${field} must be a function`)
}
