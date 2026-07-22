// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'

import { ValidationError } from '../core/errors.js'

export const DEFAULT_TOOL_CALL_ID_LENGTH = 16
export const DEFAULT_TOOL_CALL_ID_PREFIX = 'call_'
export const SHA256_HEX_LENGTH = 64

/** Optional namespace and hash length for deterministic tool-call IDs. */
export interface ToolCallIdOptions {
  readonly length?: number
  readonly prefix?: string
}

/**
 * Serialize a value into stable, valid JSON suitable for tool-call hashing.
 *
 * Object keys are sorted recursively. Values outside JSON's data model are
 * represented by tagged JSON objects, so callers never need a lossy ambient
 * `JSON.stringify` fallback and circular values cannot throw.
 */
export function canonicalizeToolCallArguments(value: unknown): string {
  try {
    return serializeValue(value, new WeakSet<object>())
  } catch {
    return taggedValue('unserializable')
  }
}

/**
 * Produce a stable SHA-256 ID from a tool name and canonicalized arguments.
 *
 * Provider-issued IDs should be preserved. This helper is for fallback IDs
 * where the provider did not supply one, keeping replays and audits stable.
 */
export function deterministicToolCallId(
  name: string,
  arguments_: unknown,
  options: ToolCallIdOptions = {},
): string {
  if (typeof name !== 'string') {
    throw new ValidationError('toolName', 'must be a string', name)
  }
  const { length, prefix } = normalizeOptions(options)
  const payload = `${name}|${canonicalizeToolCallArguments(arguments_)}`
  const digest = createHash('sha256').update(payload, 'utf8').digest('hex')
  return prefix + digest.slice(0, length)
}

function normalizeOptions(options: ToolCallIdOptions): Required<ToolCallIdOptions> {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    throw new ValidationError('toolCallIdOptions', 'must be an options object', options)
  }
  const prefix = options.prefix ?? DEFAULT_TOOL_CALL_ID_PREFIX
  if (typeof prefix !== 'string') {
    throw new ValidationError('prefix', 'must be a string', prefix)
  }
  const length = options.length ?? DEFAULT_TOOL_CALL_ID_LENGTH
  if (!Number.isSafeInteger(length) || length < 1 || length > SHA256_HEX_LENGTH) {
    throw new ValidationError('length', `must be an integer from 1 to ${SHA256_HEX_LENGTH}`, length)
  }
  return { length, prefix }
}

function serializeValue(value: unknown, ancestors: WeakSet<object>): string {
  if (value === null) return 'null'
  if (typeof value === 'string') return quote(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'number') {
    return Number.isFinite(value) ? JSON.stringify(value) : taggedValue('number', String(value))
  }
  if (typeof value === 'undefined') return taggedValue('undefined')
  if (typeof value === 'bigint') return taggedValue('bigint', value.toString())
  if (typeof value === 'symbol') return taggedValue('symbol', value.description ?? '')
  if (typeof value === 'function') return taggedValue('function', functionName(value))
  return serializeObject(value, ancestors)
}

function serializeObject(value: object, ancestors: WeakSet<object>): string {
  if (ancestors.has(value)) {
    return taggedValue('circular')
  }
  ancestors.add(value)
  try {
    if (Array.isArray(value)) return serializeArray(value, ancestors)
    if (value instanceof Date) return serializeDate(value)
    if (value instanceof Map) return serializeMap(value, ancestors)
    if (value instanceof Set) return serializeSet(value, ancestors)
    if (!isPlainRecord(value)) return taggedValue('object', objectDescription(value))
    return serializeRecord(value, ancestors)
  } finally {
    ancestors.delete(value)
  }
}

function serializeArray(value: readonly unknown[], ancestors: WeakSet<object>): string {
  const items: string[] = []
  for (let index = 0; index < value.length; index += 1) {
    const descriptor = Object.getOwnPropertyDescriptor(value, String(index))
    if (descriptor === undefined) {
      items.push('null')
    } else if ('value' in descriptor) {
      items.push(serializeValue(descriptor.value, ancestors))
    } else {
      items.push(taggedValue('accessor'))
    }
  }
  return `[${items.join(', ')}]`
}

function serializeDate(value: Date): string {
  if (Number.isNaN(value.getTime())) {
    return taggedValue('invalid_date')
  }
  return taggedValue('date', value.toISOString())
}

function serializeMap(value: Map<unknown, unknown>, ancestors: WeakSet<object>): string {
  const entries = [...value.entries()]
    .map(([key, entryValue]) => [serializeValue(key, ancestors), serializeValue(entryValue, ancestors)] as const)
    .sort(([leftKey, leftValue], [rightKey, rightValue]) => (
      compareText(leftKey, rightKey) || compareText(leftValue, rightValue)
    ))
  const serializedEntries = entries.map(([key, entryValue]) => `[${key}, ${entryValue}]`).join(', ')
  return `{"$xerxes_type": "map", "entries": [${serializedEntries}]}`
}

function serializeSet(value: Set<unknown>, ancestors: WeakSet<object>): string {
  const entries = [...value].map(entry => serializeValue(entry, ancestors)).sort(compareText)
  return `{"$xerxes_type": "set", "values": [${entries.join(', ')}]}`
}

function serializeRecord(value: object, ancestors: WeakSet<object>): string {
  const entries: string[] = []
  for (const key of Object.keys(value).sort(compareText)) {
    const descriptor = Object.getOwnPropertyDescriptor(value, key)
    const serializedValue = descriptor === undefined
      ? taggedValue('missing')
      : 'value' in descriptor
        ? serializeValue(descriptor.value, ancestors)
        : taggedValue('accessor')
    entries.push(`${quote(key)}: ${serializedValue}`)
  }
  return `{${entries.join(', ')}}`
}

function taggedValue(type: string, value?: string): string {
  if (value === undefined) {
    return `{"$xerxes_type": ${quote(type)}}`
  }
  return `{"$xerxes_type": ${quote(type)}, "value": ${quote(value)}}`
}

function quote(value: string): string {
  return JSON.stringify(value)
}

function compareText(left: string, right: string): number {
  if (left < right) return -1
  if (left > right) return 1
  return 0
}

function functionName(value: Function): string {
  try {
    return value.name || 'anonymous'
  } catch {
    return 'anonymous'
  }
}

function isPlainRecord(value: object): boolean {
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

function objectDescription(value: object): string {
  try {
    return String(value)
  } catch {
    return 'unserializable'
  }
}
