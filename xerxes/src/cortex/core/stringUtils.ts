// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** One native value that can be interpolated into a Cortex task template. */
export type TemplateInput = unknown

export interface TemplateInputValidation {
  readonly errors: readonly string[]
  readonly valid: boolean
}

const TEMPLATE_VARIABLE = /\{([a-zA-Z_][a-zA-Z0-9_]*)\}/g

/**
 * Substitute `{variable}` placeholders with values from one input record.
 *
 * Scalar values use the source runtime's human-readable rendering: booleans
 * are `True`/`False`, null is empty, and records/arrays are JSON encoded with
 * Unicode left intact. Non-JSON nested values fall back to their native
 * string representation, just as Python's JSON fallback did.
 */
export function interpolateInputs(
  input: string | null | undefined,
  inputs: Readonly<Record<string, TemplateInput>>,
): string {
  if (!input) return ''
  assertInputRecord(inputs)

  return input.replace(TEMPLATE_VARIABLE, (_placeholder, key: string) => {
    if (!Object.hasOwn(inputs, key)) {
      throw new Error(`Missing required template variable '${key}'`)
    }
    return stringifyTemplateInput(inputs[key])
  })
}

/** Return each unique `{variable}` placeholder in source order. */
export function extractTemplateVariables(input: string | null | undefined): Set<string> {
  if (!input) return new Set()
  const variables = new Set<string>()
  for (const match of input.matchAll(TEMPLATE_VARIABLE)) {
    const variable = match[1]
    if (variable !== undefined) variables.add(variable)
  }
  return variables
}

/**
 * Check whether an input record meets a template's placeholder requirements.
 *
 * Extra entries are accepted unless `allowExtra` is explicitly false.
 */
export function validateInputsForTemplate(
  template: string,
  inputs: Readonly<Record<string, TemplateInput>>,
  allowExtra = true,
): TemplateInputValidation {
  assertInputRecord(inputs)
  const required = extractTemplateVariables(template)
  const provided = new Set(Object.keys(inputs))
  const errors: string[] = []

  for (const variable of required) {
    if (!provided.has(variable)) errors.push(`Missing required variable: ${variable}`)
  }
  if (!allowExtra) {
    for (const variable of provided) {
      if (!required.has(variable)) errors.push(`Unexpected variable: ${variable}`)
    }
  }

  return { valid: errors.length === 0, errors }
}

function stringifyTemplateInput(value: TemplateInput): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string' || typeof value === 'number') return String(value)
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  if (Array.isArray(value) || isPlainRecord(value)) return stringifyStructuredValue(value)
  throw new TypeError(`Unsupported type ${describeType(value)} for template variable`)
}

function stringifyStructuredValue(value: readonly unknown[] | Readonly<Record<string, unknown>>): string {
  if (!isJsonCompatible(value, new Set())) return nativeString(value)
  try {
    const serialized = JSON.stringify(value)
    return serialized === undefined ? nativeString(value) : withPythonJsonSpacing(serialized)
  } catch {
    return nativeString(value)
  }
}

function isJsonCompatible(value: unknown, ancestors: Set<object>): boolean {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return true
  if (typeof value === 'number') return Number.isFinite(value)
  if (Array.isArray(value)) {
    if (ancestors.has(value)) return false
    ancestors.add(value)
    const compatible = value.every(item => isJsonCompatible(item, ancestors))
    ancestors.delete(value)
    return compatible
  }
  if (isPlainRecord(value)) {
    if (ancestors.has(value)) return false
    ancestors.add(value)
    const compatible = Object.values(value).every(item => isJsonCompatible(item, ancestors))
    ancestors.delete(value)
    return compatible
  }
  return false
}

/** JSON.stringify omits the spaces that Python json.dumps inserts by default. */
function withPythonJsonSpacing(serialized: string): string {
  let result = ''
  let escaped = false
  let inString = false
  for (const character of serialized) {
    if (inString) {
      result += character
      if (escaped) escaped = false
      else if (character === '\\') escaped = true
      else if (character === '"') inString = false
      continue
    }
    if (character === '"') {
      inString = true
      result += character
      continue
    }
    if (character === ':' || character === ',') {
      result += `${character} `
      continue
    }
    result += character
  }
  return result
}

function assertInputRecord(value: unknown): asserts value is Readonly<Record<string, TemplateInput>> {
  if (!isPlainRecord(value)) throw new TypeError('inputs must be a plain record')
}

function isPlainRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

function nativeString(value: unknown): string {
  try {
    return String(value)
  } catch {
    return Object.prototype.toString.call(value)
  }
}

function describeType(value: unknown): string {
  if (value === null) return 'null'
  if (Array.isArray(value)) return 'array'
  return typeof value
}
