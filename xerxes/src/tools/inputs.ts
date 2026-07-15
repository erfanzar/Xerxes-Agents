// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import type { JsonObject } from '../types/toolCalls.js'

export function requiredString(inputs: JsonObject, name: string): string {
  const value = inputs[name]
  if (typeof value !== 'string' || !value) {
    throw new ValidationError(name, 'must be a non-empty string', value)
  }
  return value
}

export function optionalString(inputs: JsonObject, name: string): string | undefined {
  const value = inputs[name]
  if (value === undefined) {
    return undefined
  }
  if (typeof value !== 'string') {
    throw new ValidationError(name, 'must be a string', value)
  }
  return value
}

export function optionalBoolean(inputs: JsonObject, name: string, defaultValue: boolean): boolean {
  const value = inputs[name]
  if (value === undefined) {
    return defaultValue
  }
  if (typeof value !== 'boolean') {
    throw new ValidationError(name, 'must be a boolean', value)
  }
  return value
}

export function optionalInteger(inputs: JsonObject, name: string, defaultValue: number): number {
  const value = inputs[name]
  if (value === undefined) {
    return defaultValue
  }
  if (typeof value !== 'number' || !Number.isInteger(value)) {
    throw new ValidationError(name, 'must be an integer', value)
  }
  return value
}

export function optionalStringArray(inputs: JsonObject, name: string): string[] {
  const value = inputs[name]
  if (value === undefined) {
    return []
  }
  if (!Array.isArray(value) || value.some(item => typeof item !== 'string')) {
    throw new ValidationError(name, 'must be an array of strings', value)
  }
  const strings: string[] = []
  for (const item of value) {
    if (typeof item !== 'string') {
      throw new ValidationError(name, 'must be an array of strings', value)
    }
    strings.push(item)
  }
  return strings
}

export function requireRange(value: number, name: string, minimum: number, maximum: number): number {
  if (value < minimum || value > maximum) {
    throw new ValidationError(name, `must be between ${minimum} and ${maximum}`, value)
  }
  return value
}
