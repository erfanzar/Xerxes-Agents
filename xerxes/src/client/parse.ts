// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { isRecord } from '../hub/protocol.js'

/** Tolerant readers for untrusted Hub payloads. Browser-safe. */

export function rec(value: unknown): Readonly<Record<string, unknown>> {
  return isRecord(value) ? value : {}
}

export function arr(value: unknown): readonly unknown[] {
  return Array.isArray(value) ? value : []
}

export function str(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined
}

export function num(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

export function clip(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text
}

/** An ISO timestamp from a value, or the fallback when it does not parse. */
export function isoOr(value: unknown, fallback: string): string {
  return typeof value === 'string' && Number.isFinite(Date.parse(value)) ? value : fallback
}
