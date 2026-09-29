// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * A workflow agent asked for a schema answers with one JSON object. Shown
 * raw it is a single unreadable line; its fields read better as sections.
 * Older sessions saved only an excerpt of that JSON, so a cut-short object
 * is closed up and its complete fields kept, marked as incomplete.
 */
export interface AgentResult {
  readonly fields: Readonly<Record<string, unknown>>
  /** False when the saved text ended mid-object and later fields are gone. */
  readonly complete: boolean
}

const MAX_REPAIRS = 32

export function agentResult(raw: string): AgentResult | null {
  const text = raw.trim()
  if (!text.startsWith('{')) return null
  const whole = objectOf(text)
  if (whole) return { fields: whole, complete: true }
  // An excerpt ends with an ellipsis the trimming added, not the agent.
  let excerpt = text.replace(/…$/, '')
  for (let attempt = 0; attempt < MAX_REPAIRS && excerpt.length > 1; attempt++) {
    const closed = objectOf(closeJson(excerpt))
    if (closed) return { fields: closed, complete: false }
    // The tail is a key or a value cut mid-token: drop back one field.
    const comma = excerpt.lastIndexOf(',')
    if (comma <= 0) return null
    excerpt = excerpt.slice(0, comma)
  }
  return null
}

function objectOf(text: string): Record<string, unknown> | null {
  try {
    const value: unknown = JSON.parse(text)
    return value !== null && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length > 0
      ? value as Record<string, unknown>
      : null
  } catch {
    return null
  }
}

/** Close the strings, arrays and objects a cut-off JSON text left open. */
function closeJson(text: string): string {
  const closers: string[] = []
  let inString = false
  let escaped = false
  for (const char of text) {
    if (inString) {
      if (escaped) escaped = false
      else if (char === '\\') escaped = true
      else if (char === '"') inString = false
      continue
    }
    if (char === '"') inString = true
    else if (char === '{') closers.push('}')
    else if (char === '[') closers.push(']')
    else if (char === '}' || char === ']') closers.pop()
  }
  let closed = escaped ? text.slice(0, -1) : text
  if (inString) closed += '"'
  closed = closed.replace(/[,:]\s*$/, '')
  return closed + closers.reverse().join('')
}

/** `root_cause` → `Root cause`. */
export function resultLabel(key: string): string {
  const words = key.replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/[_-]+/g, ' ').trim().toLowerCase()
  return words ? words[0]!.toUpperCase() + words.slice(1) : key
}
