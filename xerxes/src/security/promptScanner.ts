// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readFile } from 'node:fs/promises'
import { basename } from 'node:path'

interface ThreatPattern {
  readonly id: string
  readonly pattern: RegExp
}

interface ThreatSpan {
  readonly end: number
  readonly id: string
  readonly start: number
}

interface MergedThreatSpan {
  end: number
  readonly ids: string[]
  readonly start: number
}

/** Deterministic, best-effort patterns for hostile instructions in imported context. */
export const CONTEXT_THREAT_PATTERNS: readonly ThreatPattern[] = [
  { id: 'prompt_injection', pattern: /ignore\s+(all\s+)?(previous|above|prior)\s+instructions/gi },
  { id: 'deception_hide', pattern: /do\s+not\s+tell\s+the\s+user/gi },
  { id: 'sys_prompt_override', pattern: /system\s+prompt\s+override/gi },
  { id: 'disregard_rules', pattern: /disregard\s+(your\s+)?(all\s+)?(any\s+)?(instructions|rules|guidelines)/gi },
  {
    id: 'bypass_restrictions',
    pattern: /act\s+as\s+(if|though)\s+you\s+(have\s+no|don't\s+have)\s+(any\s+)?(restrictions|limits|rules)/gi,
  },
  { id: 'html_comment_injection', pattern: /<!--[^>]*(?:ignore|override|system|secret|hidden)[^>]*-->/gi },
  { id: 'hidden_div', pattern: /<\s*div\s+style\s*=\s*["'][\s\S]*?display\s*:\s*none/gi },
  // Wildcards are bounded: stacked unbounded `.*\s+` quantifiers backtrack
  // quadratically (ReDoS) on long attacker-controlled single lines.
  { id: 'translate_execute', pattern: /translate\s+[^\n]{0,200}?\s+into\s+[^\n]{0,200}?\s+and\s+(execute|run|eval)/gi },
  { id: 'exfil_curl', pattern: /curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)/gi },
  { id: 'read_secrets', pattern: /cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass)/gi },
]

/** Invisible directional and joiner codepoints that can conceal context instructions. */
export const CONTEXT_INVISIBLE_CHARS: ReadonlySet<string> = new Set([
  '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff', '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
])

/**
 * Neutralise detected prompt-injection spans while keeping surrounding context intact.
 */
export function scanContextContent(content: string, filename = 'unknown'): string {
  const displayName = sanitizeDisplayName(filename)
  const spans: ThreatSpan[] = []

  for (let index = 0; index < content.length; index += 1) {
    const character = content[index]
    if (character !== undefined && CONTEXT_INVISIBLE_CHARS.has(character)) {
      spans.push({ start: index, end: index + 1, id: `invisible_unicode_U+${character.charCodeAt(0).toString(16).toUpperCase().padStart(4, '0')}` })
    }
  }

  for (const threat of CONTEXT_THREAT_PATTERNS) {
    threat.pattern.lastIndex = 0
    let match = threat.pattern.exec(content)
    while (match !== null) {
      spans.push({ start: match.index, end: match.index + match[0].length, id: threat.id })
      match = threat.pattern.exec(content)
    }
  }

  if (spans.length === 0) {
    return content
  }

  const merged = mergeThreatSpans(spans)
  const parts: string[] = []
  let cursor = 0
  for (const span of merged) {
    parts.push(content.slice(cursor, span.start))
    parts.push(`[BLOCKED: ${displayName} ${span.ids.join(', ')}]`)
    cursor = span.end
  }
  parts.push(content.slice(cursor))
  return parts.join('')
}

/** Read UTF-8 context and neutralise it; failed reads are represented as a safe placeholder. */
export async function scanContextFile(path: string, filename?: string): Promise<string> {
  const name = sanitizeDisplayName(filename ?? basename(path))
  try {
    return scanContextContent(await readFile(path, 'utf8'), name)
  } catch (error) {
    return `[BLOCKED: ${name} unreadable (${errorMessage(error)})]`
  }
}

/**
 * Keep attacker-controlled filenames from breaking out of the `[BLOCKED: ...]`
 * placeholder or injecting instruction-like text into model context: strip the
 * placeholder's bracket delimiters and control/format characters.
 */
function sanitizeDisplayName(name: string): string {
  return name.replace(/[[\]]/g, '').replace(/[\p{Cc}\p{Cf}]/gu, '') || 'unknown'
}

function mergeThreatSpans(spans: readonly ThreatSpan[]): MergedThreatSpan[] {
  const sorted = [...spans].sort((left, right) => left.start - right.start || right.end - left.end)
  const merged: MergedThreatSpan[] = []
  for (const span of sorted) {
    const previous = merged.at(-1)
    if (previous !== undefined && span.start <= previous.end) {
      previous.end = Math.max(previous.end, span.end)
      if (!previous.ids.includes(span.id)) {
        previous.ids.push(span.id)
      }
      continue
    }
    merged.push({ start: span.start, end: span.end, ids: [span.id] })
  }
  return merged
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
