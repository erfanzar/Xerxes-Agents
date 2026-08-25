// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readFile } from 'node:fs/promises'
import { basename } from 'node:path'

import { REDACTED, type RedactionRule } from './redact.js'

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
  // Bounded like translate_execute below: unbounded stacked `[^>]*` quantifiers
  // backtrack quadratically (ReDoS) on input such as a long run of `<!--`.
  { id: 'html_comment_injection', pattern: /<!--[^>]{0,400}?(?:ignore|override|system|secret|hidden)[^>]{0,200}?-->/gi },
  { id: 'hidden_div', pattern: /<\s*div\s+style\s*=\s*["'][\s\S]{0,400}?display\s*:\s*none/gi },
  // Every anchor-to-tail wildcard in this list is bounded: unbounded quantifiers
  // backtrack quadratically (ReDoS) on long attacker-controlled lines, freezing
  // the event loop for seconds per scan. Realistic injections sit far inside the
  // caps; comments too wide to inspect are flagged wholesale by the
  // oversized-comment sweep in scanContextContent instead of being trusted.
  { id: 'translate_execute', pattern: /translate\s+[^\n]{0,200}?\s+into\s+[^\n]{0,200}?\s+and\s+(execute|run|eval)/gi },
  { id: 'exfil_curl', pattern: /curl\s+[^\n]{0,400}?\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)/gi },
  { id: 'read_secrets', pattern: /cat\s+[^\n]{0,400}?(\.env|credentials|\.netrc|\.pgpass)/gi },
]

/**
 * Credential-only subset of the redaction rules, for content a human or an
 * agent asked to persist verbatim.
 *
 * This deliberately excludes the email and phone rules in
 * DEFAULT_REDACTION_RULES: a durable USER.md exists precisely to hold the
 * user's own contact details, and running the full rule set over memory
 * writes would reject the one file whose whole purpose is that data.
 *
 * Every pattern is anchored on an issuer prefix or an explicit field label, so
 * a match means "this is a secret", not "this looks high-entropy". Field rules
 * capture the label in group 1 and the secret in group 2 and replace only the
 * secret; collapsing the whole match rewrites the surrounding prose, turning
 * `API key: stored in Vault as api_key=<value>` into a sentence the user never
 * wrote.
 */
export const CREDENTIAL_REDACTION_RULES: readonly RedactionRule[] = Object.freeze([
  Object.freeze({ name: 'anthropic_token', pattern: /\bsk-ant-[A-Za-z0-9_-]{16,}\b/g, replacement: REDACTED }),
  Object.freeze({ name: 'openai_token', pattern: /\bsk-[A-Za-z0-9_-]{16,}\b/g, replacement: REDACTED }),
  Object.freeze({ name: 'aws_access_key', pattern: /\bAKIA[0-9A-Z]{16}\b/g, replacement: REDACTED }),
  Object.freeze({ name: 'github_pat', pattern: /\bghp_[A-Za-z0-9]{20,}\b/g, replacement: REDACTED }),
  Object.freeze({ name: 'github_fine_grained_pat', pattern: /\bgithub_pat_[A-Za-z0-9_]{20,}\b/g, replacement: REDACTED }),
  Object.freeze({ name: 'slack_token', pattern: /\bxox[abprs]-[A-Za-z0-9-]{10,}\b/g, replacement: REDACTED }),
  Object.freeze({ name: 'google_api_key', pattern: /\bAIza[0-9A-Za-z_-]{35}\b/g, replacement: REDACTED }),
  Object.freeze({
    name: 'jwt_token',
    pattern: /\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b/g,
    replacement: REDACTED,
  }),
  Object.freeze({
    name: 'bearer_token',
    pattern: /(\bbearer\s+)([A-Za-z0-9._~+/-]{16,}={0,2})/gi,
    replacement: '$1' + REDACTED,
  }),
  Object.freeze({
    name: 'credential_field',
    // The value threshold keeps prose such as `api key: ask Erfan` inert, and
    // the lookahead keeps an environment-variable reference — the correct way
    // to record where a key lives — from being treated as the key itself.
    pattern: /((?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret)["']?\s*[:=]\s*["']?)(?!process\.env|os\.environ|import\.meta)([A-Za-z0-9._~+/-]{16,}={0,2})/gi,
    replacement: '$1' + REDACTED,
  }),
])

/** Names of the credential rules that match somewhere in the text, in rule order. */
export function findCredentialPatterns(content: string): string[] {
  const found: string[] = []
  for (const rule of CREDENTIAL_REDACTION_RULES) {
    rule.pattern.lastIndex = 0
    if (rule.pattern.test(content)) found.push(rule.name)
    rule.pattern.lastIndex = 0
  }
  return found
}

/** Whether the text carries something that must never be written to durable storage. */
export function containsCredential(content: string): boolean {
  return findCredentialPatterns(content).length > 0
}

/** Replace credential values while leaving field labels, emails, and phone numbers intact. */
export function redactCredentials(content: string): string {
  let result = content
  for (const rule of CREDENTIAL_REDACTION_RULES) {
    rule.pattern.lastIndex = 0
    result = result.replace(rule.pattern, rule.replacement)
  }
  return result
}

/** Invisible directional and joiner codepoints that can conceal context instructions. */
export const CONTEXT_INVISIBLE_CHARS: ReadonlySet<string> = new Set([
  // Zero-width joiners/separators and word joiner/BOM.
  '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
  // Legacy bidi embedding controls...
  '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
  // ...and the modern bidi isolates and directional marks (LRI/RLI/FSI/PDI, LRM/RLM).
  '\u200e', '\u200f', '\u2066', '\u2067', '\u2068', '\u2069',
])

/**
 * Detector id for terminated HTML comments wider than the bounded
 * html_comment_injection pattern can inspect.
 */
const OVERSIZED_HTML_COMMENT_ID = 'oversized_html_comment'

/**
 * Comments wider than this exceed every window the bounded injection pattern can
 * reach (400 + keyword + trailing characters), so padding instructions deep inside
 * a huge comment would otherwise slip past it. Any terminated comment wider than
 * this many characters is flagged wholesale: length must not become a place to hide.
 */
const OVERSIZED_HTML_COMMENT_CHARS = 600

interface RawSpan {
  readonly end: number
  readonly start: number
}

/**
 * Linear sweep for terminated `<!-- ... -->` comments wider than
 * {@link OVERSIZED_HTML_COMMENT_CHARS}. Pure `indexOf` scanning, so a flood of
 * unterminated openers costs one pass and never backtracks.
 */
function oversizedHtmlCommentSpans(content: string): RawSpan[] {
  const spans: RawSpan[] = []
  let cursor = 0
  while (cursor < content.length) {
    const open = content.indexOf('<!--', cursor)
    if (open < 0) return spans
    const close = content.indexOf('-->', open + 4)
    if (close < 0) return spans
    const end = close + 3
    if (end - open > OVERSIZED_HTML_COMMENT_CHARS) spans.push({ start: open, end })
    cursor = end
  }
  return spans
}

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

  // Second pass closing the recall gap the bounded comment pattern leaves behind:
  // an oversized comment is flagged in full even when its keywords sit past the
  // pattern's reachable windows.
  for (const span of oversizedHtmlCommentSpans(content)) {
    spans.push({ start: span.start, end: span.end, id: OVERSIZED_HTML_COMMENT_ID })
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
